"""Ewaluacja retrievalu (i opcjonalnie generacji) na złotym zbiorze.

Format złotego zbioru (JSONL, jeden obiekt na linię):
    {"id": "q1", "question": "...", "answer": "...",
     "source_ids": ["plik.pdf#rozdzial/sekcja/fireball"], "tags": ["easy"]}

`source_ids` to identyfikatory źródeł niezależne od chunkingu (patrz `ingest.source_ids_for`
oraz `rpg.py`). Pusta lista oznacza pytanie spoza dokumentów – oczekujemy, że model odmówi.

Metryki retrievalu (liczone tylko dla pytań z niepustym `source_ids`):
- recall@k: jaka część oczekiwanych źródeł pojawiła się w top-k chunkach,
- MRR (Mean Reciprocal Rank): średnia z 1/pozycja pierwszego trafnego chunka (0, gdy brak).
- ctx_chars: ile znaków kontekstu trafia do modelu (top-k chunków) – przy porównywaniu
  strategii chunkingu recall@k ma sens tylko obok tego budżetu (większe chunki = wyższy recall).
- retrieved_sources: źródła i indeksy chunków zwrócone przez retrieval, w kolejności trafności.

Metryki generacji (`--generate`):
- refusal_ok: model odpowiedział na pytanie z dokumentów / odmówił na pytanie spoza nich,
- citation_hit: wśród cytowanych chunków jest oczekiwane źródło,
- inline_cited: odpowiedź ma w tekście znaczniki [n] przy twierdzeniach,
- latency_s: czas retrieval + generacja.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag.config import Settings
from rag.generate import answer_question
from rag.retrieve import Hit, retrieve

HUMAN_SCORES: dict[str, float] = {
    "correct": 1.0,
    "refusal_ok": 1.0,
    "incorrect": 0.0,
    "refusal_bad": 0.0,
}


@dataclass
class GoldItem:
    id: str
    question: str
    answer: str
    source_ids: list[str]
    tags: list[str] = field(default_factory=list)
    needs_review: bool = False
    section_ids: list[str] = field(default_factory=list)


def load_golden(path: Path) -> list[GoldItem]:
    """Wczytuje złoty zbiór z pliku JSONL (puste linie są pomijane)."""
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(GoldItem(**json.loads(line)))
    return items


# --------------------------------------------------------------------------- metryki


def recall_at_k(retrieved: list[list[str]], relevant: list[str], k: int) -> float:
    """Ułamek oczekiwanych źródeł obecnych w top-k (każdy chunk może mieć kilka source_ids)."""
    found = {sid for ids in retrieved[:k] for sid in ids}
    return len(found & set(relevant)) / len(relevant)


def reciprocal_rank(retrieved: list[list[str]], relevant: list[str]) -> float:
    """1 / pozycja pierwszego chunka zawierającego którekolwiek oczekiwane źródło."""
    for rank, ids in enumerate(retrieved, start=1):
        if set(ids) & set(relevant):
            return 1.0 / rank
    return 0.0


def to_level(source_ids: list[str], level: str) -> list[str]:
    """Sprowadza identyfikatory do wybranej granulacji.

    `level="section"`: `plik.pdf#rozdzial/sekcja/fireball` → `plik.pdf#rozdzial/sekcja`.
    Przydatne przy porównaniu chunkingu: duży chunk fixed-size „zahacza” o kilka wpisów naraz,
    co na poziomie wpisów zawyża jego recall.
    """
    if level == "source":
        return list(dict.fromkeys(source_ids))

    def collapse(source_id: str) -> str:
        source, sep, path = source_id.partition("#")
        if not sep or "/" not in path:
            return source_id
        return f"{source}#{path.rsplit('/', 1)[0]}"

    return list(dict.fromkeys(collapse(source_id) for source_id in source_ids))


def gold_ids(item: GoldItem, level: str = "source") -> list[str]:
    """Zwraca oczekiwane identyfikatory źródeł dla danego pytania, w wybranej granulacji."""
    if level == "section" and item.section_ids:
        return list(dict.fromkeys(item.section_ids))
    return to_level(item.source_ids, level)


def retrieved_sources(hits: list[Hit]) -> list[dict[str, Any]]:
    """Zwraca jeden wpis na chunk wraz z należącymi do niego `source_ids`."""
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for hit in hits:
        chunk_index = hit.metadata.get("chunk_index")
        source = hit.metadata.get("source")
        key = (source, chunk_index)
        if key not in seen:
            seen.add(key)
            sources.append({"source_ids": hit.source_ids, "chunk_index": chunk_index})
    return sources


def apply_human_grades(results: dict[str, Any]) -> dict[str, Any]:
    """Przelicza statystyki ręcznych ocen w `summary`."""
    rows = results.get("rows", [])

    def review_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [r.get("human_label") for r in group]
        counts = {label: labels.count(label) for label in HUMAN_SCORES}
        graded = sum(counts.values())
        return {
            "graded": graded,
            "ungraded": len(group) - graded,
            "counts": counts,
            "percentages": {
                label: round(count / graded, 4) if graded else None
                for label, count in counts.items()
            },
            "human_acc": (
                round(
                    sum(
                        HUMAN_SCORES[r["human_label"]]
                        for r in group
                        if r.get("human_label") in HUMAN_SCORES
                    )
                    / graded,
                    4,
                )
                if graded
                else None
            ),
        }

    by_tag: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        tags = row.get("tags") or ["bez_tagu"]
        by_tag.setdefault(str(tags[0]), []).append(row)

    overall = review_stats(rows)
    results.setdefault("summary", {}).update(
        {
            "human_acc": overall["human_acc"],
            "human_review": {
                **overall,
                "by_tag": {tag: review_stats(tag_rows) for tag, tag_rows in sorted(by_tag.items())},
            },
        }
    )
    return results


# --------------------------------------------------------------------------- przebieg


def evaluate(
    items: list[GoldItem],
    settings: Settings,
    ks: tuple[int, ...] = (1, 3, 5),
    generate: bool = False,
    collection: str | None = None,
    level: str = "source",
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Liczy metryki dla całego złotego zbioru. Zwraca słownik z wynikami per pytanie i średnimi.

    Args:
        level: "source" – identyfikatory jak w złotym zbiorze (np. wpis `.../fireball`);
            "section" – porównanie na poziomie sekcji: używamy `section_id` z chunka
            i opcjonalnego pola `section_ids` w złotym zbiorze, a `to_level` jest fallbackiem.
    """
    max_k = max(ks)
    rows: list[dict[str, Any]] = []

    def hit_ids(hit: Hit) -> list[str]:
        if level == "section" and hit.metadata.get("section_id"):
            return [str(hit.metadata["section_id"])]
        return to_level(hit.source_ids, level)

    for item in items:
        row: dict[str, Any] = {"id": item.id, "question": item.question, "tags": item.tags}
        hits: list[Hit] = retrieve(
            item.question, settings, k=max_k, collection=collection, filters=filters
        )
        row["retrieved_sources"] = retrieved_sources(hits)
        if generate:
            result = answer_question(
                item.question,
                settings,
                k=settings.top_k,
                collection=collection,
                filters=filters,
            )
            cited = {sid for h in result.cited_hits() for sid in hit_ids(h)}
            gold = set(gold_ids(item, level))
            row |= {
                "answer": result.answer.answer,
                "citations": result.cited_numbers,
                "latency_s": result.retrieve_s + result.generate_s,
                "structured_mode": result.mode,
                # Odmowa poprawna, gdy pytanie jest spoza dokumentów (pusty source_ids).
                "refusal_ok": result.answered_from_context == bool(item.source_ids),
                "citation_hit": bool(cited & gold) if item.source_ids else None,
                # Czy odpowiedź ma znaczniki [n] w tekście (tylko dla pytań z odpowiedzią).
                "inline_cited": result.has_inline_citations if item.source_ids else None,
            }
        row["ctx_chars"] = sum(len(h.text) for h in hits[: settings.top_k])
        if item.source_ids:
            retrieved = [hit_ids(h) for h in hits]
            relevant = gold_ids(item, level)
            for k in ks:
                row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            row["mrr"] = reciprocal_rank(retrieved, relevant)
        rows.append(row)
    return {"rows": rows, "summary": summarize(rows, ks)}


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(statistics.mean(values), 3) if values else None


def summarize(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, Any]:
    """Średnie metryk po wszystkich pytaniach."""
    summary: dict[str, Any] = {"questions": len(rows)}
    keys = ["mrr", "ctx_chars", "refusal_ok", "citation_hit", "inline_cited", "latency_s"]
    for key in [*(f"recall@{k}" for k in ks), *keys]:
        summary[key] = _mean(rows, key)
    return summary


def to_markdown(results: dict[str, Any], title: str) -> str:
    """Tabela wyników w Markdownie (do wklejenia w raport / slajdy)."""
    rows, summary = results["rows"], results["summary"]
    cols = [
        c
        for c in summary
        if c not in {"questions", "human_review"} and (summary[c] is not None or c == "human_acc")
    ]
    display_cols = ["human_score" if c == "human_acc" else c for c in cols]
    lines = [f"# {title}", "", "| id | pytanie | " + " | ".join(display_cols) + " |"]
    lines.append("|" + "---|" * (len(cols) + 2))
    for r in rows:
        values = " | ".join(
            format_cell(r.get("human_score") if c == "human_acc" else r.get(c)) for c in cols
        )
        lines.append(f"| {r['id']} | {r['question']} | {values} |")
    means = " | ".join(f"**{format_cell(summary[c])}**" for c in cols)
    lines.append(f"| **średnia** | {summary['questions']} pytań | {means} |")
    if review := summary.get("human_review"):
        lines.extend(
            [
                "",
                "## Oceny ręczne",
                "",
                f"- Ocenione: {review['graded']}/{review['graded'] + review['ungraded']}",
                f"- Pominięte: {review['ungraded']}",
                "",
                "| pierwszy tag | ocenione | pominięte | poprawne | niepoprawne | poprawne odmowy | niepoprawne odmowy | human_score |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tag, stats in review.get("by_tag", {}).items():
            lines.append(
                f"| {tag} | {stats['graded']} | {stats['ungraded']} | "
                f"{stats['counts']['correct']} | {stats['counts']['incorrect']} | "
                f"{stats['counts']['refusal_ok']} | {stats['counts']['refusal_bad']} | "
                f"{format_cell(stats['human_acc'])} |"
            )
        lines.append(
            f"| **suma** | **{review['graded']}** | **{review['ungraded']}** | "
            f"**{review['counts']['correct']}** | **{review['counts']['incorrect']}** | "
            f"**{review['counts']['refusal_ok']}** | **{review['counts']['refusal_bad']}** | "
            f"**{format_cell(review['human_acc'])}** |"
        )
    return "\n".join(lines) + "\n"


def format_cell(value: Any) -> str:
    """Wartość metryki jako krótki tekst (tabela w terminalu i w Markdownie)."""
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, float):
        return f"{value:.0f}" if value >= 100 else f"{value:.2f}"
    return str(value)


def save_results(results: dict[str, Any], settings: Settings, out_dir: Path, label: str) -> Path:
    """Zapisuje wyniki do `eval/results/<timestamp>_<label>.json` (+ `.md`) i zwraca ścieżkę."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}_{label}.json"
    config = asdict(settings) | {"llm_api_key": "***", "embed_api_key": "***"}
    path.write_text(
        json.dumps({"config": config, **results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    title = f"{label} · {settings.llm_model} · {stamp}"
    path.with_suffix(".md").write_text(to_markdown(results, title), encoding="utf-8")
    return path
