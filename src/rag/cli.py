"""Interfejs wiersza poleceń: `uv run rag <komenda>` (Makefile tylko to opakowuje)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from rag.config import CHUNKING_STRATEGIES, get_settings

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    from rag.ingest import ingest

    stats = ingest(
        [Path(p) for p in (args.paths or [get_settings().data_dir])],
        get_settings(),
        strategy=args.chunking,
        collection=args.collection,
        recreate=args.recreate,
    )
    console.print(
        f"✅ Zaindeksowano {stats['chunks']} chunków z {stats['files']} plików do "
        f"kolekcji [bold]{stats['collection']}[/] w {stats['total_s']}s "
        f"(embedding: {stats['embed_s']}s)"
    )


def cmd_ask(args: argparse.Namespace) -> None:
    from rag.config import parse_filters
    from rag.generate import answer_question, best_quote, render_answer

    result = answer_question(
        args.question,
        get_settings(),
        k=args.k,
        collection=args.collection,
        filters=parse_filters(",".join(args.filter)),
    )
    console.print(Markdown(render_answer(result)))
    if result.answered_from_context:
        console.print("\n[bold]Źródła cytowane:[/]")
        for n in result.cited_numbers:
            hit = result.hits[n - 1]
            quote = best_quote(hit.text, result.answer.answer)
            console.print(f"  [{n}] {hit.label()}  [dim](score {hit.score:.3f})[/]")
            console.print(f"      [italic]„{quote}”[/]")
        if not result.has_inline_citations:
            console.print("[yellow]Uwaga: brak znaczników [n] w tekście odpowiedzi.[/]")
    else:
        console.print(
            "\n[yellow]Model nie zacytował żadnego fragmentu – kontekst nie zawiera odpowiedzi.[/]"
        )
    if retrieved_only := result.retrieved_only():
        console.print("[bold]Pobrane (retrieved), niecytowane:[/]")
        for i, hit in retrieved_only:
            console.print(f"  [dim][{i}] {hit.label()}  (score {hit.score:.3f})[/]")
    console.print(
        f"[dim]retrieval {result.retrieve_s}s · generacja {result.generate_s}s · "
        f"structured output: {result.mode}[/]"
    )
    if args.show_context:
        for i, hit in enumerate(result.hits, start=1):
            console.rule(f"[{i}] {hit.label()}")
            console.print(hit.text)


def cmd_agent(args: argparse.Namespace) -> None:
    from rag.agent import AgentStep, run_agent

    def show(step: AgentStep) -> None:
        if step.kind == "tool_call":
            console.print(f"[cyan]🔧 {step.arguments}[/]")
            console.print(f"[dim]{step.content[:300]}…[/]")
        else:
            console.print(Markdown(step.content))

    settings = get_settings()
    if args.question:
        run_agent(args.question, settings, on_step=show)
        return
    console.print("Tryb interaktywny agenta (pusta linia kończy).")
    while question := console.input("[bold]> [/]").strip():
        run_agent(question, settings, on_step=show)


# Krótkie nagłówki, żeby tabela mieściła się w wąskim terminalu (legenda pod tabelą).
SHORT_HEADERS = {
    "mrr": "MRR",
    "ctx_chars": "ctx",
    "refusal_ok": "odm.",
    "citation_hit": "cyt.",
    "inline_cited": "[n]",
    "latency_s": "t[s]",
    "human_acc": "ręczna",
}


def cmd_eval(args: argparse.Namespace) -> None:
    from rag.eval import evaluate, format_cell, load_golden, save_results

    settings = get_settings()
    items = load_golden(Path(args.golden))
    results = evaluate(
        items, settings, generate=args.generate, collection=args.collection, level=args.level
    )
    summary = results["summary"]
    # Kolumny = metryki policzone w tym przebiegu (bez --generate nie ma metryk odpowiedzi).
    metric_cols = [c for c in summary if c != "questions" and summary[c] is not None]
    table = Table(
        title=f"Eval: {args.golden} · {settings.llm_model} · {args.collection or settings.collection}"
    )
    table.add_column("id", no_wrap=True, min_width=7)
    # Kolumna z pytaniem dostaje to, co zostaje po metrykach (pełne pytania są w pliku .md).
    question_width = max(10, min(50, console.width - 10 - 8 * len(metric_cols)))
    table.add_column("pytanie", width=question_width, no_wrap=True, overflow="ellipsis")
    for col in metric_cols:
        header = SHORT_HEADERS.get(col, col.replace("recall@", "R@"))
        # escape: „[n]” i „t[s]” rich wziąłby za znaczniki formatowania.
        table.add_column(
            escape(header), justify="right", no_wrap=True, min_width=max(4, len(header))
        )
    for row in results["rows"]:
        table.add_row(row["id"], row["question"], *(format_cell(row.get(c)) for c in metric_cols))
    table.add_section()
    table.add_row(
        "ŚREDNIA",
        f"{summary['questions']} pytań",
        *(format_cell(summary.get(c)) for c in metric_cols),
    )
    console.print(table)
    legend = "R@k = recall@k · MRR = mean reciprocal rank · ctx = znaki kontekstu w top-k"
    if args.generate:
        legend += (
            " · odm. = poprawna odmowa/odpowiedź · cyt. = cytat z oczekiwanego źródła"
            " · [n] = znaczniki źródeł w tekście"
        )
    console.print(f"[dim]{escape(legend)}[/]")

    label = f"{Path(args.golden).stem}_{args.collection or settings.collection}"
    if args.level != "source":
        label += f"_{args.level}"
    if args.generate:
        label += f"_{settings.llm_model.replace(':', '-').replace('/', '-')}"
    results |= {
        "golden": args.golden,
        "collection": args.collection or settings.collection,
        "level": args.level,
        "generate": args.generate,
    }
    path = save_results(results, settings, Path("eval/results"), label)
    console.print(f"[dim]Zapisano: {path}[/]")


def cmd_review(args: argparse.Namespace) -> None:
    import json

    from rich.panel import Panel

    from rag.eval import (
        HUMAN_SCORES,
        apply_human_grades,
        gold_ids,
        load_golden,
        to_markdown,
    )

    path = Path(args.result_file)
    if not path.exists():
        raise ValueError(f"Plik nie istnieje: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    if not rows or "answer" not in rows[0]:
        raise ValueError(
            "Ten plik wyników nie zawiera wygenerowanych odpowiedzi. "
            "Uruchom najpierw: uv run rag eval --generate"
        )

    golden_path = Path(args.golden or data.get("golden", "eval/golden_rpg.jsonl"))
    gold_by_id = (
        {item.id: item for item in load_golden(golden_path)} if golden_path.exists() else {}
    )

    choices = {
        "y": ("correct", "poprawna"),
        "1": ("correct", "poprawna"),
        "n": ("incorrect", "niepoprawna"),
        "0": ("incorrect", "niepoprawna"),
        "r": ("refusal_ok", "poprawna odmowa"),
        "b": ("refusal_bad", "błędna odmowa / halucynacja"),
    }

    def show_review_summary() -> None:
        review = data["summary"]["human_review"]
        descriptions = {
            "correct": "poprawne",
            "incorrect": "niepoprawne",
            "refusal_ok": "poprawne odmowy",
            "refusal_bad": "błędne odmowy",
        }
        parts = [
            f"{descriptions[label]}: {review['counts'][label]} ({review['percentages'][label]:.1%})"
            for label in descriptions
            if review["counts"][label]
        ]
        acc = data["summary"].get("human_acc")
        console.print(
            f"[bold]Podsumowanie ocen:[/] {review['graded']}/{len(rows)} ocenionych · "
            f"{review['ungraded']} pominiętych · "
            f"human_score = {f'{acc:.1%}' if acc is not None else 'brak'}"
        )
        if parts:
            console.print("[dim]" + " · ".join(parts) + "[/]")
        by_tag = review.get("by_tag", {})
        if by_tag:
            table = Table(title="Oceny według tagów")
            table.add_column("tag")
            table.add_column("ocenione", justify="right")
            table.add_column("pominięte", justify="right")
            table.add_column("poprawne", justify="right")
            table.add_column("niepoprawne", justify="right")
            table.add_column("poprawne odmowy", justify="right")
            table.add_column("niepoprawne odmowy", justify="right")
            table.add_column("human_score", justify="right")
            for tag, stats in by_tag.items():
                acc = stats["human_acc"]
                table.add_row(
                    tag,
                    str(stats["graded"]),
                    str(stats["ungraded"]),
                    str(stats["counts"]["correct"]),
                    str(stats["counts"]["incorrect"]),
                    str(stats["counts"]["refusal_ok"]),
                    str(stats["counts"]["refusal_bad"]),
                    f"{acc:.1%}" if acc is not None else "–",
                )
            console.print(table)

    def save_review_outputs() -> None:
        apply_human_grades(data)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        md_path = path.with_suffix(".md")
        title = path.stem
        if md_path.exists():
            first_line = md_path.read_text(encoding="utf-8").splitlines()
            if first_line and first_line[0].startswith("# "):
                title = first_line[0][2:]
        md_path.write_text(to_markdown(data, title), encoding="utf-8")

    legend = "[y/1] poprawna · [n/0] niepoprawna · [r] poprawna odmowa · [b] błędna odmowa · [Enter/s] pomiń · [q] zapisz i wyjdź"
    console.print(
        f"[bold]Ręczna ocena:[/] {escape(path.name)} ({len(rows)} pytań)\n"
        f"[dim]{escape(legend)}[/]\n"
    )

    level = data.get("level", "source")

    for idx, row in enumerate(rows, start=1):
        if args.only_ungraded and row.get("human_label"):
            continue

        gold = gold_by_id.get(row["id"])
        expected_ids = gold_ids(gold, level) if gold else []
        expected_refusal = bool(gold and not expected_ids)
        gold_answer = gold.answer if gold else "(brak wzorca)"
        gold_sources = ", ".join(expected_ids) if expected_ids else "BRAK (oczekiwana odmowa)"

        body = (
            f"[bold]Pytanie:[/] {escape(row['question'])}\n"
            f"[bold]Wzorzec:[/] [green]{escape(gold_answer)}[/]\n"
            f"[bold]Oczekiwane źródła:[/] [dim]{escape(gold_sources)}[/]\n\n"
            f"[bold]Odpowiedź modelu:[/]\n{escape(row.get('answer') or '(pusta)')}\n\n"
            f"[dim]Cytowania: {row.get('citations', [])} · "
            f"auto refusal_ok={row.get('refusal_ok')} · "
            f"citation_hit={row.get('citation_hit')}[/]"
        )
        sources = row.get("retrieved_sources", [])
        if sources:
            source_lines = ["[bold]Źródła z retrieval:[/]"]
            cited_numbers = set(row.get("citations", []))
            for number, source in enumerate(sources, start=1):
                source_ids = ", ".join(source.get("source_ids", [])) or "brak source_ids"
                source_line = (
                    f"[{number}] {source_ids} (chunk_index={source.get('chunk_index', '?')})"
                )
                style = "bold" if number in cited_numbers else "dim"
                source_lines.append(f"[{style}]{escape(source_line)}[/{style}]")
            body += "\n\n" + "\n".join(source_lines)
        current = row.get("human_label")
        title = escape(f"[{idx}/{len(rows)}] {row['id']}")
        if expected_refusal:
            title += " (pytanie spoza korpusu)"
        if current:
            title += f" · obecna ocena: {current}"

        console.print(Panel(body, title=title, border_style="cyan"))

        while True:
            raw = console.input(r"[bold]Ocena [y/n/r/b/s/q]: [/]").strip().lower()
            if raw in ("", "s"):
                break
            if raw == "q":
                save_review_outputs()
                console.print(
                    f"[green]Zapisano dotychczasowe oceny w {path} i {path.with_suffix('.md')}[/]"
                )
                show_review_summary()
                return
            if raw in choices:
                label, desc = choices[raw]
                row["human_label"] = label
                row["human_score"] = HUMAN_SCORES[label]
                note = console.input("[dim]Notatka (opcjonalnie, Enter pomija): [/]").strip()
                if note:
                    row["human_note"] = note
                console.print(f"  → zapisano: [bold]{desc}[/]\n")
                break
            console.print("[yellow]Wpisz y, n, r, b, s lub q.[/]")

    save_review_outputs()
    console.print(f"✅ Zapisano oceny w [bold]{path}[/] i [bold]{path.with_suffix('.md')}[/]")
    show_review_summary()


def cmd_check(_: argparse.Namespace) -> None:
    from rag.check import run_checks

    if not run_checks(get_settings()):
        sys.exit(1)


def cmd_reset_cache(_: argparse.Namespace) -> None:
    from rag.structured import CACHE_PATH, reset_cache

    if reset_cache():
        console.print(f"Usunięto {CACHE_PATH} – tryb structured output zostanie wykryty od nowa.")
    else:
        console.print(f"Brak cache ({CACHE_PATH}).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="RAG dla podręczników RPG – D&D 5e, Mythras, SW5e"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="więcej logów")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="zaindeksuj dokumenty")
    p.add_argument("paths", nargs="*", help="pliki lub katalogi (domyślnie DATA_DIR z .env)")
    p.add_argument("--chunking", choices=CHUNKING_STRATEGIES, help="strategia chunkingu")
    p.add_argument("--collection", help="nazwa kolekcji (domyślnie COLLECTION z .env)")
    p.add_argument("--recreate", action="store_true", help="usuń i utwórz kolekcję od nowa")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="zadaj pytanie (RAG)")
    p.add_argument("question")
    p.add_argument("-k", type=int, help="liczba chunków w kontekście (domyślnie TOP_K)")
    p.add_argument("--collection")
    p.add_argument("--show-context", action="store_true", help="pokaż pełny tekst chunków")
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="POLE=WARTOŚĆ",
        help="filtr metadanych, np. --filter system=D&D 5e (można powtarzać)",
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("review", help="ręczna ocena wygenerowanych odpowiedzi w pliku wyników JSON")
    p.add_argument("result_file", help="ścieżka do pliku eval/results/*.json (z --generate)")
    p.add_argument("--golden", help="opcjonalna ścieżka do złotego zbioru (domyślnie z pliku JSON)")
    p.add_argument(
        "--only-ungraded",
        action="store_true",
        help="pokazuj tylko pytania, które nie mają jeszcze ręcznej oceny",
    )
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("agent", help="agent z narzędziem search_docs")
    p.add_argument("question", nargs="?")
    p.set_defaults(func=cmd_agent)

    p = sub.add_parser("eval", help="ewaluacja na złotym zbiorze")
    p.add_argument("--golden", default="eval/golden_rpg.jsonl")
    p.add_argument("--generate", action="store_true", help="oceniaj też odpowiedzi (wolniej)")
    p.add_argument(
        "--level",
        choices=("source", "section"),
        default="source",
        help="granulacja source_ids: jak w złotym zbiorze albo na poziomie sekcji",
    )
    p.add_argument("--collection")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("check", help="smoke test środowiska")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("reset-cache", help="usuń cache wykrytego trybu structured output")
    p.set_defaults(func=cmd_reset_cache)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(show_path=False, show_time=False)],
    )
    # Wykrycie trybu structured output pokazujemy zawsze (jedna linia na INFO) – to ważna
    # informacja dydaktyczna. Szczegóły każdej próby: `rag -v ...` (DEBUG).
    logging.getLogger("rag.structured").setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        args.func(args)
    except (RuntimeError, ValueError) as err:
        console.print(f"[red]Błąd:[/] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
