"""Podręczniki RPG: struktura rozdziałów/sekcji i wpisy (zaklęcia, stat blocki)."""

from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag.chunking import Block, Chunk, fixed_size, recursive_split
from rag.normalize import repair_missing_ligatures

STRUCTURE_SIZE_FACTOR = 1.5


@dataclass(frozen=True)
class OutlineEntry:
    """Pozycja z outline PDF albo ze spisu treści: poziom, tytuł, strona fizyczna (od 1)."""

    level: int
    title: str
    page: int


@dataclass
class Section:
    """Sekcja podręcznika: fragment tekstu od nagłówka do następnego nagłówka."""

    title: str
    chapter: str | None
    level: int
    start: int  # offsets w RpgDoc.text
    end: int
    page: int  # strona fizyczna PDF (do source_ids)
    page_label: str  # numer wydrukowany w książce (do cytowań)
    path: str  # slug ścieżki, np. chapter-11-spells/fireball
    kind: str = "section"


@dataclass
class Entry:
    """Wpis w sekcji: zaklęcie, stat block potwora albo inny wyróżniony fragment."""

    title: str
    kind: str  # "spell" | "power" | "monster"
    start: int
    end: int
    page: int
    page_label: str
    path: str  # np. chapter-11-spells/fireball


@dataclass
class RpgDoc:
    """Cały podręcznik jako jeden tekst + mapa stron i rozpoznane sekcje."""

    text: str
    page_starts: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    page_labels: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)

    def page_at(self, offset: int) -> tuple[int, str]:
        """Strona fizyczna i numer wydrukowany dla danego offsetu w tekście."""
        if not self.page_starts:
            return 0, ""
        index = max(0, bisect.bisect_right(self.page_starts, offset) - 1)
        return self.pages[index], self.page_labels[index]


# --------------------------------------------------------------------------- spis treści


def read_outline(path: Path) -> list[OutlineEntry]:
    """Czyta zakładki (outline) PDF. Pusta lista = książka nie ma outline."""
    reader = PdfReader(str(path))
    entries: list[OutlineEntry] = []

    def walk(items: list[Any], level: int) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item, level + 1)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:  # noqa: BLE001 – uszkodzony bookmark nie powinien psuć ingestu
                continue
            title = " ".join(str(item.title).split())
            if title and 1 <= page <= len(reader.pages):
                entries.append(OutlineEntry(level=level, title=title, page=page))

    walk(reader.outline, 0)
    return entries if len(entries) >= 2 else []


def parse_toc(blocks: list[Block]) -> list[OutlineEntry]:
    """Buduje spis treści z tekstu stron „TABLE OF CONTENTS” (fallback dla PDF bez outline)."""
    if not blocks:
        return []
    total_pages = max((int(b.metadata.get("page", 0)) for b in blocks), default=0)
    entries: list[OutlineEntry] = []
    started = False

    for block in blocks:
        text = block.text or ""
        if "TABLE OF CONTENTS" not in text.upper():
            continue
        for raw_line in text.splitlines():
            line = repair_missing_ligatures(raw_line.rstrip())
            if not line.strip():
                continue
            match = re.match(r"^(?P<indent> *)(?P<title>.+?)\s+(?P<page>\d{1,3})\s*$", line)
            if not match:
                continue
            title = " ".join(match.group("title").split())
            page = int(match.group("page"))
            if not title or page < 1 or page > total_pages:
                continue
            if title.upper().startswith("TABLE OF CONTENTS"):
                continue
            if not started:
                if title.lower() not in ("introduction", "chapter 1"):
                    continue
                started = True
            if title.lower() in ("credits", "preface"):
                continue
            if entries and page < entries[-1].page:
                continue
            level = min(len(match.group("indent")), 2)
            entries.append(OutlineEntry(level=level, title=title, page=page))

    seen: set[tuple[int, str, int]] = set()
    unique: list[OutlineEntry] = []
    for entry in entries:
        key = (entry.level, entry.title, entry.page)
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique if len(unique) >= 3 else []


# --------------------------------------------------------------------------- tekst i strony

_RUNNING_HEADER = (
    re.compile(r"^\s*\d{0,3}\s*D&D (?:Player['’]s|DM['’]s) Basic Rules.*\|.*$", re.I),
    re.compile(r"^Not for resale\..*$", re.I),
    re.compile(r"^Mythras:\s+.*$", re.I),
    re.compile(r"^CHAPTER\s+\d+\s*\|.*\d+\s*$", re.I),
    re.compile(r"^\s*TABLE OF CONTENTS.*$", re.I),
)


def _is_page_number(line: str, physical_page: int) -> bool:
    match = re.fullmatch(r"\s*(\d{1,3})\s*", line)
    return bool(match) and abs(int(match.group(1)) - physical_page) <= 3


def _detect_page_label(lines: list[str], physical_page: int) -> str:
    """Numer wydrukowany w książce (Mythras ma przesunięcie o 1 względem PDF)."""
    for line in lines[:3]:
        stripped = line.strip()
        match = re.match(r"^(\d{1,3})\s*D&D", stripped, re.I)
        if match and abs(int(match.group(1)) - physical_page) <= 3:
            return match.group(1)
        if _is_page_number(stripped, physical_page):
            return stripped
    for line in lines[-4:]:
        match = re.search(r"\b(\d{1,3})\s*$", line.strip())
        if match and abs(int(match.group(1)) - physical_page) <= 3:
            return match.group(1)
    return str(physical_page)


def _repeated_edge_lines(blocks: list[Block], first_page: int) -> set[str]:
    """Linie powtarzane na początku/końcu stron (żywa pagina, stopka) – z cyframi jako #."""
    counts: Counter[str] = Counter()
    pages = 0
    for block in blocks:
        page = int(block.metadata.get("page", 0))
        if page < first_page:
            continue
        lines = [line.strip() for line in (block.text or "").splitlines() if line.strip()]
        if not lines:
            continue
        pages += 1
        for line in [*lines[:2], *lines[-2:]]:
            counts[re.sub(r"\d+", "#", line)] += 1
    threshold = max(3, pages // 4)
    return {line for line, count in counts.items() if count >= threshold}


def _clean_page(text: str, physical_page: int, repeated: set[str]) -> tuple[str, str]:
    """Usuwa żywą paginę/stopkę i zwraca (czysty tekst, numer wydrukowany)."""
    lines = text.splitlines()
    label = _detect_page_label(lines, physical_page)
    kept: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_page_number(stripped, physical_page):
            continue
        if any(pattern.match(stripped) for pattern in _RUNNING_HEADER):
            continue
        edge = index < 2 or index >= len(lines) - 2
        if edge and re.sub(r"\d+", "#", stripped) in repeated:
            continue
        kept.append(stripped)
    return "\n".join(kept).strip(), label


def build_stream(blocks: list[Block], outline: list[OutlineEntry], source: str) -> RpgDoc:
    """Skleja strony w jeden tekst, usuwa paginę i wyznacza sekcje z outline.

    Args:
        blocks: strony PDF po normalizacji (metadata zawiera `page`).
        outline: wynik `read_outline` albo `parse_toc`; pusta lista = podział po stronach.
        source: nazwa pliku (do budowy `source_ids` w chunkach).
    """
    by_page: dict[int, str] = {}
    for block in blocks:
        page = int(block.metadata.get("page", 0))
        if page and block.text.strip():
            by_page[page] = "\n".join(filter(None, (by_page.get(page), block.text.strip())))
    if not by_page:
        return RpgDoc(text="")

    page_numbers = sorted(by_page)
    first_page = min((e.page for e in outline), default=min(page_numbers))
    first_page = max(min(page_numbers), first_page)
    repeated = _repeated_edge_lines(blocks, first_page)

    parts: list[str] = []
    page_starts: list[int] = []
    pages: list[int] = []
    page_labels: list[str] = []
    offset = 0
    for physical_page in range(first_page, max(page_numbers) + 1):
        raw = by_page.get(physical_page)
        if not raw:
            continue
        cleaned, label = _clean_page(raw, physical_page, repeated)
        if not cleaned:
            continue
        page_starts.append(offset)
        pages.append(physical_page)
        page_labels.append(label)
        parts.append(cleaned)
        offset += len(cleaned) + 1

    doc = RpgDoc(
        text="\n".join(parts),
        page_starts=page_starts,
        pages=pages,
        page_labels=page_labels,
    )
    doc.sections = _sections_from_outline(doc, outline, source)
    doc.entries = _collect_entries(doc)
    return doc


def _sections_from_outline(doc: RpgDoc, outline: list[OutlineEntry], source: str) -> list[Section]:
    """Zamienia outline na sekcje z offsetami; bez outline dzieli po stronach."""
    if not outline:
        sections = []
        for i, (start, page, label) in enumerate(
            zip(doc.page_starts, doc.pages, doc.page_labels, strict=True)
        ):
            end = doc.page_starts[i + 1] if i + 1 < len(doc.page_starts) else len(doc.text)
            sections.append(
                Section(
                    title=f"s. {label}",
                    chapter=None,
                    level=0,
                    start=start,
                    end=end,
                    page=page,
                    page_label=label,
                    path=f"p{page}",
                )
            )
        return sections

    sections: list[Section] = []
    stack: list[OutlineEntry] = []
    previous_start = 0
    for entry in outline:
        page_index = bisect.bisect_left(doc.pages, entry.page)
        if page_index >= len(doc.pages):
            continue
        page_start = doc.page_starts[page_index]
        page_end = (
            doc.page_starts[page_index + 1]
            if page_index + 1 < len(doc.page_starts)
            else len(doc.text)
        )
        search_from = max(page_start, previous_start)
        pattern = re.compile(r"\s+".join(map(re.escape, entry.title.split())), re.I)
        match = pattern.search(doc.text, search_from, page_end) if search_from <= page_end else None
        start = match.start() if match else page_start
        start = max(start, previous_start)
        previous_start = start

        while stack and stack[-1].level >= entry.level:
            stack.pop()
        chapter = stack[0].title if stack else None
        path_parts = [item.title for item in stack] + [entry.title]
        page, label = doc.page_at(start)
        sections.append(
            Section(
                title=entry.title,
                chapter=chapter,
                level=entry.level,
                start=start,
                end=len(doc.text),
                page=page,
                page_label=label,
                path="/".join(_slug(part) for part in path_parts),
            )
        )
        stack.append(entry)

    for i, section in enumerate(sections):
        section.end = sections[i + 1].start if i + 1 < len(sections) else len(doc.text)
    return sections


def _slug(text: str) -> str:
    """Czytelny slug do source_ids (bez znaków specjalnych, max 90 znaków).

    Ukośnik w tytule (np. „Force Push/Pull”) zamieniamy na myślnik, bo `/` rozdziela
    poziomy ścieżki w `source_ids`.
    """
    text = text.replace("/", "-")
    slug = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
    slug = re.sub(r"[-\s_]+", "-", slug).strip("-/")
    return slug[:90].strip("-/")


# --------------------------------------------------------------------------- wpisy


_SPELL = re.compile(
    r"^(?P<title>[A-Z][A-Za-z0-9'’\-/& ]{2,45})\n"
    r"(?P<kind>\d(?:st|nd|rd|th)-level [a-z -]+|[A-Za-z]+ cantrip|At-will [a-z -]+)\n"
    r"Casting Time:",
    re.MULTILINE,
)
_MONSTER = re.compile(
    r"^(?P<title>[A-Z][A-Za-z0-9'’\- ]{2,45})\n"
    r"(?P<subtitle>[^\n]{3,90})\n"
    r"Armor Class \d+",
    re.MULTILINE,
)


def _find_entries(doc: RpgDoc, section: Section) -> list[Entry]:
    """Znajduje zaklęcia/moce i stat blocki w sekcji; reszta zostaje zwykłym tekstem."""
    body = doc.text[section.start : section.end]
    found: list[tuple[int, str, str]] = []  # (offset w body, tytuł, rodzaj)
    for match in _SPELL.finditer(body):
        kind = "power" if "power" in match.group("kind").lower() else "spell"
        found.append((match.start(), match.group("title").strip(), kind))
    if not found:
        for match in _MONSTER.finditer(body):
            found.append((match.start(), match.group("title").strip(), "monster"))

    found.sort()
    entries: list[Entry] = []
    used_starts: set[int] = set()
    for index, (offset, title, kind) in enumerate(found):
        if offset in used_starts:
            continue
        used_starts.add(offset)
        next_offset = found[index + 1][0] if index + 1 < len(found) else len(body)
        start = section.start + offset
        end = section.start + next_offset
        if end - start < 40:
            continue
        page, label = doc.page_at(start)
        entries.append(
            Entry(
                title=title,
                kind=kind,
                start=start,
                end=end,
                page=page,
                page_label=label,
                path=f"{section.path}/{_slug(title)}",
            )
        )
    for i, entry in enumerate(entries):
        entry.end = entries[i + 1].start if i + 1 < len(entries) else section.end
    return entries


def _collect_entries(doc: RpgDoc) -> list[Entry]:
    """Wpisy ze wszystkich sekcji – używane też przez `span_chunks` do mapowania source_ids."""
    entries: list[Entry] = []
    for section in doc.sections:
        entries += _find_entries(doc, section)
    return entries


# --------------------------------------------------------------------------- chunking


def _strip_heading(text: str, title: str) -> str:
    """Usuwa z początku tekstu powtórzony nagłówek sekcji/wpisu."""
    pattern = re.compile(r"^\s*" + r"\s+".join(map(re.escape, title.split())) + r"\s*", re.I)
    return pattern.sub("", text, count=1).lstrip("\n ")


def _context_header(meta: dict[str, Any], section: Section, entry: Entry | None = None) -> str:
    """Nagłówek doklejany do każdego chunka: książka › rozdział › sekcja › wpis."""
    parts = [str(meta.get("doc_label") or meta.get("source") or "")]
    if section.chapter:
        parts.append(section.chapter)
    parts.append(section.title)
    if entry:
        parts.append(entry.title)
    return " › ".join(part for part in parts if part)


def _split_body(body: str, size: int) -> list[tuple[str, int]]:
    """Dzieli treść i zwraca (kawałek, offset w `body`) – offset daje poprawną stronę chunka."""
    body = body.strip()
    if not body:
        return []
    pieces = [body] if len(body) <= size else recursive_split(body, size, overlap=0)
    result: list[tuple[str, int]] = []
    cursor = 0
    for piece in pieces:
        if not piece.strip():
            continue
        offset = body.find(piece, cursor)
        if offset < 0:
            offset = cursor
        result.append((piece, offset))
        cursor = offset + len(piece)
    return result


def structure_chunks(doc: RpgDoc, meta: dict[str, Any], size: int) -> list[Chunk]:
    """Jeden chunk = sekcja albo wpis; długie sekcje dzielimy rekurencyjnie."""
    chunks: list[Chunk] = []
    source = str(meta["source"])

    for section in doc.sections:
        if section.start >= section.end:
            continue
        section_id = f"{source}#{section.path}"
        entries = [e for e in doc.entries if section.start <= e.start < section.end]
        body_start = section.start
        if entries:
            intro = doc.text[section.start : entries[0].start]
            intro = _strip_heading(intro, section.title).strip()
            if len(intro) >= 40:
                chunks += _make_chunks(
                    doc,
                    meta,
                    intro,
                    section,
                    None,
                    [section_id],
                    section.start,
                    size,
                    len(chunks),
                )
            body_start = entries[0].start

        if not entries:
            body = _strip_heading(doc.text[body_start : section.end], section.title)
            if len(body.strip()) < 20:
                continue
            chunks += _make_chunks(
                doc,
                meta,
                body,
                section,
                None,
                [section_id],
                body_start,
                size,
                len(chunks),
            )
            continue

        for entry in entries:
            body = _strip_heading(doc.text[entry.start : entry.end], entry.title)
            if len(body.strip()) < 20:
                continue
            ids = [f"{source}#{entry.path}", section_id]
            chunks += _make_chunks(
                doc,
                meta,
                body,
                section,
                entry,
                ids,
                entry.start,
                size,
                len(chunks),
            )
    return chunks


def _make_chunks(
    doc: RpgDoc,
    meta: dict[str, Any],
    body: str,
    section: Section,
    entry: Entry | None,
    source_ids: list[str],
    body_start: int,
    size: int,
    start_index: int,
) -> list[Chunk]:
    """Buduje chunki z jednym zestawem metadanych (wspólne dla sekcji i wpisów)."""
    header = _context_header(meta, section, entry)
    chunks = []
    for piece, offset in _split_body(body, size):
        page, page_label = doc.page_at(body_start + offset)
        chunks.append(
            Chunk(
                text=f"{header}\n{piece}".strip(),
                metadata={
                    **meta,
                    "chunking": "structure",
                    "chunk_index": start_index + len(chunks),
                    "chapter": section.chapter,
                    "section": section.title,
                    "entry": entry.title if entry else None,
                    "content_type": entry.kind if entry else "section",
                    "page": page,
                    "page_label": page_label,
                    "source_ids": source_ids,
                    "section_id": f"{meta['source']}#{section.path}",
                },
            )
        )
    return chunks


def _locate(doc: RpgDoc, offset: int) -> tuple[Section | None, Entry | None]:
    """Najgłębsza sekcja i wpis zawierające offset (do metadanych chunków fixed/recursive)."""
    section = next(
        (s for s in reversed(doc.sections) if s.start <= offset < s.end),
        None,
    )
    entry = next(
        (e for e in reversed(doc.entries) if e.start <= offset < e.end),
        None,
    )
    return section, entry


def span_chunks(doc: RpgDoc, meta: dict[str, Any], texts: list[str], strategy: str) -> list[Chunk]:
    """Chunki fixed/recursive + `source_ids` jednostek, na które chunk zachodzi."""
    source = str(meta["source"])
    units = [(s.start, s.end, f"{source}#{s.path}") for s in doc.sections]
    units += [(e.start, e.end, f"{source}#{e.path}") for e in doc.entries]
    cursor = 0
    chunks: list[Chunk] = []
    for text in texts:
        if not text.strip():
            continue
        start = doc.text.find(text[:80], cursor)
        if start < 0:
            start = doc.text.find(text[:40], cursor)
        if start < 0:
            start = cursor
        cursor = start + 1
        end = start + len(text)
        ids: list[str] = []
        for unit_start, unit_end, unit_id in units:
            overlap = min(end, unit_end) - max(start, unit_start)
            if overlap >= min(20, max(1, unit_end - unit_start)):
                ids.append(unit_id)
        page, label = doc.page_at(start)
        if not ids:
            ids = [f"{source}#p{page}"]
        section, entry = _locate(doc, start)
        section_id = f"{source}#{section.path}" if section else None
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    **meta,
                    "chunking": strategy,
                    "chunk_index": len(chunks),
                    "page": page,
                    "page_label": label,
                    "chapter": section.chapter if section else None,
                    "section": section.title if section else None,
                    "entry": entry.title if entry else None,
                    "content_type": entry.kind if entry else "section",
                    "section_id": section_id,
                    "source_ids": list(dict.fromkeys(ids)),
                },
            )
        )
    return chunks


def chunk_rpg(
    blocks: list[Block],
    meta: dict[str, Any],
    strategy: str,
    size: int,
    overlap: int,
    outline: list[OutlineEntry] | None = None,
) -> list[Chunk]:
    """Punkt wejścia dla `ingest`: normalizowane bloki → chunki z metadanymi RPG."""
    doc = build_stream(blocks, outline or [], str(meta["source"]))
    if not doc.text:
        return []
    if strategy == "structure":
        return structure_chunks(doc, meta, size=int(size * STRUCTURE_SIZE_FACTOR))
    split = fixed_size if strategy == "fixed" else recursive_split
    texts = split(doc.text, size, overlap)
    return span_chunks(doc, meta, texts, strategy)


def citation(metadata: dict[str, Any]) -> str | None:
    """Krótkie cytowanie do tekstu odpowiedzi, np. „PHB, s. 82”."""
    label = metadata.get("short_label") or metadata.get("doc_label") or metadata.get("source")
    page = metadata.get("page_label") or metadata.get("page")
    if not label or not page:
        return None
    return f"{label}, s. {page}"
