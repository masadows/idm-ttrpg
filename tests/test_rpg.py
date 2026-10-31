"""Testy parsera podręczników RPG na syntetycznych stronach (bez PDF-ów i sieci)."""

from rag.chunking import Block
from rag.rpg import (
    OutlineEntry,
    build_stream,
    chunk_rpg,
    citation,
    parse_toc,
    span_chunks,
    structure_chunks,
)

PAGES = [
    Block(
        "1\nChapter 1: Start\nTo jest krótki wstęp do rozdziału.",
        {"source": "test.pdf", "page": 1},
    ),
    Block(
        "2\nSpells\nFireball\n3rd-level evocation\nCasting Time: 1 action\n"
        "Range: 150 feet\nComponents: V, S\nDuration: Instantaneous\n"
        "A bright streak flashes from your pointing finger.",
        {"source": "test.pdf", "page": 2},
    ),
    Block(
        "3\nMonsters\nGoblin\nSmall humanoid (goblinoid), neutral evil\n"
        "Armor Class 15 (leather armor, shield)\nHit Points 7 (2d6)\nSpeed 30 ft.\n"
        "STR 8 (-1) DEX 14 (+2) CON 10 (+0) INT 10 (+0) WIS 8 (-1) CHA 8 (-1)",
        {"source": "test.pdf", "page": 3},
    ),
]

OUTLINE = [
    OutlineEntry(level=0, title="Chapter 1: Start", page=1),
    OutlineEntry(level=1, title="Spells", page=2),
    OutlineEntry(level=1, title="Monsters", page=3),
]

META = {
    "source": "test.pdf",
    "doc_label": "Test RPG",
    "short_label": "TRPG",
    "system": "Test",
    "doc_type": "rpg",
}


def test_parse_toc_levels_and_ligature_repair():
    blocks = [
        Block(
            "TABLE OF CONTENTS\nIntroduction 1\nChapter 1: Start 2\n"
            " Spells 3\n  Con\x00rmation 4\nIndex 5",
            {"source": "test.pdf", "page": 1},
        ),
        Block("x", {"source": "test.pdf", "page": 5}),
    ]
    entries = parse_toc(blocks)
    assert [(e.level, e.title, e.page) for e in entries] == [
        (0, "Introduction", 1),
        (0, "Chapter 1: Start", 2),
        (1, "Spells", 3),
        (2, "Confirmation", 4),
        (0, "Index", 5),
    ]


def test_build_stream_strips_page_number_and_builds_sections():
    doc = build_stream(PAGES, OUTLINE, "test.pdf")
    assert [s.title for s in doc.sections] == ["Chapter 1: Start", "Spells", "Monsters"]
    assert doc.sections[0].chapter is None
    assert doc.sections[1].chapter == "Chapter 1: Start"
    assert doc.sections[1].path == "chapter-1-start/spells"
    assert "Fireball" in doc.text
    assert "\n1\n" not in doc.text


def test_structure_chunks_detect_spell_and_monster():
    doc = build_stream(PAGES, OUTLINE, "test.pdf")
    chunks = structure_chunks(doc, META, size=1000)
    by_entry = {c.metadata["entry"]: c for c in chunks if c.metadata.get("entry")}
    assert set(by_entry) == {"Fireball", "Goblin"}
    assert by_entry["Fireball"].metadata["content_type"] == "spell"
    assert by_entry["Goblin"].metadata["content_type"] == "monster"
    assert by_entry["Fireball"].metadata["source_ids"][0] == (
        "test.pdf#chapter-1-start/spells/fireball"
    )
    assert by_entry["Goblin"].metadata["source_ids"][0] == (
        "test.pdf#chapter-1-start/monsters/goblin"
    )
    assert by_entry["Fireball"].metadata["page_label"] == "2"
    assert by_entry["Fireball"].metadata["section_id"] == ("test.pdf#chapter-1-start/spells")


def test_span_chunks_assigns_entry_ids_to_small_chunks():
    chunks = chunk_rpg(PAGES, META, strategy="fixed", size=40, overlap=5, outline=OUTLINE)
    ids = {sid for chunk in chunks for sid in chunk.metadata["source_ids"]}
    assert "test.pdf#chapter-1-start/spells/fireball" in ids
    assert "test.pdf#chapter-1-start/monsters/goblin" in ids
    assert all(chunk.metadata["section_id"] for chunk in chunks)
    assert all(chunk.metadata["chunking"] == "fixed" for chunk in chunks)


def test_span_chunks_falls_back_to_page_id_for_tiny_fragment():
    doc = build_stream(PAGES, OUTLINE, "test.pdf")
    chunks = span_chunks(doc, META, ["x"], "fixed")
    assert chunks[0].metadata["source_ids"] == ["test.pdf#p1"]


def test_citation_uses_short_label_and_printed_page():
    assert citation({"short_label": "PHB", "page_label": "82", "page": 84}) == "PHB, s. 82"
    assert citation({"source": "x.pdf", "page": 3}) == "x.pdf, s. 3"


def test_summoned_stat_block_inside_powers_stays_in_power_entry():
    pages = [
        Block(
            "1\nTech Powers\nConstruct Droid\nAt-will tech power\nCasting Time: 1 action\n"
            "You construct a servant droid.\n"
            "SERVANT\nTiny construct, unaligned\nArmor Class 13\nHit Points 4 (1d4 + 2)",
            {"source": "sw5e.pdf", "page": 1},
        )
    ]
    outline = [OutlineEntry(level=0, title="Tech Powers", page=1)]
    doc = build_stream(pages, outline, "sw5e.pdf")
    assert [(e.title, e.kind) for e in doc.entries] == [("Construct Droid", "power")]
    assert "SERVANT" in doc.text[doc.entries[0].start : doc.entries[0].end]
