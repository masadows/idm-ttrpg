import pytest

from rag.eval import retrieved_sources, recall_at_k, reciprocal_rank, to_level
from rag.retrieve import Hit


def test_recall_at_k():
    retrieved = [["a"], ["b", "c"], ["d"]]
    assert recall_at_k(retrieved, ["a", "d"], k=1) == 0.5
    assert recall_at_k(retrieved, ["a", "d"], k=3) == 1.0
    assert recall_at_k(retrieved, ["c"], k=2) == 1.0


def test_reciprocal_rank():
    retrieved = [["x"], ["y"], ["a"]]
    assert reciprocal_rank(retrieved, ["a"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["zzz"]) == 0.0
    assert reciprocal_rank(retrieved, ["y", "a"]) == 0.5


def test_to_level_collapses_entry_to_section():
    ids = ["phb.pdf#chapter-1/spells/fireball", "phb.pdf#chapter-1/spells/shield"]
    assert to_level(ids, "section") == ["phb.pdf#chapter-1/spells"]
    assert to_level(ids, "source") == ids


def test_to_level_keeps_page_and_top_level_ids():
    assert to_level(["x.pdf#p3", "x.pdf#chapter-1"], "section") == [
        "x.pdf#p3",
        "x.pdf#chapter-1",
    ]


def test_retrieved_sources_groups_source_ids_per_chunk():
    hits = [
        Hit(
            text="Pierwszy chunk.",
            score=0.8,
            metadata={"source_ids": ["rules.md#combat", "rules.md#chapter"], "chunk_index": 3},
        ),
        Hit(
            text="Drugi chunk.",
            score=0.7,
            metadata={"source_ids": ["rules.md#combat"], "chunk_index": 3},
        ),
        Hit(
            text="Trzeci chunk.",
            score=0.6,
            metadata={"source_ids": ["rules.md#combat"], "chunk_index": 8},
        ),
    ]

    assert retrieved_sources(hits) == [
        {"source_ids": ["rules.md#combat", "rules.md#chapter"], "chunk_index": 3},
        {"source_ids": ["rules.md#combat"], "chunk_index": 8},
    ]
