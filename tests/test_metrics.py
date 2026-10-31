import pytest

from rag.eval import recall_at_k, reciprocal_rank, to_level


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
