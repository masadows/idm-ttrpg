import pytest

from rag.eval import (
    apply_human_grades,
    recall_at_k,
    reciprocal_rank,
    retrieved_sources,
    to_level,
    to_markdown,
)
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


def test_apply_human_grades_adds_review_statistics():
    results = {
        "rows": [
            {"human_label": "correct", "tags": ["dnd", "combat"]},
            {"human_label": "refusal_ok", "tags": ["dnd"]},
            {"human_label": "incorrect", "tags": ["combat"]},
            {"tags": ["dnd"]},
        ],
        "summary": {},
    }

    apply_human_grades(results)

    assert results["summary"]["human_acc"] == pytest.approx(0.6667)
    review = results["summary"]["human_review"]
    assert review["graded"] == 3
    assert review["ungraded"] == 1
    assert review["counts"] == {"correct": 1, "refusal_ok": 1, "incorrect": 1, "refusal_bad": 0}
    assert review["percentages"]["correct"] == pytest.approx(1 / 3, abs=1e-4)
    assert review["percentages"]["refusal_ok"] == pytest.approx(1 / 3, abs=1e-4)
    assert review["percentages"]["incorrect"] == pytest.approx(1 / 3, abs=1e-4)
    assert review["percentages"]["refusal_bad"] == 0.0
    assert "human_acc" not in results["rows"][0]
    assert review["by_tag"]["dnd"]["graded"] == 2
    assert review["by_tag"]["dnd"]["ungraded"] == 1
    assert review["by_tag"]["dnd"]["counts"] == {
        "correct": 1,
        "refusal_ok": 1,
        "incorrect": 0,
        "refusal_bad": 0,
    }
    assert review["by_tag"]["combat"]["counts"]["incorrect"] == 1
    assert review["by_tag"]["combat"]["counts"]["correct"] == 0


def test_markdown_human_acc_column_uses_human_score():
    results = {
        "rows": [{"id": "q1", "question": "Pytanie", "human_score": 1.0}],
        "summary": {"questions": 1, "human_acc": 1.0},
    }

    markdown = to_markdown(results, "Wyniki")

    assert "| id | pytanie | human_score |" in markdown
    assert "human_acc" not in markdown
    assert "| q1 | Pytanie | 1.00 |" in markdown


def test_markdown_contains_manual_review_totals():
    results = {
        "rows": [],
        "summary": {
            "questions": 2,
            "human_acc": 0.5,
            "human_review": {
                "graded": 2,
                "ungraded": 0,
                "human_acc": 0.5,
                "counts": {"correct": 1, "refusal_ok": 0, "incorrect": 1, "refusal_bad": 0},
                "by_tag": {},
            },
        },
    }

    markdown = to_markdown(results, "Wyniki")

    assert "| **suma** | **2** | **0** | **1** | **1** | **0** | **0** | **0.50** |" in markdown
