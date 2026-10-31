"""Testy formatu złotych zbiorów – bez sieci i bez Qdranta."""

from pathlib import Path

import pytest

from rag.eval import load_golden

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["golden_sample.jsonl", "golden_rpg.jsonl"])
def test_golden_jsonl_parses_and_has_section_ids(name: str) -> None:
    items = load_golden(ROOT / "eval" / name)
    assert items
    assert len({item.id for item in items}) == len(items)
    for item in items:
        assert item.question.strip()
        assert item.answer.strip()
        if item.source_ids:
            assert item.section_ids, f"{item.id}: brak section_ids dla pytania z odpowiedzią"


def test_rpg_golden_covers_all_books() -> None:
    items = load_golden(ROOT / "eval" / "golden_rpg.jsonl")
    books = {source_id.split("#", 1)[0] for item in items for source_id in item.source_ids}
    assert {
        "PlayerDnDBasicRules.pdf",
        "DMBasicRulesv.pdf",
        "Mythras - 2E (v3.1).pdf",
        "SW5e - Player's Handbook.pdf",
    } <= books
