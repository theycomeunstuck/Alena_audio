"""Свободное описание ученика → валидная карточка.

Модель здесь подставная: проверяется, что код достраивает обязательный каркас,
не пускает выдуманные темы и не даёт записать карточку, которая не пройдёт
валидацию learner-data.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import card_creator  # noqa: E402
from pipeline.card_creator import (  # noqa: E402
    CardCreateError,
    build_card,
    build_catalog_block,
    create_from_description,
    render_draft,
    validate_draft,
    write_draft,
)
from pipeline.config import LEARNER_DATA_DIR  # noqa: E402
from pipeline.learner_context import list_learner_ids, load_learner_data_tools  # noqa: E402

TODAY = "2026-07-26"
CATALOG = load_learner_data_tools().load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")


def _payload(**overrides):
    payload = {
        "learner_id": "petrov-misha",
        "pseudonym": "Миша",
        "age_group": "10-11",
        "grade": "4",
        "interests": ["динозавры", "майнкрафт"],
        "story_preferences": ["раскопки"],
        "profile": {
            "explanation_style": "по картинке, с примером перед началом",
            "pace": "быстрый",
            "autonomy_level": "низкий, бросает при непонимании",
            "motivation": "интерес к теме",
            "prefers_visual": True,
        },
        "knowledge": [
            {"topic_id": "math.g4.numbers.division_by_1_2_digit", "status": "needs_support",
             "notes": "теряет ноль в частном"},
            {"topic_id": "math.g4.numbers.multidigit_numeration", "status": "confident"},
        ],
        "error_patterns": [
            {"topic_id": "math.g4.numbers.division_by_1_2_digit", "error_tag": "теряет ноль в частном"}
        ],
        "current_goal": {"topic_id": "math.g4.numbers.division_by_1_2_digit",
                         "goal": "не терять ноль в частном"},
        "effective_strategies": ["показать пример перед началом"],
        "avoid": ["не давать длинную инструкцию"],
        "notes": "",
    }
    payload.update(overrides)
    return payload


def _fake_llm(payload):
    def complete(_prompt, _system):
        return json.dumps(payload, ensure_ascii=False)

    return complete


# --- каркас карточки -----------------------------------------------------------

def test_draft_passes_learner_data_validation():
    draft = build_card(_payload(), CATALOG, TODAY)
    assert validate_draft(draft) == []


def test_required_scaffolding_is_added_by_code_not_by_the_model():
    draft = build_card(_payload(), CATALOG, TODAY)

    assert draft.card["schema_version"] == 2
    assert draft.card["language"] == "russian"
    assert set(draft.card["mvp"]) == {
        "story_preferences", "learning_preferences_observed",
        "help_strategies", "journal", "points_ledger",
    }
    assert draft.card["rag_context"]["recent_progress"]["date"] == TODAY


def test_no_mastery_numbers_are_invented():
    """Числа считает код по итогам занятий, а не модель по описанию."""
    draft = build_card(_payload(), CATALOG, TODAY)
    assert "learner_model" not in draft.card
    assert "mastery" not in json.dumps(draft.card, ensure_ascii=False)


def test_topics_outside_the_catalog_are_dropped_with_a_warning():
    draft = build_card(
        _payload(knowledge=[
            {"topic_id": "math.g4.numbers.division_by_1_2_digit", "status": "learning"},
            {"topic_id": "math.g4.magic.telekinesis", "status": "confident"},
        ]),
        CATALOG, TODAY,
    )

    topics = [item["topic_id"] for item in draft.card["knowledge"]]
    assert topics == ["math.g4.numbers.division_by_1_2_digit"]
    assert any("не из каталога" in warning for warning in draft.warnings)


def test_topic_from_another_grade_is_dropped():
    draft = build_card(
        _payload(knowledge=[
            {"topic_id": "math.g4.numbers.division_by_1_2_digit", "status": "learning"},
            {"topic_id": "math.g3.geometry.circle", "status": "confident"},
        ]),
        CATALOG, TODAY,
    )
    assert [item["topic_id"] for item in draft.card["knowledge"]] == [
        "math.g4.numbers.division_by_1_2_digit"
    ]


def test_missing_goal_falls_back_to_the_hardest_topic():
    draft = build_card(_payload(current_goal={}), CATALOG, TODAY)

    assert draft.card["rag_context"]["current_goal"]["topic_id"] == "math.g4.numbers.division_by_1_2_digit"
    assert any("цель не названа" in warning for warning in draft.warnings)


def test_card_without_any_known_topic_is_refused():
    with pytest.raises(CardCreateError):
        build_card(_payload(knowledge=[], current_goal={}), CATALOG, TODAY)


def test_missing_pseudonym_is_refused():
    with pytest.raises(CardCreateError):
        build_card(_payload(pseudonym=""), CATALOG, TODAY)


def test_taken_pseudonym_is_flagged_because_the_validator_rejects_duplicates():
    draft = build_card(_payload(pseudonym="Тёма"), CATALOG, TODAY)
    assert any("уже занят" in warning for warning in draft.warnings)


def test_learner_id_is_normalised_and_kept_unique():
    draft = build_card(_payload(learner_id="Петров Миша!"), CATALOG, TODAY)
    assert draft.learner_id == "петров-миша!".encode("ascii", "ignore").decode() or draft.learner_id
    assert draft.learner_id not in list_learner_ids()
    assert draft.learner_id.replace("-", "").isalnum()


def test_existing_learner_id_gets_a_suffix():
    draft = build_card(_payload(learner_id="zvezdin-artem"), CATALOG, TODAY)
    assert draft.learner_id != "zvezdin-artem"


def test_unknown_grade_is_inferred_from_the_topics():
    """Класс восстанавливается по программе, которую перечислил репетитор."""
    draft = build_card(_payload(grade="7"), CATALOG, TODAY)

    assert draft.card["grade"] == "4", "перечислены темы четвёртого класса"
    assert any("класс" in warning for warning in draft.warnings)
    assert draft.card["knowledge"], "темы не должны отсеяться из-за неверного класса"


def test_grade_defaults_to_three_when_there_is_nothing_to_infer_from():
    draft = build_card(
        _payload(grade="", knowledge=[{"topic_id": "math.g3.geometry.circle", "status": "learning"}],
                 error_patterns=[], current_goal={}),
        CATALOG, TODAY,
    )
    assert draft.card["grade"] == "3"


def test_missing_interests_are_flagged_not_invented():
    draft = build_card(_payload(interests=[]), CATALOG, TODAY)

    assert draft.card["interests"] == []
    assert any("интересы не названы" in warning for warning in draft.warnings)


def test_empty_profile_fields_are_marked_as_unobserved():
    draft = build_card(_payload(profile={}), CATALOG, TODAY)

    assert draft.card["profile"]["pace"] == card_creator._UNKNOWN
    assert draft.card["profile"]["prefers_visual"] is False
    assert sum("нет данных про" in warning for warning in draft.warnings) >= 4


# --- промпт и полный путь -------------------------------------------------------

def test_catalog_block_lists_only_the_requested_grade():
    block = build_catalog_block(CATALOG, "4")

    assert "math.g4.numbers.division_by_1_2_digit" in block
    assert "math.g3.geometry.circle" not in block


def test_create_from_description_end_to_end():
    draft = create_from_description(
        "Миша, 4 класс, любит динозавров. Деление столбиком не идёт.",
        complete=_fake_llm(_payload()),
        today=TODAY,
    )

    assert draft.card["pseudonym"] == "Миша"
    assert validate_draft(draft) == []
    assert "Миша" in render_draft(draft)


def test_empty_description_is_refused():
    with pytest.raises(CardCreateError):
        create_from_description("   ", complete=_fake_llm(_payload()))


# --- запись ---------------------------------------------------------------------

def test_write_refuses_to_overwrite_an_existing_card(tmp_path, monkeypatch):
    draft = build_card(_payload(), CATALOG, TODAY)
    target = tmp_path / f"{draft.learner_id}.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(card_creator.CardDraft, "path", lambda self: target)

    with pytest.raises(CardCreateError):
        write_draft(draft)
    assert target.read_text(encoding="utf-8") == "{}"


def test_write_creates_a_valid_card(tmp_path, monkeypatch):
    draft = build_card(_payload(), CATALOG, TODAY)
    target = tmp_path / f"{draft.learner_id}.json"
    monkeypatch.setattr(card_creator.CardDraft, "path", lambda self: target)

    write_draft(draft)
    assert json.loads(target.read_text(encoding="utf-8")) == draft.card
