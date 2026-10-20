"""Literal grounding and disclosure regressions without model or database calls."""
from copy import deepcopy

import pytest

from local_app.checked_division_context import (
    CHAT_CONTEXT_INSTRUCTIONS, DIVISION_TOPIC, VERSION, checked_division_context,
    extract_literal_division_pair, validate_checked_division_context,
)


TASK = "Попробуй вычислить 1224 : 4. Напиши только ответ."


def context(text="хз", *, anchors=(TASK,), subject="math", topic=DIVISION_TOPIC):
    return checked_division_context(text, subject, topic, anchors=anchors)


def test_first_group_is_twelve_hundreds_without_quotient_or_assessment():
    result = context()
    assert result == {
        "version": VERSION, "dividend": 1224, "divisor": 4,
        "first_group": {"digits": "12", "place_value": 100, "place_name": "сотен",
                        "amount": 12, "real_value": 1200},
        "complete_answer_included": False, "student_work_assessed": False,
    }
    assert not {"quotient", "quotient_digit", "product_check", "long_division_steps", "quotient_places"} & result.keys()


@pytest.mark.parametrize("current", ["хз", "ага", "306", "3", "ну давай", "Подскажи только первый шаг."])
def test_short_reply_is_unassessed_and_does_not_choose_a_new_phase(current):
    result = context(current)
    assert result == context("хз")
    assert result["student_work_assessed"] is False
    assert result["complete_answer_included"] is False


@pytest.mark.parametrize("text", ["1224 : 4", "Вычисли 1224 ÷ 4.", "Раздели 1224 на 4.",
                                  "Деление 1224 на 4", "Подели 1224 на 4.", "Проверь 1224 : 4 = 305."])
def test_supported_literal_syntax_preserves_operands_without_verifying_asserted_answer(text):
    assert extract_literal_division_pair(text) == (1224, 4)
    assert context(text, anchors=()) == context()


@pytest.mark.parametrize("dividend,divisor,digits,place,name,amount,value", [
    (2842, 14, "28", 100, "сотен", 28, 2800),
    (1005, 5, "10", 100, "сотен", 10, 1000),
    (3456, 12, "34", 100, "сотен", 34, 3400),
    (36, 4, "36", 1, "единиц", 36, 36),
    (840, 4, "8", 100, "сотен", 8, 800),
    (120000, 12, "12", 10000, "десятков тысяч", 12, 120000),
])
def test_group_unit_comes_from_consumed_positions(dividend, divisor, digits, place, name, amount, value):
    result = context(f"Вычисли {dividend} : {divisor}", anchors=())
    assert result["first_group"] == {
        "digits": digits, "place_value": place, "place_name": name, "amount": amount, "real_value": value}


def test_current_literal_wins_over_conflicting_older_active_anchors():
    result = context("А теперь 2842 : 14.", anchors=(TASK, "Раньше было 3456 : 12."))
    assert (result["dividend"], result["divisor"]) == (2842, 14)
    assert context("1224 : 4", anchors=({"bad": "older anchor"},)) == context()


def test_identical_active_pairs_are_allowed_but_conflicting_pairs_are_not():
    assert context(anchors=(TASK, "Раздели 1224 на 4.")) == context()
    assert context(anchors=(TASK, "Теперь 2842 : 14.")) is None


def test_non_numeric_active_anchor_is_not_mined_or_invented_as_a_task():
    assert context(anchors=("Дай ещё пример на деление.", TASK)) == context()
    assert context(anchors=("Дай ещё пример на деление.",)) is None
    assert context(anchors=()) is None


@pytest.mark.parametrize("text", [
    "1224 : 0", "1224 : 100", "1000000 : 4", "-1224 : 4", "−1224 : 4", "1224 : -4",
    "1224.0 : 4", "1224 : 4,0", "1 224 : 4", "01224 : 4", "1224 : 04",
    "1224 : 4 + 5", "(1224 : 4) × 2", "2**3 : 4", "1224 : 4 = 306 + 1",
    "1224 : 4 плюс пять", "К 1224 : 4 прибавь пять", "1224 : 4 умножить на два",
    "1224 : 4 = 306,5", "1224 : 4 = 0306", "1224 : 4 = 10000000 + 3",
    "12e4 : 2", "1_224 : 4", "1224 : 4e2", "1224 : 4_0",
    "x1224 : 4", "я1224 : 4", "1224 : 4x", "1224 : 4я", "1224 : 4.0x",
    "Раздели 1224 на 4e2", "١٢٢٤ : ٤",
    "1224 : 4 и 2842 : 14", "1224 : 4 и 1224 : 4", "1224 // 4", "1224 / / 4",
    "0 : 4", "5 : 12", "1237 : 12", "1224 : 4; получилось 3,5",
    "Чему равен числитель дроби 12/4?", "Сократи 12/4.", "Раздели 1224 на 0.",
    "Раздели 1224 на 4,5.", "Раздели 1224 на 100.", "Раздели на 3.",
])
def test_invalid_current_expression_never_falls_back_to_old_task(text):
    assert context(text) is None
    assert extract_literal_division_pair(text, division_context=True) is None


def test_invalid_mathematical_anchor_is_not_silently_ignored():
    assert context(anchors=("Вычисли 1224 : 4 + 5.", TASK)) is None


@pytest.mark.parametrize("text", [
    "Давай тот же пример, но делитель теперь 3.",
    "Давай тот же пример, но делитель теперь три.",
    "Теперь делимое 3636.", "Замени делимое на другое.", "Делитель три.",
])
def test_operand_reference_without_complete_current_operation_blocks_older_fallback(text):
    assert context(text) is None


@pytest.mark.parametrize("text", [
    "3636 : 12. Сначала взял 36, написал 3, вычел 36. Потом снёс 3.",
    "В 3636 : 12 уже получил 3, дальше частичная группа 3, а делитель 12.",
    "3636 : 12. В ответе написал 300 и остановился.",
    "3636 : 12. Делитель должен быть 3.",
])
def test_narrative_integers_do_not_claim_phase_or_constraint_interpretation(text):
    result = context(text, anchors=())
    assert extract_literal_division_pair(text, division_context=True) == (3636, 12)
    assert result == context("3636 : 12", anchors=())
    assert result["student_work_assessed"] is False
    assert result["complete_answer_included"] is False


def test_literal_anchor_with_narrative_is_grounded_without_treating_reported_work_as_true():
    text = "3636 : 12. Сначала взял 36, написал 3, вычел 36. Потом снёс 3."
    assert context("хз", anchors=(text,)) == context("3636 : 12", anchors=())


def test_division_topic_and_subject_boundaries_do_not_turn_generic_slash_into_a_fraction_tool():
    assert context("12/4", anchors=(), topic="math.g4.fractions") is None
    assert context("12/4", anchors=(), topic=None) is None
    assert context("Раздели 12/4", anchors=(), topic=None) is not None
    assert context("12/4", anchors=()) is not None
    assert context("12 : 4", anchors=(), subject="python") is None
    assert context("12 : 4", anchors=(), subject=None, topic=None) is None
    assert context("12 : 4", anchors=(), subject=None) is not None
    assert context("12 : 4", anchors=(), topic=None) is not None
    assert context("12 : 4", anchors=(), topic="math.g3.numbers.non_table_mult_div") is not None


@pytest.mark.parametrize("text", ["12/4", "Запиши 12/4"])
def test_shared_literal_parser_requires_slash_disambiguation(text):
    assert extract_literal_division_pair(text) is None
    assert extract_literal_division_pair(text, division_context=True) == (12, 4)


@pytest.mark.parametrize("kwargs", [
    {"anchors": TASK}, {"anchors": ({"text": TASK},)}, {"anchors": (TASK,) * 21},
    {"anchors": ("а" * 4001,)}, {"text": "а" * 4001}, {"text": None}, {"subject": False},
])
def test_malformed_or_oversized_input_fails_closed(kwargs):
    assert context(**kwargs) is None


def test_validator_recomputes_and_returns_a_distinct_canonical_mapping():
    result = context()
    validated = validate_checked_division_context(result)
    assert validated == result and validated is not result
    assert validated["first_group"] is not result["first_group"]
    assert validate_checked_division_context(context("2842 : 14", anchors=()))["first_group"]["real_value"] == 2800


@pytest.mark.parametrize("field,value", [
    ("version", "old"), ("dividend", True), ("divisor", 4.0),
    ("complete_answer_included", True), ("complete_answer_included", 0),
    ("student_work_assessed", True), ("student_work_assessed", 0),
])
def test_validator_rejects_invalid_top_level_types_and_disclosure(field, value):
    result = context()
    result[field] = value
    with pytest.raises(ValueError):
        validate_checked_division_context(result)


@pytest.mark.parametrize("field,value", [
    ("digits", "1"), ("place_value", 1000), ("place_name", "тысяч"),
    ("amount", 1), ("real_value", 1000), ("amount", True), ("real_value", 1200.0),
    ("digits", 12), ("place_name", None),
])
def test_validator_rejects_false_derived_facts_and_coercible_types(field, value):
    result = context()
    result["first_group"][field] = value
    with pytest.raises(ValueError):
        validate_checked_division_context(result)


@pytest.mark.parametrize("edit", [
    lambda value: value.update(quotient=306),
    lambda value: value["first_group"].update(quotient_digit=3),
    lambda value: value.pop("student_work_assessed"),
    lambda value: value["first_group"].pop("place_name"),
    lambda value: value.update(divisor=0),
    lambda value: value.update(divisor=100),
    lambda value: value.update(dividend=0),
    lambda value: value.update(dividend=1000000),
    lambda value: value.update(dividend=1237, divisor=12),
])
def test_validator_rejects_unknown_missing_or_unsupported_fields(edit):
    result = deepcopy(context())
    edit(result)
    with pytest.raises(ValueError):
        validate_checked_division_context(result)


def test_instructions_keep_numeric_grounding_separate_from_dialogue_phase_and_evidence():
    assert "не выбор текущего этапа" in CHAT_CONTEXT_INSTRUCTIONS
    assert "не повторяй уже пройденный шаг" in CHAT_CONTEXT_INSTRUCTIONS
    assert "без раскрытия частного" in CHAT_CONTEXT_INSTRUCTIONS
    assert "ответил самостоятельно" in CHAT_CONTEXT_INSTRUCTIONS
    assert "в этом разряде нужен ноль" in CHAT_CONTEXT_INSTRUCTIONS
    assert "не занимай из соседнего разряда" in CHAT_CONTEXT_INSTRUCTIONS
    assert "Уважай просьбу о паузе" in CHAT_CONTEXT_INSTRUCTIONS
    assert "1200" not in CHAT_CONTEXT_INSTRUCTIONS and "digits='12'" not in CHAT_CONTEXT_INSTRUCTIONS
