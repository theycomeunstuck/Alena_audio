"""ЗБР stage 4: the projection, the retrieval fallback and the generated card.

No services are required: the LLM is a fake and retrieval falls back to the
stdlib lexical scan of ``knowledge-data/``.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.learner_context import (  # noqa: E402
    LearnerContextError,
    load_task_design_context,
)
from pipeline.retrieval import retrieve_material  # noqa: E402
from pipeline.task_designer import (  # noqa: E402
    TaskDesignError,
    build_correction,
    build_task_prompt,
    design_task_card,
    parse_task_card,
    render_task_card,
)

LEARNER = "zvezdin-artem"
TOPIC = "math.g3.geometry.point_line_ray_segment"


def _model_reply(task_count=3, hint_count=2, topic_id=TOPIC):
    return {
        "topic_id": topic_id,
        "story": {"title": "Чертежи ракеты", "intro": "Тёма, вор спрятал чертежи ракеты. Проверь их."},
        "tasks": [
            {
                # Сюжет должен быть в самом задании, а не только во вступлении —
                # это проверяется отдельно, поэтому фикстура «хорошей» карточки
                # обязана упоминать интерес ребёнка в каждом задании.
                "statement": f"Задание {number}: начерти траекторию ракеты в виде луча от точки старта.",
                "expected_answer": "Луч из одной точки, второй конец не ставим.",
                "checks_error": "рисует у луча два конца",
                "hints": [
                    {"level": index, "type": "hint_question", "text": f"Подсказка {index}"}
                    for index in range(1, hint_count + 1)
                ],
                "materials": ["линейка"],
            }
            for number in range(1, task_count + 1)
        ],
        "reflection_question": "Что было самым простым?",
        "tutor_notes": "Смотреть, ставит ли второй конец.",
    }


def _fake_llm(payload):
    def complete(_prompt, _system):
        return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

    return complete


# --- projection ---------------------------------------------------------------

def test_context_carries_zpd_verdict_and_interests():
    context = load_task_design_context(LEARNER, TOPIC)

    assert context.topic_id == TOPIC
    assert context.zone == "zpd"
    assert context.difficulty == "medium"
    assert context.hint_depth == 2
    assert "космос" in context.interests
    assert "детектив" in context.story_preferences
    assert context.error_tags, "типичные ошибки по теме должны попадать в контекст"


def test_context_never_leaks_identity_journal_or_points():
    context = load_task_design_context(LEARNER, TOPIC)
    text = context.prompt_text

    assert "Звездин" not in text, "фамилия не должна попадать в промпт"
    assert "points_ledger" not in text and "Баллы" not in text
    assert "journal" not in text and "Знакомство" not in text
    assert len(text) < 4000, "проекция должна оставаться компактной"


def test_context_picks_target_from_policy_when_topic_is_omitted():
    context = load_task_design_context(LEARNER)
    assert context.topic_id == TOPIC


def test_unknown_topic_is_rejected():
    with pytest.raises(LearnerContextError):
        load_task_design_context(LEARNER, "math.g3.geometry.no_such_topic")


def test_mastered_topic_gets_harder_task_than_zone_topic():
    zone_topic = load_task_design_context(LEARNER, TOPIC)
    mastered_topic = load_task_design_context(LEARNER, "math.g3.quantities.length")

    assert mastered_topic.zone == "mastered"
    assert mastered_topic.difficulty == "hard"
    assert mastered_topic.hint_depth < zone_topic.hint_depth


# --- retrieval ----------------------------------------------------------------

def test_lexical_fallback_finds_on_topic_material_without_services():
    materials = retrieve_material("луч и отрезок", topic_id=TOPIC, grade="3", limit=3)

    assert materials, "локальный лексический поиск должен находить материал"
    assert all(material.topic_id == TOPIC for material in materials)
    assert materials[0].origin == "lexical"


def test_retrieval_prefers_the_requested_topic_over_other_grades():
    materials = retrieve_material("деление столбиком", topic_id=TOPIC, grade="3", limit=3)
    assert all(material.topic_id == TOPIC for material in materials)


# --- prompt -------------------------------------------------------------------

def test_prompt_contains_zone_material_and_explicit_counts():
    context = load_task_design_context(LEARNER, TOPIC)
    materials = retrieve_material(context.retrieval_query, topic_id=TOPIC, grade="3", limit=2)
    prompt = build_task_prompt(context, materials, 3)

    assert "ровно 3 задания" in prompt
    assert "ровно 2 ступени подсказок" in prompt
    assert "<curriculum_material>" in prompt
    assert TOPIC in prompt


# --- generated card -----------------------------------------------------------

def test_design_task_card_end_to_end_with_fake_model():
    card = design_task_card(LEARNER, TOPIC, count=3, complete=_fake_llm(_model_reply()))

    assert card.pseudonym == "Тёма"
    assert len(card.tasks) == 3
    assert all(len(task.hints) == 2 for task in card.tasks)
    assert card.sources, "должен быть указан источник материала"
    assert not card.warnings, card.warnings


def test_hint_ladder_order_is_reimposed():
    reply = _model_reply(hint_count=2)
    # The model returns the rungs in the wrong order and with the wrong types.
    reply["tasks"][0]["hints"] = [
        {"level": 2, "type": "joint", "text": "Первый шаг делаю я"},
        {"level": 1, "type": "visual", "text": "Посмотри на концы линии"},
    ]
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(reply, context, 3)

    hints = card.tasks[0].hints
    assert [hint.level for hint in hints] == [1, 2]
    assert [hint.type for hint in hints] == ["hint_question", "visual"]


def test_wrong_hint_count_is_repaired_and_reported():
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(_model_reply(hint_count=4), context, 3)

    assert all(len(task.hints) == 2 for task in card.tasks)
    assert any("подсказ" in warning for warning in card.warnings)


def test_extra_tasks_are_trimmed_and_reported():
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(_model_reply(task_count=5), context, 3)

    assert len(card.tasks) == 3
    assert any("заданий пришло 5" in warning for warning in card.warnings)


def test_wrong_topic_id_is_overridden_by_the_policy():
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(_model_reply(topic_id="math.g4.geometry.angles"), context, 3)

    assert card.topic_id == TOPIC
    assert any("topic_id" in warning for warning in card.warnings)


def test_internal_vocabulary_in_child_text_is_flagged():
    reply = _model_reply()
    reply["tasks"][0]["statement"] = "Тёма, твоя ЗБР сегодня — лучи. Начерти луч."
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(reply, context, 3)

    assert any("збр" in warning.lower() for warning in card.warnings)


def test_hint_that_gives_away_the_answer_is_flagged():
    reply = _model_reply()
    answer = reply["tasks"][0]["expected_answer"]
    reply["tasks"][0]["hints"][0]["text"] = f"Просто напиши: {answer}"
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(reply, context, 3)

    assert any("готовый ответ" in warning for warning in card.warnings)


def test_broken_arithmetic_in_the_answer_is_flagged():
    """Неверный ответ выглядит как верный — форму карточки он не нарушает."""
    reply = _model_reply()
    reply["tasks"][0]["expected_answer"] = "Считаем длину: 5 · 4 = 25 см."
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(reply, context, 3)

    assert any("проверьте счёт" in warning for warning in card.warnings), card.warnings


def test_story_without_interests_is_flagged_as_not_personalised():
    reply = _model_reply()
    reply["story"] = {"title": "Задание", "intro": "Реши задачи по геометрии."}
    for task in reply["tasks"]:
        task["statement"] = "Начерти луч."
    context = load_task_design_context(LEARNER, TOPIC)
    card = parse_task_card(reply, context, 3)

    assert any("не опирается на интересы" in warning for warning in card.warnings), card.warnings


def test_story_built_on_interests_passes_personalization_check():
    card = parse_task_card(_model_reply(), load_task_design_context(LEARNER, TOPIC), 3)
    assert not any("не опирается на интересы" in warning for warning in card.warnings)


def test_story_only_in_the_intro_is_flagged():
    """По сценарию урока тема ребёнка должна быть в самих заданиях, а не в шапке."""
    reply = _model_reply()
    for task in reply["tasks"]:
        task["statement"] = "Начерти луч от точки A."
    card = parse_task_card(reply, load_task_design_context(LEARNER, TOPIC), 3)

    assert any("только во вступлении" in warning for warning in card.warnings), card.warnings


def test_statement_copied_from_the_material_is_flagged():
    reply = _model_reply()
    material = "Отметь точку A и начерти луч, который начинается в точке A."
    reply["tasks"][0]["statement"] = material
    card = parse_task_card(reply, load_task_design_context(LEARNER, TOPIC), 3, [material])

    assert any("дословно повторяет" in warning for warning in card.warnings), card.warnings


def test_correction_names_the_actual_violations():
    correction = build_correction((
        "задание 1: проверьте счёт — «8 · 5 = 45» не сходится (40 = 45)",
        "заданий пришло 5 вместо 3 — лишние отброшены",
        "какое-то не относящееся к делу замечание",
    ))

    assert "проверьте счёт" in correction and "заданий пришло" in correction
    assert "не относящееся" not in correction
    assert build_correction(()) == ""


def test_retry_replaces_a_broken_card_with_a_better_one():
    """Второй запрос стоит одну генерацию и спасает карточку целиком."""
    broken = _model_reply()
    broken["tasks"][0]["expected_answer"] = "Считаем: 5 · 4 = 25 см."
    replies = [broken, _model_reply()]

    def flaky_llm(_prompt, _system):
        return json.dumps(replies.pop(0) if replies else _model_reply(), ensure_ascii=False)

    card = design_task_card(LEARNER, TOPIC, count=3, complete=flaky_llm)

    assert card.retries == 1
    assert not card.warnings, card.warnings


def test_empty_task_list_is_a_hard_failure():
    context = load_task_design_context(LEARNER, TOPIC)
    with pytest.raises(TaskDesignError):
        parse_task_card({"story": {"title": "t", "intro": "i"}, "tasks": []}, context, 3)


def test_child_copy_hides_answers_and_hints():
    card = design_task_card(LEARNER, TOPIC, count=3, complete=_fake_llm(_model_reply()))
    child_text = render_task_card(card, show_answers=False)
    tutor_text = render_task_card(card, show_answers=True)

    assert "Подсказка" not in child_text
    assert "Ответ (для взрослого)" not in child_text
    assert TOPIC not in child_text, "служебный topic_id не показываем ребёнку"
    assert "Подсказка 1" in tutor_text


# --- ранжирование материала ------------------------------------------------

def _bank(tmp_path, records):
    (tmp_path / "bank.json").write_text(
        json.dumps({"tasks": records}, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_difficulty_from_the_zpd_rule_lifts_matching_material(tmp_path):
    """Сложность вычислена правилом ЗБР — материал той же ступени должен быть выше."""
    directory = _bank(tmp_path, [
        {"task_id": "лёгкая", "topic_id": TOPIC, "grade": "3", "difficulty": "easy",
         "content": "Начерти луч из точки A.", "answer": "луч"},
        {"task_id": "средняя", "topic_id": TOPIC, "grade": "3", "difficulty": "medium",
         "content": "Начерти луч из точки A.", "answer": "луч"},
    ])
    ranked = retrieve_material("начерти луч", topic_id=TOPIC, grade="3", limit=2,
                               knowledge_dir=directory, books_dir=directory,
                               difficulty="medium")

    assert ranked[0].topic == "средняя"


def test_material_about_the_child_s_own_mistake_wins(tmp_path):
    directory = _bank(tmp_path, [
        {"task_id": "общая", "topic_id": TOPIC, "grade": "3", "difficulty": "easy",
         "content": "Начерти луч из точки A.", "answer": "луч", "tags": ["чертёж"]},
        {"task_id": "про-ошибку", "topic_id": TOPIC, "grade": "3", "difficulty": "easy",
         "content": "Начерти луч из точки A.", "answer": "луч",
         "tags": ["у луча только одно начало", "второй конец не ставим"]},
    ])
    ranked = retrieve_material("начерти луч", topic_id=TOPIC, grade="3", limit=2,
                               knowledge_dir=directory, books_dir=directory,
                               error_tags=("рисует у луча два конца, как у отрезка",))

    assert ranked[0].topic == "про-ошибку"
