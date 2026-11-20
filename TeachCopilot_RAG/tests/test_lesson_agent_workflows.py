"""Ported lesson workflows: validation, generation, guided chat, summary contract."""
from __future__ import annotations

import pytest

from local_app.lesson_agent import (generate_lesson, guided_history, guided_messages, guided_reply,
                                    parse_stage_marker, student_lesson_content, summary_sections,
                                    validate_lesson, validate_summary, GUIDED_STAGES,
                                    LESSON_REQUIRED_SECTIONS, SUMMARY_HEADINGS)
from local_app.lesson_agent.validation import answers_section, split_lesson_sections

VALID_LESSON = """# Урок: Умножение
## Цель
Научиться умножать двузначное число на однозначное.
## Разминка
Сосчитай: 2 × 3 и 5 × 4.
## Объяснение
24 × 3 = (20 + 4) × 3 = 60 + 12 = 72. Десятки двузначного числа умножаем отдельно.
## Практика
1. 24 × 3 = ?
2. 13 × 2 = ?
3. 31 × 3 = ?
## Проверка
- Чему равно 12 × 4?
- Как разложить 15 × 6 по разрядам?
## Ответы
24 × 3 = 72
13 × 2 = 26
31 × 3 = 93
12 × 4 = 48
"""


def _broken_lesson() -> str:
    return VALID_LESSON.replace("## Проверка\n", "").replace("24 × 3 = 72", "24 × 3 = 70")


class TestValidateLesson:
    def test_valid_lesson_has_no_problems(self):
        assert validate_lesson(VALID_LESSON, "math") == []

    def test_missing_title_and_sections(self):
        problems = validate_lesson("## Цель\nтекст", "math")
        assert any("заголовок" in problem for problem in problems)
        assert any("## Разминка" in problem for problem in problems)
        assert any("## Ответы" in problem for problem in problems)

    def test_wrong_arithmetic_is_reported(self):
        problems = validate_lesson(VALID_LESSON.replace("24 × 3 = 72", "24 × 3 = 70"), "math")
        assert any("равенство 24 × 3 = 70" in problem for problem in problems)

    def test_praise_before_answer_is_reported(self):
        problems = validate_lesson(VALID_LESSON.replace("## Цель\n", "## Цель\nМолодец! "), "math")
        assert any("похвалу" in problem for problem in problems)

    def test_multiplication_topic_needs_three_examples(self):
        problems = validate_lesson(VALID_LESSON.replace("3. 31 × 3 = ?\n", ""), "math.умножение")
        assert any("ровно 3 задания" in problem for problem in problems)

    def test_multiplication_practice_answer_required(self):
        problems = validate_lesson(VALID_LESSON.replace("31 × 3 = 93", "31 × 3 = 90"), "math.умножение")
        assert any("добавь правильный ответ 31 × 3 = 93" in problem for problem in problems)

    def test_division_story_task_rejected_for_multiplication_topic(self):
        problems = validate_lesson(VALID_LESSON.replace("3. 31 × 3 = ?", "3. Сколько дней в 3 циклах?"), "math.умножение")
        assert any("сюжетные задачи на деление" in problem for problem in problems)

    def test_addition_equalities_checked(self):
        lesson = VALID_LESSON.replace("## Проверка\n", "## Проверка\nЗапиши: 23 + 48 = 61.\n")
        problems = validate_lesson(lesson, "math")
        assert any("23 + 48 = 61" in problem for problem in problems)


class TestSections:
    def test_student_content_strips_answers_only(self):
        student = student_lesson_content(VALID_LESSON)
        assert "## Ответы" not in student
        assert "## Проверка" in student and "## Практика" in student

    def test_lesson_without_answers_section_unchanged(self):
        lesson = VALID_LESSON.split("## Ответы", 1)[0]
        assert student_lesson_content(lesson) == lesson

    def test_split_sections_is_defensive(self):
        sections = split_lesson_sections(VALID_LESSON)
        assert set(sections) == {"_title", "Цель", "Разминка", "Объяснение", "Практика", "Проверка", "Ответы"}
        assert "Урок: Умножение" in sections["_title"]
        assert answers_section(VALID_LESSON).startswith("24 × 3 = 72")
        assert answers_section("нет ответов") is None


class TestGenerationWorkflow:
    def test_generate_validate_and_one_corrective_attempt(self):
        calls = []

        def model_call(messages, purpose, schema):
            calls.append((purpose, schema))
            assert purpose == "lesson_generate" and schema is None
            assert messages[0]["role"] == "system"
            if len(calls) == 1:
                return _broken_lesson()
            return VALID_LESSON

        result = generate_lesson(model_call, profile={"grade": 3, "interests": ["космос"]},
                                 topic_id="math", topic="Умножение", minutes=20)
        assert result["attempts"] == 2 and len(calls) == 2
        assert result["lesson_text"] == VALID_LESSON.strip()
        assert result["validation"]["ok"] and result["validation"]["attempts_log"][0]["problems"]
        assert result["validation"]["attempts_log"][1]["problems"] == []

    def test_failed_after_two_attempts_is_not_stored(self):
        result = generate_lesson(lambda *_: _broken_lesson(), profile={}, topic_id="math",
                                 topic="Умножение", minutes=30)
        assert result["lesson_text"] is None and not result["validation"]["ok"]
        assert result["attempts"] == 2

    def test_fast_memory_feeds_bounded_profile(self):
        seen = {}

        def model_call(messages, purpose, schema):
            seen["prompt"] = messages[1]["content"]
            return VALID_LESSON

        generate_lesson(model_call, profile={}, topic_id="math", topic="Умножение", minutes=20,
                        fast_memory={"method": "unified_fast_memory", "revision": "mem-rev-1",
                                     "personal": {"interests": ["роботы"],
                                                  "preferences": [{"value": "не рисовать задачи про самолёты"}]}})
        assert "роботы" in seen["prompt"]
        assert "не рисовать задачи про самолёты" in seen["prompt"]
        assert "mem-rev-1" not in seen["prompt"]

    def test_invalid_minutes_rejected(self):
        with pytest.raises(ValueError):
            generate_lesson(lambda *_: VALID_LESSON, profile={}, topic_id="math",
                            topic="Умножение", minutes=120)

    def test_non_callable_model_call_rejected(self):
        with pytest.raises(ValueError):
            generate_lesson(None, profile={}, topic_id="math", topic="Умножение", minutes=20)

    def test_empty_model_answer_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            generate_lesson(lambda *_: "  ", profile={}, topic_id="math", topic="Умножение", minutes=20)


class TestGuidedChat:
    def test_stage_marker_parsed_and_stripped(self):
        reply, stage = parse_stage_marker("Давай решим первый пример.\n[Этап: разминка]")
        assert stage == "warmup" and reply == "Давай решим первый пример."

    def test_invalid_marker_keeps_text_and_no_stage(self):
        for marker in ("[Этап: неизвестно]", "[Этап:", "текст", ""):
            text, stage = parse_stage_marker("Ответ\n" + marker)
            assert stage is None and text.startswith("Ответ")

    def test_stage_labels_cover_enum(self):
        russian_labels = ("разминка", "объяснение", "практика", "проверка", "завершено")
        parsed = {parse_stage_marker(f"т\n[Этап: {label}]")[1] for label in russian_labels}
        assert parsed == {"warmup", "explanation", "practice", "check", "completed"}
        assert GUIDED_STAGES == ("not_started", "warmup", "explanation", "practice", "check", "completed")

    def test_guided_history_replaces_opening_and_merges_runs(self):
        history = guided_history([{"role": "assistant", "content": "Начнём урок!"},
                                  {"role": "user", "content": "привет"},
                                  {"role": "user", "content": "готов"}], guided_mode=True)
        assert history[0]["role"] == "user" and "Начни подготовленный урок" in history[0]["content"]
        assert history[1] == {"role": "assistant", "content": "Начнём урок!"}
        assert len(history) == 3 and "привет" in history[2]["content"] and "готов" in history[2]["content"]

    def test_guided_reply_strips_marker(self):
        def model_call(messages, purpose, schema):
            assert purpose == "lesson_guided_chat" and schema is None
            return "Попробуй сам: 24 × 3.\n[Этап: практика]"

        reply, stage = guided_reply(model_call, profile={"grade": 3}, lesson_text=VALID_LESSON,
                                    guided_mode=True, history=[{"role": "assistant", "content": "Начнём!"}],
                                    student_message="помоги")
        assert stage == "practice" and "[Этап" not in reply

    def test_guided_reply_rejects_long_question(self):
        with pytest.raises(ValueError):
            guided_reply(lambda *_: "ответ", profile={}, lesson_text=VALID_LESSON, guided_mode=True,
                         history=[], student_message="х" * 4001)

    def test_guided_reply_empty_question_rejected(self):
        with pytest.raises(ValueError):
            guided_reply(lambda *_: "ответ", profile={}, lesson_text=VALID_LESSON, guided_mode=True,
                         history=[], student_message="   ")

    def test_guided_messages_keep_lesson_and_profile_as_data(self):
        messages = guided_messages({"grade": 3}, "Текст урока", True, [], "вопрос")
        assert messages[0]["role"] == "system"
        assert "Текст урока" in messages[0]["content"]
        assert "Профиль ребёнка (JSON)" in messages[0]["content"]


class TestSummaryContract:
    def test_valid_summary_has_no_problems(self):
        text = "".join(f"## {heading}\nсодержание\n" for heading in SUMMARY_HEADINGS)
        assert validate_summary(text) == []

    def test_missing_heading_detected(self):
        text = "## Что изучали\nx\n## Что получилось\ny"
        problems = validate_summary(text)
        assert any("Что стоит повторить" in problem for problem in problems)

    def test_wrong_order_detected(self):
        text = ("## Что получилось\ny\n## Что изучали\nx\n"
                "## Что стоит повторить\nz\n## Следующий шаг\nw")
        assert any("порядок" in problem for problem in validate_summary(text))

    def test_empty_summary_detected(self):
        assert validate_summary("") != [] and validate_summary("   ") != []

    def test_summary_sections_split_for_ui(self):
        text = "".join(f"## {heading}\nсодержание-{index}\n" for index, heading in enumerate(SUMMARY_HEADINGS))
        sections = summary_sections(text)
        assert set(sections) == set(SUMMARY_HEADINGS)
        assert sections["Что получилось"].startswith("## Что получилось")
