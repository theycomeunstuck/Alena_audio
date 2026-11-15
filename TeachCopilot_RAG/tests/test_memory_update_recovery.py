"""Failed local extraction must leave a lesson eligible for a later retry."""

import pytest

from local_app.memory_ported.contracts import ContractValidationError
from local_app.memory_ported.methods import Memory


@pytest.mark.parametrize("method", ["summary", "structured", "structured_v2"])
def test_failed_update_rolls_back_source_ids_before_retry(method):
    lesson = {
        "lesson_id": "lesson",
        "observed_on": "2026-01-01",
        "topic_ids": ["math"],
        "allowed_topic_ids": ["math"],
        "turns": [
            {"turn_id": "q", "role": "teacher", "text": "Сколько частей?"},
            {"turn_id": "a", "role": "student", "text": "Четыре."},
        ],
    }
    attempts = []

    def call(messages, purpose, schema):
        attempts.append(purpose)
        if len(attempts) == 2:
            raise RuntimeError("synthetic model interruption")
        return '{"records":[]}' if method.startswith("structured") else "Краткая запись."

    memory = Memory(method, call, 4000, "math")
    previous = {**lesson, "lesson_id": "previous", "turns": [
        {"turn_id": "old-q", "role": "teacher", "text": "Другой вопрос?"},
        {"turn_id": "old-a", "role": "student", "text": "Ответ."},
    ]}
    memory.update(previous)
    previous_sources = dict(memory.sources)
    previous_summary = memory.summary
    with pytest.raises(RuntimeError, match="synthetic model interruption"):
        memory.update(lesson)
    assert memory.sources == previous_sources
    assert memory.summary == previous_summary
    memory.update(lesson)
    assert set(memory.sources) == {"old-q", "old-a", "q", "a"}
    assert len(attempts) == 3


def test_summary_budget_failure_keeps_diagnostic_event_and_allows_retry():
    outputs = iter(["x" * 129, "Краткая запись."])
    memory = Memory("summary", lambda *args: next(outputs), 128, "math")
    lesson = {"lesson_id": "lesson", "observed_on": "2026-01-01", "turns": [
        {"turn_id": "answer", "role": "student", "text": "Четыре."}]}
    with pytest.raises(ContractValidationError, match="Summary exceeds"):
        memory.update(lesson)
    assert memory.sources == {}
    assert memory.events[-1]["event"] == "summary_exceeds_budget"
    memory.update(lesson)
    assert set(memory.sources) == {"answer"}
    assert memory.events[-1]["event"] == "summary_exceeds_budget"
