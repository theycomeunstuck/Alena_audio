"""An unanswered final assistant turn survives as reviewed context, not evidence."""
from copy import deepcopy

import pytest

from local_app.memory_service import (SESSION_END_CONTEXT_NOTICE, _tail_with_dialogue_context,
    approved_chat_memory, memory_state)
from local_app.memory_ported.methods import Memory
from tests.test_dialogue_memory_context import TOPIC, prepare, source


TASK = "Попробуй вычислить 1224 : 4. Напиши только ответ."


def last_turn(text=TASK, *, question=None):
    row = source(text="ну давай ещё", question=question)
    row["session_end_context"] = {"assistant_source_id": "assistant-last", "session_id": row["session_id"],
        "role": "assistant", "usage": "context_only", "text": text, "excerpt_mode": "whole", "truncated": False}
    return row


def test_last_exercise_is_preserved_without_inventing_a_student_answer_or_unfinished_status():
    row = last_turn()
    before = deepcopy(row)
    state, context = prepare([row])
    assert TASK in context["text"] and SESSION_END_CONTEXT_NOTICE in context["text"]
    assert context["text"].index(row["text"]) < context["text"].index(TASK)
    assert context["session_end_context"] == [{"source_id": row["source_id"], **row["session_end_context"]}]
    assert context["evidence_ids"] == [row["source_id"]]
    assert context["sources"][0]["role"] == "student"
    assert TASK not in str(state) and "assistant-last" not in str(state)
    assert "session_end_context" not in context["sources"][0]
    assert row == before


def test_previous_question_and_last_instruction_keep_their_order_and_distinct_context_roles():
    row = last_turn(question="Хочешь ещё один пример?")
    _, context = prepare([row])
    assert context["dialogue_context"][0]["text"] == "Хочешь ещё один пример?"
    assert context["session_end_context"][0]["text"] == TASK
    assert context["text"].index("Хочешь ещё один пример?") < context["text"].index(row["text"]) < context["text"].index(TASK)
    assert len(context["sources"]) == 1 and len(context["evidence_ids"]) == 1


def test_budget_never_keeps_a_request_while_detaching_its_last_exercise():
    row = last_turn()
    memory = Memory("tail", None, 4000, TOPIC)
    memory.update({"lesson_id": "lesson", "observed_on": "2026-10-20", "turns": [
        {"turn_id": row["source_id"], "role": "student", "text": row["text"]}]})
    whole, ids, _ = _tail_with_dialogue_context(memory, [row])
    assert ids == {row["source_id"]}
    memory.budget = len(whole) - 1
    before = memory_state(memory)
    closing, events = [], []
    text, ids, dialogue = _tail_with_dialogue_context(memory, [row],
        selection_events=events, session_end_context=closing)
    assert (text, ids, dialogue, closing) == ("", set(), [], [])
    assert events[0]["event"] == "session_end_context_group_skipped"
    assert memory_state(memory) == before
    memory.budget = len(whole)
    assert _tail_with_dialogue_context(memory, [row])[0] == whole


def test_source_correction_and_last_exercise_share_one_budget_group():
    row = last_turn()
    correction = source("teacher-correction", "Уточнение записи: помощь неизвестна.", role="teacher", question=None)
    correction["corrects_source_id"] = row["source_id"]
    state, context = prepare([row, correction])
    assert set(context["evidence_ids"]) == {row["source_id"], correction["source_id"]}
    assert context["session_end_context"][0]["text"] == TASK
    assert TASK not in str(state)


@pytest.mark.parametrize("change", [
    {"role": "student"}, {"usage": "evidence"}, {"text": "x" * 1201},
    {"excerpt_mode": "tail"}, {"truncated": True}, {"assistant_source_id": ""},
])
def test_invalid_or_partial_last_turn_is_not_silently_used(change):
    row = last_turn()
    row["session_end_context"].update(change)
    with pytest.raises(ValueError, match="invalid_session_end_context"):
        prepare([row])


def test_overlong_last_turn_has_a_visible_omission_without_fabricating_an_excerpt():
    row = source(text="ага", question=None)
    row["session_end_context_omitted"] = {"reason": "exceeds_context_excerpt_limit", "limit_chars": 1200}
    _, context = prepare([row])
    assert "session_end_context" not in context
    assert context["sources"][0]["text"] == "ага"
    assert context["selection_events"] == [{"event": "session_end_context_omitted",
        "source_id": row["source_id"], "reason": "exceeds_context_excerpt_limit", "limit_chars": 1200}]


def test_legacy_approval_is_not_silently_enriched_with_new_assistant_context():
    row = last_turn()
    old_state, old_context = prepare([source(text=row["text"], question=None)])
    old = {"version_id": "old", "approved": True, "method": "tail", "topic_id": TOPIC,
           "base_revision": 1, "state": old_state, "context": old_context}
    new_state, new_context = prepare([row], [old])
    draft = {**old, "version_id": "new", "approved": False, "state": new_state, "context": new_context}
    snapshot = {"revision": 1, "sources": [row], "versions": [draft, old]}
    assert TASK not in approved_chat_memory(snapshot, TOPIC)["text"]
    draft["approved"] = True
    selected = approved_chat_memory(snapshot, TOPIC)
    assert selected["session_end_context"][0]["text"] == TASK and TASK in selected["text"]
    assert "session_end_context" not in old["context"]


def test_profile_only_does_not_acquire_assistant_context_or_evidence():
    state, context = prepare([last_turn()], method="profile_only")
    assert context["text"] == "" and context["sources"] == []
    assert "session_end_context" not in context and TASK not in str(state)
