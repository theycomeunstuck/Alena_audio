"""Short replies retain their question without making assistant speech evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from psycopg2.extras import Json

from local_app.memory_service import (DIALOGUE_CONTEXT_NOTICE, MEMORY_BUDGET,
    TAIL_PRESENTATION_VERSION, MemoryWorker, _tail_with_dialogue_context,
    approved_chat_memory, memory_state)
from local_app.memory_store import DIALOGUE_CONTEXT_CHARS
from local_app.store import RevisionConflict
from research_memory.methods import Memory
from tests.test_memory_api_flow import flow_store


TOPIC = "math.g4.fractions.numerator_denominator"
QUESTION = "Понятно, что такое знаменатель?"


def source(source_id="student-one", text="ага", *, question=QUESTION, role="student", session="session-one"):
    value = {"source_id": source_id, "author_role": role, "text": text,
             "session_id": session, "observed_on": "2026-10-19", "topic_id": TOPIC}
    if question is not None:
        value["dialogue_context"] = {"role": "assistant", "text": question, "truncated": False}
    return value


class SnapshotStore:
    def __init__(self, sources, versions=None):
        self.snapshot = {"revision": 1, "sources": deepcopy(sources), "versions": deepcopy(versions or [])}

    def get_learner_memory(self, learner_id, topic_id):
        assert learner_id == "learner-one" and topic_id == TOPIC
        return deepcopy(self.snapshot)


def prepare(sources, versions=None, method="tail"):
    store = SnapshotStore(sources, versions)
    return MemoryWorker(store, SimpleNamespace())._prepare(
        {"learner_id": "learner-one", "topic_id": TOPIC, "method": method})


def test_question_and_ambiguous_agreement_are_distinct_from_evidence():
    raw = source(text="ага  \nну вроде")
    state, context = prepare([raw, source("assistant-one", "Ты всё освоил!", role="assistant")])
    assert state["sources"]["student-one"]["text"] == raw["text"]
    assert QUESTION not in str(state)
    assert context["sources"] == [{"source_id": "student-one", "role": "student",
                                   "text": raw["text"], "observed_on": "2026-10-19"}]
    assert context["evidence_ids"] == ["student-one"]
    assert "assistant-one" not in str(state) and "Ты всё освоил!" not in context["text"]
    assert QUESTION in context["text"] and DIALOGUE_CONTEXT_NOTICE in context["text"]
    assert "не доказательство" in context["text"]
    assert context["dialogue_context"] == [{"source_id": "student-one", "role": "assistant",
        "usage": "context_only", "text": QUESTION, "excerpt_mode": "tail", "truncated": False}]
    assert context["presentation_version"] == TAIL_PRESENTATION_VERSION


def test_budget_never_leaves_short_reply_without_its_known_question():
    memory = Memory("tail", None, 128, TOPIC)
    row = source(question="К" * DIALOGUE_CONTEXT_CHARS)
    memory.update({"lesson_id": "one", "observed_on": "2026-10-19",
                   "turns": [{"turn_id": "student-one", "role": "student", "text": "ага"}]})
    assert "ага" in memory.context(TOPIC).text  # The original baseline fits.
    text, ids, contexts = _tail_with_dialogue_context(memory, [row])
    assert (text, ids, contexts) == ("", set(), [])  # The whole pair does not.


def test_contextual_tail_is_bounded_and_does_not_mutate_frozen_memory_state():
    rows = [source(f"student-{i}", "ну вроде", question=f"Вопрос {i}: " + "К" * 900)
            for i in range(20)]
    memory = Memory("tail", None, MEMORY_BUDGET, TOPIC)
    for row in rows:
        memory.update({"lesson_id": row["source_id"], "observed_on": row["observed_on"],
                       "turns": [{"turn_id": row["source_id"], "role": "student", "text": row["text"]}]})
    original = memory_state(memory)
    text, ids, contexts = _tail_with_dialogue_context(memory, rows)
    assert len(text) <= MEMORY_BUDGET and 1 < len(ids) < len(rows)
    assert {note["source_id"] for note in contexts} == ids
    assert all(note["truncated"] and len(note["text"]) == DIALOGUE_CONTEXT_CHARS for note in contexts)
    assert "Показан конец предыдущей реплики, до 600 символов" in text
    assert "student-19" in ids and "student-0" not in ids
    assert memory_state(memory) == original


def test_long_assistant_turn_preserves_the_trailing_question_as_literal_context():
    question = "Длинное вступление. " * 90 + QUESTION
    _, context = prepare([source(question=question)])
    note = context["dialogue_context"][0]
    assert note["text"] == question[-DIALOGUE_CONTEXT_CHARS:]
    assert note["text"].endswith(QUESTION) and note["truncated"]
    assert note["excerpt_mode"] == "tail"
    assert QUESTION in context["text"]


def test_context_is_ignored_for_teacher_sources_or_wrong_preceding_role():
    teacher = source("teacher-one", "Проверки пока не было.", role="teacher", question="WRONG TEACHER CONTEXT")
    student = source("student-one", "хз", question="WRONG ROLE")
    student["dialogue_context"]["role"] = "student"
    _, context = prepare([teacher, student])
    assert context["dialogue_context"] == []
    assert "WRONG" not in context["text"] and DIALOGUE_CONTEXT_NOTICE not in context["text"]
    assert set(context["evidence_ids"]) == {"teacher-one", "student-one"}


def test_new_draft_can_present_pairs_without_rewriting_legacy_approval():
    old_state, old_context = prepare([source(question=None)])
    version = {"version_id": "old-approved", "method": "tail", "topic_id": TOPIC,
               "state": old_state, "context": old_context, "approved": True, "base_revision": 1}
    snapshot = {"revision": 1, "versions": [version]}
    before = deepcopy(snapshot)
    _, new_context = prepare([source()], [version])
    assert QUESTION in new_context["text"]
    approved = approved_chat_memory(snapshot, TOPIC)
    assert approved["version_id"] == "old-approved" and QUESTION not in approved["text"]
    assert approved["dialogue_context"] == []
    assert snapshot == before


def test_approved_memory_topic_comes_from_matching_version_and_staleness_survives():
    snapshot = {"revision": 3, "versions": [
        {"approved": True, "method": "tail", "topic_id": TOPIC, "version_id": "old-approved",
         "base_revision": 2, "context": {"text": "LEGACY", "topic_id": "forged-topic"}}]}
    selected = approved_chat_memory(snapshot, TOPIC)
    assert selected["topic_id"] == TOPIC and selected["stale"]
    assert approved_chat_memory(snapshot, "scratch.conditions")["text"] == ""


def test_profile_only_remains_empty_and_does_not_receive_dialogue_context():
    _, context = prepare([source()], method="profile_only")
    assert context["text"] == "" and context["evidence_ids"] == [] and context["sources"] == []
    assert "dialogue_context" not in context and "presentation_version" not in context


def _turn(store, session, learner, message_id, text, answer=QUESTION):
    row = store.reserve_chat_message(session["id"], learner, message_id, text, "student")
    store.complete_chat_message(session["id"], message_id, {"answer": answer})
    return row


def test_store_pairs_only_immediate_same_session_learner_and_topic(flow_store):
    store, learner = flow_store, "demo-nika-04"
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,%s,4,%s)",
                    ("foreign-context-learner", "Synthetic", Json({"learner_id": "foreign-context-learner"})))
    foreign = store.create_learning_session("foreign-context-learner", "student", TOPIC)
    _turn(store, foreign, "foreign-context-learner", "one", "другая история", "FOREIGN QUESTION")
    store.complete_learning_session(foreign["id"], "foreign-context-learner", "student")
    other_topic = store.create_learning_session(learner, "student", "python.variables")
    _turn(store, other_topic, learner, "one", "переменная", "OTHER TOPIC QUESTION")
    store.complete_learning_session(other_topic["id"], learner, "student")
    previous_session = store.create_learning_session(learner, "student", TOPIC)
    _turn(store, previous_session, learner, "one", "раньше", "PREVIOUS SESSION QUESTION")
    store.complete_learning_session(previous_session["id"], learner, "student")

    session = store.create_learning_session(learner, "student", TOPIC)
    first = _turn(store, session, learner, "first", "что такое знаменатель?")
    second = _turn(store, session, learner, "second", "ага", "СЛЕДУЮЩИЙ ВОПРОС")
    # A genuine intervening student source prevents reaching back over it.
    with store._connect() as conn, conn.cursor() as cur:
        store._memory_learner(cur, learner)
        standalone = store._memory_source(cur, learner, TOPIC, "student", "ну", author_role="student",
                                         session_id=session["id"], source_key="standalone-student")
    third = _turn(store, session, learner, "third", "хз", "К" * 900 + QUESTION)
    fourth = _turn(store, session, learner, "fourth", "ну вроде")
    with store._connect() as conn, conn.cursor() as cur:
        store._memory_learner(cur, learner)
        store._memory_source(cur, learner, TOPIC, "teacher", "Сначала разберём рисунок.", author_role="teacher",
                             session_id=session["id"], approved=True, source_key="intervening-teacher")
    fifth = _turn(store, session, learner, "fifth", "неа")
    # Defensive topic validation also covers malformed imported sources in the
    # same session: never skip a different-topic predecessor to find an older one.
    with store._connect() as conn, conn.cursor() as cur:
        store._memory_learner(cur, learner)
        store._memory_source(cur, learner, "python.variables", "assistant", "MISMATCHED TOPIC QUESTION",
                             author_role="assistant", session_id=session["id"], source_key="mismatched-topic")
    sixth = _turn(store, session, learner, "sixth", "да")
    store.complete_learning_session(session["id"], learner, "student")
    active = store.create_learning_session(learner, "student", TOPIC)
    active_source = _turn(store, active, learner, "unfinished", "UNFINISHED SESSION")

    snapshot = store.get_learner_memory(learner, TOPIC)
    rows = {row["source_id"]: row for row in snapshot["sources"]}
    assert "dialogue_context" not in rows[first["source_id"]]
    assert rows[second["source_id"]]["dialogue_context"] == {"role": "assistant", "text": QUESTION, "excerpt_mode": "tail", "truncated": False}
    assert rows[standalone["source_id"]]["dialogue_context"]["text"] == "СЛЕДУЮЩИЙ ВОПРОС"
    assert "dialogue_context" not in rows[third["source_id"]]
    assert rows[fourth["source_id"]]["dialogue_context"] == {
        "role": "assistant", "text": ("К" * 900 + QUESTION)[-DIALOGUE_CONTEXT_CHARS:], "excerpt_mode": "tail", "truncated": True}
    assert "dialogue_context" not in rows[fifth["source_id"]]
    assert "dialogue_context" not in rows[sixth["source_id"]]
    assert active_source["source_id"] not in rows
    assert all(row["author_role"] != "assistant" for row in snapshot["sources"])
    assert "FOREIGN QUESTION" not in str(snapshot) and "OTHER TOPIC QUESTION" not in str(snapshot)
    assert "PREVIOUS SESSION QUESTION" not in str(snapshot) and "MISMATCHED TOPIC QUESTION" not in str(snapshot)


def test_persisted_pair_remains_draft_and_stale_approval_is_rejected(flow_store):
    store, learner = flow_store, "demo-nika-04"
    session = store.create_learning_session(learner, "student", TOPIC)
    _turn(store, session, learner, "first", "что это?")
    reply = _turn(store, session, learner, "second", "ага")
    store.complete_learning_session(session["id"], learner, "student")
    state, context = MemoryWorker(store, SimpleNamespace())._prepare(
        {"learner_id": learner, "topic_id": TOPIC, "method": "tail"})
    before = deepcopy(store.get_card(learner)["knowledge"])
    snapshot = store.get_learner_memory(learner, TOPIC)
    version = store.save_memory_version(learner, TOPIC, state, context=context,
        base_revision=snapshot["revision"], source_revision=snapshot["source_revision"])
    assert not version["approved"] and QUESTION in version["context"]["text"]
    assert reply["source_id"] in version["context"]["evidence_ids"]
    assert store.get_card(learner)["knowledge"] == before
    assert approved_chat_memory(store.get_learner_memory(learner, TOPIC), TOPIC)["text"] == ""
    store.add_memory_source(learner, TOPIC, "Нужна ещё одна проверка.", "teacher", approved=True)
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(learner, version["id"], "teacher", snapshot["revision"])

