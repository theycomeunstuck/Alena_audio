"""Approved topic memory freshness follows eligible durable sources, not active chat."""
import os

import pytest

from local_app.memory_service import MemoryWorker
from local_app.store import RevisionConflict
from tests.test_memory_store import conversation_card, memory_store  # noqa: F401


pytestmark = pytest.mark.skipif(not os.getenv("TEACHCOPILOT_TEST_DSN"),
                                reason="set TEACHCOPILOT_TEST_DSN for PostgreSQL integration tests")
TOPIC = "math.synthetic"
OTHER_TOPIC = "math.other-synthetic"


def _version(store, learner, topic=TOPIC, *, approved=True):
    snapshot = store.get_learner_memory(learner,topic)
    state, context = MemoryWorker(store,object())._prepare(
        {"learner_id":learner,"topic_id":topic,"method":"tail"})
    version = store.save_memory_version(learner,topic,state,context=context,
        base_revision=snapshot["revision"],source_revision=snapshot["source_revision"])
    if approved:
        store.approve_memory_version(learner,version["id"],"teacher",snapshot["revision"])
    return version


def _stored_version(store, learner, version, topic=TOPIC):
    return next(row for row in store.get_learner_memory(learner,topic)["versions"]
                if row["id"] == version["id"])


def _approved_source(store, learner, text="Ученик начал решение. Помощь неизвестна.", topic=TOPIC):
    return store.add_memory_source(learner,topic,text,"teacher",approved=True)


def test_active_student_and_assistant_turns_keep_approved_memory_fresh(memory_store):
    store, learner, _ = memory_store
    original = _approved_source(store,learner)
    version = _version(store,learner)
    session = store.create_learning_session(learner,"student",TOPIC)
    store.reserve_chat_message(session["id"],learner,"first","хз","student")
    assert not _stored_version(store,learner,version)["stale"]
    store.complete_chat_message(session["id"],"first",{"answer":"Какой шаг пока неясен?"})
    snapshot = store.get_learner_memory(learner,TOPIC)
    assert snapshot["source_revision"] > version["source_revision"]
    assert {row["source_id"] for row in snapshot["sources"]} == {original["id"]}
    assert not _stored_version(store,learner,version)["stale"]


def test_completed_session_makes_approved_memory_stale_and_new_snapshot_fresh(memory_store):
    store, learner, _ = memory_store
    _approved_source(store,learner)
    version = _version(store,learner)
    session = store.create_learning_session(learner,"student",TOPIC)
    message = store.reserve_chat_message(session["id"],learner,"first","Ответ пока не проверил.","student")
    store.complete_chat_message(session["id"],"first",{"answer":"Проверь обратным действием."})
    store.complete_learning_session(session["id"],learner,"student")
    assert _stored_version(store,learner,version)["stale"]
    updated = _version(store,learner)
    stored = _stored_version(store,learner,updated)
    assert not stored["stale"]
    assert message["source_id"] in stored["state"]["sources"]
    assert all(row["role"] != "assistant" for row in stored["state"]["sources"].values())


def test_teacher_correction_changes_eligible_sources_but_unapproved_note_does_not(memory_store):
    store, learner, _ = memory_store
    _approved_source(store,learner)
    version = _version(store,learner)
    store.add_memory_source(learner,TOPIC,"Черновая заметка, ещё не проверена.","teacher",approved=False)
    assert not _stored_version(store,learner,version)["stale"]
    _approved_source(store,learner,"Исправляю заметку: решение показал учитель, самостоятельность неизвестна.")
    assert _stored_version(store,learner,version)["stale"]


def test_other_topic_approved_sources_do_not_expire_current_topic_with_or_without_filter(memory_store):
    store, learner, _ = memory_store
    _approved_source(store,learner)
    version = _version(store,learner)
    _approved_source(store,learner,"Другая тема: новая попытка без проверки.",OTHER_TOPIC)
    other = _version(store,learner,OTHER_TOPIC)
    for topic in (TOPIC,None):
        assert not _stored_version(store,learner,version,topic)["stale"]
    assert not _stored_version(store,learner,other,None)["stale"]
    _approved_source(store,learner,"Вторая заметка только по другой теме.",OTHER_TOPIC)
    assert not _stored_version(store,learner,version,None)["stale"]
    assert _stored_version(store,learner,other,None)["stale"]


def test_learner_card_change_still_expires_approved_native_memory(memory_store):
    store, learner, _ = memory_store
    card = conversation_card(store,learner)
    _approved_source(store,learner)
    version = _version(store,learner)
    card["interests"] = ["космос"]
    store.save_card(card,"teacher",1)
    assert _stored_version(store,learner,version)["stale"]


@pytest.mark.parametrize("state",[
    {"method":"tail"},
    {"method":"tail","topic_id":OTHER_TOPIC,"sources":{}},
    {"method":"tail","topic_id":TOPIC,"sources":{"incomplete":{}}},
    {"method":"legacy","topic_id":TOPIC,"sources":{}},
])
def test_legacy_or_malformed_snapshot_keeps_conservative_global_revision_check(memory_store,state):
    store, learner, _ = memory_store
    version = store.save_memory_version(learner,TOPIC,state,base_revision=1,approved=True)
    assert not _stored_version(store,learner,version)["stale"]
    session = store.create_learning_session(learner,"student",TOPIC)
    store.reserve_chat_message(session["id"],learner,"first","ага","student")
    assert _stored_version(store,learner,version)["stale"]


def test_native_empty_snapshot_is_current_until_eligible_source_exists(memory_store):
    store, learner, _ = memory_store
    version = _version(store,learner)
    session = store.create_learning_session(learner,"student",TOPIC)
    store.reserve_chat_message(session["id"],learner,"first","не понял","student")
    assert not _stored_version(store,learner,version)["stale"]
    store.fail_chat_message(session["id"],"first","synthetic provider unavailable",unknown_outcome=False)
    store.complete_learning_session(session["id"],learner,"student")
    assert _stored_version(store,learner,version)["stale"]


def test_drafts_save_and_approval_keep_strict_global_revision_guards(memory_store):
    store, learner, _ = memory_store
    _approved_source(store,learner)
    draft = _version(store,learner,approved=False)
    session = store.create_learning_session(learner,"student",TOPIC)
    store.reserve_chat_message(session["id"],learner,"first","хз","student")
    assert _stored_version(store,learner,draft)["stale"]
    with pytest.raises(RevisionConflict,match="no longer matches"):
        store.approve_memory_version(learner,draft["id"],"teacher",1)
    with pytest.raises(RevisionConflict,match="sources changed"):
        store.save_memory_version(learner,TOPIC,draft["state"],base_revision=1,
                                  source_revision=draft["source_revision"])
    assert not _stored_version(store,learner,draft)["approved"]
