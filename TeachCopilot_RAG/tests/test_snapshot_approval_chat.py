"""A reviewed snapshot can be confirmed after new child chat, with explicit acknowledgement."""
from copy import deepcopy
from uuid import uuid4

import pytest

from local_app.memory_service import MemoryWorker
from local_app.auth import hash_password
from local_app.store import RevisionConflict
from tests.test_memory_api_flow import flow_store  # noqa: F401
from tests.test_personal_memory_store_postgres import database, draft as personal_draft  # noqa: F401
from tests.test_personal_memory_store import TOPIC as PERSONAL_TOPIC


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"


def draft_for(store):
    source = store.add_memory_source(LEARNER, TOPIC, "Исходная проверенная заметка.", "teacher", approved=True)
    before = store.get_learner_memory(LEARNER, TOPIC)
    state, context = MemoryWorker(store, object())._prepare({"learner_id": LEARNER, "topic_id": TOPIC, "method": "tail"})
    draft = store.save_memory_version(LEARNER, TOPIC, state, context=context,
                                      base_revision=before["revision"], source_revision=before["source_revision"])
    return source, draft


def send_chat(store, *, actor="student", role="student"):
    if actor == "student":
        account = store.create_account("snapshot-student@synthetic.test", "Student",
                                       hash_password("Synthetic snapshot password 123"), "student")
        actor = str(account["id"])
        store.assign_learner(actor, LEARNER, "student")
    session = store.create_learning_session(LEARNER, actor, TOPIC)
    message = store.reserve_chat_message(session["id"], LEARNER, "new-message", "Новая реплика.", actor,
                                         author_role=role)
    store.complete_chat_message(session["id"], "new-message", {"answer": "Ответ помощника."})
    return session, message


def current_version(store, version_id):
    snapshot = store.get_learner_memory(LEARNER, TOPIC)
    version = next(row for row in snapshot["versions"] if row["id"] == version_id)
    return snapshot, version


def test_child_chat_requires_acknowledgement_and_preserves_reviewed_snapshot(flow_store):
    store = flow_store
    source, draft = draft_for(store)
    saved_state, saved_sources = deepcopy(draft["state"]), deepcopy(draft["context"]["sources"])
    _, message = send_chat(store)
    snapshot, visible = current_version(store, draft["id"])
    assert visible["stale"] is True
    assert visible["can_confirm_previous_snapshot"] is True
    assert visible["new_chat_since_snapshot"] is True

    with pytest.raises(RevisionConflict):
        store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"])
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"],
            acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"] - 1)
    approved = store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"],
        acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"])
    assert approved["approved"] is True
    assert approved["state"] == saved_state
    assert approved["context"]["sources"] == saved_sources
    assert approved["context"]["snapshot_review"]["refresh_required"] is True
    assert message["source_id"] not in {row["source_id"] for row in approved["context"]["sources"]}
    assert source["id"] in {row["source_id"] for row in approved["context"]["sources"]}


@pytest.mark.parametrize("change", ["teacher_note", "teacher_chat", "standalone_student", "unlinked_student", "assistant_checkin", "unexplained_revision"])
def test_non_child_chat_changes_cannot_use_snapshot_override(flow_store, change):
    store = flow_store
    _, draft = draft_for(store)
    if change == "teacher_note":
        store.add_memory_source(LEARNER, TOPIC, "Новая заметка учителя.", "teacher", approved=True)
    elif change == "teacher_chat":
        send_chat(store, actor="teacher", role="teacher")
    elif change == "standalone_student":
        store.add_memory_source(LEARNER, TOPIC, "Непривязанная запись.", "student", author_role="student")
    elif change == "unlinked_student":
        account = store.create_account("unlinked-student@synthetic.test", "Student",
                                       hash_password("Synthetic snapshot password 123"), "student")
        actor = str(account["id"])
        session = store.create_learning_session(LEARNER, actor, TOPIC)
        store.reserve_chat_message(session["id"], LEARNER, "unlinked", "Чужая запись.", actor)
    elif change == "assistant_checkin":
        store.add_memory_source(LEARNER, TOPIC, "Вопрос помощника.", "assistant", author_role="assistant")
    else:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE teachcopilot.learners SET memory_source_revision=memory_source_revision+1 WHERE learner_id=%s", (LEARNER,))
    snapshot, visible = current_version(store, draft["id"])
    assert not visible.get("can_confirm_previous_snapshot", False)
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"],
            acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"])


def test_completed_child_chat_can_confirm_the_old_tail_snapshot(flow_store):
    store = flow_store
    _, draft = draft_for(store)
    session, message = send_chat(store)
    store.complete_learning_session(session["id"], LEARNER, session["actor_id"])
    snapshot, visible = current_version(store, draft["id"])
    assert visible["can_confirm_previous_snapshot"] is True
    approved = store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"],
        acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"])
    assert approved["context"]["snapshot_review"]["refresh_required"] is True
    assert message["source_id"] not in {row["source_id"] for row in approved["context"]["sources"]}


def test_old_draft_without_chat_revision_cannot_use_the_exception(flow_store):
    store = flow_store
    _, draft = draft_for(store)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.memory_versions SET chat_source_revision=NULL WHERE id=%s", (draft["id"],))
    send_chat(store)
    snapshot, visible = current_version(store, draft["id"])
    assert visible["can_confirm_previous_snapshot"] is False
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"],
            acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"])


def test_card_change_still_blocks_acknowledged_old_snapshot(flow_store):
    store = flow_store
    _, draft = draft_for(store)
    send_chat(store)
    card = store.get_card(LEARNER)
    card["interests"] = ["космос"]
    store.save_card(card, "teacher", expected_revision=1)
    snapshot, visible = current_version(store, draft["id"])
    assert visible["can_confirm_previous_snapshot"] is False
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"],
            acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"])


def test_new_draft_cannot_forge_review_acknowledgement(flow_store):
    store = flow_store
    _, draft = draft_for(store)
    snapshot, _ = current_version(store, draft["id"])
    context = {**draft["context"], "snapshot_review": {"refresh_required": True}}
    with pytest.raises(ValueError, match="reserved"):
        store.save_memory_version(LEARNER, TOPIC, draft["state"], context=context,
                                  base_revision=snapshot["revision"], source_revision=snapshot["source_revision"])


def test_personal_memory_approval_keeps_only_reviewed_source_ids_after_active_child_chat(database):
    from local_app.personal_memory import approved_personal_memory_version

    store, learners = database
    learner = learners[0]
    snapshot, version, original, correction = personal_draft(store, learner)
    student = store.create_account(f"personal-snapshot-{uuid4().hex}@synthetic.test", "Student",
                                   hash_password("Synthetic snapshot password 123"), "student")
    actor = str(student["id"])
    store.assign_learner(actor, learner, "student")
    session = store.create_learning_session(learner, actor, PERSONAL_TOPIC)
    new_message = store.reserve_chat_message(session["id"], learner, "turn", "Новая реплика", actor)
    store.complete_chat_message(session["id"], "turn", {"answer": "Продолжим"})
    current = store.get_learner_memory(learner, PERSONAL_TOPIC, include_personal_graph_snapshot=True)
    visible = next(row for row in current["versions"] if row["id"] == version["id"])
    assert visible["can_confirm_previous_snapshot"] is True
    with pytest.raises(RevisionConflict):
        store.approve_memory_version(learner, version["id"], "teacher", snapshot["revision"],
            acknowledge_new_messages=True, expected_source_revision=current["source_revision"],
            full_archive_access=True, expected_archive_fingerprint=snapshot["personal_graph_snapshot_fingerprint"])
    approved = store.approve_memory_version(learner, version["id"], "teacher", snapshot["revision"],
        acknowledge_new_messages=True, expected_source_revision=current["source_revision"])
    assert set(approved["context"]["source_access"]["source_ids"]) == {original["source_id"], correction["source_id"]}
    assert new_message["source_id"] not in approved["context"]["source_access"]["source_ids"]
    current = store.get_learner_memory(learner, PERSONAL_TOPIC, include_personal_graph_snapshot=True)
    canonical = {**current, "sources": current["personal_graph_sources"]}
    assert approved_personal_memory_version(canonical, learner, PERSONAL_TOPIC) is not None


def test_personal_memory_after_completed_child_session_uses_only_saved_context(database):
    from local_app.personal_memory import PersonalMemoryProvider, approved_personal_memory_version

    store, learners = database
    learner = learners[0]
    snapshot, version, original, correction = personal_draft(store, learner)
    student = store.create_account(f"personal-snapshot-{uuid4().hex}@synthetic.test", "Student",
                                   hash_password("Synthetic snapshot password 123"), "student")
    actor = str(student["id"])
    store.assign_learner(actor, learner, "student")
    session = store.create_learning_session(learner, actor, PERSONAL_TOPIC)
    new_message = store.reserve_chat_message(session["id"], learner, "turn", "НОВАЯ СЕКРЕТНАЯ РЕПЛИКА", actor)
    store.complete_chat_message(session["id"], "turn", {"answer": "Продолжим"})
    store.complete_learning_session(session["id"], learner, actor)
    current = store.get_learner_memory(learner, PERSONAL_TOPIC, include_personal_graph_snapshot=True)
    visible = next(row for row in current["versions"] if row["id"] == version["id"])
    assert visible["can_confirm_previous_snapshot"] is True
    approved = store.approve_memory_version(learner, version["id"], "teacher", snapshot["revision"],
        acknowledge_new_messages=True, expected_source_revision=current["source_revision"])
    assert set(approved["context"]["source_access"]["source_ids"]) == {original["source_id"], correction["source_id"]}
    current = store.get_learner_memory(learner, PERSONAL_TOPIC, include_personal_graph_snapshot=True)
    canonical = {**current, "sources": current["personal_graph_sources"]}
    assert approved_personal_memory_version(canonical, learner, PERSONAL_TOPIC) is not None
    recalled = PersonalMemoryProvider(store, object()).chat_context(current, learner, PERSONAL_TOPIC, "Что было?")
    assert recalled["text"] == version["context"]["text"]
    assert new_message["source_id"] not in {row["source_id"] for row in recalled["sources"]}
    assert "НОВАЯ СЕКРЕТНАЯ РЕПЛИКА" not in str(recalled)
    store.add_memory_source(learner, PERSONAL_TOPIC, "Поправка преподавателя", "teacher", approved=True)
    current = store.get_learner_memory(learner, PERSONAL_TOPIC, include_personal_graph_snapshot=True)
    assert approved_personal_memory_version({**current, "sources": current["personal_graph_sources"]}, learner, PERSONAL_TOPIC) is None
