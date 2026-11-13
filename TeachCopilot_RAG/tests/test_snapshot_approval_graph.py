"""A reviewed graph snapshot never expands to later child messages."""
from local_app.personal_graph_memory import PersonalGraphProvider, approved_graph_version
from tests.test_personal_graph_archive_api import archive_case, flow_store, LEARNER, TOPIC  # noqa: F401


def test_completed_child_chat_keeps_only_reviewed_graph_and_correction_revokes_it(archive_case):
    case = archive_case
    student = case.store.get_account(email="archive-student@synthetic.test")
    actor = str(student["id"])
    session = case.store.create_learning_session(LEARNER, actor, TOPIC)
    message = case.store.reserve_chat_message(session["id"], LEARNER, "after-draft", "Новая реплика.", actor)
    case.store.complete_chat_message(session["id"], "after-draft", {"answer": "Продолжим."})
    case.store.complete_learning_session(session["id"], LEARNER, actor)
    snapshot = case.store.get_learner_memory(LEARNER, TOPIC, include_personal_graph_snapshot=True)
    draft = next(row for row in snapshot["versions"] if row["id"] == case.version["id"])
    assert draft["can_confirm_previous_snapshot"] is True
    approved = case.store.approve_memory_version(LEARNER, case.version["id"], "teacher", snapshot["revision"],
        acknowledge_new_messages=True, expected_source_revision=snapshot["source_revision"])
    assert approved["context"]["source_access"]["source_ids"] == approved["context"]["evidence_ids"]
    snapshot = case.store.get_learner_memory(LEARNER, TOPIC, include_personal_graph_snapshot=True)
    graph_snapshot = {**snapshot, "sources": snapshot["personal_graph_sources"]}
    assert approved_graph_version(graph_snapshot, LEARNER, TOPIC) is not None
    recalled = PersonalGraphProvider(case.store, object()).chat_context(snapshot, LEARNER, TOPIC, "Что было?")
    assert recalled["snapshot_is_frozen"] is True
    assert recalled["text"] == case.version["context"]["text"]
    assert message["source_id"] not in recalled["evidence_ids"]
    case.store.add_memory_source(LEARNER, TOPIC, "Уточнение преподавателя.", "teacher", approved=True,
                                 corrects_source_id=case.selected["source_id"])
    changed = case.store.get_learner_memory(LEARNER, TOPIC, include_personal_graph_snapshot=True)
    assert approved_graph_version({**changed, "sources": changed["personal_graph_sources"]}, LEARNER, TOPIC) is None
