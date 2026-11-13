"""Optional integration coverage against the isolated PostgreSQL test database."""
from datetime import datetime, timezone

from local_app.session_summaries import get_or_build_summaries
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"


def _lesson(store, text, *, finish=True):
    session = store.create_learning_session(LEARNER, "summary-test-actor", TOPIC)
    message = store.reserve_chat_message(session["id"], LEARNER, "one", text, "summary-test-actor")
    store.complete_chat_message(session["id"], "one", {"answer": "Попробуй следующий шаг."})
    if finish:
        store.complete_learning_session(session["id"], LEARNER, "summary-test-actor")
    return session["id"], message["source_id"]


def test_real_migration_ties_persistence_corrections_and_grant_boundaries(flow_store):
    store = flow_store
    lessons = [_lesson(store, f"Ответ ученика {i}") for i in range(8)]
    active_session, _ = _lesson(store, "Активное занятие", finish=False)
    store.add_memory_source(LEARNER, TOPIC, "Старая заметка без занятия", "teacher", approved=True)
    fixed_time = datetime(2026, 11, 7, 10, tzinfo=timezone.utc)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_sessions SET completed_at=%s WHERE id=ANY(%s::uuid[])",
                    (fixed_time, [sid for sid, _ in lessons]))

    first = get_or_build_summaries(store, LEARNER, limit=7)
    expected_ids = sorted([sid for sid, _ in lessons], reverse=True)[:7]
    assert [row["learning_session_id"] for row in first["sessions"]] == expected_ids
    assert first["count"] == 7 and active_session not in str(first)
    assert "Старая заметка без занятия" not in str(first)
    assert get_or_build_summaries(store, LEARNER, limit=7)["version_id"] == first["version_id"]
    latest_session = expected_ids[0]
    target = next(source_id for session_id, source_id in lessons if session_id == latest_session)
    before = get_or_build_summaries(store, LEARNER, limit=1)
    correction = store.correct_memory_source(LEARNER, target, "Исправление: работали с подсказкой.", "teacher",
        expected_revision=store.get_revision(LEARNER),
        expected_source_revision=store.get_learner_source_revision(LEARNER),
        client_correction_id="summary-correction")
    after = get_or_build_summaries(store, LEARNER, limit=1)
    assert after["version_id"] != before["version_id"]
    fact = after["sessions"][0]["facts"][0]
    assert fact["source_id"] == correction["source"]["id"] and fact["role"] == "teacher"
    assert fact["correction_chain"][0] == {"source_id": target, "role": "student", "status": "superseded"}
    assert fact["source_session_id"] is None and fact["corrects_source_id"] == target
    assert get_or_build_summaries(store, LEARNER, limit=7)["version_id"] != first["version_id"]
    denied = get_or_build_summaries(store, LEARNER, limit=1, topic_id=TOPIC, approved_source_ids=[target])
    assert not denied["available"] and denied["count"] == 0 and "подсказкой" not in str(denied)
    allowed = get_or_build_summaries(store, LEARNER, limit=1, topic_id=TOPIC,
                                     approved_source_ids=[target, correction["source"]["id"]])
    assert allowed["count"] == 1 and "подсказкой" in allowed["text"]
    assert allowed["sessions"][0]["ending"] is None
