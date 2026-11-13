"""Real persistence checks use the explicitly configured isolated database."""
from concurrent.futures import ThreadPoolExecutor

import psycopg2
import pytest

from local_app.checkins import (
    CheckinConflict, QUESTION, get_or_create_checkin, mark_checkin_answered,
    mark_checkin_shown, skip_checkin,
)
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"


@pytest.fixture
def accounts(flow_store):
    teacher = flow_store.create_account("checkin-teacher@synthetic.test", "Teacher", "unused", "teacher")
    flow_store.assign_learner(teacher["id"], LEARNER)
    students = [flow_store.create_student_account(f"checkin-{number}@synthetic.test", "Student", "unused",
                                                  LEARNER, teacher["id"]) for number in range(2)]
    return flow_store, teacher, students


def test_daily_question_concurrent_retry_archive_and_answer_are_durable(accounts):
    store, _, students = accounts
    actor = str(students[0]["id"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        offered = list(pool.map(lambda _: get_or_create_checkin(store, LEARNER, actor), range(8)))
    assert len({item["checkin_id"] for item in offered}) == 1
    checkin_id = offered[0]["checkin_id"]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM teachcopilot.learning_sessions")
        assert cur.fetchone()["n"] == 0
        cur.execute("SELECT count(*) AS n FROM teachcopilot.memory_sources")
        assert cur.fetchone()["n"] == 0

    with ThreadPoolExecutor(max_workers=4) as pool:
        shown = list(pool.map(lambda _: mark_checkin_shown(store, LEARNER, actor, checkin_id), range(8)))
    assert len({item["assistant_source_id"] for item in shown}) == 1
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM teachcopilot.memory_sources WHERE id=%s", (shown[0]["assistant_source_id"],))
        source = cur.fetchone()
        assert source["content"] == QUESTION and source["author_role"] == "assistant"
        assert source["learning_session_id"] is None and not source["approved"]
        assert source["activity_timezone"] == "Asia/Tomsk"

    session = store.create_learning_session(LEARNER, actor, TOPIC)
    message = store.reserve_chat_message(session["id"], LEARNER, "answer", "Я построил корабль.", actor)
    answer = mark_checkin_answered(store, LEARNER, actor, checkin_id, message["source_id"])
    assert answer["status"] == "answered" and answer["answered_source_id"] == message["source_id"]
    assert mark_checkin_answered(store, LEARNER, actor, checkin_id, message["source_id"]) == answer
    store.fail_chat_message(session["id"], "answer", error="synthetic-model-failure")
    assert get_or_create_checkin(store, LEARNER, actor)["status"] == "answered"
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT event_type FROM teachcopilot.daily_checkin_events WHERE checkin_id=%s ORDER BY created_at,id", (checkin_id,))
        assert {row["event_type"] for row in cur.fetchall()} == {"offered", "shown", "answered"}
        cur.execute("SELECT content,approved FROM teachcopilot.memory_sources WHERE id=%s", (message["source_id"],))
        source = cur.fetchone()
        assert source["content"] == "Я построил корабль." and not source["approved"]
    with pytest.raises(psycopg2.IntegrityError), store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.daily_checkins SET question='rewritten' WHERE id=%s", (checkin_id,))
    with pytest.raises(psycopg2.IntegrityError), store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.daily_checkin_events SET event_type='skipped' WHERE checkin_id=%s AND event_type='answered'", (checkin_id,))


def test_daily_skip_is_account_scoped_terminal_and_survives_new_conversation(accounts):
    store, teacher, students = accounts
    actor, other = (str(item["id"]) for item in students)
    item = get_or_create_checkin(store, LEARNER, actor)
    skipped = skip_checkin(store, LEARNER, actor, item["checkin_id"])
    session = store.create_learning_session(LEARNER, actor, "general")
    assert get_or_create_checkin(store, LEARNER, actor, session["id"]) == skipped
    assert skip_checkin(store, LEARNER, actor, item["checkin_id"]) == skipped
    with pytest.raises(PermissionError):
        mark_checkin_shown(store, LEARNER, other, item["checkin_id"])
    with pytest.raises(PermissionError):
        get_or_create_checkin(store, LEARNER, str(teacher["id"]))
    other_offer = get_or_create_checkin(store, LEARNER, other)
    assert other_offer["checkin_id"] != item["checkin_id"] and other_offer["available"]
    message = store.reserve_chat_message(session["id"], LEARNER, "later", "Хочу к заданиям.", actor)
    with pytest.raises(CheckinConflict, match="skipped"):
        mark_checkin_answered(store, LEARNER, actor, item["checkin_id"], message["source_id"])
