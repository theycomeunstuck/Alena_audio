"""Scoped time/history checks for an explicitly isolated PostgreSQL database."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg2
import pytest
from psycopg2.extras import Json

from local_app.store import Store


DSN = os.getenv("TEACHCOPILOT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")


@pytest.fixture
def history_store():
    store = Store(DSN)
    store.migrate()
    learners = ["history-time-" + uuid4().hex for _ in range(2)]
    with store._connect() as conn, conn.cursor() as cur:
        for learner in learners:
            cur.execute("""INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card)
                VALUES (%s,'Synthetic',3,%s)""", (learner, Json({"learner_id": learner})))
    accounts = [store.create_account(uuid4().hex + "@synthetic.test", "Synthetic", "unused", role)
                for role in ("student", "student", "teacher")]
    yield store, learners, accounts
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id=ANY(%s)", (learners,))
        cur.execute("DELETE FROM teachcopilot.accounts WHERE id=ANY(%s::uuid[])",
                    ([str(account["id"]) for account in accounts],))


def historical_turn(store, learner, actor, session, timestamp, *, status="completed", author_role="student"):
    ids = [str(uuid4()), str(uuid4())]
    content = "Recorded words " + ids[0]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO teachcopilot.memory_sources
            (id,learner_id,topic_id,learning_session_id,actor_id,author_role,content,created_at,event_timezone,activity_timezone)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'Europe/Moscow','Asia/Tomsk')""",
            (ids[0], learner, session["topic_id"], session["id"], str(actor), author_role, content, timestamp))
        assistant = None
        if status == "completed":
            assistant = ids[1]
            cur.execute("""INSERT INTO teachcopilot.memory_sources
                (id,learner_id,topic_id,learning_session_id,actor_id,author_role,content,created_at,event_timezone,activity_timezone)
                VALUES (%s,%s,%s,%s,'assistant','assistant','Response',%s,'Europe/Moscow','Asia/Tomsk')""",
                (assistant, learner, session["topic_id"], session["id"], timestamp + timedelta(minutes=1)))
        cur.execute("""INSERT INTO teachcopilot.learning_messages
            (learning_session_id,learner_id,client_message_id,source_id,assistant_source_id,content_hash,status,created_at,completed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session["id"], learner, str(uuid4()), ids[0], assistant, hashlib.sha256(content.encode()).hexdigest(),
             status, timestamp, timestamp + timedelta(minutes=1) if status != "pending" else None))
    return ids[0], assistant


def test_timezones_capture_once_and_history_is_append_only(history_store):
    store, learners, accounts = history_store
    learner, student = learners[0], accounts[0]["id"]
    assert store.get_account_display_timezone(student) == "Asia/Tomsk"
    store.set_account_display_timezone(student, "Europe/Moscow")
    session = store.create_learning_session(learner, student, "math.test", event_timezone="Europe/Moscow")
    first = store.reserve_chat_message(session["id"], learner, "first", "Hello", student)
    store.set_account_display_timezone(student, "UTC")
    store.complete_chat_message(session["id"], "first", {"choices": [{"message": {"content": "Hello back"}}]})
    messages = store.get_own_conversation_messages(learner, student)
    assert len(messages) == 2
    assert all(message["created_at"].utcoffset() == timedelta(0) for message in messages)
    assert {message["event_timezone"] for message in messages} == {"Europe/Moscow"}
    assert {message["activity_timezone"] for message in messages} == {"Asia/Tomsk"}
    assert messages[0]["usage"] == "self_report"
    assert messages[1]["usage"] == "context_only"
    assert messages[0]["pair_source_id"] == messages[1]["pair_source_id"] == first["source_id"]
    before = [(message["source_id"], message["text"], message["created_at"], message["activity_day"]) for message in messages]
    for column, value in (("content", "Changed"), ("event_timezone", "UTC"), ("activity_timezone", "UTC")):
        with pytest.raises(psycopg2.errors.CheckViolation):
            with store._connect() as conn, conn.cursor() as cur:
                cur.execute("UPDATE teachcopilot.memory_sources SET " + column + "=%s WHERE id=%s",
                            (value, first["source_id"]))
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learners SET learning_timezone='Europe/Moscow' WHERE learner_id=%s", (learner,))
    after = store.get_own_conversation_messages(learner, student, learning_timezone="Europe/Moscow")
    assert before == [(message["source_id"], message["text"], message["created_at"], message["activity_day"]) for message in after]
    source = store.add_memory_source(learner, "math.test", "New observation", student)
    assert source["activity_timezone"] == "Europe/Moscow"
    assert store.get_session("nonexistent") is None


def test_twenty_individual_messages_and_seven_active_days_keep_scopes(history_store):
    store, learners, accounts = history_store
    learner, student, other, teacher = learners[0], accounts[0]["id"], accounts[1]["id"], accounts[2]["id"]
    sessions = [store.create_learning_session(learner, student, topic) for topic in ("math.test", "reading.test")]
    start = datetime(2026, 1, 3, 18, tzinfo=timezone.utc)
    cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc)
    own = []
    for number in range(9):
        own.extend(historical_turn(store, learner, student, sessions[number % 2], start + timedelta(days=number * 5)))
    for number in range(12):
        own.extend(historical_turn(store, learner, student, sessions[number % 2], start + timedelta(days=40, minutes=number * 2)))
    failed = historical_turn(store, learner, student, sessions[0], start + timedelta(days=41), status="failed")[0]
    pending = historical_turn(store, learner, student, sessions[0], start + timedelta(days=42), status="pending")[0]
    foreign_session = store.create_learning_session(learner, other, "math.test")
    foreign = historical_turn(store, learner, other, foreign_session, start + timedelta(days=42))[0]
    teacher_session = store.create_learning_session(learner, teacher, "math.test")
    teacher_source = historical_turn(store, learner, teacher, teacher_session, start, author_role="teacher")[0]
    teacher_legacy = historical_turn(store, learner, teacher, teacher_session, start + timedelta(days=1), author_role="student")[0]
    private_note = store.add_memory_source(learner, "math.test", "Private teacher note", teacher, approved=True)
    other_learner_session = store.create_learning_session(learners[1], student, "math.test")
    foreign_learner = historical_turn(store, learners[1], student, other_learner_session, start)[0]
    messages = store.get_own_conversation_messages(learner, student, limit=None, as_of=cutoff)
    ids = {message["source_id"] for message in messages}
    assert failed in ids and pending in ids
    assert {foreign, teacher_source, foreign_learner}.isdisjoint(ids)
    assert teacher_legacy not in ids and private_note["source_id"] not in ids
    teacher_history = store.get_own_conversation_messages(learner, teacher, limit=None, as_of=cutoff)
    teacher_rows = {message["source_id"]: message for message in teacher_history}
    assert private_note["source_id"] not in teacher_rows
    assert teacher_rows[teacher_legacy]["author_role"] == "teacher"
    assert teacher_rows[teacher_legacy]["original_source_role"] == "student"
    assert teacher_rows[teacher_source]["actor_role"] == "teacher"
    assert all(message["learner_id"] == learner and message["actor_id"] == str(student) for message in messages)
    assert len(store.get_own_conversation_messages(learner, student, limit=20, as_of=cutoff)) == 20
    days = store.get_own_conversation_messages(learner, student, limit=None, active_day_limit=7, as_of=cutoff)
    assert len({message["activity_day"] for message in days}) == 7
    assert own[0] not in {message["source_id"] for message in days}
    assert {message["topic_id"] for message in days} == {"math.test", "reading.test"}
    paged_ids, boundary = set(), None
    while True:
        page = store.get_own_conversation_messages(learner, student, limit=5,
                                                  before_source_order=boundary, as_of=cutoff)
        if not page:
            break
        page_ids = {message["source_id"] for message in page}
        assert paged_ids.isdisjoint(page_ids)
        paged_ids.update(page_ids)
        boundary = page[0]["source_order"]
    assert paged_ids == ids
    excluded = store.get_own_conversation_messages(learner, student, limit=None, exclude_source_id=pending, as_of=cutoff)
    assert pending not in {message["source_id"] for message in excluded}
    earlier = store.get_own_conversation_messages(learner, student, limit=None, as_of=start + timedelta(seconds=30))
    assert [message["source_id"] for message in earlier] == [own[0]]


def test_ordinary_channel_creation_is_concurrent_and_scoped(history_store):
    store, learners, accounts = history_store
    learner, student = learners[0], accounts[0]["id"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        sessions = list(pool.map(lambda _: store.ensure_conversation_session(learner, student, "general", "UTC"), range(4)))
    assert len({session["id"] for session in sessions}) == 1
    assert sessions[0]["event_timezone"] == "UTC"
    other = store.ensure_conversation_session(learner, accounts[1]["id"], "general", "Asia/Tomsk")
    assert other["id"] != sessions[0]["id"]
    different_topic = store.ensure_conversation_session(learner, student, "reading.test", "Asia/Tomsk")
    assert different_topic["id"] != sessions[0]["id"]
