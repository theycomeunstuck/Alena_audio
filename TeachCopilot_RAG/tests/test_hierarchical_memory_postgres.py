"""Run only against an explicitly configured isolated PostgreSQL test database."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import os
from uuid import uuid4

from psycopg2.extras import Json
import pytest

from local_app.hierarchical_memory import build_conversation_memory
from local_app.store import Store


DSN = os.getenv("TEACHCOPILOT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")
NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)


@pytest.fixture
def database(disposable_store):
    # audit F-017: guarded shared fixture (per-test disposable database).
    store = disposable_store
    learner = "hierarchy-test-" + uuid4().hex
    actors = {role: str(uuid4()) for role in ("student", "teacher", "other")}
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,'Synthetic',3,%s)",
                    (learner, Json({"learner_id": learner})))
        for label, actor in actors.items():
            role = "teacher" if label == "teacher" else "student"
            cur.execute("""INSERT INTO teachcopilot.accounts(id,email,display_name,password_hash,role)
                VALUES (%s,%s,'Synthetic','not-an-authentication-fixture',%s)""", (actor, actor + "@synthetic.test", role))
            cur.execute("INSERT INTO teachcopilot.account_learners(account_id,learner_id,relationship) VALUES (%s,%s,%s)",
                        (actor, learner, role))
    try:
        yield store, learner, actors
    finally:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id=%s", (learner,))
            cur.execute("DELETE FROM teachcopilot.accounts WHERE id=ANY(%s::uuid[])", (list(actors.values()),))


def insert_turn(store, learner, actor, when, text, *, topic="math", source_role="student", status="completed"):
    session = store.create_learning_session(learner, actor, topic)
    source_ids = []
    with store._connect() as conn, conn.cursor() as cur:
        rows = [(source_role, actor, text, when)]
        if status == "completed":
            rows.append(("assistant", "assistant", "Расскажи подробнее.", when + timedelta(seconds=2)))
        for role, source_actor, content, timestamp in rows:
            cur.execute("""INSERT INTO teachcopilot.memory_sources
                (learner_id,topic_id,learning_session_id,actor_id,author_role,content,created_at,event_timezone,activity_timezone)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'Asia/Tomsk','Asia/Tomsk') RETURNING id""",
                (learner, topic, session["id"], source_actor, role, content, timestamp))
            source_ids.append(cur.fetchone()["id"])
        cur.execute("""INSERT INTO teachcopilot.learning_messages
            (learning_session_id,learner_id,client_message_id,source_id,assistant_source_id,content_hash,status,created_at,completed_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (session["id"], learner, str(uuid4()), source_ids[0], source_ids[1] if len(source_ids) > 1 else None,
             hashlib.sha256(text.encode("utf-8")).hexdigest(), status, when,
             when + timedelta(seconds=2) if status == "completed" else None))
    return source_ids


def test_real_calendar_merge_sparse_days_recent_individual_count_and_scope(database):
    store, learner, actors = database
    for number in range(8):
        insert_turn(store, learner, actors["student"], NOW - timedelta(days=(7 - number) * 8), f"День {number}")
    insert_turn(store, learner, actors["student"], NOW + timedelta(minutes=1), "Другая тема в тот же день", topic="reading")
    insert_turn(store, learner, actors["other"], NOW, "OTHER ACCOUNT SECRET")
    store.add_memory_source(learner, "math", "PRIVATE TEACHER NOTE", actors["teacher"], approved=True)
    result = build_conversation_memory(store, learner, actors["student"], context_budget_bytes=16000,
                                       as_of=NOW + timedelta(hours=1))
    assert result["last_seven_active_days"]["count"] == 7
    assert result["last_day"]["source_count"] == 4
    assert result["last_seven_active_days"]["source_count"] == 16
    assert len(result["recent_messages"]) == 18
    assert "SECRET" not in str(result) and "PRIVATE" not in str(result)
    assert result["last_day"]["timezone"] == "Asia/Tomsk"


def test_real_readthrough_cache_concurrency_and_append_only_versions(database):
    store, learner, actors = database
    insert_turn(store, learner, actors["student"], NOW, "Корабль Сокол")

    def build():
        return build_conversation_memory(store, learner, actors["student"], as_of=NOW + timedelta(hours=1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, concurrent = list(pool.map(lambda _: build(), range(2)))
    assert first["last_day"]["version_id"] == concurrent["last_day"]["version_id"]
    assert first["last_seven_active_days"]["version_id"] == concurrent["last_seven_active_days"]["version_id"]
    insert_turn(store, learner, actors["student"], NOW + timedelta(minutes=1), "Теперь корабль зовут Искра")
    second = build()
    assert second["last_day"]["supersedes_version_id"] == first["last_day"]["version_id"]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT payload FROM teachcopilot.conversation_summary_versions WHERE id=%s",
                    (first["last_day"]["version_id"],))
        assert "Сокол" in cur.fetchone()["payload"]["text"]
        cur.execute("SELECT count(*) AS n FROM teachcopilot.conversation_summary_versions WHERE learner_id=%s", (learner,))
        assert cur.fetchone()["n"] == 4


def test_real_old_day_keeps_zone_when_learning_and_display_preferences_change(database):
    store, learner, actors = database
    when = NOW.replace(hour=18)
    insert_turn(store, learner, actors["student"], when, "После полуночи по Томску")
    first = build_conversation_memory(store, learner, actors["student"], as_of=when + timedelta(minutes=1))
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learners SET learning_timezone='Europe/Moscow' WHERE learner_id=%s", (learner,))
        cur.execute("UPDATE teachcopilot.accounts SET display_timezone='America/New_York' WHERE id=%s", (actors["student"],))
    second = build_conversation_memory(store, learner, actors["student"], as_of=when + timedelta(minutes=1))
    assert second["learning_timezone"] == "Europe/Moscow"
    assert first["last_day"]["day"] == second["last_day"]["day"] == "2026-11-08"
    assert first["last_day"]["version_id"] == second["last_day"]["version_id"]
    assert first["system_text"] == second["system_text"]


def test_real_failed_message_and_legacy_teacher_role_projection(database):
    store, learner, actors = database
    insert_turn(store, learner, actors["student"], NOW, "Сохранённый неотвеченный вопрос", status="unknown_outcome")
    insert_turn(store, learner, actors["teacher"], NOW, "Мой пробный вопрос", source_role="student")
    student = build_conversation_memory(store, learner, actors["student"], as_of=NOW + timedelta(minutes=1))
    teacher = build_conversation_memory(store, learner, actors["teacher"], as_of=NOW + timedelta(minutes=1))
    assert len(student["recent_messages"]) == 1
    assert student["recent_messages"][0]["delivery_status"] == "unknown_outcome"
    assert "Преподаватель, собственная реплика" in teacher["system_text"]
    assert teacher["recent_messages"][0]["author_role"] == "teacher"
    assert teacher["recent_messages"][0]["original_source_role"] == "student"
    assert "Мой пробный вопрос" not in student["system_text"]
