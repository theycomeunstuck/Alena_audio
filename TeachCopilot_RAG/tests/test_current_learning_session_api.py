"""Read-only dialogue restoration against isolated synthetic PostgreSQL data."""
import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.store import Store
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
ROOT_KEYS = {"available", "learning_session_id", "topic_id", "topic_title", "subject_id",
             "session_mode", "status", "messages", "history_truncated", "unresolved_status"}


class ForbiddenGeneration:
    def chat(self, *args, **kwargs):
        raise AssertionError("restoring a dialogue must not call a model")

    def answer(self, *args, **kwargs):
        raise AssertionError("restoring a dialogue must not call RAG")


def _browser(store):
    forbidden = ForbiddenGeneration()
    return TestClient(create_app(settings=Settings(), store=store, runtime=forbidden, rag_service=forbidden))


@pytest.fixture
def history_app(flow_store):
    store = flow_store
    password = "Synthetic dialogue password 123"
    password_hash = hash_password(password)
    teacher = store.create_account("history-teacher@synthetic.test", "Teacher", password_hash, "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account("history-student@synthetic.test", "Student", password_hash,
                                           LEARNER, teacher["id"])
    browser = _browser(store)

    def auth(account):
        response = browser.post("/auth/login", json={"email": account["email"], "password": password})
        assert response.status_code == 200, response.text
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    with browser:
        yield SimpleNamespace(store=store, browser=browser, student=student, teacher=teacher,
                              student_headers=auth(student), teacher_headers=auth(teacher))


def _session(app, *, actor=None, topic=TOPIC):
    actor = actor or app.student
    return app.store.create_learning_session(LEARNER, actor["id"], topic,
                                             client_session_id="history-" + uuid4().hex)["learning_session_id"]


def _turn(app, session_id, user_text, assistant_text=None, *, actor=None, status="completed", private=None):
    actor = actor or app.student
    message_id = "message-" + uuid4().hex
    reserved = app.store.reserve_chat_message(session_id, LEARNER, message_id, user_text, actor["id"],
                                               author_role="teacher" if actor["role"] == "teacher" else "student")
    if status == "completed":
        response = {"choices": [{"message": {"role": "assistant", "content": assistant_text or "Дальше?"}}]}
        response.update(private or {})
        return app.store.complete_chat_message(session_id, message_id, response)
    if status != "pending":
        return app.store.fail_chat_message(session_id, message_id, "PRIVATE provider diagnostic",
                                           unknown_outcome=status == "unknown_outcome")
    return reserved


def _current(app, *, headers=None, learner=LEARNER, browser=None):
    response = (browser or app.browser).get(f"/learners/{learner}/learning-sessions/current",
                                            headers=headers or app.student_headers)
    assert response.status_code == 200, response.text
    return response.json()


def test_current_requires_access_and_does_not_create_an_empty_session(history_app):
    app = history_app
    url = f"/learners/{LEARNER}/learning-sessions/current"
    assert app.browser.get(url).status_code == 401
    assert _current(app) == {"available": False}
    assert _current(app, headers=app.teacher_headers) == {"available": False}
    for headers in (app.student_headers, app.teacher_headers):
        assert app.browser.get("/learners/another-child/learning-sessions/current", headers=headers).status_code == 403
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT count(*) AS count FROM teachcopilot.learning_sessions")
        assert cur.fetchone()["count"] == 0


def test_current_selects_latest_own_active_session_across_topics(history_app):
    app = history_app
    older = _session(app)
    _turn(app, older, "старый вопрос", "старый ответ")
    current = _session(app, topic="programming.python.loops")
    _turn(app, current, "ну а цикл?", "Повторяем команды.")
    completed = _session(app, topic="scratch")
    _turn(app, completed, "завершённый вопрос", "завершённый ответ")
    app.store.complete_learning_session(completed, LEARNER, app.student["id"])
    teacher_session = _session(app, actor=app.teacher, topic="math")
    _turn(app, teacher_session, "заметка преподавателя", "ответ преподавателю", actor=app.teacher)

    restored = _current(app)
    assert restored["learning_session_id"] == current
    assert restored["topic_id"] == "programming.python.loops"
    assert [message["content"] for message in restored["messages"]] == ["ну а цикл?", "Повторяем команды."]
    teacher_restored = _current(app, headers=app.teacher_headers)
    assert teacher_restored["learning_session_id"] == teacher_session
    assert teacher_restored["messages"][0] == {
        "role": "user", "content": "заметка преподавателя", "delivery_status": "completed"}


def test_current_projects_literal_sources_and_never_private_response(history_app):
    app = history_app
    session = _session(app)
    user_text = "ну хз\nя 3/4 написал\n" + "не знаю – " * 700
    assistant_text = "Покажи одну часть.\nБез спешки."
    _turn(app, session, user_text, assistant_text, private={
        "_guided_division_provenance": {"task_id": "PRIVATE task"},
        "diagnostics": "PRIVATE diagnostics", "usage": {"internal": "PRIVATE usage"},
        "tool_calls": [{"id": "PRIVATE tool"}], "reasoning_content": "PRIVATE reasoning",
    })
    restored = _current(app)
    assert set(restored) == ROOT_KEYS
    assert restored["session_mode"] == "standard"
    assert restored["messages"] == [
        {"role": "user", "content": user_text, "delivery_status": "completed"},
        {"role": "assistant", "content": assistant_text},
    ]
    assert restored["history_truncated"] is False
    assert restored["unresolved_status"] is None
    assert "PRIVATE" not in json.dumps(restored)


def test_current_bounds_whole_turns_and_orders_by_immutable_sources(history_app):
    app = history_app
    session = _session(app)
    for index in range(23):
        _turn(app, session, f"вопрос {index}", f"ответ {index}")
    # Message timestamps are mutable metadata and must not determine transcript order.
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("""UPDATE teachcopilot.learning_messages m
            SET created_at=timestamp '2026-01-01' - (s.source_order * interval '1 second')
            FROM teachcopilot.memory_sources s WHERE s.id=m.source_id AND m.learning_session_id=%s""", (session,))
    restored = _current(app)
    assert restored["history_truncated"] is True
    assert len(restored["messages"]) == 40
    assert [(message["role"], message["content"]) for message in restored["messages"]] == [
        pair for index in range(3, 23) for pair in (("user", f"вопрос {index}"), ("assistant", f"ответ {index}"))]
    exact_window = _session(app)
    for index in range(20):
        _turn(app, exact_window, f"ровно {index}", "ок")
    assert _current(app)["history_truncated"] is False


def test_current_unresolved_status_includes_hidden_turns_and_pending_wins(history_app):
    app = history_app
    session = _session(app)
    _turn(app, session, "старый ответ неизвестен", status="unknown_outcome")
    for index in range(21):
        _turn(app, session, f"новый {index}", "ок")
    pending = _turn(app, session, "ещё думает", status="pending")
    restored = _current(app)
    assert restored["unresolved_status"] == "pending"
    assert restored["messages"][-1] == {"role": "user", "content": "ещё думает", "delivery_status": "pending"}
    app.store.fail_chat_message(session, pending["client_message_id"], "PRIVATE failed request", unknown_outcome=False)
    restored = _current(app)
    assert restored["unresolved_status"] == "unknown_outcome"
    assert restored["messages"][-1]["delivery_status"] == "failed"
    assert "старый ответ неизвестен" not in [message["content"] for message in restored["messages"]]
    assert "PRIVATE" not in json.dumps(restored)
    failed_session = _session(app)
    _turn(app, failed_session, "без ответа", status="failed")
    assert _current(app)["unresolved_status"] is None


def test_current_rejects_assistant_link_to_another_session_or_wrong_role(history_app):
    app = history_app
    other = _session(app)
    foreign = _turn(app, other, "прошлый вопрос", "PRIVATE other session answer")
    current = _session(app)
    message = _turn(app, current, "текущий вопрос", "исходный ответ")
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_messages SET assistant_source_id=%s WHERE id=%s",
                    (foreign["assistant_source_id"], message["id"]))
    assert _current(app)["messages"] == [{"role": "user", "content": "текущий вопрос", "delivery_status": "completed"}]
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_messages SET assistant_source_id=source_id WHERE id=%s", (message["id"],))
    assert _current(app)["messages"] == [{"role": "user", "content": "текущий вопрос", "delivery_status": "completed"}]


def test_current_resolves_topic_titles_without_exposing_unknown_technical_ids(history_app):
    app = history_app
    topics = [
        (TOPIC, "Дробь, числитель и знаменатель", "math"),
        ("programming.python.loops", "Циклы в Python", "python"),
        ("programming.scratch.sequence", "Последовательность команд в Scratch", "scratch"),
        ("math", "Математика", "math"), ("python", "Python", "python"),
        ("scratch", "Scratch", "scratch"), ("programming", "Программирование", "programming"),
        ("general", "Общее занятие", "general"),
        ("legacy.opaque-topic-173", "Прошлое занятие", "general"),
    ]
    for topic_id, title, subject in topics:
        session = _session(app, topic=topic_id)
        restored = _current(app)
        assert restored["learning_session_id"] == session
        assert restored["topic_id"] == topic_id
        assert restored["topic_title"] == title
        assert restored["subject_id"] == subject
        assert restored["messages"] == []


def test_current_survives_new_store_and_remains_read_only_with_expired_pending(history_app):
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn

    app = history_app
    session = _session(app)
    _turn(app, session, "помнишь?", "Да, вот эта переписка.")
    _turn(app, session, "ещё вопрос", status="pending")
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_messages SET created_at=now()-interval '1 day' WHERE status='pending'")

    def snapshot():
        tables = ("learners", "learning_sessions", "learning_messages", "memory_sources", "memory_jobs",
                  "memory_versions", "next_steps", "sessions")
        with app.store._connect() as connection, connection.cursor() as cur:
            result = {}
            for table in tables:
                cur.execute(sql.SQL("SELECT to_jsonb(t) AS row FROM teachcopilot.{} t ORDER BY to_jsonb(t)::text")
                            .format(sql.Identifier(table)))
                result[table] = [row["row"] for row in cur.fetchall()]
            return result

    before = snapshot()
    fresh_store = Store(make_dsn(app.store.dsn, options="-c default_transaction_read_only=on"))
    with _browser(fresh_store) as fresh_browser:
        for _ in range(2):
            restored = _current(app, browser=fresh_browser)
            assert restored["learning_session_id"] == session
            assert restored["unresolved_status"] == "pending"
            assert restored["messages"][1]["content"] == "Да, вот эта переписка."
    assert snapshot() == before
