"""Explicit calculator actions survive real authenticated PostgreSQL sessions."""
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.store import Store
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.numbers.division_by_1_2_digit"
CHECK = {
    "operation": "division_check",
    "dividend": 7560,
    "divisor": 24,
    "stated_quotient": 315,
}


class NoModelOrRetrieval:
    """Fail at every injected model/retrieval boundary."""

    def __getattr__(self, name):
        raise AssertionError(f"Explicit calculator action called model/retrieval: {name}")

    def answer(self, *args, **kwargs):
        raise AssertionError("Explicit calculator action called RAG.answer")

    def chat(self, *args, **kwargs):
        raise AssertionError("Explicit calculator action called runtime.chat")

    def embed(self, *args, **kwargs):
        raise AssertionError("Explicit calculator action called runtime.embed")


def _new_browser(store):
    forbidden = NoModelOrRetrieval()
    return TestClient(create_app(
        settings=Settings(), store=store, runtime=forbidden, rag_service=forbidden,
    ))


@pytest.fixture
def explicit_action_api(flow_store):
    store = flow_store
    password = "Synthetic explicit calculator password 123"
    teacher = store.create_account(
        "action-teacher@synthetic.test", "Action teacher", hash_password(password), "teacher",
    )
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account(
        "action-student@synthetic.test", "Action student", hash_password(password),
        LEARNER, teacher["id"],
    )
    with _new_browser(store) as browser:
        def auth(email):
            response = browser.post("/auth/login", json={"email": email, "password": password})
            assert response.status_code == 200, response.text
            return {"Authorization": "Bearer " + response.json()["access_token"]}

        yield browser, store, auth(student["email"]), auth(teacher["email"])


def _start_session(browser, headers):
    response = browser.post(
        f"/learners/{LEARNER}/learning-sessions", headers=headers,
        json={"topic_id": TOPIC, "client_session_id": "explicit-actions"},
    )
    assert response.status_code == 201, response.text
    return response.json()["learning_session_id"]


def _payload(session_id, *, message_id="explicit-check", action=None):
    return {
        "learning_session_id": session_id,
        "client_message_id": message_id,
        "learning_action": deepcopy(CHECK if action is None else action),
        "messages": [],
        "metadata": {"subject_id": "math"},
    }


def test_explicit_check_is_durable_and_replays_after_app_restart(explicit_action_api):
    browser, store, student_headers, teacher_headers = explicit_action_api
    original_card = deepcopy(store.get_card(LEARNER))
    original_revision = store.get_revision(LEARNER)
    session_id = _start_session(browser, student_headers)
    payload = _payload(session_id)

    first = browser.post("/v1/chat/completions", headers=student_headers, json=payload)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["learning_action"]["operation"] == "division_check"
    assert body["learning_action"]["version"]
    request_text = body["learning_action"]["request_text"]
    assert isinstance(request_text, str) and request_text.strip()
    assert all(str(value) in request_text for value in (7560, 24, 315))
    assert body["learning_tool"]["student_competence_assessed"] is False
    assert body["learning_tool"]["facts"]["quotient"] == 315
    assert body["answer_routing"]["backend"] == "deterministic_local"
    assert body["usage"]["total_tokens"] == 0
    assert body["learning_session_id"] == session_id
    assert body["client_message_id"] == payload["client_message_id"]

    rows = store.list_learning_messages(session_id)
    assert [row["author_role"] for row in rows] == ["student", "assistant"]
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert [row["content"] for row in rows] == [
        request_text, body["choices"][0]["message"]["content"],
    ]
    source_revision = store.get_learner_source_revision(LEARNER)

    replay = browser.post("/v1/chat/completions", headers=student_headers, json=payload)
    assert replay.status_code == 200, replay.text
    assert replay.json() == body
    assert store.list_learning_messages(session_id) == rows
    assert store.get_learner_source_revision(LEARNER) == source_revision

    # A new app and Store instance must use the PostgreSQL cache and persisted auth.
    restarted_store = Store(store.dsn)
    with _new_browser(restarted_store) as restarted:
        replay = restarted.post("/v1/chat/completions", headers=student_headers, json=payload)
        assert replay.status_code == 200, replay.text
        assert replay.json() == body
    assert restarted_store.list_learning_messages(session_id) == rows
    assert restarted_store.get_learner_source_revision(LEARNER) == source_revision
    assert restarted_store.get_card(LEARNER) == original_card
    assert restarted_store.get_revision(LEARNER) == original_revision

    completed = browser.post(f"/learning-sessions/{session_id}/complete", headers=student_headers)
    assert completed.status_code == 202, completed.text
    job_id = completed.json()["job_id"]
    job = store.get_memory_job(job_id)
    assert job["kind"] == "memory_update" and job["status"] == "queued"
    snapshot = browser.get(f"/learners/{LEARNER}/memory", headers=teacher_headers)
    assert snapshot.status_code == 200, snapshot.text
    assert [(row["author_role"], row["content"]) for row in snapshot.json()["sources"]] == [
        ("student", request_text),
    ]
    assert snapshot.json()["versions"] == []
    completed_revision = store.get_learner_source_revision(LEARNER)
    replay = browser.post("/v1/chat/completions", headers=student_headers, json=payload)
    assert replay.status_code == 200, replay.text
    assert replay.json() == body
    assert store.get_memory_job(job_id) == job
    assert store.get_learner_source_revision(LEARNER) == completed_revision
    assert store.list_learning_messages(session_id) == rows
    assert store.get_card(LEARNER) == original_card
    assert store.get_revision(LEARNER) == original_revision


def test_persisted_action_identity_and_actor_are_checked_before_cache(explicit_action_api):
    browser, store, student_headers, teacher_headers = explicit_action_api
    session_id = _start_session(browser, student_headers)
    payload = _payload(session_id)
    first = browser.post('/v1/chat/completions', headers=student_headers, json=payload)
    assert first.status_code == 200, first.text
    rows = store.list_learning_messages(session_id)
    source_revision = store.get_learner_source_revision(LEARNER)
    changed = deepcopy(payload)
    changed['learning_action']['dividend'] = 7561
    response = browser.post('/v1/chat/completions', headers=student_headers, json=changed)
    assert response.status_code == 409, response.text
    changed = deepcopy(payload)
    changed['learning_action'] = {'operation': 'division_places', 'dividend': 7560, 'divisor': 24}
    assert browser.post('/v1/chat/completions', headers=student_headers, json=changed).status_code == 409
    changed = deepcopy(payload)
    del changed['learning_action']
    changed['messages'] = [{'role': 'user', 'content': first.json()['learning_action']['request_text']}]
    assert browser.post('/v1/chat/completions', headers=student_headers, json=changed).status_code == 409
    # An assigned teacher may inspect reviewed memory, but cannot replay a student's chat request.
    changed = dict(payload, learner_id=LEARNER)
    assert browser.post('/v1/chat/completions', headers=teacher_headers, json=changed).status_code == 403
    assert store.list_learning_messages(session_id) == rows
    assert store.get_learner_source_revision(LEARNER) == source_revision
