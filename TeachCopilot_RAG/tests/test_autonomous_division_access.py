"""Account-scoped autonomous access, including revocation before message replay."""
from copy import deepcopy
from dataclasses import replace
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.autonomous_division import VERSION as PROTOCOL
from local_app.config import Settings
from tests.test_local_api import auth, token
from tests.test_memory_api_flow import flow_store  # noqa: F401
from tests.test_verified_learning_api import LearningApiStore, LEARNER, TOPIC


LISTED = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
EXCLUDED = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
TEACHER = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
SESSION = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"


class NoInference:
    def answer(self, *args, **kwargs):
        raise AssertionError("Access checks must not call RAG")

    def chat(self, *args, **kwargs):
        raise AssertionError("Access checks must not call a model")


class GuardedStore(LearningApiStore):
    """Authentication and owner reads work; crossing into an action is an error."""

    def create_learning_session(self, *args, **kwargs):
        raise AssertionError("Denied access must precede session creation or replay")

    def autonomous_division_context(self, *args, **kwargs):
        raise AssertionError("Denied access must precede saved-step restoration")

    def reserve_chat_message(self, *args, **kwargs):
        raise AssertionError("Denied access must precede reservation, including cached replay")

    def enqueue_memory_job(self, *args, **kwargs):
        raise AssertionError("Denied access must precede job admission")


def browser(store, settings):
    forbidden = NoInference()
    return TestClient(create_app(settings=settings, store=store, runtime=forbidden, rag_service=forbidden))


def settings_from_env(monkeypatch, value):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    for name in ("OLLAMA_URL", "LOCAL_CONTEXT_TOKENS", "LOCAL_REQUEST_TIMEOUT",
                 "LOCAL_MAX_OUTPUT_TOKENS", "LOCAL_SESSION_HOURS", "LOCAL_MIN_SIMILARITY"):
        monkeypatch.delenv(name, raising=False)
    if value is None:
        monkeypatch.delenv("LOCAL_AUTONOMOUS_DIVISION_ACCOUNT_IDS", raising=False)
    else:
        monkeypatch.setenv("LOCAL_AUTONOMOUS_DIVISION_ACCOUNT_IDS", value)
    return Settings.from_env()


@pytest.mark.parametrize("value,expected", [
    (None, None), ("[]", ()), (json.dumps([LISTED]), (LISTED,)),
    (json.dumps([LISTED, EXCLUDED]), (LISTED, EXCLUDED)),
])
def test_account_list_environment_preserves_explicit_scope(monkeypatch, value, expected):
    assert settings_from_env(monkeypatch, value).autonomous_division_account_ids == expected


@pytest.mark.parametrize("value", [
    "", "null", "{}", "true", "1", LISTED, json.dumps(LISTED),
    "[null]", "[true]", "[1]", '[""]', '["student-1"]',
    json.dumps([LISTED.upper()]), json.dumps(["{" + LISTED + "}"]),
    json.dumps([LISTED, LISTED]),
])
def test_malformed_environment_never_falls_back_to_global_access(monkeypatch, value):
    monkeypatch.setenv("LOCAL_AUTONOMOUS_DIVISION_ENABLED", "1")
    with pytest.raises(ValueError):
        settings_from_env(monkeypatch, value)


@pytest.mark.parametrize("value", [
    [], [LISTED], LISTED, True, (None,), (True,), (1,), ("student-1",),
    (LISTED.upper(),), (LISTED, LISTED),
])
def test_constructor_rejects_invalid_or_mutable_scope(value):
    with pytest.raises(ValueError):
        Settings(autonomous_division_enabled=True, autonomous_division_account_ids=value)


def test_feature_is_account_scoped_and_does_not_advertise_teacher_access():
    store = GuardedStore()
    for account_id, role in ((LISTED, "student"), (EXCLUDED, "student"), (TEACHER, "teacher")):
        store.add_account(account_id, account_id + "@synthetic.test", role, LEARNER)
    settings = Settings(autonomous_division_enabled=True, autonomous_division_account_ids=(LISTED, TEACHER))
    with browser(store, settings) as client:
        for account_id, expected in ((LISTED, True), (EXCLUDED, False), (TEACHER, False)):
            headers = auth(token(client, account_id + "@synthetic.test"))
            response = client.get("/auth/me", headers=headers)
            assert response.status_code == 200
            assert response.json()["features"]["autonomous_division"] is expected


@pytest.mark.parametrize("enabled,scope,account_id,role,status", [
    (False, (LISTED,), LISTED, "student", 404),
    (True, (), LISTED, "student", 404),
    (True, (LISTED,), EXCLUDED, "student", 404),
    (True, (TEACHER,), TEACHER, "teacher", 403),
])
def test_denied_accounts_cannot_bypass_scope_with_direct_requests_or_replays(
        enabled, scope, account_id, role, status):
    store = GuardedStore()
    account = store.add_account(account_id, "access@synthetic.test", role, LEARNER)
    store.learning_sessions[SESSION] = {
        "learning_session_id": SESSION, "learner_id": LEARNER, "actor_id": account_id,
        "topic_id": TOPIC, "status": "active", "session_mode": "autonomous_division",
        "client_session_id": "already-created",
    }
    store.learning_messages[(SESSION, "already-completed")] = {
        "status": "completed", "response": {"PRIVATE": "cached reply"}}
    settings = Settings(autonomous_division_enabled=enabled, autonomous_division_account_ids=scope)
    with browser(store, settings) as client:
        headers = auth(token(client, account["email"]))
        before = deepcopy((store.learning_sessions, store.learning_messages, store.sources, store.memory_jobs))
        for client_id in ("new-session", "already-created"):
            result = client.post(f"/learners/{LEARNER}/learning-sessions", headers=headers, json={
                "topic_id": TOPIC, "mode": "autonomous_division", "client_session_id": client_id})
            assert result.status_code == status, result.text
        for session_id in (None, SESSION):
            query = {"topic_id": TOPIC, "protocol": PROTOCOL}
            if session_id:
                query["learning_session_id"] = session_id
            result = client.get(f"/learners/{LEARNER}/guided-division", headers=headers, params=query)
            assert result.status_code == status, result.text
        for message_id in ("new-message", "already-completed"):
            for explicit, protocol in ((True, PROTOCOL), (False, PROTOCOL), (False, None)):
                payload = {"learning_session_id": SESSION, "client_message_id": message_id,
                           "learner_id": LEARNER, "metadata": {"account_id": LISTED, "subject_id": "math"}}
                if role == "teacher":
                    payload["learner_id"] = LEARNER
                if protocol:
                    payload["learning_protocol"] = protocol
                if explicit:
                    payload["learning_action"] = {"operation": "autonomous_division_start"}
                else:
                    payload["messages"] = [{"role": "user", "content": "Продолжи пример."}]
                result = client.post("/v1/chat/completions", headers=headers, json=payload)
                assert result.status_code == status, result.text
                assert "PRIVATE" not in result.text
        assert (store.learning_sessions, store.learning_messages, store.sources, store.memory_jobs) == before
        assert client.get("/auth/me", headers=headers).json()["features"]["autonomous_division"] is False


def test_revocation_blocks_live_actions_and_cached_reply_but_preserves_owner_history(flow_store, monkeypatch):
    """Use real PostgreSQL state to ensure the gate cannot hide a write or reroute."""
    store = flow_store
    password = "Synthetic scoped pilot password 123"
    teacher = store.create_account("scope-teacher@synthetic.test", "Teacher", hash_password(password), "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account("scope-student@synthetic.test", "Student", hash_password(password),
                                           LEARNER, teacher["id"])
    settings = Settings(autonomous_division_enabled=True, autonomous_division_account_ids=(str(student["id"]),))
    with browser(store, settings) as allowed:
        headers = auth(token(allowed, student["email"], password))
        response = allowed.post(f"/learners/{LEARNER}/learning-sessions", headers=headers, json={
            "topic_id": TOPIC, "client_session_id": uuid4().hex, "mode": "autonomous_division"})
        assert response.status_code == 201, response.text
        sid = response.json()["learning_session_id"]
        payload = {"learning_session_id": sid, "client_message_id": "scope-start", "learning_protocol": PROTOCOL,
                   "learning_action": {"operation": "autonomous_division_start"}}
        first = allowed.post("/v1/chat/completions", headers=headers, json=payload)
        assert first.status_code == 200, first.text
        assert allowed.post("/v1/chat/completions", headers=headers, json=payload).json() == first.json()
        assert allowed.get(f"/learners/{LEARNER}/guided-division", headers=headers, params={
            "topic_id": TOPIC, "protocol": PROTOCOL, "learning_session_id": sid}).json()["available"] is True
    saved = deepcopy(store.list_learning_messages(sid))
    denied_settings = replace(settings, autonomous_division_account_ids=())
    with browser(store, denied_settings) as revoked, monkeypatch.context() as guard:
        for method in ("reserve_chat_message", "enqueue_memory_job", "autonomous_division_context"):
            guard.setattr(store, method, lambda *a, **k: pytest.fail("Revoked access reached an action"))
        assert revoked.post("/v1/chat/completions", headers=headers, json=payload).status_code == 404
        changed = deepcopy(payload)
        changed["client_message_id"] = "scope-after-revocation"
        assert revoked.post("/v1/chat/completions", headers=headers, json=changed).status_code == 404
        result = revoked.get(f"/learners/{LEARNER}/learning-sessions/current", headers=headers)
        assert result.status_code == 200 and result.json()["learning_session_id"] == sid
        assert store.list_learning_messages(sid) == saved
        completed = revoked.post(f"/learning-sessions/{sid}/complete", headers=headers)
        assert completed.status_code == 202, completed.text
        assert completed.json()["job"] is None and completed.json()["job_id"] is None
        history = revoked.get(f"/learners/{LEARNER}/learning-sessions/{sid}/history", headers=headers)
        assert history.status_code == 200, history.text
        assert history.json()["messages"]
        assert store.list_learning_messages(sid) == saved
    # The scope is specific to autonomous actions: a former owner can still use the legacy protocol.
    with browser(store, denied_settings) as legacy:
        standard = legacy.post(f"/learners/{LEARNER}/learning-sessions", headers=headers,
                               json={"topic_id": TOPIC, "client_session_id": "legacy-after-revocation"})
        assert standard.status_code == 201 and standard.json()["session_mode"] == "standard"
        response = legacy.post("/v1/chat/completions", headers=headers, json={
            "learning_session_id": standard.json()["learning_session_id"], "client_message_id": "legacy-start",
            "learning_action": {"operation": "division_guided_start", "dividend": 3636, "divisor": 12}})
        assert response.status_code == 200, response.text
        assert response.json()["guided_division"]["phase"] == "quotient_digit"
