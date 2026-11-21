"""Audit F-074: login timing, chat message count and session id validation."""
import hashlib

import pytest

import local_app.auth as auth
from tests.test_api_capacity_and_links import _FakeApiStore, _make_client


@pytest.fixture
def pbkdf2_calls(monkeypatch):
    real = hashlib.pbkdf2_hmac
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(auth.hashlib, "pbkdf2_hmac", counting)
    return calls


@pytest.mark.parametrize("case", ["unknown", "inactive", "wrong_password"])
def test_every_failed_login_costs_one_password_hash(pbkdf2_calls, case):
    """Without a dummy hash, an unknown or inactive email answered without
    PBKDF2 work – the response time revealed which emails are registered."""
    store = _FakeApiStore()
    client, _ = _make_client(store)
    store.add_account("student-1", "student@example.org", "student")
    if case == "inactive":
        store.accounts["student@example.org"]["active"] = False
    email = "nobody@example.org" if case == "unknown" else "student@example.org"
    password = "correct horse battery staple" if case == "inactive" else "wrong password value"
    pbkdf2_calls.clear()
    response = client.post("/auth/login", json={"email": email, "password": password})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid email or password"
    assert len(pbkdf2_calls) == 1


def test_successful_login_still_costs_one_password_hash(pbkdf2_calls):
    store = _FakeApiStore()
    client, _ = _make_client(store)
    store.add_account("student-1", "student@example.org", "student")
    pbkdf2_calls.clear()
    response = client.post("/auth/login", json={"email": "student@example.org",
                                                "password": "correct horse battery staple"})
    assert response.status_code == 200
    assert len(pbkdf2_calls) == 1


def _messages(count):
    return [{"role": "user", "content": f"сообщение {index}"} for index in range(count)]


def test_chat_rejects_more_messages_than_the_limit_before_any_work():
    from local_app.api import MAX_CHAT_MESSAGES
    store = _FakeApiStore()
    client, headers = _make_client(store)
    response = client.post("/v1/chat/completions", headers=headers,
                           json={"learner_id": "learner-a", "messages": _messages(MAX_CHAT_MESSAGES + 1)})
    assert response.status_code == 422
    assert str(MAX_CHAT_MESSAGES) in response.json()["detail"]


def test_chat_accepts_a_list_at_the_limit():
    from local_app.api import MAX_CHAT_MESSAGES
    store = _FakeApiStore()
    client, headers = _make_client(store)
    response = client.post("/v1/chat/completions", headers=headers,
                           json={"learner_id": "learner-a", "messages": _messages(MAX_CHAT_MESSAGES)})
    assert not (response.status_code == 422 and str(MAX_CHAT_MESSAGES) in str(response.json().get("detail")))


@pytest.mark.parametrize("field", ["learning_session_id", "metadata"])
def test_malformed_learning_session_id_is_a_422_not_a_fake_502(field):
    store = _FakeApiStore()
    client, headers = _make_client(store)
    payload = {"learner_id": "learner-a", "client_message_id": "m-1", "messages": _messages(1)}
    if field == "metadata":
        payload["metadata"] = {"learning_session_id": "not-a-uuid"}
    else:
        payload["learning_session_id"] = "not-a-uuid"
    response = client.post("/v1/chat/completions", headers=headers, json=payload)
    assert response.status_code == 422, response.text
    assert "learning_session_id" in response.json()["detail"]
