"""Local demo buttons use ordinary sessions and only synthetic accounts."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from tests.test_local_api import FakeRag, FakeRuntime, MemoryStore


DEMO_PASSWORD = "test-only synthetic demo password"
LOCAL_HEADERS = {"Origin": "http://127.0.0.1", "X-TeachCopilot-Demo": "1"}


def demo_store():
    store = MemoryStore()
    store.cards = {"demo-nika-04": {
        "learner_id": "demo-nika-04", "pseudonym": "ДЕМО – Ника Комета", "grade": "4",
        "learner_model": {"engagement": {"record_type": "synthetic_demo"}},
    }}
    teacher = store.add_account("demo-teacher", "demo.rich.teacher@local.test", "teacher", password=DEMO_PASSWORD)
    teacher["payload"] = {"synthetic_demo": True}
    store.add_account("demo-student", "demo.nika04@local.test", "student", "demo-nika-04", password=DEMO_PASSWORD)
    store.assign_learner("demo-teacher", "demo-nika-04")
    return store


def demo_credentials(path):
    path.write_text(json.dumps({
        "teacher": {"email": "demo.rich.teacher@local.test", "password": DEMO_PASSWORD},
        "students": [{"learner_id": "demo-nika-04", "email": "demo.nika04@local.test", "password": DEMO_PASSWORD}],
    }), encoding="utf-8")
    return path


@pytest.fixture
def demo_client(tmp_path):
    store = demo_store()
    path = demo_credentials(tmp_path / "private-credentials.json")
    settings = SimpleNamespace(session_hours=2, rag_limit=3, demo_login_enabled=True, demo_credentials_file=str(path))
    app = create_app(settings=settings, store=store, runtime=FakeRuntime(), rag_service=FakeRag())
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as browser:
        yield browser, store, settings


def test_demo_capability_and_both_buttons_create_normal_server_sessions(demo_client):
    browser, store, _ = demo_client
    response = browser.get("/auth/demo")
    assert response.status_code == 200
    assert response.json() == {"enabled": True, "roles": ["teacher", "student"]}
    assert response.headers["cache-control"] == "no-store"
    assert "@" not in response.text and DEMO_PASSWORD not in response.text
    for role in ("teacher", "student"):
        result = browser.post(f"/auth/demo/{role}", json={}, headers=LOCAL_HEADERS)
        assert result.status_code == 200, result.text
        assert result.json()["account"]["role"] == role
        assert DEMO_PASSWORD not in result.text and "password_hash" not in result.text
        auth = {"Authorization": "Bearer " + result.json()["access_token"]}
        assert browser.get("/auth/me", headers=auth).json()["account"]["role"] == role
        assert browser.get("/learners", headers=auth).json()["learners"][0]["learner_id"] == "demo-nika-04"
        assert browser.post("/auth/logout", headers=auth).status_code == 200
        assert browser.get("/auth/me", headers=auth).status_code == 401
    assert not store.sessions


def test_demo_login_is_opt_in_and_regular_password_login_still_works(demo_client):
    browser, _, settings = demo_client
    settings.demo_login_enabled = False
    assert browser.get("/auth/demo").json() == {"enabled": False}
    assert browser.post("/auth/demo/teacher", json={}, headers=LOCAL_HEADERS).status_code == 404
    assert browser.post("/auth/login", json={"email": "demo.rich.teacher@local.test", "password": DEMO_PASSWORD}).status_code == 200


def test_demo_checks_full_authorized_cards_when_roster_only_contains_summaries(demo_client):
    browser, store, _ = demo_client
    store.cards["learner-private"] = {"learner_id": "learner-private", "pseudonym": "Private"}
    store.list_cards = lambda: [{key: card[key] for key in ("learner_id", "pseudonym")} for card in store.cards.values()]
    original_get_card, reads = store.get_card, []

    def full_card(learner_id):
        reads.append(learner_id)
        assert learner_id != "learner-private", "unassigned full profiles must not be loaded"
        return original_get_card(learner_id)

    store.get_card = full_card
    assert browser.get("/auth/demo").json() == {"enabled": True, "roles": ["teacher", "student"]}
    for role in ("teacher", "student"):
        assert browser.post(f"/auth/demo/{role}", json={}, headers=LOCAL_HEADERS).status_code == 200
    assert reads and set(reads) == {"demo-nika-04"}


@pytest.mark.parametrize("headers", [
    {}, {"X-TeachCopilot-Demo": "1"}, {"Origin": "http://evil.test", "X-TeachCopilot-Demo": "1"},
    {**LOCAL_HEADERS, "Host": "evil.test"},
    {**LOCAL_HEADERS, "Sec-Fetch-Site": "cross-site"},
    {**LOCAL_HEADERS, "Origin": "http://127.0.0.1.evil.test"},
    {**LOCAL_HEADERS, "Origin": "http://127.0.0.1:9000"},
])
def test_demo_rejects_nonlocal_or_cross_origin_requests(demo_client, headers):
    browser, store, _ = demo_client
    assert browser.post("/auth/demo/teacher", json={}, headers=headers).status_code == 403
    assert not store.sessions


def test_demo_rejects_external_client_even_with_forged_local_headers(demo_client):
    browser, store, _ = demo_client
    with TestClient(browser.app, base_url="http://127.0.0.1", client=("203.0.113.8", 50000)) as remote:
        assert remote.get("/auth/demo").status_code == 403
        assert remote.post("/auth/demo/teacher", json={}, headers=LOCAL_HEADERS).status_code == 403
    assert not store.sessions


@pytest.mark.parametrize("change", ["password", "teacher_role", "inactive", "teacher_marker", "learner_marker", "extra_student_access", "extra_teacher_access"])
def test_demo_rejects_changed_credentials_or_non_synthetic_access(demo_client, change):
    browser, store, _ = demo_client
    teacher = store.accounts["demo.rich.teacher@local.test"]
    student = store.accounts["demo.nika04@local.test"]
    role = "teacher"
    if change == "password":
        teacher["password_hash"] = "not a valid password hash"
    elif change == "teacher_role":
        teacher["role"] = "admin"
    elif change == "inactive":
        teacher["active"] = False
    elif change == "teacher_marker":
        teacher["payload"] = {}
    elif change == "learner_marker":
        store.cards["demo-nika-04"]["learner_model"] = {}
    else:
        store.cards["learner-private"] = {"learner_id": "learner-private", "pseudonym": "Private"}
        account = student if change == "extra_student_access" else teacher
        store.assign_learner(account["account_id"], "learner-private")
        role = "student" if change == "extra_student_access" else "teacher"
    assert browser.post(f"/auth/demo/{role}", json={}, headers=LOCAL_HEADERS).status_code == 503
    assert not store.sessions


def test_demo_rejects_role_or_account_overrides_and_missing_file(demo_client):
    browser, store, settings = demo_client
    assert browser.post("/auth/demo/admin", json={}, headers=LOCAL_HEADERS).status_code == 422
    assert browser.post("/auth/demo/student", json={"email": "teacher@example.org"}, headers=LOCAL_HEADERS).status_code == 422
    settings.demo_credentials_file += ".missing"
    assert browser.get("/auth/demo").json() == {"enabled": False}
    assert browser.post("/auth/demo/teacher", json={}, headers=LOCAL_HEADERS).status_code == 503
    assert not store.sessions


def test_demo_settings_default_disabled_and_read_explicit_environment(monkeypatch, tmp_path):
    from local_app.config import Settings

    assert Settings().demo_login_enabled is False
    path = str(tmp_path / "private-demo.json")
    monkeypatch.setenv("LOCAL_DEMO_LOGIN", "1")
    monkeypatch.setenv("LOCAL_DEMO_CREDENTIALS_FILE", path)
    settings = Settings.from_env()
    assert settings.demo_login_enabled is True
    assert settings.demo_credentials_file == path
    monkeypatch.setenv("LOCAL_DEMO_LOGIN", "true")
    assert Settings.from_env().demo_login_enabled is False
