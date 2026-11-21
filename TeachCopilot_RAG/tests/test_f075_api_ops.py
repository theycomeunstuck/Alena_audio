"""Audit F-075: small API operations defects."""
import logging
from datetime import date, datetime, timezone

import pytest

import local_app.api as api
import local_app.auth as auth
from local_app.generation_run_budget import GenerationRunBudgetExceeded
from tests.test_alena_api import ConversationStore, browser, memory_calls, payload  # noqa: F401
from tests.test_personal_memory_api import ObservedRag


class BudgetExhaustedRag(ObservedRag):
    def answer(self, messages, **kwargs):
        raise GenerationRunBudgetExceeded("local pilot run reached its generation limit")


def test_run_budget_exhaustion_is_a_distinct_retryable_rejection(memory_calls):
    store = ConversationStore()
    client, _ = browser(store, rag=BudgetExhaustedRag(store))
    response = client.post("/v1/chat/completions", json=payload(session=True))
    assert response.status_code == 429, response.text
    assert "budget" in str(response.json()["detail"]).lower()
    failure = store.reservations["request-1"]["failure"]
    assert failure["unknown_outcome"] is False  # no provider call happened; the message stays retryable


def test_learner_local_day_uses_the_learner_timezone(monkeypatch):
    class Store:
        def get_learner_learning_timezone(self, learner_id):
            return "Pacific/Kiritimati"  # UTC+14: already "tomorrow" for most of the world

    fixed = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)
    assert api.learner_local_day(Store(), "learner-a", now=fixed) == date(2026, 11, 22)


def test_learner_local_day_falls_back_for_stores_without_timezones():
    fixed = datetime(2026, 11, 21, 20, 0, tzinfo=timezone.utc)
    # Asia/Tomsk (UTC+7) is the documented default learning timezone.
    assert api.learner_local_day(object(), "learner-a", now=fixed) == date(2026, 11, 22)


def test_auth_has_no_dead_session_ttl():
    assert not hasattr(auth, "SESSION_TTL")


def test_readiness_log_names_the_failing_subsystem(caplog):
    class Store:
        def ping(self):
            return True

        def schema_version(self):
            return 10_000

    class Runtime:
        def check_models(self):
            return {"ready": True}

    class Rag:
        def status(self):
            return {"ready": False}

    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    app = api.create_app(settings=SimpleNamespace(), store=Store(), runtime=Runtime(), rag_service=Rag())
    with caplog.at_level(logging.ERROR):
        response = TestClient(app).get("/ready")
    assert response.status_code == 503
    messages = [record.getMessage() for record in caplog.records]
    assert any("embedding index" in message for message in messages), messages
    assert not any(message == "Database readiness check failed" for message in messages)
