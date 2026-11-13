"""Normal HTTP entry points for reviewed combined-source personal memory."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.personal_memory import PersonalMemoryStale
from tests.test_personal_graph_integration import Journal, LocalRag
from tests.test_personal_memory_service import MemoryProvider, MemoryRuntime, MemoryStore
from tests.test_personal_memory_store import context_for, state_for


class SessionStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.reservations = {}
        self.history = []
        self.completed_sessions = []
        self.session = {"learning_session_id": "session-1", "learner_id": "learner-a",
                        "actor_id": "teacher-a", "topic_id": "math.test", "status": "active"}

    def get_learning_session(self, session_id, learner_id=None):
        if session_id != "session-1" or learner_id not in (None, "learner-a"):
            return None
        return deepcopy(self.session)

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, **kwargs):
        if client_message_id in self.reservations:
            return {**self.reservations[client_message_id], "created": False}
        self.history.append({"role": "user", "content": content})
        self.reservations[client_message_id] = {"status": "pending"}
        return {"status": "pending", "created": True}

    def list_learning_messages(self, session_id):
        return deepcopy(self.history)

    def complete_chat_message(self, session_id, client_message_id, response):
        self.events.append("complete_chat")
        self.reservations[client_message_id] = {"status": "completed", "response": deepcopy(response)}

    def fail_chat_message(self, session_id, client_message_id, **kwargs):
        self.events.append("fail_chat")
        self.reservations[client_message_id] = {"status": "unknown_outcome", "failure": kwargs}

    def complete_learning_session(self, session_id, learner_id, actor_id):
        self.events.append("complete_session")
        self.completed_sessions.append((session_id, learner_id, actor_id))
        self.session["status"] = "completed"
        return {"learning_session_id": session_id, "status": "completed", "job_id": "memory-update-1"}


class ObservedRag(LocalRag):
    def __init__(self, store, external=False):
        super().__init__(external=external)
        self.store = store

    def answer(self, messages, **kwargs):
        self.store.events.append("answer")
        return super().answer(messages, **kwargs)


def approve(store):
    sources = store.snapshot["personal_graph_sources"]
    state = state_for(sources, store.snapshot["source_revision"])
    state["topic_id"] = "math.test"
    context = context_for(sources, [row["source_id"] for row in sources])
    context.update(topic_id="math.test", source_access={"mode": "selected_context", "source_ids": ["source-1"]})
    store.snapshot["versions"] = [{"id": "version-1", "version_id": "version-1", "method": "personal_memory",
        "approved": True, "topic_id": "math.test", "base_revision": 1, "source_revision": 4,
        "state": state, "context": context}]


def browser(store, *, enabled=True, external=False, provider=MemoryProvider):
    settings = SimpleNamespace(personal_memory_enabled=enabled, experimental_personal_graph_enabled=True)
    rag = ObservedRag(store, external=external)
    app = create_app(settings=settings, store=store, runtime=MemoryRuntime(store, enabled), rag_service=rag,
                     chat_response_journal=Journal(), personal_memory_factory=provider)
    return TestClient(app, headers={"Authorization": "Bearer test"}), rag


def chat_payload(*, session=False, method="personal_memory"):
    payload = {"learner_id": "learner-a", "messages": [{"role": "user", "content": "Продолжим?"}],
               "metadata": {"memory_method": method}}
    if session:
        payload.update(learning_session_id="session-1", client_message_id="message-1")
    return payload


@pytest.mark.parametrize("enabled, expected", [(True, 202), (False, 503)])
def test_teacher_next_step_feature_gate(enabled, expected):
    store = MemoryStore()
    client, _ = browser(store, enabled=enabled)
    response = client.post("/learners/learner-a/next-step", json={"topic_id": "math.test", "method": "personal_memory"})

    assert response.status_code == expected
    assert bool(store.jobs) == enabled
    if enabled:
        assert store.jobs["queued"]["method"] == "personal_memory"


@pytest.mark.parametrize("role, learner", [("teacher", "learner-b"), ("student", "learner-a")])
def test_teacher_next_step_enforces_role_and_learner_access(role, learner):
    store = MemoryStore()
    store.user.update(role=role, learner_id="learner-a")
    client, _ = browser(store)

    response = client.post(f"/learners/{learner}/next-step", json={"topic_id": "math.test", "method": "personal_memory"})

    assert response.status_code == 403 and not store.jobs


def test_chat_requires_approved_current_memory_and_local_generation():
    store = MemoryStore()
    client, rag = browser(store)
    assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 409
    assert rag.received is None and "retrieve" not in store.events

    approve(store)
    client, rag = browser(store, external=True)
    assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 503
    assert rag.received is None and "retrieve" not in store.events


def test_chat_passes_reviewed_sources_and_revalidates_identity_before_completion():
    store = SessionStore()
    approve(store)
    client, rag = browser(store)
    response = client.post("/v1/chat/completions", json=chat_payload(session=True))

    assert response.status_code == 200, response.text
    assert response.json()["memory_method"] == "personal_memory"
    assert response.json()["memory_version_id"] == "version-1"
    assert rag.received["memory_context"]["evidence_ids"] == ["source-1"]
    assert rag.received["memory_projection"][0]["source_id"] == "source-1"
    assert store.provider_question == "Продолжим?"
    assert store.events.index("answer") < store.events.index("validate_identity")
    assert store.events.index("validate_identity") < store.events.index("finish_success")
    assert store.events.index("finish_success") < store.events.index("complete_chat")
    assert store.validated_state == store.snapshot["versions"][0]["state"]


def test_student_chat_uses_account_learner_and_session_with_personal_memory():
    store = SessionStore()
    store.user.update(account_id="student-a", role="student", learner_id="learner-a")
    store.session["actor_id"] = "student-a"
    approve(store)
    client, rag = browser(store)
    payload = chat_payload(session=True)
    payload["learner_id"] = "learner-b"
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["memory_method"] == "personal_memory"
    assert response.json()["memory_version_id"] == "version-1"
    assert rag.received["memory_context"]["sources"][0]["source_id"] == "source-1"
    assert store.validated_state["learner_id"] == "learner-a"


def test_active_noncanonical_source_revision_keeps_current_approval_usable():
    store = SessionStore()
    approve(store)
    store.snapshot["source_revision"] += 1
    store.snapshot["sources"].append({"source_id": "active-unreviewed", "author_role": "student",
                                      "text": "CURRENT ACTIVE SESSION", "topic_id": "math.test"})
    client, rag = browser(store)
    response = client.post("/v1/chat/completions", json=chat_payload(session=True))

    assert response.status_code == 200, response.text
    assert rag.received["memory_context"]["evidence_ids"] == ["source-1"]
    assert "CURRENT ACTIVE SESSION" not in str(rag.received["memory_context"])
    assert store.validated_state["source_revision"] == 4


def test_chat_identity_change_after_answer_is_withheld_without_retry_or_saved_reply():
    class ChangedProvider(MemoryProvider):
        def validate_state_identity(self, state):
            super().validate_state_identity(state)
            raise PersonalMemoryStale()

    store = SessionStore()
    approve(store)
    client, _ = browser(store, provider=ChangedProvider)
    response = client.post("/v1/chat/completions", json=chat_payload(session=True))

    assert response.status_code == 409
    assert store.events.count("answer") == 1
    assert "complete_chat" not in store.events and "finish_success" not in store.events
    assert store.reservations["message-1"]["status"] == "unknown_outcome"
    assert client.post("/v1/chat/completions", json=chat_payload(session=True)).status_code == 409
    assert store.events.count("answer") == 1


@pytest.mark.parametrize("changed_method", ["tail", "personal_graph"])
def test_cached_reply_cannot_switch_memory_method(changed_method):
    store = SessionStore()
    approve(store)
    client, _ = browser(store)
    payload = chat_payload(session=True)
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 200

    payload["metadata"]["memory_method"] = changed_method
    assert client.post("/v1/chat/completions", json=payload).status_code == 409
    assert store.events.count("answer") == 1


def test_legacy_cached_reply_cannot_switch_to_personal_memory():
    store = SessionStore()
    approve(store)
    store.reservations["message-1"] = {"status": "completed", "response": {
        "choices": [{"message": {"role": "assistant", "content": "Сохранённый ответ."}}],
    }}
    client, _ = browser(store)
    assert client.post("/v1/chat/completions", json=chat_payload(session=True)).status_code == 409
    assert "answer" not in store.events


def test_student_session_completion_materializes_summaries_without_returning_teacher_content(monkeypatch):
    store = SessionStore()
    store.user.update(account_id="student-a", role="student", learner_id="learner-a")
    store.session["actor_id"] = "student-a"
    calls = []

    def summaries(received_store, learner_id, **kwargs):
        assert received_store is store
        assert store.session["status"] == "completed"
        calls.append((learner_id, kwargs))
        return {"last_session": {"text": "PRIVATE TEACHER OBSERVATION"},
                "last_seven_sessions": {"text": "PRIVATE TEACHER OBSERVATION"}}

    monkeypatch.setattr("local_app.memory_service.session_summary_views", summaries)
    client, _ = browser(store)
    response = client.post("/learning-sessions/session-1/complete")

    assert response.status_code == 202
    assert response.json()["session_summary_status"] == "ready"
    assert "PRIVATE" not in response.text and "session_summaries" not in response.json()
    assert calls == [("learner-a", {})]
    assert store.completed_sessions == [("session-1", "learner-a", "student-a")]


def test_summary_materialization_failure_keeps_completed_session_for_refresh(monkeypatch):
    store = SessionStore()

    def unavailable(*args, **kwargs):
        raise RuntimeError("summary storage unavailable")

    monkeypatch.setattr("local_app.memory_service.session_summary_views", unavailable)
    client, _ = browser(store)
    response = client.post("/learning-sessions/session-1/complete")

    assert response.status_code == 202
    assert response.json()["session_summary_status"] == "pending_refresh"
    assert store.session["status"] == "completed"


@pytest.mark.parametrize("role, learner", [("student", "learner-a"), ("teacher", "learner-b")])
def test_summary_teacher_view_checks_access_before_materialization(monkeypatch, role, learner):
    def unexpected(*args, **kwargs):
        pytest.fail("unauthorized summary was materialized")

    monkeypatch.setattr("local_app.memory_service.session_summary_views", unexpected)
    store = MemoryStore()
    store.user.update(role=role, learner_id="learner-a")
    client, _ = browser(store)
    assert client.get(f"/learners/{learner}/memory").status_code == 403


def test_teacher_memory_view_contains_both_summary_windows_and_default_method(monkeypatch):
    store = MemoryStore()
    store.catalog = lambda: {}
    expected = {"last_session": {"count": 1}, "last_seven_sessions": {"count": 7}}
    monkeypatch.setattr("local_app.memory_service.session_summary_views", lambda *_args, **_kwargs: expected)
    client, _ = browser(store)
    response = client.get("/learners/learner-a/memory")

    assert response.status_code == 200
    assert response.json()["session_summaries"] == expected
    assert response.json()["default_method"] == "personal_memory"


def test_disabled_feature_uses_tail_default_and_skips_summary_materialization(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled summary materialization")

    monkeypatch.setattr("local_app.memory_service.session_summary_views", unexpected)
    store = MemoryStore()
    store.catalog = lambda: {}
    client, _ = browser(store, enabled=False)
    response = client.get("/learners/learner-a/memory")

    assert response.status_code == 200
    assert response.json()["default_method"] == "tail"
    assert response.json()["session_summaries"] is None
    assert client.post("/v1/chat/completions", json=chat_payload()).status_code == 503
