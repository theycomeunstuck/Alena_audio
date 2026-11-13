from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.memory_service import MemoryWorker
from local_app.personal_graph_contract import (PERSONAL_GRAPH_CONTEXT_PROTOCOL,
                                               personal_graph_snapshot_fingerprint)
from local_app.personal_graph_memory import PersonalGraphSelectionBudgetExceeded
from local_app.rag import RagService
from local_app.runtime import IncompleteCompletionError


class Journal:
    def begin_request(self, **kwargs): return "request"
    def prepare(self, **kwargs): return "call"
    def record(self, *args, **kwargs): pass


class Store:
    dsn = "postgresql://example@127.0.0.1/example"
    def __init__(self):
        self.user = {"account_id": "teacher-a", "role": "teacher"}
        self.job = {"job_id": "job-1", "learner_id": "learner-a", "topic_id": "math.test",
                    "kind": "next_step", "method": "personal_graph", "base_revision": 1,
                    "source_revision": 4, "provider": "local"}
        self.snapshot = {"learner_id": "learner-a", "revision": 1, "source_revision": 4,
                         "sources": [], "versions": [], "personal_graph_sources": [],
                         "personal_graph_snapshot_fingerprint": personal_graph_snapshot_fingerprint([]),
                         "personal_graph_hidden_correction_count": 0}
        self.finished = None
        self.jobs = {}
    def get_session(self, token): return {"account": self.user}
    def authorize_learner_access(self, account, learner): return learner == "learner-a"
    def get_card(self, learner):
        return {"learner_id": learner, "grade": 4, "rag_context": {"current_goal": {"topic_id": "math.test"}}}
    def get_learner_memory(self, learner, topic_id=None, *, include_personal_graph_snapshot=False):
        snapshot = deepcopy(self.snapshot)
        if not include_personal_graph_snapshot:
            for key in ("personal_graph_sources", "personal_graph_snapshot_fingerprint",
                        "personal_graph_hidden_correction_count"):
                snapshot.pop(key, None)
        return snapshot
    def claim_memory_job(self, worker_id, lease_seconds=600): return self.job
    def get_memory_job(self, job_id): return self.finished
    def finish_memory_job(self, job_id, **kwargs):
        self.finished = kwargs
        if job_id in self.jobs:
            return {"status": "failed" if kwargs.get("error") else "completed", "job_id": job_id}
        return kwargs
    def enqueue_memory_job(self, learner_id, topic_id, kind, payload, actor_id, **kwargs):
        job = {"job_id": "queued", "learner_id": learner_id, "topic_id": topic_id, "kind": kind,
               "status": "queued", "method": kwargs.get("method", "tail")}
        self.jobs["queued"] = job
        return job
    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        return {**self.jobs[job_id], "status": "running", "worker_id": worker_id}
    def start_memory_job_call(self, job_id, worker_id): pass
    def heartbeat_memory_job(self, job_id, worker_id, lease_seconds=600): return True


class Runtime:
    context_tokens = 16384
    def __init__(self, enabled=True):
        self.settings = SimpleNamespace(experimental_personal_graph_enabled=enabled)
    def chat(self, messages, **kwargs):
        action = {"action": "clarify_missing_observation", "reason": "Уточнить условия помощи.",
                  "evidence_turn_ids": [], "unknowns": ["Помощь неизвестна."], "review_required": True}
        return {"done": True, "message": {"content": json.dumps(action)}, "prompt_eval_count": 200, "eval_count": 30}


class Provider:
    calls = []
    def __init__(self, store, runtime, **kwargs): self.store = store
    def prepare(self, job, snapshot, question):
        self.calls.append((job["learner_id"], snapshot["learner_id"], question))
        return ({"method": "personal_graph", "topic_id": job["topic_id"], "learner_id": job["learner_id"],
                 "archive_run_id": "app-test", "source_revision": 4, "source_ids": [],
                 "source_context_protocol": PERSONAL_GRAPH_CONTEXT_PROTOCOL},
                {"text": "", "sources": [], "evidence_ids": [], "method": "personal_graph",
                 "topic_id": job["topic_id"]})
    def chat_context(self, snapshot, learner_id, topic_id, question):
        self.calls.append((learner_id, topic_id, question))
        return {"text": "", "sources": [], "evidence_ids": [], "version_id": "v1",
                "stale": False, "topic_id": topic_id}


class LocalRag(RagService):
    def __init__(self, external=False):
        self.answer_adapter = SimpleNamespace(external_client=object() if external else None)
        self.received = None
    def answer(self, messages, **kwargs):
        self.received = kwargs
        return {"choices": [{"message": {"role": "assistant", "content": "Попробуем вместе."}}]}


class SessionStore(Store):
    def __init__(self):
        super().__init__()
        self.reservations = {}
        self.history = []
    def get_learning_session(self, session_id, learner_id=None):
        return {"learning_session_id": session_id, "learner_id": "learner-a", "actor_id": "teacher-a",
                "topic_id": "math.test", "status": "active"}
    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, **kwargs):
        if client_message_id in self.reservations:
            return {**self.reservations[client_message_id], "created": False}
        self.history.append({"role": "user", "content": content})
        self.reservations[client_message_id] = {"status": "pending"}
        return {"status": "pending", "created": True}
    def list_learning_messages(self, session_id): return self.history[:]
    def complete_chat_message(self, session_id, client_message_id, response):
        self.reservations[client_message_id] = {"status": "completed", "response": deepcopy(response)}
    def fail_chat_message(self, session_id, client_message_id, **kwargs):
        self.reservations[client_message_id] = {"status": "unknown_outcome"}


def approved(store):
    store.snapshot["versions"] = [{"id": "v1", "version_id": "v1", "method": "personal_graph",
        "approved": True, "topic_id": "math.test", "base_revision": 1, "source_revision": 4,
        "context": {"evidence_ids": []}, "state": {"method": "personal_graph", "learner_id": "learner-a",
        "archive_run_id": "app-test", "source_revision": 4, "source_ids": [],
        "source_context_protocol": PERSONAL_GRAPH_CONTEXT_PROTOCOL}}]


def browser(store, *, enabled=True, external=False):
    settings = SimpleNamespace(experimental_personal_graph_enabled=enabled)
    rag = LocalRag(external=external)
    app = create_app(settings=settings, store=store, runtime=Runtime(enabled), rag_service=rag,
                     chat_response_journal=Journal(), personal_graph_factory=Provider)
    return TestClient(app, headers={"Authorization": "Bearer test"}), rag


def test_teacher_job_uses_experimental_provider_and_remains_a_draft():
    Provider.calls.clear()
    store = Store()
    result = MemoryWorker(store, Runtime(), response_journal=Journal(), personal_graph_factory=Provider).run_once()
    assert not result.get("error")
    assert result["memory_state"]["method"] == "personal_graph"
    assert result["approved"] is False
    assert result["next_step"]["review_required"] is True
    assert Provider.calls[0][:2] == ("learner-a", "learner-a")


def test_teacher_job_keeps_uncited_context_and_corrections_as_review_material():
    action = {"action": "clarify_missing_observation", "reason": "Исправил после вопроса учителя.",
              "evidence_turn_ids": ["note"], "unknowns": [], "review_required": True}

    class AttributingRuntime(Runtime):
        def chat(self, messages, **kwargs):
            return {"done": True, "message": {"content": json.dumps(action)},
                    "prompt_eval_count": 200, "eval_count": 30}

    class AttributingProvider(Provider):
        def prepare(self, job, snapshot, question):
            state, context = super().prepare(job, snapshot, question)
            sources = [{"source_id": "note", "role": "teacher", "text": "Исправил после вопроса."},
                       {"source_id": "reply", "role": "student", "text": "Решил самостоятельно."},
                       {"source_id": "fix", "role": "teacher", "text": "Была подсказка.",
                        "corrects_source_id": "reply"}]
            context.update(text="synthetic selected memory", sources=sources,
                           evidence_ids=[row["source_id"] for row in sources],
                           dialogue_context=[{"source_id": "reply", "role": "assistant", "usage": "context_only",
                                              "text": "Сколько частей?"}],
                           session_end_context=[{"source_id": "reply", "role": "assistant", "usage": "context_only",
                                                 "text": "Продолжим?"}])
            return state, context

    result = MemoryWorker(Store(), AttributingRuntime(), response_journal=Journal(),
                          personal_graph_factory=AttributingProvider).run_once()
    assert not result.get("error")
    saved = result["next_step"]
    assert saved["reason"] == action["reason"] and saved["evidence_turn_ids"] == ["note"]
    review = saved["evidence_review"]
    assert review["meaning_verified"] is False and review["attribution_checks"] == [{"quote": "вопроса учителя"}]
    assert [row["source_id"] for row in review["sources"]] == ["note"]
    extra = review["attribution_contexts"]
    assert [row["source_id"] for row in extra] == ["reply", "fix"]
    assert extra[0]["corrected_by"] == ["fix"]
    assert extra[0]["dialogue_context"]["text"] == "Сколько частей?"
    assert extra[0]["session_end_context"]["text"] == "Продолжим?"
    assert all(not row["cited"] for row in extra)


@pytest.mark.parametrize("done_field", [{"done": False}, {"done": None}, {}])
def test_graph_extraction_requires_a_confirmed_complete_response(done_field):
    class IncompleteRuntime(Runtime):
        def chat(self, messages, **kwargs):
            return {**done_field, "done_reason": "stop", "message": {"content": '{"facts": []}'},
                    "prompt_eval_count": 100}
    store = Store()
    worker = MemoryWorker(store, IncompleteRuntime(), response_journal=Journal(), personal_graph_factory=Provider)
    worker._active_job = store.job
    with pytest.raises(IncompleteCompletionError):
        worker._call([{"role": "user", "content": "synthetic"}], "personal_graph_extract", None)


def test_teacher_job_keeps_memory_selection_warning():
    class WarningProvider(Provider):
        def prepare(self, job, snapshot, question):
            state, context = super().prepare(job, snapshot, question)
            context["selection_warnings"] = [{"code": "memory_budget_exceeded", "count": 1}]
            return state, context
    result = MemoryWorker(Store(), Runtime(), response_journal=Journal(), personal_graph_factory=WarningProvider).run_once()
    assert result["result"]["memory_selection_warnings"][0]["count"] == 1
    assert result["next_step"]["memory_selection_warnings"][0]["count"] == 1


def test_disabled_worker_and_foreign_or_changed_snapshot_fail_closed():
    store = Store()
    result = MemoryWorker(store, Runtime(False), response_journal=Journal(), personal_graph_factory=Provider).run_once()
    assert result["error"]["message"] == "experimental_personal_graph_disabled"
    store = Store()
    store.snapshot["revision"] = 2
    result = MemoryWorker(store, Runtime(), response_journal=Journal(), personal_graph_factory=Provider).run_once()
    assert "revision changed" in result["error"]["message"]
    assert "memory_state" not in result


def test_teacher_endpoint_is_opt_in_and_checks_access():
    store = Store()
    client, _ = browser(store, enabled=False)
    body = {"topic_id": "math.test", "method": "personal_graph"}
    assert client.post("/learners/learner-a/next-step", json=body).status_code == 503
    client, _ = browser(store, enabled=True)
    assert client.post("/learners/learner-b/next-step", json=body).status_code == 403
    result = client.post("/learners/learner-a/next-step", json=body)
    assert result.status_code == 202 and store.jobs["queued"]["method"] == "personal_graph"


def test_chat_requires_approved_current_version_and_local_answer_provider():
    Provider.calls.clear()
    store = Store()
    client, rag = browser(store)
    payload = {"learner_id": "learner-a", "messages": [{"role": "user", "content": "Продолжим?"}],
               "metadata": {"memory_method": "personal_graph"}}
    assert client.post("/v1/chat/completions", json=payload).status_code == 409
    assert not Provider.calls and rag.received is None
    approved(store)
    client, rag = browser(store, external=True)
    assert client.post("/v1/chat/completions", json=payload).status_code == 503
    assert not Provider.calls and rag.received is None
    client, rag = browser(store)
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.json()["memory_method"] == "personal_graph"
    assert rag.received["memory_context"]["version_id"] == "v1"
    assert Provider.calls[-1] == ("learner-a", "math.test", "Продолжим?")


def test_chat_preflight_uses_the_graph_specific_source_snapshot():
    store = Store()
    approved(store)
    store.snapshot["sources"] = [{"source_id": "excluded-unfinished", "author_role": "student",
                                  "text": "Не входит в снимок графа.", "topic_id": "math.test"}]
    store.snapshot["personal_graph_sources"] = []
    client, rag = browser(store)
    payload = {"learner_id": "learner-a", "messages": [{"role": "user", "content": "Продолжим?"}],
               "metadata": {"memory_method": "personal_graph"}}
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert rag.received["memory_context"]["version_id"] == "v1"


def test_chat_rejects_an_old_graph_context_contract():
    Provider.calls.clear()
    store = Store()
    approved(store)
    store.snapshot["versions"][0]["state"].pop("source_context_protocol")
    client, rag = browser(store)
    payload = {"learner_id": "learner-a", "messages": [{"role": "user", "content": "Продолжим?"}],
               "metadata": {"memory_method": "personal_graph"}}
    assert client.post("/v1/chat/completions", json=payload).status_code == 409
    assert not Provider.calls and rag.received is None


def test_chat_explains_when_approved_sources_do_not_fit(monkeypatch):
    store = Store()
    approved(store)
    def over_budget(*args, **kwargs):
        raise PersonalGraphSelectionBudgetExceeded("Фрагменты не поместились в бюджет памяти.")
    monkeypatch.setattr(Provider, "chat_context", over_budget)
    client, rag = browser(store)
    payload = {"learner_id": "learner-a", "messages": [{"role": "user", "content": "Продолжим?"}],
               "metadata": {"memory_method": "personal_graph"}}
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 409 and "бюджет памяти" in response.json()["detail"]
    assert rag.received is None


def test_chat_returns_refresh_instruction_for_stale_personal_graph_index(monkeypatch):
    from local_app.personal_graph_index import PersonalGraphIndexStale
    store = Store()
    approved(store)
    def stale_index(*args, **kwargs):
        raise PersonalGraphIndexStale()
    monkeypatch.setattr(Provider, 'chat_context', stale_index)
    client, rag = browser(store)
    payload = {'learner_id': 'learner-a', 'messages': [{'role': 'user', 'content': 'Продолжим?'}],
               'metadata': {'memory_method': 'personal_graph'}}
    response = client.post('/v1/chat/completions', json=payload)
    assert response.status_code == 409 and 'обновить' in response.json()['detail']
    assert rag.received is None


def test_chat_replay_does_not_switch_memory_method():
    store = SessionStore()
    approved(store)
    client, rag = browser(store)
    payload = {"learner_id": "learner-a", "learning_session_id": "session-1", "client_message_id": "message-1",
               "messages": [{"role": "user", "content": "Продолжим?"}],
               "metadata": {"memory_method": "personal_graph"}}
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    assert client.post("/v1/chat/completions", json=payload).status_code == 200
    payload["metadata"] = {}
    assert client.post("/v1/chat/completions", json=payload).status_code == 409
