"""CPU-only worker and bounded session-summary integration regressions."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from local_app.memory_service import MemoryWorker, attach_session_summaries, session_summary_views
from local_app.personal_graph_contract import personal_graph_snapshot_fingerprint
from local_app.personal_memory import PersonalMemoryStale
from local_app.personal_memory_contract import personal_memory_budgets
from tests.test_personal_graph_integration import Journal, Runtime, Store
from tests.test_personal_memory_store import context_for, source, state_for


class MemoryStore(Store):
    def __init__(self):
        super().__init__()
        self.events = []
        self.job["method"] = "personal_memory"
        note = source("source-1", "math.test", "Ребёнок попросил показать один пример.")
        self.snapshot.update(
            sources=[note], personal_graph_sources=[note],
            personal_graph_snapshot_fingerprint=personal_graph_snapshot_fingerprint([note]),
        )

    def finish_memory_job(self, job_id, **kwargs):
        self.events.append("finish_error" if kwargs.get("error") else "finish_success")
        return super().finish_memory_job(job_id, **kwargs)


class MemoryRuntime(Runtime):
    def __init__(self, store, enabled=True):
        super().__init__()
        self.settings = SimpleNamespace(personal_memory_enabled=enabled)
        self.store = store
        self.scope_active = False
        self.calls = []

    @contextmanager
    def approved_memory_request(self):
        self.store.events.append("scope_enter")
        self.scope_active = True
        try:
            yield
        finally:
            self.scope_active = False
            self.store.events.append("scope_exit")

    def chat(self, messages, **kwargs):
        assert self.scope_active
        self.store.events.append("generation")
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return super().chat(messages, **kwargs)


class MemoryProvider:
    def __init__(self, store, runtime, **kwargs):
        self.store = store

    def prepare(self, job, snapshot, question):
        self.store.events.append("prepare")
        self.store.provider_snapshot = deepcopy(snapshot)
        self.store.provider_question = question
        sources = snapshot["personal_graph_sources"]
        state = state_for(sources, snapshot["source_revision"])
        state["topic_id"] = job["topic_id"]
        context = context_for(sources, [row["source_id"] for row in sources])
        context["topic_id"] = job["topic_id"]
        return state, context

    def chat_context(self, snapshot, learner_id, topic_id, question):
        self.store.events.append("retrieve")
        self.store.provider_question = question
        context = deepcopy(snapshot["versions"][0]["context"])
        context.update(version_id="version-1", stale=False, topic_id=topic_id)
        return context

    def validate_state_identity(self, state):
        self.store.events.append("validate_identity")
        self.store.validated_state = deepcopy(state)


def test_personal_worker_saves_review_draft_after_scoped_generation_and_identity_check():
    store = MemoryStore()
    runtime = MemoryRuntime(store)
    result = MemoryWorker(store, runtime, response_journal=Journal(),
                          personal_memory_factory=MemoryProvider).run_once()

    assert not result.get("error")
    assert result["approved"] is False
    assert result["memory_state"]["method"] == "personal_memory"
    assert result["next_step"]["review_required"] is True
    assert result["context"]["evidence_ids"] == ["source-1"]
    assert store.events == ["prepare", "scope_enter", "generation", "scope_exit",
                            "validate_identity", "finish_success"]
    assert store.validated_state == result["memory_state"]
    assert "personal_graph_sources" in store.provider_snapshot
    assert runtime.calls[0][1]["truncate_history"] is False
    assert not runtime.scope_active


def test_personal_worker_disabled_feature_does_not_retrieve_or_generate():
    store = MemoryStore()
    result = MemoryWorker(store, MemoryRuntime(store, enabled=False), response_journal=Journal(),
                          personal_memory_factory=MemoryProvider).run_once()

    assert result["error"]["message"] == "personal_memory_disabled"
    assert "memory_state" not in result
    assert store.events == ["finish_error"]


def test_personal_worker_identity_change_after_generation_does_not_save_a_draft():
    class ChangedProvider(MemoryProvider):
        def validate_state_identity(self, state):
            super().validate_state_identity(state)
            raise PersonalMemoryStale()

    store = MemoryStore()
    runtime = MemoryRuntime(store)
    result = MemoryWorker(store, runtime, response_journal=Journal(),
                          personal_memory_factory=ChangedProvider).run_once()

    assert result["error"]["type"] == "PersonalMemoryStale"
    assert "memory_state" not in result and "next_step" not in result
    assert len(runtime.calls) == 1
    assert store.events[-2:] == ["validate_identity", "finish_error"]


@pytest.mark.parametrize("done", [False, None, "missing"])
def test_personal_worker_requires_explicit_complete_generation(done):
    class IncompleteRuntime(MemoryRuntime):
        def chat(self, messages, **kwargs):
            response = super().chat(messages, **kwargs)
            response.pop("done")
            if done != "missing":
                response["done"] = done
            return response

    store = MemoryStore()
    runtime = IncompleteRuntime(store)
    result = MemoryWorker(store, runtime, response_journal=Journal(),
                          personal_memory_factory=MemoryProvider).run_once()

    assert result["error"]["type"] == "IncompleteCompletionError"
    assert "memory_state" not in result
    assert "validate_identity" not in store.events
    assert len(runtime.calls) == 1 and not runtime.scope_active


def test_summary_views_materialize_one_and_seven_with_same_trusted_scope(monkeypatch):
    calls = []

    def build(store, learner_id, **kwargs):
        calls.append((store, learner_id, kwargs))
        return {"limit": kwargs["limit"], "sessions": []}

    monkeypatch.setattr("local_app.session_summaries.get_or_build_summaries", build)
    store = SimpleNamespace(_connect=object())
    result = session_summary_views(store, "learner-a", topic_id="math.test",
                                   approved_source_ids=["source-1"])

    assert result == {"last_session": {"limit": 1, "sessions": []},
                      "last_seven_sessions": {"limit": 7, "sessions": []}}
    assert [options["limit"] for _, _, options in calls] == [1, 7]
    assert all(learner == "learner-a" and options["topic_id"] == "math.test"
               and options["approved_source_ids"] == ["source-1"]
               for _, learner, options in calls)


def test_legacy_in_memory_store_without_summary_persistence_keeps_context_contract():
    context = {"method": "personal_memory", "text": "source", "sources": []}
    assert session_summary_views(object(), "learner-a") is None
    assert attach_session_summaries(object(), context, "learner-a", "math.test", 8192) is context


@pytest.mark.parametrize("context_tokens", [8192, 16384, 32768])
def test_summary_attachment_is_scoped_deduplicated_and_within_utf8_budget(monkeypatch, context_tokens):
    _, budget = personal_memory_budgets(context_tokens)
    short = {"learning_session_id": "session-new", "text": "Последняя цитата ребёнка."}
    large = {"learning_session_id": "session-old", "text": "Я" * budget}
    summary_views = {"last_session": {"sessions": [short]},
                     "last_seven_sessions": {"sessions": [short, large]}}
    calls = []

    def views(store, learner_id, **kwargs):
        calls.append((learner_id, kwargs))
        return deepcopy(summary_views)

    monkeypatch.setattr("local_app.memory_service.session_summary_views", views)
    context = {"method": "personal_memory", "text": "Подтверждённая цитата.",
               "sources": [{"source_id": "source-1"}], "evidence_ids": ["source-1"]}
    original = deepcopy(context)
    result = attach_session_summaries(object(), context, "learner-a", "math.test", context_tokens)

    assert context == original
    assert calls == [("learner-a", {"topic_id": "math.test", "approved_source_ids": ["source-1"]})]
    assert result["text"].count(short["text"]) == 1
    assert large["text"] not in result["text"]
    assert result["session_summary_context"] == {
        "included_session_ids": ["session-new"], "omitted_session_count": 1,
    }
    assert result["selection_budget"]["used"] == len(result["text"].encode("utf-8")) <= budget
    assert result["evidence_ids"] == ["source-1"]


def test_other_memory_method_does_not_materialize_session_summaries(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("summaries were requested outside personal_memory")

    monkeypatch.setattr("local_app.memory_service.session_summary_views", unexpected)
    context = {"method": "tail", "text": "legacy", "sources": []}
    assert attach_session_summaries(object(), context, "learner-a", "math.test", 8192) is context
