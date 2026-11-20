"""Worker dispatch for the unified job kinds against injected fakes.

The worker claims a job, runs the ported lesson workflow through its own
``model_call`` hook (journal, heartbeat and admission stay effective) and
saves valid drafts through the contract fake. Failed validation and a missing
facade fail the job honestly.
"""
from __future__ import annotations

from pathlib import Path
import json

import pytest

from local_app.memory_service import MemoryWorker
from tests.test_unified_app_api import (LESSON_TEXT, UnifiedMemoryUnavailable,
                                        UnifiedMemoryValidationError, FakeUnifiedMemory)

pytest.importorskip("pydantic")


class WorkerStore:
    """Minimal durable queue surface: claim/heartbeat/start/finish/get."""

    def __init__(self):
        self.jobs = {}
        self.claimed = []
        self.heartbeats = []
        self.calls_started = []
        self.finished = {}
        self.next_claim = []

    def enqueue(self, kind, payload=None, learner_id="learner-a", actor_id="teacher-t",
                topic_id="math", job_id=None):
        job_id = job_id or f"job-{len(self.jobs) + 1}"
        job = {"job_id": job_id, "id": job_id, "learner_id": learner_id, "topic_id": topic_id,
               "kind": kind, "method": "tail", "provider": "local", "status": "queued",
               "payload": dict(payload or {}), "actor_id": actor_id}
        self.jobs[job_id] = job
        return job

    def claim_memory_job(self, worker_id, lease_seconds=120):
        job_id = self.next_claim.pop(0) if self.next_claim else None
        if job_id is None:
            return None
        job = self.jobs[job_id]
        job["status"] = "running"
        job["worker_id"] = worker_id
        self.claimed.append(job_id)
        return dict(job)

    def heartbeat_memory_job(self, job_id, worker_id, lease_seconds=120):
        self.heartbeats.append((job_id, worker_id, lease_seconds))
        return True

    def start_memory_job_call(self, job_id, worker_id):
        self.calls_started.append((job_id, worker_id))
        return dict(self.jobs[job_id])

    def finish_memory_job(self, job_id, **kwargs):
        job = self.jobs[job_id]
        job.update(kwargs)
        job["status"] = "failed" if kwargs.get("error") else "completed"
        self.finished[job_id] = job
        return dict(job)

    def get_memory_job(self, job_id):
        return dict(self.jobs.get(job_id))

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system",
                           expected_revision=None, method="tail", idempotency_key=None,
                           provider="local"):
        # Minimal unified-queue surface for the worker's stale-retry path.
        return self.enqueue(kind, payload=payload, learner_id=learner_id,
                            actor_id=actor_id, topic_id=topic_id)

    def get_card(self, learner_id):
        return {"learner_id": learner_id, "grade": 4, "language": "ru",
                "interests": ["роботы"],
                "rag_context": {"current_goal": {"topic_id": "math", "goal": "выучить умножение"}}}

    def catalog(self):
        return {"math": {"title_ru": "Математика"}}


class WorkerRuntime:
    context_tokens = 8192
    settings_job_lease = 600

    def __init__(self, answers):
        self.answers = list(answers)
        self.chat_calls = []

    @property
    def settings(self):
        from types import SimpleNamespace
        return SimpleNamespace(job_lease_seconds=self.settings_job_lease)

    def chat(self, messages, response_format=None, max_output_tokens=None, **kwargs):
        self.chat_calls.append({"messages": messages, "max_output_tokens": max_output_tokens,
                                "response_format": response_format})
        if not self.answers:
            raise AssertionError("unexpected extra model call")
        return {"message": {"content": self.answers.pop(0)},
                "prompt_eval_count": 100, "eval_count": 20, "done": True}


class RecordingJournal:
    def __init__(self):
        self.prepared = []
        self.recorded = []

    def prepare(self, **kwargs):
        self.prepared.append(kwargs)
        return f"call-{len(self.prepared)}"

    def record(self, call_id, **kwargs):
        self.recorded.append({"call_id": call_id, **kwargs})


def make_worker(store, runtime, facade=None):
    journal = RecordingJournal()
    if facade is not None:
        factory = lambda s, r, model_call=None: facade
    else:
        # Simulate the peer's backend being absent: an explicit None factory
        # keeps the honest-failure path testable after the merge.
        factory = lambda s, r, model_call=None: None
    worker = MemoryWorker(store, runtime, "worker-unified", response_journal=journal,
                          unified_memory_factory=factory)
    return worker, journal


class TestLessonGenerationJob:
    def test_valid_generation_saves_draft_through_facade(self):
        store, runtime, fake = WorkerStore(), WorkerRuntime([LESSON_TEXT]), FakeUnifiedMemory()
        worker, journal = make_worker(store, runtime, fake)
        store.enqueue("lesson_generate", payload={"topic_id": "math", "minutes": 20,
                                                  "client_request_id": "req-1"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "completed"
        assert fake.lessons and list(fake.lessons.values())[0]["status"] == "draft"
        saved_call = [entry for entry in fake.calls if entry[0] == "save_lesson"]
        assert saved_call and saved_call[0][3] == "math"
        assert finished["result"]["lesson_id"] == "lesson-1"
        assert finished["result"]["validation"]["ok"] is True
        # The model call went through the worker hook: journal + heartbeat + admission.
        assert journal.prepared and journal.prepared[0]["purpose"] == "lesson_generate"
        assert store.heartbeats and store.calls_started
        assert runtime.chat_calls[0]["max_output_tokens"] == 2048
        # The generation prompt carries the effective fast-memory records; the
        # card-only interest "роботы" must not resurrect through a fallback
        # while the facade projection is present (round-2 P0.1).
        system_prompt = runtime.chat_calls[0]["messages"][0]["content"]
        assert "Ты опытный педагог" in system_prompt
        profile_line = runtime.chat_calls[0]["messages"][1]["content"]
        assert "выучить умножение" in profile_line and "космос" in profile_line
        assert "страшные истории" in profile_line
        assert "digit_swap" in profile_line and "разбор по разрядам" in profile_line
        assert "роботы" not in profile_line
        result_block = finished["result"]
        diagnostics = result_block.get("fast_memory_diagnostics") or {}
        assert diagnostics.get("included", {}).get("interests") == 1
        assert diagnostics.get("included", {}).get("avoided_themes") == 1

    def test_failed_validation_fails_job_without_saving(self):
        broken = LESSON_TEXT.replace("## Проверка\n", "").replace("24 × 3 = 72", "24 × 3 = 70")
        store, runtime, fake = WorkerStore(), WorkerRuntime([broken, broken]), FakeUnifiedMemory()
        worker = make_worker(store, runtime, fake)[0]
        store.enqueue("lesson_generate", payload={"topic_id": "math", "minutes": 20,
                                                  "client_request_id": "req-2"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert not fake.lessons  # a failed lesson is never stored
        assert "validation" in json.dumps(finished["error"]["message"]) or "проверки" in finished["error"]["message"] or "Проверка" in finished["error"]["message"]
        assert len(runtime.chat_calls) == 2  # one corrective attempt, no hidden retries

    def test_missing_facade_fails_job_honestly(self):
        store, runtime = WorkerStore(), WorkerRuntime([LESSON_TEXT])
        worker = make_worker(store, runtime, None)[0]
        store.enqueue("lesson_generate", payload={"topic_id": "math", "minutes": 20,
                                                  "client_request_id": "req-3"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert "unified memory backend is unavailable" in finished["error"]["message"]
        assert runtime.chat_calls == []


class TestSessionProcessJob:
    def test_process_session_delegates_with_model_hook(self):
        store, runtime, fake = WorkerStore(), WorkerRuntime([]), FakeUnifiedMemory()
        worker = make_worker(store, runtime, fake)[0]
        store.enqueue("session_process", payload={"learning_session_id": "session-9",
                                                  "client_request_id": "proc-1"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "completed"
        process_calls = [entry for entry in fake.calls if entry[0] == "process_session"]
        assert process_calls and process_calls[0][2] == "session-9"
        assert finished["result"]["session_id"] == "session-9"
        assert finished["result"]["summary"]["summary_id"] == "sum-1"
        assert finished["result"]["calls"] == []  # no model calls were needed by the fake

    def test_session_process_requires_session_id(self):
        store, runtime, fake = WorkerStore(), WorkerRuntime([]), FakeUnifiedMemory()
        worker = make_worker(store, runtime, fake)[0]
        store.enqueue("session_process", payload={})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert "learning_session_id" in finished["error"]["message"]

    def test_process_without_facade_fails_honestly(self):
        store, runtime = WorkerStore(), WorkerRuntime([])
        worker = make_worker(store, runtime, None)[0]
        store.enqueue("session_process", payload={"learning_session_id": "s1",
                                                  "client_request_id": "p1"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert "unified memory backend is unavailable" in finished["error"]["message"]


class TestLegacyKindsUnaffected:
    def test_worker_still_reports_unsupported_legacy_preparation(self):
        store, runtime = WorkerStore(), WorkerRuntime([LESSON_TEXT])
        worker = make_worker(store, runtime, None)[0]
        store.enqueue("memory_update", payload={"learning_session_id": "s1"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed"  # the fake store has no memory snapshot methods
        assert "unified memory backend" not in str(finished["error"])
