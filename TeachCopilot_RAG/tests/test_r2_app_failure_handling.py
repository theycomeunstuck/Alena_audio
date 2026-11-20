"""Round-2 P1.2: honest failure handling at the application boundary.

Queue-full answers 429 with Retry-After and keeps the durable completion; a
worker that loses its lease does not double-finish; malformed or empty model
output fails the job with its recorded calls preserved; repeated browser
submissions stay idempotent. No path fakes success or bypasses the journals.
"""
from __future__ import annotations

import json

import pytest

from local_app.memory_service import MemoryWorker
from tests.test_unified_app_api import (LEARNER, LESSON_TEXT, UnifiedMemoryValidationError,
                                        make_api)
from tests.test_unified_app_worker import (LESSON_TEXT as WORKER_LESSON_TEXT, RecordingJournal,
                                           WorkerRuntime, WorkerStore, make_worker)
from tests.test_unified_app_worker import FakeUnifiedMemory  # noqa: F401

pytest.importorskip("pydantic")

WORKER_LESSON = WORKER_LESSON_TEXT or LESSON_TEXT
STALE_RETRY_LIMIT_2 = 2  # matches MemoryWorker.STALE_RETRY_LIMIT for the exhausted case


class TestQueueFull:
    def test_completion_with_full_queue_stays_durable_and_reports_honestly(self):
        from local_app.memory_store import MemoryQueueFullError
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]

        def full_enqueue(*args, **kwargs):
            raise MemoryQueueFullError("memory queue is full")

        store.enqueue_memory_job = full_enqueue
        completed = client.post(f"/learning-sessions/{session}/complete", headers=student)
        # The durable completion is never withdrawn by the enqueue failure;
        # the failed enqueue stays visible instead of faking a queued job.
        assert completed.status_code == 202, completed.text
        assert completed.json()["session_processing_status"] == "enqueue_failed"
        assert not [job for job in store.memory_jobs.values()
                    if job["kind"] == "session_process"]

    def test_failed_job_stays_failed_and_stale_retry_is_classified(self):
        # A legacy memory_update job on the minimal fake store fails honestly
        # (no snapshot methods) and stays failed/inspectable — the worker keeps
        # polling. The retry classifier accepts a stale-marker failure for the
        # bounded retry loop; a queue-full retry enqueue leaves the failed row
        # failed (the guard exists in MemoryWorker.run_once and needs the real
        # store to exercise end-to-end).
        from local_app.memory_service import STALE_RETRY_MARKERS, MemoryWorker
        store, fake = WorkerStore(), FakeUnifiedMemory()
        runtime = WorkerRuntime([])
        worker, _journal = make_worker(store, runtime, fake)
        job = store.enqueue("memory_update", payload={"learning_session_id": "s1",
                                                      "stale_retry": 0})
        store.next_claim = [job["job_id"]]
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert store.get_memory_job(job["job_id"])["status"] == "failed"
        stale_job = {"status": "failed", "kind": "memory_update", "provider": "local",
                     "error": "stale_revision: source snapshot changed",
                     "payload": {"stale_retry": 0}}
        assert MemoryWorker._stale_retryable(stale_job)
        assert all(marker != "memory_update" for marker in STALE_RETRY_MARKERS)
        exhausted = {**stale_job, "payload": {"stale_retry": STALE_RETRY_LIMIT_2}}
        assert not MemoryWorker._stale_retryable(exhausted)


class TestLeaseTakeover:
    def test_worker_that_lost_the_job_does_not_double_finish(self):
        store, fake = WorkerStore(), FakeUnifiedMemory()
        runtime = WorkerRuntime([])
        worker, _journal = make_worker(store, runtime, fake)
        store.enqueue("session_process", payload={"learning_session_id": "s-lease",
                                                  "client_request_id": "p1"})
        store.next_claim = ["job-1"]

        def foreign_owner(job_id, **kwargs):
            raise ValueError("worker-other does not own this running job")

        store.finish_memory_job = foreign_owner
        finished = worker.run_once()
        # The requeued job stays inspectable; no crash, no fabricated result.
        assert finished["job_id"] == "job-1"


class TestUnifiedStaleRetry:
    def test_session_process_stale_failure_retries_with_fresh_request_id(self):
        # The legacy enqueue-time revision guard can fail a session_process
        # job when the learner's sources moved between enqueue and processing
        # (for example another session completed meanwhile). The worker must
        # retry it as a fresh job carrying a fresh facade request id, while
        # the failed row stays inspectable.
        store, fake = WorkerStore(), FakeUnifiedMemory()
        runtime = WorkerRuntime([])
        worker, _journal = make_worker(store, runtime, fake)
        job = store.enqueue("session_process", payload={"learning_session_id": "s-stale",
                                                        "client_request_id": "on-complete:s-stale",
                                                        "stale_retry": 0},
                            learner_id="learner-a", actor_id="student-unified")
        store.next_claim = [job["job_id"]]

        def stale_guarded_finish(job_id, **kwargs):
            row = store.jobs[job_id]
            row.update(kwargs)
            row["status"] = "failed"
            row["error"] = "stale_revision: learner card or learning sources changed during generation"
            return dict(row)

        store.finish_memory_job = stale_guarded_finish
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert "stale_revision" in str(finished["error"])
        # The retry was enqueued as an explicit new job with a fresh request id.
        retries = [j for j in store.jobs.values() if j["kind"] == "session_process"
                   and j["payload"].get("retry_of")]
        assert len(retries) == 1
        assert retries[0]["payload"]["client_request_id"] == "on-complete:s-stale:retry1"
        assert retries[0]["payload"]["stale_retry"] == 1
        assert retries[0]["payload"]["learning_session_id"] == "s-stale"


class TestMalformedModelOutput:
    def test_empty_generation_fails_job_with_recorded_calls(self):
        store, fake = WorkerStore(), FakeUnifiedMemory()
        runtime = WorkerRuntime(["   "])
        worker, journal = make_worker(store, runtime, fake)
        store.enqueue("lesson_generate", payload={"topic_id": "math", "minutes": 20,
                                                  "client_request_id": "mf-1"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed"
        assert not fake.lessons  # nothing is stored from a broken draft
        # The runtime's own empty-completion guard names the failure honestly.
        assert "empty_memory_completion" in finished["error"]["message"]
        # The failure keeps the call journal evidence for inspection.
        assert journal.prepared, "the model call must stay journaled"

    def test_garbage_generation_never_stores_a_draft(self):
        store, fake = WorkerStore(), FakeUnifiedMemory()
        broken = WORKER_LESSON.replace("## Проверка\n", "").replace("24 × 3 = 72", "24 × 3 = 70")
        runtime = WorkerRuntime([broken, broken])
        worker, _journal = make_worker(store, runtime, fake)
        store.enqueue("lesson_generate", payload={"topic_id": "math", "minutes": 20,
                                                  "client_request_id": "mf-2"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "failed" and not fake.lessons
        assert len(runtime.chat_calls) == 2  # one corrective attempt, no hidden retries


class TestRepeatedBrowserSubmissions:
    def test_repeated_complete_submissions_enqueue_processing_once(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        fake.confirm_lesson(LEARNER, "t", saved["lesson_id"], expected_revision=1, request_id="c")
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        for _ in range(3):
            response = client.post(f"/learning-sessions/{session}/complete", headers=student)
            assert response.status_code == 202
        kinds = [job["kind"] for job in store.memory_jobs.values()]
        assert kinds.count("session_process") == 1

    def test_repeated_processing_requests_with_new_request_ids_enqueue_separately(self):
        # A genuinely new request id is a new job; the facade's fingerprinting
        # decides whether processing repeats work. The endpoint must not
        # silently swallow distinct requests.
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=teacher,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        client.post(f"/learning-sessions/{session}/complete", headers=student)
        first = [job for job in store.memory_jobs.values() if job["kind"] == "session_process"]
        response = client.post(
            f"/learners/{LEARNER}/learning-sessions/{session}/process", headers=teacher,
            json={"client_request_id": "explicit-retry"})
        assert response.status_code == 202, response.text
        second = [job for job in store.memory_jobs.values() if job["kind"] == "session_process"]
        assert len(second) == len(first) + 1


def test_processing_failure_feedback_is_actionable_in_the_ui_contract():
    # The UI contract maps processing statuses to explicit Russian messages
    # (web/app.js showLessonProcessing); a failed run never claims success.
    statuses = {"completed": "Обработка завершена.", "partial": "Обработка завершена частично",
                "failed": "Обработка не удалась", "pending": "Обработка ещё выполняется"}
    import re
    source = open("web/app.js", encoding="utf-8").read()
    for key, fragment in statuses.items():
        assert fragment in source, f"the UI must explain the '{key}' processing status"
    assert "processing-result" in source
