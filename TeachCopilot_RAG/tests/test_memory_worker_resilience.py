"""Memory worker resilience: lease-takeover failures, full-queue stale retries, drift retries.

Drives the real MemoryWorker.run_once with an in-memory fake store that mimics only
the documented local_app/memory_store.py behaviours under test (the finish_memory_job
ownership check at line 1303 and the queue cap at line 966). This is the
same pure-Python pattern as .audit/20261115-0002/repro/V6/repro_cl2_cl4_cl5_worker_crashes.py:
no database, no network.

Covers audit findings:
- F-028: the generic failure handler's compensating finish_memory_job must treat the
  lease-takeover ownership rejection ("worker does not own this running job") as
  benign instead of crashing the polling worker.
- F-030: a MemoryQueueFullError from the stale-retry enqueue must not escape run_once
  and lose the retry; the job stays failed and the worker keeps polling.
- F-016: _stale_retryable must also match the three real pre-generation drift errors
  (personal_memory.py, personal_graph_memory.py, memory_service.py), which never carry
  the store's finish-time "stale_revision" marker.

Test-only import shim: the committed research_memory.methods does not define
APPLICATION_METHODS, which local_app.memory_service imports (audit finding F-004).
The shim below mirrors the user's uncommitted methods.py line 18
(`APPLICATION_METHODS = METHODS + ("structured_v2",)`) verbatim, exactly as the
verified V6 repro applies it. It is guarded so it becomes a no-op once the real
module exports the name, and it is a test-only patch – not a product change. It
must run before local_app.memory_service is first imported in this process; pytest
collects this file first in the worker test batch, which also lets the pre-existing
memory worker test modules collect at this commit.
"""
import logging
from types import SimpleNamespace

# --- test-only shim; mirrors the user's uncommitted research_memory/methods.py line 18 ---
import research_memory.methods as _research_methods
if not hasattr(_research_methods, "APPLICATION_METHODS"):
    _research_methods.APPLICATION_METHODS = _research_methods.METHODS + ("structured_v2",)

import pytest

from local_app.memory_service import MemoryWorker, STALE_RETRY_LIMIT
from local_app.memory_store import MemoryQueueFullError

# The three real drift failures that never carry the store's "stale_revision" marker.
DRIFT_MESSAGES = (
    "source snapshot changed before personal memory selection",      # personal_memory.py:405
    "source snapshot changed before personal graph indexing",        # personal_graph_memory.py:973
    "learner revision changed before personal graph indexing",       # memory_service.py:390
)


class FakeStore:
    """In-memory stand-in mimicking only the memory_store.py behaviours under test."""

    def __init__(self, *, rows=(), finish_raises=None, stale_on_finish=False, queue_full=False):
        self.jobs = {row["job_id"]: dict(row) for row in rows}
        self.finish_raises = finish_raises        # a store fault raised inside finish_memory_job
        self.stale_on_finish = stale_on_finish    # the memory_store.py:1309 marker path
        self.queue_full = queue_full              # the memory_store.py:966 waiting-request cap
        self.enqueued = []

    def claim_memory_job(self, worker_id, lease_seconds=120):
        for job in self.jobs.values():
            if job["status"] == "queued":
                job["status"] = "running"
                job["worker_id"] = str(worker_id)
                return dict(job)
        return None

    def get_memory_job(self, job_id):
        job = self.jobs.get(job_id)
        return dict(job) if job else None

    def finish_memory_job(self, job_id, *, result=None, error=None, unknown_outcome=False,
                          worker_id=None, memory_state=None, next_step=None, context=None,
                          approved=False):
        job = self.jobs[job_id]
        if self.finish_raises is not None:
            raise self.finish_raises
        if job["status"] in ("completed", "failed", "unknown_outcome"):
            return dict(job)
        # memory_store.py:1302-1303 – after a lease-expiry takeover requeued the row
        # under another claim, the losing worker's finish is rejected.
        if job["status"] != "running" or job.get("worker_id") != str(worker_id):
            raise ValueError("worker does not own this running job")
        if error is None and self.stale_on_finish:
            # memory_store.py:1307-1309 – learner revisions moved during generation.
            error = "stale_revision: learner card or learning sources changed during generation"
        job["status"] = ("unknown_outcome" if unknown_outcome
                         else "failed" if error else "completed")
        job["error"] = str(error) if error else None
        job["result"] = result
        return dict(job)

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system",
                           expected_revision=None, method="tail", idempotency_key=None,
                           provider="local"):
        if self.queue_full:
            raise MemoryQueueFullError("memory queue contains 40 waiting requests; retry later")
        self.enqueued.append({"learner_id": learner_id, "topic_id": topic_id, "kind": kind,
                              "payload": dict(payload or {}), "actor_id": actor_id,
                              "method": method, "idempotency_key": idempotency_key,
                              "provider": provider})
        retry_id = f"retry-{len(self.enqueued)}"
        self.jobs[retry_id] = job_row(retry_id, kind=kind, method=method, payload=payload or {})
        return {"id": retry_id, "job_id": retry_id}


def job_row(job_id, *, kind="memory_update", method="tail", payload=None, status="queued"):
    """Real job rows carry both keys (memory_store._row sets job_id = id)."""
    return {"id": job_id, "job_id": job_id, "learner_id": "child", "topic_id": "math",
            "actor_id": "teacher", "kind": kind, "method": method, "provider": "local",
            "payload": dict(payload or {}), "status": status, "error": None, "worker_id": None}


def build_worker(store):
    return MemoryWorker(store, SimpleNamespace(context_tokens=8192), worker_id="w1",
                        response_journal=object())


def prepare_raises(message):
    def prepare(job):
        raise ValueError(message)
    return prepare


def prepare_completes(job):
    """A tail generation that succeeds; the store then finds the learner revision moved."""
    return None, {"text": "x", "evidence_ids": [], "sources": [], "method": "tail",
                  "topic_id": "t"}


# --- F-028: lease-takeover rejection of the compensating failure record ---


def test_lease_takeover_rejecting_the_failure_record_is_benign(caplog):
    store = FakeStore(rows=[job_row("J1")])
    worker = build_worker(store)

    def prepare(job):
        # A concurrent claimer's lease-expiry recovery requeued the row between our
        # claim and our finish (memory_store.py:1010-1011); the heartbeat in _call then
        # reports memory_job_lease_expired (memory_service.py:317-319).
        store.jobs[job["job_id"]]["status"] = "queued"
        raise ValueError("memory_job_lease_expired")

    worker._prepare = prepare
    with caplog.at_level(logging.WARNING):
        result = worker.run_once()
    assert result["status"] == "queued"            # reported as-is, left for the new claim
    assert "lease takeover" in caplog.text
    assert store.enqueued == []


def test_failure_record_without_takeover_still_fails_the_job():
    store = FakeStore(rows=[job_row("J1b")])
    worker = build_worker(store)
    worker._prepare = prepare_raises("memory_job_lease_expired")
    result = worker.run_once()
    assert result["status"] == "failed"
    assert "memory_job_lease_expired" in result["error"]


def test_only_the_ownership_rejection_is_treated_as_benign():
    store = FakeStore(rows=[job_row("J1c")], finish_raises=ValueError("synthetic store fault"))
    worker = build_worker(store)
    worker._prepare = prepare_raises("boom")
    with pytest.raises(ValueError, match="synthetic store fault"):
        worker.run_once()


# --- F-030: stale-retry enqueue against a full queue ---


def test_full_queue_leaves_the_stale_job_failed_and_the_worker_alive(caplog):
    store = FakeStore(rows=[job_row("J2")], stale_on_finish=True, queue_full=True)
    worker = build_worker(store)
    worker._prepare = prepare_completes
    with caplog.at_level(logging.WARNING):
        result = worker.run_once()
    assert result["status"] == "failed"
    assert "stale_revision" in result["error"]
    assert store.enqueued == []
    assert "memory queue is full" in caplog.text


def test_stale_retry_control_enqueues_bounded_retries_when_queue_has_room():
    store = FakeStore(rows=[job_row("J3")], stale_on_finish=True)
    worker = build_worker(store)
    worker._prepare = prepare_completes
    result = worker.run_once()
    assert result["status"] == "failed"
    assert result["payload"]["stale_retry"] == STALE_RETRY_LIMIT
    assert len(store.enqueued) == STALE_RETRY_LIMIT


# --- F-016: the three real drift failures must become bounded stale retries ---


@pytest.mark.parametrize("message", DRIFT_MESSAGES)
def test_each_real_drift_error_is_stale_retryable(message):
    stored_error = str({"type": "ValueError", "message": message, "calls": []})
    job = {"status": "failed", "kind": "next_step", "provider": "local",
           "error": stored_error, "payload": {}}
    assert MemoryWorker._stale_retryable(job)


@pytest.mark.parametrize("message", DRIFT_MESSAGES)
def test_drift_failure_is_retried_as_bounded_explicit_jobs(message):
    store = FakeStore(rows=[job_row("J4", kind="next_step", method="personal_memory",
                                    payload={"request": "next step"})])
    worker = build_worker(store)
    worker._prepare = prepare_raises(message)      # the provider drift check inside _prepare
    result = worker.run_once()
    assert result["status"] == "failed"
    assert message in result["error"]
    assert len(store.enqueued) == STALE_RETRY_LIMIT
    [first] = [call for call in store.enqueued if call["idempotency_key"] == "stale-retry:J4"]
    assert first["payload"]["stale_retry"] == 1 and first["payload"]["retry_of"] == "J4"
    assert first["payload"]["request"] == "next step"   # the original payload is kept
    assert (first["kind"], first["method"], first["provider"], first["actor_id"]) == \
        ("next_step", "personal_memory", "local", "teacher")


def test_store_marker_stays_retryable():
    job = {"status": "failed", "kind": "memory_update", "provider": "local",
           "error": "stale_revision: learner card or learning sources changed during generation",
           "payload": {}}
    assert MemoryWorker._stale_retryable(job)


def test_unrelated_failures_are_never_retried():
    store = FakeStore(rows=[job_row("J5", kind="next_step", method="personal_memory")])
    worker = build_worker(store)
    worker._prepare = prepare_raises("learner_not_found")
    result = worker.run_once()
    assert result["status"] == "failed"
    assert store.enqueued == []
