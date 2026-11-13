"""A memory job that finishes as stale_revision is retried as an explicit new job (bounded), never silently.

Since direct chat requests are persisted, any chat message sent while a job generates bumps the source revision and
used to make the job fail. The retry is a separate job row with its own recorded calls, only for background memory
jobs, and capped, so there is no hidden repeated model request inside one job.
"""
from types import SimpleNamespace

import pytest

from local_app.memory_service import MemoryWorker, STALE_RETRY_LIMIT

STALE = "stale_revision: learner card or learning sources changed during generation"


def job(job_id, *, status="failed", error=STALE, kind="memory_update", provider="local", retries=0, **extra):
    return {"id": job_id, "learner_id": "child", "topic_id": "math", "actor_id": "teacher", "kind": kind,
            "method": "personal_memory", "provider": provider, "status": status, "error": error,
            "payload": {"stale_retry": retries} if retries else {"learning_session_id": "s1"}, **extra}


class FakeStore:
    def __init__(self):
        self.enqueued = []

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system", expected_revision=None,
                           method="tail", idempotency_key=None, provider="local"):
        self.enqueued.append({"learner_id": learner_id, "topic_id": topic_id, "kind": kind, "payload": payload,
                              "actor_id": actor_id, "method": method, "idempotency_key": idempotency_key,
                              "provider": provider})
        return {"id": f"retry-{len(self.enqueued)}"}


def worker(results):
    """A worker whose single-run step returns the given finished jobs in order."""
    w = object.__new__(MemoryWorker)
    w.store = FakeStore()
    queue = list(results)
    w._run_once_single = lambda: queue.pop(0) if queue else None
    return w


def test_limit_is_two_retries():
    assert STALE_RETRY_LIMIT == 2


@pytest.mark.parametrize("finished", [
    job("a", status="completed", error=None),
    job("b", error="something else"),
    job("c", kind="chat"),
    job("d", provider="external"),
    job("e", retries=STALE_RETRY_LIMIT),
    job("f", status="unknown_outcome"),
])
def test_only_stale_background_local_jobs_below_the_limit_are_retryable(finished):
    w = worker([finished])
    assert w.run_once() is finished and w.store.enqueued == []


def test_stale_failure_is_retried_as_a_new_explicit_job_and_the_retry_result_is_returned():
    first, second = job("first"), job("retry-1", status="completed", error=None)
    w = worker([first, second])
    result = w.run_once()
    assert result is second
    [call] = w.store.enqueued
    assert call["idempotency_key"] == "stale-retry:first" and call["kind"] == "memory_update"
    assert call["payload"]["stale_retry"] == 1 and call["payload"]["retry_of"] == "first"
    assert call["payload"]["learning_session_id"] == "s1"       # the original payload is kept
    assert (call["learner_id"], call["topic_id"], call["actor_id"], call["method"], call["provider"]) == \
        ("child", "math", "teacher", "personal_memory", "local")


def test_retries_stop_at_the_limit_and_the_last_stale_failure_is_returned():
    jobs = [job("j0"), job("retry-1", retries=1), job("retry-2", retries=2)]
    w = worker(jobs)
    assert w.run_once() is jobs[-1]
    assert len(w.store.enqueued) == STALE_RETRY_LIMIT


def test_if_another_job_is_claimed_instead_of_the_retry_the_loop_stops():
    first, other = job("first"), job("unrelated", status="completed", error=None)
    w = worker([first, other])
    assert w.run_once() is first            # the retry stays queued for the worker's next cycle
    assert len(w.store.enqueued) == 1


def test_nothing_to_run_returns_none():
    assert worker([]).run_once() is None
