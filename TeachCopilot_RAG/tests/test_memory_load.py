"""Guard the honesty and safety of the finite simulated load harness."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_memory import ControlledRuntime, percentile, require_disposable_dsn, timing_summary


def test_timing_metrics_retain_missing_data_and_interpolate_p95():
    assert timing_summary([]) == {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    measured = timing_summary([4, 1, 3, 2])
    assert measured["count"] == 4
    assert measured["p50_ms"] == 2.5
    assert measured["p95_ms"] == pytest.approx(3.85)
    assert measured["max_ms"] == 4
    with pytest.raises(ValueError):
        percentile([1], 1.1)


@pytest.mark.parametrize("dsn", [
    "postgresql://user@127.0.0.1:5434/teachcopilot_local",
    "postgresql://user@example.org/teachcopilot_memory_verify_0123456789ab",
    "postgresql://user@127.0.0.1/teachcopilot_memory_verify_0123456789xy",
    "postgresql://user@127.0.0.1/teachcopilot_memory_verify_0123",
])
def test_load_rejects_existing_shared_and_nonlocal_databases(dsn):
    with pytest.raises(ValueError):
        require_disposable_dsn(dsn)


def test_load_accepts_only_disposable_loopback_database_name():
    require_disposable_dsn("postgresql://user@127.0.0.1:5434/teachcopilot_memory_verify_0123456789ab")


def test_controlled_provider_cites_only_the_actual_request_schema():
    runtime = ControlledRuntime(delay_ms=0)
    schema = {"properties": {"evidence_turn_ids": {"items": {"enum": ["learner-a-source"]}}}}
    result = runtime.chat([{"role": "user", "content": "synthetic source"}], schema, 1536)
    action = json.loads(result["message"]["content"])
    assert action["evidence_turn_ids"] == ["learner-a-source"]
    assert action["review_required"] is True
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["messages"][0]["content"] == "synthetic source"
    extract = runtime.chat([], {"properties": {"records": {"type": "array"}}}, 1536)
    assert json.loads(extract["message"]["content"]) == {"records": []}


def test_controlled_provider_measures_concurrency_and_retains_every_call():
    runtime = ControlledRuntime(delay_ms=20)
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda index: runtime.chat([{"role": "user", "content": str(index)}]), range(8)))
    assert len(responses) == len(runtime.calls) == 8
    assert runtime.active == 0
    assert 1 < runtime.peak_active <= 4
    assert all(row["elapsed_ms"] >= 15 for row in runtime.calls)


@pytest.fixture
def queue_store():
    """Every queue scenario owns a separate DB, including its pending jobs."""
    dsn = os.getenv("TEACHCOPILOT_TEST_DSN")
    if not dsn:
        pytest.skip("set TEACHCOPILOT_TEST_DSN for local PostgreSQL queue checks")
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from local_app.store import Store
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn
    require_local_dsn(dsn)
    database = DATABASE_PREFIX + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(dsn, dbname="postgres", connect_timeout=5))
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        store = Store(make_dsn(dsn, dbname=database))
        store.migrate()
        yield store
    finally:
        assert database.startswith(DATABASE_PREFIX) and len(database) == len(DATABASE_PREFIX) + 12
        with admin.cursor() as cursor:
            cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
            cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        admin.close()


def seed_queue_learner(store, learner):
    from psycopg2.extras import Json
    card = {"learner_id": learner, "grade": 4, "language": "russian", "interests": []}
    with store._connect() as conn, conn.cursor() as cursor:
        cursor.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,%s,4,%s)",
                       (learner, learner, Json(card)))


def job_id(job):
    return job.get("job_id") or job["id"]


def test_real_queue_rejects_41st_waiting_request_but_accepts_idempotent_retry(queue_store):
    from local_app.memory_store import MemoryQueueFullError
    seed_queue_learner(queue_store, "queue-cap")
    first = None
    for index in range(40):
        job = queue_store.enqueue_memory_job("queue-cap", "synthetic", "next_step", idempotency_key=str(index))
        first = first or job
    repeat = queue_store.enqueue_memory_job("queue-cap", "synthetic", "next_step", idempotency_key="0")
    assert job_id(repeat) == job_id(first)
    with pytest.raises(MemoryQueueFullError):
        queue_store.enqueue_memory_job("queue-cap", "synthetic", "next_step", idempotency_key="overflow")


def test_real_queue_enforces_provider_limits_and_serializes_each_learner(queue_store):
    for index in range(7):
        seed_queue_learner(queue_store, f"queue-limit-{index}")
    queue_store.enqueue_memory_job("queue-limit-0", "synthetic", "next_step")
    queue_store.enqueue_memory_job("queue-limit-1", "synthetic", "next_step")
    local = queue_store.claim_memory_job("local-owner")
    assert local["provider"] == "local"
    assert queue_store.claim_memory_job("blocked-local") is None
    queue_store.finish_memory_job(job_id(local), result={}, worker_id="local-owner")
    second = queue_store.claim_memory_job("local-second")
    queue_store.finish_memory_job(job_id(second), result={}, worker_id="local-second")
    for index in range(2, 7):
        queue_store.enqueue_memory_job(f"queue-limit-{index}", "synthetic", "next_step", provider="external")
    external = [queue_store.claim_memory_job("external-" + str(index)) for index in range(4)]
    assert all(row and row["provider"] == "external" for row in external)
    assert queue_store.claim_memory_job("blocked-external") is None
    queue_store.finish_memory_job(job_id(external[0]), result={}, worker_id="external-0")
    fifth = queue_store.claim_memory_job("external-fifth")
    assert fifth and fifth["provider"] == "external"
    # A second job for an already running learner cannot occupy the freed slot.
    queue_store.enqueue_memory_job(external[1]["learner_id"], "synthetic", "next_step", provider="external")
    queue_store.finish_memory_job(job_id(fifth), result={}, worker_id="external-fifth")
    assert queue_store.claim_memory_job("same-learner-must-wait") is None


def test_real_queue_prioritizes_interactive_job_before_older_memory_update(queue_store):
    seed_queue_learner(queue_store, "queue-background")
    seed_queue_learner(queue_store, "queue-interactive")
    background = queue_store.enqueue_memory_job("queue-background", "synthetic", "memory_update")
    interactive = queue_store.enqueue_memory_job("queue-interactive", "synthetic", "next_step")
    claimed = queue_store.claim_memory_job("priority-check")
    assert job_id(claimed) == job_id(interactive)
    queue_store.finish_memory_job(job_id(claimed), result={}, worker_id="priority-check")
    assert job_id(queue_store.claim_memory_job("background-check")) == job_id(background)


def test_real_worker_restart_requeues_only_before_call_and_preserves_unknown_outcome(queue_store):
    seed_queue_learner(queue_store, "queue-restart")
    queued = queue_store.enqueue_memory_job("queue-restart", "synthetic", "next_step", provider="external")
    claimed = queue_store.claim_memory_job("crashed-before-call")
    with queue_store._connect() as conn, conn.cursor() as cursor:
        cursor.execute("UPDATE teachcopilot.memory_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (job_id(claimed),))
    assert queue_store.recover_memory_jobs() == {"unknown_outcome": 0, "requeued": 1}
    reclaimed = queue_store.claim_memory_job("crashed-after-call")
    assert job_id(reclaimed) == job_id(queued)
    queue_store.start_memory_job_call(job_id(reclaimed), "crashed-after-call")
    with queue_store._connect() as conn, conn.cursor() as cursor:
        cursor.execute("UPDATE teachcopilot.memory_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (job_id(reclaimed),))
    assert queue_store.recover_memory_jobs() == {"unknown_outcome": 1, "requeued": 0}
    assert queue_store.get_memory_job(job_id(queued))["status"] == "unknown_outcome"
    assert queue_store.claim_memory_job("no-paid-replay") is None


def test_real_worker_keeps_learner_sources_out_of_other_actual_prompts(queue_store):
    from local_app.memory_service import MemoryWorker
    for learner in ("isolation-a", "isolation-b"):
        seed_queue_learner(queue_store, learner)
        queue_store.add_memory_source(learner, "synthetic", "Unique synthetic marker " + learner,
                                      "teacher", approved=True)
        queue_store.enqueue_memory_job(learner, "synthetic", "next_step")
    runtime = ControlledRuntime(0)
    worker = MemoryWorker(queue_store, runtime)
    first = worker.run_once()
    second = worker.run_once()
    assert first["status"] == second["status"] == "completed", (first.get("error"), second.get("error"))
    assert len(runtime.calls) == 2
    for call in runtime.calls:
        actual_payload = json.dumps(call["messages"], ensure_ascii=False)
        assert ("isolation-a" in actual_payload) != ("isolation-b" in actual_payload)
    assert all(row["author_role"] != "assistant" for row in queue_store.get_learner_memory("isolation-a")["sources"])


def test_real_worker_preserves_failed_provider_and_database_outcomes(queue_store, monkeypatch):
    import psycopg2
    from local_app.memory_service import MemoryWorker

    class UnavailableRuntime:
        context_tokens = 8192
        calls = 0

        def chat(self, *args, **kwargs):
            self.calls += 1
            raise TimeoutError("controlled provider outage")

    runtime = UnavailableRuntime()
    for learner, provider in (("outage-local", "local"), ("outage-external", "external")):
        seed_queue_learner(queue_store, learner)
        queue_store.enqueue_memory_job(learner, "synthetic", "next_step", provider=provider)
    worker = MemoryWorker(queue_store, runtime)
    results = [worker.run_once(), worker.run_once()]
    assert {row["status"] for row in results} == {"failed", "unknown_outcome"}
    assert runtime.calls == 2
    assert worker.run_once() is None
    seed_queue_learner(queue_store, "outage-db")
    job = queue_store.enqueue_memory_job("outage-db", "synthetic", "next_step")
    with monkeypatch.context() as temporary:
        temporary.setattr(queue_store, "_connect", lambda: (_ for _ in ()).throw(psycopg2.OperationalError("controlled DB outage")))
        with pytest.raises(psycopg2.OperationalError):
            worker.run_once()
    assert queue_store.get_memory_job(job_id(job))["status"] == "queued"
    assert runtime.calls == 2
