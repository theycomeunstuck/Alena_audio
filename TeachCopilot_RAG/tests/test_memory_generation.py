"""Public runtime admission across processes with synthetic HTTP only."""
from __future__ import annotations

import multiprocessing
import time
from types import SimpleNamespace

import pytest

from local_app.generation_scheduler import GenerationCapacityError, GenerationScheduler
from local_app.runtime import OllamaRuntime, generation_priority
from tests.test_memory_load import queue_store, seed_queue_learner


class _Response:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": '{"answer":"synthetic","citations":[]}'}, "done": True, "done_reason": "stop"}


class _SyntheticHTTP:
    def __init__(self, label, counters, events, release, hold=False):
        self.label, self.counters, self.events = label, counters, events
        self.release, self.hold = release, hold

    def post(self, *args, **kwargs):
        with self.counters.get_lock():
            self.counters[0] += 1
            self.counters[1] = max(self.counters[1], self.counters[0])
            self.counters[2] += 1
        self.events.put(("entered", self.label))
        try:
            if self.hold and not self.release.wait(15):
                raise TimeoutError("test did not release the synthetic provider")
            time.sleep(0.04)
            return _Response()
        finally:
            with self.counters.get_lock():
                self.counters[0] -= 1


def _settings(dsn):
    return SimpleNamespace(database_url=dsn, ollama_url="http://127.0.0.1:11434",
        chat_model="synthetic-no-model", embedding_model="synthetic-no-embedding",
        request_timeout=5.0, context_tokens=8192)


def _run_runtime(dsn, label, priority, counters, events, release, hold=False):
    try:
        runtime = OllamaRuntime(_settings(dsn), session=_SyntheticHTTP(label, counters, events, release, hold))
        with generation_priority(priority):
            runtime.chat_json([{"role": "user", "content": "synthetic lesson"}])
        events.put(("finished", label))
    except Exception as exc:
        events.put(("error", label, type(exc).__name__, str(exc)))


def _run_external(dsn, label, counters, events, release):
    from local_app.answer_adapter import AnswerAdapter, EducationalProjection
    try:
        class Client:
            model = "synthetic-external"

            def complete(self, *args, **kwargs):
                response = _SyntheticHTTP(label, counters, events, release, hold=True).post()
                return SimpleNamespace(text=response.json()["message"]["content"], usage={},
                                       metadata={}, elapsed_seconds=0.04)

        runtime = SimpleNamespace(generation_scheduler=GenerationScheduler(dsn, wait_timeout=8))
        adapter = AnswerAdapter(runtime, Client(), privacy_checker=lambda _: {"safe_to_send": True, "uncertain": False})
        adapter.answer([{"role": "user", "content": "Explain division"}],
            projection=EducationalProjection(topic="division", current_message="Explain division",
                identity_context_complete=True), known_identifiers=("synthetic-learner",))
        events.put(("finished", label))
    except Exception as exc:
        events.put(("error", label, type(exc).__name__, str(exc)))


def _wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Synthetic admission scenario did not reach its expected state")


def _rows(store, where="TRUE"):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM teachcopilot.generation_admission WHERE " + where)
        return cur.fetchall()


def _join(processes, events):
    for process in processes:
        process.join(15)
    result = []
    while not events.empty():
        result.append(events.get(timeout=1))
    assert all(not process.is_alive() and process.exitcode == 0 for process in processes), result
    assert not [row for row in result if row[0] == "error"], result
    return result


def _stop(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
        process.join(5)


def test_actual_local_runtime_across_processes_prioritizes_interactive_over_writer(queue_store):
    ctx = multiprocessing.get_context("spawn")
    counters, events, release = ctx.Array("i", [0, 0, 0]), ctx.Queue(), ctx.Event()
    processes = []
    try:
        holder = ctx.Process(target=_run_runtime, args=(queue_store.dsn,"holder","interactive",counters,events,release,True))
        processes.append(holder); holder.start()
        _wait_until(lambda: counters[2] == 1)
        writer = ctx.Process(target=_run_runtime, args=(queue_store.dsn,"writer","background",counters,events,release))
        processes.append(writer); writer.start()
        _wait_until(lambda: len(_rows(queue_store,"status='waiting' AND priority=0")) == 1)
        interactive = ctx.Process(target=_run_runtime, args=(queue_store.dsn,"interactive","interactive",counters,events,release))
        processes.append(interactive); interactive.start()
        _wait_until(lambda: len(_rows(queue_store,"status='waiting'")) == 2)
        release.set()
        result = _join(processes, events)
        assert [row[1] for row in result if row[0] == "entered"] == ["holder","interactive","writer"]
        assert counters[1] == 1 and counters[2] == 3
        assert not _rows(queue_store)
    finally:
        release.set(); _stop(processes)


def test_external_adapter_admits_four_actual_calls_across_processes(queue_store):
    ctx = multiprocessing.get_context("spawn")
    counters, events, release = ctx.Array("i", [0, 0, 0]), ctx.Queue(), ctx.Event()
    processes = [ctx.Process(target=_run_external,args=(queue_store.dsn,str(index),counters,events,release)) for index in range(5)]
    try:
        for process in processes:
            process.start()
        _wait_until(lambda: counters[0] == 4)
        _wait_until(lambda: len(_rows(queue_store,"lane='external' AND status='waiting'")) == 1)
        assert counters[2] == 4
        release.set()
        _join(processes, events)
        assert counters[1] == 4 and counters[2] == 5
        assert not _rows(queue_store)
    finally:
        release.set(); _stop(processes)


def test_dead_waiter_expires_without_blocking_next_public_runtime_call(queue_store):
    ctx = multiprocessing.get_context("spawn")
    counters, events, release = ctx.Array("i", [0, 0, 0]), ctx.Queue(), ctx.Event()
    processes = []
    try:
        holder = ctx.Process(target=_run_runtime,args=(queue_store.dsn,"holder","interactive",counters,events,release,True))
        processes.append(holder); holder.start()
        _wait_until(lambda: counters[2] == 1)
        crashed = ctx.Process(target=_run_runtime,args=(queue_store.dsn,"crashed","interactive",counters,events,release))
        processes.append(crashed); crashed.start()
        _wait_until(lambda: len(_rows(queue_store,"status='waiting'")) == 1)
        crashed.terminate(); crashed.join(5)
        with queue_store._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE teachcopilot.generation_admission SET lease_until=clock_timestamp()-interval '1 second' WHERE status='waiting'")
        release.set(); holder.join(10)
        runtime = OllamaRuntime(_settings(queue_store.dsn),session=_SyntheticHTTP("new",counters,events,release))
        runtime.chat([{"role":"user","content":"synthetic"}])
        assert counters[1] == 1 and counters[2] == 2
        assert not _rows(queue_store)
    finally:
        release.set(); _stop(processes)


def test_expired_active_lease_keeps_session_lock_until_actual_call_finishes(queue_store):
    ctx = multiprocessing.get_context("spawn")
    counters, events, release = ctx.Array("i", [0, 0, 0]), ctx.Queue(), ctx.Event()
    processes = []
    try:
        holder = ctx.Process(target=_run_runtime,args=(queue_store.dsn,"holder","interactive",counters,events,release,True))
        processes.append(holder); holder.start()
        _wait_until(lambda: counters[2] == 1)
        with queue_store._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE teachcopilot.generation_admission SET lease_until=clock_timestamp()-interval '1 second' WHERE status='active'")
        next_call = ctx.Process(target=_run_runtime,args=(queue_store.dsn,"next","interactive",counters,events,release))
        processes.append(next_call); next_call.start()
        _wait_until(lambda: len(_rows(queue_store,"status='waiting'")) == 1)
        assert counters[2] == 1
        release.set(); _join(processes,events)
        assert counters[1] == 1 and counters[2] == 2
        assert not _rows(queue_store)
    finally:
        release.set(); _stop(processes)


@pytest.mark.parametrize("queued_jobs", [0, 20])
def test_full_actual_generation_queue_refuses_before_http_and_expires_crashed_waiters(queue_store, queued_jobs):
    import uuid
    if queued_jobs:
        seed_queue_learner(queue_store,"shared-cap")
        for index in range(queued_jobs):
            queue_store.enqueue_memory_job("shared-cap","synthetic","next_step",idempotency_key=str(index))
    with queue_store._connect() as conn, conn.cursor() as cur:
        for _ in range(40-queued_jobs):
            cur.execute("""INSERT INTO teachcopilot.generation_admission(id,lane,priority,lease_until)
                VALUES (%s,'local',10,clock_timestamp()+interval '30 seconds')""", (str(uuid.uuid4()),))
    class HTTP:
        calls = 0
        def post(self,*args,**kwargs):
            self.calls += 1
            return _Response()
    http = HTTP()
    runtime = OllamaRuntime(_settings(queue_store.dsn),session=http)
    with pytest.raises(GenerationCapacityError,match="40 waiting"):
        runtime.chat([{"role":"user","content":"synthetic"}])
    assert http.calls == 0
    with queue_store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.generation_admission SET lease_until=clock_timestamp()-interval '1 second'")
    runtime.chat([{"role":"user","content":"synthetic"}])
    assert http.calls == 1 and not _rows(queue_store)


def test_actual_runtime_keeps_single_failed_call_and_conservative_unknown_slot(queue_store):
    class HTTP:
        calls = 0
        def post(self,*args,**kwargs):
            self.calls += 1
            raise TimeoutError("synthetic unknown model outcome")
    http = HTTP()
    runtime = OllamaRuntime(_settings(queue_store.dsn),session=http)
    with pytest.raises(TimeoutError):
        runtime.chat_json([{"role":"user","content":"synthetic"}])
    assert http.calls == 1
    assert len(_rows(queue_store,"status='unknown_outcome'")) == 1
    runtime.generation_scheduler.wait_timeout = 0.05
    with pytest.raises(GenerationCapacityError,match="wait expired"):
        runtime.chat([{"role":"user","content":"new request while old outcome is unknown"}])
    assert http.calls == 1
    with queue_store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.generation_admission SET lease_until=clock_timestamp()-interval '1 second'")
    with pytest.raises(TimeoutError):
        runtime.chat([{"role":"user","content":"explicit later request"}])
    assert http.calls == 2


def test_partial_completion_keeps_literal_response_for_worker_diagnostics():
    from local_app.runtime import IncompleteCompletionError
    raw = {"message":{"content":"partial"},"done_reason":"length","eval_count":1536}
    class HTTP:
        def post(self,*args,**kwargs):
            response = _Response()
            response.json = lambda: raw
            return response
    runtime = OllamaRuntime(_settings(None),session=HTTP())
    with pytest.raises(IncompleteCompletionError) as error:
        runtime.chat([{"role":"user","content":"synthetic"}])
    assert error.value.raw_response == raw
    assert error.value.raw_response is not raw


def test_external_admission_database_failure_prevents_actual_provider_request(monkeypatch):
    import psycopg2
    from unittest.mock import Mock
    from local_app.answer_adapter import AnswerAdapter, EducationalProjection
    scheduler = GenerationScheduler("synthetic-dsn")
    monkeypatch.setattr(scheduler,"_connect",Mock(side_effect=psycopg2.OperationalError("synthetic DB outage")))
    provider = Mock()
    adapter = AnswerAdapter(SimpleNamespace(generation_scheduler=scheduler),provider,
        privacy_checker=lambda _: {"safe_to_send":True,"uncertain":False})
    with pytest.raises(psycopg2.OperationalError):
        adapter.answer([{"role":"user","content":"Explain division"}],
            projection=EducationalProjection(
                topic="division",current_message="Explain division",identity_context_complete=True),
            known_identifiers=("synthetic-learner",))
    provider.complete.assert_not_called()
