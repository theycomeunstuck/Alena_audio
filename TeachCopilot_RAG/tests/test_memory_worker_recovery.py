"""Real worker polling survives a DB disconnect without replaying generation.

Every scenario uses a disposable PostgreSQL database. Model endpoints are
unavailable; an empty tail update needs no model call.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time

import psutil
import psycopg2
from psycopg2.extensions import make_dsn, parse_dsn
import pytest

from local_app.memory_service import MemoryWorker
from local_app.memory_response_journal import MemoryResponseJournal
from scripts.load_memory import ControlledRuntime
from tests.test_memory_load import queue_store, seed_queue_learner  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


class DatabaseGate:
    """Interrupt only this child's loopback DB transport, never PostgreSQL."""
    def __init__(self, dsn):
        params = parse_dsn(dsn)
        assert params["host"] in {"127.0.0.1", "localhost", "::1"}
        self.target = (params["host"], int(params.get("port", 5432)))
        self.allowed = threading.Event()
        self.attempted = threading.Event()
        self.refused = threading.Event()
        self.stopped = threading.Event()
        gate = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                gate.attempted.set()
                if not gate.allowed.is_set():
                    gate.refused.set()
                    return
                try:
                    with socket.create_connection(gate.target, timeout=2) as upstream:
                        sockets = [self.request, upstream]
                        while not gate.stopped.is_set():
                            ready, _, _ = select.select(sockets, [], [], 0.1)
                            for source in ready:
                                data = source.recv(65536)
                                if not data:
                                    return
                                (upstream if source is self.request else self.request).sendall(data)
                except OSError:
                    return

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True

        self.server = Server(("127.0.0.1", 0), Handler)
        self.dsn = make_dsn(dsn, host="127.0.0.1", port=self.server.server_address[1], connect_timeout=1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@contextmanager
def running_worker(dsn, directory, *arguments):
    env = dict(os.environ, DATABASE_URL=dsn, OLLAMA_URL="http://127.0.0.1:9",
               TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG="", LOCAL_GENERATION_SCHEDULER_DSN="",
               LOCAL_GENERATION_RUN_BUDGET_PATH="", PYTHONIOENCODING="utf-8")
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "scripts/memory_worker.py", "--idle-seconds", "0.05", *arguments],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        owned = psutil.Process(process.pid)
        try:
            yield process
        finally:
            try:
                children = [child for child in owned.children(recursive=True)
                            if child.name().casefold() in {"python", "python.exe", "python3"}]
                for child in reversed(children):
                    assert Path(child.cwd()).resolve() == ROOT
                    assert "scripts/memory_worker.py" in child.cmdline()
                    child.terminate()
                    child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
            if process.poll() is None:
                owned.terminate()
            process.wait(timeout=5)


def test_daemon_recovers_db_polling_in_same_process(queue_store, tmp_path):
    store = queue_store
    seed_queue_learner(store, "db-recovery")
    job = store.enqueue_memory_job("db-recovery", "synthetic", "memory_update", idempotency_key="empty-tail")
    with DatabaseGate(store.dsn) as gate, running_worker(gate.dsn, tmp_path) as process:
        assert gate.refused.wait(10), "Worker never encountered its isolated DB outage"
        gate.allowed.set()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            assert process.poll() is None, "Worker exited instead of recovering DB polling; see worker.log"
            if store.get_memory_job(job["id"])["status"] == "completed":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Worker did not process the queued update after DB recovery")
        assert process.poll() is None
        completed = store.get_memory_job(job["id"])
        assert completed["result"]["calls"] == []
        assert completed["call_started_at"] is None
        log = (tmp_path / "worker.log").read_text(encoding="utf-8")
        assert "Database connection unavailable" in log and "Database connection restored" in log


@pytest.mark.parametrize("phase", ["connect", "before_commit", "after_commit"])
@pytest.mark.parametrize("provider", ["local", "external"])
def test_db_loss_after_response_never_replays_or_overwrites_result(queue_store, monkeypatch, phase, provider):
    store = queue_store
    seed_queue_learner(store, "response-db-loss")
    job = store.enqueue_memory_job("response-db-loss", "synthetic", "next_step", provider=provider)
    real_connect = psycopg2.connect
    received = []

    class Runtime(ControlledRuntime):
        def chat(self, *args, **kwargs):
            response = super().chat(*args, **kwargs)
            received.append(response)

            def fail_one_connection(*args, **kwargs):
                monkeypatch.setattr(psycopg2, "connect", real_connect)
                raise psycopg2.OperationalError("Synthetic lost database connection after response")

            if phase == "connect":
                monkeypatch.setattr(psycopg2, "connect", fail_one_connection)
            else:
                lose_commit(monkeypatch, phase=phase)
            return response

    runtime = Runtime(0)
    worker = MemoryWorker(store, runtime)
    with pytest.raises(psycopg2.OperationalError):
        worker.run_once()
    pending = store.get_memory_job(job["id"])
    assert pending["call_started_at"] is not None
    assert pending["status"] == ("completed" if phase == "after_commit" else "running")
    assert worker.run_once() is None
    if phase != "after_commit":
        with store._connect() as conn, conn.cursor() as cur:
            # Advance only this synthetic job's lease; never alter production.
            cur.execute("UPDATE teachcopilot.memory_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (job["id"],))
        assert worker.run_once() is None
        assert store.get_memory_job(job["id"])["status"] == "unknown_outcome"
    else:
        assert len(store.get_learner_memory("response-db-loss", "synthetic")["versions"]) == 1
    assert len(runtime.calls) == 1
    reopened = MemoryResponseJournal(worker.response_journal.path,
                                     database_identity=worker.response_journal.database_identity)
    saved = reopened.list_calls(job_id=job["id"])
    assert len(saved) == 1 and saved[0]["raw_response"] == received[0]
    assert saved[0]["outcome"] == "returned"


def lose_commit(monkeypatch, *, phase, statement=None):
    """Lose one DB acknowledgement around a real PostgreSQL transaction."""
    real_connect = psycopg2.connect
    armed = True

    class Connection:
        def __init__(self, connection):
            self.connection = connection
            self.selected = statement is None

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def cursor(self, *args, **kwargs):
            connection = self

            class Cursor:
                def __init__(self):
                    self.cursor = connection.connection.cursor(*args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self.cursor, name)

                def __enter__(self):
                    self.cursor.__enter__()
                    return self

                def __exit__(self, *exc):
                    return self.cursor.__exit__(*exc)

                def execute(self, query, *args, **kwargs):
                    result = self.cursor.execute(query, *args, **kwargs)
                    if statement and statement in str(query):
                        connection.selected = True
                    return result

            return Cursor()

        def commit(self):
            nonlocal armed
            if armed and self.selected:
                armed = False
                monkeypatch.setattr(psycopg2, "connect", real_connect)
                if phase == "after_commit":
                    self.connection.commit()
                raise psycopg2.OperationalError("Synthetic lost transaction acknowledgement")
            self.connection.commit()

    monkeypatch.setattr(psycopg2, "connect", lambda *args, **kwargs: Connection(real_connect(*args, **kwargs)))


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_interest_call_marker_db_loss_is_not_hidden_as_success(queue_store, monkeypatch, phase):
    store = queue_store
    seed_queue_learner(store, "interest-db-loss")
    session = store.create_learning_session("interest-db-loss", "synthetic", "synthetic")
    store.reserve_chat_message(session["id"], "interest-db-loss", "one", "I enjoy drawing.", "synthetic")
    store.complete_chat_message(session["id"], "one", {"answer": "Synthetic context."})
    ended = store.complete_learning_session(session["id"], "interest-db-loss", "synthetic")
    runtime = ControlledRuntime(0)
    worker = MemoryWorker(store, runtime)
    lose_commit(monkeypatch, phase=phase, statement="SET call_started_at")
    with pytest.raises(psycopg2.OperationalError):
        worker.run_once()
    job = store.get_memory_job(ended["job_id"])
    assert job["status"] == "running"
    assert (job["call_started_at"] is not None) == (phase == "after_commit")
    assert runtime.calls == []


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_lost_claim_acknowledgement_requeues_only_after_unstarted_lease(queue_store, monkeypatch, phase):
    store = queue_store
    seed_queue_learner(store, "claim-db-loss")
    job = store.enqueue_memory_job("claim-db-loss", "synthetic", "next_step")
    runtime = ControlledRuntime(0)
    worker = MemoryWorker(store, runtime)
    lose_commit(monkeypatch, phase=phase, statement="SET status='running'")
    with pytest.raises(psycopg2.OperationalError):
        worker.run_once()
    pending = store.get_memory_job(job["id"])
    assert pending["status"] == ("queued" if phase == "before_commit" else "running")
    assert pending["call_started_at"] is None and runtime.calls == []
    if phase == "after_commit":
        assert worker.run_once() is None
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE teachcopilot.memory_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (job["id"],))
    assert worker.run_once()["status"] == "completed"
    assert len(runtime.calls) == 1


def test_finite_worker_exits_on_db_loss_without_silent_retry(queue_store, tmp_path):
    with DatabaseGate(queue_store.dsn) as gate, running_worker(gate.dsn, tmp_path, "--max-jobs", "1") as process:
        assert process.wait(timeout=10) == 1
        assert gate.refused.is_set()
    log = (tmp_path / "worker.log").read_text(encoding="utf-8")
    assert "finite worker stopped without replay" in log
    assert "Traceback" not in log


def test_daemon_stops_before_provider_when_response_journal_is_unavailable(queue_store, tmp_path, monkeypatch):
    store = queue_store
    seed_queue_learner(store, "journal-unavailable")
    job = store.enqueue_memory_job("journal-unavailable", "synthetic", "next_step")
    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"preserve this file")
    monkeypatch.setenv("LOCAL_MEMORY_RESPONSE_JOURNAL", str(blocked / "responses.sqlite3"))
    with running_worker(store.dsn, tmp_path) as process:
        assert process.wait(timeout=10) == 1
    saved = store.get_memory_job(job["id"])
    assert saved["status"] == "running" and saved["call_started_at"] is None
    assert blocked.read_bytes() == b"preserve this file"
    log = (tmp_path / "worker.log").read_text(encoding="utf-8")
    assert "response journal unavailable" in log and "stopped without replay" in log
    assert "Traceback" not in log


def test_polling_backoff_is_capped_resets_and_redacts_connection_details(queue_store, monkeypatch, capsys):
    from scripts import memory_worker
    real_connect = psycopg2.connect
    events = iter(["fail"] * 9 + ["healthy", "fail", "stop"])
    delays = []
    secret = "synthetic-password-must-not-be-logged"

    def connect(*args, **kwargs):
        event = next(events)
        if event == "fail":
            raise psycopg2.OperationalError(secret)
        if event == "stop":
            raise RuntimeError("end of finite clock fixture")
        return real_connect(*args, **kwargs)

    monkeypatch.setenv("DATABASE_URL", queue_store.dsn)
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LOCAL_GENERATION_SCHEDULER_DSN", "")
    monkeypatch.setenv("LOCAL_GENERATION_RUN_BUDGET_PATH", "")
    monkeypatch.setattr(psycopg2, "connect", connect)
    monkeypatch.setattr(memory_worker.time, "sleep", delays.append)
    monkeypatch.setattr(sys, "argv", ["memory_worker.py", "--idle-seconds", "0.25"])
    with pytest.raises(RuntimeError, match="end of finite clock fixture"):
        memory_worker.main()
    assert delays == [0.25, 0.5, 1, 2, 4, 8, 16, 30, 30, 0.25, 0.25]
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert "Database connection restored" in output.err


@pytest.mark.parametrize("error", [psycopg2.ProgrammingError, ValueError])
def test_non_connection_errors_are_not_silently_polled(monkeypatch, error):
    from scripts import memory_worker
    delays = []

    def fail(*args, **kwargs):
        raise error("Synthetic schema or programming fault")

    monkeypatch.setenv("DATABASE_URL", "host=127.0.0.1 dbname=synthetic_unreachable")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LOCAL_GENERATION_SCHEDULER_DSN", "")
    monkeypatch.setenv("LOCAL_GENERATION_RUN_BUDGET_PATH", "")
    monkeypatch.setattr(psycopg2, "connect", fail)
    monkeypatch.setattr(memory_worker.time, "sleep", delays.append)
    monkeypatch.setattr(sys, "argv", ["memory_worker.py"])
    with pytest.raises(error, match="Synthetic schema or programming fault"):
        memory_worker.main()
    assert delays == []
