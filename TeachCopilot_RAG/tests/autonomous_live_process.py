"""Real API/worker child processes for autonomous browser acceptance.

Only model-facing services are replaced. Routes, sessions, arithmetic, Store and
MemoryWorker are the production implementations. Every forbidden service call
is durably recorded before it raises; readiness is explicitly adapter readiness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AuditLog:
    """Append only safe metadata, never credentials, prompts or learner data."""

    def __init__(self, directory: Path, role: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.role = role
        self.forbidden_calls = 0
        self._lock = threading.Lock()

    def write(self, event: str, *, forbidden: bool = False, **metadata) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "role": self.role,
            "pid": os.getpid(),
            "event": event,
            **metadata,
        }
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        # Windows CRT append writes are not atomic between API and worker.
        # Each process owns its files; the parent aggregates them after exit.
        kind = "forbidden_calls" if forbidden else "process_events"
        filename = f"{kind}-{self.role}-{os.getpid()}.jsonl"
        with self._lock:
            if forbidden:
                self.forbidden_calls += 1
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
            fd = os.open(self.directory / filename, flags, 0o600)
            try:
                if os.write(fd, payload) != len(payload):
                    raise OSError("incomplete acceptance audit write")
                os.fsync(fd)
            finally:
                os.close(fd)


class ForbiddenServices:
    """Unknown service operations fail closed as well as named model methods."""

    def __init__(self, audit: AuditLog, service: str):
        self.audit = audit
        self.service = service

    def __getattr__(self, method: str):
        if method.startswith("__"):
            raise AttributeError(method)

        def forbidden(*args, **kwargs):
            self.audit.write("forbidden_service_call", forbidden=True,
                             service=self.service, method=method)
            raise AssertionError(f"autonomous acceptance forbids {self.service}.{method}")

        return forbidden


class ForbiddenRuntime(ForbiddenServices):
    def __init__(self, audit: AuditLog, settings):
        super().__init__(audit, "runtime")
        self.settings = settings
        self.context_tokens = settings.context_tokens
        self.max_output_tokens = settings.max_output_tokens
        self.generation_scheduler = None
        self.generation_run_budget = None
        self.embedding_prompt_recipe = "forbidden-acceptance-adapter"
        self.embedding_prompt_hash = "forbidden-acceptance-adapter"

    def check_models(self):
        return {"ready": True, "adapter": "forbidden-acceptance-adapter",
                "model_requests_allowed": False}

    def manifest(self):
        return {"local_only": True, "adapter": "forbidden-acceptance-adapter",
                "model_requests_allowed": False,
                "embedding_prompt_recipe": self.embedding_prompt_recipe,
                "embedding_prompt_hash": self.embedding_prompt_hash}


class ForbiddenRag(ForbiddenServices):
    def __init__(self, audit: AuditLog):
        super().__init__(audit, "rag")

    def status(self):
        return {"ready": True, "adapter": "forbidden-acceptance-adapter",
                "retrieval_requests_allowed": False}


def environment():
    # Requiring explicit overrides keeps a misconfigured child away from the
    # normal application database and response journal before dotenv is read.
    required = ("DATABASE_URL", "AUTONOMOUS_LIVE_ARTIFACTS", "LOCAL_MEMORY_RESPONSE_JOURNAL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError("missing explicit acceptance environment: " + ", ".join(missing))
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn

    params = require_local_dsn(os.environ["DATABASE_URL"])
    database_name = params.get("dbname", "")
    if not re.fullmatch(re.escape(DATABASE_PREFIX) + r"[0-9a-f]{12}", database_name):
        raise ValueError("acceptance requires the generated disposable verification database")
    artifacts = Path(os.environ["AUTONOMOUS_LIVE_ARTIFACTS"]).resolve()
    journal = Path(os.environ["LOCAL_MEMORY_RESPONSE_JOURNAL"]).resolve()
    if not journal.is_relative_to(artifacts):
        raise ValueError("acceptance response journal must be inside its artifact directory")
    from dataclasses import replace
    from local_app.config import Settings

    settings = replace(Settings.from_env(), autonomous_division_enabled=True,
                       demo_login_enabled=False)
    return settings, artifacts


def serve_api(settings, store, audit: AuditLog):
    import uvicorn
    from local_app.api import create_app

    port = int(os.environ["AUTONOMOUS_LIVE_PORT"])
    if not 1024 <= port <= 65535:
        raise ValueError("acceptance API port must be between 1024 and 65535")
    app = create_app(settings=settings, store=store,
                     runtime=ForbiddenRuntime(audit, settings),
                     rag_service=ForbiddenRag(audit))
    audit.write("api_start", host="127.0.0.1", port=port,
                production_routes=True, model_requests_allowed=False)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info", access_log=False)


def run_worker(settings, store, audit: AuditLog):
    from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
    from local_app.memory_service import MemoryWorker

    stopping = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopping.set())
    journal = MemoryResponseJournal(settings.memory_response_journal_path,
                                    database_identity=database_fingerprint(store.dsn))
    worker = MemoryWorker(store, ForbiddenRuntime(audit, settings),
                          worker_id=f"autonomous-live-{os.getpid()}", response_journal=journal)
    audit.write("worker_start", production_worker=True, model_requests_allowed=False)
    polls = 0
    idle_polls = 0
    jobs_returned = 0
    next_snapshot = 0.0
    while not stopping.is_set():
        result = worker.run_once()
        polls += 1
        if result is None:
            idle_polls += 1
        else:
            jobs_returned += 1
            audit.write("worker_job_returned", job_id=str(result.get("job_id")),
                        status=result.get("status"), kind=result.get("kind"),
                        runtime_calls=len(worker.calls))
        if audit.forbidden_calls:
            raise AssertionError("worker attempted a forbidden model service")
        now = time.monotonic()
        if now >= next_snapshot or result is not None:
            audit.write("worker_poll_snapshot", polls=polls, idle_polls=idle_polls,
                        jobs_returned=jobs_returned,
                        last_poll_claimed_job=result is not None,
                        forbidden_calls=audit.forbidden_calls)
            next_snapshot = now + 1.0
        stopping.wait(0.2)
    audit.write("worker_stop", polls=polls, idle_polls=idle_polls,
                jobs_returned=jobs_returned, forbidden_calls=audit.forbidden_calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("api", "worker"))
    args = parser.parse_args()
    settings, artifacts = environment()
    audit = AuditLog(artifacts, args.role)
    from local_app.store import Store

    store = Store(settings.database_url)
    store.ping()
    audit.write("database_connected", schema_version=store.schema_version())
    try:
        if args.role == "api":
            serve_api(settings, store, audit)
        else:
            run_worker(settings, store, audit)
    except BaseException as exc:
        audit.write("process_error", error_type=type(exc).__name__)
        raise
    finally:
        audit.write("process_exit", forbidden_calls=audit.forbidden_calls)


if __name__ == "__main__":
    main()
