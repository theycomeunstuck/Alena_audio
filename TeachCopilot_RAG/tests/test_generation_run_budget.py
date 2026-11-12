"""Free synthetic checks of durable local-run accounting and runtime opt-in."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest

from local_app.config import Settings
from local_app.generation_run_budget import (
    GenerationRunBudget, GenerationRunBudgetError, GenerationRunBudgetExceeded,
)
from local_app.runtime import OllamaRuntime


def _reserve_many(path, count, events):
    journal = GenerationRunBudget(path)
    accepted = 0
    for _ in range(count):
        try:
            journal.reserve_generation({"model": "synthetic", "messages": []})
            accepted += 1
        except GenerationRunBudgetExceeded:
            pass
    events.put(accepted)


def _rows(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM calls ORDER BY id")]


def test_concurrent_processes_share_limit_and_unknown_reservations(tmp_path):
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    original = journal.initialize(max_calls=9)
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    processes = [context.Process(target=_reserve_many, args=(path, 8, events)) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(15)
        assert all(process.exitcode == 0 for process in processes)
        assert sum(events.get(timeout=2) for _ in processes) == 9
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
    reopened = GenerationRunBudget(path)
    after = reopened.initialize(max_calls=9)
    assert after["generation_calls"] == 9
    assert after["unknown_outcomes"] == 9
    assert after["deadline_at"] == original["deadline_at"]
    with pytest.raises(GenerationRunBudgetExceeded):
        reopened.reserve_generation({})
    with pytest.raises(GenerationRunBudgetError, match="cannot be changed"):
        reopened.initialize(max_calls=10)


def test_deadline_is_persistent_and_embedding_count_is_separate(tmp_path, monkeypatch):
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: 1000.0)
    monkeypatch.setattr("local_app.generation_run_budget.time.monotonic", lambda: 100.0)
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    journal.initialize(max_calls=1, max_seconds=60)
    call_id = journal.reserve_generation({"messages": [{"role": "user", "content": "synthetic"}]})
    embedding_id = journal.record_embedding({"input": ["a", "b"]}, item_count=2)
    journal.finish_embedding(embedding_id, response={"embeddings": [[1], [2]]})
    journal.finish_generation(call_id, error=TimeoutError("synthetic unknown transport outcome"))
    state = journal.snapshot()
    assert (state["generation_calls"], state["embedding_calls"], state["embedding_items"]) == (1, 1, 2)
    assert state["unknown_outcomes"] == 1
    with pytest.raises(GenerationRunBudgetError, match="already finalized"):
        journal.finish_generation(call_id, response={})
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: 1061.0)
    with pytest.raises(GenerationRunBudgetExceeded, match="deadline"):
        journal.record_embedding({"input": ["c"]}, item_count=1)
    # A restart and a backward wall clock cannot replace the original deadline.
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: 900.0)
    restarted = GenerationRunBudget(path)
    assert restarted.initialize(max_calls=1, max_seconds=60)["deadline_reached"] is True
    with pytest.raises(GenerationRunBudgetExceeded, match="deadline"):
        restarted.reserve_generation({})


class _HTTP:
    def __init__(self, error=None):
        self.calls = []
        self.timeouts = []
        self.error = error

    def post(self, url, *, json, **kwargs):
        self.calls.append((url, json))
        self.timeouts.append(kwargs["timeout"])
        if self.error:
            raise self.error
        data = ({"embeddings": [[1.0, 0.0] for _ in json["input"]]} if url.endswith("/api/embed")
                else {"message": {"content": "Synthetic answer"}, "done": True, "done_reason": "stop"})
        return SimpleNamespace(status_code=200, json=lambda: data, raise_for_status=lambda: None)


def _settings(path=None, dsn=None, scheduler_dsn=None):
    return SimpleNamespace(database_url=dsn, generation_scheduler_dsn=scheduler_dsn,
        generation_run_budget_path=path, ollama_url="http://127.0.0.1:11435",
        chat_model="synthetic", embedding_model="synthetic-embedding",
        request_timeout=5, context_tokens=8192)


def test_runtime_records_exact_payload_response_and_never_retries(tmp_path):
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    journal.initialize(max_calls=2)
    http = _HTTP()
    runtime = OllamaRuntime(_settings(str(path)), session=http)
    runtime.chat([{"role": "user", "content": "synthetic learner"}])
    runtime.embed(["a", "b"])
    failing_http = _HTTP(error=TimeoutError("synthetic timeout"))
    restarted = OllamaRuntime(_settings(str(path)), session=failing_http)
    with pytest.raises(TimeoutError):
        restarted.chat([{"role": "user", "content": "synthetic continuation"}])
    with pytest.raises(GenerationRunBudgetExceeded):
        restarted.chat([{"role": "user", "content": "must not reach HTTP"}])
    assert len(http.calls) == 2 and len(failing_http.calls) == 1
    rows = _rows(path)
    assert [row["outcome"] for row in rows] == ["returned", "returned", "unknown_outcome"]
    for row, (_, payload) in zip(rows, http.calls + failing_http.calls):
        assert json.loads(row["request_json"]) == payload
        assert row["request_sha256"] == hashlib.sha256(row["request_json"].encode("utf-8")).hexdigest()
    assert json.loads(rows[2]["error_json"])["type"] == "TimeoutError"
    assert journal.snapshot()["generation_calls"] == 2
    assert journal.snapshot()["embedding_calls"] == 1


def test_runtime_fail_closed_before_initialization_and_before_http(tmp_path):
    http = _HTTP()
    path = tmp_path / "absent.sqlite3"
    with pytest.raises(GenerationRunBudgetError, match="explicitly initialized"):
        OllamaRuntime(_settings(str(path)), session=http)
    assert not path.exists() and not http.calls
    journal = GenerationRunBudget(path)
    journal.initialize()
    runtime = OllamaRuntime(_settings(str(path)), session=http)
    with pytest.raises(ValueError, match="At least one"):
        runtime.chat([])
    assert journal.snapshot()["generation_calls"] == 0 and not http.calls


def test_scheduler_override_is_explicit_and_exhaustion_releases_cleanly(tmp_path, monkeypatch):
    events = []

    class Scheduler:
        def __init__(self, dsn, **kwargs):
            self.dsn = dsn

        @contextmanager
        def acquire(self, *args, **kwargs):
            events.append("entered")
            try:
                yield
            except BaseException:
                events.append("unknown_outcome")
                raise
            else:
                events.append("released")

    monkeypatch.setattr("local_app.generation_scheduler.GenerationScheduler", Scheduler)
    default = OllamaRuntime(_settings(dsn="isolated-synthetic-dsn"), session=_HTTP())
    assert default.generation_scheduler.dsn == "isolated-synthetic-dsn"
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    journal.initialize(max_calls=1)
    runtime = OllamaRuntime(_settings(str(path), "isolated-synthetic-dsn", "shared-synthetic-dsn"), session=_HTTP())
    assert runtime.generation_scheduler.dsn == "shared-synthetic-dsn"
    runtime.chat([{"role": "user", "content": "synthetic"}])
    with pytest.raises(GenerationRunBudgetExceeded):
        runtime.chat([{"role": "user", "content": "synthetic"}])
    assert events == ["entered", "released", "entered", "released"]
    assert "shared-synthetic-dsn" not in json.dumps(runtime.manifest())


def test_settings_optional_environment_and_secret_repr(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("LOCAL_GENERATION_SCHEDULER_DSN", raising=False)
    monkeypatch.delenv("LOCAL_GENERATION_RUN_BUDGET_PATH", raising=False)
    assert Settings.from_env().generation_scheduler_dsn is None
    assert Settings.from_env().generation_run_budget_path is None
    monkeypatch.setenv("LOCAL_GENERATION_SCHEDULER_DSN", "postgresql://secret-value")
    monkeypatch.setenv("LOCAL_GENERATION_RUN_BUDGET_PATH", "synthetic/run.sqlite3")
    settings = Settings.from_env()
    assert settings.generation_scheduler_dsn == "postgresql://secret-value"
    assert settings.generation_run_budget_path == "synthetic/run.sqlite3"
    assert "secret-value" not in repr(settings)


def test_http_timeout_uses_remaining_deadline_after_admission_and_failure_stays_unknown(tmp_path, monkeypatch):
    clock = {"wall": 1000.0, "mono": 100.0}
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: clock["wall"])
    monkeypatch.setattr("local_app.generation_run_budget.time.monotonic", lambda: clock["mono"])
    events = []

    class Scheduler:
        def __init__(self, *args, **kwargs):
            pass

        @contextmanager
        def acquire(self, *args, **kwargs):
            clock["wall"] += 59
            clock["mono"] += 59
            try:
                yield
            except BaseException:
                events.append("unknown_outcome")
                raise
            else:
                events.append("released")

    monkeypatch.setattr("local_app.generation_scheduler.GenerationScheduler", Scheduler)
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    journal.initialize(max_seconds=60)
    http = _HTTP(error=TimeoutError("synthetic deadline-limited transport timeout"))
    runtime = OllamaRuntime(_settings(str(path), "synthetic-dsn"), session=http)
    with pytest.raises(TimeoutError):
        runtime.chat([{"role": "user", "content": "synthetic"}])
    assert len(http.calls) == 1 and 0 < http.timeouts[0] <= 1
    assert events == ["unknown_outcome"]
    assert journal.snapshot()["generation_calls"] == 1
    assert _rows(path)[0]["outcome"] == "unknown_outcome"


@pytest.mark.parametrize("expire_before_reservation", [True, False])
def test_deadline_without_http_releases_admission_cleanly(tmp_path, monkeypatch, expire_before_reservation):
    clock = {"wall": 1000.0, "mono": 100.0}
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: clock["wall"])
    monkeypatch.setattr("local_app.generation_run_budget.time.monotonic", lambda: clock["mono"])
    events = []

    def expire():
        clock["wall"], clock["mono"] = 1061.0, 161.0

    class Scheduler:
        def __init__(self, *args, **kwargs):
            pass

        @contextmanager
        def acquire(self, *args, **kwargs):
            if expire_before_reservation:
                expire()
            try:
                yield
            except BaseException:
                events.append("unknown_outcome")
                raise
            else:
                events.append("released")

    class DelayedRuntime(OllamaRuntime):
        def _post(self, *args, **kwargs):
            expire()
            return super()._post(*args, **kwargs)

    monkeypatch.setattr("local_app.generation_scheduler.GenerationScheduler", Scheduler)
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    journal.initialize(max_seconds=60)
    http = _HTTP()
    runtime = DelayedRuntime(_settings(str(path), "synthetic-dsn"), session=http)
    with pytest.raises(GenerationRunBudgetExceeded):
        runtime.chat([{"role": "user", "content": "synthetic"}])
    assert not http.calls and events == ["released"]
    assert journal.snapshot()["deadline_reached"] is True
    assert journal.snapshot()["generation_calls"] == (0 if expire_before_reservation else 1)


def _extension_receipt(tmp_path, journal, original, **overrides):
    receipt = {"schema_version": "local_generation_time_extension_v1", "authorization_id": "synthetic-user-recovery",
               "authorized_by": "user", "authorized_at": "1970-01-01T01:46:40+00:00",
               "authorization_text": "Разрешить ещё 60 минут, общий предел 160 сохранить",
               "scope": "operational_recovery_after_usage_interruption", "ledger_path": str(journal.path),
               "original_started_at": original["started_at"], "original_deadline_at": original["deadline_at"],
               "extension_seconds": 3600, "max_generations": 160, **overrides}
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_explicit_extension_preserves_all_calls_original_limits_and_unknown_once(tmp_path, monkeypatch):
    clock = {"wall": 1000.0, "mono": 100.0}
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: clock["wall"])
    monkeypatch.setattr("local_app.generation_run_budget.time.monotonic", lambda: clock["mono"])
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    original = journal.initialize()
    for index in range(30):
        call_id = journal.reserve_generation({"synthetic": index})
        if index:
            journal.finish_generation(call_id, response={"synthetic": index})
    for index in range(10):
        call_id = journal.record_embedding({"input": [str(index)]}, item_count=1)
        journal.finish_embedding(call_id, response={"embeddings": [[1.0]]})
    before_calls = _rows(path)
    clock["wall"], clock["mono"] = 6500.0, 5600.0
    assert journal.snapshot()["deadline_reached"]
    authorization = _extension_receipt(tmp_path, journal, original)
    with sqlite3.connect(path) as conn:
        before_run = conn.execute("SELECT * FROM run").fetchall()
    preview = journal.preview_authorized_extension(authorization)
    assert preview["journal_modified"] is False and preview["generation_calls"] == 30
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM run").fetchall() == before_run
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='authorized_extensions'").fetchone()
    after = journal.extend_authorized(authorization)
    assert (after["generation_calls"], after["embedding_calls"], after["unknown_outcomes"]) == (30, 10, 1)
    assert _rows(path) == before_calls
    assert after["started_at"] == original["started_at"] and after["max_seconds"] == 5400
    assert after["max_generations"] == 160 and after["remaining_generations"] == 130
    assert after["original_deadline_at"] == original["deadline_at"]
    assert datetime.fromisoformat(after["deadline_at"]).timestamp() == 6500.0 + 3600
    assert len(after["authorized_extensions"]) == 1
    assert after["authorized_extensions"][0]["authorization_sha256"] == hashlib.sha256(authorization.read_bytes()).hexdigest()
    reopened = GenerationRunBudget(path)
    assert reopened.initialize()["deadline_at"] == after["deadline_at"]
    with pytest.raises(GenerationRunBudgetError, match="already been used"):
        reopened.extend_authorized(authorization)
    for statement in ("DELETE FROM authorized_extensions", "UPDATE authorized_extensions SET deadline_at=99999"):
        with sqlite3.connect(path) as conn, pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(statement)
    for index in range(130):
        reopened.reserve_generation({"new_synthetic": index})
    with pytest.raises(GenerationRunBudgetExceeded, match="generation limit"):
        reopened.reserve_generation({})
    assert reopened.snapshot()["generation_calls"] == 160
    clock["wall"], clock["mono"] = 10101.0, 9201.0
    with pytest.raises(GenerationRunBudgetExceeded, match="deadline"):
        reopened.record_embedding({"input": ["late"]}, item_count=1)


@pytest.mark.parametrize("overrides", [
    {"authorization_text": "Просто продолжить"}, {"authorized_by": "agent"}, {"extension_seconds": 3601},
    {"max_generations": 161}, {"authorized_at": "1970-01-01"}, {"authorized_at": "2999-01-01T00:00:00+00:00"},
    {"original_deadline_at": "1970-01-01T00:00:00+00:00"}, {"ledger_path": "another-run.sqlite3"},
])
def test_extension_rejects_wrong_authorization_without_any_history_change(tmp_path, monkeypatch, overrides):
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: 1000.0)
    monkeypatch.setattr("local_app.generation_run_budget.time.monotonic", lambda: 100.0)
    path = tmp_path / "run.sqlite3"
    journal = GenerationRunBudget(path)
    original = journal.initialize()
    journal.reserve_generation({"synthetic_unknown": True})
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: 6500.0)
    authorization = _extension_receipt(tmp_path, journal, original, **overrides)
    with pytest.raises(GenerationRunBudgetError):
        journal.extend_authorized(authorization)
    state = journal.snapshot()
    assert state["authorized_extensions"] == []
    assert state["deadline_at"] == original["deadline_at"]
    assert state["generation_calls"] == state["unknown_outcomes"] == 1


def test_extension_cannot_start_before_expiry_or_on_changed_original_limit(tmp_path, monkeypatch):
    clock = {"wall": 1000.0}
    monkeypatch.setattr("local_app.generation_run_budget.time.time", lambda: clock["wall"])
    monkeypatch.setattr("local_app.generation_run_budget.time.monotonic", lambda: 100.0)
    journal = GenerationRunBudget(tmp_path / "run.sqlite3")
    original = journal.initialize()
    authorization = _extension_receipt(tmp_path, journal, original, authorized_at="1970-01-01T00:16:40+00:00")
    with pytest.raises(GenerationRunBudgetError, match="already expired"):
        journal.extend_authorized(authorization)
    clock["wall"] = 6500.0
    other = GenerationRunBudget(tmp_path / "smaller-run.sqlite3")
    smaller = other.initialize(max_calls=159)
    clock["wall"] = 12000.0
    authorization = _extension_receipt(tmp_path, other, smaller, authorized_at="1970-01-01T01:48:20+00:00")
    with pytest.raises(GenerationRunBudgetError, match="original local run"):
        other.extend_authorized(authorization)
