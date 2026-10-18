"""Durable synthetic-only batches. No automatic educational quality assessment.

The time budget is a persisted wall-clock deadline starting with the first worker
session. Time between sessions counts too. Token checks use known usage and a
full configured context/output reservation; they are not a tokenizer guarantee.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

from .clients import Completion, CompletionValidationError, FixtureClient, OllamaClient
from .dataset import load_dataset, require
from .methods import METHODS, PROMPT_DIR
from .runner import ROOT, run


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _versions():
    return {"code": {p.name: _sha(p) for p in sorted(ROOT.glob("*.py"))},
            "prompts": {p.name: _sha(p) for p in sorted(PROMPT_DIR.glob("*.txt"))}}


@contextmanager
def _connect(directory):
    conn = sqlite3.connect(Path(directory) / "queue.sqlite3", timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _meta(conn, key):
    return json.loads(conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()[0])


def _setmeta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, _json(value)))


@contextmanager
def _worker_lock(directory):
    """An OS lock survives neither process death nor a closed descriptor."""
    stream = (Path(directory) / "worker.lock").open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            raise RuntimeError("Another worker owns this batch") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def create_batch(histories_path, requests_path, output, *, backend="fixture", history_ids=None,
                 methods=METHODS, budget_chars=4000, max_calls=1000, max_seconds=3600,
                 max_input_tokens=None, max_output_tokens_total=None, config_path=None,
                 model=None, base_url=None):
    """Freeze inputs and configuration; create one durable history×method job."""
    dataset = load_dataset(histories_path, requests_path)
    require(backend in {"fixture", "ollama"}, "Unsupported backend")
    require(isinstance(budget_chars, int) and budget_chars >= 128, "Invalid character budget")
    methods = tuple(methods)
    require(bool(methods) and len(methods) == len(set(methods)) and set(methods) <= set(METHODS),
            "Invalid methods")
    selected = list(history_ids) if history_ids is not None else list(dataset.histories)
    require(bool(selected) and len(selected) == len(set(selected)) and set(selected) <= set(dataset.histories),
            "Unknown or empty history selection")
    require(all(h["data_origin"].startswith("synthetic") for h in dataset.histories.values()),
            "This batch accepts synthetic data only; traces contain full inputs and outputs")
    for name, value in (("max_calls", max_calls), ("max_seconds", max_seconds),
                        ("max_input_tokens", max_input_tokens),
                        ("max_output_tokens_total", max_output_tokens_total)):
        require(value is None or isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0,
                f"{name} must be positive or None")
    require(max_calls is None or isinstance(max_calls, int), "max_calls must be an integer")
    require(all(value is None or isinstance(value, int) for value in (max_input_tokens, max_output_tokens_total)),
            "Token limits must be integers")
    config = json.loads(Path(config_path or ROOT / "config.json").read_text(encoding="utf-8"))
    if model is not None:
        config["model"] = model
    if base_url is not None:
        config["base_url"] = base_url
    config["budget_chars"] = budget_chars
    require(set(config) == {"protocol_version", "model", "base_url", "timeout_seconds", "context_tokens",
                            "max_output_tokens", "budget_chars"}, "Unexpected configuration fields")
    OllamaClient(base_url=config["base_url"], model=config["model"], timeout=config["timeout_seconds"],
                 context_tokens=config["context_tokens"], max_output_tokens=config["max_output_tokens"])
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    snapshots = directory / "snapshots"
    snapshots.mkdir()
    for name, source in (("histories.jsonl", histories_path), ("requests.jsonl", requests_path)):
        (snapshots / name).write_bytes(Path(source).read_bytes())
    _write(snapshots / "config.json", config)
    versions = _versions()
    for folder, paths in (("code", ROOT.glob("*.py")), ("prompts", PROMPT_DIR.glob("*.txt"))):
        (snapshots / folder).mkdir()
        for path in paths:
            (snapshots / folder / path.name).write_bytes(path.read_bytes())
    definition = {"batch_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "backend": backend, "history_ids": selected, "methods": list(methods),
                  "split": dataset.split, "budget_chars": budget_chars, "resolved_config": config,
                  "limits": {"max_calls": max_calls, "max_seconds": max_seconds,
                             "max_input_tokens": max_input_tokens,
                             "max_output_tokens_total": max_output_tokens_total},
                  "time_budget_kind": "wall_clock_deadline_from_first_work_including_downtime",
                  "token_budget_kind": "known_usage_plus_full_context_and_output_reservation_not_hard_guarantee",
                  "versions": versions,
                  "snapshot_sha256": {p.name: _sha(p) for p in snapshots.iterdir() if p.is_file()},
                  "scientific_result": False,
                  "run_kind": "contract_only" if backend == "fixture" else "unreviewed_development_pilot",
                  "original_input_sha256": {"histories": _sha(histories_path), "requests": _sha(requests_path)}}
    _write(directory / "batch.json", definition)
    with _connect(directory) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
          CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE jobs(id INTEGER PRIMARY KEY, history_id TEXT NOT NULL, method TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', attempt INTEGER NOT NULL DEFAULT 0,
            artifact_path TEXT, error TEXT, UNIQUE(history_id,method));
          CREATE TABLE calls(job_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, fingerprint TEXT NOT NULL,
            status TEXT NOT NULL, started REAL NOT NULL, completion TEXT, error TEXT, error_code TEXT,
            PRIMARY KEY(job_id,ordinal));
        """)
        _setmeta(conn, "definition_sha256", _sha(directory / "batch.json"))
        _setmeta(conn, "deadline", None)
        _setmeta(conn, "cache_hits", 0)
        for history_id in selected:
            for method in methods:
                conn.execute("INSERT INTO jobs(history_id,method) VALUES (?,?)", (history_id, method))
    return batch_report(directory)


def _validate_frozen(directory, conn):
    require(_sha(directory / "batch.json") == _meta(conn, "definition_sha256"), "Batch definition hash mismatch")
    definition = json.loads((directory / "batch.json").read_text(encoding="utf-8"))
    for name, expected in definition["snapshot_sha256"].items():
        require(_sha(directory / "snapshots" / name) == expected, f"Frozen input/config hash mismatch: {name}")
    require(_versions() == definition["versions"], "Current code or prompt hash mismatch; create a new batch")
    for folder, versions in definition["versions"].items():
        for name, expected in versions.items():
            require(_sha(directory / "snapshots" / folder / name) == expected,
                    f"Frozen {folder} hash mismatch: {name}")
    return definition


def _usage(conn, backend):
    calls = conn.execute("SELECT status,completion FROM calls").fetchall()
    result = {"new_calls": len(calls), "cache_hits": _meta(conn, "cache_hits"),
              "actual_model_calls": 0,
              "preflight_refusals": 0,
              "fixture_calls": len(calls) if backend == "fixture" else 0,
              "known_input_tokens": 0, "known_output_tokens": 0,
              "unknown_usage_calls": 0, "failed_calls": 0, "unknown_outcome_calls": 0}
    for row in calls:
        completion = json.loads(row["completion"]) if row["completion"] else {}
        usage = completion.get("usage", {})
        performed = completion.get("metadata", {}).get("network_request_performed") is not False
        result["actual_model_calls"] += backend == "ollama" and performed
        result["preflight_refusals"] += backend == "ollama" and not performed
        unknown = False
        for key in ("input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                if backend == "ollama":
                    result["known_" + key] += value
            else:
                unknown = True
        result["unknown_usage_calls"] += unknown and backend == "ollama"
        result["failed_calls"] += row["status"] == "failed"
        result["unknown_outcome_calls"] += row["status"] in {"started", "unknown"}
    result["token_usage_complete"] = backend == "ollama" and result["unknown_usage_calls"] == 0
    return result


class _BudgetStop(RuntimeError):
    pass


class _CachedClient:
    def __init__(self, conn, definition, job_id, client):
        self.conn, self.definition, self.job_id, self.client = conn, definition, job_id, client
        self.ordinal = 0
        self.stop_reason = None

    def _reserve(self):
        limits = self.definition["limits"]
        usage = _usage(self.conn, self.definition["backend"])
        deadline = _meta(self.conn, "deadline")
        if deadline is not None and time.time() >= deadline:
            raise _BudgetStop("deadline_exceeded")
        if limits["max_calls"] is not None and usage["new_calls"] >= limits["max_calls"]:
            raise _BudgetStop("max_calls_exceeded")
        if self.definition["backend"] == "ollama":
            if usage["unknown_usage_calls"] and (limits["max_input_tokens"] is not None or
                                                limits["max_output_tokens_total"] is not None):
                raise _BudgetStop("token_usage_unknown")
            for used, limit, reserve in (("known_input_tokens", "max_input_tokens", "context_tokens"),
                                        ("known_output_tokens", "max_output_tokens_total", "max_output_tokens")):
                if limits[limit] is not None and usage[used] + self.definition["resolved_config"][reserve] > limits[limit]:
                    raise _BudgetStop(limit + "_reservation_exceeded")
        return None if deadline is None else max(0.001, deadline - time.time())

    def complete(self, messages, purpose, response_schema=None):
        self.ordinal += 1
        if self.stop_reason:
            raise RuntimeError(self.stop_reason)
        fingerprint = hashlib.sha256(_json({"messages": messages, "purpose": purpose,
                                             "response_schema": response_schema,
                                             "model_config": self.definition["resolved_config"]}).encode()).hexdigest()
        row = self.conn.execute("SELECT * FROM calls WHERE job_id=? AND ordinal=?",
                                (self.job_id, self.ordinal)).fetchone()
        if row:
            if row["fingerprint"] != fingerprint:
                self.stop_reason = "cache_fingerprint_mismatch"
                raise RuntimeError(self.stop_reason)
            if row["status"] not in {"succeeded", "failed"}:
                self.stop_reason = "unknown_prior_call_outcome"
                raise RuntimeError(self.stop_reason)
            with self.conn:
                _setmeta(self.conn, "cache_hits", _meta(self.conn, "cache_hits") + 1)
            completion = Completion(**json.loads(row["completion"])) if row["completion"] else None
            if row["status"] == "succeeded":
                return completion
            self.stop_reason = "known_call_failure"
            if completion:
                raise CompletionValidationError("Cached failed completion", completion, row["error_code"] or "cached_failure")
            raise RuntimeError("Cached model call failed")
        try:
            remaining = self._reserve()
        except _BudgetStop as exc:
            self.stop_reason = str(exc)
            raise
        with self.conn:
            self.conn.execute("INSERT INTO calls(job_id,ordinal,fingerprint,status,started) VALUES (?,?,?,'started',?)",
                              (self.job_id, self.ordinal, fingerprint, time.time()))
        original_timeout = getattr(self.client, "timeout", None)
        if remaining is not None and original_timeout is not None:
            self.client.timeout = min(original_timeout, remaining)
        try:
            completion = self.client.complete(messages, purpose, response_schema)
            serialized = _json(asdict(completion))
        except Exception as exc:
            completion = exc.completion if isinstance(exc, CompletionValidationError) else None
            with self.conn:
                self.conn.execute("UPDATE calls SET status='failed',completion=?,error=?,error_code=? WHERE job_id=? AND ordinal=?",
                                  (_json(asdict(completion)) if completion else None, type(exc).__name__,
                                   exc.code if isinstance(exc, CompletionValidationError) else None,
                                   self.job_id, self.ordinal))
            self.stop_reason = "known_call_failure"
            raise
        finally:
            if original_timeout is not None:
                self.client.timeout = original_timeout
        with self.conn:
            self.conn.execute("UPDATE calls SET status='succeeded',completion=? WHERE job_id=? AND ordinal=?",
                              (serialized, self.job_id, self.ordinal))
        return completion


def work_batch(batch_dir, *, client=None, max_jobs=None):
    """Execute pending jobs once. Resume uses call-ordinal scoped cached results.

    Unknown interrupted calls fail their job rather than causing an unrecorded
    repeat. A stopped budget leaves the job pending and never resets the limits.
    """
    directory = Path(batch_dir)
    require(max_jobs is None or isinstance(max_jobs, int) and not isinstance(max_jobs, bool) and max_jobs >= 0,
            "Invalid max_jobs")
    with _worker_lock(directory), _connect(directory) as conn:
        definition = _validate_frozen(directory, conn)
        config = definition["resolved_config"]
        if client is None:
            client = FixtureClient() if definition["backend"] == "fixture" else OllamaClient(
                base_url=config["base_url"], model=config["model"], timeout=config["timeout_seconds"],
                context_tokens=config["context_tokens"], max_output_tokens=config["max_output_tokens"])
        with conn:
            if _meta(conn, "deadline") is None and definition["limits"]["max_seconds"] is not None:
                _setmeta(conn, "deadline", time.time() + definition["limits"]["max_seconds"])
            conn.execute("UPDATE calls SET status='unknown',error='unknown_after_interruption' WHERE status='started'")
            conn.execute("UPDATE jobs SET status='failed',error='unknown_model_call_outcome' WHERE id IN "
                         "(SELECT job_id FROM calls WHERE status='unknown')")
            conn.execute("UPDATE jobs SET status='pending',error='resuming_interrupted_job' WHERE status='running'")
        processed = 0
        pending = conn.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY id").fetchall()
        for job in pending:
            if max_jobs is not None and processed >= max_jobs:
                break
            deadline = _meta(conn, "deadline")
            if deadline is not None and time.time() >= deadline:
                break
            attempt = job["attempt"] + 1
            relative = Path("attempts") / f"job-{job['id']:04d}" / f"attempt-{attempt:04d}"
            with conn:
                conn.execute("UPDATE jobs SET status='running',attempt=?,artifact_path=?,error=NULL WHERE id=?",
                             (attempt, relative.as_posix(), job["id"]))
            cached = _CachedClient(conn, definition, job["id"], client)
            try:
                manifest = run(directory / "snapshots/histories.jsonl", directory / "snapshots/requests.jsonl",
                               directory / relative, backend=definition["backend"], history_ids=[job["history_id"]],
                               methods=[job["method"]], budget_chars=definition["budget_chars"],
                               config_path=directory / "snapshots/config.json", client=cached)
                status = "failed" if manifest["error_result_count"] else "succeeded"
                error = "result_contract_or_call_failure" if status == "failed" else None
                if cached.stop_reason and cached.stop_reason not in {"known_call_failure", "cache_fingerprint_mismatch",
                                                                    "known_prior_call_failure", "unknown_prior_call_outcome"}:
                    status, error = "pending", cached.stop_reason
                elif cached.stop_reason:
                    status, error = "failed", cached.stop_reason
            except Exception as exc:
                status, error = "failed", type(exc).__name__
            with conn:
                conn.execute("UPDATE jobs SET status=?,error=? WHERE id=?", (status, error, job["id"]))
            processed += 1
            if status == "pending":
                break
        return batch_report(directory)


def batch_status(batch_dir):
    """Read-only snapshot; it is safe while another worker owns the batch."""
    directory = Path(batch_dir)
    definition = json.loads((directory / "batch.json").read_text(encoding="utf-8"))
    with _connect(directory) as conn:
        # One SQLite read transaction keeps job and call counters consistent.
        conn.execute("BEGIN")
        jobs = [dict(x) for x in conn.execute("SELECT * FROM jobs ORDER BY id")]
        usage = _usage(conn, definition["backend"])
        deadline = _meta(conn, "deadline")
    dataset = load_dataset(directory / "snapshots/histories.jsonl", directory / "snapshots/requests.jsonl")
    expected = sum(q["history_id"] in set(definition["history_ids"]) for q in dataset.requests) * len(definition["methods"])
    counts = {state: sum(j["status"] == state for j in jobs)
              for state in ("pending", "running", "succeeded", "failed", "blocked")}
    return {"batch_dir": str(directory.resolve()), "backend": definition["backend"],
            "scientific_result": False, "run_kind": definition["run_kind"],
            "real_model_calls": usage["actual_model_calls"] > 0,
            "job_counts": counts, "jobs": jobs, "expected_results": expected,
            "assessment_status": "not_assessed", "limits": definition["limits"],
            "deadline_unix": deadline, "deadline_exceeded": deadline is not None and time.time() >= deadline,
            "time_budget_kind": definition["time_budget_kind"], "usage": usage}


def batch_report(batch_dir):
    """Materialize all expected rows, including pending and failed cases."""
    directory = Path(batch_dir)
    status = batch_status(directory)
    dataset = load_dataset(directory / "snapshots/histories.jsonl", directory / "snapshots/requests.jsonl")
    rows = []
    for job in status["jobs"]:
        artifact = directory / job["artifact_path"] / "results.jsonl" if job["artifact_path"] else None
        actual = {}
        if job["status"] in {"succeeded", "failed"} and artifact and artifact.exists():
            for line in artifact.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                actual[value["request_id"]] = value
        for request in dataset.requests:
            if request["history_id"] != job["history_id"]:
                continue
            row = {"result_id": f"{job['history_id']}:{request['request_id']}:{job['method']}",
                   "history_id": job["history_id"], "request_id": request["request_id"],
                   "method": job["method"], "request_text": request["text"], "action": None,
                   "status": "not_executed", "scientific_result": False, "run_kind": status["run_kind"]}
            if job["status"] in {"succeeded", "failed"} and request["request_id"] in actual:
                row.update(actual[request["request_id"]])
            row.update(job_status=job["status"], job_error=job["error"],
                       assessment_status="not_assessed", attempt=job["attempt"])
            rows.append(row)
    temporary = directory / ("results.jsonl." + uuid.uuid4().hex + ".tmp")
    temporary.write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, directory / "results.jsonl")
    status["result_count"] = len(rows)
    status["executed_results"] = sum(row["status"] != "not_executed" for row in rows)
    status["unexecuted_results"] = len(rows) - status["executed_results"]
    status["execution_error_results"] = sum(row["status"] not in {"ok", "not_executed"} for row in rows)
    status["assessed_results"] = 0
    status["not_assessed_results"] = len(rows)
    _write(directory / "batch_manifest.json", status)
    usage = status["usage"]
    text = ("# Отчёт фонового пакета\n\n"
            f"Тип запуска: `{status['run_kind']}`. Научный результат: нет. "
            "Педагогическое качество не оценено (`not_assessed`).\n\n"
            f"Ожидалось ответов: {len(rows)}. Выполнено: {status['executed_results']}; "
            f"не выполнено: {status['unexecuted_results']}; ошибок выполнения: {status['execution_error_results']}. "
            f"Не оценено: {len(rows)}.\n\n"
            f"Задания: {_json(status['job_counts'])}.\n\n"
            f"Новых вызовов клиента: {usage['new_calls']}; возвратов из кеша: {usage['cache_hits']}; "
            f"реальных попыток модели: {usage['actual_model_calls']}; fixture: {usage['fixture_calls']}. "
            f"Неизвестный исход вызова: {usage['unknown_outcome_calls']}; ошибок вызовов: {usage['failed_calls']}.\n\n"
            f"Известные токены: вход {usage['known_input_tokens']}, выход {usage['known_output_tokens']}. "
            f"Реальных вызовов с неполным usage: {usage['unknown_usage_calls']}. "
            "Fixture не создаёт модельных токенов. Денежная стоимость не рассчитывается.\n\n"
            "Лимит времени – сохранённый wall-clock deadline от первого worker; паузы и простой тоже входят. "
            "Перед вызовом проверяется deadline и сокращается сетевой timeout; это не жёсткое принудительное завершение процесса. "
            "Лимиты не сбрасываются при возобновлении. Вызовы с неизвестным исходом не повторяются. "
            "Токены проверяются по известному usage с резервом полного контекста и максимального выхода; "
            "это консервативная остановка, а не доказанная жёсткая граница токенизации. "
            "Все ожидаемые случаи сохранены в results.jsonl, включая pending и failed.\n")
    (directory / "report.md").write_text(text, encoding="utf-8")
    return status
