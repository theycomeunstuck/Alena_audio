"""Durable, optional limits for one explicitly initialized local pilot run.

The SQLite file is shared by API and worker processes. A reservation is never
refunded: a process can disappear after Ollama accepted its HTTP request.
Requests and responses belong in an ignored runtime directory, not in Git.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


class GenerationRunBudgetError(RuntimeError):
    """The explicit local pilot journal cannot authorize another request."""


class GenerationRunBudgetExceeded(GenerationRunBudgetError):
    """The local pilot reached its persistent count or deadline."""


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class GenerationRunBudget:
    """Append-only actual-call accounting with explicit, recorded time recovery."""

    SCHEMA_VERSION = 1
    MAX_GENERATIONS = 160
    MAX_SECONDS = 90 * 60
    EXTENSION_SECONDS = 60 * 60
    EXTENSION_AUTHORIZATION_TEXT = "Разрешить ещё 60 минут, общий предел 160 сохранить"

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._clock_anchor_wall = time.time()
        self._clock_anchor_mono = time.monotonic()

    @contextmanager
    def _connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        if not create and not self.path.is_file():
            raise GenerationRunBudgetError("Local generation run journal must be explicitly initialized")
        try:
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
        except sqlite3.Error as exc:
            raise GenerationRunBudgetError("Local generation run journal is unavailable") from exc
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self, *, max_calls: int = MAX_GENERATIONS, max_seconds: float = MAX_SECONDS) -> dict:
        """Create the run once; reopening with the same limits never resets it."""
        if type(max_calls) is not int or not 1 <= max_calls <= self.MAX_GENERATIONS:
            raise ValueError("Local generation limit must be between 1 and 160")
        if not math.isfinite(float(max_seconds)) or not 0 < max_seconds <= self.MAX_SECONDS:
            raise ValueError("Local generation duration must be positive and at most 5400 seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(create=True) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("""CREATE TABLE IF NOT EXISTS run (
                    id INTEGER PRIMARY KEY CHECK (id=1), schema_version INTEGER NOT NULL,
                    started_at REAL NOT NULL, deadline_at REAL NOT NULL,
                    max_generations INTEGER NOT NULL, max_seconds REAL NOT NULL,
                    last_observed_at REAL NOT NULL)""")
                conn.execute("""CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('generation','embedding')),
                    requested_at REAL NOT NULL, item_count INTEGER NOT NULL,
                    request_json TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'unknown_outcome',
                    finished_at REAL, response_json TEXT, response_sha256 TEXT, error_json TEXT)""")
                row = conn.execute("SELECT * FROM run WHERE id=1").fetchone()
                if row is None:
                    now = time.time()
                    conn.execute("INSERT INTO run VALUES (1,?,?,?,?,?,?)",
                                 (self.SCHEMA_VERSION, now, now + max_seconds, max_calls, float(max_seconds), now))
                elif (row["schema_version"] != self.SCHEMA_VERSION
                      or row["max_generations"] != max_calls or row["max_seconds"] != float(max_seconds)):
                    raise GenerationRunBudgetError("Existing local run limits cannot be changed or reset")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return self.snapshot()

    def _run(self, conn: sqlite3.Connection) -> sqlite3.Row:
        try:
            row = conn.execute("SELECT * FROM run WHERE id=1").fetchone()
        except sqlite3.Error as exc:
            raise GenerationRunBudgetError("Local generation run journal is not initialized") from exc
        if row is None or row["schema_version"] != self.SCHEMA_VERSION:
            raise GenerationRunBudgetError("Local generation run journal has an unsupported configuration")
        return row

    def _now(self, conn: sqlite3.Connection, row: sqlite3.Row) -> float:
        # A backward wall-clock adjustment cannot refund elapsed time in this
        # process, nor move the last persisted observation backward on restart.
        mono = time.monotonic()
        elapsed_now = self._clock_anchor_wall + max(0, mono - self._clock_anchor_mono)
        now = max(time.time(), elapsed_now, row["last_observed_at"])
        if now > elapsed_now:
            self._clock_anchor_wall, self._clock_anchor_mono = now, mono
        conn.execute("UPDATE run SET last_observed_at=? WHERE id=1", (now,))
        return now

    @staticmethod
    def _extensions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='authorized_extensions'").fetchone()
        return list(conn.execute("SELECT * FROM authorized_extensions ORDER BY id")) if exists else []

    def _effective_deadline(self, conn: sqlite3.Connection, row: sqlite3.Row) -> tuple[float, list[sqlite3.Row]]:
        extensions = self._extensions(conn)
        if not extensions:
            return row["deadline_at"], []
        extension = extensions[0]
        if (len(extensions) != 1 or extension["previous_deadline_at"] != row["deadline_at"]
                or extension["extension_seconds"] != self.EXTENSION_SECONDS
                or extension["deadline_at"] != extension["applied_at"] + self.EXTENSION_SECONDS
                or row["max_generations"] != self.MAX_GENERATIONS):
            raise GenerationRunBudgetError("Local run extension does not match its immutable original limits")
        return extension["deadline_at"], extensions

    def _extension_authorization(self, authorization_path: str | Path) -> tuple[dict, str, str]:
        raw = Path(authorization_path).read_bytes()
        authorization = json.loads(raw.decode("utf-8"))
        required = {"schema_version", "authorization_id", "authorized_by", "authorized_at",
                    "authorization_text", "scope", "ledger_path", "original_started_at",
                    "original_deadline_at", "extension_seconds", "max_generations"}
        if (not isinstance(authorization, dict) or set(authorization) != required
                or authorization["schema_version"] != "local_generation_time_extension_v1"
                or authorization["authorized_by"] != "user"
                or authorization["authorization_text"] != self.EXTENSION_AUTHORIZATION_TEXT
                or authorization["scope"] != "operational_recovery_after_usage_interruption"
                or type(authorization["extension_seconds"]) is not int
                or authorization["extension_seconds"] != self.EXTENSION_SECONDS
                or type(authorization["max_generations"]) is not int
                or authorization["max_generations"] != self.MAX_GENERATIONS
                or not isinstance(authorization["authorization_id"], str)
                or not authorization["authorization_id"].strip()
                or Path(authorization["ledger_path"]).resolve() != self.path):
            raise GenerationRunBudgetError("Explicit user authorization for this exact local run is required")
        try:
            authorized_at = datetime.fromisoformat(authorization["authorized_at"])
            if authorized_at.tzinfo is None:
                raise ValueError("Authorization timestamp needs a timezone")
        except (TypeError, ValueError) as exc:
            raise GenerationRunBudgetError("User authorization needs a dated timezone-aware timestamp") from exc
        return authorization, raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()

    def _validate_extension(self, conn: sqlite3.Connection, row: sqlite3.Row, authorization: dict, now: float) -> None:
        if self._extensions(conn):
            raise GenerationRunBudgetError("The single authorized local run extension has already been used")
        if (row["max_generations"] != self.MAX_GENERATIONS or row["max_seconds"] != self.MAX_SECONDS
                or authorization["original_started_at"] != _utc(row["started_at"])
                or authorization["original_deadline_at"] != _utc(row["deadline_at"])):
            raise GenerationRunBudgetError("User authorization does not match the original local run")
        authorized_at = datetime.fromisoformat(authorization["authorized_at"]).timestamp()
        if not row["started_at"] <= authorized_at <= now:
            raise GenerationRunBudgetError("User authorization date must belong to this local run's history")
        if now < row["deadline_at"]:
            raise GenerationRunBudgetError("Operational recovery may extend only an already expired local run")

    def preview_authorized_extension(self, authorization_path: str | Path) -> dict:
        """Validate a receipt without modifying the journal or starting its window."""
        authorization, _, digest = self._extension_authorization(authorization_path)
        with self._connect() as conn:
            conn.execute("BEGIN")
            row = self._run(conn)
            now = max(time.time(), row["last_observed_at"])
            self._validate_extension(conn, row, authorization, now)
            count = conn.execute("SELECT count(*) FROM calls WHERE kind='generation'").fetchone()[0]
        return {"status": "dry_run", "ledger_path": str(self.path), "authorization_sha256": digest,
                "authorization_id": authorization["authorization_id"], "extension_seconds": self.EXTENSION_SECONDS,
                "original_deadline_at": _utc(row["deadline_at"]), "proposed_deadline_at": _utc(now + self.EXTENSION_SECONDS),
                "generation_calls": count, "max_generations": row["max_generations"], "journal_modified": False}

    def extend_authorized(self, authorization_path: str | Path) -> dict:
        """Consume one explicit user recovery authorization; never reset a call."""
        authorization, raw, digest = self._extension_authorization(authorization_path)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._run(conn)
                now = self._now(conn, row)
                self._validate_extension(conn, row, authorization, now)
                conn.execute("""CREATE TABLE IF NOT EXISTS authorized_extensions (
                    id INTEGER PRIMARY KEY CHECK(id=1), applied_at REAL NOT NULL,
                    previous_deadline_at REAL NOT NULL, deadline_at REAL NOT NULL,
                    extension_seconds INTEGER NOT NULL CHECK(extension_seconds=3600),
                    authorization_id TEXT NOT NULL UNIQUE, authorization_json TEXT NOT NULL,
                    authorization_sha256 TEXT NOT NULL)""")
                conn.execute("""CREATE TRIGGER IF NOT EXISTS authorized_extensions_no_update
                    BEFORE UPDATE ON authorized_extensions BEGIN
                    SELECT RAISE(ABORT,'Authorized extensions are append-only'); END""")
                conn.execute("""CREATE TRIGGER IF NOT EXISTS authorized_extensions_no_delete
                    BEFORE DELETE ON authorized_extensions BEGIN
                    SELECT RAISE(ABORT,'Authorized extensions are append-only'); END""")
                conn.execute("INSERT INTO authorized_extensions VALUES (1,?,?,?,?,?,?,?)",
                             (now, row["deadline_at"], now + self.EXTENSION_SECONDS, self.EXTENSION_SECONDS,
                              authorization["authorization_id"], raw, digest))
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._run(conn)
                now = self._now(conn, row)
                deadline, extensions = self._effective_deadline(conn, row)
                counts = {item["kind"]: item for item in conn.execute(
                    "SELECT kind,count(*) AS calls,sum(item_count) AS items FROM calls GROUP BY kind")}
                generation_calls = counts.get("generation", {"calls": 0})["calls"]
                embeddings = counts.get("embedding", {"calls": 0, "items": 0})
                unresolved = conn.execute("SELECT count(*) FROM calls WHERE outcome='unknown_outcome'").fetchone()[0]
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {"schema_version": row["schema_version"], "started_at": _utc(row["started_at"]),
                "deadline_at": _utc(deadline), "original_deadline_at": _utc(row["deadline_at"]),
                "authorized_extensions": [{"authorization_id": item["authorization_id"],
                    "authorization_sha256": item["authorization_sha256"], "applied_at": _utc(item["applied_at"]),
                    "previous_deadline_at": _utc(item["previous_deadline_at"]), "deadline_at": _utc(item["deadline_at"]),
                    "extension_seconds": item["extension_seconds"]} for item in extensions],
                "max_generations": row["max_generations"],
                "max_seconds": row["max_seconds"], "generation_calls": generation_calls,
                "embedding_calls": embeddings["calls"], "embedding_items": embeddings["items"],
                "remaining_generations": max(0, row["max_generations"] - generation_calls),
                "remaining_seconds": max(0.0, deadline - now),
                "deadline_reached": now >= deadline, "unknown_outcomes": unresolved}

    def _reserve(self, kind: str, payload: dict, *, item_count: int = 1) -> int:
        encoded = _json(payload)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._run(conn)
                now = self._now(conn, row)
                deadline, _ = self._effective_deadline(conn, row)
                if now >= deadline:
                    raise GenerationRunBudgetExceeded("Local generation run reached its persistent deadline")
                if kind == "generation":
                    count = conn.execute("SELECT count(*) FROM calls WHERE kind='generation'").fetchone()[0]
                    if count >= row["max_generations"]:
                        raise GenerationRunBudgetExceeded("Local generation run reached its generation limit")
                call_id = conn.execute("""INSERT INTO calls
                    (kind,requested_at,item_count,request_json,request_sha256) VALUES (?,?,?,?,?)""",
                    (kind, now, item_count, encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest())).lastrowid
                conn.commit()
                return int(call_id)
            except BaseException:
                # Persist the monotonic clock observation even on exhaustion.
                conn.commit()
                raise

    def reserve_generation(self, payload: dict) -> int:
        return self._reserve("generation", payload)

    def record_embedding(self, payload: dict, *, item_count: int) -> int:
        if type(item_count) is not int or item_count < 1:
            raise ValueError("Embedding item count must be positive")
        return self._reserve("embedding", payload, item_count=item_count)

    def finish(self, call_id: int, *, response: Any = None, error: BaseException | None = None) -> None:
        """Record transport outcome once; unknown reservations never disappear."""
        encoded = None if response is None else _json(response)
        error_json = None if error is None else _json({"type": type(error).__name__, "message": str(error)})
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._run(conn)
                now = self._now(conn, row)
                updated = conn.execute("""UPDATE calls SET outcome=?,finished_at=?,response_json=?,
                    response_sha256=?,error_json=? WHERE id=? AND finished_at IS NULL""",
                    ("returned" if error is None else "unknown_outcome", now, encoded,
                     None if encoded is None else hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                     error_json, call_id)).rowcount
                if updated != 1:
                    raise GenerationRunBudgetError("Local generation call is absent or already finalized")
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    finish_generation = finish
    finish_embedding = finish
