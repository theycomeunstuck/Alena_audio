"""Local durable worker responses, independent of PostgreSQL and never replayed."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from uuid import uuid4


APPLICATION_ID = 0x54434D52
SCHEMA_VERSION = 1
DEFAULT_PATH = Path(__file__).resolve().parents[1] / ".local" / "memory-worker-responses.sqlite3"


class ResponseJournalError(RuntimeError):
    """A local durability failure must stop this worker without a provider retry."""


def default_journal_path() -> Path:
    value = os.getenv("LOCAL_MEMORY_RESPONSE_JOURNAL")
    if value is not None and not value.strip():
        raise ResponseJournalError("Worker response journal path is empty")
    return Path(value) if value is not None else DEFAULT_PATH


def database_fingerprint(dsn: str | None) -> str:
    """Bind diagnostics to a DB without storing its DSN, password or username."""
    if not dsn:
        return ""
    from psycopg2.extensions import parse_dsn
    try:
        params = parse_dsn(dsn)
        # Do not guess a server/database from service files or login defaults.
        # The application's configured URL supplies an explicit loopback host.
        if (params.get("service") or os.getenv("PGSERVICE") or not params.get("dbname")
                or not (params.get("host") or params.get("hostaddr"))):
            raise ValueError("explicit endpoint and database required")
        identity = {key: params.get(key, "") for key in ("host", "hostaddr", "dbname")}
        identity["hostaddr"] = params.get("hostaddr") or os.getenv("PGHOSTADDR") or ""
        identity["port"] = params.get("port") or os.getenv("PGPORT") or "5432"
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        raise ResponseJournalError("Worker response journal database identity is invalid") from None


class MemoryResponseJournal:
    def __init__(self, path: str | Path, database_identity: str = ""):
        if not str(path).strip():
            raise ResponseJournalError("Worker response journal path is empty")
        self.path = Path(path).resolve()
        self.database_identity = database_identity

    @contextmanager
    def _connect(self, *, write: bool = False):
        connection = None
        try:
            if write:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = sqlite3.connect(self.path.as_uri() + ("?mode=rwc" if write else "?mode=ro"),
                                         uri=True, isolation_level=None, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
            self._schema(connection, create=write)
            yield connection
            if write:
                connection.commit()
        except (OSError, sqlite3.Error):
            raise ResponseJournalError("Worker response journal is unavailable; no automatic replay") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _schema(connection, *, create: bool):
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        objects = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone()
        if create and objects is None and application_id == 0 and version == 0:
            connection.execute("""CREATE TABLE calls (
                call_id TEXT PRIMARY KEY, database_identity TEXT NOT NULL,
                job_id TEXT NOT NULL, worker_id TEXT NOT NULL, purpose TEXT NOT NULL,
                call_index INTEGER NOT NULL, prepared_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE outcomes (
                call_id TEXT PRIMARY KEY REFERENCES calls(call_id),
                payload TEXT NOT NULL, recorded_at TEXT NOT NULL)""")
            for table in ("calls", "outcomes"):
                for operation in ("UPDATE", "DELETE"):
                    connection.execute(f"""CREATE TRIGGER {table}_no_{operation.lower()}
                        BEFORE {operation} ON {table} BEGIN
                        SELECT RAISE(ABORT, 'immutable worker response journal'); END""")
            connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        elif application_id != APPLICATION_ID or version != SCHEMA_VERSION or tables != {"calls", "outcomes"}:
            raise ResponseJournalError("Worker response journal has an unsupported identity or schema")

    def prepare(self, *, job_id: str, worker_id: str, purpose: str, call_index: int) -> str:
        if (not all(isinstance(value, str) and value and len(value) <= 200
                    for value in (job_id, worker_id, purpose))
                or type(call_index) is not int or call_index < 1):
            raise ResponseJournalError("Worker response journal metadata is invalid")
        call_id = uuid4().hex
        with self._connect(write=True) as connection:
            connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?,?)",
                (call_id, self.database_identity, job_id, worker_id, purpose, call_index,
                 datetime.now(timezone.utc).isoformat()))
        return call_id

    def record(self, call_id: str, *, outcome: str, raw_response=None,
               error_type: str | None = None, elapsed_seconds: float = 0.0) -> None:
        if (outcome not in {"returned", "raised"} or not isinstance(elapsed_seconds, (int, float))
                or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
                or error_type is not None and not isinstance(error_type, str)):
            raise ResponseJournalError("Worker response journal outcome is invalid")
        try:
            payload = json.dumps({"outcome": outcome, "raw_response": raw_response,
                                  "error_type": error_type, "elapsed_seconds": elapsed_seconds},
                                 ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            raise ResponseJournalError("Worker response cannot be serialized durably") from None
        with self._connect(write=True) as connection:
            parent = connection.execute("SELECT 1 FROM calls WHERE call_id=? AND database_identity=?",
                                        (call_id, self.database_identity)).fetchone()
            if parent is None:
                raise ResponseJournalError("Worker response does not belong to this journal database")
            existing = connection.execute("SELECT payload FROM outcomes WHERE call_id=?", (call_id,)).fetchone()
            if existing is not None:
                if existing["payload"] != payload:
                    raise ResponseJournalError("Worker response journal outcome is immutable")
                return
            connection.execute("INSERT INTO outcomes VALUES (?,?,?)",
                               (call_id, payload, datetime.now(timezone.utc).isoformat()))

    @staticmethod
    def _project(row) -> dict:
        result = dict(row)
        payload = result.pop("payload")
        try:
            result.update(json.loads(payload) if payload is not None else {
                "outcome": None, "raw_response": None, "error_type": None, "elapsed_seconds": None})
        except (TypeError, ValueError):
            raise ResponseJournalError("Worker response journal outcome cannot be read") from None
        return result

    def get(self, call_id: str) -> dict:
        if not self.path.exists():
            raise KeyError(call_id)
        with self._connect() as connection:
            row = connection.execute("""SELECT c.*,o.payload,o.recorded_at FROM calls c
                LEFT JOIN outcomes o USING(call_id) WHERE c.call_id=? AND c.database_identity=?""",
                (call_id, self.database_identity)).fetchone()
            if row is None:
                raise KeyError(call_id)
            return self._project(row)

    def list_calls(self, job_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        with self._connect() as connection:
            query = """SELECT c.*,o.payload,o.recorded_at FROM calls c
                LEFT JOIN outcomes o USING(call_id) WHERE c.database_identity=?"""
            values = [self.database_identity]
            if job_id is not None:
                query += " AND c.job_id=?"
                values.append(job_id)
            rows = connection.execute(query + " ORDER BY c.prepared_at,c.call_id", values).fetchall()
            return [self._project(row) for row in rows]
