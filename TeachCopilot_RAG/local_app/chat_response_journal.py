"""Durable private chat outcomes, independent of PostgreSQL and never replayed."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from time import monotonic
from uuid import uuid4

from local_app.memory_response_journal import ResponseJournalError, database_fingerprint


APPLICATION_ID = 0x54434352
SCHEMA_VERSION = 1
DEFAULT_PATH = Path(__file__).resolve().parents[1] / ".local" / "chat-responses.sqlite3"


def _journal_default(value):
    """Known non-JSON-native database types stay in the durable evidence.

    Captured responses may carry PostgreSQL values (Decimal scores, datetime
    stamps) inside retrieval diagnostics; losing the whole journal record to
    a TypeError would defeat the journal's purpose. Anything else stays
    rejected: opaque objects are refused, not silently stringified.
    """
    from decimal import Decimal
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

_TABLES = {
    "requests": """CREATE TABLE requests (
        request_id TEXT PRIMARY KEY, database_identity TEXT NOT NULL,
        actor_id TEXT NOT NULL, learner_id TEXT NOT NULL, job_id TEXT,
        learning_session_id TEXT, client_message_id TEXT, requested_at TEXT NOT NULL)""",
    "calls": """CREATE TABLE calls (
        call_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(request_id),
        purpose TEXT NOT NULL, backend TEXT NOT NULL, call_index INTEGER NOT NULL CHECK(call_index > 0),
        prepared_at TEXT NOT NULL, UNIQUE(request_id, call_index))""",
    "outcomes": """CREATE TABLE outcomes (
        call_id TEXT PRIMARY KEY REFERENCES calls(call_id),
        payload TEXT NOT NULL, recorded_at TEXT NOT NULL)""",
}
_TRIGGERS = {
    f"{table}_no_{operation.lower()}": f"""CREATE TRIGGER {table}_no_{operation.lower()}
        BEFORE {operation} ON {table} BEGIN
        SELECT RAISE(ABORT, 'immutable chat response journal'); END"""
    for table in _TABLES for operation in ("UPDATE", "DELETE")
}
_SELECT = """SELECT r.*, c.call_id, c.purpose, c.backend, c.call_index, c.prepared_at,
    o.payload, o.recorded_at FROM requests r JOIN calls c USING(request_id)
    LEFT JOIN outcomes o USING(call_id) WHERE r.database_identity=?"""


def default_chat_journal_path() -> Path:
    value = os.getenv("LOCAL_CHAT_RESPONSE_JOURNAL")
    if value is not None and not value.strip():
        raise ResponseJournalError("Chat response journal path is empty")
    return Path(value) if value is not None else DEFAULT_PATH


default_journal_path = default_chat_journal_path


def _metadata(value, *, optional: bool = False) -> bool:
    if value is None:
        return optional
    if not isinstance(value, str) or not value or len(value) > 200:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


class ChatResponseJournal:
    def __init__(self, path: str | Path, database_identity: str = ""):
        if not str(path).strip():
            raise ResponseJournalError("Chat response journal path is empty")
        if not isinstance(database_identity, str) or len(database_identity) > 200:
            raise ResponseJournalError("Chat response journal database identity is invalid")
        try:
            database_identity.encode("utf-8")
            self.path = Path(path).resolve()
        except (OSError, TypeError, ValueError, RuntimeError, UnicodeError):
            raise ResponseJournalError("Chat response journal path or identity is invalid") from None
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
            if write:
                # Audit F-072: the journal holds full child chat payloads. mkdir's
                # mode is ignored for an existing directory and sqlite3 uses the
                # umask, so restrict the file and its sidecars explicitly.
                for private in (self.path, *(self.path.with_name(self.path.name + suffix)
                                             for suffix in ("-wal", "-shm", "-journal"))):
                    if private.exists():
                        os.chmod(private, 0o600)
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
            self._schema(connection, create=write)
            yield connection
            if write:
                connection.commit()
        except (OSError, sqlite3.Error, UnicodeError):
            raise ResponseJournalError("Chat response journal is unavailable; no automatic replay") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _schema(connection, *, create: bool):
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        objects = {row["name"]: (row["type"], row["sql"]) for row in connection.execute(
            "SELECT name,type,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
        expected = {**{name: ("table", sql) for name, sql in _TABLES.items()},
                    **{name: ("trigger", sql) for name, sql in _TRIGGERS.items()}}
        if create and not objects and application_id == 0 and version == 0:
            for _, statement in expected.values():
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            return
        normalized = lambda value: " ".join(value.split()) if isinstance(value, str) else value
        if (application_id != APPLICATION_ID or version != SCHEMA_VERSION
                or objects.keys() != expected.keys()
                or any(objects[name][0] != kind or normalized(objects[name][1]) != normalized(sql)
                       for name, (kind, sql) in expected.items())):
            raise ResponseJournalError("Chat response journal has an unsupported identity or schema")

    def begin_request(self, *, actor_id: str, learner_id: str, job_id: str | None = None,
                      learning_session_id: str | None = None,
                      client_message_id: str | None = None) -> str:
        if (not all(_metadata(value) for value in (actor_id, learner_id))
                or not all(_metadata(value, optional=True)
                           for value in (job_id, learning_session_id, client_message_id))):
            raise ResponseJournalError("Chat response journal request metadata is invalid")
        request_id = uuid4().hex
        with self._connect(write=True) as connection:
            connection.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?)",
                (request_id, self.database_identity, actor_id, learner_id, job_id,
                 learning_session_id, client_message_id, datetime.now(timezone.utc).isoformat()))
        return request_id

    def prepare(self, *, request_id: str, purpose: str, backend: str) -> str:
        if not all(_metadata(value) for value in (request_id, purpose, backend)):
            raise ResponseJournalError("Chat response journal call metadata is invalid")
        call_id = uuid4().hex
        with self._connect(write=True) as connection:
            parent = connection.execute("SELECT 1 FROM requests WHERE request_id=? AND database_identity=?",
                                        (request_id, self.database_identity)).fetchone()
            if parent is None:
                raise ResponseJournalError("Chat request does not belong to this journal database")
            call_index = connection.execute(
                "SELECT COALESCE(MAX(call_index),0)+1 FROM calls WHERE request_id=?", (request_id,)).fetchone()[0]
            connection.execute("INSERT INTO calls VALUES (?,?,?,?,?,?)",
                (call_id, request_id, purpose, backend, call_index, datetime.now(timezone.utc).isoformat()))
        return call_id

    def record(self, call_id: str, *, outcome: str, raw_response=None,
               error_type: str | None = None, elapsed_seconds: float = 0.0) -> None:
        if (not _metadata(call_id) or not isinstance(outcome, str) or outcome not in {"returned", "raised"}
                or type(elapsed_seconds) not in (int, float)
                or error_type is not None and (not isinstance(error_type, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,199}", error_type) is None)):
            raise ResponseJournalError("Chat response journal outcome is invalid")
        try:
            valid_elapsed = math.isfinite(elapsed_seconds) and elapsed_seconds >= 0
        except (OverflowError, TypeError, ValueError):
            valid_elapsed = False
        if not valid_elapsed:
            raise ResponseJournalError("Chat response journal outcome is invalid")
        try:
            payload = json.dumps({"outcome": outcome, "raw_response": raw_response,
                                  "error_type": error_type, "elapsed_seconds": elapsed_seconds},
                                 ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False, default=_journal_default)
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise ResponseJournalError("Chat response cannot be serialized durably") from None
        with self._connect(write=True) as connection:
            parent = connection.execute("""SELECT 1 FROM calls c JOIN requests r USING(request_id)
                WHERE c.call_id=? AND r.database_identity=?""", (call_id, self.database_identity)).fetchone()
            if parent is None:
                raise ResponseJournalError("Chat response does not belong to this journal database")
            existing = connection.execute("SELECT payload FROM outcomes WHERE call_id=?", (call_id,)).fetchone()
            if existing is not None:
                if existing["payload"] != payload:
                    raise ResponseJournalError("Chat response journal outcome is immutable")
                return
            connection.execute("INSERT INTO outcomes VALUES (?,?,?)",
                               (call_id, payload, datetime.now(timezone.utc).isoformat()))

    @staticmethod
    def _project(row) -> dict:
        result = dict(row)
        payload = result.pop("payload")
        if payload is None:
            result.update(outcome=None, raw_response=None, error_type=None, elapsed_seconds=None)
            return result
        try:
            decoded = json.loads(payload)
            if (not isinstance(decoded, dict) or set(decoded) != {
                    "outcome", "raw_response", "error_type", "elapsed_seconds"}
                    or not isinstance(decoded["outcome"], str)
                    or decoded["outcome"] not in {"returned", "raised"}):
                raise ValueError("invalid payload")
            result.update(decoded)
        except (TypeError, ValueError, RecursionError):
            raise ResponseJournalError("Chat response journal outcome cannot be read") from None
        return result

    def get(self, call_id: str) -> dict:
        if not self.path.exists():
            raise KeyError(call_id)
        with self._connect() as connection:
            row = connection.execute(_SELECT + " AND c.call_id=?",
                                     (self.database_identity, call_id)).fetchone()
            if row is None:
                raise KeyError(call_id)
            return self._project(row)

    def list_calls(self, *, request_id: str | None = None, job_id: str | None = None,
                   learning_session_id: str | None = None,
                   client_message_id: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        with self._connect() as connection:
            query = _SELECT
            values = [self.database_identity]
            for name, value in (("request_id", request_id), ("job_id", job_id),
                                ("learning_session_id", learning_session_id),
                                ("client_message_id", client_message_id)):
                if value is not None:
                    if not _metadata(value):
                        raise ResponseJournalError("Chat response journal selector is invalid")
                    query += f" AND r.{name}=?"
                    values.append(value)
            rows = connection.execute(query + " ORDER BY r.requested_at,r.request_id,c.call_index", values).fetchall()
            return [self._project(row) for row in rows]


@dataclass
class CaptureContext:
    journal: ChatResponseJournal
    request_id: str
    failure: ResponseJournalError | None = None

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure from None

    def failed(self, error: ResponseJournalError) -> None:
        if self.failure is None:
            self.failure = error


_context: ContextVar[CaptureContext | None] = ContextVar("chat_response_journal", default=None)


def current_capture_context() -> CaptureContext | None:
    return _context.get()


@contextmanager
def journal_request(journal: ChatResponseJournal, **metadata):
    previous = current_capture_context()
    if previous is not None:
        previous.check()
    try:
        request_id = journal.begin_request(**metadata)
    except ResponseJournalError as error:
        if previous is not None:
            previous.failed(error)
        raise
    context = CaptureContext(journal, request_id)
    token = _context.set(context)
    try:
        yield context
        context.check()
    except BaseException:
        if context.failure is not None and previous is not None:
            previous.failed(context.failure)
        context.check()
        raise
    finally:
        _context.reset(token)


class CapturedCall:
    def __init__(self, context: CaptureContext, call_id: str):
        self.context = context
        self.call_id = call_id
        self.started = monotonic()
        self.completed = False

    def _record(self, *, outcome, raw_response=None, error_type=None):
        self.context.check()
        if self.completed:
            return
        try:
            self.context.journal.record(self.call_id, outcome=outcome, raw_response=raw_response,
                                       error_type=error_type, elapsed_seconds=max(0.0, monotonic() - self.started))
        except ResponseJournalError as error:
            self.context.failed(error)
            raise
        self.completed = True

    def returned(self, raw_response) -> None:
        self._record(outcome="returned", raw_response=raw_response)

    def raised(self, exc: BaseException, raw_response=None) -> None:
        if isinstance(exc, ResponseJournalError):
            self.context.failed(exc)
            raise exc from None
        name = type(exc).__name__
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,199}", name) is None:
            name = "Exception"
        if raw_response is None:
            raw_response = getattr(exc, "raw_response", None)
        completion = getattr(exc, "completion", None)
        if raw_response is None and is_dataclass(completion) and not isinstance(completion, type):
            try:
                raw_response = {"completion": asdict(completion)}
            except (TypeError, ValueError, RecursionError):
                error = ResponseJournalError("Chat completion diagnostics cannot be serialized durably")
                self.context.failed(error)
                raise error from None
        self._record(outcome="raised", raw_response=raw_response, error_type=name)


@contextmanager
def captured_call(purpose: str, backend: str):
    context = current_capture_context()
    if context is None:
        yield None
        return
    context.check()
    try:
        call_id = context.journal.prepare(request_id=context.request_id, purpose=purpose, backend=backend)
    except ResponseJournalError as error:
        context.failed(error)
        raise
    capture = CapturedCall(context, call_id)
    try:
        yield capture
        context.check()
    except BaseException as error:
        if isinstance(error, ResponseJournalError):
            context.failed(error)
        elif not capture.completed:
            capture.raised(error)
        context.check()
        raise
