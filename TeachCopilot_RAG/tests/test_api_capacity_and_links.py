"""Regression tests for audit fixes F-032, F-033 and F-035.

F-032 – AnswerCapacityError from the external answer adapter must classify as
generation capacity (retryable, never unknown_outcome).
F-033 – store.assign_learner must guard and insert inside one transaction with
the account's links locked, so two concurrent link requests cannot attach one
student account to two learners.
F-035 – chat admission polling must reuse one connection per waiter and bound
the number of waiters; excess requests get an immediate 503 chat-busy.
"""
from __future__ import annotations

import re
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

import local_app.api as api_module
import local_app.store as store_module
from local_app.api import create_app
from local_app.auth import hash_password, hash_session_token


# ---------------------------------------------------------------------------
# F-032 – capacity classification
# ---------------------------------------------------------------------------

def test_answer_capacity_error_classifies_as_generation_capacity():
    from local_app.answer_adapter import AnswerCapacityError
    from local_app.generation_scheduler import GenerationCapacityError
    from local_app.api import _is_generation_capacity

    exc = AnswerCapacityError("External answer capacity is full; retry explicitly later")
    # The adapter's capacity rejection is the same retryable capacity family as
    # the scheduler's; RuntimeError handling must be preserved either way.
    assert issubclass(AnswerCapacityError, GenerationCapacityError)
    assert issubclass(AnswerCapacityError, RuntimeError)
    assert _is_generation_capacity(exc) is True
    assert str(exc) == "External answer capacity is full; retry explicitly later"


def test_answer_capacity_error_keeps_started_calls_out_of_unknown_outcome():
    from local_app.answer_adapter import AnswerCapacityError
    from local_app.api import _is_generation_capacity

    # Mirrors the api.py cleanup rule: unknown_outcome = call_started and not capacity.
    exc = AnswerCapacityError("External answer capacity is full; retry explicitly later")
    capacity = _is_generation_capacity(exc)
    assert capacity is True
    assert (True and not capacity) is False


def test_generation_run_budget_exceeded_stays_outside_capacity_classification():
    from local_app.api import _is_generation_capacity
    from local_app.generation_run_budget import GenerationRunBudgetExceeded

    assert _is_generation_capacity(GenerationRunBudgetExceeded("deadline reached")) is False


# ---------------------------------------------------------------------------
# F-033 – store.assign_learner serialization
# ---------------------------------------------------------------------------

class _RowLock:
    """Exclusive lock with owner tracking, modelling SELECT ... FOR UPDATE rows."""

    def __init__(self):
        self._lock = threading.Lock()
        self._owner = None
        self._depth = 0

    def acquire(self, owner, timeout=10.0):
        if self._owner == owner:
            self._depth += 1
            return
        if not self._lock.acquire(timeout=timeout):
            raise RuntimeError("fake row lock timed out – transaction did not serialize")
        self._owner = owner
        self._depth = 1

    def release(self, owner):
        if self._owner != owner:
            raise RuntimeError("fake row lock released by a non-owner")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


class _FakeLinkDB:
    """In-memory teachcopilot.accounts/account_learners with READ COMMITTED rules."""

    def __init__(self):
        self.accounts = {}
        self.links = []  # committed (account_id, learner_id, relationship)
        self.locks = {}

    def lock(self, key, owner):
        self.locks.setdefault(key, _RowLock()).acquire(owner)

    def release_all(self, owner):
        for key in list(self.locks):
            lock = self.locks[key]
            if lock._owner == owner:
                lock.release(owner)


class _FakeLinkCursor:
    def __init__(self, conn):
        self._conn = conn
        self._rows: list = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params=()):
        db = self._conn.db
        conn = self._conn
        sql = " ".join(query.split())
        if re.search(r"SELECT\s+id\s+FROM\s+teachcopilot\.accounts\s+WHERE\s+id=%s\s+FOR\s+UPDATE", sql, re.I):
            (account_id,) = params
            db.lock(("accounts", account_id), conn.owner)
            self._rows = [dict(db.accounts[account_id])] if account_id in db.accounts else []
            self.rowcount = len(self._rows)
        elif re.search(r"SELECT\s+learner_id\s+FROM\s+teachcopilot\.account_learners"
                       r"\s+WHERE\s+account_id=%s\s+FOR\s+UPDATE", sql, re.I):
            (account_id,) = params
            db.lock(("account_learners", account_id), conn.owner)
            self._rows = [{"learner_id": learner_id}
                          for (account, learner_id, _relationship) in conn.visible_links(account_id)]
            self.rowcount = len(self._rows)
        elif re.search(r"INSERT\s+INTO\s+teachcopilot\.account_learners", sql, re.I):
            account_id, learner_id, relationship = params
            conn.staged.append((str(account_id), str(learner_id), relationship))
            self.rowcount = 1
        else:
            raise AssertionError(f"fake store received unexpected SQL: {sql[:160]}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeLinkConnection:
    """One transaction against the fake database (a fresh connection per call)."""

    def __init__(self, db: _FakeLinkDB):
        self.db = db
        self.owner = id(self)
        self.staged: list = []
        self.finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    def cursor(self, *args, **kwargs):
        return _FakeLinkCursor(self)

    def visible_links(self, account_id):
        # READ COMMITTED: committed rows plus this transaction's own writes.
        rows = [(a, lid, rel) for (a, lid, rel) in self.db.links if a == account_id]
        for (a, lid, rel) in self.staged:
            if a == account_id:
                rows = [(x, y, z) for (x, y, z) in rows if y != lid]
                rows.append((a, lid, rel))
        return rows

    def commit(self):
        for (a, lid, rel) in self.staged:
            self.db.links = [(x, y, z) for (x, y, z) in self.db.links if not (x == a and y == lid)]
            self.db.links.append((a, lid, rel))
        self.staged = []
        self.finished = True
        self.db.release_all(self.owner)

    def rollback(self):
        self.staged = []
        self.finished = True
        self.db.release_all(self.owner)

    def close(self):
        if not self.finished:
            self.rollback()


@pytest.fixture()
def link_store(monkeypatch):
    db = _FakeLinkDB()
    db.accounts["student-1"] = {"id": "student-1", "role": "student"}
    db.accounts["teacher-1"] = {"id": "teacher-1", "role": "teacher"}
    monkeypatch.setattr(store_module, "connect", lambda dsn: _FakeLinkConnection(db))
    store = store_module.Store("dbname=unused")
    return store, db


def test_concurrent_student_links_serialize_to_a_single_learner(link_store):
    store, db = link_store
    barrier = threading.Barrier(2, timeout=5)
    results = {}

    def link(name, learner_id):
        barrier.wait()
        try:
            store.assign_learner("student-1", learner_id, relationship="student")
            results[name] = ("ok", learner_id)
        except Exception as exc:  # the losing racer must fail, not double-link
            results[name] = (type(exc).__name__, str(exc))

    threads = [threading.Thread(target=link, args=("t1", "learner-1")),
               threading.Thread(target=link, args=("t2", "learner-2"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "assign_learner deadlocked"

    ok = [value for value in results.values() if value[0] == "ok"]
    failed = [value for value in results.values() if value[0] != "ok"]
    assert len(ok) == 1, results
    assert len(failed) == 1 and failed[0][0] == "PermissionError", results
    assert "already linked" in failed[0][1]
    assert len(db.links) == 1, db.links
    assert db.links[0][1] == ok[0][1]


def test_concurrent_relink_of_the_same_learner_stays_idempotent(link_store):
    store, db = link_store
    barrier = threading.Barrier(2, timeout=5)
    errors = []

    def relink():
        barrier.wait()
        try:
            store.assign_learner("student-1", "learner-1", relationship="student")
        except Exception as exc:  # pragma: no cover – reported for diagnosis
            errors.append(exc)

    threads = [threading.Thread(target=relink), threading.Thread(target=relink)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    assert [(a, lid) for (a, lid, _rel) in db.links] == [("student-1", "learner-1")]


def test_student_link_to_a_second_learner_is_rejected_by_the_store_guard(link_store):
    store, db = link_store
    db.links.append(("student-1", "learner-9", "student"))
    with pytest.raises(PermissionError, match="already linked"):
        store.assign_learner("student-1", "learner-1", relationship="student")
    assert [(a, lid) for (a, lid, _rel) in db.links] == [("student-1", "learner-9")]


def test_teacher_relationship_still_allows_multiple_learners(link_store):
    store, db = link_store
    store.assign_learner("teacher-1", "learner-1")
    store.assign_learner("teacher-1", "learner-2")
    assert sorted((lid, rel) for (_a, lid, rel) in db.links) == [("learner-1", "teacher"), ("learner-2", "teacher")]


# ---------------------------------------------------------------------------
# F-035 – admission waiter cap and single-connection polling
# ---------------------------------------------------------------------------

class _PollCursor:
    """Executes the real durable-claim SQL against an in-memory queue model."""

    def __init__(self, model):
        self._model = model
        self._rows: list = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, query, params=()):
        sql = " ".join(query.split())
        model = self._model
        if "pg_advisory_xact_lock" in sql:
            self._rows, self.rowcount = [], 0
        elif sql.startswith("UPDATE teachcopilot.memory_jobs SET status='running'"):
            model.job_row["status"] = "running"
            self._rows, self.rowcount = [dict(model.job_row)], 1
        elif sql.startswith("UPDATE teachcopilot.memory_jobs SET status='queued'"):
            self._rows, self.rowcount = [], 0
        elif sql.startswith("UPDATE teachcopilot.memory_jobs SET status='unknown_outcome'"):
            self._rows, self.rowcount = [], 0
        elif sql.startswith("UPDATE teachcopilot.memory_jobs j SET status='failed'"):
            self._rows, self.rowcount = [], 0
        elif sql.startswith("UPDATE teachcopilot.learning_messages"):
            self._rows, self.rowcount = [], 0
        elif sql.startswith("SELECT j.*") and "FOR UPDATE OF j" in sql:
            model.poll_selects += 1
            if model.poll_selects >= model.claim_after_selects:
                self._rows, self.rowcount = [dict(model.job_row)], 1
            else:
                self._rows, self.rowcount = [], 0
        else:
            raise AssertionError(f"fake poll connection received unexpected SQL: {sql[:160]}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _PollConnection:
    def __init__(self, model):
        self._model = model
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    def cursor(self, *args, **kwargs):
        return _PollCursor(self._model)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


@pytest.fixture()
def poll_store(monkeypatch):
    model = types.SimpleNamespace(
        poll_selects=0,
        claim_after_selects=3,
        job_row={"id": "job-1", "kind": "chat", "provider": "local", "status": "queued",
                 "learner_id": "learner-a", "topic_id": "general", "priority": 10},
    )
    connects = []
    monkeypatch.setattr(store_module, "connect", lambda dsn: (connects.append(1), _PollConnection(model))[1])
    return store_module.Store("dbname=unused"), model, connects


def test_chat_admission_poll_reuses_a_single_connection(poll_store):
    store, model, connects = poll_store
    claimed = store.claim_interactive_chat_job_polling("job-1", "worker-1", lease_seconds=600,
                                                       poll_interval=0.001, timeout=5.0)
    assert str(claimed["job_id"]) == "job-1"
    assert claimed["status"] == "running"
    assert model.poll_selects == 3, "the claim should be retried until the job becomes claimable"
    assert len(connects) == 1, f"expected one connection for the whole poll, got {len(connects)}"
    assert model.job_row["status"] == "running"


def test_chat_admission_poll_stops_on_deadline_on_the_same_connection(poll_store):
    store, model, connects = poll_store
    model.claim_after_selects = 10 ** 9  # never claimable
    result = store.claim_interactive_chat_job_polling("job-1", "worker-1", lease_seconds=600,
                                                      poll_interval=0.001, timeout=0.03)
    assert result is None
    assert model.poll_selects >= 1
    assert len(connects) == 1, "deadline polling must not open further connections"


class _FakeApiStore:
    """Fake store covering login, learner access and the legacy chat admission path."""

    def __init__(self):
        self.accounts = {}
        self.sessions = {}
        self.assignments = set()
        self.cards = {
            "learner-a": {"learner_id": "learner-a", "pseudonym": "A", "grade": 4},
            "learner-b": {"learner_id": "learner-b", "pseudonym": "B", "grade": 4},
        }
        self.assigned = []
        self.enqueue_calls = []
        self.claim_calls = 0
        self.claim_result = None
        self.cancelled = []

    def add_account(self, account_id, email, role, password="correct horse battery staple"):
        self.accounts[email] = {"account_id": account_id, "email": email, "display_name": email,
                                "role": role, "password_hash": hash_password(password), "active": True}
        return self.accounts[email]

    def get_account(self, email=None, account_id=None):
        if email:
            return self.accounts.get(email)
        return next((a for a in self.accounts.values() if a["account_id"] == account_id), None)

    def create_session(self, account_id, token_hash, expires_at):
        self.sessions[token_hash] = {"account_id": account_id, "expires_at": expires_at}

    def get_session(self, token_hash):
        return self.sessions.get(token_hash)

    def list_cards(self):
        return [dict(card) for card in self.cards.values()]

    def get_card(self, learner_id):
        return self.cards.get(learner_id)

    def authorize_learner_access(self, account_id, learner_id):
        return (account_id, learner_id) in self.assignments

    def assign_learner(self, account_id, learner_id, relationship="teacher"):
        self.assigned.append((account_id, learner_id, relationship))
        self.assignments.add((account_id, learner_id))

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system", **kwargs):
        self.enqueue_calls.append((learner_id, topic_id, kind))
        return {"job_id": "job-1", "status": "queued", "learner_id": learner_id, "kind": kind}

    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        self.claim_calls += 1
        return None if self.claim_result is None else dict(self.claim_result)

    def cancel_memory_job(self, job_id, error="cancelled before provider call"):
        self.cancelled.append((job_id, error))


class _FakeRagService:
    def answer(self, messages, **kwargs):
        return {"id": "chatcmpl-local", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}]}


def _make_client(store):
    settings = types.SimpleNamespace(session_hours=2, rag_limit=3, request_timeout=0.05)
    runtime = types.SimpleNamespace(complete=lambda prompt, system_prompt="": "{}",
                                    check_models=lambda: {"ready": True, "models": ["chat"]})
    app = create_app(settings=settings, store=store, runtime=runtime, rag_service=_FakeRagService())
    client = TestClient(app)
    teacher = store.add_account("teacher-1", "teacher@example.org", "teacher")
    store.assignments.add(("teacher-1", "learner-a"))
    login = client.post("/auth/login", json={"email": teacher["email"], "password": "correct horse battery staple"})
    assert login.status_code == 200
    return client, {"Authorization": f"Bearer {login.json()['access_token']}"}


def _chat_payload():
    return {"learner_id": "learner-a", "messages": [{"role": "user", "content": "hello"}]}


def test_admission_waiter_cap_rejects_immediately_with_503(monkeypatch):
    store = _FakeApiStore()
    client, headers = _make_client(store)
    monkeypatch.setattr(api_module, "_chat_admission_waiters", api_module.CHAT_ADMISSION_WAITER_CAP)

    response = client.post("/v1/chat/completions", headers=headers, json=_chat_payload())

    assert response.status_code == 503
    assert "chat busy" in response.json()["detail"].lower()
    # Rejected before silently queueing work and before claiming any slot.
    assert store.enqueue_calls == []
    assert store.claim_calls == 0
    assert api_module._chat_admission_waiters == api_module.CHAT_ADMISSION_WAITER_CAP


def test_admission_slot_counter_reserves_and_releases():
    before = api_module._chat_admission_waiters
    assert api_module._reserve_admission_slot() is True
    assert api_module._chat_admission_waiters == before + 1
    api_module._release_admission_slot()
    assert api_module._chat_admission_waiters == before
    monkey_cap = api_module.CHAT_ADMISSION_WAITER_CAP
    saved = api_module._chat_admission_waiters
    try:
        api_module._chat_admission_waiters = monkey_cap
        assert api_module._reserve_admission_slot() is False
        assert api_module._chat_admission_waiters == monkey_cap
    finally:
        api_module._chat_admission_waiters = saved


def test_chat_admission_deadline_cancels_the_job_with_429_and_releases_the_slot():
    store = _FakeApiStore()
    store.claim_result = None
    client, headers = _make_client(store)

    response = client.post("/v1/chat/completions", headers=headers, json=_chat_payload())

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "10"
    assert store.enqueue_calls == [("learner-a", "general", "chat")]
    assert store.claim_calls >= 1
    assert store.cancelled == [("job-1", "chat_queue_wait_timeout")]
    assert api_module._chat_admission_waiters == 0


def test_link_student_endpoint_flow_is_unchanged():
    store = _FakeApiStore()
    store.add_account("student-9", "student@example.org", "student")
    client, headers = _make_client(store)

    linked = client.post("/learners/learner-a/student", headers=headers,
                         json={"student_email": "student@example.org"})
    assert linked.status_code == 200
    assert linked.json() == {"learner_id": "learner-a", "student_account_id": "student-9", "linked": True}
    assert store.assigned == [("student-9", "learner-a", "student")]

    store.assignments.add(("teacher-1", "learner-b"))
    conflict = client.post("/learners/learner-b/student", headers=headers,
                           json={"student_email": "student@example.org"})
    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "student account is already linked"