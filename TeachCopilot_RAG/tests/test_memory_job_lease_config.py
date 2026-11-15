"""F-029: the memory job lease is configurable via LOCAL_JOB_LEASE_SECONDS.

The worker claim, the worker's pre-model-call heartbeat and both interactive
chat claim paths hardcoded a 600-second lease, while the lease is only renewed
between model calls, so a claim-to-finish window longer than the lease
deterministically discarded completed generations as unknown_outcome. The lease
now comes from Settings (LOCAL_JOB_LEASE_SECONDS, default 600, minimum 60,
ceiling 3600 to match the memory store's applied clamp) at every site.

Pure-Python tests: no database, no network. The stores are in-memory fakes
that record the lease_seconds every claim/heartbeat call receives.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from local_app.config import Settings
from local_app.memory_service import MemoryWorker


# --- Settings: default, env override, sane minimum ---------------------------


def test_settings_default_job_lease_is_600():
    assert Settings.job_lease_seconds == 600
    assert Settings().job_lease_seconds == 600


def test_from_env_reads_local_job_lease_seconds(monkeypatch):
    monkeypatch.delenv("LOCAL_JOB_LEASE_SECONDS", raising=False)
    assert Settings.from_env().job_lease_seconds == 600
    monkeypatch.setenv("LOCAL_JOB_LEASE_SECONDS", "900")
    assert Settings.from_env().job_lease_seconds == 900


def test_job_lease_below_minimum_is_rejected(monkeypatch):
    monkeypatch.setenv("LOCAL_JOB_LEASE_SECONDS", "30")
    with pytest.raises(ValueError):
        Settings.from_env()
    with pytest.raises(ValueError):
        Settings(job_lease_seconds=30)


def test_job_lease_above_the_store_ceiling_is_rejected():
    # memory_store.py clamps every applied lease to 3600; Settings must not
    # advertise a value the store would silently shorten.
    with pytest.raises(ValueError):
        Settings(job_lease_seconds=3601)


# --- Worker claim and pre-model-call heartbeat -------------------------------

ACTION = {"action": "clarify_missing_observation", "reason": "Уточнить условия самостоятельной работы.",
          "evidence_turn_ids": [], "unknowns": ["Условия помощи неизвестны."], "review_required": True}


class WorkerLeaseStore:
    """test_memory_service.WorkerStore plus lease capture on claim and heartbeat."""

    def __init__(self):
        self.job = {"job_id": "job-1", "learner_id": "learner-a", "topic_id": "math.test",
                    "kind": "next_step", "method": "tail", "base_revision": 1, "provider": "local"}
        self.snapshot = {"revision": 1, "versions": [], "sources": [
            {"source_id": "student-1", "author_role": "student",
             "text": "Начал решать пример.", "observed_on": "2026-10-01"}]}
        self.finished = None
        self.claim_leases = []
        self.heartbeat_leases = []

    def claim_memory_job(self, worker_id, lease_seconds=600):
        self.claim_leases.append(lease_seconds)
        return deepcopy(self.job)

    def get_learner_memory(self, learner_id, topic_id=None):
        return deepcopy(self.snapshot)

    def get_card(self, learner_id):
        return {"grade": 4, "legal_name": "Never prompt this", "journal": ["private"]}

    def heartbeat_memory_job(self, job_id, worker_id, lease_seconds=600):
        self.heartbeat_leases.append(lease_seconds)
        return True

    def start_memory_job_call(self, job_id, worker_id):
        return dict(self.job)

    def finish_memory_job(self, job_id, **kwargs):
        self.finished = kwargs
        return {"job_id": job_id, "status": "completed"}

    def get_memory_job(self, job_id):
        return self.finished


class ActionRuntime:
    """The test_memory_service.Runtime shape plus an injectable settings object."""
    context_tokens = 8192

    def __init__(self, settings=None):
        self.action = dict(ACTION)
        if settings is not None:
            self.settings = settings

    def chat(self, messages, **kwargs):
        return {"message": {"content": json.dumps(self.action)},
                "prompt_eval_count": 250, "eval_count": 40}


def run_worker(settings=None):
    store = WorkerLeaseStore()
    MemoryWorker(store, ActionRuntime(settings=settings)).run_once()
    return store


def test_worker_claims_with_the_default_lease_and_renews_it_before_model_calls():
    store = run_worker()
    assert store.claim_leases == [600]
    assert store.heartbeat_leases == [600]


def test_worker_claim_and_heartbeat_use_the_configured_lease():
    store = run_worker(settings=SimpleNamespace(job_lease_seconds=900))
    assert store.claim_leases == [900]
    assert store.heartbeat_leases == [900]


def test_worker_lease_floor_holds_for_a_runtime_below_the_minimum():
    # Settings rejects LOCAL_JOB_LEASE_SECONDS below 60, but the worker helper is
    # the last guard before the store's own clamp: a runtime carrying a smaller
    # value must still claim and heartbeat with the 60-second minimum, never a
    # lease shorter than the minimum claim window.
    store = run_worker(settings=SimpleNamespace(job_lease_seconds=10))
    assert store.claim_leases == [60]
    assert store.heartbeat_leases == [60]


# --- Interactive chat claim (both api.py claim branches) ---------------------


class ChatLeaseStore:
    """Minimal durable-chat store driving the legacy /v1/chat/completions path."""

    def __init__(self):
        self.accounts = {}
        self.sessions = {}
        self.assignments = set()
        self.cards = {"learner-a": {"learner_id": "learner-a", "pseudonym": "A", "grade": 4,
                                    "rag_context": {}, "legal_name": "Private A"}}
        self.chat_claim_leases = []

    def add_account(self, account_id, email, role, learner_id=None,
                    password="correct horse battery staple"):
        from local_app.auth import hash_password
        self.accounts[email] = {"account_id": account_id, "email": email, "display_name": email,
                                "role": role, "learner_id": learner_id,
                                "password_hash": hash_password(password)}
        if learner_id:
            self.assignments.add((account_id, learner_id))
        return self.accounts[email]

    def get_account(self, email=None, account_id=None):
        if email:
            return self.accounts.get(email)
        return next((a for a in self.accounts.values() if a["account_id"] == account_id), None)

    def create_session(self, account_id, token_hash, expires_at):
        self.sessions[token_hash] = {"account_id": account_id, "expires_at": expires_at}

    def get_session(self, token_hash):
        return self.sessions.get(token_hash)

    def revoke_session(self, token_hash):
        return self.sessions.pop(token_hash, None) is not None

    def authorize_learner_access(self, account_id, learner_id):
        return (account_id, learner_id) in self.assignments

    def list_cards(self):
        return list(self.cards.values())

    def get_card(self, learner_id):
        return self.cards.get(learner_id)

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system", **kwargs):
        return {"job_id": "chat-job-1", "learner_id": learner_id, "kind": kind, "status": "queued"}

    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        self.chat_claim_leases.append(lease_seconds)
        return {"job_id": job_id, "learner_id": "learner-a", "kind": "chat", "status": "running"}

    def start_memory_job_call(self, job_id, worker_id):
        return {"job_id": job_id, "status": "running"}

    def finish_memory_job(self, job_id, **kwargs):
        return {"job_id": job_id, "status": "completed"}


class PollingChatLeaseStore(ChatLeaseStore):
    """Adds the connection-plumbing polling hook api.py prefers when present."""

    def claim_interactive_chat_job_polling(self, job_id, worker_id, lease_seconds=600,
                                           poll_interval=0.05, timeout=0.0):
        self.chat_claim_leases.append(lease_seconds)
        return {"job_id": job_id, "learner_id": "learner-a", "kind": "chat", "status": "running"}


class FakeRag:
    def answer(self, messages, **kwargs):
        return {"id": "chatcmpl-local", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
                "tutor_notes": "internal",
                "sources": [{"id": "source-1", "content": "protected excerpt",
                             "answer": "private answer", "source_file": "book.pdf"}]}

    def search(self, **kwargs):
        return []

    def reindex(self):
        return {"materials": 0}

    def status(self):
        return {"ready": True, "total": 1, "indexed": 1}


class FakeRuntime:
    def complete(self, prompt, system_prompt=""):
        return "{}"

    def check_models(self):
        return {"ready": True, "models": ["chat", "embedding"]}


def chat_client(store, settings):
    from fastapi.testclient import TestClient
    from local_app.api import create_app
    store.add_account("student-1", "student@example.org", "student", "learner-a")
    app = create_app(settings=settings, store=store, runtime=FakeRuntime(), rag_service=FakeRag())
    return TestClient(app)


def post_chat(client):
    result = client.post("/auth/login", json={"email": "student@example.org",
                                              "password": "correct horse battery staple"})
    assert result.status_code == 200
    headers = {"Authorization": f"Bearer {result.json()['access_token']}"}
    return client.post("/v1/chat/completions", headers=headers,
                       json={"messages": [{"role": "user", "content": "Привет"}]})


def test_chat_claim_defaults_to_600_when_the_setting_is_absent():
    store = ChatLeaseStore()
    response = post_chat(chat_client(store, SimpleNamespace(session_hours=2, rag_limit=3)))
    assert response.status_code == 200
    assert store.chat_claim_leases == [600]


def test_chat_claim_uses_the_configured_lease():
    store = ChatLeaseStore()
    response = post_chat(chat_client(store, SimpleNamespace(session_hours=2, rag_limit=3,
                                                            job_lease_seconds=900)))
    assert response.status_code == 200
    assert store.chat_claim_leases == [900]


def test_chat_polling_claim_uses_the_configured_lease():
    store = PollingChatLeaseStore()
    response = post_chat(chat_client(store, SimpleNamespace(session_hours=2, rag_limit=3,
                                                            job_lease_seconds=900)))
    assert response.status_code == 200
    assert store.chat_claim_leases == [900]
