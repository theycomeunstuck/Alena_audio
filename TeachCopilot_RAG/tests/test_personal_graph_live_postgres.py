"""Live personal-graph checks against real Ollama, Neo4j and a disposable PostgreSQL database.

Every test is marked ``live`` and additionally skips when the graph stack is
unusable in this environment (graphiti-core package, Neo4j bolt endpoint or the
graph password file). Skips never fail: without Neo4j the product itself
degrades the unified memory channel with graceful reasons, and this module must
reflect that contract instead of demanding a service that is not running.
"""
import importlib.util
import json
import os
import shutil
import socket
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import requests
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.chat_response_journal import ChatResponseJournal
from local_app.config import Settings
from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
from local_app.memory_service import MemoryWorker
from local_app.memory_store import get_learner_memory_snapshot
from local_app.personal_memory import PersonalMemoryProvider
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from local_app.store import Store


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
TEACHER_NOTE = (
    "Ученица решала примеры на сложение дробей с одинаковыми знаменателями "
    "и уточнила у преподавателя, как подписывать числитель и знаменатель."
)
PASSWORD = "Live personal graph password 123"
NEO4J_BOLT = ("127.0.0.1", 17687)
NEO4J_SKIP_REASON = "Neo4j is not running / password file missing"


def _skip_unless_explicitly_selected(request):
    """Keep these checks out of default runs; only an explicit -m selection runs them.

    The ``live`` marker is registered in pyproject.toml, but pytest still runs
    marked tests when no ``-m`` expression is given. A default collection must
    skip them; ``-m live`` (or any expression pytest used to select this item)
    executes them.
    """
    if not str(request.config.getoption("markexpr") or "").strip():
        pytest.skip("live checks are deselected by default; run with -m live")


def _ollama_reachable() -> bool:
    try:
        response = requests.get(Settings.from_env().ollama_url.rstrip("/") + "/api/version", timeout=2)
    except requests.RequestException:
        return False
    return getattr(response, "status_code", None) == 200


def _require_neo4j():
    """Bolt endpoint and graph password file must both be available."""
    try:
        with socket.create_connection(NEO4J_BOLT, timeout=2):
            pass
    except OSError:
        pytest.skip(NEO4J_SKIP_REASON)
    if not Path(Settings.from_env().personal_graph_password_file).is_file():
        pytest.skip(NEO4J_SKIP_REASON)


class RecordingRuntime(OllamaRuntime):
    """Delegate to the real runtime while keeping every chat request payload."""

    def __init__(self, settings):
        super().__init__(settings)
        self.chat_payloads = []

    def chat(self, messages, *args, **kwargs):
        self.chat_payloads.append([dict(message) for message in messages])
        return super().chat(messages, *args, **kwargs)

    def captured_requests(self):
        return [json.dumps(payload, ensure_ascii=False) for payload in self.chat_payloads]


def _registry_state(store, run_id):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT state FROM teachcopilot.personal_graph_indexes WHERE run_id=%s", (run_id,))
        row = cur.fetchone()
    return row["state"] if row else None


@pytest.fixture(autouse=True)
def _live_requires_selected_services(request):
    _skip_unless_explicitly_selected(request)
    if not _ollama_reachable():
        pytest.skip("Ollama is not running")


@pytest.fixture(scope="module")
def graph_store(request, tmp_path_factory):
    """One disposable PostgreSQL database shared by the module's live checks."""
    _skip_unless_explicitly_selected(request)
    if not _ollama_reachable():
        pytest.skip("Ollama is not running")
    if importlib.util.find_spec("graphiti_core") is None:
        pytest.skip("graphiti-core is not installed in this environment")
    if importlib.util.find_spec("neo4j") is None:
        pytest.skip("neo4j driver is not installed in this environment")
    _require_neo4j()
    dsn = os.getenv("TEACHCOPILOT_TEST_DSN")
    if not dsn:
        pytest.skip("set TEACHCOPILOT_TEST_DSN for the disposable PostgreSQL database")
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn
    require_local_dsn(dsn)
    name = DATABASE_PREFIX + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(dsn, dbname="postgres"))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        store = Store(make_dsn(dsn, dbname=name))
        store.migrate()
        repo = Path(__file__).resolve().parents[2]
        scratch = tmp_path_factory.mktemp("graph-live-repo")
        shutil.copytree(repo / "learner-data/catalog", scratch / "learner-data/catalog")
        learners = scratch / "learner-data/learners"
        learners.mkdir()
        shutil.copy(repo / "learner-data/demo/learners/demo-nika-04.json", learners)
        store.import_repository(scratch)
        yield store
    finally:
        with admin.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        admin.close()


@pytest.fixture(scope="module")
def graph_cycle(graph_store, tmp_path_factory):
    """Build and approve one real personal graph index, then reuse it across checks."""
    store = graph_store
    base = Settings.from_env()
    settings = replace(base, database_url=str(store.dsn), context_tokens=max(base.context_tokens, 16384),
                       experimental_personal_graph_enabled=True)
    runtime = RecordingRuntime(settings)
    teacher = store.create_account("live-graph-teacher@synthetic.test", "Teacher",
                                   hash_password(PASSWORD), "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account("live-graph-student@synthetic.test", "Student",
                                           hash_password(PASSWORD), LEARNER, teacher["id"])
    store.add_memory_source(LEARNER, TOPIC, TEACHER_NOTE, "teacher", approved=True, author_role="teacher")
    journal_dir = tmp_path_factory.mktemp("graph-live-journals")
    app = create_app(settings=settings, store=store, runtime=runtime,
                     rag_service=RagService(store=store, settings=settings, runtime=runtime),
                     chat_response_journal=ChatResponseJournal(
                         journal_dir / "chat-responses.sqlite3",
                         database_identity=database_fingerprint(store.dsn)))
    with TestClient(app) as client:

        def auth(account):
            response = client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
            assert response.status_code == 200
            return {"Authorization": "Bearer " + response.json()["access_token"]}

        teacher_headers, student_headers = auth(teacher), auth(student)
        queued = client.post(f"/learners/{LEARNER}/next-step", headers=teacher_headers,
                             json={"topic_id": TOPIC, "method": "personal_graph"})
        assert queued.status_code == 202, queued.text
        worker = MemoryWorker(store, runtime, response_journal=MemoryResponseJournal(
            journal_dir / "memory-responses.sqlite3", database_identity=database_fingerprint(store.dsn)))
        finished = worker.run_once()
        assert finished["status"] == "completed", finished.get("error")
        snapshot = get_learner_memory_snapshot(store, LEARNER, None, include_personal_graph_snapshot=True)
        version = snapshot["versions"][0]
        confirmed = client.post(f"/learners/{LEARNER}/memory/versions/{version['id']}/confirm",
                                headers=teacher_headers, json={"expected_revision": snapshot["revision"]})
        assert confirmed.status_code == 200, confirmed.text
        yield SimpleNamespace(store=store, settings=settings, runtime=runtime, client=client,
                              teacher_headers=teacher_headers, student_headers=student_headers,
                              version=version, state=version["state"],
                              run_id=str(version["state"]["archive_run_id"]))


@pytest.mark.live
def test_live_personal_graph_cycle(graph_cycle):
    c = graph_cycle
    assert _registry_state(c.store, c.run_id) == "ready"
    session = c.client.post(f"/learners/{LEARNER}/learning-sessions", headers=c.student_headers,
                            json={"topic_id": TOPIC}).json()["learning_session_id"]
    response = c.client.post("/v1/chat/completions", headers=c.student_headers, json={
        "learning_session_id": session, "client_message_id": "live-graph-chat",
        "metadata": {"memory_method": "personal_graph"},
        "messages": [{"role": "user", "content": "Продолжим занятие по дробям: что нам стоит повторить?"}]})
    assert response.status_code == 200, response.text
    answer = response.json()["choices"][0]["message"]["content"]
    assert answer.strip()
    with_memory = [payload for payload in c.runtime.captured_requests()
                   if "<approved_memory_untrusted>" in payload]
    assert with_memory
    assert any("числитель" in payload for payload in with_memory)


@pytest.mark.live
def test_live_unified_hybrid_uses_graph_channel(graph_cycle):
    c = graph_cycle
    snapshot = get_learner_memory_snapshot(c.store, LEARNER, None, include_personal_graph_snapshot=True)
    job = {"job_id": "live-hybrid-probe", "learner_id": LEARNER, "topic_id": TOPIC, "kind": "next_step",
           "method": "personal_memory", "provider": "local",
           "base_revision": snapshot["revision"], "source_revision": snapshot["source_revision"]}
    question = "Что дальше по теме дробей?"
    state, context = PersonalMemoryProvider(c.store, c.runtime).prepare(job, snapshot, question)
    diagnostics = context["retrieval_diagnostics"]
    channel = diagnostics["graph_channel"]
    # graph_channel "used" means non-empty source_ids with no degradation reason:
    # the unified hybrid consumed the ready graph index built by this module.
    assert channel.get("source_ids"), channel
    assert channel.get("reason") is None
    assert diagnostics["channels"].get("graph")
    assert state.get("graph_channel") == channel

    # The same snapshot without the graph channel degrades with a graceful reason.
    disabled_provider = PersonalMemoryProvider(c.store, OllamaRuntime(replace(c.settings, memory_graph_channel=False)))
    disabled_state, disabled_context = disabled_provider.prepare(job, snapshot, question)
    disabled_channel = disabled_context["retrieval_diagnostics"]["graph_channel"]
    assert disabled_channel.get("reason") == "channel_disabled"
    assert not disabled_channel.get("source_ids")
    assert not disabled_context["retrieval_diagnostics"]["channels"].get("graph")
    assert disabled_state.get("retrieval_mode") == "hybrid"
