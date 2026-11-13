"""Teacher-only memory diagnostics endpoints against an isolated PostgreSQL store."""
import hashlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.personal_graph_contract import personal_graph_snapshot_fingerprint
from local_app.personal_memory import PersonalMemoryProvider
from local_app.personal_memory_contract import (PERSONAL_MEMORY_CONTEXT_PROTOCOL,
                                                PERSONAL_MEMORY_SELECTION_VERSION)
from tests.test_memory_api_flow import InspectableRag, flow_store  # noqa: F401

LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
PASSWORD = "Synthetic diagnostics password 123"
SOURCE_TEXT = "Мы сравнивали дроби с одинаковым знаменателем и числителем на рисунке."


class FakeModel:
    """Deterministic local-model stand-in: no Ollama, only pure hashing."""

    context = 16384
    model = "gemma4:12b"
    base_url = "http://127.0.0.1:11434"
    model_identities = {"chat": {"name": "gemma4:12b", "digest": "a" * 64},
                        "embedding": {"name": "bge-m3", "digest": "b" * 64}}

    def check_model_identities(self):
        return None

    def embed_many(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            vector = [0.0] * 1024
            vector[digest[0]] = 1.0
            vectors.append(vector)
        return vectors


class DiagnosticsProvider(PersonalMemoryProvider):
    def _application_model(self):
        return FakeModel()


def _lesson(store):
    session = store.create_learning_session(LEARNER, "student", TOPIC)
    store.reserve_chat_message(session["id"], session["learner_id"], "turn-1", SOURCE_TEXT,
                               session["actor_id"])
    store.complete_chat_message(session["id"], "turn-1", {"answer": "Попробуй сравнить числители."})
    store.complete_learning_session(session["id"], LEARNER, "student")


def _approve_personal_memory(store):
    """Save and approve a valid personal_memory version without any model calls."""
    snapshot = store.get_learner_memory(LEARNER, None)
    sources = snapshot["sources"]
    source_ids = sorted(row["source_id"] for row in sources)
    fingerprint = personal_graph_snapshot_fingerprint(sources)
    state = {"method": "personal_memory", "topic_id": TOPIC, "learner_id": LEARNER,
             "source_revision": snapshot["source_revision"], "source_ids": source_ids,
             "source_context_protocol": PERSONAL_MEMORY_CONTEXT_PROTOCOL,
             "archive_snapshot_fingerprint": fingerprint,
             "source_snapshot_id": "personal-memory-" + fingerprint,
             "retrieval_mode": "hybrid", "model_identities": FakeModel.model_identities,
             "selection_recipe_fingerprint": "c" * 64,
             "selection_version": PERSONAL_MEMORY_SELECTION_VERSION}
    evidence = source_ids[:1]
    context = {"method": "personal_memory", "topic_id": TOPIC, "text": sources[0]["text"],
               "evidence_ids": evidence,
               "sources": [{"source_id": row["source_id"], "role": row["author_role"],
                            "text": row["text"], "topic_id": row["topic_id"] or "general"}
                           for row in sources if row["source_id"] in evidence]}
    draft = store.save_memory_version(LEARNER, TOPIC, state, context=context,
                                      base_revision=snapshot["revision"],
                                      source_revision=snapshot["source_revision"])
    store.approve_memory_version(LEARNER, draft["id"], "teacher", snapshot["revision"])


@pytest.fixture
def diagnostics_client(flow_store, tmp_path):
    store = flow_store
    teacher = store.create_account("diagnostics-teacher@synthetic.test", "Teacher",
                                   hash_password(PASSWORD), "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account("diagnostics-student@synthetic.test", "Student",
                                           hash_password(PASSWORD), LEARNER, teacher["id"])
    settings = Settings(personal_memory_enabled=True, context_tokens=16384, session_hours=2,
                        personal_graph_password_file=str(tmp_path / "absent-neo4j-password.txt"))
    runtime = SimpleNamespace(settings=settings, context_tokens=settings.context_tokens)
    app = create_app(settings=settings, store=store, runtime=runtime, rag_service=InspectableRag(),
                     personal_memory_factory=lambda bound_store, bound_runtime:
                         DiagnosticsProvider(bound_store, bound_runtime))
    client = TestClient(app)
    headers = {}
    for name, account in (("teacher", teacher), ("student", student)):
        login = client.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
        assert login.status_code == 200
        headers[name] = {"Authorization": "Bearer " + login.json()["access_token"]}
    return SimpleNamespace(store=store, client=client, headers=headers, settings=settings)


def test_teacher_index_reports_flags_and_readonly_registry_status(diagnostics_client):
    case = diagnostics_client
    _lesson(case.store)
    response = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/index",
                               headers=case.headers["teacher"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["learner_id"] == LEARNER
    flags = data["flags"]
    assert flags["personal_memory_enabled"] is True
    assert flags["retrieval_mode"] == "hybrid"
    assert flags["memory_graph_channel"] is True
    assert flags["experimental_personal_graph_enabled"] is False
    assert flags["context_tokens"] == 16384
    assert flags["models"] == {"chat": "gemma4:12b", "embedding": "bge-m3"}
    assert data["budgets"]["selection_bytes"] == 10000 and data["budgets"]["memory_bytes"] == 12000
    assert data["snapshot"]["source_count"] == 1
    assert data["snapshot"]["lesson_count"] == 1
    assert len(data["snapshot"]["snapshot_fingerprint"]) == 64
    index = data["index"]
    assert index["ready"] is False and index["run_id"] is None and index["manifest"] is None
    # No Neo4j password file and no ready registry row: a clean, leaked-free reason.
    assert index["password_file_available"] is False
    assert isinstance(index["reason"], str) and index["reason"]
    assert "postgres" not in index["reason"].lower() and "password" not in str(index["manifest"]).lower()
    # The plain retrieval endpoint stays conservative without an approved version.
    empty = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/retrieval",
                            headers=case.headers["teacher"], params={"q": SOURCE_TEXT})
    assert empty.status_code == 200 and empty.json()["status"] == "not_approved"
    assert empty.json()["hint"]


def test_student_and_foreign_requests_are_forbidden(diagnostics_client):
    case = diagnostics_client
    for path in (f"/learners/{LEARNER}/memory/diagnostics/index",
                 f"/learners/{LEARNER}/memory/diagnostics/retrieval"):
        assert case.client.get(path, headers=case.headers["student"],
                               params={"q": SOURCE_TEXT}).status_code == 403
    assert case.client.get(f"/learners/{LEARNER}/memory/diagnostics/index").status_code == 401


def test_retrieval_with_approved_version_returns_channels_and_graph_reason(diagnostics_client):
    case = diagnostics_client
    _lesson(case.store)
    _approve_personal_memory(case.store)
    response = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/retrieval",
                               headers=case.headers["teacher"],
                               params={"q": "какие дроби мы сравнивали", "mode": "hybrid"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok" and data["topic_id"] == TOPIC
    assert data["requested_mode"] == "hybrid" and data["retrieval_mode"] == "hybrid"
    assert data["budget"]["requested"] is None and data["budget"]["applied"] == 10000
    channels = data["channels"]
    assert set(channels) >= {"chronological", "lexical", "vector"}
    assert all(channels[name] for name in ("chronological", "lexical", "vector"))
    ranked = data["ranked_source_ids"]
    snapshot = case.store.get_learner_memory(LEARNER, None)
    allowed = {row["source_id"] for row in snapshot["sources"]}
    assert ranked and set(ranked) <= allowed
    graph = data["graph_channel"]
    assert graph["source_ids"] is None and graph["run_id"] is None
    # Without Neo4j the graph channel degrades to plain hybrid with a recorded,
    # credential-free reason (for example password_file_unavailable).
    assert isinstance(graph["reason"], str) and graph["reason"]
    assert "@" not in graph["reason"] and "postgresql" not in graph["reason"].lower()
    assert data["selection_warnings"] == []
    context = data["context"]
    assert context["method"] == "personal_memory" and context["topic_id"] == TOPIC
    assert context["evidence_ids"] and set(context["evidence_ids"]) <= allowed
    assert context["text"] and SOURCE_TEXT in context["text"]
    assert context["retrieval_diagnostics"]["graph_channel"]["reason"] == graph["reason"]
    assert context["selection_budget"]["limit"] == 10000

    # graph_hybrid is accepted and stays on the same unified path with a reason.
    hybrid = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/retrieval",
                             headers=case.headers["teacher"],
                             params={"q": "какие дроби мы сравнивали", "mode": "graph_hybrid"})
    assert hybrid.status_code == 200 and hybrid.json()["requested_mode"] == "graph_hybrid"
    assert isinstance(hybrid.json()["graph_channel"]["reason"], str)


def test_retrieval_without_approved_memory_stays_conservative(diagnostics_client):
    case = diagnostics_client
    _lesson(case.store)
    response = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/retrieval",
                               headers=case.headers["teacher"], params={"q": SOURCE_TEXT})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "not_approved" and data["hint"]
    assert "context" not in data and "channels" not in data


def test_retrieval_honors_explicit_smaller_and_larger_budgets(diagnostics_client):
    case = diagnostics_client
    _lesson(case.store)
    _approve_personal_memory(case.store)
    params = {"q": "какие дроби мы сравнивали", "budget": 20000}
    large = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/retrieval",
                            headers=case.headers["teacher"], params=params)
    assert large.status_code == 200, large.text
    assert large.json()["budget"] == {"requested": 20000, "applied": 20000}
    assert large.json()["context"]["selection_budget"]["limit"] == 20000
    tiny = case.client.get(f"/learners/{LEARNER}/memory/diagnostics/retrieval",
                           headers=case.headers["teacher"],
                           params={"q": "какие дроби мы сравнивали", "budget": 500})
    # Nothing fits into a 500-byte payload: a clean 409 instead of a broken draft.
    assert tiny.status_code == 409 and tiny.json()["detail"]


def test_retrieval_rejects_oversized_blank_or_invalid_input(diagnostics_client):
    case = diagnostics_client
    base = f"/learners/{LEARNER}/memory/diagnostics/retrieval"
    headers = case.headers["teacher"]
    assert case.client.get(base, headers=headers, params={"q": "x" * 2001}).status_code == 422
    assert case.client.get(base, headers=headers, params={"q": "   "}).status_code == 422
    assert case.client.get(base, headers=headers).status_code == 422
    assert case.client.get(base, headers=headers,
                           params={"q": "вопрос", "mode": "vector"}).status_code == 422
    assert case.client.get(base, headers=headers,
                           params={"q": "вопрос", "budget": 100}).status_code == 422
    assert case.client.get(base, headers=headers,
                           params={"q": "вопрос", "budget": 40000}).status_code == 422
