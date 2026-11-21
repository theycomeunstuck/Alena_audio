"""Live checks against the real local Ollama runtime and a disposable PostgreSQL database.

Every test here is marked ``live`` (registered in pyproject.toml) and skips
instead of failing when a service is unavailable. The model identity is never
hardcoded: settings come from ``Settings.from_env()`` (LOCAL_CHAT_MODEL /
LOCAL_EMBEDDING_MODEL with the config.py defaults), which the strict
``SourceModel`` gate requires to resolve to the installed local models.
"""
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from tests.live_guard import skip_unless_live_selected
import requests
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.chat_response_journal import ChatResponseJournal
from local_app.config import Settings
from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
from local_app.memory_service import MemoryWorker
from local_app.personal_graph_retrieval import VECTOR_DIMENSIONS
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
TEACHER_NOTE = (
    "Ученица решала примеры на сложение дробей с одинаковыми знаменателями "
    "и уточнила у преподавателя, как подписывать числитель и знаменатель."
)
STUDENT_TURN = "Я подписала числитель три, а знаменатель четыре, и сложила доли."
PASSWORD = "Live personal memory password 123"


def _skip_unless_explicitly_selected(request):
    """Keep these checks out of default runs; only an explicit -m selection runs them.

    The ``live`` marker is registered in pyproject.toml, but pytest still runs
    marked tests when no ``-m`` expression is given. A default collection must
    skip them; ``-m live`` (or any expression pytest used to select this item)
    executes them.
    """
    # Audit F-080: any unrelated -m filter used to enable live model calls.
    skip_unless_live_selected(request)


def _ollama_base_url() -> str:
    return Settings.from_env().ollama_url


def _ollama_reachable() -> bool:
    try:
        response = requests.get(_ollama_base_url().rstrip("/") + "/api/version", timeout=2)
    except requests.RequestException:
        return False
    return getattr(response, "status_code", None) == 200


def _db_free_settings() -> Settings:
    """Settings for runtime-only checks: no database, therefore no scheduler DSN."""
    return replace(Settings.from_env(), database_url="")


def _cycle_settings(store) -> Settings:
    # The unified memory chat path is verified at 16k context; 8k overflows the
    # prompt budget (413/502). Local env may raise it, never lower it below 16384.
    base = Settings.from_env()
    return replace(base, database_url=str(store.dsn), context_tokens=max(base.context_tokens, 16384))


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


@pytest.fixture(autouse=True)
def _live_requires_selected_services(request):
    _skip_unless_explicitly_selected(request)
    if not _ollama_reachable():
        pytest.skip("Ollama is not running")


@pytest.mark.live
def test_live_embed_matches_configured_model():
    runtime = OllamaRuntime(_db_free_settings())
    vectors = runtime.embed(["Дробь показывает часть целого."])
    assert len(vectors) == 1
    vector = vectors[0]
    assert len(vector) == VECTOR_DIMENSIONS
    assert all(math.isfinite(value) for value in vector)


@pytest.mark.live
def test_live_chat_generation_answers_in_language():
    runtime = OllamaRuntime(_db_free_settings())
    response = runtime.chat([
        {"role": "user",
         "content": "Ответь одним коротким предложением по-русски: что такое числитель дроби?"},
    ])
    content = response["message"]["content"]
    assert content.strip()
    assert any("а" <= character <= "я" or "А" <= character <= "Я" or character == "ё" for character in content)


@pytest.mark.live
def test_live_personal_memory_cycle(flow_store, tmp_path):
    store = flow_store
    settings = _cycle_settings(store)
    runtime = RecordingRuntime(settings)
    teacher = store.create_account("live-memory-teacher@synthetic.test", "Teacher",
                                   hash_password(PASSWORD), "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account("live-memory-student@synthetic.test", "Student",
                                           hash_password(PASSWORD), LEARNER, teacher["id"])
    # Seed literal approved sources: a teacher note and one learner turn inside a
    # completed session (only completed sessions join the personal graph archive).
    store.add_memory_source(LEARNER, TOPIC, TEACHER_NOTE, "teacher", approved=True, author_role="teacher")
    seeded = store.create_learning_session(LEARNER, "student", TOPIC)
    store.reserve_chat_message(seeded["id"], LEARNER, "seed-one", STUDENT_TURN, "student")
    store.complete_chat_message(seeded["id"], "seed-one", {"answer": "Подписывай части дроби по порядку."})
    completion = store.complete_learning_session(seeded["id"], LEARNER, "student")
    store.cancel_memory_job(completion["job_id"], "live test seeds sources directly")

    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    app = create_app(settings=settings, store=store, runtime=runtime,
                     rag_service=RagService(store=store, settings=settings, runtime=runtime),
                     chat_response_journal=ChatResponseJournal(
                         journal_dir / "chat-responses.sqlite3",
                         database_identity=database_fingerprint(store.dsn)))
    browser = TestClient(app)

    def auth(account):
        response = browser.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
        assert response.status_code == 200
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    teacher_headers, student_headers = auth(teacher), auth(student)

    queued = browser.post(f"/learners/{LEARNER}/next-step", headers=teacher_headers,
                          json={"topic_id": TOPIC, "method": "personal_memory"})
    assert queued.status_code == 202, queued.text
    worker = MemoryWorker(store, runtime, response_journal=MemoryResponseJournal(
        journal_dir / "memory-responses.sqlite3", database_identity=database_fingerprint(store.dsn)))
    finished = worker.run_once()
    assert finished["status"] == "completed", finished.get("error")

    snapshot = browser.get(f"/learners/{LEARNER}/memory", headers=teacher_headers).json()
    version_id = next(row["version_id"] for row in snapshot["versions"]
                      if (row.get("method") or (row.get("state") or {}).get("method")) == "personal_memory")
    confirmed = browser.post(f"/learners/{LEARNER}/memory/versions/{version_id}/confirm",
                             headers=teacher_headers, json={"expected_revision": snapshot["revision"]})
    assert confirmed.status_code == 200, confirmed.text

    learning_session = browser.post(f"/learners/{LEARNER}/learning-sessions", headers=student_headers,
                                    json={"topic_id": TOPIC}).json()["learning_session_id"]
    first = browser.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": learning_session, "client_message_id": "live-memory-one",
        "metadata": {"memory_method": "personal_memory"},
        "messages": [{"role": "user", "content": "Как найти числитель и знаменатель дроби?"}]})
    assert first.status_code == 200, first.text
    answer = first.json()["choices"][0]["message"]["content"]
    assert answer.strip()
    # The approved personal memory must reach the local generation request.
    assert any("<approved_memory_untrusted>" in payload for payload in runtime.captured_requests())

    # Forbidden-fact probe: a fact that no source ever contained must not be
    # invented by the model. The unique nonsense token never appears in any
    # seeded source, so an honest answer cannot contain it either.
    token = "квилворд-" + uuid4().hex[:8]
    probe_question = f"Правда, что на прошлом занятии мне объясняли правило {token}?"
    probe = browser.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": learning_session, "client_message_id": "live-memory-probe",
        "metadata": {"memory_method": "personal_memory"},
        "messages": [{"role": "user", "content": probe_question}]})
    assert probe.status_code == 200, probe.text
    probe_reply = probe.json()["choices"][0]["message"]["content"]
    assert probe_reply.strip()
    assert token.casefold() not in probe_reply.casefold(), probe_reply

    report_dir = os.getenv("MEMORY_LIVE_REPORT_DIR")
    if report_dir:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        (report_path / "personal-memory-live.json").write_text(json.dumps({
            "learner_id": LEARNER, "topic_id": TOPIC, "version_id": version_id,
            "chat_model": settings.chat_model, "embedding_model": settings.embedding_model,
            "context_tokens": settings.context_tokens,
            "student_question": "Как найти числитель и знаменатель дроби?",
            "answer": answer,
            "probe_question": probe_question, "probe_reply": probe_reply,
            "probe_token": token,
            "captured_chat_request_count": len(runtime.chat_payloads),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
