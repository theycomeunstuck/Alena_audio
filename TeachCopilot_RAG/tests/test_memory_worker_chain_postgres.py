"""Full personal-memory chain on disposable PostgreSQL without any model service.

The controlled runtime provides deterministic embeddings, the /api/tags stub
serves immutable local model digests, and the real worker, provider, retrieval,
pgvector cache, API approval and chat chain run against real PostgreSQL.
"""
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import random
import threading
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from local_app.personal_memory import PersonalMemoryProvider
from local_app.rag import RagService
from scripts.load_memory import ControlledRuntime
from tests.test_hybrid_retrieval_postgres import documented_graph_channel_degrade
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
PASSWORD = "Synthetic chain password 123"
SOURCE_TEXT = "Ученик сам составил дробь 3/4 и назвал числитель."
CORRECTION_TEXT = "Дробь 3/4 собрал преподаватель вместе с учеником; самостоятельность неизвестна."
CHAT_ANSWER = "Вспоминаем: в дроби 3/4 числитель стоит сверху."
QUESTION = "Что мы делали с дробью 3/4?"
VECTOR_DIMENSIONS = 1024


def deterministic_embedding(text):
    """Stable unit vector per exact text; no Ollama and no model download."""
    generator = random.Random("teachcopilot-chain-embed:" + text)
    values = [generator.uniform(-1.0, 1.0) for _ in range(VECTOR_DIMENSIONS)]
    norm = sum(value * value for value in values) ** 0.5
    return [value / norm for value in values]


class _ModelTagsHandler(BaseHTTPRequestHandler):
    """Loopback /api/tags stub with immutable digests for identity resolution."""

    def do_GET(self):
        if self.path.split("?")[0] != "/api/tags":
            self.send_error(404)
            return
        body = json.dumps({"models": [
            {"name": "gemma4:12b", "model": "gemma4:12b", "digest": "sha256:" + "a" * 64},
            {"name": "bge-m3", "model": "bge-m3", "digest": "b" * 64},
        ]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture
def model_tags_url():
    server = HTTPServer(("127.0.0.1", 0), _ModelTagsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


class ChainRuntime(ControlledRuntime):
    """ControlledRuntime with settings, deterministic embed and explicit action completion."""

    # The real answer pipeline bounds the whole prompt in UTF-8 bytes against
    # the model context; 16k leaves room for profile, own-conversation summary
    # and the approved memory body.
    context_tokens = 16384

    def __init__(self, settings):
        super().__init__(0)
        self.settings = settings
        self.embed_calls = []
        self.chat_prompts = []

    def chat(self, messages, response_format=None, max_output_tokens=None, **kwargs):
        fields = (response_format or {}).get("properties", {})
        if "answer" in fields and "evidence_turn_ids" not in fields:
            self.chat_prompts.append(deepcopy(messages))
            content = json.dumps({"answer": CHAT_ANSWER, "citations": []}, ensure_ascii=False)
            return {"message": {"role": "assistant", "content": content},
                    "prompt_eval_count": 150, "eval_count": 40, "done": True}
        response = super().chat(messages, response_format=response_format,
                                max_output_tokens=max_output_tokens, **kwargs)
        response["done"] = True
        return response

    def embed(self, texts):
        items = [texts] if isinstance(texts, str) else list(texts)
        self.embed_calls.append(list(items))
        return [deterministic_embedding(text) for text in items]

    @contextmanager
    def approved_memory_request(self):
        yield

    def resolve_chat_query(self, query, history, subject_id, **kwargs):
        return {"in_scope": True, "uses_prior_context": False, "uses_approved_memory": True,
                "personal_memory_recall": False, "needs_clarification": False,
                "query_source": "original_learner_message", "search_query": query}


class LocalMemoryRag(RagService):
    """The real answer pipeline; only curriculum search is replaced by a stub."""

    def search(self, *args, **kwargs):
        return []


def chain_settings(tags_url, tmp_path, *, graph_channel):
    return Settings(ollama_url=tags_url, context_tokens=16384,
                    memory_graph_channel=graph_channel,
                    personal_graph_password_file=str(tmp_path / "chain-missing-password.txt"))


def seed_chain_accounts_and_source(store):
    secret = hash_password(PASSWORD)
    teacher = store.create_account(f"chain-teacher-{uuid4().hex[:8]}@synthetic.test",
                                   "Chain teacher", secret, "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account(f"chain-student-{uuid4().hex[:8]}@synthetic.test",
                                           "Chain student", secret, LEARNER, teacher["id"])
    source = store.add_memory_source(LEARNER, TOPIC, SOURCE_TEXT, "teacher",
                                     approved=True, observed_on="2026-02-20")
    return teacher, student, source


def run_personal_memory_job(store, runtime, actor_id):
    store.enqueue_memory_job(LEARNER, TOPIC, "next_step", {}, actor_id=str(actor_id),
                             method="personal_memory")
    result = MemoryWorker(store, runtime).run_once()
    assert result["status"] == "completed", result.get("error")
    return result


def draft_version(store, version_id):
    snapshot = store.get_learner_memory(LEARNER, TOPIC)
    return snapshot, next(row for row in snapshot["versions"] if row["id"] == version_id)


def approved_memory_body(system_prompt):
    marker = "<approved_memory_untrusted>"
    start = system_prompt.index(marker) + len(marker)
    return system_prompt[start:system_prompt.index("</approved_memory_untrusted>", start)].strip()


def recorded_plain_hybrid_channel(channel):
    """The channel recorded why it degraded and contributed no graph hits."""
    return (channel["source_ids"] is None and channel["run_id"] is None
            and documented_graph_channel_degrade(channel["reason"]))


def test_worker_draft_api_approval_and_correction_chain(flow_store, model_tags_url, tmp_path):
    store = flow_store
    settings = chain_settings(model_tags_url, tmp_path, graph_channel=True)
    runtime = ChainRuntime(settings)
    teacher, student, source = seed_chain_accounts_and_source(store)

    # 1. The personal_memory job completes as an unapproved draft with recorded
    #    graph-channel diagnostics and an approved=False memory version.
    result = run_personal_memory_job(store, runtime, teacher["id"])
    assert result["method"] == "personal_memory"
    assert result["result"]["review_required"] is True
    version_id = result["result"]["memory_version_id"]
    _, draft = draft_version(store, version_id)
    assert draft["approved"] is False
    state = draft["state"]
    assert state["method"] == "personal_memory" and state["topic_id"] == TOPIC
    assert recorded_plain_hybrid_channel(state["graph_channel"])
    context = draft["context"]
    assert context["method"] == "personal_memory" and SOURCE_TEXT in context["text"]
    assert recorded_plain_hybrid_channel(context["retrieval_diagnostics"]["graph_channel"])
    assert set(context["evidence_ids"]) <= set(state["source_ids"])
    assert SOURCE_TEXT in str(context["sources"])

    # 2. Teacher approval through the API, then student recall through chat.
    browser = TestClient(create_app(settings=settings, store=store, runtime=runtime,
                                    rag_service=LocalMemoryRag(store=store, runtime=runtime,
                                                               settings=settings)))

    def auth(account):
        response = browser.post("/auth/login", json={"email": account["email"], "password": PASSWORD})
        assert response.status_code == 200
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    teacher_headers, student_headers = auth(teacher), auth(student)
    confirmed = browser.post(f"/learners/{LEARNER}/memory/versions/{version_id}/confirm",
                             headers=teacher_headers,
                             json={"expected_revision": store.get_revision(LEARNER)})
    assert confirmed.status_code == 200, confirmed.text

    session = browser.post(f"/learners/{LEARNER}/learning-sessions", headers=student_headers,
                           json={"topic_id": TOPIC}).json()["learning_session_id"]
    first = browser.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session, "client_message_id": "chain-recall",
        "metadata": {"memory_method": "personal_memory"},
        "messages": [{"role": "user", "content": QUESTION}]})
    assert first.status_code == 200, first.text
    assert first.json()["memory_method"] == "personal_memory"
    assert first.json()["choices"][0]["message"]["content"] == CHAT_ANSWER
    assert first.json()["answer_routing"]["backend"] == "local"
    system_prompt = runtime.chat_prompts[-1][0]["content"]
    assert "<approved_memory_untrusted>" in system_prompt
    assert SOURCE_TEXT in approved_memory_body(system_prompt)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, result FROM teachcopilot.memory_jobs WHERE idempotency_key=%s",
                    (f"chat:{session}:chain-recall",))
        chat_job = cur.fetchone()
    assert chat_job["status"] == "completed"
    assert chat_job["result"]["response_ready"] is True

    # 3. A teacher correction withdraws the grant: the old version goes stale
    #    and chat fails closed until a fresh draft is approved.
    correction = browser.post(f"/learners/{LEARNER}/memory/sources/{source['id']}/corrections",
                              headers=teacher_headers,
                              json={"text": CORRECTION_TEXT,
                                    "expected_revision": store.get_revision(LEARNER),
                                    "expected_source_revision": store.get_learner_source_revision(LEARNER),
                                    "client_correction_id": "chain-correction"})
    assert correction.status_code == 202, correction.text
    assert MemoryWorker(store, runtime).run_once()["status"] == "completed"
    _, old_version = draft_version(store, version_id)
    assert old_version["stale"] is True
    blocked = browser.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session, "client_message_id": "chain-blocked",
        "metadata": {"memory_method": "personal_memory"},
        "messages": [{"role": "user", "content": QUESTION}]})
    assert blocked.status_code == 409, blocked.text

    refreshed = run_personal_memory_job(store, runtime, teacher["id"])
    new_version_id = refreshed["result"]["memory_version_id"]
    _, new_draft = draft_version(store, new_version_id)
    assert new_draft["approved"] is False and new_draft["id"] != version_id
    assert CORRECTION_TEXT in new_draft["context"]["text"]
    assert SOURCE_TEXT in new_draft["context"]["text"]
    confirmed_again = browser.post(f"/learners/{LEARNER}/memory/versions/{new_version_id}/confirm",
                                   headers=teacher_headers,
                                   json={"expected_revision": store.get_revision(LEARNER)})
    assert confirmed_again.status_code == 200, confirmed_again.text
    recalled = browser.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session, "client_message_id": "chain-recall-after-correction",
        "metadata": {"memory_method": "personal_memory"},
        "messages": [{"role": "user", "content": QUESTION}]})
    assert recalled.status_code == 200, recalled.text
    recalled_body = approved_memory_body(runtime.chat_prompts[-1][0]["content"])
    assert CORRECTION_TEXT in recalled_body and SOURCE_TEXT in recalled_body


def test_disabled_graph_channel_keeps_plain_hybrid_selection(flow_store, model_tags_url, tmp_path):
    store = flow_store
    settings_on = chain_settings(model_tags_url, tmp_path, graph_channel=True)
    runtime_on = ChainRuntime(settings_on)
    teacher, student, _source = seed_chain_accounts_and_source(store)
    result = run_personal_memory_job(store, runtime_on, teacher["id"])
    version_id = result["result"]["memory_version_id"]
    _, draft = draft_version(store, version_id)
    assert recorded_plain_hybrid_channel(draft["state"]["graph_channel"])

    review_app = TestClient(create_app(settings=settings_on, store=store, runtime=runtime_on,
                                       rag_service=LocalMemoryRag(store=store, runtime=runtime_on,
                                                                  settings=settings_on)))
    login = review_app.post("/auth/login", json={"email": teacher["email"], "password": PASSWORD})
    assert login.status_code == 200
    teacher_headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    confirmed = review_app.post(f"/learners/{LEARNER}/memory/versions/{version_id}/confirm",
                                headers=teacher_headers,
                                json={"expected_revision": store.get_revision(LEARNER)})
    assert confirmed.status_code == 200, confirmed.text
    snapshot = store.get_learner_memory(LEARNER, TOPIC, include_personal_graph_snapshot=True)

    # The enabled-but-degraded channel and the disabled channel must select the
    # identical plain-hybrid sources for the same approved grant and question.
    provider_on = PersonalMemoryProvider(store, runtime_on)
    context_on = provider_on.chat_context(snapshot, LEARNER, TOPIC, QUESTION)
    assert recorded_plain_hybrid_channel(context_on["retrieval_diagnostics"]["graph_channel"])
    settings_off = chain_settings(model_tags_url, tmp_path, graph_channel=False)
    runtime_off = ChainRuntime(settings_off)
    context_off = PersonalMemoryProvider(store, runtime_off).chat_context(
        snapshot, LEARNER, TOPIC, QUESTION)
    disabled_channel = context_off["retrieval_diagnostics"]["graph_channel"]
    assert disabled_channel["source_ids"] is None
    assert disabled_channel["reason"] == "channel_disabled"
    assert context_off["sources"] == context_on["sources"]
    assert context_off["evidence_ids"] == context_on["evidence_ids"]
    assert context_off["text"] == context_on["text"]
    assert SOURCE_TEXT in context_off["text"]

    # The complete chat chain still works with the feature flag off and sends
    # exactly the same approved memory body to the model.
    browser = TestClient(create_app(settings=settings_off, store=store, runtime=runtime_off,
                                    rag_service=LocalMemoryRag(store=store, runtime=runtime_off,
                                                               settings=settings_off)))
    student_login = browser.post("/auth/login", json={"email": student["email"], "password": PASSWORD})
    assert student_login.status_code == 200
    student_headers = {"Authorization": "Bearer " + student_login.json()["access_token"]}
    session = browser.post(f"/learners/{LEARNER}/learning-sessions", headers=student_headers,
                           json={"topic_id": TOPIC}).json()["learning_session_id"]
    response = browser.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session, "client_message_id": "chain-disabled-recall",
        "metadata": {"memory_method": "personal_memory"},
        "messages": [{"role": "user", "content": QUESTION}]})
    assert response.status_code == 200, response.text
    assert response.json()["memory_method"] == "personal_memory"
    assert approved_memory_body(runtime_off.chat_prompts[-1][0]["content"]) == context_off["text"]
