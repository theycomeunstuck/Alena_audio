"""Ordinary chat durability against disposable PostgreSQL and controlled providers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import psycopg2
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.chat_response_journal import ChatResponseJournal
from local_app.config import Settings
from local_app.memory_response_journal import ResponseJournalError, database_fingerprint
from local_app.rag import RagService
from local_app.runtime import IncompleteCompletionError, OllamaRuntime
from tests.test_memory_api_flow import flow_store  # noqa: F401
from tests.test_memory_worker_recovery import DatabaseGate
from tests.test_social_chat import LocalSession, MaterialsDatabase


LEARNER = "demo-nika-04"
TOPIC = "math.g4.numbers.division_by_1_2_digit"
RAW_THINKING = "PRIVATE raw model reasoning"
RAW_DIAGNOSTIC = "PRIVATE resolver decision"
RAW_ENVELOPE = {
    "choices": [{"message": {"role": "assistant", "content": "Проверь один шаг.",
                               "reasoning_content": RAW_THINKING}}],
    "_local_diagnostics": {"type": "InvalidQueryResolutionError", "model_response": RAW_DIAGNOSTIC},
    "usage": {"completion_tokens": 7},
}


class ControlledRag:
    def __init__(self, response=None):
        self.response = deepcopy(RAW_ENVELOPE if response is None else response)
        self.calls = []

    def answer(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return deepcopy(self.response)


class NoModels:
    def chat(self, *args, **kwargs):
        raise AssertionError("The API test must not invoke any model")

    def embed(self, *args, **kwargs):
        raise AssertionError("The API test must not invoke embeddings")


@pytest.fixture
def chat_api(flow_store, tmp_path):
    store = flow_store
    password = "Synthetic chat durability password 123"
    password_hash = hash_password(password)
    teacher = store.create_account("journal-teacher@synthetic.test", "Teacher", password_hash, "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    student = store.create_student_account("journal-student@synthetic.test", "Student", password_hash,
                                           LEARNER, teacher["id"])
    foreign = store.create_account("journal-foreign@synthetic.test", "Foreign", password_hash, "teacher")
    settings = Settings(database_url=store.dsn, autonomous_division_enabled=True,
                        chat_response_journal_path=str(tmp_path / "chat-responses.sqlite3"))
    journal = ChatResponseJournal(settings.chat_response_journal_path,
                                  database_identity=database_fingerprint(store.dsn))
    rag = ControlledRag()
    app = create_app(settings=settings, store=store, runtime=NoModels(), rag_service=rag,
                     chat_response_journal=journal)
    with TestClient(app) as browser:
        def auth(account):
            response = browser.post("/auth/login", json={"email": account["email"], "password": password})
            assert response.status_code == 200, response.text
            return {"Authorization": "Bearer " + response.json()["access_token"]}

        yield SimpleNamespace(store=store, settings=settings, app=app, browser=browser,
                              journal=journal, rag=rag, student=student, teacher=teacher,
                              student_headers=auth(student), teacher_headers=auth(teacher),
                              foreign_headers=auth(foreign))


def session(api, *, mode="standard"):
    response = api.browser.post(f"/learners/{LEARNER}/learning-sessions", headers=api.student_headers,
                                json={"topic_id": TOPIC, "mode": mode})
    assert response.status_code == 201, response.text
    return response.json()["learning_session_id"]


def payload(session_id=None, message_id="chat-one"):
    result = {"messages": [{"role": "user", "content": "Как проверить деление?"}]}
    if session_id is not None:
        result.update(learning_session_id=session_id, client_message_id=message_id)
    return result


def send(api, body, *, headers=None):
    return api.browser.post("/v1/chat/completions", headers=headers or api.student_headers, json=body)


def message_row(api, session_id):
    with api.store._connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT status,response FROM teachcopilot.learning_messages WHERE learning_session_id=%s",
                    (session_id,))
        return dict(cur.fetchone())


def assert_private_journal_ids_absent(public, rows):
    serialized = json.dumps(public, ensure_ascii=False)
    for row in rows:
        assert row["call_id"] not in serialized
        assert row["request_id"] not in serialized
    assert "chat_response_journal" not in serialized


def test_full_service_response_is_durable_before_projection_and_replay_does_not_generate(chat_api):
    api = chat_api
    sid = session(api)
    body = payload(sid)
    response = send(api, body)
    assert response.status_code == 200, response.text
    rows = api.journal.list_calls(learning_session_id=sid, client_message_id="chat-one")
    assert len(rows) == 1
    row = rows[0]
    assert row["purpose"] == "rag_answer" and row["backend"] == "service"
    assert row["raw_response"] == RAW_ENVELOPE and row["outcome"] == "returned"
    assert row["actor_id"] == api.student["id"] and row["learner_id"] == LEARNER
    assert row["job_id"]
    public = response.json()
    assert RAW_DIAGNOSTIC not in response.text and RAW_THINKING not in response.text
    assert_private_journal_ids_absent(public, rows)
    assert_private_journal_ids_absent(message_row(api, sid)["response"], rows)
    replay = send(api, body)
    assert replay.status_code == 200 and replay.json() == public
    assert len(api.rag.calls) == 1 and api.journal.list_calls() == rows
    completed = api.browser.post(f"/learning-sessions/{sid}/complete", headers=api.student_headers)
    assert completed.status_code == 202, completed.text
    archive = api.browser.get(f"/learners/{LEARNER}/learning-sessions/{sid}/history", headers=api.student_headers)
    assert archive.status_code == 200, archive.text
    assert RAW_DIAGNOSTIC not in archive.text and RAW_THINKING not in archive.text
    assert_private_journal_ids_absent(archive.json(), rows)
    student_job = api.browser.get(f"/memory/jobs/{row['job_id']}", headers=api.student_headers)
    assert student_job.status_code == 403
    teacher_job = api.browser.get(f"/memory/jobs/{row['job_id']}", headers=api.teacher_headers)
    assert teacher_job.status_code == 200, teacher_job.text
    assert_private_journal_ids_absent(teacher_job.json(), rows)
    assert api.journal.get(row["call_id"])["raw_response"] == RAW_ENVELOPE


@pytest.mark.parametrize("method,after_commit", [
    ("finish_memory_job", False), ("finish_memory_job", True),
    ("complete_chat_message", False), ("complete_chat_message", True),
])
def test_postgres_commit_ack_failure_preserves_raw_response_without_second_generation(chat_api, monkeypatch,
                                                                                     method, after_commit):
    api = chat_api
    sid = session(api)
    body = payload(sid)
    original = getattr(api.store, method)

    def unavailable(*args, **kwargs):
        # Cleanup may finish the job with an error after the first failure.
        if method == "finish_memory_job" and kwargs.get("error") is not None:
            return original(*args, **kwargs)
        if after_commit:
            original(*args, **kwargs)
        raise psycopg2.OperationalError("PRIVATE database password/path must not escape")

    with monkeypatch.context() as fault:
        fault.setattr(api.store, method, unavailable)
        response = send(api, body)
    assert response.status_code == 503, response.text
    assert "PRIVATE" not in response.text
    rows = api.journal.list_calls(learning_session_id=sid)
    assert len(rows) == len(api.rag.calls) == 1 and rows[0]["raw_response"] == RAW_ENVELOPE
    row = message_row(api, sid)
    committed_message = method == "complete_chat_message" and after_commit
    # A lost database acknowledgement must not cause another terminal write.
    # The persisted pending reservation blocks retries until normal recovery.
    assert row["status"] == ("completed" if committed_message else "pending")
    repeated = send(api, body)
    assert repeated.status_code == (200 if committed_message else 409), repeated.text
    assert len(api.rag.calls) == 1 and api.journal.list_calls() == rows
    if not committed_message:
        with api.store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE teachcopilot.learning_messages SET created_at=now()-interval '601 seconds' "
                           "WHERE learning_session_id=%s", (sid,))
            cursor.execute("UPDATE teachcopilot.memory_jobs SET lease_until=now()-interval '1 second' "
                           "WHERE status='running'")
        api.store.recover_memory_jobs()
        api.store.recover_chat_messages()
        assert message_row(api, sid)["status"] == "unknown_outcome"
        assert send(api, body).status_code == 409
        assert len(api.rag.calls) == 1 and api.journal.list_calls() == rows


def test_database_and_cleanup_outage_still_returns_sanitized_503_and_fresh_process_reads_raw(chat_api, monkeypatch):
    api = chat_api
    sid = session(api)

    def unavailable(*args, **kwargs):
        raise psycopg2.OperationalError("PRIVATE cleanup failure")

    with monkeypatch.context() as fault:
        fault.setattr(api.store, "finish_memory_job", unavailable)
        fault.setattr(api.store, "fail_chat_message", unavailable)
        response = send(api, payload(sid))
    assert response.status_code == 503, response.text
    assert "PRIVATE" not in response.text
    rows = api.journal.list_calls(learning_session_id=sid)
    assert len(rows) == len(api.rag.calls) == 1
    assert rows[0]["raw_response"] == RAW_ENVELOPE
    code = """
import json, sys
from local_app.chat_response_journal import ChatResponseJournal
journal = ChatResponseJournal(sys.argv[1], database_identity=sys.argv[2])
print(json.dumps(journal.get(sys.argv[3]), ensure_ascii=True))
"""
    child = subprocess.run([sys.executable, "-c", code, str(api.journal.path), api.journal.database_identity,
                            rows[0]["call_id"]], cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["raw_response"] == RAW_ENVELOPE
    # A failed cleanup leaves the original reservation blocked, never regenerated.
    repeated = send(api, payload(sid))
    assert repeated.status_code == 409 and len(api.rag.calls) == 1


def test_real_postgres_transport_loss_after_reply_keeps_journal_readable_offline(chat_api):
    api = chat_api
    original_dsn = api.store.dsn
    with DatabaseGate(original_dsn) as gate:
        gate.allowed.set()
        api.store.dsn = gate.dsn
        api.journal.database_identity = database_fingerprint(gate.dsn)
        try:
            sid = session(api)

            class DisconnectAfterReply(ControlledRag):
                def answer(self, *args, **kwargs):
                    response = super().answer(*args, **kwargs)
                    gate.allowed.clear()
                    return response

            rag = DisconnectAfterReply()
            api.app.state.rag_service = rag
            response = send(api, payload(sid))
            assert response.status_code == 503 and gate.refused.wait(1)
            row, = api.journal.list_calls()
            assert row["raw_response"] == RAW_ENVELOPE and len(rag.calls) == 1
            env = dict(os.environ, DATABASE_URL=gate.dsn,
                       LOCAL_CHAT_RESPONSE_JOURNAL=str(api.journal.path), OLLAMA_URL="http://127.0.0.1:9")
            child = subprocess.run([sys.executable, "scripts/inspect_chat_responses.py", "--call-id", row["call_id"],
                                    "--include-response"], cwd=Path(__file__).resolve().parents[1], env=env,
                                   capture_output=True, text=True, encoding="utf-8", timeout=15)
            assert child.returncode == 0, child.stderr
            assert json.loads(child.stdout)["calls"][0]["raw_response"] == RAW_ENVELOPE
            gate.allowed.set()
            assert send(api, payload(sid)).status_code == 409 and len(rag.calls) == 1
        finally:
            gate.allowed.set()
            api.store.dsn = original_dsn


def test_legacy_chat_without_session_has_durable_owned_service_response(chat_api):
    api = chat_api
    response = send(api, payload())
    assert response.status_code == 200, response.text
    rows = api.journal.list_calls()
    assert len(rows) == 1 and rows[0]["raw_response"] == RAW_ENVELOPE
    assert rows[0]["learning_session_id"] is None and rows[0]["client_message_id"] is None
    assert rows[0]["actor_id"] == api.student["id"] and rows[0]["job_id"]
    assert_private_journal_ids_absent(response.json(), rows)


def test_auth_foreign_invalid_explicit_and_autonomous_paths_create_no_chat_journal(chat_api):
    api = chat_api
    body = payload()
    assert api.browser.post("/v1/chat/completions", json=body).status_code == 401
    assert send(api, {**body, "learner_id": LEARNER}, headers=api.foreign_headers).status_code == 403
    assert send(api, {"messages": []}).status_code == 422
    sid = session(api)
    explicit = {"learning_session_id": sid, "client_message_id": "explicit", "messages": [],
                "learning_action": {"operation": "division_check", "dividend": 12, "divisor": 3}}
    response = send(api, explicit)
    assert response.status_code == 200, response.text
    assert send(api, explicit).json() == response.json()
    auto_sid = session(api, mode="autonomous_division")
    autonomous = {"learning_session_id": auto_sid, "client_message_id": "auto-one", "messages": [],
                  "learning_protocol": "autonomous-division-v1",
                  "learning_action": {"operation": "autonomous_division_start", "divisor_digits": 1}}
    response = send(api, autonomous)
    assert response.status_code == 200, response.text
    assert send(api, autonomous).json() == response.json()
    clarify = send(api, {**payload(auto_sid, "auto-clarify"), "learning_protocol": "autonomous-division-v1"})
    assert clarify.status_code == 200, clarify.text
    assert api.rag.calls == [] and api.journal.list_calls() == [] and not api.journal.path.exists()


class RawHttpSession(LocalSession):
    def post(self, url, json, timeout, allow_redirects):
        assert url == "http://127.0.0.1:11435/api/chat"
        self.requests.append(json)
        return self.response(deepcopy(self.replies.pop(0)))


def runtime_reply(content, **extra):
    return {"message": {"role": "assistant", "content": content, "thinking": RAW_THINKING},
            "done_reason": "stop", "model": "controlled-chat", "eval_count": 7, **extra}


def install_local_runtime(api, raw, *, structured=False, resolver=False):
    http = RawHttpSession(raw)
    runtime = OllamaRuntime(api.settings, session=http)
    runtime.generation_scheduler = None

    class DirectRuntimeRag:
        def __init__(self):
            self.calls = 0

        def answer(self, messages, **kwargs):
            self.calls += 1
            if structured:
                runtime.chat_json(messages)
                raise AssertionError("Malformed structured output must be rejected")
            result = runtime.chat(messages)
            return {"choices": [{"message": {"role": "assistant", "content": result["message"]["content"]}}]}

    rag = RagService(MaterialsDatabase(), runtime, api.settings) if resolver else DirectRuntimeRag()
    api.app.state.rag_service = rag
    return http, runtime, rag


@pytest.mark.parametrize("failure", ["truncated", "malformed_structured", "invalid_resolver", "none"])
def test_local_http_raw_response_is_captured_before_validation_and_normalization(chat_api, failure):
    api = chat_api
    sid = session(api)
    content = {"truncated": "Незавершённый ответ", "malformed_structured": '{"answer":',
               "invalid_resolver": '{"in_scope": "wrong", "uses_prior_context": false, "search_query": ""}',
               "none": "Проверь " + chr(0x2014) + " один шаг."}[failure]
    raw = runtime_reply(content, **({"done_reason": "length"} if failure == "truncated" else {}))
    http, runtime, rag = install_local_runtime(api, raw,
        structured=failure == "malformed_structured", resolver=failure == "invalid_resolver")
    response = send(api, payload(sid))
    assert response.status_code == (503 if failure in {"truncated", "malformed_structured"} else 200), response.text
    rows = api.journal.list_calls(learning_session_id=sid)
    local = [row for row in rows if row["purpose"] == "ollama_chat"]
    service = [row for row in rows if row["purpose"] == "rag_answer"]
    assert len(rows) == 2 and len(local) == len(service) == len(http.requests) == 1
    assert local[0]["backend"] == "local" and local[0]["raw_response"] == raw
    assert local[0]["outcome"] == "returned"
    assert local[0]["call_index"] > service[0]["call_index"]
    assert RAW_THINKING not in response.text
    if failure == "none":
        assert response.json()["choices"][0]["message"]["content"] == "Проверь – один шаг."
    elif failure == "invalid_resolver":
        assert service[0]["raw_response"]["_local_diagnostics"]["type"] == "InvalidQueryResolutionError"
        assert '"resolver_decision":' not in response.text
    else:
        assert service[0]["outcome"] == "raised" and service[0]["error_type"] == "IncompleteCompletionError"
    assert_private_journal_ids_absent(response.json(), rows)


def test_local_raw_response_precedes_run_budget_finish_failure(chat_api):
    api = chat_api
    sid = session(api)
    raw = runtime_reply("Проверь один шаг.")
    http, runtime, rag = install_local_runtime(api, raw)

    class FailedBudget:
        def reserve_generation(self, request):
            return 1

        def snapshot(self):
            return {"remaining_seconds": 30}

        def finish(self, *args, **kwargs):
            raise ConnectionError("PRIVATE budget database unavailable")

    runtime.generation_run_budget = FailedBudget()
    response = send(api, payload(sid))
    assert response.status_code == 503, response.text
    assert "PRIVATE" not in response.text and len(http.requests) == 1
    local = [row for row in api.journal.list_calls() if row["purpose"] == "ollama_chat"]
    assert len(local) == 1 and local[0]["raw_response"] == raw


@pytest.mark.parametrize("stage", ["begin_request", "prepare", "record"])
def test_journal_failure_aborts_request_with_sanitized_503_and_no_retry(chat_api, monkeypatch, stage):
    api = chat_api
    sid = session(api)

    def unavailable(*args, **kwargs):
        raise ResponseJournalError("PRIVATE journal filename and raw output")

    monkeypatch.setattr(api.journal, stage, unavailable)
    response = send(api, payload(sid))
    assert response.status_code == 503, response.text
    assert "PRIVATE" not in response.text
    assert len(api.rag.calls) == (1 if stage == "record" else 0)
    replay = send(api, payload(sid))
    assert replay.status_code == 409 and len(api.rag.calls) == (1 if stage == "record" else 0)


def test_poisoned_local_journal_stops_swallowed_failure_before_a_second_http_call(chat_api, monkeypatch):
    api = chat_api
    sid = session(api)
    raw = runtime_reply("Ответ один.")
    http, runtime, rag = install_local_runtime(api, raw)
    original_record = api.journal.record

    def fail_local_record(call_id, **kwargs):
        if api.journal.get(call_id)["purpose"] == "ollama_chat":
            raise ResponseJournalError("PRIVATE local raw durability failure")
        return original_record(call_id, **kwargs)

    class SwallowingService:
        def answer(self, messages, **kwargs):
            try:
                runtime.chat(messages)
            except Exception:
                # A legacy fallback catch must not authorize another provider call.
                try:
                    runtime.chat(messages)
                except Exception:
                    pass
            return {"choices": [{"message": {"role": "assistant", "content": "Недопустимый fallback"}}]}

    api.app.state.rag_service = SwallowingService()
    monkeypatch.setattr(api.journal, "record", fail_local_record)
    response = send(api, payload(sid))
    assert response.status_code == 503, response.text
    assert "PRIVATE" not in response.text and "fallback" not in response.text
    assert len(http.requests) == 1
    assert message_row(api, sid)["status"] == "unknown_outcome"


def test_service_exception_raw_attribute_is_durable_without_database_diagnostics_dependency(chat_api, monkeypatch):
    api = chat_api
    sid = session(api)
    raw = runtime_reply('{"partial":', done_reason="length")

    def truncated(messages, **kwargs):
        api.rag.calls.append((deepcopy(messages), deepcopy(kwargs)))
        raise IncompleteCompletionError("PRIVATE incomplete", raw_response=raw)

    def unavailable(*args, **kwargs):
        raise psycopg2.OperationalError("PRIVATE database cleanup")

    monkeypatch.setattr(api.rag, "answer", truncated)
    monkeypatch.setattr(api.store, "finish_memory_job", unavailable)
    monkeypatch.setattr(api.store, "fail_chat_message", unavailable)
    response = send(api, payload(sid))
    assert response.status_code == 503, response.text
    assert "PRIVATE" not in response.text and len(api.rag.calls) == 1
    row = api.journal.list_calls()[0]
    assert row["outcome"] == "raised" and row["error_type"] == "IncompleteCompletionError"
    assert row["raw_response"] == raw
