"""Durable private API response diagnostics, with no PostgreSQL or model calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import uuid

import pytest

from local_app.chat_response_journal import (
    ChatResponseJournal, ResponseJournalError, captured_call, journal_request,
)
from local_app.memory_response_journal import database_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DATABASE_A = "a" * 64
DATABASE_B = "b" * 64


def request(journal, **overrides):
    metadata = dict(actor_id="actor-1", learner_id="learner-1", job_id="job-1",
                    learning_session_id="lesson-1", client_message_id="message-1")
    metadata.update(overrides)
    return journal.begin_request(**metadata)


def prepare(journal, request_id=None, *, purpose="chat", backend="synthetic"):
    return journal.prepare(request_id=request_id or request(journal), purpose=purpose, backend=backend)


def test_missing_journal_reads_do_not_create_any_files(tmp_path):
    path = tmp_path / "absent" / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    assert journal.list_calls() == []
    with pytest.raises(KeyError):
        journal.get(uuid.uuid4().hex)
    assert not path.parent.exists()


def test_request_metadata_and_call_order_survive_reopening(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    request_id = request(journal)
    assert uuid.UUID(request_id).version == 4
    call_id = prepare(journal, request_id)
    assert uuid.UUID(call_id).version == 4
    pending = journal.get(call_id)
    assert {key: pending[key] for key in (
        "request_id", "actor_id", "learner_id", "job_id", "learning_session_id", "client_message_id",
        "database_identity", "call_index", "purpose", "backend",
    )} == dict(request_id=request_id, actor_id="actor-1", learner_id="learner-1", job_id="job-1",
              learning_session_id="lesson-1", client_message_id="message-1", database_identity=DATABASE_A,
              call_index=1, purpose="chat", backend="synthetic")
    for key in ("outcome", "raw_response", "error_type", "elapsed_seconds", "recorded_at"):
        assert pending[key] is None
    raw = {"message": {"content": "Сырой ответ – 42"}, "done_reason": "stop", "usage": {"output": 17}}
    journal.record(call_id, outcome="returned", raw_response=raw, elapsed_seconds=0.125)
    second_id = prepare(journal, request_id, purpose="resolver", backend="other-synthetic")
    reopened = ChatResponseJournal(path, database_identity=DATABASE_A)
    assert reopened.get(call_id)["raw_response"] == raw
    assert reopened.get(call_id)["elapsed_seconds"] == 0.125
    assert reopened.get(second_id)["call_index"] == 2
    assert [row["call_id"] for row in reopened.list_calls(request_id=request_id)] == [call_id, second_id]
    for filters in (
        {"job_id": "job-1"}, {"learning_session_id": "lesson-1"},
        {"client_message_id": "message-1"},
        {"learning_session_id": "lesson-1", "client_message_id": "message-1"},
    ):
        assert {row["call_id"] for row in reopened.list_calls(**filters)} == {call_id, second_id}
    assert reopened.list_calls(request_id="absent") == []
    assert reopened.list_calls(job_id="absent") == []
    assert reopened.list_calls(learning_session_id="absent", client_message_id="message-1") == []


def test_lone_surrogates_and_nested_unicode_responses_roundtrip(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    call_id = prepare(journal)
    raw = {"message": {"content": "Деление – \ud800"}, "parts": ["\udfff", "漢字", "😀"], "nested": {"empty": None}}
    journal.record(call_id, outcome="raised", raw_response=raw,
                   error_type="IncompleteCompletionError", elapsed_seconds=1.25)
    assert ChatResponseJournal(path, database_identity=DATABASE_A).get(call_id)["raw_response"] == raw


def test_completed_outcome_is_idempotent_and_immutable(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    call_id = prepare(journal)
    raw = {"message": {"content": "original"}}
    journal.record(call_id, outcome="returned", raw_response=raw, elapsed_seconds=0.25)
    before = journal.get(call_id)
    journal.record(call_id, outcome="returned", raw_response=deepcopy(raw), elapsed_seconds=0.25)
    with pytest.raises(ResponseJournalError):
        journal.record(call_id, outcome="returned", raw_response={"content": "modified"}, elapsed_seconds=0.25)
    with pytest.raises(ResponseJournalError):
        journal.record(call_id, outcome="raised", raw_response=raw, elapsed_seconds=0.25)
    assert journal.get(call_id) == before


@pytest.mark.parametrize("statement,selector", [
    ("UPDATE requests SET actor_id='changed' WHERE request_id=?", "request"),
    ("DELETE FROM requests WHERE request_id=?", "request"),
    ("UPDATE calls SET purpose='changed' WHERE call_id=?", "call"),
    ("DELETE FROM calls WHERE call_id=?", "call"),
    ("UPDATE outcomes SET payload='{}' WHERE call_id=?", "call"),
    ("DELETE FROM outcomes WHERE call_id=?", "call"),
])
def test_sql_update_delete_cannot_mutate_any_journal_record(tmp_path, statement, selector):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    request_id = request(journal)
    call_id = prepare(journal, request_id)
    journal.record(call_id, outcome="returned", raw_response={"content": "preserve"})
    before = journal.get(call_id)
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(statement, (request_id if selector == "request" else call_id,))
    assert journal.get(call_id) == before


def test_each_write_uses_full_synchronous_commit(tmp_path, monkeypatch):
    real_connect = sqlite3.connect
    traces = []

    def observed_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        statements = []
        connection.set_trace_callback(statements.append)
        traces.append(statements)
        return connection

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    call_id = prepare(journal)
    journal.record(call_id, outcome="returned", raw_response={"content": "durable"})
    writes = [statements for statements in traces if any("INSERT INTO" in sql.upper() for sql in statements)]
    assert len(writes) == 3
    for statements in writes:
        normalized = ["".join(sql.upper().split()) for sql in statements]
        assert "PRAGMASYNCHRONOUS=FULL" in normalized
        assert "BEGINIMMEDIATE" in normalized
        assert "COMMIT" in normalized
        assert normalized.index("PRAGMASYNCHRONOUS=FULL") < normalized.index("BEGINIMMEDIATE")


@pytest.mark.parametrize("schema", [
    "CREATE TABLE unrelated (secret TEXT)", "CREATE VIEW unrelated AS SELECT 42 AS answer",
])
def test_foreign_sqlite_files_are_preserved_byte_for_byte(tmp_path, schema):
    path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(schema)
    before = path.read_bytes()
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    for action in (lambda: request(journal), journal.list_calls):
        with pytest.raises(ResponseJournalError):
            action()
        assert path.read_bytes() == before


def test_worker_journal_is_foreign_to_chat_journal(tmp_path):
    from local_app.memory_response_journal import MemoryResponseJournal

    path = tmp_path / "worker.sqlite3"
    worker_journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    call_id = worker_journal.prepare(job_id="job-1", worker_id="worker", purpose="action", call_index=1)
    worker_journal.record(call_id, outcome="returned", raw_response={"private": "worker response"})
    before = path.read_bytes()
    with pytest.raises(ResponseJournalError):
        request(ChatResponseJournal(path, database_identity=DATABASE_A))
    assert path.read_bytes() == before
    assert worker_journal.get(call_id)["raw_response"] == {"private": "worker response"}


@pytest.mark.parametrize("mutation", [
    "DROP TRIGGER outcomes_no_update", "PRAGMA user_version=999",
    "PRAGMA application_id=123", "CREATE TABLE unexpected (secret TEXT)",
])
def test_modified_schema_is_rejected_without_further_mutation(tmp_path, mutation):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    prepare(journal)
    with sqlite3.connect(path) as connection:
        connection.execute(mutation)
    before = path.read_bytes()
    for action in (journal.list_calls, lambda: request(journal)):
        with pytest.raises(ResponseJournalError):
            action()
        assert path.read_bytes() == before


@pytest.mark.parametrize("metadata", [
    {"actor_id": ""}, {"learner_id": None}, {"job_id": ""},
    {"learning_session_id": "x" * 201}, {"client_message_id": "\ud800"},
])
def test_invalid_request_metadata_is_rejected_before_file_creation(tmp_path, metadata):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    with pytest.raises(ResponseJournalError):
        request(journal, **metadata)
    assert not path.exists()


def test_request_metadata_does_not_accept_prompts_or_authorization(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    for extra in ({"prompt": "private prompt"}, {"authorization": "Bearer private-token"}):
        with pytest.raises(TypeError):
            request(journal, **extra)
    assert not path.exists()


def test_database_binding_stores_fingerprint_without_dsn_or_credentials(tmp_path):
    path = tmp_path / "responses.sqlite3"
    dsn = "host=127.0.0.1 port=9 dbname=journal user=private_journal_user password=private_journal_secret"
    fingerprint = database_fingerprint(dsn)
    journal = ChatResponseJournal(path, database_identity=fingerprint)
    call_id = prepare(journal)
    journal.record(call_id, outcome="returned", raw_response={"content": "synthetic raw return"})
    assert journal.get(call_id)["database_identity"] == fingerprint
    disk = path.read_bytes()
    for private in (dsn, "private_journal_user", "private_journal_secret"):
        assert private.encode() not in disk
        assert private not in json.dumps(journal.get(call_id))


@pytest.mark.parametrize("kwargs", [
    {"outcome": "unsupported"}, {"elapsed_seconds": float("nan")},
    {"elapsed_seconds": -1}, {"elapsed_seconds": True},
    {"error_type": "private error body"}, {"raw_response": {"content": object()}},
])
def test_invalid_outcomes_preserve_pending_call_and_hide_private_values(tmp_path, kwargs):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    call_id = prepare(journal)
    outcome = dict(outcome="returned", raw_response={"content": "private raw response"})
    outcome.update(kwargs)
    with pytest.raises(ResponseJournalError) as error:
        journal.record(call_id, **outcome)
    assert "private" not in str(error.value)
    assert journal.get(call_id)["outcome"] is None


def test_corrupt_payload_read_is_sanitized_and_read_only(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    call_id = prepare(journal)
    private = "invalid-json-private-marker"
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO outcomes VALUES (?,?,?)", (call_id, private, "2026-10-24"))
    before = path.read_bytes()
    for action in (lambda: journal.get(call_id), journal.list_calls):
        with pytest.raises(ResponseJournalError) as error:
            action()
        assert private not in str(error.value)
        assert str(path) not in str(error.value)
    assert path.read_bytes() == before


def test_database_scope_filters_all_selectors_and_rejects_foreign_calls(tmp_path):
    path = tmp_path / "responses.sqlite3"
    first = ChatResponseJournal(path, database_identity=DATABASE_A)
    second = ChatResponseJournal(path, database_identity=DATABASE_B)
    first_request, second_request = request(first), request(second)
    first_id, second_id = prepare(first, first_request), prepare(second, second_request)
    assert [row["call_id"] for row in first.list_calls()] == [first_id]
    assert [row["call_id"] for row in second.list_calls(job_id="job-1")] == [second_id]
    assert second.list_calls(request_id=first_request) == []
    with pytest.raises(KeyError):
        second.get(first_id)
    with pytest.raises(ResponseJournalError):
        second.record(first_id, outcome="returned", raw_response={"foreign": True})
    with pytest.raises(ResponseJournalError):
        prepare(second, first_request)
    assert first.get(first_id)["outcome"] is None


def test_response_survives_abrupt_writer_exit(tmp_path):
    path = tmp_path / "responses.sqlite3"
    code = """
import os, sys
from local_app.chat_response_journal import ChatResponseJournal
journal = ChatResponseJournal(sys.argv[1], database_identity=sys.argv[2])
request_id = journal.begin_request(actor_id='actor-process', learner_id='learner-process', job_id='job-process')
call_id = journal.prepare(request_id=request_id, purpose='chat', backend='synthetic')
journal.record(call_id, outcome='raised', raw_response={'content': '{partial'}, error_type='IncompleteCompletionError', elapsed_seconds=1.5)
print(call_id, flush=True)
os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path), DATABASE_A], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 17, result.stderr
    record = ChatResponseJournal(path, database_identity=DATABASE_A).get(result.stdout.strip())
    assert record["outcome"] == "raised"
    assert record["raw_response"] == {"content": "{partial"}
    assert record["error_type"] == "IncompleteCompletionError"
    assert record["actor_id"] == "actor-process"
    assert record["job_id"] == "job-process"


def test_concurrent_processes_allocate_unique_per_request_call_indexes(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = ChatResponseJournal(path, database_identity=DATABASE_A)
    request_id = request(journal)
    code = """
import json, sys
from local_app.chat_response_journal import ChatResponseJournal
journal = ChatResponseJournal(sys.argv[1], database_identity=sys.argv[2])
print('ready', flush=True)
sys.stdin.readline()
ids = []
for index in range(3):
    call_id = journal.prepare(request_id=sys.argv[3], purpose='chat', backend='synthetic')
    journal.record(call_id, outcome='returned', raw_response={'writer': sys.argv[4], 'index': index})
    ids.append(call_id)
print(json.dumps(ids), flush=True)
"""
    processes = []
    try:
        for writer in range(4):
            processes.append(subprocess.Popen(
                [sys.executable, "-c", code, str(path), DATABASE_A, request_id, str(writer)], cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ))
        for process in processes:
            assert process.stdout.readline().strip() == "ready"
        for process in processes:
            process.stdin.write("start\n")
            process.stdin.flush()
        call_ids = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            call_ids.extend(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    assert len(call_ids) == len(set(call_ids)) == 12
    records = journal.list_calls(request_id=request_id)
    assert {record["call_id"] for record in records} == set(call_ids)
    assert [record["call_index"] for record in records] == list(range(1, 13))
    assert {tuple(sorted(record["raw_response"].items())) for record in records} == {
        tuple(sorted({"writer": str(writer), "index": index}.items()))
        for writer in range(4) for index in range(3)
    }


def test_capture_outside_request_is_noop_and_does_not_create_files(tmp_path):
    path = tmp_path / "responses.sqlite3"
    ChatResponseJournal(path, database_identity=DATABASE_A)
    with captured_call("chat", "synthetic") as capture:
        assert capture is None
    assert not path.exists()


def test_capture_prepares_before_provider_and_records_exact_raw_response(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    raw = {"message": {"content": "private model return"}, "usage": {"input_tokens": 100}}
    with journal_request(journal, actor_id="actor-1", learner_id="learner-1") as scope:
        with captured_call("chat", "synthetic") as capture:
            pending = journal.list_calls(request_id=scope.request_id)
            assert len(pending) == 1 and pending[0]["outcome"] is None
            capture.returned(raw)
            assert journal.get(pending[0]["call_id"])["raw_response"] == raw
    assert journal.list_calls()[0]["outcome"] == "returned"
    assert journal.list_calls()[0]["elapsed_seconds"] >= 0
    assert journal.list_calls()[0]["error_type"] is None


def test_capture_exception_is_recorded_without_private_exception_message(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    private_message = "private prompt and bearer token"

    class TruncatedError(RuntimeError):
        raw_response = {"content": "partial return"}

    with pytest.raises(TruncatedError, match=private_message):
        with journal_request(journal, actor_id="actor-1", learner_id="learner-1"):
            with captured_call("chat", "synthetic"):
                raise TruncatedError(private_message)
    record = journal.list_calls()[0]
    assert record["outcome"] == "raised"
    assert record["error_type"] == "TruncatedError"
    assert record["raw_response"] == {"content": "partial return"}
    assert private_message not in json.dumps(record)
    assert private_message.encode() not in journal.path.read_bytes()


def test_completion_validation_error_keeps_its_dataclass_without_exception_message(tmp_path):
    from local_app.memory_ported.clients import Completion, CompletionValidationError
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    completion = Completion(text="Private returned completion", usage={"output_tokens": 3},
                            elapsed_seconds=0.1, metadata={"backend": "synthetic"})
    with pytest.raises(CompletionValidationError):
        with journal_request(journal, actor_id="actor", learner_id="learner"):
            with captured_call("chat", "synthetic"):
                raise CompletionValidationError("Private exception text", completion, "invalid_completion")
    row, = journal.list_calls()
    assert row["outcome"] == "raised" and row["error_type"] == "CompletionValidationError"
    assert row["raw_response"]["completion"]["text"] == completion.text
    assert "Private exception text" not in json.dumps(row)


def test_explicit_raised_capture_is_not_overwritten_on_context_exit(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    with journal_request(journal, actor_id="actor-1", learner_id="learner-1"):
        with captured_call("external_chat", "synthetic") as capture:
            capture.raised(ValueError("private failure"), raw_response={"content": "provider error body"})
    record = journal.list_calls()[0]
    assert record["outcome"] == "raised"
    assert record["error_type"] == "ValueError"
    assert record["raw_response"] == {"content": "provider error body"}


def test_unmarked_normal_exit_retains_unknown_outcome(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    with journal_request(journal, actor_id="actor-1", learner_id="learner-1"):
        with captured_call("chat", "synthetic"):
            pass
    assert journal.list_calls()[0]["outcome"] is None


def test_nested_context_restores_parent_and_exit_clears_binding(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    with journal_request(journal, actor_id="actor-parent", learner_id="learner-parent") as parent:
        with captured_call("chat", "synthetic") as capture:
            capture.returned("parent first")
        with journal_request(journal, actor_id="actor-child", learner_id="learner-child") as child:
            with captured_call("chat", "synthetic") as capture:
                capture.returned("child")
        with captured_call("chat", "synthetic") as capture:
            capture.returned("parent second")
    assert [r["raw_response"] for r in journal.list_calls(request_id=parent.request_id)] == ["parent first", "parent second"]
    assert [r["raw_response"] for r in journal.list_calls(request_id=child.request_id)] == ["child"]
    with captured_call("chat", "synthetic") as capture:
        assert capture is None


def test_failed_nested_request_poison_cannot_be_swallowed_by_parent(tmp_path, monkeypatch):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    error = ResponseJournalError("durability failure")
    with pytest.raises(ResponseJournalError, match="durability failure"):
        with journal_request(journal, actor_id="parent", learner_id="parent"):
            try:
                with journal_request(journal, actor_id="child", learner_id="child"):
                    with monkeypatch.context() as fault:
                        def fail(*args, **kwargs):
                            raise error
                        fault.setattr(journal, "record", fail)
                        with captured_call("chat", "synthetic") as capture:
                            capture.returned("response")
            except ResponseJournalError:
                pass
            with captured_call("forbidden-followup", "synthetic"):
                pytest.fail("failed child request must prevent further provider calls")


def test_thread_contexts_are_isolated_and_worker_threads_do_not_inherit_parent(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    barrier = threading.Barrier(4)

    def worker(index):
        with captured_call("outside", "synthetic") as outside:
            assert outside is None
        with journal_request(journal, actor_id=f"actor-{index}", learner_id=f"learner-{index}") as scope:
            barrier.wait(timeout=10)
            with captured_call("chat", "synthetic") as capture:
                capture.returned({"writer": index})
        with captured_call("outside", "synthetic") as outside:
            assert outside is None
        return scope.request_id

    with journal_request(journal, actor_id="actor-parent", learner_id="learner-parent") as parent:
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(worker, range(4)))
    assert journal.list_calls(request_id=parent.request_id) == []
    assert len(set(ids)) == 4
    for index, request_id in enumerate(ids):
        records = journal.list_calls(request_id=request_id)
        assert len(records) == 1
        assert records[0]["actor_id"] == f"actor-{index}"
        assert records[0]["raw_response"] == {"writer": index}


@pytest.mark.parametrize("failure_phase", ["prepare", "record"])
def test_swallowed_durability_failure_poison_stops_further_calls_and_request(tmp_path, failure_phase):
    class FailingJournal(ChatResponseJournal):
        def prepare(self, **kwargs):
            if failure_phase == "prepare":
                raise ResponseJournalError("journal unavailable")
            return super().prepare(**kwargs)

        def record(self, *args, **kwargs):
            if failure_phase == "record":
                raise ResponseJournalError("journal unavailable")
            return super().record(*args, **kwargs)

    journal = FailingJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    provider_calls = []
    with pytest.raises(ResponseJournalError):
        with journal_request(journal, actor_id="actor-1", learner_id="learner-1"):
            try:
                with captured_call("chat", "synthetic") as capture:
                    provider_calls.append("first")
                    capture.returned({"content": "first raw response"})
            except ResponseJournalError:
                pass
            with pytest.raises(ResponseJournalError):
                with captured_call("chat", "synthetic"):
                    provider_calls.append("forbidden subsequent call")
    assert provider_calls == ([] if failure_phase == "prepare" else ["first"])
    records = journal.list_calls()
    assert len(records) == (0 if failure_phase == "prepare" else 1)
    assert all(record["outcome"] is None for record in records)
    with captured_call("outside", "synthetic") as capture:
        assert capture is None


def test_context_does_not_relabel_journal_failure_as_provider_failure(tmp_path):
    journal = ChatResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    with pytest.raises(ResponseJournalError):
        with journal_request(journal, actor_id="actor-1", learner_id="learner-1"):
            with captured_call("chat", "synthetic"):
                raise ResponseJournalError("inner journal unavailable")
    record = journal.list_calls()[0]
    assert record["outcome"] is None
    assert record["error_type"] is None


def cli(path, *arguments, dsn=None):
    dsn = dsn or "host=127.0.0.1 port=9 dbname=offline_chat user=offline-user password=private-dsn-secret"
    env = dict(os.environ, DATABASE_URL=dsn, LOCAL_CHAT_RESPONSE_JOURNAL=str(path),
               OLLAMA_URL="http://127.0.0.1:9", PYTHONIOENCODING="utf-8")
    env.pop("PGSERVICE", None)
    return subprocess.run([sys.executable, "scripts/inspect_chat_responses.py", "--journal", str(path), *arguments],
                          cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True, timeout=20)


def offline_journal(path):
    dsn = "host=127.0.0.1 port=9 dbname=offline_chat user=offline-user password=private-dsn-secret"
    return ChatResponseJournal(path, database_identity=database_fingerprint(dsn))


@pytest.mark.parametrize("selector", ["--call-id", "--request-id", "--job-id"])
def test_cli_is_offline_read_only_and_hides_raw_without_explicit_flag(tmp_path, selector):
    path = tmp_path / "responses.sqlite3"
    journal = offline_journal(path)
    request_id = request(journal)
    call_id = prepare(journal, request_id)
    raw = {"message": {"content": "PRIVATE response \ud800"}, "private_debug": "PRIVATE debug"}
    journal.record(call_id, outcome="returned", raw_response=raw)
    value = {"--call-id": call_id, "--request-id": request_id, "--job-id": "job-1"}[selector]
    before = path.read_bytes()
    result = cli(path, selector, value)
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert body["automatic_replay"] is False
    assert len(body["calls"]) == 1 and body["calls"][0]["call_id"] == call_id
    assert "raw_response" not in body["calls"][0]
    assert "PRIVATE" not in result.stdout + result.stderr
    assert "private-dsn-secret" not in result.stdout + result.stderr
    result = cli(path, selector, value, "--include-response")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["calls"][0]["raw_response"] == raw
    assert path.read_bytes() == before


def test_cli_missing_inspection_does_not_create_directories(tmp_path):
    path = tmp_path / "absent" / "responses.sqlite3"
    result = cli(path, "--request-id", "missing")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["calls"] == []
    assert not path.parent.exists()


def test_cli_different_database_cannot_disclose_matching_selector(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = offline_journal(path)
    call_id = prepare(journal)
    journal.record(call_id, outcome="returned", raw_response={"content": "PRIVATE foreign response"})
    other_dsn = "host=127.0.0.1 port=9 dbname=different_chat user=foreign-user password=foreign-secret"
    result = cli(path, "--call-id", call_id, "--include-response", dsn=other_dsn)
    assert result.returncode != 0
    assert "PRIVATE" not in result.stdout + result.stderr
    assert "foreign-secret" not in result.stdout + result.stderr
    result = cli(path, "--job-id", "job-1", "--include-response", dsn=other_dsn)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["calls"] == []


def test_cli_corruption_errors_hide_payload_and_path_even_when_raw_requested(tmp_path):
    path = tmp_path / "private-journal-path" / "responses.sqlite3"
    journal = offline_journal(path)
    call_id = prepare(journal)
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO outcomes VALUES (?,?,?)", (call_id, "PRIVATE corrupt payload", "2026-10-24"))
    before = path.read_bytes()
    for arguments in (("--call-id", call_id), ("--job-id", "job-1", "--include-response")):
        result = cli(path, *arguments)
        assert result.returncode != 0
        output = result.stdout + result.stderr
        assert "PRIVATE" not in output
        assert str(path) not in output
        assert "private-dsn-secret" not in output
        assert "Traceback" not in output
    assert path.read_bytes() == before
