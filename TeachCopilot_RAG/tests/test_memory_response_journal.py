"""Durable response capture and fail-closed worker behavior without a provider."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid

import pytest

from local_app.memory_response_journal import MemoryResponseJournal, ResponseJournalError, database_fingerprint
from local_app.memory_service import MemoryWorker
from local_app.runtime import IncompleteCompletionError
from tests.test_memory_service import ACTION, Runtime, WorkerStore


DATABASE_A = "a" * 64
DATABASE_B = "b" * 64


def prepare(journal, *, job_id="job-1", call_index=1):
    return journal.prepare(job_id=job_id, worker_id="worker-1", purpose="action", call_index=call_index)


def test_reading_missing_journal_does_not_create_files_or_directories(tmp_path):
    path = tmp_path / "absent" / "responses.sqlite3"
    journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    assert journal.list_calls() == []
    with pytest.raises(KeyError):
        journal.get(str(uuid.uuid4()))
    assert not path.parent.exists()


def test_prepared_call_and_immutable_outcome_survive_reopening(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    call_id = prepare(journal)
    assert uuid.UUID(call_id).version == 4
    pending = journal.get(call_id)
    assert pending["job_id"] == "job-1"
    assert pending["worker_id"] == "worker-1"
    assert pending["purpose"] == "action"
    assert pending["call_index"] == 1
    assert pending["outcome"] is None

    raw = {"message": {"content": "Пробный ответ – 42"}, "eval_count": 17}
    journal.record(call_id, outcome="returned", raw_response=raw, elapsed_seconds=0.25)
    stored = MemoryResponseJournal(path, database_identity=DATABASE_A).get(call_id)
    assert stored["raw_response"] == raw
    assert stored["outcome"] == "returned"
    assert stored["elapsed_seconds"] == 0.25
    journal.record(call_id, outcome="returned", raw_response=deepcopy(raw), elapsed_seconds=0.25)
    with pytest.raises(ResponseJournalError):
        journal.record(call_id, outcome="returned", raw_response={"message": {"content": "changed"}}, elapsed_seconds=0.25)
    assert journal.get(call_id) == stored
    assert len(journal.list_calls(job_id="job-1")) == 1
    assert journal.list_calls(job_id="another-job") == []


def test_raw_response_with_lone_surrogates_and_russian_text_roundtrips(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    call_id = prepare(journal)
    raw = {"message": {"content": "Проверка \ud800: деление – 42"}, "parts": ["\udfff", "обычный текст"]}
    journal.record(call_id, outcome="returned", raw_response=raw)
    assert MemoryResponseJournal(path, database_identity=DATABASE_A).get(call_id)["raw_response"] == raw


def test_corrupt_outcome_read_raises_sanitized_error_without_modifying_file(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    call_id = prepare(journal)
    private_corrupt_payload = "invalid-json-private-response-marker"
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO outcomes VALUES (?,?,?)", (call_id, private_corrupt_payload, "2026-10-24"))
    before = path.read_bytes()
    for read in (lambda: journal.get(call_id), journal.list_calls):
        with pytest.raises(ResponseJournalError) as error:
            read()
        assert private_corrupt_payload not in str(error.value)
        assert str(path) not in str(error.value)
    assert path.read_bytes() == before


def test_response_is_durable_after_writer_process_exits(tmp_path):
    path = tmp_path / "responses.sqlite3"
    code = """
import os
import sys
from local_app.memory_response_journal import MemoryResponseJournal
journal = MemoryResponseJournal(sys.argv[1], database_identity=sys.argv[2])
call_id = journal.prepare(job_id='job-process', worker_id='worker-process', purpose='extract', call_index=2)
journal.record(call_id, outcome='raised', raw_response={'message': {'content': '{"partial":'}}, error_type='IncompleteCompletionError', elapsed_seconds=1.5)
print(call_id, flush=True)
os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), DATABASE_A],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 17, result.stderr
    call_id = result.stdout.strip()
    record = MemoryResponseJournal(path, database_identity=DATABASE_A).get(call_id)
    assert record["outcome"] == "raised"
    assert record["error_type"] == "IncompleteCompletionError"
    assert record["raw_response"] == {"message": {"content": '{"partial":'}}
    assert record["call_index"] == 2


def test_concurrent_process_writers_preserve_all_calls_and_responses(tmp_path):
    path = tmp_path / "responses.sqlite3"
    code = """
import json
import sys
from local_app.memory_response_journal import MemoryResponseJournal
journal = MemoryResponseJournal(sys.argv[1], database_identity=sys.argv[2])
worker = sys.argv[3]
print('ready', flush=True)
sys.stdin.readline()
ids = []
for index in range(1, 4):
    call_id = journal.prepare(job_id='shared-job', worker_id=worker, purpose='action', call_index=index)
    journal.record(call_id, outcome='returned', raw_response={'worker': worker, 'index': index, 'content': 'response-' + worker})
    ids.append(call_id)
print(json.dumps(ids), flush=True)
"""
    processes = []
    try:
        for index in range(4):
            processes.append(subprocess.Popen(
                [sys.executable, "-c", code, str(path), DATABASE_A, f"worker-{index}"],
                cwd=Path(__file__).resolve().parents[1], text=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
    records = MemoryResponseJournal(path, database_identity=DATABASE_A).list_calls(job_id="shared-job")
    assert {record["call_id"] for record in records} == set(call_ids)
    expected = {(f"worker-{worker}", index) for worker in range(4) for index in range(1, 4)}
    assert {(record["worker_id"], record["call_index"]) for record in records} == expected
    for record in records:
        assert record["outcome"] == "returned"
        assert record["raw_response"] == {
            "worker": record["worker_id"], "index": record["call_index"],
            "content": "response-" + record["worker_id"],
        }


@pytest.mark.parametrize("statement", [
    "UPDATE calls SET purpose='tampered' WHERE call_id=?",
    "DELETE FROM calls WHERE call_id=?",
    "UPDATE outcomes SET payload='{}' WHERE call_id=?",
    "DELETE FROM outcomes WHERE call_id=?",
])
def test_sql_update_and_delete_cannot_mutate_journal_records(tmp_path, statement):
    path = tmp_path / "responses.sqlite3"
    journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    call_id = prepare(journal)
    journal.record(call_id, outcome="returned", raw_response={"content": "preserve response"})
    before = journal.get(call_id)
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(statement, (call_id,))
    assert MemoryResponseJournal(path, database_identity=DATABASE_A).get(call_id) == before


def test_database_identity_scopes_reads_and_rejects_foreign_writes(tmp_path):
    path = tmp_path / "responses.sqlite3"
    first = MemoryResponseJournal(path, database_identity=DATABASE_A)
    second = MemoryResponseJournal(path, database_identity=DATABASE_B)
    first_id = prepare(first, job_id="shared-job")
    second_id = prepare(second, job_id="shared-job")
    assert [row["call_id"] for row in first.list_calls()] == [first_id]
    assert [row["call_id"] for row in second.list_calls(job_id="shared-job")] == [second_id]
    with pytest.raises(KeyError):
        second.get(first_id)
    with pytest.raises(ResponseJournalError):
        second.record(first_id, outcome="returned", raw_response={"foreign": True})
    assert first.get(first_id)["outcome"] is None


def test_database_fingerprint_ignores_credentials_and_changes_with_database_name():
    first = database_fingerprint(
        "host=127.0.0.1 port=5432 dbname=journal_database_a user=journal_user_a password=journal_secret_a"
    )
    changed_credentials = database_fingerprint(
        "host=127.0.0.1 port=5432 dbname=journal_database_a user=journal_user_b password=journal_secret_b"
    )
    changed_database = database_fingerprint(
        "host=127.0.0.1 port=5432 dbname=journal_database_b user=journal_user_a password=journal_secret_a"
    )
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")
    assert first == changed_credentials
    assert first != changed_database


def test_database_fingerprint_distinguishes_explicit_host_addresses():
    first = database_fingerprint("host=database.internal hostaddr=192.0.2.1 port=5432 dbname=journal")
    second = database_fingerprint("host=database.internal hostaddr=192.0.2.2 port=5432 dbname=journal")
    assert first != second
    assert database_fingerprint("hostaddr=192.0.2.1 port=5432 dbname=journal") != database_fingerprint(
        "hostaddr=192.0.2.2 port=5432 dbname=journal"
    )


def test_database_fingerprint_uses_pghostaddr_only_without_explicit_hostaddr(monkeypatch):
    monkeypatch.delenv("PGSERVICE", raising=False)
    implicit = "host=database.internal port=5432 dbname=journal"
    explicit = "host=database.internal hostaddr=192.0.2.1 port=5432 dbname=journal"
    monkeypatch.setenv("PGHOSTADDR", "192.0.2.1")
    initial = database_fingerprint(implicit)
    assert initial == database_fingerprint(explicit)
    monkeypatch.setenv("PGHOSTADDR", "192.0.2.2")
    assert database_fingerprint(implicit) != initial
    assert database_fingerprint(explicit) == initial


def test_database_fingerprint_rejects_pgservice_environment_with_explicit_url(monkeypatch):
    monkeypatch.setenv("PGSERVICE", "private-production-service")
    dsn = "postgresql://private_user:private_password@127.0.0.1:5432/journal"
    with pytest.raises(ResponseJournalError) as error:
        database_fingerprint(dsn)
    assert dsn not in str(error.value)
    assert "private-production-service" not in str(error.value)
    assert "private_password" not in str(error.value)


@pytest.mark.parametrize("dsn", [
    "service=production",
    "service=test",
    "service=production host=127.0.0.1 dbname=journal",
    "dbname=journal user=private_journal_user password=private_journal_secret",
    "host=127.0.0.1 user=private_journal_user password=private_journal_secret",
])
def test_database_fingerprint_rejects_ambiguous_connection_identity_without_leaking_dsn(dsn):
    with pytest.raises(ResponseJournalError) as error:
        database_fingerprint(dsn)
    assert dsn not in str(error.value)
    assert "private_journal_user" not in str(error.value)
    assert "private_journal_secret" not in str(error.value)


def test_database_fingerprint_normalizes_default_port_and_respects_pgport(monkeypatch):
    monkeypatch.delenv("PGPORT", raising=False)
    implicit = "host=127.0.0.1 dbname=journal"
    explicit_default = "host=127.0.0.1 port=5432 dbname=journal"
    default_fingerprint = database_fingerprint(explicit_default)
    assert database_fingerprint(implicit) == default_fingerprint
    monkeypatch.setenv("PGPORT", "5433")
    assert database_fingerprint(implicit) == database_fingerprint("host=127.0.0.1 port=5433 dbname=journal")
    assert database_fingerprint(implicit) != default_fingerprint
    assert database_fingerprint(explicit_default) == default_fingerprint


def test_journal_stores_only_database_fingerprint_without_connection_credentials(tmp_path):
    path = tmp_path / "responses.sqlite3"
    dsn = "host=127.0.0.1 port=5432 dbname=journal_database user=unique_journal_user password=unique_journal_secret"
    fingerprint = database_fingerprint(dsn)
    journal = MemoryResponseJournal(path, database_identity=fingerprint)
    call_id = prepare(journal)
    journal.record(call_id, outcome="returned", raw_response={"content": "synthetic response"})
    stored = journal.get(call_id)
    assert stored["database_identity"] == fingerprint
    disk = path.read_bytes()
    for private_value in (dsn, "unique_journal_user", "unique_journal_secret"):
        assert private_value not in json.dumps(stored)
        assert private_value.encode("utf-8") not in disk


def test_unrelated_sqlite_database_is_rejected_without_modification(tmp_path):
    path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (secret TEXT NOT NULL)")
        connection.execute("INSERT INTO unrelated VALUES (?)", ("preserve existing content",))
    before = path.read_bytes()
    with pytest.raises(ResponseJournalError):
        prepare(MemoryResponseJournal(path, database_identity=DATABASE_A))
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT secret FROM unrelated").fetchall() == [("preserve existing content",)]


def test_unrelated_view_only_sqlite_database_is_rejected_without_modification(tmp_path):
    path = tmp_path / "unrelated-view.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE VIEW report AS SELECT 42 AS answer")
    before = path.read_bytes()
    with pytest.raises(ResponseJournalError):
        prepare(MemoryResponseJournal(path, database_identity=DATABASE_A))
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT answer FROM report").fetchall() == [(42,)]


class OrderedStore(WorkerStore):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def heartbeat_memory_job(self, *args, **kwargs):
        return True

    def start_memory_job_call(self, *args, **kwargs):
        self.events.append("marker")

    def finish_memory_job(self, *args, **kwargs):
        self.events.append("finish")
        return super().finish_memory_job(*args, **kwargs)


class ObservedJournal(MemoryResponseJournal):
    def __init__(self, path, events):
        super().__init__(path, database_identity=DATABASE_A)
        self.events = events

    def prepare(self, **kwargs):
        call_id = super().prepare(**kwargs)
        self.events.append("prepare")
        return call_id

    def record(self, *args, **kwargs):
        result = super().record(*args, **kwargs)
        self.events.append("record")
        return result


def test_worker_durably_prepares_before_marker_and_records_before_job_finish(tmp_path):
    events = []
    journal = ObservedJournal(tmp_path / "responses.sqlite3", events)
    store, runtime = OrderedStore(events), Runtime()
    original_chat = runtime.chat

    def chat(*args, **kwargs):
        events.append("provider")
        assert journal.list_calls(job_id="job-1")[0]["outcome"] is None
        return original_chat(*args, **kwargs)

    runtime.chat = chat
    MemoryWorker(store, runtime, response_journal=journal).run_once()
    assert events == ["prepare", "marker", "provider", "record", "finish"]
    record = journal.list_calls(job_id="job-1")[0]
    assert record["outcome"] == "returned"
    assert json.loads(record["raw_response"]["message"]["content"]) == ACTION
    assert len(runtime.calls) == 1


def test_journal_prepare_failure_stops_worker_before_provider_or_db_marker(tmp_path):
    events = []

    class FailingJournal(ObservedJournal):
        def prepare(self, **kwargs):
            raise ResponseJournalError("response journal unavailable")

    store, runtime = OrderedStore(events), Runtime()
    journal = FailingJournal(tmp_path / "responses.sqlite3", events)
    with pytest.raises(ResponseJournalError):
        MemoryWorker(store, runtime, response_journal=journal).run_once()
    assert runtime.calls == []
    assert events == []
    assert store.finished is None


def test_journal_record_failure_propagates_after_one_provider_call(tmp_path):
    events = []

    class FailingJournal(ObservedJournal):
        def record(self, *args, **kwargs):
            raise ResponseJournalError("response journal unavailable")

    store, runtime = OrderedStore(events), Runtime()
    journal = FailingJournal(tmp_path / "responses.sqlite3", events)
    with pytest.raises(ResponseJournalError):
        MemoryWorker(store, runtime, response_journal=journal).run_once()
    assert len(runtime.calls) == 1
    assert events == ["prepare", "marker"]
    assert store.finished is None
    assert journal.list_calls()[0]["outcome"] is None


def test_truncated_provider_response_is_durable_without_retry(tmp_path):
    path = tmp_path / "responses.sqlite3"
    journal = MemoryResponseJournal(path, database_identity=DATABASE_A)
    store, runtime = WorkerStore(), Runtime()
    raw = {"message": {"content": '{"action":'}, "done_reason": "length", "eval_count": 1536}

    def truncated(*args, **kwargs):
        runtime.calls.append((args, kwargs))
        raise IncompleteCompletionError("truncated", raw_response=raw)

    runtime.chat = truncated
    MemoryWorker(store, runtime, response_journal=journal).run_once()
    record = MemoryResponseJournal(path, database_identity=DATABASE_A).list_calls()[0]
    assert record["raw_response"] == raw
    assert record["outcome"] == "raised"
    assert record["error_type"] == "IncompleteCompletionError"
    assert len(runtime.calls) == 1
    assert store.finished["error"]["type"] == "IncompleteCompletionError"


def test_invalid_model_output_is_journaled_before_contract_validation(tmp_path):
    journal = MemoryResponseJournal(tmp_path / "responses.sqlite3", database_identity=DATABASE_A)
    store, runtime = WorkerStore(), Runtime({**ACTION, "evidence_turn_ids": ["assistant-1"]})
    MemoryWorker(store, runtime, response_journal=journal).run_once()
    record = journal.list_calls()[0]
    assert record["outcome"] == "returned"
    assert json.loads(record["raw_response"]["message"]["content"])["evidence_turn_ids"] == ["assistant-1"]
    assert store.finished["error"]["type"] == "ContractValidationError"
    assert "memory_state" not in store.finished
    assert len(runtime.calls) == 1
