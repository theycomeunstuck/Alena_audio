"""Private diagnostics can be inspected without a DB or model connection."""
import json
import os
from pathlib import Path
import subprocess
import sys

from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def test_inspection_is_read_only_offline_and_raw_response_requires_explicit_flag(tmp_path):
    path = tmp_path / "responses.sqlite3"
    dsn = "host=127.0.0.1 port=9 dbname=offline user=test password=synthetic-secret"
    journal = MemoryResponseJournal(path, database_identity=database_fingerprint(dsn))
    call_id = journal.prepare(job_id="offline-job", worker_id="worker", purpose="action", call_index=1)
    raw = {"message": {"content": "PRIVATE original response \ud800"}}
    journal.record(call_id, outcome="returned", raw_response=raw)
    before = path.read_bytes()
    env = dict(os.environ, DATABASE_URL=dsn, LOCAL_MEMORY_RESPONSE_JOURNAL=str(path), PYTHONIOENCODING="utf-8")
    command = [sys.executable, "scripts/inspect_memory_responses.py", "--job-id", "offline-job"]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    assert len(body["calls"]) == 1 and body["calls"][0]["call_id"] == call_id
    assert body["automatic_replay"] is False
    assert "PRIVATE" not in result.stdout and "synthetic-secret" not in result.stdout
    assert "raw_response" not in body["calls"][0]
    result = subprocess.run(command + ["--include-response"], cwd=ROOT, env=env,
                            text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["calls"][0]["raw_response"] == raw
    assert path.read_bytes() == before


def test_missing_inspection_does_not_create_journal(tmp_path):
    path = tmp_path / "absent" / "responses.sqlite3"
    env = dict(os.environ, DATABASE_URL="host=127.0.0.1 port=9 dbname=offline",
               LOCAL_MEMORY_RESPONSE_JOURNAL=str(path), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "scripts/inspect_memory_responses.py", "--job-id", "missing"],
                            cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["calls"] == []
    assert not path.parent.exists()
