"""Concurrent acceptance logs stay complete without sharing append files."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECORDS_PER_PROCESS = 100
UNICODE_TEXT = "\u041f\u0440\u043e\u0432\u0435\u0440\u043a\u0430 \u2013 UTF8\n\u4e2d\u6587 " * 20
WRITER = r"""
import os
import sys
import time
from pathlib import Path
from tests.autonomous_live_process import AuditLog

directory, role, kind, count, text = sys.argv[1:]
directory = Path(directory)
audit = AuditLog(directory, role)
ready = directory / (role + '.ready')
temporary = directory / (role + '.ready.tmp')
temporary.write_text(str(os.getpid()), encoding='utf-8')
temporary.replace(ready)
deadline = time.monotonic() + 15
while not (directory / 'start').exists():
    if time.monotonic() >= deadline:
        raise RuntimeError('synthetic audit start barrier timed out')
    time.sleep(0.01)
for index in range(int(count)):
    audit.write('synthetic_audit_record', forbidden=kind == 'forbidden_calls',
                index=index, text=text)
assert audit.forbidden_calls == (int(count) if kind == 'forbidden_calls' else 0)
"""


@pytest.mark.parametrize("kind", ["process_events", "forbidden_calls"])
def test_concurrent_audit_writers_use_separate_complete_utf8_jsonl(tmp_path, kind):
    processes = {}
    writer_pids = {}
    try:
        for role in ("api", "worker"):
            processes[role] = subprocess.Popen(
                [sys.executable, "-c", WRITER, str(tmp_path), role, kind,
                 str(RECORDS_PER_PROCESS), UNICODE_TEXT],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        deadline = time.monotonic() + 10
        while not all((tmp_path / (role + ".ready")).exists() for role in processes):
            assert time.monotonic() < deadline, "synthetic audit writers did not become ready"
            assert all(process.poll() is None for process in processes.values())
            time.sleep(0.01)
        # A Windows venv launcher can have a different PID from its interpreter.
        writer_pids = {role: int((tmp_path / (role + ".ready")).read_text(encoding="utf-8"))
                       for role in processes}
        assert len(set(writer_pids.values())) == 2
        # Both processes wait at this barrier before either begins appending.
        (tmp_path / "start").write_text("start", encoding="utf-8")
        for process in processes.values():
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=20)

    paths = sorted(tmp_path.glob("*.jsonl"))
    assert {path.name for path in paths} == {
        f"{kind}-{role}-{pid}.jsonl" for role, pid in writer_pids.items()
    }
    records = []
    for role, pid in writer_pids.items():
        raw = (tmp_path / f"{kind}-{role}-{pid}.jsonl").read_bytes()
        assert raw.endswith(b"\n") and b"\r\n" not in raw
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
        assert len(rows) == RECORDS_PER_PROCESS
        assert [row["index"] for row in rows] == list(range(RECORDS_PER_PROCESS))
        assert all(row["role"] == role and row["pid"] == pid
                   and row["event"] == "synthetic_audit_record"
                   and row["text"] == UNICODE_TEXT for row in rows)
        records.extend(rows)
    assert len(records) == 2 * RECORDS_PER_PROCESS
