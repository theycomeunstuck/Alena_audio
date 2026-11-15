"""Regression tests for audit findings F-034 and F-036 (assessment v5 workflow).

F-034 – the irreversible cross-workspace phase claim must be written only after the
durable workspace setup (phase directory, protocol snapshots, schema, settings and
manifest) has succeeded, so a transient OSError during setup cannot consume the paid
phase with zero model calls. The second-run-refuses semantics for actually started
phases must be preserved.

F-036 – expected-label files must be schema-validated in prepare/check_inputs, before
any paid call, with a clear error naming the offending rows.

The frozen v4 dataset (TeachCopilot_RAG/.local/research_memory/final_v4_20261018) is
not committed to the repository, so these tests synthesize a minimal frozen fixture in
a temporary directory and stub only workflow.frozen_checks; prepare, run_phase, the
claim and setup ordering and the label validation all run for real.
"""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from research_assessment_v5 import workflow
from research_assessment_v5.protocol import lines, read, rows, sha, write
from research_memory.clients import Completion

AUTHORIZED = {"model": "openai/gpt-6.1-sol", "reasoning_effort": "medium",
              "max_output_tokens": 8192, "max_cost_usd": "16", "max_calls": 126}
CALIBRATION_LABELS = "calibration.expected.private.jsonl"
DEV_LABELS = "dev.expected.private.jsonl"


class CountingJudge:
    """Fake provider client: every completion is invalid; only the call count matters."""
    def __init__(self):
        self.calls = 0

    def preflight(self):
        return {"blocked": False, "calls_reserved": self.calls, "charged_or_reserved_usd": "0"}

    def complete(self, *args):
        self.calls += 1
        return Completion("{invalid", {}, 0, {"backend": "test_only"})


def _frozen_fixture(root):
    """Minimal stand-in for the frozen v4 held-out dataset consumed by prepare()."""
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    profile = {"grade": 4, "topic_id": "fixture.topic", "status": "learning"}
    histories = []
    for number in range(16):
        histories.append({"history_id": f"H{number:02d}", "data_origin": "synthetic_fixture",
                          "profile_as_of_cutoff": profile,
                          "lessons": [{"lesson_id": f"H{number:02d}L1", "observed_on": "2026-10-01",
                                       "turns": [{"turn_id": f"H{number:02d}s", "role": "student",
                                                  "text": "Я решил пример с подсказкой."},
                                                 {"turn_id": f"H{number:02d}t", "role": "teacher",
                                                  "text": "Самостоятельность пока не подтверждена."}]}]})
    results = []
    for number in range(128):
        history = histories[number % 16]
        ok = number < 94
        results.append({"result_id": f"R{number:03d}", "method": f"m{number % 4}",
                        "history_id": history["history_id"], "request_id": f"Q{number:03d}",
                        "status": "ok" if ok else "failed",
                        "error": None if ok else "fixture_technical_failure",
                        "request_text": "Что делать дальше на уроке?",
                        "action": {"action": "independent_check",
                                   "reason": "Нужно подтвердить самостоятельность."},
                        "profile_as_of_cutoff": profile,
                        "memory_context": f"H{number % 16:02d}s: Я решил пример с подсказкой."})
    lines(frozen / "heldout/snapshots/histories.jsonl", histories)
    lines(frozen / "heldout/results.jsonl", results)
    write(frozen / "heldout/batch.json", {"snapshot_sha256":
          {"histories.jsonl": sha(frozen / "heldout/snapshots/histories.jsonl")}})
    return frozen


@pytest.fixture()
def env(tmp_path, monkeypatch):
    frozen = _frozen_fixture(tmp_path)
    monkeypatch.setattr(workflow, "frozen_checks", lambda *args, **kwargs: {})
    monkeypatch.setattr(workflow, "FROZEN", frozen)
    config, auth = tmp_path / "config.json", tmp_path / "receipt.json"
    write(config, AUTHORIZED)
    write(auth, {})
    return SimpleNamespace(tmp_path=tmp_path, frozen=frozen, config=config, auth=auth,
                           claims=tmp_path / "assessment_v5_execution_claims")


def _label_rows(count, prefix="K"):
    """Expected-label rows shaped like the shipped files, extra fields included."""
    return [{"answer_code": f"{prefix}{number:03d}", "situation_type": f"s{number % 8}",
             "expected_verdict": ("acceptable", "critical_error", "uncertain")[number % 3],
             "critical": number % 3 == 1,
             "required_utility": ("useful", "limited", "uncertain")[number % 3],
             "evidence_turn_ids": ["t1"], "explanation": "fixture label"}
            for number in range(1, count + 1)]


def _write_labels(path, table):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in table), encoding="utf-8")


def _rewrite_labels(path, table):
    """Replace a labels file and keep the preparation stamp consistent with it."""
    _write_labels(path, table)
    stamp = read(path.parent.parent / "preparation.json")
    stamp["input_sha256"][path.name] = sha(path)
    write(path.parent.parent / "preparation.json", stamp)


# F-034 – claim must not be written while the fallible durable setup can still fail.

def test_setup_mkdir_failure_writes_no_claim_and_phase_is_retryable(env):
    target = env.tmp_path / "workspace"
    workflow.prepare(target, frozen=env.frozen)
    client = CountingJudge()
    real_mkdir = Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if not kwargs.get("exist_ok", False):
            raise OSError("simulated transient setup failure")
        return real_mkdir(self, *args, **kwargs)

    with patch("pathlib.Path.mkdir", failing_mkdir):
        with pytest.raises(OSError):
            workflow.run_phase(target, "dev", env.config, env.auth, execute=True, client=client)
    assert client.calls == 0
    assert not (env.claims / "dev.json").exists()
    assert not (target / "dev").exists()
    manifest = workflow.run_phase(target, "dev", env.config, env.auth, execute=True, client=client)
    assert manifest["attempted_calls"] == 8
    assert client.calls == 8
    claim = read(env.claims / "dev.json")
    assert Path(claim["workspace"]).resolve() == target.resolve()


def test_snapshot_copy_failure_writes_no_claim_and_fresh_workspace_can_run(env):
    target = env.tmp_path / "workspace"
    workflow.prepare(target, frozen=env.frozen)
    client = CountingJudge()

    def failing_copyfile(src, dst, *args, **kwargs):
        raise OSError("simulated transient failure copying the protocol snapshot")

    with patch("shutil.copyfile", failing_copyfile):
        with pytest.raises(OSError):
            workflow.run_phase(target, "dev", env.config, env.auth, execute=True, client=client)
    assert client.calls == 0
    assert not (env.claims / "dev.json").exists()
    assert (target / "dev").exists()
    second = env.tmp_path / "second_workspace"
    workflow.prepare(second, frozen=env.frozen)
    manifest = workflow.run_phase(second, "dev", env.config, env.auth, execute=True, client=client)
    assert manifest["attempted_calls"] == 8
    assert client.calls == 8
    claim = read(env.claims / "dev.json")
    assert Path(claim["workspace"]).resolve() == second.resolve()


# F-036 – expected-label schema must be checked before any paid call.

def test_prepare_rejects_malformed_calibration_labels(env, monkeypatch):
    real_copyfile = shutil.copyfile
    poisoned = _label_rows(24)
    del poisoned[7]["expected_verdict"]

    def poisoned_copyfile(src, dst, *args, **kwargs):
        if Path(src).name == CALIBRATION_LABELS:
            _write_labels(Path(dst), poisoned)
            return
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", poisoned_copyfile)
    client = CountingJudge()
    with pytest.raises(ValueError) as excinfo:
        workflow.prepare(env.tmp_path / "workspace", frozen=env.frozen)
    message = str(excinfo.value)
    assert "row 8" in message and "expected_verdict" in message
    assert client.calls == 0


def test_prepare_rejects_malformed_dev_labels(env, monkeypatch):
    real_copyfile = shutil.copyfile
    poisoned = _label_rows(8, prefix="D")
    del poisoned[2]["answer_code"]

    def poisoned_copyfile(src, dst, *args, **kwargs):
        if Path(src).name == DEV_LABELS:
            _write_labels(Path(dst), poisoned)
            return
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", poisoned_copyfile)
    client = CountingJudge()
    with pytest.raises(ValueError) as excinfo:
        workflow.prepare(env.tmp_path / "workspace", frozen=env.frozen)
    message = str(excinfo.value)
    assert "row 3" in message and "answer_code" in message
    assert client.calls == 0


def test_run_phase_rejects_malformed_dev_labels_before_any_paid_call(env):
    target = env.tmp_path / "workspace"
    workflow.prepare(target, frozen=env.frozen)
    labels = target / "inputs" / DEV_LABELS
    table = rows(labels)
    del table[2]["answer_code"]
    _rewrite_labels(labels, table)
    client = CountingJudge()
    with pytest.raises(ValueError) as excinfo:
        workflow.run_phase(target, "dev", env.config, env.auth, execute=True, client=client)
    message = str(excinfo.value)
    assert "row 3" in message and "answer_code" in message
    assert client.calls == 0
    assert not (env.claims / "dev.json").exists()
    assert not (target / "dev").exists()


def test_run_phase_rejects_malformed_calibration_labels_before_any_paid_call(env):
    target = env.tmp_path / "workspace"
    workflow.prepare(target, frozen=env.frozen)
    client = CountingJudge()
    dev = workflow.run_phase(target, "dev", env.config, env.auth, execute=True, client=client)
    assert dev["attempted_calls"] == 8
    workflow.freeze(target, env.config, env.auth)
    labels = target / "inputs" / CALIBRATION_LABELS
    table = rows(labels)
    del table[7]["expected_verdict"]
    _rewrite_labels(labels, table)
    with pytest.raises(ValueError) as excinfo:
        workflow.run_phase(target, "calibration", env.config, env.auth, execute=True, client=client)
    message = str(excinfo.value)
    assert "row 8" in message and "expected_verdict" in message
    assert client.calls == 8
    assert not (env.claims / "calibration.json").exists()
    assert not (target / "calibration").exists()
