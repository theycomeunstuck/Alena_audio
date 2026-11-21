"""Audit F-077: small defects in the research/evaluation scripts."""
import sys
from pathlib import Path

import pytest

from scripts import build_memory_delivery, evaluate_local, monitor_personal_memory_run, run_product_sessions_v1
from scripts import verify_local

EXPERIMENT = [sys.executable, "-m", "research_memory.personal_memory_lab.experiment"]


def test_monitor_matches_the_run_by_its_output_location_not_a_substring():
    final = EXPERIMENT + ["--output", ".local/research_memory/runs/final_v4/results.json"]
    warm = EXPERIMENT + ["--output=.local/research_memory/runs/final_v4_warm/results.json"]
    other = [sys.executable, "-m", "something.else", "--output", "runs/final_v4/x.json"]
    assert monitor_personal_memory_run.is_target_run(final, "final_v4")
    assert not monitor_personal_memory_run.is_target_run(warm, "final_v4")
    assert monitor_personal_memory_run.is_target_run(warm, "final_v4_warm")
    assert not monitor_personal_memory_run.is_target_run(other, "final_v4")


def test_verify_local_subprocesses_have_a_finite_timeout():
    output, code = verify_local.run_check([sys.executable, "-c", "import time; time.sleep(30)"],
                                          env=None, timeout=1)
    assert code == 124
    assert "timeout" in output.lower()


def test_verify_local_reports_a_normal_exit():
    output, code = verify_local.run_check([sys.executable, "-c", "print('ok')"], env=None, timeout=60)
    assert code == 0 and "ok" in output


def test_delivery_requires_the_pytest_check_before_writing():
    with pytest.raises(RuntimeError, match="pytest"):
        build_memory_delivery.pytest_passed_tests({"checks": [{"name": "policy", "passed": True}]})
    assert build_memory_delivery.pytest_passed_tests({"checks": [{"name": "pytest", "passed_tests": 12}]}) == 12


def test_evaluator_final_report_write_is_atomic(tmp_path, monkeypatch):
    output = tmp_path / "eval.json"
    output.write_text('{"status": "running", "checkpoint": true}\n', encoding="utf-8")
    real_write = Path.write_text

    def failing(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            real_write(self, '{"truncated', encoding="utf-8")
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing)
    with pytest.raises(OSError):
        evaluate_local.write_report_atomically(output, '{"status": "complete"}')
    monkeypatch.setattr(Path, "write_text", real_write)
    assert output.read_text(encoding="utf-8") == '{"status": "running", "checkpoint": true}\n'

    evaluate_local.write_report_atomically(output, '{"status": "complete"}')
    assert output.read_text(encoding="utf-8") == '{"status": "complete"}\n'
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("created,keep,reused,status,expected", [
    (True, False, False, "completed_with_findings", True),   # own temporary database
    (True, False, False, "failed", True),
    (True, True, False, "failed", False),                    # --keep-database
    (True, False, True, "completed_with_findings", True),    # regression reuse that finished
    (True, False, True, "failed", False),                    # failed regression keeps the retained DB
    (False, False, False, "failed", False),
])
def test_regression_failure_keeps_the_retained_database(created, keep, reused, status, expected):
    assert run_product_sessions_v1.should_drop_database(
        created=created, keep_database=keep, reused=reused, status=status) is expected
