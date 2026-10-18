"""Transport recovery uses immutable local artifacts, never a model or API."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from research_memory.batch import _versions
from research_memory.judge import CONTEXT_PROTOCOL, judge_cases, prompt_path, summarize_calibration
from research_memory.reuse import describe_calibration_reuse, import_calibration_reuse
from research_memory.spiderssense import ProviderRequestError
from research_memory.tests.test_judge_context import StubClient, case, judgement, write_rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class FailedClient:
    def __init__(self, http_status=None):
        self.http_status = http_status

    def complete(self, *args, **kwargs):
        raise ProviderRequestError("Safe test failure", code="provider_http_error" if self.http_status
            else "provider_connection_error", http_status=self.http_status, ledger_call_id=1)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.target = self.root / "source", self.root / "target"
        self.inputs = self.source / "inputs"
        self.inputs.mkdir(parents=True)
        batch = self.root / "batch"
        put(batch / "batch.json", {"frozen": True})
        cases = [case(f"C0{i}") for i in (1, 2, 3)]
        expected = [{"answer_code": "C01", "expected_verdict": "acceptable", "required_utility": "useful"},
                    {"answer_code": "C02", "expected_verdict": "critical_error", "required_utility": "limited"},
                    {"answer_code": "C03", "expected_verdict": "acceptable", "required_utility": "limited"}]
        write_rows(self.inputs / "calibration.cases.jsonl", cases)
        write_rows(self.inputs / "calibration.expected.jsonl", expected)
        self.definition = {"workflow_version": 1, "batch_dir": str(batch.resolve()), "judge_config": {
            "backend": "spiderssense", "model": "stub", "protocol": CONTEXT_PROTOCOL,
            "base_url": "http://127.0.0.1:11435", "max_calls": 73, "timeout_seconds": 180,
            "max_output_tokens": 8192, "provider_configuration_hash": "immutable-pricing",
            "provider_ledger_path": str(self.root / "ledger.sqlite3")},
            "sha256": {"batch_definition": sha(batch / "batch.json"),
                "calibration_cases": sha(self.inputs / "calibration.cases.jsonl"),
                "calibration_expected": sha(self.inputs / "calibration.expected.jsonl"),
                "judge_prompt": sha(prompt_path(CONTEXT_PROTOCOL))}, "versions": _versions()}
        put(self.source / "workflow.json", self.definition)
        rows = []
        for ordinal, item in enumerate(cases, 1):
            step = self.source / "steps/calibration" / f"{ordinal:05d}"
            step.mkdir(parents=True)
            write_rows(step / "input.jsonl", [item])
            client = FailedClient() if ordinal == 1 else StubClient(
                judgement("critical_error" if ordinal == 2 else "acceptable", "limited"))
            judge_cases(step / "input.jsonl", step / "output", backend="spiderssense", model="stub",
                        max_calls=1, max_output_tokens=8192, client=client, protocol=CONTEXT_PROTOCOL)
            put(step / "step.json", {"status": "completed", "input_sha256": sha(step / "input.jsonl"),
                                      "judgements_sha256": sha(step / "output/judgements.jsonl")})
            rows.append(read(step / "output/judgements.jsonl"))
        (self.source / "calibration").mkdir()
        write_rows(self.source / "calibration/judgements.jsonl", rows)
        self.refresh_summary()
        put(self.source / "state.json", {"status": "calibration_not_qualified", "judge_calls_reserved": 3,
            "real_client_attempts": 3, "fixture_call_reservations": 0,
            "provider_accounting": {"calls_reserved": 3, "charged_or_reserved_usd": "0.05"}})
        put(self.source / "report.json", {"status": "calibration_not_qualified", "contract_only": False,
                                          "calibration_qualified": False})

    def refresh_summary(self):
        summary = self.source / "calibration/summary.json"
        if summary.exists():
            summary.unlink()
        summarize_calibration(self.source / "calibration/judgements.jsonl",
            self.inputs / "calibration.expected.jsonl", summary, protocol=CONTEXT_PROTOCOL)

    def target_inputs(self):
        put(self.target / "workflow.json", self.definition)
        (self.target / "inputs").mkdir(exist_ok=True)
        for path in self.inputs.iterdir():
            shutil.copy2(path, self.target / "inputs" / path.name)

    def all_hashes(self):
        return {str(path.relative_to(self.source)): sha(path) for path in self.source.rglob("*") if path.is_file()}

    def test_reuses_all_assessed_only_and_preserves_source_and_counters(self):
        before = self.all_hashes()
        descriptor = describe_calibration_reuse(self.source, self.definition)
        self.target_inputs()
        put(self.target / "state.json", {"judge_calls_reserved": 0, "real_client_attempts": 0})
        counts = import_calibration_reuse(descriptor, self.target)
        self.assertEqual(counts, {"reused_calibration_cases": 2, "transport_retry_cases": 1})
        self.assertFalse((self.target / "steps/calibration/00001").exists())
        self.assertTrue((self.target / "steps/calibration/00002/step.json").exists())
        self.assertTrue((self.target / "steps/calibration/00003/step.json").exists())
        self.assertEqual(import_calibration_reuse(descriptor, self.target), counts)
        self.assertEqual(self.all_hashes(), before)
        self.assertEqual(read(self.target / "state.json"), {"judge_calls_reserved": 0, "real_client_attempts": 0})
        self.assertEqual(descriptor["source_accounting"]["real_client_attempts"], 3)

    def test_unknown_or_format_failure_cannot_be_retried(self):
        path = self.source / "steps/calibration/00001/output/calls.jsonl"
        call = read(path)
        for error, code in (("CompletionValidationError", "invalid_judgement"),
                            ("RuntimeError", "judge_error_unknown"), ("ProviderRequestError", "provider_transport_error")):
            write_rows(path, [{**call, "error": error, "error_code": code}])
            with self.subTest(error=error, code=code), self.assertRaises(ValueError):
                describe_calibration_reuse(self.source, self.definition)

    def test_schema_and_label_errors_are_not_selected_away(self):
        path = self.source / "steps/calibration/00002/output/judgements.jsonl"
        original = read(path)
        for mutate in (lambda row: row["judgement"].update(extra="invalid"),
                       lambda row: row["judgement"].update(verdict="acceptable")):
            row = copy.deepcopy(original)
            mutate(row)
            write_rows(path, [row])
            aggregate = [read(self.source / f"steps/calibration/{i:05d}/output/judgements.jsonl") for i in (1, 2, 3)]
            write_rows(self.source / "calibration/judgements.jsonl", aggregate)
            stamp_path = path.parents[1] / "step.json"
            stamp = read(stamp_path)
            stamp["judgements_sha256"] = sha(path)
            put(stamp_path, stamp)
            self.refresh_summary()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                describe_calibration_reuse(self.source, self.definition)

    def test_different_model_code_inputs_or_profile_are_refused(self):
        for mutate in (lambda target: target["judge_config"].update(model="other"),
                       lambda target: target["versions"]["code"].update({"judge.py": "changed"}),
                       lambda target: target["sha256"].update(calibration_cases="changed"),
                       lambda target: target["versions"]["prompts"].update({"judge_context.txt": "changed"})):
            target = copy.deepcopy(self.definition)
            mutate(target)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                describe_calibration_reuse(self.source, target)
        cases = [json.loads(line) for line in (self.inputs / "calibration.cases.jsonl").read_text().splitlines()]
        cases[0]["payload"]["generator_context"]["profile_as_of_cutoff"]["grade"] = 8
        write_rows(self.inputs / "calibration.cases.jsonl", cases)
        with self.assertRaisesRegex(ValueError, "input"):
            describe_calibration_reuse(self.source, self.definition)

    def test_chaining_and_changed_frozen_source_are_refused(self):
        descriptor = describe_calibration_reuse(self.source, self.definition)
        self.target_inputs()
        source_definition = read(self.source / "workflow.json")
        source_definition["calibration_reuse"] = {}
        put(self.source / "workflow.json", source_definition)
        with self.assertRaisesRegex(ValueError, "Chained"):
            describe_calibration_reuse(self.source, self.definition)
        with self.assertRaisesRegex(ValueError, "source changed"):
            import_calibration_reuse(descriptor, self.target)

    def test_existing_target_steps_and_inputs_cannot_be_overwritten(self):
        descriptor = describe_calibration_reuse(self.source, self.definition)
        self.target_inputs()
        import_calibration_reuse(descriptor, self.target)
        output = self.target / "steps/calibration/00002/output/judgements.jsonl"
        output.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Existing target"):
            import_calibration_reuse(descriptor, self.target)
        self.assertEqual(output.read_text(), "changed")


if __name__ == "__main__":
    unittest.main()
