"""Synthetic coverage tests for comparison aggregation."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_memory.analysis import METHODS, _money, build_comparison


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AnalysisTests(unittest.TestCase):
    def fixture(self, root, *, status="completed", contract=False, qualified=True, context=False):
        root.mkdir(parents=True, exist_ok=True)
        batch = root / "batch"
        batch.mkdir()
        results, cases, mapping, judgements = [], [], {}, []
        for mindex, method in enumerate(METHODS):
            for offset in range(2):
                index = mindex * 2 + offset
                code, rid = f"J{index:04}", f"h{index // 2}:q{offset}:{method}"
                generation = "ok" if offset == 0 else "action_error"
                results.append({"result_id": rid, "history_id": f"h{index // 2}",
                    "request_id": f"q{offset}", "method": method, "status": generation,
                    "split": "dev", "action": {"reason": "x"} if generation == "ok" else None})
                cases.append({"answer_code": code, "execution_status": generation,
                    "contract_only": contract})
                mapping[code] = {"result_id": rid, "method": method,
                    "history_id": f"h{index // 2}", "request_id": f"q{offset}"}
                if generation == "ok":
                    verdict = ("critical_error", "acceptable", "uncertain", "acceptable")[mindex]
                    judgement = {"verdict": verdict, "findings": [{"explanation": "reason",
                        "evidence_turn_ids": ["t1"]}] if verdict == "critical_error" else []}
                    if context:
                        judgement["utility"] = ("useful", "limited", "uncertain", "useful")[mindex]
                    judgements.append({"answer_code": code, "status": "assessed", "judgement": judgement})
                else:
                    judgements.append({"answer_code": code, "status": "not_assessed", "judgement": None})
        result_path = batch / "results.jsonl"
        cases_path = root / "prepared" / "cases.jsonl"
        write_rows(result_path, results)
        write_rows(cases_path, cases)
        write_rows(root / "answers" / "judgements.jsonl", judgements)
        write_json(root / "prepared" / "mapping.private.json", mapping)
        write_json(root / "prepared" / "manifest.json", {"input_sha256": {"results":
            hashlib.sha256(result_path.read_bytes()).hexdigest()}, "cases_sha256":
            hashlib.sha256(cases_path.read_bytes()).hexdigest()})
        report = {"status": status, "contract_only": contract,
            "calibration_qualified": qualified, "provider_accounting":
            {"charged_or_reserved_usd": "0.2"}}
        if context:
            report["judge_protocol"] = "context-v2"
        write_json(root / "report.json", report)
        write_json(batch / "batch.json", {"split": "dev"})
        workflow = {"batch_dir": str(batch), "split": "dev"}
        if context:
            workflow["judge_config"] = {"protocol": "context-v2"}
        write_json(root / "workflow.json", workflow)
        return mapping

    def test_method_coverage_failures_and_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workflow"
            self.fixture(root)
            result = build_comparison(root, Path(temp) / "out")
            self.assertEqual(result["split"], "dev")
            self.assertEqual(result["distinct_history_count"], 4)
            report_text = (Path(temp) / "out" / "report.md").read_text(encoding="utf-8")
            self.assertIn("split=dev, 4 разных histories", report_text)
            self.assertIn("critical_error/assessed", report_text)
            self.assertNotIn("Оценка одного машинного судьи", report_text)
            for method in METHODS:
                stats = result["methods"][method]
                self.assertEqual((stats["planned"], stats["technical_ok"], stats["generation_error"]), (2, 1, 1))
                self.assertEqual(stats["not_assessed"], 1)
            self.assertEqual(len(result["critical_error_examples"]), 1)
            self.assertFalse(result["scientific_result"])
            self.assertFalse(result["score_axes_combined"])
            self.assertNotIn("utility_counts", result["methods"]["profile_only"])
            self.assertTrue((Path(temp) / "out" / "report.md").exists())

    def test_missing_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "w"
            mapping = self.fixture(root)
            del mapping["J0000"]
            write_json(root / "prepared/mapping.private.json", mapping)
            with self.assertRaises(ValueError):
                build_comparison(root, Path(temp) / "out")

    def test_unqualified_refuses_before_mapping_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "w"
            self.fixture(root, status="calibration_not_qualified", qualified=False)
            original = Path.read_text
            def guarded(path, *args, **kwargs):
                if path.name == "mapping.private.json":
                    raise AssertionError("mapping opened before gate")
                return original(path, *args, **kwargs)
            with patch.object(Path, "read_text", guarded):
                with self.assertRaises(ValueError):
                    build_comparison(root, Path(temp) / "out")

    def test_contract_only_refuses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "w"
            self.fixture(root, contract=True)
            with self.assertRaises(ValueError):
                build_comparison(root, Path(temp) / "out")

    def test_money_uses_decimal_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "w", Path(temp) / "out"
            self.fixture(root)
            receipt = Path(temp) / "receipt.json"
            write_json(receipt, {"cumulative_prior_charged_or_reserved_usd": "0.1", "new_cap_usd": "3.8"})
            original = Path.read_text
            def reject_hidden_transition(path, *args, **kwargs):
                if ".local" in path.parts or "budget_transition" in path.name:
                    raise AssertionError("unexpected read outside specified receipt")
                return original(path, *args, **kwargs)
            with patch.object(Path, "read_text", reject_hidden_transition):
                result = build_comparison(root, out, budget_receipt_path=receipt)
            self.assertEqual(result["budget"]["cumulative_charged_or_reserved_usd"], "0.3")
            self.assertEqual(result["budget"]["current_phase_charged_or_reserved_usd"], "0.2")
            self.assertEqual(result["budget"]["new_phase_cap_usd"], "3.8")
            self.assertEqual(result["budget"]["prior_stages"], "accounted_from_receipt")
            self.assertIn("Накопительно charged or reserved: $0.3", (out / "report.md").read_text(encoding="utf-8"))

    def test_unknown_cost_is_unavailable_not_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "w", Path(temp) / "out"
            self.fixture(root)
            write_json(root / "report.json", {"status": "completed", "contract_only": False,
                "calibration_qualified": True, "provider_accounting": None})
            result = build_comparison(root, out)
            self.assertIsNone(result["budget"]["current_phase_charged_or_reserved_usd"])
            self.assertEqual(result["budget"]["accounting_status"], "unavailable")
            self.assertIn("недоступна", (out / "report.md").read_text(encoding="utf-8"))

    def test_assessment_on_generation_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "w"
            self.fixture(root)
            rows = [json.loads(line) for line in (root / "answers/judgements.jsonl").read_text().splitlines()]
            rows[1] = {"answer_code": "J0001", "status": "assessed",
                       "judgement": {"verdict": "acceptable"}}
            write_rows(root / "answers/judgements.jsonl", rows)
            with self.assertRaises(ValueError):
                build_comparison(root, Path(temp) / "out")

    def test_money_must_be_finite_and_nonnegative(self):
        for value in ("NaN", "Infinity", "-0.01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _money(value, "test amount")

    def test_context_v2_summarizes_utility_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp) / "w", Path(temp) / "out"
            self.fixture(root, context=True)
            result = build_comparison(root, out)
            self.assertEqual(result["judge_protocol"], "context-v2")
            self.assertEqual(result["methods"]["profile_only"]["utility_counts"],
                             {"useful": 1, "limited": 0, "uncertain": 0})
            self.assertEqual(result["methods"]["summary"]["utility_counts"],
                             {"useful": 0, "limited": 0, "uncertain": 1})
            report_text = (out / "report.md").read_text(encoding="utf-8")
            self.assertIn("Оценка одного машинного судьи", report_text)
            self.assertNotIn("Научный результат: нет", report_text)
            self.assertIn("utility_uncertain", report_text)
            self.assertIn("generation_error", report_text)
            self.assertIn("not_assessed", report_text)

    def test_context_v2_rejects_missing_or_invalid_utility(self):
        for invalid in (None, "excellent"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "w"
                self.fixture(root, context=True)
                rows_path = root / "answers/judgements.jsonl"
                rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
                judgement = rows[0]["judgement"]
                if invalid is None:
                    judgement.pop("utility")
                else:
                    judgement["utility"] = invalid
                write_rows(rows_path, rows)
                with self.assertRaises(ValueError):
                    build_comparison(root, Path(temp) / "out")


if __name__ == "__main__":
    unittest.main()
