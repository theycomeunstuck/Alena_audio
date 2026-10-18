from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

from research_memory.clients import Completion
from research_memory.judge import (CRITERIA, judge_cases, prepare_judging,
                                   summarize_calibration, validate_judgement)


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def history():
    return {"profile_as_of_cutoff": {"grade": 4, "topic_id": "math.division", "status": "learning"},
            "lessons": [{"lesson_id": "l1", "observed_on": "2026-09-01", "turns": [
                {"turn_id": "t1", "role": "teacher", "text": "Учитель задал наводящий вопрос."},
                {"turn_id": "t2", "role": "student", "text": "После вопроса исправил ответ."}]}]}


def case(code="C01", status="ok"):
    return {"answer_code": code, "data_origin": "synthetic_unreviewed", "contract_only": False,
            "execution_status": status, "payload": {"history": history(), "question": "Что дальше?",
            "answer": {"reason": "Проверить без подсказки.", "evidence_turn_ids": ["t2"]}
            if status == "ok" else None}}


def judgement(verdict="acceptable"):
    return {"verdict": verdict, "scores": {key: 2 for key in CRITERIA}, "findings": []
            if verdict != "critical_error" else [{"criterion": "help_attribution",
                "claim": "Самостоятельно исправил.", "evidence_turn_ids": ["t2"],
                "quote": "После вопроса исправил", "explanation": "Помощь ошибочно названа самостоятельностью."}],
            "rationale": "История подтверждает помощь преподавателя."}


class StubClient:
    def __init__(self, value=None, error=None):
        self.value = value if value is not None else judgement()
        self.error, self.seen = error, []

    def complete(self, messages, purpose, response_schema=None):
        self.seen.append((messages, purpose, response_schema))
        if self.error:
            raise self.error
        return Completion(json.dumps(self.value, ensure_ascii=False),
                          {"input_tokens": 101, "output_tokens": 37}, 0.05,
                          {"backend": "ollama", "model": "stub"})


class JudgePreparationTests(unittest.TestCase):
    def test_export_is_blind_full_history_and_keeps_generation_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(json.dumps({
                "status": "completed_with_errors", "backend": "ollama"}), encoding="utf-8")
            h = {**history(), "history_id": "h1", "data_origin": "synthetic_unreviewed"}
            write_rows(root / "histories.jsonl", [h])
            base = {"history_id": "h1", "request_id": "q1", "method": "METHOD_SECRET",
                    "request_text": "Что дальше?", "action": {"reason": "Проверить."},
                    "source_turns": [h["lessons"][0]["turns"][1]], "status": "ok"}
            write_rows(run_dir / "results.jsonl", [{**base, "result_id": "r1"},
                       {**base, "result_id": "r2", "status": "action_error", "action": None}])
            report = prepare_judging(run_dir, root / "histories.jsonl", root / "blind")
            text = (root / "blind/cases.jsonl").read_text(encoding="utf-8")
            exported = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(report["generation_error_count"], 1)
            self.assertNotIn("METHOD_SECRET", text)
            self.assertNotIn("source_turns", text)
            self.assertEqual(len(exported), 2)
            self.assertEqual(exported[0]["payload"]["history"], history())
            self.assertIn("METHOD_SECRET", (root / "blind/mapping.private.json").read_text())

    def test_non_synthetic_or_unfinished_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finish"):
                prepare_judging(root, root / "unused", root / "out")
            (root / "manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            write_rows(root / "histories.jsonl", [{**history(), "history_id": "h1", "data_origin": "real"}])
            with self.assertRaisesRegex(ValueError, "synthetic"):
                prepare_judging(root, root / "histories.jsonl", root / "out")

    def test_partial_batch_keeps_pending_rows_and_checks_frozen_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h = {**history(), "history_id": "h1", "data_origin": "synthetic_unreviewed"}
            write_rows(root / "histories.jsonl", [h])
            (root / "batch.json").write_text(json.dumps({"snapshot_sha256": {
                "histories.jsonl": hashlib.sha256((root / "histories.jsonl").read_bytes()).hexdigest()}}))
            (root / "batch_manifest.json").write_text(json.dumps({"backend": "fixture",
                "expected_results": 2, "job_counts": {"pending": 1, "succeeded": 1}}))
            base = {"history_id": "h1", "request_text": "Что дальше?", "action": None,
                    "status": "not_executed", "method": "METHOD_SECRET"}
            write_rows(root / "results.jsonl", [{**base, "result_id": "r1"},
                       {**base, "result_id": "r2", "status": "ok", "action": {"reason": "Проверить."}}])
            report = prepare_judging(root, root / "histories.jsonl", root / "blind")
            self.assertEqual(report["source_kind"], "batch")
            self.assertTrue(report["source_partial"])
            self.assertTrue(report["contract_only"])
            self.assertEqual(report["case_count"], 2)
            self.assertEqual(report["generation_error_count"], 1)
            (root / "histories.jsonl").write_text((root / "histories.jsonl").read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                prepare_judging(root, root / "histories.jsonl", root / "changed")


class JudgeContractTests(unittest.TestCase):
    def setUp(self):
        self.sources = {turn["turn_id"]: turn for turn in history()["lessons"][0]["turns"]}

    def test_valid_findings_and_unquoted_omission(self):
        value = judgement("critical_error")
        self.assertEqual(validate_judgement(json.dumps(value), self.sources), value)
        value["findings"][0].update(evidence_turn_ids=[], quote="")
        self.assertEqual(validate_judgement(json.dumps(value), self.sources), value)

    def test_invalid_citation_quote_and_schema_are_rejected(self):
        for mutation in (
                lambda v: v["findings"][0].update(evidence_turn_ids=["other"]),
                lambda v: v["findings"][0].update(quote="Новая самостоятельная попытка."),
                lambda v: v["findings"][0].update(evidence_turn_ids=[]),
                lambda v: v.update(extra="hidden"),
                lambda v: v.update(rationale=" "),
                lambda v: v.update(findings=[]),
                lambda v: v["scores"].update(factual_accuracy=True)):
            with self.subTest(mutation=mutation):
                value = judgement("critical_error")
                mutation(value)
                with self.assertRaises(ValueError):
                    validate_judgement(json.dumps(value), self.sources)


class JudgeExecutionTests(unittest.TestCase):
    def test_fixture_is_uncertain_contract_only_and_skips_failed_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "cases.jsonl", [case(), case("C02", "action_error")])
            manifest = judge_cases(root / "cases.jsonl", root / "judged")
            rows = [json.loads(line) for line in (root / "judged/judgements.jsonl").read_text().splitlines()]
            self.assertTrue(manifest["contract_only"])
            self.assertFalse(manifest["real_model_calls"])
            self.assertEqual(manifest["performed_calls"], 1)
            self.assertEqual(rows[0]["judgement"]["verdict"], "uncertain")
            self.assertEqual(rows[1]["status"], "not_assessed")
            self.assertTrue((root / "judged/report.md").exists())

    def test_judge_gets_full_history_only_payload_and_preserves_usage_on_bad_judgement(self):
        client = StubClient(judgement("critical_error"))
        client.value["findings"][0]["quote"] = "Несуществующая цитата."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "cases.jsonl", [case()])
            manifest = judge_cases(root / "cases.jsonl", root / "out", backend="ollama", model="stub", client=client)
            payload = json.loads(client.seen[0][0][-1]["content"])
            self.assertEqual(set(payload), {"history", "question", "answer"})
            self.assertIn("t1", json.dumps(payload))
            self.assertNotIn("C01", json.dumps(payload))
            call = json.loads((root / "out/calls.jsonl").read_text())
            self.assertEqual(call["usage"], {"input_tokens": 101, "output_tokens": 37})
            self.assertEqual(call["error_code"], "invalid_judgement")
            self.assertEqual(manifest["counts"]["judge_error"], 1)

    def test_limit_and_provider_error_do_not_leak_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "cases.jsonl", [case(), case("C02")])
            client = StubClient(error=RuntimeError("SECRET_PROVIDER_BODY"))
            manifest = judge_cases(root / "cases.jsonl", root / "out", backend="ollama",
                                   model="stub", client=client, max_calls=1)
            self.assertEqual(len(client.seen), 1)
            self.assertEqual(manifest["counts"]["judge_error"], 2)
            for path in (root / "out").glob("*"):
                self.assertNotIn("SECRET_PROVIDER_BODY", path.read_text(encoding="utf-8"))

    def test_metadata_or_expected_label_cannot_enter_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = case()
            value["payload"]["expected_verdict"] = "acceptable"
            write_rows(root / "cases.jsonl", [value])
            with self.assertRaisesRegex(ValueError, "payload"):
                judge_cases(root / "cases.jsonl", root / "out")

    def test_backend_authorship_and_explicit_model_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "cases.jsonl", [case()])
            for options in ({"backend": "ollama", "client": StubClient()},
                            {"backend": "spiderssense", "model": "stub"},
                            {"backend": "fixture", "client": StubClient()}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    judge_cases(root / "cases.jsonl", root / "out", **options)


class CalibrationTests(unittest.TestCase):
    def test_actual_calibration_files_roundtrip_with_fixture(self):
        data = Path(__file__).parents[1] / "data"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = judge_cases(data / "calibration.cases.jsonl", root / "judged")
            report = summarize_calibration(root / "judged/judgements.jsonl",
                                           data / "calibration.expected.jsonl", root / "report.json")
            self.assertEqual(manifest["case_count"], 12)
            self.assertEqual(report["expected_count"], 12)
            self.assertEqual(report["assessed_count"], 12)
            self.assertEqual(report["uncertain_count"], 12)
            self.assertTrue(report["contract_only"])
            self.assertFalse(report["qualified"])

    def summarize(self, root, rows):
        write_rows(root / "judgements.jsonl", rows)
        write_rows(root / "expected.jsonl", [
            {"answer_code": "C01", "expected_verdict": "acceptable"},
            {"answer_code": "C02", "expected_verdict": "critical_error"},
            {"answer_code": "C03", "expected_verdict": "uncertain"}])
        return summarize_calibration(root / "judgements.jsonl", root / "expected.jsonl", root / "report.json")

    def row(self, code, verdict, contract=False):
        return {"answer_code": code, "status": "assessed", "judgement": judgement(verdict),
                "contract_only": contract}

    def test_known_case_gate_and_uncertain_is_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory), [self.row("C01", "acceptable"),
                     self.row("C02", "critical_error"), self.row("C03", "uncertain")])
            self.assertTrue(report["qualified"])
            self.assertEqual(report["uncertain_count"], 1)
            self.assertEqual(report["determinate_fraction_of_expected"], 2 / 3)

    def test_critical_miss_fails_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory), [self.row("C01", "acceptable"),
                     self.row("C02", "acceptable"), self.row("C03", "uncertain")])
            self.assertFalse(report["qualified"])
            self.assertFalse(report["gate"]["all_designated_critical_errors_found"])

    def test_missing_and_failed_cases_remain_not_assessed(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory), [self.row("C01", "acceptable"),
                     {"answer_code": "C02", "status": "judge_error", "judgement": None,
                      "contract_only": False}])
            self.assertFalse(report["qualified"])
            self.assertEqual(report["not_assessed_count"], 2)
            self.assertEqual(report["confusion"]["critical_error"]["not_assessed"], 1)

    def test_fixture_cannot_qualify_even_with_matching_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory), [self.row("C01", "acceptable", True),
                     self.row("C02", "critical_error", True), self.row("C03", "uncertain", True)])
            self.assertFalse(report["qualified"])
            self.assertFalse(report["gate"]["not_contract_only"])

    def test_no_judgements_is_zero_coverage_instead_of_success(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory), [])
            self.assertEqual(report["assessed_count"], 0)
            self.assertEqual(report["not_assessed_count"], 3)
            self.assertFalse(report["qualified"])


if __name__ == "__main__":
    unittest.main()
