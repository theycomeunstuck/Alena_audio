from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research_memory.clients import Completion
from research_memory.judge import (CONTEXT_PROTOCOL, CRITERIA, PROMPT, UTILITIES,
                                   judge_cases, prepare_judging, prompt_path,
                                   summarize_calibration, validate_judgement)


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def full_history():
    return {"profile_as_of_cutoff": {"grade": 4, "topic_id": "math.division", "status": "learning"},
            "lessons": [{"lesson_id": "l1", "observed_on": "2026-09-01", "turns": [
                {"turn_id": "t1", "role": "teacher", "text": "Учитель задал вопрос о разряде."},
                {"turn_id": "t2", "role": "student", "text": "После вопроса исправил ответ."}]}]}


def case(code="C01"):
    return {"answer_code": code, "data_origin": "synthetic_unreviewed", "execution_status": "ok",
            "contract_only": False, "payload": {"history": full_history(), "question": "Что дальше?",
                "answer": {"reason": "Нужна новая проверка без помощи."},
                "generator_context": {"profile_as_of_cutoff": full_history()["profile_as_of_cutoff"],
                                      "selected_memory": "ACTUAL_MEMORY_SENTINEL"}}}


def judgement(verdict="acceptable", utility="limited"):
    return {"verdict": verdict, "utility": utility, "scores": {key: 2 for key in CRITERIA},
            "findings": [] if verdict != "critical_error" else [{"criterion": "help_attribution",
                "claim": "Ученик исправил ответ самостоятельно.", "evidence_turn_ids": ["t2"],
                "quote": "После вопроса исправил ответ.", "explanation": "Помощь названа самостоятельностью."}],
            "rationale": "Учитываются факты полной истории и контекст генератора."}


class StubClient:
    def __init__(self, value=None):
        self.value, self.seen = value or judgement(), []

    def complete(self, messages, purpose, response_schema=None):
        self.seen.append((messages, purpose, response_schema))
        return Completion(json.dumps(self.value, ensure_ascii=False),
                          {"input_tokens": 100, "output_tokens": 50}, 0.01,
                          {"backend": "ollama", "model": "stub"})


class ContextPreparationTests(unittest.TestCase):
    def source(self, root, row):
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({"status": "completed", "backend": "ollama",
                                                          "result_count": 1}), encoding="utf-8")
        write_rows(root / "histories.jsonl", [{**full_history(), "history_id": "h1",
                                               "data_origin": "synthetic_unreviewed"}])
        write_rows(run_dir / "results.jsonl", [row])
        return run_dir

    def row(self):
        return {"result_id": "r1", "history_id": "h1", "request_id": "q1", "method": "METHOD_SECRET",
                "model": "MODEL_SECRET", "request_text": "Что дальше?", "status": "ok",
                "action": {"reason": "Предложить новую проверку."},
                "profile_as_of_cutoff": full_history()["profile_as_of_cutoff"],
                "memory_context": "ACTUAL_MEMORY_SENTINEL"}

    def test_export_uses_actual_context_without_method_or_model_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.row()
            row["profile_as_of_cutoff"] = {**row["profile_as_of_cutoff"], "model": "PROFILE_SECRET"}
            run_dir = self.source(root, row)
            report = prepare_judging(run_dir, root / "histories.jsonl", root / "blind", protocol=CONTEXT_PROTOCOL)
            exported = json.loads((root / "blind/cases.jsonl").read_text(encoding="utf-8"))
            payload = exported["payload"]
            self.assertEqual(set(payload), {"history", "question", "answer", "generator_context"})
            self.assertEqual(payload["generator_context"], case()["payload"]["generator_context"])
            self.assertEqual(report["judge_protocol"], CONTEXT_PROTOCOL)
            self.assertNotEqual(report["protocol_version"], "memory-judge-v1")
            for hidden in ("METHOD_SECRET", "MODEL_SECRET", "PROFILE_SECRET", "r1", "J0001"):
                self.assertNotIn(hidden, json.dumps(payload))

    def test_missing_invalid_or_mismatched_actual_context_fails_before_export(self):
        mutations = (lambda row: row.pop("memory_context"), lambda row: row.pop("profile_as_of_cutoff"),
                     lambda row: row.update(memory_context=None), lambda row: row.update(memory_context=5),
                     lambda row: row["profile_as_of_cutoff"].update(grade=8))
        for mutate in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                row = self.row()
                mutate(row)
                run_dir = self.source(root, row)
                with self.assertRaises(ValueError):
                    prepare_judging(run_dir, root / "histories.jsonl", root / "blind", protocol=CONTEXT_PROTOCOL)
                self.assertFalse((root / "blind").exists())

    def test_non_ok_unknown_context_is_null_and_never_judged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.row()
            row.update(status="not_executed", action=None)
            row.pop("memory_context")
            row.pop("profile_as_of_cutoff")
            run_dir = self.source(root, row)
            prepare_judging(run_dir, root / "histories.jsonl", root / "blind", protocol=CONTEXT_PROTOCOL)
            exported = json.loads((root / "blind/cases.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(exported["payload"]["generator_context"],
                             {"profile_as_of_cutoff": None, "selected_memory": None})
            manifest = judge_cases(root / "blind/cases.jsonl", root / "judged", protocol=CONTEXT_PROTOCOL)
            self.assertEqual(manifest["performed_calls"], 0)
            self.assertEqual(manifest["counts"]["not_assessed"], 1)

    def test_recorded_empty_memory_is_preserved_without_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = self.row()
            row.update(method="structured", memory_context="")
            run_dir = self.source(root, row)
            prepare_judging(run_dir, root / "histories.jsonl", root / "blind", protocol=CONTEXT_PROTOCOL)
            exported = json.loads((root / "blind/cases.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(exported["payload"]["generator_context"]["selected_memory"], "")


class ContextExecutionTests(unittest.TestCase):
    def setUp(self):
        self.sources = {turn["turn_id"]: turn for turn in full_history()["lessons"][0]["turns"]}

    def test_acceptable_and_limited_are_valid_separate_results(self):
        result = judgement("acceptable", "limited")
        self.assertEqual(validate_judgement(json.dumps(result), self.sources, protocol=CONTEXT_PROTOCOL), result)
        with self.assertRaises(ValueError):
            validate_judgement(json.dumps(result), self.sources)

    def test_invalid_utility_extra_finding_and_damaged_json_are_rejected(self):
        for mutate in (lambda value: value.update(utility="weak"),
                       lambda value: value.pop("utility"),
                       lambda value: value["findings"][0].update(utility="limited"),
                       lambda value: value.update(findings=[])):
            with self.subTest(mutation=mutate):
                value = judgement("critical_error")
                mutate(value)
                with self.assertRaises(ValueError):
                    validate_judgement(json.dumps(value), self.sources, protocol=CONTEXT_PROTOCOL)
        with self.assertRaises(ValueError):
            validate_judgement('{"verdict":"acceptable",}', self.sources, protocol=CONTEXT_PROTOCOL)

    def test_model_payload_contains_actual_context_and_schema_requires_utility(self):
        client = StubClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "cases.jsonl", [case()])
            manifest = judge_cases(root / "cases.jsonl", root / "out", backend="ollama", model="stub",
                                   client=client, protocol=CONTEXT_PROTOCOL)
            messages, _, schema = client.seen[0]
            payload = json.loads(messages[-1]["content"])
            self.assertEqual(payload, case()["payload"])
            self.assertIn("utility", schema["required"])
            self.assertEqual(manifest["utility_counts"], {"limited": 1})
            self.assertEqual(manifest["judge_protocol"], CONTEXT_PROTOCOL)
            row = json.loads((root / "out/judgements.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["judgement"]["verdict"], "acceptable")
            self.assertEqual(row["judgement"]["utility"], "limited")

    def test_missing_context_rejected_before_any_client_call(self):
        client = StubClient()
        for mutate in (lambda value: value["payload"].pop("generator_context"),
                       lambda value: value["payload"]["generator_context"].update(selected_memory=None)):
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                value = case()
                mutate(value)
                write_rows(root / "cases.jsonl", [value])
                with self.assertRaises(ValueError):
                    judge_cases(root / "cases.jsonl", root / "out", backend="ollama", model="stub",
                                client=client, protocol=CONTEXT_PROTOCOL)
                self.assertFalse(client.seen)
                self.assertFalse((root / "out").exists())

    def test_context_fixture_is_uncertain_and_legacy_fixture_has_no_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(root / "context.jsonl", [case()])
            manifest = judge_cases(root / "context.jsonl", root / "context", protocol=CONTEXT_PROTOCOL)
            result = json.loads((root / "context/judgements.jsonl").read_text())["judgement"]
            self.assertEqual((result["verdict"], result["utility"]), ("uncertain", "uncertain"))
            self.assertTrue(manifest["contract_only"])
            old_case = case()
            old_case["payload"].pop("generator_context")
            write_rows(root / "legacy.jsonl", [old_case])
            old = judge_cases(root / "legacy.jsonl", root / "legacy")
            legacy_result = json.loads((root / "legacy/judgements.jsonl").read_text())["judgement"]
            self.assertNotIn("utility", legacy_result)
            self.assertNotIn("judge_protocol", old)
            self.assertEqual(old["protocol_version"], "memory-judge-v1")


class ContextCalibrationTests(unittest.TestCase):
    def test_actual_context_calibration_files_roundtrip_with_fixture(self):
        data = Path(__file__).parents[1] / "data" / "v3"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = judge_cases(data / "calibration.cases.jsonl", root / "judged", protocol=CONTEXT_PROTOCOL)
            report = summarize_calibration(root / "judged/judgements.jsonl",
                                           data / "calibration.expected.jsonl", root / "summary.json",
                                           protocol=CONTEXT_PROTOCOL)
            self.assertEqual(manifest["case_count"], 18)
            self.assertEqual(report["assessed_count"], 18)
            self.assertEqual(report["uncertain_count"], 18)
            self.assertFalse(report["qualified"])
            self.assertTrue(report["contract_only"])

    def summarize(self, root, *, actual="limited", required="limited"):
        rows = [{"answer_code": "C01", "status": "assessed", "contract_only": False,
                 "protocol": CONTEXT_PROTOCOL, "judgement": judgement("acceptable", actual)},
                {"answer_code": "C02", "status": "assessed", "contract_only": False,
                 "protocol": CONTEXT_PROTOCOL, "judgement": judgement("critical_error", "limited")}]
        expected = [{"answer_code": "C01", "expected_verdict": "acceptable", "required_utility": required},
                    {"answer_code": "C02", "expected_verdict": "critical_error", "required_utility": "limited"}]
        write_rows(root / "judgements.jsonl", rows)
        write_rows(root / "expected.jsonl", expected)
        return summarize_calibration(root / "judgements.jsonl", root / "expected.jsonl", root / "out.json",
                                     protocol=CONTEXT_PROTOCOL)

    def test_required_utility_mismatch_blocks_otherwise_passing_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory), actual="useful", required="limited")
            self.assertFalse(report["qualified"])
            self.assertFalse(report["gate"]["all_required_utilities_matched"])
            self.assertTrue(all(value for key, value in report["gate"].items()
                                if key != "all_required_utilities_matched"))

    def test_matching_utilities_and_verdicts_pass_known_case_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.summarize(Path(directory))
            self.assertTrue(report["qualified"])
            self.assertEqual(report["judge_protocol"], CONTEXT_PROTOCOL)
            self.assertEqual(report["utility_counts"], {"limited": 2})
            with self.assertRaises(ValueError):
                summarize_calibration(Path(directory) / "judgements.jsonl", Path(directory) / "expected.jsonl",
                                      Path(directory) / "legacy.json")

    def test_invalid_required_utility_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "utility"):
                self.summarize(Path(directory), required="weak")


class ProtocolTests(unittest.TestCase):
    def test_unknown_protocol_refused_before_io(self):
        self.assertEqual(prompt_path(), PROMPT)
        self.assertEqual(UTILITIES, ("useful", "limited", "uncertain"))
        for operation in (lambda: prompt_path("context-v99"),
                          lambda: prepare_judging("absent", "absent", "absent", protocol="context-v99"),
                          lambda: judge_cases("absent", "absent", protocol="context-v99"),
                          lambda: summarize_calibration("absent", "absent", "absent", protocol="context-v99")):
            with self.assertRaisesRegex(ValueError, "protocol"):
                operation()

    def test_prompt_examples_are_valid_complete_json(self):
        prompt = prompt_path(CONTEXT_PROTOCOL).read_text(encoding="utf-8")
        examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"verdict"')]
        self.assertEqual(len(examples), 2)
        sources = {"t2": {"text": "После вопроса исправил ответ."}}
        for example in examples:
            validate_judgement(json.dumps(example), sources, protocol=CONTEXT_PROTOCOL)
        self.assertEqual([(value["verdict"], value["utility"]) for value in examples],
                         [("acceptable", "limited"), ("critical_error", "limited")])


if __name__ == "__main__":
    unittest.main()
