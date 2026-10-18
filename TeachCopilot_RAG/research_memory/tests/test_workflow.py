"""Finite workflow checks; fake local clients and synthetic cases only."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_memory.batch import create_batch
from research_memory.clients import Completion, FixtureClient
from research_memory.judge import CRITERIA
from research_memory import workflow


class Crash(BaseException):
    pass


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def history():
    return {"profile_as_of_cutoff": {"topic_id": "math.test", "grade": 4},
            "lessons": [{"lesson_id": "l1", "observed_on": "2026-09-01", "turns": [
                {"turn_id": "t1", "role": "student", "text": "Исправил после помощи."}]}]}


def make_calibration(root):
    cases = []
    for code, marker in (("C001", "CALIBRATION_ACCEPT"), ("C002", "CALIBRATION_CRITICAL")):
        cases.append({"answer_code": code, "data_origin": "synthetic_unreviewed", "execution_status": "ok",
                      "contract_only": False, "payload": {"history": history(), "question": "Что дальше?",
                      "answer": {"reason": marker}}})
    paths = root / "calibration.cases.jsonl", root / "calibration.expected.jsonl"
    write_rows(paths[0], cases)
    write_rows(paths[1], [{"answer_code": "C001", "expected_verdict": "acceptable"},
                         {"answer_code": "C002", "expected_verdict": "critical_error", "critical": True}])
    return paths


class JudgeStub:
    def __init__(self, *, all_uncertain=False, crash_answer_number=None):
        self.seen, self.all_uncertain, self.crash_answer_number = [], all_uncertain, crash_answer_number
        self.main_calls = 0

    def complete(self, messages, purpose, response_schema=None):
        payload = json.loads(messages[-1]["content"])
        reason = payload["answer"]["reason"]
        self.seen.append(reason)
        if "CALIBRATION_" not in reason:
            self.main_calls += 1
            if self.main_calls == self.crash_answer_number:
                raise Crash()
        verdict = "uncertain" if self.all_uncertain else "critical_error" if reason == "CALIBRATION_CRITICAL" else "acceptable"
        value = {"verdict": verdict, "scores": {key: None if verdict == "uncertain" else 2 for key in CRITERIA},
                 "findings": [{"criterion": "help_attribution", "claim": "Ошибка в ответе.",
                               "evidence_turn_ids": ["t1"], "quote": "после помощи",
                               "explanation": "В истории есть помощь."}] if verdict == "critical_error" else [],
                 "rationale": "Тестовый судья без модели."}
        return Completion(json.dumps(value, ensure_ascii=False), {"input_tokens": 10, "output_tokens": 5}, 0, {})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calibration = make_calibration(self.root)
        hist = {**history(), "history_id": "h1", "learner_id": "synthetic1", "split": "dev",
                "data_origin": "synthetic_unreviewed"}
        self.histories, self.requests = self.root / "histories.jsonl", self.root / "requests.jsonl"
        write_rows(self.histories, [hist])
        write_rows(self.requests, [{"request_id": f"q{i}", "history_id": "h1", "topic_id": "math.test",
                                   "text": f"Следующее действие {i}?"} for i in range(1, 4)])
        self.batch = self.root / "batch"
        self.output = self.root / "workflow"

    def create(self, **kwargs):
        return create_batch(self.histories, self.requests, self.batch, methods=["profile_only"], **kwargs)

    def run_workflow(self, **kwargs):
        return workflow.run_workflow(self.batch, self.output,
            calibration_cases_path=self.calibration[0], calibration_expected_path=self.calibration[1], **kwargs)

    def test_fixture_end_to_end_is_contract_only_and_resume_does_not_repeat(self):
        self.create()
        first = self.run_workflow()
        self.assertEqual(first["status"], "completed_contract_only")
        self.assertTrue(first["contract_only"])
        self.assertFalse(first["calibration_qualified"])
        self.assertEqual(first["judge_calls_reserved"], 5)
        self.assertEqual(first["real_client_attempts"], 0)
        self.assertEqual(first["answer_verdict_counts"], {"uncertain": 3})
        second = self.run_workflow()
        self.assertEqual(second["judge_calls_reserved"], 5)
        self.assertEqual(second["fixture_call_reservations"], 5)
        self.assertFalse(second["mapping_opened"])

    def test_nonqualified_judge_does_not_assess_main_answers(self):
        self.create(backend="ollama")
        client = JudgeStub(all_uncertain=True)
        result = self.run_workflow(judge_backend="ollama", judge_model="fake", judge_client=client,
                                   generation_client=FixtureClient())
        self.assertEqual(result["status"], "calibration_not_qualified")
        self.assertEqual(len(client.seen), 2)
        self.assertEqual(result["answer_counts"], {"not_assessed": 3})
        self.assertEqual(result["answer_error_counts"], {"calibration_not_qualified": 3})
        self.assertFalse(result["calibration_qualified"])

    def test_partial_resume_reuses_success_and_never_repeats_unknown_case(self):
        self.create(backend="ollama")
        first = JudgeStub(crash_answer_number=2)
        with self.assertRaises(Crash):
            self.run_workflow(judge_backend="ollama", judge_model="fake", judge_client=first,
                              generation_client=FixtureClient())
        self.assertEqual(len(first.seen), 4)
        resumed = JudgeStub()
        result = self.run_workflow(judge_backend="ollama", judge_model="fake", judge_client=resumed,
                                   generation_client=FixtureClient())
        self.assertEqual(len(resumed.seen), 1)
        self.assertEqual(result["judge_calls_reserved"], 5)
        self.assertEqual(result["real_client_attempts"], 5)
        self.assertEqual(result["answer_counts"], {"assessed": 2, "not_assessed": 1})
        self.assertEqual(result["answer_error_counts"], {"judge_error_unknown": 1})
        self.assertEqual(result["status"], "completed_with_unassessed")

    def test_hash_mismatch_stops_before_another_model_call(self):
        self.create()
        self.run_workflow()
        with self.calibration[0].open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash or configuration mismatch"):
            self.run_workflow()

    def test_judge_call_limit_is_global_and_persistent(self):
        self.create(backend="ollama")
        client = JudgeStub()
        first = self.run_workflow(judge_backend="ollama", judge_model="fake", judge_client=client,
                                  generation_client=FixtureClient(), judge_max_calls=3)
        self.assertEqual(len(client.seen), 3)
        self.assertEqual(first["answer_counts"], {"assessed": 1, "not_assessed": 2})
        self.assertEqual(first["answer_error_counts"], {"judge_max_calls_exceeded": 2})
        resumed = JudgeStub()
        second = self.run_workflow(judge_backend="ollama", judge_model="fake", judge_client=resumed,
                                   generation_client=FixtureClient(), judge_max_calls=3)
        self.assertFalse(resumed.seen)
        self.assertEqual(second["judge_calls_reserved"], 3)

    def test_generation_budget_incomplete_stops_before_calibration(self):
        self.create(max_calls=1)
        result = self.run_workflow()
        self.assertEqual(result["status"], "generation_incomplete")
        self.assertEqual(result["judge_calls_reserved"], 0)
        self.assertFalse((self.output / "prepared").exists())
        second = self.run_workflow()
        self.assertEqual(second["status"], "generation_incomplete")

    def test_completed_judgement_tampering_is_detected(self):
        self.create()
        self.run_workflow()
        path = self.output / "steps/answers/00001/output/judgements.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Completed judgement hash mismatch"):
            self.run_workflow()

    def test_mapping_is_never_read_by_workflow(self):
        self.create()
        original = Path.read_text

        def guarded(path, *args, **kwargs):
            self.assertNotIn("mapping.private", path.name)
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded):
            result = self.run_workflow()
        self.assertFalse(result["mapping_opened"])

    def test_provider_configuration_and_ledger_are_frozen_but_usage_can_change(self):
        self.create(backend="ollama")

        class ProviderStub(JudgeStub):
            model = "fake"
            max_output_tokens = 100

            def __init__(self, ledger):
                super().__init__()
                self.ledger_path = ledger
                self.ready = False

            def preflight(self):
                return {"configuration_hash": "immutable-config", "key_present": self.ready,
                        "calls_reserved": len(self.seen), "charged_or_reserved_usd": str(len(self.seen))}

        client = ProviderStub(self.root / "ledger.sqlite3")
        first = self.run_workflow(judge_backend="spiderssense", judge_model="fake", judge_client=client,
                                  generation_client=FixtureClient())
        self.assertEqual(first["status"], "awaiting_credentials")
        self.assertEqual(first["judge_calls_reserved"], 0)
        self.assertFalse(client.seen)
        client.ready = True
        ready = self.run_workflow(judge_backend="spiderssense", judge_model="fake", judge_client=client,
                                 generation_client=FixtureClient())
        self.assertEqual(ready["status"], "completed")
        self.run_workflow(judge_backend="spiderssense", judge_model="fake", judge_client=client,
                          generation_client=FixtureClient())
        self.assertEqual(len(client.seen), 5)
        client.ledger_path = self.root / "different-ledger.sqlite3"
        with self.assertRaisesRegex(ValueError, "hash or configuration mismatch"):
            self.run_workflow(judge_backend="spiderssense", judge_model="fake", judge_client=client)

    def test_derived_preparation_and_summary_recover_without_repeating_calls(self):
        self.create()
        first = self.run_workflow()
        state_path = self.output / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        del state["prepared_cases_sha256"]
        state_path.write_text(json.dumps(state), encoding="utf-8")
        (self.output / "calibration/summary.json").write_text("interrupted derived file", encoding="utf-8")
        resumed = self.run_workflow()
        self.assertEqual(resumed["status"], "completed_contract_only")
        self.assertEqual(resumed["judge_calls_reserved"], first["judge_calls_reserved"])

    def test_empty_unstarted_step_directory_can_resume(self):
        self.create()
        (self.output / "steps/calibration/00001").mkdir(parents=True)
        result = self.run_workflow()
        self.assertEqual(result["status"], "completed_contract_only")


if __name__ == "__main__":
    unittest.main()
