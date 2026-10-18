"""Context-aware judging must remain gated and resumable without new generation."""

import json
import unittest

from research_memory.clients import Completion, FixtureClient
from research_memory.judge import CRITERIA
from research_memory.tests import test_workflow as legacy

history, write_rows = legacy.history, legacy.write_rows


class ContextJudge:
    def __init__(self, state_path):
        self.state_path, self.seen, self.phases = state_path, [], []

    def complete(self, messages, purpose, response_schema=None):
        payload = json.loads(messages[-1]["content"])
        self.seen.append(payload)
        self.phases.append(json.loads(self.state_path.read_text(encoding="utf-8"))["status"])
        critical = payload["answer"]["reason"] == "CALIBRATION_CRITICAL"
        calibration = payload["answer"]["reason"].startswith("CALIBRATION_")
        utility = "useful" if calibration and not critical else "limited"
        value = {"verdict": "critical_error" if critical else "acceptable", "utility": utility,
            "scores": {key: 2 for key in CRITERIA}, "findings": [{
                "criterion": "help_attribution", "claim": "False independence.",
                "evidence_turn_ids": ["t1"], "quote": "после помощи",
                "explanation": "Assistance was observed."}] if critical else [],
            "rationale": "Synthetic stub judgement."}
        return Completion(json.dumps(value, ensure_ascii=False),
                          {"input_tokens": 10, "output_tokens": 5}, 0, {})


class ContextWorkflowTests(unittest.TestCase):
    create = legacy.WorkflowTests.create
    run_workflow = legacy.WorkflowTests.run_workflow

    def setUp(self):
        legacy.WorkflowTests.setUp(self)
        cases = [json.loads(line) for line in self.calibration[0].read_text(encoding="utf-8").splitlines()]
        for case in cases:
            case["payload"]["generator_context"] = {
                "profile_as_of_cutoff": history()["profile_as_of_cutoff"],
                "selected_memory": "t1: Исправил после помощи."}
        self.context_calibration = (self.root / "context.cases.jsonl", self.root / "context.expected.jsonl")
        write_rows(self.context_calibration[0], cases)
        write_rows(self.context_calibration[1], [
            {"answer_code": "C001", "expected_verdict": "acceptable", "required_utility": "useful"},
            {"answer_code": "C002", "expected_verdict": "critical_error", "critical": True,
             "required_utility": "limited"}])

    def context_run(self, **kwargs):
        from research_memory.workflow import run_workflow

        return run_workflow(self.batch, self.output, judge_protocol="context-v2",
            calibration_cases_path=self.context_calibration[0],
            calibration_expected_path=self.context_calibration[1], **kwargs)

    def test_context_roundtrip_utility_is_separate_and_resume_does_not_repeat(self):
        self.create(backend="ollama")
        client = ContextJudge(self.output / "state.json")
        report = self.context_run(judge_backend="ollama", judge_model="stub", judge_client=client,
                                  generation_client=FixtureClient())
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["calibration_qualified"])
        self.assertEqual(report["judge_protocol"], "context-v2")
        self.assertEqual(report["answer_verdict_counts"], {"acceptable": 3})
        self.assertEqual(report["answer_utility_counts"], {"limited": 3})
        self.assertEqual(client.phases, ["calibrating"] * 2 + ["judging"] * 3)
        self.assertTrue(all(p["generator_context"]["selected_memory"] == "" for p in client.seen[2:]))
        second = self.context_run(judge_backend="ollama", judge_model="stub", judge_client=client)
        self.assertEqual(second["judge_calls_reserved"], 5)
        self.assertEqual(len(client.seen), 5)

    def test_context_utility_calibration_failure_blocks_main_calls(self):
        self.create(backend="ollama")
        expected = [json.loads(line) for line in self.context_calibration[1].read_text().splitlines()]
        expected[0]["required_utility"] = "limited"
        write_rows(self.context_calibration[1], expected)
        client = ContextJudge(self.output / "state.json")
        report = self.context_run(judge_backend="ollama", judge_model="stub", judge_client=client,
                                  generation_client=FixtureClient())
        self.assertEqual(report["status"], "calibration_not_qualified")
        self.assertEqual(len(client.seen), 2)
        self.assertEqual(report["answer_counts"], {"not_assessed": 3})

    def test_context_fixture_remains_contract_only(self):
        self.create()
        result = self.context_run()
        self.assertTrue(result["contract_only"])
        self.assertFalse(result["calibration_qualified"])
        self.assertEqual(result["answer_utility_counts"], {"uncertain": 3})

    def test_protocol_mismatch_cannot_reuse_completed_steps(self):
        self.create()
        self.run_workflow()
        with self.assertRaisesRegex(ValueError, "hash or configuration mismatch"):
            self.context_run()
