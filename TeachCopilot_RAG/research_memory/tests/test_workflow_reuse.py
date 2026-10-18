"""A transport-only calibration recovery must not repeat assessed responses."""

import hashlib
import json
import unittest

from research_memory.clients import FixtureClient
from research_memory.spiderssense import ProviderRequestError
from research_memory.tests import test_workflow_context as base


class RecoveryProvider(base.ContextJudge):
    model = "stub"
    max_output_tokens = 100

    def __init__(self, state_path, ledger_path, *, fail_first=False):
        super().__init__(state_path)
        self.ledger_path, self.fail_first = ledger_path, fail_first

    def preflight(self):
        return {"configuration_hash": "test-provider-immutable", "key_present": True,
                "blocked": False, "calls_reserved": len(self.seen), "charged_or_reserved_usd": "0"}

    def complete(self, messages, purpose, response_schema=None):
        payload = json.loads(messages[-1]["content"])
        if self.fail_first and payload["answer"]["reason"] == "CALIBRATION_ACCEPT":
            self.fail_first = False
            self.seen.append(payload)
            raise ProviderRequestError("Safe connection failure", code="provider_connection_error",
                                       ledger_call_id=1)
        return super().complete(messages, purpose, response_schema)


class WorkflowReuseTests(unittest.TestCase):
    create = base.ContextWorkflowTests.create
    context_run = base.ContextWorkflowTests.context_run

    def setUp(self):
        base.ContextWorkflowTests.setUp(self)

    def test_recovery_imports_assessed_cases_and_calls_only_missing_once(self):
        self.create(backend="ollama")
        source = self.output
        ledger = self.root / "test-ledger.sqlite3"
        first_client = RecoveryProvider(source / "state.json", ledger, fail_first=True)
        first = self.context_run(judge_backend="spiderssense", judge_model="stub",
            judge_client=first_client, generation_client=FixtureClient())
        self.assertEqual(first["status"], "calibration_not_qualified")
        source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in source.rglob("*") if p.is_file() and p.name != "workflow.lock"}
        self.output = self.root / "recovered"
        second_client = RecoveryProvider(self.output / "state.json", ledger)
        second = self.context_run(judge_backend="spiderssense", judge_model="stub",
            judge_client=second_client, calibration_reuse_source=source)
        self.assertEqual(second["status"], "completed")
        self.assertTrue(second["calibration_qualified"])
        self.assertEqual(second["reused_calibration_cases"], 1)
        self.assertEqual(second["transport_retry_cases"], 1)
        self.assertEqual(second["real_client_attempts"], 4)
        self.assertEqual(second["judge_calls_reserved"], 4)
        self.assertEqual(len(second_client.seen), 4)
        self.assertNotIn("CALIBRATION_CRITICAL", [p["answer"]["reason"] for p in second_client.seen])
        self.assertEqual(source_hashes, {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*") if p.is_file() and p.name != "workflow.lock"})
        resumed = self.context_run(judge_backend="spiderssense", judge_model="stub",
            judge_client=second_client, calibration_reuse_source=source)
        self.assertEqual(resumed["real_client_attempts"], 4)
        self.assertEqual(len(second_client.seen), 4)

    def test_recovery_is_not_allowed_for_semantic_calibration_failure(self):
        self.create(backend="ollama")
        source = self.output
        ledger = self.root / "test-ledger.sqlite3"
        expected = [json.loads(line) for line in self.context_calibration[1].read_text().splitlines()]
        expected[0]["required_utility"] = "limited"
        base.write_rows(self.context_calibration[1], expected)
        client = RecoveryProvider(source / "state.json", ledger)
        first = self.context_run(judge_backend="spiderssense", judge_model="stub",
            judge_client=client, generation_client=FixtureClient())
        self.assertEqual(first["status"], "calibration_not_qualified")
        self.output = self.root / "rejected"
        next_client = RecoveryProvider(self.output / "state.json", ledger)
        with self.assertRaises(ValueError):
            self.context_run(judge_backend="spiderssense", judge_model="stub",
                judge_client=next_client, calibration_reuse_source=source)
        self.assertFalse(next_client.seen)
