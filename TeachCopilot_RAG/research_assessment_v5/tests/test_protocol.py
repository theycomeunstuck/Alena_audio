from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from research_assessment_v5.protocol import CRITERIA, calibration_gate, sources, validate, write
from research_assessment_v5.workflow import (_RecordedResponse, _RecordingOpener, _settings, _tree_hashes,
                                            check_freeze, freeze, prepare, run_phase)
from research_memory.clients import Completion
from research_memory.spiderssense import BudgetError, ProviderRequestError


def case(code="X01"):
    profile = {"grade": 4, "topic_id": "synthetic.test", "status": "learning"}
    return {"answer_code": code, "data_origin": "synthetic_contract_only", "execution_status": "ok", "contract_only": False,
            "payload": {"history": {"profile_as_of_cutoff": profile, "lessons": [{"lesson_id": "l1", "observed_on": "2026-10-22",
                        "turns": [{"turn_id": "s1", "role": "student", "text": "Я  решил с подсказкой."},
                                  {"turn_id": "t1", "role": "teacher", "text": "Самостоятельная проверка ещё не проведена."}]}]},
                        "question": "Что дальше?", "answer": {"action": "independent_check", "reason": "Нужно уточнить самостоятельность.",
                        "evidence_turn_ids": ["s1"], "unknowns": ["самостоятельность"], "review_required": True},
                        "generator_context": {"profile_as_of_cutoff": profile, "selected_memory": "s1: Я  решил с подсказкой."}}}


def judgement(verdict="acceptable", utility="useful"):
    evidence = [{"source_turn_id": "s1", "quote": "Я  решил с подсказкой."},
                {"source_turn_id": "t1", "quote": "Самостоятельная проверка ещё не проведена."}]
    return {"verdict": verdict, "utility": utility, "substantial_distortion": verdict == "critical_error",
            "criteria": {key: {"score": 0 if verdict == "critical_error" else 2,
                         "explanation": "Синтетическая проверка контракта.", "evidence": deepcopy(evidence)} for key in CRITERIA},
            "findings": [{"criterion": "factual_accuracy", "claim": "reason", "explanation": "Проверка контракта.",
                          "evidence": deepcopy(evidence)}] if verdict == "critical_error" else [], "rationale": "Проверка контракта."}


class StrictGroundsTests(unittest.TestCase):
    def test_all_five_criteria_have_literal_grounds(self):
        result = judgement()
        self.assertEqual(validate(json.dumps(result), case()), result)
        self.assertEqual(len(result["criteria"]), 5)

    def test_paraphrase_or_space_normalization_is_rejected(self):
        result = judgement()
        result["criteria"]["factual_accuracy"]["evidence"][0]["quote"] = "Я решил с подсказкой."
        with self.assertRaises(ValueError):
            validate(json.dumps(result), case())

    def test_every_cited_id_is_checked_not_just_one_matching_quote(self):
        result = judgement()
        result["criteria"]["source_support"]["evidence"].append({"source_turn_id": "not-real", "quote": "Я  решил"})
        with self.assertRaises(ValueError):
            validate(json.dumps(result), case())

    def test_empty_missing_duplicate_and_inconsistent_output_rejected(self):
        mutations = [lambda x: x["criteria"].pop("unknowns_handling"),
                     lambda x: x["criteria"]["source_support"].update(evidence=[]),
                     lambda x: x.update(verdict="critical_error"),
                     lambda x: x["criteria"]["source_support"].update(score=True)]
        for mutate in mutations:
            result = judgement()
            mutate(result)
            with self.subTest(result=result), self.assertRaises(ValueError):
                validate(json.dumps(result), case())
        with self.assertRaises(ValueError):
            validate('{"verdict":"acceptable","verdict":"uncertain"}', case())
        with self.assertRaises(ValueError):
            validate("```json\n" + json.dumps(judgement()) + "\n```", case())

    def test_no_method_or_expected_labels_in_payload(self):
        payload = case()
        self.assertNotIn("expected", json.dumps(payload))
        payload["method"] = "tail"
        with self.assertRaises(ValueError):
            sources(payload)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.cases, self.expected, self.results = [], [], []
        for typ in range(8):
            for index, verdict in enumerate(("acceptable", "critical_error", "uncertain")):
                code = f"X{typ}{index}"
                utility = {"acceptable": "useful", "critical_error": "limited", "uncertain": "uncertain"}[verdict]
                self.cases.append(case(code))
                self.expected.append({"answer_code": code, "situation_type": str(typ), "expected_verdict": verdict,
                                      "critical": verdict == "critical_error", "required_utility": utility,
                                      "evidence_turn_ids": ["s1", "t1"]})
                self.results.append({"answer_code": code, "status": "assessed", "contract_only": False,
                                     "judgement": judgement(verdict, utility)})

    def test_qualification_requires_all_prespecified_conditions(self):
        self.assertTrue(calibration_gate(self.cases, self.expected, self.results)["qualified"])
        for mutate in (lambda rs: rs.pop(),
                       lambda rs: rs[0]["judgement"].update(verdict="uncertain"),
                       lambda rs: rs[1]["judgement"].update(verdict="uncertain"),
                       lambda rs: rs[2]["judgement"].update(utility="useful"),
                       lambda rs: rs[0].update(contract_only=True),
                       lambda rs: rs[0]["judgement"]["criteria"]["source_support"].update(evidence=[])):
            result = deepcopy(self.results)
            mutate(result)
            self.assertFalse(calibration_gate(self.cases, self.expected, result)["qualified"])

    def test_uncertain_verdict_is_diagnostic_but_utility_is_required(self):
        self.results[2]["judgement"]["verdict"] = "acceptable"
        self.assertTrue(calibration_gate(self.cases, self.expected, self.results)["qualified"])

    def test_correct_id_ground_must_be_covered(self):
        for item in self.results[0]["judgement"]["criteria"].values():
            item["evidence"] = item["evidence"][:1]
        self.assertFalse(calibration_gate(self.cases, self.expected, self.results)["qualified"])


class FiniteExecutionTests(unittest.TestCase):
    def test_never_attempted_completion_retains_unknown_and_uses_eight_calls_total(self):
        from research_assessment_v5.protocol import ROOT, read, rows, sha
        import shutil
        class SingleAttemptJudge:
            def __init__(self): self.calls, self.payloads = 0, []
            def preflight(self):
                return {"blocked": False, "calls_reserved": self.calls, "charged_or_reserved_usd": str(self.calls / 10)}
            def complete(self, messages, *args):
                self.calls += 1
                self.payloads.append(json.loads(messages[1]["content"]))
                if self.calls in {6, 9}:
                    raise ProviderRequestError("unknown", code="provider_connection_error", ledger_call_id=self.calls)
                return Completion("{invalid", {}, 0, {"backend": "test_only"})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            prepare(target)
            config, auth = Path(directory) / "config.json", Path(directory) / "receipt.json"
            write(config, {"model": "openai/gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192,
                           "max_cost_usd": "16", "max_calls": 126})
            write(auth, {})
            client = SingleAttemptJudge()
            dev = run_phase(target, "dev", config, auth, execute=True, client=client)
            self.assertEqual(dev["attempted_calls"], 6)
            original_hashes = _tree_hashes(target / "dev")
            unknown = rows(target / "dev/judgements.jsonl")[5]
            self.assertEqual(unknown["new_reserved_calls"], 1)
            second = Path(directory) / "other"
            shutil.copytree(target, second)
            with self.assertRaises(ValueError):
                run_phase(second, "dev_completion", config, auth, execute=True, client=client)
            plan = run_phase(target, "dev_completion", config, auth)
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["case_count"], 2)
            self.assertEqual(client.calls, 6)
            completed = run_phase(target, "dev_completion", config, auth, execute=True, client=client)
            self.assertEqual(completed["attempted_calls"], 2)
            self.assertEqual(client.calls, 8)
            self.assertEqual(_tree_hashes(target / "dev"), original_hashes)
            merged = rows(target / "dev_merged/judgements.jsonl")
            self.assertEqual(len(merged), 8)
            self.assertEqual(merged[5], unknown)
            self.assertEqual(len({json.dumps(value, sort_keys=True) for value in client.payloads}), 8)
            self.assertEqual(read(target / "report.json")["cost_accounting"]["calls_reserved"], 8)
            with self.assertRaises((FileExistsError, ValueError)):
                run_phase(target, "dev_completion", config, auth, execute=True, client=client)
            frozen = freeze(target, config, auth, correction_note="One execution/diagnostic correction from dev interruption.")
            self.assertEqual(frozen["correction_count"], 1)
            self.assertEqual(frozen["dev_receipt"]["attempted_calls"], 8)
            check_freeze(target, config, auth)
            original_sha = sha
            with patch("research_assessment_v5.workflow.sha", side_effect=lambda path:
                       "changed" if Path(path) == ROOT / "__init__.py" else original_sha(path)):
                with self.assertRaises(ValueError):
                    check_freeze(target, config, auth)
            calibration = run_phase(target, "calibration", config, auth, execute=True, client=client)
            self.assertEqual(calibration["attempted_calls"], 24)
            self.assertEqual(client.calls, 32)
            # First calibration transport unknown is retained; the next independent cases still run once.
            self.assertEqual(rows(target / "calibration/judgements.jsonl")[0]["error"]["code"], "provider_connection_error")
            self.assertEqual(len(rows(target / "calibration/judgements.jsonl")), 24)
            self.assertEqual(_tree_hashes(target / "dev"), original_hashes)

    def test_completion_rejects_any_attempt_artifact_or_new_reservation_on_unsent_case(self):
        from research_assessment_v5.protocol import rows, lines
        class UnknownSixth:
            def __init__(self): self.calls = 0
            def preflight(self): return {"blocked": False, "calls_reserved": self.calls, "charged_or_reserved_usd": "0"}
            def complete(self, *args):
                self.calls += 1
                if self.calls == 6:
                    raise ProviderRequestError("unknown", code="provider_connection_error", ledger_call_id=6)
                return Completion("{invalid", {}, 0, {})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            prepare(target)
            config, auth = Path(directory) / "config.json", Path(directory) / "receipt.json"
            write(config, {"model": "openai/gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192,
                           "max_cost_usd": "16", "max_calls": 126})
            write(auth, {})
            client = UnknownSixth()
            run_phase(target, "dev", config, auth, execute=True, client=client)
            artifact = target / "dev/calls/007/attempt.json"
            write(artifact, {})
            with self.assertRaises(ValueError):
                run_phase(target, "dev_completion", config, auth, execute=True, client=client)
            artifact.unlink()
            original = rows(target / "dev/judgements.jsonl")
            original[6]["new_reserved_calls"] = 1
            lines(target / "dev/judgements.jsonl", original)
            with self.assertRaises(ValueError):
                run_phase(target, "dev_completion", config, auth, execute=True, client=client)
            self.assertEqual(client.calls, 6)

    def test_transport_diagnostics_are_recorded_before_client_hides_reason(self):
        from research_assessment_v5.protocol import read
        for failure in (URLError(OSError("socket failed test-secret")), TimeoutError("deadline test-secret")):
            class Opener:
                def open(self, *args, **kwargs): raise failure
            class Client:
                _opener = Opener()
                def _key(self): return "test-secret"
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                recorder = _RecordingOpener(Client())
                recorder.directory = Path(directory)
                with self.assertRaises(type(failure)):
                    recorder.open(Request("https://example.invalid", data=b"{}", method="POST"))
                receipt = read(Path(directory) / "http.transport.error.json")
                self.assertNotIn("test-secret", json.dumps(receipt))
                self.assertIn("[REDACTED_CREDENTIAL]", receipt["reason"])
                self.assertEqual(receipt["stage"], "open")
        class TimeoutResponse:
            def read(self, *args): raise TimeoutError("read timeout test-secret")
        with tempfile.TemporaryDirectory() as directory:
            response = _RecordedResponse(TimeoutResponse(), Path(directory) / "http.response.body.raw", b"test-secret")
            with self.assertRaises(TimeoutError):
                response.read()
            diagnostic = read(Path(directory) / "http.transport.error.json")
            self.assertEqual(diagnostic["stage"], "read")
            self.assertNotIn("test-secret", json.dumps(diagnostic))

    def test_actual_http_error_body_is_saved_hashed_redacted_and_reraised(self):
        from research_assessment_v5.protocol import read
        raw = b'{"error":"test-secret"}'
        class Opener:
            def open(self, *args, **kwargs):
                raise HTTPError("https://example.invalid", 429, "limited", {}, BytesIO(raw))
        class Client:
            _opener = Opener()
            def _key(self): return "test-secret"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            recorder = _RecordingOpener(Client())
            recorder.directory = target
            request_body = b'{"model":"test","messages":[{"role":"user","content":"current turn"}]}'
            request = Request("https://example.invalid", data=request_body, method="POST")
            with self.assertRaises(HTTPError):
                recorder.open(request)
            self.assertEqual((target / "http.request.body.json").read_bytes(), request_body)
            self.assertEqual((target / "http.error.body.raw").read_bytes(), b'{"error":"[REDACTED_CREDENTIAL]"}')
            receipt = read(target / "http.error.body.sha256.json")
            self.assertEqual(receipt["original_body_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt["http_status"], 429)
            self.assertTrue(receipt["credential_redacted"])

    def test_failed_single_calibration_blocks_main_and_retains_twenty_review_places(self):
        from research_assessment_v5.protocol import read, rows
        class InvalidJudge:
            def __init__(self): self.calls = 0
            def preflight(self):
                return {"blocked": False, "calls_reserved": self.calls, "charged_or_reserved_usd": "0"}
            def complete(self, *args):
                if self.calls == 8:
                    raise BudgetError("Persisted budget exhausted before calibration reservation")
                self.calls += 1
                return Completion("{invalid", {}, 0, {"backend": "test_only"})
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            prepare(target)
            config = Path(directory) / "config.json"
            auth = Path(directory) / "receipt.json"
            write(config, {"model": "openai/gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192,
                           "max_cost_usd": "16", "max_calls": 126})
            write(auth, {})
            client = InvalidJudge()
            dev = run_phase(target, "dev", config, auth, execute=True, client=client)
            self.assertEqual(dev["attempted_calls"], 8)
            frozen = freeze(target, config, auth)
            self.assertEqual(frozen["correction_count"], 0)
            check_freeze(target, config, auth)
            calibration = run_phase(target, "calibration", config, auth, execute=True, client=client)
            self.assertEqual(calibration["attempted_calls"], 1)
            self.assertEqual(client.calls, 8)
            self.assertFalse(read(target / "calibration/qualification.json")["qualified"])
            packet = rows(target / "human_review/packet.jsonl")
            self.assertEqual(len(packet), 20)
            self.assertTrue(all("method" not in item and "expected_verdict" not in item for item in packet))
            self.assertEqual(read(target / "human_review/manifest.json")["histories"], 5)
            self.assertEqual(read(target / "report.json")["assessed"], 0)
            with self.assertRaises(ValueError):
                run_phase(target, "main", config, auth, execute=True, client=client)
            with self.assertRaises(FileExistsError):
                run_phase(target, "calibration", config, auth, execute=True, client=client)
            self.assertEqual(client.calls, 8)
            changed = read(config)
            changed["timeout"] = 900
            write(config, changed)
            with self.assertRaises(ValueError):
                check_freeze(target, config, auth)

    def test_budget_and_model_limits_are_fixed(self):
        valid = {"model": "openai/gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192,
                 "max_cost_usd": "16", "max_calls": 126}
        _settings(valid)
        for key, value in (("max_calls", 127), ("max_output_tokens", 1536), ("max_cost_usd", "20"),
                           ("model", "different"), ("reasoning_effort", "high")):
            with self.assertRaises(ValueError):
                _settings({**valid, key: value})

    def test_prepare_retains_denominator_blinds_and_does_not_modify_sources(self):
        from research_assessment_v5.protocol import rows
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            result = prepare(target)
            self.assertEqual((result["frozen_rows"], result["accepted_rows"], result["technical_failures"]), (128, 94, 34))
            cases = rows(target / "inputs/main.cases.jsonl")
            self.assertEqual(len(cases), 94)
            self.assertEqual(len(rows(target / "mapping.private.jsonl")), 128)
            for item in cases:
                self.assertNotIn("method", item)
                self.assertNotIn("expected_verdict", json.dumps(item))
                sources(item)
            with self.assertRaises(FileExistsError):
                prepare(target)

    def test_transport_unknown_stops_and_cannot_repeat_even_new_workspace(self):
        class UnknownClient:
            def __init__(self): self.calls = 0
            def preflight(self):
                return {"blocked": False, "calls_reserved": self.calls, "charged_or_reserved_usd": "0.1"}
            def complete(self, *args):
                self.calls += 1
                raise ProviderRequestError("unknown", code="provider_connection_error", ledger_call_id=1)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            prepare(target)
            config_path = Path(directory) / "config.json"
            auth_path = Path(directory) / "receipt.json"
            write(config_path, {"model": "openai/gpt-6.1-sol", "reasoning_effort": "medium", "max_output_tokens": 8192,
                                "max_cost_usd": "16", "max_calls": 126})
            write(auth_path, {})
            client = UnknownClient()
            self.assertTrue(run_phase(target, "dev", config_path, auth_path)["dry_run"])
            self.assertEqual(client.calls, 0)
            result = run_phase(target, "dev", config_path, auth_path, execute=True, client=client)
            self.assertEqual(result["attempted_calls"], 1)
            self.assertEqual(client.calls, 1)
            with self.assertRaises(FileExistsError):
                run_phase(target, "dev", config_path, auth_path, execute=True, client=client)
            second = Path(directory) / "other"
            prepare(second)
            with self.assertRaises(FileExistsError):
                run_phase(second, "dev", config_path, auth_path, execute=True, client=client)
            self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
