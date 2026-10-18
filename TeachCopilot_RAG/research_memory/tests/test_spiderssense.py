import json
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler

from research_memory.clients import CompletionValidationError, MAX_RESPONSE_BYTES
from research_memory.spiderssense import (BASE_URL, KEY_ENV, BudgetError,
                                         ProviderRequestError, SpiderssenseClient)


FAKE_KEY = "not-a-real-spiderssense-credential-123456"
MESSAGES = [{"role": "user", "content": "Вымышленный учебный пример"}]


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        self.limit = limit
        return self.body


def reply(*, finish="stop", text='{"answer":"example"}', usage=True):
    value = {"choices": [{"message": {"content": text}, "finish_reason": finish}]}
    if usage is True:
        value["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
    elif isinstance(usage, dict):
        value["usage"] = usage
    return json.dumps(value).encode()


class SpiderssenseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {KEY_ENV: FAKE_KEY})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.factory = patch("research_memory.spiderssense.build_opener")
        self.build_opener = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.transport = Mock()
        self.transport.open.return_value = Response(reply())
        self.build_opener.return_value = self.transport

    def client(self, **overrides):
        config = {"model": "explicit-test-model", "input_usd_per_million": "1",
                  "output_usd_per_million": "2", "pricing_verified": True,
                  "pricing_source": "https://spiderssense.com/v1/models",
                  "ledger_path": self.root / "budget.sqlite3", "max_output_tokens": 64,
                  "key_file": self.root / "dummy.key"}
        return SpiderssenseClient(**{**config, **overrides})

    def test_unknown_prices_and_invalid_money_refuse_before_network(self):
        cases = [{"pricing_verified": False}, {"input_usd_per_million": None},
                 {"input_usd_per_million": "NaN"}, {"output_usd_per_million": "Infinity"},
                 {"output_usd_per_million": -1}, {"max_cost_usd": 0},
                 {"max_cost_usd": True}, {"max_cost_usd": "NaN"},
                 {"pricing_source": "https://other.example/prices"},
                 {"pricing_source": "https://spiderssense.com/docs?key=secret"},
                 {"token_field": "automatic"}, {"max_calls": 1.5},
                 {"timeout": float("nan")}, {"input_cost_multiplier": "1.1"}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.client(**case)
        self.transport.open.assert_not_called()

    def test_preflight_neither_reads_credential_contents_nor_uses_network(self):
        client = self.client()
        with patch.object(Path, "open", side_effect=AssertionError("No credential read")):
            status = client.preflight()
        self.assertTrue(status["key_present"])
        self.assertFalse(status["network_used"])
        self.assertEqual(status["available_budget_usd"], "4")
        self.assertNotIn(FAKE_KEY, json.dumps(status))
        self.transport.open.assert_not_called()

    def test_empty_key_file_is_not_reported_as_ready(self):
        key_path = self.root / "empty.key"
        key_path.write_text("", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            client = self.client(key_file=key_path)
            with patch.object(Path, "open", side_effect=AssertionError("No credential read")):
                self.assertFalse(client.preflight()["key_present"])
        self.transport.open.assert_not_called()

    def test_endpoint_headers_nonstream_schema_and_token_field_are_explicit(self):
        client = self.client(token_field="max_tokens")
        result = client.complete(MESSAGES, "action", {"type": "object"})
        request = self.transport.open.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, BASE_URL + "/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_KEY)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_tokens"], 64)
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("response_format", payload)
        self.assertIn('"type": "object"', payload["messages"][-1]["content"])
        self.assertEqual(MESSAGES, [{"role": "user", "content": "Вымышленный учебный пример"}])
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertEqual(result.usage["output_tokens"], 20)
        self.assertEqual(result.usage["cost_status"], "settled")
        self.assertNotIn(FAKE_KEY, json.dumps(result.__dict__))

    def test_structured_output_is_only_enabled_explicitly(self):
        client = self.client(use_json_schema=True)
        client.complete(MESSAGES, "action", {"type": "object"})
        payload = json.loads(self.transport.open.call_args.args[0].data)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["messages"], MESSAGES)

    def test_json_object_mode_keeps_schema_instruction_without_strict_schema(self):
        client = self.client(use_json_object=True)
        client.complete(MESSAGES, "judge", {"type": "object"})
        payload = json.loads(self.transport.open.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIn('"type": "object"', payload["messages"][-1]["content"])
        self.assertEqual(payload["messages"][:-1], MESSAGES)
        self.assertTrue(client.preflight()["use_json_object"])

    def test_json_object_mode_rejects_conflict_and_nonboolean_before_network(self):
        for options in ({"use_json_object": True, "use_json_schema": True},
                        {"use_json_object": "true"}, {"use_json_object": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.client(**options)
        self.transport.open.assert_not_called()

    def test_json_object_false_preserves_legacy_configuration_fingerprint(self):
        implicit = self.client()
        explicit = self.client(use_json_object=False)
        self.assertNotIn("use_json_object", implicit._config)
        self.assertNotIn("use_json_object", explicit._config)
        self.assertEqual(implicit._config_json, explicit._config_json)
        self.assertEqual(implicit.preflight()["configuration_hash"], explicit.preflight()["configuration_hash"])
        with self.assertRaises(BudgetError):
            self.client(use_json_object=True)

    def test_reasoning_effort_is_explicit_and_frozen_without_temperature(self):
        client = self.client(reasoning_effort="medium")
        client.complete(MESSAGES, "judge")
        payload = json.loads(self.transport.open.call_args.args[0].data)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertNotIn("temperature", payload)
        with self.assertRaises(BudgetError):
            self.client(reasoning_effort="high")

    def test_redirects_and_environment_proxies_are_disabled(self):
        self.client()
        handlers = self.build_opener.call_args.args
        self.assertIsInstance(handlers[0], ProxyHandler)
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsNone(handlers[1].redirect_request(Mock(), None, 307, "redirect", {},
                                                     "https://other.example/steal"))

    def test_unknown_network_result_keeps_reservation_across_restart(self):
        client = self.client(max_calls=1)
        self.transport.open.side_effect = URLError("secret body " + FAKE_KEY)
        with self.assertRaises(RuntimeError) as error:
            client.complete(MESSAGES, "action")
        self.assertNotIn(FAKE_KEY, str(error.exception))
        before = client.preflight()
        self.assertGreater(float(before["charged_or_reserved_usd"]), 0)
        resumed = self.client(max_calls=1)
        self.assertEqual(before["charged_or_reserved_usd"], resumed.preflight()["charged_or_reserved_usd"])
        with self.assertRaises(BudgetError):
            resumed.complete(MESSAGES, "action")
        self.assertEqual(self.transport.open.call_count, 1)

    def test_atomic_reservation_allows_only_one_of_two_concurrent_clients(self):
        clients = [self.client(max_calls=1), self.client(max_calls=1)]
        self.transport.open.side_effect = URLError("unknown outcome")

        def call(client):
            try:
                client.complete(MESSAGES, "action")
            except Exception as error:
                return type(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(call, clients))
        self.assertCountEqual(outcomes, [ProviderRequestError, BudgetError])
        self.assertEqual(self.transport.open.call_count, 1)
        self.assertEqual(clients[0].preflight()["calls_reserved"], 1)

    def test_partial_token_usage_keeps_full_reservation(self):
        self.transport.open.return_value = Response(reply(usage={"prompt_tokens": 100, "cost": "0.0001"}))
        result = self.client().complete(MESSAGES, "action")
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertIsNone(result.usage["output_tokens"])
        self.assertEqual(result.usage["cost_status"], "reserved_unknown")
        self.assertEqual(result.usage["cost_usd"], result.metadata["reserved_usd"])

    def test_usage_above_output_limit_blocks_even_when_reported_cost_is_small(self):
        self.transport.open.return_value = Response(reply(usage={"prompt_tokens": 100,
                                                   "completion_tokens": 65, "cost": "0.0001"}))
        client = self.client()
        with self.assertRaises(CompletionValidationError) as error:
            client.complete(MESSAGES, "action")
        self.assertEqual(error.exception.code, "budget_accounting_mismatch")
        self.assertTrue(client.preflight()["blocked"])

    def test_overbudget_is_refused_before_request(self):
        client = self.client(max_cost_usd="0.0000001")
        with self.assertRaises(BudgetError):
            client.complete(MESSAGES, "action")
        self.transport.open.assert_not_called()
        self.assertEqual(client.preflight()["calls_reserved"], 0)

    def test_provider_transport_diagnostics_are_safe_and_keep_reservations(self):
        client = self.client()
        failures = [
            (HTTPError(BASE_URL + "?secret=" + FAKE_KEY, 503, FAKE_KEY,
                       {"Authorization": FAKE_KEY}, None), "provider_http_error", 503),
            (URLError(FAKE_KEY), "provider_connection_error", None),
            (RuntimeError(FAKE_KEY), "provider_transport_error", None),
        ]
        previous_spend = 0.0
        for call_id, (failure, code, status) in enumerate(failures, 1):
            self.transport.open.side_effect = failure
            with self.subTest(code=code), self.assertRaises(ProviderRequestError) as raised:
                client.complete(MESSAGES, "judge")
            error = raised.exception
            self.assertIsInstance(error, RuntimeError)
            self.assertEqual(error.code, code)
            self.assertEqual(error.http_status, status)
            self.assertEqual(error.ledger_call_id, call_id)
            self.assertNotIn(FAKE_KEY, str(error))
            self.assertNotIn(FAKE_KEY, json.dumps(error.__dict__))
            spent = float(client.preflight()["charged_or_reserved_usd"])
            self.assertGreater(spent, previous_spend)
            previous_spend = spent
        self.assertEqual(client.preflight()["calls_reserved"], 3)

    def test_resume_cannot_reset_spend_by_changing_configuration(self):
        self.client()
        for change in ({"max_cost_usd": 6}, {"model": "different-model"},
                       {"input_usd_per_million": "0.1"}, {"max_calls": 99}):
            with self.subTest(change=change), self.assertRaises(BudgetError):
                self.client(**change)

    def test_missing_usage_retains_full_reservation(self):
        self.transport.open.return_value = Response(reply(usage=False))
        client = self.client()
        result = client.complete(MESSAGES, "action")
        self.assertEqual(result.usage["cost_status"], "reserved_unknown")
        self.assertEqual(result.usage["cost_usd"], result.metadata["reserved_usd"])

    def test_invalid_provider_cost_does_not_release_unknown_cost(self):
        self.transport.open.return_value = Response(reply(usage={"prompt_tokens": 100,
                                                   "completion_tokens": 20, "cost": "unknown"}))
        result = self.client().complete(MESSAGES, "action")
        self.assertEqual(result.usage["cost_status"], "reserved_unknown")
        self.assertEqual(result.usage["cost_usd"], result.metadata["reserved_usd"])

    def test_reported_cost_is_compared_against_conservative_rate(self):
        self.transport.open.return_value = Response(reply(usage={"prompt_tokens": 100,
                                                   "completion_tokens": 20, "cost": "0.00001"}))
        result = self.client().complete(MESSAGES, "action")
        self.assertEqual(result.usage["rate_upper_cost_usd"], "0.000165")
        self.assertEqual(result.usage["cost_usd"], "0.000165")

    def test_overreservation_reported_cost_blocks_future_calls(self):
        self.transport.open.return_value = Response(reply(usage={"prompt_tokens": 100,
                                                   "completion_tokens": 20, "cost": "2"}))
        client = self.client()
        with self.assertRaises(CompletionValidationError) as error:
            client.complete(MESSAGES, "action")
        self.assertEqual(error.exception.code, "budget_accounting_mismatch")
        self.assertTrue(client.preflight()["blocked"])
        self.assertEqual(client.preflight()["charged_or_reserved_usd"], "2")
        with self.assertRaises(BudgetError):
            self.client().complete(MESSAGES, "action")
        self.assertEqual(self.transport.open.call_count, 1)

    def test_partial_response_preserves_text_and_billed_usage(self):
        self.transport.open.return_value = Response(reply(finish="length", text="partial reply"))
        client = self.client()
        with self.assertRaises(CompletionValidationError) as error:
            client.complete(MESSAGES, "action")
        self.assertEqual(error.exception.code, "truncated_completion")
        self.assertEqual(error.exception.completion.text, "partial reply")
        self.assertEqual(error.exception.completion.usage["output_tokens"], 20)
        self.assertGreater(float(client.preflight()["charged_or_reserved_usd"]), 0)

    def test_malformed_and_oversized_response_retain_reservations(self):
        client = self.client()
        for response in (b"invalid JSON " + FAKE_KEY.encode(), b"x" * (MAX_RESPONSE_BYTES + 1)):
            self.transport.open.return_value = Response(response)
            with self.assertRaises(ValueError) as error:
                client.complete(MESSAGES, "action")
            self.assertNotIn(FAKE_KEY, str(error.exception))
        self.assertEqual(client.preflight()["calls_reserved"], 2)
        self.assertGreater(float(client.preflight()["charged_or_reserved_usd"]), 0)

    def test_errors_and_echoed_credentials_do_not_reach_completion_trace(self):
        client = self.client()
        self.transport.open.side_effect = HTTPError(BASE_URL, 302, FAKE_KEY, {}, None)
        with self.assertRaises(RuntimeError) as error:
            client.complete(MESSAGES, "action")
        self.assertNotIn(FAKE_KEY, str(error.exception))
        self.transport.open.side_effect = None
        self.transport.open.return_value = Response(reply(text="echo " + FAKE_KEY))
        with self.assertRaises(CompletionValidationError) as error:
            client.complete(MESSAGES, "action")
        self.assertEqual(error.exception.code, "credential_echo")
        self.assertNotIn(FAKE_KEY, json.dumps(error.exception.completion.__dict__))
        with closing(sqlite3.connect(client.ledger_path)) as connection:
            self.assertNotIn(FAKE_KEY, str(connection.execute("SELECT * FROM budget_config").fetchall()))
            self.assertNotIn(FAKE_KEY, str(connection.execute("SELECT * FROM budget_calls").fetchall()))

    def test_credentials_are_read_at_call_time_and_missing_key_does_not_reserve(self):
        self.env.stop()
        with patch.dict(os.environ, {}, clear=True):
            client = self.client()
            self.assertFalse(client.preflight()["key_present"])
            with self.assertRaises(RuntimeError):
                client.complete(MESSAGES, "action")
            self.assertEqual(client.preflight()["calls_reserved"], 0)
            (self.root / "dummy.key").write_text(FAKE_KEY, encoding="utf-8")
            client.complete(MESSAGES, "action")
            request = self.transport.open.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer " + FAKE_KEY)


if __name__ == "__main__":
    unittest.main()
