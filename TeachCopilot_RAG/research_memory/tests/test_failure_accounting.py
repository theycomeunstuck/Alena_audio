"""Failed responses must remain failures without hiding known model costs."""

import io
import json
import unittest
from unittest.mock import Mock, patch

from research_memory.clients import Completion, CompletionValidationError, InputBudgetError, OllamaClient
from research_memory.runner import CallJournal
from research_memory.spiderssense import ProviderRequestError


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return json.dumps({"done": True, "done_reason": "length",
                           "message": {"content": '{"records": ['},
                           "prompt_eval_count": 103, "eval_count": 512,
                           "total_duration": 500000000}).encode()


class FailureAccountingTests(unittest.TestCase):
    def test_preflight_writer_refusal_is_traced_but_not_a_model_call(self):
        diagnostic = {"method": "cl100k_base_with_safety_reserve_v1", "guarded_input_tokens": 9000,
                      "available_input_tokens": 6656, "within_guard": False, "exact_token_count": False}
        client = Mock()
        client.complete.side_effect = InputBudgetError("Writer budget exceeded", Completion(
            "", {"input_tokens": 0, "output_tokens": 0}, 0,
            {"network_request_performed": False, "input_budget": diagnostic}), "writer_input_budget_exceeded")
        calls, trace = io.StringIO(), io.StringIO()
        journal = CallJournal(client, calls, max_calls=1, trace_stream=trace)
        messages = [{"role": "user", "content": "SYNTHETIC_FULL_INPUT"}]
        with self.assertRaises(RuntimeError) as rejected:
            journal(messages, "extract")
        row, raw = json.loads(calls.getvalue()), json.loads(trace.getvalue())
        self.assertEqual(rejected.exception.code, "writer_input_budget_exceeded")
        self.assertFalse(row["performed"])
        self.assertEqual(journal.performed, 0)
        self.assertEqual(journal.attempted, 1)
        self.assertEqual(raw["messages"], messages)
        self.assertEqual(raw["completion_metadata"]["input_budget"], diagnostic)
        self.assertFalse(raw["completion_metadata"]["network_request_performed"])
        self.assertEqual(row["usage"], {"input_tokens": 0, "output_tokens": 0})

    def test_provider_failure_preserves_safe_status_and_reservation_id(self):
        client = Mock()
        client.complete.side_effect = ProviderRequestError(
            "Untrusted body with sk-secret must never be saved", code="provider_http_error",
            http_status=429, ledger_call_id=7)
        calls, trace = io.StringIO(), io.StringIO()
        journal = CallJournal(client, calls, max_calls=1, trace_stream=trace)
        with self.assertRaises(RuntimeError):
            journal([{"role": "user", "content": "synthetic"}], "judge")
        row, raw = json.loads(calls.getvalue()), json.loads(trace.getvalue())
        self.assertEqual(row["error_code"], "provider_http_error")
        self.assertEqual(row["http_status"], 429)
        self.assertEqual(row["ledger_call_id"], 7)
        self.assertIsNone(row["usage"]["input_tokens"])
        self.assertEqual(raw["completion_metadata"]["provider_error_code"], "provider_http_error")
        self.assertNotIn("sk-secret", calls.getvalue() + trace.getvalue())

    def test_arbitrary_exception_attributes_are_not_persisted(self):
        for exception in (RuntimeError("sk-secret"),
                ProviderRequestError("sk-secret", code="sk-secret", http_status="sk-secret")):
            client = Mock()
            client.complete.side_effect = exception
            calls, trace = io.StringIO(), io.StringIO()
            journal = CallJournal(client, calls, trace_stream=trace)
            with self.assertRaises(RuntimeError):
                journal([], "judge")
            self.assertNotIn("sk-secret", calls.getvalue() + trace.getvalue())

    @patch("research_memory.clients.build_opener")
    def test_truncated_reply_retains_usage_and_trace_but_is_not_accepted(self, build_opener):
        server = Mock()
        server.open.return_value = Response()
        build_opener.return_value = server
        client = OllamaClient()
        with self.assertRaises(CompletionValidationError) as rejected:
            client.complete([], "extract")
        self.assertEqual(rejected.exception.completion.usage,
                         {"input_tokens": 103, "output_tokens": 512})
        self.assertEqual(rejected.exception.code, "truncated_completion")

        calls, trace = io.StringIO(), io.StringIO()
        journal = CallJournal(client, calls, max_calls=1, trace_stream=trace)
        with self.assertRaises(RuntimeError):
            journal([{"role": "user", "content": "Вымышленная история"}], "extract")
        row = json.loads(calls.getvalue())
        raw = json.loads(trace.getvalue())
        self.assertTrue(row["performed"])
        self.assertEqual(row["error_code"], "truncated_completion")
        self.assertEqual(row["usage"], {"input_tokens": 103, "output_tokens": 512})
        self.assertEqual(raw["completion_text"], '{"records": [')
        self.assertEqual(raw["error"], "CompletionValidationError")
        self.assertEqual(journal.performed, 1)
