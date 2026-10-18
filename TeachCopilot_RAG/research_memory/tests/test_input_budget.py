"""Writer refusals preserve complete input and never trigger hidden model retries."""

import json
import unittest
from unittest.mock import Mock, patch

from research_memory.clients import (
    CompletionValidationError,
    InputBudgetError,
    OllamaClient,
    input_budget_diagnostic,
)


class Response:
    def __init__(self, *, input_tokens=40, output_tokens=8):
        self.payload = json.dumps({"done": True, "done_reason": "stop",
            "message": {"content": '{"records": []}'},
            "prompt_eval_count": input_tokens, "eval_count": output_tokens}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        return self.payload


class InputBudgetTests(unittest.TestCase):
    @patch("research_memory.clients.build_opener")
    def test_large_writer_input_is_refused_before_a_network_request(self, build_opener):
        server = Mock()
        build_opener.return_value = server
        client = OllamaClient(max_output_tokens=1536)
        messages = [{"role": "user", "content": "unrelated words " * 20000}]
        before = json.dumps(messages)
        for purpose in ("summary", "extract"):
            with self.subTest(purpose=purpose), self.assertRaises(InputBudgetError) as raised:
                client.complete(messages, purpose, {"type": "object"})
            error = raised.exception
            self.assertEqual(error.code, "writer_input_budget_exceeded")
            self.assertEqual(error.completion.usage, {"input_tokens": 0, "output_tokens": 0})
            self.assertEqual(error.completion.elapsed_seconds, 0)
            self.assertEqual(error.completion.text, "")
            self.assertFalse(error.completion.metadata["network_request_performed"])
            budget = error.completion.metadata["input_budget"]
            self.assertEqual(budget["available_input_tokens"], 8192 - 1536)
            self.assertGreater(budget["guarded_input_tokens"], budget["available_input_tokens"])
            self.assertFalse(budget["exact_token_count"])
            self.assertEqual(budget["token_budget_kind"], "heuristic_not_model_tokenizer")
        self.assertEqual(json.dumps(messages), before)
        server.open.assert_not_called()

    @patch("research_memory.clients.build_opener")
    def test_schema_alone_can_exceed_the_full_writer_budget(self, build_opener):
        server = Mock()
        build_opener.return_value = server
        schema = {"type": "string", "enum": [f"source_{index:06d}" for index in range(3000)]}
        with self.assertRaises(InputBudgetError) as raised:
            OllamaClient(max_output_tokens=1536).complete([], "extract", schema)
        diagnostic = raised.exception.completion.metadata["input_budget"]
        self.assertGreater(diagnostic["schema_utf8_bytes"], 30000)
        self.assertGreater(diagnostic["guarded_input_tokens"], diagnostic["available_input_tokens"])
        server.open.assert_not_called()

    @patch("research_memory.clients.build_opener")
    def test_ordinary_cyrillic_history_is_sent_whole_with_transparent_estimate(self, build_opener):
        server = Mock()
        server.open.return_value = Response()
        build_opener.return_value = server
        messages = [{"role": "user", "content": "Учитель: ученик решил задачу с подсказкой.\n" * 120}]
        completion = OllamaClient(max_output_tokens=1536).complete(messages, "summary")
        request = server.open.call_args.args[0]
        self.assertEqual(json.loads(request.data)["messages"], messages)
        self.assertEqual(server.open.call_count, 1)
        diagnostic = completion.metadata["input_budget"]
        self.assertEqual(diagnostic["estimator"], "cl100k_base")
        self.assertGreater(diagnostic["message_utf8_bytes"], diagnostic["available_input_tokens"])
        self.assertTrue(diagnostic["within_guard"])
        self.assertEqual(diagnostic["actual_input_tokens"], 40)
        self.assertEqual(diagnostic["actual_output_tokens"], 8)

    @patch("research_memory.clients._estimate_encoding")
    def test_guard_boundary_and_output_reservation(self, encoding):
        fake = Mock()
        encoding.return_value = fake
        fake.encode.side_effect = lambda text, **kwargs: list(range(832)) if text else []
        messages = [{"role": "user", "content": "lesson"}]
        at_limit = input_budget_diagnostic(messages, None, context_tokens=2048,
                                           max_output_tokens=512, purpose="extract")
        self.assertEqual(at_limit["guarded_input_tokens"], 1536)
        self.assertTrue(at_limit["within_guard"])
        fake.encode.side_effect = lambda text, **kwargs: list(range(833)) if text else []
        over_limit = input_budget_diagnostic(messages, None, context_tokens=2048,
                                             max_output_tokens=512, purpose="extract")
        self.assertFalse(over_limit["within_guard"])
        self.assertEqual(over_limit["guarded_input_tokens"], 1537)

    @patch("research_memory.clients._estimate_encoding", side_effect=ImportError)
    def test_offline_fallback_is_explicit_and_still_counts_schema(self, encoding):
        diagnostic = input_budget_diagnostic([{"role": "user", "content": "урок"}],
            {"type": "object"}, context_tokens=8192, max_output_tokens=1536, purpose="extract")
        self.assertEqual(diagnostic["estimator"], "utf8_bytes_fallback")
        self.assertFalse(diagnostic["exact_token_count"])
        self.assertEqual(diagnostic["estimated_payload_tokens"],
                         diagnostic["message_utf8_bytes"] + diagnostic["schema_utf8_bytes"])

    @patch("research_memory.clients.build_opener")
    def test_context_saturation_preserves_response_and_actual_cost_without_retry(self, build_opener):
        server = Mock()
        server.open.return_value = Response(input_tokens=6656, output_tokens=7)
        build_opener.return_value = server
        with self.assertRaises(CompletionValidationError) as raised:
            OllamaClient(max_output_tokens=1536).complete([], "extract")
        error = raised.exception
        self.assertEqual(error.code, "context_saturation_requires_review")
        self.assertEqual(error.completion.text, '{"records": []}')
        self.assertEqual(error.completion.usage, {"input_tokens": 6656, "output_tokens": 7})
        self.assertTrue(error.completion.metadata["network_request_performed"])
        self.assertEqual(server.open.call_count, 1)

    @patch("research_memory.clients.build_opener")
    def test_action_requests_are_diagnosed_but_writer_policy_does_not_change_them(self, build_opener):
        server = Mock()
        server.open.return_value = Response(input_tokens=6656)
        build_opener.return_value = server
        completion = OllamaClient(max_output_tokens=1536).complete(
            [{"role": "user", "content": "unrelated words " * 20000}], "action")
        self.assertFalse(completion.metadata["input_budget"]["enforced"])
        self.assertFalse(completion.metadata["input_budget"]["within_guard"])
        self.assertEqual(server.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
