import json
import unittest
from unittest.mock import Mock, patch

from research_memory.clients import FixtureClient, OllamaClient
from urllib.request import ProxyHandler


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        self.limit = limit
        return self.payload


class OllamaClientTests(unittest.TestCase):
    def test_restricts_urls_to_loopback_http_without_secrets(self):
        for url in (
            "https://127.0.0.1:11435", "http://example.com", "http://127.0.0.1:11435/x?key=x",
            "http://user:pass@127.0.0.1:11435", "http://127.0.0.1:11435/#fragment",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                OllamaClient(base_url=url)
        self.assertEqual(OllamaClient(base_url="http://[::1]:11435").base_url,
                         "http://[::1]:11435")

    @patch("research_memory.clients.build_opener")
    def test_sends_nonstreaming_request_and_maps_usage(self, build_opener):
        server = Mock()
        server.open.return_value = _Response(json.dumps({
            "message": {"content": "Готово"}, "done_reason": "stop", "done": True,
            "prompt_eval_count": 17, "eval_count": 4,
            "total_duration": 123, "load_duration": 20,
        }).encode())
        build_opener.return_value = server
        client = OllamaClient()
        result = client.complete([{"role": "user", "content": "пример"}], "extract",
                                 {"type": "object"})
        request = server.open.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11435/api/chat")
        self.assertEqual(sent["stream"], False)
        self.assertEqual(sent["think"], False)
        self.assertNotIn("temperature", sent)
        self.assertEqual(sent["options"], {"num_ctx": 8192, "num_predict": 1024,
                                            "seed": 42, "temperature": 0})
        self.assertEqual(sent["format"], {"type": "object"})
        self.assertEqual(result.text, "Готово")
        self.assertEqual(result.usage, {"input_tokens": 17, "output_tokens": 4})
        self.assertEqual(result.metadata["backend"], "ollama")
        self.assertNotIn("messages", result.metadata)
        self.assertEqual(server.open.call_args.kwargs["timeout"], 180)
        handlers = build_opener.call_args.args
        self.assertIsInstance(handlers[0], ProxyHandler)
        self.assertEqual(handlers[0].proxies, {})

    @patch("research_memory.clients.build_opener")
    def test_rejects_malformed_empty_and_truncated_results(self, build_opener):
        client = OllamaClient()
        server = Mock()
        build_opener.return_value = server
        build_opener.return_value = server
        for payload in (b"not json", b'{"message":{"content":"  "},"done":true}',
                        b'{"message":{"content":"x"},"done":true,"done_reason":"LeNgTh"}',
                        b'{"message":{"content":"x"},"done":false}'):
            server.open.return_value = _Response(payload)
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                client.complete([], "summary")

    def test_validates_limits(self):
        for kwargs in ({"timeout": 0}, {"context_tokens": 1024},
                       {"max_output_tokens": 0}, {"context_tokens": 2048,
                                                   "max_output_tokens": 2048}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OllamaClient(**kwargs)


class FixtureClientTests(unittest.TestCase):
    def test_fixture_outputs_are_explicitly_non_scientific(self):
        client = FixtureClient()
        summary = client.complete([], "summary")
        extract = client.complete([], "extract")
        action = client.complete([], "action")
        self.assertIn("Демонстрационное", summary.text)
        self.assertEqual(json.loads(extract.text), {"records": []})
        action_obj = json.loads(action.text)
        self.assertEqual(action_obj["action"], "clarify_missing_observation")
        self.assertTrue(action_obj["review_required"])
        for completion in (summary, extract, action):
            self.assertEqual(completion.metadata["backend"], "fixture")
            self.assertFalse(completion.metadata["scientific_result"])
            self.assertEqual(completion.elapsed_seconds, 0)
            self.assertEqual(completion.usage, {"input_tokens": None, "output_tokens": None})


if __name__ == "__main__":
    unittest.main()
