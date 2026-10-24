"""Raw local HTTP evidence before validation, without network or PostgreSQL."""
import json

import pytest
import requests

from local_app.chat_response_journal import ChatResponseJournal, journal_request
from local_app.runtime import OllamaRuntime
from tests.test_social_chat import Settings


class HttpResponse:
    def __init__(self, body, status=200):
        self.text, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("private transport error")

    def json(self):
        return json.loads(self.text)


class HttpSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_nonfinite_local_json_preserves_original_body_before_rejection(tmp_path, value):
    body = '{"message":{"content":"Сырой ответ"},"value":' + value + '}'
    http = HttpSession(HttpResponse(body))
    runtime = OllamaRuntime(Settings(), session=http)
    journal = ChatResponseJournal(tmp_path / "chat.sqlite3", database_identity="a" * 64)
    with pytest.raises(RuntimeError, match="unsupported JSON"):
        with journal_request(journal, actor_id="actor", learner_id="learner"):
            runtime.chat([{"role": "user", "content": "Private prompt not persisted"}])
    record, = journal.list_calls()
    assert record["outcome"] == "returned"
    assert record["raw_response"] == {"http_status": 200, "body": body}
    assert http.calls == 1


@pytest.mark.parametrize("status,body,error_type", [
    (200, '{"message":', "JSONDecodeError"),
    (503, "Сырой отказ локального провайдера", "HTTPError"),
])
def test_bad_local_http_reply_is_saved_before_raising(tmp_path, status, body, error_type):
    http = HttpSession(HttpResponse(body, status))
    runtime = OllamaRuntime(Settings(), session=http)
    journal = ChatResponseJournal(tmp_path / "chat.sqlite3", database_identity="a" * 64)
    with pytest.raises((ValueError, requests.HTTPError)):
        with journal_request(journal, actor_id="actor", learner_id="learner"):
            runtime.chat([{"role": "user", "content": "Private prompt not persisted"}])
    record, = journal.list_calls()
    assert record["outcome"] == "raised" and record["error_type"] == error_type
    assert record["raw_response"] == {"http_status": status, "body": body}
    assert http.calls == 1


def test_local_timeout_records_type_without_private_exception_or_automatic_retry(tmp_path):
    http = HttpSession(error=requests.Timeout("private path password and prompt"))
    runtime = OllamaRuntime(Settings(), session=http)
    journal = ChatResponseJournal(tmp_path / "chat.sqlite3", database_identity="a" * 64)
    with pytest.raises(requests.Timeout):
        with journal_request(journal, actor_id="actor", learner_id="learner"):
            runtime.chat([{"role": "user", "content": "Private prompt not persisted"}])
    record, = journal.list_calls()
    assert record["outcome"] == "raised" and record["error_type"] == "Timeout"
    assert record["raw_response"] is None
    assert "private path" not in json.dumps(record) and http.calls == 1
