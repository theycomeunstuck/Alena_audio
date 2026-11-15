"""Private provider evidence with synthetic HTTP only, without paid requests."""
from __future__ import annotations

import base64
import io
import json
import re
import sqlite3
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from local_app.chat_response_journal import ChatResponseJournal, ResponseJournalError, journal_request
from local_app.memory_ported.clients import CompletionValidationError
from local_app.memory_ported.spiderssense import KEY_ENV, ProviderRequestError
from tests.test_answer_adapter import (
    IDENTITIES, SCHEMA, local_messages, projection, setup_adapter,  # noqa: F401 – shared fake HTTP fixture
)


class ByteResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return self.body[:limit]


@pytest.fixture
def journal(tmp_path):
    return ChatResponseJournal(tmp_path / "chat-responses.sqlite3", database_identity="a" * 64)


def request(journal):
    return journal_request(journal, actor_id="actor-1", learner_id="learner-1",
                           job_id="job-1", learning_session_id="lesson-1",
                           client_message_id="message-1")


def answer(adapter):
    value = projection()
    return adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES,
                          response_schema=SCHEMA)


def provider_body(content=None, *, finish_reason="stop"):
    if content is None:
        content = json.dumps({"answer": "Сырой ответ – проверь умножением.", "citations": ["S1", "S2"]},
                             ensure_ascii=False)
    return json.dumps({
        "id": "provider-call-raw-1", "model": "synthetic-provider-model",
        "choices": [{"message": {"role": "assistant", "content": content,
                                  "reasoning_content": "Сырой промежуточный текст."},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }, ensure_ascii=False).encode("utf-8")


def sole_call(journal):
    calls = ChatResponseJournal(journal.path, journal.database_identity).list_calls()
    assert len(calls) == 1
    return calls[0]


def test_raw_body_is_durable_before_provider_budget_settlement_failure(
        setup_adapter, journal, monkeypatch):
    adapter, local, client, transport = setup_adapter
    raw = provider_body()
    transport.open.return_value = ByteResponse(raw)

    def unavailable_settlement(*args):
        observed = sole_call(journal)
        assert observed["outcome"] == "returned"
        assert base64.b64decode(observed["raw_response"]["body_base64"]) == raw
        assert observed["raw_response"]["parsed"] == json.loads(raw)
        raise sqlite3.OperationalError("synthetic budget database unavailable")

    settle = Mock(side_effect=unavailable_settlement)
    monkeypatch.setattr(client, "_settle", settle)
    with pytest.raises(sqlite3.OperationalError, match="synthetic budget"):
        with request(journal):
            answer(adapter)
    assert sole_call(journal)["outcome"] == "returned"
    assert client.preflight()["calls_reserved"] == 1
    assert local.requests == []
    assert transport.open.call_count == settle.call_count == 1


def test_provider_aliases_and_original_usage_survive_before_dealias(setup_adapter, journal):
    adapter, local, _, transport = setup_adapter
    raw = provider_body()
    transport.open.return_value = ByteResponse(raw)
    with request(journal):
        result = answer(adapter)
    record = sole_call(journal)
    captured = record["raw_response"]
    assert record["purpose"] == "chat" and record["backend"] == "spiderssense"
    assert record["outcome"] == "returned" and record["error_type"] is None
    assert captured["http_status"] == 200
    assert captured["body_text"] == raw.decode("utf-8")
    assert base64.b64decode(captured["body_base64"]) == raw
    content = captured["parsed"]["choices"][0]["message"]["content"]
    assert json.loads(content)["citations"] == ["S1", "S2"]
    assert captured["parsed"]["usage"] == {"prompt_tokens": 100, "completion_tokens": 30}
    expected_ids = [item["source_id"] for item in [*projection().memory, *projection().materials]]
    assert json.loads(result["message"]["content"])["citations"] == expected_ids
    serialized = json.dumps(record, ensure_ascii=False)
    for private in (*IDENTITIES, *expected_ids, "RAW CARD", "Authorization", "Bearer "):
        assert private not in serialized
    assert local.requests == [] and transport.open.call_count == 1


@pytest.mark.parametrize("kind", ["malformed", "length", "non_object", "non_finite"])
def test_invalid_or_truncated_provider_body_is_retained_before_validation(
        setup_adapter, journal, kind):
    adapter, local, client, transport = setup_adapter
    bodies = {
        "malformed": b'{"choices": [{"message": {"content": "unfinished',
        "length": provider_body(finish_reason="length"),
        "non_object": b'["unexpected", "provider", "value"]',
        "non_finite": b'{"extra": NaN, "choices": []}',
    }
    raw = bodies[kind]
    transport.open.return_value = ByteResponse(raw)
    expected_error = CompletionValidationError if kind in {"length", "non_finite"} else ValueError
    with pytest.raises(expected_error):
        with request(journal):
            answer(adapter)
    record = sole_call(journal)
    assert record["outcome"] == "returned"
    assert base64.b64decode(record["raw_response"]["body_base64"]) == raw
    if kind in {"malformed", "non_finite"}:
        assert "parsed" not in record["raw_response"]
    assert client.preflight()["calls_reserved"] == 1
    assert transport.open.call_count == 1 and local.requests == []


def test_prepare_failure_prevents_http_and_budget_reservation(setup_adapter, journal, monkeypatch):
    adapter, local, client, transport = setup_adapter
    prepare = Mock(side_effect=ResponseJournalError("synthetic journal prepare failure"))
    monkeypatch.setattr(journal, "prepare", prepare)
    with pytest.raises(ResponseJournalError, match="prepare failure"):
        with request(journal):
            answer(adapter)
    assert prepare.call_count == 1
    assert journal.list_calls() == []
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()
    assert local.requests == []


def test_record_failure_stops_settlement_fallback_and_second_generation(
        setup_adapter, journal, monkeypatch):
    adapter, local, client, transport = setup_adapter
    record = Mock(side_effect=ResponseJournalError("synthetic journal record failure"))
    settle = Mock(wraps=client._settle)
    monkeypatch.setattr(journal, "record", record)
    monkeypatch.setattr(client, "_settle", settle)
    with pytest.raises(ResponseJournalError, match="record failure"):
        with request(journal):
            with pytest.raises(ResponseJournalError, match="record failure"):
                answer(adapter)
            with pytest.raises(ResponseJournalError, match="record failure"):
                answer(adapter)
    assert record.call_count == 1
    settle.assert_not_called()
    assert client.preflight()["calls_reserved"] == 1
    assert transport.open.call_count == 1 and local.requests == []
    assert sole_call(journal)["outcome"] is None


def test_privacy_checker_journal_error_cannot_start_local_fallback(setup_adapter, journal):
    adapter, local, client, transport = setup_adapter
    local.decision = ResponseJournalError("synthetic privacy journal failure")
    with pytest.raises(ResponseJournalError, match="privacy journal failure"):
        with request(journal):
            answer(adapter)
    assert len(local.checks) == 1 and local.requests == []
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()
    assert journal.list_calls() == []


@pytest.mark.parametrize("encoding", ["literal", "unicode", "nested_unicode", "mixed"])
def test_reflected_actual_credential_is_absent_even_after_decoding_journal(
        setup_adapter, journal, monkeypatch, encoding):
    adapter, local, _, transport = setup_adapter
    key = "synthetic-reflected-credential-932"
    monkeypatch.setenv(KEY_ENV, key)
    escaped = "".join(f"\\u{ord(char):04x}" for char in key)
    if encoding == "literal":
        content = json.dumps({"answer": key, "citations": ["S1"]})
    elif encoding == "unicode":
        content = '{"answer":"' + escaped + '","citations":["S1"]}'
    elif encoding == "nested_unicode":
        nested = json.dumps({"echo": escaped})
        content = json.dumps({"answer": nested, "citations": ["S1"]})
    else:
        content = json.dumps({"answer": key + " / " + escaped + " / " + json.dumps({"echo": escaped}),
                              "citations": ["S1"]})
    raw = provider_body(content)
    transport.open.return_value = ByteResponse(raw)
    with request(journal):
        with pytest.raises(CompletionValidationError, match="credential echo"):
            answer(adapter)
    record = sole_call(journal)
    assert record["outcome"] == "returned"
    captured = record["raw_response"]
    texts = [captured["body_text"], base64.b64decode(captured["body_base64"]).decode("utf-8"),
             json.dumps(record, ensure_ascii=False)]
    if "parsed" in captured:
        texts.append(json.dumps(captured["parsed"], ensure_ascii=False))
    for text in texts:
        for _ in range(8):
            assert key not in text
            text = re.sub(r'\\(?:u[0-9a-fA-F]{4}|[\\/"bfnrt])',
                          lambda match: json.loads('"' + match.group(0) + '"'), text)
        assert key not in text
    assert key.encode() not in journal.path.read_bytes()
    assert "REDACTED_CREDENTIAL" in captured["body_text"]
    assert transport.open.call_count == 1 and local.requests == []


def test_http_error_body_is_durable_with_safe_diagnostics_only(setup_adapter, journal):
    adapter, local, client, transport = setup_adapter
    raw = b'{"error":{"message":"synthetic provider failure"}}'
    transport.open.side_effect = HTTPError("https://spiderssense.com/v1/chat/completions", 503,
                                          "synthetic failure", {"Authorization": "secret-header"},
                                          io.BytesIO(raw))
    with pytest.raises(ProviderRequestError, match="HTTP 503"):
        with request(journal):
            answer(adapter)
    record = sole_call(journal)
    assert record["outcome"] == "raised" and record["error_type"] == "ProviderRequestError"
    assert record["raw_response"]["http_status"] == 503
    assert base64.b64decode(record["raw_response"]["body_base64"]) == raw
    assert "secret-header" not in json.dumps(record)
    assert client.preflight()["calls_reserved"] == 1
    assert transport.open.call_count == 1 and local.requests == []
