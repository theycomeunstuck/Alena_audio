"""Synthetic request-boundary checks; no paid or network calls."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from local_app.answer_adapter import AnswerAdapter, AnswerCapacityError, EducationalProjection
from research_memory.spiderssense import KEY_ENV, ProviderRequestError, SpiderssenseClient


class LocalRuntime:
    def __init__(self):
        self.requests = []
        self.checks = []
        self.decision = {"safe_to_send": True, "uncertain": False}

    def chat(self, messages, response_format=None):
        self.requests.append((messages, response_format))
        return {"model": "synthetic-local", "message": {"role": "assistant", "content": '{"answer":"Локальный ответ","citations":[]}'}}

    def chat_json(self, messages, response_schema=None, max_output_tokens=None):
        self.checks.append((messages, response_schema, max_output_tokens))
        if isinstance(self.decision, Exception):
            raise self.decision
        return self.decision


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit):
        return json.dumps({
            "choices": [{"message": {"content": '{"answer":"Реши похожий пример – проверь сам.","citations":["S1","S2"]}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        }).encode()


@pytest.fixture
def setup_adapter(tmp_path, monkeypatch):
    transport = Mock()
    transport.open.return_value = Response()
    monkeypatch.setenv(KEY_ENV, "synthetic-test-key-123")
    monkeypatch.setattr("research_memory.spiderssense.build_opener", lambda *args: transport)
    client = SpiderssenseClient(
        model="synthetic-test-model", input_usd_per_million="1", output_usd_per_million="2",
        pricing_verified=True, pricing_source="https://spiderssense.com/v1/models",
        ledger_path=tmp_path / "integration.sqlite3", max_cost_usd="0.40", max_calls=20,
        max_output_tokens=128, use_json_schema=True,
    )
    local = LocalRuntime()
    adapter = AnswerAdapter(local, client)
    return adapter, local, client, transport


def projection(**changes):
    base = EducationalProjection(
        topic="Деление", current_message="Меня зовут Ирина Соколова, learner-synthetic-7. Как проверить деление?",
        dialogue=({"role": "user", "content": "Ирина решила пример с подсказкой."},
                  {"role": "assistant", "content": "Проверь умножением."}),
        memory=({"source_id": "learner-synthetic-7/source/private-lesson", "claim": "Была подсказка.",
                 "quote": "Ирина Соколова получила подсказку учителя."},),
        materials=({"source_id": "local/material/private-file", "title": "Деление", "content": "Проверь результат умножением."},),
        identity_context_complete=True,
    )
    return replace(base, **changes)


def local_messages(value):
    return [{"role": "system", "content": "RAW CARD legal_name=Ирина Соколова; tutor_notes=private note; raw_snapshot=secret"},
            *value.dialogue, {"role": "user", "content": value.current_message}]


IDENTITIES = ("Ирина Соколова", "learner-synthetic-7")
SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}},
}}


def test_actual_outbound_request_redacts_current_dialogue_quotes_and_ids(setup_adapter):
    adapter, local, client, transport = setup_adapter
    value = projection()
    schema = json.loads(json.dumps(SCHEMA))
    schema["properties"]["citations"]["items"]["enum"] = [item["source_id"] for item in [*value.memory, *value.materials]]
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES,
                            response_schema=schema)
    request = transport.open.call_args.args[0]
    # Inspect exactly what the real client handed to its HTTP transport.
    outgoing_bytes = request.data.decode("utf-8")
    for private in (*IDENTITIES, "Ирина", "Соколова", "RAW CARD", "legal_name", "tutor_notes", "raw_snapshot", "private note", "private-lesson", "private-file"):
        assert private not in outgoing_bytes
    outgoing = json.loads(outgoing_bytes)
    data = json.loads(outgoing["messages"][1]["content"])
    assert data["current_message"].count("[REDACTED]") == 2
    assert "[REDACTED]" in data["memory"][0]["quote"]
    assert "[REDACTED]" in data["dialogue"][0]["content"]
    assert [data["memory"][0]["source_id"], data["materials"][0]["source_id"]] == ["S1", "S2"]
    assert outgoing["response_format"]["json_schema"]["schema"]["properties"]["citations"]["items"]["enum"] == ["S1", "S2"]
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == outgoing["messages"]
    assert checked["response_schema"] == outgoing["response_format"]["json_schema"]["schema"]
    assert json.loads(result["message"]["content"])["citations"] == [item["source_id"] for item in [*value.memory, *value.materials]]
    assert result["answer_routing"]["backend"] == "spiderssense"
    assert client.preflight()["calls_reserved"] == 1
    assert local.requests == []
    assert "Ирина Соколова" in local_messages(value)[-1]["content"]


def test_confirmed_interests_and_avoided_themes_are_checked_in_actual_outbound_request(setup_adapter):
    adapter, local, _, transport = setup_adapter
    value = projection(interests=("Minecraft", "Шахматы Ирины Соколовой"),
                       avoided_interests=("Футбол",))
    result = adapter.answer(local_messages(value), projection=value,
                            known_identifiers=(*IDENTITIES, "Ирины Соколовой"))
    outgoing = json.loads(transport.open.call_args.args[0].data)
    data = json.loads(outgoing["messages"][1]["content"])
    assert data["interests"] == ["Minecraft", "Шахматы [REDACTED]"]
    assert data["avoided_interests"] == ["Футбол"]
    assert "RAW CARD" not in json.dumps(outgoing, ensure_ascii=False)
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == outgoing["messages"]
    assert result["answer_routing"]["backend"] == "spiderssense"


@pytest.mark.parametrize("unsafe", [
    "Как проверить деление? Напиши test-person@example.test",
    "Как проверить деление? Мой номер +7 (999) 123-45-67",
    "Как проверить деление? https://private.example.test/student/42",
    "Как проверить деление? @private_person",
    "Как проверить деление? Паспорт 1234 567890",
    "Как проверить деление? C:\\Users\\PrivateName\\lesson.txt",
])
def test_unresolved_risk_in_current_message_never_reaches_transport(setup_adapter, unsafe):
    adapter, local, client, transport = setup_adapter
    value = projection(current_message=unsafe)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["backend"] == "local"
    assert len(local.requests) == 1
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


def test_unresolved_risk_in_quoted_source_never_reaches_transport(setup_adapter):
    adapter, local, _, transport = setup_adapter
    value = projection(memory=({"source_id": "s", "quote": "Напиши learner@example.test", "claim": "Проверка"},))
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    transport.open.assert_not_called()
    assert len(local.requests) == 1


def test_schema_is_part_of_privacy_boundary(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = projection()
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES,
                            response_schema={"type": "string", "description": "Contact undisclosed@example.test"})
    assert result["answer_routing"]["reason"] == "free_text_risk"
    transport.open.assert_not_called()


@pytest.mark.parametrize("decision", [
    {"safe_to_send": False, "uncertain": False}, {"safe_to_send": True, "uncertain": True},
    {"safe_to_send": 1, "uncertain": False}, {"safe_to_send": True},
    {"safe_to_send": True, "uncertain": False, "extra": "field"}, None,
    ConnectionError("local checker unavailable"), ValueError("malformed JSON"),
])
def test_unknown_name_or_ambiguous_free_text_uses_local_on_gate_failure(setup_adapter, decision):
    adapter, local, _, transport = setup_adapter
    value = projection(current_message="Помоги разделить. Мой сосед из соседнего класса тоже решал.")
    local.decision = decision
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["backend"] == "local"
    assert len(local.requests) == 1
    transport.open.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"identity_context_complete": False},
    {"memory": ({"source_id": "s", "quote": "ok", "tutor_notes": "raw"},)},
    {"memory": ({"source_id": "s", "quote": "x" * 801},)},
    {"materials": ({"source_id": "s", "title": "ok", "content": "ok", "raw_snapshot": {}},)},
    {"current_message": "hidden\u200bidentity"},
])
def test_incomplete_identity_or_nonallowlisted_fields_fail_closed(setup_adapter, changes):
    adapter, _, _, transport = setup_adapter
    value = projection(**changes)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["backend"] == "local"
    transport.open.assert_not_called()


def test_projection_cannot_omit_actual_current_message_or_prior_dialogue(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = projection()
    for messages in ([{"role": "user", "content": "private@example.test"}],
                     local_messages(value)[-1:]):
        result = adapter.answer(messages, projection=value, known_identifiers=IDENTITIES)
        assert result["answer_routing"]["backend"] == "local"
    transport.open.assert_not_called()


def test_provider_unknown_outcome_is_not_retried_or_silently_fallback(setup_adapter):
    adapter, local, client, transport = setup_adapter
    transport.open.side_effect = URLError("synthetic unknown outcome")
    value = projection()
    with pytest.raises(ProviderRequestError):
        adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert transport.open.call_count == 1
    assert local.requests == []
    assert client.preflight()["calls_reserved"] == 1
    assert float(client.preflight()["charged_or_reserved_usd"]) > 0


def test_external_capacity_refusal_is_explicit_and_never_reserves_money(setup_adapter, monkeypatch):
    adapter, local, client, transport = setup_adapter
    slots = Mock()
    slots.acquire.return_value = False
    monkeypatch.setattr("local_app.answer_adapter._EXTERNAL_SLOTS", slots)
    value = projection()
    with pytest.raises(AnswerCapacityError):
        adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()
    slots.release.assert_not_called()
    assert local.requests == []


def test_short_source_ids_do_not_rewrite_arithmetic_in_excerpts(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = projection(memory=({"source_id": "1", "quote": "1 + 1 = 2"},),
                       materials=({"source_id": "2", "title": "Пример", "content": "2 + 1 = 3"},))
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    outgoing = json.loads(transport.open.call_args.args[0].data)
    data = json.loads(outgoing["messages"][1]["content"])
    assert data["memory"][0] == {"source_id": "S1", "claim": "", "quote": "1 + 1 = 2"}
    assert data["materials"][0]["content"] == "2 + 1 = 3"


def test_schema_instruction_mode_sends_only_checked_aliases(setup_adapter, tmp_path):
    adapter, local, _, transport = setup_adapter
    adapter.external_client = SpiderssenseClient(
        model="synthetic-test-model", input_usd_per_million="1", output_usd_per_million="2",
        pricing_verified=True, pricing_source="https://spiderssense.com/v1/models",
        ledger_path=tmp_path / "json-object.sqlite3", max_cost_usd="0.40", max_calls=20,
        max_output_tokens=128, use_json_schema=False, use_json_object=True,
    )
    value = projection()
    schema = {"type": "string", "enum": [value.memory[0]["source_id"]]}
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES,
                   purpose="next_step", response_schema=schema)
    request = transport.open.call_args.args[0]
    outgoing = json.loads(request.data)
    assert outgoing["response_format"] == {"type": "json_object"}
    assert "S1" in outgoing["messages"][-1]["content"]
    assert value.memory[0]["source_id"] not in request.data.decode()
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["response_schema"] == {"type": "string", "enum": ["S1"]}
    assert "review_required=true" in outgoing["messages"][0]["content"]


def test_local_failure_is_not_retried_by_gate_error_handling(setup_adapter):
    adapter, local, _, transport = setup_adapter
    local.chat = Mock(side_effect=ConnectionError("local runtime unavailable"))
    value = projection(current_message="person@example.test")
    with pytest.raises(ConnectionError):
        adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert local.chat.call_count == 1
    transport.open.assert_not_called()


def test_default_and_disabled_config_do_not_construct_provider(tmp_path, monkeypatch):
    provider = Mock(side_effect=AssertionError("must remain local"))
    monkeypatch.setattr("local_app.answer_adapter.SpiderssenseClient", provider)
    local = LocalRuntime()
    monkeypatch.delenv("TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG", raising=False)
    assert AnswerAdapter.from_env(local).external_client is None
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps({"kind": "local_app_external_answers_v1", "enabled": False, "spiderssense": {}}))
    assert AnswerAdapter.from_config(local, path).external_client is None
    provider.assert_not_called()


@pytest.mark.parametrize("override", [
    {"max_cost_usd": "0.41"}, {"max_cost_usd": "NaN"}, {"max_cost_usd": "unknown"}, {"max_calls": 21},
    {"max_calls": True}, {"max_output_tokens": 1537}, {"ledger_path": None},
])
def test_integration_configuration_rejects_budget_expansion(tmp_path, monkeypatch, override):
    config = {"kind": "local_app_external_answers_v1", "enabled": True, "spiderssense": {
        "model": "synthetic-test-model", "pricing_verified": True,
        "pricing_source": "https://spiderssense.com/v1/models", "input_usd_per_million": "1",
        "output_usd_per_million": "2", "ledger_path": str(tmp_path / "ledger.sqlite3"),
        "max_cost_usd": "0.40", "max_calls": 20, **override,
    }}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises((ValueError, TypeError)):
        AnswerAdapter.from_config(LocalRuntime(), path)


def test_rag_adapter_does_not_route_embedding_or_retrieval_outside_local():
    from local_app.rag import RagService
    from tests.test_local_rag import FakeRuntime, FakeStore, Settings

    class CapturingAdapter:
        def answer(self, messages, **kwargs):
            self.kwargs = kwargs
            self.messages = messages
            return runtime.chat(messages, response_format=kwargs["response_schema"])

    store, runtime, adapter = FakeStore(), FakeRuntime(), CapturingAdapter()
    service = RagService(store, runtime, Settings(), answer_adapter=adapter)
    result = service.answer([{"role": "user", "content": "Что такое отрезок?"}],
                            profile={"legal_name": "Private Name", "tutor_notes": "raw note"}, subject_id="math",
                            known_identifiers=("Private Name",), identity_context_complete=True)
    assert runtime.embedded == ["Что такое отрезок?"]
    assert store.search_query == "Что такое отрезок?"
    assert adapter.kwargs["projection"].current_message == "Что такое отрезок?"
    assert adapter.kwargs["projection"].materials[0] == {
        "source_id": "chunk-1", "title": "", "content": "Отрезок – часть прямой с двумя концами.",
    }
    assert "legal_name" not in json.dumps(adapter.kwargs["projection"].__dict__)
    assert result["citations"][0]["id"] == "chunk-1"
