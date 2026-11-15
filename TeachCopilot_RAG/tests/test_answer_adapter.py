"""Synthetic request-boundary checks; no paid or network calls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import pytest

from local_app.answer_adapter import AnswerAdapter, AnswerCapacityError, EducationalProjection
from local_app.checked_division_context import (
    CHAT_CONTEXT_INSTRUCTIONS, DIVISION_TOPIC, checked_division_context,
)
from local_app.memory_ported.spiderssense import KEY_ENV, ProviderRequestError, SpiderssenseClient


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
    monkeypatch.setattr("local_app.memory_ported.spiderssense.build_opener", lambda *args: transport)
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


def contextual_projection(**changes):
    source_id = "learner-synthetic-7/source/private-lesson"
    return projection(
        memory=({"source_id": source_id, "role": "student", "quote": "ага", "claim": ""},),
        memory_dialogue_context=({
            "source_id": source_id, "role": "assistant", "usage": "context_only",
            "text": "Ирина Соколова, мы выделили 4 части круга. Всего частей 10 – какое число внизу?",
            "excerpt_mode": "tail", "truncated": False,
        },), **changes,
    )


def test_actual_outbound_request_keeps_assistant_memory_context_separate_and_redacted(setup_adapter):
    adapter, local, _, transport = setup_adapter
    value = contextual_projection(memory_stale=True)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    outgoing_bytes = transport.open.call_args.args[0].data.decode("utf-8")
    outgoing = json.loads(outgoing_bytes)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["memory"] == [{"source_id": "S1", "role": "student", "claim": "", "quote": "ага"}]
    assert payload["memory_dialogue_context"] == [{
        "source_id": "S1", "role": "assistant", "usage": "context_only",
        "text": "[REDACTED], мы выделили 4 части круга. Всего частей 10 – какое число внизу?",
        "excerpt_mode": "tail", "truncated": False,
    }]
    assert payload["memory_stale"] is True
    for private in (*IDENTITIES, "Ирина", "Соколова", value.memory[0]["source_id"], "private-lesson"):
        assert private not in outgoing_bytes
    assert "какое число внизу" not in payload["memory"][0]["quote"]
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == outgoing["messages"]
    system = outgoing["messages"][0]["content"]
    assert "контекст вопроса" in system and "не наблюдение об ученике" in system
    assert "не выдавай её за актуальную" in system
    assert "один уточняющий вопрос" in system and "Согласие ребёнка не означает" in system
    assert "может быть ошибочным" in system
    assert result["answer_routing"]["policy"] == "educational-projection-v6"


@pytest.mark.parametrize("unsafe", ["Напиши private-person@example.test", "hidden\u200bidentity"])
def test_unresolved_risk_in_assistant_memory_context_never_reaches_transport(setup_adapter, unsafe):
    adapter, local, client, transport = setup_adapter
    value = contextual_projection()
    value = replace(value, memory_dialogue_context=({**value.memory_dialogue_context[0], "text": unsafe},))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["backend"] == "local"
    assert len(local.requests) == 1
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


def test_free_text_gate_reads_private_story_in_assistant_context_before_any_external_call(setup_adapter):
    adapter, local, client, transport = setup_adapter
    value = contextual_projection()
    text = "Ты пропустила урок, потому что у мамы неприятности со здоровьем. Продолжим?"
    value = replace(value, memory_dialogue_context=({**value.memory_dialogue_context[0], "text": text},))
    local.decision = {"safe_to_send": False, "uncertain": True}
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    checked = json.loads(local.checks[0][0][1]["content"])
    assert json.loads(checked["messages"][1]["content"])["memory_dialogue_context"][0]["text"] == text
    assert result["answer_routing"]["reason"] == "privacy_uncertain"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("context_changes", [
    {"source_id": "not-selected-foreign-source"}, {"role": "student"}, {"usage": "evidence"},
    {"excerpt_mode": "full"}, {"truncated": "false"}, {"truncated": 0},
    {"text": ""}, {"text": "x" * 601}, {"text": "\ufb03" * 201},
    {"raw_teacher_note": "not allowlisted"},
])
def test_invalid_assistant_context_provenance_or_bounds_fails_closed(setup_adapter, context_changes):
    adapter, _, client, transport = setup_adapter
    value = contextual_projection()
    value = replace(value, memory_dialogue_context=({**value.memory_dialogue_context[0], **context_changes},))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("memory_role", [None, "teacher", "assistant"])
def test_assistant_context_requires_matching_explicit_student_evidence_role(setup_adapter, memory_role):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection()
    memory = dict(value.memory[0])
    if memory_role is None:
        del memory["role"]
    else:
        memory["role"] = memory_role
    value = replace(value, memory=(memory,))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


@pytest.mark.parametrize("duplicate_kind", ["context", "source"])
def test_duplicate_source_or_context_is_rejected_without_repair(setup_adapter, duplicate_kind):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection()
    value = (replace(value, memory=tuple(value.memory) * 2) if duplicate_kind == "source" else
             replace(value, memory_dialogue_context=tuple(value.memory_dialogue_context) * 2))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


@pytest.mark.parametrize("stale", [None, "false", 0, 1])
def test_memory_staleness_must_be_boolean(setup_adapter, stale):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection(memory_stale=stale)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


def test_legacy_projection_remains_without_context_or_invented_source_role(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = projection()
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["memory_dialogue_context"] == []
    assert payload["memory_session_end_context"] == []
    assert payload["checked_division_context"] is None
    assert CHAT_CONTEXT_INSTRUCTIONS not in outgoing["messages"][0]["content"]
    assert payload["memory_stale"] is False
    assert "role" not in payload["memory"][0]


def checked_projection():
    return projection(topic=DIVISION_TOPIC, current_message="хз", dialogue=(), memory=(), materials=(),
                      checked_division_context=checked_division_context("1224 : 4", "math", DIVISION_TOPIC))


@pytest.mark.parametrize("ground", ["current", "dialogue", "memory", "preceding", "closing"])
def test_checked_first_group_is_grounded_in_actual_outbound_text_without_full_answer(setup_adapter, ground):
    adapter, local, _, transport = setup_adapter
    value = checked_projection()
    task = "Ирина Соколова, подскажи первый шаг: 1224 : 4."
    source_id = "learner-synthetic-7/source/current-task"
    if ground == "current":
        value = replace(value, current_message=task)
    elif ground == "dialogue":
        value = replace(value, dialogue=({"role": "user", "content": task},
                                         {"role": "assistant", "content": "С какого шага начнём?"}))
    elif ground == "memory":
        value = replace(value, memory=({"source_id": source_id, "role": "teacher", "quote": task},))
    else:
        context = {"source_id": source_id, "role": "assistant", "usage": "context_only", "text": task,
                   "excerpt_mode": "tail" if ground == "preceding" else "whole", "truncated": False}
        value = replace(value, memory=({"source_id": source_id, "role": "student", "quote": "хз"},),
                        **{"memory_dialogue_context" if ground == "preceding" else "memory_session_end_context": (context,)})
    before = deepcopy(value.checked_division_context)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    outgoing_bytes = transport.open.call_args.args[0].data.decode("utf-8")
    outgoing = json.loads(outgoing_bytes)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["checked_division_context"] == {
        "version": "checked-division-first-group-v1", "dividend": 1224, "divisor": 4,
        "first_group": {"digits": "12", "place_value": 100, "place_name": "сотен", "amount": 12,
                        "real_value": 1200},
        "complete_answer_included": False, "student_work_assessed": False,
    }
    checked_json = json.dumps(payload["checked_division_context"], ensure_ascii=False)
    assert "quotient" not in checked_json and "306" not in checked_json
    assert "Ирина" not in outgoing_bytes and "Соколова" not in outgoing_bytes
    assert source_id not in outgoing_bytes and "[REDACTED]" in outgoing_bytes
    assert CHAT_CONTEXT_INSTRUCTIONS in outgoing["messages"][0]["content"]
    checked_request = json.loads(local.checks[0][0][1]["content"])
    assert checked_request["messages"] == outgoing["messages"]
    assert value.checked_division_context == before
    assert result["answer_routing"]["policy"] == "educational-projection-v6"
    assert result["answer_routing"]["backend"] == "spiderssense"


def test_checked_context_metadata_is_not_corrupted_by_identifier_collisions(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = replace(checked_projection(), current_message="Подскажи 1224 : 4.")
    adapter.answer(local_messages(value), projection=value, known_identifiers=(
        *IDENTITIES, "checked_division_context", "first_group", "сотен", "checked-division-first-group-v1"))
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["checked_division_context"] == value.checked_division_context


@pytest.mark.parametrize("changes", [
    {"quotient": 306}, {"source_id": "private-source"}, {"version": "private-person@example.test"},
    {"dividend": 2842}, {"divisor": True}, {"complete_answer_included": True},
    {"complete_answer_included": "false"}, {"student_work_assessed": True},
    {"first_group": {"quotient": 306}},
    {"first_group": {"digits": "12", "place_value": 100, "place_name": "сотен", "amount": 1, "real_value": 1200}},
    {"first_group": {"digits": "12", "place_value": 100, "place_name": "private-person@example.test", "amount": 12, "real_value": 1200}},
])
def test_invalid_or_full_answer_checked_context_falls_back_before_transport(setup_adapter, changes):
    adapter, local, client, transport = setup_adapter
    value = checked_projection()
    value = replace(value, current_message="Подскажи 1224 : 4.",
                    checked_division_context={**value.checked_division_context, **changes})
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    assert len(local.requests) == 1 and client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("hidden_ground", ["missing", "materials", "claim", "topic", "schema", "system", "split"])
def test_checked_operands_cannot_be_grounded_in_nonpublic_fields_or_separate_fragments(setup_adapter, hidden_ground):
    adapter, _, client, transport = setup_adapter
    value = checked_projection()
    schema = None
    if hidden_ground == "materials":
        value = replace(value, materials=({"source_id": "material-one", "title": "Деление", "content": "1224 : 4"},))
    elif hidden_ground == "claim":
        value = replace(value, memory=({"source_id": "source-one", "role": "teacher", "claim": "1224 : 4", "quote": "хз"},))
    elif hidden_ground == "topic":
        value = replace(value, topic="1224 : 4")
    elif hidden_ground == "schema":
        schema = {"type": "string", "description": "1224 : 4"}
    elif hidden_ground == "split":
        value = replace(value, current_message="1224", dialogue=({"role": "user", "content": "4"},))
    messages = local_messages(value)
    if hidden_ground == "system":
        messages[0]["content"] += " Model search rewrite: 1224 : 4"
    result = adapter.answer(messages, projection=value, known_identifiers=IDENTITIES, response_schema=schema)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("current", ["2842 : 14", "4 : 1224", "1224 : 4 + 5", "1224.0 : 4", "-1224 : 4"])
def test_checked_context_requires_the_same_complete_literal_pair(setup_adapter, current):
    adapter, _, _, transport = setup_adapter
    value = replace(checked_projection(), current_message=current)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


def test_checked_context_is_grounded_after_redaction_not_before_it(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = replace(checked_projection(), current_message="1224 : 4")
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=(*IDENTITIES, "1224"))
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


def test_redaction_placeholder_does_not_assemble_a_literal_operation(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = replace(checked_projection(), current_message="1224 private-name : 4")
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=(*IDENTITIES, "private-name"))
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


@pytest.mark.parametrize("whole_ground", [None, "current", "closing"])
def test_truncated_assistant_operand_is_not_checked_ground_but_whole_sources_remain_usable(setup_adapter, whole_ground):
    adapter, _, _, transport = setup_adapter
    source_id = "learner-synthetic-7/source/truncated-task"
    # The stored tail could be cut from "1224 : 4"; its first visible digit is
    # insufficient to claim the literal task was actually "224 : 4".
    tail = {"source_id": source_id, "role": "assistant", "usage": "context_only",
            "text": "224 : 4", "excerpt_mode": "tail", "truncated": True}
    value = replace(checked_projection(), memory=({"source_id": source_id, "role": "student", "quote": "хз"},),
                    memory_dialogue_context=(tail,),
                    checked_division_context=checked_division_context("224 : 4", "math", DIVISION_TOPIC))
    if whole_ground == "current":
        value = replace(value, current_message="Подскажи 224 : 4.")
    elif whole_ground == "closing":
        value = replace(value, memory_session_end_context=({**tail, "excerpt_mode": "whole", "truncated": False},))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    if whole_ground is None:
        assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
        transport.open.assert_not_called()
    else:
        outgoing = json.loads(transport.open.call_args.args[0].data)
        payload = json.loads(outgoing["messages"][1]["content"])
        assert payload["checked_division_context"]["dividend"] == 224
        # The truncated excerpt stays available as conversation context; it
        # simply cannot justify the independently verified arithmetic object.
        assert payload["memory_dialogue_context"][0]["truncated"] is True
        assert result["answer_routing"]["backend"] == "spiderssense"


@pytest.mark.parametrize("topic,allowed", [(DIVISION_TOPIC, True), ("math.g4.fractions", False)])
def test_slash_ground_requires_exact_selected_division_topic(setup_adapter, topic, allowed):
    adapter, _, _, transport = setup_adapter
    value = replace(checked_projection(), current_message="1224 / 4", topic=topic)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    if allowed:
        assert result["answer_routing"]["backend"] == "spiderssense"
    else:
        assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
        transport.open.assert_not_called()


def test_checked_context_does_not_bypass_actual_body_privacy_checks(setup_adapter):
    adapter, local, client, transport = setup_adapter
    value = replace(checked_projection(), current_message="1224 : 4. Отправь на private@example.test")
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "free_text_risk"
    assert client.preflight()["calls_reserved"] == 0 and len(local.requests) == 1
    transport.open.assert_not_called()


def test_checked_context_is_in_whole_request_seen_by_conservative_local_gate(setup_adapter):
    adapter, local, client, transport = setup_adapter
    value = replace(checked_projection(), current_message="1224 : 4. Потом расскажу о неприятностях в семье.")
    local.decision = {"safe_to_send": False, "uncertain": True}
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    checked = json.loads(local.checks[0][0][1]["content"])
    assert json.loads(checked["messages"][1]["content"])["checked_division_context"] == value.checked_division_context
    assert result["answer_routing"]["reason"] == "privacy_uncertain"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


def test_shared_chat_division_instructions_do_not_change_next_step_prompt(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = replace(checked_projection(), current_message="1224 : 4")
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES, purpose="next_step")
    outgoing = json.loads(transport.open.call_args.args[0].data)
    assert CHAT_CONTEXT_INSTRUCTIONS not in outgoing["messages"][0]["content"]


def session_end_projection():
    value = contextual_projection(memory_stale=True)
    return replace(value, memory_session_end_context=({
        "source_id": value.memory[0]["source_id"], "role": "assistant", "usage": "context_only",
        "text": "Ирина Соколова, знаменатель – 4. Какое число поставишь снизу?",
        "excerpt_mode": "whole", "truncated": False,
    },))


def test_actual_request_keeps_session_end_context_separate_from_evidence_and_previous_question(setup_adapter):
    adapter, local, _, transport = setup_adapter
    value = session_end_projection()
    result = adapter.answer(local_messages(value), projection=value,
                            known_identifiers=(*IDENTITIES, "assistant", "context_only", "whole",
                                               "source_id", "S1", "memory_session_end_context"))
    outgoing_bytes = transport.open.call_args.args[0].data.decode("utf-8")
    outgoing = json.loads(outgoing_bytes)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["memory"] == [{"source_id": "S1", "role": "student", "claim": "", "quote": "ага"}]
    assert payload["memory_session_end_context"] == [{
        "source_id": "S1", "role": "assistant", "usage": "context_only",
        "text": "[REDACTED], знаменатель – 4. Какое число поставишь снизу?",
        "excerpt_mode": "whole", "truncated": False,
    }]
    assert payload["memory_dialogue_context"][0]["source_id"] == "S1"
    assert payload["memory_dialogue_context"][0]["excerpt_mode"] == "tail"
    assert "Всего частей 10" in payload["memory_dialogue_context"][0]["text"]
    assert payload["memory_stale"] is True
    assert "assistant_source_id" not in outgoing_bytes and "session_id" not in outgoing_bytes
    for private in (*IDENTITIES, "Ирина", "Соколова", value.memory[0]["source_id"]):
        assert private not in outgoing_bytes
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == outgoing["messages"]
    system = outgoing["messages"][0]["content"]
    assert "не доказательство действия или освоения ученика" in system
    assert "не установленная незавершённая работа" in system
    assert "не додумывай его ответ" in system
    assert "Последняя реплика ассистента тоже может быть ошибочной" in system
    assert result["answer_routing"]["policy"] == "educational-projection-v6"


@pytest.mark.parametrize("changes", [
    {"source_id": "not-selected-foreign-source"}, {"source_id": None}, {"role": "student"},
    {"role": "teacher"}, {"usage": "evidence"}, {"excerpt_mode": "tail"},
    {"truncated": True}, {"truncated": "false"}, {"truncated": 0},
    {"text": ""}, {"text": "  "}, {"text": "x" * 1201}, {"text": "\ufb03" * 401},
    {"assistant_source_id": "internal-assistant-id"}, {"session_id": "internal-session-id"},
])
def test_invalid_session_end_context_never_reaches_transport(setup_adapter, changes):
    adapter, _, client, transport = setup_adapter
    value = session_end_projection()
    value = replace(value, memory_session_end_context=({**value.memory_session_end_context[0], **changes},))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("memory_role", [None, "teacher"])
def test_session_end_context_requires_explicit_student_evidence_role(setup_adapter, memory_role):
    adapter, _, _, transport = setup_adapter
    value = session_end_projection()
    memory = dict(value.memory[0])
    if memory_role is None:
        del memory["role"]
    else:
        memory["role"] = memory_role
    value = replace(value, memory=(memory,), memory_dialogue_context=())
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


def test_duplicate_session_end_context_is_rejected_without_deduplication(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = session_end_projection()
    value = replace(value, memory_session_end_context=tuple(value.memory_session_end_context) * 2)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


@pytest.mark.parametrize("unsafe", ["Напиши private-person@example.test", "hidden\u200bidentity"])
def test_private_text_in_session_end_context_never_reaches_transport(setup_adapter, unsafe):
    adapter, local, client, transport = setup_adapter
    value = session_end_projection()
    value = replace(value, memory_session_end_context=({**value.memory_session_end_context[0], "text": unsafe},))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["backend"] == "local" and len(local.requests) == 1
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


def test_privacy_gate_checks_whole_session_end_context_before_external_generation(setup_adapter):
    adapter, local, client, transport = setup_adapter
    value = session_end_projection()
    text = "Ты пропустила урок, потому что у мамы неприятности со здоровьем. Продолжим?"
    value = replace(value, memory_session_end_context=({**value.memory_session_end_context[0], "text": text},))
    local.decision = {"safe_to_send": False, "uncertain": True}
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    checked = json.loads(local.checks[0][0][1]["content"])
    assert json.loads(checked["messages"][1]["content"])["memory_session_end_context"][0]["text"] == text
    assert result["answer_routing"]["reason"] == "privacy_uncertain"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("invalid_collection", [None, "", {}])
def test_session_end_context_requires_a_sequence_of_rows(setup_adapter, invalid_collection):
    adapter, _, _, transport = setup_adapter
    value = replace(session_end_projection(), memory_session_end_context=invalid_collection)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


def test_session_end_context_whole_text_limit_preserves_all_1200_characters(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = session_end_projection()
    text = "А" * 1181 + " Какое число снизу?"
    assert len(text) == 1200
    value = replace(value, memory_session_end_context=({**value.memory_session_end_context[0], "text": text},))
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["memory_session_end_context"][0]["text"] == text


@pytest.mark.parametrize("count", [20, 21])
def test_session_end_context_collection_is_bounded(setup_adapter, count):
    adapter, _, _, transport = setup_adapter
    value = session_end_projection()
    value = replace(value, memory_dialogue_context=(),
        memory=tuple({"source_id": f"source-{i}", "role": "student", "quote": "хз"} for i in range(20)),
        memory_session_end_context=tuple({**value.memory_session_end_context[0], "source_id": f"source-{i}",
                                         "text": "Сначала посчитай все части."} for i in range(count)))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    if count == 20:
        outgoing = json.loads(transport.open.call_args.args[0].data)
        payload = json.loads(outgoing["messages"][1]["content"])
        assert len(payload["memory_session_end_context"]) == 20
        assert result["answer_routing"]["backend"] == "spiderssense"
    else:
        assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
        transport.open.assert_not_called()


def correction_projection():
    original = "learner-synthetic-7/source/original-record"
    correction = "learner-synthetic-7/source/source-correction"
    return projection(memory=(
        {"source_id": original, "role": "teacher",
         "quote": "Ирина Соколова пропустила ноль в ответе 303."},
        {"source_id": correction, "role": "teacher", "corrects_source_id": original,
         "quote": "Исправление записи: Ирина Соколова записала 303, ноль был. Условия помощи неизвестны."},
    ), memory_stale=True)


@pytest.mark.parametrize("original_role", ["student", "teacher"])
def test_actual_request_preserves_correction_pair_and_redacts_both_quotes(setup_adapter, original_role):
    adapter, local, _, transport = setup_adapter
    value = correction_projection()
    value = replace(value, memory=({**value.memory[0], "role": original_role}, value.memory[1]))
    schema = {"type": "string", "enum": [item["source_id"] for item in value.memory]}
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES,
                            response_schema=schema)
    outgoing_bytes = transport.open.call_args.args[0].data.decode("utf-8")
    outgoing = json.loads(outgoing_bytes)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["memory"] == [
        {"source_id": "S1", "role": original_role, "claim": "",
         "quote": "[REDACTED] пропустила ноль в ответе 303."},
        {"source_id": "S2", "role": "teacher", "claim": "", "corrects_source_id": "S1",
         "quote": "Исправление записи: [REDACTED] записала 303, ноль был. Условия помощи неизвестны."},
    ]
    assert payload["memory_stale"] is True and payload["memory_dialogue_context"] == []
    assert outgoing["response_format"]["json_schema"]["schema"]["enum"] == ["S1", "S2"]
    assert "а не новое действие ученика" in outgoing["messages"][0]["content"]
    assert "отменённый тезис не используй" in outgoing["messages"][0]["content"]
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == outgoing["messages"]
    assert checked["response_schema"] == outgoing["response_format"]["json_schema"]["schema"]
    for private in (*IDENTITIES, "Ирина", "Соколова", *(item["source_id"] for item in value.memory)):
        assert private not in outgoing_bytes
    assert result["answer_routing"]["policy"] == "educational-projection-v6"


@pytest.mark.parametrize("bad_role", ["student", "assistant", None])
def test_source_correction_requires_explicit_teacher_role(setup_adapter, bad_role):
    adapter, _, _, transport = setup_adapter
    value = correction_projection()
    correction = dict(value.memory[1])
    if bad_role is None:
        del correction["role"]
    else:
        correction["role"] = bad_role
    value = replace(value, memory=(value.memory[0], correction))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


@pytest.mark.parametrize("bad_target", [None, 1, "", "not-selected-foreign-source", "x" * 513])
def test_source_correction_rejects_missing_or_invalid_target(setup_adapter, bad_target):
    adapter, _, client, transport = setup_adapter
    value = correction_projection()
    value = replace(value, memory=(value.memory[0], {**value.memory[1], "corrects_source_id": bad_target}))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    assert client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


@pytest.mark.parametrize("invalid_pair", ["unpaired", "self", "forward", "cycle", "legacy_target_role", "duplicate"])
def test_source_correction_rejects_unpaired_or_nonchronological_records_without_repair(setup_adapter, invalid_pair):
    adapter, _, _, transport = setup_adapter
    value = correction_projection()
    original, correction = (dict(item) for item in value.memory)
    if invalid_pair == "unpaired":
        memory = (correction,)
    elif invalid_pair == "self":
        correction["corrects_source_id"] = correction["source_id"]
        memory = (original, correction)
    elif invalid_pair == "forward":
        memory = (correction, original)
    elif invalid_pair == "cycle":
        original["corrects_source_id"] = correction["source_id"]
        memory = (original, correction)
    elif invalid_pair == "legacy_target_role":
        del original["role"]
        memory = (original, correction)
    else:
        memory = (original, correction, correction)
    value = replace(value, memory=memory)
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()


@pytest.mark.parametrize("unsafe_index", [0, 1])
def test_private_text_in_either_correction_pair_quote_never_reaches_transport(setup_adapter, unsafe_index):
    adapter, local, client, transport = setup_adapter
    value = correction_projection()
    memory = [dict(item) for item in value.memory]
    memory[unsafe_index]["quote"] += " Напиши private-person@example.test"
    value = replace(value, memory=tuple(memory))
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    assert result["answer_routing"]["reason"] == "free_text_risk"
    assert len(local.requests) == 1 and client.preflight()["calls_reserved"] == 0
    transport.open.assert_not_called()


def test_correction_metadata_alias_survives_collision_with_a_known_identity(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = correction_projection()
    adapter.answer(local_messages(value), projection=value,
                   known_identifiers=(*IDENTITIES, "teacher", "corrects_source_id", "S1"))
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["memory"][1]["role"] == "teacher"
    assert payload["memory"][1]["corrects_source_id"] == payload["memory"][0]["source_id"] == "S1"


def test_actual_request_keeps_context_linked_to_correct_source_among_multiple_roles(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection()
    value = replace(value, memory=(
        {"source_id": "private-teacher-source", "role": "teacher", "quote": "Самостоятельной проверки не было."},
        *value.memory,
        {"source_id": "private-second-student-source", "role": "student", "quote": "хз"},
    ), memory_dialogue_context=(
        value.memory_dialogue_context[0],
        {**value.memory_dialogue_context[0], "source_id": "private-second-student-source",
         "text": "Как проверить ответ?", "truncated": True},
    ))
    adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert [(item["source_id"], item["role"]) for item in payload["memory"]] == [
        ("S1", "teacher"), ("S2", "student"), ("S3", "student")]
    assert [(item["source_id"], item["truncated"]) for item in payload["memory_dialogue_context"]] == [
        ("S2", False), ("S3", True)]
    assert all(item["role"] == "assistant" and item["usage"] == "context_only"
               for item in payload["memory_dialogue_context"])
    assert payload["materials"][0]["source_id"] == "S4"


def test_identity_collisions_redact_prose_without_corrupting_outbound_protocol(setup_adapter):
    adapter, local, _, transport = setup_adapter
    value = contextual_projection()
    # These are actual identities for redaction purposes, even when the same
    # strings also happen to be trusted protocol keys, roles or source aliases.
    collisions = ("student", "teacher", "assistant", "context_only", "source_id", "tail", "S1", "string")
    value = replace(value,
        current_message="student teacher assistant context_only source_id tail S1 string",
        memory=(value.memory[0], {"source_id": "teacher-source-private", "role": "teacher",
                                 "quote": "teacher source_id"}),
    )
    schema = {"type": "object", "properties": {
        "source_id": {"type": "string", "description": "student teacher source_id",
                      "enum": [value.memory[0]["source_id"], "teacher-source-private"]},
        "answer": {"type": "string", "examples": ["assistant context_only"]},
    }, "required": ["source_id", "answer"]}
    result = adapter.answer(local_messages(value), projection=value,
                            known_identifiers=(*IDENTITIES, *collisions), response_schema=schema)
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    actual_schema = outgoing["response_format"]["json_schema"]["schema"]
    assert payload["current_message"] == " ".join(["[REDACTED]"] * len(collisions))
    assert [item["role"] for item in payload["memory"]] == ["student", "teacher"]
    assert payload["memory"][0]["source_id"] == "S1"
    assert payload["memory_dialogue_context"][0]["source_id"] == "S1"
    assert payload["memory_dialogue_context"][0]["role"] == "assistant"
    assert payload["memory_dialogue_context"][0]["usage"] == "context_only"
    assert payload["memory_dialogue_context"][0]["excerpt_mode"] == "tail"
    assert payload["dialogue"][0]["role"] == "user" and payload["dialogue"][1]["role"] == "assistant"
    assert actual_schema["required"] == ["source_id", "answer"]
    assert actual_schema["properties"]["source_id"] == {
        "type": "string", "description": "[REDACTED] [REDACTED] [REDACTED]", "enum": ["S1", "S2"]}
    assert actual_schema["properties"]["answer"]["examples"] == ["[REDACTED] [REDACTED]"]
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == outgoing["messages"] and checked["response_schema"] == actual_schema
    assert result["answer_routing"]["backend"] == "spiderssense"


def test_conservative_id_fragment_redaction_keeps_protocol_roles_intact(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection(current_message="The student can try one example.")
    adapter.answer(local_messages(value), projection=value,
                   known_identifiers=(*IDENTITIES, "demo-projection-student"))
    outgoing = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(outgoing["messages"][1]["content"])
    assert payload["current_message"] == "The [REDACTED] can try one example."
    assert payload["memory"][0]["role"] == "student"


def test_schema_property_names_that_match_metadata_keywords_remain_structural(setup_adapter):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection()
    names = ["description", "title", "enum", "const"]
    schema = {"type": "object", "properties": {
        name: {"type": "string", "description": "student"} for name in names
    }, "required": names, "dependentRequired": {"description": ["title"]}}
    adapter.answer(local_messages(value), projection=value,
                   known_identifiers=(*IDENTITIES, *names, "student", "string"), response_schema=schema)
    outgoing = json.loads(transport.open.call_args.args[0].data)
    actual = outgoing["response_format"]["json_schema"]["schema"]
    assert list(actual["properties"]) == names and actual["required"] == names
    assert actual["dependentRequired"] == {"description": ["title"]}
    assert all(item == {"type": "string", "description": "[REDACTED]"}
               for item in actual["properties"].values())


@pytest.mark.parametrize("count", [20, 21])
def test_assistant_context_collection_has_explicit_count_limit(setup_adapter, count):
    adapter, _, _, transport = setup_adapter
    value = contextual_projection()
    value = replace(value,
        memory=tuple({"source_id": f"source-{i}", "role": "student", "quote": "хз"}
                     for i in range(20)),
        memory_dialogue_context=tuple({**value.memory_dialogue_context[0], "source_id": f"source-{i}",
                                       "text": "Как проверить ответ?"}
                                      for i in range(count)),
    )
    result = adapter.answer(local_messages(value), projection=value, known_identifiers=IDENTITIES)
    if count == 20:
        outgoing = json.loads(transport.open.call_args.args[0].data)
        payload = json.loads(outgoing["messages"][1]["content"])
        assert len(payload["memory_dialogue_context"]) == 20
        assert result["answer_routing"]["backend"] == "spiderssense"
    else:
        assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
        transport.open.assert_not_called()


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
