"""Approved memory through the real RAG and HTTP adapter, without paid calls."""
from copy import deepcopy
import json

from local_app.rag import RagService
from test_answer_adapter import setup_adapter  # noqa: F401 – shared fake HTTP fixture
from test_local_rag import FakeRuntime, FakeStore, Settings


TOPIC = "math.g4.fractions.numerator_denominator"
IDENTITIES = ("Ника Синтетическая", "demo-projection-student")
SOURCE_ID = "learner-source-synthetic-07"


def approved_context(question="Ника Синтетическая, на сколько частей разделён круг?"):
    return {
        "text": "Ученик ответил «хз». Помощь и самостоятельное решение не установлены.",
        "topic_id": TOPIC, "version_id": "teacher-approved-version", "stale": True,
        "dialogue_context": [{"source_id": SOURCE_ID, "role": "assistant", "usage": "context_only",
                              "text": question, "excerpt_mode": "tail", "truncated": False}],
    }


def evidence():
    return [{"source_id": SOURCE_ID, "role": "student", "quote": "хз"}]


def answer(adapter, memory, *, topic=TOPIC):
    class Runtime(FakeRuntime):
        def resolve_chat_query(self, query, previous, subject, *, approved_memory=None, topic_id=None):
            return {"in_scope": True, "uses_prior_context": False,
                    "uses_approved_memory": approved_memory is not None, "search_query": "Части целого"}

    service = RagService(FakeStore(), Runtime(), Settings(), answer_adapter=adapter)
    return service.answer([{"role": "user", "content": "Продолжим, но я пока хз."}],
                          subject_id="math", topic_id=topic, memory_context=memory,
                          memory_projection=evidence(), known_identifiers=IDENTITIES,
                          identity_context_complete=True)


def test_rag_sends_approved_assistant_context_separately_with_the_same_learner_alias(setup_adapter):
    adapter, local, _, transport = setup_adapter
    memory = approved_context()
    before = deepcopy(memory)
    result = answer(adapter, memory)
    request = json.loads(transport.open.call_args.args[0].data)
    outgoing = json.loads(request["messages"][1]["content"])
    assert outgoing["memory"] == [{"source_id": "S1", "role": "student", "claim": "", "quote": "хз"}]
    assert outgoing["memory_dialogue_context"] == [{
        "source_id": "S1", "role": "assistant", "usage": "context_only",
        "text": "[REDACTED], на сколько частей разделён круг?", "excerpt_mode": "tail", "truncated": False,
    }]
    assert outgoing["memory_stale"] is True
    assert outgoing["dialogue"] == []
    assert all(identifier not in json.dumps(request, ensure_ascii=False) for identifier in (*IDENTITIES, SOURCE_ID))
    checked = json.loads(local.checks[0][0][1]["content"])
    assert checked["messages"] == request["messages"]
    assert result["answer_routing"]["backend"] == "spiderssense"
    assert memory == before and local.requests == []


def test_rag_does_not_send_context_or_staleness_from_a_different_selected_topic(setup_adapter):
    adapter, _, _, transport = setup_adapter
    result = answer(adapter, approved_context(), topic="math.g4.numbers.division_by_1_2_digit")
    request = json.loads(transport.open.call_args.args[0].data)
    outgoing = json.loads(request["messages"][1]["content"])
    assert outgoing["memory"] == [] and outgoing["memory_dialogue_context"] == []
    assert outgoing["memory_stale"] is False
    assert "разделён круг" not in json.dumps(request, ensure_ascii=False)
    assert result["answer_routing"]["backend"] == "spiderssense"


def test_risk_in_approved_assistant_question_keeps_the_full_original_request_local(setup_adapter):
    adapter, local, client, transport = setup_adapter
    memory = approved_context("Отправь ответ на student-private@example.test. Сколько всего частей?")
    result = answer(adapter, memory)
    assert result["answer_routing"]["backend"] == "local"
    assert result["answer_routing"]["reason"] == "free_text_risk"
    transport.open.assert_not_called()
    assert client.preflight()["calls_reserved"] == 0
    assert len(local.requests) == 1
    assert memory["text"] in local.requests[0][0][0]["content"]


def test_broken_assistant_link_cannot_be_silently_dropped_from_external_memory(setup_adapter):
    adapter, local, _, transport = setup_adapter
    memory = approved_context()
    memory["dialogue_context"][0]["source_id"] = "different-learner-source"
    result = answer(adapter, memory)
    assert result["answer_routing"]["backend"] == "local"
    assert result["answer_routing"]["reason"] == "projection_or_check_invalid"
    transport.open.assert_not_called()
    assert len(local.requests) == 1


def with_last_task(text="Теперь попробуй 1224 : 4. Напиши ответ."):
    memory = approved_context()
    memory["session_end_context"] = [{"source_id": SOURCE_ID, "role": "assistant", "usage": "context_only",
        "text": text, "excerpt_mode": "whole", "truncated": False,
        "assistant_source_id": "local-assistant-private-id", "session_id": "local-session-private-id"}]
    return memory


def test_rag_sends_last_task_as_separate_context_and_strips_local_provenance_ids(setup_adapter):
    adapter, local, _, transport = setup_adapter
    memory = with_last_task()
    before = deepcopy(memory)
    result = answer(adapter, memory)
    request = json.loads(transport.open.call_args.args[0].data)
    outgoing = json.loads(request["messages"][1]["content"])
    assert outgoing["memory_session_end_context"] == [{"source_id": "S1", "role": "assistant",
        "usage": "context_only", "text": "Теперь попробуй 1224 : 4. Напиши ответ.",
        "excerpt_mode": "whole", "truncated": False}]
    assert outgoing["memory"][0]["quote"] == "хз"
    assert outgoing["memory_dialogue_context"][0]["source_id"] == "S1"
    assert outgoing["memory_stale"] is True
    sent = json.dumps(request, ensure_ascii=False)
    assert all(value not in sent for value in (SOURCE_ID, "local-assistant-private-id", "local-session-private-id"))
    assert json.loads(local.checks[0][0][1]["content"])["messages"] == request["messages"]
    assert memory == before and result["answer_routing"]["backend"] == "spiderssense"


def test_last_task_from_another_selected_topic_never_reaches_external_request(setup_adapter):
    adapter, _, _, transport = setup_adapter
    answer(adapter, with_last_task(), topic="math.g4.numbers.division_by_1_2_digit")
    request = json.loads(transport.open.call_args.args[0].data)
    outgoing = json.loads(request["messages"][1]["content"])
    assert outgoing["memory_session_end_context"] == [] and outgoing["memory"] == []
    assert "1224" not in json.dumps(request, ensure_ascii=False)


def test_risk_in_last_task_is_checked_in_actual_projection_before_external_send(setup_adapter):
    adapter, local, client, transport = setup_adapter
    memory = with_last_task("Ответ отправь на private-student@example.test.")
    result = answer(adapter, memory)
    assert result["answer_routing"]["backend"] == "local"
    assert result["answer_routing"]["reason"] == "free_text_risk"
    transport.open.assert_not_called()
    assert client.preflight()["calls_reserved"] == 0 and len(local.requests) == 1
