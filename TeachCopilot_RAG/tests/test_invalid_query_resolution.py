"""A complete contradictory resolver decision asks a fixed question once."""
from copy import deepcopy
import json

import pytest
import requests

from local_app.rag import RagService
from local_app.runtime import IncompleteCompletionError, InvalidQueryResolutionError, OllamaRuntime
from tests.test_social_chat import LocalSession, Settings


# Exact complete resolver content from the saved synthetic session smoke.
SMOKE_JSON = ('{"in_scope": true, "uses_prior_context": true, "search_query": "", '
              '"prior_learner_question_index": 0, "prior_assistant_question_index": 1, '
              '"needs_clarification": true}')
SMOKE_DECISION = json.loads(SMOKE_JSON)
PRIOR = [{"role": "user", "content": "Помоги начать деление 1224 : 4 без ответа."},
         {"role": "assistant", "content": "С какого разряда мы начнём деление?"}]


class NoRetrieval:
    def hybrid_search(self, **kwargs):
        raise AssertionError("Invalid resolution must not retrieve materials")


class NoAnswer:
    def answer(self, *args, **kwargs):
        raise AssertionError("Invalid resolution must not generate an answer")


def answer(runtime, *, prior=PRIOR, memory=None):
    return RagService(NoRetrieval(), runtime, Settings(), answer_adapter=NoAnswer()).answer(
        [*prior, {"role": "user", "content": "хз"}], subject_id="math",
        topic_id="math.g4.numbers.division_by_1_2_digit", memory_context=memory)


def test_smoke_complete_json_has_a_typed_validation_error_with_unchanged_decision():
    session = LocalSession(SMOKE_JSON)
    with pytest.raises(InvalidQueryResolutionError) as caught:
        OllamaRuntime(Settings(), session=session).resolve_chat_query("хз", PRIOR, "math")
    assert isinstance(caught.value, IncompleteCompletionError)
    assert str(caught.value) == "Clarification resolver must not request retrieval"
    assert caught.value.raw_response == {"resolver_decision": SMOKE_DECISION}
    assert len(session.requests) == 1


def test_smoke_contract_violation_gets_one_marked_deterministic_question_without_retry():
    session = LocalSession(SMOKE_JSON)
    result = answer(OllamaRuntime(Settings(), session=session))
    assert result["query_resolution"] == {
        "source": "invalid_resolver_decision", "reason": "resolver_contract_violation"}
    assert result["choices"][0]["message"]["content"] == (
        "Хочешь другой пример по этой теме или разберём непонятный шаг?")
    assert result["sources"] == result["citations"] == []
    assert result["_local_diagnostics"] == {
        "type": "InvalidQueryResolutionError", "message": "Clarification resolver must not request retrieval",
        "model_response": {"resolver_decision": SMOKE_DECISION}}
    assert result["usage_scope"] == "deterministic_clarification_excludes_resolver"
    assert len(session.requests) == 1


@pytest.mark.parametrize("decision,prior", [
    ({"in_scope": "yes", "uses_prior_context": False, "search_query": ""}, PRIOR),
    ({"in_scope": True}, PRIOR),
    ({"in_scope": True, "uses_prior_context": False, "search_query": "x", "unknown": True}, PRIOR),
    ({"in_scope": True, "uses_prior_context": True, "search_query": "Деление",
      "prior_learner_question_index": 9}, PRIOR),
    ({"in_scope": True, "uses_prior_context": False, "search_query": "Деление",
      "prior_assistant_question_index": 0}, PRIOR),
    ({"in_scope": True, "uses_prior_context": False, "search_query": "Деление",
      "prior_assistant_turn_index": 1}, PRIOR),
    ({"in_scope": True, "uses_prior_context": False, "search_query": "Деление",
      "uses_approved_memory": True}, PRIOR),
    ({"in_scope": False, "uses_prior_context": False, "search_query": "Деление", "social_turn": True}, PRIOR),
    ({"in_scope": False, "uses_prior_context": True, "search_query": ""}, PRIOR),
    ({"in_scope": True, "uses_prior_context": True, "search_query": ""}, PRIOR),
    ({"in_scope": True, "uses_prior_context": False, "search_query": "x" * 501}, PRIOR),
    ({"in_scope": True, "uses_prior_context": True, "search_query": "Деление"}, []),
])
def test_post_json_contract_errors_remain_distinct_from_incomplete_generation(decision, prior):
    session = LocalSession(json.dumps(decision))
    with pytest.raises(InvalidQueryResolutionError) as caught:
        OllamaRuntime(Settings(), session=session).resolve_chat_query("хз", prior, "math")
    assert caught.value.raw_response == {"resolver_decision": decision}
    assert len(session.requests) == 1


def test_invalid_decision_without_previous_assistant_asks_for_the_actual_exercise():
    session = LocalSession(json.dumps({"in_scope": True, "uses_prior_context": False,
                                      "search_query": "Деление", "uses_approved_memory": True}))
    result = answer(OllamaRuntime(Settings(), session=session), prior=[])
    assert "какое задание" in result["choices"][0]["message"]["content"]
    assert result["query_resolution"]["source"] == "invalid_resolver_decision"
    assert len(session.requests) == 1


def test_valid_model_clarification_keeps_its_original_source_and_has_no_error_diagnostics():
    session = LocalSession(json.dumps({"in_scope": False, "uses_prior_context": False,
                                      "search_query": "", "needs_clarification": True}))
    result = answer(OllamaRuntime(Settings(), session=session))
    assert result["query_resolution"] == {"source": "clarification", "reason": "learning_context_unclear"}
    assert "_local_diagnostics" not in result
    assert len(session.requests) == 1


@pytest.mark.parametrize("decision", [SMOKE_DECISION,
    {"in_scope": True, "uses_prior_context": False, "search_query": "Деление",
     "prior_assistant_turn_index": 99}])
def test_rag_validation_of_an_injected_decision_is_not_converted_to_a_model_fallback(decision):
    class InjectedRuntime:
        def resolve_chat_query(self, *args, **kwargs):
            return deepcopy(decision)

    with pytest.raises(IncompleteCompletionError) as caught:
        answer(InjectedRuntime())
    assert type(caught.value) is IncompleteCompletionError


def test_generic_incomplete_error_from_resolver_is_not_caught_as_invalid_decision():
    error = IncompleteCompletionError("completion stopped", raw_response={"done_reason": "length"})

    class InjectedRuntime:
        def resolve_chat_query(self, *args, **kwargs):
            raise error

    with pytest.raises(IncompleteCompletionError) as caught:
        answer(InjectedRuntime())
    assert caught.value is error


@pytest.mark.parametrize("raw,expected", [
    ({"done_reason": "stop", "message": {"content": '{"in_scope":'}}, "valid structured JSON"),
    ({"done_reason": "length", "message": {"content": SMOKE_JSON}}, "truncated"),
])
def test_malformed_or_truncated_runtime_response_keeps_the_existing_failure(raw, expected):
    class CompletionSession:
        trust_env = True

        def __init__(self):
            self.requests = []

        def post(self, url, **kwargs):
            self.requests.append(kwargs)
            return LocalSession.response(deepcopy(raw))

    session = CompletionSession()
    with pytest.raises(IncompleteCompletionError, match=expected) as caught:
        answer(OllamaRuntime(Settings(), session=session))
    assert type(caught.value) is IncompleteCompletionError
    assert caught.value.raw_response == raw
    assert len(session.requests) == 1


def test_transport_timeout_propagates_without_clarification_or_retry():
    error = requests.Timeout("synthetic timeout")

    class TimeoutSession:
        trust_env = True
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise error

    session = TimeoutSession()
    with pytest.raises(requests.Timeout) as caught:
        answer(OllamaRuntime(Settings(), session=session))
    assert caught.value is error and session.calls == 1


def test_invalid_memory_input_is_not_reclassified_as_a_model_decision_failure():
    session = LocalSession()
    with pytest.raises(ValueError, match="Approved memory text exceeds its context budget"):
        answer(OllamaRuntime(Settings(), session=session), memory={"text": "x" * 4001})
    assert session.requests == []
