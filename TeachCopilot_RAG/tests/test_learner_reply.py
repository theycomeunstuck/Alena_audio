"""A bare agreement must not become a fabricated correct lesson answer."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from local_app.answer_adapter import AnswerAdapter
from local_app.learner_reply import unanswered_question_followup
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_social_chat import CurriculumSession, ExternalProvider, LessonMaterials, LocalSession, Settings


QUESTION = "Если мы разделили круг на 4 части, какое число должно быть внизу?"
HISTORY = [{"role": "user", "content": "Путаюсь в числителе и знаменателе."},
           {"role": "assistant", "content": QUESTION}]


class NoRetrieval:
    def hybrid_search(self, **kwargs):
        raise AssertionError("A request for the missing learner answer needs no retrieval")


@pytest.mark.parametrize("reply", ["ага", "Ага!", " угу... ", "да", "ну вроде", "Кажется, понятно.", "окей"])
@pytest.mark.parametrize("question", [
    QUESTION,
    "Нам нужно разделить 3 на 12. Как ты думаешь, какую цифру нужно записать в разряд десятков?",
    "12 × 1 = 12\n12 × 2 = 24\n12 × 3 = 36\nКакое из этих чисел самое близкое к 36 и не больше его?",
    "Что выведет print('Маяк')?",
])
def test_bare_agreement_after_an_open_question_gets_no_answer_generation(reply, question):
    decision = {"in_scope": True, "uses_prior_context": False,
                "prior_assistant_question_index": 1, "search_query": "Продолжение учебного вопроса"}
    session = LocalSession(json.dumps(decision))
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(NoRetrieval(), runtime, Settings(), AnswerAdapter(runtime, ExternalProvider()))
    history = [HISTORY[0], {"role": "assistant", "content": question}, {"role": "user", "content": reply}]
    before = deepcopy(history)

    result = service.answer(history, subject_id="python" if "print" in question else "math")

    text = result["choices"][0]["message"]["content"]
    assert "Напиши свой ответ" in text and "подсказка" in text
    assert "Правильно" not in text and "Верно" not in text
    assert result["answer_routing"] == {"backend": "deterministic_local", "reason": "brief_agreement_not_answer"}
    assert result["answer_validation"] == {"status": "clarification", "reason": "learner_answer_missing"}
    assert result["citations"] == result["sources"] == []
    assert result["usage"]["total_tokens"] == 0 and "excludes_resolver" in result["usage_scope"]
    assert len(session.requests) == 1 and history == before


def test_guard_also_applies_when_resolver_selects_only_the_learner_question():
    session = LocalSession(json.dumps({"in_scope": True, "uses_prior_context": True,
        "prior_learner_question_index": 0, "search_query": "Дроби"}))
    result = RagService(NoRetrieval(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [*HISTORY, {"role": "user", "content": "ага"}], subject_id="math")
    assert result["answer_validation"]["reason"] == "learner_answer_missing"
    assert len(session.requests) == 1


@pytest.mark.parametrize("reply", ["4?", "ага, 4", "Четыре", "не знаю", "Нужна подсказка", "ага, объясни ещё раз"])
def test_actual_answer_or_request_keeps_the_normal_generation_route(reply):
    session = CurriculumSession(json.dumps({"in_scope": True, "uses_prior_context": False,
        "prior_assistant_question_index": 1, "search_query": "Числитель и знаменатель"}),
        json.dumps({"answer": "Разберём текущий вопрос.", "citations": []}))
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [*HISTORY, {"role": "user", "content": reply}], subject_id="math")
    assert result["choices"][0]["message"]["content"] == "Разберём текущий вопрос."
    assert len(session.requests) == 2 and "answer_validation" not in result


@pytest.mark.parametrize("question", [
    "Продолжим?", "Хочешь подсказку, как найти знаменатель?", "Тогда разберём, как это сделать?",
    "Сколько частей в круге? Хочешь подсказку?", "Равен ли знаменатель четырём?",
    "Любишь рисовать?", "Попробуй вычислить 1224 : 4.",
])
def test_readiness_binary_social_and_statement_contexts_are_not_answer_graded(question):
    assert unanswered_question_followup("ага", question) is None


@pytest.mark.parametrize("social", [False, True])
def test_out_of_scope_and_social_decisions_are_not_overridden_by_an_old_lesson_question(social):
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": ""}
    if social:
        decision["social_turn"] = True
    session = LocalSession(json.dumps(decision), "Хорошо, можем поговорить.")
    result = RagService(NoRetrieval(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [*HISTORY, {"role": "user", "content": "ага"}], subject_id="math")
    assert "answer_validation" not in result and len(session.requests) == (2 if social else 1)


def test_agreement_to_an_offer_of_help_keeps_the_normal_generation_route():
    session = CurriculumSession(json.dumps({"in_scope": True, "uses_prior_context": False,
        "prior_assistant_question_index": 1, "search_query": "Объяснение знаменателя"}),
        json.dumps({"answer": "Начни с подсчёта всех равных частей круга.", "citations": []}))
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [HISTORY[0], {"role": "assistant", "content": "Хочешь подсказку, как найти знаменатель?"},
         {"role": "user", "content": "ага"}], subject_id="math")
    assert result["choices"][0]["message"]["content"] == "Начни с подсчёта всех равных частей круга."
    assert len(session.requests) == 2 and "answer_validation" not in result


def test_invalid_reference_is_rejected_before_the_new_local_guard():
    session = LocalSession(json.dumps({"in_scope": True, "uses_prior_context": True,
        "prior_learner_question_index": 1, "search_query": "Дроби"}))
    result = RagService(NoRetrieval(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [*HISTORY, {"role": "user", "content": "ага"}], subject_id="math")
    assert result["query_resolution"]["reason"] == "resolver_contract_violation"
    assert "answer_validation" not in result and len(session.requests) == 1
