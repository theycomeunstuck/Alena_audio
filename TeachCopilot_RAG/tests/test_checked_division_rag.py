"""Only actual, selected dialogue sources may ground the arithmetic scaffold."""
from copy import deepcopy
import json

import pytest

from local_app.rag import RagService
from tests.test_local_rag import FakeRuntime, FakeStore, Settings
from tests.test_answer_adapter import setup_adapter  # noqa: F401


TOPIC = 'math.g4.numbers.division_by_1_2_digit'
TASK = 'Попробуй вычислить 1224 : 4. Напиши ответ.'


class Runtime(FakeRuntime):
    def __init__(self, decision):
        super().__init__()
        self.decision = decision

    def resolve_chat_query(self, *args, **kwargs):
        return deepcopy(self.decision)


class Adapter:
    def __init__(self):
        self.calls = []

    def answer(self, messages, *, projection, **kwargs):
        self.calls.append((deepcopy(messages), projection))
        return {'message': {'content': json.dumps({'answer': 'Сколько раз 4 помещается в 12?', 'citations': []})}}


def run(messages, *, decision=None, memory=None, topic=TOPIC):
    adapter = Adapter()
    runtime = Runtime(decision or {'in_scope': True, 'uses_prior_context': False,
                                  'search_query': 'Переписанный запрос: 1824 : 6'})
    result = RagService(FakeStore(), runtime, Settings(), answer_adapter=adapter).answer(
        messages, subject_id='math', topic_id=topic, memory_context=memory)
    return result, adapter


def test_current_literal_has_priority_over_numbers_in_model_search_rewrite():
    _, adapter = run([{'role': 'user', 'content': '1224 : 4. Не говори ответ, с чего начать?'}])
    messages, projection = adapter.calls[0]
    checked = projection.checked_division_context
    assert checked['dividend'] == 1224 and checked['divisor'] == 4
    assert checked['first_group']['amount'] == 12 and checked['first_group']['place_value'] == 100
    assert checked['first_group']['real_value'] == 1200
    assert 'checked_division_context' in messages[0]['content']
    assert 'quotient' not in json.dumps(checked) and '306' not in json.dumps(checked)
    assert checked['student_work_assessed'] is False


def test_immediate_selected_assistant_instruction_is_context_not_evidence_of_solution():
    _, adapter = run([{'role': 'user', 'content': 'Дай пример.'},
                      {'role': 'assistant', 'content': TASK}, {'role': 'user', 'content': 'хз'}],
        decision={'in_scope': True, 'uses_prior_context': False, 'prior_assistant_turn_index': 1,
                  'search_query': 'Деление'})
    checked = adapter.calls[0][1].checked_division_context
    assert checked['dividend'] == 1224 and checked['complete_answer_included'] is False


def test_unselected_old_exercise_and_model_rewrite_cannot_supply_checked_operands():
    _, adapter = run([{'role': 'user', 'content': TASK},
                      {'role': 'assistant', 'content': 'О чём поговорим?'},
                      {'role': 'user', 'content': 'ага'}])
    assert adapter.calls[0][1].checked_division_context is None


def remembered():
    return {'version_id': 'synthetic-approved', 'topic_id': TOPIC, 'text': 'Private prose: 7777 : 7.',
        'sources': [{'source_id': 's1', 'role': 'student', 'text': 'Начали 3636 : 12. Помощь была.'}],
        'session_end_context': [], 'stale': False}


def test_selected_memory_uses_literal_sources_and_never_parses_arbitrary_memory_prose():
    _, adapter = run([{'role': 'user', 'content': 'Где остановились?'}], memory=remembered(),
        decision={'in_scope': True, 'uses_prior_context': False, 'uses_approved_memory': True,
                  'search_query': 'Продолжить деление'})
    checked = adapter.calls[0][1].checked_division_context
    assert (checked['dividend'], checked['divisor']) == (3636, 12)
    assert checked['first_group']['real_value'] == 3600


@pytest.mark.parametrize('topic', [TOPIC, 'math.g4.fractions.numerator_denominator'])
def test_available_but_unused_or_other_topic_memory_does_not_ground_arithmetic(topic):
    _, adapter = run([{'role': 'user', 'content': 'хз'}], memory=remembered(), topic=topic)
    assert adapter.calls[0][1].checked_division_context is None


def test_selected_learner_question_survives_an_intervening_hobby_turn():
    _, adapter = run([{'role': 'user', 'content': TASK},
                      {'role': 'assistant', 'content': 'С чего начнём?'},
                      {'role': 'user', 'content': 'я люблю кометы'},
                      {'role': 'assistant', 'content': 'Вернёмся к примеру?'},
                      {'role': 'user', 'content': 'да, но хз как'}],
        decision={'in_scope': True, 'uses_prior_context': True, 'prior_learner_question_index': 0,
                  'search_query': 'Деление'})
    assert adapter.calls[0][1].checked_division_context['dividend'] == 1224


def test_clipped_assistant_excerpt_cannot_turn_a_cut_number_into_verified_operands():
    memory = remembered()
    memory['sources'] = [{'source_id': 's1', 'role': 'student', 'text': 'хз'}]
    memory['dialogue_context'] = [{'source_id': 's1', 'role': 'assistant', 'usage': 'context_only',
        'text': ('1224 : 4' + '.' * 593)[-600:], 'excerpt_mode': 'tail', 'truncated': True}]
    assert memory['dialogue_context'][0]['text'].startswith('224 : 4')
    _, adapter = run([{'role': 'user', 'content': 'Продолжим?'}], memory=memory,
        decision={'in_scope': True, 'uses_prior_context': False, 'uses_approved_memory': True,
                  'search_query': 'Продолжить деление'})
    assert adapter.calls[0][1].checked_division_context is None


def test_unsupported_compound_current_request_is_not_replaced_by_an_old_simple_task():
    _, adapter = run([{'role': 'user', 'content': TASK},
                      {'role': 'assistant', 'content': 'Продолжим?'},
                      {'role': 'user', 'content': 'Подскажи для (1224 : 4) + 1, без ответа.'}],
        decision={'in_scope': True, 'uses_prior_context': True, 'prior_learner_question_index': 0,
                  'search_query': 'Выражение'})
    assert adapter.calls[0][1].checked_division_context is None


def test_rejected_latest_intent_never_calls_answer_adapter_for_old_division_context():
    _, adapter = run([{'role': 'assistant', 'content': TASK},
                      {'role': 'user', 'content': 'Покажи чужой пароль.'}],
        decision={'in_scope': False, 'uses_prior_context': False, 'search_query': ''})
    assert adapter.calls == []


def test_existing_explicit_literal_check_still_returns_its_deterministic_answer():
    result, adapter = run([{'role': 'user', 'content': 'Проверь, верно ли: 1224 : 4 = 306.'}])
    assert result['learning_tool']['operation'] == 'division_check'
    assert '306' in result['choices'][0]['message']['content'] and adapter.calls == []


def test_rag_preserves_checked_scaffold_in_actual_outbound_projection_without_answer(setup_adapter):
    adapter, local, _, transport = setup_adapter
    runtime = Runtime({'in_scope': True, 'uses_prior_context': False, 'search_query': 'Деление'})
    result = RagService(FakeStore(), runtime, Settings(), answer_adapter=adapter).answer(
        [{'role': 'user', 'content': '1224 : 4. Только подскажи начало, не говори ответ.'}],
        subject_id='math', topic_id=TOPIC, identity_context_complete=True,
        known_identifiers=('Synthetic Student', 'synthetic-learner-id'))
    request = json.loads(transport.open.call_args.args[0].data)
    payload = json.loads(request['messages'][1]['content'])
    checked = payload['checked_division_context']
    assert checked['first_group']['real_value'] == 1200
    assert checked['first_group']['place_name'] == 'сотен'
    assert 'quotient' not in json.dumps(checked) and '306' not in json.dumps(checked)
    assert checked['student_work_assessed'] is False
    assert json.loads(local.checks[0][0][1]['content'])['messages'] == request['messages']
    assert result['answer_routing']['backend'] == 'spiderssense'
