"""Verified learning responses through the normal chat and HTTP boundaries."""
import json
import pytest

from local_app.answer_adapter import AnswerAdapter
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_social_chat import CurriculumSession, ExternalProvider, LessonMaterials, LocalSession, Settings


class NoMaterials:
    def hybrid_search(self, **kwargs):
        raise AssertionError('A literal arithmetic check needs no retrieved materials')


def test_direct_division_check_returns_computed_result_without_answer_generation():
    session = LocalSession(json.dumps({'in_scope': True, 'uses_prior_context': False,
                                      'search_query': 'Проверка деления 3663 на 11'}))
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(NoMaterials(), runtime, Settings(), AnswerAdapter(runtime, ExternalProvider()))
    result = service.answer([{'role': 'user', 'content': 'Проверь, пожалуйста, 3663 : 11 = 333.'}], subject_id='math')
    assert '333 × 11 = 3663' in result['choices'][0]['message']['content']
    assert result['answer_routing']['backend'] == 'deterministic_local'
    assert result['learning_tool']['operation'] == 'division_check'
    assert result['citations'] == result['sources'] == []


def test_literal_print_returns_exact_quotes_and_separator_without_model_answer():
    session = LocalSession(json.dumps({'in_scope': True, 'uses_prior_context': False,
                                      'search_query': 'Python print и разделитель'}))
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(NoMaterials(), runtime, Settings(), AnswerAdapter(runtime, ExternalProvider()))
    result = service.answer([{'role': 'user', 'content':
        'Что выведет этот код?\n```python\nprint(\'"Туман"\', \'лес\', sep=\'|\')\n```'}], subject_id='python')
    assert result['learning_tool']['facts']['stdout'] == '"Туман"|лес\n'
    assert '"Туман"|лес' in result['choices'][0]['message']['content']
    assert result['learning_tool']['operation'] == 'print_output'


def test_place_values_come_from_the_quotient_and_keep_the_zero():
    session = LocalSession(json.dumps({'in_scope': True, 'uses_prior_context': False,
                                      'search_query': 'Разряды частного 4928 на 16'}))
    runtime = OllamaRuntime(Settings(), session=session)
    result = RagService(NoMaterials(), runtime, Settings()).answer([
        {'role': 'user', 'content': 'Поясни разряды частного 4928 : 16 = 308.'}], subject_id='math')
    text = result['choices'][0]['message']['content']
    assert '3 × 100 = 300' in text and '0 × 10 = 0' in text and '8 × 1 = 8' in text
    assert result['learning_tool']['operation'] == 'division_places'


@pytest.mark.parametrize('zero_property', ['посередине ноль', 'ноль в середине', 'с нулём внутри'])
def test_new_practice_has_the_requested_zero_without_revealing_the_quotient(zero_property):
    session = LocalSession(json.dumps({'in_scope': True, 'uses_prior_context': False,
                                      'search_query': 'Новый пример с внутренним нулём'}))
    runtime = OllamaRuntime(Settings(), session=session)
    result = RagService(NoMaterials(), runtime, Settings()).answer([{'role': 'user', 'content':
        'Подбери один новый пример на деление. В частном три цифры, ' + zero_property + ', '
        'делитель двузначный. Ответ не показывай.'}], subject_id='math')
    assert result['learning_tool']['operation'] == 'division_practice'
    operands = result['learning_tool']['facts']
    assert set(operands) == {'dividend', 'divisor'}
    quotient, remainder = divmod(operands['dividend'], operands['divisor'])
    assert remainder == 0 and 10 <= operands['divisor'] <= 99
    assert 100 <= quotient <= 999 and str(quotient)[1] == '0' and str(quotient)[-1] != '0'
    assert '=' not in result['choices'][0]['message']['content']


@pytest.mark.parametrize('message', [
    'Проверь 120 : 12 + 5 = 15.',
    'Проверь 120 : 12 = 10 + 5.',
    'Проверь (120 : 12) * 2.',
    'Проверь -(120 : 12).',
    'Проверь 120 : 12 = 10,5.',
    'Проверь 120 : 12 = 10000000.',
])
def test_compound_or_unsupported_equality_is_not_silently_shortened(message):
    session = CurriculumSession(
        json.dumps({'in_scope': True, 'uses_prior_context': False, 'search_query': message}),
        json.dumps({'answer': 'Разберём всё выражение, включая остальные действия.', 'citations': ['division-1']}),
    )
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['choices'][0]['message']['content'] == 'Разберём всё выражение, включая остальные действия.'
    assert 'learning_tool' not in result


@pytest.mark.parametrize('hint_request', ['Хочу только намёк.', 'Не называй сам ответ.',
                                   'Оставь продолжение мне, скажи только первый шаг.'])
def test_request_for_a_hint_does_not_reveal_a_computed_quotient(hint_request):
    message = 'Проверь 120 : 12. ' + hint_request
    session = CurriculumSession(
        json.dumps({'in_scope': True, 'uses_prior_context': False, 'search_query': message}),
        json.dumps({'answer': 'Сначала посмотри, какая начальная часть делимого не меньше делителя.', 'citations': []}),
    )
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['choices'][0]['message']['content'] == 'Сначала посмотри, какая начальная часть делимого не меньше делителя.'
    assert 'learning_tool' not in result


@pytest.mark.parametrize('extra', ['Делитель должен быть 19.', 'Делитель нечётный.', 'С трёхзначным делителем.',
                                 'Не дай мне новый пример, я пока об этом не прошу.'])
def test_practice_does_not_ignore_additional_or_negated_requirements(extra):
    message = 'Дай новый пример на деление с трёхзначным частным и нулём в середине. ' + extra
    session = CurriculumSession(
        json.dumps({'in_scope': True, 'uses_prior_context': False, 'search_query': message}),
        json.dumps({'answer': 'Уточним все условия нового примера.', 'citations': []}),
    )
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['choices'][0]['message']['content'] == 'Уточним все условия нового примера.'
    assert 'learning_tool' not in result


def test_another_practice_does_not_repeat_the_visible_previous_exercise():
    message = 'Дай новый пример на деление. В частном три цифры и ноль внутри. Ответ не показывай.'
    session = LocalSession(*[json.dumps({'in_scope': True, 'uses_prior_context': False,
                                        'search_query': 'Новый пример с внутренним нулём'})] * 2)
    service = RagService(NoMaterials(), OllamaRuntime(Settings(), session=session), Settings())
    first = service.answer([{'role': 'user', 'content': message}], subject_id='math')
    second = service.answer([{'role': 'user', 'content': message},
                             first['choices'][0]['message'],
                             {'role': 'user', 'content': message}], subject_id='math')
    assert first['learning_tool']['facts'] != second['learning_tool']['facts']
    assert set(second['learning_tool']['facts']) == {'dividend', 'divisor'}
