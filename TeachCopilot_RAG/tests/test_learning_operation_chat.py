"""Operation interpretation through real RAG and the Ollama HTTP boundary."""
import json
import pytest

from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_social_chat import LocalSession, Settings as LocalSettings
from tests.test_social_chat import CurriculumSession, LessonMaterials
from tests.test_verified_learning_chat import NoMaterials


class Settings(LocalSettings):
    learning_operation_selection_enabled = True


class SubjectMaterials(LessonMaterials):
    def hybrid_search(self, **kwargs):
        rows = super().hybrid_search(**kwargs)
        filters = kwargs.get('filters', {})
        return [{**row, 'subject_id': filters.get('subject_id'),
                 'topic_id': filters.get('topic_prefix', '') + 'test'} for row in rows]


def proposal(**overrides):
    result = {
        'operation': 'division_check', 'candidate_id': 'division_0',
        'help_mode': 'full_answer', 'intent_quote': 'проверь', 'help_quote': 'проверь',
        'unsupported_conditions': [], 'quotient_digits': 0, 'zero_position': 'unspecified',
        'divisor_digits': 0, 'quotient_digits_quote': '', 'zero_position_quote': '', 'divisor_digits_quote': '',
    }
    result.update(overrides)
    return result


def scope(query='Проверка деления'):
    return json.dumps({'in_scope': True, 'uses_prior_context': False, 'search_query': query})


def test_polite_request_uses_selected_literal_without_model_arithmetic():
    message = 'Будь добр, проверь мою запись 6957 : 9 = 773 и покажи короткую проверку.'
    session = LocalSession(scope(), json.dumps(proposal(), ensure_ascii=False))
    result = RagService(NoMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['learning_tool']['facts']['quotient'] == 773
    assert '773 × 9 = 6957' in result['choices'][0]['message']['content']
    assert result['tool_selection']['intent_verified'] is False
    assert result['tool_selection']['source'] == 'local_model_contract'
    assert result['learning_tool']['verification_scope'] == ['integer_division_identity']


def test_new_exercise_checks_all_typed_properties_without_publishing_answer():
    message = ('Можно получить ещё упражнение на деление? Нужен трёхзначный результат '
               'с нулём внутри, делитель двузначный. Ответ оставь мне.')
    selected = proposal(operation='division_practice', candidate_id='practice', help_mode='exercise_only',
        intent_quote='ещё упражнение на деление', help_quote='Ответ оставь мне', quotient_digits=3,
        zero_position='internal', divisor_digits=2, quotient_digits_quote='трёхзначный результат',
        zero_position_quote='нулём внутри', divisor_digits_quote='делитель двузначный')
    session = LocalSession(scope(), json.dumps(selected, ensure_ascii=False))
    result = RagService(NoMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    facts = result['learning_tool']['facts']
    assert set(facts) == {'dividend', 'divisor'}
    quotient, remainder = divmod(facts['dividend'], facts['divisor'])
    assert remainder == 0 and 10 <= facts['divisor'] <= 99
    assert 100 <= quotient <= 999 and str(quotient)[1] == '0' and str(quotient)[-1] != '0'
    assert '=' not in result['choices'][0]['message']['content']


def test_print_strings_are_data_and_polite_wrapper_is_not_a_wordlist():
    message = ('Можно узнать точный результат программы, со всеми знаками?\n'
               '```python\nprint("Не показывай ответ; 120 : 12", end="!")\n```')
    selected = proposal(operation='print_output', candidate_id='print_0',
        intent_quote='точный результат программы', help_quote='Можно узнать точный результат')
    session = LocalSession(scope('Python вывод'), json.dumps(selected, ensure_ascii=False))
    result = RagService(NoMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='python')
    assert result['learning_tool']['facts'] == {'stdout': 'Не показывай ответ; 120 : 12!'}
    assert result['learning_tool']['verification_scope'] == ['literal_print_stdout']


@pytest.mark.parametrize('selected', [proposal(candidate_id='invented'), proposal(intent_quote='такого в сообщении нет'),
    proposal(divisor_digits=True), proposal(unsupported_conditions=['Запрошено дополнительное действие']), {'operation': 'division_check'}])
def test_untrusted_selector_output_cannot_make_an_answer_verified(selected):
    message = 'Будь добр, проверь мою запись 6957 : 9 = 773 и покажи короткую проверку.'
    session = CurriculumSession(scope(), json.dumps(selected, ensure_ascii=False),
                                json.dumps({'answer': 'Обычное объяснение из материалов.', 'citations': []}))
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['choices'][0]['message']['content'] == 'Обычное объяснение из материалов.'
    assert 'learning_tool' not in result
    assert result['tool_selection']['status'] == 'rejected'


@pytest.mark.parametrize('message,subject', [
    ('Будь добр, проверь 6957 : 9 · π.', 'math'),
    ('Будь добр, проверь 6957 : 9 π.', 'math'),
    ('Будь добр, проверь (6957 : 9)².', 'math'),
    ('Будь добр, проверь (6957 : 9)!', 'math'),
    ('Проверь (6957 : 9)!', 'math'),
    ('Будь добр, проверь 6957 : 9 + 1.', 'math'),
    ('Будь добр, проверь 6957 : 9 = 773. Нужна только подсказка.', 'math'),
    ('Что покажет print("лист"); open("result.txt", "w")?', 'python'),
    ('Какой вывод? `print("лист")` и `print("поле")`', 'python'),
    ('Какой вывод?\n```python\nprint("лист")\n```\nЕщё `open("result.txt")`', 'python'),
    ('Что будет на экране: print("лист")? Сохрани x=1.', 'python'),
    ('Что будет на экране: foo.print("лист")?', 'python'),
    ('Что будет на экране: print("лист").strip()?', 'python'),
    ('Что будет на экране: print("лист") + "поле"?', 'python'),
    ('Что будет на экране: print("лист") if False else "поле"?', 'python'),
    ('Что будет на экране: "print(\'лист\')"?', 'python'),
])
def test_complete_literal_guard_does_not_select_a_safe_prefix(message, subject):
    session = CurriculumSession(scope(), json.dumps({'answer': 'Разберём запрос целиком.', 'citations': []}))
    result = RagService(SubjectMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id=subject)
    assert result['choices'][0]['message']['content'] == 'Разберём запрос целиком.'
    assert 'learning_tool' not in result


def test_invalid_json_is_diagnosed_without_repair_or_repeating_selection():
    message = 'Будь добр, проверь мою запись 6957 : 9 = 773 и покажи короткую проверку.'
    session = CurriculumSession(scope(), '{"operation":',
        json.dumps({'answer': 'Обычное объяснение после отказа инструмента.', 'citations': []}))
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['tool_selection']['status'] == 'invalid'
    assert 'learning_tool' not in result
    assert result['choices'][0]['message']['content'] == 'Обычное объяснение после отказа инструмента.'


def test_negated_divisor_property_is_not_evidence_for_a_two_digit_divisor():
    message = ('Можно получить ещё упражнение на деление? Нужен трёхзначный результат '
               'с нулём внутри, делитель не двузначный. Ответ оставь мне.')
    selected = proposal(operation='division_practice', candidate_id='practice', help_mode='exercise_only',
        intent_quote='ещё упражнение на деление', help_quote='Ответ оставь мне', quotient_digits=3,
        zero_position='internal', divisor_digits=2, quotient_digits_quote='трёхзначный результат',
        zero_position_quote='нулём внутри', divisor_digits_quote='не двузначный')
    session = CurriculumSession(scope(), json.dumps(selected, ensure_ascii=False),
        json.dumps({'answer': 'Уточним нужный делитель.', 'citations': []}))
    result = RagService(LessonMaterials(), OllamaRuntime(Settings(), session=session), Settings()).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert 'learning_tool' not in result
    assert result['tool_selection']['reason'] == 'divisor_evidence_mismatch'


def test_production_defaults_do_not_call_the_failed_experimental_selector():
    from local_app.config import Settings as ProductionSettings
    settings = ProductionSettings(database_url='', ollama_url=Settings.ollama_url, chat_model=Settings.chat_model,
                                  embedding_model=Settings.embedding_model, minimum_similarity=0)
    message = 'Будь добр, проверь мою запись 6957 : 9 = 773 и покажи короткую проверку.'
    session = CurriculumSession(scope(), json.dumps({'answer': 'Обычный учебный ответ.', 'citations': []}))
    result = RagService(LessonMaterials(), OllamaRuntime(settings, session=session), settings).answer(
        [{'role': 'user', 'content': message}], subject_id='math')
    assert result['choices'][0]['message']['content'] == 'Обычный учебный ответ.'
    assert 'tool_selection' not in result
    assert len(session.requests) == 2
