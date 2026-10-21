from copy import deepcopy
import json

import pytest

from local_app.memory_retrieval import memory_retrieval_query
from local_app.rag import RagService
from tests.test_local_rag import FakeRuntime, FakeStore, Settings

TOPIC = 'math.g4.numbers.division_by_1_2_digit'


def memory():
    return {'version_id': 'approved', 'topic_id': TOPIC, 'text': 'DO NOT PARSE: 999 : 3.',
            'sources': [{'source_id': 's1', 'role': 'student', 'text': 'Начал 1224 : 4. Помощь неизвестна.'}]}


class Runtime(FakeRuntime):
    def resolve_chat_query(self, *args, **kwargs):
        return {'in_scope': True, 'uses_prior_context': False, 'uses_approved_memory': True,
                'search_query': '999 : 3'}


class Store(FakeStore):
    def hybrid_search(self, **kwargs):
        self.filters, self.search_query = kwargs['filters'], kwargs['query']
        return [dict(id='wrong', subject_id='math', topic_id='math.geometry', score=.99,
                     content='Треугольник'),
                dict(id='right', subject_id='math', topic_id=TOPIC, score=.7, content='Деление')]


def test_actual_embedding_uses_literal_approved_source_and_exact_topic_not_rewrite_or_prose():
    runtime, store = Runtime(), Store()
    response = RagService(store, runtime, Settings()).answer(
        [{'role': 'user', 'content': 'Продолжим тот пример.'}], subject_id='math', topic_id=TOPIC,
        memory_context=memory())
    assert '1224 : 4' in runtime.embedded[0] and '999 : 3' not in runtime.embedded[0]
    assert 'Продолжим тот пример.' in runtime.embedded[0]
    assert store.filters == {'subject_id': 'math', 'topic_id': TOPIC}
    assert [row['id'] for row in response['sources']] == ['right']
    assert response['retrieval_context']['mode'] == 'approved_memory_exact_topic'
    assert '1224' not in json.dumps(response['retrieval_context'])


def test_no_exact_materials_does_not_broaden_or_make_second_embedding():
    class Empty(Store):
        def hybrid_search(self, **kwargs):
            super().hybrid_search(**kwargs)
            return []
    runtime = Runtime()
    result = RagService(Empty(), runtime, Settings()).answer(
        [{'role': 'user', 'content': 'Продолжим.'}], subject_id='math', topic_id=TOPIC, memory_context=memory())
    assert len(runtime.embedded) == 1 and result['sources'] == []
    assert result['retrieval_context']['outcome'] == 'memory_without_topic_materials'


@pytest.mark.parametrize('in_scope', [True, False])
def test_unused_memory_does_not_change_a_new_question_or_a_rejection(in_scope):
    class NewQuestion(Runtime):
        def resolve_chat_query(self, *args, **kwargs):
            return {'in_scope':in_scope,'uses_prior_context':False,'search_query':'Что такое отрезок?' if in_scope else ''}
    runtime, store = NewQuestion(), Store()
    result = RagService(store,runtime,Settings()).answer([{'role':'user','content':'Новый вопрос.'}],
        subject_id='math',topic_id=TOPIC,memory_context=memory())
    assert 'retrieval_context' not in result
    assert '1224' not in str(runtime.embedded)
    if in_scope:
        assert store.filters == {'subject_id':'math'}
    else:
        assert runtime.embedded == [] and result['sources'] == []


def test_legacy_memory_without_quotes_uses_topic_and_current_message_only():
    value = memory()
    value['sources'] = []
    result = memory_retrieval_query('Где остановились?', TOPIC, value, topic_title='Деление')
    assert '999' not in result['query'] and 'Деление' in result['query']
    assert result['diagnostics']['anchor_status'] == 'literal_anchor_unavailable'


def test_correction_group_is_paired_and_not_presented_as_current_step():
    value = memory()
    value['sources'].append({'source_id': 'fix', 'role': 'teacher', 'text': 'Это было 3636 : 12.',
                             'corrects_source_id': 's1'})
    original = deepcopy(value)
    result = memory_retrieval_query('Продолжим.', TOPIC, value)
    assert 'Исправленная исходная запись' in result['query']
    assert 'Исправление преподавателя' in result['query']
    assert '1224' in result['query'] and '3636' in result['query']
    assert result['diagnostics']['literal_source_count'] == 2 and value == original


@pytest.mark.parametrize('kind', ['missing_target', 'cycle', 'wrong_role', 'oversized'])
def test_broken_or_over_budget_correction_group_is_not_partially_used(kind):
    value = memory()
    fix = {'source_id': 'fix', 'role': 'teacher', 'text': '3636 : 12.', 'corrects_source_id': 's1'}
    value['sources'].append(fix)
    if kind == 'missing_target':
        value['sources'] = [fix]
    elif kind == 'cycle':
        value['sources'][0]['corrects_source_id'] = 'fix'
    elif kind == 'wrong_role':
        fix['role'] = 'student'
    else:
        fix['text'] *= 1000
    result = memory_retrieval_query('Продолжим.', TOPIC, value)
    assert '1224' not in result['query'] and '3636' not in result['query']
    assert result['diagnostics']['literal_source_count'] == 0


def test_only_complete_linked_assistant_task_context_is_a_retrieval_anchor():
    value = memory()
    value['sources'][0]['text'] = 'хз'
    value['dialogue_context'] = [
        {'source_id': 's1', 'role': 'assistant', 'usage': 'context_only', 'truncated': False, 'text': 'Реши 3636 : 12.'},
        {'source_id': 'foreign', 'role': 'assistant', 'usage': 'context_only', 'truncated': False, 'text': '777 : 7'},
        {'source_id': 's1', 'role': 'assistant', 'usage': 'context_only', 'truncated': True, 'text': '224 : 4'},
    ]
    result = memory_retrieval_query('Продолжим.', TOPIC, value)
    assert '3636 : 12' in result['query'] and 'не действие ученика' in result['query']
    assert '777' not in result['query'] and '224' not in result['query']
    assert result['diagnostics']['context_only_count'] == 1
