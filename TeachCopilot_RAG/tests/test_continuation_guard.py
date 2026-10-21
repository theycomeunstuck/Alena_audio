from copy import deepcopy
import json

import pytest

from local_app.continuation_guard import foreign_exercise_guard
from local_app.rag import RagService
from tests.test_memory_retrieval import Store, Settings, Runtime, memory, TOPIC

CURRENT = 'Продолжим тот пример. Только один шаг, без ответа.'
REJECTED = 'Давай вернёмся к примеру с делением 85 на 4. Чтобы начать, нужно посмотреть на самое большое число в 85 – это десятки. Сколько раз число 4 помещается в 80?'


def test_saved_live_failure_is_detected_without_changing_either_input():
    value = memory()
    before = deepcopy(value)
    result = foreign_exercise_guard(CURRENT,value,REJECTED)
    assert result['expected_pair'] == [1224,4] and result['referenced_pair'] == [85,4]
    assert '1224 : 4' in result['answer'] and '306' not in result['answer']
    assert value == before


@pytest.mark.parametrize('answer', ['Сейчас делим 12 на 4. Какую цифру запишешь?',
    'В примере 1224 : 4 начни с первой подходящей группы.', 'Нужно поставить ноль в десятках.'])
def test_ordinary_subcalculations_and_the_actual_task_are_not_relabelled(answer):
    assert foreign_exercise_guard(CURRENT,memory(),answer) is None


@pytest.mark.parametrize('latest', ['Дай другой пример.', 'Продолжим с новым примером.',
    'Не продолжай тот пример.', 'Продолжим деление 4512 : 12.'])
def test_a_new_task_request_is_not_forced_to_use_the_old_operands(latest):
    assert foreign_exercise_guard(latest,memory(),REJECTED) is None


def test_ambiguous_or_corrected_memory_does_not_choose_a_single_exercise():
    value = memory()
    value['sources'].append({'source_id':'s2','role':'student','text':'Ещё решали 3636 : 12.'})
    assert foreign_exercise_guard(CURRENT,value,REJECTED) is None
    value['sources'][1].update(role='teacher',corrects_source_id='s1')
    assert foreign_exercise_guard(CURRENT,value,REJECTED) is None


def test_rag_retains_rejected_response_but_returns_one_local_question_without_retry():
    class Bad(Runtime):
        def __init__(self):
            super().__init__(); self.calls = 0
        def chat(self,*args,**kwargs):
            self.calls += 1
            return {'message':{'content':json.dumps({'answer':REJECTED,'citations':['right']})},
                    'prompt_eval_count':123,'eval_count':44}
    runtime = Bad()
    result = RagService(Store(),runtime,Settings()).answer([{'role':'user','content':CURRENT}],
        subject_id='math',topic_id=TOPIC,memory_context=memory())
    assert '1224 : 4' in result['choices'][0]['message']['content']
    assert result['sources'] == result['citations'] == []
    assert result['answer_validation']['reason'] == 'foreign_exercise_reference'
    assert result['_local_diagnostics']['model_response']['rejected_answer'] == REJECTED
    assert result['usage']['total_tokens'] == 167
    assert runtime.calls == 1 and len(runtime.embedded) == 1
    assert result['answer_routing']['backend'] == 'deterministic_local'
