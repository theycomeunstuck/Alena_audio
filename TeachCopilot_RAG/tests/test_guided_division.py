import json
import pytest

from local_app.guided_division import start_state, transition, render_prompt, action_response, validate_state
from local_app.explicit_learning_actions import validate_action


def move(state, value):
    return transition(state, 'division_guided_answer', state_id=state['state_id'] + 'n', value=value)


def test_internal_zero_is_a_separate_check_before_bringing_down_the_last_digit():
    state = start_state(3636, 12, state_id='start')
    for value in (3, 0, 3):
        state, assessed = move(state, value)
        assert assessed['status'] == 'correct' and assessed['independent_work'] == 'unknown'
    assert state['step_index'] == 1 and state['phase'] == 'quotient_digit'
    hint = render_prompt(state, hint=True)
    assert '3 на 12' in hint and '3 меньше 12' in hint and 'десятков' in hint
    assert '36' not in hint and '303' not in hint
    before = dict(state)
    wrong, assessment = move(state, 3)
    assert assessment['status'] == 'incorrect'
    assert {k:v for k,v in wrong.items() if k != 'state_id'} == {k:v for k,v in before.items() if k != 'state_id'}
    state, _ = move(wrong, 0)
    assert state['phase'] == 'remainder' and 'из 3' in render_prompt(state)
    state, _ = move(state, 3)
    assert state['phase'] == 'bring_down'
    state, _ = move(state, 6)
    assert state['step_index'] == 2 and '36 на 12' in render_prompt(state)
    assert before['phase'] == 'quotient_digit'


def test_entire_exercise_finishes_only_after_the_actual_last_remainder():
    state = start_state(3636, 12, state_id='s')
    for value in (3, 0, 3, 0, 3, 6, 3, 0):
        state, assessment = move(state, value)
        response = action_response(state, assessment, 'Мой ответ на шаг.', 'division_guided_answer')
        assert response['learning_tool']['facts'] == {}
        assert '303' not in json.dumps(response, ensure_ascii=False)
        assert assessment['competence_assessed'] is False
    assert state['status'] == 'completed'
    with pytest.raises(ValueError, match='завершён'):
        transition(state, 'division_guided_resume', state_id='next')


def test_hint_pause_and_explicit_resume_do_not_advance_a_cursor():
    original = start_state(1224, 4, state_id='s')
    hinted, assessment = transition(original, 'division_guided_hint', state_id='h')
    assert assessment['status'] == 'not_assessed' and hinted['phase'] == original['phase']
    paused, _ = transition(hinted, 'division_guided_pause', state_id='p')
    with pytest.raises(ValueError, match='приостановлен'):
        move(paused, 3)
    resumed, assessment = transition(paused, 'division_guided_resume', state_id='r')
    assert resumed['status'] == 'active' and resumed['step_index'] == 0
    assert assessment['status'] == 'not_assessed'
    active_resume, _ = transition(resumed, 'division_guided_resume', state_id='again')
    assert active_resume['phase'] == resumed['phase']


@pytest.mark.parametrize('value', ['хз', 'ага', '3', True, None, -1, 1.5])
def test_ambiguous_or_non_typed_answers_cannot_advance(value):
    state = start_state(3636, 12, state_id='s')
    with pytest.raises(ValueError):
        move(state, value)
    assert state['state_id'] == 's'


@pytest.mark.parametrize('a,b', [(0,4), (17,4), (1,4), (True,1), (1224,0), (1000000,4)])
def test_unsupported_start_does_not_silently_change_the_problem(a,b):
    with pytest.raises(ValueError):
        start_state(a,b,state_id='s')


def test_client_cannot_supply_a_cursor_and_source_text_does_not_show_internal_ids():
    action = {'operation':'division_guided_answer', 'expected_state_id':'private-step-id', 'value':3}
    canonical, text = validate_action(action)
    assert canonical == action and 'private-step-id' not in text
    with pytest.raises(ValueError):
        validate_action({**action, 'step_index': 5})
    state = start_state(3636,12,state_id='s')
    with pytest.raises(ValueError):
        validate_state({**state, 'quotient':303})
