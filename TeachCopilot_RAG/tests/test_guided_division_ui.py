"""Guided controls with real static UI/auth and controlled action transport.

The PostgreSQL/API tests verify persistence and review. These test actual browser
requests and state restoration without invoking a model or touching live data.
"""
from copy import deepcopy
import json

import pytest
from playwright.sync_api import expect
from test_ui_acceptance import _login, ui_server  # noqa: F401
from local_app.explicit_learning_actions import validate_action
from local_app.guided_division import start_state, transition, action_response, render_prompt


def install_transport(page, initial=None, *, unavailable_reason=None, resume_required=False):
    data = {'state':initial, 'origin':'approved_memory' if initial else 'current_session',
            'session':None, 'sent':[], 'chat_sent':[], 'cache':{},
            'unavailable_reason':unavailable_reason, 'resume_required':resume_required}
    def context(route):
        cursor = data['state']
        reason = data['unavailable_reason']
        payload = {'available':not reason and cursor is not None and cursor['status'] != 'completed'}
        if reason:
            payload['reason'] = reason
        if cursor and (not reason or reason == 'exercise_completed'):
            payload.update(guided_division=cursor, origin=data['origin'],
                source_learning_session_id=data['session'] or 'reviewed-previous-session', prompt=render_prompt(cursor))
            if data['resume_required']:
                payload.update(resume_required=True, resume_reason='conversation_continued')
            if data['session']:
                payload['active_learning_session_id'] = data['session']
        route.fulfill(status=200, content_type='application/json', body=json.dumps(payload))
    def submit(route):
        request = route.request.post_data_json
        action = request.get('learning_action', {})
        if not action.get('operation','').startswith('division_guided_'):
            # A controlled ordinary reply tests the actual browser transition;
            # this is not an inference-quality or PostgreSQL persistence test.
            data['chat_sent'].append(deepcopy(request))
            assert not action and data['state'] is not None
            data['resume_required'] = True
            route.fulfill(status=200, content_type='application/json', body=json.dumps({
                'choices':[{'message':{'role':'assistant',
                    'content':'Хорошо. Если хочешь, вернёмся к примеру.'}}]}))
            return
        data['sent'].append(deepcopy(request))
        mid = request['client_message_id']
        if mid not in data['cache']:
            canonical, text = validate_action(action)
            if canonical['operation'] == 'division_guided_start':
                cursor = start_state(canonical['dividend'], canonical['divisor'], state_id=mid)
                assessment = {'status':'not_assessed','competence_assessed':False,'independent_work':'unknown'}
            else:
                assert not data['resume_required'] or canonical['operation'] == 'division_guided_resume'
                assert canonical['expected_state_id'] == data['state']['state_id']
                cursor, assessment = transition(data['state'], canonical['operation'], state_id=mid, value=canonical.get('value'))
            data['state'], data['session'], data['origin'] = cursor, request['learning_session_id'], 'current_session'
            data['unavailable_reason'] = None
            data['resume_required'] = False
            data['cache'][mid] = action_response(cursor,assessment,text,canonical['operation'])
        route.fulfill(status=200, content_type='application/json', body=json.dumps(data['cache'][mid]))
    page.route('**/guided-division?*', context)
    page.route('**/v1/chat/completions', submit)
    return data


def click_action(page, selector):
    with page.expect_response(lambda r:r.url.endswith('/v1/chat/completions')) as result:
        page.locator(selector).click()
    assert result.value.status == 200
    expect(page.locator(selector)).to_be_enabled() if page.locator(selector).is_visible() else None


def test_student_keeps_zero_step_through_hints_pause_and_page_reload(chromium, ui_server):
    page = chromium.new_page(viewport={'width':1365,'height':1050})
    errors = []
    page.on('pageerror',lambda e:errors.append(str(e)))
    transport = install_transport(page)
    try:
        _login(page,ui_server.base_url,'student.ui@example.test')
        page.locator('#chat-mode').select_option('division_guided')
        expect(page.locator('#guided-division-status')).to_contain_text('Сохранённого шага')
        page.locator('#action-dividend').fill('3636'); page.locator('#action-divisor').fill('12')
        click_action(page,'#send-learning-action')
        assert transport['sent'][0]['learning_action'] == {'operation':'division_guided_start','dividend':3636,'divisor':12}
        sid = transport['session']
        for number in ('3','0','3'):
            page.locator('#guided-division-answer').fill(number)
            click_action(page,'#guided-division-check')
        expect(page.locator('#guided-division-prompt')).to_contain_text('3 на 12')
        expect(page.locator('#guided-division-prompt')).to_contain_text('десятков')
        click_action(page,'#guided-division-hint')
        expect(page.locator('#guided-division-prompt')).to_contain_text('3 меньше 12')
        assert transport['state']['step_index'] == 1
        click_action(page,'#guided-division-pause')
        expect(page.locator('#guided-division-answer-form')).to_be_hidden()
        expect(page.locator('#guided-division-resume')).to_be_visible()
        click_action(page,'#guided-division-resume')
        page.reload()
        expect(page.locator('#learner-name')).to_contain_text('Ника')
        page.locator('#chat-mode').select_option('division_guided')
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        expect(page.locator('#guided-division-prompt')).to_contain_text('3 на 12')
        page.locator('#guided-division-answer').fill('0')
        click_action(page,'#guided-division-check')
        assert transport['sent'][-1]['learning_session_id'] == sid
        assert len(ui_server.store.learning_sessions) == 1
        assert '303' not in page.locator('#guided-division-panel').inner_text()
        assert transport['state']['state_id'] not in page.locator('#guided-division-panel').inner_text()
        assert not errors
    finally:
        page.close()


def test_reviewed_previous_exercise_waits_for_explicit_resume(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636,12,state_id='reviewed-last-step')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page,cursor)
    try:
        _login(page,ui_server.base_url,'student.ui@example.test')
        page.locator('#chat-mode').select_option('division_guided')
        expect(page.locator('#guided-division-resume')).to_be_visible()
        expect(page.locator('#guided-division-answer-form')).to_be_hidden()
        assert not transport['sent']
        click_action(page,'#guided-division-resume')
        request = transport['sent'][0]['learning_action']
        assert request == {'operation':'division_guided_resume', 'expected_state_id':'reviewed-last-step',
                           'source_learning_session_id':'reviewed-previous-session'}
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        assert transport['state']['step_index'] == 1
    finally:
        page.close()


@pytest.mark.parametrize('aside', ['ага', 'хз', '3', 'я в майнкрафте строю дом'])
def test_return_after_ordinary_conversation_requires_resume_without_advancing(chromium, ui_server, aside):
    """Use controlled ordinary replies; browser actions and reload are real."""
    page = chromium.new_page(viewport={'width':1365, 'height':1050})
    transport = install_transport(page)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        page.locator('#chat-mode').select_option('division_guided')
        expect(page.locator('#guided-division-status')).to_contain_text('Сохранённого шага')
        page.locator('#action-dividend').fill('3636')
        page.locator('#action-divisor').fill('12')
        click_action(page, '#send-learning-action')
        for number in ('3', '0', '3'):
            page.locator('#guided-division-answer').fill(number)
            click_action(page, '#guided-division-check')
        saved = deepcopy(transport['state'])
        assert (saved['step_index'], saved['phase'], saved['status']) == (1, 'quotient_digit', 'active')
        typed_calls = len(transport['sent'])
        session_id = transport['session']

        page.locator('#chat-mode').select_option('chat')
        assert len(transport['sent']) == typed_calls and not transport['chat_sent']
        page.locator('#chat-input').fill(aside)
        click_action(page, '#send-chat')
        assert transport['chat_sent'][0]['messages'] == [{'role':'user', 'content':aside}]
        assert transport['chat_sent'][0]['learning_session_id'] == session_id
        assert transport['state'] == saved

        for reload_first in (False, True):
            if reload_first:
                page.reload()
                expect(page.locator('#learner-name')).to_contain_text('Ника')
            page.locator('#chat-mode').select_option('division_guided')
            expect(page.locator('#guided-division-status')).to_have_text(
                'Последний проверенный шаг сохранён. После беседы можно вернуться к нему.')
            expect(page.locator('#guided-division-resume')).to_have_text('Вернуться к сохранённому шагу')
            expect(page.locator('#guided-division-resume')).to_be_visible()
            for selector in ('#guided-division-answer-form', '#guided-division-hint', '#guided-division-pause'):
                expect(page.locator(selector)).to_be_hidden()
            expect(page.locator('#guided-division-prompt')).to_contain_text('3 на 12')
            assert len(transport['sent']) == typed_calls and len(transport['chat_sent']) == 1
            assert transport['state'] == saved

        click_action(page, '#guided-division-resume')
        assert transport['sent'][-1]['learning_action'] == {
            'operation':'division_guided_resume', 'expected_state_id':saved['state_id']}
        assert transport['sent'][-1]['learning_session_id'] == session_id
        assert transport['state']['step_index'] == saved['step_index']
        assert transport['state']['phase'] == saved['phase']
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        expect(page.locator('#guided-division-status')).to_have_text('Один вопрос – один шаг. Готовый ответ не показываем.')
        expect(page.locator('#guided-division-resume')).to_have_text('Продолжить пример')
        page.locator('#guided-division-answer').fill('0')
        click_action(page, '#guided-division-check')
        assert (transport['state']['step_index'], transport['state']['phase']) == (1, 'remainder')
        assert len(ui_server.store.learning_sessions) == 1
        assert '303' not in page.locator('#guided-division-panel').inner_text()
    finally:
        page.close()


def test_reviewed_conversation_return_keeps_cross_session_resume_source(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636, 12, state_id='reviewed-step-before-conversation')
    cursor.update(step_index=1, phase='quotient_digit')
    transport = install_transport(page, cursor, resume_required=True)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        page.locator('#chat-mode').select_option('division_guided')
        expect(page.locator('#guided-division-resume')).to_have_text('Вернуться к сохранённому шагу')
        expect(page.locator('#guided-division-answer-form')).to_be_hidden()
        assert not transport['sent']
        click_action(page, '#guided-division-resume')
        assert transport['sent'][0]['learning_action'] == {
            'operation':'division_guided_resume', 'expected_state_id':cursor['state_id'],
            'source_learning_session_id':'reviewed-previous-session'}
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        assert (transport['state']['step_index'], transport['state']['phase']) == (1, 'quotient_digit')
    finally:
        page.close()


@pytest.mark.parametrize(('reason', 'message'), [
    ('teacher_review_required', 'Для продолжения учителю нужно проверить записи прошлого занятия. Пока можно начать новый пример.'),
    ('corrected_guided_source', 'Запись сохранённого шага исправлена. Продолжение этого примера недоступно. Можно начать новый.'),
    ('later_message_barrier', 'После этого шага выбрано другое учебное действие. Можно начать новый пример.'),
    ('message_outcome_unavailable', 'Результат предыдущего сообщения пока не подтверждён. Продолжение недоступно.'),
    ('invalid_guided_chain', 'Сохранённый шаг не удалось проверить. Можно начать новый пример.'),
    ('another_actor', 'Сохранённого шага для этого входа нет. Можно начать новый пример.'),
    ('exercise_completed', 'Этот пример завершён. Можно начать новый.'),
])
def test_unavailable_step_explains_reason_without_offering_resume(chromium, ui_server, reason, message):
    page = chromium.new_page()
    cursor = None
    if reason == 'exercise_completed':
        cursor = start_state(12, 4, state_id='private-completed-step-id')
        cursor.update(step_index=0, phase='remainder', status='completed')
    transport = install_transport(page, cursor, unavailable_reason=reason)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        page.locator('#chat-mode').select_option('division_guided')
        expect(page.locator('#guided-division-status')).to_have_text(message)
        for selector in ('#guided-division-resume', '#guided-division-answer-form',
                         '#guided-division-hint', '#guided-division-pause'):
            expect(page.locator(selector)).to_be_hidden()
        expect(page.locator('#send-learning-action')).to_be_enabled()
        assert not transport['sent']
        panel_text = page.locator('#guided-division-panel').inner_text()
        assert reason not in panel_text
        assert 'private-completed-step-id' not in panel_text
        assert 'reviewed-previous-session' not in panel_text
        page.locator('#action-dividend').fill('12')
        page.locator('#action-divisor').fill('4')
        click_action(page, '#send-learning-action')
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        assert transport['sent'][0]['learning_action'] == {
            'operation':'division_guided_start', 'dividend':12, 'divisor':4}
    finally:
        page.close()


def test_guided_mode_loads_saved_step_before_catalog_and_keeps_topic_after_catalog(chromium, ui_server):
    page = chromium.new_page()
    held_catalog = []
    page.route('**/catalog', lambda route: held_catalog.append(route))
    cursor = start_state(3636, 12, state_id='reviewed-step-before-catalog')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page, cursor)
    topic = 'math.g4.numbers.division_by_1_2_digit'
    title = 'Деление на однозначное и двузначное число'
    try:
        with page.expect_request(lambda request: request.url.endswith('/catalog')):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#learner-name')).to_contain_text('Ника')
        assert len(held_catalog) == 1
        with page.expect_response(lambda response: '/guided-division?' in response.url) as context:
            page.locator('#chat-mode').select_option('division_guided')
        assert context.value.status == 200
        assert f'topic_id={topic}' in context.value.url
        expect(page.locator('#chat-topic')).to_have_value(topic)
        expect(page.locator('#chat-topic option:checked')).to_have_text(title)
        expect(page.locator('#guided-division-resume')).to_be_visible()
        expect(page.locator('#guided-division-answer-form')).to_be_hidden()
        assert not transport['sent']

        with page.expect_response(lambda response: response.url.endswith('/catalog')):
            held_catalog.pop().continue_()
        expect(page.locator('#chat-topic option:checked')).to_have_text(f'{title} · 4 класс')
        expect(page.locator('#chat-topic')).to_have_value(topic)
        expect(page.locator('#chat-mode')).to_have_value('division_guided')
        expect(page.locator('#guided-division-resume')).to_be_visible()
        click_action(page, '#guided-division-resume')
        session_id = transport['sent'][0]['learning_session_id']
        assert ui_server.store.learning_sessions[session_id]['topic_id'] == topic
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        assert transport['state']['step_index'] == 1
    finally:
        for route in held_catalog:
            route.abort()
        page.close()
