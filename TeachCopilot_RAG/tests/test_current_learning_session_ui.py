"""A saved current conversation is restored without new paid or source writes.

The browser and chat routes use isolated in-memory fixtures. Controlled current
session responses cover delayed reads and unavailable outcomes; no real model
or live learner database is contacted.
"""
from copy import deepcopy
import json
import re

import pytest
from playwright.sync_api import expect

from test_ui_acceptance import _login, ui_server  # noqa: F401


CURRENT = '/learning-sessions/current'
FRACTIONS = 'math.g4.fractions.numerator_denominator'
PYTHON = 'programming.python.output'
NOTE = '#learning-session-recovery-note'
REFRESH = '#refresh-learning-session'


def is_current(response):
    return response.url.endswith(CURRENT)


def watch_writes(page):
    writes = {'sessions':[], 'chat':[], 'complete':[]}

    def capture(request):
        if request.method != 'POST':
            return
        if request.url.endswith('/learning-sessions'):
            writes['sessions'].append(deepcopy(request.post_data_json))
        elif request.url.endswith('/v1/chat/completions'):
            writes['chat'].append(deepcopy(request.post_data_json))
        elif request.url.endswith('/complete'):
            writes['complete'].append(request.url)

    page.on('request', capture)
    return writes


def controlled_current(page, payload, *, status=200):
    control = {'payload':payload, 'status':status, 'reads':[]}

    def respond(route):
        control['reads'].append(route.request.url)
        route.fulfill(status=control['status'], content_type='application/json',
                      body=json.dumps(control['payload'], ensure_ascii=False))

    page.route('**/learning-sessions/current', respond)
    return control


def active_payload(ui_server, *, topic=FRACTIONS, subject='math', unresolved=None, truncated=False):
    created = ui_server.store.create_learning_session('demo-nika-04', 'student-ui', topic, 'seed-current')
    sid = created['learning_session_id']
    title = ui_server.store.catalog_rows[topic]['title_ru']
    messages = [{'role':'user', 'content':'Мы остановились на этом вопросе.', 'delivery_status':'completed'},
                {'role':'assistant', 'content':'Покажи один следующий шаг.', 'delivery_status':'completed'}]
    if unresolved:
        messages = [{'role':'user', 'content':'Сообщение без подтверждённого результата.', 'delivery_status':unresolved}]
    ui_server.store.learning_turns[sid] = deepcopy(messages)
    return {'available':True, 'learning_session_id':sid, 'topic_id':topic, 'topic_title':title,
            'subject_id':subject, 'status':'active', 'messages':messages,
            'history_truncated':truncated, 'unresolved_status':unresolved}


def send(page, text):
    page.locator('#chat-input').fill(text)
    with page.expect_response(lambda response: response.url.endswith('/v1/chat/completions')) as result:
        page.locator('#send-chat').click()
    assert result.value.status == 200, result.value.text()
    expect(page.locator('#send-chat')).to_be_enabled()
    return result.value.json()


def settle(page):
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')


def test_reload_restores_actual_messages_and_continues_the_same_session(chromium, ui_server):
    page = chromium.new_page(viewport={'width':1365, 'height':1050})
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current) as initial:
            _login(page, ui_server.base_url, 'student.ui@example.test')
        assert initial.value.status == 200
        expect(page.locator('#send-chat')).to_be_enabled()
        first = send(page, 'Объясняем дробь. Пока хз, с чего начать.')
        reply = first['choices'][0]['message']['content']
        sid = writes['chat'][0]['learning_session_id']
        assert len(writes['sessions']) == 1 and len(ui_server.rag.calls) == 1

        with page.expect_response(is_current) as restored:
            page.reload()
        assert restored.value.status == 200
        expect(page.locator('#chat-mode')).to_have_value('chat')
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        expect(page.locator('#chat-topic')).to_be_disabled()
        expect(page.locator('#chat-subject')).to_be_disabled()
        expect(page.locator('#complete-learning-session')).to_be_visible()
        expect(page.locator('#chat-log .chat-bubble.user')).to_have_count(1)
        expect(page.locator('#chat-log')).to_contain_text('Объясняем дробь. Пока хз, с чего начать.')
        expect(page.locator('#chat-log')).to_contain_text(reply)
        assert page.locator('#chat-log img').count() == 0
        assert page.evaluate('window.__ui_xss') is None
        assert 'Synthetic hidden chain of thought' not in page.locator('#chat-log').inner_text()
        assert 'Synthetic teacher-only note' not in page.locator('#chat-log').inner_text()
        assert len(writes['sessions']) == len(writes['chat']) == len(ui_server.rag.calls) == 1
        assert not writes['complete']

        send(page, 'ага, теперь вижу целое')
        assert writes['chat'][-1]['learning_session_id'] == sid
        assert len(writes['sessions']) == 1 and len(ui_server.store.learning_sessions) == 1
        assert len(ui_server.rag.calls) == 2
        assert all(job['kind'] == 'chat' for job in ui_server.store.memory_jobs.values())
    finally:
        page.close()


def test_restored_python_session_preserves_subject_topic_and_original_history(chromium, ui_server):
    page = chromium.new_page()
    payload = active_payload(ui_server, topic=PYTHON, subject='python')
    controlled_current(page, payload)
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#chat-subject')).to_have_value('python')
        expect(page.locator('#chat-topic')).to_have_value(PYTHON)
        expect(page.locator('#learning-session-status')).to_contain_text('Вывод данных в Python')
        expect(page.locator('#chat-log')).to_contain_text('Мы остановились на этом вопросе.')
        expect(page.locator('#chat-log')).to_contain_text('Покажи один следующий шаг.')
        assert not writes['sessions'] and not writes['chat'] and not ui_server.rag.calls
        send(page, 'а что напечатает print?')
        assert writes['chat'][0]['learning_session_id'] == payload['learning_session_id']
        assert ui_server.rag.last_answer_kwargs['subject_id'] == 'python'
        assert ui_server.rag.last_answer_kwargs['topic_id'] == PYTHON
        assert not writes['sessions']
    finally:
        page.close()


@pytest.mark.parametrize('outcome', ['pending', 'unknown_outcome'])
def test_unresolved_current_message_blocks_new_requests_and_refresh_is_read_only(chromium, ui_server, outcome):
    page = chromium.new_page()
    payload = active_payload(ui_server, unresolved=outcome)
    control = controlled_current(page, payload)
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#chat-log')).to_contain_text('Сообщение без подтверждённого результата.')
        expect(page.locator('#send-chat')).to_be_disabled()
        expect(page.locator('#chat-input')).to_be_disabled()
        expect(page.locator('#chat-mode')).to_be_disabled()
        expect(page.locator(NOTE)).to_be_visible()
        expect(page.locator(REFRESH)).to_be_enabled()
        if outcome == 'pending':
            expect(page.locator('#complete-learning-session')).to_be_disabled()
        else:
            expect(page.locator('#complete-learning-session')).to_be_enabled()
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls

        control['payload'] = {**payload, 'unresolved_status':None, 'messages':[
            {'role':'user','content':'Сообщение без подтверждённого результата.','delivery_status':'completed'},
            {'role':'assistant','content':'Сохранённый результат готов.','delivery_status':'completed'}]}
        with page.expect_response(is_current):
            page.locator(REFRESH).click()
        expect(page.locator('#send-chat')).to_be_enabled()
        expect(page.locator('#chat-log')).to_contain_text('Сохранённый результат готов.')
        assert len(control['reads']) >= 2
        assert not writes['chat'] and not writes['sessions'] and not writes['complete']
        assert not ui_server.rag.calls
    finally:
        page.close()


def test_unknown_outcome_can_be_closed_explicitly_without_replaying_a_request(chromium, ui_server):
    page = chromium.new_page()
    payload = active_payload(ui_server, unresolved='unknown_outcome')
    control = controlled_current(page, payload)
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#complete-learning-session')).to_be_enabled()
        control['payload'] = {'available':False}
        with page.expect_response(lambda response: response.url.endswith('/complete')) as completed:
            page.locator('#complete-learning-session').click()
        assert completed.value.status == 202
        expect(page.locator('#send-chat')).to_be_enabled()
        assert ui_server.store.learning_sessions[payload['learning_session_id']]['status'] == 'completed'
        assert len(writes['complete']) == 1
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls
    finally:
        page.close()


def test_truncated_saved_history_is_explained_without_generating_more_text(chromium, ui_server):
    page = chromium.new_page()
    payload = active_payload(ui_server, truncated=True)
    controlled_current(page, payload)
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#chat-log')).to_contain_text('Мы остановились на этом вопросе.')
        expect(page.locator(NOTE)).to_be_visible()
        expect(page.locator(NOTE)).to_contain_text(re.compile('часть|последн|ранн|не вся', re.IGNORECASE))
        expect(page.locator('#send-chat')).to_be_enabled()
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls
    finally:
        page.close()


def test_no_current_session_keeps_the_goal_and_normal_new_conversation(chromium, ui_server):
    page = chromium.new_page()
    controlled_current(page, {'available':False})
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        expect(page.locator('#chat-topic')).to_be_enabled()
        expect(page.locator('#chat-mode')).to_have_value('chat')
        expect(page.locator('#send-chat')).to_be_enabled()
        expect(page.locator('#chat-log')).to_contain_text('Привет!')
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls
    finally:
        page.close()


def test_failed_current_read_stays_closed_until_explicit_refresh_succeeds(chromium, ui_server):
    page = chromium.new_page()
    control = controlled_current(page, {'detail':'synthetic database unavailable'}, status=503)
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current) as failed:
            _login(page, ui_server.base_url, 'student.ui@example.test')
        assert failed.value.status == 503
        expect(page.locator('#send-chat')).to_be_disabled()
        expect(page.locator('#chat-input')).to_be_disabled()
        expect(page.locator('#chat-mode')).to_be_disabled()
        expect(page.locator(NOTE)).to_be_visible()
        expect(page.locator(REFRESH)).to_be_enabled()
        assert not writes['chat'] and not writes['sessions']
        control.update(status=200, payload={'available':False})
        with page.expect_response(is_current):
            page.locator(REFRESH).click()
        expect(page.locator('#send-chat')).to_be_enabled()
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls
    finally:
        page.close()


def test_loading_current_read_disables_sending_until_the_read_completes(chromium, ui_server):
    page = chromium.new_page()
    held = []
    page.route('**/learning-sessions/current', lambda route: held.append(route))
    writes = watch_writes(page)
    try:
        with page.expect_request(lambda request: request.url.endswith(CURRENT)):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#send-chat')).to_be_disabled()
        expect(page.locator('#chat-input')).to_be_disabled()
        expect(page.locator('#chat-mode')).to_be_disabled()
        assert held and not writes['chat'] and not writes['sessions']
        for route in held:
            route.fulfill(status=200, content_type='application/json', body='{"available":false}')
        held.clear()
        expect(page.locator('#send-chat')).to_be_enabled()
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls
    finally:
        for route in held:
            route.abort()
        page.close()


@pytest.mark.parametrize('old_status', [200, 401])
def test_late_current_read_cannot_restore_old_history_after_a_new_login(chromium, ui_server, old_status):
    page = chromium.new_page()
    held = []
    writes = watch_writes(page)

    def respond(route):
        if '/learners/demo-nika-04/' in route.request.url:
            held.append(route)
        else:
            route.fulfill(status=200, content_type='application/json', body='{"available":false}')

    page.route('**/learning-sessions/current', respond)
    try:
        with page.expect_request(lambda request: request.url.endswith(CURRENT)):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#learner-name')).to_contain_text('Ника')
        page.locator('#logout').click()
        expect(page.locator('#login-view')).to_be_visible()
        page.get_by_label('Электронная почта').fill('other.student@example.test')
        page.get_by_label('Пароль').fill('test-only passphrase 123')
        with page.expect_response(is_current):
            page.get_by_role('button', name='Войти', exact=True).click()
        expect(page.locator('#learner-name')).to_have_text(ui_server.store.cards['demo-luna-03']['pseudonym'])
        expect(page.locator('#send-chat')).to_be_enabled()
        old = {'available':True,'learning_session_id':'private-old-session','topic_id':PYTHON,
               'topic_title':'Вывод данных в Python','subject_id':'python','status':'active',
               'messages':[{'role':'user','content':'PRIVATE OLD CONVERSATION','delivery_status':'completed'}],
               'history_truncated':False,'unresolved_status':None}
        for route in held:
            route.fulfill(status=old_status, content_type='application/json',
                          body=json.dumps(old if old_status == 200 else {'detail':'old login expired'}))
        held.clear()
        settle(page)
        expect(page.locator('#workspace')).to_be_visible()
        expect(page.locator('#login-view')).to_be_hidden()
        expect(page.locator('#learner-name')).to_have_text(ui_server.store.cards['demo-luna-03']['pseudonym'])
        expect(page.locator('#chat-log')).not_to_contain_text('PRIVATE OLD CONVERSATION')
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls
    finally:
        for route in held:
            route.abort()
        page.close()


def test_legacy_unknown_topic_uses_a_friendly_title_and_does_not_leak_into_a_new_conversation(chromium, ui_server):
    page = chromium.new_page()
    legacy_topic = 'legacy.private-import-topic'
    session = ui_server.store.create_learning_session('demo-nika-04', 'student-ui', legacy_topic, 'seed-legacy')
    sid = session['learning_session_id']
    ui_server.store.reserve_chat_message(sid, 'demo-nika-04', 'legacy-message', 'ну я забыл', 'student-ui')
    ui_server.store.complete_chat_message(sid, 'legacy-message', {
        'choices':[{'message':{'role':'assistant', 'content':'Начнём с одного вопроса.'}}]})
    writes = watch_writes(page)
    try:
        with page.expect_response(is_current) as restored:
            _login(page, ui_server.base_url, 'student.ui@example.test')
        # Exercise the real API fallback as well as its temporary UI options.
        assert restored.value.status == 200
        assert restored.value.json()['subject_id'] == 'general'
        assert restored.value.json()['topic_title'] == 'Прошлое занятие'
        expect(page.locator('#chat-subject')).to_have_value('general')
        expect(page.locator('#chat-topic')).to_have_value(legacy_topic)
        expect(page.locator('#chat-topic option:checked')).to_have_text('Прошлое занятие')
        expect(page.locator('#learning-session-status')).to_contain_text('Прошлое занятие')
        expect(page.locator('#learning-session-status')).not_to_contain_text(legacy_topic)
        expect(page.locator('#chat-log')).to_contain_text('ну я забыл')
        expect(page.locator('#chat-log')).to_contain_text('Начнём с одного вопроса.')
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls

        with page.expect_response(lambda response: response.url.endswith('/complete')) as completed:
            page.locator('#complete-learning-session').click()
        assert completed.value.status == 202
        expect(page.locator('#send-chat')).to_be_enabled()
        expect(page.locator('#chat-subject')).to_be_enabled()
        expect(page.locator('#chat-subject option[data-restored-subject]')).to_have_count(0)
        expect(page.locator('#chat-subject')).not_to_have_value('general')
        expect(page.locator(f'#chat-topic option[value="{legacy_topic}"]')).to_have_count(0)
        assert ui_server.store.learning_sessions[sid]['status'] == 'completed'

        page.locator('#chat-subject').select_option('python')
        page.locator('#chat-topic').select_option(PYTHON)
        send(page, 'ну а print что делает?')
        assert len(writes['sessions']) == len(writes['chat']) == 1
        assert writes['sessions'][0]['topic_id'] == PYTHON
        assert writes['chat'][0]['learning_session_id'] != sid
        assert writes['chat'][0]['metadata']['subject_id'] == 'python'
        assert ui_server.rag.last_answer_kwargs['subject_id'] == 'python'
        assert ui_server.rag.last_answer_kwargs['topic_id'] == PYTHON
        expect(page.locator('#chat-log')).not_to_contain_text('ну я забыл')
    finally:
        page.close()


def test_delayed_completion_blocks_refresh_and_new_messages_until_the_session_is_closed(chromium, ui_server):
    page = chromium.new_page()
    payload = active_payload(ui_server)
    control = controlled_current(page, payload)
    writes = watch_writes(page)
    held = []
    page.route('**/learning-sessions/*/complete', lambda route: held.append(route))
    try:
        with page.expect_response(is_current):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#complete-learning-session')).to_be_enabled()
        with page.expect_request(lambda request: request.url.endswith('/complete')):
            page.locator('#complete-learning-session').click()
        assert len(held) == 1
        for selector in (REFRESH, '#chat-input', '#send-chat', '#chat-mode', '#chat-topic',
                         '#chat-subject', '#complete-learning-session'):
            expect(page.locator(selector)).to_be_disabled()
        expect(page.locator(NOTE)).to_contain_text('Сохраняем завершение занятия')
        prior_reads = len(control['reads'])
        # Deliberately bypass disabled controls to check the action guards too.
        page.locator(REFRESH).evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        page.locator('#chat-input').evaluate("el => { el.value = 'ещё вопрос во время завершения'; }")
        page.locator('#chat-form').evaluate("el => el.dispatchEvent(new Event('submit', {bubbles:true, cancelable:true}))")
        settle(page)
        assert len(control['reads']) == prior_reads
        assert len(writes['complete']) == 1
        assert not writes['chat'] and not writes['sessions'] and not ui_server.rag.calls

        sid = payload['learning_session_id']
        result = ui_server.store.complete_learning_session(sid, 'demo-nika-04', 'student-ui')
        control['payload'] = {'available':False}
        held.pop().fulfill(status=202, content_type='application/json', body=json.dumps(result))
        expect(page.locator('#send-chat')).to_be_enabled()
        expect(page.locator('#chat-mode')).to_be_enabled()
        assert ui_server.store.learning_sessions[sid]['status'] == 'completed'
        send(page, 'теперь начнём заново')
        assert len(writes['sessions']) == len(writes['chat']) == 1
        assert writes['chat'][0]['learning_session_id'] != sid
    finally:
        for route in held:
            route.abort()
        page.close()
