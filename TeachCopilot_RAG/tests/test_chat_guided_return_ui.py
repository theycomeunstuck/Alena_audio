"""Ordinary chat offers an explicit return to a checked division step.

The browser and session routes are real. Action/model transport is controlled,
so these UI checks do not call inference or write to a live learner database.
"""
from copy import deepcopy
import json

import pytest
from playwright.sync_api import expect

from local_app.guided_division import render_prompt, start_state
from test_guided_division_ui import click_action, install_transport
from test_ui_acceptance import _login, ui_server  # noqa: F401


TOPIC = 'math.g4.numbers.division_by_1_2_digit'
TITLE = 'Деление на однозначное и двузначное число'
BOOKMARK = '#chat-guided-return'
RETURN = '#chat-guided-return-button'
FRACTIONS = 'math.g4.fractions.numerator_denominator'


def _settle_browser(page):
    # Allow the fetch JSON continuation and its rendering to run before a
    # negative visibility assertion; an already-hidden card is not enough.
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')


def _start_zero_step(page, transport):
    page.locator('#chat-mode').select_option('division_guided')
    expect(page.locator('#guided-division-status')).to_contain_text('Сохранённого шага')
    page.locator('#action-dividend').fill('3636')
    page.locator('#action-divisor').fill('12')
    click_action(page, '#send-learning-action')
    for number in ('3', '0', '3'):
        page.locator('#guided-division-answer').fill(number)
        click_action(page, '#guided-division-check')
    assert (transport['state']['step_index'], transport['state']['phase']) == (1, 'quotient_digit')
    page.locator('#chat-mode').select_option('chat')
    expect(page.locator('#chat-form')).to_be_visible()


def _assert_bookmark(page, cursor):
    expect(page.locator(BOOKMARK)).to_be_visible()
    expect(page.locator('#chat-guided-return-label')).to_contain_text('3636 : 12')
    expect(page.locator(BOOKMARK)).to_contain_text(TITLE)
    expect(page.locator(RETURN)).to_have_text('Вернуться к примеру')
    expect(page.locator(RETURN)).to_be_enabled()
    text = page.locator(BOOKMARK).inner_text()
    assert '303' not in text
    assert cursor['state_id'] not in text
    assert 'reviewed-previous-session' not in text
    assert TOPIC not in text


@pytest.mark.parametrize('aside', ['хз', 'я в майнкрафте строю дом'])
def test_chat_bookmark_resumes_same_checked_step_without_automatic_action(chromium, ui_server, aside):
    page = chromium.new_page(viewport={'width':1365, 'height':1050})
    transport = install_transport(page)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        _start_zero_step(page, transport)
        saved = deepcopy(transport['state'])
        session_id = transport['session']
        typed_calls = len(transport['sent'])
        page.locator('#chat-input').fill(aside)
        click_action(page, '#send-chat')
        _assert_bookmark(page, saved)
        assert transport['state'] == saved
        assert len(transport['sent']) == typed_calls
        assert len(transport['chat_sent']) == 1
        assert len(ui_server.store.learning_sessions) == 1

        click_action(page, RETURN)
        expect(page.locator('#chat-mode')).to_have_value('division_guided')
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        expect(page.locator(BOOKMARK)).to_be_hidden()
        assert transport['sent'][-1]['learning_action'] == {
            'operation':'division_guided_resume', 'expected_state_id':saved['state_id']}
        assert transport['sent'][-1]['learning_session_id'] == session_id
        assert (transport['state']['step_index'], transport['state']['phase']) == (1, 'quotient_digit')
        page.locator('#guided-division-answer').fill('0')
        click_action(page, '#guided-division-check')
        assert (transport['state']['step_index'], transport['state']['phase']) == (1, 'remainder')
        assert len(ui_server.store.learning_sessions) == 1
    finally:
        page.close()


def test_chat_bookmark_restores_after_reload_without_selecting_guided_mode(chromium, ui_server):
    page = chromium.new_page()
    transport = install_transport(page)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        _start_zero_step(page, transport)
        page.locator('#chat-input').fill('ага')
        click_action(page, '#send-chat')
        saved = deepcopy(transport['state'])
        session_id = transport['session']
        typed_calls = len(transport['sent'])
        page.reload()
        expect(page.locator('#learner-name')).to_contain_text('Ника')
        expect(page.locator('#chat-mode')).to_have_value('chat')
        _assert_bookmark(page, saved)
        assert transport['state'] == saved
        assert len(transport['sent']) == typed_calls
        click_action(page, RETURN)
        assert transport['sent'][-1]['learning_session_id'] == session_id
        assert len(ui_server.store.learning_sessions) == 1
    finally:
        page.close()


def test_reviewed_chat_bookmark_requires_explicit_cross_session_resume(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636, 12, state_id='private-reviewed-bookmark-state')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page, cursor, resume_required=True)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator(f'#chat-topic option[value="{TOPIC}"]')).to_have_count(1)
        page.locator('#chat-topic').select_option(TOPIC)
        expect(page.locator('#chat-mode')).to_have_value('chat')
        _assert_bookmark(page, cursor)
        expect(page.locator('#chat-topic')).to_have_value(TOPIC)
        assert not transport['sent']
        assert not ui_server.store.learning_sessions
        click_action(page, RETURN)
        assert transport['sent'][0]['learning_action'] == {
            'operation':'division_guided_resume', 'expected_state_id':cursor['state_id'],
            'source_learning_session_id':'reviewed-previous-session'}
        assert ui_server.store.learning_sessions[transport['session']]['topic_id'] == TOPIC
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        assert (transport['state']['step_index'], transport['state']['phase']) == (1, 'quotient_digit')
    finally:
        page.close()


def test_chat_discovers_bookmark_without_changing_default_fraction_topic(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636, 12, state_id='private-reviewed-fractions-bookmark')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page, cursor)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        expect(page.locator('#chat-mode')).to_have_value('chat')
        _assert_bookmark(page, cursor)
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        assert not transport['sent'] and not ui_server.store.learning_sessions

        # Choosing another topic is an explicit decision to leave this bookmark.
        page.locator('#chat-topic').select_option('math')
        expect(page.locator(BOOKMARK)).to_be_hidden()
        assert not transport['sent'] and not ui_server.store.learning_sessions
    finally:
        page.close()


def test_chat_return_rechecks_server_context_before_creating_a_session(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636, 12, state_id='private-bookmark-revoked-before-click')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page, cursor)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        _assert_bookmark(page, cursor)
        transport['unavailable_reason'] = 'corrected_guided_source'
        with page.expect_response(lambda response: '/guided-division?' in response.url):
            page.locator(RETURN).click()
        _settle_browser(page)
        expect(page.locator(BOOKMARK)).to_be_hidden()
        expect(page.locator('#chat-mode')).to_have_value('chat')
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        assert not transport['sent'] and not ui_server.store.learning_sessions
    finally:
        page.close()


def test_changed_chat_bookmark_requires_a_second_explicit_click(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636, 12, state_id='private-first-bookmark-state')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page, cursor)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        _assert_bookmark(page, cursor)
        replacement = start_state(1224, 4, state_id='private-replacement-bookmark-state')
        replacement['status'] = 'paused'
        transport['state'] = deepcopy(replacement)
        with page.expect_response(lambda response: '/guided-division?' in response.url):
            page.locator(RETURN).click()
        expect(page.locator('#chat-guided-return-label')).to_contain_text('1224 : 4')
        expect(page.locator(RETURN)).to_be_enabled()
        expect(page.locator('#chat-mode')).to_have_value('chat')
        assert not transport['sent'] and not ui_server.store.learning_sessions
        assert transport['state'] == replacement

        click_action(page, RETURN)
        assert transport['sent'][0]['learning_action'] == {
            'operation':'division_guided_resume', 'expected_state_id':replacement['state_id'],
            'source_learning_session_id':'reviewed-previous-session'}
        expect(page.locator('#guided-division-example')).to_have_text('1224 : 4')
        assert (transport['state']['step_index'], transport['state']['phase']) == (0, 'quotient_digit')
    finally:
        page.close()


def test_double_click_on_chat_return_sends_only_one_resume(chromium, ui_server):
    page = chromium.new_page()
    cursor = start_state(3636, 12, state_id='private-double-click-bookmark')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    transport = install_transport(page, cursor)
    held = []
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        _assert_bookmark(page, cursor)
        expect(page.locator(f'#chat-topic option[value="{TOPIC}"]')).to_have_count(1)
        page.route('**/guided-division?*', lambda route: held.append(route))
        with page.expect_request(lambda request: '/guided-division?' in request.url):
            page.locator(RETURN).dblclick()
        expect(page.locator(RETURN)).to_be_disabled()
        _settle_browser(page)
        assert len(held) == 1
        assert not transport['sent'] and not ui_server.store.learning_sessions
        with page.expect_response(lambda response: response.url.endswith('/v1/chat/completions')):
            held.pop().fallback()
        expect(page.locator('#guided-division-answer-form')).to_be_visible()
        assert len(transport['sent']) == 1
        assert transport['sent'][0]['learning_action']['operation'] == 'division_guided_resume'
        assert len(ui_server.store.learning_sessions) == 1
    finally:
        for route in held:
            route.abort()
        page.close()


@pytest.mark.parametrize('reason', [
    'exercise_completed', 'invalid_guided_chain', 'corrected_guided_source',
    'message_outcome_unavailable', 'teacher_review_required', 'another_actor',
])
def test_unavailable_chat_context_does_not_offer_a_bookmark(chromium, ui_server, reason):
    page = chromium.new_page()
    cursor = None
    if reason == 'exercise_completed':
        cursor = start_state(3636, 12, state_id='private-completed-bookmark-state')
        cursor.update(step_index=2, phase='remainder', status='completed')
    transport = install_transport(page, cursor, unavailable_reason=reason)
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator(f'#chat-topic option[value="{TOPIC}"]')).to_have_count(1)
        with page.expect_response(lambda response: '/guided-division?' in response.url):
            page.locator('#chat-topic').select_option(TOPIC)
        _settle_browser(page)
        expect(page.locator(BOOKMARK)).to_be_hidden()
        assert not transport['sent']
        assert not ui_server.store.learning_sessions
        assert 'private-completed-bookmark-state' not in page.locator('#student-actions').inner_text()
    finally:
        page.close()


def test_chat_bookmark_is_not_actionable_while_message_outcome_is_pending(chromium, ui_server):
    page = chromium.new_page()
    transport = install_transport(page)
    held = []
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        _start_zero_step(page, transport)
        saved = deepcopy(transport['state'])
        _assert_bookmark(page, saved)

        def hold_ordinary(route):
            action = route.request.post_data_json.get('learning_action')
            if action:
                route.fallback()
            else:
                held.append(route)

        page.route('**/v1/chat/completions', hold_ordinary)
        with page.expect_request(lambda request: request.url.endswith('/v1/chat/completions')):
            page.locator('#chat-input').fill('погоди')
            page.locator('#send-chat').click()
        expect(page.locator('#send-chat')).to_be_disabled()
        assert not page.locator(RETURN).is_visible() or not page.locator(RETURN).is_enabled()
        assert transport['state'] == saved
        assert len(held) == 1
        with page.expect_response(lambda response: response.url.endswith('/v1/chat/completions')):
            held.pop().fallback()
        _assert_bookmark(page, saved)
    finally:
        for route in held:
            route.abort()
        page.close()


@pytest.mark.parametrize('leave', ['subject', 'logout'])
def test_late_guided_lookup_does_not_restore_bookmark_after_leaving_math(chromium, ui_server, leave):
    page = chromium.new_page()
    held = []
    transport = install_transport(page)
    cursor = start_state(3636, 12, state_id='private-late-bookmark-state')
    cursor.update(step_index=1, phase='quotient_digit', status='paused')
    page.route('**/guided-division?*', lambda route: held.append(route))
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator(f'#chat-topic option[value="{TOPIC}"]')).to_have_count(1)
        page.locator('#chat-topic').select_option(TOPIC)
        expect(page.locator('#chat-topic')).to_have_value(TOPIC)
        assert held
        if leave == 'subject':
            page.locator('#chat-subject').select_option('python')
            expect(page.locator('#chat-subject')).to_have_value('python')
        else:
            page.locator('#logout').click()
            expect(page.locator('#login-view')).to_be_visible()
        for route in held:
            route.fulfill(status=200, content_type='application/json', body=json.dumps({
                'available':True, 'origin':'approved_memory', 'guided_division':cursor,
                'source_learning_session_id':'private-reviewed-source-session',
                'prompt':render_prompt(cursor)}))
        held.clear()
        _settle_browser(page)
        expect(page.locator(BOOKMARK)).to_be_hidden()
        assert not transport['sent']
        assert not ui_server.store.learning_sessions
        if leave == 'subject':
            expect(page.locator('#chat-subject')).to_have_value('python')
            assert '3636 : 12' not in page.locator('#student-actions').inner_text()
        else:
            expect(page.locator('#workspace')).to_be_hidden()
    finally:
        for route in held:
            route.abort()
        page.close()


def test_old_bookmark_lookup_401_cannot_log_out_a_new_student(chromium, ui_server):
    page = chromium.new_page()
    held = []
    transport = install_transport(page)

    def hold_first_student_only(route):
        if '/learners/demo-nika-04/' in route.request.url:
            held.append(route)
        else:
            route.fulfill(status=200, content_type='application/json',
                          body=json.dumps({'available':False, 'reason':'no_learning_session'}))

    page.route('**/guided-division?*', hold_first_student_only)
    try:
        with page.expect_request(lambda request: '/guided-division?' in request.url):
            _login(page, ui_server.base_url, 'student.ui@example.test')
        expect(page.locator('#learner-name')).to_contain_text('Ника')
        assert held
        page.locator('#logout').click()
        expect(page.locator('#login-view')).to_be_visible()
        page.get_by_label('Электронная почта').fill('other.student@example.test')
        page.get_by_label('Пароль').fill('test-only passphrase 123')
        page.get_by_role('button', name='Войти', exact=True).click()
        expect(page.locator('#workspace')).to_be_visible()
        expect(page.locator('#learner-name')).to_have_text(ui_server.store.cards['demo-luna-03']['pseudonym'])

        with page.expect_response(lambda response: '/learners/demo-nika-04/' in response.url
                                  and '/guided-division?' in response.url and response.status == 401):
            for route in held:
                route.fulfill(status=401, content_type='application/json',
                              body=json.dumps({'detail':'expired previous login'}))
            held.clear()
        _settle_browser(page)
        expect(page.locator('#workspace')).to_be_visible()
        expect(page.locator('#login-view')).to_be_hidden()
        expect(page.locator('#learner-name')).to_have_text(ui_server.store.cards['demo-luna-03']['pseudonym'])
        expect(page.locator(BOOKMARK)).to_be_hidden()
        assert not transport['sent'] and not ui_server.store.learning_sessions
    finally:
        for route in held:
            route.abort()
        page.close()
