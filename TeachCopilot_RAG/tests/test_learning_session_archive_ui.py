"""Completed conversations are read without resuming or replaying their calls.

All data and generated replies stay in the synthetic browser fixture. Archive
transport is controlled for pagination, failures, and delayed-response races.
"""
from copy import deepcopy
import json
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("playwright.sync_api", reason="install requirements-dev-local.txt for browser acceptance tests")
from playwright.sync_api import expect  # noqa: E402

from test_ui_acceptance import _login, ui_server  # noqa: F401


LEARNER = 'demo-nika-04'
FRACTIONS = 'math.g4.fractions.numerator_denominator'
PYTHON = 'programming.python.output'
A = 'bc8e9fb2-7010-4f7e-9ae5-4d75e381a5b1'
B = 'bc8e9fb2-7010-4f7e-9ae5-4d75e381a5b2'
ARCHIVE = '#learning-archive'
LIST = '#learning-archive-list'
LOG = '#learning-archive-log'


def listed(sid=A, topic=FRACTIONS, title='Дробь, числитель и знаменатель'):
    return {'learning_session_id':sid, 'topic_id':topic, 'topic_title':title,
            'subject_id':'python' if topic == PYTHON else 'math',
            'completed_at':'2026-10-21T12:00:00+00:00'}


def history(sid=A, *, messages=None, next_cursor=None, topic=FRACTIONS, title='Дробь, числитель и знаменатель'):
    return {**{key:value for key,value in listed(sid, topic, title).items() if key != 'completed_at'},
            'status':'completed', 'messages':messages or [
                {'role':'user', 'content':'ну я хз', 'delivery_status':'completed'},
                {'role':'assistant', 'content':'Посмотри на одну часть рисунка.'}],
            'next_cursor':next_cursor}


def transport(page, *, rows=None, detail=None, list_pages=None, detail_pages=None):
    control = {'list_status':200, 'list_pages':list_pages or {None:{'sessions':rows or [], 'next_cursor':None}},
               'detail_status':200, 'detail_pages':detail_pages or {(A,None):detail or history()},
               'list_reads':[], 'history_reads':[]}

    def respond_list(route):
        if route.request.method != 'GET':
            route.fallback()
            return
        control['list_reads'].append(route.request.url)
        before = parse_qs(urlsplit(route.request.url).query).get('before', [None])[0]
        body = control['list_pages'].get(before, {'detail':'unexpected list cursor'})
        route.fulfill(status=control['list_status'], content_type='application/json', body=json.dumps(body))

    def respond_history(route):
        control['history_reads'].append(route.request.url)
        url = urlsplit(route.request.url)
        sid = url.path.split('/')[-2]
        before = parse_qs(url.query).get('before', [None])[0]
        body = control['detail_pages'].get((sid,before), {'detail':'unexpected history cursor'})
        route.fulfill(status=control['detail_status'], content_type='application/json', body=json.dumps(body))

    page.route('**/learning-sessions', respond_list)
    page.route('**/learning-sessions?*', respond_list)
    page.route('**/learning-sessions/*/history*', respond_history)
    return control


def watch_writes(page):
    writes = []
    page.on('request', lambda request: writes.append(request) if request.method == 'POST' and
            ('/v1/chat/completions' in request.url or '/learning-sessions' in request.url) else None)
    return writes


def login(page, ui_server):
    _login(page, ui_server.base_url, 'student.ui@example.test')
    expect(page.locator('#send-chat')).to_be_enabled()


def open_archive(page):
    with page.expect_response(lambda response: urlsplit(response.url).path.endswith('/learning-sessions') and response.request.method == 'GET'):
        page.locator(ARCHIVE+' > summary').click()


def open_row(page, index=0):
    with page.expect_response(lambda response: '/history' in response.url):
        page.locator(LIST+' button').nth(index).click()


def settle(page):
    page.evaluate('() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')


def seed_completed(store, sid=A, *, topic=FRACTIONS, user='ну я хз', assistant='Посмотри на одну часть рисунка.'):
    store.learning_sessions[sid] = {'learning_session_id':sid, 'learner_id':LEARNER, 'actor_id':'student-ui',
        'topic_id':topic, 'client_session_id':'seed-'+sid, 'status':'active',
        'created_at':'2026-10-21T10:00:00+00:00', 'completed_at':None}
    store.learning_turns[sid] = []
    store.reserve_chat_message(sid, LEARNER, 'archive-seed', user, 'student-ui')
    store.complete_chat_message(sid, 'archive-seed', {'choices':[{'message':{'role':'assistant','content':assistant}}],
        'reasoning_content':'PRIVATE cached reasoning', '_guided_division_provenance':{'private':'PRIVATE cursor'},
        'diagnostics':'PRIVATE provider log'})
    store.complete_learning_session(sid, LEARNER, 'student-ui')
    store.learning_sessions[sid]['completed_at'] = '2026-10-21T12:00:00+00:00'


def test_archive_is_lazy_and_empty_archive_does_not_create_a_session(chromium, ui_server):
    page = chromium.new_page()
    control = transport(page)
    writes = watch_writes(page)
    try:
        login(page, ui_server)
        settle(page)
        assert not control['list_reads'] and not control['history_reads']
        open_archive(page)
        expect(page.locator('#learning-archive-status')).to_contain_text('нет')
        expect(page.locator(LIST+' button')).to_have_count(0)
        expect(page.locator('#learning-archive-view')).to_be_hidden()
        assert len(control['list_reads']) == 1 and not writes and not ui_server.rag.calls
        assert not ui_server.store.learning_sessions
    finally:
        page.close()


def test_real_completed_sources_are_shown_literally_without_private_cache_or_writes(chromium, ui_server):
    page = chromium.new_page()
    user = 'хз <img src=x onerror="window.__archive_xss=1">'
    assistant = 'Вот исходный ответ.\nБез изменения текста.'
    seed_completed(ui_server.store, user=user, assistant=assistant)
    writes = watch_writes(page)
    before = deepcopy(ui_server.store.learning_messages)
    try:
        login(page, ui_server)
        open_archive(page)
        expect(page.locator(LIST)).to_contain_text('Дробь, числитель и знаменатель')
        open_row(page)
        expect(page.locator(LOG)).to_contain_text(user)
        expect(page.locator(LOG)).to_contain_text(assistant)
        expect(page.locator('#learning-archive-title')).to_contain_text('Дробь, числитель и знаменатель')
        assert page.locator(LOG+' img').count() == 0
        assert page.evaluate('window.__archive_xss') is None
        assert 'PRIVATE' not in page.locator(LOG).inner_text()
        assert not writes and not ui_server.rag.calls
        assert ui_server.store.learning_messages == before
        assert ui_server.store.learning_sessions[A]['status'] == 'completed'
        page.locator('#close-learning-archive').click()
        expect(page.locator('#learning-archive-view')).to_be_hidden()
        expect(page.locator('#send-chat')).to_be_enabled()
    finally:
        page.close()


def test_archive_does_not_change_the_active_topic_draft_mode_or_session(chromium, ui_server):
    page = chromium.new_page()
    seed_completed(ui_server.store, topic=PYTHON)
    current = ui_server.store.create_learning_session(LEARNER, 'student-ui', FRACTIONS, 'current-during-archive')
    sid = current['learning_session_id']
    ui_server.store.reserve_chat_message(sid, LEARNER, 'current-message', 'Текущий вопрос про дробь', 'student-ui')
    ui_server.store.complete_chat_message(sid, 'current-message', {'choices':[{'message':{
        'role':'assistant', 'content':'Продолжаем эту дробь.'}}]})
    writes = watch_writes(page)
    try:
        login(page, ui_server)
        page.locator('#chat-input').fill('ага, текущий черновик')
        open_archive(page)
        open_row(page)
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        expect(page.locator('#chat-subject')).to_have_value('math')
        expect(page.locator('#chat-mode')).to_have_value('chat')
        expect(page.locator('#chat-input')).to_have_value('ага, текущий черновик')
        expect(page.locator('#chat-log')).to_contain_text('Текущий вопрос про дробь')
        assert not writes and not ui_server.rag.calls
        with page.expect_response(lambda response: response.url.endswith('/v1/chat/completions')) as sent:
            page.locator('#send-chat').click()
        assert sent.value.status == 200
        assert len(writes) == 1 and writes[0].post_data_json['learning_session_id'] == sid
        assert len(ui_server.rag.calls) == 1
        expect(page.locator('#learning-archive-view')).to_be_visible()
        assert ui_server.store.learning_sessions[A]['status'] == 'completed'
    finally:
        page.close()


def test_history_pagination_prepends_older_turns_in_order_without_duplicate_requests(chromium, ui_server):
    page = chromium.new_page()
    latest = history(messages=[{'role':'user','content':'вопрос 21'}, {'role':'assistant','content':'ответ 21'}], next_cursor='older-turns')
    older = history(messages=[{'role':'user','content':'вопрос 1'}, {'role':'assistant','content':'ответ 1'}])
    control = transport(page, rows=[listed()], detail_pages={(A,None):latest,(A,'older-turns'):older})
    writes = watch_writes(page)
    try:
        login(page, ui_server)
        open_archive(page)
        open_row(page)
        expect(page.locator(LOG)).not_to_contain_text('вопрос 1')
        with page.expect_response(lambda response: 'before=older-turns' in response.url):
            page.locator('#older-learning-messages').click()
        expect(page.locator(LOG)).to_contain_text('вопрос 1')
        text = page.locator(LOG).inner_text()
        assert text.index('вопрос 1') < text.index('ответ 1') < text.index('вопрос 21') < text.index('ответ 21')
        expect(page.locator('#older-learning-messages')).to_be_hidden()
        assert len(control['history_reads']) == 2 and not writes and not ui_server.rag.calls
    finally:
        page.close()


def test_completed_session_list_loads_the_next_page_without_changing_chat(chromium, ui_server):
    page = chromium.new_page()
    control = transport(page, list_pages={
        None:{'sessions':[listed()], 'next_cursor':'older-sessions'},
        'older-sessions':{'sessions':[listed(B,PYTHON,'Вывод данных в Python')], 'next_cursor':None}})
    writes = watch_writes(page)
    try:
        login(page, ui_server)
        open_archive(page)
        with page.expect_response(lambda response: 'before=older-sessions' in response.url):
            page.locator('#more-learning-sessions').click()
        expect(page.locator(LIST+' button')).to_have_count(2)
        expect(page.locator(LIST)).to_contain_text('Вывод данных в Python')
        expect(page.locator('#more-learning-sessions')).to_be_hidden()
        expect(page.locator('#chat-topic')).to_have_value(FRACTIONS)
        assert len(control['list_reads']) == 2 and not control['history_reads']
        assert not writes and not ui_server.rag.calls
    finally:
        page.close()


def test_list_and_history_failures_can_be_retried_with_reads_only(chromium, ui_server):
    page = chromium.new_page()
    control = transport(page, rows=[listed()])
    control['list_status'] = 503
    writes = watch_writes(page)
    try:
        login(page, ui_server)
        open_archive(page)
        expect(page.locator('#learning-archive-status')).not_to_be_empty()
        expect(page.locator('#refresh-learning-archive')).to_be_enabled()
        expect(page.locator('#send-chat')).to_be_enabled()
        control['list_status'] = 200
        with page.expect_response(lambda response: urlsplit(response.url).path.endswith('/learning-sessions')):
            page.locator('#refresh-learning-archive').click()
        expect(page.locator(LIST+' button')).to_have_count(1)
        control['detail_status'] = 503
        open_row(page)
        expect(page.locator(LOG)).not_to_contain_text('ну я хз')
        expect(page.locator('#send-chat')).to_be_enabled()
        control['detail_status'] = 200
        open_row(page)
        expect(page.locator(LOG)).to_contain_text('ну я хз')
        assert len(control['list_reads']) == 2 and len(control['history_reads']) == 2
        assert not writes and not ui_server.rag.calls
    finally:
        page.close()


def test_late_history_of_previous_selection_cannot_replace_the_selected_conversation(chromium, ui_server):
    page = chromium.new_page()
    control = transport(page, rows=[listed(),listed(B,PYTHON,'Вывод данных в Python')], detail_pages={
        (B,None):history(B,messages=[{'role':'user','content':'выбранная переписка Python'}],topic=PYTHON,title='Вывод данных в Python')})
    writes = watch_writes(page)
    held = []
    page.route('**/learning-sessions/'+A+'/history', lambda route: held.append(route))
    try:
        login(page, ui_server)
        open_archive(page)
        with page.expect_request(lambda request: A+'/history' in request.url):
            page.locator(LIST+' button').nth(0).click()
        open_row(page,1)
        expect(page.locator(LOG)).to_contain_text('выбранная переписка Python')
        for route in held:
            route.fulfill(status=200,content_type='application/json',body=json.dumps(history(messages=[{'role':'user','content':'OLD SELECTION'}])))
        held.clear()
        settle(page)
        expect(page.locator(LOG)).to_contain_text('выбранная переписка Python')
        expect(page.locator(LOG)).not_to_contain_text('OLD SELECTION')
        assert not writes and not ui_server.rag.calls
    finally:
        for route in held:
            route.abort()
        page.close()


def test_late_older_page_cannot_prepend_messages_to_a_different_conversation(chromium, ui_server):
    page = chromium.new_page()
    control = transport(page, rows=[listed(),listed(B,PYTHON,'Вывод данных в Python')], detail_pages={
        (A,None):history(next_cursor='older-a'),
        (B,None):history(B,messages=[{'role':'user','content':'только Python'}],topic=PYTHON,title='Вывод данных в Python')})
    held = []
    writes = watch_writes(page)
    page.route('**/history?before=older-a', lambda route: held.append(route))
    try:
        login(page, ui_server)
        open_archive(page)
        open_row(page)
        with page.expect_request(lambda request: 'before=older-a' in request.url):
            page.locator('#older-learning-messages').click()
        open_row(page,1)
        expect(page.locator(LOG)).to_contain_text('только Python')
        for route in held:
            route.fulfill(status=200,content_type='application/json',body=json.dumps(history(messages=[{'role':'user','content':'PRIVATE OTHER PAGE'}])))
        held.clear()
        settle(page)
        expect(page.locator(LOG)).not_to_contain_text('PRIVATE OTHER PAGE')
        expect(page.locator(LOG)).to_contain_text('только Python')
        assert not writes and not ui_server.rag.calls
    finally:
        for route in held:
            route.abort()
        page.close()


@pytest.mark.parametrize('old_status',[200,401])
def test_late_archive_history_cannot_reappear_or_log_out_a_new_account(chromium, ui_server, old_status):
    page = chromium.new_page()
    transport(page, rows=[listed()])
    held = []
    writes = watch_writes(page)
    page.route('**/learning-sessions/'+A+'/history', lambda route: held.append(route))
    try:
        login(page, ui_server)
        open_archive(page)
        with page.expect_request(lambda request: A+'/history' in request.url):
            page.locator(LIST+' button').click()
        page.locator('#logout').click()
        expect(page.locator('#login-view')).to_be_visible()
        page.get_by_label('Электронная почта').fill('other.student@example.test')
        page.get_by_label('Пароль').fill('test-only passphrase 123')
        page.get_by_role('button',name='Войти',exact=True).click()
        expect(page.locator('#learner-name')).to_have_text(ui_server.store.cards['demo-luna-03']['pseudonym'])
        expect(page.locator('#send-chat')).to_be_enabled()
        for route in held:
            route.fulfill(status=old_status,content_type='application/json',body=json.dumps(
                history(messages=[{'role':'user','content':'PRIVATE PREVIOUS ACCOUNT'}]) if old_status == 200 else {'detail':'old login expired'}))
        held.clear()
        settle(page)
        expect(page.locator('#workspace')).to_be_visible()
        expect(page.locator('#login-view')).to_be_hidden()
        expect(page.locator(LOG)).not_to_contain_text('PRIVATE PREVIOUS ACCOUNT')
        expect(page.locator('#learning-archive-view')).to_be_hidden()
        assert not writes and not ui_server.rag.calls
    finally:
        for route in held:
            route.abort()
        page.close()


def test_late_list_cannot_fill_the_new_accounts_archive(chromium, ui_server):
    page = chromium.new_page()
    held = []
    writes = watch_writes(page)
    page.route('**/learners/'+LEARNER+'/learning-sessions', lambda route: held.append(route))
    try:
        login(page,ui_server)
        with page.expect_request(lambda request: request.url.endswith('/learning-sessions')):
            page.locator(ARCHIVE+' > summary').click()
        page.locator('#logout').click()
        expect(page.locator('#login-view')).to_be_visible()
        page.get_by_label('Электронная почта').fill('other.student@example.test')
        page.get_by_label('Пароль').fill('test-only passphrase 123')
        page.get_by_role('button',name='Войти',exact=True).click()
        expect(page.locator('#learner-name')).to_have_text(ui_server.store.cards['demo-luna-03']['pseudonym'])
        for route in held:
            route.fulfill(status=200,content_type='application/json',body=json.dumps({'sessions':[listed(title='PRIVATE OLD LIST')],'next_cursor':None}))
        held.clear()
        settle(page)
        expect(page.locator(LIST)).not_to_contain_text('PRIVATE OLD LIST')
        assert not writes and not ui_server.rag.calls
    finally:
        for route in held:
            route.abort()
        page.close()
