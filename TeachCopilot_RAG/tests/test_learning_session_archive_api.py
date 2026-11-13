"""Archived conversations are bounded canonical reads, never model replays."""
import base64
import json
from uuid import uuid4

from local_app.store import Store
from tests.test_current_learning_session_api import (
    LEARNER, TOPIC, _browser, _session, _turn, history_app,  # noqa: F401
)
from tests.test_memory_api_flow import flow_store  # noqa: F401


def _url(session_id=None, *, learner=LEARNER):
    root = f'/learners/{learner}/learning-sessions'
    return root if session_id is None else f'{root}/{session_id}/history'


def _get(app, session_id=None, *, before=None, headers=None, browser=None):
    response = (browser or app.browser).get(_url(session_id),
        headers=headers or app.student_headers, params={} if before is None else {'before':before})
    assert response.status_code == 200,response.text
    data = response.json()
    # Time fields (7 November time layer) are not part of what these tests pin. source_id is not ignored on
    # purpose: the learner-facing history must not return it.
    for message in data.get('messages', []):
        for key in ('activity_day', 'activity_timezone', 'created_at', 'event_timezone'):
            message.pop(key, None)
    return data


def _close(app, session_id, *, actor=None):
    return app.store.complete_learning_session(session_id,LEARNER,(actor or app.student)['id'])


def _decode(cursor):
    return json.loads(base64.urlsafe_b64decode(cursor+'='*(-len(cursor)%4)))


def _encode(payload):
    return base64.urlsafe_b64encode(json.dumps(payload,separators=(',',':')).encode()).decode().rstrip('=')


def test_archive_requires_access_and_only_the_same_actor_can_read_completed_history(history_app):
    app = history_app
    sid = _session(app)
    _turn(app,sid,'мой вопрос','мой ответ')
    assert _get(app) == {'sessions':[],'next_cursor':None}
    active = app.browser.get(_url(sid),headers=app.student_headers)
    assert active.status_code == 409
    _close(app,sid)
    teacher_sid = _session(app,actor=app.teacher)
    _turn(app,teacher_sid,'заметка учителя','ответ учителю',actor=app.teacher)
    _close(app,teacher_sid,actor=app.teacher)
    assert [s['learning_session_id'] for s in _get(app)['sessions']] == [sid]
    assert [s['learning_session_id'] for s in _get(app,headers=app.teacher_headers)['sessions']] == [teacher_sid]
    for path in (_url(),_url(sid)):
        assert app.browser.get(path).status_code == 401
        foreign = path.replace('/'+LEARNER+'/','/another-child/')
        assert app.browser.get(foreign,headers=app.student_headers).status_code == 403
        assert app.browser.get(foreign,headers=app.teacher_headers).status_code == 403
    assert app.browser.get(_url(sid),headers=app.teacher_headers).status_code == 403
    assert app.browser.get(_url(teacher_sid),headers=app.student_headers).status_code == 403
    assert app.browser.get(_url(str(uuid4())),headers=app.student_headers).status_code == 404
    assert app.browser.get(_url('not-a-uuid'),headers=app.student_headers).status_code == 422


def test_archive_list_keysets_are_stable_with_tied_timestamps_and_later_completions(history_app):
    app = history_app
    original = []
    for index in range(23):
        sid = _session(app)
        _turn(app,sid,f'вопрос {index}','ок')
        _close(app,sid)
        original.append(sid)
    # Tie-breaking by UUID is required; timestamps alone must not skip a page.
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_sessions SET completed_at='2026-01-01T00:00:00Z' WHERE learner_id=%s",(LEARNER,))
    first = _get(app)
    assert len(first['sessions']) == 10 and first['next_cursor']
    assert all(set(row) == {'learning_session_id','topic_id','topic_title','subject_id','session_mode','completed_at',
                            'created_at','event_timezone'}  # the last two come from the 7 November time layer
               for row in first['sessions'])
    assert all(row['session_mode'] == 'standard' for row in first['sessions'])
    assert all(row['topic_title'] == 'Дробь, числитель и знаменатель' for row in first['sessions'])
    later = _session(app)
    _close(app,later)
    pages = [first]
    while pages[-1]['next_cursor']:
        pages.append(_get(app,before=pages[-1]['next_cursor']))
    assert [len(page['sessions']) for page in pages] == [10,10,3]
    ids = [row['learning_session_id'] for page in pages for row in page['sessions']]
    assert ids == sorted(original,reverse=True)
    assert len(ids) == len(set(ids)) == 23 and later not in ids
    assert _get(app)['sessions'][0]['learning_session_id'] == later


def test_archive_history_pages_preserve_every_whole_turn_in_source_order(history_app):
    app = history_app
    sid = _session(app,topic='programming.python.loops')
    for index in range(45):
        _turn(app,sid,f'ну {index}?',f'Один шаг {index}.')
    _close(app,sid)
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute("""UPDATE teachcopilot.learning_messages m SET created_at=
            timestamp '2026-01-01'-(s.source_order*interval '1 second')
            FROM teachcopilot.memory_sources s WHERE s.id=m.source_id AND m.learning_session_id=%s""",(sid,))
    pages = [_get(app,sid)]
    while pages[-1]['next_cursor']:
        pages.append(_get(app,sid,before=pages[-1]['next_cursor']))
    assert [len(page['messages']) for page in pages] == [40,40,10]
    assert all(page['topic_title'] == 'Циклы в Python' and page['subject_id'] == 'python'
               and page['status'] == 'completed' and page['learning_session_id'] == sid for page in pages)
    messages = [message for page in reversed(pages) for message in page['messages']]
    assert [(m['role'],m['content']) for m in messages] == [
        pair for index in range(45) for pair in [('user',f'ну {index}?'),('assistant',f'Один шаг {index}.')]]
    assert set(pages[0]) == {'learning_session_id','topic_id','topic_title','subject_id','session_mode','status','messages','next_cursor'}
    assert pages[0]['session_mode'] == 'standard'
    assert _get(app,sid) == pages[0]


def test_archive_history_uses_literal_sources_and_excludes_private_cached_fields(history_app):
    app = history_app
    sid = _session(app,topic='legacy.opaque-topic')
    text = 'ну хз\nя 3/4 написал\n'+'не знаю – '*700
    _turn(app,sid,text,'Покажи одну часть.\nНе спешим.',private={
        'reasoning_content':'PRIVATE reasoning','diagnostics':'PRIVATE diagnostic',
        'usage':{'tokens':'PRIVATE usage'},'_guided_division_provenance':{'id':'PRIVATE source'},
        'tool_calls':[{'id':'PRIVATE tool'}]})
    _turn(app,sid,'ответ не получился',status='failed')
    _turn(app,sid,'а этот ответ неизвестен',status='unknown_outcome')
    _close(app,sid)
    result = _get(app,sid)
    assert result['topic_title'] == 'Прошлое занятие' and result['subject_id'] == 'general'
    assert result['messages'] == [
        {'role':'user','content':text,'delivery_status':'completed'},
        {'role':'assistant','content':'Покажи одну часть.\nНе спешим.'},
        {'role':'user','content':'ответ не получился','delivery_status':'failed'},
        {'role':'user','content':'а этот ответ неизвестен','delivery_status':'unknown_outcome'}]
    assert result['next_cursor'] is None
    assert 'PRIVATE' not in json.dumps(result)
    empty = _session(app)
    _close(app,empty)
    assert _get(app,empty)['messages'] == []


def test_archive_does_not_accept_an_assistant_link_from_another_session_or_wrong_role(history_app):
    app = history_app
    other = _session(app)
    wrong = _turn(app,other,'старый вопрос','PRIVATE other session')
    _close(app,other)
    current = _session(app)
    message = _turn(app,current,'мой вопрос','правильный ответ')
    _close(app,current)
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute('UPDATE teachcopilot.learning_messages SET assistant_source_id=%s WHERE id=%s',
                    (wrong['assistant_source_id'],message['id']))
    assert _get(app,current)['messages'] == [{'role':'user','content':'мой вопрос','delivery_status':'completed'}]
    with app.store._connect() as connection, connection.cursor() as cur:
        cur.execute('UPDATE teachcopilot.learning_messages SET assistant_source_id=source_id WHERE id=%s',(message['id'],))
    assert _get(app,current)['messages'] == [{'role':'user','content':'мой вопрос','delivery_status':'completed'}]


def test_archive_cursors_reject_malformed_values_and_cross_actor_learner_session_binding(history_app):
    app = history_app
    sid = _session(app)
    for index in range(23):
        _turn(app,sid,f'вопрос {index}','ок')
    _close(app,sid)
    for _ in range(10):
        extra = _session(app)
        _close(app,extra)
    list_cursor = _get(app)['next_cursor']
    history_cursor = _get(app,sid)['next_cursor']
    assert list_cursor and history_cursor
    for path,cursor in [(_url(),list_cursor),(_url(sid),history_cursor)]:
        for invalid in ('','x','{}',';;;;','A'*1025,_encode([]),cursor+'=',history_cursor if path == _url() else list_cursor):
            assert app.browser.get(path,headers=app.student_headers,params={'before':invalid}).status_code == 422
        for key,value in [('actor',str(app.teacher['id'])),('learner','another-child'),('kind','other')]:
            payload = _decode(cursor)
            payload[key] = value
            assert app.browser.get(path,headers=app.student_headers,params={'before':_encode(payload)}).status_code == 422
    # A valid cursor from one archived dialogue is invalid for another.
    assert app.browser.get(_url(extra),headers=app.student_headers,params={'before':history_cursor}).status_code == 422
    for bad_order in (-1,0,True,1.0,9223372036854775808):
        payload = _decode(history_cursor)
        payload['order'] = bad_order
        assert app.browser.get(_url(sid),headers=app.student_headers,params={'before':_encode(payload)}).status_code == 422
    for bad_time in ('not-time','2026-01-01',None,True):
        payload = _decode(list_cursor)
        payload['time'] = bad_time
        assert app.browser.get(_url(),headers=app.student_headers,params={'before':_encode(payload)}).status_code == 422
    # Python accepts wider offsets than PostgreSQL; normalize before binding.
    payload = _decode(list_cursor)
    payload['time'] = '2026-10-22T00:00:00+23:59'
    assert app.browser.get(_url(),headers=app.student_headers,params={'before':_encode(payload)}).status_code == 200
    payload['time'] = '0001-01-01T00:00:00+23:59'
    assert app.browser.get(_url(),headers=app.student_headers,params={'before':_encode(payload)}).status_code == 422


def test_archive_reopens_with_a_fresh_read_only_store_and_never_mutates_durable_rows(history_app):
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    app = history_app
    sid = _session(app)
    for index in range(23):
        _turn(app,sid,f'вопрос {index}','прошлый ответ')
    _close(app,sid)
    _session(app)  # Active conversation remains separate from archive reads.

    def snapshot():
        tables = ('learners','learning_sessions','learning_messages','memory_sources','memory_jobs',
                  'memory_versions','next_steps','sessions')
        result = {}
        with app.store._connect() as connection,connection.cursor() as cur:
            for table in tables:
                cur.execute(sql.SQL('SELECT to_jsonb(t) AS row FROM teachcopilot.{} t ORDER BY to_jsonb(t)::text')
                            .format(sql.Identifier(table)))
                result[table] = [row['row'] for row in cur.fetchall()]
        return result

    before = snapshot()
    expected_list = _get(app)
    expected_history = _get(app,sid)
    fresh = Store(make_dsn(app.store.dsn,options='-c default_transaction_read_only=on'))
    with _browser(fresh) as browser:
        assert _get(app,browser=browser) == expected_list
        assert _get(app,sid,browser=browser) == expected_history
        older = _get(app,sid,before=expected_history['next_cursor'],browser=browser)
        assert len(older['messages']) == 6 and older['next_cursor'] is None
    assert snapshot() == before
