from copy import deepcopy
from pathlib import Path
import shutil

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from local_app.store import Store
from local_app.autonomous_division import VERSION as AUTONOMOUS_PROTOCOL
from local_app.learning_math import division_facts
from scripts.load_memory import ControlledRuntime
from tests.test_memory_api_flow import flow_store  # noqa: F401

LEARNER = 'demo-nika-04'
TOPIC = 'math.g4.numbers.division_by_1_2_digit'


class Forbidden:
    def answer(self, *args, **kwargs):
        raise AssertionError('Guided actions must not call RAG')
    def chat(self, *args, **kwargs):
        raise AssertionError('Guided actions must not call models')


@pytest.fixture
def guided(flow_store):
    store = flow_store
    password = 'Synthetic guided division password 123'
    teacher = store.create_account('guided-teacher@synthetic.test', 'Teacher', hash_password(password), 'teacher')
    store.assign_learner(teacher['id'], LEARNER)
    student = store.create_student_account('guided-student@synthetic.test', 'Student', hash_password(password), LEARNER, teacher['id'])
    def browser(store):
        return TestClient(create_app(settings=Settings(autonomous_division_enabled=True), store=store, runtime=Forbidden(), rag_service=Forbidden()))
    client = browser(store)
    def auth(account):
        response = client.post('/auth/login', json={'email':account['email'], 'password':password})
        assert response.status_code == 200
        return {'Authorization':'Bearer '+response.json()['access_token']}
    yield client, store, auth(student), auth(teacher), browser
    client.close()


def session(client, headers):
    result = client.post(f'/learners/{LEARNER}/learning-sessions', headers=headers, json={'topic_id':TOPIC})
    assert result.status_code == 201, result.text
    return result.json()['learning_session_id']


def autonomous_session(client, headers, client_session_id='autonomous-session'):
    result = client.post(f'/learners/{LEARNER}/learning-sessions', headers=headers, json={
        'topic_id': TOPIC, 'client_session_id': client_session_id, 'mode': 'autonomous_division'})
    assert result.status_code == 201, result.text
    assert result.json()['session_mode'] == 'autonomous_division'
    return result.json()['learning_session_id']


def autonomous_action(client, headers, sid, mid, operation, **values):
    return client.post('/v1/chat/completions', headers=headers, json={
        'learning_session_id': sid, 'client_message_id': mid,
        'learning_protocol': AUTONOMOUS_PROTOCOL,
        'learning_action': {'operation': operation, **values}})


def autonomous_context(client, headers, sid=None):
    query = {'topic_id': TOPIC, 'protocol': AUTONOMOUS_PROTOCOL}
    if sid:
        query['learning_session_id'] = sid
    result = client.get(f'/learners/{LEARNER}/guided-division', headers=headers, params=query)
    assert result.status_code == 200, result.text
    return result.json()


def correct_value(state):
    facts = division_facts(state['dividend'], state['divisor'])
    step = facts['long_division_steps'][state['step_index']]
    if state['phase'] == 'quotient_digit':
        return step['quotient_digit']
    if state['phase'] == 'remainder':
        return step['remainder']
    return facts['long_division_steps'][state['step_index'] + 1]['brought_down_digit']


def action(client, headers, sid, mid, operation, **values):
    return client.post('/v1/chat/completions', headers=headers, json={
        'learning_session_id':sid, 'client_message_id':mid, 'learning_action':{'operation':operation, **values}})


def start(client, headers, sid):
    result = action(client, headers, sid, 'start', 'division_guided_start', dividend=3636, divisor=12)
    assert result.status_code == 200, result.text
    return result.json()['guided_division']


def advance_to_zero(client, headers, sid):
    state = start(client,headers,sid)
    for i,value in enumerate((3,0,3)):
        result = action(client,headers,sid,f'answer-{i}','division_guided_answer',expected_state_id=state['state_id'],value=value)
        assert result.status_code == 200, result.text
        assert '_guided_division_provenance' not in result.json()
        state = result.json()['guided_division']
    return state


def context(client, headers, sid=None):
    query = {'topic_id':TOPIC}
    if sid: query['learning_session_id'] = sid
    result = client.get(f'/learners/{LEARNER}/guided-division', headers=headers, params=query)
    assert result.status_code == 200, result.text
    return result.json()


def approve_latest(client, store, teacher_headers):
    job = MemoryWorker(store, ControlledRuntime(0)).run_once()
    assert job['status'] == 'completed', job
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    version = snapshot['versions'][0]
    response = client.post(f"/learners/{LEARNER}/memory/versions/{version['version_id']}/confirm", headers=teacher_headers,
                           json={'expected_revision':snapshot['revision']})
    assert response.status_code == 200, response.text


def test_zero_step_survives_recreation_and_duplicate_or_stale_inputs_do_not_advance(guided):
    client,store,student,teacher,browser = guided
    sid = session(client,student)
    knowledge = deepcopy(store.get_card(LEARNER).get('knowledge'))
    state = advance_to_zero(client,student,sid)
    reopened = Store(store.dsn)
    with browser(reopened) as fresh:
        restored = context(fresh,student,sid)
        assert restored['guided_division'] == state
        assert context(fresh,student)['active_learning_session_id'] == sid
        assert '3 на 12' in restored['prompt'] and 'десятков' in restored['prompt']
        hint = action(fresh,student,sid,'hint','division_guided_hint',expected_state_id=state['state_id'])
        replay = action(fresh,student,sid,'hint','division_guided_hint',expected_state_id=state['state_id'])
        assert hint.status_code == replay.status_code == 200 and hint.json() == replay.json()
        assert '3 меньше 12' in hint.json()['choices'][0]['message']['content']
        wrong_cache = action(fresh,student,sid,'hint','division_guided_hint',expected_state_id='other')
        assert wrong_cache.status_code == 409
        stale = action(fresh,student,sid,'stale','division_guided_answer',expected_state_id=state['state_id'],value=0)
        assert stale.status_code == 409
        current = context(fresh,student,sid)['guided_division']
        assert current['state_id'] == 'hint' and current['phase'] == 'quotient_digit'
        accepted = action(fresh,student,sid,'zero','division_guided_answer',expected_state_id='hint',value=0)
        assert accepted.status_code == 200 and accepted.json()['guided_division']['phase'] == 'remainder'
        assert fresh.get(f'/learners/foreign/guided-division', headers=student, params={'topic_id':TOPIC}).status_code == 403
        assert fresh.get(f'/learners/{LEARNER}/guided-division', headers=teacher,
                         params={'topic_id':TOPIC,'learning_session_id':sid}).status_code == 403
    assert store.get_card(LEARNER).get('knowledge') == knowledge


def test_future_session_needs_review_and_explicit_resume_then_preserves_exact_phase(guided):
    client,store,student,teacher,_ = guided
    sid = session(client,student)
    state = advance_to_zero(client,student,sid)
    assert client.post(f'/learning-sessions/{sid}/complete', headers=student).status_code == 202
    assert not context(client,student)['available']
    approve_latest(client,store,teacher)
    previous = context(client,student)
    assert previous['available'] and previous['origin'] == 'approved_memory'
    second = session(client,student)
    forbidden = action(client,student,second,'not-resumed','division_guided_answer',expected_state_id=state['state_id'],value=0)
    assert forbidden.status_code == 409
    resumed = action(client,student,second,'resume','division_guided_resume',expected_state_id=state['state_id'],source_learning_session_id=sid)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['guided_division']['step_index'] == 1
    assert resumed.json()['guided_assessment']['status'] == 'not_assessed'
    assert '3 на 12' in resumed.json()['choices'][0]['message']['content']


@pytest.mark.parametrize('conversation', [False, True])
def test_correction_of_an_earlier_chain_source_blocks_saved_cursor(guided, conversation):
    client,store,student,teacher,_ = guided
    sid = session(client,student)
    advance_to_zero(client,student,sid)
    if conversation:
        ordinary(store,sid)
    assert client.post(f'/learning-sessions/{sid}/complete', headers=student).status_code == 202
    approve_latest(client,store,teacher)
    assert context(client,student)['available']
    first_source = store.list_learning_messages(sid)[0]['source_id']
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    correction = client.post(f'/learners/{LEARNER}/memory/sources/{first_source}/corrections', headers=teacher,
        json={'text':'В исходном примере была другая запись; этот курсор использовать нельзя.',
              'expected_revision':snapshot['revision'],'expected_source_revision':snapshot['source_revision'],
              'client_correction_id':'guided-source-fix'})
    assert correction.status_code == 202, correction.text
    assert context(client,student)['reason'] == 'corrected_guided_source'


@pytest.mark.parametrize('text', ['ага', 'хз', '0', 'Вчера построил станцию в Minecraft.',
                                  'Я уже всё решил, получилось 303.', 'не хочу уже, потом'])
def test_ordinary_conversation_keeps_bookmark_but_requires_explicit_resume(guided, text):
    client,store,student,_,_ = guided
    sid = session(client,student)
    knowledge = deepcopy(store.get_card(LEARNER).get('knowledge'))
    state = advance_to_zero(client,student,sid)
    invalid = action(client,student,sid,'bad','division_guided_answer',expected_state_id=state['state_id'],value='хз')
    assert invalid.status_code == 422 and context(client,student,sid)['guided_division'] == state

    class Dialogue:
        def answer(self, messages, **kwargs):
            assert messages[-1] == {'role':'user', 'content':text}
            return {'choices':[{'message':{'role':'assistant','content':'Можем вернуться к примеру позже.'}}]}

    client.app.state.rag_service = Dialogue()
    aside = client.post('/v1/chat/completions', headers=student, json={'learning_session_id':sid,
        'client_message_id':'aside', 'messages':[{'role':'user','content':text}]})
    assert aside.status_code == 200, aside.text
    bookmark = context(client,student,sid)
    assert bookmark['available'] and bookmark['resume_required']
    assert bookmark['resume_reason'] == 'conversation_continued'
    assert bookmark['guided_division'] == state
    assert '3 на 12' in bookmark['prompt']
    for index, operation in enumerate(('division_guided_answer', 'division_guided_hint', 'division_guided_pause')):
        result = action(client,student,sid,f'not-resumed-{index}',operation,
                        expected_state_id=state['state_id'],**({'value':0} if operation.endswith('answer') else {}))
        assert result.status_code == 409, result.text
        assert context(client,student,sid)['guided_division'] == state
    resumed = action(client,student,sid,'return','division_guided_resume',expected_state_id=state['state_id'])
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['guided_division']['phase'] == state['phase']
    assert resumed.json()['guided_division']['step_index'] == state['step_index']
    assert resumed.json()['guided_assessment']['status'] == 'not_assessed'
    assert '_guided_division_provenance' not in resumed.json()
    sources = store.list_learning_messages(sid)
    replay = action(client,student,sid,'return','division_guided_resume',expected_state_id=state['state_id'])
    assert replay.status_code == 200 and replay.json() == resumed.json()
    assert store.list_learning_messages(sid) == sources
    assert context(client,student,sid)['resume_required'] is False
    assert action(client,student,sid,'stale','division_guided_answer',expected_state_id=state['state_id'],value=0).status_code == 409
    accepted = action(client,student,sid,'zero','division_guided_answer',expected_state_id='return',value=0)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()['guided_division']['phase'] == 'remainder'
    assert accepted.json()['guided_assessment']['independent_work'] == 'unknown'
    assert store.get_card(LEARNER).get('knowledge') == knowledge


def test_real_rag_asks_for_a_missing_answer_and_replay_preserves_guided_state(guided):
    from local_app.rag import RagService

    client, store, student, _, _ = guided
    sid = session(client, student)
    state = advance_to_zero(client, student, sid)
    knowledge = deepcopy(store.get_card(LEARNER).get('knowledge'))
    original_sources = store.list_learning_messages(sid)

    class ScopeOnly:
        calls = 0

        def resolve_chat_query(self, query, prior, subject_id, **kwargs):
            self.calls += 1
            assert query == 'ага' and prior[-1]['role'] == 'assistant'
            return {'in_scope': True, 'uses_prior_context': False,
                    'prior_assistant_question_index': len(prior) - 1, 'search_query': 'Деление по шагам'}

        def chat(self, *args, **kwargs):
            raise AssertionError('A bare agreement must not generate a correct numerical answer')

    runtime = ScopeOnly()
    client.app.state.rag_service = RagService(store, runtime, Settings())
    request = {'learning_session_id': sid, 'client_message_id': 'agreement-not-answer',
               'messages': [{'role': 'user', 'content': 'ага'}]}
    response = client.post('/v1/chat/completions', headers=student, json=request)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 'Напиши свой ответ' in body['choices'][0]['message']['content']
    assert body['answer_validation']['reason'] == 'learner_answer_missing'
    assert body['citations'] == body['sources'] == []
    saved_sources = store.list_learning_messages(sid)
    assert len(saved_sources) == len(original_sources) + 2
    assert [row['author_role'] for row in saved_sources[-2:]] == ['student', 'assistant']
    assert saved_sources[-2]['content'] == 'ага'
    replay = client.post('/v1/chat/completions', headers=student, json=request)
    assert replay.status_code == 200 and replay.json() == body
    assert runtime.calls == 1 and store.list_learning_messages(sid) == saved_sources
    bookmark = context(client, student, sid)
    assert bookmark['guided_division'] == state and bookmark['resume_required'] is True
    assert store.get_card(LEARNER).get('knowledge') == knowledge


def ordinary(store, sid, text='Кстати, люблю кометы.', *, mid='aside', status='completed'):
    actor = store.get_learning_session(sid)['actor_id']
    store.reserve_chat_message(sid,LEARNER,mid,text,actor)
    if status == 'completed':
        store.complete_chat_message(sid,mid,{'choices':[{'message':{'role':'assistant','content':'Что тебя в них заинтересовало?'}}]})
    elif status != 'pending':
        store.fail_chat_message(sid,mid,unknown_outcome=status == 'unknown_outcome')


@pytest.mark.parametrize('status', ['pending', 'unknown_outcome'])
def test_unresolved_message_outcome_prevents_return_to_earlier_step(guided, status):
    client,store,student,_,_ = guided
    sid = session(client,student)
    state = advance_to_zero(client,student,sid)
    ordinary(store,sid,status=status)
    blocked = context(client,student,sid)
    assert not blocked['available'] and blocked['reason'] == 'message_outcome_unavailable'
    resumed = action(client,student,sid,'return','division_guided_resume',expected_state_id=state['state_id'])
    assert resumed.status_code in (409,422)
    assert not context(client,student,sid)['available']


def test_known_failed_message_does_not_change_or_hide_the_saved_step(guided):
    client,store,student,_,_ = guided
    sid = session(client,student)
    state = advance_to_zero(client,student,sid)
    ordinary(store,sid,status='failed')
    restored = context(client,student,sid)
    assert restored['available'] and restored['resume_required'] is False
    assert restored['guided_division'] == state


@pytest.mark.parametrize('latest', ['new_start', 'completed', 'corrupt'])
def test_conversation_never_revives_a_step_before_a_newer_typed_exercise(guided, latest):
    client,store,student,_,_ = guided
    sid = session(client,student)
    advance_to_zero(client,student,sid)
    result = action(client,student,sid,'new-start','division_guided_start',dividend=12,divisor=4)
    assert result.status_code == 200
    state = result.json()['guided_division']
    if latest == 'completed':
        for mid,value in (('digit',3),('remainder',0)):
            result = action(client,student,sid,mid,'division_guided_answer',expected_state_id=state['state_id'],value=value)
            assert result.status_code == 200
            state = result.json()['guided_division']
    elif latest == 'corrupt':
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE teachcopilot.learning_messages SET response=response-'guided_division' WHERE learning_session_id=%s AND client_message_id='new-start'",(sid,))
    ordinary(store,sid)
    restored = context(client,student,sid)
    if latest == 'new_start':
        assert restored['available'] and restored['resume_required']
        assert restored['guided_division'] == state
        assert restored['guided_division']['dividend'] == 12
    else:
        assert not restored['available']
        assert restored['reason'] == ('exercise_completed' if latest == 'completed' else 'invalid_guided_chain')


def test_completed_conversation_and_new_greeting_still_need_current_teacher_review(guided):
    client,store,student,teacher,_ = guided
    sid = session(client,student)
    state = advance_to_zero(client,student,sid)
    ordinary(store,sid)
    assert client.post(f'/learning-sessions/{sid}/complete',headers=student).status_code == 202
    assert context(client,student)['reason'] == 'teacher_review_required'
    approve_latest(client,store,teacher)
    reviewed = context(client,student)
    assert reviewed['available'] and reviewed['resume_required'] and reviewed['origin'] == 'approved_memory'
    assert reviewed['guided_division'] == state
    second = session(client,student)
    ordinary(store,second,'привет',mid='greeting')
    bookmark = context(client,student,second)
    assert bookmark['available'] and bookmark['origin'] == 'approved_memory'
    assert bookmark['active_learning_session_id'] == second
    assert bookmark['source_learning_session_id'] == sid
    assert bookmark['resume_required'] and bookmark['guided_division'] == state
    assert action(client,student,second,'no-return','division_guided_hint',expected_state_id=state['state_id']).status_code == 409
    resumed = action(client,student,second,'return','division_guided_resume',expected_state_id=state['state_id'],source_learning_session_id=sid)
    assert resumed.status_code == 200, resumed.text
    assert context(client,student,second)['origin'] == 'current_session'
    assert context(client,student,second)['resume_required'] is False


def test_new_explicit_non_guided_action_blocks_older_bookmark(guided):
    client,_,student,_,_ = guided
    sid = session(client,student)
    advance_to_zero(client,student,sid)
    result = action(client,student,sid,'check-other','division_check',dividend=84,divisor=2)
    assert result.status_code == 200, result.text
    restored = context(client,student,sid)
    assert not restored['available'] and restored['reason'] == 'later_message_barrier'


def test_review_does_not_restore_a_bookmark_evicted_from_selected_memory(guided):
    client,store,student,teacher,_ = guided
    sid = session(client,student)
    state = advance_to_zero(client,student,sid)
    bookmark_source = [row['source_id'] for row in store.list_learning_messages(sid) if row['author_role'] == 'student'][-1]
    for index in range(5):
        ordinary(store,sid,'Рассказываю про станцию в игре. ' * 30,mid=f'long-aside-{index}')
    assert context(client,student,sid)['guided_division'] == state
    assert client.post(f'/learning-sessions/{sid}/complete',headers=student).status_code == 202
    approve_latest(client,store,teacher)
    snapshot = store.get_learner_memory(LEARNER,TOPIC)
    selected = {row['source_id'] for row in snapshot['versions'][0]['context']['sources']}
    assert bookmark_source not in selected
    restored = context(client,student)
    assert not restored['available'] and restored['reason'] == 'teacher_review_required'


def test_changed_saved_cursor_is_rejected_and_client_cannot_set_one(guided):
    client,store,student,_,_ = guided
    sid = session(client,student)
    state = start(client,student,sid)
    forged = action(client,student,sid,'forged','division_guided_answer',expected_state_id=state['state_id'],value=3,step_index=1)
    assert forged.status_code == 422
    with store._connect() as connection, connection.cursor() as cur:
        cur.execute("UPDATE teachcopilot.learning_messages SET response=jsonb_set(response,'{guided_division,step_index}','1') WHERE learning_session_id=%s AND client_message_id='start'",(sid,))
    assert context(client,student,sid)['reason'] == 'invalid_guided_chain'


def test_new_unreviewed_completed_lesson_prevents_reviving_an_older_cursor(guided):
    client,store,student,teacher,_ = guided
    first = session(client,student)
    advance_to_zero(client,student,first)
    assert client.post(f'/learning-sessions/{first}/complete', headers=student).status_code == 202
    approve_latest(client,store,teacher)
    assert context(client,student)['available']
    second = session(client,student)
    start(client,student,second)
    assert client.post(f'/learning-sessions/{second}/complete', headers=student).status_code == 202
    assert not context(client,student)['available']


def test_autonomous_protocol_isolated_from_guided_mode_and_default_flag_is_off(guided):
    client, store, student, teacher, browser = guided
    protocol_without_session = client.post('/v1/chat/completions', headers=student, json={
        'learning_protocol': AUTONOMOUS_PROTOCOL,
        'messages': [{'role': 'user', 'content': 'Can you solve this?'}]})
    assert protocol_without_session.status_code == 409
    standard = client.post(f'/learners/{LEARNER}/learning-sessions', headers=student,
        json={'topic_id': TOPIC, 'client_session_id': 'same-session-id'})
    assert standard.status_code == 201
    assert standard.json()['session_mode'] == 'standard'
    conflict = client.post(f'/learners/{LEARNER}/learning-sessions', headers=student,
        json={'topic_id': TOPIC, 'client_session_id': 'same-session-id', 'mode': 'autonomous_division'})
    assert conflict.status_code == 409
    default_topic = client.post(f'/learners/{LEARNER}/learning-sessions', headers=student,
        json={'client_session_id': 'auto-default-topic', 'mode': 'autonomous_division'})
    assert default_topic.status_code == 201
    assert default_topic.json()['topic_id'] == TOPIC

    disabled = TestClient(create_app(settings=Settings(), store=store, runtime=Forbidden(), rag_service=Forbidden()))
    try:
        response = disabled.post(f'/learners/{LEARNER}/learning-sessions', headers=student,
            json={'topic_id': TOPIC, 'client_session_id': 'disabled-auto', 'mode': 'autonomous_division'})
        assert response.status_code == 404
        assert disabled.get('/auth/me', headers=student).json()['features']['autonomous_division'] is False
    finally:
        disabled.close()


def test_autonomous_api_rejects_ambiguous_text_and_persists_only_public_fields(guided):
    client, store, student, _, browser = guided
    sid = autonomous_session(client, student, 'ambiguous-session')
    start_request = {'learning_session_id': sid, 'client_message_id': 'auto-start',
        'learning_protocol': AUTONOMOUS_PROTOCOL,
        'learning_action': {'operation': 'autonomous_division_start'}}
    started = client.post('/v1/chat/completions', headers=student, json=start_request)
    assert started.status_code == 200, started.text
    first = started.json()
    assert first['learning_protocol'] == AUTONOMOUS_PROTOCOL
    assert first['learning_tool']['facts'] == {}
    assert '_autonomous_division_state' not in first
    assert '_autonomous_division_provenance' not in first
    replay = client.post('/v1/chat/completions', headers=student, json=start_request)
    assert replay.status_code == 200 and replay.json() == first
    assert len(store.list_learning_messages(sid)) == 2

    cursor = first['autonomous_division']
    saved_position = (cursor['step_index'], cursor['phase'])
    for index, text in enumerate(('Ага', 'хз', '4?')):
        response = client.post('/v1/chat/completions', headers=student, json={
            'learning_session_id': sid, 'client_message_id': f'ambiguous-{index}',
            'learning_protocol': AUTONOMOUS_PROTOCOL,
            'messages': [{'role': 'user', 'content': text}]})
        assert response.status_code == 200, f'index={index}, cursor={cursor}, body={response.text}'
        assert response.json()['learning_action']['operation'] == 'autonomous_division_clarify'
        assert 'не могу проверить этот текст как числовой ответ' in response.json()['choices'][0]['message']['content']
        assert (response.json()['autonomous_division']['step_index'],
                response.json()['autonomous_division']['phase']) == saved_position
        cursor = response.json()['autonomous_division']
    restored = autonomous_context(client, student, sid)
    assert restored['state']['state_id'] == cursor['state_id']
    assert restored['available_actions'] == ['answer', 'explain', 'show_step', 'pause', 'switch']
    assert 'source_session_id' not in restored and 'origin' not in restored

    show = autonomous_action(client, student, sid, 'auto-show', 'autonomous_division_show_step',
                              expected_state_id=cursor['state_id'])
    assert show.status_code == 200, show.text
    show_text = show.json()['choices'][0]['message']['content']
    assert 'Текущий шаг:' in show_text
    assert str(division_facts(cursor['dividend'], cursor['divisor'])['quotient']) not in show_text
    assert show.json()['autonomous_division']['phase'] == cursor['phase']
    assert show.json()['learning_tool']['facts'] == {}
    assert '_autonomous_division_state' not in show.json()

    bad_action = autonomous_action(client, student, sid, 'bad-ambiguous', 'autonomous_division_answer',
                                   expected_state_id=show.json()['autonomous_division']['state_id'], value='4?')
    assert bad_action.status_code == 422
    wrong_value = correct_value(show.json()['autonomous_division']) + 1
    wrong = autonomous_action(client, student, sid, 'auto-wrong', 'autonomous_division_answer',
        expected_state_id=show.json()['autonomous_division']['state_id'], value=wrong_value)
    assert wrong.status_code == 200 and wrong.json()['autonomous_assessment']['status'] == 'incorrect'
    assert (wrong.json()['autonomous_division']['step_index'], wrong.json()['autonomous_division']['phase']) == saved_position
    assert wrong.json()['autonomous_assessment']['competence_assessed'] is False
    assert wrong.json()['autonomous_assessment']['independent_work'] == 'unknown'


def test_autonomous_continuation_pause_next_example_and_no_memory_worker_job(guided):
    client, store, student, _, browser = guided
    first_session = autonomous_session(client, student, 'continuation-1')
    started = autonomous_action(client, student, first_session, 'first-start', 'autonomous_division_start')
    assert started.status_code == 200, started.text
    state = started.json()['autonomous_division']
    pause = autonomous_action(client, student, first_session, 'first-pause', 'autonomous_division_pause',
                              expected_state_id=state['state_id'])
    assert pause.status_code == 200
    paused = pause.json()['autonomous_division']
    in_session = autonomous_context(client, student, first_session)
    assert in_session['state']['state_id'] == paused['state_id']
    ended = client.post(f'/learning-sessions/{first_session}/complete', headers=student)
    assert ended.status_code == 202 and ended.json()['job'] is None and ended.json()['job_id'] is None
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT count(*) AS n FROM teachcopilot.memory_jobs WHERE idempotency_key=%s",
                       (f'session:{first_session}',))
        assert cursor.fetchone()['n'] == 0

    second_session = autonomous_session(client, student, 'continuation-2')
    restored = autonomous_context(client, student, second_session)
    assert restored['available'] and restored['resume_required'] is True
    assert restored['state']['state_id'] == paused['state_id']
    assert restored['state']['phase'] == paused['phase']
    resumed = autonomous_action(client, student, second_session, 'second-resume', 'autonomous_division_resume',
                                expected_state_id=restored['state']['state_id'])
    assert resumed.status_code == 200, resumed.text
    state = resumed.json()['autonomous_division']
    assert state['status'] == 'active' and state['step_index'] == paused['step_index']
    assert resumed.json()['autonomous_assessment']['status'] == 'not_assessed'

    while state['status'] != 'completed':
        answer = autonomous_action(client, student, second_session, f"complete-{state['state_id']}",
            'autonomous_division_answer', expected_state_id=state['state_id'], value=correct_value(state))
        assert answer.status_code == 200, f"state={state}, body={answer.text}"
        state = answer.json()['autonomous_division']
    ended_again = client.post(f'/learning-sessions/{second_session}/complete', headers=student)
    assert ended_again.status_code == 202 and ended_again.json()['job_id'] is None

    third_session = autonomous_session(client, student, 'continuation-3')
    completed = autonomous_context(client, student, third_session)
    assert completed['available'] and completed['state']['status'] == 'completed'
    next_problem = autonomous_action(client, student, third_session, 'third-next', 'autonomous_division_next',
                                     expected_state_id=completed['state']['state_id'])
    assert next_problem.status_code == 200, next_problem.text
    assert next_problem.json()['autonomous_division']['status'] == 'active'
    assert next_problem.json()['autonomous_division']['state_id'] == 'third-next'
    assert next_problem.json()['learning_tool']['facts'] == {}
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT response->'_autonomous_division_state'->>'example_kind' AS kind "
            "FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND client_message_id='third-next'",
            (third_session,))
        assert cursor.fetchone()['kind'] in {'ordinary', 'internal_zero', 'terminal_zero'}


def test_new_example_chain_is_independent_of_corrections_to_older_example(guided):
    client, store, student, teacher, _ = guided
    sid = autonomous_session(client, student, 'independent-example-chain')
    started = autonomous_action(client, student, sid, 'chain-start', 'autonomous_division_start')
    assert started.status_code == 200
    state = started.json()['autonomous_division']
    index = 0
    while state['status'] != 'completed':
        response = autonomous_action(client, student, sid, f'chain-answer-{index}',
            'autonomous_division_answer', expected_state_id=state['state_id'], value=correct_value(state))
        assert response.status_code == 200, f'index={index}, state={state}, body={response.text}'
        state = response.json()['autonomous_division']
        index += 1
    next_example = autonomous_action(client, student, sid, 'chain-next', 'autonomous_division_next',
        expected_state_id=state['state_id'])
    assert next_example.status_code == 200, next_example.text
    current_id = next_example.json()['autonomous_division']['state_id']
    completed_session = client.post(f'/learning-sessions/{sid}/complete', headers=student)
    assert completed_session.status_code == 202 and completed_session.json()['job_id'] is None

    old_start_source = store.list_learning_messages(sid)[0]['source_id']
    snapshot = store.get_learner_memory(LEARNER, TOPIC)
    correction = client.post(f'/learners/{LEARNER}/memory/sources/{old_start_source}/corrections',
        headers=teacher, json={'text': 'Correction to the previous division example.',
            'expected_revision': snapshot['revision'],
            'expected_source_revision': snapshot['source_revision'],
            'client_correction_id': 'old-autonomous-example-fix'})
    assert correction.status_code == 202, correction.text
    restored = autonomous_context(client, student, sid)
    assert restored['available'] is True
    assert restored['state']['state_id'] == current_id


def test_autonomous_two_tabs_stale_state_unknown_outcome_and_foreign_access(guided):
    client, store, student, teacher, browser = guided
    second_tab = browser(Store(store.dsn))
    try:
        sid = autonomous_session(client, student, 'two-tabs')
        started = autonomous_action(client, student, sid, 'tab-start', 'autonomous_division_start')
        assert started.status_code == 200
        old_cursor = started.json()['autonomous_division']
        correct = correct_value(old_cursor)
        first = autonomous_action(client, student, sid, 'tab-one-answer', 'autonomous_division_answer',
                                  expected_state_id=old_cursor['state_id'], value=correct)
        second = autonomous_action(second_tab, student, sid, 'tab-two-answer', 'autonomous_division_answer',
                                   expected_state_id=old_cursor['state_id'], value=correct)
        assert first.status_code == 200
        assert second.status_code == 409
        assert autonomous_context(second_tab, student, sid)['state']['state_id'] == 'tab-one-answer'

        ordinary(store, sid, 'unresolved action', mid='autonomous-unknown', status='unknown_outcome')
        blocked = autonomous_context(client, student, sid)
        assert blocked['available'] is False and blocked['reason'] == 'unknown_outcome'
        denied = autonomous_action(client, student, sid, 'after-unknown', 'autonomous_division_start')
        assert denied.status_code == 409
        assert autonomous_context(client, student, sid)['reason'] == 'unknown_outcome'

        assert client.get(f'/learners/foreign/guided-division', headers=student,
            params={'topic_id': TOPIC, 'protocol': AUTONOMOUS_PROTOCOL}).status_code == 403
        assert client.get(f'/learners/{LEARNER}/guided-division', headers=teacher,
            params={'topic_id': TOPIC, 'protocol': AUTONOMOUS_PROTOCOL, 'learning_session_id': sid}).status_code == 403
    finally:
        second_tab.close()


def test_six_sequential_autonomous_sessions_for_two_synthetic_learners(guided, tmp_path):
    client, store, student, teacher, browser = guided
    password = 'Synthetic autonomous division password 123'
    second_learner = 'demo-luna-03'
    learner_card = Path(__file__).resolve().parents[2] / 'learner-data/demo/learners/demo-luna-03.json'
    shutil.copy(learner_card, tmp_path / 'learner-data/learners' / learner_card.name)
    store.import_repository(tmp_path)
    teacher_account = store.get_account(email='guided-teacher@synthetic.test')
    store.assign_learner(teacher_account['id'], second_learner)
    store.create_student_account('autonomous-luna@synthetic.test', 'Synthetic Luna', hash_password(password),
                                 second_learner, teacher_account['id'])
    second_account = client.post('/auth/login', json={'email': 'autonomous-luna@synthetic.test', 'password': password})
    assert second_account.status_code == 200
    second_student = {'Authorization': 'Bearer ' + second_account.json()['access_token']}

    for learner_id, headers in ((LEARNER, student), (second_learner, second_student)):
        prior_session = None
        for lesson in range(3):
            client_session_id = f'six-session-{learner_id}-{lesson}'
            created = client.post(f'/learners/{learner_id}/learning-sessions', headers=headers, json={
                'topic_id': TOPIC, 'client_session_id': client_session_id, 'mode': 'autonomous_division'})
            assert created.status_code == 201, created.text
            sid = created.json()['learning_session_id']
            if prior_session is None:
                initial = client.get(f'/learners/{learner_id}/guided-division', headers=headers,
                    params={'topic_id': TOPIC, 'protocol': AUTONOMOUS_PROTOCOL, 'learning_session_id': sid}).json()
                assert initial['available'] is False
                start = client.post('/v1/chat/completions', headers=headers, json={
                    'learning_session_id': sid, 'client_message_id': f'six-start-{learner_id}',
                    'learning_protocol': AUTONOMOUS_PROTOCOL,
                    'learning_action': {'operation': 'autonomous_division_start'}})
                assert start.status_code == 200, start.text
                cursor = start.json()['autonomous_division']
            else:
                restored = client.get(f'/learners/{learner_id}/guided-division', headers=headers,
                    params={'topic_id': TOPIC, 'protocol': AUTONOMOUS_PROTOCOL, 'learning_session_id': sid})
                assert restored.status_code == 200 and restored.json()['available']
                assert restored.json()['resume_required'] is True
                resume = client.post('/v1/chat/completions', headers=headers, json={
                    'learning_session_id': sid, 'client_message_id': f'six-resume-{learner_id}-{lesson}',
                    'learning_protocol': AUTONOMOUS_PROTOCOL,
                    'learning_action': {'operation': 'autonomous_division_resume',
                                        'expected_state_id': restored.json()['state']['state_id']}})
                assert resume.status_code == 200, resume.text
                cursor = resume.json()['autonomous_division']
            wrong = client.post('/v1/chat/completions', headers=headers, json={
                'learning_session_id': sid, 'client_message_id': f'six-wrong-{learner_id}-{lesson}',
                'learning_protocol': AUTONOMOUS_PROTOCOL,
                'learning_action': {'operation': 'autonomous_division_answer',
                    'expected_state_id': cursor['state_id'], 'value': correct_value(cursor) + 1}})
            assert wrong.status_code == 200 and wrong.json()['autonomous_assessment']['status'] == 'incorrect'
            cursor = wrong.json()['autonomous_division']
            help_result = autonomous_action(client, headers, sid, f'six-help-{learner_id}-{lesson}',
                'autonomous_division_explain', expected_state_id=cursor['state_id'])
            assert help_result.status_code == 200
            assert help_result.json()['autonomous_division']['phase'] == cursor['phase']
            cursor = help_result.json()['autonomous_division']
            paused = autonomous_action(client, headers, sid, f'six-pause-{learner_id}-{lesson}',
                'autonomous_division_pause', expected_state_id=cursor['state_id'])
            assert paused.status_code == 200
            ended = client.post(f'/learning-sessions/{sid}/complete', headers=headers)
            assert ended.status_code == 202 and ended.json()['job'] is None and ended.json()['job_id'] is None
            prior_session = sid


def test_concurrent_answers_accept_only_one_transition(guided):
    from concurrent.futures import ThreadPoolExecutor
    client,store,student,_,_ = guided
    sid = session(client,student)
    start(client,student,sid)
    def send(mid):
        return action(client,student,sid,mid,'division_guided_answer',expected_state_id='start',value=3)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(send,['concurrent-a','concurrent-b']))
    assert sorted(result.status_code for result in results)[0] == 200
    assert sum(result.status_code == 200 for result in results) == 1
    assert all(result.status_code in (200,409,422) for result in results)
    saved = context(client,student,sid)['guided_division']
    assert saved['step_index'] == 0 and saved['phase'] == 'remainder'


def test_stale_reviewed_memory_does_not_allow_cross_session_resume(guided):
    client,store,student,teacher,_ = guided
    sid = session(client,student)
    advance_to_zero(client,student,sid)
    assert client.post(f'/learning-sessions/{sid}/complete', headers=student).status_code == 202
    approve_latest(client,store,teacher)
    card = store.get_card(LEARNER)
    card['interests'] = list(card.get('interests', [])) + ['кометы']
    store.save_card(card,'synthetic-teacher',expected_revision=store.get_revision(LEARNER))
    assert context(client,student)['reason'] == 'teacher_review_required'
