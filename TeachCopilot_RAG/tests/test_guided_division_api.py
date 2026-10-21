from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from local_app.store import Store
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
        return TestClient(create_app(settings=Settings(), store=store, runtime=Forbidden(), rag_service=Forbidden()))
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


def test_correction_of_an_earlier_chain_source_blocks_saved_cursor(guided):
    client,store,student,teacher,_ = guided
    sid = session(client,student)
    advance_to_zero(client,student,sid)
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


def test_ordinary_message_is_a_barrier_and_ambiguous_text_is_not_a_typed_answer(guided):
    client,store,student,_,_ = guided
    sid = session(client,student)
    state = start(client,student,sid)
    invalid = action(client,student,sid,'bad','division_guided_answer',expected_state_id=state['state_id'],value='хз')
    assert invalid.status_code == 422 and context(client,student,sid)['guided_division'] == state
    # A separate ordinary reply is conversational data, never an imported cursor.
    actor = store.get_learning_session(sid)['actor_id']
    store.reserve_chat_message(sid,LEARNER,'aside','ага',actor)
    store.complete_chat_message(sid,'aside',{'choices':[{'message':{'role':'assistant','content':'Хорошо, сделаем паузу.'}}]})
    assert context(client,student,sid)['reason'] == 'later_message_barrier'


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
