"""Contradictory scope output becomes one clarification; diagnostics stay local."""
from copy import deepcopy
import json

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_memory_api_flow import flow_store  # noqa: F401
from tests.test_social_chat import LocalSession, Settings as RuntimeSettings


LEARNER = 'demo-nika-04'
TOPIC = 'math.g4.numbers.division_by_1_2_digit'
DECISION = {'in_scope': True, 'uses_prior_context': True, 'search_query': '',
            'prior_learner_question_index': 0, 'prior_assistant_question_index': 1,
            'needs_clarification': True}


@pytest.mark.parametrize('role', ['teacher', 'student'])
def test_invalid_resolver_diagnostic_is_in_local_job_but_not_response_or_cached_replay(flow_store, role, monkeypatch):
    store = flow_store
    password = 'Synthetic invalid resolver test password 123'
    teacher = store.create_account('invalid-teacher@synthetic.test', 'Synthetic Teacher',
                                   hash_password(password), 'teacher')
    store.assign_learner(teacher['id'], LEARNER)
    student = store.create_student_account('invalid-student@synthetic.test', 'Synthetic Student',
        hash_password(password), LEARNER, teacher['id'])
    account = teacher if role == 'teacher' else student
    http = LocalSession(json.dumps(DECISION))
    runtime = OllamaRuntime(RuntimeSettings(), session=http)

    def forbidden(*args, **kwargs):
        pytest.fail('An invalid resolver decision must not reach retrieval or answer generation')

    monkeypatch.setattr(store, 'hybrid_search', forbidden)
    monkeypatch.setattr(runtime, 'embed', forbidden)
    rag = RagService(store, runtime, RuntimeSettings())
    monkeypatch.setattr(rag.answer_adapter, 'answer', forbidden)
    with TestClient(create_app(settings=Settings(database_url=store.dsn), store=store,
                               runtime=runtime, rag_service=rag)) as client:
        def auth(selected):
            response = client.post('/auth/login', json={'email': selected['email'], 'password': password})
            assert response.status_code == 200
            return {'Authorization': 'Bearer ' + response.json()['access_token']}

        headers, student_headers, teacher_headers = auth(account), auth(student), auth(teacher)
        created = client.post(f'/learners/{LEARNER}/learning-sessions', headers=headers,
                              json={'topic_id': TOPIC})
        assert created.status_code == 201
        session_id = created.json()['learning_session_id']
        store.reserve_chat_message(session_id, LEARNER, 'earlier-message', '1224 : 4. С чего начать?',
                                   account['id'], author_role=role)
        store.complete_chat_message(session_id, 'earlier-message', {'choices': [{'message': {
            'role': 'assistant', 'content': 'С какого разряда начнём деление?'}}]})
        knowledge = deepcopy(store.get_card(LEARNER).get('knowledge'))
        payload = {'learner_id': LEARNER, 'learning_session_id': session_id, 'client_message_id': 'uncertain-message',
                   'messages': [{'role': 'user', 'content': 'хз'}]}
        first = client.post('/v1/chat/completions', headers=headers, json=payload)
        replay = client.post('/v1/chat/completions', headers=headers, json=payload)
        assert first.status_code == replay.status_code == 200, first.text
        assert first.json() == replay.json() and len(http.requests) == 1
        public = first.json()
        assert public['query_resolution'] == {'source': 'invalid_resolver_decision',
                                               'reason': 'resolver_contract_violation'}
        assert public['sources'] == public['citations'] == []
        assert '_local_diagnostics' not in public and 'model_response' not in first.text
        assert 'Clarification resolver must not request retrieval' not in first.text
        assert len(store.list_learning_messages(session_id)) == 4
        assert store.get_card(LEARNER).get('knowledge') == knowledge
        with store._connect() as connection, connection.cursor() as cur:
            cur.execute("SELECT id FROM teachcopilot.memory_jobs WHERE kind='chat'")
            job_ids = [str(row['id']) for row in cur.fetchall()]
        assert len(job_ids) == 1
        job_id = job_ids[0]
        job = store.get_memory_job(job_id)
        assert job['status'] == 'completed' and job['result']['response_ready'] is True
        diagnostic = job['result']['local_diagnostics']
        assert diagnostic['type'] == 'InvalidQueryResolutionError'
        assert diagnostic['model_response'] == {'resolver_decision': DECISION}
        assert client.get(f'/memory/jobs/{job_id}', headers=student_headers).status_code == 403
        teacher_job = client.get(f'/memory/jobs/{job_id}', headers=teacher_headers)
        assert teacher_job.status_code == 200
        assert teacher_job.json()['result']['local_diagnostics'] == diagnostic


@pytest.mark.parametrize('role', ['teacher', 'student'])
@pytest.mark.parametrize('diagnostic_type', ['InvalidQueryResolutionError','ForeignExerciseReference'])
def test_legacy_chat_removes_private_diagnostics_without_logging_the_raw_decision(role, diagnostic_type, caplog, monkeypatch):
    from types import SimpleNamespace
    from tests.test_local_api import FakeRag, auth, make_client, token

    monkeypatch.setattr('pipeline.learner_context.load_learner_context',
                        lambda *args, **kwargs: SimpleNamespace(prompt_text='Synthetic safe context'))

    private = 'SYNTHETIC_PRIVATE_RESOLVER_TEXT'
    rag = FakeRag(answer={'choices': [{'message': {'role': 'assistant', 'content': 'Какой шаг непонятен?'}}],
        'query_resolution': {'source': 'invalid_resolver_decision', 'reason': 'resolver_contract_violation'},
        '_local_diagnostics': {'type': diagnostic_type, 'message': private,
                               'model_response': {'resolver_decision': {'search_query': private}}}})
    client, _, _ = make_client(rag=rag)
    with client:
        headers = auth(token(client, role + '@example.org'))
        response = client.post('/v1/chat/completions', headers=headers, json={
            'learner_id': 'learner-a', 'messages': [{'role': 'user', 'content': 'хз'}]})
    assert response.status_code == 200, response.text
    assert '_local_diagnostics' not in response.json() and private not in response.text
    assert ('invalid decision' if diagnostic_type == 'InvalidQueryResolutionError' else 'different exercise') in caplog.text
    assert private not in caplog.text
