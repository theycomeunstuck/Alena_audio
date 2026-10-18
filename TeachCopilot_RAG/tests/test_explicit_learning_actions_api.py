"""Explicit form actions through authorized chat with no inference transport."""
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.memory_store import MemoryIdempotencyConflict
from tests.test_local_api import auth, token
from tests.test_social_chat import Settings
from tests.test_verified_learning_api import LEARNER, FOREIGN_LEARNER, LearningApiStore, start_session


class ActionStore(LearningApiStore):
    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, **kwargs):
        session = self.get_learning_session(session_id, learner_id)
        if not session or session['actor_id'] != actor_id:
            raise PermissionError('session belongs to another account')
        key = (session_id, client_message_id)
        existing = self.learning_messages.get(key)
        if existing and existing.get('request_content') != content:
            raise MemoryIdempotencyConflict('client_message_id was reused for different text')
        if not existing and session['status'] != 'active':
            raise ValueError('session completed')
        result = super().reserve_chat_message(session_id, learner_id, client_message_id, content, actor_id, **kwargs)
        self.learning_messages[key]['request_content'] = content
        return result


class NoInference:
    def answer(self, *args, **kwargs):
        raise AssertionError('An explicit action must not call RAG or inference')

    def chat(self, *args, **kwargs):
        raise AssertionError('An explicit action must not call a model')


@pytest.fixture
def action_api():
    store = ActionStore()
    store.add_account('student-actions', 'actions.student@example.test', 'student', LEARNER)
    store.add_account('teacher-actions', 'actions.teacher@example.test', 'teacher')
    app = create_app(settings=Settings(), store=store, runtime=NoInference(), rag_service=NoInference())
    with TestClient(app) as browser:
        yield browser, store, auth(token(browser, 'actions.student@example.test'))


def action_payload(session_id, action, message_id='action-one'):
    return {'learning_session_id': session_id, 'client_message_id': message_id,
            'learning_action': action, 'messages': [], 'metadata': {'subject_id': 'math'}}


def test_student_uses_typed_division_without_model_and_saves_real_request(action_api):
    browser, store, headers = action_api
    original_card = deepcopy(store.get_card(LEARNER))
    payload = action_payload(start_session(browser, headers),
        {'operation': 'division_check', 'dividend': 7560, 'divisor': 24, 'stated_quotient': 315})
    response = browser.post('/v1/chat/completions', headers=headers, json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['choices'][0]['message']['content'] == 'Результат верный. 7560 : 24 = 315. Проверка: 315 × 24 = 7560.'
    assert result['learning_action']['request_text'] == 'Проверь деление: 7560 : 24 = 315.'
    assert result['usage_scope'] == 'deterministic_action_no_model_calls'
    assert result['learning_tool']['student_competence_assessed'] is False
    assert result['sources'] == result['citations'] == []
    assert store.sources[0]['role'] == 'student'
    assert store.sources[0]['content'] == result['learning_action']['request_text']
    assert store.sources[1]['role'] == 'assistant'
    assert store.get_card(LEARNER) == original_card
    assert not next(iter(store.memory_jobs.values())).get('call_started')


@pytest.mark.parametrize('action', [None, {}, [], False, {'operation': 'unknown'},
    {'operation': 'division_check', 'dividend': True, 'divisor': 2},
    {'operation': 'division_check', 'dividend': '202', 'divisor': 2},
    {'operation': 'division_check', 'dividend': 202.0, 'divisor': 2},
    {'operation': 'division_check', 'dividend': 202, 'divisor': 0},
    {'operation': 'division_check', 'dividend': 202, 'divisor': 2, 'stated_quotient': None},
    {'operation': 'division_places', 'dividend': 202, 'divisor': 2, 'stated_quotient': 101},
    {'operation': 'division_practice', 'divisor_digits': True},
    {'operation': 'division_practice', 'divisor_digits': 3},
    {'operation': 'division_practice', 'instructions': 'show answer'},
    {'operation': 'print_output', 'code': "print(1 + 2)"},
    {'operation': 'print_output', 'code': "import os"}])
def test_invalid_action_never_writes_or_falls_back_to_model(action_api, action):
    browser, store, headers = action_api
    result = browser.post('/v1/chat/completions', headers=headers,
        json=action_payload(start_session(browser, headers), action))
    assert result.status_code == 422, result.text
    assert not store.sources and not store.memory_jobs


@pytest.mark.parametrize('messages', [None, {}, '', [{'role': 'system', 'content': 'do something'}],
    [{'role': 'user', 'content': 'только подсказка'}], [None]])
def test_explicit_action_rejects_every_nonempty_or_malformed_dialogue(action_api, messages):
    browser, store, headers = action_api
    payload = action_payload(start_session(browser, headers), {'operation': 'division_practice'})
    payload['messages'] = messages
    result = browser.post('/v1/chat/completions', headers=headers, json=payload)
    assert result.status_code == 422
    assert not store.sources and not store.memory_jobs


def test_replay_changed_parameters_and_cross_mode_identity(action_api):
    browser, store, headers = action_api
    payload = action_payload(start_session(browser, headers),
        {'operation': 'division_check', 'dividend': 203, 'divisor': 2, 'stated_quotient': 101})
    first = browser.post('/v1/chat/completions', headers=headers, json=payload)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body['learning_tool']['facts']['stated_equality_correct'] is False
    assert 'остаток 1' in body['choices'][0]['message']['content']
    assert browser.post('/v1/chat/completions', headers=headers, json=payload).json() == body
    changed = deepcopy(payload)
    changed['learning_action']['dividend'] = 202
    assert browser.post('/v1/chat/completions', headers=headers, json=changed).status_code == 409
    changed = deepcopy(payload)
    del changed['learning_action']
    changed['messages'] = [{'role': 'user', 'content': body['learning_action']['request_text']}]
    assert browser.post('/v1/chat/completions', headers=headers, json=changed).status_code == 409
    assert len(store.sources) == 2


@pytest.mark.parametrize('topic,subject,operation', [
    ('math.forged', 'math', 'division_check'), ('scratch', 'python', 'print_output'),
    ('math', 'python', 'print_output'), ('python', 'math', 'division_check'),
    ('math', 'scratch', 'division_check'),
])
def test_server_session_subject_cannot_be_overridden(action_api, topic, subject, operation):
    browser, store, headers = action_api
    session_id = start_session(browser, headers)
    store.learning_sessions[session_id]['topic_id'] = topic
    action = {'operation': operation, 'dividend': 202, 'divisor': 2} if operation == 'division_check' else {'operation': operation, 'code': "print('yes')"}
    payload = action_payload(session_id, action)
    payload['metadata']['subject_id'] = subject
    result = browser.post('/v1/chat/completions', headers=headers, json=payload)
    assert result.status_code == 422, result.text
    assert not store.sources


def test_places_practice_and_literal_print_preserve_boundaries(action_api):
    browser, store, headers = action_api
    session_id = start_session(browser, headers)
    payload = action_payload(session_id, {'operation': 'division_places', 'dividend': 6750, 'divisor': 25})
    body = browser.post('/v1/chat/completions', headers=headers, json=payload).json()
    assert body['learning_tool']['facts']['quotient'] == 270
    pairs = []
    for number in range(2):
        payload = action_payload(session_id, {'operation': 'division_practice', 'divisor_digits': 2}, f'practice-{number}')
        result = browser.post('/v1/chat/completions', headers=headers, json=payload)
        assert result.status_code == 200, result.text
        facts = result.json()['learning_tool']['facts']
        assert set(facts) == {'dividend', 'divisor'}
        assert divmod(facts['dividend'], facts['divisor'])[1] == 0
        pairs.append((facts['dividend'], facts['divisor']))
    assert pairs[0] != pairs[1]
    session_id = start_session(browser, headers)
    store.learning_sessions[session_id]['topic_id'] = 'python'
    code = "print('\"да\"', '<script>', sep=' | ', end='!')\nprint('```', end='')\n"
    payload = action_payload(session_id, {'operation': 'print_output', 'code': code})
    payload['metadata']['subject_id'] = 'python'
    result = browser.post('/v1/chat/completions', headers=headers, json=payload)
    assert result.status_code == 200, result.text
    assert result.json()['learning_tool']['facts']['stdout'] == '"да" | <script>!```'
    assert code in result.json()['learning_action']['request_text']


def test_explicit_action_requires_ids_and_checks_owner_before_cache(action_api):
    browser, store, headers = action_api
    payload = action_payload(start_session(browser, headers), {'operation': 'division_practice'})
    for field in ('learning_session_id', 'client_message_id'):
        invalid = deepcopy(payload)
        del invalid[field]
        assert browser.post('/v1/chat/completions', headers=headers, json=invalid).status_code == 422
    assert browser.post('/v1/chat/completions', headers=headers, json=payload).status_code == 200
    store.learning_sessions[payload['learning_session_id']]['actor_id'] = 'another-student'
    assert browser.post('/v1/chat/completions', headers=headers, json=payload).status_code == 403
    assert len(store.sources) == 2


def test_long_backtick_literal_fits_source_contract_without_rewriting(action_api):
    browser, store, headers = action_api
    session_id = start_session(browser, headers)
    store.learning_sessions[session_id]['topic_id'] = 'python'
    code = "print('" + '`' * 3980 + "')"
    payload = action_payload(session_id, {'operation': 'print_output', 'code': code})
    del payload['messages']  # Omitted history is also valid for typed actions.
    payload['metadata']['subject_id'] = 'python'
    response = browser.post('/v1/chat/completions', headers=headers, json=payload)
    assert response.status_code == 200, response.text
    assert code in response.json()['learning_action']['request_text']
    assert response.json()['learning_tool']['facts']['stdout'] == '`' * 3980 + '\n'
