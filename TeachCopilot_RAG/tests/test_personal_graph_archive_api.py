"""Live HTTP approval checks against an isolated synthetic PostgreSQL database."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from scripts.load_memory import ControlledRuntime
from tests.test_memory_api_flow import flow_store, InspectableRag  # noqa: F401
from tests.test_session_end_context_store import LEARNER, TOPIC, OTHER_TOPIC, _lesson, _graph_memory, _personal_graph_draft

PASSWORD = 'Synthetic archive password 123'


@pytest.fixture
def archive_case(flow_store):
    store = flow_store
    teacher = store.create_account('archive-teacher@synthetic.test', 'Teacher', hash_password(PASSWORD), 'teacher')
    store.assign_learner(teacher['id'], LEARNER)
    outsider = store.create_account('archive-outsider@synthetic.test', 'Outsider', hash_password(PASSWORD), 'teacher')
    student = store.create_student_account('archive-student@synthetic.test', 'Student', hash_password(PASSWORD), LEARNER, teacher['id'])
    _, selected, _ = _lesson(store, text='SELECTED SYNTHETIC SOURCE')
    old = store.add_memory_source(LEARNER, OTHER_TOPIC, 'OLD SYNTHETIC SOURCE', 'teacher', author_role='teacher', approved=True)
    snapshot = _graph_memory(store, None)
    version, fingerprint = _personal_graph_draft(store, snapshot, selected['source_id'])
    app = create_app(settings=SimpleNamespace(session_hours=2, rag_limit=3, experimental_personal_graph_enabled=True), store=store, runtime=ControlledRuntime(0), rag_service=InspectableRag())
    with TestClient(app) as client:
        headers = {}
        for name, account in [('teacher', teacher), ('outsider', outsider), ('student', student)]:
            login = client.post('/auth/login', json={'email': account['email'], 'password': PASSWORD})
            assert login.status_code == 200
            headers[name] = {'Authorization': 'Bearer ' + login.json()['access_token']}
        yield SimpleNamespace(store=store, app=app, client=client, headers=headers, snapshot=snapshot,
            version=version, fingerprint=fingerprint, selected=selected, old=old)


def _url(case):
    return f"/learners/{LEARNER}/memory/versions/{case.version['id']}/confirm"


@pytest.mark.parametrize('full', [False, True])
def test_snapshot_authorization_and_confirmation_modes(archive_case, full):
    c = archive_case
    path = f'/learners/{LEARNER}/memory?include_personal_graph_snapshot=true'
    for actor in ('outsider', 'student'):
        assert c.client.get(path, headers=c.headers[actor]).status_code == 403
        assert c.client.post(_url(c), headers=c.headers[actor], json={'expected_revision': c.snapshot['revision']}).status_code == 403
    preview = c.client.get(path, headers=c.headers['teacher'])
    assert preview.status_code == 200
    data = preview.json()
    assert data['personal_graph_snapshot_fingerprint'] == c.fingerprint
    ids = {s['source_id'] for s in data['personal_graph_sources']}
    assert c.selected['source_id'] in ids and c.old['source_id'] in ids
    base = {'expected_revision': data['revision']}
    assert c.client.post(_url(c), headers=c.headers['teacher'], json={**base, 'allow_full_archive': True}).status_code == 422
    assert c.client.post(_url(c), headers=c.headers['teacher'], json={**base, 'reviewed_full_archive': True, 'allow_full_archive': True}).status_code == 422
    choice = {**base, 'reviewed_full_archive': full, 'allow_full_archive': full}
    if full:
        choice['archive_snapshot_fingerprint'] = c.fingerprint
    result = c.client.post(_url(c), headers=c.headers['teacher'], json=choice)
    assert result.status_code == 200, result.text
    access = result.json()['version']['context']['source_access']
    assert access['mode'] == ('full_archive' if full else 'selected_context')
    allowed = set(access['source_ids']) if full else set(result.json()['version']['context']['evidence_ids'])
    assert c.selected['source_id'] in allowed
    assert (c.old['source_id'] in allowed) == full
    assert c.client.post(_url(c), headers=c.headers['teacher'], json=choice).status_code == 200
    opposite = {**base, 'reviewed_full_archive': not full, 'allow_full_archive': not full}
    if not full:
        opposite['archive_snapshot_fingerprint'] = c.fingerprint
    assert c.client.post(_url(c), headers=c.headers['teacher'], json=opposite).status_code == 409


def test_changed_snapshot_cannot_be_confirmed(archive_case):
    c = archive_case
    c.store.add_memory_source(LEARNER, OTHER_TOPIC, 'NEW SYNTHETIC SOURCE', 'teacher', author_role='teacher', approved=True)
    response = c.client.post(_url(c), headers=c.headers['teacher'], json={
        'expected_revision': c.snapshot['revision'], 'reviewed_full_archive': True,
        'allow_full_archive': True, 'archive_snapshot_fingerprint': c.fingerprint})
    assert response.status_code == 409, response.text
