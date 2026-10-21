"""Completed guided actions survive actual API process shutdown and restart.

Only a disposable loopback PostgreSQL database is used. The child API forbids
model/RAG calls; the memory worker uses a controlled interest-extraction fixture.
This covers committed results, not an unknown in-flight provider outcome.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import requests

from local_app.auth import hash_password
from local_app.memory_service import MemoryWorker
from scripts.load_memory import ControlledRuntime, require_disposable_dsn
if __name__ != '__main__':
    from tests.test_memory_api_flow import flow_store  # noqa: F401

LEARNER = 'demo-nika-04'
TOPIC = 'math.g4.numbers.division_by_1_2_digit'


def serve(directory):
    dsn = os.environ['TEACHCOPILOT_TEST_DSN']
    require_disposable_dsn(dsn)
    import socket
    import threading
    import uvicorn
    from local_app.api import create_app
    from local_app.config import Settings
    from local_app.store import Store

    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError('Restart probe must not call model or RAG: ' + name)

    # The serving app uses only this explicit DSN and forbidden adapters.
    app = create_app(settings=Settings(database_url=dsn), store=Store(dsn),
                     runtime=Forbidden(), rag_service=Forbidden())
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
    finished = threading.Event()

    def watch_stop():
        while not finished.wait(0.05):
            if (directory / 'stop').exists():
                server.should_exit = True
                return

    watcher = threading.Thread(target=watch_stop, daemon=True)
    watcher.start()
    (directory / 'address.json').write_text(json.dumps({
        'pid': os.getpid(), 'port': sock.getsockname()[1],
        'nonce': os.environ['TEACHCOPILOT_RESTART_NONCE'],
    }), encoding='utf-8')
    try:
        server.run(sockets=[sock])
    finally:
        finished.set()
        watcher.join(timeout=1)
        sock.close()


@contextmanager
def running_api(store, directory):
    require_disposable_dsn(store.dsn)
    directory.mkdir()
    nonce = uuid4().hex
    env = dict(os.environ, TEACHCOPILOT_TEST_DSN=store.dsn, DATABASE_URL=store.dsn,
               TEACHCOPILOT_RESTART_NONCE=nonce, PYTHONIOENCODING='utf-8',
               TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG='', LOCAL_DEMO_LOGIN='0',
               OLLAMA_URL='http://127.0.0.1:9')
    with (directory / 'server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve', str(directory)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        owned = psutil.Process(process.pid)
        client = requests.Session()
        client.trust_env = False
        info = None
        api_process = None
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError('API child exited before readiness; see server.log')
                if (directory / 'address.json').exists():
                    try:
                        info = json.loads((directory / 'address.json').read_text(encoding='utf-8'))
                    except json.JSONDecodeError:
                        time.sleep(0.05)
                        continue
                    assert info['nonce'] == nonce
                    assert info['pid'] in {process.pid, *(child.pid for child in owned.children(recursive=True))}
                    api_process = psutil.Process(info['pid'])
                    base = 'http://127.0.0.1:' + str(info['port'])
                    try:
                        if client.get(base + '/health/live', timeout=0.5).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                time.sleep(0.05)
            else:
                raise AssertionError('API child exceeded finite readiness timeout')
            yield client, base, info['pid']
        finally:
            client.close()
            try:
                children = owned.children(recursive=True)
            except psutil.NoSuchProcess:
                children = []
            if api_process is not None and all(child.pid != api_process.pid for child in children):
                children.append(api_process)
            (directory / 'stop').touch()
            try:
                process.wait(timeout=10)
                if api_process is not None:
                    api_process.wait(timeout=3)
            except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
                # Retained Process identities also guard against PID reuse.
                for child in reversed(children):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                raise AssertionError('API child did not shut down cleanly')
            assert process.returncode == 0


class EmptyInterests(ControlledRuntime):
    def chat(self, messages, response_format=None, **kwargs):
        assert 'candidates' in (response_format or {}).get('properties', {})
        self.calls.append({'kind':'controlled_empty_interest_extraction'})
        return {'message':{'content':'{"candidates":[]}'}, 'prompt_eval_count':0, 'eval_count':0}


def test_step_tokens_replays_and_reviewed_future_lesson_survive_api_process_restarts(flow_store, tmp_path):
    store = flow_store
    password = 'Synthetic restart password 123'
    teacher = store.create_account('restart-teacher@synthetic.test', 'Teacher', hash_password(password), 'teacher')
    store.assign_learner(teacher['id'], LEARNER)
    student = store.create_student_account('restart-student@synthetic.test', 'Student', hash_password(password), LEARNER, teacher['id'])
    knowledge = deepcopy(store.get_card(LEARNER).get('knowledge'))

    def durable_snapshot():
        from psycopg2 import sql
        snapshot = {}
        with store._connect() as connection, connection.cursor() as cursor:
            for table in ('memory_sources', 'learning_messages', 'memory_jobs'):
                cursor.execute(sql.SQL('SELECT to_jsonb(t) AS row FROM teachcopilot.{} t WHERE learner_id=%s ORDER BY id')
                               .format(sql.Identifier(table)), (LEARNER,))
                snapshot[table] = [row['row'] for row in cursor.fetchall()]
        snapshot['source_revision'] = store.get_learner_source_revision(LEARNER)
        return snapshot

    def send(client, base, headers, sid, mid, operation, **values):
        return client.post(base+'/v1/chat/completions', headers=headers, json={
            'learning_session_id':sid, 'client_message_id':mid,
            'learning_action':{'operation':operation, **values}}, timeout=10)

    def context(client, base, headers):
        result = client.get(base+f'/learners/{LEARNER}/guided-division', headers=headers,
                            params={'topic_id':TOPIC}, timeout=10)
        assert result.status_code == 200, result.text
        assert '_guided_division_provenance' not in result.text
        return result.json()

    def new_session(client, base, headers):
        result = client.post(base+f'/learners/{LEARNER}/learning-sessions', headers=headers,
                             json={'topic_id':TOPIC}, timeout=10)
        assert result.status_code == 201, result.text
        return result.json()['learning_session_id']

    def recorded_conversation(sid, mid, text):
        # A persisted controlled conversation fixture, not a model-quality run.
        actor = store.get_learning_session(sid)['actor_id']
        store.reserve_chat_message(sid,LEARNER,mid,text,actor)
        store.complete_chat_message(sid,mid,{'choices':[{'message':{
            'role':'assistant','content':'Можем вернуться к сохранённому шагу позже.'}}]})

    with running_api(store, tmp_path/'process-1') as (client, base, first_pid):
        def login(account):
            result = client.post(base+'/auth/login', json={'email':account['email'],'password':password}, timeout=10)
            assert result.status_code == 200
            return {'Authorization':'Bearer '+result.json()['access_token']}
        student_headers, teacher_headers = login(student), login(teacher)
        sid = new_session(client,base,student_headers)
        reply = send(client,base,student_headers,sid,'start','division_guided_start',dividend=3636,divisor=12)
        assert reply.status_code == 200, reply.text
        state = reply.json()['guided_division']
        for index,value in enumerate((3,0,3)):
            reply = send(client,base,student_headers,sid,f'answer-{index}','division_guided_answer',
                         expected_state_id=state['state_id'],value=value)
            assert reply.status_code == 200, reply.text
            state = reply.json()['guided_division']
        before_pause_id = state['state_id']
        paused = send(client,base,student_headers,sid,'pause','division_guided_pause',expected_state_id=before_pause_id)
        assert paused.status_code == 200, paused.text
        saved = paused.json()
        state = saved['guided_division']
        assert state['step_index'] == 1 and state['phase'] == 'quotient_digit' and state['status'] == 'paused'
        assert '_guided_division_provenance' not in saved
        recorded_conversation(sid,'aside','Я вчера построил станцию в Minecraft.')
        source_ids = [row['source_id'] for row in store.list_learning_messages(sid)]
        saved_snapshot = durable_snapshot()

    with running_api(store, tmp_path/'process-2') as (client, base, second_pid):
        assert second_pid != first_pid
        restored = context(client,base,student_headers)
        assert restored['guided_division'] == state and restored['active_learning_session_id'] == sid
        assert restored['resume_required'] is True
        replay = send(client,base,student_headers,sid,'pause','division_guided_pause',expected_state_id=before_pause_id)
        assert replay.status_code == 200 and replay.json() == saved
        assert [row['source_id'] for row in store.list_learning_messages(sid)] == source_ids
        assert durable_snapshot() == saved_snapshot
        assert context(client,base,student_headers)['resume_required'] is True
        denied = client.get(base+f'/learners/{LEARNER}/guided-division', headers=teacher_headers,
                            params={'topic_id':TOPIC,'learning_session_id':sid}, timeout=10)
        assert denied.status_code == 403
        foreign = client.get(base+'/learners/foreign/guided-division', headers=student_headers,
                             params={'topic_id':TOPIC}, timeout=10)
        assert foreign.status_code == 403
        completed = client.post(base+f'/learning-sessions/{sid}/complete', headers=student_headers, timeout=10)
        assert completed.status_code == 202
        assert context(client,base,student_headers)['reason'] == 'teacher_review_required'
        runtime = EmptyInterests(0)
        job = MemoryWorker(store,runtime).run_once()
        assert job['status'] == 'completed', job
        assert runtime.calls and all(call['kind'] == 'controlled_empty_interest_extraction' for call in runtime.calls)
        snapshot = client.get(base+f'/learners/{LEARNER}/memory',headers=teacher_headers,timeout=10).json()
        version = snapshot['versions'][0]
        assert version['context']['interest_extraction']['status'] == 'ready'
        confirmation = client.post(base+f"/learners/{LEARNER}/memory/versions/{version['version_id']}/confirm",
            headers=teacher_headers,json={'expected_revision':snapshot['revision']},timeout=10)
        assert confirmation.status_code == 200, confirmation.text

    with running_api(store, tmp_path/'process-3') as (client, base, third_pid):
        assert third_pid not in {first_pid,second_pid}
        reviewed = context(client,base,student_headers)
        assert reviewed['available'] and reviewed['origin'] == 'approved_memory'
        assert reviewed['guided_division'] == state
        assert reviewed['resume_required'] is True
        second_sid = new_session(client,base,student_headers)
        recorded_conversation(second_sid,'greeting','привет')
        assert context(client,base,student_headers)['resume_required'] is True
        blocked = send(client,base,student_headers,second_sid,'not-resumed','division_guided_answer',
                       expected_state_id=state['state_id'],value=0)
        assert blocked.status_code == 409
        resume = send(client,base,student_headers,second_sid,'resume','division_guided_resume',
                      expected_state_id=state['state_id'],source_learning_session_id=sid)
        assert resume.status_code == 200, resume.text
        assert resume.json()['guided_assessment']['status'] == 'not_assessed'
        assert resume.json()['guided_division']['step_index'] == 1
        assert '3 на 12' in resume.json()['choices'][0]['message']['content']
        answer = send(client,base,student_headers,second_sid,'zero','division_guided_answer',expected_state_id='resume',value=0)
        assert answer.status_code == 200, answer.text
        assert answer.json()['guided_division']['phase'] == 'remainder'
        assert answer.json()['guided_assessment']['independent_work'] == 'unknown'
    assert store.get_card(LEARNER).get('knowledge') == knowledge
    final_sources = {row['id']:row for row in durable_snapshot()['memory_sources']}
    assert all(final_sources[row['id']] == row for row in saved_snapshot['memory_sources'])


if __name__ == '__main__':
    assert len(sys.argv) == 3 and sys.argv[1] == '--serve'
    serve(Path(sys.argv[2]))
