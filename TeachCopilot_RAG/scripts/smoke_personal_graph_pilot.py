"""Synthetic local pilot for graph index reuse and teacher-controlled archive access.

Create a fresh database first with create_personal_graph_test_db.py, then pass its
name file here. This script intentionally leaves that isolated database and its
Neo4j scopes in place for inspection. It never prints credentials or the DSN.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import json
import logging
from pathlib import Path
import secrets
import shutil
import sys
import time
from uuid import uuid4

from fastapi.testclient import TestClient
from psycopg2.extensions import make_dsn, parse_dsn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.answer_adapter import AnswerAdapter
from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
from local_app.memory_service import MemoryWorker
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from local_app.store import Store


class ObservedRag(RagService):
    def answer(self, messages, **kwargs):
        self.observed = {"messages": messages, "memory_context": kwargs.get("memory_context")}
        return super().answer(messages, **kwargs)


class ObservedRuntime(OllamaRuntime):
    """Capture model chat payloads only during selected synthetic pilot steps."""
    captured_chat_calls = None
    capture_last_only = False

    def _post(self, path, payload):
        calls = self.captured_chat_calls if path == '/api/chat' else None
        if calls is None:
            return super()._post(path, payload)
        record = {'path': path, 'request': deepcopy(payload)}
        if self.capture_last_only:
            calls[:] = [record]
        else:
            calls.append(record)
        try:
            response = super()._post(path, payload)
            record['response'] = deepcopy(response)
            return response
        except BaseException as exc:
            record['error_type'] = type(exc).__name__
            raise


class SmokeCheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise SmokeCheckError(message)


def run(args):
    # Exclusive creation makes the artifact immutable with respect to prior runs.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"scientific_result": False, "synthetic": True, "ok": False, "technical_ok": False,
              "ok_meaning": "technical checks only", "stages": [],
              "conflicting_corrections": bool(args.conflicting_corrections),
              "quality": {"assessment": "keyword heuristics only; human review required",
                          "selected_chat": None, "old_event_chat": None}}
    with args.output.open('x', encoding='utf-8') as output:
        def save():
            output.seek(0)
            json.dump(report, output, ensure_ascii=False, indent=2, default=str)
            output.write('\n')
            output.truncate()
            output.flush()

        @contextmanager
        def stage(name):
            row = {"stage": name, "status": "running"}
            report['stages'].append(row)
            save()
            started = time.perf_counter()
            try:
                yield row
                row['status'] = 'passed'
            except BaseException as exc:
                # Exception strings can contain connection credentials.
                row.update(status='failed', error_type=type(exc).__name__)
                if isinstance(exc, SmokeCheckError):
                    row['error'] = str(exc)
                raise
            finally:
                row['seconds'] = round(time.perf_counter() - started, 3)
                save()
                print(f"{name}: {row['status']}", flush=True)

        save()
        print(f"artifact: {args.output.resolve()}", flush=True)
        try:
            with stage('fixture') as row:
                dbname = args.db_name_file.read_text(encoding='ascii').strip()
                require(dbname.startswith('nirs_personal_graph_it_') and dbname.replace('_', '').isalnum(),
                        'Expected an isolated synthetic database')
                settings = Settings.from_env()
                params = parse_dsn(settings.database_url)
                import os
                hosts = (params.get('host'), params.get('hostaddr'), os.getenv('PGHOSTADDR'))
                require(any(hosts) and all(not h or h in {'localhost', '127.0.0.1', '::1'} for h in hosts),
                        'A local PostgreSQL endpoint is required')
                dsn = make_dsn(settings.database_url, dbname=dbname)
                nonce = uuid4().hex[:12]
                settings = replace(settings, database_url=dsn, generation_scheduler_dsn=dsn,
                    context_tokens=16384, experimental_personal_graph_enabled=True, request_timeout=300,
                    memory_response_journal_path=str(args.output.parent / ('pilot-memory-' + nonce + '.sqlite3')),
                    chat_response_journal_path=str(args.output.parent / ('pilot-chat-' + nonce + '.sqlite3')))
                store = Store(dsn)
                with store._connect() as conn, conn.cursor() as cur:
                    cur.execute('SELECT count(*) AS n FROM teachcopilot.accounts')
                    require(cur.fetchone()['n'] == 0, 'Database is not fresh')
                    cur.execute('SELECT count(*) AS n FROM teachcopilot.memory_jobs')
                    require(cur.fetchone()['n'] == 0, 'Database is not fresh')
                fixture = args.output.parent / ('pilot-fixture-' + nonce)
                (fixture / 'learner-data/learners').mkdir(parents=True)
                shutil.copytree(ROOT.parent / 'learner-data/catalog', fixture / 'learner-data/catalog')
                card = json.loads((ROOT.parent / 'learner-data/demo/learners/demo-nika-04.json').read_text(encoding='utf-8-sig'))
                learner, topic = 'graph-pilot-' + nonce, 'math.g4.fractions.numerator_denominator'
                card.update(learner_id=learner, pseudonym='Синтетический ученик')
                (fixture / 'learner-data/learners' / (learner + '.json')).write_text(
                    json.dumps(card, ensure_ascii=False), encoding='utf-8')
                store.import_repository(fixture)
                password = secrets.token_urlsafe(18)
                teacher = store.create_account('teacher-' + nonce + '@synthetic.test', 'Synthetic teacher',
                                               hash_password(password), 'teacher')
                store.assign_learner(teacher['id'], learner)
                student = store.create_student_account('student-' + nonce + '@synthetic.test', 'Synthetic student',
                                                       hash_password(password), learner, teacher['id'])
                note = store.add_memory_source(learner, topic,
                    'На старом занятии ребёнок придумал фиолетового дракона по имени Лунтик.',
                    teacher['id'], approved=True)
                def lesson(label, pairs):
                    session = store.create_learning_session(learner, student['id'], topic,
                                                            client_session_id=label + '-' + nonce)
                    source_ids = []
                    for number, (user, assistant) in enumerate(pairs):
                        message_id = label + '-' + str(number)
                        reserved = store.reserve_chat_message(session['id'], learner, message_id, user, student['id'])
                        source_ids.append(reserved['source_id'])
                        store.complete_chat_message(session['id'], message_id, {'choices': [{'message': {
                            'role': 'assistant', 'content': assistant}}]})
                    store.complete_learning_session(session['id'], learner, student['id'])
                    return source_ids
                old_student_source_ids = lesson('old', [('Я придумал фиолетового дракона Лунтика.', 'Как звали твоего дракона?'),
                               ('Лунтик.', 'Запомнили имя дракона.')])
                lesson('recent', [('Я думаю, три восьмых.', 'Сколько равных частей у целого?'),
                                  ('Четыре, значит три четверти.', 'Хорошо, продолжим в следующий раз.')])
                runtime = ObservedRuntime(settings)
                rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
                journal = MemoryResponseJournal(settings.memory_response_journal_path,
                                                database_identity=database_fingerprint(dsn))
                worker = MemoryWorker(store, runtime, response_journal=journal)
                client = TestClient(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag))
                def auth(account):
                    response = client.post('/auth/login', json={'email': account['email'], 'password': password})
                    require(response.status_code == 200, 'Synthetic login failed')
                    return {'Authorization': 'Bearer ' + response.json()['access_token']}
                th, sh = auth(teacher), auth(student)
                row.update(database=dbname, learner=learner, source_id=note['source_id'])
                if args.conflicting_corrections:
                    row['old_name_source_ids'] = [note['source_id'], *old_student_source_ids]

            def teacher_step(row):
                queued = client.post(f'/learners/{learner}/next-step', headers=th,
                                     json={'topic_id': topic, 'method': 'personal_graph'})
                row['queue_status'] = queued.status_code
                require(queued.status_code == 202, 'Teacher job rejected')
                job_id = queued.json()['job_id']
                calls = []
                runtime.captured_chat_calls = calls
                runtime.capture_last_only = True
                try:
                    job = worker.run_once()
                finally:
                    runtime.captured_chat_calls = None
                    runtime.capture_last_only = False
                    row['last_local_chat_call'] = calls[-1] if calls else None
                row['job'] = job
                require(job and str(job.get('job_id')) == str(job_id) and job.get('status') == 'completed',
                        'Teacher worker failed')
                result = job.get('result') or {}
                version_id = result.get('memory_version_id')
                require(version_id, 'Teacher memory version missing')
                versions = store.get_learner_memory(learner, None)['versions']
                version = next((item for item in versions if str(item.get('id') or item.get('version_id')) == str(version_id)), None)
                require(version is not None, 'Stored memory version missing')
                row['version'] = version
                row['extract_calls'] = sum(call.get('purpose') == 'personal_graph_extract'
                                           for call in (result.get('calls') or []))
                return version_id, version

            def preview():
                response = client.get(f'/learners/{learner}/memory?include_personal_graph_snapshot=true', headers=th)
                require(response.status_code == 200, 'Teacher archive preview failed')
                return response.json()

            def approve(row, version_id, full=False):
                snapshot = preview()
                row['preview'] = snapshot
                payload = {'expected_revision': snapshot['revision']}
                if full:
                    payload.update(reviewed_full_archive=True, allow_full_archive=True,
                                   archive_snapshot_fingerprint=snapshot['personal_graph_snapshot_fingerprint'])
                response = client.post(f'/learners/{learner}/memory/versions/{version_id}/confirm',
                                       headers=th, json=payload)
                row.update(request=payload, status_code=response.status_code, response=response.json())
                require(response.status_code == 200, 'Teacher approval failed')
                expected = 'full_archive' if full else 'selected_context'
                require(response.json()['version']['context']['source_access']['mode'] == expected,
                        'Unexpected approval mode')

            def chat(row, session_id, message_id, question):
                payload = {'learning_session_id': session_id, 'client_message_id': message_id,
                           'messages': [{'role': 'user', 'content': question}],
                           'metadata': {'memory_method': 'personal_graph'}}
                rag.observed = None
                calls = []
                runtime.captured_chat_calls = calls
                try:
                    response = client.post('/v1/chat/completions', headers=sh, json=payload)
                finally:
                    runtime.captured_chat_calls = None
                    row['local_chat_calls'] = calls
                row.update(request=payload, status_code=response.status_code, response=response.json(),
                           observed=rag.observed)
                require(response.status_code == 200, 'Student chat failed')
                answer = response.json().get('choices', [{}])[0].get('message', {}).get('content')
                require(isinstance(answer, str) and answer.strip(), 'Student answer missing')
                row['answer'] = answer
                return response

            with stage('teacher_first') as row:
                first_id, first = teacher_step(row)
                require(first['state'].get('index_reused') is False, 'First index must be new')
            with stage('approve_selected') as row:
                approve(row, first_id)
                before_chat_preview = row['preview']
            with stage('chat_selected') as row:
                response = client.post(f'/learners/{learner}/learning-sessions', headers=sh,
                    json={'topic_id': topic, 'client_session_id': 'pilot-chat-' + nonce})
                row['session_status'] = response.status_code
                require(response.status_code == 201, 'Student session failed')
                session_id = response.json()['learning_session_id']
                chat(row, session_id, 'selected-1', 'Какой вопрос помог мне исправить дробь?')
                context = ((row.get('observed') or {}).get('memory_context') or {}).get('text') or ''
                answer = row['answer'].lower()
                report['quality']['selected_chat'] = {
                    'expected_question_in_context': 'сколько равных частей у целого' in context.lower(),
                    'answer_mentions_equal_parts': 'равн' in answer and 'част' in answer,
                    'answer_says_information_missing': any(phrase in answer for phrase in
                        ('нет информации', 'не знаю', 'не помню', 'не могу вспомнить', 'неизвестно')),
                }
                after_chat_preview = preview()
                row['snapshot_after_chat'] = {
                    'fingerprint': after_chat_preview['personal_graph_snapshot_fingerprint'],
                    'source_revision': after_chat_preview['source_revision'],
                    'unchanged_from_before_chat': after_chat_preview['personal_graph_snapshot_fingerprint'] ==
                                                  before_chat_preview['personal_graph_snapshot_fingerprint'],
                }
                require(row['snapshot_after_chat']['unchanged_from_before_chat'],
                        'Active chat changed the eligible graph snapshot')
            with stage('teacher_unchanged') as row:
                second_id, second = teacher_step(row)
                row['identity_comparison'] = {
                    'snapshot_unchanged': second['state'].get('archive_snapshot_fingerprint') ==
                                          first['state'].get('archive_snapshot_fingerprint'),
                    'config_unchanged': second['state'].get('index_config_fingerprint') ==
                                        first['state'].get('index_config_fingerprint'),
                    'source_revision_before': first['state'].get('source_revision'),
                    'source_revision_after': second['state'].get('source_revision'),
                }
                require(row['identity_comparison']['snapshot_unchanged'], 'Graph snapshot changed during pilot')
                require(row['identity_comparison']['config_unchanged'], 'Index configuration changed during pilot')
                require(second['state'].get('index_reused') is True, 'Unchanged snapshot was not reused')
                require(second['state'].get('archive_run_id') == first['state'].get('archive_run_id'), 'Run differs')
                require(row['extract_calls'] == 0, 'Index extraction unexpectedly repeated')
            with stage('approve_full_archive') as row:
                approve(row, second_id, full=True)
            with stage('chat_old_event') as row:
                chat(row, session_id, 'old-event-1', 'Какого дракона я придумал и как его звали?')
                context = ((row.get('observed') or {}).get('memory_context') or {}).get('text') or ''
                row['old_event_in_context'] = 'дракон' in context.lower() and 'лунтик' in context.lower()
                answer = row['answer'].lower()
                report['quality']['old_event_chat'] = {
                    'old_event_in_context': row['old_event_in_context'],
                    'answer_mentions_name': 'лунтик' in answer,
                    'answer_mentions_colour': 'фиолетов' in answer,
                    'answer_says_information_missing': any(phrase in answer for phrase in
                        ('нет информации', 'не знаю', 'не помню', 'не могу вспомнить', 'неизвестно')),
                }
                require(row['old_event_in_context'], 'Old event was not retrieved into the approved context')
            with stage('correction') as row:
                source_ids = ([note['source_id'], *old_student_source_ids]
                              if args.conflicting_corrections else [note['source_id']])
                correction_rows = []
                row['corrections'] = correction_rows
                for position, source_id in enumerate(source_ids):
                    snapshot = preview()
                    payload = {'text': ('В исходной записи имя указано ошибочно; правильное имя дракона – Искорка.'
                                        if args.conflicting_corrections else
                                        'Уточнение: фиолетового дракона звали Лунтик.'),
                               'expected_revision': snapshot['revision'],
                               'expected_source_revision': snapshot['source_revision'],
                               'client_correction_id': ('pilot-correction-' + nonce + '-' + str(position)
                                                        if args.conflicting_corrections else
                                                        'pilot-correction-' + nonce)}
                    response = client.post(f'/learners/{learner}/memory/sources/{source_id}/corrections',
                                           headers=th, json=payload)
                    item = {'source_id': source_id, 'request': payload, 'status_code': response.status_code,
                            'response': response.json()}
                    correction_rows.append(item)
                    if not args.conflicting_corrections:
                        row.update(request=payload, status_code=response.status_code, response=response.json())
                    require(response.status_code == 202, 'Correction failed')
                    require(str(item['response']['source'].get('corrects_source_id')) == str(source_id),
                            'Correction target differs from the requested source')
            with stage('stale_grant') as row:
                payload = {'learning_session_id': session_id, 'client_message_id': 'after-correction',
                           'messages': [{'role': 'user', 'content': 'Напомни про дракона.'}],
                           'metadata': {'memory_method': 'personal_graph'}}
                response = client.post('/v1/chat/completions', headers=sh, json=payload)
                row.update(request=payload, status_code=response.status_code, response=response.json())
                require(response.status_code == 409, 'Stale archive grant was not rejected')
            with stage('teacher_changed') as row:
                changed_id, changed = teacher_step(row)
                require(changed['state'].get('index_reused') is False, 'Changed snapshot reused old index')
                require(changed['state'].get('archive_run_id') != second['state'].get('archive_run_id'),
                        'Changed snapshot used old scope')
            if args.conflicting_corrections:
                with stage('approve_corrected_full_archive') as row:
                    approve(row, changed_id, full=True)
                with stage('chat_corrected_event') as row:
                    original_ids = [note['source_id'], *old_student_source_ids]
                    with store._connect() as conn, conn.cursor() as cur:
                        cur.execute('''SELECT id::text AS source_id, content, author_role,
                            corrects_source_id::text AS corrects_source_id
                            FROM teachcopilot.memory_sources WHERE learner_id=%s
                            AND (id::text=ANY(%s) OR corrects_source_id::text=ANY(%s))
                            ORDER BY source_order''', (learner, original_ids, original_ids))
                        original_and_corrections = [dict(source) for source in cur.fetchall()]
                    row['original_sources_and_corrections'] = original_and_corrections
                    originals = {source['source_id']: source for source in original_and_corrections
                                 if source['source_id'] in original_ids}
                    corrections = {source['corrects_source_id']: source for source in original_and_corrections
                                   if source['corrects_source_id'] in original_ids}
                    originals_preserved = (set(originals) == set(original_ids) and
                                           all('лунтик' in source['content'].lower() for source in originals.values()))
                    corrected_sources_present = (set(corrections) == set(original_ids) and
                        all('искорка' in source['content'].lower() and source['author_role'] == 'teacher'
                            for source in corrections.values()))
                    row['source_checks'] = {'old_name_sources_preserved': originals_preserved,
                                            'corrected_teacher_sources_present': corrected_sources_present}
                    require(originals_preserved and corrected_sources_present, 'Correction source records are incomplete')
                    chat(row, session_id, 'corrected-event-1', 'Как зовут дракона с учётом исправления записи?')
                    context = ((row.get('observed') or {}).get('memory_context') or {}).get('text') or ''
                    answer = row['answer'].lower()
                    report['quality']['corrected_event_chat'] = {
                        'old_name_sources_preserved': originals_preserved,
                        'corrected_teacher_sources_present': corrected_sources_present,
                        'corrected_name_in_context': 'искорк' in context.lower(),
                        'old_name_in_context': 'лунтик' in context.lower(),
                        'answer_mentions_corrected_name': 'искорк' in answer,
                        'answer_mentions_old_name': 'лунтик' in answer,
                    }
            report['ok'] = True
            report['technical_ok'] = True
        except BaseException as exc:
            report['error_type'] = type(exc).__name__
        finally:
            save()
    return report['ok']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-name-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--conflicting-corrections', action='store_true',
                        help='Run the additional synthetic conflicting-name correction and chat scenario')
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    if not run(args):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
