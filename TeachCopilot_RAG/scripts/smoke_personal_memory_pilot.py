"""Live local source-memory pilot on a fresh, isolated synthetic PostgreSQL DB.

Run --prepare first. Only --run --code-frozen permits Ollama generation, after
the application recipe has been frozen. Reports contain synthetic prompts and
exact source IDs; console output never contains accounts, credentials or DSNs.
No database or shared service is deleted or stopped.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import time
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.answer_adapter import AnswerAdapter
from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
from local_app.memory_service import MemoryWorker, session_summary_views
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from local_app.store import Store


TOPIC = "math.g4.fractions.numerator_denominator"
OTHER_TOPIC = "math.g3.geometry.perimeter"
DB_PATTERN = re.compile(r"nirs_personal_memory_it_[0-9a-f]{12}")


class PilotFailure(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise PilotFailure(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def safe_settings(directory, dbname=None):
    settings = Settings.from_env()
    params = parse_dsn(settings.database_url)
    hosts = (params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR"))
    require(any(hosts) and all(not host or host in {"localhost", "127.0.0.1", "::1"} for host in hosts),
            "Only a loopback PostgreSQL endpoint is allowed")
    require(settings.ollama_url.rstrip("/") == "http://127.0.0.1:11435", "Expected the reserved local Ollama endpoint")
    require(settings.chat_model == "gemma4:12b" and settings.embedding_model.split(":")[0] == "bge-m3",
            "Expected local Gemma4:12b and BGE-M3")
    dsn = make_dsn(settings.database_url, dbname=dbname) if dbname else settings.database_url
    return replace(settings, database_url=dsn, generation_scheduler_dsn=dsn, context_tokens=16384,
                   request_timeout=300, personal_memory_enabled=True, personal_memory_retrieval_mode="hybrid",
                   experimental_personal_graph_enabled=False,
                   personal_graph_password_file=str(directory / "neo4j-unavailable-never-created"),
                   generation_run_budget_path=None,
                   memory_response_journal_path=str(directory / "memory-responses.sqlite3"),
                   chat_response_journal_path=str(directory / "chat-responses.sqlite3"))


def isolated_store(directory):
    dbname = (directory / "db-name.txt").read_text(encoding="ascii").strip()
    require(DB_PATTERN.fullmatch(dbname), "Only the dedicated synthetic database is allowed")
    settings = safe_settings(directory, dbname)
    return dbname, settings, Store(settings.database_url)


class DatedFixtureStore(Store):
    """Set historical observation dates on insertion, respecting immutability."""
    fixture_observed_on = None

    def _memory_source(self, cursor, *args, **kwargs):
        if self.fixture_observed_on is not None and kwargs.get("observed_on") is None:
            kwargs["observed_on"] = self.fixture_observed_on
        return super()._memory_source(cursor, *args, **kwargs)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    name_file = directory / "db-name.txt"
    require(not name_file.exists(), "The pilot database name file already exists")
    settings = safe_settings(directory)
    dbname = "nirs_personal_memory_it_" + uuid4().hex[:12]
    connection = psycopg2.connect(settings.database_url, connect_timeout=10)
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    finally:
        connection.close()
    name_file.write_text(dbname + "\n", encoding="ascii")
    _, settings, store = isolated_store(directory)
    applied = store.migrate()
    store = DatedFixtureStore(settings.database_url)
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT count(*) AS n FROM teachcopilot.accounts")
        require(cursor.fetchone()["n"] == 0, "The database is not fresh")
    nonce = uuid4().hex[:12]
    learner, other_learner = "source-pilot-" + nonce, "source-pilot-other-" + nonce
    fixture = directory / ("fixture-" + nonce)
    (fixture / "learner-data/learners").mkdir(parents=True)
    shutil.copytree(ROOT.parent / "learner-data/catalog", fixture / "learner-data/catalog")
    template = json.loads((ROOT.parent / "learner-data/demo/learners/demo-nika-04.json").read_text(encoding="utf-8-sig"))
    for identifier in (learner, other_learner):
        card = deepcopy(template)
        card.update(learner_id=identifier, pseudonym="Синтетический ученик")
        write_json(fixture / "learner-data/learners" / (identifier + ".json"), card)
    store.import_repository(fixture)
    password = secrets.token_urlsafe(24)
    teacher = store.create_account("teacher-" + nonce + "@synthetic.test", "Synthetic teacher", hash_password(password), "teacher")
    students = []
    for index, identifier in enumerate((learner, other_learner)):
        store.assign_learner(teacher["id"], identifier)
        students.append(store.create_student_account(f"student-{index}-{nonce}@synthetic.test", "Synthetic student",
                                                    hash_password(password), identifier, teacher["id"]))
    write_json(directory / "PRIVATE-auth.json", {"synthetic": True, "password": password,
        "accounts": [{"id": teacher["id"], "role": "teacher", "email": teacher["email"], "learner_id": None},
                     *[{"id": account["id"], "role": "student", "email": account["email"], "learner_id": identifier}
                       for account, identifier in zip(students, (learner, other_learner))]]})
    sessions, old_source = [], None
    baseline = datetime(2026, 11, 1, 8, tzinfo=timezone.utc)
    for index in range(8):
        finished = baseline + timedelta(days=min(index, 6), hours=int(index == 7))
        store.fixture_observed_on = finished.date().isoformat()
        session = store.create_learning_session(learner, students[0]["id"], TOPIC,
                                                client_session_id=f"seed-{index}-{nonce}")
        if index == 0:
            pairs = [("Я нарисовала синий корабль по имени Сокол.", "Спасибо за рассказ о корабле.")]
        elif index == 7:
            pairs = [("Не знаю, как различить числитель и знаменатель.", "Сколько равных частей у целого?"),
                     ("Четыре, тогда закрашены три четверти.", "Ты назвала число частей. Самостоятельность ещё не проверяли.")]
        else:
            pairs = [(f"На занятии {index + 1} я подписала числитель дроби и попросила проверить ответ.",
                      "Покажи, какие части закрашены на рисунке.")]
        source_ids = []
        for turn, (question, answer) in enumerate(pairs):
            key = f"seed-{index}-{turn}"
            reserved = store.reserve_chat_message(session["id"], learner, key, question, students[0]["id"])
            source_ids.append(reserved["source_id"])
            store.complete_chat_message(session["id"], key, {"choices": [{"message": {"role": "assistant", "content": answer}}]})
        completed = store.complete_learning_session(session["id"], learner, students[0]["id"])
        store.cancel_memory_job(completed["job_id"], "Synthetic fixture: no legacy writer generation requested")
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE teachcopilot.learning_sessions SET created_at=%s,completed_at=%s WHERE id=%s",
                           (finished - timedelta(minutes=10), finished, session["id"]))
        sessions.append({"session_id": session["id"], "completed_at": finished.isoformat(), "source_ids": source_ids})
        if index == 0:
            old_source = source_ids[0]
    store.fixture_observed_on = baseline.date().isoformat()
    other_note = store.add_memory_source(learner, OTHER_TOPIC, "В другой теме был макет красного маяка.", teacher["id"], approved=True)
    private_note = store.add_memory_source(other_learner, TOPIC, "Другой ученик назвал аппарат Янтарь.", teacher["id"], approved=True)
    manifest = {"synthetic": True, "database": dbname, "schema_version": store.schema_version(),
                "migrations_applied": applied, "learner": learner, "other_learner": other_learner,
                "topic": TOPIC, "other_topic": OTHER_TOPIC, "sessions": sessions, "old_source_id": old_source,
                "other_topic_source_id": other_note["source_id"], "other_learner_source_id": private_note["source_id"],
                "ollama_called": False, "summary_baseline": session_summary_views(store, learner)}
    write_json(directory / "fixture-manifest.json", manifest)
    print("prepare: passed; isolated fixture and persisted summaries created; Ollama was not called", flush=True)


class ObservedRuntime(OllamaRuntime):
    def __init__(self, settings):
        super().__init__(settings)
        self.captured = []

    def _post(self, path, payload):
        row = {"path": path, "request": deepcopy(payload)}
        self.captured.append(row)
        started = time.perf_counter()
        try:
            result = super()._post(path, payload)
            if path == "/api/embed":
                row["response"] = {key: value for key, value in result.items() if key != "embeddings"}
                row["response"]["embedding_dimensions"] = [len(vector) for vector in result.get("embeddings", [])]
            else:
                row["response"] = deepcopy(result)
            return result
        except BaseException as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["seconds"] = round(time.perf_counter() - started, 3)


class ObservedRag(RagService):
    observed = None

    def answer(self, messages, **kwargs):
        self.observed = {"messages": deepcopy(messages), "memory_context": deepcopy(kwargs.get("memory_context"))}
        return super().answer(messages, **kwargs)


def run(directory):
    report_path = directory / "live-report.json"
    require(not report_path.exists(), "The immutable live report already exists")
    dbname, settings, store = isolated_store(directory)
    fixture = json.loads((directory / "fixture-manifest.json").read_text(encoding="utf-8"))
    learner, other_learner = fixture["learner"], fixture["other_learner"]
    report = {"synthetic": True, "scientific_result": False, "technical_ok": False, "database": dbname,
              "quality": {"assessment": "manual review required; separate from technical success", "answers": []},
              "stages": [], "neo4j_factory_calls": 0}

    def save():
        write_json(report_path, report)

    @contextmanager
    def stage(name):
        row = {"stage": name, "status": "running"}
        report["stages"].append(row)
        save()
        started = time.perf_counter()
        call_start = len(runtime.captured)
        try:
            yield row
            row["status"] = "passed"
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, PilotFailure):
                row["error"] = str(exc)
            raise
        finally:
            row["seconds"] = round(time.perf_counter() - started, 3)
            row["local_calls"] = runtime.captured[call_start:]
            save()
            print(name + ": " + row["status"], flush=True)

    def unavailable_graph(*args, **kwargs):
        report["neo4j_factory_calls"] += 1
        raise PilotFailure("The Neo4j-backed provider is intentionally unavailable")

    runtime = ObservedRuntime(settings)
    rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    journal = MemoryResponseJournal(settings.memory_response_journal_path,
                                    database_identity=database_fingerprint(settings.database_url))
    worker = MemoryWorker(store, runtime, response_journal=journal, personal_graph_factory=unavailable_graph)
    client = TestClient(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag,
                                   personal_graph_factory=unavailable_graph))
    private = json.loads((directory / "PRIVATE-auth.json").read_text(encoding="utf-8"))
    password, accounts = private["password"], private["accounts"]

    def auth(role, identifier=None):
        account = next(item for item in accounts if item["role"] == role and (identifier is None or item["learner_id"] == identifier))
        response = client.post("/auth/login", json={"email": account["email"], "password": password})
        require(response.status_code == 200, "Synthetic account authentication failed")
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    teacher_headers, student_headers, other_headers = auth("teacher"), auth("student", learner), auth("student", other_learner)

    def preview():
        response = client.get(f"/learners/{learner}/memory?include_personal_graph_snapshot=true", headers=teacher_headers)
        require(response.status_code == 200, "Teacher preview failed")
        return response.json()

    def cache_state():
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT learner_id,count(*) AS count FROM teachcopilot.personal_graph_source_embedding_cache GROUP BY learner_id ORDER BY learner_id")
            return [dict(row) for row in cursor.fetchall()]

    def teacher_step(row):
        queued = client.post(f"/learners/{learner}/next-step", headers=teacher_headers,
                             json={"topic_id": TOPIC, "method": "personal_memory"})
        row["queue_status"] = queued.status_code
        require(queued.status_code == 202, "Teacher next-step queue failed")
        job_id = queued.json()["job_id"]
        result = worker.run_once()
        row["worker_result"] = result
        require(result and str(result.get("job_id")) == job_id and result.get("status") == "completed", "Teacher MemoryWorker failed")
        response = client.get("/memory/jobs/" + job_id, headers=teacher_headers)
        require(response.status_code == 200, "Teacher draft job preview failed")
        version_id = result["result"]["memory_version_id"]
        version = next(item for item in preview()["versions"] if str(item.get("id") or item.get("version_id")) == str(version_id))
        row.update(version=version, cache=cache_state())
        require(version["state"]["method"] == "personal_memory", "Wrong draft memory method")
        require(version["state"]["model_identities"]["chat"]["name"].startswith("gemma4:12b"), "Wrong generation model")
        report["model_identities"] = version["state"]["model_identities"]
        return version_id, version

    def approve(row, version_id, full=False):
        snapshot = preview()
        payload = {"expected_revision": snapshot["revision"]}
        if full:
            payload.update(reviewed_full_archive=True, allow_full_archive=True,
                           archive_snapshot_fingerprint=snapshot["personal_graph_snapshot_fingerprint"])
        response = client.post(f"/learners/{learner}/memory/versions/{version_id}/confirm", headers=teacher_headers, json=payload)
        row.update(request=payload, status_code=response.status_code, response=response.json())
        require(response.status_code == 200, "Teacher approval failed")
        granted = response.json()["version"]
        require(granted["context"]["source_access"]["mode"] == ("full_archive" if full else "selected_context"), "Unexpected approval mode")
        return granted

    def chat(row, key, question, *, expected_status=200):
        before = len(runtime.captured)
        payload = {"learning_session_id": active_session, "client_message_id": key,
                   "messages": [{"role": "user", "content": question}], "metadata": {"memory_method": "personal_memory"}}
        rag.observed = None
        response = client.post("/v1/chat/completions", headers=student_headers, json=payload)
        row.update(request=payload, status_code=response.status_code, response=response.json(), observed=rag.observed,
                   cache=cache_state(), model_call_count=len(runtime.captured) - before)
        require(response.status_code == expected_status, "Unexpected student chat HTTP status")
        row["embedding_input_count"] = sum(len(call["request"].get("input", [])) for call in runtime.captured[before:] if call["path"] == "/api/embed")
        row["source_embedding_input_count"] = sum(text.startswith(("student:", "teacher:"))
            for call in runtime.captured[before:] if call["path"] == "/api/embed"
            for text in call["request"].get("input", []))
        if expected_status == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            require(isinstance(answer, str) and answer.strip(), "The student answer is empty")
            context = (rag.observed or {}).get("memory_context") or {}
            row.update(answer=answer, exact_evidence_ids=context.get("evidence_ids"),
                       exact_summary_session_ids=(context.get("session_summary_context") or {}).get("included_session_ids"))
            require(fixture["other_learner_source_id"] not in json.dumps(context), "Another learner's source reached the prompt")
            require(all(item.get("role") != "assistant" for item in context.get("sources", [])), "An assistant turn became evidence")
            require(len(context.get("text", "").encode("utf-8")) <= 12000, "The complete memory context exceeded 16k UTF8 budget")
            report["quality"]["answers"].append({"stage": row["stage"], "question": question, "answer": answer,
                                                    "manual_assessment": None})
        return response

    save()
    try:
        with stage("persisted_summaries_repeat_and_restart") as row:
            snapshot = preview()
            summaries = snapshot["session_summaries"]
            one, seven = summaries["last_session"], summaries["last_seven_sessions"]
            ids = [item["learning_session_id"] for item in seven["sessions"]]
            expected = [item["session_id"] for item in reversed(fixture["sessions"][1:])]
            require(one["count"] == 1 and seven["count"] == 7 and ids == expected, "The completed-session windows differ from the fixture")
            require(fixture["sessions"][0]["session_id"] not in ids, "The eighth-oldest session entered the seven-session window")
            require(one["sessions"][0]["completed_at"][:10] == seven["sessions"][1]["completed_at"][:10], "Same-day sessions were collapsed")
            require(summaries == fixture["summary_baseline"], "Persisted summaries changed after process restart")
            require(preview()["session_summaries"] == summaries, "Repeated read created different summaries")
            restarted = session_summary_views(Store(settings.database_url), learner)
            require(restarted == summaries, "A new Store did not restore persisted summary versions")
            for item in seven["sessions"]:
                require(item["completed_at"] and item["version_id"] and item["fingerprint"], "Missing summary date/version provenance")
                for fact in item["facts"]:
                    require(fact["source_id"] in next(seed["source_ids"] for seed in fixture["sessions"] if seed["session_id"] == item["learning_session_id"]), "Summary source provenance mismatch")
                    require(fact["role"] in {"student", "teacher"} and fact["observed_on"], "Summary author/date provenance missing")
            forbidden = client.get(f"/learners/{learner}/memory", headers=other_headers)
            require(forbidden.status_code == 403, "Another learner accessed the teacher summary view")
            row.update(summaries=summaries, other_learner_http_status=forbidden.status_code)
        with stage("local_models_ready") as row:
            row["models"] = runtime.check_models()
            row["manifest"] = runtime.manifest()
            row["neo4j_disabled_configuration"] = not settings.experimental_personal_graph_enabled and not Path(settings.personal_graph_password_file).exists()
            require(row["neo4j_disabled_configuration"], "Neo4j-dependent configuration unexpectedly enabled")
            report["code_hashes_at_start"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("local_app/runtime.py", "local_app/rag.py", "local_app/api.py", "local_app/personal_memory.py",
                             "local_app/memory_service.py", "local_app/session_summaries.py")}
        with stage("teacher_draft_first") as row:
            first_id, first = teacher_step(row)
        with stage("teacher_selected_approval") as row:
            selected = approve(row, first_id)
            approved_before_active = preview()
        with stage("first_active_turn_with_selected_memory") as row:
            started = client.post(f"/learners/{learner}/learning-sessions", headers=student_headers,
                                  json={"topic_id": TOPIC, "client_session_id": "live-active-" + uuid4().hex})
            require(started.status_code == 201, "Student active session could not start")
            active_session = started.json()["learning_session_id"]
            row["learning_session_id"] = active_session
            chat(row, "selected-1", "Какой вопрос ты задал перед моим ответом про три четверти? Самостоятельность уже проверена?")
            context = row["observed"]["memory_context"]
            require(set(context["evidence_ids"]) <= set(selected["context"]["source_access"]["source_ids"]), "Selected approval expanded its source grant")
            after = preview()
            require(after["personal_graph_snapshot_fingerprint"] == approved_before_active["personal_graph_snapshot_fingerprint"], "The first active turn invalidated the source grant")
            approved_after = next(item for item in after["versions"]
                                  if str(item.get("id") or item.get("version_id")) == str(first_id))
            require(approved_after["approved"] and not approved_after["stale"], "The active turn made its reviewed memory stale")
            row["source_revision_diagnostics"] = {"before": approved_before_active["source_revision"],
                                                   "after": after["source_revision"],
                                                   "meaning": "Global revision may grow; eligible fingerprint and approval stay current"}
            row["active_turn_preserved_grant"] = True
        with stage("repeat_recall_cache") as row:
            chat(row, "selected-repeat", "Какой вопрос ты задал перед моим ответом про три четверти? Самостоятельность уже проверена?")
            require(row["source_embedding_input_count"] == 0, "Cached source vectors were embedded again")
            row["source_cache_reuse_verified_by_embedding_inputs"] = True
        with stage("teacher_refresh_draft") as row:
            refreshed_id, refreshed = teacher_step(row)
            require(refreshed["state"]["archive_snapshot_fingerprint"] == first["state"]["archive_snapshot_fingerprint"], "Active chat changed the teacher source snapshot")
        with stage("teacher_full_archive_approval") as row:
            full = approve(row, refreshed_id, full=True)
        with stage("old_event_full_archive_recall") as row:
            chat(row, "old-event", "Какой корабль я нарисовала и как его назвала?")
            require(fixture["old_source_id"] in row["exact_evidence_ids"], "Old event was not retrieved under full archive approval")
        with stage("unknown_event_recall") as row:
            chat(row, "unknown-event", "Где я была в воскресенье? Есть ли это в нашей записи?")
        with stage("teacher_correction") as row:
            snapshot = preview()
            payload = {"text": "В исходной записи имя корабля указано ошибочно. Правильное имя синего корабля – Искра.",
                       "expected_revision": snapshot["revision"], "expected_source_revision": snapshot["source_revision"],
                       "client_correction_id": "live-correction-" + uuid4().hex}
            response = client.post(f"/learners/{learner}/memory/sources/{fixture['old_source_id']}/corrections", headers=teacher_headers, json=payload)
            row.update(request=payload, status_code=response.status_code, response=response.json())
            require(response.status_code == 202, "Teacher correction failed")
            correction_id = response.json()["source"]["source_id"]
            store.cancel_memory_job(response.json()["job_id"], "Synthetic pilot requests the source-memory refresh explicitly")
        with stage("correction_invalidates_old_grant") as row:
            chat(row, "stale-grant", "Как сейчас называется мой корабль?", expected_status=409)
            require(row["model_call_count"] == 0, "A stale source grant reached Ollama")
        with stage("teacher_corrected_draft") as row:
            corrected_id, corrected = teacher_step(row)
            require(corrected["state"]["archive_snapshot_fingerprint"] != first["state"]["archive_snapshot_fingerprint"], "Correction did not change the approved snapshot")
        with stage("teacher_corrected_full_archive_approval") as row:
            approve(row, corrected_id, full=True)
        with stage("corrected_fact_recall") as row:
            chat(row, "corrected-event", "Как называется мой синий корабль с учётом исправления записи?")
            require({fixture["old_source_id"], correction_id} <= set(row["exact_evidence_ids"]), "The correction and original were detached")
        require(report["neo4j_factory_calls"] == 0, "The source-memory path called the unavailable Neo4j provider")
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT count(*) AS n FROM teachcopilot.personal_graph_indexes")
            require(cursor.fetchone()["n"] == 0, "The source-memory path created a graph index")
        report["technical_ok"] = True
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        if isinstance(exc, PilotFailure):
            report["error"] = str(exc)
    finally:
        save()
        client.close()
    return report["technical_ok"]


def pilot_app():
    """Serve this isolated fixture for browser QA without exposing its DSN."""
    directory = Path(os.environ["PERSONAL_MEMORY_PILOT_DIR"]).resolve()
    _, settings, store = isolated_store(directory)
    runtime = ObservedRuntime(settings)
    rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    return create_app(settings=settings, store=store, runtime=runtime, rag_service=rag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--code-frozen", action="store_true", help="Explicit live-generation gate after the recipe is frozen")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    directory = args.work_dir.resolve()
    if args.prepare:
        prepare(directory)
    else:
        require(args.code_frozen, "Wait for the parent agent's fingerprint code frozen signal")
        if not run(directory):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
