"""One bounded live proof of automatic Alena conversation memory.

This creates a new synthetic database after --go-for-live, runs four ordinary
student chats and one twelve-chunk BGE catalog build, and preserves exact local
traces. No production rows or shared service configuration are changed.
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
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn
from psycopg2.extras import Json
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Capture the exact implementation before importing any application module.
if __name__ == "__main__" and "--go-for-live" in sys.argv:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--work-dir", type=Path, required=True)
    location, _ = preliminary.parse_known_args()
    proof_directory = location.work_dir.resolve()
    copied_code = proof_directory / "frozen-code-v1"
    copied_code.mkdir(parents=True, exist_ok=False)
    code_paths = [*sorted((ROOT / "local_app").glob("*.py")),
                  *sorted((ROOT / "db/migrations").glob("*.sql")), Path(__file__),
                  ROOT / "scripts/smoke_personal_memory_pilot.py"]
    captured_hashes = {}
    for code_path in code_paths:
        relative = code_path.relative_to(ROOT)
        data = code_path.read_bytes()
        target = copied_code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        captured_hashes[str(relative).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
    (proof_directory / "code-hashes-before-import.json").write_text(
        json.dumps(captured_hashes, indent=2) + "\n", encoding="utf-8")

from scripts.smoke_personal_memory_pilot import ObservedRuntime, require, safe_settings, write_json, PilotFailure
from local_app.answer_adapter import AnswerAdapter
from local_app.api import create_app
from local_app.auth import hash_password
from local_app.conversation_retrieval import chunk_conversation_messages, read_own_messages
from local_app.dialogue_catalog import rebuild_dialogue_catalog
from local_app.hierarchical_memory import build_conversation_memory
from local_app.rag import RagService
from local_app.store import Store


TOPICS = ("math.g4.fractions.numerator_denominator", "math.g3.geometry.perimeter")
DB_PATTERN = re.compile(r"nirs_alena_it_[0-9a-f]{12}")
OLD_FACT = "В сентябре я нарисовала синий корабль и назвала его Северная Искра."
OLD_QUERY = "Как назывался синий корабль, который я нарисовала раньше?"


def settings_for(directory, dbname=None):
    return replace(safe_settings(directory, dbname), context_tokens=8192, request_timeout=300)


def load_store(directory):
    dbname = (directory / "db-name.txt").read_text(encoding="ascii").strip()
    require(DB_PATTERN.fullmatch(dbname), "Only the dedicated Alena synthetic database is allowed")
    settings = settings_for(directory, dbname)
    return settings, Store(settings.database_url)


def seed_pair(store, learner, actor, topic, when, user_text, assistant_text, label):
    """Insert immutable historical source instants directly in the test fixture."""
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("""INSERT INTO teachcopilot.learning_sessions
            (learner_id,actor_id,topic_id,client_session_id,status,created_at,completed_at,event_timezone)
            VALUES(%s,%s,%s,%s,'completed',%s,%s,'Europe/Moscow') RETURNING id""",
            (learner, actor, topic, label, when - timedelta(minutes=1), when + timedelta(minutes=1)))
        session_id = cursor.fetchone()["id"]
        sources = []
        for index, (role, text) in enumerate((("student", user_text), ("assistant", assistant_text))):
            cursor.execute("""INSERT INTO teachcopilot.memory_sources
                (learner_id,topic_id,learning_session_id,actor_id,author_role,content,created_at,
                 observed_on,source_key,event_timezone,activity_timezone)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'Europe/Moscow','Asia/Tomsk') RETURNING id""",
                (learner, topic, session_id, actor if role == "student" else "assistant", role, text,
                 when + timedelta(seconds=index), when.date().isoformat(), label + "-" + role))
            sources.append(cursor.fetchone()["id"])
        response = {"choices": [{"message": {"role": "assistant", "content": assistant_text}}]}
        cursor.execute("""INSERT INTO teachcopilot.learning_messages
            (learning_session_id,learner_id,client_message_id,source_id,assistant_source_id,
             content_hash,status,response,created_at,completed_at)
            VALUES(%s,%s,%s,%s,%s,%s,'completed',%s,%s,%s)""",
            (session_id, learner, label, sources[0], sources[1], hashlib.sha256(user_text.encode("utf-8")).hexdigest(),
             Json(response), when, when + timedelta(seconds=2)))
        cursor.execute("UPDATE teachcopilot.learners SET memory_source_revision=memory_source_revision+2 WHERE learner_id=%s", (learner,))
    return {"learning_session_id": str(session_id), "source_ids": sources, "created_at": when.isoformat(), "topic_id": topic}


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / "db-name.txt").exists(), "This immutable pilot already has a database")
    base = settings_for(directory)
    dbname = "nirs_alena_it_" + uuid4().hex[:12]
    connection = psycopg2.connect(base.database_url, connect_timeout=10)
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    finally:
        connection.close()
    (directory / "db-name.txt").write_text(dbname + "\n", encoding="ascii")
    settings, store = load_store(directory)
    applied = store.migrate()
    require(store.schema_version() >= 23, "Wait for migrations 020–023 to be frozen")
    nonce = uuid4().hex[:12]
    fixture_root = directory / "fixture"
    (fixture_root / "learner-data/learners").mkdir(parents=True)
    shutil.copytree(ROOT.parent / "learner-data/catalog", fixture_root / "learner-data/catalog")
    template = json.loads((ROOT.parent / "learner-data/demo/learners/demo-nika-04.json").read_text(encoding="utf-8-sig"))
    learners = {"fresh": "alena-fresh-" + nonce, "recall": "alena-recall-" + nonce}
    for identifier in learners.values():
        card = deepcopy(template)
        card.update(learner_id=identifier, pseudonym="Синтетический ученик")
        write_json(fixture_root / "learner-data/learners" / (identifier + ".json"), card)
    store.import_repository(fixture_root)
    password = secrets.token_urlsafe(24)
    teacher = store.create_account("teacher-" + nonce + "@synthetic.test", "Synthetic teacher", hash_password(password), "teacher")
    for identifier in learners.values():
        store.assign_learner(teacher["id"], identifier)
    accounts = {"teacher": {**teacher, "learner_id": None}}
    for label in ("fresh", "recall", "foreign"):
        learner = learners["fresh" if label == "fresh" else "recall"]
        account = store.create_student_account(label + "-" + nonce + "@synthetic.test", "Synthetic student",
                                               hash_password(password), learner, teacher["id"])
        accounts[label] = {**account, "learner_id": learner}
    write_json(directory / "PRIVATE-auth.json", {"synthetic": True, "password": password,
        "accounts": {label: {key: str(value) if key == "id" else value for key, value in account.items()
                            if key in {"id", "email", "role", "learner_id"}}
                     for label, account in accounts.items()}})
    actor = str(accounts["recall"]["id"])
    store.set_account_display_timezone(actor, "Europe/Moscow")
    instants = [datetime.fromisoformat(value) for value in (
        "2026-08-31T18:30:00+00:00", "2026-09-09T10:00:00+00:00", "2026-09-20T10:00:00+00:00",
        "2026-10-01T10:00:00+00:00", "2026-10-15T10:00:00+00:00", "2026-10-31T10:00:00+00:00",
        "2026-11-02T10:00:00+00:00", "2026-11-03T08:00:00+00:00", "2026-11-03T09:00:00+00:00",
        "2026-11-03T10:00:00+00:00", "2026-11-03T11:00:00+00:00", "2026-11-03T12:00:00+00:00")]
    pairs = []
    for index, when in enumerate(instants):
        user = OLD_FACT if index == 0 else f"На занятии {index + 1} я считала закрашенные части рисунка и просила проверить дробь."
        assistant = "Спасибо за рассказ о твоём корабле." if index == 0 else "Сколько одинаковых частей у целого?"
        pairs.append(seed_pair(store, learners["recall"], actor, TOPICS[index % 2], when, user, assistant, f"seed-{index}"))
    foreign = seed_pair(store, learners["recall"], str(accounts["foreign"]["id"]), TOPICS[0], instants[-1],
                        "ТОЛЬКО_ЧУЖОЙ_ПОЛЬЗОВАТЕЛЬ: мой аппарат называется Янтарь.", "Чужая реплика.", "foreign")
    private = store.add_memory_source(learners["recall"], TOPICS[0], "ПРИВАТНАЯ_ЗАМЕТКА_ПРЕПОДАВАТЕЛЯ", teacher["id"], approved=True)
    own = read_own_messages(store, learners["recall"], actor)
    chunks = chunk_conversation_messages(own, learners["recall"], actor)
    require(len(chunks) == 12, "Expected exactly twelve short archive chunks")
    hierarchy = build_conversation_memory(store, learners["recall"], actor)
    old_source = pairs[0]["source_ids"][0]
    require(old_source not in {row["source_id"] for row in hierarchy["recent_messages"]}, "Old fact remains in the recent window")
    require("Северная Искра" not in hierarchy["system_text"], "Old fact remains in the seven-active-day summary")
    manifest = {"synthetic": True, "database": dbname, "schema_version": store.schema_version(), "migrations_applied": applied,
                "learners": learners, "actor_ids": {label: str(row["id"]) for label, row in accounts.items()},
                "pairs": pairs, "old_source_id": old_source, "private_source_id": private["source_id"],
                "foreign_source_ids": foreign["source_ids"], "seed_chunk_count": len(chunks),
                "seed_chunk_texts": [chunk["text"] for chunk in chunks], "initial_hierarchy": hierarchy,
                "old_instant": {"utc": instants[0].isoformat(), "display_moscow": instants[0].astimezone(ZoneInfo("Europe/Moscow")).isoformat(),
                                "study_tomsk": instants[0].astimezone(ZoneInfo("Asia/Tomsk")).isoformat()},
                "fixture_clock_note": "Historical instants are deterministic fixture data; the live server clock is unchanged"}
    write_json(directory / "fixture-manifest.json", manifest)
    return settings, store, manifest


class AlenaRag(RagService):
    observed = None
    def answer(self, messages, **kwargs):
        self.observed = {"conversation_context": deepcopy(kwargs.get("conversation_context")), "retrievals": []}
        callback = kwargs.get("conversation_retriever")
        if callable(callback):
            def observed(query):
                result = callback(query)
                self.observed["retrievals"].append(deepcopy(result))
                return result
            kwargs["conversation_retriever"] = observed
        return super().answer(messages, **kwargs)


def run(directory):
    report_path = directory / "live-report.json"
    require(not report_path.exists(), "The immutable Alena live report already exists")
    settings, store, fixture = prepare(directory)
    runtime = ObservedRuntime(settings)
    rag = AlenaRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    client = TestClient(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag))
    private = json.loads((directory / "PRIVATE-auth.json").read_text(encoding="utf-8"))
    report = {"synthetic": True, "scientific_result": False, "technical_ok": False, "quality_ok": None,
              "context_tokens": 8192, "stages": [], "answers": [], "independent_benchmark": False}
    def save(): write_json(report_path, report)
    @contextmanager
    def stage(name):
        row = {"stage": name, "status": "running"}
        report["stages"].append(row)
        save()
        started, before = time.perf_counter(), len(runtime.captured)
        try:
            yield row
            row["status"] = "passed"
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__)
            if isinstance(exc, PilotFailure): row["error"] = str(exc)
            raise
        finally:
            row.update(seconds=round(time.perf_counter() - started, 3), local_calls=runtime.captured[before:])
            save()
            print(name + ": " + row["status"], flush=True)
    def auth(label):
        account = private["accounts"][label]
        result = client.post("/auth/login", json={"email": account["email"], "password": private["password"]})
        require(result.status_code == 200, "Synthetic login failed")
        return {"Authorization": "Bearer " + result.json()["access_token"]}
    fresh_headers, recall_headers = auth("fresh"), auth("recall")
    def history(label):
        return store.get_own_conversation_messages(fixture["learners"][label], fixture["actor_ids"][label], limit=None)
    def chat(row, label, question, metadata=None):
        headers = fresh_headers if label == "fresh" else recall_headers
        before = history(label)
        rag.observed = None
        payload = {"messages": [{"role": "user", "content": question}], "metadata": metadata or {}}
        result = client.post("/v1/chat/completions", headers=headers, json=payload)
        row.update(request=payload, status_code=result.status_code, response=result.json(), observed=deepcopy(rag.observed))
        require(result.status_code == 200, "The ordinary legacy chat did not complete")
        after = history(label)
        appended = [item for item in after if item["source_id"] not in {old["source_id"] for old in before}]
        row["appended_messages"] = appended
        require([item["role"] for item in appended] == ["user", "assistant"], "Ordinary chat did not append the complete question and reply")
        require(appended[0]["text"] == question, "Stored question text changed")
        row["answer"] = result.json()["choices"][0]["message"]["content"]
        require(appended[1]["text"] == row["answer"], "Stored assistant reply differs from its HTTP response")
        model_calls = [call for call in runtime.captured if call["path"] == "/api/chat"]
        final_messages = model_calls[-1]["request"]["messages"]
        require(sum(message["role"] == "user" and message["content"] == question for message in final_messages) == 1,
                "Current question was duplicated in the model dialogue")
        require(len([message for message in final_messages if message["role"] != "system"]) <= 20, "Raw model history exceeds twenty individual messages")
        context = row["observed"]["conversation_context"]
        forbidden = {fixture["private_source_id"], *fixture["foreign_source_ids"]}
        require(not forbidden & set(context["provenance"]["source_ids"]), "Private or foreign sources entered own hierarchy")
        system = final_messages[0]["content"]
        require("ПРИВАТНАЯ_ЗАМЕТКА" not in system and "ТОЛЬКО_ЧУЖОЙ" not in system, "Private or foreign text entered the prompt")
        require(not re.search(r"\d{2}:\d{2}:\d{2}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", context["system_text"]),
                "Exact timestamps or source UUIDs leaked into summary text")
        row["raw_model_message_count"] = len(final_messages) - 1
        report["answers"].append({"stage": row["stage"], "question": question, "answer": row["answer"], "manual_assessment": None})
    save()
    try:
        with stage("local_models_and_frozen_code") as row:
            row["models"] = runtime.check_models()
            row["identities"] = {kind: runtime.model_identity(model) for kind, model in (("chat", settings.chat_model), ("embedding", settings.embedding_model))}
            report["code_hashes"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
                "local_app/runtime.py", "local_app/rag.py", "local_app/api.py", "local_app/memory_store.py",
                "local_app/hierarchical_memory.py", "local_app/checkins.py", "local_app/conversation_retrieval.py", "local_app/dialogue_catalog.py")}
        with stage("fresh_checkin_shown_and_legacy_answer") as row:
            learner = fixture["learners"]["fresh"]
            offered = client.get(f"/learners/{learner}/checkin", headers=fresh_headers)
            require(offered.status_code == 200, "Fresh check-in offer failed")
            checkin = offered.json()["checkin"]
            shown = client.post(f"/learners/{learner}/checkin/{checkin['checkin_id']}/shown", headers=fresh_headers, json={})
            require(shown.status_code == 200, "Check-in display was not archived")
            row["checkin_shown"] = shown.json()["checkin"]
            chat(row, "fresh", "Сегодня я немного устала, но хочу спокойно порисовать.", {"checkin_id": checkin["checkin_id"]})
            state = client.get(f"/learners/{learner}/checkin", headers=fresh_headers).json()["checkin"]
            require(state["status"] == "answered" and state["assistant_source_id"] and state["answered_source_id"], "Check-in answer was not linked to ordinary history")
            row["checkin_answered"] = state
            with store._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT count(*) AS n FROM teachcopilot.memory_versions WHERE learner_id=%s", (learner,))
                require(cursor.fetchone()["n"] == 0, "Fresh conversation unexpectedly required an approved memory version")
        with stage("twelve_short_chunks_bge_catalog") as row:
            label = "recall"
            row["catalog"] = rebuild_dialogue_catalog(store, runtime, fixture["learners"][label], fixture["actor_ids"][label])
            require(row["catalog"]["chunk_count"] == 12, "Bounded catalog proof changed the twelve-chunk fixture")
            row["score_caveat"] = "Cosine similarity and engineering defaults; not calibrated confidence or a quality benchmark"
        with stage("old_fact_outside_day7_and_recent20") as row:
            chat(row, "recall", OLD_QUERY)
            retrieved = row["observed"]["retrievals"]
            require(len(retrieved) == 1, "Explicit own recall did not call the full archive exactly once")
            require(fixture["old_source_id"] in retrieved[0]["source_ids"], "Old literal source was not returned from the full archive")
            row["old_name_in_model_context"] = "Северная Искра" in retrieved[0]["system_text"]
            require(row["old_name_in_model_context"], "The old name was absent from the literal model retrieval context")
            row["answer_mentions_expected_name"] = "северн" in row["answer"].lower() and "искр" in row["answer"].lower()
        with stage("unknown_fact_honesty") as row:
            chat(row, "recall", "В каком городе я была на осенних каникулах? Есть ли это в нашей записи?")
            unknown_context = row["observed"]["conversation_context"]
        with stage("ordinary_social_and_stable_older_prefix") as row:
            chat(row, "recall", "Спасибо! Сегодня хочу спокойно порисовать.")
            require(not row["observed"]["retrievals"], "Ordinary sharing unnecessarily retrieved the full archive")
            current_context = row["observed"]["conversation_context"]
            day = current_context["last_day"]["day"]
            marker = "Сеанс за " + day + "."
            before = unknown_context["system_text"].split(marker, 1)[0]
            after = current_context["system_text"].split(marker, 1)[0]
            row["older_prefix_equal"] = before == after
            require(row["older_prefix_equal"], "Frozen older-day prefix changed during the same activity day")
        report["technical_ok"] = True
        report["utc_display_study_fixture_proof"] = fixture["old_instant"]
    except BaseException as exc:
        report["error_type"] = type(exc).__name__
        if isinstance(exc, PilotFailure): report["error"] = str(exc)
    finally:
        save()
        client.close()
    return report["technical_ok"]


def pilot_app():
    directory = Path(os.environ["ALENA_MEMORY_PILOT_DIR"]).resolve()
    settings, store = load_store(directory)
    runtime = ObservedRuntime(settings)
    rag = AlenaRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    return create_app(settings=settings, store=store, runtime=runtime, rag_service=rag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--go-for-live", action="store_true")
    args = parser.parse_args()
    require(args.go_for_live, "Wait for the explicit GOFORLIVE signal before migrations, code hashes or GPU work")
    logging.disable(logging.CRITICAL)
    if not run(args.work_dir.resolve()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
