"""Create and run a local, synthetic-only TeachCopilot personal-graph demo."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.auth import hash_password
from local_app.config import Settings
from local_app.store import Store

PREFIX = "nirs_personal_graph_demo_"
KIND = "personal-graph-synthetic-demo-v1"


def local_dsn(database=None):
    settings = Settings.from_env()
    params = parse_dsn(settings.database_url)
    hosts = (params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR"))
    if not any(hosts) or any(host and host not in {"127.0.0.1", "localhost", "::1"} for host in hosts):
        raise ValueError("The synthetic demo requires a loopback PostgreSQL server")
    return make_dsn(settings.database_url, dbname=database) if database else settings.database_url


def setup(state_path):
    if state_path.exists():
        raise FileExistsError(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid4().hex[:12]
    dbname, learner = PREFIX + nonce, "graph-demo-" + nonce
    fixture = state_path.parent / ("demo-fixture-" + nonce)
    state = {"kind": KIND, "status": "creating", "database": dbname, "learner": learner,
             "warning": "Synthetic demonstration only"}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    admin = psycopg2.connect(local_dsn(), connect_timeout=10)
    try:
        admin.autocommit = True
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
    finally:
        admin.close()
    store = Store(local_dsn(dbname))
    store.migrate()
    (fixture / "learner-data/learners").mkdir(parents=True)
    shutil.copytree(ROOT.parent / "learner-data/catalog", fixture / "learner-data/catalog")
    card = json.loads((ROOT.parent / "learner-data/demo/learners/demo-nika-04.json").read_text(encoding="utf-8-sig"))
    card.update(learner_id=learner, pseudonym="Синтетический ученик")
    (fixture / "learner-data/learners" / (learner + ".json")).write_text(
        json.dumps(card, ensure_ascii=False), encoding="utf-8")
    store.import_repository(fixture)
    password = secrets.token_urlsafe(18)
    teacher_email, student_email = "teacher-" + nonce + "@synthetic.test", "student-" + nonce + "@synthetic.test"
    teacher = store.create_account(teacher_email, "Синтетический преподаватель", hash_password(password), "teacher")
    store.assign_learner(teacher["id"], learner)
    student = store.create_student_account(student_email, "Синтетический ученик", hash_password(password),
                                           learner, teacher["id"])
    topic = "math.g4.fractions.numerator_denominator"
    store.add_memory_source(learner, topic,
        "В примере с дробями ребёнок сначала ответил три восьмых, после вопроса о равных частях исправил на три четверти.",
        teacher["id"], approved=True)
    session = store.create_learning_session(learner, student["id"], topic, client_session_id="synthetic-first")
    store.reserve_chat_message(session["id"], learner, "demo-1", "Хз.", student["id"])
    store.complete_chat_message(session["id"], "demo-1", {"choices": [{"message": {
        "role": "assistant", "content": "Сколько равных частей у целого?"}}]})
    store.reserve_chat_message(session["id"], learner, "demo-2", "Четыре, значит три четверти.", student["id"])
    store.complete_chat_message(session["id"], "demo-2", {"choices": [{"message": {
        "role": "assistant", "content": "Хорошо, продолжим в следующий раз."}}]})
    store.complete_learning_session(session["id"], learner, student["id"])
    state.update(status="ready", topic=topic, teacher_email=teacher_email, student_email=student_email,
                 synthetic_password=password)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Synthetic demo prepared; local login details:", state_path.resolve())


def load_settings(state_path):
    state = json.loads(state_path.read_text(encoding="utf-8"))
    database = state.get("database", "")
    if (state.get("kind") != KIND or state.get("status") != "ready" or not database.startswith(PREFIX)
            or not database.replace("_", "").isalnum()):
        raise ValueError("The state file is not a ready synthetic demo")
    dsn = local_dsn(database)
    settings = Settings.from_env()
    return replace(settings, database_url=dsn, generation_scheduler_dsn=dsn, context_tokens=16384,
        experimental_personal_graph_enabled=True, request_timeout=300,
        memory_response_journal_path=str(state_path.with_suffix(".memory.sqlite3")),
        chat_response_journal_path=str(state_path.with_suffix(".chat.sqlite3")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "serve", "worker"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("invalid local port")
    if args.action == "setup":
        setup(args.state)
        return
    settings = load_settings(args.state)
    from local_app.runtime import OllamaRuntime
    store, runtime = Store(settings.database_url), OllamaRuntime(settings)
    if args.action == "serve":
        import uvicorn
        from local_app.answer_adapter import AnswerAdapter
        from local_app.rag import RagService
        # Keep the module-level default app local too, even if .local.env has
        # an external-answer configuration.
        os.environ["TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG"] = ""
        from local_app.api import create_app
        rag = RagService(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
        uvicorn.run(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag),
                    host="127.0.0.1", port=args.port, log_level="warning")
    else:
        from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
        from local_app.memory_service import MemoryWorker
        journal = MemoryResponseJournal(settings.memory_response_journal_path,
                                         database_identity=database_fingerprint(settings.database_url))
        worker = MemoryWorker(store, runtime, response_journal=journal)
        print("Synthetic demo memory worker started.", flush=True)
        while True:
            result = worker.run_once()
            if result:
                print(result["job_id"], result["status"], flush=True)
            else:
                time.sleep(1)


if __name__ == "__main__":
    main()
