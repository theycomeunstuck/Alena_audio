"""Real API, PostgreSQL, Neo4j and local Gemma with synthetic accounts and data."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
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
        self.observed = kwargs.get("memory_context")
        return super().answer(messages, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-name-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dbname = args.db_name_file.read_text(encoding="ascii").strip()
    if not dbname.startswith("nirs_personal_graph_it_") or not dbname.replace("_", "").isalnum():
        raise ValueError("Only an isolated synthetic database is allowed")
    nonce = uuid4().hex[:12]
    fixture = args.output.parent / ("api-fixture-" + nonce)
    (fixture / "learner-data/learners").mkdir(parents=True)
    shutil.copytree(ROOT.parent / "learner-data/catalog", fixture / "learner-data/catalog")
    card = json.loads((ROOT.parent / "learner-data/demo/learners/demo-nika-04.json").read_text(encoding="utf-8-sig"))
    learner = "graph-api-" + nonce
    card.update(learner_id=learner, pseudonym="Синтетический ученик")
    (fixture / "learner-data/learners" / (learner + ".json")).write_text(
        json.dumps(card, ensure_ascii=False), encoding="utf-8")
    settings = Settings.from_env()
    # Audit F-049: only a loopback PostgreSQL endpoint is allowed (same guard
    # as smoke_personal_graph_pilot.py / smoke_personal_memory_pilot.py).
    params = parse_dsn(settings.database_url)
    hosts = (params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR"))
    if not (any(hosts) and all(not host or host in {"localhost", "127.0.0.1", "::1"} for host in hosts)):
        raise SystemExit("Refusing to run: only a loopback PostgreSQL endpoint is allowed")
    dsn = make_dsn(settings.database_url, dbname=dbname)
    settings = replace(settings, database_url=dsn, generation_scheduler_dsn=dsn, context_tokens=16384,
        experimental_personal_graph_enabled=True, request_timeout=300,
        memory_response_journal_path=str(args.output.parent / ("api-memory-" + nonce + ".sqlite3")),
        chat_response_journal_path=str(args.output.parent / ("api-chat-" + nonce + ".sqlite3")))
    store = Store(dsn)
    runtime = OllamaRuntime(settings)
    rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    journal = MemoryResponseJournal(settings.memory_response_journal_path, database_identity=database_fingerprint(dsn))
    worker = MemoryWorker(store, runtime, response_journal=journal)
    report = {"database": dbname, "learner": learner, "scientific_result": False, "uses_real_rag": True}
    try:
        store.import_repository(fixture)
        topic = "math.g4.fractions.numerator_denominator"
        password = "Synthetic-only-" + nonce
        teacher = store.create_account("teacher-" + nonce + "@synthetic.test", "Synthetic teacher",
                                       hash_password(password), "teacher")
        store.assign_learner(teacher["id"], learner)
        student = store.create_student_account("student-" + nonce + "@synthetic.test", "Synthetic student",
                                              hash_password(password), learner, teacher["id"])
        note = store.add_memory_source(learner, topic,
            "Ребёнок назвал дробь три восьмых; после вопроса о равных частях исправил на три четверти.",
            teacher["id"], approved=True)
        first = store.create_learning_session(learner, student["id"], topic, client_session_id="first-" + nonce)
        store.reserve_chat_message(first["id"], learner, "first-1", "Хз.", student["id"])
        store.complete_chat_message(first["id"], "first-1", {"choices": [{"message": {
            "role": "assistant", "content": "Сколько равных частей у целого?"}}]})
        store.reserve_chat_message(first["id"], learner, "first-2", "Четыре, значит три четверти.", student["id"])
        store.complete_chat_message(first["id"], "first-2", {"choices": [{"message": {
            "role": "assistant", "content": "Хорошо, продолжим в следующий раз."}}]})
        store.complete_learning_session(first["id"], learner, student["id"])
        client = TestClient(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag))
        def auth(account):
            result = client.post("/auth/login", json={"email": account["email"], "password": password})
            if result.status_code != 200:
                raise RuntimeError("synthetic login failed: " + str(result.status_code))
            return {"Authorization": "Bearer " + result.json()["access_token"]}
        th, sh = auth(teacher), auth(student)
        result = client.post(f"/learners/{learner}/next-step", headers=th,
                             json={"topic_id": topic, "method": "personal_graph"})
        report["queue_status"] = result.status_code
        if result.status_code != 202:
            raise RuntimeError("teacher queue rejected: " + result.text[:300])
        job_id = result.json()["job_id"]
        job = worker.run_once()
        report["teacher_job_status"] = job.get("status") if job else None
        if not job or job.get("job_id") != job_id or job.get("status") != "completed":
            raise RuntimeError("teacher worker failed: " + json.dumps(job, ensure_ascii=True, default=str)[:600])
        version_id = job["result"]["memory_version_id"]
        report["teacher_result"] = {k: job["result"].get(k) for k in
                                    ("action", "reason", "unknowns", "evidence_turn_ids", "review_required")}
        report["teacher_source_review"] = job["result"].get("evidence_review")
        report["teacher_sources"] = job["result"].get("sources")
        snap = client.get(f"/learners/{learner}/memory", headers=th).json()
        from local_app.personal_graph_contract import PERSONAL_GRAPH_CONTEXT_PROTOCOL
        stored = next(item for item in store.get_learner_memory(learner, None)["versions"]
                      if str(item.get("id") or item.get("version_id")) == str(version_id))
        report["current_context_protocol"] = (
            (stored.get("state") or {}).get("source_context_protocol") == PERSONAL_GRAPH_CONTEXT_PROTOCOL)
        selected = json.loads((stored.get("context") or {}).get("text") or "{}")
        report["selected_context_has_topic"] = any(
            turn.get("role") in {"teacher", "student"} and turn.get("topic_id") == topic
            for block in selected.get("lessons", []) for turn in block.get("turns", []))
        confirm_url = f"/learners/{learner}/memory/versions/{version_id}/confirm"
        report["student_cannot_confirm"] = client.post(confirm_url, headers=sh,
            json={"expected_revision": snap["revision"]}).status_code == 403
        confirmed = client.post(confirm_url, headers=th, json={"expected_revision": snap["revision"]})
        report["teacher_confirmed"] = confirmed.status_code == 200
        session = client.post(f"/learners/{learner}/learning-sessions", headers=sh,
                              json={"topic_id": topic, "client_session_id": "second-" + nonce})
        if session.status_code != 201:
            raise RuntimeError("new learning session failed: " + session.text[:300])
        payload = {"learning_session_id": session.json()["learning_session_id"], "client_message_id": "chat-1",
            "messages": [{"role": "user", "content": "Напомни, какой вопрос помог мне исправить дробь?"}],
            "metadata": {"memory_method": "personal_graph"}}
        answer = client.post("/v1/chat/completions", headers=sh, json=payload)
        report["chat_status"] = answer.status_code
        if answer.status_code != 200:
            raise RuntimeError("real chat failed: " + answer.text[:500])
        body = answer.json()
        report["chat_answer"] = body.get("choices", [{}])[0].get("message", {}).get("content")
        report["chat_version"] = body.get("memory_version_id")
        report["chat_method"] = body.get("memory_method")
        observed = rag.observed or {}
        report["chat_memory_context"] = observed
        approved_ids = set(confirmed.json()["version"]["context"]["evidence_ids"])
        report["real_rag_received_approved_memory"] = bool(observed.get("text")) and \
            observed.get("version_id") == version_id and set(observed.get("evidence_ids", [])) <= approved_ids
        report["chat_replay_identical"] = client.post("/v1/chat/completions", headers=sh, json=payload).json() == body
        payload["metadata"] = {}
        report["different_method_replay_rejected"] = client.post("/v1/chat/completions", headers=sh, json=payload).status_code == 409
        report["student_memory_view_forbidden"] = client.get(f"/learners/{learner}/memory", headers=sh).status_code == 403
        latest = store.get_learner_memory(learner, None)
        correction = client.post(f"/learners/{learner}/memory/sources/{note['source_id']}/corrections", headers=th,
            json={"text": "Уточнение: вопрос был о количестве равных частей.", "expected_revision": latest["revision"],
                  "expected_source_revision": latest["source_revision"], "client_correction_id": "correct-" + nonce})
        report["correction_status"] = correction.status_code
        payload["metadata"] = {"memory_method": "personal_graph"}
        payload["client_message_id"] = "chat-after-correction"
        report["chat_rejects_stale_memory"] = client.post("/v1/chat/completions", headers=sh, json=payload).status_code == 409
        checks = ("student_cannot_confirm", "teacher_confirmed", "real_rag_received_approved_memory",
                  "chat_replay_identical", "different_method_replay_rejected", "student_memory_view_forbidden",
                  "chat_rejects_stale_memory", "current_context_protocol", "selected_context_has_topic")
        report["ok"] = all(report.get(k) for k in checks) and bool(report["chat_answer"]) \
            and report["chat_version"] == version_id and report["chat_method"] == "personal_graph" \
            and report["correction_status"] == 202
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:800])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("ok", "queue_status", "teacher_job_status", "chat_status",
                                                "error_type", "error")}, ensure_ascii=True))
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
