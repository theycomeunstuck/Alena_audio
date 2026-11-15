"""Exercise the opt-in app bridge on a new synthetic learner in an isolated test DB."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from psycopg2.extensions import make_dsn, parse_dsn
from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.config import Settings
from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
from local_app.memory_service import MemoryWorker
from local_app.personal_graph_memory import PersonalGraphProvider, approved_graph_version
from local_app.runtime import OllamaRuntime
from local_app.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-name-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dbname = args.db_name_file.read_text(encoding="ascii").strip()
    if not dbname.startswith("nirs_personal_graph_it_") or not dbname.replace("_", "").isalnum():
        raise ValueError("This smoke only accepts the isolated synthetic database")
    settings = Settings.from_env()
    # Audit F-049: only a loopback PostgreSQL endpoint is allowed (same guard
    # as smoke_personal_graph_pilot.py / smoke_personal_memory_pilot.py).
    params = parse_dsn(settings.database_url)
    hosts = (params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR"))
    if not (any(hosts) and all(not host or host in {"localhost", "127.0.0.1", "::1"} for host in hosts)):
        raise SystemExit("Refusing to run: only a loopback PostgreSQL endpoint is allowed")
    dsn = make_dsn(settings.database_url, dbname=dbname)
    nonce = uuid4().hex[:12]
    journal_path = args.output.parent / ("app-smoke-journal-" + nonce + ".sqlite3")
    settings = replace(settings, database_url=dsn, generation_scheduler_dsn=dsn, context_tokens=16384,
                       experimental_personal_graph_enabled=True, memory_response_journal_path=str(journal_path))
    store = Store(dsn)
    learner = "personal-graph-smoke-" + nonce
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,%s,4,%s)",
                    (learner, "Синтетический ученик", Json({"learner_id": learner, "grade": 4,
                     "rag_context": {"current_goal": {"topic_id": "math.fractions"}}})))
    note = store.add_memory_source(learner, "math.fractions",
        "В примере с дробями ребёнок сначала ответил три восьмых, после вопроса о равных частях исправил ответ.",
        "teacher:test", approved=True)
    store.add_memory_source(learner, "math.geometry", "В другой теме ребёнок попросил нарисовать угол.",
                            "teacher:test", approved=True)
    session = store.create_learning_session(learner, "student:test", "math.fractions", client_session_id=nonce)
    sid = session["id"]
    store.reserve_chat_message(sid, learner, "msg-1", "Хз.", "student:test")
    store.complete_chat_message(sid, "msg-1", {"choices": [{"message": {"role": "assistant",
        "content": "Какой размер у одинаковых частей?"}}]})
    store.reserve_chat_message(sid, learner, "msg-2", "Четверти, тогда три четверти.", "student:test")
    store.complete_chat_message(sid, "msg-2", {"choices": [{"message": {"role": "assistant",
        "content": "Спасибо, сравним другой пример позже."}}]})
    store.complete_learning_session(sid, learner, "student:test")
    queued = store.enqueue_memory_job(learner, "math.fractions", "next_step", {}, "teacher:test", method="personal_graph")
    runtime = OllamaRuntime(settings)
    journal = MemoryResponseJournal(journal_path, database_identity=database_fingerprint(dsn))
    worker = MemoryWorker(store, runtime, response_journal=journal)
    report = {"database": dbname, "learner": learner, "job_id": queued["job_id"], "scientific_result": False}
    try:
        result = worker.run_once()
        report["job_status"] = result.get("status")
        report["job_error"] = result.get("error")
        report["model_calls"] = len(worker.calls)
        if result.get("status") != "completed":
            raise RuntimeError("personal graph teacher job did not complete")
        report["teacher_result"] = result.get("result")
        version_id = result["result"]["memory_version_id"]
        version = store.approve_memory_version(learner, version_id, "teacher:test", expected_revision=1)
        report["teacher_approved"] = version["approved"]
        topic_view = store.get_learner_memory(learner, "math.fractions", include_personal_graph_snapshot=True)
        report["topic_view_graph_fresh"] = any(v.get("version_id") == version_id and not v["stale"]
                                                for v in topic_view["versions"])
        active = store.create_learning_session(learner, "student:test", "math.fractions",
                                               client_session_id="active-" + nonce)
        store.reserve_chat_message(active["id"], learner, "msg-active", "Вспомним дроби?", "student:test")
        snapshot = store.get_learner_memory(learner, None, include_personal_graph_snapshot=True)
        graph_snapshot = {**snapshot, "sources": snapshot["personal_graph_sources"]}
        provider = PersonalGraphProvider(store, runtime)
        context = provider.chat_context(snapshot, learner, "math.fractions", "Какой вопрос помог ребёнку?")
        report["chat_context"] = context
        report["chat_sources_from_approved_version"] = set(context["evidence_ids"]) <= set(version["context"]["evidence_ids"])
        report["assistant_not_evidence"] = all(source["role"] != "assistant" for source in context["sources"])
        report["active_version_fresh"] = approved_graph_version(graph_snapshot, learner, "math.fractions") is not None
        from local_app.memory_ported.personal_memory_lab.archive import Archive
        archive = Archive(dsn)
        report["other_child_has_no_sources"] = not archive.lessons(version["state"]["archive_run_id"], "other-child")
        correction = store.correct_memory_source(learner, note["source_id"], "Уточнение: речь шла о знаменателе.",
            "teacher:test", expected_revision=1, expected_source_revision=snapshot["source_revision"],
            client_correction_id="correction-" + nonce)
        changed = store.get_learner_memory(learner, None, include_personal_graph_snapshot=True)
        changed_graph_snapshot = {**changed, "sources": changed["personal_graph_sources"]}
        report["correction_id"] = correction["source"]["source_id"]
        report["correction_invalidates_approval"] = approved_graph_version(changed_graph_snapshot, learner, "math.fractions") is None
        withheld = provider.chat_context(changed, learner, "math.fractions", "Что было исправлено?")
        report["stale_context_withheld"] = not withheld["text"] and withheld["version_id"] is None
        checks = ("teacher_approved", "topic_view_graph_fresh", "chat_sources_from_approved_version", "assistant_not_evidence",
                  "active_version_fresh", "other_child_has_no_sources", "correction_invalidates_approval",
                  "stale_context_withheld")
        report["ok"] = all(report.get(key) for key in checks)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)[:250]
        report["ok"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("database", "job_status", "model_calls", "ok", "error_type", "error")},
                     ensure_ascii=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
