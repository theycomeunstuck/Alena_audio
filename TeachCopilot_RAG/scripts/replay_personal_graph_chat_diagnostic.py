"""Retry a synthetic chat in a new session using an existing approved graph.

This never retries the original client message or rewrites its result. The
input must be a report from the isolated selection diagnostic.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psycopg2.extensions import make_dsn
from fastapi.testclient import TestClient
from local_app.answer_adapter import AnswerAdapter
from local_app.api import create_app
from local_app.config import Settings
from local_app.personal_graph_memory import (PersonalGraphProvider, application_lesson_order,
    approved_graph_version, select_context, source_lessons)
from local_app.personal_graph_contract import personal_graph_budgets
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from local_app.store import Store
from scripts.diagnose_personal_graph_selection import (DB_PREFIX, SYNTHETIC_LEARNER_PREFIX,
    approval_boundary_report, possible_refusal_heuristic, selected_context_report,
    validate_database_name, validate_local_ollama_url, validate_local_postgres_dsn,
    validate_output_path, write_json_exclusive)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, choices=(16384, 32768))
    parser.add_argument("--version-id", help="Reuse an explicitly identified current synthetic approved version")
    parser.add_argument("--diagnostic-query-draft", action="store_true",
                        help="Create and confirm a synthetic query-specific draft for diagnostic purposes")
    args = parser.parse_args(argv)
    output = validate_output_path(args.output)
    report = {"diagnostic_only": True, "scientific_result": False, "ok": False}
    try:
        previous = json.loads(args.input.read_text(encoding="utf-8"))
        dbname = validate_database_name(previous["database"])
        if not previous.get("diagnostic_only") or previous.get("scientific_result") is not False:
            raise ValueError("input_must_be_a_synthetic_diagnostic")
        context_tokens = args.context_tokens or previous["runtime"]["context_tokens"]
        if context_tokens not in (16384, 32768):
            raise ValueError("invalid_context_window")
        base = Settings.from_env()
        validate_local_postgres_dsn(base.database_url, pg_hostaddr=os.getenv("PGHOSTADDR"))
        validate_local_ollama_url(base.ollama_url)
        if (base.chat_model != previous["runtime"]["chat_model"]
                or base.embedding_model != previous["runtime"]["embedding_model"]):
            raise ValueError("diagnostic_model_tags_changed")
        dsn = make_dsn(base.database_url, dbname=dbname)
        settings = replace(base, database_url=dsn, generation_scheduler_dsn=dsn,
            context_tokens=context_tokens, experimental_personal_graph_enabled=True, request_timeout=300,
            chat_response_journal_path=str(output.parent / ("diagnostic-replay-chat-" + uuid4().hex + ".sqlite3")))
        store, runtime = Store(dsn), OllamaRuntime(settings)
        source_version_id = str(previous["teacher_job"]["version_id"])
        version_id = args.version_id or source_version_id
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT learner_id,topic_id FROM teachcopilot.memory_versions WHERE id=%s", (version_id,))
            row = cursor.fetchone()
        if not row or not row["learner_id"].startswith(SYNTHETIC_LEARNER_PREFIX):
            raise ValueError("version_is_not_in_the_synthetic_scope")
        learner_id, topic_id = row["learner_id"], row["topic_id"]
        nonce = learner_id.removeprefix(SYNTHETIC_LEARNER_PREFIX)
        if len(nonce) != 12 or any(c not in "0123456789abcdef" for c in nonce):
            raise ValueError("synthetic_learner_id_is_invalid")
        snapshot = store.get_learner_memory(learner_id, None, include_personal_graph_snapshot=True)
        graph_snapshot = {**snapshot, "sources": snapshot["personal_graph_sources"]}
        version = approved_graph_version(graph_snapshot, learner_id, topic_id)
        if not version or str(version.get("id") or version.get("version_id")) != version_id:
            raise ValueError("synthetic_memory_is_no_longer_approved_and_current")

        class ObservedRag(RagService):
            def answer(self, messages, *positional, **kwargs):
                self.last_memory_context = kwargs.get("memory_context")
                return super().answer(messages, *positional, **kwargs)

        rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
        client = TestClient(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag))
        diagnostic_draft = None
        if args.diagnostic_query_draft:
            async def build_query_context():
                lessons, _ = source_lessons(graph_snapshot, learner_id)
                _, _, graph, _, workflow = PersonalGraphProvider(store, runtime)._objects(
                    source_order=application_lesson_order(lessons))
                try:
                    selection_limit, memory_limit = personal_graph_budgets(context_tokens)
                    found = await workflow.search(version["state"]["archive_run_id"], learner_id,
                        previous["scenario"]["query"], mode="graphiti", budget_tokens=selection_limit)
                    context = select_context(found, graph_snapshot, learner_id, topic_id, memory_byte_budget=memory_limit)
                    return found, context
                finally:
                    await graph.close()

            found, context = asyncio.run(build_query_context())
            target = previous["scenario"]["synthetic_target_source_id"]
            if target not in context["evidence_ids"]:
                raise ValueError("diagnostic_query_did_not_select_target")
            state = {**version["state"], "source_revision": snapshot["source_revision"]}
            draft = store.save_memory_version(learner_id, topic_id, state, base_revision=snapshot["revision"],
                source_revision=snapshot["source_revision"], context=context)
            teacher_auth = client.post("/auth/login", json={"email": f"teacher-{nonce}@synthetic.invalid",
                                                            "password": "Synthetic-only-" + nonce})
            if teacher_auth.status_code != 200:
                raise RuntimeError("synthetic_teacher_login_failed")
            teacher_headers = {"Authorization": "Bearer " + teacher_auth.json()["access_token"]}
            approval = client.post(f"/learners/{learner_id}/memory/versions/{draft['id']}/confirm",
                headers=teacher_headers, json={"expected_revision": snapshot["revision"]})
            if approval.status_code != 200:
                raise RuntimeError("synthetic_diagnostic_draft_approval_failed")
            version = approval.json()["version"]
            version_id = str(version.get("id") or version.get("version_id"))
            diagnostic_draft = {"query_specific": True, "created_via_store_for_synthetic_diagnostic": True,
                                "teacher_approved_via_api": True, "target_selected": True,
                                "selection_budget_bytes": personal_graph_budgets(context_tokens)[0],
                                "context_budget": context.get("selection_budget"),
                                "workflow_skipped": found.get("skipped", [])}
        approved_ids = set((version.get("context") or {}).get("evidence_ids") or [])
        auth = client.post("/auth/login", json={"email": f"student-{nonce}@synthetic.invalid",
                                                "password": "Synthetic-only-" + nonce})
        if auth.status_code != 200:
            raise RuntimeError("synthetic_student_login_failed")
        headers = {"Authorization": "Bearer " + auth.json()["access_token"]}
        session = client.post(f"/learners/{learner_id}/learning-sessions", headers=headers,
            json={"topic_id": topic_id, "client_session_id": "diagnostic-replay-" + uuid4().hex})
        if session.status_code != 201:
            raise RuntimeError("synthetic_new_session_failed")
        started = time.perf_counter()
        response = client.post("/v1/chat/completions", headers=headers, json={
            "learning_session_id": session.json()["learning_session_id"],
            "client_message_id": "diagnostic-replay-" + uuid4().hex,
            "messages": [{"role": "user", "content": previous["scenario"]["query"]}],
            "metadata": {"memory_method": "personal_graph"},
        })
        elapsed = time.perf_counter() - started
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        answer = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") if response.status_code == 200 else ""
        selected = selected_context_report(getattr(rag, "last_memory_context", None) or {})
        boundary = approval_boundary_report(previous["scenario"]["synthetic_target_source_id"],
                                            approved_ids, selected.get("evidence_ids") or [])
        report.update(database=dbname, source_report=str(args.input), context_tokens=context_tokens,
            original_version_id=source_version_id, active_version_id=version_id,
            http_status=response.status_code, elapsed_seconds=elapsed,
            answer=answer or body.get("detail"), usage=body.get("usage"),
            possible_refusal_heuristic=possible_refusal_heuristic(answer),
            memory_context=selected, approval_boundary=boundary, diagnostic_draft=diagnostic_draft,
            ok=response.status_code == 200 and bool(answer) and boundary["chat_ids_within_approved_set"])
        client.close()
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error_code="synthetic_replay_failed")
    write_json_exclusive(output, report)
    print(json.dumps({key: report.get(key) for key in ("ok", "http_status", "error_type")}, ensure_ascii=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
