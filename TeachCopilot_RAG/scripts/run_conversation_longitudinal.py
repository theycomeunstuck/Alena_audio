"""Finite synthetic sessions through the API, PostgreSQL and real local Ollama.

Only public curriculum/index tables are copied from the local working database.
Accounts, cards, dialogues and simulated reviews exist in a disposable database.
Expected preferences never enter a generation prompt. No automatic retries.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def normal(value, aliases):
    value = " ".join(value.casefold().split())
    for canonical, variants in aliases.items():
        if value in {canonical.casefold(), *(item.casefold() for item in variants)}:
            return canonical.casefold()
    return value


def run(args):
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from fastapi.testclient import TestClient
    from local_app.answer_adapter import AnswerAdapter
    from local_app.api import create_app
    from local_app.auth import hash_password
    from local_app.config import Settings
    from local_app.memory_service import MemoryWorker
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from scripts.verify_memory import require_local_dsn

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture_path = ROOT / "tests/fixtures/conversation_longitudinal_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    learners = fixture["learners"]
    assert len(learners) == 5 and sum(len(item["sessions"]) for item in learners) == 20
    assert sum(len(s["messages"]) for l in learners for s in l["sessions"]) == 80
    assert all(0 < len(m) <= 800 for l in learners for s in l["sessions"] for m in s["messages"])
    aliases = fixture.get("expected_aliases", {})
    baseline = Settings.from_env()
    require_local_dsn(baseline.database_url)
    database = "teachcopilot_conversation_run_" + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(baseline.database_url, dbname="postgres"))
    admin.autocommit = True
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    counters = {"generation_calls": 0, "embedding_calls": 0}
    http_path = out / "ollama_http.jsonl"
    report = {"status": "running", "data_origin": "synthetic_unreviewed", "human_reviewed": False,
              "teacher_review": "simulated_by_preregistered_expected_preferences",
              "expected_sessions": 20, "expected_messages": 80, "messages": [], "sessions": [],
              "paid_provider_calls": 0, "isolation_checks": [], "max_generation_calls": args.max_calls,
              "max_minutes": args.max_minutes, "preflight_only": args.preflight_only}

    class TracedRuntime(OllamaRuntime):
        def _post(self, endpoint, payload):
            if endpoint == "/api/chat":
                if counters["generation_calls"] >= args.max_calls or time.monotonic() >= deadline:
                    raise RuntimeError("finite_longitudinal_budget_exhausted")
                counters["generation_calls"] += 1
            elif "embed" in endpoint:
                counters["embedding_calls"] += 1
            row = {"endpoint": endpoint, "case": current.get("case"), "request": payload}
            call_started = time.perf_counter()
            try:
                response = super()._post(endpoint, payload)
                # Vector values are not needed for conversational audit.
                row["response"] = response if endpoint == "/api/chat" else {"embedding_count": len(response.get("embeddings", []))}
                return response
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                row["elapsed_seconds"] = time.perf_counter() - call_started
                with http_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    class ObservedRag(RagService):
        def answer(self, messages, **kwargs):
            observation = {"case": current.get("case"), "messages": deepcopy(messages),
                           "preferences": deepcopy(kwargs.get("preference_context")),
                           "memory_context": deepcopy(kwargs.get("memory_context"))}
            observations.append(observation)
            return super().answer(messages, **kwargs)

    current, observations = {}, []
    with admin.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        settings = replace(baseline, database_url=make_dsn(baseline.database_url, dbname=database), request_timeout=120)
        store = Store(settings.database_url)
        store.migrate()
        # Three explicitly allowlisted public tables, with actual local vectors.
        source = Store(baseline.database_url)
        public_counts = {}
        with source._connect() as conn, conn.cursor() as cursor, store._connect() as target, target.cursor() as destination:
            for table in ("catalog_topics", "embedding_models", "materials"):
                buffer = io.StringIO()
                cursor.execute("""SELECT column_name FROM information_schema.columns
                    WHERE table_schema='teachcopilot' AND table_name=%s AND is_generated='NEVER'
                    ORDER BY ordinal_position""", (table,))
                columns = sql.SQL(",").join(sql.Identifier(item["column_name"]) for item in cursor.fetchall())
                copy_table = sql.SQL("teachcopilot.{} ({})").format(sql.Identifier(table), columns)
                cursor.copy_expert(sql.SQL("COPY {} TO STDOUT WITH (FORMAT CSV)").format(copy_table), buffer)
                buffer.seek(0)
                destination.copy_expert(sql.SQL("COPY {} FROM STDIN WITH (FORMAT CSV)").format(copy_table), buffer)
                destination.execute(sql.SQL("SELECT count(*) AS n FROM teachcopilot.{}").format(sql.Identifier(table)))
                public_counts[table] = destination.fetchone()["n"]
        runtime = TracedRuntime(settings)
        rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
        template = json.loads((ROOT.parent / "learner-data/demo/learners/demo-nika-04.json").read_text(encoding="utf-8-sig"))
        password = "Synthetic isolated night fixture 2026"
        teacher = store.create_account("night-teacher@synthetic.test", "Synthetic night teacher", hash_password(password), "teacher")
        accounts, initial_knowledge = {}, {}
        for learner in learners:
            learner_id = "night-" + learner["code"].lower()
            card = deepcopy(template)
            card.update(learner_id=learner_id, pseudonym="SYNTHETIC " + learner["code"], interests=[], knowledge=[], error_patterns=[])
            card.pop("legal_name", None)
            card.pop("learner_model", None)
            card["mvp"] = {key: [] for key in card["mvp"]}
            card["profile"].update(explanation_style="Кратко, по одному шагу", autonomy_level="Условия самостоятельности пока неизвестны", motivation="Уточняется в диалоге")
            card["rag_context"].update(
                current_goal={"topic_id": learner["sessions"][0]["topic_id"], "goal": "Разобрать учебный вопрос, сохраняя неизвестные условия помощи.", "updated_at": "2026-10-18"},
                current_topics=[], priority_difficulties=[], effective_strategies=[], avoid=[],
                recent_progress={"date": "2026-10-18", "note": "Начальные учебные наблюдения отсутствуют."})
            store.save_card(card, teacher["id"])
            initial_knowledge[learner_id] = deepcopy(card["knowledge"])
            store.assign_learner(teacher["id"], learner_id)
            accounts[learner_id] = store.create_student_account(learner_id + "@synthetic.test", "Synthetic student", hash_password(password), learner_id, teacher["id"])
        with TestClient(create_app(settings=settings, store=store, runtime=runtime, rag_service=rag)) as client:
            def auth(email):
                response = client.post("/auth/login", json={"email": email, "password": password})
                assert response.status_code == 200, "synthetic login failed"
                return {"Authorization": "Bearer " + response.json()["access_token"]}

            teacher_headers = auth(teacher["email"])
            headers = {learner_id: auth(account["email"]) for learner_id, account in accounts.items()}
            ids = list(accounts)
            for learner_id in ids:
                foreign = next(item for item in ids if item != learner_id)
                denied = client.get(f"/learners/{foreign}/dashboard", headers=headers[learner_id]).status_code
                private = client.get(f"/learners/{learner_id}/memory", headers=headers[learner_id]).status_code
                report["isolation_checks"].append({"learner": learner_id, "foreign_dashboard_status": denied,
                                                   "student_memory_status": private, "passed": denied == private == 403})
            tracked = [fixture_path, Path(__file__), *[ROOT / "local_app" / name for name in
                       ("api.py", "runtime.py", "rag.py", "memory_service.py", "memory_store.py", "conversation_memory.py", "answer_adapter.py")],
                       ROOT / "local_app/prompts/conversation_interests.txt"]
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
            prereg = {"created_at": datetime.now(timezone.utc).isoformat(), "fixture": fixture,
                      "git_commit": commit, "model_identity": runtime.model_identity(settings.chat_model),
                      "manifest": runtime.manifest(), "seed": 42, "temperature": 0, "think": False,
                      "max_generation_calls": args.max_calls, "max_minutes": args.max_minutes,
                      "public_tables_copied": public_counts, "external_provider_disabled": True,
                      "review_policy": report["teacher_review"], "human_reviewed": False,
                      "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}}
            write_json(out / "preregistered.json", prereg)
            assert rag.status()["ready"], "copied real embedding index is not ready"
            assert all(item["passed"] for item in report["isolation_checks"])
            if args.preflight_only:
                report["status"] = "preflight_passed"
                return report
            for learner in learners:
                learner_id = "night-" + learner["code"].lower()
                for session_spec in learner["sessions"]:
                    code = session_spec["code"]
                    current["case"] = code
                    row = {"code": code, "learner_id": learner_id, "status": "failed", "simulated_teacher_review": True}
                    try:
                        created = client.post(f"/learners/{learner_id}/learning-sessions", headers=headers[learner_id],
                                              json={"topic_id": session_spec["topic_id"], "client_session_id": code})
                        assert created.status_code == 201, created.text
                        session_id = created.json()["learning_session_id"]
                        for index, message in enumerate(session_spec["messages"]):
                            current["case"] = code + "M" + str(index + 1)
                            before = counters["generation_calls"]
                            payload = {"learning_session_id": session_id, "client_message_id": current["case"],
                                       "metadata": {"subject_id": "python" if ".python." in session_spec["topic_id"] else "scratch" if ".scratch." in session_spec["topic_id"] else "math"},
                                       "messages": [{"role": "user", "content": message}]}
                            response = client.post("/v1/chat/completions", headers=headers[learner_id], json=payload)
                            result = {"code": current["case"], "learner_id": learner_id, "session_code": code,
                                      "user_message": message, "status_code": response.status_code,
                                      "response": response.json(), "generation_calls": counters["generation_calls"] - before,
                                      "observed_context": observations[-1] if observations and observations[-1]["case"] == current["case"] else None}
                            if result["observed_context"]:
                                memory_sources = (result["observed_context"].get("memory_context") or {}).get("sources", [])
                                own_ids = {item["source_id"] for item in store.get_learner_memory(learner_id)["sources"]}
                                result["memory_source_scope_ok"] = {item["source_id"] for item in memory_sources} <= own_ids
                            if index == 0 and session_spec is learner["sessions"][0]:
                                retry_before = counters["generation_calls"]
                                replay = client.post("/v1/chat/completions", headers=headers[learner_id], json=payload)
                                result["replay_cached"] = response.status_code == replay.status_code == 200 and response.json() == replay.json() and counters["generation_calls"] == retry_before
                            report["messages"].append(result)
                            write_json(out / "PROCESSING.json", {"status": "running", "completed_messages": len(report["messages"]), **counters})
                            write_json(out / (current["case"] + ".json"), result)
                            print(current["case"] + ": HTTP " + str(response.status_code), flush=True)
                        current["case"] = code + "-memory"
                        completed = client.post(f"/learning-sessions/{session_id}/complete", headers=headers[learner_id])
                        repeated = client.post(f"/learning-sessions/{session_id}/complete", headers=headers[learner_id])
                        assert completed.status_code == repeated.status_code == 202, completed.text
                        row["completion_idempotent"] = completed.json()["job_id"] == repeated.json()["job_id"]
                        job = MemoryWorker(store, runtime, worker_id="night-" + code).run_once()
                        row["memory_job"] = job
                        snapshot = store.get_learner_memory(learner_id, session_spec["topic_id"])
                        version = next((v for v in snapshot["versions"] if (v.get("context", {}).get("interest_extraction") or {}).get("learning_session_id") == session_id), None)
                        assert version is not None, "completed session has no interest extraction receipt"
                        candidates = version["context"].get("interest_candidates", [])
                        expected = {(operation, normal(value, aliases)) for operation, values in session_spec["expected_preferences"].items() for value in values}
                        actual = {(item["operation"], normal(item["interest"], aliases)) for item in candidates}
                        selected = [item["candidate_id"] for item in candidates if (item["operation"], normal(item["interest"], aliases)) in expected]
                        row.update(candidates=candidates, missing_expected=sorted(expected - actual), extra_candidates=sorted(actual - expected),
                                   interest_extraction=version["context"].get("interest_extraction"))
                        choice = {"expected_revision": snapshot["revision"], "interest_candidate_ids": selected}
                        url = f"/learners/{learner_id}/memory/versions/{version['version_id']}/confirm"
                        approved = client.post(url, headers=teacher_headers, json=choice)
                        row["approval_status"] = approved.status_code
                        assert approved.status_code == 200, approved.text
                        revision = store.get_revision(learner_id)
                        repeated = client.post(url, headers=teacher_headers, json=choice)
                        row["approval_idempotent"] = repeated.status_code == 200 and store.get_revision(learner_id) == revision
                        card = store.get_card(learner_id)
                        from local_app.conversation_memory import preference_projection
                        preferences = preference_projection(card)
                        actual_interests = {normal(item, aliases) for item in card["interests"]}
                        actual_avoided = {normal(item, aliases) for item in preferences["avoided_interests"]}
                        row.update(final_interests=card["interests"], final_preferences=preferences,
                                   final_interests_match=actual_interests == {normal(item, aliases) for item in session_spec["expected_final_interests"]},
                                   final_avoided_match=actual_avoided == {normal(item, aliases) for item in session_spec["expected_final_avoided"]},
                                   competencies_unchanged=card.get("knowledge") == initial_knowledge[learner_id],
                                   selected_memory_chars=len(version["context"].get("text", "")),
                                   topic_source_chars=sum(len(item.get("text", "")) for item in snapshot["sources"]),
                                   selected_context=version["context"], status="completed")
                    except Exception as exc:
                        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    report["sessions"].append(row)
                    write_json(out / (code + "-session.json"), row)
                    write_json(out / "report.partial.json", report)
                    print(code + ": " + row["status"], flush=True)
            report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        assert database.startswith("teachcopilot_conversation_run_") and len(database) == len("teachcopilot_conversation_run_") + 12
        with admin.cursor() as cursor:
            cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
            cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        admin.close()
        report.update(counters, elapsed_seconds=time.monotonic() - started, temporary_database_removed=True)
        if not args.preflight_only:
            sessions, messages = report["sessions"], report["messages"]
            report["technical_summary"] = {
                "http_200": sum(item["status_code"] == 200 for item in messages), "expected_messages": 80,
                "messages_not_attempted": 80 - len(messages),
                "sessions_completed": sum(item["status"] == "completed" for item in sessions), "expected_sessions": 20,
                "interest_extraction_ready": sum((item.get("interest_extraction") or {}).get("status") == "ready" for item in sessions),
                "final_interests_match": sum(item.get("final_interests_match") is True for item in sessions),
                "final_avoided_match": sum(item.get("final_avoided_match") is True for item in sessions),
                "competencies_unchanged": sum(item.get("competencies_unchanged") is True for item in sessions),
                "source_scope_failures": sum(item.get("memory_source_scope_ok") is False for item in messages),
                "cached_replays_passed": sum(item.get("replay_cached") is True for item in messages),
            }
            report["semantic_review"] = "pending_separate_ai_review_not_human_reviewed"
        write_json(out / "report.json", report)
        write_json(out / "PROCESSING.json", {"status": report["status"], "messages": len(report["messages"]),
                   "sessions": len(report["sessions"]), **counters, "human_reviewed": False, "paid_provider_calls": 0})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-calls", type=int, default=200)
    parser.add_argument("--max-minutes", type=float, default=45)
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 240 or not 1 <= args.max_minutes <= 60:
        parser.error("finite caps must be 1..240 calls and 1..60 minutes")
    result = run(args)
    print(json.dumps({key: result.get(key) for key in ("status", "generation_calls", "embedding_calls", "elapsed_seconds", "fatal_error")}), flush=True)
    return 0 if result["status"] in {"completed", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
