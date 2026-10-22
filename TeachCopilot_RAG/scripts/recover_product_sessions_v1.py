"""Continue an interrupted product pilot without repeating its unknown request.

The original package is read-only except its shared generation journal. This
recovery closes the interrupted fourth lesson, preserves two unexecuted turns,
then executes the two already preregistered third lessons. It is a continuation
of the same pilot, not its engineering iteration or an independent experiment.
Only an explicit --execute may start the existing local models.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_product_sessions_v1 import (FIXTURE, Pilot, load_fixture, normalized,
                                           owned_services, require_owned_database, write_json)

COMPLETED_CODES = ("L01S1", "L02S1", "L01S2")
INTERRUPTED_CODE = "L02S2"
UNKNOWN_MESSAGE_ID = "L02S2-E02"
SKIPPED_MESSAGE_IDS = ("L02S2-E03", "L02S2-E04")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_files(directory, budget_path):
    """Exclude only the authorized shared SQLite journal and its sidecars."""
    directory, budget_path = Path(directory).resolve(), Path(budget_path).resolve()
    mutable = {budget_path, Path(str(budget_path) + "-wal"), Path(str(budget_path) + "-shm"),
               Path(str(budget_path) + "-journal")}
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.resolve() not in mutable}


def load_checkpoint(source):
    source = Path(source).resolve()
    path = source / "report.partial.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    prereg = json.loads((source / "preregistered.json").read_text(encoding="utf-8"))
    require_owned_database(report["database_name"])
    fixture, _ = load_fixture()
    if report.get("status") != "running" or report.get("preflight_only"):
        raise ValueError("Recovery requires the original interrupted full pilot checkpoint")
    if prereg.get("fixture") != fixture or report.get("fixture_sha256") != sha256(FIXTURE):
        raise ValueError("The original preregistered six-lesson fixture must remain unchanged")
    rows = report.get("sessions", [])
    if [row.get("code") for row in rows] != [*COMPLETED_CODES, INTERRUPTED_CODE]:
        raise ValueError("Recovery only supports the documented fourth-lesson interruption boundary")
    if any(row.get("status") != "attempted" for row in rows[:3]) or rows[3].get("status") != "running":
        raise ValueError("Original completed and interrupted lesson states do not match")
    processes = report.get("processes", [])
    if len(processes) != 2 or any(processes[0][role]["pid"] == processes[1][role]["pid"] for role in ("api", "worker")):
        raise ValueError("The original real API and worker restart evidence is required")
    ledger = Path(report["budget_ledger"]).resolve()
    if not ledger.is_file():
        raise ValueError("The original shared generation journal is required")
    return report, prereg, fixture, ledger


def recovery_plan(report):
    return {"schema_version": "product_sessions_v1_interruption_recovery", "independent_experiment": False,
            "engineering_iteration": False, "database_name": report["database_name"],
            "preserved_completed_lessons": list(COMPLETED_CODES),
            "interrupted_lesson": INTERRUPTED_CODE, "unknown_request_never_repeated": UNKNOWN_MESSAGE_ID,
            "not_executed_message_ids": list(SKIPPED_MESSAGE_IDS),
            "remaining_new_lessons": ["L01S3", "L02S3"], "remaining_new_ordinary_messages": 8,
            "remaining_new_explicit_actions": 0, "remaining_teacher_recommendations": 1,
            "new_physical_lessons": 2, "physical_lesson_total": 6,
            "shared_journal": report["budget_ledger"], "max_generations_unchanged": 160,
            "automatic_retries": False, "new_external_provider_calls": 0,
            "content_review": "pending_separate_agent_review_not_human_expertise"}


def read_database_state(store, fixture):
    codes = [s["code"] for learner in fixture["learners"] for s in learner["sessions"]]
    learner_ids = [learner["learner_id"] for learner in fixture["learners"]]
    with store._connect() as connection, connection.cursor() as cursor:
        connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
        cursor.execute("""SELECT to_jsonb(s) AS session FROM teachcopilot.learning_sessions s
            WHERE client_session_id=ANY(%s) AND learner_id=ANY(%s) ORDER BY created_at,id""", (codes, learner_ids))
        sessions = [row["session"] for row in cursor.fetchall()]
        cursor.execute("""SELECT to_jsonb(m) AS message,s.content FROM teachcopilot.learning_messages m
            JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
            JOIN teachcopilot.learning_sessions ls ON ls.id=m.learning_session_id
            WHERE ls.client_session_id=%s ORDER BY m.created_at,m.id""", (INTERRUPTED_CODE,))
        messages = [{**row["message"], "literal_user_text": row["content"]} for row in cursor.fetchall()]
        cursor.execute("""SELECT to_jsonb(j) AS job FROM teachcopilot.memory_jobs j
            WHERE learner_id=ANY(%s) ORDER BY created_at,id""", (learner_ids,))
        jobs = [row["job"] for row in cursor.fetchall()]
        cursor.execute("""SELECT ls.client_session_id AS session_code,m.client_message_id,m.status
            FROM teachcopilot.learning_messages m JOIN teachcopilot.learning_sessions ls ON ls.id=m.learning_session_id
            WHERE ls.client_session_id=ANY(%s) AND ls.learner_id=ANY(%s) ORDER BY m.created_at,m.id""", (codes, learner_ids))
        delivery_statuses = [dict(row) for row in cursor.fetchall()]
    return {"sessions": sessions, "interrupted_messages": messages, "jobs": jobs, "delivery_statuses": delivery_statuses}


def ordinary_message_coverage(delivery_statuses, fixture):
    planned = [f"{session['code']}-E{index:02d}" for learner in fixture["learners"]
               for session in learner["sessions"] for index, event in enumerate(session["events"], 1)
               if event["kind"] == "message"]
    actual = {row["client_message_id"]: row["status"] for row in delivery_statuses}
    counts = {status: sum(actual.get(mid) == status for mid in planned)
              for status in ("completed", "unknown_outcome", "failed", "pending")}
    missing = [mid for mid in planned if mid not in actual]
    return {"planned": len(planned), **counts, "not_executed": len(missing), "not_executed_message_ids": missing,
            "coverage_complete": counts["completed"] == len(planned)}


def validate_database_boundary(state, fixture, *, recovered=False):
    sessions = {s["client_session_id"]: s for s in state["sessions"]}
    if len(state["sessions"]) != 4 or set(sessions) != {*COMPLETED_CODES, INTERRUPTED_CODE}:
        raise ValueError("Refusing to recreate or repeat an already attempted later lesson")
    if any(sessions[code]["status"] != "completed" for code in COMPLETED_CODES):
        raise ValueError("The original three completed lessons must remain completed")
    if sessions[INTERRUPTED_CODE]["status"] != "active":
        raise ValueError("The interrupted lesson must still be active before explicit completion")
    learner = fixture["learners"][1]
    spec = learner["sessions"][1]
    messages = state["interrupted_messages"]
    if [message["client_message_id"] for message in messages] != ["L02S2-E01", UNKNOWN_MESSAGE_ID]:
        raise ValueError("Only E01 and the unknown E02 may have been attempted")
    for message, expected in zip(messages, spec["events"][:2]):
        if message["literal_user_text"] != expected["text"]:
            raise ValueError("Persisted source differs from the original preregistered utterance")
    if messages[0]["status"] != "completed":
        raise ValueError("The first interrupted-lesson response must already be complete")
    allowed_unknown = {"unknown_outcome"} if recovered else {"pending", "unknown_outcome"}
    if messages[1]["status"] not in allowed_unknown:
        raise ValueError("The second response must retain an unknown outcome without retry")
    sid = sessions[INTERRUPTED_CODE]["id"]
    chat_jobs = [j for j in state["jobs"] if j.get("idempotency_key") == f"chat:{sid}:{UNKNOWN_MESSAGE_ID}"]
    if len(chat_jobs) != 1 or chat_jobs[0].get("call_started_at") is None:
        raise ValueError("Missing durable evidence that the unknown request started its provider call")
    if recovered and chat_jobs[0]["status"] != "unknown_outcome":
        raise ValueError("Expired started chat job must be recovered as unknown, never queued")
    active = [j for j in state["jobs"] if j["status"] in {"queued", "running"}]
    if recovered and active:
        raise ValueError("Recovery cannot silently execute any pre-existing queued or running job")
    if not recovered and any(j["id"] != chat_jobs[0]["id"] for j in active):
        raise ValueError("Unexpected outstanding job outside the documented interruption")
    return sid


def restored_interrupted_events(messages, session, source):
    evidence = {message["client_message_id"]: message for message in messages}
    events = []
    for index, event in enumerate(session["events"], 1):
        mid = f"{session['code']}-E{index:02d}"
        row = evidence.get(mid)
        if row:
            events.append({"id": mid, "kind": event["kind"], "status": row["status"],
                           "response_completed": row["status"] == "completed", "response": row.get("response"),
                           "reconstructed_from_durable_database": True, "original_package": str(source),
                           "source_id": row["source_id"], "error": row.get("error"),
                           "generation_repeated": False})
        else:
            events.append({"id": mid, "kind": event["kind"], "status": "not_executed",
                           "reason": "original_usage_interruption_remaining_turns_deferred_to_single_regression",
                           "literal_preregistered_text": event["text"], "generation_repeated": False})
    return events


def assert_original_children_absent(report):
    import psutil
    for phase in report["processes"]:
        for role in ("api", "worker"):
            try:
                process = psutil.Process(phase[role]["pid"])
                command = " ".join(process.cmdline())
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as exc:
                raise RuntimeError("Cannot prove the previous owned process is absent") from exc
            if "run_product_sessions_v1.py" in command and "--child-role" in command:
                raise RuntimeError("Original owned API or worker still exists; recovery will not stop it implicitly")


def finish_interrupted_lesson(pilot, learner, spec, row):
    lid, code = learner["learner_id"], spec["code"]
    sid = row["learning_session_id"]
    pilot.session_ids[code] = sid
    completed = pilot.request("POST", f"/learning-sessions/{sid}/complete", headers=pilot.headers[lid], expected=202,
                              label="explicit_close_interrupted_lesson_no_chat_replay")
    pilot.approve(learner, spec, completed["job_id"])
    from local_app.conversation_memory import preference_projection
    card = pilot.store.get_card(lid)
    aliases = pilot.fixture["review_aliases"]
    preferences = preference_projection(card)
    pilot.check(code + "-interests", {normalized(x, aliases) for x in card["interests"]} ==
                {normalized(x, aliases) for x in spec["expected_final_interests"]}, actual=card["interests"])
    pilot.check(code + "-avoided-stories", {normalized(x, aliases) for x in preferences["avoided_interests"]} ==
                {normalized(x, aliases) for x in spec["expected_final_avoided"]}, actual=preferences["avoided_interests"],
                original_expected_fact_not_rewritten=True, relevant_turn="L02S2-E03-not_executed")
    pilot.check(code + "-knowledge-unchanged", card.get("knowledge") == [])
    prepared = pilot.request("POST", f"/learners/{lid}/next-step", headers=pilot.teacher_headers,
        payload={"topic_id": spec["topic_id"], "method": "tail", "expected_revision": pilot.store.get_revision(lid)},
        expected=202, label="second_and_last_initial_teacher_recommendation")
    row["recommendation"] = pilot.wait_job(prepared["job_id"])
    pilot.check(code + "-recommendation-technical", row["recommendation"]["status"] == "completed",
                status=row["recommendation"]["status"], content_review="pending_separate_agent_review_not_human")
    row.update(status="attempted_after_interruption_recovery", unknown_request_never_repeated=UNKNOWN_MESSAGE_ID,
               not_executed_message_ids=list(SKIPPED_MESSAGE_IDS), engineering_iteration=False)
    write_json(pilot.output / (code + "-session.json"), row)
    write_json(pilot.output / "report.partial.json", pilot.report)


def run(args):
    source = args.continue_from.resolve()
    original, original_prereg, fixture, budget_path = load_checkpoint(source)
    plan = recovery_plan(original)
    if not args.execute:
        return {"status": "dry_run", "execute_required": True, **plan}

    from psycopg2.extensions import make_dsn
    from local_app.config import Settings
    from local_app.generation_run_budget import GenerationRunBudget
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from scripts.verify_memory import require_local_dsn

    output = args.output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("Recovery output must be a fresh directory outside the read-only original package")
    output.mkdir(parents=True, exist_ok=False)
    hashes_before = immutable_files(source, budget_path)
    budget = GenerationRunBudget(budget_path)
    budget_before = budget.snapshot()
    if budget_before["deadline_reached"] or budget_before["remaining_generations"] <= 0:
        raise ValueError("The explicit additional-time authorization must already be appended to the original journal")
    if budget_before["max_generations"] != 160:
        raise ValueError("Recovery cannot replace or expand the original 160-call limit")
    baseline = Settings.from_env()
    require_local_dsn(baseline.database_url)
    settings = replace(baseline, database_url=make_dsn(baseline.database_url, dbname=original["database_name"]),
        request_timeout=180, generation_scheduler_dsn=baseline.database_url,
        generation_run_budget_path=str(budget_path), learning_operation_selection_enabled=False,
        checked_division_context_enabled=False, demo_login_enabled=False)
    report = deepcopy(original)
    report.update(status="recovery_running", continuation_of=str(source), recovery_plan=plan,
                  independent_experiment=False, engineering_iteration=False, temporary_database_removed=False,
                  retained_synthetic_database=original["database_name"], budget_before_recovery=budget_before,
                  original_file_sha256=hashes_before, recovery_output=str(output))
    report.pop("fatal_error", None)
    started = time.monotonic()
    try:
        assert_original_children_absent(original)
        store = Store(settings.database_url)
        store.ping()
        before = read_database_state(store, fixture)
        write_json(output / "database-before-recovery.json", before)
        validate_database_boundary(before, fixture)
        recovered = store.recover_memory_jobs()
        after = read_database_state(store, fixture)
        sid = validate_database_boundary(after, fixture, recovered=True)
        write_json(output / "database-after-lease-recovery.json", after)
        write_json(output / "interruption-receipt.json", {"at": datetime.now(timezone.utc).isoformat(),
            "original_package": str(source), "original_terminal_report_missing": not (source / "report.json").exists(),
            "recovery": recovered, "unknown_request_id": UNKNOWN_MESSAGE_ID, "unknown_request_repeated": False,
            "not_executed_message_ids": list(SKIPPED_MESSAGE_IDS), "original_budget_before_recovery": budget_before,
            "original_artifact_sha256": hashes_before})
        teacher = store.get_account(email="product-teacher@synthetic.test")
        students = {l["learner_id"]: store.get_account(email=l["learner_id"] + "@synthetic.test") for l in fixture["learners"]}
        runtime = OllamaRuntime(settings)
        if settings.chat_model != "gemma4:12b":
            raise ValueError("The original gemma4:12b model is required")
        models = runtime.check_models()
        known = {m["name"]: m["digest"] for m in original_prereg["models"]["models"]}
        observed = {m["name"]: m["digest"] for m in models["models"]}
        if observed != known:
            raise ValueError("Recovery model identity differs from the original preregistration")
        index = RagService(store, runtime, settings).status()
        if not index.get("ready"):
            raise ValueError("The original isolated teaching-material index is not ready")
        write_json(output / "preregistered.json", {"created_at": datetime.now(timezone.utc).isoformat(), "plan": plan,
            "fixture": fixture, "models": models, "manifest": runtime.manifest(), "index_state": index,
            "budget": budget.snapshot(), "source_report_sha256": sha256(source / "report.partial.json"),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            "sha256": {Path(__file__).relative_to(ROOT).as_posix(): sha256(__file__), FIXTURE.relative_to(ROOT).as_posix(): sha256(FIXTURE)},
            "human_reviewed": False, "external_answers_enabled": False})
        pilot = Pilot(store, fixture, budget, output, settings, report)
        pilot.teacher_id = teacher["id"]
        pilot.student_ids = {lid: account["id"] for lid, account in students.items()}
        pilot.session_ids = {s["code"]: s["learning_session_id"] for s in report["sessions"]}
        interrupted = report["sessions"][3]
        interrupted["learning_session_id"] = sid
        interrupted["events"] = restored_interrupted_events(after["interrupted_messages"], fixture["learners"][1]["sessions"][1], source)
        with owned_services(settings, baseline.database_url, budget_path, output, "recovery") as (client, base, identities):
            report["processes"].append({"phase": "recovery", **identities})
            pilot.bind(client, base)
            pilot.check("interruption-unknown-outcome-preserved", after["interrupted_messages"][1]["status"] == "unknown_outcome",
                        request_id=UNKNOWN_MESSAGE_ID, generation_repeated=False)
            pilot.check("interruption-two-turns-not-executed", sum(e["status"] == "not_executed" for e in interrupted["events"]) == 2)
            # The correction source and approved draft already exist. No
            # correction, prior reply or prior recommendation is submitted again.
            finish_interrupted_lesson(pilot, fixture["learners"][1], fixture["learners"][1]["sessions"][1], interrupted)
            for learner in fixture["learners"]:
                pilot.session(learner, learner["sessions"][2])
            for learner in fixture["learners"]:
                lid = learner["learner_id"]
                archive = pilot.request("GET", f"/learners/{lid}/learning-sessions", headers=pilot.headers[lid], expected=200)
                pilot.check(lid + "-archive-sessions-after-recovery", len(archive["sessions"]) == 3)
                for item in archive["sessions"]:
                    history = pilot.request("GET", f"/learners/{lid}/learning-sessions/{item['learning_session_id']}/history", headers=pilot.headers[lid], expected=200)
                    pilot.check(lid + "-archive-history-after-recovery", bool(history["messages"]) and history["status"] == "completed")
        pilot.verify_observations()
        report["status"] = "completed_with_interruption_and_findings"
        report["all_mechanical_checks_passed"] = all(c["passed"] for c in report["checks"])
        report["initial_failures_preserved"] = [c for c in report["checks"] if not c["passed"]]
        final_state = read_database_state(store, fixture)
        report["ordinary_message_coverage"] = ordinary_message_coverage(final_state["delivery_statuses"], fixture)
        write_json(output / "database-final-delivery-state.json", final_state)
        write_json(output / "database-snapshot.json", {l["learner_id"]: {"card": store.get_card(l["learner_id"]),
                   "memory": store.get_learner_memory(l["learner_id"])} for l in fixture["learners"]})
    except Exception as exc:
        report["status"] = "recovery_failed"
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["budget"] = budget.snapshot()
        report["recovery_elapsed_seconds"] = time.monotonic() - started
        hashes_after = immutable_files(source, budget_path)
        report["original_immutable_artifacts_unchanged"] = hashes_after == hashes_before
        if hashes_after != hashes_before:
            report["status"] = "recovery_failed"
            report["original_file_sha256_after"] = hashes_after
        write_json(output / "report.json", report)
        write_json(output / "PROCESSING.json", {"status": report["status"], "budget": report["budget"], "paid_provider_calls": 0})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-from", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and not args.output_dir:
        parser.error("--execute requires a fresh --output-dir")
    result = run(args)
    print(json.dumps({key: result.get(key) for key in ("status", "recovery_plan", "budget", "fatal_error")}, ensure_ascii=False), flush=True)
    return 1 if result["status"] == "recovery_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
