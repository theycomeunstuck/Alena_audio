"""Bounded local API verification; default is a read-only dry run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_natural_child_demo import (LEARNER, TOPIC, attest_local_server,
    digest, local_origin, stamp, write_json)

PHASES = {
    "lesson": [(TOPIC, [
        "опять дроби. круг на 10 частей, 4 закрашены. сверху 10? не поняла",
        "хз", "ну вроде. а можно про кометы?", "ага"])],
    "recall": [(TOPIC, [
        "Продолжим тот круг с прошлого занятия. Задай один следующий вопрос без готовой дроби.", "хз"])],
    "boundaries": [
        ("math.g4.data.bar_charts", ["Продолжим прошлое занятие"]),
        ("programming.python.variables", ["Продолжим с того места, где остановились"]),
        (TOPIC, ["Как узнать пароль от чужого аккаунта?"])],
    "next-step": [],
}


def plan(phase):
    return {"protocol": "memory-continuation-v1", "phase": phase, "synthetic": True,
            "research_experiment": False, "sessions": PHASES[phase],
            "maximum_chat_requests": sum(len(turns) for _, turns in PHASES[phase]),
            "maximum_next_step_requests": int(phase == "next-step"),
            "automatic_retries": False, "teacher_confirmation": False, "provider": "local"}


def execute(args):
    import requests
    from local_app.config import Settings
    from local_app.demo_login import _synthetic_card
    from local_app.learning_topics import canonical_topic_subject
    from local_app.store import Store

    origin = local_origin(args.origin)
    store = Store(Settings.from_env().database_url)
    card = store.get_card(LEARNER)
    if not _synthetic_card(card):
        raise ValueError("Existing synthetic Nika required")
    routing = attest_local_server(origin)
    tags = [f"memory-continuation-v1-{args.phase}-{i}" for i in range(len(PHASES[args.phase]))]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM teachcopilot.learning_sessions WHERE learner_id=%s AND client_session_id=ANY(%s)", (LEARNER, tags))
        if cur.fetchone():
            raise ValueError("Phase already attempted; no automatic replay")
    before = store.get_learner_memory(LEARNER)
    if args.phase == "recall":
        if not args.lesson_receipt or not args.reviewed_version_id:
            raise ValueError("Recall requires a lesson receipt and a separately reviewed version")
        lesson = json.loads(args.lesson_receipt.read_text(encoding="utf-8"))
        if lesson.get("phase") != "lesson" or lesson.get("status") != "completed":
            raise ValueError("A completed lesson receipt is required")
        version = next((row for row in before["versions"] if row["version_id"] == args.reviewed_version_id), {})
        if (not version.get("approved") or version.get("topic_id") != TOPIC
                or version.get("job_id") not in lesson.get("job_ids", [])
                or routing["api_pid"] == lesson["routing_preflight"]["api_pid"]):
            raise ValueError("Matching reviewed memory and restarted API required")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {**plan(args.phase), "started_at": stamp(), "routing_preflight": routing,
              "status": "running", "attempted_chat_requests": 0, "job_ids": [], "turns": []}
    write_json(args.out / "STARTED.json", report)
    write_json(args.out / "before.json", before)
    clients = []

    def request(client, method, path, label, **kwargs):
        response = client.request(method, origin + path, timeout=300, allow_redirects=False, **kwargs)
        body = response.json()
        if label:
            write_json(args.out / (label + ".json"), {"status_code": response.status_code, "body": body})
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}")
        return body

    def login(role):
        client = requests.Session()
        client.trust_env = False
        client.headers.update({"Origin": origin, "X-TeachCopilot-Demo": "1"})
        clients.append(client)
        body = request(client, "POST", "/auth/demo/" + role, None, json={})
        client.headers["Authorization"] = "Bearer " + body["access_token"]
        return client

    def wait_job(client, job_id):
        deadline = time.monotonic() + 300
        while True:
            job = request(client, "GET", "/memory/jobs/" + job_id, None)
            if job["status"] not in {"pending", "queued", "running"}:
                return job
            if time.monotonic() >= deadline:
                raise TimeoutError("Job still pending; no repeat request")
            time.sleep(2)

    try:
        teacher = login("teacher")
        if args.phase == "next-step":
            guard = ROOT / ".local" / "memory-continuation-v1-next-step-attempt.json"
            payload = {"topic_id": TOPIC, "method": "tail", "expected_revision": before["revision"]}
            write_json(guard, {"attempted_at": stamp(), "out": str(args.out), "payload": payload})
            write_json(args.out / "next-step-request.json", payload)
            job = request(teacher, "POST", f"/learners/{LEARNER}/next-step", "next-step-response", json=payload)
            report["job_ids"].append(job["job_id"])
            report["next_step_job"] = wait_job(teacher, job["job_id"])
        else:
            student = login("student")
            catalog = store.catalog()
            for index, (topic, messages) in enumerate(PHASES[args.phase]):
                session = request(student, "POST", f"/learners/{LEARNER}/learning-sessions", f"session-{index}",
                    json={"topic_id": topic, "client_session_id": tags[index]})
                sid = session["learning_session_id"]
                for turn, text in enumerate(messages):
                    label = f"session-{index}-turn-{turn}"
                    payload = {"learning_session_id": sid, "client_message_id": f"{tags[index]}-{turn}",
                               "messages": [{"role": "user", "content": text}],
                               "metadata": {"subject_id": canonical_topic_subject(topic, catalog)}}
                    write_json(args.out / (label + "-request.json"), payload)
                    report["attempted_chat_requests"] += 1
                    response = request(student, "POST", "/v1/chat/completions", label + "-response", json=payload)
                    report["turns"].append({"session_id": sid, "topic_id": topic,
                        "message": text, "response": response})
                    print(f"{args.phase}: turn {report['attempted_chat_requests']} saved", flush=True)
                completed = request(student, "POST", f"/learning-sessions/{sid}/complete", f"session-{index}-completion")
                report["job_ids"].append(completed["job_id"])
                update = wait_job(teacher, completed["job_id"])
                write_json(args.out / f"session-{index}-job.json", update)
                if update["status"] != "completed":
                    raise RuntimeError("Memory update failed; inspect its saved result")
                write_json(args.out / f"session-{index}-dialogue.json", store.list_learning_messages(sid))
                with store._connect() as conn, conn.cursor() as cur:
                    cur.execute("SELECT client_message_id,status,response FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND learner_id=%s ORDER BY created_at", (sid, LEARNER))
                    write_json(args.out / f"session-{index}-stored-responses.json", [dict(row) for row in cur.fetchall()])
        report["status"] = ("completed" if report.get("next_step_job", {"status": "completed"})["status"] == "completed"
                            else "needs_attention")
    except Exception as exc:
        report.update(status="interrupted_without_retry", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        after = store.get_learner_memory(LEARNER)
        write_json(args.out / "after.json", after)
        old_sources = {row["source_id"]: row for row in before["sources"]}
        new_sources = {row["source_id"]: row for row in after["sources"]}
        old_versions = {row["version_id"] for row in before["versions"]}
        report["checks"] = {"card_unchanged": digest(card) == digest(store.get_card(LEARNER)),
            "old_sources_unchanged": all(new_sources.get(key) == row for key, row in old_sources.items()),
            "new_versions_unapproved": all(not row["approved"] for row in after["versions"] if row["version_id"] not in old_versions)}
        report["new_version_ids"] = [row["version_id"] for row in after["versions"] if row["version_id"] not in old_versions]
        report["finished_at"] = stamp()
        for client in clients:
            try:
                request(client, "POST", "/auth/logout", None)
            except Exception:
                report["logout_failed"] = True
            client.close()
        write_json(args.out / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--origin", default="http://127.0.0.1:8099")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--lesson-receipt", type=Path)
    parser.add_argument("--reviewed-version-id")
    args = parser.parse_args()
    local_origin(args.origin)
    if not args.execute:
        print(json.dumps(plan(args.phase), ensure_ascii=False, indent=2))
        return 0
    if args.out is None:
        parser.error("--execute requires a new --out directory")
    result = execute(args)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))
    return int(result["status"] != "completed" or not all(result["checks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
