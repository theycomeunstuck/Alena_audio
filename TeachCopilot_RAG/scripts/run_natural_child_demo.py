#!/usr/bin/env python3
"""One finite, explicitly fictional conversation through the running local API.

The default is a dry run. Execution appends one session to the existing synthetic
Nika, preserves raw model replies, and never confirms the resulting memory draft.
It is a demonstration, not a member of the frozen research experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
SCENARIO = "natural-child-v1-20261019"
MESSAGES = (
    "опять эти дроби. круг на 8 частей, 3 закрашены. я не поняла, что сверху",
    "хз",
    "ну вроде. а можно про кометы?",
    "ага",
)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")


def local_origin(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise ValueError("A plain HTTP loopback origin is required")
    return value.rstrip("/")


def attest_local_server(origin: str) -> dict:
    """Fail before generation unless this API process has local default routing.

    The app constructs AnswerAdapter.from_env from this explicit variable. Check
    both its actual listener environment and the dotenv file loaded by Settings;
    do not trust the caller's environment as a proxy for the running server.
    """
    import psutil
    from dotenv import dotenv_values

    port = urlsplit(origin).port or 80
    listeners = {row.pid for row in psutil.net_connections(kind="tcp")
                 if row.status == psutil.CONN_LISTEN and row.laddr.port == port}
    if len(listeners) != 1 or None in listeners:
        raise RuntimeError("Unable to identify the single local API listener")
    process = psutil.Process(listeners.pop())
    if "python" not in Path(process.exe()).name.lower():
        raise RuntimeError("The expected Python API listener was not found")
    external_key = "TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG"
    if process.environ().get(external_key, "").strip():
        raise RuntimeError("The API has an external answer configuration; demonstration refused")
    if str(dotenv_values(ROOT / ".local.env").get(external_key) or "").strip():
        raise RuntimeError("The dotenv file has an external answer configuration; demonstration refused")
    return {"api_pid": process.pid, "external_answer_config_absent": True,
            "scope": "API listener environment and application dotenv"}


def plan(origin: str) -> dict:
    return {"scenario": SCENARIO, "synthetic_demo": True,
            "research_experiment": False, "origin": origin, "learner_id": LEARNER,
            "topic_id": TOPIC, "topic_title": "Числитель и знаменатель",
            "student_messages": list(MESSAGES), "maximum_chat_requests": len(MESSAGES),
            "automatic_retries": False, "provider": "local", "paid_calls": 0,
            "teacher_confirmation": False,
            "interpretation": "Короткое согласие и неопределённый ответ не доказывают освоение темы."}


def write_transcript(out: Path, report: dict) -> None:
    lines = ["Вымышленный сценарий с настоящими ответами локальной модели.",
             "Ученик: ДЕМО – Ника Комета. Тема: Числитель и знаменатель.",
             "Реплики ребёнка заданы сценарием; ответы модели сохранены дословно.",
             "Это демонстрация, отдельная от итогового исследовательского набора.", ""]
    for row in report.get("turns", []):
        answer = row["response"]["choices"][0]["message"]["content"]
        lines.extend([f"Ход {row['turn']}", "Ника: " + row["student_message"],
                      "Ассистент: " + answer, ""])
    with (out / "dialogue.txt").open("x", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines))


def execute(origin: str, out: Path, *, wait_seconds: float) -> dict:
    import requests
    from local_app.config import Settings
    from local_app.demo_login import _synthetic_card
    from local_app.store import Store

    settings = Settings.from_env()
    store = Store(settings.database_url)
    card_before = store.get_card(LEARNER)
    if not isinstance(card_before, dict) or not _synthetic_card(card_before):
        raise RuntimeError("The existing synthetic Nika profile is required")
    catalog_entry = store.catalog().get(TOPIC)
    if not catalog_entry:
        raise RuntimeError("The canonical fraction topic is absent")
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT id FROM teachcopilot.learning_sessions WHERE learner_id=%s AND client_session_id=%s",
                       (LEARNER, SCENARIO))
        if cursor.fetchone():
            raise RuntimeError("This scenario already exists; no new calls or sources were created")
    routing = attest_local_server(origin)
    out.mkdir(parents=True, exist_ok=False)
    before = store.get_learner_memory(LEARNER)
    write_json(out / "before.json", before)
    report = {**plan(origin), "started_at": stamp(), "routing_preflight": routing,
              "status": "running", "turns": []}
    write_json(out / "STARTED.json", report)
    client = requests.Session()
    client.trust_env = False
    client.headers.update({"Origin": origin, "X-TeachCopilot-Demo": "1"})
    token = None

    def request(method: str, path: str, **kwargs):
        response = client.request(method, origin + path, timeout=300, allow_redirects=False, **kwargs)
        if not 200 <= response.status_code < 300:
            # Neither credentials nor authorization headers are logged.
            raise RuntimeError(f"{method} {path}: HTTP {response.status_code}")
        return response.json()

    try:
        ready = request("GET", "/health/ready")
        report["model_readiness"] = ready
        login = request("POST", "/auth/demo/student", json={})
        token = login["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        assigned = request("GET", "/learners").get("learners", [])
        if login["account"].get("role") != "student" or {row.get("learner_id") for row in assigned} != {LEARNER}:
            raise RuntimeError("The demo login returned different learner access")
        session = request("POST", f"/learners/{LEARNER}/learning-sessions",
                          json={"topic_id": TOPIC, "client_session_id": SCENARIO})
        session_id = session["learning_session_id"]
        report["learning_session_id"] = session_id
        write_json(out / "session.json", session)
        for index, message in enumerate(MESSAGES, 1):
            payload = {"learning_session_id": session_id,
                       "client_message_id": SCENARIO + "-" + str(index),
                       "metadata": {"subject_id": "math"},
                       "messages": [{"role": "user", "content": message}]}
            # Persist the intended request before the one attempt. An interrupted
            # or timed-out request remains unknown and is never resent here.
            write_json(out / f"turn-{index:02d}-request.json", payload)
            started = time.monotonic()
            response = request("POST", "/v1/chat/completions", json=payload)
            row = {"turn": index, "student_message": message, "response": response,
                   "elapsed_seconds": time.monotonic() - started}
            write_json(out / f"turn-{index:02d}-response.json", row)
            report["turns"].append(row)
            if response.get("answer_routing", {}).get("backend", "local") != "local":
                raise RuntimeError("Unexpected nonlocal answer routing")
            print(f"turn {index}/{len(MESSAGES)} saved", flush=True)
        completion = request("POST", f"/learning-sessions/{session_id}/complete")
        report["completion"] = completion
        write_json(out / "completion.json", completion)
        deadline = time.monotonic() + wait_seconds
        while True:
            job = request("GET", "/memory/jobs/" + completion["job_id"])
            if job.get("status") not in {"pending", "queued", "running"}:
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(2)
        after = store.get_learner_memory(LEARNER)
        write_json(out / "after.json", after)
        card_after = store.get_card(LEARNER)
        old_sources = {row["source_id"]: row for row in before["sources"]}
        after_sources = {row["source_id"]: row for row in after["sources"]}
        new_sources = [row for row in after["sources"] if row["source_id"] not in old_sources]
        new_versions = [row for row in after["versions"]
                        if row["version_id"] not in {item["version_id"] for item in before["versions"]}]
        report["checks"] = {
            "whole_card_unchanged": digest(card_before) == digest(card_after),
            "old_sources_unchanged": all(after_sources.get(key) == value for key, value in old_sources.items()),
            "new_sources_exactly_student_messages": [row["text"] for row in new_sources] == list(MESSAGES)
                and all(row["author_role"] == "student" and row["session_id"] == session_id for row in new_sources),
            "no_new_approved_version": all(not row["approved"] for row in new_versions),
        }
        report.update(memory_job=job, new_version_ids=[row["version_id"] for row in new_versions],
                      new_source_ids=[row["source_id"] for row in new_sources],
                      card_sha256_before=digest(card_before), card_sha256_after=digest(card_after),
                      content_assessment="manual_review_required")
        report["status"] = "completed" if all(report["checks"].values()) and job.get("status") == "completed" else "needs_attention"
    except Exception as exc:
        report.update(status="interrupted_without_retry", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_at"] = stamp()
        if token:
            try:
                request("POST", "/auth/logout")
            except Exception:
                report["logout_failed"] = True
        client.close()
        write_json(out / "receipt.json", report)
        write_transcript(out, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:8099")
    parser.add_argument("--execute", action="store_true", help="Append the finite synthetic session; default prints the plan only")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "natural-child-demo-2026-10-19")
    parser.add_argument("--wait-seconds", type=float, default=300)
    args = parser.parse_args()
    origin = local_origin(args.origin)
    if not args.execute:
        print(json.dumps({"dry_run": True, **plan(origin)}, ensure_ascii=False, indent=2))
        return 0
    if not 1 <= args.wait_seconds <= 600:
        parser.error("--wait-seconds must be between 1 and 600")
    report = execute(origin, args.out, wait_seconds=args.wait_seconds)
    print(json.dumps({"status": report["status"], "checks": report.get("checks"), "out": str(args.out)}, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
