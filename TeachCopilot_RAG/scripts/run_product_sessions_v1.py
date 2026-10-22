"""Finite synthetic product sessions in an isolated PostgreSQL database.

Dry-run is the default. --execute is required even with --preflight-only.
The real API and memory worker run in owned child processes. The shared local
generation ledger and working database admission lane survive their restart.
No request with a failed or unknown outcome is automatically repeated.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests/fixtures/product_sessions_v1.json"
REGRESSION_FIXTURE = ROOT / "tests/fixtures/product_sessions_v1_regression.json"
DATABASE_PREFIX = "teachcopilot_product_sessions_"
COPY_TABLES = ("catalog_topics", "embedding_models", "materials")
PASSWORD = "Synthetic product sessions 2026"


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def append_json(path: Path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def load_fixture(path=FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    learners = fixture["learners"]
    sessions = [session for learner in learners for session in learner["sessions"]]
    events = [event for session in sessions for event in session["events"]]
    counts = {"ordinary_messages": sum(e["kind"] == "message" for e in events),
              "explicit_actions": sum(e["kind"] == "action" for e in events),
              "teacher_recommendations": sum(bool(s.get("recommendation")) for s in sessions)}
    if (len(learners) != 2 or len(sessions) != 6 or
            any(len(learner["sessions"]) != 3 for learner in learners)):
        raise ValueError("Product fixture must contain two learners and six sessions")
    if counts != {"ordinary_messages": 24, "explicit_actions": 12, "teacher_recommendations": 2}:
        raise ValueError("Product fixture exceeds or changes preregistered finite counts")
    if len({s["code"] for s in sessions}) != 6:
        raise ValueError("Session codes must be unique")
    if any(len(e.get("text", "")) > 800 for e in events):
        raise ValueError("Synthetic messages must be bounded")
    return fixture, counts


def normalized(value, aliases):
    value = " ".join(str(value).casefold().split())
    for canonical, variants in aliases.items():
        if value in {canonical.casefold(), *(v.casefold() for v in variants)}:
            return canonical.casefold()
    return value


def require_owned_database(database):
    if not re.fullmatch(re.escape(DATABASE_PREFIX) + r"[0-9a-f]{12}", database):
        raise ValueError("Refusing a database outside the runner's exact namespace")
    return database


def load_regression_fixture(path=REGRESSION_FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    events = [e for s in fixture["sessions"] for e in s["events"]]
    counts = {"ordinary_messages": sum(e["kind"] == "message" for e in events),
              "explicit_actions": sum(e["kind"] == "action" for e in events),
              "teacher_recommendations": sum(bool(s.get("recommendation")) for s in fixture["sessions"])}
    if (fixture.get("independent_experiment") is not False or
            fixture.get("max_engineering_iterations") != 1 or len(fixture["sessions"]) != 2 or
            counts["ordinary_messages"] > 8 or counts["explicit_actions"] > 4 or counts["teacher_recommendations"]):
        raise ValueError("Regression must remain one dependent iteration, at most eight messages and four actions")
    return fixture, counts


def make_card(learner):
    """A new synthetic card, with no learner profile copied from any database."""
    return {"schema_version": 2, "learner_id": learner["learner_id"],
            "pseudonym": "SYNTHETIC " + learner["code"], "age_group": "9-10", "grade": "4",
            "language": "russian", "interests": [], "knowledge": [], "error_patterns": [],
            "profile": {"explanation_style": "Кратко, по одному шагу", "pace": "Уточняется",
                        "autonomy_level": "Условия помощи неизвестны", "motivation": "Уточняется",
                        "prefers_visual": False},
            "rag_context": {"current_goal": {"topic_id": learner["sessions"][0]["topic_id"],
                            "goal": "Разобрать учебный вопрос, сохраняя неизвестные условия помощи.",
                            "updated_at": "2026-10-22"}, "current_topics": [],
                            "priority_difficulties": [], "effective_strategies": [], "avoid": [],
                            "recent_progress": {"date": "2026-10-22", "note": "Наблюдений пока нет."}},
            "mvp": {key: [] for key in ("story_preferences", "learning_preferences_observed", "help_strategies",
                                        "journal", "points_ledger")}}


def child_process(role, directory):
    """Entry point invoked only in a fresh child with an explicit temporary DSN."""
    from psycopg2.extensions import parse_dsn
    from local_app.config import Settings
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store

    settings = Settings.from_env()
    database = parse_dsn(settings.database_url)["dbname"]
    require_owned_database(database)
    if not settings.generation_scheduler_dsn or settings.generation_scheduler_dsn == settings.database_url:
        raise RuntimeError("The pilot must use the working database shared generation scheduler")
    if not settings.generation_run_budget_path:
        raise RuntimeError("Missing persistent generation budget")
    runtime = OllamaRuntime(settings)
    store = Store(settings.database_url)
    nonce = os.environ["TEACHCOPILOT_PRODUCT_NONCE"]
    identity = {"pid": os.getpid(), "nonce": nonce, "role": role, "database": database}
    if role == "worker":
        from local_app.memory_service import MemoryWorker
        worker = MemoryWorker(store, runtime, worker_id="product-" + nonce)
        write_json(directory / "address.json", identity)
        while not (directory / "stop").exists():
            job = worker.run_once()
            if job:
                append_json(directory / "jobs.jsonl", job)
            else:
                time.sleep(0.1)
        return

    import uvicorn
    from local_app.api import create_app
    from local_app.rag import RagService
    from local_app.answer_adapter import AnswerAdapter
    lock = threading.Lock()

    class ObservedRag(RagService):
        def answer(self, messages, **kwargs):
            observation = {"at": datetime.now(timezone.utc).isoformat(), "messages": deepcopy(messages),
                           "topic_id": kwargs.get("topic_id"), "preferences": deepcopy(kwargs.get("preference_context")),
                           "memory_context": deepcopy(kwargs.get("memory_context")),
                           "memory_projection": deepcopy(kwargs.get("memory_projection"))}
            with lock:
                append_json(directory / "rag_observations.jsonl", observation)
            return super().answer(messages, **kwargs)

    rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    app = create_app(settings=settings, store=store, runtime=runtime, rag_service=rag)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    identity["port"] = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    finished = threading.Event()

    def watch():
        while not finished.wait(0.1):
            if (directory / "stop").exists():
                server.should_exit = True
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    write_json(directory / "address.json", identity)
    try:
        server.run(sockets=[sock])
    finally:
        finished.set()
        watcher.join(timeout=1)
        sock.close()


@contextmanager
def owned_services(settings, scheduler_dsn, budget_path, output, phase):
    """Start and stop only Process objects and nonce-proven children we created."""
    import psutil
    import requests
    paths, children, logs = {}, {}, []
    client = requests.Session()
    client.trust_env = False
    common = dict(os.environ, DATABASE_URL=settings.database_url, OLLAMA_URL=settings.ollama_url,
                  LOCAL_CHAT_MODEL=settings.chat_model, LOCAL_EMBEDDING_MODEL=settings.embedding_model,
                  LOCAL_CONTEXT_TOKENS=str(settings.context_tokens), LOCAL_MAX_OUTPUT_TOKENS=str(settings.max_output_tokens),
                  LOCAL_REQUEST_TIMEOUT=str(settings.request_timeout),
                  LOCAL_LEARNING_OPERATION_SELECTION="0", LOCAL_CHECKED_DIVISION_CONTEXT="0", LOCAL_DEMO_LOGIN="0",
                  TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG="", LOCAL_GENERATION_SCHEDULER_DSN=scheduler_dsn,
                  LOCAL_GENERATION_RUN_BUDGET_PATH=str(budget_path), PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
    identities = {}
    try:
        for role in ("api", "worker"):
            directory = output / f"process-{phase}-{role}"
            directory.mkdir()
            nonce = uuid4().hex
            log = (directory / "process.log").open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child-role", role,
                                        "--child-dir", str(directory)], cwd=ROOT,
                env=dict(common, TEACHCOPILOT_PRODUCT_NONCE=nonce), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            owned = psutil.Process(process.pid)
            children[role] = (process, owned)
            paths[role] = directory
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Owned {role} exited before readiness; see {directory}/process.log")
                try:
                    info = json.loads((directory / "address.json").read_text(encoding="utf-8"))
                    owned_pids = {process.pid, *(child.pid for child in owned.children(recursive=True))}
                    if info["nonce"] != nonce or info["pid"] not in owned_pids or info["role"] != role:
                        raise RuntimeError("Child identity did not match the owned process")
                    if role == "worker" or client.get(f"http://127.0.0.1:{info['port']}/health/live", timeout=0.5).status_code == 200:
                        identities[role] = info
                        break
                except (OSError, ValueError, requests.RequestException):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Owned {role} exceeded finite readiness timeout")
        yield client, f"http://127.0.0.1:{identities['api']['port']}", identities
    finally:
        client.close()
        errors = []
        for directory in paths.values():
            (directory / "stop").touch()
        for role, (process, owned) in children.items():
            try:
                process.wait(timeout=settings.request_timeout + 35)
            except subprocess.TimeoutExpired:
                # Identity objects retain their creation time, preventing PID reuse.
                try:
                    descendants = owned.children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for child in reversed(descendants):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    owned.kill()
                except psutil.NoSuchProcess:
                    pass
                process.wait(timeout=5)
                errors.append(role + " exceeded clean shutdown timeout")
            if process.returncode:
                errors.append(role + " exited with " + str(process.returncode))
        for log in logs:
            log.close()
        if errors and sys.exc_info()[0] is None:
            raise RuntimeError("; ".join(errors))


def copy_curriculum(source, target, topics):
    from psycopg2 import sql
    counts = {}
    with source._connect() as connection, connection.cursor() as origin, target._connect() as destination_connection, destination_connection.cursor() as destination:
        connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
        topic_array = sql.Literal(sorted(set(topics)))
        for table in COPY_TABLES:
            origin.execute("""SELECT column_name FROM information_schema.columns
                WHERE table_schema='teachcopilot' AND table_name=%s AND is_generated='NEVER'
                ORDER BY ordinal_position""", (table,))
            columns = sql.SQL(",").join(sql.Identifier(row["column_name"]) for row in origin.fetchall())
            named = sql.SQL("teachcopilot.{} ({})").format(sql.Identifier(table), columns)
            if table == "catalog_topics":
                query = sql.SQL("""WITH RECURSIVE needed AS (
                    SELECT topic_id,parent_topic_id FROM teachcopilot.catalog_topics WHERE topic_id=ANY({topics})
                    UNION SELECT c.topic_id,c.parent_topic_id FROM teachcopilot.catalog_topics c
                    JOIN needed n ON c.topic_id=n.parent_topic_id)
                    SELECT {columns} FROM teachcopilot.catalog_topics WHERE topic_id IN (SELECT topic_id FROM needed)""").format(topics=topic_array, columns=columns)
            elif table == "embedding_models":
                query = sql.SQL("""SELECT {columns} FROM teachcopilot.embedding_models WHERE model_name IN
                    (SELECT DISTINCT embedding_model FROM teachcopilot.materials WHERE active AND topic_id=ANY({topics}))""").format(topics=topic_array, columns=columns)
            else:
                query = sql.SQL("SELECT {columns} FROM teachcopilot.materials WHERE active AND topic_id=ANY({topics})").format(topics=topic_array, columns=columns)
            buffer = io.StringIO()
            origin.copy_expert(sql.SQL("COPY ({}) TO STDOUT WITH (FORMAT CSV)").format(query), buffer)
            buffer.seek(0)
            destination.copy_expert(sql.SQL("COPY {} FROM STDIN WITH (FORMAT CSV)").format(named), buffer)
            destination.execute(sql.SQL("SELECT count(*) AS n FROM teachcopilot.{}").format(sql.Identifier(table)))
            counts[table] = destination.fetchone()["n"]
    return counts


class Pilot:
    def __init__(self, store, fixture, budget, output, settings, report):
        self.store, self.fixture, self.budget = store, fixture, budget
        self.output, self.settings, self.report = output, settings, report
        self.exchange_index = 0
        self.session_ids, self.guided = {}, {}

    def request(self, method, path, *, headers=None, payload=None, params=None, expected=None, label=None):
        self.exchange_index += 1
        entry = {"label": label, "method": method, "path": path, "request": payload, "params": params,
                 "started_at": datetime.now(timezone.utc).isoformat(), "budget_before": self.budget.snapshot()}
        try:
            response = self.client.request(method, self.base + path, headers=headers, json=payload, params=params,
                                           timeout=self.settings.request_timeout + 20)
            entry["status_code"] = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {"raw_text": response.text}
            entry["response"] = ({**body, "access_token": "[omitted runtime credential]"}
                                 if path == "/auth/login" and "access_token" in body else body)
            if expected is not None and response.status_code not in (expected if isinstance(expected, tuple) else (expected,)):
                raise RuntimeError(f"{label or path}: expected {expected}, received {response.status_code}: {response.text[:600]}")
            return body
        except BaseException as exc:
            entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            if path == "/auth/login" and isinstance(entry.get("request"), dict):
                entry["request"] = {**entry["request"], "password": "[omitted runtime credential]"}
            entry["budget_after"] = self.budget.snapshot()
            write_json(self.output / f"http-{self.exchange_index:04d}.json", entry)

    def bind(self, client, base):
        self.client, self.base = client, base

        def login(email):
            data = self.request("POST", "/auth/login", payload={"email": email, "password": PASSWORD}, expected=200)
            return {"Authorization": "Bearer " + data["access_token"]}

        self.teacher_headers = login("product-teacher@synthetic.test")
        self.headers = {learner["learner_id"]: login(learner["learner_id"] + "@synthetic.test")
                        for learner in self.fixture["learners"]}

    def check(self, name, condition, **details):
        self.report["checks"].append({"name": name, "passed": bool(condition), **details})
        write_json(self.output / "report.partial.json", self.report)
        return bool(condition)

    def wait_job(self, job_id):
        deadline = time.monotonic() + min(600, self.settings.request_timeout * 3 + 30)
        while time.monotonic() < deadline:
            # Polling durable job status never resubmits a generation request.
            job = self.store.get_memory_job(job_id)
            if job and job["status"] not in {"queued", "running"}:
                write_json(self.output / ("job-" + str(job_id) + ".json"), job)
                return job
            if self.budget.snapshot()["deadline_reached"]:
                raise RuntimeError("Finite local package deadline reached while waiting for job")
            time.sleep(0.2)
        raise RuntimeError("Durable memory job exceeded finite wait; outcome retained, no retry")

    def memory(self, learner_id, topic_id):
        return self.request("GET", f"/learners/{learner_id}/memory", headers=self.teacher_headers,
                            params={"topic_id": topic_id}, expected=200)

    def approve(self, learner, session, job_id, *, preferences=True):
        job = self.wait_job(job_id)
        self.check(session["code"] + "-memory-job", job["status"] == "completed", job_id=job_id, status=job["status"])
        if job["status"] != "completed":
            return None
        lid, aliases = learner["learner_id"], self.fixture["review_aliases"]
        snapshot = self.memory(lid, session["topic_id"])
        version = next((v for v in snapshot["versions"] if v.get("job_id") == job_id), None)
        if version is None:
            # Job id is not projected by some store versions. Select the receipt
            # for this exact completed lesson; never approve an unrelated draft.
            sid = self.session_ids.get(session["code"])
            version = next((v for v in snapshot["versions"] if
                ((v.get("context") or {}).get("interest_extraction") or {}).get("learning_session_id") == sid), None)
        if version is None:
            raise RuntimeError("Completed job has no matching memory draft")
        selected = []
        candidates = (version.get("context") or {}).get("interest_candidates", [])
        expected = {(operation, normalized(value, aliases)) for operation, values in
                    (session["review_preferences"] if preferences else {}).items() for value in values}
        actual = {(item["operation"], normalized(item["interest"], aliases)) for item in candidates}
        if preferences:
            self.check(session["code"] + "-interest-candidates", expected <= actual,
                       missing=sorted(expected - actual), extra=sorted(actual - expected), candidates=candidates,
                       synthetic_review=True, human_reviewed=False)
        for item in candidates:
            if (item["operation"], normalized(item["interest"], aliases)) in expected:
                selected.append(item["candidate_id"])
        path = f"/learners/{lid}/memory/versions/{version['version_id']}/confirm"
        payload = {"expected_revision": snapshot["revision"], "interest_candidate_ids": selected}
        if session["code"] == "L01S1":
            self.request("POST", path, headers=self.teacher_headers,
                         payload={**payload, "expected_revision": snapshot["revision"] + 1}, expected=409,
                         label="synthetic_stale_revision")
            self.check("memory-revision-conflict", True)
        approved = self.request("POST", path, headers=self.teacher_headers, payload=payload, expected=200)
        revision = self.store.get_revision(lid)
        replay = self.request("POST", path, headers=self.teacher_headers, payload=payload, expected=200)
        self.check(session["code"] + "-confirmation-idempotent", approved == replay and self.store.get_revision(lid) == revision,
                   synthetic_review=True, human_reviewed=False)
        return version

    def correct_note(self, learner, session):
        lid = learner["learner_id"]
        snapshot = self.memory(lid, session["topic_id"])
        target = next(s for s in snapshot["sources"] if s["text"] == learner["erroneous_teacher_note"])
        payload = {"text": learner["teacher_note_correction"], "expected_revision": snapshot["revision"],
                   "expected_source_revision": snapshot["source_revision"], "client_correction_id": "synthetic-l02-note-correction"}
        path = f"/learners/{lid}/memory/sources/{target['source_id']}/corrections"
        self.request("POST", path, headers=self.teacher_headers,
                     payload={**payload, "expected_source_revision": snapshot["source_revision"] + 1}, expected=409)
        corrected = self.request("POST", path, headers=self.teacher_headers, payload=payload, expected=202)
        from local_app.memory_service import approved_chat_memory
        after = self.store.get_learner_memory(lid, session["topic_id"])
        visible = approved_chat_memory(after, session["topic_id"])
        self.check("withdrawn-note-withheld-before-review", visible.get("withheld_reason") == "source_correction_requires_review")
        replay = self.request("POST", path, headers=self.teacher_headers, payload=payload, expected=202)
        self.check("source-correction-idempotent", replay["job_id"] == corrected["job_id"])
        # This explicit synthetic correction review precedes the second lesson.
        self.approve(learner, {**session, "code": "L02-correction"}, corrected["job_id"], preferences=False)
        final = self.store.get_learner_memory(lid, session["topic_id"])
        sources = {s["source_id"]: s for s in final["sources"]}
        self.check("source-correction-append-only", sources[target["source_id"]]["text"] == learner["erroneous_teacher_note"]
                   and any(s.get("corrects_source_id") == target["source_id"] for s in sources.values()))

    def session(self, learner, session):
        lid, code, topic = learner["learner_id"], session["code"], session["topic_id"]
        row = {"code": code, "topic_title": session["topic_title"], "status": "running", "events": [],
               "synthetic_teacher_review": True, "human_reviewed": False}
        self.report["sessions"].append(row)
        if session.get("correct_teacher_note"):
            self.correct_note(learner, session)
        created = self.request("POST", f"/learners/{lid}/learning-sessions", headers=self.headers[lid],
                payload={"topic_id": topic, "client_session_id": code}, expected=201)
        sid = created["learning_session_id"]
        self.session_ids[code] = sid
        row["learning_session_id"] = sid
        self.check(code + "-readable-topic", created.get("topic_title") != topic and bool(created.get("topic_title")),
                   topic_title=created.get("topic_title"))
        if code == "L01S2":
            context = self.request("GET", f"/learners/{lid}/guided-division", headers=self.headers[lid], params={"topic_id": topic}, expected=200)
            self.check("reviewed-example-survives-restart", context.get("origin") == "approved_memory" and
                       context.get("resume_required") is True and context.get("guided_division") == self.guided.get("L01S1"), context=context)
        message_index = 0
        for index, event in enumerate(session["events"], 1):
            mid = f"{code}-E{index:02d}"
            before = self.budget.snapshot()
            if before["deadline_reached"] or before["remaining_generations"] <= 0:
                raise RuntimeError("Finite shared local generation budget exhausted")
            payload = {"learning_session_id": sid, "client_message_id": mid, "metadata": {"subject_id": session["subject_id"]}}
            saved = self.store.guided_division_context(lid, topic, self.student_ids[lid], learning_session_id=sid)
            if event["kind"] == "message":
                payload["messages"] = [{"role": "user", "content": event["text"]}]
                message_index += 1
            else:
                action = {key: value for key, value in event.items() if key != "kind"}
                if action["operation"] != "division_guided_start":
                    state = saved.get("guided_division")
                    if not state:
                        raise RuntimeError("No exact persisted guided state for preregistered action")
                    action["expected_state_id"] = state["state_id"]
                    if action["operation"] == "division_guided_resume":
                        action["source_learning_session_id"] = saved["source_learning_session_id"]
                payload["learning_action"] = action
            try:
                response = self.request("POST", "/v1/chat/completions", headers=self.headers[lid], payload=payload, label=mid)
                good = isinstance(response, dict) and bool(response.get("choices"))
                self.check(mid + "-completed-response", good)
                after = self.store.guided_division_context(lid, topic, self.student_ids[lid], learning_session_id=sid)
                if event["kind"] == "message":
                    self.check(mid + "-free-chat-does-not-advance", saved.get("guided_division") == after.get("guided_division"))
                elif good:
                    self.check(mid + "-explicit-action-no-generation", self.budget.snapshot()["generation_calls"] == before["generation_calls"])
                    self.check(mid + "-unknown-help", response.get("guided_assessment", {}).get("independent_work") == "unknown"
                               and response.get("guided_assessment", {}).get("competence_assessed") is False)
                row["events"].append({"id": mid, "kind": event["kind"], "response_completed": good,
                                      "response": response, "budget_before": before, "budget_after": self.budget.snapshot()})
                if code == "L01S1" and message_index == 1 and event["kind"] == "message" and good:
                    calls = self.budget.snapshot()["generation_calls"]
                    replay = self.request("POST", "/v1/chat/completions", headers=self.headers[lid], payload=payload, expected=200, label=mid + "-cached-replay")
                    self.check("completed-message-replay-no-generation", replay == response and self.budget.snapshot()["generation_calls"] == calls)
            except Exception as exc:
                # Preserve the first error; a later event is a new preregistered
                # utterance, never a repeated paid/local generation request.
                row["events"].append({"id": mid, "kind": event["kind"], "error": {"type": type(exc).__name__, "message": str(exc)}})
                self.check(mid + "-completed-response", False, error=str(exc))
                if event["kind"] == "action":
                    raise
            if self.budget.snapshot()["deadline_reached"]:
                raise RuntimeError("Finite local package deadline reached")
        final = self.store.guided_division_context(lid, topic, self.student_ids[lid], learning_session_id=sid)
        if "expected_guided" in session:
            state = final.get("guided_division") or {}
            self.guided[code] = state
            self.check(code + "-exact-guided-state", all(state.get(key) == value for key, value in session["expected_guided"].items()), state=state)
        if code == "L02S1":
            source = self.store.add_memory_source(lid, topic, learner["erroneous_teacher_note"], self.teacher_id,
                                                 approved=True, source_key="synthetic-preregistered-erroneous-note")
            write_json(self.output / "synthetic-erroneous-teacher-note.json", source)
        completed = self.request("POST", f"/learning-sessions/{sid}/complete", headers=self.headers[lid], expected=202)
        replay = self.request("POST", f"/learning-sessions/{sid}/complete", headers=self.headers[lid], expected=202)
        self.check(code + "-completion-idempotent", completed["job_id"] == replay["job_id"])
        self.approve(learner, session, completed["job_id"])
        from local_app.conversation_memory import preference_projection
        card = self.store.get_card(lid)
        pref = preference_projection(card)
        aliases = self.fixture["review_aliases"]
        self.check(code + "-interests", {normalized(x, aliases) for x in card["interests"]} == {normalized(x, aliases) for x in session["expected_final_interests"]}, actual=card["interests"])
        self.check(code + "-avoided-stories", {normalized(x, aliases) for x in pref["avoided_interests"]} == {normalized(x, aliases) for x in session["expected_final_avoided"]}, actual=pref["avoided_interests"])
        self.check(code + "-knowledge-unchanged", card.get("knowledge") == [])
        if session.get("recommendation"):
            prepared = self.request("POST", f"/learners/{lid}/next-step", headers=self.teacher_headers,
                payload={"topic_id": topic, "method": "tail", "expected_revision": self.store.get_revision(lid)}, expected=202)
            job = self.wait_job(prepared["job_id"])
            row["recommendation"] = job
            self.check(code + "-recommendation-technical", job["status"] == "completed", status=job["status"],
                       content_review="pending_separate_agent_review_not_human")
        row["status"] = "attempted"
        write_json(self.output / (code + "-session.json"), row)
        write_json(self.output / "report.partial.json", self.report)
        print(code + ": attempted", flush=True)

    def isolation_and_outcomes(self):
        """Deterministic synthetic unresolved states, with no model requests."""
        learners = self.fixture["learners"]
        for index, learner in enumerate(learners):
            lid, foreign = learner["learner_id"], learners[1-index]["learner_id"]
            denied = self.request("GET", f"/learners/{foreign}/dashboard", headers=self.headers[lid], expected=403)
            self.request("GET", f"/learners/{lid}/memory", headers=self.headers[lid], expected=403)
            self.check(lid + "-learner-isolation", True)
        lid, topic = learners[0]["learner_id"], learners[0]["sessions"][0]["topic_id"]
        actor = self.student_ids[lid]
        for outcome in ("pending", "unknown_outcome"):
            sid = self.store.create_learning_session(lid, actor, topic, client_session_id="synthetic-" + outcome)["learning_session_id"]
            text = "Синтетическая техническая проверка неизвестного исхода; не учебное сообщение."
            self.store.reserve_chat_message(sid, lid, "synthetic-unresolved", text, actor)
            if outcome == "unknown_outcome":
                self.store.fail_chat_message(sid, "synthetic-unresolved", "synthetic explicit unknown outcome", unknown_outcome=True)
            before = self.budget.snapshot()["generation_calls"]
            body = {"learning_session_id": sid, "client_message_id": "synthetic-unresolved",
                    "metadata": {"subject_id": "math"}, "messages": [{"role": "user", "content": text}]}
            self.request("POST", "/v1/chat/completions", headers=self.headers[lid], payload=body, expected=409)
            self.check(outcome + "-does-not-repeat-generation", self.budget.snapshot()["generation_calls"] == before,
                       synthetic_state_injection=True)
            if outcome == "pending":
                self.store.fail_chat_message(sid, "synthetic-unresolved", "synthetic fixture closed", unknown_outcome=False)
        # These state fixtures remain active and unreviewed. They never enter
        # the six product lessons or their confirmed memory.

    def verify_observations(self):
        for path in sorted(self.output.glob("process-*-api/rag_observations.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                context = row.get("memory_context") or {}
                sources = context.get("sources") or []
                source_ids = {s.get("source_id") for s in sources}
                matching = []
                for learner in self.fixture["learners"]:
                    own = self.store.get_learner_memory(learner["learner_id"], row.get("topic_id"))
                    if source_ids <= {s["source_id"] for s in own["sources"]}:
                        matching.append(learner["learner_id"])
                self.check("observed-memory-source-scope", not source_ids or len(matching) == 1,
                           topic_id=row.get("topic_id"), source_ids=sorted(source_ids), matching_learners=matching)


def run(args):
    fixture, counts = load_fixture()
    regression, previous = None, None
    if args.regression_from:
        regression, counts = load_regression_fixture()
        previous = json.loads((args.regression_from.resolve() / "report.json").read_text(encoding="utf-8"))
        if not previous.get("retained_synthetic_database") or previous.get("temporary_database_removed"):
            raise ValueError("Regression requires the retained original synthetic database")
    dry = {"status": "dry_run", "data_origin": "synthetic", "human_reviewed": False,
           "counts": counts, "limits": fixture["limits"], "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
           "sequence": [l["sessions"][index]["code"] for index in range(3) for l in fixture["learners"]],
           "real_restart_after": ["L01S1", "L02S1"], "paid_provider_calls": 0,
           "execute_required": True, "database_created": False, "model_calls": 0}
    if regression:
        dry.update(sequence=[s["code"] for s in regression["sessions"]], independent_experiment=False,
                   max_engineering_iterations=1, dependent_on=str(args.regression_from.resolve()),
                   regression_fixture_sha256=hashlib.sha256(REGRESSION_FIXTURE.read_bytes()).hexdigest())
    if not args.execute:
        return dry

    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from local_app.auth import hash_password
    from local_app.config import Settings
    from local_app.generation_run_budget import GenerationRunBudget
    from local_app.runtime import OllamaRuntime
    from local_app.rag import RagService
    from local_app.store import Store
    from scripts.verify_memory import require_local_dsn

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = Settings.from_env()
    require_local_dsn(baseline.database_url)
    if baseline.chat_model != "gemma4:12b":
        raise ValueError("This preregistered product package requires gemma4:12b")
    database = require_owned_database(previous["retained_synthetic_database"] if previous else DATABASE_PREFIX + uuid4().hex[:12])
    budget_path = (Path(previous["budget_ledger"]) if previous else args.budget_ledger or output / "local-generation-budget.sqlite3").resolve()
    if previous and args.budget_ledger and args.budget_ledger.resolve() != budget_path:
        raise ValueError("Regression must use the original persistent generation journal")
    if previous:
        # One durable attempt marker is consumed before any generation or
        # schema operation. A failed regression is never silently repeated.
        with (args.regression_from.resolve() / "REGRESSION_ATTEMPTED.json").open("x", encoding="utf-8") as marker:
            json.dump({"output": str(output), "created_at": datetime.now(timezone.utc).isoformat()}, marker)
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    budget = GenerationRunBudget(budget_path)
    budget.initialize(max_calls=160, max_seconds=5400)
    settings = replace(baseline, database_url=make_dsn(baseline.database_url, dbname=database), request_timeout=180,
                       generation_scheduler_dsn=baseline.database_url, generation_run_budget_path=str(budget_path),
                       learning_operation_selection_enabled=False, checked_division_context_enabled=False, demo_login_enabled=False)
    report = {**dry, "status": "running", "execute_required": False, "checks": [], "sessions": [], "processes": [],
              "database_name": database, "budget_ledger": str(budget_path), "preflight_only": args.preflight_only,
              "content_review": "pending_separate_agent_review_not_human_expertise", "throughput_20_users_measured": False}
    admin = psycopg2.connect(make_dsn(baseline.database_url, dbname="postgres"))
    admin.autocommit = True
    created = False
    started = time.monotonic()
    try:
        if not previous:
            with admin.cursor() as cursor:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        created = True
        report["database_created"] = not bool(previous)
        report["reused_original_synthetic_database"] = bool(previous)
        store = Store(settings.database_url)
        if not previous:
            store.migrate()
        else:
            store.ping()
        topics = sorted({s["topic_id"] for learner in fixture["learners"] for s in learner["sessions"]})
        report["public_tables_copied"] = (previous["public_tables_copied"] if previous else
                                         copy_curriculum(Store(baseline.database_url), store, topics))
        report["curriculum_copy_filter"] = {"materials": {"active": True, "topic_id_in": topics},
            "catalog_topics": "three exact topics and their required parent chain",
            "embedding_models": "only models referenced by copied active material rows",
            "source_transaction": "read_only_repeatable_read", "personal_tables_copied": []}
        teacher = (store.get_account(email="product-teacher@synthetic.test") if previous else
                   store.create_account("product-teacher@synthetic.test", "SYNTHETIC product teacher", hash_password(PASSWORD), "teacher"))
        students = {}
        for learner in fixture["learners"]:
            lid = learner["learner_id"]
            if previous:
                students[lid] = store.get_account(email=lid + "@synthetic.test")
            else:
                store.save_card(make_card(learner), teacher["id"])
                store.assign_learner(teacher["id"], lid)
                students[lid] = store.create_student_account(lid + "@synthetic.test", "SYNTHETIC student", hash_password(PASSWORD), lid, teacher["id"])
        runtime = OllamaRuntime(settings)
        model_state = runtime.check_models()
        index_state = RagService(store, runtime, settings).status()
        if not index_state.get("ready"):
            raise RuntimeError("Copied curriculum embedding index is not ready")
        tracked = [Path(__file__), FIXTURE, REGRESSION_FIXTURE, *sorted((ROOT / "local_app").glob("*.py")),
                   *sorted((ROOT / "local_app/prompts").glob("*.txt"))]
        prereg = {"created_at": datetime.now(timezone.utc).isoformat(), "fixture": regression or fixture,
                  "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
                  "models": model_state, "manifest": runtime.manifest(), "index_state": index_state,
                  "database_name": database, "public_tables_copied": report["public_tables_copied"],
                  "curriculum_copy_filter": report["curriculum_copy_filter"],
                  "scheduler": "working_database_generation_admission_only", "budget": budget.snapshot(),
                  "external_answers_enabled": False, "human_reviewed": False,
                  "sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}}
        write_json(output / "preregistered.json", prereg)
        pilot = Pilot(store, fixture, budget, output, settings, report)
        pilot.teacher_id = teacher["id"]
        pilot.student_ids = {lid: account["id"] for lid, account in students.items()}
        before_restart = {}
        for phase in ((3,) if previous else (1, 2)):
            with owned_services(settings, baseline.database_url, budget_path, output, phase) as (client, base, identities):
                report["processes"].append({"phase": phase, **identities})
                pilot.bind(client, base)
                if previous:
                    for session in regression["sessions"]:
                        learner = next(l for l in fixture["learners"] if l["code"] == session["learner_code"])
                        pilot.session(learner, session)
                elif phase == 1:
                    pilot.isolation_and_outcomes()
                    if not args.preflight_only:
                        for learner in fixture["learners"]:
                            pilot.session(learner, learner["sessions"][0])
                    for learner in fixture["learners"]:
                        lid = learner["learner_id"]
                        before_restart[lid] = {"card": store.get_card(lid), "memory": store.get_learner_memory(lid),
                            "current": pilot.request("GET", f"/learners/{lid}/learning-sessions/current", headers=pilot.headers[lid], expected=200)}
                    write_json(output / "phase-1-summary.json", {"sessions": report["sessions"], "budget": budget.snapshot(),
                               "checks": report["checks"], "restart_pending": True})
                else:
                    first = report["processes"][0]
                    pilot.check("real-api-and-worker-restart", all(first[role]["pid"] != identities[role]["pid"] for role in ("api", "worker")))
                    for learner in fixture["learners"]:
                        lid = learner["learner_id"]
                        current = pilot.request("GET", f"/learners/{lid}/learning-sessions/current", headers=pilot.headers[lid], expected=200)
                        pilot.check(lid + "-durable-state-after-restart", before_restart[lid] == {
                            "card": store.get_card(lid), "memory": store.get_learner_memory(lid), "current": current})
                    if not args.preflight_only:
                        for session_index in (1, 2):
                            for learner in fixture["learners"]:
                                pilot.session(learner, learner["sessions"][session_index])
                    for learner in fixture["learners"]:
                        lid = learner["learner_id"]
                        archive = pilot.request("GET", f"/learners/{lid}/learning-sessions", headers=pilot.headers[lid], expected=200)
                        pilot.check(lid + "-archive-sessions", len(archive["sessions"]) == (0 if args.preflight_only else 3))
                        for session in archive["sessions"]:
                            history = pilot.request("GET", f"/learners/{lid}/learning-sessions/{session['learning_session_id']}/history", headers=pilot.headers[lid], expected=200)
                            pilot.check(lid + "-archive-history", bool(history["messages"]) and history["status"] == "completed")
                write_json(output / "report.partial.json", report)
            # Both Popen objects have exited before the second phase begins.
        pilot.verify_observations()
        report["status"] = ("preflight_passed" if args.preflight_only else "completed_with_findings")
        report["all_mechanical_checks_passed"] = all(c["passed"] for c in report["checks"])
        report["initial_failures_preserved"] = [c for c in report["checks"] if not c["passed"]]
        write_json(output / "database-snapshot.json", {l["learner_id"]: {"card": store.get_card(l["learner_id"]),
                   "memory": store.get_learner_memory(l["learner_id"])} for l in fixture["learners"]})
    except Exception as exc:
        report["status"] = "failed"
        report["fatal_error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if created and not args.keep_database:
            require_owned_database(database)
            with admin.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
                cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
            report["temporary_database_removed"] = True
        else:
            report["temporary_database_removed"] = not created
            report["retained_synthetic_database"] = database if created else None
        admin.close()
        report["budget"] = budget.snapshot()
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "report.json", report)
        write_json(output / "PROCESSING.json", {"status": report["status"], "budget": report["budget"], "paid_provider_calls": 0})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Real DB/process checks without generation requests")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budget-ledger", type=Path, help="One persistent ledger for pilot, regression and integration")
    parser.add_argument("--regression-from", type=Path, help="One dependent iteration using the original output's retained synthetic DB and budget")
    parser.add_argument("--keep-database", action="store_true", help="Retain only this isolated synthetic database for follow-up UI checks")
    parser.add_argument("--child-role", choices=("api", "worker"), help=argparse.SUPPRESS)
    parser.add_argument("--child-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child_role:
        if not args.child_dir:
            parser.error("Child directory required")
        child_process(args.child_role, args.child_dir)
        return 0
    if args.execute and not args.output_dir:
        parser.error("--execute requires a new --output-dir")
    if args.regression_from and args.preflight_only:
        parser.error("A regression attempt is finite and cannot be used as infrastructure preflight")
    result = run(args)
    print(json.dumps({key: result.get(key) for key in ("status", "counts", "limits", "budget", "fatal_error")}, ensure_ascii=False), flush=True)
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
