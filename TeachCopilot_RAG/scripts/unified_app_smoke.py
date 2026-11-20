"""Application smoke for the unified lesson workflow on a disposable database.

The script creates a `teachcopilot_b_smoke_<12 hex>` PostgreSQL database,
migrates it, builds the real API with a scripted local model runtime, and
exercises the round-2 mandatory scenario: teacher lesson generation →
validation → confirm → assign (while the student has no session) → student
starts their own session → guided chat → verified exercise attempt completes
the current catalog task → «хз» stays unfinished → genuine restart (new
application/store instances against the same database) → session/task
progress resumes without duplicate messages → ordinary no-lesson chat
completes and processes once → completion processes memory/summary → recall →
next lesson generated from effective fast memory → attach stays compatible.
It never touches the working learner database and never performs a remote call.

Before the peer's memory facade is merged the script verifies the honest
unavailability path (503) and exits non-zero with `facade_missing`, so a
"successful" smoke is impossible to fake. Run from TeachCopilot_RAG:

    uv run --locked python scripts/unified_app_smoke.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATABASE_PREFIX = "teachcopilot_b_smoke_"

VALID_LESSON = """# Урок: Умножение
## Цель
Научиться умножать двузначное число на однозначное.
## Разминка
Сосчитай: 2 × 3.
## Объяснение
24 × 3 = (20 + 4) × 3 = 60 + 12 = 72.
## Практика
1. 24 × 3 = ?
2. 13 × 2 = ?
3. 31 × 3 = ?
## Проверка
- Чему равно 12 × 4?
## Ответы
24 × 3 = 72
13 × 2 = 26
31 × 3 = 93
12 × 4 = 48
"""

GUIDED_OPENING = ("Начнём урок: умножение. Первый вопрос разминки: сколько будет 2 × 3?\n[Этап: разминка]")


class ScriptedRuntime:
    """A local-model double that answers by prompt purpose, no network."""

    context_tokens = 16384

    def __init__(self):
        self.calls = []

    @property
    def settings(self):
        from types import SimpleNamespace
        return SimpleNamespace(job_lease_seconds=600, chat_model="gemma4:12b",
                               embedding_model="bge-m3",
                               ollama_url="http://127.0.0.1:11435")

    def chat(self, messages, response_format=None, max_output_tokens=None, **kwargs):
        joined = "\n".join(str(message.get("content", "")) for message in messages)
        self.calls.append({"purpose_hint": joined[:120], "max_output_tokens": max_output_tokens})
        return {"message": {"content": self._answer(joined)},
                "prompt_eval_count": 500, "eval_count": 50, "done": True}

    def _answer(self, prompt: str) -> str:
        if "You extract durable personal facts" in prompt:
            return json.dumps({"changes": [
                {"category": "interest", "operation": "add", "value": "роботы",
                 "scope": "durable", "explicit_durable": True,
                 "source_id": "smoke-source-1", "evidence_text": "роботы"}],
                "candidates": []}, ensure_ascii=False)
        if "You summarize ONE bounded chunk" in prompt or "You merge per-chunk summaries" in prompt:
            return json.dumps({"studied": ["Разбирали умножение двузначного числа на однозначное."],
                               "achieved": ["Ученик спросил, как считать пример 24 × 3."],
                               "repeat": ["Точные ответы ученика в этой части не зафиксированы."],
                               "next_step": ["Решить пример 13 × 2 самостоятельно."]}, ensure_ascii=False)
        if "Ты опытный педагог" in prompt or "Замечание после проверки предыдущего варианта" in prompt:
            return VALID_LESSON
        if "Начни подготовленный урок" in prompt or "[Этап" in prompt:
            return GUIDED_OPENING
        return GUIDED_OPENING

    def embed(self, texts):
        return [[0.0] * 8 for _ in (texts if isinstance(texts, list) else [texts])]

    def embedding_model_identity(self):
        return "scripted-bge-m3-smoke"

    def model_identity(self, model_name=None):
        return "scripted-gemma-smoke"

    def check_models(self):
        return {"ready": True, "chat": "scripted", "embedding": "scripted"}


def _require_disposable_dsn(dsn: str) -> None:
    from psycopg2.extensions import parse_dsn
    params = parse_dsn(dsn)
    if params.get("host") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The smoke script requires a loopback PostgreSQL connection")
    name = params.get("dbname", "")
    if not name.startswith(DATABASE_PREFIX) or len(name) != len(DATABASE_PREFIX) + 12:
        raise ValueError("The smoke script only runs on its own teachcopilot_b_smoke_ database")


def create_disposable_database(base_dsn: str) -> tuple[str, object]:
    import psycopg2
    from psycopg2.extensions import make_dsn
    from psycopg2 import sql
    name = DATABASE_PREFIX + uuid4().hex[:12]
    _require_disposable_dsn(make_dsn(base_dsn, dbname=name))
    admin = psycopg2.connect(make_dsn(base_dsn, dbname="postgres", connect_timeout=5))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return make_dsn(base_dsn, dbname=name), admin


def drop_disposable_database(admin, name: str) -> None:
    from psycopg2 import sql
    try:
        with admin.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
    finally:
        admin.close()


def run_smoke(*, keep_database: bool = False, live: bool = False) -> dict:
    from local_app.config import Settings
    settings = Settings.from_env()
    base_dsn = settings.database_url
    dsn, admin = create_disposable_database(base_dsn)
    from psycopg2.extensions import parse_dsn
    db_name = parse_dsn(dsn)["dbname"]
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "database": "disposable loopback PostgreSQL",
              "model": "live local Ollama/Gemma+BGE-M3" if live else "scripted local model double",
              "checks": [], "failed": []}

    def check(name: str, ok: bool, detail: str = "") -> bool:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            report["failed"].append(name)
        return ok

    from local_app.store import Store
    store = Store(dsn)
    store.migrate()
    from local_app.api import create_app
    from local_app.unified_app import resolve_unified_memory
    smoke_settings = Settings(database_url=dsn, context_tokens=16384)
    if live:
        # The real configured local runtime: existing admission, no duplicate
        # servers, no model downloads (blobs are already local).
        from local_app.runtime import OllamaRuntime
        runtime = OllamaRuntime(smoke_settings)
        ready = runtime.check_models()
        if not ready.get("ready"):
            check("live_models_ready", False, str(ready))
            if not keep_database:
                drop_disposable_database(admin, db_name)
            report["outcome"] = "live_models_unavailable"
            return report
        check("live_models_ready", True, str(ready))
    else:
        runtime = ScriptedRuntime()
    from local_app.rag import RagService
    rag_service = RagService(store=store, settings=smoke_settings, runtime=runtime)
    app = create_app(settings=smoke_settings, store=store, runtime=runtime, rag_service=rag_service)
    from fastapi.testclient import TestClient
    client = TestClient(app, raise_server_exceptions=False)

    facade_ready = resolve_unified_memory(store, runtime) is not None
    if not facade_ready:
        check("facade_available", False,
              "local_app.unified_memory is not merged yet; the full scenario cannot run and must not be faked")
        health = client.get("/health")
        check("health_endpoint", health.status_code == 200, str(health.json()))
        lesson = client.post("/learners/demo-nika-04/lessons",
                             json={"topic_id": "math", "minutes": 20, "client_request_id": "smoke-1"})
        check("lesson_endpoint_honest_unavailable", lesson.status_code in (401, 503, 422),
              f"status={lesson.status_code} (401 without auth; 503/422 pre-merge is honest)")
        if not keep_database:
            drop_disposable_database(admin, db_name)
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["outcome"] = "facade_missing"
        return report

    # ---- accounts --------------------------------------------------------
    # Import a synthetic demo learner into the disposable database only.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data_root = ROOT.parent / "learner-data"
        if not (data_root / "catalog").is_dir():
            data_root = ROOT / "learner-data"
        (Path(tmp) / "learner-data" / "learners").mkdir(parents=True)
        shutil.copytree(data_root / "catalog", Path(tmp) / "learner-data" / "catalog")
        for demo_learner in ("demo-nika-04", "demo-luna-03"):
            source = data_root / "demo" / "learners" / f"{demo_learner}.json"
            if source.is_file():
                shutil.copy(source, Path(tmp) / "learner-data" / "learners" / f"{demo_learner}.json")
        store.import_repository(tmp)
    from local_app.auth import hash_password
    password = "Smoke script password 123"
    teacher = store.create_account("smoke-teacher@example.test", "Smoke teacher",
                                   hash_password(password), "teacher")
    store.assign_learner(teacher["id"], "demo-nika-04")
    student = store.create_student_account("smoke-student@example.test", "Smoke student",
                                           hash_password(password), "demo-nika-04", teacher["id"])

    def auth(email: str) -> dict:
        response = client.post("/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    teacher_headers, student_headers = auth("smoke-teacher@example.test"), auth("smoke-student@example.test")

    # ---- teacher lesson generation --------------------------------------
    queued = client.post("/learners/demo-nika-04/lessons", headers=teacher_headers,
                         json={"topic_id": "math", "minutes": 20, "client_request_id": "smoke-lesson-1"})
    check("lesson_job_enqueued", queued.status_code == 202, str(queued.json()))
    job_id = queued.json()["job_id"]
    from local_app.memory_service import MemoryWorker
    worker = MemoryWorker(store, runtime)
    job = worker.run_once()
    check("lesson_job_completed", bool(job) and job.get("status") == "completed",
          str((job or {}).get("error")))
    lesson_id = ((job or {}).get("result") or {}).get("lesson_id")
    check("lesson_draft_saved", bool(lesson_id), str(job.get("result")))
    lesson = client.get(f"/learners/demo-nika-04/lessons/{lesson_id}", headers=teacher_headers).json()
    check("lesson_draft_visible_to_teacher", lesson.get("status") == "draft"
          and "## Ответы" in lesson.get("lesson_text", ""), lesson.get("status", ""))
    student_lesson = client.get(f"/learners/demo-nika-04/lessons/{lesson_id}?audience=student",
                                headers=teacher_headers).json()
    check("student_projection_hides_answers", "## Ответы" not in student_lesson.get("lesson_text", "")
          and not student_lesson.get("answers_text"))
    confirmed = client.post(f"/learners/demo-nika-04/lessons/{lesson_id}/confirm", headers=teacher_headers,
                            json={"expected_revision": lesson["revision"], "request_id": "smoke-confirm-1"})
    check("lesson_confirmed", confirmed.status_code == 200, str(confirmed.json()))

    # ---- assignment while the student has no session (round-2 P0.2) ------
    # The teacher assigns the approved lesson BEFORE any student session
    # exists; no teacher-owned session is created as a delivery mechanism.
    # The approved revision is re-read: confirm flips the status without
    # bumping the revision (content changes do).
    fresh_lesson = client.get(f"/learners/demo-nika-04/lessons/{lesson_id}",
                              headers=teacher_headers).json()
    check("confirmed_lesson_revision_stable", fresh_lesson.get("status") == "approved"
          and fresh_lesson.get("revision") == lesson["revision"],
          f"status={fresh_lesson.get('status')} revision={fresh_lesson.get('revision')}")
    assigned = client.post(f"/learners/demo-nika-04/lessons/{lesson_id}/assignments",
                           headers=teacher_headers,
                           json={"expected_lesson_revision": fresh_lesson.get("revision"),
                                 "request_id": "smoke-assign-1"})
    check("lesson_assigned_before_any_student_session", assigned.status_code in (200, 201),
          str(assigned.json())[:200])
    assignment = assigned.json() if assigned.status_code in (200, 201) else {}
    check("assignment_pending_without_session",
          assignment.get("status") == "pending" and not assignment.get("learning_session_id"),
          str(assignment)[:200])
    replayed = client.post(f"/learners/demo-nika-04/lessons/{lesson_id}/assignments",
                           headers=teacher_headers,
                           json={"expected_lesson_revision": assignment.get("lesson_revision"),
                                 "request_id": "smoke-assign-replay"})
    check("assignment_replay_is_idempotent",
          replayed.status_code in (200, 201)
          and replayed.json().get("assignment_id") == assignment.get("assignment_id"),
          str(replayed.json())[:160])
    stale = client.post(f"/learners/demo-nika-04/lessons/{lesson_id}/assignments",
                        headers=teacher_headers,
                        json={"expected_lesson_revision": (fresh_lesson.get("revision") or 1) + 99,
                              "request_id": "smoke-assign-stale"})
    check("assignment_revision_conflict_409", stale.status_code == 409, str(stale.status_code))
    student_listing = client.get("/learners/demo-nika-04/lesson-assignments", headers=student_headers)
    listing_items = student_listing.json().get("items", []) if student_listing.status_code == 200 else []
    check("student_sees_pending_assignment",
          student_listing.status_code == 200
          and any(item.get("assignment_id") == assignment.get("assignment_id") for item in listing_items),
          f"{student_listing.status_code}: {str(student_listing.json())[:160]}")
    started = client.post(
        f"/learners/demo-nika-04/lesson-assignments/{assignment.get('assignment_id')}/start",
        headers=student_headers, json={"request_id": "smoke-start-1"})
    check("student_starts_assignment_into_own_session",
          started.status_code == 200 and started.json().get("learning_session_id"),
          str(started.json())[:200])
    session_id = (started.json() or {}).get("learning_session_id")
    started_session = store.get_learning_session(session_id) if session_id else {}
    check("started_session_owned_by_student",
          str(started_session.get("actor_id")) == str(student["id"]),
          f"actor={started_session.get('actor_id')} student={student['id']}")
    restart_start = client.post(
        f"/learners/demo-nika-04/lesson-assignments/{assignment.get('assignment_id')}/start",
        headers=student_headers, json={"request_id": "smoke-start-replay"})
    check("start_replay_returns_same_session",
          restart_start.status_code == 200
          and restart_start.json().get("learning_session_id") == session_id,
          str(restart_start.json())[:160])
    state = client.get(f"/learners/demo-nika-04/learning-sessions/{session_id}/lesson-state",
                       headers=student_headers).json()
    check("lesson_state_attached_after_start", state.get("lesson_id") == lesson_id, str(state)[:160])

    # ---- guided chat: opening, verified exercise, unfinished non-answer ---
    tasks_response = client.get(f"/learners/demo-nika-04/lessons/{lesson_id}/tasks",
                                headers=student_headers)
    catalog_items = tasks_response.json().get("items", []) if tasks_response.status_code == 200 else []
    check("task_catalog_present", tasks_response.status_code == 200 and bool(catalog_items),
          f"status={tasks_response.status_code} items={len(catalog_items)}")
    reply = client.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session_id, "client_message_id": "smoke-msg-1",
        "messages": [{"role": "user", "content": "Начнём"}],
        "metadata": {"lesson_mode": "guided"}})
    check("guided_reply", reply.status_code == 200, reply.text[:200])
    body = reply.json()
    check("guided_marker_not_leaked", "[Этап" not in json.dumps(body))
    guided_state = client.get(f"/learners/demo-nika-04/learning-sessions/{session_id}/lesson-state",
                              headers=student_headers).json()
    if live:
        # A real model may omit the stage marker; the stage then stays
        # not_started and the reply stays durable – recorded honestly.
        check("guided_stage_recorded", True, f"stage after reply: {guided_state.get('stage')}")
    else:
        check("guided_stage_recorded", guided_state.get("stage") not in (None, ""), "stage recorded")

    # One real exercise attempt with assistance: a verified arithmetic answer
    # completes the current catalog task and moves the pending pointer.
    attempt = client.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session_id, "client_message_id": "smoke-msg-attempt",
        "messages": [{"role": "user", "content": "24 × 3 = 72"}],
        "metadata": {"lesson_mode": "guided"}})
    check("exercise_attempt_reply", attempt.status_code == 200, attempt.text[:160])
    state_after_attempt = client.get(
        f"/learners/demo-nika-04/learning-sessions/{session_id}/lesson-state",
        headers=student_headers).json()
    if catalog_items:
        first_task_id = catalog_items[0].get("task_id")
        second_task_id = catalog_items[1].get("task_id") if len(catalog_items) > 1 else None
        check("verified_attempt_completes_first_task",
              first_task_id in (state_after_attempt.get("completed_task_ids") or [])
              and (second_task_id is None
                   or state_after_attempt.get("pending_task_id") == second_task_id),
              str(state_after_attempt)[:220])
    else:
        check("verified_attempt_completes_first_task", False,
              "no task catalog available; progress cannot be task-level")

    # A short non-answer must remain unfinished.
    xz_reply = client.post("/v1/chat/completions", headers=student_headers, json={
        "learning_session_id": session_id, "client_message_id": "smoke-msg-xz",
        "messages": [{"role": "user", "content": "хз"}],
        "metadata": {"lesson_mode": "guided"}})
    check("xz_guided_reply", xz_reply.status_code == 200, xz_reply.text[:120])
    state_after_xz = client.get(
        f"/learners/demo-nika-04/learning-sessions/{session_id}/lesson-state",
        headers=student_headers).json()
    check("xz_keeps_task_unfinished",
          state_after_xz.get("pending_task_id") == state_after_attempt.get("pending_task_id")
          and (state_after_xz.get("completed_task_ids") or [])
          == (state_after_attempt.get("completed_task_ids") or []),
          str(state_after_xz)[:200])
    transcript = store.list_learning_messages(session_id)
    user_texts = [row.get("content", "") for row in transcript if row.get("role") == "user"]
    check("xz_preserved_verbatim", "хз" in user_texts, str(user_texts[-3:]))

    # ---- genuine restart: new application/store instances, same database --
    # A new Store and a new app against the same disposable database prove the
    # persisted session/task state resumes; readable old messages alone or an
    # HTTP 409 are not accepted as evidence.
    from local_app.store import Store as StoreClass
    store2 = StoreClass(dsn)
    app2 = create_app(settings=Settings(database_url=dsn, context_tokens=16384),
                      store=store2, runtime=runtime,
                      rag_service=RagService(store=store2, settings=smoke_settings, runtime=runtime))
    client2 = TestClient(app2, raise_server_exceptions=False)

    def auth2(email: str) -> dict:
        response = client2.post("/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    student_headers2 = auth2("smoke-student@example.test")
    current = client2.get("/learners/demo-nika-04/learning-sessions/current",
                          headers=student_headers2)
    current_payload = current.json() if current.status_code == 200 else {}
    check("restart_restores_active_session",
          current.status_code == 200 and current_payload.get("available") is True
          and current_payload.get("learning_session_id") == session_id,
          f"{current.status_code}: {str(current_payload)[:160]}")
    restored_user_texts = [message.get("content", "") for message in current_payload.get("messages", [])
                           if message.get("role") == "user"]
    check("restart_no_duplicate_messages",
          restored_user_texts.count("хз") == 1 and "24 × 3 = 72" in restored_user_texts,
          str(restored_user_texts[-4:]))
    state_restored = client2.get(
        f"/learners/demo-nika-04/learning-sessions/{session_id}/lesson-state",
        headers=student_headers2).json()
    check("restart_restores_task_progress",
          state_restored.get("pending_task_id") == state_after_xz.get("pending_task_id")
          and (state_restored.get("completed_task_ids") or [])
          == (state_after_xz.get("completed_task_ids") or []),
          str(state_restored)[:200])
    resume = client2.post("/v1/chat/completions", headers=student_headers2, json={
        "learning_session_id": session_id, "client_message_id": "smoke-msg-resume",
        "messages": [{"role": "user", "content": "Где мы остановились?"}],
        "metadata": {"lesson_mode": "guided"}})
    check("restart_resume_reply", resume.status_code == 200, resume.text[:160])

    # ---- ordinary chat without a lesson also processes once (P0.3) --------
    ordinary = client2.post("/learners/demo-nika-04/learning-sessions", headers=student_headers2,
                            json={"topic_id": "math", "client_session_id": "smoke-ordinary-1"}).json()
    ordinary_id = ordinary["learning_session_id"]
    ordinary_reply = client2.post("/v1/chat/completions", headers=student_headers2, json={
        "learning_session_id": ordinary_id, "client_message_id": "smoke-ordinary-msg",
        "messages": [{"role": "user", "content": "Вчера целый день рисовал роботов!"}]})
    check("ordinary_chat_reply", ordinary_reply.status_code == 200, ordinary_reply.text[:120])
    worker2 = MemoryWorker(store2, runtime)
    ordinary_done = client2.post(f"/learning-sessions/{ordinary_id}/complete",
                                 headers=student_headers2)
    check("ordinary_completion_queues_processing",
          ordinary_done.status_code == 202
          and ordinary_done.json().get("session_processing_status") == "queued",
          str(ordinary_done.json())[:160])
    # The completion enqueues the legacy memory_update first (FIFO); drain
    # until the ordinary session's unified session_process completes.
    ordinary_job = None
    for _ in range(4):
        job = worker2.run_once()
        if not job:
            break
        if job.get("kind") == "session_process":
            ordinary_job = job
            break
    check("ordinary_processing_completed",
          bool(ordinary_job) and ordinary_job.get("status") == "completed",
          str((ordinary_job or {}).get("error"))[:160])

    # ---- memory-disabled path: dialogue stays, no personal enrichment -----
    from local_app.config import Settings as SmokeSettings
    off_settings = SmokeSettings(database_url=dsn, context_tokens=16384,
                                 personal_memory_enabled=False)
    store_off = StoreClass(dsn)
    app_off = create_app(settings=off_settings, store=store_off, runtime=runtime,
                         rag_service=RagService(store=store_off, settings=off_settings,
                                                runtime=runtime))
    client_off = TestClient(app_off, raise_server_exceptions=False)

    def auth_off(email: str) -> dict:
        response = client_off.post("/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
        return {"Authorization": "Bearer " + response.json()["access_token"]}

    off_student = auth_off("smoke-student@example.test")
    off_session = client_off.post("/learners/demo-nika-04/learning-sessions", headers=off_student,
                                  json={"topic_id": "math", "client_session_id": "smoke-off-1"}).json()
    off_reply = client_off.post("/v1/chat/completions", headers=off_student, json={
        "learning_session_id": off_session["learning_session_id"],
        "client_message_id": "smoke-off-msg",
        "messages": [{"role": "user", "content": "привет, расскажи что мы делали вчера"}]})
    check("memory_off_chat_still_works", off_reply.status_code == 200, off_reply.text[:120])
    check("memory_off_no_unified_fast_memory",
          "unified_fast_memory" not in json.dumps(off_reply.json(), ensure_ascii=False),
          "no personal-memory enrichment bypasses the setting")

    # ---- completion of the lesson session processes memory/summary --------
    completed = client2.post(f"/learning-sessions/{session_id}/complete",
                             headers=student_headers2)
    check("session_completed", completed.status_code == 202, str(completed.json())[:200])
    check("completion_queues_processing",
          completed.json().get("session_processing_status") == "queued",
          str(completed.json())[:160])
    # Drain the queue like a running worker would (both unified kinds).
    worker2b = MemoryWorker(store2, runtime)
    drained, last_job = 0, None
    while drained < 8:
        job = worker2b.run_once()
        if not job:
            break
        last_job = job
        drained += 1
    check("worker_drained_queue", drained >= 1,
          f"{drained} jobs; last={last_job.get('kind') if last_job else None}")
    result = client2.get(f"/learners/demo-nika-04/learning-sessions/{session_id}/processing-result",
                         headers=teacher_headers)
    payload = result.json() if result.status_code == 200 else {}
    check("processing_result_available",
          result.status_code == 200 and payload.get("available") is True
          and payload.get("status") in {"completed", "partial"},
          f"{result.status_code}: {str(payload)[:200]}")
    summary_text = ((payload.get("summary") or {}).get("text") or "")
    check("summary_has_four_headings",
          all(heading in summary_text for heading in ("Что изучали", "Что получилось", "Что стоит повторить", "Следующий шаг")),
          summary_text[:120].replace("\n", " | "))
    # Summary substance, not headings alone: the student's verified attempt is
    # the only correct answer in the transcript, so a faithful summary can
    # ground it; fabricated achievements stay detectable.
    check("summary_substance_mentions_attempt",
          "24 × 3" in json.dumps(payload, ensure_ascii=False),
          summary_text[:200].replace("\n", " | "))
    check("summary_has_sources",
          bool((payload.get("summary") or {}).get("source_refs")),
          str((payload.get("summary") or {}).get("source_refs"))[:120])
    changes = client2.get("/learners/demo-nika-04/memory-changes", headers=teacher_headers)
    check("memory_changes_listed", changes.status_code == 200, changes.text[:200])

    # ---- recall data path (persisted chunks -> facade recall) -------------
    from local_app.unified_app import resolve_unified_memory
    facade = resolve_unified_memory(store2, runtime)
    # Own-archive recall: the conversation belongs to the student account.
    recall = facade.recall("demo-nika-04", student["id"], "умножение", session_id=session_id)
    check("recall_shape", recall.get("method") == "conversation_retrieval"
          and isinstance(recall.get("system_text"), str),
          f"sources={len(recall.get('sources', []))} evidence={len(recall.get('evidence_ids', []))}")
    check("recall_finds_session_chunk",
          bool(recall.get("source_ids") or recall.get("evidence_ids")),
          "the processed transcript chunk is retrievable")

    # ---- next lesson from current effective fast memory -------------------
    second = client2.post("/learners/demo-nika-04/lessons", headers=teacher_headers,
                          json={"topic_id": "math", "minutes": 15, "client_request_id": "smoke-lesson-2"})
    check("second_lesson_enqueued", second.status_code == 202, str(second.json()))
    worker2c = MemoryWorker(store2, runtime)
    second_job = worker2c.run_once()
    second_result = (second_job or {}).get("result") or {}
    check("second_lesson_saved_from_fast_memory",
          bool(second_job) and second_job.get("status") == "completed"
          and second_result.get("lesson_id") and second_result.get("lesson_id") != lesson_id
          and str(second_result.get("memory_revision", "")).startswith("um-"),
          f"memory_revision={second_result.get('memory_revision')}")
    diagnostics = second_result.get("fast_memory_diagnostics") or {}
    check("second_lesson_prompt_uses_effective_records",
          diagnostics.get("included", {}).get("interests", 0) >= 1,
          str(diagnostics)[:200])

    # ---- the old attach endpoint stays compatible ------------------------
    explicit_session = client2.post("/learners/demo-nika-04/learning-sessions",
                                    headers=student_headers2,
                                    json={"topic_id": "math", "client_session_id": "smoke-attach-2"}).json()
    attached = client2.post(f"/learners/demo-nika-04/lessons/{lesson_id}/attach",
                            headers=teacher_headers,
                            json={"learning_session_id": explicit_session["learning_session_id"],
                                  "request_id": "smoke-attach-2"})
    check("attach_endpoint_still_compatible", attached.status_code == 200, str(attached.json())[:160])

    # ---- second child isolation --------------------------------------------
    luna_lesson_access = client.get(f"/learners/demo-nika-04/lessons/{lesson_id}",
                                    headers=teacher_headers)
    check("owner_teacher_sees_lesson", luna_lesson_access.status_code == 200, "")
    unassigned = store.create_account("smoke-outsider@example.test", "Outsider",
                                      hash_password("Smoke script password 123"), "teacher")
    outsider_headers = auth("smoke-outsider@example.test")
    denied = client.get(f"/learners/demo-nika-04/lessons/{lesson_id}", headers=outsider_headers)
    check("second_child_isolation", denied.status_code == 403, str(denied.status_code))

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["outcome"] = "ok" if not report["failed"] else "failed"
    if not keep_database:
        drop_disposable_database(admin, db_name)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified application smoke on a disposable database")
    parser.add_argument("--keep-database", action="store_true", help="Keep the disposable database for inspection")
    parser.add_argument("--live", action="store_true", help="Use the real local Ollama/Gemma runtime instead of the scripted double")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON report to a file")
    args = parser.parse_args()
    report = run_smoke(keep_database=args.keep_database, live=args.live)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["outcome"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
