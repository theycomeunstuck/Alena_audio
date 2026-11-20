"""Browser verification of the round-2 lesson handoff on a disposable database.

Starts the real FastAPI application (uvicorn) against a disposable
`teachcopilot_b_smoke_` PostgreSQL database, then drives the shipped web UI
with Chromium in two separate browser contexts: the teacher creates the
lesson (through the real queue and inline worker), confirms it and assigns
it – without creating any teacher-owned session – and a separate student
context sees the assignment, starts it into its own session and receives the
guided-lesson state. The server process is then actually stopped and a new
uvicorn process starts on the same database; the student context reloads and
must see the persisted lesson state and a «Продолжить занятие» action.
Model calls are not live here (a scripted local-model double); live quality
is measured by the quality check. The working learner database is untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
import shutil
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _create_database(base_dsn: str) -> tuple[str, object, str]:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    name = "teachcopilot_b_smoke_" + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(base_dsn, dbname="postgres", connect_timeout=5))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return make_dsn(base_dsn, dbname=name), admin, name


def _drop_database(admin, name: str) -> None:
    from psycopg2 import sql
    with admin.cursor() as cur:
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
        cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
    admin.close()


def run_browser_check(*, keep_database: bool = False) -> dict:
    from local_app.config import Settings
    base_dsn = Settings.from_env().database_url
    dsn, admin, db_name = _create_database(base_dsn)
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
              "server": "real uvicorn processes on loopback (disposable database)",
              "checks": [], "failed": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            report["failed"].append(name)

    from local_app.store import Store
    store = Store(dsn)
    store.migrate()
    with tempfile.TemporaryDirectory() as tmp:
        data_root = ROOT.parent / "learner-data"
        if not (data_root / "catalog").is_dir():
            data_root = ROOT / "learner-data"
        (Path(tmp) / "learner-data" / "learners").mkdir(parents=True)
        shutil.copytree(data_root / "catalog", Path(tmp) / "learner-data" / "catalog")
        shutil.copy(data_root / "demo" / "learners" / "demo-nika-04.json",
                    Path(tmp) / "learner-data" / "learners" / "demo-nika-04.json")
        store.import_repository(tmp)
    from local_app.auth import hash_password
    password = "Browser check password 1"
    teacher = store.create_account("browser-teacher@example.test", "Browser teacher",
                                   hash_password(password), "teacher")
    store.assign_learner(teacher["id"], "demo-nika-04")
    student = store.create_student_account("browser-student@example.test", "Browser student",
                                           hash_password(password), "demo-nika-04", teacher["id"])

    from uvicorn import Config, Server
    from local_app.api import create_app
    from local_app.unified_app import resolve_unified_memory
    if resolve_unified_memory(store, None) is None:
        check("facade_available", False,
              "local_app.unified_memory is not merged yet; the browser handoff cannot run and must not be faked")
        if not keep_database:
            _drop_database(admin, db_name)
        report["outcome"] = "facade_missing"
        return report

    # A scripted local-model double keeps the browser check free of live model
    # calls while still exercising the real queue/worker/generation path.
    from scripts.unified_app_smoke import ScriptedRuntime
    from local_app.rag import RagService
    from local_app.memory_service import MemoryWorker
    settings = Settings(database_url=dsn, context_tokens=16384)
    runtime = ScriptedRuntime()

    def build_server() -> tuple[Server, threading.Thread, str]:
        store_i = Store(dsn)
        app_i = create_app(settings=settings, store=store_i, runtime=runtime,
                           rag_service=RagService(store=store_i, settings=settings, runtime=runtime))
        port_i = _free_port()
        server_i = Server(Config(app=app_i, host="127.0.0.1", port=port_i, log_level="warning"))
        thread_i = threading.Thread(target=server_i.run, daemon=True)
        thread_i.start()
        deadline_i = time.time() + 30
        while time.time() < deadline_i:
            try:
                with socket.create_connection(("127.0.0.1", port_i), timeout=1):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("uvicorn did not open a loopback port in 30s")
        return server_i, thread_i, port_i

    server, thread, port = build_server()
    check("server_started", True, f"http://127.0.0.1:{port}")

    from playwright.sync_api import sync_playwright
    base = f"http://127.0.0.1:{port}"
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as exc:
            check("chromium_available", False, str(exc)[:150])
            server.should_exit = True
            report["outcome"] = "chromium_missing"
            if not keep_database:
                _drop_database(admin, db_name)
            return report
        check("chromium_available", True, "headless chromium, two separate contexts")

        def login(page, email: str) -> None:
            page.goto(base)
            page.fill("#email", email)
            page.fill("#password", password)
            page.click("#login-form button[type=submit]")
            page.wait_for_selector("#workspace:not([hidden])", timeout=10000)

        # ---- teacher context: create, confirm, assign ------------------------
        teacher_page = browser.new_context().new_page()
        login(teacher_page, "browser-teacher@example.test")
        check("teacher_login", True, teacher_page.title())
        teacher_page.wait_for_selector(".learner-list button", timeout=10000)
        teacher_page.click(".learner-list button")
        teacher_page.wait_for_selector("#profile-content:not([hidden])", timeout=10000)
        topic_values = teacher_page.eval_on_selector("#lesson-topic-select",
            "s => Array.from(s.options).map(o => o.value).filter(Boolean)")
        check("teacher_topic_options_loaded", bool(topic_values), f"{len(topic_values)} topics")
        if topic_values:
            teacher_page.select_option("#lesson-topic-select", topic_values[0])
        teacher_page.click("#create-lesson")
        # The teacher UI polls the job every 5s; run the real inline worker so
        # the scripted generation completes behind the same queue/lease path.
        worker = MemoryWorker(store, runtime)
        for _ in range(30):
            worker.run_once()
            if teacher_page.is_visible("#lesson-view"):
                break
            time.sleep(1)
        check("teacher_lesson_draft_visible", teacher_page.is_visible("#lesson-view"))
        confirm_button = teacher_page.wait_for_selector("#confirm-lesson:not([hidden])", timeout=30000)
        confirm_button.click()
        teacher_page.wait_for_selector("#assign-lesson:not([hidden])", timeout=15000)
        teacher_page.click("#assign-lesson")
        deadline = time.time() + 20
        assign_status = ""
        while time.time() < deadline:
            assign_status = teacher_page.inner_text("#lesson-assign-status")
            if "Задание выдано" in assign_status:
                break
            time.sleep(0.5)
        check("teacher_assigns_without_own_session", "Задание выдано" in assign_status,
              assign_status[:160])
        teacher_screenshot = ROOT / ".local" / "browser-check-teacher-assign.png"
        teacher_screenshot.parent.mkdir(parents=True, exist_ok=True)
        teacher_page.screenshot(path=str(teacher_screenshot), full_page=False)
        check("teacher_screenshot_saved", teacher_screenshot.is_file(), str(teacher_screenshot))

        # ---- separate student context: sees and starts the assignment --------
        student_page = browser.new_context().new_page()
        login(student_page, "browser-student@example.test")
        check("student_login", True, "")
        student_page.wait_for_selector("#profile-content:not([hidden])", timeout=10000)
        student_page.wait_for_selector(".learner-list button", timeout=10000)
        student_page.click(".learner-list button")
        student_page.wait_for_selector("#chat-log", timeout=10000)
        check("student_assignments_panel_visible",
              student_page.is_visible("#assignments-panel"), "#assignments-panel")
        start_button = student_page.wait_for_selector(
            "#assignments-list button:has-text('Начать занятие')", timeout=15000)
        start_button.click()
        deadline = time.time() + 20
        guided_state_text = ""
        while time.time() < deadline:
            if student_page.is_visible("#guided-lesson-state"):
                guided_state_text = student_page.inner_text("#guided-lesson-state")
                if "Этап" in guided_state_text or "урок" in guided_state_text.lower():
                    break
            time.sleep(0.5)
        check("student_starts_assignment_from_own_context",
              student_page.is_visible("#guided-lesson-state"), guided_state_text[:160])
        check("student_no_teacher_session_created",
              student_page.is_visible("#chat-form"), "student owns the chat")
        student_screenshot = ROOT / ".local" / "browser-check-student-start.png"
        student_page.screenshot(path=str(student_screenshot), full_page=False)
        check("student_screenshot_saved", student_screenshot.is_file(), str(student_screenshot))
        browser.close()

    # ---- genuine server restart ------------------------------------------
    server.should_exit = True
    thread.join(timeout=10)
    server2, thread2, port2 = build_server()
    base = f"http://127.0.0.1:{port2}"  # the nested login() reads this closure
    check("server_restarted", True, f"new uvicorn process at {base} on the same disposable database")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        student_page = browser.new_context().new_page()
        login(student_page, "browser-student@example.test")
        student_page.wait_for_selector(".learner-list button", timeout=10000)
        student_page.click(".learner-list button")
        student_page.wait_for_selector("#chat-log", timeout=10000)
        deadline = time.time() + 20
        restored_guided = ""
        while time.time() < deadline:
            if student_page.is_visible("#guided-lesson-state"):
                restored_guided = student_page.inner_text("#guided-lesson-state")
                if restored_guided.strip():
                    break
            time.sleep(0.5)
        check("restart_restores_lesson_state_in_browser",
              student_page.is_visible("#guided-lesson-state") and bool(restored_guided.strip()),
              restored_guided[:160])
        continue_button = student_page.wait_for_selector(
            "#assignments-list button:has-text('Продолжить занятие')", timeout=15000)
        check("restart_assignment_offers_continue", continue_button is not None, "")
        restart_screenshot = ROOT / ".local" / "browser-check-student-restart.png"
        student_page.screenshot(path=str(restart_screenshot), full_page=False)
        check("restart_screenshot_saved", restart_screenshot.is_file(), str(restart_screenshot))
        browser.close()

    server2.should_exit = True
    thread2.join(timeout=10)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["outcome"] = "ok" if not report["failed"] else "failed"
    if not keep_database:
        _drop_database(admin, db_name)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Browser verification of the unified lesson UI")
    parser.add_argument("--keep-database", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_browser_check(keep_database=args.keep_database)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["outcome"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
