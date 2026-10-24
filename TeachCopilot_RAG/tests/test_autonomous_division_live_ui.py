"""Unmocked browser HTTP through production API/Store and a real worker process."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import psutil
import pytest
import requests
from psycopg2.extensions import parse_dsn
from playwright.sync_api import expect

from local_app.auth import hash_password
from local_app.autonomous_division import TOPIC, VERSION
from tests.autonomous_network_fault import ResponseLossProxy
from tests.test_memory_api_flow import flow_store  # noqa: F401
from tests.test_memory_load import seed_queue_learner


ROOT = Path(__file__).resolve().parents[1]
LEARNERS = ("demo-nika-04", "demo-luna-03")
PASSWORD = "Synthetic live browser password 123"
PUBLIC_STATE_FIELDS = {"state_id", "dividend", "divisor", "step_index", "phase", "status",
                       "question_number", "question_count"}


def independent_steps(dividend, divisor):
    """School division using only decimal digits and integer arithmetic."""
    partial, started, steps = 0, False, []
    for digit in map(int, str(dividend)):
        partial = partial * 10 + digit
        if not started and partial < divisor:
            continue
        started = True
        quotient_digit = partial // divisor
        remainder = partial - quotient_digit * divisor
        steps.append({"digit": digit, "quotient_digit": quotient_digit, "remainder": remainder,
                      "partial": partial, "product": quotient_digit * divisor})
        partial = remainder
    assert partial == 0 and dividend == divisor * int("".join(str(s["quotient_digit"]) for s in steps))
    return steps


def next_answer(state):
    steps = independent_steps(state["dividend"], state["divisor"])
    index = state["step_index"]
    return steps[index + 1]["digit"] if state["phase"] == "bring_down" else steps[index][state["phase"]]


def position(state):
    return state["dividend"], state["divisor"], state["step_index"], state["phase"]


def prompt_arithmetic(state):
    step = independent_steps(state["dividend"], state["divisor"])[state["step_index"]]
    if state["phase"] == "quotient_digit":
        return f"Сейчас делим {step['partial']} на {state['divisor']}."
    if state["phase"] == "remainder":
        return f"Вычти {step['product']} из {step['partial']}."
    return f"Остаток этого шага – {step['remainder']}."


def shown_arithmetic(state):
    steps = independent_steps(state["dividend"], state["divisor"])
    step = steps[state["step_index"]]
    if state["phase"] == "quotient_digit":
        return f"{state['divisor']} × {step['quotient_digit']} = {step['product']}"
    if state["phase"] == "remainder":
        return f"{step['partial']} − {step['product']} = {step['remainder']}"
    return f"снести цифру {steps[state['step_index'] + 1]['digit']}"


def assert_public(value):
    if isinstance(value, dict):
        assert not any(key.startswith("_autonomous") or key in {
            "long_division_steps", "example_kind", "actions_used", "previous_message_id",
            "previous_session_id", "raw_response", "reasoning_content"} for key in value)
        for child in value.values():
            assert_public(child)
    elif isinstance(value, list):
        for child in value:
            assert_public(child)


class LiveStack:
    def __init__(self, store, artifacts):
        self.store, self.artifacts = store, artifacts
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.api_url = f"http://127.0.0.1:{self.port}"
        self.processes = []
        self.starts = []
        self.checkpoints = []
        self.responses = []
        self.browser_errors = []
        self.console_errors = []
        self.help_phases = set()
        self.cards_before = {learner: store.get_card(learner) for learner in LEARNERS}
        self.legacy_before = self.legacy_snapshot()
        self.database_evidence = {}
        self.completion_refreshes = 0
        self.proxy = None
        self.completed = False

    def start(self):
        generation = len(self.starts) + 1
        env = dict(os.environ, DATABASE_URL=self.store.dsn, OLLAMA_URL="http://127.0.0.1:9",
            AUTONOMOUS_LIVE_ARTIFACTS=str(self.artifacts), AUTONOMOUS_LIVE_PORT=str(self.port),
            LOCAL_MEMORY_RESPONSE_JOURNAL=str(self.artifacts / "worker-responses.sqlite3"),
            LOCAL_AUTONOMOUS_DIVISION_ENABLED="1", LOCAL_DEMO_LOGIN="0", PYTHONIOENCODING="utf-8",
            TEACHCOPILOT_EXTERNAL_ANSWERS_CONFIG="", LOCAL_GENERATION_RUN_BUDGET_PATH="",
            LOCAL_GENERATION_SCHEDULER_DSN="")
        for role in ("api", "worker"):
            log = (self.artifacts / f"{role}-{generation}.log").open("w", encoding="utf-8")
            command = [sys.executable, "tests/autonomous_live_process.py", role]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.processes.append((process, log))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            self.assert_alive()
            try:
                if requests.get(self.api_url + "/health/ready", timeout=1).status_code == 200:
                    self.starts.append({"generation": generation,
                        "pids": [process.pid for process, _ in self.processes]})
                    return
            except requests.RequestException:
                pass
            time.sleep(0.05)
        raise AssertionError(f"API did not start; see {self.artifacts}")

    def assert_alive(self):
        assert all(process.poll() is None for process, _ in self.processes), str(self.artifacts)

    def stop(self):
        for process, log in reversed(self.processes):
            try:
                owned = psutil.Process(process.pid)
                children = [child for child in owned.children(recursive=True)
                            if child.name().casefold() in {"python", "python.exe", "python3"}]
                for child in reversed(children):
                    assert Path(child.cwd()).resolve() == ROOT
                    assert any("autonomous_live_process.py" in part for part in child.cmdline())
                    child.terminate()
                psutil.wait_procs(children, timeout=5)
                if process.poll() is None:
                    owned.terminate()
                process.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
            finally:
                log.close()
        self.processes.clear()

    def restart(self):
        self.stop()
        self.start()
        self.checkpoints.append({"event": "api_and_worker_restarted", "generation": len(self.starts)})

    def saved(self, session_id, message_id=None):
        with self.store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("""SELECT m.client_message_id,m.response,s.source_order
                FROM teachcopilot.learning_messages m JOIN teachcopilot.memory_sources s ON s.id=m.source_id
                WHERE m.learning_session_id=%s AND m.status='completed'
                  AND (%s::text IS NULL OR m.client_message_id=%s)
                ORDER BY s.source_order DESC LIMIT 1""", (session_id, message_id, message_id))
            row = cursor.fetchone()
        assert row is not None
        return row["response"]["_autonomous_division_state"]

    def legacy_snapshot(self):
        with self.store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("""SELECT m.learner_id,m.status,m.response,s.content,s.author_role
                FROM teachcopilot.learning_messages m JOIN teachcopilot.memory_sources s ON s.id=m.source_id
                WHERE m.client_message_id='legacy-turn' ORDER BY m.learner_id""")
            return [dict(row) for row in cursor.fetchall()]

    def verify_response(self, body):
        assert_public(body)
        state = body["autonomous_division"]
        assert set(state) == PUBLIC_STATE_FIELDS
        saved = self.saved(body["learning_session_id"], body["client_message_id"])
        assert all(saved[key] == state[key] for key in state if key in saved), {
            "public_state": state, "saved_fields": {key: saved[key] for key in state if key in saved}}
        assert state["question_count"] == len(independent_steps(state["dividend"], state["divisor"]))
        assert body["learning_tool"]["facts"] == {}
        assert body["autonomous_assessment"]["competence_assessed"] is False
        self.responses.append(body)
        return state

    def write_artifacts(self):
        # Each helper writes its own file. Merge only after every helper stops;
        # Windows append mode does not serialize writes from separate processes.
        events = []
        audit_files = sorted(self.artifacts.glob("process_events-*.jsonl"))
        for path in audit_files:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                assert path.name == f"process_events-{event['role']}-{event['pid']}.jsonl"
                events.append(event)
        events.sort(key=lambda event: (event["time"], event["pid"]))
        forbidden_files = sorted(self.artifacts.glob("forbidden_calls-*.jsonl"))
        forbidden_count = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in forbidden_files)
        if self.completed:
            assert not forbidden_files
            assert len(audit_files) == 2 * len(self.starts)
            for event_name in ("api_start", "worker_start"):
                assert sum(event["event"] == event_name for event in events) == len(self.starts)
            assert sum(event["event"] == "database_connected" for event in events) == len(audit_files)
            assert any(event["event"] == "worker_job_returned" and event["status"] == "completed"
                       and event["runtime_calls"] == 0 for event in events)
        for name, records in (("responses.jsonl", self.responses),
                              ("network.jsonl", self.proxy.request_log if self.proxy else []),
                              ("checkpoints.jsonl", self.checkpoints),
                              ("process_events.jsonl", events)):
            with (self.artifacts / name).open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        receipt = {"passed": self.completed, "api_worker_starts": self.starts,
            "database_name": parse_dsn(self.store.dsn)["dbname"],
            "completed_actions": len(self.responses), "browser_errors": self.browser_errors,
            "console_errors": self.console_errors, "mocked_learning_http_responses": 0,
            "database_evidence": self.database_evidence,
            "help_phases_checked": sorted(self.help_phases),
            "autonomous_refreshes_after_completion": self.completion_refreshes,
            "oracle": "decimal-digit integer arithmetic independent of production math",
            "process_audit_files": len(audit_files), "process_audit_events": len(events),
            "forbidden_calls": forbidden_count}
        (self.artifacts / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def live_stack(flow_store, tmp_path):
    store = flow_store
    source = ROOT.parent / "learner-data/demo/learners/demo-luna-03.json"
    shutil.copy(source, tmp_path / "learner-data/learners" / source.name)
    store.import_repository(tmp_path)
    password_hash = hash_password(PASSWORD)
    teacher = store.create_account("live-teacher@synthetic.test", "Setup only", password_hash, "teacher")
    for index, learner in enumerate(LEARNERS):
        store.assign_learner(teacher["id"], learner)
        account = store.create_student_account(f"live-{index}@synthetic.test", f"Live student {index}",
                                               password_hash, learner, teacher["id"])
        legacy = store.create_learning_session(learner, account["id"], TOPIC, client_session_id="legacy-standard")
        store.reserve_chat_message(legacy["id"], learner, "legacy-turn", "Earlier synthetic question", account["id"])
        store.complete_chat_message(legacy["id"], "legacy-turn", {"choices": [{"message": {
            "role": "assistant", "content": "Earlier synthetic answer"}}], "diagnostics": "LEGACY_PRIVATE"})
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("UPDATE teachcopilot.learning_sessions SET status='completed',completed_at=now() WHERE id=%s",
                           (legacy["id"],))
    seed_queue_learner(store, "worker-sentinel")
    sentinel = store.enqueue_memory_job("worker-sentinel", "synthetic", "memory_update", idempotency_key="alive")
    parent = Path(os.getenv("TEACHCOPILOT_LIVE_ARTIFACTS", str(ROOT / ".local/autonomous_division_live")))
    artifacts = parent.resolve() / uuid4().hex[:12]
    artifacts.mkdir(parents=True)
    print(f"AUTONOMOUS_LIVE_ARTIFACTS={artifacts}", flush=True)
    stack = LiveStack(store, artifacts)
    try:
        stack.start()
        deadline = time.monotonic() + 10
        while store.get_memory_job(sentinel["id"])["status"] != "completed" and time.monotonic() < deadline:
            stack.assert_alive()
            time.sleep(0.05)
        completed = store.get_memory_job(sentinel["id"])
        assert completed["status"] == "completed" and completed["call_started_at"] is None
        assert completed["result"]["calls"] == []
        with ResponseLossProxy(stack.api_url) as proxy:
            stack.proxy = proxy
            yield stack
    finally:
        try:
            stack.stop()
        finally:
            stack.write_artifacts()


@contextmanager
def browser_lesson(chromium, stack, learner_index, lesson):
    # Each lesson uses a new browser process and a real student login.
    browser = chromium.browser_type.launch(headless=True)
    label = f"learner-{learner_index}-lesson-{lesson}"
    context = browser.new_context(viewport={"width": 1365, "height": 1050},
        record_video_dir=str(stack.artifacts / "videos"),
        record_har_path=str(stack.artifacts / (label + ".har")))
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    def observe(page):
        page.on("pageerror", lambda error: stack.browser_errors.append(str(error)))
        page.on("console", lambda message: stack.console_errors.append(message.text) if message.type == "error" else None)
    context.on("page", observe)
    page = context.new_page()
    try:
        page.goto(stack.proxy.base_url, wait_until="networkidle")
        page.locator("#email").fill(f"live-{learner_index}@synthetic.test")
        page.locator("#password").fill(PASSWORD)
        page.locator('#login-form button[type="submit"]').click()
        expect(page.locator("#workspace")).to_be_visible()
        expect(page.locator("#profile-content")).to_be_visible()
        expect(page.locator("#refresh-learning-session")).to_be_enabled()
        page.locator("#chat-mode").select_option("autonomous_division")
        expect(page.locator("#autonomous-division-refresh")).to_be_enabled()
        yield page
    except BaseException as error:
        stack.checkpoints.append({"event": "browser_failure", "lesson": label,
            "error_type": type(error).__name__, "traceback": traceback.format_exc()})
        page.screenshot(path=str(stack.artifacts / (label + "-failure.png")), full_page=True)
        raise
    finally:
        context.tracing.stop(path=str(stack.artifacts / (label + "-trace.zip")))
        context.close()
        browser.close()


def action(page, stack, name, *, value=None, status=200):
    if value is not None:
        page.locator("#autonomous-division-answer").fill(str(value))
    selectors = {"answer": "check", "show_step": "show-step"}
    selector = "#autonomous-division-" + selectors.get(name, name)
    operation = "autonomous_division_" + name
    with page.expect_response(lambda response: response.url.endswith("/v1/chat/completions")
            and response.request.post_data_json.get("learning_action", {}).get("operation") == operation,
            timeout=60000) as pending:
        page.locator(selector).click()
    response = pending.value
    assert response.status == status, response.text()
    body = response.json()
    if status != 200:
        return body
    state = stack.verify_response(body)
    if state["status"] == "active":
        expect(page.locator("#autonomous-division-check")).to_be_enabled()
        expect(page.locator("#autonomous-division-prompt")).to_contain_text(prompt_arithmetic(state))
        if name == "show_step":
            expect(page.locator("#autonomous-division-prompt")).to_contain_text(shown_arithmetic(state))
            stack.help_phases.add(state["phase"])
    elif state["status"] == "paused":
        expect(page.locator("#autonomous-division-resume")).to_be_visible()
    else:
        expect(page.locator("#autonomous-division-next")).to_be_visible()
    expect(page.locator("#autonomous-division-example")).to_contain_text(f"{state['dividend']} : {state['divisor']}")
    return body


def finish_example(page, stack, body):
    initial = body["autonomous_division"]
    quotient = initial["dividend"] // initial["divisor"]
    kind = "terminal_zero" if str(quotient).endswith("0") else "internal_zero" if "0" in str(quotient) else "ordinary"
    for _ in range(30):
        before = body["autonomous_division"]
        if before["status"] == "completed":
            stack.checkpoints.append({"event": "example_completed", "kind": kind,
                "dividend": before["dividend"], "divisor": before["divisor"], "quotient": quotient})
            return body
        if before["phase"] not in stack.help_phases:
            body = action(page, stack, "show_step")
            assert position(body["autonomous_division"]) == position(before)
            before = body["autonomous_division"]
        body = action(page, stack, "answer", value=next_answer(before))
        after = body["autonomous_division"]
        assert body["autonomous_assessment"]["status"] == "correct"
        if before["phase"] == "quotient_digit":
            assert after["phase"] == "remainder" and after["step_index"] == before["step_index"]
        elif before["phase"] == "bring_down":
            assert after["phase"] == "quotient_digit" and after["step_index"] == before["step_index"] + 1
        elif before["step_index"] + 1 < before["question_count"]:
            assert after["phase"] == "bring_down" and after["step_index"] == before["step_index"]
        else:
            assert after["status"] == "completed"
    raise AssertionError("Example did not finish within its independently calculated path")


def headers(page):
    return {"Authorization": "Bearer " + page.evaluate("sessionStorage.getItem('teachcopilot.session')")}


def public_context(page, stack, learner, sid=None):
    params = {"topic_id": TOPIC, "protocol": VERSION}
    if sid:
        params["learning_session_id"] = sid
    response = page.context.request.get(stack.proxy.base_url + f"/learners/{learner}/guided-division",
                                        params=params, headers=headers(page))
    assert response.status == 200
    body = response.json()
    assert_public(body)
    return body


def end_lesson(page, stack, sid, learner, lesson):
    with page.expect_response(lambda response: "/guided-division?" in response.url
            and "protocol=autonomous-division-v1" in response.url) as refreshed:
        with page.expect_response(lambda response: response.url.endswith(f"/learning-sessions/{sid}/complete")) as pending:
            page.locator("#complete-learning-session").click()
    response = pending.value
    assert response.status == 202
    assert response.json()["job"] is None and response.json()["job_id"] is None
    assert refreshed.value.status == 200
    stack.completion_refreshes += 1
    expect(page.locator("#complete-learning-session")).to_be_hidden()
    expect(page.locator("#learning-session-status")).to_contain_text("без подтверждения памяти")
    expect(page.locator("#autonomous-division-refresh")).to_be_enabled()
    restored = public_context(page, stack, learner)
    if not restored["available"]:
        expect(page.locator("#autonomous-division-answer-form")).to_be_hidden()
        expect(page.locator("#autonomous-division-example")).to_be_hidden()
        expect(page.locator("#autonomous-division-prompt")).to_be_hidden()
    elif restored["state"]["status"] != "completed":
        expect(page.locator("#autonomous-division-resume")).to_be_visible()
        expect(page.locator("#autonomous-division-answer-form")).to_be_hidden()
    page.screenshot(path=str(stack.artifacts / f"{learner}-lesson-{lesson}-closed.png"), full_page=True)
    stack.checkpoints.append({"event": "lesson_completed", "learner": learner, "lesson": lesson, "session_id": sid})


def test_six_browser_lessons_real_api_database_worker_and_response_loss(chromium, live_stack):
    stack = live_stack
    sessions = []
    for learner_index, learner in enumerate(LEARNERS):
        with browser_lesson(chromium, stack, learner_index, 1) as page:
            assert public_context(page, stack, learner)["available"] is False
            expect(page.locator("#autonomous-division-digits")).to_have_value("1")
            body = action(page, stack, "start")
            sid = body["learning_session_id"]
            sessions.append(sid)
            initial = body["autonomous_division"]
            assert 1 <= initial["divisor"] <= 9
            for text in ("ага", "хз", "4?"):
                count = len(stack.proxy.request_log)
                page.locator("#autonomous-division-answer").fill(text)
                page.locator("#autonomous-division-check").click()
                expect(page.locator("#learning-action-error")).to_contain_text("Введи одно целое число")
                assert stack.saved(sid)["state_id"] == initial["state_id"]
                assert not any(row.get("operation") for row in stack.proxy.request_log[count:])
            for text in ("ага", "хз", "4?", "Расскажи о погоде"):
                response = page.context.request.post(stack.proxy.base_url + "/v1/chat/completions",
                    headers=headers(page), data={"learning_session_id": sid, "client_message_id": uuid4().hex,
                        "learning_protocol": VERSION, "messages": [{"role": "user", "content": text}]})
                assert response.status == 200
                clarified = response.json()
                state = stack.verify_response(clarified)
                assert position(state) == position(initial)
                assert clarified["learning_action"]["operation"] == "autonomous_division_clarify"
            page.locator("#autonomous-division-refresh").click()
            expect(page.locator("#autonomous-division-check")).to_be_enabled()
            body = action(page, stack, "answer", value=next_answer(initial) + 1)
            assert body["autonomous_assessment"]["status"] == "incorrect"
            assert position(body["autonomous_division"]) == position(initial)
            for name in ("explain", "show_step"):
                before = body["autonomous_division"]
                body = action(page, stack, name)
                assert position(body["autonomous_division"]) == position(before)
                assert str(initial["dividend"] // initial["divisor"]) not in body["choices"][0]["message"]["content"]
            body = action(page, stack, "answer", value=next_answer(body["autonomous_division"]))
            body = action(page, stack, "pause")
            paused = body["autonomous_division"]
            end_lesson(page, stack, sid, learner, 1)
        stack.restart()

        with browser_lesson(chromium, stack, learner_index, 2) as page:
            restored = public_context(page, stack, learner)
            assert restored["resume_required"] and restored["state"] == paused
            expect(page.locator("#autonomous-division-resume")).to_be_visible()
            body = action(page, stack, "resume")
            sid = body["learning_session_id"]
            sessions.append(sid)
            assert position(body["autonomous_division"]) == position(paused)
            body = finish_example(page, stack, body)
            body = action(page, stack, "next")
            body = finish_example(page, stack, body)
            body = action(page, stack, "next")
            body = action(page, stack, "pause")
            paused = body["autonomous_division"]
            end_lesson(page, stack, sid, learner, 2)

        with browser_lesson(chromium, stack, learner_index, 3) as page:
            restored = public_context(page, stack, learner)
            assert restored["state"] == paused and restored["resume_required"]
            body = action(page, stack, "resume")
            sid = body["learning_session_id"]
            sessions.append(sid)
            body = finish_example(page, stack, body)
            page.locator("#autonomous-division-digits").select_option("2")
            body = action(page, stack, "next")
            assert 10 <= body["autonomous_division"]["divisor"] <= 99
            old = body["autonomous_division"]
            body = action(page, stack, "switch")
            assert position(body["autonomous_division"]) != position(old)
            with stack.store._connect() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT response->'_autonomous_division_provenance'->>'interrupted' AS value "
                    "FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND client_message_id=%s",
                    (sid, body["client_message_id"]))
                assert cursor.fetchone()["value"] == "true"

            if learner_index == 0:
                second_tab = page.context.new_page()
                second_tab.goto(stack.proxy.base_url, wait_until="networkidle")
                second_tab.locator("#email").fill("live-0@synthetic.test")
                second_tab.locator("#password").fill(PASSWORD)
                second_tab.locator('#login-form button[type="submit"]').click()
                expect(second_tab.locator("#profile-content")).to_be_visible()
                expect(second_tab.locator("#autonomous-division-check")).to_be_enabled()
                stale = body["autonomous_division"]
                body = action(page, stack, "answer", value=next_answer(stale))
                action(second_tab, stack, "answer", value=next_answer(stale), status=409)
                expect(second_tab.locator("#learning-action-error")).to_contain_text("Состояние примера изменилось")
                assert public_context(second_tab, stack, learner, sid)["state"] == body["autonomous_division"]
                second_tab.close()
                stack.checkpoints.append({"event": "stale_second_tab_rejected", "session_id": sid})

                stack.proxy.arm(mode="truncate_body")
                page.locator("#autonomous-division-answer").fill(str(next_answer(body["autonomous_division"])))
                page.locator("#autonomous-division-check").click()
                assert stack.proxy.dropped.wait(60)
                captured = stack.proxy.captured
                committed = captured["response_json"]
                assert stack.saved(sid)["state_id"] == committed["client_message_id"]
                expect(page.locator("#retry-chat")).to_be_visible()
                stack.restart()
                with page.expect_response(lambda response: response.url.endswith("/v1/chat/completions")) as replay:
                    page.locator("#retry-chat").click()
                assert replay.value.status == 200 and replay.value.json() == committed
                body = replay.value.json()
                stack.verify_response(body)
                expect(page.locator("#autonomous-division-check")).to_be_enabled()
                with stack.store._connect() as connection, connection.cursor() as cursor:
                    cursor.execute("SELECT count(*) AS n FROM teachcopilot.learning_messages "
                        "WHERE learning_session_id=%s AND client_message_id=%s", (sid, committed["client_message_id"]))
                    assert cursor.fetchone()["n"] == 1
                stack.checkpoints.append({"event": "committed_response_lost_then_replayed_after_restart",
                    "session_id": sid, "message_id": committed["client_message_id"]})
            body = finish_example(page, stack, body)
            if learner_index == 1:
                body = action(page, stack, "next")
                stack.store.reserve_chat_message(sid, learner, "synthetic-unknown", "Unresolved synthetic action",
                                                 stack.store.get_learning_session(sid)["actor_id"])
                stack.store.fail_chat_message(sid, "synthetic-unknown", "synthetic lost outcome", unknown_outcome=True)
                page.locator("#refresh-learning-session").click()
                expect(page.locator("#learning-session-recovery-note")).to_contain_text("Результат предыдущего запроса неизвестен")
                expect(page.locator("#autonomous-division-answer-form")).to_be_hidden()
                expect(page.locator("#autonomous-division-example")).to_be_hidden()
                expect(page.locator("#autonomous-division-prompt")).to_be_hidden()
                blocked = public_context(page, stack, learner, sid)
                assert blocked == {"learning_protocol": VERSION, "available": False, "reason": "unknown_outcome"}
                stack.checkpoints.append({"event": "unknown_outcome_blocks_old_cursor", "session_id": sid})

            # Authorization/mode checks use real browser HTTP, without teacher login.
            auth = headers(page)
            foreign = page.context.request.get(stack.proxy.base_url + f"/learners/{LEARNERS[1 - learner_index]}/guided-division",
                params={"topic_id": TOPIC, "protocol": VERSION}, headers=auth)
            assert foreign.status == 403
            forged = page.context.request.post(stack.proxy.base_url + "/v1/chat/completions", headers=auth, data={
                "learning_session_id": sid, "client_message_id": "forged-mode", "learning_protocol": "guided-division-v1",
                "learning_action": {"operation": "division_guided_pause", "expected_state_id": body["client_message_id"]}})
            assert forged.status == 409
            end_lesson(page, stack, sid, learner, 3)
            archive = page.context.request.get(stack.proxy.base_url + f"/learners/{learner}/learning-sessions/{sid}/history",
                                                headers=auth)
            assert archive.status == 200
            pages = [archive.json()]
            while pages[-1]["next_cursor"]:
                archive = page.context.request.get(stack.proxy.base_url + f"/learners/{learner}/learning-sessions/{sid}/history",
                    params={"before": pages[-1]["next_cursor"]}, headers=auth)
                assert archive.status == 200
                pages.append(archive.json())
            for history in pages:
                assert_public(history)
                assert "LEGACY_PRIVATE" not in json.dumps(history)
            restored_messages = [message for history in reversed(pages) for message in history["messages"]]
            assert [(row["role"], row["content"]) for row in restored_messages] == [
                (row["role"], row["content"]) for row in stack.store.list_learning_messages(sid)]

    with stack.store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT count(*) AS n FROM teachcopilot.learning_sessions WHERE session_mode='autonomous_division' AND status='completed'")
        assert cursor.fetchone()["n"] == 6 and len(set(sessions)) == 6
        cursor.execute("SELECT count(*) AS n FROM teachcopilot.memory_jobs WHERE learner_id=ANY(%s)", (list(LEARNERS),))
        assert cursor.fetchone()["n"] == 0
        cursor.execute("SELECT count(*) AS n FROM teachcopilot.memory_versions WHERE learner_id=ANY(%s)", (list(LEARNERS),))
        assert cursor.fetchone()["n"] == 0
        cursor.execute("SELECT count(*) AS n FROM teachcopilot.learning_messages WHERE client_message_id='legacy-turn' AND response->>'diagnostics'='LEGACY_PRIVATE'")
        assert cursor.fetchone()["n"] == 2
    completed_examples = [row for row in stack.checkpoints if row["event"] == "example_completed"]
    assert {row["kind"] for row in completed_examples} == {"ordinary", "internal_zero", "terminal_zero"}
    assert any(row["divisor"] < 10 for row in completed_examples) and any(row["divisor"] >= 10 for row in completed_examples)
    assert not list(stack.artifacts.glob("forbidden_calls-*.jsonl"))
    assert stack.help_phases == {"quotient_digit", "remainder", "bring_down"}
    assert all(stack.store.get_card(learner) == stack.cards_before[learner] for learner in LEARNERS)
    assert stack.legacy_snapshot() == stack.legacy_before
    stack.database_evidence = {"completed_autonomous_sessions": 6, "sessions_per_learner": 3,
        "learner_memory_jobs": 0, "learner_memory_versions": 0, "cards_unchanged": True,
        "legacy_messages_unchanged": True, "archive_matches_literal_sources": True,
        "legacy_sha256": hashlib.sha256(json.dumps(stack.legacy_before, sort_keys=True).encode()).hexdigest()}
    assert not stack.browser_errors
    assert stack.completion_refreshes == 6
    assert all(re.search(r"409|ERR_EMPTY_RESPONSE|ERR_CONTENT_LENGTH_MISMATCH|Failed to fetch", message)
               for message in stack.console_errors), stack.console_errors
    assert not any("/confirm" in row["path"] for row in stack.proxy.request_log)
    stack.assert_alive()
    stack.completed = True
