"""Browser acceptance for the student-only deterministic division flow."""
from copy import deepcopy
import json

import pytest
from playwright.sync_api import expect

from local_app.autonomous_division import (
    VERSION, action_response, public_state, render_prompt, start_state, transition,
)
from local_app.explicit_learning_actions import validate_action
from local_app.learning_math import division_facts
from test_ui_acceptance import ui_server


def install_autonomous_transport(page, server):
    data = {"state": None, "sent": [], "cache": {}, "resume_required": False}

    def context(route):
        state = data["state"]
        if state is None:
            body = {"learning_protocol": VERSION, "available": False,
                    "reason": "no_saved_autonomous_step"}
        else:
            body = {"learning_protocol": VERSION, "available": True,
                    "state": public_state(state), "prompt": render_prompt(state),
                    "resume_required": data["resume_required"] or state["status"] == "paused",
                    "limit_reached": state["actions_used"] >= 60,
                    "available_actions": ["resume", "switch"] if state["status"] == "paused" else
                        ["next"] if state["status"] == "completed" else
                        ["answer", "explain", "show_step", "pause", "switch"]}
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body, ensure_ascii=False))

    def submit(route):
        request = route.request.post_data_json
        assert request["learning_protocol"] == VERSION
        action = request["learning_action"]
        assert action and action["operation"].startswith("autonomous_division_")
        data["sent"].append(deepcopy(request))
        message_id = request["client_message_id"]
        if message_id not in data["cache"]:
            canonical, action_text = validate_action(action)
            operation = canonical["operation"]
            if operation == "autonomous_division_start":
                assert canonical["divisor_digits"] == 2
                cursor = start_state(3636, 12, state_id=message_id,
                    kind="internal_zero", divisor_digits=2)
                assessment = {"status": "not_assessed"}
            else:
                cursor, assessment = transition(data["state"], canonical, state_id=message_id)
            data["state"] = cursor
            data["resume_required"] = False
            response = action_response(cursor, assessment, operation, action_text,
                                       interrupted=bool(assessment.get("interrupted")))
            data["cache"][message_id] = response
            session_id = request["learning_session_id"]
            session = server.store.get_learning_session(session_id)
            reservation = server.store.reserve_chat_message(session_id, "demo-nika-04", message_id,
                action_text, session["actor_id"], author_role="student")
            if reservation.get("created"):
                server.store.complete_chat_message(session_id, message_id, response)
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps(data["cache"][message_id], ensure_ascii=False))

    page.route("**/learners/*/guided-division?*", context)
    page.route("**/v1/chat/completions", submit)
    return data


def test_student_completes_autonomous_practice_without_teacher_or_memory_job(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    transport = install_autonomous_transport(page, ui_server)
    try:
        page.goto(ui_server.base_url, wait_until="networkidle")
        page.locator("#email").fill("student.ui@example.test")
        page.locator("#password").fill("test-only passphrase 123")
        page.locator('#login-form button[type="submit"]').click()
        expect(page.locator("#workspace")).to_be_visible()
        page.locator("#chat-mode").select_option("autonomous_division")
        expect(page.locator("#autonomous-division-start")).to_be_visible()
        expect(page.locator("#autonomous-division-status")).to_contain_text("Сохранённого примера нет")
        page.locator("#autonomous-division-digits").select_option("2")
        page.locator("#autonomous-division-start").click()
        expect(page.locator("#autonomous-division-example")).to_contain_text("3636 : 12")
        expect(page.locator("#autonomous-division-prompt")).to_contain_text("Какую цифру")
        assert transport["sent"][0]["learning_protocol"] == VERSION
        assert transport["sent"][0]["learning_action"] == {
            "operation": "autonomous_division_start", "divisor_digits": 2}

        # A wrong number keeps the same phase. Rule and step help also stay put.
        page.locator("#autonomous-division-answer").fill("4")
        page.locator("#autonomous-division-check").click()
        expect(page.locator("#chat-log")).to_contain_text("Это число не подходит")
        assert transport["state"]["step_index"] == 0 and transport["state"]["phase"] == "quotient_digit"
        page.locator("#autonomous-division-explain").click()
        expect(page.locator("#chat-log")).to_contain_text("Правило:")
        page.locator("#autonomous-division-show-step").click()
        expect(page.locator("#chat-log")).to_contain_text("Текущий шаг:")
        assert transport["state"]["step_index"] == 0 and transport["state"]["phase"] == "quotient_digit"
        page.locator("#autonomous-division-pause").click()
        expect(page.locator("#autonomous-division-answer-form")).to_be_hidden()
        expect(page.locator("#autonomous-division-resume")).to_be_visible()

        page.reload()
        expect(page.locator("#workspace")).to_be_visible()
        expect(page.locator("#chat-mode")).to_have_value("autonomous_division")
        page.locator("#autonomous-division-resume").scroll_into_view_if_needed()
        expect(page.locator("#autonomous-division-resume")).to_be_visible()
        expect(page.locator("#autonomous-division-example")).to_contain_text("3636 : 12")
        expect(page.locator("#autonomous-division-prompt")).to_be_visible()
        page.locator("#autonomous-division-resume").click()
        expect(page.locator("#autonomous-division-answer-form")).to_be_visible()

        # Finish 3636 : 12, retaining its internal zero, then choose another example.
        for value in (3, 0, 3, 0, 3, 6, 3, 0):
            page.locator("#autonomous-division-answer").fill(str(value))
            page.locator("#autonomous-division-check").click()
        expect(page.locator("#autonomous-division-next")).to_be_visible()
        assert transport["state"]["status"] == "completed"
        page.locator("#autonomous-division-next").click()
        expect(page.locator("#autonomous-division-answer-form")).to_be_visible()
        assert transport["state"]["status"] == "active"
        assert transport["state"]["example_kind"] == "terminal_zero"

        sid = ui_server.store.learning_sessions[next(reversed(ui_server.store.learning_sessions))]["learning_session_id"]
        complete = page.locator("#complete-learning-session")
        expect(complete).to_be_visible()
        complete.click()
        expect(page.locator("#learning-session-status")).to_contain_text("без подтверждения памяти")
        expect(page.locator("#toast")).to_contain_text("Автономная практика сохранена")
        assert ui_server.store.learning_sessions[sid]["session_mode"] == "autonomous_division"
        assert ui_server.store.learning_sessions[sid]["status"] == "completed"
        assert not ui_server.store.memory_jobs
        assert all("_autonomous_division_state" not in item for item in transport["cache"].values())
        assert all("_autonomous_division_provenance" not in item for item in transport["cache"].values())
        assert not errors
    finally:
        page.close()


def test_autonomous_mode_is_hidden_when_server_flag_is_off():
    from local_app.config import Settings
    assert Settings().autonomous_division_enabled is False


def test_excluded_student_cannot_select_or_start_autonomous_mode(chromium, ui_server):
    from test_ui_acceptance import _login

    page = chromium.new_page()
    autonomous_requests = []

    def excluded_account(route):
        response = route.fetch()
        body = response.json()
        body["features"]["autonomous_division"] = False
        route.fulfill(response=response, json=body)

    page.route("**/auth/me", excluded_account)
    page.on("request", lambda request: autonomous_requests.append(request.url) if
        "/guided-division?" in request.url and "autonomous-division-v1" in request.url or
        request.method == "POST" and ("/v1/chat/completions" in request.url or "/learning-sessions" in request.url) else None)
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        expect(page.locator("#send-chat")).to_be_enabled()
        option = page.locator('#chat-mode option[value="autonomous_division"]')
        expect(option).to_be_hidden()
        expect(option).to_be_disabled()
        page.locator("#chat-mode").evaluate("node => { node.value = 'autonomous_division'; node.dispatchEvent(new Event('change', {bubbles:true})); }")
        expect(page.locator("#chat-mode")).to_have_value("chat")
        expect(page.locator("#autonomous-division-panel")).to_be_hidden()
        assert not autonomous_requests and not ui_server.store.learning_sessions and not ui_server.rag.calls
    finally:
        page.close()


def test_revoked_autonomous_session_stays_readable_and_can_be_closed(chromium, ui_server):
    """A restored autonomous session cannot silently turn into ordinary chat."""
    from test_ui_acceptance import _login

    session = ui_server.store.create_learning_session("demo-nika-04", "student-ui",
        "math.g4.numbers.division_by_1_2_digit", client_session_id="revoked-session",
        mode="autonomous_division")
    sid = session["learning_session_id"]
    ui_server.store.reserve_chat_message(sid, "demo-nika-04", "saved-answer", "Сделаю паузу.", "student-ui")
    ui_server.store.complete_chat_message(sid, "saved-answer",
        {"choices": [{"message": {"role": "assistant", "content": "Место остановки сохранено."}}]})
    page = chromium.new_page()
    reads, writes, errors = [], [], []

    def excluded_account(route):
        response = route.fetch()
        body = response.json()
        body["features"]["autonomous_division"] = False
        route.fulfill(response=response, json=body)

    page.route("**/auth/me", excluded_account)
    page.on("request", lambda request: reads.append(request.url) if "/guided-division?" in request.url else None)
    page.on("request", lambda request: writes.append(request.url) if request.method == "POST" and
        ("/v1/chat/completions" in request.url or "/learning-sessions" in request.url) else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        expect(page.locator("#chat-mode")).to_have_value("autonomous_division")
        expect(page.locator("#chat-mode")).to_be_disabled()
        expect(page.locator("#chat-form")).to_be_hidden()
        expect(page.locator("#autonomous-division-panel")).to_be_visible()
        expect(page.locator("#autonomous-division-status")).to_contain_text("недоступна для этой учётной записи")
        expect(page.locator("#autonomous-division-start")).to_be_disabled()
        expect(page.locator("#chat-log")).to_contain_text("Место остановки сохранено")
        expect(page.locator("#complete-learning-session")).to_be_enabled()

        # Even a stale form event cannot submit an ordinary question in this session.
        page.locator("#chat-input").evaluate("node => { node.value = 'Объясни деление'; node.form.dispatchEvent(new Event('submit', {bubbles:true, cancelable:true})); }")
        page.locator("#autonomous-division-start").dispatch_event("click")
        page.locator("#complete-learning-session").click()
        expect(page.locator("#toast")).to_contain_text("Автономная практика сохранена")
        assert ui_server.store.learning_sessions[sid]["status"] == "completed"
        assert len(writes) == 1 and writes[0].endswith(f"/learning-sessions/{sid}/complete")
        assert not reads and not ui_server.rag.calls and not ui_server.store.memory_jobs

        page.locator("#learning-archive > summary").click()
        page.locator("#learning-archive-list button").first.click()
        expect(page.locator("#learning-archive-log")).to_contain_text("Место остановки сохранено")
        assert not errors
    finally:
        page.close()


def test_stale_tab_loses_autonomous_actions_without_retry_or_chat_fallback(chromium, ui_server):
    from test_ui_acceptance import _login

    page = chromium.new_page()
    transport = install_autonomous_transport(page, ui_server)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        page.locator("#chat-mode").select_option("autonomous_division")
        page.locator("#autonomous-division-digits").select_option("2")
        page.locator("#autonomous-division-start").click()
        expect(page.locator("#autonomous-division-explain")).to_be_enabled()
        rejected = []

        def reject_action(route):
            rejected.append(route.request.post_data_json)
            route.fulfill(status=404, json={"detail": "autonomous division is not enabled for this account"})

        page.route("**/v1/chat/completions", reject_action)
        page.locator("#autonomous-division-explain").click()
        expect(page.locator("#autonomous-division-status")).to_contain_text("недоступна для этой учётной записи")
        expect(page.locator("#chat-form")).to_be_hidden()
        expect(page.locator("#chat-mode")).to_have_value("autonomous_division")
        expect(page.locator("#autonomous-division-explain")).to_be_disabled()
        expect(page.locator("#retry-chat")).to_be_hidden()
        expect(page.locator("#complete-learning-session")).to_be_enabled()
        page.locator("#retry-chat").dispatch_event("click")
        page.locator("#complete-learning-session").click()
        expect(page.locator("#toast")).to_contain_text("Автономная практика сохранена")
        assert len(rejected) == 1 and len(transport["sent"]) == 1
        assert not ui_server.rag.calls and not ui_server.store.memory_jobs and not errors
    finally:
        page.close()
