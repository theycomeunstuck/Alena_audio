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
