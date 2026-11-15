"""Student action flows through the real local API and static frontend.

The shared browser fixture supplies an isolated in-memory store and records
RAG calls. Only the lost-response test intercepts transport, after the real
server has already completed and cached the requested action.
"""
from __future__ import annotations

import copy
import re

import pytest

pytest.importorskip("playwright.sync_api", reason="install requirements-dev-local.txt for browser acceptance tests")
from playwright.sync_api import expect  # noqa: E402

from test_ui_acceptance import _login, _screenshot, ui_server  # noqa: E402


CHAT_PATH = "/v1/chat/completions"


def _record_actions(page):
    sent = []
    page.on(
        "request",
        lambda request: sent.append(copy.deepcopy(request.post_data_json))
        if request.url.endswith(CHAT_PATH) and request.method == "POST"
        else None,
    )
    return sent


def _choose_action(page, operation):
    page.locator("#chat-mode").select_option(operation)
    expect(page.locator("#learning-action-form")).to_be_visible()
    # Form fields stay disabled until the session-recovery state finishes
    # loading; one lost request under load used to block them for good.
    try:
        expect(page.locator("#action-dividend")).to_be_enabled(timeout=10000)
    except AssertionError:
        page.reload()
        expect(page.locator("#chat-mode")).to_be_visible(timeout=15000)
        page.locator("#chat-mode").select_option(operation)
        expect(page.locator("#learning-action-form")).to_be_visible()
        expect(page.locator("#action-dividend")).to_be_enabled(timeout=15000)


def _submit_action(page):
    with page.expect_response(lambda response: response.url.endswith(CHAT_PATH)) as response:
        page.locator("#send-learning-action").click()
    assert response.value.status == 200, response.value.text()
    expect(page.locator("#send-learning-action")).to_be_enabled()
    return response.value.json()


def test_student_can_run_four_actions_and_start_python_after_math_session(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    sent = _record_actions(page)
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        expect(page.locator("#chat-mode")).to_have_value("chat")
        expect(page.locator("#learning-action-form")).to_be_hidden()

        _choose_action(page, "division_check")
        page.locator("#action-dividend").fill("7560")
        page.locator("#action-divisor").fill("24")
        page.locator("#action-quotient").fill("315")
        checked = _submit_action(page)
        assistant = page.locator("#chat-log .chat-bubble.assistant").last
        expect(assistant).to_contain_text("7560 : 24 = 315")
        expect(assistant).to_contain_text("315 × 24 = 7560")
        assert checked["learning_tool"]["operation"] == "division_check"
        assert checked["learning_tool"]["status"] == "verified"
        assert checked["learning_tool"]["facts"]["quotient"] == 315
        expect(page.locator("#chat-subject")).to_be_disabled()

        _choose_action(page, "division_places")
        page.locator("#action-dividend").fill("6750")
        page.locator("#action-divisor").fill("25")
        places = _submit_action(page)
        assistant = page.locator("#chat-log .chat-bubble.assistant").last
        expect(assistant).to_contain_text("6750 : 25 = 270")
        expect(assistant).to_contain_text("2 × 100 = 200")
        expect(assistant).to_contain_text("7 × 10 = 70")
        expect(assistant).to_contain_text("0 × 1 = 0")
        assert places["learning_tool"]["operation"] == "division_places"

        _choose_action(page, "division_practice")
        page.locator("#action-divisor-digits").select_option("2")
        practice = _submit_action(page)
        facts = practice["learning_tool"]["facts"]
        quotient, remainder = divmod(facts["dividend"], facts["divisor"])
        assert 10 <= facts["divisor"] <= 99
        assert remainder == 0 and re.fullmatch(r"[1-9]0[1-9]", str(quotient))
        assistant = page.locator("#chat-log .chat-bubble.assistant").last
        expect(assistant).to_contain_text(str(facts["dividend"]))
        expect(assistant).to_contain_text(str(facts["divisor"]))
        assert not re.search(rf"=\s*{quotient}\b", assistant.inner_text()), "practice must leave the answer for the student"
        assert "quotient" not in facts
        assert len(ui_server.store.learning_sessions) == 1
        math_session = next(iter(ui_server.store.learning_sessions))
        assert len(ui_server.store.learning_turns[math_session]) == 6
        _screenshot(page, "student-explicit-math-actions.png")

        with page.expect_response(lambda response: response.url.endswith("/complete")) as completed:
            page.get_by_role("button", name="Завершить занятие").click()
        assert completed.value.status == 202
        expect(page.locator("#chat-subject")).to_be_enabled()
        assert ui_server.store.learning_sessions[math_session]["status"] == "completed"
        page.locator("#chat-subject").select_option("python")
        _choose_action(page, "print_output")
        code = "print('\"да\"', 'нет', sep=' | ', end='!')\nprint('ещё', end='')"
        page.locator("#action-code").fill(code)
        printed = _submit_action(page)
        expected_stdout = '"да" | нет!ещё'
        assert printed["learning_tool"]["facts"]["stdout"] == expected_stdout
        expect(page.locator("#chat-log pre.learning-tool-stdout").last).to_have_text(expected_stdout)
        expect(page.locator("#chat-log")).not_to_contain_text("7560 : 24")
        expect(page.locator("#chat-log .chat-bubble.user")).to_have_count(1)
        assert len(ui_server.store.learning_sessions) == 2
        assert sent[-1]["learning_session_id"] != math_session
        assert sent[-1]["metadata"]["subject_id"] == "python"
        assert [request["learning_action"] for request in sent] == [
            {"operation": "division_check", "dividend": 7560, "divisor": 24, "stated_quotient": 315},
            {"operation": "division_places", "dividend": 6750, "divisor": 25},
            {"operation": "division_practice", "divisor_digits": 2},
            {"operation": "print_output", "code": code},
        ]
        assert all(request["messages"] == [] for request in sent)
        assert ui_server.rag.calls == [], "explicit actions must not invoke the chat model"
        assert not errors, errors
        _screenshot(page, "student-explicit-print-action.png")
    finally:
        page.close()


def test_invalid_division_and_unsupported_print_do_not_produce_an_answer(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    sent = _record_actions(page)
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        _choose_action(page, "division_check")
        page.locator("#action-dividend").fill("7560")
        page.locator("#action-divisor").fill("0")
        page.locator("#send-learning-action").click()
        invalid_field = page.locator("#action-divisor").evaluate("element => !element.checkValidity()")
        visible_error = page.locator("#learning-action-error").is_visible() and bool(page.locator("#learning-action-error").inner_text())
        assert invalid_field or visible_error, "invalid operands need visible form feedback"
        assert sent == []
        assert ui_server.store.learning_sessions == {}
        expect(page.locator("#chat-log .chat-bubble.assistant")).to_have_count(1)

        page.locator("#chat-subject").select_option("python")
        _choose_action(page, "print_output")
        page.locator("#action-code").fill("print(1 + 2)")
        with page.expect_response(lambda response: response.url.endswith(CHAT_PATH)) as rejected:
            page.locator("#send-learning-action").click()
        assert rejected.value.status == 422, rejected.value.text()
        expect(page.locator("#learning-action-error")).to_be_visible()
        expect(page.locator("#learning-action-error")).not_to_have_text("")
        expect(page.locator("#action-code")).to_be_enabled()
        expect(page.locator("#send-learning-action")).to_be_enabled()
        expect(page.locator("#chat-log .chat-bubble.assistant")).to_have_count(1)
        expect(page.locator("#chat-log pre.learning-tool-stdout")).to_have_count(0)
        assert ui_server.store.learning_messages == {}
        assert ui_server.rag.calls == []

        # Correct the rejected action without reloading or leaving a stuck retry.
        page.locator("#action-code").fill("print('Исправлено', end='')")
        _submit_action(page)
        expect(page.locator("#learning-action-error")).to_be_hidden()
        expect(page.locator("#chat-retry")).to_be_hidden()
        expect(page.locator("#chat-log pre.learning-tool-stdout").last).to_have_text("Исправлено")
        assert len(ui_server.store.learning_messages) == 1
        assert ui_server.rag.calls == []
    finally:
        page.close()


def test_lost_completed_action_response_retries_the_same_payload_and_cached_turn(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    sent, server_results = [], []

    def lose_first_response(route):
        sent.append(copy.deepcopy(route.request.post_data_json))
        response = route.fetch()
        assert response.status == 200, response.text()
        server_results.append(response.json())
        if len(sent) == 1:
            route.abort("failed")
        else:
            route.fulfill(response=response)

    page.route("**/v1/chat/completions", lose_first_response)
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        _choose_action(page, "division_check")
        page.locator("#action-dividend").fill("7560")
        page.locator("#action-divisor").fill("24")
        page.locator("#action-quotient").fill("315")
        page.locator("#send-learning-action").click()
        expect(page.locator("#chat-retry")).to_be_visible()
        expect(page.locator("#retry-chat")).to_be_visible()
        expect(page.locator("#chat-mode")).to_be_disabled()
        expect(page.locator("#action-dividend")).to_be_disabled()
        expect(page.locator("#chat-subject")).to_be_disabled()
        expect(page.locator("#complete-learning-session")).to_be_disabled()
        expect(page.locator("#chat-log .chat-bubble.user")).to_have_count(1)
        assert len(ui_server.store.learning_messages) == 1
        cached = next(iter(ui_server.store.learning_messages.values()))
        assert cached["status"] == "completed", "the server must finish before the browser loses its response"

        page.locator("#retry-chat").click()
        expect(page.locator("#chat-retry")).to_be_hidden()
        expect(page.locator("#chat-log .chat-bubble.assistant").last).to_contain_text("7560 : 24 = 315")
        expect(page.locator("#chat-mode")).to_be_enabled()
        expect(page.locator("#send-learning-action")).to_be_enabled()
        expect(page.locator("#complete-learning-session")).to_be_enabled()
        assert len(sent) == 2 and sent[0] == sent[1]
        assert sent[0]["messages"] == []
        assert sent[0]["learning_action"] == {
            "operation": "division_check", "dividend": 7560, "divisor": 24, "stated_quotient": 315,
        }
        assert server_results[0] == server_results[1]
        assert len(ui_server.store.learning_sessions) == 1
        assert len(ui_server.store.learning_messages) == 1
        assert len(next(iter(ui_server.store.learning_turns.values()))) == 2
        assert ui_server.rag.calls == []
    finally:
        page.close()


def test_student_action_works_with_keyboard_on_narrow_mobile_screen(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        page.get_by_role("button", name="Занятие", exact=True).click()
        expect(page.locator("#student-actions")).to_be_visible()
        page.locator("#chat-mode").focus()
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Enter")
        expect(page.locator("#chat-mode")).to_have_value("division_check")
        page.locator("#action-dividend").focus()
        page.keyboard.type("7560")
        page.keyboard.press("Tab")
        expect(page.locator("#action-divisor")).to_be_focused()
        page.keyboard.type("24")
        page.keyboard.press("Tab")
        expect(page.locator("#action-quotient")).to_be_focused()
        page.keyboard.type("315")
        page.keyboard.press("Tab")
        expect(page.locator("#send-learning-action")).to_be_focused()
        with page.expect_response(lambda response: response.url.endswith(CHAT_PATH)) as checked:
            page.keyboard.press("Enter")
        assert checked.value.status == 200, checked.value.text()
        expect(page.locator("#chat-log .chat-bubble.assistant").last).to_contain_text("7560 : 24 = 315")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        _screenshot(page, "student-explicit-action-mobile-390.png")
        page.set_viewport_size({"width": 320, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        _screenshot(page, "student-explicit-action-mobile-320.png")
        assert ui_server.rag.calls == []
        assert not errors, errors
    finally:
        page.close()


def test_print_is_plain_text_and_action_fields_are_cleared_on_logout(chromium, ui_server):
    page = chromium.new_page(viewport={'width': 1365, 'height': 1000})
    try:
        _login(page, ui_server.base_url, 'student.ui@example.test')
        page.locator('#chat-subject').select_option('python')
        _choose_action(page, 'print_output')
        code = "print('<img src=x onerror=window.actionInjected=true>\\n\\n', end='')"
        page.locator('#action-code').fill(code)
        body = _submit_action(page)
        stdout = '<img src=x onerror=window.actionInjected=true>\n\n'
        assert body['learning_tool']['facts']['stdout'] == stdout
        assert page.locator('pre.learning-tool-stdout').text_content() == stdout
        assert page.locator('#chat-log img').count() == 0
        assert page.evaluate('window.actionInjected === undefined')
        page.locator('#logout').click()
        _login(page, ui_server.base_url, 'student.ui@example.test')
        assert page.locator('#action-code').input_value() == ''
        expect(page.locator('#chat-mode')).to_have_value('chat')
        assert ui_server.rag.calls == []
    finally:
        page.close()
