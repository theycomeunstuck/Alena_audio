"""Explicit child topic selection uses the real API with model-free browser fixtures."""
import pytest

pytest.importorskip("playwright.sync_api", reason="install requirements-dev-local.txt for browser acceptance tests")
from playwright.sync_api import expect  # noqa: E402

from test_ui_acceptance import _login, ui_server  # noqa: E402


FRACTIONS = "math.g4.fractions.numerator_denominator"
PYTHON = "programming.python.output"


def _send(page, content):
    page.locator("#chat-input").fill(content)
    with page.expect_response(lambda response: response.url.endswith("/v1/chat/completions")) as response:
        page.locator("#send-chat").click()
    assert response.value.status == 200, response.value.text()
    expect(page.locator("#send-chat")).to_be_enabled()


def _complete(page):
    with page.expect_response(lambda response: response.url.endswith("/complete")) as response:
        page.locator("#complete-learning-session").click()
    assert response.value.status == 202


def test_visible_goal_topic_is_saved_and_locked_until_a_new_session(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 1050})
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        expect(page.locator("#chat-topic")).to_have_value(FRACTIONS)
        expect(page.locator("#chat-topic option:checked")).to_contain_text("Дробь, числитель и знаменатель")
        _send(page, "хз")
        session = next(iter(ui_server.store.learning_sessions.values()))
        assert session["topic_id"] == FRACTIONS
        assert ui_server.rag.last_answer_kwargs["topic_id"] == FRACTIONS
        assert ui_server.rag.last_answer_kwargs["subject_id"] == "math"
        expect(page.locator("#chat-topic")).to_be_disabled()
        expect(page.locator("#chat-subject")).to_be_disabled()
        _complete(page)
        expect(page.locator("#chat-topic")).to_be_enabled()
        page.locator("#chat-subject").select_option("python")
        # A math goal must not silently label a Python conversation as fractions.
        expect(page.locator("#chat-topic")).to_have_value("python")
        assert not page.locator(f'#chat-topic option[value="{FRACTIONS}"]').count()
        page.locator("#chat-topic").select_option(PYTHON)
        _send(page, "ну вроде")
        assert list(ui_server.store.learning_sessions.values())[-1]["topic_id"] == PYTHON
        assert ui_server.rag.last_answer_kwargs["subject_id"] == "python"
        _complete(page)
        page.locator("#chat-subject").select_option("math")
        page.locator("#chat-topic").select_option("math")
        _send(page, "а можно просто поболтать о математике")
        assert list(ui_server.store.learning_sessions.values())[-1]["topic_id"] == "math"
    finally:
        page.close()


def test_late_catalog_does_not_override_an_explicit_general_conversation(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 1050})
    held = []
    page.route("**/catalog", lambda route: held.append(route))
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        expect(page.locator("#chat-topic")).to_have_value(FRACTIONS)
        page.locator("#chat-topic").select_option("math")
        assert held
        with page.expect_response(lambda response: response.url.endswith("/catalog")):
            held.pop().continue_()
        expect(page.locator(f'#chat-topic option[value="{FRACTIONS}"]')).to_have_count(1)
        expect(page.locator("#chat-topic")).to_have_value("math")
        _send(page, "ага")
        assert next(iter(ui_server.store.learning_sessions.values()))["topic_id"] == "math"
        page.get_by_role("button", name="Выйти").click()
        _login(page, ui_server.base_url, "other.student@example.test")
        # Logging out clears choices; another child's goal is never the prior selection.
        own_goal = ui_server.store.cards["demo-luna-03"]["rag_context"]["current_goal"]["topic_id"]
        if own_goal.startswith("math."):
            expect(page.locator("#chat-topic")).to_have_value(own_goal)
    finally:
        page.close()
