"""Both role buttons exercise real local authentication from a browser."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
pytest.importorskip("playwright.sync_api", reason="install requirements-dev-local.txt for browser acceptance tests")
from playwright.sync_api import expect

from local_app.api import create_app
from local_app.auth import hash_password
from tests.test_demo_login import DEMO_PASSWORD, demo_credentials
from tests.test_ui_acceptance import BrowserRag, BrowserRuntime, BrowserStore, _install_pipeline_stubs, _login, _screenshot, ui_server


@pytest.fixture
def demo_ui_server(monkeypatch, tmp_path):
    import uvicorn

    _install_pipeline_stubs(monkeypatch)
    store = BrowserStore()
    for old_email, email in (("teacher.ui@example.test", "demo.rich.teacher@local.test"), ("student.ui@example.test", "demo.nika04@local.test")):
        account = store.accounts.pop(old_email)
        account.update(email=email, password_hash=hash_password(DEMO_PASSWORD))
        store.accounts[email] = account
    store.accounts["demo.rich.teacher@local.test"]["payload"] = {"synthetic_demo": True}
    credentials = demo_credentials(tmp_path / "demo-credentials.json")
    settings = SimpleNamespace(session_hours=2, rag_limit=3, demo_login_enabled=True, demo_credentials_file=str(credentials))
    app = create_app(settings=settings, store=store, runtime=BrowserRuntime(), rag_service=BrowserRag())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("local demo API test server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield SimpleNamespace(base_url=f"http://127.0.0.1:{port}", store=store)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_demo_buttons_enter_actual_teacher_and_student_accounts(chromium, demo_ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(demo_ui_server.base_url, wait_until="networkidle")
    expect(page.locator("#demo-login")).to_be_visible()
    _screenshot(page, "role-login-desktop.png")
    with page.expect_response(lambda response: response.url.endswith("/auth/demo/teacher")) as login:
        page.get_by_role("button", name="Войти как учитель").click()
    assert login.value.status == 200
    assert login.value.request.post_data_json == {}
    assert login.value.request.headers["x-teachcopilot-demo"] == "1"
    expect(page.locator("#workspace")).to_be_visible()
    expect(page.locator("#role-label")).to_have_text("Учитель")
    expect(page.locator("#learner-list .learner-item")).to_have_count(6)
    expect(page.locator("#teacher-actions")).to_be_visible()
    page.get_by_role("button", name="Выйти").click()
    expect(page.locator("#email")).to_have_value("")
    expect(page.locator("#password")).to_have_value("")
    with page.expect_response(lambda response: response.url.endswith("/auth/demo/student")) as login:
        page.get_by_role("button", name="Войти как ученик").click()
    assert login.value.status == 200
    expect(page.locator("#role-label")).to_have_text("Ученик")
    expect(page.locator("#learner-list .learner-item")).to_have_count(1)
    expect(page.locator("#learner-name")).to_have_text(demo_ui_server.store.cards["demo-nika-04"]["pseudonym"])
    expect(page.locator("#teacher-actions")).to_be_hidden()
    expect(page.locator("#student-actions")).to_be_visible()
    assert DEMO_PASSWORD not in page.content()
    assert not errors, errors
    page.close()


def test_demo_disabled_keeps_ordinary_password_form(chromium, ui_server):
    page = chromium.new_page()
    page.goto(ui_server.base_url, wait_until="networkidle")
    expect(page.locator("#demo-login")).to_be_hidden()
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    expect(page.locator("#role-label")).to_have_text("Учитель")
    page.close()


def test_demo_failure_leaves_form_and_other_button_usable(chromium, demo_ui_server):
    page = chromium.new_page()
    page.goto(demo_ui_server.base_url, wait_until="networkidle")
    expect(page.locator("#demo-login")).to_be_visible()
    demo_ui_server.store.accounts["demo.rich.teacher@local.test"]["active"] = False
    page.get_by_role("button", name="Войти как учитель").click()
    expect(page.locator("#login-error")).to_be_visible()
    expect(page.locator("#email")).to_be_enabled()
    expect(page.get_by_role("button", name="Войти как ученик")).to_be_enabled()
    page.get_by_role("button", name="Войти как ученик").click()
    expect(page.locator("#role-label")).to_have_text("Ученик")
    page.close()


def test_demo_buttons_fit_mobile_and_accept_keyboard(chromium, demo_ui_server):
    page = chromium.new_page(viewport={"width": 320, "height": 740}, is_mobile=True, has_touch=True)
    page.goto(demo_ui_server.base_url, wait_until="networkidle")
    expect(page.locator("#demo-login")).to_be_visible()
    student = page.get_by_role("button", name="Войти как ученик")
    student.focus()
    expect(student).to_be_focused()
    _screenshot(page, "role-login-mobile.png")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    student.press("Enter")
    expect(page.locator("#role-label")).to_have_text("Ученик")
    expect(page.locator("#teacher-actions")).to_be_hidden()
    page.close()
