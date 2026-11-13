"""Chromium exercises the teacher-only memory diagnostics UI against real HTTP."""
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from playwright.sync_api import expect

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from tests.test_memory_api_flow import InspectableRag, flow_store  # noqa: F401
from tests.test_memory_diagnostics_api import (DiagnosticsProvider, LEARNER, PASSWORD, TOPIC,
                                               _approve_personal_memory, _lesson)
from tests.test_ui_acceptance import _login


@pytest.fixture
def diagnostics_server(flow_store, tmp_path):
    store = flow_store
    teacher = store.create_account("diagnostics-teacher@synthetic.test", "Teacher",
                                   hash_password(PASSWORD), "teacher")
    store.assign_learner(teacher["id"], LEARNER)
    store.create_student_account("diagnostics-student@synthetic.test", "Student",
                                 hash_password(PASSWORD), LEARNER, teacher["id"])
    _lesson(store)
    _approve_personal_memory(store)
    settings = Settings(personal_memory_enabled=True, context_tokens=16384, session_hours=2,
                        personal_graph_password_file=str(tmp_path / "absent-neo4j-password.txt"))
    runtime = SimpleNamespace(settings=settings, context_tokens=settings.context_tokens)
    app = create_app(settings=settings, store=store, runtime=runtime, rag_service=InspectableRag(),
                     personal_memory_factory=lambda bound_store, bound_runtime:
                         DiagnosticsProvider(bound_store, bound_runtime))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0,
                                           log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("local diagnostics test server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield SimpleNamespace(store=store, app=app, base_url=f"http://127.0.0.1:{port}")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_teacher_runs_memory_diagnostics_and_sees_channels(chromium, diagnostics_server):
    case = diagnostics_server
    page = chromium.new_page(viewport={"width": 1440, "height": 1050})
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    try:
        _login(page, case.base_url, "diagnostics-teacher@synthetic.test", PASSWORD)
        page.locator(f'.learner-item[data-id="{LEARNER}"]').click()
        section = page.locator("#memory-diagnostics-section")
        expect(section).to_be_visible()
        expect(section.locator("#memory-diagnostics-title")).to_have_text("Диагностика памяти")
        index_card = section.locator("#memory-diagnostics-index")
        expect(index_card.locator("table.diagnostics-table")).to_be_visible()
        expect(index_card).to_contain_text("Статус графового индекса")
        expect(index_card).to_contain_text("Не готов – причина")
        expect(index_card).to_contain_text("Персональная память включена")
        expect(section.locator("#memory-diagnostics-status")).to_contain_text("Диагностика")

        page.locator("#diagnostics-query").fill("какие дроби мы сравнивали")
        page.locator("#diagnostics-mode").select_option("hybrid")
        with page.expect_response(lambda response:
                                  "/memory/diagnostics/retrieval" in response.url) as retrieval:
            page.locator("#run-memory-diagnostics").click()
        assert retrieval.value.status == 200, retrieval.value.text()
        results = page.locator("#memory-diagnostics-results")
        expect(results.locator("table.diagnostics-table")).to_be_visible()
        expect(results).to_contain_text("Каналы поиска")
        expect(results).to_contain_text("Хронология")
        expect(results).to_contain_text("Графовый канал не использован – причина")
        expect(results).to_contain_text("Выбранный контекст")
        assert page_errors == []
    finally:
        page.close()


def test_diagnostics_section_stays_hidden_for_students(chromium, diagnostics_server):
    case = diagnostics_server
    page = chromium.new_page(viewport={"width": 1440, "height": 1050})
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    try:
        _login(page, case.base_url, "diagnostics-student@synthetic.test", PASSWORD)
        expect(page.locator("#memory-diagnostics-section")).to_be_hidden()
        assert page_errors == []
    finally:
        page.close()
