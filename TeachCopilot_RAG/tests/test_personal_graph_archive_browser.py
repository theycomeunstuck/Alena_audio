"""Chromium exercises the real review UI, HTTP routes, and PostgreSQL store."""
import os
from pathlib import Path
import threading
import time

import pytest
import uvicorn
from playwright.sync_api import expect

from tests.test_personal_graph_archive_api import archive_case, LEARNER, OTHER_TOPIC, PASSWORD  # noqa: F401
from tests.test_ui_acceptance import _login
from tests.test_memory_api_flow import flow_store  # noqa: F401


@pytest.fixture
def live_archive(archive_case):
    server = uvicorn.Server(uvicorn.Config(archive_case.app, host='127.0.0.1', port=0,
                                            log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail('local archive test server did not start')
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield archive_case, f'http://127.0.0.1:{port}'
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _screenshot(page, name):
    directory = os.getenv("PERSONAL_GRAPH_SCREENSHOT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(path / name), full_page=True)


def _open(page, url):
    _login(page, url, 'archive-teacher@synthetic.test', PASSWORD)
    page.locator(f'.learner-item[data-id="{LEARNER}"]').click()
    version = page.locator('#memory-history .memory-version').first
    expect(version).to_be_visible()
    version.locator('summary').first.click()
    panel = version.locator('section.memory-record')
    return version, panel


@pytest.mark.parametrize('full', [False, True])
def test_teacher_reviews_and_confirms_sources(chromium, live_archive, full):
    case, url = live_archive
    page = chromium.new_page(viewport={'width': 1440, 'height': 1050})
    try:
        version, panel = _open(page, url)
        checks = panel.locator('input[type=checkbox]')
        expect(checks.nth(0)).to_be_disabled()
        expect(checks.nth(1)).to_be_disabled()
        panel.locator('button').click()
        expect(panel.locator('.memory-archive-preview')).to_contain_text('SELECTED SYNTHETIC SOURCE')
        expect(panel.locator('.memory-archive-preview')).to_contain_text('OLD SYNTHETIC SOURCE')
        expect(panel.locator('.memory-archive-preview summary')).to_have_text('Доступные исходники графа (2)')
        expect(checks.nth(0)).to_be_enabled()
        expect(checks.nth(1)).to_be_disabled()
        if full:
            checks.nth(0).check()
            expect(checks.nth(1)).to_be_enabled()
            checks.nth(1).check()
        else:
            _screenshot(page, 'selected-context-review.png')
        with page.expect_response(lambda response: response.url.endswith('/confirm') and response.request.method == 'POST') as response:
            version.get_by_role('button', name='Подтвердить память для помощника').click()
        assert response.value.status == 200, response.value.text()
        sent = response.value.request.post_data_json
        assert sent['reviewed_full_archive'] is full and sent['allow_full_archive'] is full
        assert ('archive_snapshot_fingerprint' in sent) is full
        if full:
            assert sent['archive_snapshot_fingerprint'] == case.fingerprint
        access = response.value.json()['version']['context']['source_access']
        assert access['mode'] == ('full_archive' if full else 'selected_context')
        allowed = set(access['source_ids']) if full else set(response.value.json()['version']['context']['evidence_ids'])
        assert (case.old['source_id'] in allowed) == full
        expect(page.locator('#memory-history .memory-version > summary').first).to_contain_text('подтверждена')
        if full:
            version = page.locator('#memory-history .memory-version').first
            if version.get_attribute('open') is None:
                version.locator('summary').first.click()
            version.locator('button').first.click(timeout=3000)
            expect(version.locator('.memory-archive-preview')).to_contain_text('OLD SYNTHETIC SOURCE')
            expect(version.locator('.memory-archive-preview summary')).to_have_text('Разрешённые исходники графа (2)')
            _screenshot(page, 'full-archive-approved.png')
    finally:
        page.close()


def test_changed_snapshot_is_not_accepted_in_review(chromium, live_archive):
    case, url = live_archive
    page = chromium.new_page(viewport={'width': 1440, 'height': 1050})
    try:
        _, panel = _open(page, url)
        case.store.add_memory_source(LEARNER, OTHER_TOPIC, 'NEW SYNTHETIC SOURCE', 'teacher', author_role='teacher', approved=True)
        panel.locator('button').click()
        expect(panel.locator('[role=status]')).to_contain_text('не совпадает')
        expect(panel.locator('input[type=checkbox]').nth(0)).to_be_disabled()
        expect(panel.locator('input[type=checkbox]').nth(1)).to_be_disabled()
        assert panel.locator('.memory-archive-preview').inner_text() == ''
        _screenshot(page, 'stale-archive-review.png')
    finally:
        page.close()


def test_graph_draft_renders_structured_sources_instead_of_serialized_context(chromium, live_archive):
    _, url = live_archive
    page = chromium.new_page(viewport={'width': 1440, 'height': 1050})

    def memory(route):
        response = route.fetch()
        data = response.json()
        version = data['versions'][0]
        version['context'].update({
            'text': '{"raw_marker":"SHOULD_NOT_RENDER"}',
            'sources': [
                {'source_id': 's1', 'role': 'student', 'text': 'Первая реплика', 'corrected_by': ['s2']},
                {'source_id': 's2', 'role': 'teacher', 'text': 'Исправленная реплика',
                 'corrects_source_id': 's1', 'cited': False},
            ],
            'dialogue_context': [{'source_id': 's1', 'role': 'assistant', 'usage': 'context_only',
                                  'text': 'Вопрос перед ответом'}],
            'session_end_context': [{'source_id': 's2', 'role': 'assistant', 'usage': 'context_only',
                                     'text': 'Завершение занятия'}],
        })
        route.fulfill(response=response, json=data)

    try:
        page.route('**/learners/*/memory?*', memory)
        version, _ = _open(page, url)
        expect(version).to_contain_text('Выбрано реплик: 2')
        expect(version).not_to_contain_text('SHOULD_NOT_RENDER')
        sources = version.locator('details.memory-sources').first
        assert sources.get_attribute('open') is not None
        for text in ('Первая реплика', 'Исправленная реплика', 'Вопрос перед ответом',
                     'Завершение занятия', 'Запись исправлена', 'Исправляет источник',
                     'Связанная запись для проверки исправления'):
            expect(sources).to_contain_text(text)
        sources.locator('summary').click()
        sources.locator('summary').click()
        expect(sources).to_contain_text('Исправленная реплика')
    finally:
        page.close()

