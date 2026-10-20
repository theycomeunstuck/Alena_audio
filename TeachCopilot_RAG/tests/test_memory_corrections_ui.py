"""Teacher correction forms preserve source text and idempotent request identity."""
import copy
from uuid import UUID

import pytest

from tests.test_ui_acceptance import _login, expect, ui_server


TOPIC = "math.g4.fractions.numerator_denominator"
PATH = "/learners/demo-nika-04/memory/sources/source-student/corrections"
SOURCE_TEXT = '<img src=x onerror="window.__correction_xss=1"> ну хз'


def _memory_payload(**changes):
    payload = {
        "revision": 7, "source_revision": 11, "sources": [
            {"source_id": "source-student", "author_role": "student", "text": SOURCE_TEXT},
            {"source_id": "source-teacher", "role": "teacher", "text": "В исходной записи ошибка."},
            {"source_id": "source-assistant", "role": "assistant", "text": "Посчитаем части."},
            {"source_id": "source-user-alias", "role": "user", "text": "Старая запись без роли источника."},
        ], "versions": [],
    }
    payload.update(changes)
    return payload


def _open_teacher(page, ui_server):
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
    with page.expect_response(lambda response: "/demo-nika-04/memory?topic_id=" in response.url):
        page.get_by_label("Тема", exact=True).select_option(TOPIC)
    expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
    page.locator("#memory-history .memory-sources summary").click()
    return page.locator("#memory-history .memory-source").first


def _edit(row, text):
    row.get_by_role("button", name="Исправить запись", exact=True).click()
    field = row.get_by_label("Исправление записи", exact=True)
    field.fill(text)
    return field, row.get_by_role("button", name="Сохранить исправление", exact=True)


def _roundtrip_topic(page):
    with page.expect_response(lambda response: response.url.endswith("/demo-nika-04/memory")):
        page.get_by_label("Тема", exact=True).select_option("")
    expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
    with page.expect_response(lambda response: "/demo-nika-04/memory?topic_id=" in response.url):
        page.get_by_label("Тема", exact=True).select_option(TOPIC)
    expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
    page.locator("#memory-history .memory-sources summary").click()


def test_teacher_correction_preserves_text_sends_revisions_and_keeps_one_call_in_flight(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload, pending, sent, all_posts = _memory_payload(), [], [], []
    before = copy.deepcopy(payload)

    def save(route):
        sent.append(route.request.post_data_json)
        pending.append(route)

    page.route("**/learners/*/memory**", lambda route: route.fulfill(json=payload))
    page.route("**/memory/sources/*/corrections", save)
    page.on("request", lambda request: all_posts.append(request.url) if request.method == "POST" else None)
    try:
        row = _open_teacher(page, ui_server)
        expect(page.get_by_role("button", name="Исправить запись", exact=True)).to_have_count(2)
        expect(row).to_contain_text(SOURCE_TEXT)
        text = '  Исправление: <script>window.__correction_xss=2</script>\nПомощь неизвестна.  '
        field, button = _edit(row, text)
        expect(field).to_have_attribute("maxlength", "2000")
        expect(row).to_contain_text("Исходная запись сохранится")
        expect(row).to_contain_text("не новый результат или прогресс ребёнка")
        with page.expect_request(lambda request: request.url.endswith(PATH)):
            button.click()
        expect(button).to_be_disabled()
        expect(field).to_be_disabled()
        row.locator("form").dispatch_event("submit")
        assert len(sent) == 1
        identifier = sent[0]["client_correction_id"]
        assert str(UUID(identifier)) == identifier
        assert sent == [{"text": text, "expected_revision": 7, "expected_source_revision": 11, "client_correction_id": identifier}]
        assert payload == before
        payload["sources"].append({"source_id": "source-fix", "role": "teacher", "text": text, "corrects_source_id": "source-student"})
        with page.expect_response(lambda response: response.url.endswith(PATH)):
            pending[0].fulfill(status=202, json={"source": payload["sources"][-1], "job_id": "job-fix"})
        expect(page.locator("#memory-status")).to_have_text("Исправление сохранено. Состояние обновления памяти ещё не проверено.")
        expect(page.get_by_role("button", name="Проверить обновление памяти", exact=True)).to_be_visible()
        expect(page.locator("#memory-history")).not_to_contain_text("Обновлённую память нужно подтвердить")
        page.locator("#memory-history .memory-sources summary").click()
        expect(page.locator("#memory-history")).to_contain_text("Исправляет источник source-student. Исходная запись сохранена.")
        expect(page.locator("#memory-history")).to_contain_text(SOURCE_TEXT)
        assert page.locator("#memory-history img, #memory-history script").count() == 0
        assert page.evaluate("window.__correction_xss || null") is None
        assert not any(url.endswith("/confirm") for url in all_posts)
    finally:
        page.close()


def test_uncertain_retry_reuses_identifier_and_changed_text_gets_new_identifier(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    sent = []

    def save(route):
        sent.append(route.request.post_data_json)
        if len(sent) < 3:
            route.abort("failed")
        else:
            route.fulfill(status=202, json={"source": {"source_id": "fix"}, "job_id": "job"})

    page.route("**/learners/*/memory**", lambda route: route.fulfill(json=_memory_payload()))
    page.route("**/memory/sources/*/corrections", save)
    try:
        row = _open_teacher(page, ui_server)
        field, button = _edit(row, "Помощь не была зафиксирована.")
        for _ in range(2):
            button.click()
            expect(row.get_by_role("status")).to_contain_text("Исход запроса может быть неизвестен")
            expect(field).to_be_enabled()
        assert len(sent) == 2
        assert sent[0] == sent[1]
        field.fill("В журнале другой результат; помощь неизвестна.")
        button.click()
        expect(page.locator("#memory-status")).to_contain_text("Исправление сохранено.")
        assert len(sent) == 3
        assert sent[2]["client_correction_id"] != sent[0]["client_correction_id"]
        assert sent[2]["text"] == "В журнале другой результат; помощь неизвестна."
    finally:
        page.close()


def test_revision_conflict_keeps_correction_draft_without_refresh_or_confirmation(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    reads, sent = [], []

    def memory(route):
        reads.append(route.request.url)
        route.fulfill(json=_memory_payload())

    def save(route):
        sent.append(route.request.post_data_json)
        route.fulfill(status=409, json={"detail": "source_revision_changed"})

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", save)
    try:
        row = _open_teacher(page, ui_server)
        field, button = _edit(row, "Ученик сам не исправлял пример – исправлена заметка.")
        before = len(reads)
        button.click()
        expect(row.get_by_role("status")).to_contain_text("Данные изменились")
        expect(row.get_by_role("status")).to_contain_text("Черновик исправления сохранён в форме.")
        expect(field).to_have_value("Ученик сам не исправлял пример – исправлена заметка.")
        expect(button).to_be_enabled()
        assert len(reads) == before
        assert len(sent) == 1
    finally:
        page.close()


def test_delayed_correction_does_not_replace_another_or_reopened_learner_view(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    pending, reads = [], []

    def memory(route):
        reads.append(route.request.url)
        route.fulfill(json=_memory_payload(sources=[] if "/demo-luna-03/" in route.request.url else _memory_payload()["sources"]))

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", lambda route: pending.append(route))
    try:
        row = _open_teacher(page, ui_server)
        _, button = _edit(row, "В журнале была опечатка.")
        with page.expect_request(lambda request: request.url.endswith(PATH)):
            button.click()
        page.locator('.learner-item[data-id="demo-luna-03"]').click()
        expect(page.locator("#memory-history .memory-source")).to_have_count(0)
        page.locator('.learner-item[data-id="demo-nika-04"]').click()
        expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
        page.locator("#memory-history .memory-sources summary").click()
        current = page.locator("#memory-history .memory-source").first
        field, _ = _edit(current, "Новый черновик в открытом профиле.")
        before = len(reads)
        with page.expect_response(lambda response: response.url.endswith(PATH)) as response:
            pending[0].fulfill(status=202, json={"source": {"source_id": "old-fix"}, "job_id": "old-job"})
        response.value.finished()
        expect(field).to_have_value("Новый черновик в открытом профиле.")
        expect(page.locator("#memory-status")).not_to_contain_text("Исправление сохранено.")
        assert len(reads) == before
    finally:
        page.close()


def test_missing_source_revision_keeps_legacy_sources_read_only(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload = _memory_payload()
    del payload["source_revision"]
    page.route("**/learners/*/memory**", lambda route: route.fulfill(json=payload))
    try:
        _open_teacher(page, ui_server)
        expect(page.locator("#memory-history")).to_contain_text(SOURCE_TEXT)
        expect(page.get_by_role("button", name="Исправить запись", exact=True)).to_have_count(0)
    finally:
        page.close()


def test_delayed_correction_does_not_refresh_a_different_topic(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    pending, reads = [], []

    def memory(route):
        reads.append(route.request.url)
        route.fulfill(json=_memory_payload())

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", lambda route: pending.append(route))
    try:
        row = _open_teacher(page, ui_server)
        expect(page.get_by_label("Тема", exact=True)).to_have_value(TOPIC)
        _, button = _edit(row, "Сохранить уточнение прошлого источника.")
        with page.expect_request(lambda request: request.url.endswith(PATH)):
            button.click()
        with page.expect_response(lambda response: response.url.endswith("/demo-nika-04/memory")):
            page.get_by_label("Тема", exact=True).select_option("")
        expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
        before = len(reads)
        with page.expect_response(lambda response: response.url.endswith(PATH)) as response:
            pending[0].fulfill(status=202, json={"source": {"source_id": "fix-old-topic"}, "job_id": "job"})
        response.value.finished()
        expect(page.get_by_label("Тема", exact=True)).to_have_value("")
        expect(page.locator("#memory-status")).not_to_contain_text("Исправление сохранено.")
        assert len(reads) == before
    finally:
        page.close()


def test_recommendation_sources_remain_read_only(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    page.route("**/learners/*/memory**", lambda route: route.fulfill(json=_memory_payload()))
    page.route("**/learners/*/next-step", lambda route: route.fulfill(status=202, json={"job_id": "correction-readonly", "status": "pending"}))
    page.route("**/memory/jobs/correction-readonly", lambda route: route.fulfill(json={"status": "completed", "result": {
        "topic_id": TOPIC, "action": "Уточнить условия помощи.", "reason": "В записи неизвестны условия помощи.",
        "sources": _memory_payload()["sources"], "review_required": True,
    }}))
    try:
        _open_teacher(page, ui_server)
        page.get_by_role("button", name="Предложить следующий шаг").click()
        result = page.locator("#next-step-result")
        expect(result).to_contain_text("Уточнить условия помощи.")
        result.locator("summary").click()
        expect(result).to_contain_text(SOURCE_TEXT)
        expect(result.get_by_role("button", name="Исправить запись", exact=True)).to_have_count(0)
        expect(result.get_by_label("Исправление записи", exact=True)).to_have_count(0)
    finally:
        page.close()


def test_memory_reports_a_correction_group_that_does_not_fit_the_context(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload = _memory_payload(versions=[{
        "version_id": "version-skipped", "method": "tail", "approved": False,
        "context": {"text": "Другая целая запись.", "selection_events": [{
            "event": "source_correction_group_skipped", "source_ids": ["source-student", "fix-1"],
            "reason": "group_exceeds_budget", "raw_diagnostic": "private diagnostic not displayed",
        }]},
    }])
    page.route("**/learners/*/memory**", lambda route: route.fulfill(json=payload))
    try:
        _open_teacher(page, ui_server)
        version = page.locator("#memory-history .memory-version")
        version.locator("summary").click()
        expect(version).to_contain_text("Исходная запись и исправления не поместились в выбранную память. Эта связка не будет передана помощнику.")
        expect(page.locator("#memory-history .memory-source").first).to_contain_text(SOURCE_TEXT)
        expect(version).not_to_contain_text("private diagnostic not displayed")
        expect(version.locator(".memory-record")).to_have_count(0)
    finally:
        page.close()


def test_saved_correction_tracks_queued_running_and_completed_update_without_resubmission(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload, saves, job_checks, reads = _memory_payload(), [], [], []
    statuses = iter(["queued", "running", "completed"])

    def memory(route):
        reads.append(route.request.url)
        route.fulfill(json=payload)

    def save(route):
        saves.append(route.request.post_data_json)
        route.fulfill(status=202, json={"source": {"source_id": "source-fix"}, "job_id": "update-progress", "topic_id": TOPIC})

    def job(route):
        job_checks.append(route.request.method)
        status = next(statuses)
        if status == "completed":
            payload["versions"] = [{"version_id": "new-corrected-version", "method": "tail", "approved": False, "context": {"text": "Запись учителя исправлена; помощь неизвестна."}}]
        route.fulfill(json={"job_id": "update-progress", "status": status})

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", save)
    page.route("**/memory/jobs/update-progress", job)
    try:
        row = _open_teacher(page, ui_server)
        _, button = _edit(row, "Исправлена ошибочная заметка о самостоятельной попытке.")
        button.click()
        expect(page.locator("#memory-status")).to_contain_text("Состояние обновления памяти ещё не проверено.")
        panel = page.locator(".memory-correction-job")
        check = panel.get_by_role("button", name="Проверить обновление памяти", exact=True)
        assert job_checks == [], "saving a correction must not silently poll or rerun its update"
        before = len(reads)
        check.click()
        expect(panel.get_by_role("status")).to_contain_text("Обновление памяти в очереди.")
        expect(page.get_by_role("button", name="Подтвердить память для помощника")).to_have_count(0)
        assert len(reads) == before
        check.click()
        expect(panel.get_by_role("status")).to_contain_text("Память обновляется.")
        assert len(reads) == before
        check.click()
        expect(page.locator("#memory-status")).to_contain_text("Обновление памяти завершено.")
        expect(page.locator("#memory-history .memory-version")).to_have_count(1)
        page.locator("#memory-history .memory-version summary").click()
        expect(page.locator("#memory-history .memory-version")).to_contain_text("Запись учителя исправлена; помощь неизвестна.")
        expect(page.get_by_role("button", name="Подтвердить память для помощника")).to_be_visible()
        assert len(reads) == before + 1
        assert job_checks == ["GET", "GET", "GET"]
        assert len(saves) == 1
    finally:
        page.close()


@pytest.mark.parametrize("job_status, expected", [
    ("failed", "Обновить память не удалось."),
    ("unknown_outcome", "Исход обновления памяти неизвестен."),
    ("unknown", "Исход обновления памяти неизвестен."),
])
def test_saved_correction_keeps_failure_or_unknown_outcome_without_automatic_retry(chromium, ui_server, job_status, expected):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    saves, reads, checks = [], [], []

    def memory(route):
        reads.append(route.request.url)
        route.fulfill(json=_memory_payload())

    def save(route):
        saves.append(route.request.post_data_json)
        route.fulfill(status=202, json={"source": {"source_id": "fix"}, "job_id": "update-failed", "topic_id": TOPIC})

    def job(route):
        checks.append(route.request.method)
        route.fulfill(json={"job_id": "update-failed", "status": job_status, "error": "private traceback must not be rendered"})

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", save)
    page.route("**/memory/jobs/update-failed", job)
    try:
        _, button = _edit(_open_teacher(page, ui_server), "Учитель исправляет прежнюю ошибочную запись.")
        button.click()
        panel = page.locator(".memory-correction-job")
        expect(panel).to_contain_text("Состояние обновления памяти ещё не проверено.")
        before = len(reads)
        panel.get_by_role("button", name="Проверить обновление памяти").click()
        expect(panel.get_by_role("status")).to_contain_text(expected)
        expect(panel.get_by_role("status")).to_contain_text("Исправление сохранено.")
        expect(panel.get_by_role("status")).to_contain_text("Автоматического повторного запуска не будет.")
        expect(panel).not_to_contain_text("private traceback")
        assert len(reads) == before
        assert len(saves) == 1
        assert checks == ["GET"]
        expect(page.get_by_role("button", name="Подтвердить память для помощника")).to_have_count(0)
    finally:
        page.close()


def test_delayed_update_check_does_not_replace_reopened_learner_or_draft(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    pending, reads, checks = [], [], []

    def memory(route):
        reads.append(route.request.url)
        route.fulfill(json=_memory_payload(sources=[] if "/demo-luna-03/" in route.request.url else _memory_payload()["sources"]))

    def job(route):
        checks.append(route.request.method)
        pending.append(route)

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", lambda route: route.fulfill(status=202, json={"source": {"source_id": "fix"}, "job_id": "delayed-update", "topic_id": TOPIC}))
    page.route("**/memory/jobs/delayed-update", job)
    try:
        _, save = _edit(_open_teacher(page, ui_server), "В старой заметке ошибка.")
        save.click()
        check = page.get_by_role("button", name="Проверить обновление памяти", exact=True)
        with page.expect_request(lambda request: request.url.endswith("/memory/jobs/delayed-update")):
            check.click()
        expect(check).to_be_disabled()
        check.dispatch_event("click")
        assert checks == ["GET"]
        page.locator('.learner-item[data-id="demo-luna-03"]').click()
        expect(page.locator("#memory-history .memory-source")).to_have_count(0)
        expect(page.get_by_role("button", name="Проверить обновление памяти", exact=True)).to_have_count(0)
        page.locator('.learner-item[data-id="demo-nika-04"]').click()
        with page.expect_response(lambda response: "/demo-nika-04/memory?topic_id=" in response.url):
            page.get_by_label("Тема", exact=True).select_option(TOPIC)
        expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
        expect(page.get_by_role("button", name="Проверить обновление памяти", exact=True)).to_be_enabled()
        page.locator("#memory-history .memory-sources summary").click()
        field, _ = _edit(page.locator("#memory-history .memory-source").first, "Свежий несохранённый черновик.")
        before = len(reads)
        with page.expect_response(lambda response: response.url.endswith("/memory/jobs/delayed-update")) as response:
            pending[0].fulfill(json={"status": "completed", "job_id": "delayed-update"})
        response.value.finished()
        expect(field).to_have_value("Свежий несохранённый черновик.")
        expect(page.locator(".memory-correction-job")).to_contain_text("Состояние обновления памяти ещё не проверено.")
        expect(page.locator("#memory-status")).not_to_contain_text("Обновление памяти завершено.")
        assert len(reads) == before
    finally:
        page.close()


def test_completed_update_with_failed_list_refresh_remains_saved_and_can_be_refreshed(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    fail_reads, saves, checks = [], [], []

    def memory(route):
        if fail_reads:
            route.fulfill(status=503, json={"detail": "database temporarily unavailable"})
        else:
            route.fulfill(json=_memory_payload())

    def save(route):
        saves.append(route.request.post_data_json)
        route.fulfill(status=202, json={"source": {"source_id": "fix"}, "job_id": "completed-read-failed", "topic_id": TOPIC})

    def job(route):
        checks.append(route.request.method)
        fail_reads.append(True)
        route.fulfill(json={"status": "completed", "job_id": "completed-read-failed"})

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", save)
    page.route("**/memory/jobs/completed-read-failed", job)
    try:
        _, button = _edit(_open_teacher(page, ui_server), "Исправление останется при временном сбое базы.")
        button.click()
        expect(page.locator("#memory-status")).to_contain_text("Состояние обновления памяти ещё не проверено.")
        page.get_by_role("button", name="Проверить обновление памяти").click()
        expect(page.locator("#memory-status")).to_contain_text("Обновление памяти завершено.")
        expect(page.locator("#memory-status")).to_contain_text("Не удалось обновить список памяти. Нажмите «Обновить».")
        fail_reads.clear()
        page.locator("#refresh-memory").click()
        expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
        expect(page.locator(".memory-correction-job")).to_contain_text("Обновление памяти завершено.")
        assert len(saves) == 1
        assert checks == ["GET"]
    finally:
        page.close()


@pytest.mark.parametrize("operation", ["correction_post", "update_get", "memory_get"])
def test_topic_roundtrip_rejects_previous_memory_view_response_without_losing_new_draft(chromium, ui_server, operation):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    pending, reads, hold_next_memory = [], [], []

    def memory(route):
        reads.append(route.request.url)
        if hold_next_memory:
            hold_next_memory.clear()
            pending.append(route)
        else:
            route.fulfill(json=_memory_payload())

    def save(route):
        if operation == "correction_post":
            pending.append(route)
        else:
            route.fulfill(status=202, json={"source": {"source_id": "fix"}, "job_id": "roundtrip-job", "topic_id": TOPIC})

    page.route("**/learners/*/memory**", memory)
    page.route("**/memory/sources/*/corrections", save)
    page.route("**/memory/jobs/roundtrip-job", lambda route: pending.append(route))
    try:
        row = _open_teacher(page, ui_server)
        if operation in {"correction_post", "update_get"}:
            _, button = _edit(row, "Исправление до переключения темы.")
            with page.expect_request(lambda request: request.url.endswith(PATH)):
                button.click()
            if operation == "update_get":
                check = page.get_by_role("button", name="Проверить обновление памяти", exact=True)
                with page.expect_request(lambda request: request.url.endswith("/memory/jobs/roundtrip-job")):
                    check.click()
        else:
            hold_next_memory.append(True)
            with page.expect_request(lambda request: "/demo-nika-04/memory?topic_id=" in request.url):
                page.locator("#refresh-memory").click()
        assert len(pending) == 1
        _roundtrip_topic(page)
        field, _ = _edit(page.locator("#memory-history .memory-source").first, "Новый черновик после A → B → A.")
        before = len(reads)
        url = pending[0].request.url
        with page.expect_response(lambda response: response.url == url) as response:
            if operation == "correction_post":
                pending[0].fulfill(status=202, json={"source": {"source_id": "old-fix"}, "job_id": "old-roundtrip", "topic_id": TOPIC})
            elif operation == "update_get":
                pending[0].fulfill(json={"job_id": "roundtrip-job", "status": "completed"})
            else:
                pending[0].fulfill(json=_memory_payload(revision=99, sources=[{"source_id": "stale-source", "role": "student", "text": "Устаревший снимок памяти."}]))
        response.value.finished()
        expect(field).to_have_value("Новый черновик после A → B → A.")
        expect(page.locator("#memory-status")).to_contain_text("Версия профиля 7")
        expect(page.locator("#memory-history")).not_to_contain_text("Устаревший снимок памяти.")
        expect(page.locator("#memory-history")).not_to_contain_text("Обновление памяти завершено.")
        assert len(reads) == before
    finally:
        page.close()
