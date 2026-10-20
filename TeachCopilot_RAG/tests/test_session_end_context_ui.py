"""The assistant's final session turn is visible as context, never learner evidence."""
import copy

import pytest

from tests.test_ui_acceptance import _login, expect, ui_server


TOPIC = "math.g4.fractions.numerator_denominator"
FINAL_TEXT = '  <img src=x onerror="window.__session_end_xss=1">\nПосчитаем части вместе?  '


def _payload():
    return {
        "revision": 1, "source_revision": 3, "sources": [{
            "source_id": "student-final", "author_role": "student", "topic_id": TOPIC, "text": "хз",
            "session_end_context": {"role": "assistant", "usage": "context_only", "text": FINAL_TEXT,
                "excerpt_mode": "whole", "truncated": False, "assistant_source_id": "assistant-context-only", "session_id": "completed-session"},
        }], "versions": [{"version_id": "session-context-version", "method": "tail", "approved": False,
            "state": {"records": [{"claim": "Результат ученика не установлен.", "source_ids": ["student-final"]}]},
            "context": {"selection_events": []}}],
    }


def _open_memory(page, ui_server, payload):
    page.route("**/learners/*/memory**", lambda route: route.fulfill(json=payload))
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    expect(page.locator("#memory-status")).to_contain_text("Версия профиля 1")
    page.locator("#memory-history .memory-sources summary").click()
    return page.locator("#memory-history .memory-source").first


def test_final_assistant_turn_has_its_own_safe_context_block_without_becoming_student_evidence(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload = _payload()
    original = copy.deepcopy(payload)
    try:
        row = _open_memory(page, ui_server, payload)
        expect(page.locator("#memory-history .memory-source")).to_have_count(1)
        expect(row.locator(":scope > p").first).to_have_text("хз")
        block = row.locator(".memory-session-end-context")
        expect(block).to_contain_text("Последняя реплика помощника в занятии")
        expect(block).to_contain_text("Только контекст разговора. Результат ученика по этой реплике не установлен.")
        assert block.locator("blockquote").text_content() == FINAL_TEXT
        assert block.locator("blockquote").evaluate("node => getComputedStyle(node).whiteSpace") == "pre-wrap"
        assert page.locator("#memory-history img, #memory-history script").count() == 0
        assert page.evaluate("window.__session_end_xss || null") is None
        expect(page.get_by_role("button", name="Исправить запись", exact=True)).to_have_count(1)
        page.locator("#memory-history .memory-version summary").click()
        expect(page.locator(".memory-record")).to_contain_text("Основания: student-final")
        expect(page.locator(".memory-record")).not_to_contain_text("assistant-context-only")
        expect(row).not_to_contain_text("assistant-context-only")
        assert row.locator("a").count() == 0
        assert payload == original, "displaying assistant context must not rewrite source or evidence data"
    finally:
        page.close()


def test_oversized_final_turn_keeps_student_source_and_shows_omission_without_private_metadata(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload = _payload()
    source = payload["sources"][0]
    del source["session_end_context"]
    source["session_end_context_omitted"] = {"reason": "max_limit", "diagnostic": "private size information"}
    try:
        row = _open_memory(page, ui_server, payload)
        expect(row.locator(":scope > p").first).to_have_text("хз")
        expect(row).to_contain_text("Последняя реплика помощника слишком длинная и не включена в память.")
        expect(row.locator(".memory-session-end-context")).to_have_count(0)
        expect(row).not_to_contain_text("private size information")
        expect(row.locator(":scope > p.review-note")).not_to_contain_text("исправление")
        expect(page.locator("#memory-history .memory-source")).to_have_count(1)
    finally:
        page.close()


@pytest.mark.parametrize("event, message", [
    ("session_end_context_group_skipped", "Контекст последней реплики помощника не поместился в выбранную память и не будет передан помощнику."),
    ("session_end_context_omitted", "Последняя реплика помощника слишком длинная и не включена в память. Контекст этой реплики недоступен."),
])
def test_selected_memory_reports_unavailable_final_context_as_context_not_a_correction(chromium, ui_server, event, message):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    payload = _payload()
    payload["versions"][0]["context"]["selection_events"] = [{
        "event": event, "reason": "remaining_budget_exceeded", "diagnostic": "private selection metadata",
        "source_ids": ["student-final", "assistant-context-only"],
    }]
    try:
        _open_memory(page, ui_server, payload)
        version = page.locator("#memory-history .memory-version")
        version.locator("summary").click()
        expect(version).to_contain_text(message)
        expect(version).not_to_contain_text("Исходная запись и исправления")
        expect(version).not_to_contain_text("private selection metadata")
        expect(version).not_to_contain_text("assistant-context-only")
        expect(version.locator(".memory-record")).to_have_count(1)
        expect(version.locator(".memory-record")).to_contain_text("Основания: student-final")
        expect(page.locator(".memory-session-end-context")).to_contain_text("Посчитаем части вместе?")
    finally:
        page.close()
