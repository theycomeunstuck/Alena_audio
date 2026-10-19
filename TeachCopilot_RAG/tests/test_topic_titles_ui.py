"""Topic labels must survive the child-safe API projection and memory views."""
import copy

from tests.test_ui_acceptance import _login, expect, ui_server


TOPIC = "math.g4.fractions.numerator_denominator"
TITLE = "Дробь, числитель и знаменатель"


def test_student_sees_goal_and_named_history_with_actual_outcomes(chromium, ui_server):
    card = ui_server.store.cards["demo-nika-04"]
    card["rag_context"]["current_goal"] = {"topic_id": TOPIC, "goal": "Объяснить свою дробь"}
    card["learner_model"]["competencies"] = [{
        "topic_id": TOPIC,
        "history": [
            {"date": "2026-10-19", "solved": True, "support_level": "hint_question"},
            {"date": "2026-10-18", "solved": False, "support_level": "independent"},
            {"date": "2026-10-17", "solved": True, "support_level": "independent"},
        ],
    }]
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    try:
        _login(page, ui_server.base_url, "student.ui@example.test")
        expect(page.locator("#goal-subject")).to_have_text(TITLE)
        expect(page.locator("#current-goal")).to_have_text("Объяснить свою дробь")
        history = page.locator("#history-list .history-item")
        expect(history).to_have_count(3)
        expect(history.nth(0)).to_contain_text(f"{TITLE} – Выполнено с опорой")
        expect(history.nth(1)).to_contain_text(f"{TITLE} – Попробовали, нужна ещё практика")
        expect(history.nth(2)).to_contain_text(f"{TITLE} – Выполнено самостоятельно")
        expect(history.nth(0).locator(".history-mark")).to_have_class("history-mark supported")
        expect(page.locator("#history-list")).not_to_contain_text("Тема не указана")
        page.get_by_label("Предмет").select_option("python")
        expect(page.locator("#learning-session-status")).to_contain_text("Python")
    finally:
        page.close()


def test_teacher_sees_distinct_catalog_and_general_topic_memory_labels(chromium, ui_server, monkeypatch):
    payload = {"revision": 1, "sources": [{
        "source_id": "title-source", "topic_id": TOPIC, "role": "student", "text": "ну вроде",
    }], "versions": [
        {"version_id": "specific", "topic_id": TOPIC, "method": "tail", "created_at": "2026-10-19"},
        {"version_id": "subject", "topic_id": "math", "method": "tail", "created_at": "2026-10-18"},
    ]}
    original = copy.deepcopy(payload)
    monkeypatch.setattr(ui_server.store, "get_learner_memory", lambda *args: payload)
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    try:
        page.route("**/learners/*/next-step", lambda route: route.fulfill(status=202, json={"job_id": "named-step", "status": "pending"}))
        page.route("**/memory/jobs/named-step", lambda route: route.fulfill(json={
            "status": "completed", "result": {
                "topic_id": TOPIC, "action": "Задать один уточняющий вопрос.",
                "reason": "Короткий ответ пока не подтверждает решение.", "review_required": True,
            },
        }))
        _login(page, ui_server.base_url, "teacher.ui@example.test")
        page.locator('.learner-item[data-id="demo-nika-04"]').click()
        summaries = page.locator("#memory-history .memory-version summary")
        expect(summaries).to_have_count(2)
        expect(summaries.nth(0)).to_contain_text(f"{TITLE} · Последние занятия")
        expect(summaries.nth(1)).to_contain_text("Математика · Последние занятия")
        page.locator("#memory-history .memory-sources summary").click()
        expect(page.locator("#memory-history .memory-source")).to_contain_text(TITLE)
        expect(page.locator("#memory-history .memory-source")).to_contain_text("ну вроде")
        page.get_by_label("Тема", exact=True).select_option(TOPIC)
        page.get_by_role("button", name="Предложить следующий шаг").click()
        expect(page.locator("#next-step-result")).to_contain_text(TITLE)
        expect(page.locator("#next-step-result")).to_contain_text("Задать один уточняющий вопрос.")
        assert payload == original, "display labels must not mutate stored records"
    finally:
        page.close()
