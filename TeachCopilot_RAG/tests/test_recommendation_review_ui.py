"""A teacher can inspect literal grounds despite incomplete generated unknowns."""
from copy import deepcopy

from local_app.recommendation_review import recommendation_source_review
from tests.test_ui_acceptance import _login, expect, ui_server


TOPIC = "math.g4.fractions.numerator_denominator"


def open_recommendation(page, ui_server, data):
    page.route("**/learners/*/next-step", lambda route: route.fulfill(status=202, json={"job_id": "source-review", "status": "queued"}))
    page.route("**/memory/jobs/source-review", lambda route: route.fulfill(json={"status": "completed", "result": data}))
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    page.get_by_label("Тема", exact=True).select_option(TOPIC)
    page.get_by_role("button", name="Предложить следующий шаг").click()
    expect(page.locator("#next-step-status")).to_contain_text("готова к просмотру")
    return page.locator("#next-step-result")


def data_for(sources, ids, **context_extra):
    action = {"action": "clarify_missing_observation", "reason": "Уточнить запись ученика.",
              "unknowns": [], "evidence_turn_ids": ids, "review_required": True}
    context = {"sources": sources, "evidence_ids": [row["source_id"] for row in sources], **context_extra}
    return {**action, "topic_id": TOPIC, "sources": sources,
            "evidence_review": recommendation_source_review(action, context)}


def test_empty_unknowns_still_show_source_unknowns_and_the_question_behind_four(chromium, ui_server, tmp_path):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    answer = '4? Это число всех равных частей? <img src=x onerror="window.__review_xss=1">'
    question = "Сколько всего равных частей в круге?"
    data = data_for([
        {"source_id": "help-note", "role": "teacher", "text": "Условия помощи неизвестны."},
        {"source_id": "answer", "role": "student", "text": answer},
    ], ["answer"], dialogue_context=[{"source_id": "answer", "role": "assistant", "usage": "context_only",
        "text": question, "truncated": False}])
    before = deepcopy(data)
    try:
        result = open_recommendation(page, ui_server, data)
        expect(result.locator(".next-step-action")).to_have_text("Уточнить недостающее наблюдение")
        expect(result).not_to_contain_text("clarify_missing_observation")
        expect(result).to_contain_text("Черновик обоснования")
        expect(result.locator(".recommendation-empty-unknowns")).to_be_visible()
        expect(result.locator(".recommendation-source-unknowns")).to_contain_text("Условия помощи неизвестны")
        expect(result.locator(".recommendation-source-unknowns")).to_contain_text("Источник не указан в ссылках черновика")
        evidence = result.locator(".recommendation-evidence")
        evidence.locator("summary").click()
        expect(evidence).to_contain_text(answer)
        expect(evidence).to_contain_text(question)
        expect(evidence).to_contain_text("Слова помощника не подтверждают результат ученика")
        expect(evidence.locator("img")).to_have_count(0)
        assert page.evaluate("window.__review_xss") is None
        assert data == before
        result.screenshot(path=str(tmp_path / "recommendation_review.png"))
    finally:
        page.close()


def test_cited_withdrawn_note_is_shown_with_its_correction_as_read_only(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    data = data_for([
        {"source_id": "old", "role": "teacher", "text": "Три примера решил самостоятельно."},
        {"source_id": "fix", "role": "teacher", "text": "Прежнюю запись отзываю. Самостоятельность не подтверждена.", "corrects_source_id": "old"},
    ], ["old"])
    try:
        result = open_recommendation(page, ui_server, data)
        expect(result.locator(".recommendation-source-unknowns")).to_contain_text("Самостоятельность не подтверждена")
        evidence = result.locator(".recommendation-evidence")
        evidence.locator("summary").click()
        expect(evidence.locator(".memory-source")).to_have_count(2)
        expect(evidence).to_contain_text("Запись исправлена: fix")
        expect(evidence).to_contain_text("Прежнюю запись отзываю")
        expect(evidence).to_contain_text("в ссылках черновика не указана")
        expect(result.get_by_role("button", name="Исправить запись", exact=True)).to_have_count(0)
    finally:
        page.close()


def test_graph_question_attribution_prompts_review_without_changing_reason(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1100})
    sources = [
        {"source_id": "note", "role": "teacher", "text": "Исправил после вопроса о частях."},
        {"source_id": "answer", "role": "student", "text": "Четыре, значит три четверти."},
    ]
    dialogue = [{"source_id": "answer", "role": "assistant", "usage": "context_only",
                 "text": "Сколько равных частей?", "truncated": False}]
    data = data_for(sources, ["note"], method="personal_graph", dialogue_context=dialogue)
    data["reason"] = "Исправил после вопроса учителя."
    data["evidence_review"] = recommendation_source_review(
        data, {"method": "personal_graph", "sources": sources, "evidence_ids": ["note", "answer"],
               "dialogue_context": dialogue})
    try:
        result = open_recommendation(page, ui_server, data)
        expect(result).to_contain_text(data["reason"])
        check = result.locator(".recommendation-attribution-check")
        expect(check).to_contain_text("вопроса учителя")
        expect(check).to_contain_text("Это предупреждение не определяет, есть ли ошибка")
        expect(check).to_contain_text("Другие выбранные диалоги могут относиться к другим событиям")
        other_dialogues = check.locator("details")
        other_dialogues.locator("summary").click()
        expect(other_dialogues).to_contain_text("В ссылках черновика не указана")
        expect(other_dialogues).to_contain_text("Сколько равных частей?")
    finally:
        page.close()


def test_saved_legacy_result_keeps_sources_and_does_not_imply_known_conditions(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    data = {"topic_id": TOPIC, "action": "continue_work", "reason": "Вернуться к примеру.", "unknowns": [],
            "review_required": True, "sources": [{"source_id": "legacy", "role": "student", "text": "Начал пример."}]}
    try:
        result = open_recommendation(page, ui_server, data)
        expect(result.locator(".recommendation-empty-unknowns")).to_contain_text("Это не подтверждает самостоятельность или освоение")
        expect(result.locator(".recommendation-evidence")).to_have_count(0)
        result.locator("summary").click()
        expect(result).to_contain_text("Начал пример")
    finally:
        page.close()
