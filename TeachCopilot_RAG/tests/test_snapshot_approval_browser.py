"""The teacher sees and acknowledges new chat before confirming an old snapshot."""
import pytest
from playwright.sync_api import expect

from tests.test_personal_graph_archive_browser import live_archive, archive_case, flow_store, _open  # noqa: F401
from tests.test_personal_graph_archive_api import LEARNER, TOPIC


@pytest.mark.parametrize("completed", [False, True])
def test_teacher_confirms_only_reviewed_snapshot_after_new_child_chat(chromium, live_archive, completed):
    case, url = live_archive
    student = case.store.get_account(email="archive-student@synthetic.test")
    actor = str(student["id"])
    session = case.store.create_learning_session(LEARNER, actor, TOPIC)
    message = case.store.reserve_chat_message(session["id"], LEARNER, "after-draft", "Новая реплика.", actor)
    case.store.complete_chat_message(session["id"], "after-draft", {"answer": "Продолжим."})
    if completed:
        case.store.complete_learning_session(session["id"], LEARNER, actor)
    page = chromium.new_page(viewport={"width": 1440, "height": 1050})
    try:
        version, panel = _open(page, url)
        expect(version).to_contain_text("новые сообщения")
        expect(version).to_contain_text("не входят в показанный снимок")
        expect(panel.locator(".memory-archive-preview")).to_have_count(0)
        confirm = version.get_by_role("button", name="Подтвердить память для помощника")
        expect(confirm).to_be_disabled()
        version.get_by_label("Я понимаю, что новые сообщения не входят в эту версию и память нужно обновить отдельно.").check()
        expect(confirm).to_be_enabled()
        with page.expect_response(lambda response: response.url.endswith("/confirm") and response.request.method == "POST") as response:
            confirm.click()
        assert response.value.status == 200, response.value.text()
        request = response.value.request.post_data_json
        assert request["acknowledge_new_messages"] is True
        assert request["expected_source_revision"] > case.snapshot["source_revision"]
        result = response.value.json()
        assert result["refresh_required"] is True
        access = result["version"]["context"]["source_access"]
        assert access["mode"] == "selected_context"
        assert message["source_id"] not in access["source_ids"]
        expect(page.get_by_text("Эта версия подтверждена без новых сообщений.", exact=False)).to_be_visible()
        refresh = page.get_by_role("button", name="Подготовить новую версию памяти")
        expect(refresh).to_be_visible()
        with page.expect_response(lambda result: result.url.endswith("/next-step") and result.request.method == "POST") as queued:
            refresh.click()
        assert queued.value.status == 202, queued.value.text()
        assert queued.value.request.post_data_json["method"] == "personal_graph"
        assert case.store.get_memory_job(queued.value.json()["job_id"])["source_revision"] == request["expected_source_revision"]
    finally:
        page.close()
