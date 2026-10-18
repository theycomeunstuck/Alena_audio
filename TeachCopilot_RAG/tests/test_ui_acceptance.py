"""Browser acceptance tests for the local teacher and student workspace.

These tests run the real FastAPI routes and static frontend in a local Uvicorn
process. The injected store and model services are isolated in-memory fixtures
so a browser run cannot touch the user's database or invoke local inference.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the browser download local to this ignored workspace directory.
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".local" / "ms-playwright"))

playwright_api = pytest.importorskip("playwright.sync_api", reason="install requirements-dev-local.txt for browser acceptance tests")
from playwright.sync_api import expect

from local_app.auth import hash_password
from local_app.api import create_app
from local_app.store import MIGRATIONS, RevisionConflict


DEMO_LEARNERS = ROOT.parent / "learner-data" / "demo" / "learners"
CATALOG_DIR = ROOT.parent / "learner-data" / "catalog"
SCREENSHOTS = ROOT / ".local" / "screenshots"


def _load_demo_cards() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(DEMO_LEARNERS.glob("*.json"))
    ]


def _load_catalog() -> dict[str, dict]:
    topics: dict[str, dict] = {}
    for path in sorted(CATALOG_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for topic in payload.get("topics", []):
            topics[topic["id"]] = topic
    return topics


class BrowserStore:
    """Small Store-shaped test double; all writes stay inside this fixture."""

    def __init__(self):
        self.cards = {card["learner_id"]: copy.deepcopy(card) for card in _load_demo_cards()}
        self.catalog_rows = _load_catalog()
        self.accounts: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}
        self.assignments: set[tuple[str, str]] = set()
        self.revisions = {learner_id: 1 for learner_id in self.cards}
        self.proposals: dict[str, dict] = {}
        self.missing_dashboards: set[str] = set()
        self.dashboard_delay = 0.0
        self.last_chat_subject: str | None = None
        self.confirm_calls: list[str] = []
        self.learning_sessions: dict[str, dict] = {}
        self.learning_messages: dict[tuple[str, str], dict] = {}
        self.learning_turns: dict[str, list[dict]] = {}
        self.memory_jobs: dict[str, dict] = {}

        self._add_account("teacher-ui", "teacher.ui@example.test", "teacher", None)
        self._add_account("teacher-other", "other.ui@example.test", "teacher", None)
        self._add_account("student-ui", "student.ui@example.test", "student", "demo-nika-04")
        self._add_account("student-other", "other.student@example.test", "student", "demo-luna-03")
        for learner_id in self.cards:
            self.assignments.add(("teacher-ui", learner_id))
        self.assignments.add(("student-ui", "demo-nika-04"))
        self.assignments.add(("student-other", "demo-luna-03"))

        # Keep the demo card schema-valid because the real chat route validates
        # its safe learner projection before invoking the RAG service.

    def _add_account(self, account_id: str, email: str, role: str, learner_id: str | None):
        self.accounts[email] = {
            "id": account_id,
            "account_id": account_id,
            "email": email,
            "display_name": "Учитель демо" if role == "teacher" else "Ученик демо",
            "role": role,
            "learner_id": learner_id,
            "password_hash": hash_password("test-only passphrase 123"),
            "active": True,
        }

    def get_account(self, email=None, account_id=None):
        if email:
            return self.accounts.get(email)
        return next((row for row in self.accounts.values() if row["id"] == account_id), None)

    def create_session(self, account_id, token_hash, expires_at):
        self.sessions[token_hash] = {"account_id": account_id, "expires_at": expires_at}

    def get_session(self, token_hash):
        row = self.sessions.get(token_hash)
        if not row:
            return None
        expires = row.get("expires_at")
        if expires and expires <= datetime.now(timezone.utc):
            return None
        return row

    def revoke_session(self, token_hash):
        return self.sessions.pop(token_hash, None) is not None

    def revoke_all_sessions(self):
        self.sessions.clear()

    def list_cards(self):
        return [copy.deepcopy(card) for card in self.cards.values()]

    def get_card(self, learner_id):
        if self.dashboard_delay:
            delay, self.dashboard_delay = self.dashboard_delay, 0
            time.sleep(delay)
        if learner_id in self.missing_dashboards:
            return None
        card = self.cards.get(learner_id)
        return copy.deepcopy(card) if card else None

    def catalog(self):
        return self.catalog_rows

    def get_revision(self, learner_id):
        return self.revisions.get(learner_id)

    def save_card(self, card, actor_id, expected_revision=None):
        learner_id = card["learner_id"]
        if self.revisions.get(learner_id) != expected_revision:
            raise RevisionConflict("learner card revision changed")
        self.cards[learner_id] = copy.deepcopy(card)
        self.revisions[learner_id] += 1
        return {**copy.deepcopy(card), "revision": self.revisions[learner_id]}

    def authorize_learner_access(self, account_id, learner_id):
        return (account_id, learner_id) in self.assignments

    def assign_learner(self, account_id, learner_id, relationship="teacher"):
        self.assignments.add((account_id, learner_id))

    def create_proposal(self, proposal_id, actor_id, base_revisions, proposed_cards, metadata):
        proposal = {
            "proposal_id": proposal_id,
            "actor_id": actor_id,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": copy.deepcopy(metadata),
            "cards": [
                {
                    "learner_id": learner_id,
                    "base_revision": base_revisions[learner_id],
                    "proposed_card": copy.deepcopy(card),
                    "result_card": copy.deepcopy(card),
                }
                for learner_id, card in proposed_cards.items()
            ],
        }
        self.proposals[proposal_id] = proposal
        return copy.deepcopy(proposal)

    def get_proposal(self, proposal_id):
        proposal = self.proposals.get(proposal_id)
        return copy.deepcopy(proposal) if proposal else None

    def confirm_proposal(self, proposal_id, actor_id):
        proposal = self.proposals[proposal_id]
        if proposal["actor_id"] != actor_id:
            raise PermissionError("proposal belongs to another teacher")
        self.confirm_calls.append(proposal_id)
        if proposal["status"] == "confirmed":
            return copy.deepcopy(proposal.get("result", []))
        results = []
        for row in proposal["cards"]:
            learner_id = row["learner_id"]
            if self.revisions[learner_id] != row["base_revision"]:
                raise RevisionConflict("learner changed after proposal preview")
            results.append(self.save_card(row["proposed_card"], actor_id, row["base_revision"]))
        proposal["status"] = "confirmed"
        proposal["result"] = results
        return copy.deepcopy(results)

    def get_learner_memory(self, learner_id, topic_id=None):
        return {"learner_id": learner_id, "revision": self.revisions[learner_id], "sources": [], "versions": []}

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system", **kwargs):
        job_id = f"ui-job-{len(self.memory_jobs) + 1}"
        job = {"job_id": job_id, "learner_id": learner_id, "topic_id": topic_id,
               "kind": kind, "method": kwargs.get("method", "tail"), "status": "queued"}
        self.memory_jobs[job_id] = job
        return copy.deepcopy(job)

    def get_memory_job(self, job_id):
        return copy.deepcopy(self.memory_jobs.get(job_id))

    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        job = self.memory_jobs[job_id]
        job.update(status="running", worker_id=worker_id)
        return copy.deepcopy(job)

    def start_memory_job_call(self, job_id, worker_id):
        self.memory_jobs[job_id]["call_started"] = True

    def finish_memory_job(self, job_id, **kwargs):
        job = self.memory_jobs[job_id]
        job["status"] = "unknown_outcome" if kwargs.get("unknown_outcome") else "failed" if kwargs.get("error") else "completed"
        return copy.deepcopy(job)

    def create_learning_session(self, learner_id, actor_id, topic_id, client_session_id=None):
        for row in self.learning_sessions.values():
            if row["client_session_id"] == client_session_id and row["actor_id"] == actor_id:
                return copy.deepcopy(row)
        session_id = f"learning-ui-{len(self.learning_sessions) + 1}"
        row = {"learning_session_id": session_id, "learner_id": learner_id, "actor_id": actor_id, "topic_id": topic_id, "client_session_id": client_session_id, "status": "active"}
        self.learning_sessions[session_id] = row
        self.learning_turns[session_id] = []
        return copy.deepcopy(row)

    def get_learning_session(self, session_id, learner_id=None):
        row = self.learning_sessions.get(session_id)
        return copy.deepcopy(row) if row and (learner_id is None or row["learner_id"] == learner_id) else None

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, author_role="student"):
        key = (session_id, client_message_id)
        if key in self.learning_messages:
            return {**copy.deepcopy(self.learning_messages[key]), "created": False}
        row = {"status": "pending", "content": content}
        self.learning_messages[key] = row
        self.learning_turns[session_id].append({"role": "user", "content": content})
        return {**row, "created": True}

    def list_learning_messages(self, session_id):
        return copy.deepcopy(self.learning_turns[session_id])

    def complete_chat_message(self, session_id, client_message_id, response):
        self.learning_messages[(session_id, client_message_id)].update(status="completed", response=copy.deepcopy(response))
        self.learning_turns[session_id].append(copy.deepcopy(response["choices"][0]["message"]))

    def fail_chat_message(self, session_id, client_message_id, error=None, unknown_outcome=True):
        self.learning_messages[(session_id, client_message_id)].update(status="unknown_outcome" if unknown_outcome else "failed")

    def complete_learning_session(self, session_id, learner_id, actor_id):
        self.learning_sessions[session_id]["status"] = "completed"
        return {**copy.deepcopy(self.learning_sessions[session_id]), "job_id": "ui-memory-update"}

    def ping(self):
        return True

    def schema_version(self):
        return max((int(path.name.split("_", 1)[0]) for path in MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")), default=0)


class BrowserRuntime:
    def complete(self, prompt, system_prompt=""):
        return "{}"

    def check_models(self):
        return {"ready": True, "models": ["gemma4:12b", "bge-m3"]}


class BrowserRag:
    def __init__(self):
        self.last_answer_kwargs = None
        self.calls = []

    def search(self, **kwargs):
        return []

    def answer(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        self.last_answer_kwargs = dict(kwargs)
        return {
            "id": "ui-acceptance-chat",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {
                "role": "assistant",
                "content": '<img src=x onerror="window.__ui_xss=1"> Локальный ответ по теме.',
                "reasoning_content": "Synthetic hidden chain of thought",
            }}],
            "citations": [{"id": "python-output-01", "source_file": "programming_python.json", "answer": "Synthetic private answer"}],
            "tutor_notes": "Synthetic teacher-only note",
        }

    def status(self):
        return {"ready": True, "total": 1, "indexed": 1}


def _install_pipeline_stubs(monkeypatch):
    import pipeline.task_designer as task_designer
    import pipeline.card_updater as card_updater
    calls = {"task_design": 0}

    class DraftCard:
        def as_dict(self):
            return {
                "learner_id": "demo-nika-04",
                "topic_id": "programming.python.loops",
                "tasks": [{
                    "prompt": "Выведи число 5.",
                    "expected_answer": "print(5)",
                    "source_file": "python-loop-01",
                }],
            }

    def design_task_card(*args, **kwargs):
        calls["task_design"] += 1
        return DraftCard()

    monkeypatch.setattr(task_designer, "design_task_card", design_task_card)
    monkeypatch.setattr(task_designer, "render_task_card", lambda card, show_answers=False: "Черновик задания" if not show_answers else "Ответ для учителя")

    def update_from_notes(notes, learner_ids, **kwargs):
        rows = []
        store = kwargs["store"]
        for learner_id in learner_ids:
            card = store.get_card(learner_id)
            card.setdefault("rag_context", {}).setdefault("current_goal", {})["goal"] = "Уверенно объяснять следующий шаг"
            rows.append(SimpleNamespace(
                learner_id=learner_id,
                card=card,
                changes=["Предложена новая учебная цель"],
                pending=[],
                warnings=[],
                summary="Предложена новая учебная цель",
            ))
        return rows

    monkeypatch.setattr(card_updater, "update_from_notes", update_from_notes)
    monkeypatch.setattr(card_updater, "render_update", lambda row, applied=False: row.summary)
    return calls


@pytest.fixture
def ui_server(monkeypatch):
    import uvicorn

    calls = _install_pipeline_stubs(monkeypatch)
    store = BrowserStore()
    rag = BrowserRag()
    settings = SimpleNamespace(session_hours=2, rag_limit=3, database_url="postgresql://local-ui-test/unused")
    app = create_app(settings=settings, store=store, runtime=BrowserRuntime(), rag_service=rag)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error", access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("local API test server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield SimpleNamespace(base_url=f"http://127.0.0.1:{port}", store=store, rag=rag, calls=calls)
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _screenshot(page, filename: str):
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOTS / filename), full_page=True)


def _login(page, base_url: str, email: str, password: str = "test-only passphrase 123"):
    page.goto(base_url, wait_until="networkidle")
    expect(page.get_by_role("heading", name="Войти в класс")).to_be_visible()
    page.get_by_label("Электронная почта").fill(email)
    page.get_by_label("Пароль").fill(password)
    page.get_by_role("button", name="Войти").click()
    expect(page.locator("#workspace")).to_be_visible()


def test_teacher_can_review_task_record_observation_and_confirm_proposal(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1050})
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    confirm_requests = []
    page.on("request", lambda request: confirm_requests.append(request.url) if request.url.endswith("/confirm") and request.method == "POST" else None)

    _login(page, ui_server.base_url, "teacher.ui@example.test")
    learner = page.locator('.learner-item[data-id="demo-nika-04"]')
    expect(learner).to_be_visible()
    learner.click()
    expect(page.locator("#learner-name")).to_have_text(ui_server.store.cards["demo-nika-04"]["pseudonym"])
    expect(page.locator("#zone-map h3").nth(1)).to_have_text("С поддержкой")
    expect(page.locator("#zone-map h3").nth(3)).to_have_text("Нужно проверить")
    _screenshot(page, "teacher-dashboard.png")

    topic_select = page.get_by_label("Тема")
    topic_select.select_option("math.g4.fractions.numerator_denominator")
    page.get_by_label("Как подготовить").select_option("generated")
    page.get_by_label("Заданий в черновике").select_option("1")
    with page.expect_response(lambda response: response.url.endswith("/task/design")) as task_response:
        page.get_by_role("button", name="Собрать задания").click()
    assert task_response.value.status == 200, f"{task_response.value.text()} calls={ui_server.calls}"
    expect(page.locator("#task-result")).to_be_visible()
    expect(page.locator("#task-result")).to_contain_text("Черновик заданий")
    expect(page.locator("#task-cards")).not_to_contain_text("Ответ: print(5)")
    page.get_by_label("Показать ответы").check()
    expect(page.locator("#task-cards")).to_contain_text("Ответ: print(5)")

    page.get_by_label("Наблюдение по теме").select_option("math.g4.fractions.numerator_denominator")
    page.get_by_label("Задание краткое название").fill("Сравнить две дроби")
    page.get_by_label("Решил задание").check()
    page.get_by_label("Какая опора понадобилась?").select_option("independent")
    page.get_by_label("Заметка учителя необязательно").fill("Самостоятельно объяснила выбор знаменателя.")
    page.get_by_role("button", name="Сохранить наблюдение").click()
    expect(page.locator("#revision-label")).to_contain_text("2")
    expect(page.locator("#history-list")).to_contain_text("Самостоятельно объяснила выбор знаменателя.")

    page.get_by_label("Задание краткое название").fill("Объяснить дробь после вопроса")
    page.get_by_label("Какая опора понадобилась?").select_option("hint_question")
    page.get_by_label("Заметка учителя необязательно").fill("После наводящего вопроса нашла общий знаменатель.")
    page.get_by_role("button", name="Сохранить наблюдение").click()
    expect(page.locator("#revision-label")).to_contain_text("3")
    expect(page.locator("#history-list")).to_contain_text("После наводящего вопроса нашла общий знаменатель.")
    assert ui_server.store.revisions["demo-nika-04"] == 3

    page.locator("#lesson-notes").fill("Подтвердить цель после обсуждения дробей.")
    page.get_by_role("button", name="Подготовить изменение профиля").click()
    expect(page.locator("#proposal-result")).to_contain_text("Предложена новая учебная цель")
    assert ui_server.store.revisions["demo-nika-04"] == 3, "preview must not write the learner card"
    page.get_by_role("button", name="Посмотреть профиль").click()
    expect(page.locator("#proposal-result")).to_contain_text("Проверьте профиль перед подтверждением")
    expect(page.locator("#proposal-result")).to_contain_text("Уверенно объяснять следующий шаг")
    assert ui_server.store.revisions["demo-nika-04"] == 3, "previewing proposed changes must remain read-only"
    page.get_by_role("button", name="Подтвердить изменение профиля").click()
    expect(page.locator("#revision-label")).to_contain_text("4")
    assert ui_server.store.revisions["demo-nika-04"] == 4
    assert ui_server.store.cards["demo-nika-04"]["rag_context"]["current_goal"]["goal"] == "Уверенно объяснять следующий шаг"
    assert len(confirm_requests) == 1
    assert len(ui_server.store.confirm_calls) == 1
    assert not page_errors, page_errors
    page.close()


def test_student_sees_only_own_profile_and_chat_renders_untrusted_text_safely(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    _login(page, ui_server.base_url, "student.ui@example.test")

    expect(page.locator("#teacher-actions")).to_be_hidden()
    expect(page.locator("#student-actions")).to_be_visible()
    expect(page.locator("#learner-filter")).to_be_hidden()
    expect(page.locator("#edit-profile")).to_be_hidden()
    expect(page.locator("#learner-list .learner-item")).to_have_count(1)
    visible_profile = page.locator("#profile-content").inner_text()
    assert "legal_name" not in visible_profile
    assert "answer_key" not in visible_profile
    assert "tutor_notes" not in visible_profile

    token = page.evaluate("sessionStorage.getItem('teachcopilot.session')")
    denied = page.request.get(
        f"{ui_server.base_url}/learners/demo-luna-03/dashboard",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert denied.status == 403

    page.get_by_label("Предмет").select_option("python")
    page.get_by_label("Сообщение помощнику").fill("Как вывести число в Python?")
    with page.expect_response(lambda response: response.url.endswith("/v1/chat/completions"), timeout=5000) as chat_response:
        page.get_by_role("button", name="Отправить вопрос").click()
    assert chat_response.value.status == 200, chat_response.value.text()
    assistant = page.locator("#chat-log .chat-bubble.assistant").last
    expect(assistant).to_contain_text("Локальный ответ по теме", timeout=10000)
    expect(assistant).to_contain_text("\u041c\u0430\u0442\u0435\u0440\u0438\u0430\u043b \u043f\u043e Python")
    expect(assistant).not_to_contain_text("programming_python.json")
    expect(page.locator("#chat-log img")).to_have_count(0)
    assert page.evaluate("window.__ui_xss || null") is None
    assert ui_server.rag.last_answer_kwargs["subject_id"] == "python"
    assert "Synthetic private answer" not in page.locator("#chat-log").inner_text()
    assert "Synthetic hidden chain of thought" not in page.locator("#chat-log").inner_text()
    page.get_by_label("Сообщение помощнику").fill("А можно подсказать первый шаг?")
    with page.expect_response(lambda response: response.url.endswith("/v1/chat/completions"), timeout=5000) as followup_response:
        page.get_by_role("button", name="Отправить вопрос").click()
    assert followup_response.value.status == 200
    assert len(ui_server.rag.calls) == 2
    assert ui_server.rag.calls[1] == [
        {"role": "user", "content": "Как вывести число в Python?"},
        {"role": "assistant", "content": '<img src=x onerror="window.__ui_xss=1"> Локальный ответ по теме.'},
        {"role": "user", "content": "А можно подсказать первый шаг?"},
    ]
    assert ui_server.rag.last_answer_kwargs["subject_id"] == "python"
    assert not page_errors, page_errors
    _screenshot(page, "student-chat.png")
    page.close()


def test_teacher_memory_recommendation_shows_sources_and_passes_saved_id_to_design(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1050})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    queued, designed, approved = [], [], []
    source = {"source_id": "source-1", "role": "teacher", "text": '<img src=x onerror="window.__memory_xss=1"> Помощь неизвестна.', "observed_on": "2026-10-17"}

    def memory_route(route):
        route.fulfill(json={"learner_id": "demo-nika-04", "revision": 1, "human_reviewed": False, "sources": [source], "versions": [{"version_id": "version-ui-1", "method": "tail", "approved": bool(approved), "created_at": "2026-10-17", "state": {"records": [{"claim": "Условия помощи неизвестны", "support_level": "unknown", "source_ids": ["source-1"]}]}}]})

    def enqueue(route):
        queued.append(route.request.post_data_json)
        route.fulfill(status=202, json={"job_id": "job-ui-1", "status": "pending"})

    def job(route):
        route.fulfill(json={"job_id": "job-ui-1", "status": "completed", "result": {"next_step_id": "next-ui-1", "learner_id": "demo-nika-04", "topic_id": queued[-1]["topic_id"], "method": "tail", "revision": 1, "action": "Предложить одну самостоятельную попытку.", "reason": "Результат есть, условия помощи неизвестны.", "unknowns": ["Были ли подсказки?"], "review_required": True, "sources": [source]}})

    def approve(route):
        approved.append(route.request.post_data_json)
        route.fulfill(json={"confirmed": True})

    page.route("**/learners/*/memory**", memory_route)
    page.route("**/learners/*/memory/versions/*/confirm", approve)
    page.route("**/learners/*/next-step", enqueue)
    page.route("**/memory/jobs/*", job)
    page.route("**/task/design", lambda route: route.fulfill(json={"card": {"tasks": [{"prompt": "Проверь один пример самостоятельно."}]}, "review_required": True}))
    page.on("request", lambda request: designed.append(request.post_data_json) if request.url.endswith("/task/design") else None)
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    expect(page.locator("#memory-status")).to_contain_text("требует просмотра")
    page.locator("#memory-history .memory-version summary").click()
    page.get_by_role("button", name="Подтвердить память для помощника").click()
    expect(page.locator("#memory-history")).to_contain_text("подтверждена")
    assert approved == [{"expected_revision": 1, "interest_candidate_ids": []}]
    page.get_by_label("Тема", exact=True).select_option("math.g4.fractions.numerator_denominator")
    expect(page.get_by_label("Способ памяти")).to_have_value("tail")
    page.get_by_role("button", name="Предложить следующий шаг").click()
    result = page.locator("#next-step-result")
    expect(result).to_contain_text("Предложить одну самостоятельную попытку.")
    expect(result).to_contain_text("Были ли подсказки?")
    expect(result).to_contain_text("Требуется просмотр преподавателем")
    result.locator("summary").click()
    expect(result).to_contain_text("Помощь неизвестна.")
    assert page.locator("#next-step-result img").count() == 0
    assert page.evaluate("window.__memory_xss || null") is None
    page.get_by_label("Как подготовить").select_option("generated")
    page.get_by_role("button", name="Создать задание по рекомендации").click()
    expect(page.locator("#task-result")).to_be_visible()
    assert queued == [{"topic_id": "math.g4.fractions.numerator_denominator", "method": "tail"}]
    assert designed[-1]["next_step_id"] == "next-ui-1"
    _screenshot(page, "teacher-memory.png")
    page.locator('.learner-item[data-id="demo-luna-03"]').click()
    expect(result).to_be_hidden()
    assert not errors
    page.close()


def test_teacher_selects_conversation_interests_before_confirming_memory(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    confirmed, errors = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    candidates = [
        {"candidate_id": "add-space", "operation": "add", "interest": "космос", "source_id": "turn-interest", "quote": "Мне нравится космос."},
        {"candidate_id": "remove-racing", "operation": "remove", "interest": "гонки", "source_id": "turn-interest", "quote": "Гонки мне больше не интересны."},
        {"candidate_id": "avoid-minecraft", "operation": "avoid", "interest": "Minecraft", "source_id": "turn-interest", "quote": "Minecraft люблю, но задачи про него не хочу."},
        {"candidate_id": "allow-trains", "operation": "allow", "interest": "поезда", "source_id": "turn-interest", "quote": "Про поезда снова можно."},
    ]

    def memory(route):
        route.fulfill(json={"revision": 2 if confirmed else 1, "sources": [{"source_id": "turn-interest", "role": "student", "text": " ".join(candidate["quote"] for candidate in candidates)}], "versions": [
            {"version_id": "interests-ready", "method": "tail", "approved": bool(confirmed), "context": {"interest_candidates": candidates}},
            {"version_id": "interests-failed", "method": "tail", "approved": True, "context": {"interest_extraction": {"status": "failed", "error": "private provider traceback"}}},
        ]})

    def confirm(route):
        confirmed.append(route.request.post_data_json)
        ui_server.store.cards["demo-nika-04"]["interests"] = ["космос"]
        ui_server.store.revisions["demo-nika-04"] = 2
        route.fulfill(json={"confirmed": True, "revision": 2})

    page.route("**/learners/*/memory**", memory)
    page.route("**/learners/*/memory/versions/*/confirm", confirm)
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    page.get_by_role("button", name="Ученики", exact=True).click()
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    page.get_by_role("button", name="Прогресс", exact=True).click()
    first = page.locator("#memory-history .memory-version").first
    first.locator("summary").first.click()
    expect(first).to_contain_text("В карточку попадут только отмеченные интересы")
    add = first.get_by_role("checkbox", name="Добавить интерес: космос")
    remove = first.get_by_role("checkbox", name="Убрать интерес: гонки")
    avoid = first.get_by_role("checkbox", name="Не предлагать этот сюжет: Minecraft")
    allow = first.get_by_role("checkbox", name="Снова разрешить этот сюжет: поезда")
    expect(add).not_to_be_checked()
    expect(remove).not_to_be_checked()
    expect(avoid).not_to_be_checked()
    expect(allow).not_to_be_checked()
    expect(first).to_contain_text("Мне нравится космос.")
    expect(first).to_contain_text("turn-interest")
    page.locator("#memory-history .memory-version").nth(1).locator("summary").click()
    expect(page.locator("#memory-history")).to_contain_text("Учебная память готова. Интересы из разговора извлечь не удалось.")
    expect(page.locator("#memory-history")).not_to_contain_text("private provider traceback")
    add.check()
    avoid.check()
    with page.expect_response(lambda response: response.url.endswith("/interests-ready/confirm")):
        first.get_by_role("button", name="Подтвердить память для помощника").click()
    expect(page.locator("#revision-label")).to_contain_text("2")
    expect(page.locator("#profile-meta")).to_contain_text("космос")
    expect(page.locator('.workspace-grid')).to_have_attribute("data-mobile-panel", "learning-panel")
    expect(page.locator("#memory-history .memory-version").first).to_contain_text("подтверждена")
    assert confirmed == [{"expected_revision": 1, "interest_candidate_ids": ["add-space", "avoid-minecraft"]}]
    assert not errors
    page.close()


def test_delayed_memory_confirmation_does_not_replace_reopened_learner_view(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1440, "height": 1050})
    pending, dashboard_requests = [], []
    page.on("request", lambda request: dashboard_requests.append(request.url) if request.url.endswith("/demo-nika-04/dashboard") else None)

    def memory(route):
        versions = [{"version_id": "delayed-memory", "method": "tail", "approved": False, "context": {"interest_candidates": [
            {"candidate_id": "add-space", "operation": "add", "interest": "космос", "source_id": "turn-interest", "quote": "Мне нравится космос."},
        ]}}] if "/demo-nika-04/" in route.request.url else []
        route.fulfill(json={"revision": 1, "sources": [], "versions": versions})

    page.route("**/learners/*/memory**", memory)
    page.route("**/learners/*/memory/versions/*/confirm", lambda route: pending.append(route))
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    first = page.locator("#memory-history .memory-version").first
    first.locator("summary").click()
    first.get_by_role("checkbox", name="Добавить интерес: космос").check()
    approve = first.get_by_role("button", name="Подтвердить память для помощника")
    approve.focus()
    page.keyboard.press("Enter")
    page.keyboard.press("Enter")
    expect(approve).to_be_disabled()
    assert len(pending) == 1, "one confirmation must remain in flight"
    page.locator('.learner-item[data-id="demo-luna-03"]').click()
    expect(page.locator("#memory-history .memory-version")).to_have_count(0)
    page.locator('.learner-item[data-id="demo-nika-04"]').click()
    first = page.locator("#memory-history .memory-version").first
    first.locator("summary").click()
    choice = first.get_by_role("checkbox", name="Добавить интерес: космос")
    choice.check()
    before = len(dashboard_requests)
    with page.expect_response(lambda response: response.url.endswith("/delayed-memory/confirm")) as response:
        pending[0].fulfill(json={"confirmed": True})
    response.value.finished()
    page.evaluate("() => new Promise(resolve => setTimeout(resolve, 50))")
    expect(choice).to_be_checked()
    assert len(dashboard_requests) == before, "an earlier view must not refresh the reopened learner"
    expect(page.locator("#toast")).not_to_have_class("toast visible")
    page.close()


def test_student_session_retry_reuses_message_id_and_new_session_has_new_history(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    errors, sessions, sent, completed = [], [], [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def create_session(route):
        sessions.append(route.request.post_data_json)
        route.fulfill(status=201, json={"learning_session_id": f"session-{len(sessions)}", "learner_id": "demo-nika-04", "topic_id": sessions[-1]["topic_id"], "status": "active"})

    def chat(route):
        sent.append(route.request.post_data_json)
        if len(sent) == 1:
            route.fulfill(status=503, json={"detail": "synthetic temporary failure"})
        else:
            route.fulfill(json={"choices": [{"message": {"role": "assistant", "content": "Сначала объясни свой первый шаг."}}]})

    def complete(route):
        completed.append(route.request.url)
        route.fulfill(status=202, json={"learning_session_id": "session-1", "status": "completed", "job_id": "memory-ui-1"})

    page.route("**/learners/*/learning-sessions", create_session)
    page.route("**/v1/chat/completions", chat)
    page.route("**/learning-sessions/*/complete", complete)
    _login(page, ui_server.base_url, "student.ui@example.test")
    page.get_by_label("Предмет").select_option("python")
    page.get_by_label("Сообщение помощнику").fill("Как вывести число?")
    page.get_by_role("button", name="Отправить вопрос").click()
    expect(page.locator("#chat-retry")).to_be_visible()
    expect(page.locator("#chat-input")).to_be_disabled()
    expect(page.locator("#chat-log .chat-bubble.user")).to_have_count(1)
    page.get_by_role("button", name="Повторить этот вопрос").click()
    expect(page.locator("#chat-retry")).to_be_hidden()
    expect(page.locator("#chat-log")).to_contain_text("Сначала объясни свой первый шаг.")
    assert len(sessions) == 1
    assert sent[0]["client_message_id"] == sent[1]["client_message_id"]
    assert sent[0]["learning_session_id"] == sent[1]["learning_session_id"] == "session-1"
    assert sent[1]["messages"] == [{"role": "user", "content": "Как вывести число?"}]
    page.get_by_label("Сообщение помощнику").fill("Мой шаг – print(5).")
    page.get_by_role("button", name="Отправить вопрос").click()
    expect(page.locator("#chat-log .chat-bubble.assistant")).to_have_count(3)
    assert len(sessions) == 1
    assert sent[2]["client_message_id"] != sent[1]["client_message_id"]
    assert sent[2]["messages"] == [{"role": "user", "content": "Мой шаг – print(5)."}]
    page.get_by_role("button", name="Завершить занятие").click()
    expect(page.locator("#learning-session-status")).to_contain_text("Занятие завершено")
    assert len(completed) == 1
    page.get_by_label("Предмет").select_option("math")
    page.get_by_label("Сообщение помощнику").fill("Как сравнить дроби?")
    page.get_by_role("button", name="Отправить вопрос").click()
    expect(page.locator("#chat-log")).to_contain_text("Как сравнить дроби?")
    expect(page.locator("#chat-log")).not_to_contain_text("Как вывести число?")
    expect(page.locator("#chat-log .chat-bubble.user")).to_have_count(1)
    expect(page.locator("#chat-input")).to_be_enabled()
    assert len(sessions) == 2
    assert sent[-1]["learning_session_id"] == "session-2"
    assert not errors
    _screenshot(page, "student-session.png")
    page.close()


def test_same_tab_teacher_to_student_switch_clears_teacher_draft_and_roster(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 1365, "height": 900})
    _login(page, ui_server.base_url, "teacher.ui@example.test")
    expect(page.locator("#learner-list .learner-item")).to_have_count(6)

    page.get_by_label("Тема").select_option("math.g4.fractions.numerator_denominator")
    page.get_by_label("Как подготовить").select_option("generated")
    page.get_by_label("Заданий в черновике").select_option("1")
    with page.expect_response(lambda response: response.url.endswith("/task/design")) as task_response:
        page.get_by_role("button", name="Собрать задания").click()
    assert task_response.value.status == 200
    page.get_by_label("Показать ответы").check()
    expect(page.locator("#task-cards")).to_contain_text("Ответ: print(5)")

    page.get_by_role("button", name="Выйти").click()
    expect(page.locator("#login-view")).to_be_visible()
    page.get_by_label("Электронная почта").fill("other.student@example.test")
    page.get_by_label("Пароль").fill("test-only passphrase 123")
    page.get_by_role("button", name="Войти").click()
    expect(page.locator("#student-actions")).to_be_visible()
    expect(page.locator("#teacher-actions")).to_be_hidden()
    expect(page.locator("#learner-list .learner-item")).to_have_count(1)
    previous_name = ui_server.store.cards["demo-nika-04"]["pseudonym"]
    expect(page.locator("#learner-name")).not_to_contain_text(previous_name)
    assert "print(5)" not in page.locator("body").text_content()
    expect(page.locator("#learner-name")).to_have_text(ui_server.store.cards["demo-luna-03"]["pseudonym"])
    assert "demo-nika-04" not in page.locator("body").text_content()
    page.close()


def test_keyboard_mobile_error_recovery_and_expired_session(chromium, ui_server):
    page = chromium.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(ui_server.base_url, wait_until="networkidle")
    email = page.get_by_label("Электронная почта")
    email.focus()
    page.keyboard.type("teacher.ui@example.test")
    page.keyboard.press("Tab")
    page.keyboard.type("test-only passphrase 123")
    page.keyboard.press("Enter")
    expect(page.locator("#workspace")).to_be_visible()
    expect(page.locator("#profile-content")).to_be_visible()
    page.get_by_role("button", name="Занятие").click()
    expect(page.locator("#action-panel")).to_be_visible()
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOTS / "teacher-mobile.png"), full_page=True)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

    # The roster can recover from a missing record without a reload.
    ui_server.store.missing_dashboards.add("demo-mika-03")
    page.get_by_role("button", name="Ученики").click()
    with page.expect_response(lambda response: "/learners/demo-mika-03/dashboard" in response.url) as failed_dashboard:
        page.locator('.learner-item[data-id="demo-mika-03"]').click()
    assert failed_dashboard.value.status == 404
    page.get_by_role("button", name="Прогресс").click()
    expect(page.locator("#error-state")).to_be_visible()
    expect(page.locator("#error-text")).to_contain_text("не найден")
    ui_server.store.missing_dashboards.remove("demo-mika-03")
    page.get_by_role("button", name="Ученики").click()
    page.get_by_role("button", name="Прогресс").click()
    page.get_by_role("button", name="Повторить").click()
    expect(page.locator("#profile-content")).to_be_visible()
    expect(page.locator("#learner-name")).to_have_text(ui_server.store.cards["demo-mika-03"]["pseudonym"])

    # A revoked session returns the user to the sign-in form on the next API call.
    ui_server.store.revoke_all_sessions()
    page.get_by_role("button", name="Занятие").click()
    page.locator("#lesson-notes").fill("Проверка завершения сеанса.")
    page.get_by_role("button", name="Подготовить изменение профиля").click()
    expect(page.locator("#login-view")).to_be_visible()
    expect(page.locator("#login-error")).to_contain_text("Сеанс завершён")
    assert page.evaluate("sessionStorage.getItem('teachcopilot.session')") is None
    assert not page_errors, page_errors
    page.close()
