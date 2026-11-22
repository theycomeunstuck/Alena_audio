from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
import pytest

import local_app.api as api_module
from local_app.api import create_app
from local_app.auth import hash_password, hash_session_token
from local_app.store import MIGRATIONS, RevisionConflict


class MemoryStore:
    def __init__(self):
        self.accounts = {}
        self.sessions = {}
        self.assignments = set()
        self.proposals = {}
        self.memory_jobs = {}
        self.revisions = {"learner-a": 1, "learner-b": 1}
        self.cards = {
            "learner-a": {"learner_id": "learner-a", "pseudonym": "A", "grade": 4, "language": "ru", "interests": ["football"], "rag_context": {"current_goal": {"topic_id": "math.division"}}, "learner_model": {"mastery": 0.2}, "legal_name": "Private A", "journal": [{"text": "private"}], "answer_key": "secret", "tutor_notes": "secret"},
            "learner-b": {"learner_id": "learner-b", "grade": 4, "rag_context": {}, "legal_name": "Private B"},
        }
        self._catalog = {
            "math.g4.numbers.division_by_1_2_digit": {
                "title_ru": "Деление на однозначное и двузначное число",
                "grade": 4,
                "parent": "math.g4.numbers",
                "answer": "private catalog answer",
                "content": "private curriculum content",
            },
        }

    def add_account(self, account_id, email, role, learner_id=None, password="correct horse battery staple"):
        self.accounts[email] = {"account_id": account_id, "email": email, "display_name": email, "role": role, "learner_id": learner_id, "password_hash": hash_password(password)}
        if learner_id:
            self.assignments.add((account_id, learner_id))
        return self.accounts[email]

    def create_student_account(self, email, display_name, password_hash, learner_id, teacher_id):
        if (teacher_id, learner_id) not in self.assignments:
            raise PermissionError("not assigned")
        if email in self.accounts:
            raise ValueError("duplicate email")
        account = {"account_id": "student-created", "email": email, "display_name": display_name, "role": "student", "learner_id": learner_id, "password_hash": password_hash, "active": True}
        self.accounts[email] = account
        self.assignments.add((account["account_id"], learner_id))
        return account

    def get_account(self, email=None, account_id=None):
        if email:
            return self.accounts.get(email)
        return next((a for a in self.accounts.values() if a["account_id"] == account_id), None)

    def create_session(self, account_id, token_hash, expires_at):
        self.sessions[token_hash] = {"account_id": account_id, "expires_at": expires_at}

    def get_session(self, token_hash):
        return self.sessions.get(token_hash)

    def revoke_session(self, token_hash):
        return self.sessions.pop(token_hash, None) is not None

    def authorize_learner_access(self, account_id, learner_id):
        return (account_id, learner_id) in self.assignments

    def assign_learner(self, account_id, learner_id, relationship="teacher"):
        self.assignments.add((account_id, learner_id))

    def get_card(self, learner_id):
        return self.cards.get(learner_id)

    def catalog(self):
        return self._catalog

    def list_cards(self):
        return list(self.cards.values())

    def get_revision(self, learner_id):
        return self.revisions.get(learner_id)

    def save_card(self, card, actor_id, expected_revision=None):
        learner_id = card["learner_id"]
        if self.revisions.get(learner_id) != expected_revision:
            raise RevisionConflict("revision conflict")
        self.cards[learner_id] = card
        self.revisions[learner_id] += 1
        return dict(card, revision=self.revisions[learner_id])

    def create_assigned_learner(self, card, teacher_id):
        learner_id = card["learner_id"]
        if learner_id in self.cards:
            raise ValueError("already exists")
        self.cards[learner_id] = card
        self.revisions[learner_id] = 1
        self.assign_learner(teacher_id, learner_id, relationship="teacher")
        return dict(card, revision=1)

    def get_proposal(self, proposal_id):
        return self.proposals.get(proposal_id)

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system", **kwargs):
        job_id = "api-job-" + str(len(self.memory_jobs) + 1)
        job = {"job_id": job_id, "learner_id": learner_id, "kind": kind, "status": "queued"}
        self.memory_jobs[job_id] = job
        return dict(job)

    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        self.memory_jobs[job_id]["status"] = "running"
        return dict(self.memory_jobs[job_id])

    def start_memory_job_call(self, job_id, worker_id):
        return dict(self.memory_jobs[job_id])

    def finish_memory_job(self, job_id, **kwargs):
        self.memory_jobs[job_id]["status"] = "failed" if kwargs.get("error") else "completed"
        return dict(self.memory_jobs[job_id])

    def confirm_proposal(self, proposal_id, actor_id):
        proposal = self.proposals[proposal_id]
        if proposal["actor_id"] != actor_id:
            raise PermissionError("not proposal owner")
        return []

    def ping(self):
        return True

    def schema_version(self):
        return max((int(path.name.split("_", 1)[0]) for path in MIGRATIONS.glob("[0-9][0-9][0-9]_*.sql")), default=0)


class FakeRag:
    def __init__(self, search_result=None, search_error=None, answer=None):
        self.search_result = search_result if search_result is not None else []
        self.search_error = search_error
        self.answer_result = answer or {"id": "chatcmpl-local", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello", "reasoning_content": "hidden"}}], "tutor_notes": "internal", "sources": [{"id": "source-1", "content": "protected excerpt", "answer": "private answer", "source_file": "book.pdf"}]}
        self.search_calls = []
        self.answer_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.search_error:
            raise self.search_error
        return self.search_result

    def answer(self, messages, **kwargs):
        self.answer_calls.append({"messages": messages, **kwargs})
        return self.answer_result

    def reindex(self):
        return {"materials": 0}

    def status(self):
        return {"ready": True, "total": 1, "indexed": 1}


class FakeRuntime:
    def complete(self, prompt, system_prompt=""):
        return "{}"

    def check_models(self):
        return {"ready": True, "models": ["chat", "embedding"]}


def make_client(store=None, rag=None):
    store = store or MemoryStore()
    store.add_account("teacher-1", "teacher@example.org", "teacher")
    store.add_account("teacher-2", "other@example.org", "teacher")
    store.add_account("student-1", "student@example.org", "student", "learner-a")
    store.assign_learner("teacher-1", "learner-a", relationship="teacher")
    store.assign_learner("teacher-1", "learner-b", relationship="teacher")
    app = create_app(settings=type("Settings", (), {"session_hours": 2, "rag_limit": 3})(), store=store, runtime=FakeRuntime(), rag_service=rag or FakeRag())
    return TestClient(app), store, app.state.rag_service


def token(client, email, password="correct horse battery staple"):
    result = client.post("/auth/login", json={"email": email, "password": password})
    assert result.status_code == 200
    return result.json()["access_token"]


def auth(token_value):
    return {"Authorization": f"Bearer {token_value}"}


def test_protected_routes_require_auth_and_student_cannot_escalate_role():
    client, _, _ = make_client()
    assert client.get("/profile?learner_id=learner-a").status_code == 401
    assert client.post("/rag/search", json={"query": "math"}).status_code == 401
    student = token(client, "student@example.org")
    assert client.post("/learners", headers=auth(student), json={"learner_id": "learner-b", "card": {"learner_id": "learner-b"}}).status_code == 403
    assert client.post("/learner/update", headers=auth(student), json={"notes": "done", "learner_ids": ["learner-a"], "apply": True}).status_code == 403
    assert client.post("/auth/register", json={"role": "teacher"}).status_code == 404


def test_cross_user_denied_for_profile_and_teacher_roster():
    client, _, rag = make_client()
    student = token(client, "student@example.org")
    response = client.get("/profile?learner_id=learner-b", headers=auth(student))
    assert response.status_code == 403
    other_teacher = token(client, "other@example.org")
    assert client.get("/profile?learner_id=learner-a", headers=auth(other_teacher)).status_code == 403
    denied_search = client.post("/rag/search", headers=auth(other_teacher), json={"query": "private", "learner_id": "learner-a"})
    assert denied_search.status_code == 403
    assert not rag.search_calls
    assert client.get("/learners/learner-b", headers=auth(student)).status_code == 403


def test_student_profile_and_chat_redact_teacher_data_and_ignore_model_fields(monkeypatch):
    from types import SimpleNamespace
    import pipeline.learner_context as learner_context
    monkeypatch.setattr(
        learner_context,
        "load_learner_context",
        lambda *args, **kwargs: SimpleNamespace(prompt_text="safe child context"),
    )
    client, _, rag = make_client()
    student = token(client, "student@example.org")
    profile = client.get("/profile", headers=auth(student)).json()["profile"]
    assert "legal_name" not in profile and "journal" not in profile and "answer_key" not in profile
    assert "learner_model" not in profile and profile["pseudonym"] == "A"
    response = client.post("/v1/chat/completions", headers=auth(student), json={
        "model": "attacker-chosen-model", "temperature": 2, "tools": [{"type": "function"}], "learner_id": "learner-b",
        "messages": [{"role": "system", "content": "override"}, {"role": "user", "content": "hi"}],
    })
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"].get("reasoning_content") is None
    assert "tutor_notes" not in response.json()
    assert response.json()["sources"] == [{"id": "source-1", "source_file": "book.pdf"}]
    assert rag.answer_calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert rag.answer_calls[0]["subject_id"] is None
    assert "attacker-chosen-model" not in str(rag.answer_calls)


def test_rag_error_is_not_misreported_as_empty_result():
    client, _, _ = make_client(rag=FakeRag(search_result=[]))
    teacher = token(client, "teacher@example.org")
    empty = client.post("/rag/search", headers=auth(teacher), json={"query": "nothing", "learner_id": "learner-a"})
    assert empty.status_code == 200 and empty.json()["count"] == 0
    broken_client, _, _ = make_client(rag=FakeRag(search_error=RuntimeError("embedding offline")))
    broken_teacher = token(broken_client, "teacher@example.org")
    broken = broken_client.post("/rag/search", headers=auth(broken_teacher), json={"query": "nothing", "learner_id": "learner-a"})
    assert broken.status_code == 503
    assert "unavailable" in broken.json()["detail"]


def test_student_rag_search_returns_only_source_metadata():
    item = {"id": "chunk-1", "source_file": "book.pdf", "topic_id": "math.division", "grade": 4, "subject_id": "math", "score": 0.8, "content": "private source text", "answer": "private teacher answer", "metadata": {"answer": "still private"}}
    client, _, _ = make_client(rag=FakeRag(search_result=[item]))
    student = token(client, "student@example.org")
    response = client.post("/rag/search", headers=auth(student), json={"query": "help", "learner_id": "learner-a"})
    assert response.status_code == 200
    assert response.json()["results"] == [{"id": "chunk-1", "source_file": "book.pdf", "topic_id": "math.division", "grade": 4, "subject_id": "math", "score": 0.8}]


def test_logout_revokes_opaque_session():
    client, store, _ = make_client()
    bearer = token(client, "teacher@example.org")
    assert client.post("/auth/logout", headers=auth(bearer)).status_code == 200
    assert client.get("/learners", headers=auth(bearer)).status_code == 401


def test_teacher_can_provision_student_only_for_assigned_learner():
    client, store, _ = make_client()
    teacher = token(client, "teacher@example.org")
    created = client.post("/learners/learner-a/students", headers=auth(teacher), json={"email": "new-student@example.org", "display_name": "New Student", "password": "a long passphrase 123"})
    assert created.status_code == 201
    student_token = token(client, "new-student@example.org", "a long passphrase 123")
    assert client.get("/profile", headers=auth(student_token)).status_code == 200
    rejected = client.post("/learners/learner-b/students", headers=auth(token(client, "other@example.org")), json={"email": "bad@example.org", "display_name": "Bad", "password": "another long passphrase"})
    assert rejected.status_code == 403


def test_teacher_cannot_self_assign_existing_learner():
    client, store, _ = make_client()
    other_teacher = token(client, "other@example.org")
    response = client.post("/learners", headers=auth(other_teacher), json={"learner_id": "learner-a", "card": store.cards["learner-a"]})
    assert response.status_code == 409
    assert not store.authorize_learner_access("teacher-2", "learner-a")


def test_readiness_checks_database_index_and_models_but_live_stays_separate():
    client, _, rag = make_client()
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["models"]["ready"] is True
    rag.status = lambda: {"ready": False, "total": 4, "indexed": 2}
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503


def test_login_expiration_matches_configured_session_lifetime():
    client, _, _ = make_client()
    assert client.post("/auth/login", json={"email": "teacher@example.org", "password": "correct horse battery staple"}).json()["expires_in"] == 7200


def test_auth_me_and_catalog_are_authenticated_and_catalog_has_no_answers():
    client, _, _ = make_client()
    assert client.get("/auth/me").status_code == 401
    assert client.get("/catalog").status_code == 401
    teacher = token(client, "teacher@example.org")
    me = client.get("/auth/me", headers=auth(teacher))
    assert me.status_code == 200
    assert me.json()["account"]["role"] == "teacher"
    catalog = client.get("/catalog", headers=auth(teacher))
    assert catalog.status_code == 200
    assert catalog.json()["topics"] == [{
        "topic_id": "math.g4.numbers.division_by_1_2_digit",
        "title": "Деление на однозначное и двузначное число",
        "subject_id": "math",
        "grade": 4,
        "prerequisites": ["math.g4.numbers"],
    }]
    assert "private catalog answer" not in catalog.text
    student = token(client, "student@example.org")
    assert client.get("/catalog", headers=auth(student)).status_code == 200


def test_local_ui_root_and_assets_are_served_from_repo_without_cdn():
    client, _, _ = make_client()
    page = client.get("/")
    stylesheet = client.get("/assets/app.css")
    assert page.status_code == 200 and "TeachCopilot" in page.text
    assert stylesheet.status_code == 200 and ".app-shell" in stylesheet.text
    assert "cdn.jsdelivr.net" not in page.text
    assert "fonts.googleapis.com" not in page.text


def test_dashboard_authorizes_ownership_and_redacts_student_progress():
    client, store, _ = make_client()
    store.cards["learner-a"]["rag_context"]["current_goal"] = {
        "topic_id": "math.g4.numbers.division_by_1_2_digit",
        "goal": "Разобраться с делением",
    }
    store.cards["learner-a"]["learner_model"] = {
        "competencies": [{
            "topic_id": "math.g4.numbers.division_by_1_2_digit",
            "mastery": 0.72,
            "independence": 0.55,
            "confidence": 0.7,
            "last_assessed": "2026-10-01",
            "history": [{
                "date": "2026-10-01", "solved": True,
                "support_level": "hint_question", "task_id": "task-1",
                "note": "private teacher note",
            }],
        }],
    }
    student = token(client, "student@example.org")
    response = client.get("/learners/learner-a/dashboard", headers=auth(student))
    assert response.status_code == 200
    dashboard = response.json()
    assert dashboard["profile"]["goal"]["topic"] == "Деление на однозначное и двузначное число"
    assert dashboard["zpd"]["topics"][0]["progress"] == "Продолжаем наблюдать"
    assert dashboard["history"]["observations"][0]["outcome"] == "Выполнено с опорой"
    assert "private teacher note" not in response.text
    assert "mastery" not in response.text and "confidence" not in response.text
    assert "legal_name" not in response.text and "tutor_notes" not in response.text
    assert "Выполнено самостоятельно" not in response.text
    other_teacher = token(client, "other@example.org")
    assert client.get("/learners/learner-a/dashboard", headers=auth(other_teacher)).status_code == 403
    teacher = token(client, "teacher@example.org")
    teacher_view = client.get("/learners/learner-a/dashboard", headers=auth(teacher))
    assert teacher_view.status_code == 200
    assert teacher_view.json()["profile"]["legal_name"] == "Private A"
    assert teacher_view.json()["history"]["observations"][0]["notes"] == "private teacher note"
    topic = teacher_view.json()["zpd"]["topics"][0]
    assert topic["planning_zone"] == "zpd"
    assert topic["confirmed_zone"] == "unknown"
    assert topic["evidence"]["sufficient"] is False


def test_observations_are_explicit_revision_guarded_and_teacher_only():
    client, store, _ = make_client()
    teacher = token(client, "teacher@example.org")
    payload = {
        "expected_revision": 1,
        "topic_id": "math.g4.numbers.division_by_1_2_digit",
        "task_id": "division-task-01",
        "solved": True,
        "support_level": "hint_question",
        "notes": "Used the first hint, then finished.",
        "observed_on": "2026-10-13",
    }
    response = client.post("/learners/learner-a/observations", headers=auth(teacher), json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 2
    competency = store.cards["learner-a"]["learner_model"]["competencies"][0]
    assert competency["history"][-1]["support_level"] == "hint_question"
    assert competency["history"][-1]["solved"] is True
    assert competency["history"][-1]["task_id"] == "division-task-01"
    assert store.cards["learner-a"]["mvp"]["lessons"][-1]["topic_ids"] == [payload["topic_id"]]
    student = token(client, "student@example.org")
    assert client.post("/learners/learner-a/observations", headers=auth(student), json=payload).status_code == 403
    stale = dict(payload, expected_revision=1)
    conflict = client.post("/learners/learner-a/observations", headers=auth(teacher), json=stale)
    assert conflict.status_code == 409


def test_observation_keeps_unsolved_and_support_rung_as_separate_evidence():
    client, store, _ = make_client()
    teacher = token(client, "teacher@example.org")
    response = client.post("/learners/learner-a/observations", headers=auth(teacher), json={
        "expected_revision": 1,
        "topic_id": "math.g4.numbers.division_by_1_2_digit",
        "task_id": "division-task-02",
        "solved": False,
        "support_level": "joint",
    })
    assert response.status_code == 200, response.text
    evidence = store.cards["learner-a"]["learner_model"]["competencies"][0]["history"][-1]
    assert evidence["solved"] is False
    assert evidence["support_level"] == "joint"


def test_new_learner_creation_requires_fullcard_and_assigns_only_new_card(monkeypatch):
    monkeypatch.setattr(api_module, "_validate_learner_card", lambda card, learner_id, store: None)
    client, store, _ = make_client()
    teacher = token(client, "teacher@example.org")
    card = {"learner_id": "brand-new", "grade": "4", "pseudonym": "New"}
    created = client.post("/learners", headers=auth(teacher), json={"learner_id": "brand-new", "card": card})
    assert created.status_code == 201
    assert store.authorize_learner_access("teacher-1", "brand-new")


def test_teacher_card_update_checks_owner_and_expected_revision(monkeypatch):
    monkeypatch.setattr(api_module, "_validate_learner_card", lambda card, learner_id, store: None)
    client, store, _ = make_client()
    teacher = token(client, "teacher@example.org")
    card = dict(store.cards["learner-a"])
    updated = client.put("/learners/learner-a", headers=auth(teacher), json={"expected_revision": 1, "card": card})
    assert updated.status_code == 200 and updated.json()["revision"] == 2
    stale = client.put("/learners/learner-a", headers=auth(teacher), json={"expected_revision": 1, "card": card})
    assert stale.status_code == 409
    other = token(client, "other@example.org")
    denied = client.put("/learners/learner-a", headers=auth(other), json={"expected_revision": 2, "card": card})
    assert denied.status_code == 403


def test_wrong_teacher_cannot_confirm_proposal_through_either_route():
    client, store, _ = make_client()
    store.proposals["owned-by-other"] = {"proposal_id": "owned-by-other", "actor_id": "teacher-2", "metadata": {"learner_ids": ["learner-a"], "mode": "auto_ema"}}
    teacher = token(client, "teacher@example.org")
    payload = {"notes": "reviewed", "learner_ids": ["learner-a"], "apply": True, "proposal_id": "owned-by-other"}
    legacy = client.post("/learner/update", headers=auth(teacher), json=payload)
    dedicated = client.post("/lessons/proposals/owned-by-other/confirm", headers=auth(teacher))
    assert legacy.status_code == 403
    assert dedicated.status_code == 403


def test_proposal_preview_is_owner_only_and_contains_review_snapshot():
    client, store, _ = make_client()
    store.proposals["owned-by-teacher"] = {
        "proposal_id": "owned-by-teacher",
        "actor_id": "teacher-1",
        "status": "pending",
        "created_at": "2026-10-13T10:00:00Z",
        "metadata": {"learner_ids": ["learner-a"], "notes": "confirm this"},
        "cards": [{"learner_id": "learner-a", "base_revision": 1,
                   "proposed_card": {"learner_id": "learner-a", "goal": "new goal"}}],
    }
    teacher = token(client, "teacher@example.org")
    preview = client.get("/lessons/proposals/owned-by-teacher", headers=auth(teacher))
    assert preview.status_code == 200
    assert preview.json()["cards"][0]["base_revision"] == 1
    assert preview.json()["cards"][0]["proposed_card"]["goal"] == "new goal"
    other = token(client, "other@example.org")
    assert client.get("/lessons/proposals/owned-by-teacher", headers=auth(other)).status_code == 403
    student = token(client, "student@example.org")
    assert client.get("/lessons/proposals/owned-by-teacher", headers=auth(student)).status_code == 403


def test_generated_cards_require_teacher_review_before_student_delivery():
    client, _, _ = make_client()
    student = token(client, 'student@example.org')
    response = client.post('/task/design', headers=auth(student),
                           json={'learner_id': 'learner-a', 'count': 3})
    assert response.status_code == 403


def test_source_template_task_is_grounded_and_does_not_call_runtime(monkeypatch):
    from types import SimpleNamespace
    from pipeline.task_designer import render_task_card
    from local_app.services import design_source_template

    context = SimpleNamespace(
        topic_id="programming.scratch.loops", topic_title="Циклы Scratch", pseudonym="A",
        zone="zpd", zone_ru="В зоне развития", difficulty="easy", hint_depth=2,
        interests=("космос",),
    )
    monkeypatch.setattr("pipeline.learner_context.load_task_design_context", lambda *args, **kwargs: context)

    class SourceStore:
        def material_rows(self, filters):
            assert filters == {"topic_id": "programming.scratch.loops"}
            return [{
                "source_file": "scratch-loops.md",
                "raw_snapshot": {
                    "topic_id": "programming.scratch.loops", "task_id": "scratch-01", "difficulty": "easy",
                        "content": "Повтори движение 10 раз " + chr(0x2014) + " блоком повторения, чтобы избежать десяти одинаковых блоков.",
                    "answer": "Помести блок движения внутрь цикла с числом повторений 10.",
                    "hints": ["Какую команду нужно повторять?", "Вмести её в блок цикла."],
                },
            }]

    card = design_source_template("learner-a", "programming.scratch.loops", store=SourceStore())
    assert card.sources == ("scratch-loops.md",)
    assert "числом повторений 10" in card.tasks[0].expected_answer
    assert "10" in card.tasks[0].statement
    assert len(card.tasks[0].hints) == 2
    assert "повторить 10" not in card.tasks[0].hints[-1].text
    assert "\u2014" not in card.tasks[0].statement
    assert "10" in render_task_card(card, show_answers=False)


def test_source_template_requires_matching_active_source(monkeypatch):
    from types import SimpleNamespace
    from pipeline.task_designer import TaskDesignError
    from local_app.services import design_source_template

    context = SimpleNamespace(topic_id="programming.scratch.loops", difficulty="easy")
    monkeypatch.setattr("pipeline.learner_context.load_task_design_context", lambda *args, **kwargs: context)

    class SourceStore:
        def material_rows(self, filters):
            return [{
                "source_file": "loops.md",
                "raw_snapshot": {"topic_id": "programming.scratch.loops", "difficulty": "easy", "content": "Only a generic repeat block is described."},
            }]

    with pytest.raises(TaskDesignError, match="Only 0 distinct source tasks"):
        design_source_template("learner-a", None, store=SourceStore())


@pytest.mark.parametrize(
    ("topic_id", "statement", "answer"),
    [
        ("math.g4.numbers.division_by_1_2_digit", "Divide 408 by 4 using long division.", "408 : 4 = 102."),
        ("programming.python.loops", "Print numbers from 0 through 4 using for and range(5).", "for i in range(5) produces values from 0 through 4."),
    ],
)
def test_source_template_adapts_canonical_math_and_python_rows(monkeypatch, topic_id, statement, answer):
    from types import SimpleNamespace
    from local_app.services import design_source_template

    context = SimpleNamespace(
        topic_id=topic_id, topic_title="Selected topic", pseudonym="A", zone="zpd",
        zone_ru="Zone of proximal development", difficulty="easy", hint_depth=3,
    )
    monkeypatch.setattr("pipeline.learner_context.load_task_design_context", lambda *args, **kwargs: context)

    class SourceStore:
        def material_rows(self, filters):
            return [{"source_file": "curriculum.json", "raw_snapshot": {
                "topic_id": topic_id, "task_id": "source-task-1", "difficulty": "easy",
                "content": statement, "answer": answer, "hints": None,
            }}]

    card = design_source_template("learner-a", topic_id, store=SourceStore())
    assert card.topic_id == topic_id
    assert card.tasks[0].statement == statement
    assert card.tasks[0].expected_answer == answer
    assert card.hint_depth == 3
    assert len(card.tasks[0].hints) == 3
    hint_text = " ".join(hint.text for hint in card.tasks[0].hints)
    assert answer not in hint_text
    assert any("General scaffolding" in warning for warning in card.warnings)
    if topic_id.startswith("math."):
        assert "408" not in hint_text and "102" not in hint_text


def test_task_design_source_template_api_uses_no_model(monkeypatch):
    from types import SimpleNamespace

    client, store, _ = make_client()
    teacher = token(client, "teacher@example.org")
    store.material_rows = lambda filters: [{
        "source_file": "scratch-loops.md",
        "raw_snapshot": {
            "topic_id": "programming.scratch.loops", "task_id": "scratch-01", "difficulty": "easy",
            "content": "Повтори движение 10 раз блоком повторения, чтобы избежать десяти одинаковых блоков.",
            "answer": "Помести блок движения внутрь цикла с числом повторений 10.",
            "hints": ["Какую команду нужно повторять?"],
        },
    }]
    context = SimpleNamespace(
        topic_id="programming.scratch.loops", topic_title="Циклы Scratch", pseudonym="A",
        zone="zpd", zone_ru="В зоне развития", difficulty="easy", hint_depth=1,
        interests=("космос",),
    )
    monkeypatch.setattr("pipeline.learner_context.load_task_design_context", lambda *args, **kwargs: context)
    client.app.state.runtime.complete = lambda *args, **kwargs: pytest.fail("source template invoked the model")
    response = client.post("/task/design", headers=auth(teacher), json={
        "learner_id": "learner-a", "topic_id": "programming.scratch.loops",
        "mode": "source_template",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "source_template"
    assert body["review_required"] is True
    assert body["card"]["sources"] == ["scratch-loops.md"]
    assert len(body["card"]["tasks"]) == 1
    assert body["hints_origin"] == "source_authored"


def test_task_design_source_template_rejects_unavailable_topic():
    client, _, _ = make_client()
    teacher = token(client, "teacher@example.org")
    response = client.post("/task/design", headers=auth(teacher), json={
        "learner_id": "learner-a", "topic_id": "math.g4.numbers.division_by_1_2_digit",
        "count": 1, "mode": "source_template",
    })
    assert response.status_code == 422


def test_task_context_limit_returns_explicit_client_error(monkeypatch):
    import pipeline.task_designer as designer
    def oversized(*args, **kwargs):
        raise ValueError('private prompt text that must stay out of the response')
    monkeypatch.setattr(designer, 'design_task_card', oversized)
    client, _, _ = make_client()
    teacher = token(client, 'teacher@example.org')
    response = client.post('/task/design', headers=auth(teacher), json={'learner_id': 'learner-a'})
    assert response.status_code == 422
    assert 'private prompt' not in response.text


def test_chat_subject_keeps_ambiguous_questions_in_unified_corpus():
    assert api_module._chat_subject('How do I create a list of colors?') is None
    assert api_module._chat_subject('print in Python') == 'python'
    assert api_module._chat_subject('a loop in Scratch') == 'scratch'
    assert api_module._chat_subject('Any question', 'math') == 'math'
    assert api_module._chat_subject('Any question', 'out_of_domain') is None


def test_lesson_incomplete_completion_returns_service_error(monkeypatch):
    import pipeline.card_updater as updater
    from local_app.runtime import IncompleteCompletionError
    def incomplete(*args, **kwargs):
        raise IncompleteCompletionError('private generated content')
    monkeypatch.setattr(updater, 'update_from_notes', incomplete)
    client, _, _ = make_client()
    teacher = token(client, 'teacher@example.org')
    response = client.post('/learner/update', headers=auth(teacher), json={'learner_ids': ['learner-a'], 'notes': 'reviewed'})
    assert response.status_code == 503
    assert 'private generated' not in response.text

