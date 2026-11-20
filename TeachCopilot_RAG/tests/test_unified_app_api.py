"""Unified application API against an injected contract-fake memory facade.

The fake implements exactly the documented UnifiedMemoryService surface
(SHARED_CONTRACT.md) and records every call. The store fake accepts the two
new queue kinds so the endpoints can be exercised before the real backend is
merged. PostgreSQL behavior is covered separately on a disposable database.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.rag import RagService
from local_app.unified_app import facade_import_error, resolve_unified_memory
from tests.test_local_api import MemoryStore, auth, token
from tests.test_social_chat import Settings as LocalSettings

LEARNER = "learner-a"
FOREIGN_LEARNER = "learner-b"
TEACHER_EMAIL = "teacher.unified@example.test"
STUDENT_EMAIL = "student.unified@example.test"


class UnifiedMemoryUnavailable(Exception):
    """Contract-named stand-in; the merged facade exports the real class."""

    pass

LESSON_TEXT = """# Урок: Умножение
## Цель
Научиться умножать.
## Разминка
Сосчитай 2 × 3.
## Объяснение
24 × 3 = (20 + 4) × 3 = 72.
## Практика
1. 24 × 3 = ?
2. 13 × 2 = ?
3. 31 × 3 = ?
## Проверка
- Чему равно 12 × 4?
## Ответы
24 × 3 = 72
"""


class UnifiedMemoryValidationError(Exception):
    """Contract-named stand-in; the merged facade exports the real class."""

    pass


class UnifiedMemoryConflict(Exception):
    """Contract-named stand-in; the merged facade exports the real class."""

    pass


class FakeUnifiedMemory:
    """Contract fake: same method signatures and exceptions as the facade."""

    def __init__(self, store=None):
        self.lessons = {}
        self.lesson_counter = 0
        self.states = {}
        self.state_revisions = {}
        self.processing_results = {}
        self.assignments = {}
        self.assignment_counter = 0
        self.changes = {"revision": 7, "items": [
            {"change_id": "chg-1", "kind": "personal_change", "category": "interest",
             "operation": "add", "value": "роботы", "status": "applied", "scope": "durable"},
            {"change_id": "cand-1", "kind": "academic_candidate", "category": "mastery",
             "proposed_value": 0.8, "status": "pending", "source_refs": ["s1"]},
        ]}
        self.calls = []
        self.model_call = None
        self._store = store

    # --- helpers ---------------------------------------------------------
    def _check_access(self, learner_id, actor_id):
        self.calls.append(("access", learner_id, actor_id))

    # --- core facade methods ---------------------------------------------
    def previous_lesson_context(self, learner_id, actor_id):
        self._check_access(learner_id, actor_id)
        return None

    def fast_memory(self, learner_id, actor_id, *, session_id=None):
        self._check_access(learner_id, actor_id)
        # The real facade shape: effective records with value/source_refs.
        # The card interest "роботы" is deliberately absent from the effective
        # projection so tests can prove no card fallback resurrects it.
        return {"method": "unified_fast_memory", "learner_id": learner_id, "revision": "fast-1",
                "personal": {
                    "interests": [{"value": "космос", "origin": "conversation",
                                   "source_refs": [{"kind": "unified_personal_change",
                                                    "category": "interest", "value": "космос"}]}],
                    "avoided_interests": [{"value": "страшные истории", "origin": "conversation",
                                           "source_refs": [{"kind": "unified_personal_change",
                                                            "category": "avoided_interest",
                                                            "value": "страшные истории"}]}],
                    "preferences": [{"name": "stated_preference", "value": "короткие примеры",
                                     "origin": "conversation",
                                     "source_refs": [{"kind": "unified_personal_change",
                                                      "category": "preference",
                                                      "value": "короткие примеры"}]}]},
                "learning": {
                    "goals": [{"topic_id": "math", "goal": "выучить умножение",
                               "updated_at": "2026-11-10T10:00:00Z", "origin": "profile",
                               "source_refs": [{"kind": "learner_card",
                                                "field": "rag_context.current_goal"}]}],
                    "confirmed_observations": [],
                    "errors": [{"error_tag": "digit_swap", "topic_id": "math", "subject": "math",
                                "count": 2, "last_seen": "2026-11-10",
                                "source_refs": [{"kind": "learner_card",
                                                 "field": "error_patterns"}]}],
                    "help": [{"topic_id": "math", "strategies": ["разбор по разрядам"],
                              "source_refs": [{"kind": "learner_card",
                                               "field": "scaffolding_by_topic"}]}],
                    "unfinished": None},
                "session": {}, "source_refs": {"profile_authority": "learner_card",
                                               "conversation_authority": "unified_personal_changes"}}

    def recall(self, learner_id, actor_id, query, *, session_id=None, budget_bytes=4000,
               time_filter=None):
        self._check_access(learner_id, actor_id)
        return {"method": "conversation_retrieval", "text": "", "system_text": "",
                "sources": [], "source_ids": [], "evidence_ids": [], "diagnostics": {}}

    def save_lesson(self, learner_id, actor_id, *, topic_id, minutes, lesson_text,
                    answers_text, validation, request_id, memory_revision=None):
        self._check_access(learner_id, actor_id)
        self.lesson_counter += 1
        lesson_id = f"lesson-{self.lesson_counter}"
        record = {"lesson_id": lesson_id, "learner_id": learner_id, "topic_id": topic_id,
                  "minutes": minutes, "revision": 1, "status": "draft",
                  "lesson_text": lesson_text, "answers_text": answers_text,
                  "validation": deepcopy(validation), "source_refs": [],
                  "memory_revision": memory_revision}
        self.lessons[lesson_id] = record
        self.calls.append(("save_lesson", learner_id, actor_id, topic_id, request_id))
        return deepcopy(record)

    def get_lesson(self, learner_id, actor_id, lesson_id, *, audience):
        self._check_access(learner_id, actor_id)
        record = self.lessons[lesson_id]
        # Actual actor permissions override the requested audience: only the
        # teacher account may read teacher material.
        if audience == "teacher" and actor_id != "teacher-unified":
            audience = "student"
        if audience == "student":
            projected = deepcopy(record)
            projected["lesson_text"] = projected["lesson_text"].split("## Ответы", 1)[0]
            projected.pop("answers_text", None)
            return projected
        return deepcopy(record)

    def confirm_lesson(self, learner_id, actor_id, lesson_id, *, expected_revision, request_id):
        self._check_access(learner_id, actor_id)
        record = self.lessons[lesson_id]
        if record["revision"] != expected_revision:
            raise UnifiedMemoryValidationError("lesson revision changed")
        if record["status"] != "draft":
            raise UnifiedMemoryValidationError("lesson already confirmed")
        # The real facade keeps the revision and flips the status; content
        # changes create a new revision.
        record["status"] = "approved"
        record["confirmed"] = True
        self.calls.append(("confirm_lesson", lesson_id, request_id))
        return deepcopy(record)

    def attach_lesson(self, learner_id, actor_id, session_id, lesson_id, *, request_id):
        self._check_access(learner_id, actor_id)
        record = self.lessons[lesson_id]
        if record["status"] != "approved":
            raise UnifiedMemoryValidationError("lesson must be approved before attach")
        state = {"session_id": session_id, "lesson_id": lesson_id, "revision": 1,
                 "stage": "not_started", "pending_task_id": None,
                 "last_question_source_id": None, "completed_task_ids": []}
        self.states[session_id] = deepcopy(state)
        self.state_revisions[session_id] = 1
        self.calls.append(("attach_lesson", learner_id, session_id, lesson_id, request_id))
        return deepcopy(state)

    def lesson_state(self, learner_id, actor_id, session_id):
        self._check_access(learner_id, actor_id)
        state = self.states.get(session_id)
        return deepcopy(state) if state else {"session_id": session_id, "lesson_id": None,
                                              "revision": None, "stage": None,
                                              "pending_task_id": None,
                                              "last_question_source_id": None,
                                              "completed_task_ids": []}

    def update_lesson_state(self, learner_id, actor_id, session_id, *, expected_revision,
                            state, request_id):
        self._check_access(learner_id, actor_id)
        if self.state_revisions.get(session_id) != expected_revision:
            raise UnifiedMemoryValidationError("lesson state revision changed")
        self.state_revisions[session_id] = expected_revision + 1
        stored = self.states.setdefault(session_id, {})
        stored.update(state)
        stored["revision"] = self.state_revisions[session_id]
        self.calls.append(("update_lesson_state", session_id, dict(state), request_id))
        return deepcopy(stored)

    def process_session(self, learner_id, actor_id, session_id, *, request_id):
        self._check_access(learner_id, actor_id)
        self.calls.append(("process_session", learner_id, session_id, request_id))
        return {"session_id": session_id, "status": "completed", "source_fingerprint": "fp-1",
                "summary": {"summary_id": "sum-1", "text": "## Что изучали\nумножение\n"
                            "## Что получилось\nданных пока нет\n## Что стоит повторить\nнет\n"
                            "## Следующий шаг\nодин пример", "source_refs": ["s1"],
                            "coverage": {"chunks": ["s1"], "omitted": []}, "provisional": False},
                "personal_changes": [{"change_id": "chg-1", "category": "interest",
                                      "operation": "add", "value": "роботы",
                                      "source_refs": ["s1"], "status": "applied",
                                      "scope": "durable"}],
                "academic_candidates": [{"candidate_id": "cand-1", "category": "mastery",
                                         "proposed_value": 0.8, "source_refs": ["s1"],
                                         "status": "pending"}],
                "warnings": []}

    def processing_result(self, learner_id, actor_id, session_id):
        self._check_access(learner_id, actor_id)
        result = self.processing_results.get(session_id)
        if result is None:
            raise UnifiedMemoryUnavailable("processing has not run for this session")
        return deepcopy(result)

    def list_changes(self, learner_id, actor_id):
        self._check_access(learner_id, actor_id)
        return deepcopy(self.changes)

    def review_changes(self, learner_id, actor_id, *, actions, expected_revision, request_id):
        self._check_access(learner_id, actor_id)
        if expected_revision != self.changes["revision"]:
            raise UnifiedMemoryValidationError("changes revision changed")
        for action in actions:
            if action["id"] == "chg-1" and action["action"] == "revert":
                self.changes["items"][0]["status"] = "reverted"
        self.changes["revision"] += 1
        self.calls.append(("review_changes", list(actions), request_id))
        return {"applied": [action["id"] for action in actions], "revision": self.changes["revision"]}

    def list_lessons(self, learner_id, actor_id):
        self._check_access(learner_id, actor_id)
        return {"learner_id": learner_id, "lessons": [deepcopy(record) for record in self.lessons.values()]}

    # --- assignment lifecycle (round-2 P0.2 facade contract) ---------------
    def assign_lesson(self, learner_id, actor_id, lesson_id, *, expected_lesson_revision,
                      request_id):
        self._check_access(learner_id, actor_id)
        if actor_id != "teacher-unified":
            raise PermissionError("only the teacher assigns lessons")
        record = self.lessons.get(lesson_id)
        if record is None:
            raise LookupError(lesson_id)
        if record["status"] != "approved":
            raise UnifiedMemoryValidationError("lesson must be approved before assignment")
        if record["revision"] != expected_lesson_revision:
            raise UnifiedMemoryConflict("lesson revision changed")
        for assignment in self.assignments.values():
            if (assignment["lesson_id"] == lesson_id
                    and assignment["status"] in ("pending", "started")):
                self.calls.append(("assign_lesson_replayed", lesson_id, request_id))
                return deepcopy(assignment)
        self.assignment_counter += 1
        assignment = {"assignment_id": f"assign-{self.assignment_counter}",
                      "learner_id": learner_id, "lesson_id": lesson_id,
                      "lesson_revision": expected_lesson_revision, "status": "pending",
                      "learning_session_id": None, "created_at": "2026-11-17T12:00:00Z",
                      "actor_id": actor_id}
        self.assignments[assignment["assignment_id"]] = assignment
        self.calls.append(("assign_lesson", learner_id, lesson_id, request_id))
        return deepcopy(assignment)

    def list_lesson_assignments(self, learner_id, actor_id):
        self._check_access(learner_id, actor_id)
        return {"learner_id": learner_id,
                "items": [deepcopy(item) for item in self.assignments.values()
                          if item["learner_id"] == learner_id],
                "revision": 1}

    def start_lesson_assignment(self, learner_id, actor_id, assignment_id, *, request_id):
        self._check_access(learner_id, actor_id)
        assignment = self.assignments.get(assignment_id)
        if assignment is None:
            raise LookupError(assignment_id)
        if actor_id == "teacher-unified":
            raise PermissionError("only the linked student starts an assignment")
        if assignment["status"] == "completed":
            raise UnifiedMemoryValidationError("assignment already completed")
        if assignment["status"] == "started":
            state = deepcopy(self.states.get(assignment["learning_session_id"]))
            return {"assignment_id": assignment_id,
                    "learning_session_id": assignment["learning_session_id"],
                    "lesson_id": assignment["lesson_id"],
                    "lesson_revision": assignment["lesson_revision"],
                    "status": assignment["status"], "state": state}
        if self._store is None:
            raise UnifiedMemoryUnavailable("assignment start needs the session store")
        session = self._store.create_learning_session(learner_id, actor_id,
                                                      self.lessons[assignment["lesson_id"]]["topic_id"])
        session_id = session["learning_session_id"]
        state = {"session_id": session_id, "lesson_id": assignment["lesson_id"],
                 "revision": 1, "stage": "not_started", "pending_task_id": None,
                 "last_question_source_id": None, "completed_task_ids": []}
        self.states[session_id] = deepcopy(state)
        self.state_revisions[session_id] = 1
        assignment["status"] = "started"
        assignment["learning_session_id"] = session_id
        self.calls.append(("start_lesson_assignment", learner_id, assignment_id, request_id))
        return {"assignment_id": assignment_id, "learning_session_id": session_id,
                "lesson_id": assignment["lesson_id"],
                "lesson_revision": assignment["lesson_revision"],
                "status": assignment["status"], "state": deepcopy(state)}

    def lesson_tasks(self, learner_id, actor_id, lesson_id, *, audience):
        self._check_access(learner_id, actor_id)
        record = self.lessons.get(lesson_id)
        if record is None:
            raise LookupError(lesson_id)
        if audience == "teacher" and actor_id != "teacher-unified":
            raise PermissionError("task answers stay teacher-only")
        items = [
            {"task_id": f"{lesson_id}:p1", "stage": "practice", "position": 1,
             "prompt_text": "24 × 3 = ?", "source_refs": [{"kind": "lesson_section"}]},
            {"task_id": f"{lesson_id}:p2", "stage": "practice", "position": 2,
             "prompt_text": "13 × 2 = ?", "source_refs": [{"kind": "lesson_section"}]},
            {"task_id": f"{lesson_id}:p3", "stage": "practice", "position": 3,
             "prompt_text": "31 × 3 = ?", "source_refs": [{"kind": "lesson_section"}]},
        ]
        result = {"lesson_id": lesson_id, "lesson_revision": record["revision"], "items": items}
        if audience == "teacher":
            result["answers"] = {f"{lesson_id}:p1": "72", f"{lesson_id}:p2": "26",
                                 f"{lesson_id}:p3": "93"}
        return result


class UnifiedStore(MemoryStore):
    """Adds the two unified job kinds, idempotency conflicts and sessions."""

    def __init__(self):
        super().__init__()
        data = Path(__file__).resolve().parents[2] / "learner-data"
        for learner_id in (LEARNER, FOREIGN_LEARNER, "demo-nika-04"):
            card_path = data / "demo" / "learners" / f"{learner_id}.json"
            if card_path.is_file():
                self.cards[learner_id] = json.loads(card_path.read_text(encoding="utf-8"))
                self.revisions[learner_id] = 1
        for catalog_path in sorted((data / "catalog").glob("*.json")):
            for topic in json.loads(catalog_path.read_text(encoding="utf-8")).get("topics", []):
                self._catalog[topic["id"]] = topic
        self.sessions = {}
        self.messages = {}
        self.turns = {}
        self.job_conflicts = {}
        self.processing_results = {}

    # --- queue with the unified kinds -------------------------------------
    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system",
                           expected_revision=None, method="tail", idempotency_key=None,
                           provider="local"):
        if kind not in ("memory_update", "next_step", "chat", "lesson_generate", "session_process"):
            raise ValueError("invalid memory job options")
        key = (learner_id, kind, idempotency_key)
        if idempotency_key is not None and key in self.job_conflicts:
            existing = self.job_conflicts[key]
            if existing["payload"] != (payload or {}) or existing["topic_id"] != topic_id:
                from local_app.memory_store import MemoryIdempotencyConflict
                raise MemoryIdempotencyConflict("job key was reused for different input")
            return dict(existing)
        job_id = f"unified-job-{len(self.memory_jobs) + 1}"
        job = {"job_id": job_id, "learner_id": learner_id, "topic_id": topic_id, "kind": kind,
               "method": method, "provider": provider, "status": "queued",
               "payload": deepcopy(payload or {}), "actor_id": actor_id,
               "result": None, "error": None}
        self.memory_jobs[job_id] = job
        if idempotency_key is not None:
            self.job_conflicts[key] = job
        return dict(job)

    def heartbeat_memory_job(self, job_id, worker_id, lease_seconds=120):
        return True

    def get_memory_job(self, job_id):
        return deepcopy(self.memory_jobs.get(job_id))

    # --- learning sessions (minimal chat support) --------------------------
    def create_learning_session(self, learner_id, actor_id, topic_id, client_session_id=None,
                                mode="standard", event_timezone=None):
        session_id = f"unified-session-{len(self.sessions) + 1}"
        session = {"learning_session_id": session_id, "learner_id": learner_id,
                   "actor_id": actor_id, "topic_id": topic_id,
                   "client_session_id": client_session_id, "status": "active",
                   "session_mode": mode}
        self.sessions[session_id] = session
        self.turns[session_id] = []
        return deepcopy(session)

    def get_learning_session(self, session_id, learner_id=None):
        session = self.sessions.get(session_id)
        if session and (learner_id is None or session["learner_id"] == learner_id):
            return deepcopy(session)
        return None

    def complete_learning_session(self, session_id, learner_id, actor_id):
        session = self.sessions[session_id]
        session["status"] = "completed"
        return {**deepcopy(session), "job": None, "job_id": None}

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content,
                             actor_id, *, author_role="student", event_timezone=None):
        key = (session_id, client_message_id)
        if key in self.messages:
            return {**deepcopy(self.messages[key]), "created": False}
        self.messages[key] = {"status": "pending", "source_id": f"src-{len(self.messages) + 1}"}
        self.turns.setdefault(session_id, []).append({"role": "user", "content": content})
        return {"status": "pending", "created": True, "source_id": self.messages[key]["source_id"]}

    def list_learning_messages(self, session_id):
        return deepcopy(self.turns.get(session_id, []))

    def complete_chat_message(self, session_id, client_message_id, response):
        key = (session_id, client_message_id)
        self.messages[key]["status"] = "completed"
        message = deepcopy(response["choices"][0]["message"])
        message["assistant_source_id"] = f"asrc-{len(self.messages) + 1}"
        self.turns[session_id].append(message)
        return {"assistant_source_id": message["assistant_source_id"]}

    def fail_chat_message(self, session_id, client_message_id, error=None, unknown_outcome=True):
        self.messages[(session_id, client_message_id)]["status"] = \
            "unknown_outcome" if unknown_outcome else "failed"

    def get_learner_memory(self, learner_id, topic_id=None):
        return {"learner_id": learner_id, "revision": self.revisions[learner_id],
                "sources": [], "versions": []}


class SpyRag(RagService):
    """Records the answer kwargs; real RAG behavior is out of scope here."""

    def __init__(self):
        self.calls = []

    def answer(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), "kwargs": deepcopy(kwargs)})
        return {"id": None, "object": "chat.completion", "created": None, "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Обычный ответ"},
                             "finish_reason": "stop"}],
                "citations": [], "sources": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class GuidedRag:
    """Records guided calls; ordinary chat is out of scope for these tests."""

    def __init__(self):
        self.guided_calls = []

    def guided_lesson_answer(self, history, *, profile, lesson_text, guided_mode, stage,
                             preferences=None, fast_memory_text=None):
        self.guided_calls.append({"history": deepcopy(history), "profile": deepcopy(profile),
                                  "lesson_text": lesson_text, "guided_mode": guided_mode,
                                  "stage": stage, "fast_memory_text": fast_memory_text})
        return {"id": None, "object": "chat.completion", "created": None, "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant",
                             "content": "Реши первый пример: 24 × 3."}, "finish_reason": "stop"}],
                "citations": [], "sources": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "answer_routing": {"backend": "local", "reason": "guided_lesson",
                                   "stage": stage, "next_stage": "warmup"},
                "_lesson_stage": "warmup"}


def make_api(*, facade=True, rag=None, store=None, runtime=None):
    settings = LocalSettings()
    store = store or UnifiedStore()
    store.add_account("teacher-unified", TEACHER_EMAIL, "teacher", LEARNER)
    store.add_account("student-unified", STUDENT_EMAIL, "student", LEARNER)
    fake = None
    if facade:
        fake = FakeUnifiedMemory(store)
    app = create_app(settings=settings, store=store, runtime=runtime, rag_service=rag or GuidedRag(),
                     unified_memory_factory=(lambda s, r: fake) if facade else (lambda s, r: None))
    client = TestClient(app, raise_server_exceptions=False)
    teacher = auth(token(client, TEACHER_EMAIL))
    student = auth(token(client, STUDENT_EMAIL))
    return client, store, fake, teacher, student


class TestAvailability:
    def test_endpoints_answer_503_without_facade(self):
        client, store, fake, teacher, student = make_api(facade=False)
        response = client.get(f"/learners/{LEARNER}/lessons", headers=teacher)
        assert response.status_code == 503
        assert "unified memory backend" in response.json()["detail"]
        assert client.get(f"/learners/{LEARNER}/lessons/lesson-1", headers=teacher).status_code == 503
        assert client.post(f"/learners/{LEARNER}/lessons/lesson-1/confirm", headers=teacher,
                           json={"expected_revision": 1, "request_id": "r1"}).status_code == 503
        assert client.get(f"/learners/{LEARNER}/memory-changes", headers=teacher).status_code == 503

    def test_resolve_unified_memory_reports_missing_facade_honestly(self, monkeypatch):
        # Simulate the pre-merge tree: the facade module must not be importable.
        import builtins
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "local_app.unified_memory":
                raise ImportError("No module named 'local_app.unified_memory'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert resolve_unified_memory(object(), object()) is None
        assert facade_import_error() is not None and "No module named" in facade_import_error()
        # The merged tree resolves the real facade.
        from local_app.store import Store  # noqa: F401  (import sanity only)


    def test_lesson_enqueue_requires_real_store_support(self):
        # The real (unmerged) store still rejects the new kinds; the endpoint
        # answers 422 instead of a fake success.
        client, store, fake, teacher, student = make_api(facade=True)

        class QueueLimitedStore(MemoryStore):
            def get_learner_memory(self, learner_id, topic_id=None):
                return {"learner_id": learner_id, "revision": 1, "sources": [], "versions": []}

            def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None,
                                   actor_id="system", **kwargs):
                # Mirror the production allowlist before the peer's migration.
                if kind not in ("memory_update", "next_step", "chat"):
                    raise ValueError("invalid memory job options")
                return super().enqueue_memory_job(learner_id, topic_id, kind, payload,
                                                  actor_id, **kwargs)

        plain = QueueLimitedStore()
        plain.add_account("t2", "teacher2@example.test", "teacher", LEARNER)
        settings = LocalSettings()
        app = create_app(settings=settings, store=plain, runtime=None, rag_service=GuidedRag(),
                         unified_memory_factory=lambda s, r: FakeUnifiedMemory())
        plain_client = TestClient(app, raise_server_exceptions=False)
        headers = auth(token(plain_client, "teacher2@example.test"))
        response = plain_client.post(f"/learners/{LEARNER}/lessons", headers=headers,
                                     json={"topic_id": "math", "minutes": 20, "client_request_id": "r1"})
        assert response.status_code == 422


class TestLessonEndpoints:
    def test_create_lesson_job_is_idempotent_and_conflicting_payloads_rejected(self):
        client, store, fake, teacher, student = make_api()
        first = client.post(f"/learners/{LEARNER}/lessons", headers=teacher,
                            json={"topic_id": "math", "minutes": 20, "client_request_id": "req-1"})
        assert first.status_code == 202, first.text
        job_id = first.json()["job_id"]
        replay = client.post(f"/learners/{LEARNER}/lessons", headers=teacher,
                             json={"topic_id": "math", "minutes": 20, "client_request_id": "req-1"})
        assert replay.status_code == 202 and replay.json()["job_id"] == job_id
        changed = client.post(f"/learners/{LEARNER}/lessons", headers=teacher,
                              json={"topic_id": "math", "minutes": 30, "client_request_id": "req-1"})
        assert changed.status_code == 409
        assert client.post(f"/learners/{LEARNER}/lessons", headers=teacher,
                           json={"topic_id": "math", "minutes": 5, "client_request_id": "req-2"}).status_code == 202

    def test_student_cannot_create_lesson(self):
        client, store, fake, teacher, student = make_api()
        response = client.post(f"/learners/{LEARNER}/lessons", headers=student,
                               json={"topic_id": "math", "minutes": 20, "client_request_id": "r9"})
        assert response.status_code == 403

    def test_teacher_audience_sees_answers_student_projection_hides_them(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="24 × 3 = 72", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision="fast-1")
        lesson_id = saved["lesson_id"]
        teacher_view = client.get(f"/learners/{LEARNER}/lessons/{lesson_id}", headers=teacher).json()
        assert "## Ответы" in teacher_view["lesson_text"] and teacher_view["answers_text"]
        student_view = client.get(f"/learners/{LEARNER}/lessons/{lesson_id}", headers=student).json()
        assert "## Ответы" not in student_view["lesson_text"] and "answers_text" not in student_view
        # A student cannot escalate the audience.
        escalated = client.get(f"/learners/{LEARNER}/lessons/{lesson_id}?audience=teacher", headers=student)
        assert escalated.status_code == 200
        assert "## Ответы" not in escalated.json()["lesson_text"]

    def test_confirm_is_teacher_only_and_validates_revision(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        lesson_id = saved["lesson_id"]
        assert client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/confirm", headers=student,
                           json={"expected_revision": 1, "request_id": "r"}).status_code == 403
        wrong = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/confirm", headers=teacher,
                            json={"expected_revision": 5, "request_id": "r"})
        assert wrong.status_code == 422
        good = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/confirm", headers=teacher,
                           json={"expected_revision": 1, "request_id": "req-confirm-1"})
        assert good.status_code == 200 and good.json()["status"] == "approved"
        replay = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/confirm", headers=teacher,
                             json={"expected_revision": 1, "request_id": "req-confirm-1"})
        assert replay.status_code == 422  # already approved

    def test_attach_requires_approved_lesson_and_existing_session(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        lesson_id = saved["lesson_id"]
        missing = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/attach", headers=teacher,
                              json={"learning_session_id": "nope", "request_id": "r"})
        assert missing.status_code == 404
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=teacher,
                              json={"topic_id": "math"}).json()
        early = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/attach", headers=teacher,
                            json={"learning_session_id": session["learning_session_id"], "request_id": "r"})
        assert early.status_code == 422  # draft lessons cannot attach
        fake.confirm_lesson(LEARNER, "t", lesson_id, expected_revision=1, request_id="c")
        attached = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/attach", headers=teacher,
                               json={"learning_session_id": session["learning_session_id"], "request_id": "r2"})
        assert attached.status_code == 200
        assert attached.json()["stage"] == "not_started"

    def test_lesson_state_read_and_update(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        fake.confirm_lesson(LEARNER, "t", saved["lesson_id"], expected_revision=1, request_id="c")
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        client.post(f"/learners/{LEARNER}/lessons/{saved['lesson_id']}/attach", headers=teacher,
                    json={"learning_session_id": session, "request_id": "r2"})
        state = client.get(f"/learners/{LEARNER}/learning-sessions/{session}/lesson-state",
                           headers=student)
        assert state.status_code == 200 and state.json()["stage"] == "not_started"
        updated = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/lesson-state",
                              headers=student,
                              json={"expected_revision": 1, "state": {"stage": "warmup"},
                                    "request_id": "u1"})
        assert updated.status_code == 200 and updated.json()["stage"] == "warmup"
        stale = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/lesson-state",
                            headers=student,
                            json={"expected_revision": 1, "state": {"stage": "practice"},
                                  "request_id": "u2"})
        assert stale.status_code == 422

    def test_list_lessons_uses_optional_facade_capability(self):
        client, store, fake, teacher, student = make_api()
        fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                         answers_text="", validation={"ok": True, "problems": []},
                         request_id="r1", memory_revision=None)
        listed = client.get(f"/learners/{LEARNER}/lessons", headers=teacher)
        assert listed.status_code == 200 and len(listed.json()["lessons"]) == 1


class TestSessionProcessing:
    def test_active_session_requires_teacher_request(self):
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        denied = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/process",
                             headers=student, json={"client_request_id": "p1"})
        assert denied.status_code == 403
        teacher_ok = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/process",
                                 headers=teacher, json={"client_request_id": "p1"})
        assert teacher_ok.status_code == 202

    def test_completed_session_owner_can_request_processing_once(self):
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        store.complete_learning_session(session, LEARNER, "student-unified")
        first = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/process",
                            headers=student, json={"client_request_id": "p1"})
        assert first.status_code == 202
        replay = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/process",
                             headers=student, json={"client_request_id": "p1"})
        assert replay.status_code == 202 and replay.json()["job_id"] == first.json()["job_id"]
        other = client.post(f"/learners/{LEARNER}/learning-sessions/{session}/process",
                            headers=student, json={"client_request_id": "p2"})
        assert other.status_code == 202 and other.json()["job_id"] != first.json()["job_id"]

    def test_processing_result_passthrough_and_unavailable_state(self):
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        result = client.get(f"/learners/{LEARNER}/learning-sessions/{session}/processing-result",
                            headers=student)
        assert result.status_code == 503  # explicit pending/unavailable, not a fake success
        fake.processing_results[session] = {"session_id": session, "status": "completed",
                                            "summary": {"text": "итоги"}}
        ok = client.get(f"/learners/{LEARNER}/learning-sessions/{session}/processing-result",
                        headers=student)
        assert ok.status_code == 200 and ok.json()["summary"]["text"] == "итоги"


class TestChangeReview:
    def test_changes_are_teacher_only(self):
        client, store, fake, teacher, student = make_api()
        assert client.get(f"/learners/{LEARNER}/memory-changes", headers=student).status_code == 403
        listed = client.get(f"/learners/{LEARNER}/memory-changes", headers=teacher)
        assert listed.status_code == 200 and listed.json()["revision"] == 7
        assert client.post(f"/learners/{LEARNER}/memory-changes/review", headers=student,
                           json={"expected_revision": 7, "actions": [{"id": "chg-1", "action": "revert"}],
                                 "request_id": "r1"}).status_code == 403

    def test_review_rejects_malformed_actions_and_applies_valid_ones(self):
        client, store, fake, teacher, student = make_api()
        bad = client.post(f"/learners/{LEARNER}/memory-changes/review", headers=teacher,
                          json={"expected_revision": 7, "actions": [{"id": "chg-1", "action": "delete"}],
                                "request_id": "r1"})
        assert bad.status_code == 422
        good = client.post(f"/learners/{LEARNER}/memory-changes/review", headers=teacher,
                           json={"expected_revision": 7, "actions": [{"id": "chg-1", "action": "revert"}],
                                 "request_id": "r2"})
        assert good.status_code == 200 and good.json()["applied"] == ["chg-1"]
        assert fake.changes["items"][0]["status"] == "reverted"
        stale = client.post(f"/learners/{LEARNER}/memory-changes/review", headers=teacher,
                            json={"expected_revision": 7, "actions": [{"id": "chg-1", "action": "revert"}],
                                  "request_id": "r3"})
        assert stale.status_code == 422


class TestGuidedChat:
    def _chat_payload(self, session_id, message="помоги начать"):
        return {"learning_session_id": session_id, "client_message_id": f"msg-{uuid4().hex[:8]}",
                "messages": [{"role": "user", "content": message}],
                "metadata": {"lesson_mode": "guided"}}

    def test_guided_chat_without_session_or_lesson_is_rejected(self):
        client, store, fake, teacher, student = make_api()
        assert client.post("/v1/chat/completions", headers=student,
                           json={"messages": [{"role": "user", "content": "привет"}],
                                 "metadata": {"lesson_mode": "guided"}}).status_code == 409
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        no_lesson = client.post("/v1/chat/completions", headers=student,
                                json=self._chat_payload(session))
        assert no_lesson.status_code == 409
        assert "attached lesson" in no_lesson.json()["detail"]

    def test_guided_chat_without_facade_answers_503(self):
        client, store, fake, teacher, student = make_api(facade=False)
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        response = client.post("/v1/chat/completions", headers=student,
                               json=self._chat_payload(session))
        assert response.status_code == 503

    def test_guided_chat_uses_server_state_and_records_stage(self):
        client, store, fake, teacher, student = make_api()
        rag = client.app.state.rag_service
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="24 × 3 = 72", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision="fast-1")
        fake.confirm_lesson(LEARNER, "t", saved["lesson_id"], expected_revision=1, request_id="c")
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        fake.attach_lesson(LEARNER, "t", session, saved["lesson_id"], request_id="a1")
        response = client.post("/v1/chat/completions", headers=student,
                               json=self._chat_payload(session))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["choices"][0]["message"]["content"] == "Реши первый пример: 24 × 3."
        assert "_lesson_stage" not in json.dumps(body)
        assert body["answer_routing"]["reason"] == "guided_lesson"
        call = rag.guided_calls[-1]
        # The lesson text is the student projection: teacher answers never enter the prompt.
        assert "## Ответы" not in call["lesson_text"] and "24 × 3 = 72" not in call["lesson_text"]
        assert call["guided_mode"] is True and call["stage"] == "not_started"
        # Server-built history only: the browser payload is replaced by the
        # stored session messages before the guided prompt is built.
        assert call["history"] == [{"role": "user", "content": "помоги начать"}]
        # The stage was recorded server-side through the facade with CAS. The
        # catalog task stays pinned as pending: a short reply never completes it.
        updates = [entry for entry in fake.calls if entry[0] == "update_lesson_state"]
        assert updates and updates[-1][1] == session
        assert updates[-1][2] == {"stage": "warmup", "pending_task_id": f"{saved['lesson_id']}:p1"}
        # Server-side history after the turn keeps the pairing.
        assert store.turns[session][-1]["role"] == "assistant"

    def test_guided_chat_replay_keeps_stage_from_persisted_state(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        fake.confirm_lesson(LEARNER, "t", saved["lesson_id"], expected_revision=1, request_id="c")
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        fake.attach_lesson(LEARNER, "t", session, saved["lesson_id"], request_id="a1")
        fake.states[session]["stage"] = "practice"
        payload = self._chat_payload(session, "что дальше?")
        first = client.post("/v1/chat/completions", headers=student, json=payload)
        assert first.status_code == 200
        assert client.app.state.rag_service.guided_calls[-1]["stage"] == "practice"


class TestFreeDialogueFastMemory:
    DEMO_LEARNER = "demo-nika-04"

    def _demo_api(self, fake=True):
        settings = LocalSettings()
        store = UnifiedStore()
        store.add_account("teacher-free", "teacher.free@example.test", "teacher", self.DEMO_LEARNER)
        store.add_account("student-free", "student.free@example.test", "student", self.DEMO_LEARNER)
        facade = FakeUnifiedMemory() if fake else None
        spy = SpyRag()
        app = create_app(settings=settings, store=store, runtime=None, rag_service=spy,
                         unified_memory_factory=(lambda s, r: facade) if fake
                         else None)  # None factory: the real merged facade is used
        client = TestClient(app, raise_server_exceptions=False)
        headers = auth(token(client, "student.free@example.test"))
        return client, facade, spy, headers

    def test_free_dialogue_passes_fast_memory_and_ordinary_history(self):
        client, fake, spy, headers = self._demo_api(fake=True)
        session = client.post(f"/learners/{self.DEMO_LEARNER}/learning-sessions", headers=headers,
                              json={"topic_id": "math.g4.numbers.division_by_1_2_digit"}).json()["learning_session_id"]
        reply = client.post("/v1/chat/completions", headers=headers, json={
            "learning_session_id": session, "client_message_id": "free-1",
            "messages": [{"role": "user", "content": "расскажи что-нибудь"}]})
        assert reply.status_code == 200, reply.text
        assert spy.calls, "the spy RAG service must receive the ordinary answer call"
        kwargs = spy.calls[-1]["kwargs"]
        assert kwargs["fast_memory_context"]["method"] == "unified_fast_memory"
        assert kwargs["fast_memory_context"]["learner_id"] == self.DEMO_LEARNER
        # No guided selection: the ordinary chat contract is untouched.
        assert "lesson_mode" not in json.dumps(reply.json())

    def test_fast_memory_failure_degrades_without_breaking_chat(self):
        # The real merged facade against an incompatible in-memory store must
        # degrade with a warning: the approved-memory path stays authoritative
        # and the chat keeps its durable contract.
        client, fake, spy, headers = self._demo_api(fake=False)
        session = client.post(f"/learners/{self.DEMO_LEARNER}/learning-sessions", headers=headers,
                              json={"topic_id": "math.g4.numbers.division_by_1_2_digit"}).json()["learning_session_id"]
        reply = client.post("/v1/chat/completions", headers=headers, json={
            "learning_session_id": session, "client_message_id": "free-2",
            "messages": [{"role": "user", "content": "привет"}]})
        assert reply.status_code == 200, reply.text
        kwargs = spy.calls[-1]["kwargs"]
        assert kwargs.get("fast_memory_context") is None


class TestCompletionAutoProcessing:
    def test_completion_enqueues_processing_once_for_attached_lessons(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        fake.confirm_lesson(LEARNER, "t", saved["lesson_id"], expected_revision=1, request_id="c")
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        fake.attach_lesson(LEARNER, "t", session, saved["lesson_id"], request_id="a1")
        completed = client.post(f"/learning-sessions/{session}/complete", headers=student)
        assert completed.status_code == 202
        assert completed.json()["session_processing_status"] == "queued"
        kinds = [job["kind"] for job in store.memory_jobs.values()]
        assert kinds.count("session_process") == 1
        # Completion is idempotent: no second processing job appears.
        client.post(f"/learning-sessions/{session}/complete", headers=student)
        assert [job["kind"] for job in store.memory_jobs.values()].count("session_process") == 1

    def test_completion_without_lesson_still_queues_processing_once(self):
        # Round-2 P0.3: an ordinary completed own conversation is processed
        # exactly once, with no attached lesson required.
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        completed = client.post(f"/learning-sessions/{session}/complete", headers=student)
        assert completed.status_code == 202
        assert completed.json()["session_processing_status"] == "queued"
        kinds = [job["kind"] for job in store.memory_jobs.values()]
        assert kinds.count("session_process") == 1
        job = [job for job in store.memory_jobs.values() if job["kind"] == "session_process"][0]
        assert job["payload"]["learning_session_id"] == session
        # Completion is idempotent: no second processing job appears.
        client.post(f"/learning-sessions/{session}/complete", headers=student)
        assert [job["kind"] for job in store.memory_jobs.values()].count("session_process") == 1

    def test_teacher_completion_enqueues_owner_session_processing(self):
        # The session owner's sources are processed even when the teacher
        # requested the completion; the job records the requesting actor.
        client, store, fake, teacher, student = make_api()
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        completed = client.post(f"/learning-sessions/{session}/complete", headers=teacher)
        assert completed.status_code == 202
        assert completed.json()["session_processing_status"] == "queued"
        job = [job for job in store.memory_jobs.values() if job["kind"] == "session_process"][0]
        assert job["learner_id"] == LEARNER
        assert job["payload"]["learning_session_id"] == session

    def test_completion_without_unified_backend_skips_enqueue(self):
        # Memory-disabled contract: no new processing without the backend.
        client, store, fake, teacher, student = make_api(facade=False)
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        completed = client.post(f"/learning-sessions/{session}/complete", headers=student)
        assert completed.status_code == 202
        assert "session_processing_status" not in completed.json()
        assert not [job for job in store.memory_jobs.values() if job["kind"] == "session_process"]
