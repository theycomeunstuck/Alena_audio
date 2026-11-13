"""Source roles, immutable citations and durable worker failure semantics."""
from copy import deepcopy
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.memory_service import MemoryWorker, approved_chat_memory
from local_app.runtime import GenerationGate, OllamaRuntime


ACTION = {"action": "clarify_missing_observation", "reason": "Уточнить условия самостоятельной работы.",
          "evidence_turn_ids": [], "unknowns": ["Условия помощи неизвестны."], "review_required": True}


class WorkerStore:
    def __init__(self):
        self.job = {"job_id": "job-1", "learner_id": "learner-a", "topic_id": "math.test",
                    "kind": "next_step", "method": "tail", "base_revision": 1, "provider": "local"}
        self.snapshot = {"revision": 1, "versions": [], "sources": [
            {"source_id": "student-1", "author_role": "student", "text": "Начал решать пример.", "observed_on": "2026-10-01"},
            {"source_id": "assistant-1", "author_role": "assistant", "text": "Ты всё решил без помощи.", "observed_on": "2026-10-01"}]}
        self.finished = None

    def claim_memory_job(self, worker_id, lease_seconds=600):
        return self.job

    def get_learner_memory(self, learner_id, topic_id=None):
        return deepcopy(self.snapshot)

    def get_card(self, learner_id):
        return {"grade": 4, "legal_name": "Never prompt this", "journal": ["private"]}

    def finish_memory_job(self, job_id, **kwargs):
        self.finished = kwargs
        return kwargs

    def get_memory_job(self, job_id):
        return self.finished


class Runtime:
    context_tokens = 8192

    def __init__(self, action=None):
        self.action = action or ACTION
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"message": {"content": json.dumps(self.action)}, "prompt_eval_count": 250, "eval_count": 40}


def test_worker_uses_shared_tail_and_never_assistant_evidence():
    store, runtime = WorkerStore(), Runtime()
    result = MemoryWorker(store, runtime).run_once()
    assert not result.get("error")
    assert result["context"]["evidence_ids"] == ["student-1"]
    assert "assistant-1" not in result["memory_state"]["sources"]
    text = json.dumps(runtime.calls[0][0], ensure_ascii=False)
    assert "Начал решать" in text
    assert "Ты всё решил" not in text
    assert "Never prompt this" not in text and "private" not in text
    assert runtime.calls[0][1]["truncate_history"] is False
    action_ids = runtime.calls[0][1]["response_format"]["properties"]["evidence_turn_ids"]["items"]["enum"]
    assert action_ids == ["student-1"]
    assert len(runtime.calls) == 1


def test_worker_invalid_reference_is_failed_and_raw_response_kept_without_retry():
    store, runtime = WorkerStore(), Runtime({**ACTION, "evidence_turn_ids": ["assistant-1"]})
    MemoryWorker(store, runtime).run_once()
    assert store.finished["error"]["type"] == "ContractValidationError"
    assert len(runtime.calls) == 1
    assert "assistant-1" in store.finished["error"]["calls"][0]["raw_response"]["message"]["content"]
    assert "memory_state" not in store.finished


def test_structured_v2_action_citations_are_limited_to_learning_records():
    from local_app.memory_service import MemoryWorker
    from research_memory.contracts import ContractValidationError, action_schema, validate_action

    store = WorkerStore()
    store.job["method"] = "structured_v2"
    store.snapshot["sources"] = [
        {"source_id": "teacher-task", "author_role": "teacher", "text": "Compare 1/4 and 1/8.",
         "observed_on": "2026-10-01", "topic_id": "math.test", "session_id": "session-v2"},
        {"source_id": "student-error", "author_role": "student", "text": "1/8 is bigger because 8 is bigger.",
         "observed_on": "2026-10-01", "topic_id": "math.test", "session_id": "session-v2"},
        {"source_id": "student-interest", "author_role": "student", "text": "I love comets.",
         "observed_on": "2026-10-01", "topic_id": "math.test", "session_id": "session-v2"},
        {"source_id": "student-session", "author_role": "student", "text": "Please give me one task at a time.",
         "observed_on": "2026-10-01", "topic_id": "math.test", "session_id": "session-v2"},
    ]

    def extraction_record(kind, *, topic_id, status, evidence, **values):
        return {
            "record_type": kind, "topic_id": topic_id, "context": "A short observed event.",
            "interpretation": "A cautious interpretation.", "interpretation_status": status,
            "support_level": None if topic_id is None else "unknown", "support_evidence": None,
            "independent_recheck": None if topic_id is None else "unknown", "recheck_evidence": None,
            "evidence": [{"turn_id": evidence[0], "quote": evidence[1]}], "links": [],
            "interest_operation": None, "interest": None, "preference": None, "fact": None,
            "claim_status": None, "retention_class": None, "corrects_record_id": None,
            **values,
        }

    raw_records = [
        extraction_record("child_error", topic_id="math.test", status="explicit",
                          evidence=("student-error", "1/8 is bigger because 8 is bigger.")),
        extraction_record("interest_candidate", topic_id=None, status="candidate",
                          evidence=("student-interest", "I love comets."), interest_operation="add",
                          interest="comets", claim_status="candidate", retention_class="profile_candidate"),
        extraction_record("session_preference_candidate", topic_id=None, status="candidate",
                          evidence=("student-session", "Please give me one task at a time."),
                          preference="one task at a time", claim_status="candidate", retention_class="session_only"),
    ]
    worker = MemoryWorker(store, Runtime())
    worker._call = lambda _messages, _purpose, _schema: json.dumps({"records": raw_records})

    _state, context = worker._prepare(store.job)

    assert context["evidence_ids"] == ["student-error", "student-interest"]
    assert context["action_evidence_ids"] == ["student-error"]
    memory = json.loads(context["text"])
    assert memory["interest_candidates"][0]["interest"] == "comets"
    assert memory["session_preferences"] == []

    allowed = set(context["action_evidence_ids"])
    schema_ids = action_schema(allowed)["properties"]["evidence_turn_ids"]["items"]["enum"]
    assert schema_ids == ["student-error"]
    accepted = {**ACTION, "evidence_turn_ids": ["student-error"]}
    assert validate_action(json.dumps(accepted), allowed)["evidence_turn_ids"] == ["student-error"]
    rejected = {**ACTION, "evidence_turn_ids": ["student-interest"]}
    with pytest.raises(ContractValidationError) as failure:
        validate_action(json.dumps(rejected), allowed)
    assert failure.value.code == "missing_source"


def test_non_v2_methods_keep_the_existing_action_evidence_default():
    from research_memory.methods import Memory

    call = lambda *_args: ""
    for method in ("profile_only", "tail", "summary", "structured"):
        memory = Memory(method, call, 1000, "math.test")
        context = memory.context("math.test")
        assert context.action_evidence_ids is None

    tail = Memory("tail", call, 1000, "math.test")
    tail.update({"lesson_id": "lesson-1", "observed_on": "2026-10-01",
                 "turns": [{"turn_id": "tail-student", "role": "student", "text": "An answer."}]})
    context = tail.context("math.test")
    assert context.evidence_ids == {"tail-student"}
    assert context.action_evidence_ids is None


def test_worker_preserves_raw_truncated_runtime_response_without_retry():
    from local_app.runtime import IncompleteCompletionError
    store, runtime = WorkerStore(), Runtime()
    raw = {"message": {"content": '{"action":'}, "done_reason": "length", "prompt_eval_count": 81, "eval_count": 1536}
    def truncated(*args, **kwargs):
        raise IncompleteCompletionError("truncated", raw_response=raw)
    runtime.chat = truncated
    MemoryWorker(store, runtime).run_once()
    calls = store.finished["error"]["calls"]
    assert len(calls) == 1 and calls[0]["raw_response"] == raw
    assert calls[0]["completion_tokens"] == 1536


def test_only_approved_tail_is_selected_and_staleness_remains_visible():
    snapshot = {"revision": 2, "versions": [
        {"approved": False, "method": "tail", "topic_id": "math.test", "context": {"text": "DRAFT"}},
        {"approved": True, "method": "structured", "topic_id": "math.test", "context": {"text": "DEMO"}},
        {"approved": True, "method": "tail", "topic_id": "math.test", "version_id": "v1", "base_revision": 1,
         "context": {"text": "APPROVED", "sources": []}}]}
    selected = approved_chat_memory(snapshot, "math.test")
    assert selected["text"] == "APPROVED" and selected["stale"]
    assert selected["version_id"] == "v1"


class ApiStore:
    def __init__(self):
        self.user = {"account_id": "teacher-a", "role": "teacher"}
        self.jobs = {}
        self.reservations = {}
        self.history = []
        self.session = {"learning_session_id": "session-a", "learner_id": "learner-a", "actor_id": "teacher-a", "topic_id": "math.test", "status": "active"}

    def get_session(self, token):
        return {"account": self.user}

    def authorize_learner_access(self, account, learner):
        return learner == "learner-a"

    def get_card(self, learner):
        return {"learner_id": learner, "grade": 4, "rag_context": {}}

    def get_revision(self, learner):
        return 1

    def get_learner_memory(self, learner, topic=None):
        return {"learner_id": learner, "revision": 1, "sources": [], "versions": []}

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload, actor_id, **kwargs):
        job = {"job_id": "job-a", "learner_id": learner_id, "topic_id": topic_id, "kind": kind,
               "status": "queued", "method": kwargs.get("method", "tail")}
        self.jobs["job-a"] = job
        return job

    def get_memory_job(self, job_id):
        return self.jobs.get(job_id)

    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        job = self.jobs[job_id]
        job.update(status="running", worker_id=worker_id)
        return job

    def start_memory_job_call(self, job_id, worker_id):
        self.jobs[job_id]["call_started"] = True

    def finish_memory_job(self, job_id, **kwargs):
        job = self.jobs[job_id]
        job["status"] = "unknown_outcome" if kwargs.get("unknown_outcome") else "failed" if kwargs.get("error") else "completed"
        return job

    def get_learning_session(self, session_id, learner_id=None):
        return self.session if session_id == "session-a" else None

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, **kwargs):
        if client_message_id in self.reservations:
            return {**self.reservations[client_message_id], "created": False}
        self.history.append({"role": "user", "content": content})
        self.reservations[client_message_id] = {"status": "pending"}
        return {"status": "pending", "created": True}

    def list_learning_messages(self, session_id):
        return self.history[:]

    def complete_chat_message(self, session_id, client_message_id, response):
        self.reservations[client_message_id] = {"status": "completed", "response": deepcopy(response)}
        self.history.append({"role": "assistant", "content": response["choices"][0]["message"]["content"]})

    def fail_chat_message(self, session_id, client_message_id, **kwargs):
        self.reservations[client_message_id] = {"status": "unknown_outcome"}


class Rag:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def answer(self, messages, **kwargs):
        self.calls.append(deepcopy(messages))
        if self.fail:
            raise TimeoutError("unknown provider outcome")
        return {"choices": [{"message": {"role": "assistant", "content": "Попробуй объяснить первый шаг."}}]}


def client(store, rag):
    app = create_app(settings=SimpleNamespace(), store=store, runtime=object(), rag_service=rag)
    return TestClient(app, headers={"Authorization": "Bearer test"})


def test_teacher_queue_default_tail_and_foreign_job_denied():
    store = ApiStore()
    browser = client(store, Rag())
    response = browser.post("/learners/learner-a/next-step", json={"topic_id": "math.test"})
    assert response.status_code == 202
    assert store.jobs["job-a"]["method"] == "tail"
    store.jobs["foreign"] = {"learner_id": "learner-b", "kind": "next_step"}
    assert browser.get("/memory/jobs/foreign").status_code == 403
    assert browser.get("/learners/learner-b/memory").status_code == 403


def test_teacher_confirms_only_explicitly_selected_conversation_interests():
    class ReviewStore(ApiStore):
        def approve_memory_version(self, learner_id, version_id, actor_id, expected_revision,
                                   interest_candidate_ids=None):
            self.selected = interest_candidate_ids
            return {"version_id": version_id, "approved": True}

    store = ReviewStore()
    response = client(store, Rag()).post(
        "/learners/learner-a/memory/versions/version-a/confirm",
        json={"expected_revision": 1, "interest_candidate_ids": ["interest-one"]},
    )
    assert response.status_code == 200
    assert store.selected == ["interest-one"]


def test_personal_graph_confirmation_requires_explicit_fingerprint_for_full_archive():
    class ReviewStore(ApiStore):
        def approve_memory_version(self, learner_id, version_id, actor_id, expected_revision,
                                   interest_candidate_ids=None, **options):
            self.review = {"interest_candidate_ids": interest_candidate_ids, **options}
            return {"version_id": version_id, "approved": True}

    store = ReviewStore()
    fingerprint = "a" * 64
    response = client(store, Rag()).post(
        "/learners/learner-a/memory/versions/version-a/confirm",
        json={"expected_revision": 1, "reviewed_full_archive": True, "allow_full_archive": True,
              "archive_snapshot_fingerprint": fingerprint},
    )
    assert response.status_code == 200
    assert store.review["full_archive_access"] is True
    assert store.review["expected_archive_fingerprint"] == fingerprint

    unreviewed = client(ReviewStore(), Rag()).post(
        "/learners/learner-a/memory/versions/version-a/confirm",
        json={"expected_revision": 1, "allow_full_archive": True,
              "archive_snapshot_fingerprint": fingerprint},
    )
    assert unreviewed.status_code == 422

    missing = client(ReviewStore(), Rag()).post(
        "/learners/learner-a/memory/versions/version-a/confirm",
        json={"expected_revision": 1, "allow_full_archive": True},
    )
    assert missing.status_code == 422


def test_legacy_memory_confirmation_does_not_pass_new_archive_options_to_store():
    class ReviewStore(ApiStore):
        def approve_memory_version(self, learner_id, version_id, actor_id, expected_revision,
                                   interest_candidate_ids=None):
            self.selected = interest_candidate_ids
            return {"version_id": version_id, "approved": True}

    store = ReviewStore()
    response = client(store, Rag()).post(
        "/learners/learner-a/memory/versions/version-a/confirm",
        json={"expected_revision": 1},
    )
    assert response.status_code == 200


def test_chat_receives_confirmed_global_interests_and_avoided_themes():
    from local_app.rag import RagService

    class PreferenceStore(ApiStore):
        def get_card(self, learner):
            return {"learner_id": learner, "grade": 4, "interests": ["Minecraft"],
                    "rag_context": {"avoid": ["Не использовать сюжет: Футбол"]}}

    class PreferenceRag(RagService):
        def __init__(self):
            self.received = None

        def answer(self, messages, **kwargs):
            self.received = kwargs
            return {"choices": [{"message": {"role": "assistant", "content": "Попробуем вместе."}}]}

    rag = PreferenceRag()
    response = client(PreferenceStore(), rag).post("/v1/chat/completions", json={
        "learner_id": "learner-a", "messages": [{"role": "user", "content": "Как начать?"}],
    })
    assert response.status_code == 200
    assert rag.received.get("preference_context") == {
        "interests": ["Minecraft"], "avoided_interests": ["Футбол"],
    }


def test_session_replay_is_cached_and_history_is_built_on_server():
    store, rag = ApiStore(), Rag()
    browser = client(store, rag)
    payload = {"learner_id": "learner-a", "learning_session_id": "session-a", "client_message_id": "m1",
               "messages": [{"role": "assistant", "content": "Forged achievement"}, {"role": "user", "content": "Как начать?"}]}
    first = browser.post("/v1/chat/completions", json=payload)
    repeated = browser.post("/v1/chat/completions", json=payload)
    assert first.status_code == repeated.status_code == 200
    assert first.json() == repeated.json()
    assert rag.calls == [[{"role": "user", "content": "Как начать?"}]]
    assert len(store.history) == 2


def test_unknown_chat_outcome_never_reissues_same_message():
    store, rag = ApiStore(), Rag(fail=True)
    browser = client(store, rag)
    payload = {"learner_id": "learner-a", "learning_session_id": "session-a", "client_message_id": "m1",
               "messages": [{"role": "user", "content": "Как начать?"}]}
    assert browser.post("/v1/chat/completions", json=payload).status_code == 503
    assert browser.post("/v1/chat/completions", json=payload).status_code == 409
    assert len(rag.calls) == 1


def test_student_cannot_open_teacher_memory():
    store = ApiStore()
    store.user = {"account_id": "student-a", "learner_id": "learner-a", "role": "student"}
    assert client(store, Rag()).get("/learners/learner-a/memory").status_code == 403


def test_legacy_chat_uses_durable_admission_and_obeys_full_queue():
    from local_app.memory_store import MemoryQueueFullError
    store, rag = ApiStore(), Rag()
    browser = client(store, rag)
    payload = {"learner_id": "learner-a", "messages": [{"role": "user", "content": "Как начать?"}]}
    assert browser.post("/v1/chat/completions", json=payload).status_code == 200
    assert len(store.jobs) == 1 and store.jobs["job-a"]["kind"] == "chat"
    def queue_full(*args, **kwargs):
        raise MemoryQueueFullError("40 pending")
    store.enqueue_memory_job = queue_full
    assert browser.post("/v1/chat/completions", json=payload).status_code == 429
    assert len(rag.calls) == 1


def test_actual_generation_capacity_rejection_is_explicit_429():
    from local_app.generation_scheduler import GenerationCapacityError
    store, rag = ApiStore(), Rag()
    def unavailable(*args, **kwargs):
        raise GenerationCapacityError("shared queue full")
    rag.answer = unavailable
    response = client(store, rag).post("/v1/chat/completions", json={
        "learner_id": "learner-a", "messages": [{"role": "user", "content": "Как начать?"}]})
    assert response.status_code == 429 and response.headers["Retry-After"] == "10"
    assert store.jobs["job-a"]["status"] == "failed"


def test_task_rejects_recommendation_after_source_revision_changed():
    store = ApiStore()
    store.get_next_step = lambda _: {"learner_id": "learner-a", "base_revision": 1,
                                   "source_revision": 2, "topic_id": "math.test"}
    store.get_learner_source_revision = lambda _: 3
    response = client(store, Rag()).post("/task/design", json={"learner_id": "learner-a", "next_step_id": "step-1"})
    assert response.status_code == 409


def test_application_reuses_ready_memory_when_writer_input_is_too_large(monkeypatch):
    from local_app.memory_service import memory_state, restore_memory
    store, runtime = WorkerStore(), Runtime()
    store.job["method"] = "summary"
    prior = restore_memory({}, runtime.chat, method="summary", topic_id="math.test")
    prior.sources = {"student-old": {"text": "Предыдущее наблюдение.", "role": "student"}}
    prior.summary = "[student-old] Помощь неизвестна."
    store.snapshot["sources"].insert(0, {"source_id": "student-old", "author_role": "student", "text": "Предыдущее наблюдение.", "observed_on": "2026-09-01"})
    store.snapshot["versions"] = [{"version_id": "ready-1", "method": "summary", "state": memory_state(prior)}]
    monkeypatch.setattr("local_app.memory_service.input_budget_diagnostic", lambda *args, **kwargs:
                        {"enforced": kwargs["purpose"] == "summary", "within_guard": kwargs["purpose"] != "summary"})
    MemoryWorker(store, runtime).run_once()
    assert store.finished["memory_state"] is None
    assert store.finished["result"]["memory_version_id"] == "ready-1"
    assert store.finished["next_step"]["stale"] is True
    assert store.finished["context"]["evidence_ids"] == ["student-old"]
    assert len(runtime.calls) == 1
    assert "memory_writer_input_exceeds_context" in json.dumps(runtime.calls[0][0])


def test_failed_writer_update_retains_prior_version_and_reports_failure(monkeypatch):
    from local_app.memory_service import memory_state, restore_memory
    store, runtime = WorkerStore(), Runtime()
    store.job.update(method="summary", kind="memory_update")
    prior = restore_memory({}, runtime.chat, method="summary", topic_id="math.test")
    prior.summary = "Недостаточно сведений."
    store.snapshot["versions"] = [{"version_id": "ready-1", "method": "summary", "state": memory_state(prior)}]
    monkeypatch.setattr("local_app.memory_service.input_budget_diagnostic", lambda *args, **kwargs:
                        {"enforced": True, "within_guard": False})
    MemoryWorker(store, runtime).run_once()
    assert "memory_state" not in store.finished
    assert store.finished["error"]["message"] == "memory_writer_input_exceeds_context"
    assert store.finished["result"]["stale"] and not runtime.calls
