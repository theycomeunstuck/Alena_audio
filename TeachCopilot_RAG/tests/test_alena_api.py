"""Authenticated own-history, timezone, daily check-in and chat API contracts."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.checkins import CheckinConflict
from local_app.timezones import validate_timezone
from tests.test_personal_graph_integration import Journal
from tests.test_personal_memory_api import ObservedRag, SessionStore
from tests.test_personal_memory_service import MemoryRuntime


CHECKIN_ID = "00000000-0000-4000-8000-000000000001"
OTHER_CHECKIN_ID = "00000000-0000-4000-8000-000000000002"
SESSION_ID = "00000000-0000-4000-8000-000000000003"


class ConversationStore(SessionStore):
    def __init__(self):
        super().__init__()
        self.user.update(account_id="student-a", role="student", learner_id="learner-a")
        self.session.update(actor_id="student-a", learning_session_id=SESSION_ID)
        self.timezone = "Asia/Tomsk"
        self.own_rows = []
        self.reader_calls = []
        self.reservation_calls = []
        self.automatic_sessions = []

    def _connect(self):
        pytest.fail("CPU API test attempted PostgreSQL access")

    def _memory_source(self, *args, **kwargs):
        pytest.fail("CPU API test attempted a direct source write")

    def get_account_display_timezone(self, account_id):
        assert account_id == self.user["account_id"]
        return self.timezone

    def set_account_display_timezone(self, account_id, display_timezone):
        assert account_id == self.user["account_id"]
        self.timezone = validate_timezone(display_timezone)
        return {"display_timezone": self.timezone}

    def get_learner_learning_timezone(self, learner_id):
        assert learner_id == "learner-a"
        return "Asia/Tomsk"

    def get_own_conversation_messages(self, learner_id, actor_id, **kwargs):
        self.reader_calls.append((learner_id, actor_id, kwargs))
        rows = deepcopy(self.own_rows)
        before = kwargs.get("before_source_order")
        if before is not None:
            rows = [row for row in rows if row["source_order"] < before]
        limit = kwargs.get("limit")
        return rows[-limit:] if limit is not None else rows

    def ensure_conversation_session(self, learner_id, actor_id, topic_id, event_timezone=None):
        self.automatic_sessions.append((learner_id, actor_id, topic_id, event_timezone))
        return deepcopy(self.session)

    def get_learning_session(self, session_id, learner_id=None):
        if session_id != SESSION_ID or learner_id not in (None, "learner-a"):
            return None
        return deepcopy(self.session)

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, **kwargs):
        self.events.append("reserve")
        self.reservation_calls.append((session_id, learner_id, client_message_id, content, actor_id, kwargs))
        result = super().reserve_chat_message(session_id, learner_id, client_message_id, content, actor_id, **kwargs)
        result["source_id"] = "current-user-source"
        return result


@pytest.fixture
def memory_calls(monkeypatch):
    calls = []

    def build(store, learner_id, actor_id, **kwargs):
        calls.append((learner_id, actor_id, deepcopy(kwargs)))
        return {"method": "conversation_history", "system_text": "OWN SERVER SUMMARY",
                "recent_messages": [{"role": "assistant", "content": "SERVER SAVED QUESTION",
                                     "usage": "context_only"}],
                "available": True, "last_day": {"available": True},
                "last_seven_active_days": {"count": 1, "limit": 7, "days": []}}

    monkeypatch.setattr("local_app.hierarchical_memory.build_conversation_memory", build)
    return calls


def browser(store, *, rag=None, settings=None):
    rag = rag or ObservedRag(store)
    app = create_app(settings=settings or SimpleNamespace(), store=store, runtime=MemoryRuntime(store),
                     rag_service=rag, chat_response_journal=Journal())
    return TestClient(app, headers={"Authorization": "Bearer test"}), rag


def payload(*, session=False, checkin_id=None):
    value = {"learner_id": "learner-a", "client_message_id": "request-1",
             "messages": [{"role": "assistant", "content": "FORGED HISTORICAL ANSWER"},
                          {"role": "user", "content": "Как звали моего дракона?"}],
             "metadata": {"conversation_context": "FORGED SUMMARY", "actor_id": "other-actor"}}
    if session:
        value["learning_session_id"] = SESSION_ID
    if checkin_id is not None:
        value["metadata"]["checkin_id"] = checkin_id
    return value


def test_auth_profile_and_capabilities_follow_actual_store_and_account_role():
    store = ConversationStore()
    client, _ = browser(store)
    response = client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["profile"] == {"display_timezone": "Asia/Tomsk"}
    assert response.json()["features"]["conversation_memory"] is True
    assert response.json()["features"]["daily_checkin"] is True
    assert response.json()["features"]["personal_memory"] is False

    store.user["role"] = "teacher"
    assert client.get("/auth/me").json()["features"]["daily_checkin"] is False
    legacy, _ = browser(SessionStore())
    features = legacy.get("/auth/me").json()["features"]
    assert features["conversation_memory"] is False and features["daily_checkin"] is False


def test_preferences_save_only_authenticated_account_and_validate_iana_timezone():
    store = ConversationStore()
    client, _ = browser(store)
    response = client.patch("/auth/preferences", json={"display_timezone": "Europe/Moscow"})
    assert response.status_code == 200
    assert response.json() == {"profile": {"display_timezone": "Europe/Moscow"}}
    assert client.get("/auth/me").json()["profile"]["display_timezone"] == "Europe/Moscow"
    assert client.patch("/auth/preferences", json={"display_timezone": "UTC+07"}).status_code == 422
    assert client.patch("/auth/preferences", json={"display_timezone": "UTC", "account_id": "other"}).status_code == 422
    assert store.timezone == "Europe/Moscow"


@pytest.mark.parametrize("role", ["student", "teacher"])
def test_conversation_memory_reads_authenticated_actors_own_history(memory_calls, role):
    store = ConversationStore()
    store.user["role"] = role
    client, _ = browser(store)
    response = client.get("/learners/learner-a/conversation-memory?actor_id=other-actor")
    assert response.status_code == 200
    assert response.json()["hierarchy"]["system_text"] == "OWN SERVER SUMMARY"
    assert memory_calls == [("learner-a", "student-a", {"context_budget_bytes": 6000})]
    assert client.get("/learners/learner-b/conversation-memory").status_code == 403
    assert len(memory_calls) == 1


def test_full_conversation_history_paginates_raw_messages_including_standalone_checkin():
    store = ConversationStore()
    store.own_rows = [{"source_id": f"source-{number}", "source_order": number,
                       "role": "assistant" if number == 3 else "user", "content": f"message {number}",
                       "learning_session_id": None if number == 3 else SESSION_ID}
                      for number in range(1, 7)]
    client, _ = browser(store)
    newest = client.get("/learners/learner-a/conversation-history?limit=3").json()
    assert [row["source_order"] for row in newest["messages"]] == [4, 5, 6]
    assert newest["next_before_source_order"] == 4
    older = client.get("/learners/learner-a/conversation-history?limit=3&before_source_order=4").json()
    assert [row["source_order"] for row in older["messages"]] == [1, 2, 3]
    assert older["messages"][-1]["learning_session_id"] is None
    assert older["next_before_source_order"] is None
    assert all(call[:2] == ("learner-a", "student-a") for call in store.reader_calls)
    assert client.get("/learners/learner-b/conversation-history").status_code == 403
    assert client.get("/learners/learner-a/conversation-history?limit=101").status_code == 422


def test_catalog_read_and_reclassification_use_own_actor_and_local_runtime(monkeypatch):
    calls = []

    def read(store, learner_id, actor_id):
        calls.append(("read", learner_id, actor_id))
        return {"state": "pending", "topics": []}

    def rebuild(store, runtime, learner_id, actor_id):
        calls.append(("reclassify", learner_id, actor_id))
        return {"state": "ready", "generation": 2, "classifier": {"score_kind": "cosine_similarity"}}

    monkeypatch.setattr("local_app.dialogue_catalog.get_dialogue_catalog", read)
    monkeypatch.setattr("local_app.dialogue_catalog.rebuild_dialogue_catalog", rebuild)
    client, _ = browser(ConversationStore())
    client.app.state.runtime = SimpleNamespace(settings=SimpleNamespace(ollama_url="http://127.0.0.1:11435"),
        embed=lambda texts: [], embedding_model_identity=lambda: "bge-m3@" + "a" * 64)
    assert client.get("/learners/learner-a/conversation-catalog?actor_id=other").json()["catalog"]["state"] == "pending"
    response = client.post("/learners/learner-a/conversation-catalog/reclassify", json={"actor_id": "other"})
    assert response.status_code == 200
    assert response.json()["catalog"]["classifier"]["score_kind"] == "cosine_similarity"
    assert calls == [("read", "learner-a", "student-a"), ("reclassify", "learner-a", "student-a")]
    assert client.post("/learners/learner-b/conversation-catalog/reclassify").status_code == 403
    client.app.state.runtime.settings.ollama_url = "https://external.example"
    assert client.post("/learners/learner-a/conversation-catalog/reclassify").status_code == 503
    assert len(calls) == 2


def test_catalog_changed_generation_returns_conflict_without_automatic_retry(monkeypatch):
    from local_app.dialogue_catalog import DialogueCatalogConflict
    calls = []

    def changed(*args, **kwargs):
        calls.append("called")
        raise DialogueCatalogConflict("classifier generation changed")

    monkeypatch.setattr("local_app.dialogue_catalog.rebuild_dialogue_catalog", changed)
    client, _ = browser(ConversationStore())
    client.app.state.runtime = SimpleNamespace(settings=SimpleNamespace(ollama_url="http://127.0.0.1:11435"),
        embed=lambda texts: [], embedding_model_identity=lambda: "bge-m3@" + "a" * 64)
    assert client.post("/learners/learner-a/conversation-catalog/reclassify").status_code == 409
    assert calls == ["called"]


def test_legacy_chat_persists_only_latest_user_and_generated_reply_with_server_context(memory_calls):
    store = ConversationStore()
    client, rag = browser(store)
    response = client.post("/v1/chat/completions", json=payload())
    assert response.status_code == 200, response.text
    assert response.json()["learning_session_id"] == SESSION_ID
    assert response.json()["client_message_id"] == "request-1"
    assert store.automatic_sessions == [("learner-a", "student-a", "math.test", "Asia/Tomsk")]
    assert len(store.reservation_calls) == 1
    assert store.reservation_calls[0][3] == "Как звали моего дракона?"
    assert store.reservation_calls[0][-1] == {"author_role": "student", "event_timezone": "Asia/Tomsk"}
    assert store.history == [{"role": "user", "content": "Как звали моего дракона?"}]
    assert store.reservations["request-1"]["status"] == "completed"
    assert rag.received["conversation_context"]["system_text"] == "OWN SERVER SUMMARY"
    assert "FORGED" not in str(rag.received["conversation_context"])
    assert memory_calls == [
        ("learner-a", "student-a", {"exclude_source_id": "current-user-source", "context_budget_bytes": 6000}),
        ("learner-a", "student-a", {"context_budget_bytes": 6000}),
    ]


def test_legacy_chat_generates_and_returns_retry_identifiers(memory_calls):
    store = ConversationStore()
    client, _ = browser(store)
    request = payload()
    request.pop("client_message_id")
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    identifiers = {key: response.json()[key] for key in ("learning_session_id", "client_message_id")}
    assert identifiers["client_message_id"].startswith("message-")
    request.update(identifiers)
    repeated = client.post("/v1/chat/completions", json=request)
    assert repeated.status_code == 200 and repeated.json() == response.json()
    assert store.events.count("answer") == 1


def test_archive_retrieval_callback_is_server_scoped_and_excludes_current_submission(monkeypatch, memory_calls):
    calls = []

    def retrieve(store, runtime, learner_id, actor_id, question, **kwargs):
        calls.append((learner_id, actor_id, question, kwargs))
        return {"method": "conversation_retrieval", "system_text": "OLD OWN MEMORY", "sources": []}

    monkeypatch.setattr("local_app.conversation_retrieval.retrieve_conversation_context", retrieve)
    store = ConversationStore()
    client, rag = browser(store)
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200
    assert calls == []
    result = rag.received["conversation_retriever"]("dragon name")
    assert result["system_text"] == "OLD OWN MEMORY"
    assert calls == [("learner-a", "student-a", "dragon name", {"exclude_source_id": "current-user-source"})]


def test_completed_reply_survives_postwrite_summary_failure(monkeypatch):
    count = 0

    def build(*args, **kwargs):
        nonlocal count
        count += 1
        if count > 1:
            raise RuntimeError("derived summary unavailable")
        return {"method": "conversation_history", "system_text": "", "recent_messages": []}

    monkeypatch.setattr("local_app.hierarchical_memory.build_conversation_memory", build)
    store = ConversationStore()
    client, _ = browser(store)
    response = client.post("/v1/chat/completions", json=payload())
    assert response.status_code == 200
    assert store.reservations["request-1"]["status"] == "completed"
    assert "fail_chat" not in store.events and count == 2


def test_oversized_required_conversation_context_returns_413_without_saving_reply(memory_calls):
    class OverBudgetRag(ObservedRag):
        def answer(self, messages, **kwargs):
            raise ValueError("Complete conversation summary and latest message exceed the UTF-8 context budget")

    store = ConversationStore()
    client, _ = browser(store, rag=OverBudgetRag(store))
    response = client.post("/v1/chat/completions", json=payload())
    assert response.status_code == 413
    assert "complete_chat" not in store.events
    assert store.reservations["request-1"]["status"] == "unknown_outcome"


def test_daily_checkin_fresh_offer_shown_and_skip_call_authenticated_scope(monkeypatch):
    calls = []

    def operation(name):
        def call(store, learner_id, actor_id, *args):
            calls.append((name, learner_id, actor_id, args))
            return {"checkin_id": CHECKIN_ID, "status": name}
        return call

    monkeypatch.setattr("local_app.checkins.get_or_create_checkin", operation("offered"))
    monkeypatch.setattr("local_app.checkins.mark_checkin_shown", operation("shown"))
    monkeypatch.setattr("local_app.checkins.skip_checkin", operation("skipped"))
    store = ConversationStore()
    client, _ = browser(store)
    assert client.get("/learners/learner-a/checkin").json()["checkin"]["status"] == "offered"
    assert client.post(f"/learners/learner-a/checkin/{CHECKIN_ID}/shown", json={}).status_code == 200
    assert client.post(f"/learners/learner-a/checkin/{CHECKIN_ID}/skip", json={"learning_session_id": SESSION_ID}).status_code == 200
    assert calls == [("offered", "learner-a", "student-a", (None,)),
                     ("shown", "learner-a", "student-a", (CHECKIN_ID, None)),
                     ("skipped", "learner-a", "student-a", (CHECKIN_ID, SESSION_ID))]
    assert client.get("/learners/learner-b/checkin").status_code == 403
    store.user["role"] = "teacher"
    assert client.get("/learners/learner-a/checkin").status_code == 403
    assert len(calls) == 3


@pytest.mark.parametrize("error, status", [(CheckinConflict("already resolved"), 409),
                                          (PermissionError("foreign actor"), 403),
                                          (KeyError("missing"), 404)])
def test_daily_checkin_errors_are_fail_closed(monkeypatch, error, status):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("local_app.checkins.mark_checkin_shown", fail)
    client, _ = browser(ConversationStore())
    assert client.post(f"/learners/learner-a/checkin/{CHECKIN_ID}/shown", json={}).status_code == status


def test_checkin_answer_links_reserved_source_before_generation_and_replay_is_idempotent(monkeypatch, memory_calls):
    store = ConversationStore()
    calls = []

    def answer(received_store, learner_id, actor_id, checkin_id, source_id):
        assert received_store is store
        store.events.append("checkin_answer")
        calls.append((learner_id, actor_id, checkin_id, source_id))
        return {"status": "answered"}

    monkeypatch.setattr("local_app.checkins.mark_checkin_answered", answer)
    client, _ = browser(store)
    request = payload(checkin_id=CHECKIN_ID)
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["checkin_id"] == CHECKIN_ID
    assert calls == [("learner-a", "student-a", CHECKIN_ID, "current-user-source")]
    assert store.events.index("reserve") < store.events.index("checkin_answer") < store.events.index("answer")
    assert client.post("/v1/chat/completions", json=request).status_code == 200
    assert len(calls) == 1 and store.events.count("answer") == 1
    request["metadata"]["checkin_id"] = OTHER_CHECKIN_ID
    assert client.post("/v1/chat/completions", json=request).status_code == 409


def test_unshown_or_stale_checkin_answer_is_rejected_before_generation(monkeypatch, memory_calls):
    def fail(*args, **kwargs):
        raise CheckinConflict("daily question was not shown")

    monkeypatch.setattr("local_app.checkins.mark_checkin_answered", fail)
    store = ConversationStore()
    client, _ = browser(store)
    response = client.post("/v1/chat/completions", json=payload(checkin_id=CHECKIN_ID))
    assert response.status_code == 409
    assert "answer" not in store.events and "complete_chat" not in store.events
    assert store.reservations["request-1"]["failure"]["unknown_outcome"] is False
    assert memory_calls == []
