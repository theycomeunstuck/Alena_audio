"""The memory switch affects model context, not durable conversation storage."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_alena_api import ConversationStore, ObservedRag, browser, payload


class CapturingRag(ObservedRag):
    def answer(self, messages, **kwargs):
        self.messages = deepcopy(messages)
        return super().answer(messages, **kwargs)


def test_disabled_memory_does_not_forward_prior_messages_or_archive_but_still_records(monkeypatch):
    store = ConversationStore()
    store.history.extend([
        {"role": "user", "content": "PREVIOUS PRIVATE CONVERSATION"},
        {"role": "assistant", "content": "PREVIOUS ASSISTANT REPLY"},
    ])
    rag = CapturingRag(store)
    settings = SimpleNamespace(personal_memory_enabled=False, experimental_personal_graph_enabled=True)

    def forbidden(*args, **kwargs):
        pytest.fail("disabled personal memory was read or refreshed")

    monkeypatch.setattr(store, "get_learner_memory", forbidden)
    monkeypatch.setattr("local_app.hierarchical_memory.build_conversation_memory", forbidden)
    monkeypatch.setattr("local_app.conversation_retrieval.retrieve_conversation_context", forbidden)
    client, _ = browser(store, rag=rag, settings=settings)
    response = client.post("/v1/chat/completions", json=payload())

    assert response.status_code == 200, response.text
    assert rag.messages == [{"role": "user", "content": "Как звали моего дракона?"}]
    assert rag.received["profile"] == {}
    assert rag.received["preference_context"] == {"interests": [], "avoided_interests": []}
    assert rag.received["memory_context"] == {"text": "", "sources": []}
    assert rag.received["memory_projection"] == []
    assert "conversation_context" not in rag.received
    assert "conversation_retriever" not in rag.received
    assert store.history[-1]["content"] == "Как звали моего дракона?"
    assert store.history[0]["content"] == "PREVIOUS PRIVATE CONVERSATION"
    assert store.reservations["request-1"]["status"] == "completed"
    assert response.json()["memory_version_id"] is None


@pytest.mark.parametrize("method", ["personal_memory", "personal_graph"])
def test_disabled_memory_rejects_explicit_personal_search_before_writing(method):
    store = ConversationStore()
    settings = SimpleNamespace(personal_memory_enabled=False, experimental_personal_graph_enabled=True)
    client, rag = browser(store, settings=settings)
    request = payload()
    request["metadata"]["memory_method"] = method
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 503
    assert store.reservations == {}
    assert rag.received is None


def test_disabling_memory_does_not_hide_the_user_readable_archive():
    store = ConversationStore()
    store.own_rows = [{"source_id": "source-1", "source_order": 1, "role": "user",
                       "content": "сохранённая реплика", "learning_session_id": None}]
    client, _ = browser(store, settings=SimpleNamespace(personal_memory_enabled=False))
    assert client.get("/auth/me").json()["features"]["conversation_memory_in_responses"] is False
    response = client.get("/learners/learner-a/conversation-history")
    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "сохранённая реплика"
