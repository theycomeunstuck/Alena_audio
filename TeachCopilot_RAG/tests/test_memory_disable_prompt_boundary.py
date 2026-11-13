"""Exercise real RAG prompt assembly while personal memory is disabled."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from local_app.rag import RagService
from tests.test_alena_api import ConversationStore, browser, payload


class Runtime:
    context_tokens = 8192

    def __init__(self, *, social=False):
        self.calls = []
        self.generations = []
        self.social = social

    def resolve_chat_query(self, latest, history, subject, **kwargs):
        self.calls.append((latest, deepcopy(history), deepcopy(kwargs)))
        if self.social:
            return {"in_scope": False, "uses_prior_context": False,
                    "social_turn": True, "search_query": ""}
        return {"in_scope": True, "uses_prior_context": False,
                "uses_approved_memory": False, "search_query": "дробь"}

    def chat(self, messages, **kwargs):
        self.generations.append(deepcopy(messages))
        return {"message": {"content": "Здорово! Что было интереснее всего?"}}


class Adapter:
    external_client = None

    def __init__(self):
        self.calls = []

    def answer(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return {"message": {"content": json.dumps({"answer": "Разберём дробь.", "citations": []},
                                                     ensure_ascii=False)}}


class RealPromptRag(RagService):
    def search(self, *args, **kwargs):
        return [{"id": "synthetic-material", "content": "Дробь обозначает часть целого.",
                 "topic_id": "math.test", "subject_id": "math"}]


def test_disabled_memory_stays_out_of_resolver_and_real_rag_prompt(monkeypatch):
    store = ConversationStore()
    store.history.extend([{"role": "user", "content": "PRIVATE_OLD_USER_MESSAGE"},
                          {"role": "assistant", "content": "PRIVATE_OLD_ASSISTANT_REPLY"}])
    store.get_card = lambda learner: {"learner_id": learner, "grade": 4,
        "interests": ["PRIVATE_INTEREST"],
        "rag_context": {"current_goal": {"topic_id": "math.test"}, "detail": "PRIVATE_PROFILE"}}
    monkeypatch.setattr(store, "get_learner_memory",
                        lambda *args, **kwargs: pytest.fail("approved memory must not be read"))
    refreshes = []
    monkeypatch.setattr("local_app.hierarchical_memory.build_conversation_memory",
                        lambda *args, **kwargs: refreshes.append(True) or {"system_text": "PRIVATE_SUMMARY"})
    runtime, adapter = Runtime(), Adapter()
    settings = SimpleNamespace(personal_memory_enabled=False, chat_model="synthetic-model")
    rag = RealPromptRag(store, runtime, settings, answer_adapter=adapter)
    client, _ = browser(store, rag=rag, settings=settings)
    request = payload()
    request["messages"][-1]["content"] = "Объясни дробь."

    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert len(runtime.calls) == len(adapter.calls) == 1
    latest, history, resolver_options = runtime.calls[0]
    assert latest == "Объясни дробь." and history == []
    assert resolver_options.get("approved_memory") is None
    assert "conversation_context" not in resolver_options
    messages, options = adapter.calls[0]
    assert messages[-1] == {"role": "user", "content": "Объясни дробь."}
    assert [message["role"] for message in messages] == ["system", "user"]
    projection = asdict(options["projection"])
    assert projection["dialogue"] == [] and projection["memory"] == []
    assert projection["interests"] == [] and projection["avoided_interests"] == []
    sent = json.dumps({"messages": messages, "projection": projection}, ensure_ascii=False)
    for old_value in ("PRIVATE_OLD_USER_MESSAGE", "PRIVATE_OLD_ASSISTANT_REPLY", "PRIVATE_INTEREST",
                      "PRIVATE_PROFILE", "PRIVATE_SUMMARY", "FORGED HISTORICAL ANSWER", "FORGED SUMMARY"):
        assert old_value not in sent
    assert refreshes == [True]


def test_disabled_memory_stays_out_of_the_real_social_chat_prompt(monkeypatch):
    store = ConversationStore()
    store.history.extend([{"role": "user", "content": "PRIVATE_OLD_USER_MESSAGE"},
                          {"role": "assistant", "content": "PRIVATE_OLD_ASSISTANT_REPLY"}])
    store.get_card = lambda learner: {"learner_id": learner, "grade": 4,
        "interests": ["PRIVATE_INTEREST"],
        "rag_context": {"current_goal": {"topic_id": "math.test"}, "detail": "PRIVATE_PROFILE"}}
    monkeypatch.setattr(store, "get_learner_memory",
                        lambda *args, **kwargs: pytest.fail("approved memory must not be read"))
    refreshes = []
    monkeypatch.setattr("local_app.hierarchical_memory.build_conversation_memory",
                        lambda *args, **kwargs: refreshes.append(True) or {"system_text": "PRIVATE_SUMMARY"})
    runtime, adapter = Runtime(social=True), Adapter()
    settings = SimpleNamespace(personal_memory_enabled=False, chat_model="synthetic-model")
    rag = RealPromptRag(store, runtime, settings, answer_adapter=adapter)
    client, _ = browser(store, rag=rag, settings=settings)
    request = payload()
    request["messages"][-1]["content"] = "Сегодня я построил дом."

    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert len(runtime.calls) == len(runtime.generations) == 1 and adapter.calls == []
    latest, history, resolver_options = runtime.calls[0]
    assert latest == "Сегодня я построил дом." and history == []
    assert resolver_options.get("approved_memory") is None
    assert "conversation_context" not in resolver_options
    messages = runtime.generations[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "Сегодня я построил дом."
    sent = json.dumps(messages, ensure_ascii=False)
    for old_value in ("PRIVATE_OLD_USER_MESSAGE", "PRIVATE_OLD_ASSISTANT_REPLY", "PRIVATE_INTEREST",
                      "PRIVATE_PROFILE", "PRIVATE_SUMMARY", "FORGED HISTORICAL ANSWER", "FORGED SUMMARY"):
        assert old_value not in sent
    assert refreshes == [True]
