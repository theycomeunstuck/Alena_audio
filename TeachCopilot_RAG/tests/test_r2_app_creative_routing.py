"""Round-2 P0.4: bounded creative/history routing through the existing resolver.

A request such as "continue the story we started last week" must retrieve
older own-conversation context without a literal "remember" trigger, use up
to 2048 output tokens with the complete-input budget checked, and never go
through the 192-token social route. Fiction stays fiction; greetings stay
brief; legacy decisions without the new fields keep routing unchanged.
"""
from __future__ import annotations

import json

import pytest

from local_app.rag import RagService
from local_app.runtime import IncompleteCompletionError, InvalidQueryResolutionError, OllamaRuntime
from tests.test_social_chat import LocalSession, Settings


class MaterialsDatabase:
    def hybrid_search(self, **kwargs):
        raise AssertionError("A creative continuation must not retrieve curriculum materials")


class ExternalProvider:
    def complete(self, *args, **kwargs):
        raise AssertionError("A creative continuation must stay local")


def _archive_retriever():
    calls = []

    def retriever(query):
        calls.append(query)
        return {"method": "conversation_retrieval",
                "system_text": "2026-11-03 – ученик: «Жил-был дракон по имени Искра...»",
                "sources": [{"id": "s-old-1", "kind": "self_report"}],
                "source_ids": ["s-old-1"], "evidence_ids": ["s-old-1"],
                "diagnostics": {"chunks": 1}}
    retriever.calls = calls
    return retriever


LONG_STORY = ("Продолжаем историю! Искра расправила крылья и полетела над вечерним городом. "
              * 16)


def _answer(decision, reply, **kwargs):
    session = LocalSession(json.dumps(decision), reply)
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsDatabase(), runtime, Settings())
    result = service.answer([{"role": "user", "content": kwargs.pop("message")}], **kwargs)
    return result, session


def test_creative_continuation_uses_archive_retrieval_and_longer_budget():
    retriever = _archive_retriever()
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": "",
                "social_turn": True, "creative_mode": True, "needs_archive": True}
    result, session = _answer(decision, LONG_STORY,
                              message="Продолжи историю, которую мы начали на прошлой неделе!",
                              conversation_retriever=retriever)
    # The archive was searched server-side without a "remember" trigger.
    assert retriever.calls == ["Продолжи историю, которую мы начали на прошлой неделе!"]
    request = session.requests[-1]
    assert request["options"]["num_predict"] == 2048  # not the 192-token social route
    system = request["messages"][0]["content"]
    # Fiction/real separation and honest missing-evidence behavior are instructed.
    assert "вымысел" in system
    assert "не нашёл её продолжение" in system
    # The retrieved old story fragment reached the prompt as untrusted data.
    assert "Искра" in system and "own_conversation_retrieval_untrusted" in system
    assert result["choices"][0]["message"]["content"] == LONG_STORY.strip()
    assert len(LONG_STORY) > 900  # the 900-character social limit does not apply
    assert result["answer_routing"]["reason"] == "creative_continuation"
    assert result["conversation_retrieval"]["source_count"] == 1


def test_creative_with_archive_unavailable_reports_it_honestly():
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": "",
                "social_turn": True, "creative_mode": True, "needs_archive": True}
    result, _session = _answer(decision, "Начало истории не нашлось, давай начнём новую!",
                               message="Продолжи нашу историю!")
    assert result["answer_routing"]["reason"] == "creative_continuation"
    assert result["answer_routing"]["creative_archive"] == "unavailable"


def test_greeting_stays_on_the_brief_social_route():
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": "",
                "social_turn": True}
    session = LocalSession(json.dumps(decision), "Привет! Как твоё настроение сегодня?")
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsDatabase(), runtime, Settings())
    result = service.answer([{"role": "user", "content": "привет"}])
    assert session.requests[-1]["options"]["num_predict"] <= 256
    assert result["answer_routing"]["reason"] == "brief_social_turn"


def test_legacy_decision_without_creative_fields_routes_social():
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": "",
                "social_turn": True}
    result, _session = _answer(decision, "Здорово! Расскажи ещё.",
                               message="Я сегодня рисовал кота!")
    assert result["answer_routing"]["reason"] == "brief_social_turn"


@pytest.mark.parametrize("decision", [
    # A creative decision must not smuggle retrieval or context anchors.
    {"in_scope": False, "uses_prior_context": False, "search_query": "дракон Искра",
     "social_turn": True, "creative_mode": True},
    {"in_scope": True, "uses_prior_context": False, "search_query": "",
     "creative_mode": True},
    {"in_scope": False, "uses_prior_context": True, "search_query": "",
     "creative_mode": True, "prior_learner_question_index": 0},
    {"in_scope": False, "uses_prior_context": False, "search_query": "",
     "creative_mode": True, "uses_approved_memory": True},
    # Archive lookup without creative mode is invalid.
    {"in_scope": False, "uses_prior_context": False, "search_query": "",
     "social_turn": True, "needs_archive": True},
])
def test_resolver_rejects_invalid_creative_decisions(decision):
    session = LocalSession(json.dumps(decision))
    with pytest.raises(InvalidQueryResolutionError):
        OllamaRuntime(Settings(), session=session).resolve_chat_query(
            "Продолжи историю, которую мы начали на прошлой неделе!", [])


def test_resolver_prompt_mentions_creative_and_archive_decisions():
    # The creative decision is a product feature for the accepted 16k/32k
    # contexts; an 8k window keeps its baseline prompt size exactly.
    class WideSettings(Settings):
        context_tokens = 16384
    session = LocalSession(json.dumps({"in_scope": False, "uses_prior_context": False,
                                       "search_query": "", "social_turn": True,
                                       "creative_mode": True, "needs_archive": True}))
    runtime = OllamaRuntime(WideSettings(), session=session)
    runtime.resolve_chat_query("Продолжи историю!", [])
    system = session.requests[0]["messages"][0]["content"]
    assert "creative_mode" in system and "needs_archive" in system
    assert "fictional story" in system
    schema = session.requests[0]["format"]
    properties = schema.get("schema", schema).get("properties", {})
    assert properties.get("creative_mode", {}).get("type") == "boolean"
    assert properties.get("needs_archive", {}).get("type") == "boolean"
