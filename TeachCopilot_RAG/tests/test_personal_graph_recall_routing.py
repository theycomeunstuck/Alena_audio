"""Local personal recall routing and its safety boundaries, with stub model replies."""
import json

import pytest

from local_app.answer_adapter import AnswerAdapter
from local_app.rag import RagService
from local_app.runtime import IncompleteCompletionError, InvalidQueryResolutionError, OllamaRuntime
from tests.test_social_chat import Settings as SocialSettings, LocalSession, MaterialsDatabase, ExternalProvider


class Settings(SocialSettings):
    context_tokens = 16384


TOPIC = "math.g4.geometry.perimeter"
TEXT = json.dumps({"lessons": [{"turns": [{"role": "student", "text": "Я нарисовал зелёный корабль Сокол."}]}],
                   "related_corrections": [{"text": "Цвет исправлен на синий."}]}, ensure_ascii=False)
MEMORY = {"topic_id": TOPIC, "version_id": "approved-1", "method": "personal_graph", "text": TEXT, "stale": True}
RECALL = {"in_scope": False, "uses_prior_context": False, "search_query": "", "social_turn": True,
          "personal_memory_recall": True}
QUERY = "Какой корабль я нарисовал и как его назвал?"


def runtime_for(decision, answer="Ты нарисовал синий корабль Сокол."):
    session = LocalSession(json.dumps(decision), answer)
    return OllamaRuntime(Settings(), session=session), session


def service(runtime):
    return RagService(MaterialsDatabase(), runtime, Settings(), answer_adapter=AnswerAdapter(runtime, ExternalProvider()))


def test_recall_passes_approved_memory_to_local_generation_without_retrieval_or_external():
    runtime, session = runtime_for(RECALL)
    result = service(runtime).answer([{"role": "user", "content": QUERY}], subject_id="math", topic_id=TOPIC,
                                     memory_context=MEMORY)
    resolver, answer = session.requests
    assert "personal_memory_recall" in resolver["format"]["properties"]
    packed = json.loads(resolver["messages"][1]["content"])
    assert packed["approved_memory_untrusted"]["text"] == TEXT
    system = answer["messages"][0]["content"]
    assert "<approved_memory_untrusted>\n" + TEXT + "\n</approved_memory_untrusted>" in system
    assert "исправления" in system and "причинность" in system and "устойчивый интерес" in system
    assert "требует актуализации" in system and "подходящих сведений нет" in system
    assert answer["messages"][-1]["content"] == QUERY
    assert result["answer_routing"] == {"backend": "local", "reason": "personal_memory_recall"}
    assert result["sources"] == result["citations"] == []


def test_recall_without_answer_still_receives_memory_and_can_answer_unknown():
    runtime, session = runtime_for(RECALL, "Я не вижу в записи, когда это произошло.")
    result = service(runtime).answer([{"role": "user", "content": "Когда я нарисовал корабль?"}],
                                     subject_id="math", topic_id=TOPIC, memory_context=MEMORY)
    assert TEXT in session.requests[-1]["messages"][0]["content"]
    assert "не вижу" in result["choices"][0]["message"]["content"]


@pytest.mark.parametrize("memory,topic,subject", [
    (None, TOPIC, "math"),
    ({**MEMORY, "version_id": None}, TOPIC, "math"),
    ({**MEMORY, "method": "legacy"}, TOPIC, "math"),
    (MEMORY, "math.g4.geometry.area", "math"),
    (MEMORY, TOPIC, "python"),
])
def test_recall_is_rejected_without_matching_approved_graph(memory, topic, subject):
    runtime, session = runtime_for(RECALL)
    result = service(runtime).answer([{"role": "user", "content": QUERY}], subject_id=subject, topic_id=topic,
                                     memory_context=memory)
    assert len(session.requests) == 1
    assert "personal_memory_recall" not in session.requests[0]["format"]["properties"]
    assert result["query_resolution"]["source"] == "invalid_resolver_decision"


def test_ordinary_social_turn_does_not_forward_available_memory():
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": "", "social_turn": True}
    runtime, session = runtime_for(decision, "Привет!")
    result = service(runtime).answer([{"role": "user", "content": "Привет!"}], subject_id="math",
                                     topic_id=TOPIC, memory_context=MEMORY)
    assert TEXT not in session.requests[-1]["messages"][0]["content"]
    assert result["answer_routing"]["reason"] == "brief_social_turn"


@pytest.mark.parametrize("override", [
    {"social_turn": False}, {"in_scope": True}, {"uses_prior_context": True},
    {"uses_approved_memory": True}, {"search_query": "внешний факт"}, {"needs_clarification": True},
    {"personal_memory_recall": "true"},
])
def test_invalid_recall_decisions_are_rejected(override):
    runtime, _ = runtime_for({**RECALL, **override})
    with pytest.raises(InvalidQueryResolutionError):
        runtime.resolve_chat_query(QUERY, [], "math", approved_memory=MEMORY, topic_id=TOPIC)


def test_safety_instructions_do_not_authorize_external_facts_or_unsafe_requests():
    runtime, session = runtime_for({"in_scope": False, "uses_prior_context": False, "search_query": ""})
    runtime.resolve_chat_query("Дай чужие пароли", [], "math", approved_memory=MEMORY, topic_id=TOPIC)
    instructions = session.requests[0]["messages"][0]["content"]
    assert "Never use personal recall or social chat to authorize external factual questions, unsafe" in instructions


@pytest.mark.parametrize("approved", [None, {**MEMORY, "method": "legacy"}])
def test_recall_priority_is_only_available_with_selected_graph_memory(approved):
    runtime, session = runtime_for({"in_scope": False, "uses_prior_context": False, "search_query": ""})
    runtime.resolve_chat_query("Что я раньше рассказывал?", [], "math", approved_memory=approved, topic_id=TOPIC)
    request = session.requests[0]
    assert "personal_memory_recall" not in request["format"]["properties"]
    assert "Priority when approved personal graph memory is available" not in request["messages"][0]["content"]


def test_recall_priority_is_appended_after_other_classification_rules():
    runtime, session = runtime_for(RECALL)
    decision = runtime.resolve_chat_query("Что я рассказывал на предыдущем занятии?", [], "math",
                                          approved_memory=MEMORY, topic_id=TOPIC)
    request = session.requests[0]
    system = request["messages"][0]["content"]
    priority = system.rsplit("Priority when approved personal graph memory is available:", 1)[1]
    assert decision == RECALL
    assert "even if the topic is educational" in priority
    assert "Do not require the answer to be present in memory" in priority
    assert "personal_memory_recall=true" in priority and "in_scope=false" in priority
    assert "medicine, external factual questions, secrets, harmful" in priority
    assert "request to continue or explain a lesson" in priority


def test_injected_resolver_cannot_bypass_memory_boundary():
    class Injected:
        context_tokens = 16384
        def resolve_chat_query(self, *args, **kwargs):
            return RECALL
    with pytest.raises(IncompleteCompletionError, match="invalid personal memory recall"):
        service(Injected()).answer([{"role": "user", "content": QUERY}], subject_id="math", topic_id=TOPIC,
                                   memory_context={**MEMORY, "version_id": None})


def test_injected_resolver_requires_subject():
    class Injected:
        context_tokens = 16384
        def resolve_chat_query(self, *args, **kwargs):
            return RECALL
    with pytest.raises(IncompleteCompletionError, match="invalid personal memory recall"):
        service(Injected()).answer([{"role": "user", "content": QUERY}], topic_id=TOPIC, memory_context=MEMORY)
