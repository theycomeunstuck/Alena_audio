"""Actual-delivery trace regressions: retrieved-but-unused never counts as delivered.

The recording fake session captures the real outbound /api/chat payload, so the
``answer_prompt`` event's ``final_prompt`` and its hash are checked against the
exact final answer-model call, not a resolver flag or an opaque revision.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.memory_trace import MemoryTrace
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_alena_api import ConversationStore, browser as alena_browser, payload as alena_payload
from tests.test_social_chat import LocalSession, Settings


def recall_decision():
    return json.dumps({"in_scope": False, "uses_prior_context": False,
                       "search_query": "", "social_turn": True, "personal_memory_recall": True})


def curriculum_decision(query="дроби и доли"):
    return json.dumps({"in_scope": True, "uses_prior_context": False, "search_query": query})


class TracedSession(LocalSession):
    """Records /api/tags and /api/embed like CurriculumSession, plus chat replies."""

    def post(self, url, json, timeout, allow_redirects):
        if url.endswith("/api/embed"):
            return self.response({"embeddings": [[0.1, 0.2, 0.3]]})
        return super().post(url, json, timeout, allow_redirects)

    def get(self, url, timeout, allow_redirects):
        assert url.endswith("/api/tags")
        return self.response({"models": [{"name": Settings.embedding_model, "digest": "local-digest"}]})


class MaterialsStore:
    """Five authorized curriculum rows; the caller ranks and limits them."""

    def __init__(self, rows):
        self.rows = rows

    def hybrid_search(self, **kwargs):
        return [dict(row) for row in self.rows]


def material_rows(count=5):
    return [{
        "id": f"row-{index}", "source_file": f"lesson-{index}.md", "source_version": "v1",
        "score": 1.0 - index / 10, "similarity": 0.9, "text_score": 0.0,
        "topic_id": "math.fractions", "grade": "4", "subject_id": "math",
        "title": f"Дроби {index}", "content": f"Дробь {index} показывает долю целого.",
    } for index in range(count)]


def collect():
    return []


def trace_for(events):
    return MemoryTrace(events.append, learner_id="learner-a")


def phases(events):
    return [event["phase"] for event in events]


def by_phase(events, phase):
    return [event for event in events if event["phase"] == phase]


def prompt_digest(final_messages):
    return hashlib.sha256(json.dumps(final_messages, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


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


def test_in_scope_trace_hashes_the_actual_final_call_and_excludes_unused_sources():
    events = collect()
    session = TracedSession(curriculum_decision(),
                            json.dumps({"answer": "Дробь – это доля целого.",
                                        "citations": ["row-0"]}))
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore(material_rows()), runtime, Settings())
    result = service.answer([{"role": "user", "content": "Объясни, что такое дроби"}],
                            subject_id="math", topic_id="math.fractions", limit=2, trace=trace_for(events))
    assert result["choices"][0]["message"]["content"].startswith("Дробь")
    assert phases(events) == ["resolver", "retrieval", "answer_prompt"]

    resolver = by_phase(events, "resolver")[0]
    assert resolver["route"] == "curriculum_retrieval"
    assert resolver["decision"]["in_scope"] is True

    retrieval = by_phase(events, "retrieval")[0]
    assert retrieval["eligible_source_ids"] == [f"row-{index}" for index in range(5)]
    assert retrieval["retrieved_source_ids"] == [f"row-{index}" for index in range(5)]
    assert retrieval["selected_source_ids"] == ["row-0", "row-1"]
    assert retrieval["omitted_source_ids"] == ["row-2", "row-3", "row-4"]

    prompt = by_phase(events, "answer_prompt")[0]
    assert prompt["delivered_source_ids"] == ["row-0", "row-1"]
    # The three retrieved-but-unused rows must never count as delivered evidence.
    assert not set(retrieval["omitted_source_ids"]) & set(prompt["delivered_source_ids"])
    assert prompt["generation_called"] is True
    # The recorded final input must be the actual outbound answer-model call.
    actual_outbound = session.requests[-1]["messages"]
    assert prompt["final_prompt"] == actual_outbound
    assert any("row-0" in message["content"] for message in actual_outbound)
    assert not any("row-2" in message["content"] for message in actual_outbound)
    assert prompt["final_prompt_sha256"] == prompt_digest(actual_outbound)
    assert {quote["source_id"] for quote in prompt["hydrated_quotes"]} == {"row-0", "row-1"}


def test_invalid_resolver_decision_records_fallback_without_generation():
    events = collect()
    session = TracedSession('{"in_scope": true, "uses_prior_context": true, "search_query": "", '
                            '"prior_learner_question_index": 0, "needs_clarification": true}')
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore([]), runtime, Settings())
    result = service.answer([{"role": "user", "content": "хз"}], subject_id="math", trace=trace_for(events))
    assert result["query_resolution"]["source"] == "invalid_resolver_decision"
    assert phases(events) == ["resolver", "fallback"]
    resolver = by_phase(events, "resolver")[0]
    assert resolver["route"] == "clarification"
    assert resolver["invalid_decision"]
    fallback = by_phase(events, "fallback")[0]
    assert fallback["generation_called"] is False
    assert "answer_prompt" not in phases(events)


def test_personal_recall_trace_carries_hydrated_quotes_and_hashes_the_social_call():
    events = collect()
    session = TracedSession(recall_decision(), "Ты рассказывал, что рисуешь корабль!")
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore([]), runtime, Settings())
    conversation_context = {"method": "conversation_history", "system_text": "СОБСТВЕННАЯ СВОДКА",
                            "recent_messages": [{"role": "user", "content": "Я нарисовал корабль.",
                                                 "usage": "self_report"}]}
    retrieval = {"method": "conversation_retrieval", "system_text": "[Q1] …корабль…",
                 "sources": [{"source_id": "src-1", "role": "user", "usage": "self_report",
                              "text": "Я нарисовал корабль.", "activity_day": "2026-11-10",
                              "author_role": "student"}],
                 "source_ids": ["src-1"], "evidence_ids": ["src-1"],
                 "hits": [{"source_ids": ["src-1"]}],
                 "diagnostics": {"budget_bytes": 4000, "archive_message_count": 6}}
    result = service.answer([{"role": "user", "content": "Какой корабль я нарисовала?"}],
                            conversation_context=conversation_context,
                            conversation_retriever=lambda query: retrieval,
                            trace=trace_for(events))
    assert result["answer_routing"]["reason"] == "personal_memory_recall"
    assert phases(events) == ["resolver", "retrieval", "answer_prompt"]
    resolver = by_phase(events, "resolver")[0]
    assert resolver["route"] == "personal_recall"
    retrieval_event = by_phase(events, "retrieval")[0]
    assert retrieval_event["route"] == "personal_recall"
    assert retrieval_event["selected_source_ids"] == ["src-1"]
    assert retrieval_event["budget"]["budget_bytes"] == 4000
    prompt = by_phase(events, "answer_prompt")[0]
    assert prompt["delivered_source_ids"] == ["src-1"]
    quote = prompt["hydrated_quotes"][0]
    assert quote["source_id"] == "src-1"
    assert quote["author_role"] == "student"
    assert quote["date"] == "2026-11-10"
    assert quote["kind"] == "own_conversation"
    assert prompt["final_prompt"] == session.requests[-1]["messages"]
    assert prompt["final_prompt_sha256"] == prompt_digest(session.requests[-1]["messages"])
    assert prompt["generation_called"] is True


def test_sink_failure_is_counted_and_never_breaks_the_answer():
    class FlakySink:
        def __init__(self):
            self.count = 0
            self.accepted = []

        def __call__(self, event):
            self.count += 1
            if self.count <= 2:
                raise RuntimeError("sink unavailable")
            self.accepted.append(event)

    sink = FlakySink()
    session = TracedSession(curriculum_decision(),
                            json.dumps({"answer": "Ответ про дроби.", "citations": []}))
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore(material_rows(2)), runtime, Settings())
    result = service.answer([{"role": "user", "content": "Что такое дроби?"}],
                            subject_id="math", trace=MemoryTrace(sink, learner_id="learner-a"))
    assert result["choices"][0]["message"]["content"] == "Ответ про дроби."
    assert sink.count >= 3  # the failed emissions never reach the sink
    assert len(sink.accepted) == sink.count - 2


def test_trace_events_keep_database_provenance_json_safe():
    from datetime import datetime
    from decimal import Decimal
    from local_app.memory_trace import _json_safe
    safe = _json_safe({"score": Decimal("0.75"), "at": datetime(2026, 11, 17, 20, 30),
                       "rows": [Decimal("2"), {"x": None}]})
    assert json.dumps(safe, allow_nan=False)
    assert safe["score"] == "0.75"
    assert safe["at"] == "2026-11-17T20:30:00"


def test_api_sink_emits_one_request_scoped_complete_with_correlation_id(memory_calls):
    store = ConversationStore()
    events = collect()
    client, rag = alena_browser(store)
    # Rebuild the app with the trusted constructor-only sink installed.
    app = create_app(settings=client.app.state.settings, store=store, runtime=client.app.state.runtime,
                     rag_service=rag, chat_response_journal=client.app.state.chat_response_journal,
                     memory_trace_sink=events.append)
    client = TestClient(app, headers={"Authorization": "Bearer test"})
    request = alena_payload()
    request["metadata"]["quality_row_id"] = "quality-row-9"
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200, response.text
    assert phases(events) == ["complete"]
    complete = events[0]
    assert complete["correlation_id"] == "quality-row-9"
    assert complete["learner_id"] == "learner-a"
    assert complete["request_id"]
    assert complete["output_status"] == "fallback"
    assert complete["generation_called"] is False
    # No trace data may leak into the public response.
    assert "memory_trace" not in json.dumps(response.json()).lower()


def test_http_request_cannot_enable_a_sink_and_garbage_correlation_is_dropped():
    events = collect()
    store = ConversationStore()
    client, rag = alena_browser(store)
    app = create_app(settings=client.app.state.settings, store=store, runtime=client.app.state.runtime,
                     rag_service=rag, chat_response_journal=client.app.state.chat_response_journal,
                     memory_trace_sink=events.append)
    client = TestClient(app, headers={"Authorization": "Bearer test"})
    request = alena_payload()
    request["metadata"]["quality_row_id"] = "x" * 500  # bounded validation drops it
    assert client.post("/v1/chat/completions", json=request).status_code == 200
    assert events and events[0]["correlation_id"] is None
    with pytest.raises(ValueError):
        create_app(settings=client.app.state.settings, store=store, runtime=client.app.state.runtime,
                   rag_service=rag, memory_trace_sink="not-a-callable")


def test_empty_candidate_pool_traces_retrieval_and_fallback_without_crash():
    # Live probe regression: an in-scope search with an empty authorized pool
    # raised UnboundLocalError in the trace path (minimum_similarity was read
    # inside the per-row loop only).
    events = collect()
    session = TracedSession(curriculum_decision("дроби и доли"))
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore([]), runtime, Settings())
    result = service.answer([{"role": "user", "content": "Объясни дроби"}],
                            subject_id="math", topic_id="math.fractions",
                            trace=trace_for(events))
    assert "не найдено" in result["choices"][0]["message"]["content"]
    assert phases(events) == ["resolver", "retrieval", "fallback"]
    retrieval = by_phase(events, "retrieval")[0]
    assert retrieval["selected_source_ids"] == []
    assert retrieval["budget"]["database_candidate_count"] == 0
    fallback = by_phase(events, "fallback")[0]
    assert fallback["generation_called"] is False


def test_creative_route_traces_delivered_archive_quotes():
    # Probe-all18 regression: the creative branch dropped the trace, so a real
    # generated story continuation was recorded as a fallback with no evidence.
    events = collect()
    session = TracedSession(json.dumps({"in_scope": False, "uses_prior_context": False,
                                        "search_query": "", "social_turn": True,
                                        "creative_mode": True, "needs_archive": True,
                                        "personal_memory_recall": False}),
                            "Наша история началась с лисёнка Рыжика!")
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore([]), runtime, Settings())
    conversation_context = {"method": "conversation_history", "system_text": "сводка",
                            "recent_messages": [{"role": "user", "content": "Начнём сказку.",
                                                 "usage": "self_report"}]}
    retrieval = {"method": "conversation_retrieval", "system_text": "[Q1] Начнём сказку.",
                 "sources": [{"source_id": "story-1", "role": "user", "usage": "self_report",
                              "text": "Начнём сказку.", "activity_day": "2026-11-01",
                              "author_role": "student"}],
                 "source_ids": ["story-1"], "evidence_ids": ["story-1"], "hits": [],
                 "diagnostics": {"budget_bytes": 4000}}
    result = service.answer([{"role": "user", "content": "Продолжи нашу историю!"}],
                            conversation_context=conversation_context,
                            conversation_retriever=lambda query: retrieval,
                            trace=trace_for(events))
    assert result["answer_routing"]["reason"] == "creative_continuation"
    assert phases(events) == ["resolver", "retrieval", "answer_prompt"]
    retrieval_event = by_phase(events, "retrieval")[0]
    assert retrieval_event["route"] == "creative"
    assert retrieval_event["selected_source_ids"] == ["story-1"]
    prompt = by_phase(events, "answer_prompt")[0]
    assert prompt["delivered_source_ids"] == ["story-1"]
    assert prompt["generation_called"] is True
    assert prompt["final_prompt_sha256"] == prompt_digest(session.requests[-1]["messages"])


def test_personal_recall_delivers_assistant_context_quotes_it_actually_rendered():
    # The recorded F5 bookkeeping defect: the selection admitted the assistant
    # context quote (the exact 96-candy question) and rendered it into the
    # final prompt, but delivered_source_ids recorded only the user-side
    # evidence id, so the trace claimed the expected source never reached the
    # model. Delivered evidence is everything the final prompt carries.
    events = collect()
    session = TracedSession(recall_decision(), "Ты спрашивал ровно так: сколько минут?")
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore([]), runtime, Settings())
    conversation_context = {"method": "conversation_history", "system_text": "сводка",
                            "recent_messages": []}
    retrieval = {"method": "conversation_retrieval", "system_text": "[Q1] … [Q2] …",
                 "sources": [{"source_id": "ask-1", "role": "user", "usage": "self_report",
                              "text": "Хорошо, давай продолжать.", "activity_day": "2026-11-17",
                              "author_role": "student"},
                             {"source_id": "exact-a", "role": "assistant", "usage": "context_only",
                              "text": "Вопрос ровно в такой формулировке: сколько минут?",
                              "activity_day": "2026-11-17", "author_role": "assistant"}],
                 "source_ids": ["ask-1"], "evidence_ids": ["ask-1"],
                 "hits": [{"source_ids": ["ask-1", "exact-a"]}],
                 "diagnostics": {"budget_bytes": 4000, "archive_message_count": 6}}
    result = service.answer([{"role": "user", "content": "Как дословно звучал вопрос?"}],
                            conversation_context=conversation_context,
                            conversation_retriever=lambda query: retrieval,
                            trace=trace_for(events))
    retrieval_event = by_phase(events, "retrieval")[0]
    assert retrieval_event["selected_source_ids"] == ["ask-1", "exact-a"], \
        "the assistant context quote is part of the selection actually rendered"
    prompt = by_phase(events, "answer_prompt")[0]
    assert prompt["delivered_source_ids"] == ["ask-1", "exact-a"], \
        "delivered evidence includes the assistant quote the final prompt carries"
    assert {quote["source_id"] for quote in prompt["hydrated_quotes"]} == {"ask-1", "exact-a"}
    assert prompt["generation_called"] is True
