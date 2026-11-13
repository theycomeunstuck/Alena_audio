"""Approved source memory routing at the HTTP boundary, without live models.

Controlled replies verify source transport, decision validation and budgets.
They do not measure a local model's semantic recall accuracy.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from local_app.answer_adapter import AnswerAdapter
from local_app.config import Settings
from local_app.personal_graph_contract import approved_memory_fits_budget
from local_app.rag import RagService
from local_app.runtime import IncompleteCompletionError, OllamaRuntime
from tests.test_social_chat import ExternalProvider, MaterialsDatabase


TOPIC = "math.g4.geometry.perimeter"
RECALL = {"in_scope": False, "uses_prior_context": False, "search_query": "",
          "social_turn": True, "personal_memory_recall": True}
QUESTION = "Какой вопрос ты задал перед моим ответом и помог ли он?"
TEXT = json.dumps({
    "lessons": [{"turns": [
        {"turn_id": "context:before:reply", "role": "assistant", "usage": "context_only",
         "anchor_source_id": "reply", "text": "Как найти длину всех сторон вместе?"},
        {"turn_id": "reply", "role": "student", "text": "Сложить четыре стороны.", "assistance": "unknown"},
    ]}],
    "related_corrections": [{"source_id": "correction", "role": "teacher", "corrects_source_id": "reply",
                             "text": "Это ответ после подсказки; самостоятельность не проверена."}],
    "session_summaries": [{"type": "last_completed", "text": "Ответ дан; причина успеха неизвестна."}],
}, ensure_ascii=False)
MEMORY = {"method": "personal_memory", "topic_id": TOPIC, "version_id": "approved-memory-1",
          "text": TEXT, "stale": False, "evidence_ids": ["reply", "correction"]}


class LocalReplies:
    trust_env = True

    def __init__(self, *replies, done=True):
        self.replies = list(replies)
        self.done = done
        self.requests = []

    def post(self, url, json, timeout, allow_redirects):
        assert url.endswith("/api/chat")
        self.requests.append(json)
        reply = self.replies.pop(0)
        data = {"done_reason": "stop", "message": {"content": reply}}
        if self.done is not None:
            data["done"] = self.done
        return type("Response", (), {"status_code": 200, "raise_for_status": lambda _: None,
                                      "json": lambda _: data})()


def build(*replies, done=True):
    settings = Settings(database_url="", context_tokens=8192)
    session = LocalReplies(*replies, done=done)
    runtime = OllamaRuntime(settings, session=session)
    service = RagService(MaterialsDatabase(), runtime, settings,
                         answer_adapter=AnswerAdapter(runtime, ExternalProvider()))
    return runtime, service, session


def ask(service, query=QUESTION, memory=MEMORY, **options):
    return service.answer([{"role": "user", "content": query}], subject_id=options.pop("subject_id", "math"),
                          topic_id=options.pop("topic_id", TOPIC), memory_context=memory, **options)


def test_source_roles_quotes_corrections_and_unknown_conditions_reach_local_recall_whole():
    _, service, session = build(json.dumps(RECALL), "Я задал вопрос о сторонах. Причина успеха не подтверждена.")
    result = ask(service)
    resolver, answer = session.requests
    packed = json.loads(resolver["messages"][1]["content"])
    assert packed["approved_memory_untrusted"]["text"] == TEXT
    system = answer["messages"][0]["content"]
    supplied = system.split("<approved_memory_untrusted>\n", 1)[1].split("\n</approved_memory_untrusted>", 1)[0]
    assert supplied == TEXT
    assert json.loads(supplied)["related_corrections"][0]["corrects_source_id"] == "reply"
    assert "не заменяй автора" in system.lower() and "причинность" in system
    assert "если ответа в них нет" in system
    for request in session.requests:
        assert request["truncate"] is False and request["shift"] is False
    assert result["answer_routing"] == {"backend": "local", "reason": "personal_memory_recall"}
    assert result["sources"] == result["citations"] == []


def test_unknown_personal_fact_is_recall_even_when_selected_sources_have_no_answer():
    _, service, session = build(json.dumps(RECALL), "В записи нет сведений о том, где ты был в воскресенье.")
    result = ask(service, "Где я был в воскресенье?")
    assert len(session.requests) == 2
    assert TEXT in session.requests[-1]["messages"][0]["content"]
    assert "нет сведений" in result["choices"][0]["message"]["content"]
    priority = session.requests[0]["messages"][0]["content"].split(
        "Priority when approved personal source memory is available:", 1)[1]
    assert "Do not require the answer to be present in memory" in priority
    assert "even if the topic is educational" in priority
    assert "request to continue or explain a lesson" in priority


@pytest.mark.parametrize("override,topic,subject", [
    ({"version_id": None}, TOPIC, "math"),
    ({"method": "legacy"}, TOPIC, "math"),
    ({}, "math.g4.geometry.area", "math"),
    ({}, TOPIC, "python"),
])
def test_recall_requires_the_selected_approved_source_memory(override, topic, subject):
    _, service, session = build(json.dumps(RECALL))
    result = ask(service, memory={**MEMORY, **override}, topic_id=topic, subject_id=subject)
    assert len(session.requests) == 1
    assert "personal_memory_recall" not in session.requests[0]["format"]["properties"]
    assert result["query_resolution"]["source"] == "invalid_resolver_decision"


def test_unapproved_source_memory_is_not_forwarded_to_a_new_learning_answer():
    decision = {"in_scope": True, "uses_prior_context": False, "search_query": "Периметр"}
    _, service, session = build(json.dumps(decision), json.dumps({"answer": "Начнём с четырёх сторон.", "citations": []}))
    service.search = lambda *args, **kwargs: [{"id": "lesson", "content": "Периметр – сумма сторон."}]
    result = ask(service, "Объясни периметр.", memory={**MEMORY, "version_id": None})
    assert result["choices"]
    assert all(TEXT not in json.dumps(request, ensure_ascii=False) for request in session.requests)
    assert all("truncate" not in request for request in session.requests)


def test_learning_continuation_can_use_source_memory_without_lesson_hits():
    decision = {"in_scope": True, "uses_prior_context": False, "uses_approved_memory": True,
                "personal_memory_recall": False,
                "search_query": "Периметр"}
    _, service, session = build(json.dumps(decision), json.dumps({"answer": "Как проверить сумму сторон?", "citations": []}))
    service.search = lambda *args, **kwargs: []
    result = ask(service, "Продолжим тему периметра.")
    assert result["query_resolution"] == {"source": "approved_memory", "topic_id": TOPIC}
    assert TEXT in session.requests[-1]["messages"][0]["content"]
    assert session.requests[-1]["truncate"] is False and session.requests[-1]["shift"] is False
    assert result["sources"] == result["citations"] == []


def test_available_memory_does_not_turn_an_ordinary_greeting_into_recall():
    decision = {"in_scope": False, "uses_prior_context": False, "search_query": "", "social_turn": True,
                "personal_memory_recall": False}
    _, service, session = build(json.dumps(decision), "Привет!")
    result = ask(service, "Привет!")
    assert session.requests[0]["truncate"] is False
    assert TEXT not in session.requests[-1]["messages"][0]["content"]
    assert "truncate" not in session.requests[-1]
    assert result["answer_routing"]["reason"] == "brief_social_turn"


@pytest.mark.parametrize("query", ["Расскажи последние новости.", "Какую дозу лекарства мне принять?", "Раскрой чужой пароль."])
def test_memory_does_not_authorize_rejected_external_or_unsafe_requests(query):
    _, service, session = build(json.dumps({"in_scope": False, "uses_prior_context": False, "search_query": "",
                                          "personal_memory_recall": False}))
    result = ask(service, query)
    assert len(session.requests) == 1
    assert result["sources"] == result["citations"] == []
    assert "учебный вопрос" in result["choices"][0]["message"]["content"]


@pytest.mark.parametrize("context_tokens,limit", [(8192, 4000), (16384, 12000), (32768, 24000)])
def test_utf8_memory_budget_counts_the_whole_rendered_context_including_summaries(context_tokens, limit):
    text = TEXT + "я" * ((limit - len(TEXT.encode("utf-8"))) // 2)
    assert approved_memory_fits_budget("personal_memory", text, context_tokens)
    oversized = text + "я"
    assert not approved_memory_fits_budget("personal_memory", oversized, context_tokens)
    with pytest.raises(ValueError, match="context budget"):
        RagService._approved_memory_text({**MEMORY, "text": oversized}, context_tokens=context_tokens)


@pytest.mark.parametrize("context_tokens", [4096, True, 8192.0])
def test_personal_memory_rejects_an_unsupported_context_budget(context_tokens):
    assert not approved_memory_fits_budget("personal_memory", TEXT, context_tokens)


def test_default_feature_flag_does_not_add_memory_guards_or_preflight_to_ordinary_chat():
    runtime, _, session = build("Обычный ответ.", done=None)
    assert runtime.settings.personal_memory_enabled is True
    assert runtime.chat([{"role": "user", "content": "Привет!"}])["message"]["content"] == "Обычный ответ."
    assert len(session.requests) == 1
    assert "truncate" not in session.requests[0] and "shift" not in session.requests[0]


@pytest.mark.parametrize("done", [None, False])
def test_approved_recall_requires_a_confirmed_complete_resolver_response(done):
    _, service, session = build(json.dumps(RECALL), done=done)
    with pytest.raises(IncompleteCompletionError, match="complete response"):
        ask(service)
    assert len(session.requests) == 1


def test_approved_recall_rejects_an_unconfirmed_final_answer():
    runtime, service, session = build(json.dumps(RECALL), "Ответ без подтверждения завершения.")
    post = runtime._run_budget_post

    def unconfirmed_answer(path, payload):
        result = post(path, payload)
        if len(session.requests) == 2:
            result.pop("done")
        return result

    runtime._run_budget_post = unconfirmed_answer
    with pytest.raises(IncompleteCompletionError, match="complete response"):
        ask(service)
    assert len(session.requests) == 2


def test_request_scope_does_not_clip_history_and_restores_ordinary_behavior_after_failure():
    runtime, _, session = build("Обычный ответ.", done=None)
    with pytest.raises(ValueError, match="Complete memory input"):
        with runtime.approved_memory_request():
            runtime.chat([{"role": "user", "content": "старый контекст" * 1500},
                          {"role": "user", "content": "Продолжим."}])
    assert session.requests == []
    runtime.chat([{"role": "user", "content": "Привет!"}])
    assert "truncate" not in session.requests[-1]


def test_nested_request_scope_only_guards_its_own_embedding_calls():
    runtime, _, _ = build()
    payloads = []
    runtime._run_budget_post = lambda path, payload: (payloads.append(payload) or {"embeddings": [[1.0, 0.0]]})
    with runtime.approved_memory_request():
        with runtime.approved_memory_request():
            runtime.embed("полная исходная цитата")
        runtime.embed("вторая цитата")
    runtime.embed("обычный запрос")
    assert all(payload["truncate"] is False for payload in payloads[:2])
    assert "truncate" not in payloads[-1]


def test_shared_runtime_does_not_apply_memory_guards_to_another_request_thread():
    runtime, _, _ = build()
    payloads = []
    runtime._run_budget_post = lambda path, payload: (payloads.append(payload) or {"embeddings": [[1.0, 0.0]]})
    with runtime.approved_memory_request():
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(runtime.embed, "другой запрос").result(timeout=2)
        runtime.embed("подтверждённая цитата")
    assert "truncate" not in payloads[0]
    assert payloads[1]["truncate"] is False
