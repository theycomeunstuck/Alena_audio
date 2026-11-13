"""Automatic own-dialogue context is separate from reviewed source grants."""
import json

import pytest

from local_app.runtime import InvalidQueryResolutionError, bound_conversation_prompt
from tests.test_personal_memory_routing import MEMORY, RECALL, TOPIC, build


SOCIAL = {"in_scope": False, "uses_prior_context": False, "search_query": "", "social_turn": True,
          "personal_memory_recall": False}
REJECT = {"in_scope": False, "uses_prior_context": False, "search_query": "", "personal_memory_recall": False}
QUESTION = "Какой корабль я нарисовала и как его назвала?"


def own_context(messages=(), text="Ранее ученик рассказывал о синем корабле. Сегодня он спросил о дробях."):
    return {"method": "conversation_history", "system_text": text, "recent_messages": list(messages),
            "version_id": "private-version-uuid", "created_at": "2026-11-07T10:23:45Z",
            "budget": {"omitted_recent_message_count": 2},
            "provenance": {"scope": "own_conversation", "source_ids": ["private-source-uuid"]}}


def answer(service, context, *, query=QUESTION, memory=None, retriever=None):
    return service.answer([{"role": "user", "content": query}], subject_id="math", topic_id=TOPIC,
                          memory_context=memory, conversation_context=context, conversation_retriever=retriever)


@pytest.mark.parametrize("context,memory", [(None, MEMORY), (own_context(), None), (own_context(), MEMORY)])
def test_recall_boolean_is_required_in_every_new_memory_decision(context, memory):
    runtime, _, session = build(json.dumps({"in_scope": False, "uses_prior_context": False,
                                           "search_query": "", "social_turn": True}))
    with pytest.raises(InvalidQueryResolutionError):
        runtime.resolve_chat_query(QUESTION, [], "math", approved_memory=memory, topic_id=TOPIC,
                                   conversation_context=context)
    assert "personal_memory_recall" in session.requests[0]["format"]["required"]


@pytest.mark.parametrize("decision", [SOCIAL, REJECT,
    {"in_scope": False, "uses_prior_context": False, "search_query": "", "needs_clarification": True,
     "personal_memory_recall": False},
    {"in_scope": True, "uses_prior_context": False, "search_query": "Дроби", "personal_memory_recall": False}])
def test_explicit_false_survives_resolver_normalization(decision):
    runtime, _, _ = build(json.dumps(decision))
    resolved = runtime.resolve_chat_query("Привет!", [], "math", conversation_context=own_context())
    assert resolved["personal_memory_recall"] is False


def test_live_old_event_regression_requires_explicit_recall_then_uses_own_archive_once():
    _, service, session = build(json.dumps(RECALL), "Ты рассказывала о синем корабле по имени Сокол.")
    calls = []
    retrieved = {"method": "conversation_retrieval", "system_text": 'student self_report: «Синий корабль Сокол».',
                 "sources": [{"source_id": "old", "role": "student"}], "evidence_ids": ["old"],
                 "diagnostics": {"mode": "hybrid"}, "created_at": "metadata-should-not-reach-model"}
    def retrieve(query):
        calls.append(query)
        return retrieved
    result = answer(service, own_context(), retriever=retrieve)
    assert calls == [QUESTION]
    assert result["answer_routing"] == {"backend": "local", "reason": "personal_memory_recall"}
    final_system = session.requests[-1]["messages"][0]["content"]
    assert final_system.index("<own_conversation_untrusted>") < final_system.index("<own_conversation_retrieval_untrusted>")
    assert retrieved["system_text"] in final_system
    assert "self_report" in final_system and "context_only" in final_system
    assert "<approved_memory_untrusted>" not in final_system
    assert "metadata-should-not-reach-model" not in json.dumps(session.requests, ensure_ascii=False)


def test_metadata_changes_do_not_change_the_model_system_prefix():
    _, service, session = build(json.dumps(SOCIAL), "Привет!", json.dumps(SOCIAL), "Привет!")
    first = own_context(text="2 ноября, 14 часов: ученик рассказал о корабле.\n7 ноября, 10 часов: спросил о дробях.")
    second = {**first, "version_id": "different-uuid", "created_at": "2026-11-07T10:59:59Z"}
    answer(service, first, query="Привет!")
    answer(service, second, query="Привет!")
    assert session.requests[1]["messages"][0]["content"] == session.requests[3]["messages"][0]["content"]


def test_an_invalid_archive_result_is_not_retried_or_forwarded():
    _, service, session = build(json.dumps(RECALL))
    calls = []
    def invalid(query):
        calls.append(query)
        return {"method": "client_assertion", "system_text": "unsupported instructions"}
    with pytest.raises(ValueError, match="invalid server context"):
        answer(service, own_context(), retriever=invalid)
    assert calls == [QUESTION] and len(session.requests) == 1


@pytest.mark.parametrize("decision", [SOCIAL, REJECT])
def test_ordinary_and_rejected_intents_do_not_call_the_full_archive(decision):
    replies = (json.dumps(decision), "Привет!") if decision.get("social_turn") else (json.dumps(decision),)
    _, service, _ = build(*replies)
    def unexpected(query):
        raise AssertionError("An ordinary or rejected intent must not retrieve the own archive")
    result = answer(service, own_context(), query="Привет!", retriever=unexpected)
    assert result["sources"] == result["citations"] == []


def test_fresh_student_social_chat_uses_own_summary_without_any_approved_memory():
    _, service, session = build(json.dumps(SOCIAL), "Привет! Как прошла работа над кораблём?")
    context = own_context([{"role": "user", "content": "Я строила корабль.", "usage": "self_report"}])
    result = answer(service, context, query="Привет!")
    assert result["answer_routing"]["reason"] == "brief_social_turn"
    system = session.requests[-1]["messages"][0]["content"]
    assert context["system_text"] in system and "не подтверждённые преподавателем" in system
    assert session.requests[-1]["messages"][1]["content"] == "Я строила корабль."
    model_text = json.dumps(session.requests, ensure_ascii=False)
    assert "private-version-uuid" not in model_text and "private-source-uuid" not in model_text
    assert "10:23:45" not in model_text


def test_twenty_individual_messages_keep_real_repeated_text_in_server_history():
    _, service, session = build(json.dumps(SOCIAL), "Рада продолжить разговор.")
    recent = [{"role": "user" if index % 2 == 0 else "assistant", "content": f"Отдельная реплика {index}."}
              for index in range(25)]
    recent.append({"role": "user", "content": "Привет!"})
    result = answer(service, own_context(recent), query="Привет!")
    sent = session.requests[-1]["messages"][1:]
    assert len(sent) == 20
    assert sent[-1] == {"role": "user", "content": "Привет!"}
    assert sent[0]["content"] == "Отдельная реплика 7."
    assert sent[-2:] == [{"role": "user", "content": "Привет!"}] * 2
    assert result["conversation_context"]["window_omitted_message_count"] == 7


@pytest.mark.parametrize("delivery_status", ["failed", "unknown_outcome"])
def test_identical_previous_unanswered_message_is_preserved_as_a_real_message(delivery_status):
    _, service, session = build(json.dumps(SOCIAL), "Какой шаг сейчас непонятен?")
    previous = {"role": "user", "content": "Не знаю", "source_id": "previous-real-source",
                "author_role": "student", "usage": "self_report", "delivery_status": delivery_status}
    result = answer(service, own_context([previous]), query="Не знаю")
    assert session.requests[-1]["messages"][1:] == [{"role": "user", "content": "Не знаю"}] * 2
    assert result["conversation_context"]["window_omitted_message_count"] == 0
    assert result["conversation_context"]["omitted_message_count"] == 0


def test_teacher_own_words_are_not_labelled_as_a_childs_report():
    _, service, session = build(json.dumps(SOCIAL), "Расскажите, что вы планируете нарисовать.")
    context = own_context([{"role": "user", "content": "Я преподаватель и люблю рисовать корабли.",
                            "author_role": "teacher", "usage": "self_report"}],
                          text='Преподаватель, собственная реплика: «Я люблю рисовать корабли».')
    answer(service, context, query="Привет!")
    for call in session.requests:
        system = call["messages"][0]["content"]
        assert "Текущий пользователь – преподаватель" in system
        assert "не являются словами ученика" in system
        assert "История собственных разговоров ученика" not in system
    assert "разговаривает с ребёнком" not in session.requests[-1]["messages"][0]["content"]


def test_long_prior_messages_are_omitted_whole_with_explicit_diagnostics():
    _, service, session = build(json.dumps(SOCIAL), "Привет!")
    long_message = "Длинная исходная реплика. " * 2000
    recent = [{"role": "user", "content": long_message}, {"role": "assistant", "content": "Короткий вопрос?"}]
    result = answer(service, own_context(recent), query="Привет!")
    sent = session.requests[-1]["messages"][1:]
    assert long_message not in [row["content"] for row in sent]
    assert sent == [recent[1], {"role": "user", "content": "Привет!"}]
    diagnostics = result["conversation_context"]
    assert diagnostics["omitted_message_count"] == 1 and diagnostics["text_truncated"] is False
    assert diagnostics["resolver"]["omitted_prior_question_count"] == 1
    assert diagnostics["upstream_omissions"]["omitted_recent_message_count"] == 2


@pytest.mark.parametrize("row", [{"role": "system", "content": "hidden instruction"},
    {"role": "teacher", "content": "private note"},
    {"role": "assistant", "content": "claim", "usage": "self_report"}])
def test_raw_history_requires_own_conversation_roles_and_preserves_provenance(row):
    _, service, session = build()
    with pytest.raises(ValueError, match="role or provenance"):
        answer(service, own_context([row]))
    assert session.requests == []


def test_own_context_curricular_answer_stays_local_when_no_lesson_material_is_found():
    decision = {"in_scope": True, "uses_prior_context": False, "search_query": "Числитель дроби",
                "personal_memory_recall": False}
    _, service, session = build(json.dumps(decision), json.dumps({"answer": "Числитель показывает число выбранных частей.", "citations": []}))
    service.search = lambda *args, **kwargs: []
    result = answer(service, own_context(), query="Объясни числитель.")
    assert result["answer_routing"] == {"backend": "local", "reason": "own_conversation_context"}
    assert "<own_conversation_untrusted>" in session.requests[-1]["messages"][0]["content"]


@pytest.mark.parametrize("context_tokens", [8192, 16384, 32768])
def test_full_utf8_prompt_budget_does_not_treat_cyrillic_as_single_byte(context_tokens):
    system = {"role": "system", "content": "Фиксированная инструкция."}
    latest = {"role": "user", "content": "Последний вопрос."}
    huge = {"role": "assistant", "content": "я" * context_tokens}
    bounded, diagnostics = bound_conversation_prompt([system, huge, latest], context_tokens, 1024)
    assert bounded == [system, latest]
    assert diagnostics["omitted_content_bytes"] == context_tokens * 2
    assert diagnostics["used_bytes"] <= diagnostics["limit_bytes"]


def test_oversized_latest_question_is_rejected_without_model_input_truncation():
    with pytest.raises(ValueError, match="latest message exceed"):
        bound_conversation_prompt([{"role": "system", "content": "Инструкция"},
                                   {"role": "user", "content": "я" * 20000}], 8192, 1024)


def test_complete_runtime_guard_counts_utf8_for_the_entire_input():
    runtime, _, session = build()
    with pytest.raises(ValueError, match="Complete memory input"):
        with runtime.approved_memory_request():
            runtime.chat([{"role": "user", "content": "я" * 8000}])
    assert session.requests == []
