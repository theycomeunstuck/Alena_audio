"""Verbatim-quote guard on the personal-recall answer, observed through RagService.answer.

The model is a fake Ollama session. Each scenario feeds the answer route a fixed set of
retrieval ``sources`` and a scripted model reply, then checks the public result and the
``answer_prompt`` trace event. The guard may only use the sources already delivered to the
prompt, only acts when the latest user message asks for a verbatim quote, and never adds a
model call (router plus answer).
"""
from __future__ import annotations

import json
import re

import pytest

from local_app.rag import RagService
from local_app.runtime import OllamaRuntime
from tests.test_lesson_summary_continuity import FIRST, RECALL, SECOND
from tests.test_memory_trace import (MaterialsStore, Settings, TracedSession, by_phase, phases,
                                     recall_decision, trace_for)

DAY_FIRST = "2026-11-10"
DAY_SECOND = "2026-11-12"
UNAVAILABLE = ("В найденных репликах нет фразы, которую можно привести дословно. "
               "Подскажи, когда это было или о чём шла речь?")
S1, S2, S3 = re.split(r"(?<=[.!?…])\s+", FIRST)
ASSISTANT_TEXT = ("Давай разберём пример с маяком. Если 1200 разделить на 10, получится 120, "
                  "ноль терять нельзя. Попробуй теперь решить такую же задачу сам.")
ASSISTANT_UNAVAILABLE = ("В найденных репликах нет моего ответа, который можно привести дословно. "
                         "Подскажи, когда это было или о чём шла речь?")
ASK_ASSISTANT = "Процитируй дословно, что ты мне сказал в примере с маяком."
PARAPHRASE = "Ты рассказывала, что при делении на 10 потеряла ноль, и хотела потренироваться."
ROUTING = {"backend": "local", "reason": "personal_memory_recall"}


def source(source_id, text, *, day=DAY_FIRST, role="user", usage="self_report", **extra):
    return {"source_id": source_id, "role": role, "usage": usage, "text": text,
            "activity_day": day, "author_role": "student" if role == "user" else "assistant", **extra}


def assistant_source(source_id, text=ASSISTANT_TEXT, **extra):
    extra.setdefault("usage", "context_only")
    return source(source_id, text, role="assistant", **extra)


FIRST_SOURCE = source("src-first", FIRST)
SECOND_SOURCE = source("src-second", SECOND, day=DAY_SECOND)


def run(sources, reply, question=RECALL, *, system_text="[Q1] …", history=None):
    """One recall turn. Returns (result, answer_prompt event, fake session)."""
    events = []
    session = TracedSession(recall_decision(), reply)
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsStore([]), runtime, Settings())
    conversation_context = {"method": "conversation_history", "system_text": "СОБСТВЕННАЯ СВОДКА",
                            "recent_messages": []}
    user_ids = [row["source_id"] for row in sources if row.get("role") == "user"]
    retrieval = {"method": "conversation_retrieval", "system_text": system_text, "sources": sources,
                 "source_ids": user_ids, "evidence_ids": user_ids, "hits": [],
                 "diagnostics": {"budget_bytes": 4000, "archive_message_count": 6}}
    messages = history or [{"role": "user", "content": question}]
    result = service.answer(messages, conversation_context=conversation_context,
                            conversation_retriever=lambda query: retrieval, trace=trace_for(events))
    # The guard is code, not a model call: router plus the single answer call.
    assert len(session.requests) == 2
    assert phases(events) == ["resolver", "retrieval", "answer_prompt"]
    return result, by_phase(events, "answer_prompt")[0], session


def content(result):
    return result["choices"][0]["message"]["content"]


def assert_untouched(result, prompt, reply):
    """The model answer passed through: no replacement fields at all."""
    assert content(result) == reply.strip()
    assert result["answer_routing"] == ROUTING
    for key in ("answer_validation", "_local_diagnostics", "usage_scope"):
        assert key not in result
    assert prompt["final_prompt_sha256"]


def assert_replaced(result, prompt, session, *, reason, validation, selected, candidates, rejected):
    assert result["answer_routing"] == {"backend": "deterministic_local", "reason": reason,
                                        "model_attempt_backend": "local"}
    assert result["answer_validation"] == {"status": validation, "reason": reason}
    assert result["usage_scope"] == "includes_rejected_model_answer"
    diagnostics = result["_local_diagnostics"]
    assert diagnostics["type"] == "VerbatimQuoteMismatch"
    assert isinstance(diagnostics["message"], str) and diagnostics["message"]
    assert diagnostics["model_response"]["rejected_answer"] == rejected
    assert diagnostics["model_response"]["selected_source_id"] == selected
    assert diagnostics["model_response"]["candidate_source_ids"] == candidates
    assert len(content(result)) <= 900
    # The recorded final prompt is still the actual model call, not the replacement text.
    assert prompt["final_prompt"] == session.requests[-1]["messages"]


def quote_check(prompt, status, selected, candidates, match=None, author="user"):
    assert prompt["quote_check"] == {"status": status, "selected_source_id": selected,
                                     "candidate_source_ids": candidates, "match": match,
                                     "author": author}


def test_model_that_quotes_the_whole_source_passes_unchanged():
    reply = "Вот что ты сказала: «" + FIRST + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply)
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", "src-first", ["src-first"], match="full")


def test_whitespace_differences_do_not_fail_the_verbatim_check():
    reply = "«" + FIRST.replace(". ", ".\n").replace(", ", ",  ") + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply)
    assert_untouched(result, prompt, reply)
    assert prompt["quote_check"]["status"] == "model_verbatim"
    assert prompt["quote_check"]["match"] == "full"


def test_paraphrase_is_replaced_with_the_full_source_text():
    result, prompt, session = run([FIRST_SOURCE], PARAPHRASE)
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{FIRST}»"
    assert_replaced(result, prompt, session, reason="verbatim_quote_inserted", validation="replaced",
                    selected="src-first", candidates=["src-first"], rejected=PARAPHRASE)
    quote_check(prompt, "inserted", "src-first", ["src-first"])


def test_source_without_a_day_is_inserted_without_the_date():
    row = {key: value for key, value in FIRST_SOURCE.items() if key != "activity_day"}
    result, prompt, _ = run([row], PARAPHRASE)
    assert content(result) == f"Вот твоя фраза дословно: «{FIRST}»"
    assert prompt["quote_check"]["status"] == "inserted"


@pytest.mark.parametrize("sources", [[FIRST_SOURCE, SECOND_SOURCE], [SECOND_SOURCE, FIRST_SOURCE]],
                         ids=["lighthouse-first", "bridge-first"])
def test_lighthouse_wins_over_bridge_when_both_are_delivered(sources):
    result, prompt, session = run(sources, PARAPHRASE)
    assert FIRST in content(result) and SECOND not in content(result)
    candidates = [row["source_id"] for row in sources]
    assert_replaced(result, prompt, session, reason="verbatim_quote_inserted", validation="replaced",
                    selected="src-first", candidates=candidates, rejected=PARAPHRASE)
    quote_check(prompt, "inserted", "src-first", candidates)


def test_only_user_self_report_rows_are_candidates():
    assistant = source("src-assistant", "Покажи, как ты делила 1200 на 10 в примере с маяком.",
                       role="assistant", usage="context_only")
    result, prompt, _ = run([assistant, FIRST_SOURCE], PARAPHRASE)
    assert FIRST in content(result)
    quote_check(prompt, "inserted", "src-first", ["src-first"])


def test_repeated_request_still_quotes_the_original_phrase_not_the_stored_request():
    """The first request became a stored student message; it is a request, not quotable content."""
    stored_request = source("src-request", RECALL, day="2026-11-22")
    result, prompt, session = run([stored_request, FIRST_SOURCE], PARAPHRASE)
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{FIRST}»"
    assert_replaced(result, prompt, session, reason="verbatim_quote_inserted", validation="replaced",
                    selected="src-first", candidates=["src-first"], rejected=PARAPHRASE)
    quote_check(prompt, "inserted", "src-first", ["src-first"])


def test_only_the_stored_request_is_unavailable():
    result, prompt, session = run([source("src-request", RECALL, day="2026-11-22")], PARAPHRASE)
    assert content(result) == UNAVAILABLE
    assert_replaced(result, prompt, session, reason="verbatim_quote_unavailable",
                    validation="clarification", selected=None, candidates=[], rejected=PARAPHRASE)
    quote_check(prompt, "unavailable", None, [])


def test_earlier_guard_reply_is_not_quoted_as_the_assistant_answer():
    earlier_guard_reply = assistant_source("src-guard", f"Вот мой ответ дословно: «{ASSISTANT_TEXT}»",
                                           day="2026-11-22")
    result, prompt, session = run([earlier_guard_reply, assistant_source("src-asst")], PARAPHRASE, ASK_ASSISTANT)
    assert content(result) == f"Вот мой ответ дословно (запись от {DAY_FIRST}): «{ASSISTANT_TEXT}»"
    quote_check(prompt, "inserted", "src-asst", ["src-asst"], author="assistant")


@pytest.mark.parametrize("question", [
    "Процитируй дословно мою первую фразу.",
    "Процитируй дословно, как я делила на 10.",
], ids=["no-distinguishing-feature", "tie-on-shared-number"])
def test_similar_phrases_without_a_distinguishing_feature_ask_which_one(question):
    first = source("src-a", "Я разделила 1200 на 10 и получила 12.")
    second = source("src-b", "Я разделила 450 на 10 и получила 45.", day=DAY_SECOND)
    result, prompt, session = run([first, second], PARAPHRASE, question)
    assert content(result) == (
        "Нашлось несколько похожих фраз:\n"
        f"1. ({DAY_FIRST}) «{first['text']}»\n2. ({DAY_SECOND}) «{second['text']}»\n"
        "Какую из них ты имеешь в виду?")
    assert_replaced(result, prompt, session, reason="verbatim_quote_ambiguous", validation="clarification",
                    selected=None, candidates=["src-a", "src-b"], rejected=PARAPHRASE)
    quote_check(prompt, "ambiguous", None, ["src-a", "src-b"])


def test_ambiguous_answer_that_already_quotes_a_leader_is_kept():
    first = source("src-a", "Я разделила 1200 на 10 и получила 12.")
    second = source("src-b", "Я разделила 450 на 10 и получила 45.", day=DAY_SECOND)
    reply = "Скорее всего, ты имеешь в виду: «" + second["text"] + "»"
    result, prompt, _ = run([first, second], reply, "Процитируй дословно мою первую фразу.")
    assert_untouched(result, prompt, reply)
    # Even when the model quoted one leader, an ambiguous pick never names a selected source.
    quote_check(prompt, "model_verbatim", None, ["src-a", "src-b"], match="full")


def test_correction_markers_are_not_checked_when_the_choice_is_ambiguous():
    first = source("src-a", "Я разделила 1200 на 10 и получила 12.", superseded_by="src-fix")
    second = source("src-b", "Я разделила 450 на 10 и получила 45.", day=DAY_SECOND)
    result, prompt, _ = run([first, second], PARAPHRASE, "Процитируй дословно мою первую фразу.")
    assert result["answer_validation"]["reason"] == "verbatim_quote_ambiguous"
    quote_check(prompt, "ambiguous", None, ["src-a", "src-b"])


def test_long_ambiguous_quotes_are_cut_to_300_and_the_answer_stays_within_900():
    sentence = "Я разделила число на 10 и получила ответ, а потом проверила его умножением. "
    long_a = ("Первая запись. " + sentence * 12).strip()
    long_b = ("Вторая запись. " + sentence * 12).strip()
    assert len(long_a) > 800
    result, prompt, session = run([source("src-a", long_a), source("src-b", long_b, day=DAY_SECOND)],
                                  PARAPHRASE, "Процитируй дословно мою первую фразу.")
    answer = content(result)
    assert len(answer) <= 900
    quotes = re.findall(r"«(.*?)»", answer, re.S)
    assert len(quotes) == 2
    for quote, original in zip(quotes, (long_a, long_b)):
        assert quote.endswith("…")
        assert len(quote) <= 300, "the 300-character limit includes the ellipsis"
        assert original.startswith(quote.removesuffix("…").rstrip())
    assert answer.rstrip().endswith("Какую из них ты имеешь в виду?")
    assert_replaced(result, prompt, session, reason="verbatim_quote_ambiguous", validation="clarification",
                    selected=None, candidates=["src-a", "src-b"], rejected=PARAPHRASE)


def test_empty_sources_give_the_unavailable_answer_without_a_whole_history_claim():
    result, prompt, session = run([], PARAPHRASE)
    assert content(result) == UNAVAILABLE
    assert not re.search(r"истори", content(result), re.I)
    assert_replaced(result, prompt, session, reason="verbatim_quote_unavailable",
                    validation="clarification", selected=None, candidates=[], rejected=PARAPHRASE)
    quote_check(prompt, "unavailable", None, [])


def test_irrelevant_candidates_are_unavailable_even_if_the_model_quotes_them():
    other = source("src-ship", "Я нарисовала корабль.", day="2026-11-01")
    reply = "Ты писала: «" + other["text"] + "»"
    result, prompt, session = run([other], reply)
    assert content(result) == UNAVAILABLE
    assert_replaced(result, prompt, session, reason="verbatim_quote_unavailable",
                    validation="clarification", selected=None, candidates=["src-ship"], rejected=reply)
    quote_check(prompt, "unavailable", None, ["src-ship"])


@pytest.mark.parametrize("rows", [
    [source("src-assistant", FIRST, role="assistant", usage="context_only")],
    [source("src-empty", "")],
], ids=["assistant-row-with-the-text", "empty-text-row"])
def test_rows_outside_the_candidate_rule_are_not_quoted(rows):
    result, prompt, _ = run(rows, PARAPHRASE)
    assert content(result) == UNAVAILABLE
    assert prompt["quote_check"]["status"] == "unavailable"


def test_text_that_exists_only_in_the_prompt_block_is_not_inserted():
    """The guard reads the delivered sources list, not the rendered evidence block."""
    result, prompt, _ = run([], PARAPHRASE, system_text="[Q1] " + FIRST)
    assert content(result) == UNAVAILABLE
    assert FIRST not in content(result)
    assert prompt["quote_check"]["status"] == "unavailable"


def test_a_fabricated_quote_is_never_inserted_or_kept():
    other = source("src-ship", "Я нарисовала корабль.", day="2026-11-01")
    fabricated = "«В примере с маяком я разделила 1200 на 10 и получила 120.»"
    result, prompt, _ = run([other], fabricated)
    assert "получила 120" not in content(result)
    assert FIRST not in content(result)
    assert result["_local_diagnostics"]["model_response"]["rejected_answer"] == fabricated
    assert prompt["quote_check"]["status"] == "unavailable"


@pytest.mark.parametrize("field", ["superseded_by", "corrects_source_id"])
def test_corrected_source_is_skipped_and_the_answer_is_unchanged(field):
    corrected = source("src-first", FIRST, **{field: "src-fix"})
    result, prompt, _ = run([corrected], PARAPHRASE)
    assert_untouched(result, prompt, PARAPHRASE)
    quote_check(prompt, "skipped_correction", "src-first", ["src-first"])


@pytest.mark.parametrize("question", [
    "Вспомни мой первый урок с маяком про деление на 10. Что у меня получилось?",
    "Расскажи своими словами, что я говорила про маяк и деление на 10.",
], ids=["plain-recall", "retell-request"])
def test_ordinary_recall_without_a_marker_is_not_touched(question):
    result, prompt, _ = run([FIRST_SOURCE], PARAPHRASE, question)
    assert_untouched(result, prompt, PARAPHRASE)
    assert result["answer_routing"]["reason"] == "personal_memory_recall"
    assert "quote_check" not in prompt


def test_marker_in_an_earlier_user_message_does_not_trigger_the_guard():
    history = [{"role": "user", "content": RECALL},
               {"role": "assistant", "content": "Вот что нашлось."},
               {"role": "user", "content": "А что было потом, после маяка?"}]
    result, prompt, _ = run([FIRST_SOURCE], PARAPHRASE, history=history)
    assert_untouched(result, prompt, PARAPHRASE)
    assert "quote_check" not in prompt


@pytest.mark.parametrize("question", [
    "Вспомни урок про маяк с делением на 10 и скажи слово в слово.",
    "Вспомни урок про маяк с делением на 10 и скажи теми же словами.",
    "Вспомни урок про маяк с делением на 10 и скажи точными словами.",
    "Вспомни урок про маяк с делением на 10 и напиши цитату.",
    "ВСПОМНИ УРОК ПРО МАЯК С ДЕЛЕНИЕМ НА 10, ПРОЦИТИРУЙ МЕНЯ.",
    "Вспомни урок про маяк с делением на 10, нужно дословное повторение.",
], ids=["word-for-word", "same-words", "exact-words", "citation", "uppercase", "stem-dosloven"])
def test_every_marker_form_triggers_the_guard(question):
    result, prompt, _ = run([FIRST_SOURCE], PARAPHRASE, question)
    assert FIRST in content(result)
    assert prompt["quote_check"]["status"] == "inserted"


@pytest.mark.parametrize("question", [
    "Процитируй дословно, что ты мне сказал в примере с маяком.",
    "Процитируй дословно, что ты ответила в примере с маяком.",
    "Процитируй дословно, что ассистент мне сказал в примере с маяком.",
    "Процитируй дословно, что мне сказал помощник в примере с маяком.",
    "Процитируй дословно твой ответ в примере с маяком.",
    "Процитируй дословно слова помощника в примере с маяком.",
    "Что ты мне ответил в примере с маяком? Процитируй дословно.",
    "Процитируй дословно, что сказал ассистент в примере с маяком.",
], ids=["you-said", "you-answered", "assistant-said", "helper-said-to-me", "your-answer",
        "helper-words", "you-replied-to-me", "assistant-nominative"])
def test_assistant_wording_selects_assistant_author_and_template(question):
    result, prompt, session = run([FIRST_SOURCE, assistant_source("src-asst")], PARAPHRASE, question)
    assert content(result) == f"Вот мой ответ дословно (запись от {DAY_FIRST}): «{ASSISTANT_TEXT}»"
    assert_replaced(result, prompt, session, reason="verbatim_quote_inserted", validation="replaced",
                    selected="src-asst", candidates=["src-asst"], rejected=PARAPHRASE)
    quote_check(prompt, "inserted", "src-asst", ["src-asst"], author="assistant")


@pytest.mark.parametrize("question", [
    "Процитируй дословно, что я сказала помощнику в примере с маяком.",
    "Процитируй дословно слова, которые я написала в примере с маяком.",
    "Процитируй дословно, что я спросила у ассистента в примере с маяком.",
], ids=["i-told-helper", "words-i-wrote", "i-asked-assistant"])
def test_quoting_what_the_learner_said_to_the_assistant_stays_user_authored(question):
    result, prompt, _ = run([FIRST_SOURCE, assistant_source("src-asst")], PARAPHRASE, question)
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{FIRST}»"
    quote_check(prompt, "inserted", "src-first", ["src-first"], author="user")


def test_assistant_model_answer_with_the_full_reply_or_one_sentence_is_kept():
    for reply, match in (("Я сказала так: «" + ASSISTANT_TEXT + "»", "full"),
                         ("Я говорила: «Давай разберём пример с маяком»", "sentence")):
        result, prompt, _ = run([assistant_source("src-asst")], reply, ASK_ASSISTANT)
        assert_untouched(result, prompt, reply)
        quote_check(prompt, "model_verbatim", "src-asst", ["src-asst"], match=match, author="assistant")


def test_assistant_candidates_are_only_assistant_context_rows():
    rows = [FIRST_SOURCE,
            assistant_source("src-self-report", usage="self_report"),
            source("src-user-context", ASSISTANT_TEXT, usage="context_only"),
            assistant_source("src-empty", ""),
            assistant_source("src-asst")]
    result, prompt, _ = run(rows, PARAPHRASE, ASK_ASSISTANT)
    assert ASSISTANT_TEXT in content(result) and FIRST not in content(result)
    quote_check(prompt, "inserted", "src-asst", ["src-asst"], author="assistant")


def test_two_similar_assistant_replies_ask_which_one():
    first = assistant_source("src-a", "Вот первый разбор: 1200 разделить на 10 будет 120.")
    second = assistant_source("src-b", "Вот второй разбор: 450 разделить на 10 будет 45.", day=DAY_SECOND)
    result, prompt, session = run([first, second], PARAPHRASE, "Процитируй дословно, что ты мне сказал.")
    assert content(result) == (
        "Нашлось несколько похожих моих ответов:\n"
        f"1. ({DAY_FIRST}) «{first['text']}»\n2. ({DAY_SECOND}) «{second['text']}»\n"
        "Какой из них ты имеешь в виду?")
    assert_replaced(result, prompt, session, reason="verbatim_quote_ambiguous", validation="clarification",
                    selected=None, candidates=["src-a", "src-b"], rejected=PARAPHRASE)
    quote_check(prompt, "ambiguous", None, ["src-a", "src-b"], author="assistant")


@pytest.mark.parametrize("rows", [
    [FIRST_SOURCE],
    [],
    [assistant_source("src-other", "Это совсем другой разговор про корабли.")],
], ids=["only-user-rows", "no-rows", "irrelevant-assistant-row"])
def test_missing_assistant_reply_is_unavailable_without_a_whole_history_claim(rows):
    result, prompt, session = run(rows, PARAPHRASE, ASK_ASSISTANT)
    assert content(result) == ASSISTANT_UNAVAILABLE
    assert not re.search(r"истори", content(result), re.I)
    assert FIRST not in content(result)
    candidates = [row["source_id"] for row in rows if row["role"] == "assistant"]
    assert_replaced(result, prompt, session, reason="verbatim_quote_unavailable",
                    validation="clarification", selected=None, candidates=candidates, rejected=PARAPHRASE)
    quote_check(prompt, "unavailable", None, candidates, author="assistant")


def test_assistant_reply_that_is_superseded_is_skipped_and_the_answer_is_unchanged():
    result, prompt, _ = run([assistant_source("src-asst", superseded_by="src-fix")], PARAPHRASE, ASK_ASSISTANT)
    assert_untouched(result, prompt, PARAPHRASE)
    quote_check(prompt, "skipped_correction", "src-asst", ["src-asst"], author="assistant")


@pytest.mark.parametrize("question, rows", [
    (RECALL, [FIRST_SOURCE]),
    (RECALL, []),
    ("Процитируй дословно мою первую фразу.", [FIRST_SOURCE, SECOND_SOURCE]),
    (ASK_ASSISTANT, [assistant_source("src-asst")]),
    (ASK_ASSISTANT, []),
], ids=["user-inserted", "user-unavailable", "user-ambiguous", "assistant-inserted", "assistant-unavailable"])
def test_every_replacement_names_the_quote_author_and_never_reports_not_applicable(question, rows):
    result, prompt, _ = run(rows, PARAPHRASE, question)
    check = prompt["quote_check"]
    assert "answer_validation" in result
    assert check["author"] == ("assistant" if question == ASK_ASSISTANT else "user")
    assert check["status"] != "not_applicable"
    assert check["match"] is None


def test_text_missing_from_the_sources_is_never_inserted():
    result, prompt, _ = run([SECOND_SOURCE], PARAPHRASE)
    visible = {key: value for key, value in result.items() if key != "_local_diagnostics"}
    assert FIRST not in json.dumps(visible, ensure_ascii=False)


def padded(prefix, filler, length):
    """A source of exactly ``length`` characters that still names the lighthouse and 10."""
    text = (prefix + filler * (length // len(filler) + 1))[:length]
    assert len(text) == length
    return text


def test_long_source_is_cut_at_a_sentence_and_the_answer_stays_within_900():
    sentence = "Потом я решила ещё одну задачу про маяк и ответила уверенно. "
    text = ("В примере с маяком я разделила 1200 на 10. " + sentence * 25).strip()
    assert len(text) > 1000
    result, prompt, session = run([source("src-long", text)], PARAPHRASE)
    answer = content(result)
    assert len(answer) <= 900
    assert answer.startswith(f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «")
    assert answer.endswith("…»")
    quote = answer.split("«", 1)[1].removesuffix("»")
    assert len(quote) <= 700, "the 700-character limit includes the ellipsis"
    body = quote.removesuffix("…")
    assert text.startswith(body.rstrip()) and 600 < len(body) <= 699
    cut = body.rstrip()
    assert cut[-1] in ".!?…" or text[len(cut)] in ".!?…", "cut must land on a sentence boundary"
    assert_replaced(result, prompt, session, reason="verbatim_quote_inserted", validation="replaced",
                    selected="src-long", candidates=["src-long"], rejected=PARAPHRASE)


def test_source_at_the_limit_is_untouched_and_one_character_over_is_cut():
    filler = "Потом я решила ещё одну задачу про маяк и ответила уверенно. "
    exact = padded("В примере с маяком я разделила 1200 на 10. ", filler, 700).rstrip()
    exact = exact + "!" * (700 - len(exact))
    result, _, _ = run([source("src-exact", exact)], PARAPHRASE)
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{exact}»"
    over = exact + "!"
    result, _, _ = run([source("src-over", over)], PARAPHRASE)
    quote = content(result).split("«", 1)[1].removesuffix("»")
    assert quote.endswith("…") and len(quote) <= 700 and quote != over


def test_long_source_without_sentence_boundaries_is_cut_hard_before_the_ellipsis():
    text = "Маяком и делением на 10 " + "слово " * 200
    result, _, _ = run([source("src-long", text)], PARAPHRASE)
    answer = content(result)
    assert len(answer) <= 900 and answer.endswith("…»")
    quote = answer.split("«", 1)[1].removesuffix("»")
    body = quote.removesuffix("…")
    assert len(quote) <= 700 and text.startswith(body.rstrip()) and 690 <= len(body) <= 699


@pytest.mark.parametrize("day, shown", [
    ("2026-11-10", "2026-11-10"),
    ("2026-11-10T14:30:00+03:00", "2026-11-10"),
    ("10.11.2026", None),
    ("вчера", None),
    ("", None),
    (None, None),
], ids=["iso-date", "iso-datetime", "dotted", "word", "empty", "none"])
def test_only_an_iso_activity_day_is_shown(day, shown):
    result, _, _ = run([source("src-first", FIRST, day=day)], PARAPHRASE)
    expected = (f"Вот твоя фраза дословно (запись от {shown}): «{FIRST}»" if shown
                else f"Вот твоя фраза дословно: «{FIRST}»")
    assert content(result) == expected


def test_ambiguous_answer_omits_non_iso_days():
    first = source("src-a", "Я разделила 1200 на 10 и получила 12.", day="вчера")
    second = source("src-b", "Я разделила 450 на 10 и получила 45.", day=DAY_SECOND)
    result, _, _ = run([first, second], PARAPHRASE, "Процитируй дословно мою первую фразу.")
    assert content(result) == (
        "Нашлось несколько похожих фраз:\n"
        f"1. «{first['text']}»\n2. ({DAY_SECOND}) «{second['text']}»\n"
        "Какую из них ты имеешь в виду?")


def test_sentence_split_of_the_lighthouse_row_has_three_sentences():
    assert [S1, S2, S3] == re.split(r"(?<=[.!?…])\s+", FIRST)
    assert S1.startswith("В примере с маяком") and S3.endswith("шары?")


@pytest.mark.parametrize("quoted", [S1, S1.rstrip(".!?…"), S1.rstrip(".!?…") + "!", S1.rstrip(".!?…") + "…",
                                    "  ".join(S1.split(" "))],
                         ids=["with-period", "without-period", "exclamation", "ellipsis", "extra-spaces"])
def test_model_quoting_only_the_best_sentence_is_kept(quoted):
    reply = "Ты тогда сказала: «" + quoted + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply)
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", "src-first", ["src-first"], match="sentence")


@pytest.mark.parametrize("quoted", [S2, S3], ids=["second-sentence", "third-sentence"])
def test_model_quoting_a_weaker_sentence_is_replaced_with_the_full_text(quoted):
    reply = "Ты тогда сказала: «" + quoted + "»"
    result, prompt, session = run([FIRST_SOURCE], reply)
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{FIRST}»"
    assert_replaced(result, prompt, session, reason="verbatim_quote_inserted", validation="replaced",
                    selected="src-first", candidates=["src-first"], rejected=reply)
    quote_check(prompt, "inserted", "src-first", ["src-first"], match=None)


@pytest.mark.parametrize("best, status, match", [
    ("Маяком 10 ноль потеряла.", "model_verbatim", "sentence"),
    ("Маяком 10 ноль.", "inserted", None),
], ids=["four-words-counts", "three-words-does-not"])
def test_a_sentence_needs_at_least_four_words(best, status, match):
    text = best + " Потом мы ещё долго говорили совсем о другом деле."
    reply = "Ты написала: «" + best + "»"
    result, prompt, _ = run([source("src-short", text)], reply)
    quote_check(prompt, status, "src-short", ["src-short"], match=match)
    if status == "model_verbatim":
        assert_untouched(result, prompt, reply)
    else:
        assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{text}»"


def test_sentence_match_applies_to_any_leader_when_the_choice_is_ambiguous():
    reply = "Скорее всего это было так: «" + S2 + "»"
    result, prompt, _ = run([FIRST_SOURCE, SECOND_SOURCE], reply, "Процитируй дословно мою первую фразу.")
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", None, ["src-first", "src-second"], match="sentence")


def test_ambiguous_sentence_must_be_a_best_sentence_of_a_leader():
    # Both rows tie on the shared number 10. Sentence 2 mentions 10, sentence 3 does not.
    question = "Процитируй дословно, как я делила на 10."
    result, prompt, _ = run([FIRST_SOURCE, SECOND_SOURCE], "«" + S2 + "»", question)
    quote_check(prompt, "model_verbatim", None, ["src-first", "src-second"], match="sentence")
    result, prompt, _ = run([FIRST_SOURCE, SECOND_SOURCE], "«" + S3 + "»", question)
    assert content(result).startswith("Нашлось несколько похожих фраз:")
    quote_check(prompt, "ambiguous", None, ["src-first", "src-second"], match=None)


def test_sentence_of_a_row_that_is_not_selected_does_not_count():
    result, prompt, _ = run([FIRST_SOURCE, SECOND_SOURCE], "«" + SECOND + "»")
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{FIRST}»"
    quote_check(prompt, "inserted", "src-first", ["src-first", "src-second"], match=None)


def test_without_question_features_every_long_enough_sentence_is_a_best_one():
    reply = "«" + S3 + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply, "Процитируй дословно мою первую фразу.")
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", "src-first", ["src-first"], match="sentence")


def test_punctuation_of_the_source_sentence_is_not_compared():
    best = "Мы с маяком разделили число на 10 и получили двенадцать"
    text = best + "! Потом мы ещё долго говорили совсем о другом деле."
    for ending in (".", "!", "?", "", "…"):
        reply = "Ты писала: «" + best + ending + "»"
        result, prompt, _ = run([source("src-punct", text)], reply)
        assert_untouched(result, prompt, reply)
        quote_check(prompt, "model_verbatim", "src-punct", ["src-punct"], match="sentence")


@pytest.mark.parametrize("reply", [
    "Ты писала: Мы с маяком разделили число на 100",
    "Ты писала: ВМы с маяком разделили число на 10",
    "Ты писала: Мы с маяком разделили число на 10x",
    "Ты писала: aМы с маяком разделили число на 10",
], ids=["right-digit", "left-letter", "right-latin", "left-latin"])
def test_sentence_inside_a_longer_word_or_number_does_not_count(reply):
    best = "Мы с маяком разделили число на 10"
    text = best + ". Потом мы ещё долго говорили совсем о другом деле."
    result, prompt, _ = run([source("src-bound", text)], reply)
    assert content(result) == f"Вот твоя фраза дословно (запись от {DAY_FIRST}): «{text}»"
    quote_check(prompt, "inserted", "src-bound", ["src-bound"], match=None)


@pytest.mark.parametrize("reply", [
    "Ты писала:«Мы с маяком разделили число на 10».",
    "Ты писала: (Мы с маяком разделили число на 10)",
    "Ты писала:\nМы с маяком разделили число на 10\nВот так.",
], ids=["guillemets", "parentheses", "newlines"])
def test_sentence_next_to_punctuation_or_whitespace_counts(reply):
    best = "Мы с маяком разделили число на 10"
    text = best + ". Потом мы ещё долго говорили совсем о другом деле."
    result, prompt, _ = run([source("src-bound", text)], reply)
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", "src-bound", ["src-bound"], match="sentence")


def test_full_text_wins_over_a_sentence_match_among_ambiguous_leaders():
    reply = "«" + S2 + "» и ещё «" + SECOND + "»"
    result, prompt, _ = run([FIRST_SOURCE, SECOND_SOURCE], reply, "Процитируй дословно мою первую фразу.")
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", None, ["src-first", "src-second"], match="full")


def test_full_text_and_sentence_in_one_answer_report_full():
    reply = "«" + S1 + "» и вся реплика целиком: «" + FIRST + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply)
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", "src-first", ["src-first"], match="full")
