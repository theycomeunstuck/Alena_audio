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
PARAPHRASE = "Ты рассказывала, что при делении на 10 потеряла ноль, и хотела потренироваться."
ROUTING = {"backend": "local", "reason": "personal_memory_recall"}


def source(source_id, text, *, day=DAY_FIRST, role="user", usage="self_report", **extra):
    return {"source_id": source_id, "role": role, "usage": usage, "text": text,
            "activity_day": day, "author_role": "student" if role == "user" else "assistant", **extra}


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


def quote_check(prompt, status, selected, candidates):
    assert prompt["quote_check"] == {"status": status, "selected_source_id": selected,
                                     "candidate_source_ids": candidates}


def test_model_that_quotes_the_whole_source_passes_unchanged():
    reply = "Вот что ты сказала: «" + FIRST + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply)
    assert_untouched(result, prompt, reply)
    quote_check(prompt, "model_verbatim", "src-first", ["src-first"])


def test_whitespace_differences_do_not_fail_the_verbatim_check():
    reply = "«" + FIRST.replace(". ", ".\n").replace(", ", ",  ") + "»"
    result, prompt, _ = run([FIRST_SOURCE], reply)
    assert_untouched(result, prompt, reply)
    assert prompt["quote_check"]["status"] == "model_verbatim"


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
    quote_check(prompt, "model_verbatim", None, ["src-a", "src-b"])


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
    "Процитируй дословно, что ты мне сказал про маяк.",
    "Процитируй дословно, что ты ответила про деление на 10.",
    "Процитируй дословно, что сказал ассистент про маяк.",
    "Процитируй дословно слова помощника про маяк.",
], ids=["you-said", "you-answered", "assistant", "helper"])
def test_asking_to_quote_the_assistant_is_not_applicable(question):
    result, prompt, _ = run([FIRST_SOURCE], PARAPHRASE, question)
    assert_untouched(result, prompt, PARAPHRASE)
    assert prompt["quote_check"]["status"] == "not_applicable"
    assert FIRST not in content(result)


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
