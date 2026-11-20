"""Literal recall uses only admitted excerpts, without changing ordinary social replies."""
from datetime import date, datetime

import pytest

from local_app.conversation_retrieval import _stem_list
from local_app.recall_quote_guard import (
    contains_full_quote, eligible_sources, format_ambiguous, format_inserted,
    is_assistant_quote_request, is_verbatim_request, quote_features,
    recall_quote_guard, select_quote_sources, truncate_quote,
)


FIRST = ("В примере с маяком я разделила 1200 на 10 и получила 12: потеряла ноль. "
         "В следующий раз хочу потренироваться делить на 10, чтобы не терять ноль. "
         "Можно сегодня одну задачу про воздушные шары?")
SECOND = "В примере с мостом я разделила 450 на 10 и получила 45."
RECALL = ("Вспомни мой первый урок с маяком про деление на 10. "
          "Процитируй дословно мою первую фразу о том, что у меня получилось.")


def source(identifier, text, **extra):
    return {"source_id": identifier, "role": "user", "usage": "self_report", "text": text, **extra}


@pytest.mark.parametrize("query", ["ДОСЛОВНО", "Процитируй", "ЦИТАТУ", "слово в слово",
                                    "теми же словами", "точными словами", "ПРОЦИТИРЁШЬ"])
def test_quote_markers(query):
    assert is_verbatim_request(query)


@pytest.mark.parametrize("query", ["Вспомни мой урок", "слова", "что я сказала?"])
def test_non_quote_requests(query):
    assert not is_verbatim_request(query)
    assert recall_quote_guard(query, [source("1", "фраза")], "ответ") is None


@pytest.mark.parametrize("query", ["ты сказал", "ТЫ МНЕ НАПИСАЛА", "ты ответил", "ты объяснял",
                                    "ты спрашивал", "ты показывал", "ассистент", "помощник"])
def test_assistant_requests(query):
    assert is_assistant_quote_request(query)
    check = recall_quote_guard("Дословно " + query, [source("1", "пример")], "ответ")
    assert check == {"status": "not_applicable", "selected_source_id": None, "candidate_source_ids": ["1"]}


def test_candidates_are_only_nonempty_self_report_users_in_original_order():
    first = source("1", "  привет  ", excerpt_mode="literal_span")
    last = source("2", "только переданный фрагмент")
    rows = [first, {}, source("3", " "), source("4", "текст", role="assistant"),
            source("5", "текст", usage="context_only"), last]
    assert eligible_sources(rows) == [first, last]
    check = recall_quote_guard("Дословно", [last], "пересказ")
    assert check["answer"] == "Вот твоя фраза дословно: «только переданный фрагмент»"


def test_features_stop_stems_numbers_and_lighthouse_regression():
    stop = "вспомни процитируй дословно цитата цитату фраза фразу первый первую урок уроке слово словами точно точными получилось сказала сказал написала написал"
    assert not quote_features(stop)
    assert {"1", "10", "1200"} <= quote_features("1 10 1200 10")
    features = quote_features(RECALL)
    assert len(features & quote_features(FIRST)) == 2
    assert len(features & quote_features(SECOND)) == 1
    assert set(_stem_list("Маяком")) <= quote_features("МАЯКОМ")
    assert select_quote_sources(RECALL, [source("first", FIRST), source("second", SECOND)]) == (
        "unique", [source("first", FIRST)])


@pytest.mark.parametrize(("question", "texts", "expected", "leader_ids"), [
    ("дословно маяк", [], "unavailable", []),
    ("дословно маяк", ["мост", "домик"], "unavailable", []),
    ("дословно маяк", ["маяк 10", "маяк 20"], "ambiguous", ["0", "1"]),
    ("дословно 10 маяк", ["маяк 10", "маяк 20"], "unique", ["0"]),
    ("дословно", ["что-то"], "unique", ["0"]),
    ("дословно", ["первое", "второе", "третье"], "ambiguous", ["0", "1", "2"]),
])
def test_select_leaders(question, texts, expected, leader_ids):
    rows = [source(str(i), text) for i, text in enumerate(texts)]
    kind, leaders = select_quote_sources(question, rows)
    assert kind == expected
    assert [row["source_id"] for row in leaders] == leader_ids


@pytest.mark.parametrize("field", ["superseded_by", "corrects_source_id"])
def test_selected_correction_skips_even_if_model_quotes(field):
    row = source("1", "маяк", **{field: "2"})
    check = recall_quote_guard("дословно маяк", [row], "маяк")
    assert check == {"status": "skipped_correction", "selected_source_id": "1", "candidate_source_ids": ["1"]}


def test_ambiguous_correction_does_not_skip_and_ids_cover_all_candidates():
    rows = [source("a", "маяк A", superseded_by="b"), source("b", "маяк B"), source("c", "мост")]
    check = recall_quote_guard("дословно маяк", rows, "пересказ")
    assert check["status"] == "ambiguous"
    assert check["selected_source_id"] is None
    assert check["candidate_source_ids"] == ["a", "b", "c"]
    assert "маяк A" in check["answer"] and "маяк B" in check["answer"]
    check = recall_quote_guard("дословно маяк", rows, "«маяк A»")
    assert check["status"] == "model_verbatim"
    assert check["selected_source_id"] is None
    assert "answer" not in check


@pytest.mark.parametrize(("answer", "quote", "expected"), [
    ("До Я\n рисую\tкорабль. после", "Я рисую корабль.", True),
    ("я рисую корабль.", "Я рисую корабль.", False),
    ("Я рисую корабль!", "Я рисую корабль.", False),
    ("Я рисую", "Я рисую корабль.", False),
])
def test_full_quote_only_collapses_whitespace(answer, quote, expected):
    assert contains_full_quote(answer, quote) is expected


@pytest.mark.parametrize("mark", list(".!?…"))
@pytest.mark.parametrize("limit", [300, 700])
def test_truncation_at_last_boundary_with_ellipsis_inside_limit(mark, limit):
    text = "а" * 20 + "." + "б" * 40 + mark + "в" * limit
    assert truncate_quote(text, limit) == "а" * 20 + "." + "б" * 40 + mark + "…"
    assert len(truncate_quote(text, limit)) <= limit


@pytest.mark.parametrize("limit", [300, 700])
def test_truncation_edges_and_hard_cut(limit):
    assert truncate_quote("а" * limit, limit) == "а" * limit
    assert truncate_quote("а" * (limit + 1), limit) == "а" * (limit - 1) + "…"
    assert truncate_quote("а" * (limit - 1) + ".дальше", limit) == "а" * (limit - 1) + "…"


@pytest.mark.parametrize(("value", "expected"), [
    ("2026-11-10", "2026-11-10"), ("2026-11-10T13:04:20Z", "2026-11-10"),
    (datetime(2026, 11, 10, 13, 4), "2026-11-10"), (date(2026, 11, 10), "2026-11-10"),
    ("2026-02-30", None), ("2026-11-10 invalid", None), ("x" * 1000, None), (None, None),
])
def test_inserted_date_formats(value, expected):
    result = format_inserted(source("a", "текст", activity_day=value))
    assert result == (f"Вот твоя фраза дословно (запись от {expected}): «текст»" if expected
                      else "Вот твоя фраза дословно: «текст»")


def test_deterministic_templates_are_bounded_and_show_only_two_leaders():
    rows = [source("a", "а" * 1000, activity_day="2026-11-10"),
            source("b", "б" * 1000), source("c", "не показывать")]
    inserted = format_inserted(rows[0])
    assert inserted == "Вот твоя фраза дословно (запись от 2026-11-10): «" + "а" * 699 + "…»"
    ambiguous = format_ambiguous(rows)
    assert ambiguous == ("Нашлось несколько похожих фраз:\n1. (2026-11-10) «" + "а" * 299 + "…»\n2. «"
                         + "б" * 299 + "…»\nКакую из них ты имеешь в виду?")
    assert len(inserted) <= 900 and len(ambiguous) <= 900


def test_guard_statuses_and_exact_unavailable_template():
    row = source("a", "маяк", activity_day="2026-11-10")
    assert recall_quote_guard("дословно маяк", [row], "Вот «маяк»")["status"] == "model_verbatim"
    inserted = recall_quote_guard("дословно маяк", [row], "пересказ")
    assert inserted["status"] == "inserted" and inserted["selected_source_id"] == "a"
    assert inserted["answer"] == "Вот твоя фраза дословно (запись от 2026-11-10): «маяк»"
    unavailable = recall_quote_guard("дословно корабль", [row], "маяк")
    assert unavailable == {"status": "unavailable", "selected_source_id": None, "candidate_source_ids": ["a"],
                           "answer": "В найденных репликах нет фразы, которую можно привести дословно. "
                                     "Подскажи, когда это было или о чём шла речь?"}


@pytest.mark.parametrize(("question", "rows", "model_answer", "status"), [
    ("Дословно маяк", [source("a", "маяк")], "пересказ", "inserted"),
    ("Дословно маяк", [source("a", "маяк 1"), source("b", "маяк 2")], "пересказ", "ambiguous"),
    ("Дословно маяк", [source("a", "мост")], "пересказ", "unavailable"),
    ("Дословно маяк", [source("a", "маяк")], "«маяк»", "model_verbatim"),
    ("Дословно что ты сказал", [source("a", "маяк")], "пересказ", "not_applicable"),
    ("Дословно маяк", [source("a", "маяк", corrects_source_id="b")], "пересказ", "skipped_correction"),
    ("Что было с маяком?", [source("a", "маяк")], "пересказ", None),
])
def test_social_service_result_and_trace(question, rows, model_answer, status):
    import json
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from tests.test_memory_trace import MaterialsStore, TracedSession, prompt_digest, recall_decision, trace_for
    from tests.test_social_chat import Settings

    retrieval = {"method": "conversation_retrieval", "system_text": json.dumps(rows, ensure_ascii=False),
                 "sources": rows, "source_ids": [row["source_id"] for row in rows],
                 "evidence_ids": [row["source_id"] for row in rows], "diagnostics": {}}
    context = {"method": "conversation_history", "system_text": "", "recent_messages": (
        [{"role": "user", "content": "Процитируй дословно прошлую фразу", "usage": "self_report"}]
        if status is None else [])}
    session = TracedSession(recall_decision(), model_answer)
    events = []
    service = RagService(MaterialsStore([]), OllamaRuntime(Settings(), session=session), Settings())
    result = service.answer([{"role": "user", "content": question}], conversation_context=context,
                            conversation_retriever=lambda _: retrieval, trace=trace_for(events))
    assert len(session.requests) == 2
    event = next(event for event in events if event["phase"] == "answer_prompt")
    assert event["final_prompt"] == session.requests[-1]["messages"]
    assert event["final_prompt_sha256"] == prompt_digest(session.requests[-1]["messages"])
    if status is None:
        assert "quote_check" not in event
    else:
        assert event["quote_check"]["status"] == status
        assert event["quote_check"]["candidate_source_ids"] == [row["source_id"] for row in rows]
    if status in {"inserted", "ambiguous", "unavailable"}:
        reason = "verbatim_quote_" + status
        assert result["answer_routing"] == {"backend": "deterministic_local", "reason": reason,
                                             "model_attempt_backend": "local"}
        assert result["answer_validation"] == {"status": "replaced" if status == "inserted" else "clarification",
                                               "reason": reason}
        assert result["usage_scope"] == "includes_rejected_model_answer"
        assert result["_local_diagnostics"]["type"] == "VerbatimQuoteMismatch"
        assert result["_local_diagnostics"]["model_response"]["rejected_answer"] == model_answer
        assert result["choices"][0]["message"]["content"] != model_answer
    else:
        assert result["choices"][0]["message"]["content"] == model_answer
        assert result["answer_routing"] == {"backend": "local", "reason": "personal_memory_recall"}
        assert not {"answer_validation", "_local_diagnostics", "usage_scope"} & result.keys()
