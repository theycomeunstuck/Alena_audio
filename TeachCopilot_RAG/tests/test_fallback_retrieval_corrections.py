"""Fallback retrieval attaches correction metadata without changing the delivered prompt."""
from copy import deepcopy

import pytest
from psycopg2 import Error as DatabaseError

from local_app import conversation_retrieval as retrieval
from local_app.recall_quote_guard import recall_quote_guard
from tests.test_conversation_retrieval import Runtime, Store, message


class CorrectionStore(Store):
    def __init__(self, messages, refs=None, *, fail=False):
        super().__init__(messages)
        self.refs = refs or {}
        self.fail = fail
        self.correction_calls = []

    def memory_source_correction_refs(self, learner_id, source_ids):
        self.correction_calls.append((learner_id, list(source_ids)))
        if self.fail:
            raise DatabaseError("injected correction lookup failure")
        return {key: self.refs[key] for key in source_ids if key in self.refs}


@pytest.fixture(autouse=True)
def clear_memory_cache():
    retrieval._MEMO.clear()
    yield
    retrieval._MEMO.clear()


def run(store, question="Дословно про корабль Сокол", **kwargs):
    return retrieval.retrieve_conversation_context(store, Runtime(), "child", "account", question, **kwargs)


@pytest.mark.parametrize("marker", ["corrects_source_id", "superseded_by", "both"])
def test_correction_markers_are_attached_after_selection_without_changing_prompt(monkeypatch, marker):
    row = message(0, "Корабль Сокол был нарисован мной.",
                  **({"superseded_by": "new-source"} if marker in {"superseded_by", "both"} else {}))
    refs = {row["source_id"]: "old-source"} if marker in {"corrects_source_id", "both"} else {}
    baseline = run(Store([row]))
    captured = {}
    original = retrieval._select_quotes

    def record_selection(*args, **kwargs):
        assert "correction_refs" not in kwargs
        result = original(*args, **kwargs)
        captured["original"] = deepcopy(result)
        return result

    monkeypatch.setattr(retrieval, "_select_quotes", record_selection)
    store = CorrectionStore([row], refs)
    result = run(store)
    selected_text, selected_quotes, selected_hits, _ = captured["original"]
    assert result["text"].encode("utf-8") == selected_text.encode("utf-8")
    assert result["system_text"].encode("utf-8") == selected_text.encode("utf-8")
    assert result["system_text"].encode("utf-8") == baseline["system_text"].encode("utf-8")
    assert result["text"].encode("utf-8") == baseline["text"].encode("utf-8")
    assert result["hits"] == selected_hits == baseline["hits"]
    assert store.correction_calls == [("child", [quote["source_id"] for quote in selected_quotes])]
    assert result["diagnostics"]["correction_refs_status"] == "ok"
    assert len(result["sources"]) == 1
    quote = result["sources"][0]
    assert {k: v for k, v in quote.items() if k not in {"corrects_source_id", "superseded_by"}} == selected_quotes[0]
    assert quote.get("corrects_source_id") == ("old-source" if refs else None)
    assert quote.get("superseded_by") == ("new-source" if marker in {"superseded_by", "both"} else None)
    check = recall_quote_guard("Дословно про корабль Сокол", result["sources"], "пересказ")
    assert check["status"] == "skipped_correction"


def test_lookup_exception_still_carries_superseded_by():
    row = message(0, "Корабль Сокол был нарисован мной.", superseded_by="new-source")
    store = CorrectionStore([row], fail=True)
    result = run(store)
    assert result["diagnostics"]["correction_refs_status"] == "unavailable"
    assert result["sources"][0]["superseded_by"] == "new-source"
    assert "corrects_source_id" not in result["sources"][0]
    assert recall_quote_guard("Дословно про корабль Сокол", result["sources"], "пересказ")["status"] == "skipped_correction"


def test_missing_method_and_fields_keep_existing_retrieval_working():
    result = run(Store([message(0, "Корабль Сокол был нарисован мной.")]))
    assert result["diagnostics"]["correction_refs_status"] == "not_supported"
    assert "superseded_by" not in result["sources"][0]
    assert "corrects_source_id" not in result["sources"][0]


def test_empty_refs_are_ok_and_only_delivered_source_ids_are_looked_up():
    rows = [message(0, "Корабль Сокол был нарисован мной."),
            message(1, "Корабль Сокол стоит у моста.", day="2026-11-01")]
    store = CorrectionStore(rows, {"not-delivered": "old"})
    result = run(store, time_filter={"from": "2026-10-01", "through": "2026-10-31"})
    assert [quote["source_id"] for quote in result["sources"]] == ["opaque-source-0"]
    assert store.correction_calls == [("child", ["opaque-source-0"])]
    assert result["diagnostics"]["correction_refs_status"] == "ok"
    assert "corrects_source_id" not in result["sources"][0]


def test_literal_span_and_assistant_correction_keep_the_same_prompt():
    row = message(0, "Сокол " + "длинный абзац " * 400, role="assistant")
    baseline = run(Store([row]), budget_bytes=3500)
    store = CorrectionStore([row], {row["source_id"]: "original"})
    result = run(store, budget_bytes=3500)
    assert result["sources"] and result["sources"][0]["excerpt_mode"] == "literal_span"
    assert result["sources"][0]["role"] == "assistant"
    assert result["sources"][0]["corrects_source_id"] == "original"
    assert result["sources"][0]["text"] == baseline["sources"][0]["text"]
    assert result["system_text"].encode("utf-8") == baseline["system_text"].encode("utf-8")
    assert recall_quote_guard("Процитируй дословно слова помощника про Сокол", result["sources"], "пересказ")["status"] == "skipped_correction"


def test_correction_status_is_reported_on_the_early_exit():
    """Closeout of the PR #2 limitation: retrieve_conversation_context returned
    early (empty archive or zero budget) without correction_refs_status."""
    from local_app.conversation_retrieval import retrieve_conversation_context
    from tests.test_conversation_retrieval import Runtime, Store, message

    empty = retrieve_conversation_context(Store([]), Runtime(), "child", "account", "что я рисовала?")
    assert empty["diagnostics"]["correction_refs_status"] == "not_needed"
    no_budget = retrieve_conversation_context(Store([message(0, "Я рисовала корабль.")]), Runtime(),
                                              "child", "account", "что я рисовала?", budget_bytes=0)
    assert no_budget["diagnostics"]["correction_refs_status"] == "not_needed"
