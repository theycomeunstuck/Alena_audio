"""Own archive retrieval and fixed chunk provenance without model services."""
from copy import deepcopy
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace

import pytest

from local_app import conversation_retrieval as retrieval
from local_app.dialogue_catalog import classify_chunks, summarize_catalog


class Store:
    def __init__(self, messages):
        self.messages, self.calls = messages, []

    def get_own_conversation_messages(self, learner_id, actor_id, **kwargs):
        self.calls.append((learner_id, actor_id, kwargs))
        excluded = kwargs.get("exclude_source_id")
        return deepcopy([message for message in self.messages if not excluded or
                         message["source_id"] != excluded and message.get("pair_source_id") != excluded])


class Runtime:
    settings = SimpleNamespace(embedding_model="bge-m3", ollama_url="http://127.0.0.1:11435")

    def __init__(self):
        self.calls = []

    def embedding_model_identity(self):
        return "bge-m3@" + "b" * 64

    def embed(self, texts):
        self.calls.extend(texts)
        return [([1.0, 0.0] if any(word in text.casefold() for word in ("сокол", "судно", "корабл"))
                 else [0.0, 1.0]) + [0.0] * 1022 for text in texts]


def message(index, text, *, role="user", day="2026-10-01", **extra):
    return {"source_id": f"opaque-source-{index}", "source_order": index + 1, "learner_id": "child",
            "actor_id": "account", "role": role, "author_role": "student" if role == "user" else "assistant",
            "text": text, "activity_day": date.fromisoformat(day), "activity_timezone": "Asia/Tomsk",
            "event_timezone": "Asia/Tomsk", "created_at": datetime.fromisoformat(day + "T10:15:00+00:00"),
            "learning_session_id": f"opaque-session-{index}", "topic_id": "general", "delivery_status": "completed",
            "usage": "self_report" if role == "user" else "context_only", **extra}


@pytest.fixture(autouse=True)
def clear_memory_cache():
    retrieval._MEMO.clear()
    yield
    retrieval._MEMO.clear()


def test_old_fact_is_retrieved_from_all_history_with_meta_only_ids_and_exact_instants():
    store = Store([message(0, "Мой корабль назывался Сокол.")] +
                  [message(index, "Мы считали яблоки.", day="2026-11-01") for index in range(1, 42)])
    result = retrieval.retrieve_conversation_context(store, Runtime(), "child", "account", "Как я называл судно?")
    assert store.calls[0][2]["limit"] is None and store.calls[0][2]["active_day_limit"] is None
    assert "Сокол" in result["system_text"] and result["source_ids"] == ["opaque-source-0"]
    assert result["sources"][0]["created_at"] == "2026-10-01T10:15:00+00:00"
    assert "opaque-source" not in result["system_text"] and "opaque-session" not in result["system_text"]
    assert "T10:15" not in result["system_text"] and '"quote":"Q1"' in result["system_text"]
    assert len(result["system_text"].encode("utf-8")) <= 4000
    assert result["diagnostics"]["archive_message_count"] == 42


def test_fixed_window_spans_reconstruct_literal_messages_and_keep_short_replies_in_context():
    rows = [message(0, "Я назвал корабль Сокол.", learning_session_id="session"),
            message(1, "Он был синий?", role="assistant", learning_session_id="session"),
            message(2, "Да", learning_session_id="session")]
    chunks = retrieval.chunk_conversation_messages(rows, "child", "account")
    assert len(chunks) == 1 and len(chunks[0]["spans"]) == 3
    by_id = {row["source_id"]: row for row in rows}
    for span in chunks[0]["spans"]:
        assert chunks[0]["text"][span["chunk_start"]:span["chunk_end"]] == by_id[span["source_id"]]["text"][span["start"]:span["end"]]
    assert "Ассистент: Он был синий?" in chunks[0]["text"] and "Пользователь: Да" in chunks[0]["text"]


def test_long_paragraph_uses_larger_fixed_overlap_and_literal_excerpts_when_parent_cannot_fit():
    row = message(0, "Сокол " + "длинный абзац " * 400)
    chunks = retrieval.chunk_conversation_messages([row], "child", "account")
    assert chunks[0]["end"] - chunks[1]["start"] == 240
    for chunk in chunks:
        for span in chunk["spans"]:
            assert chunk["text"][span["chunk_start"]:span["chunk_end"]] == row["text"][span["start"]:span["end"]]
    result = retrieval.retrieve_conversation_context(Store([row]), Runtime(), "child", "account", "Про судно?", budget_bytes=3500)
    assert result["sources"] and result["sources"][0]["excerpt_mode"] == "literal_span"
    assert len(result["text"].encode("utf-8")) <= 3500


def test_failed_user_text_remains_available_and_completed_pair_is_restored():
    failed = message(0, "Мой корабль Сокол.", delivery_status="failed")
    result = retrieval.retrieve_conversation_context(Store([failed]), Runtime(), "child", "account", "Как называлось судно?")
    assert result["sources"][0]["delivery_status"] == "failed"
    own = message(1, "Я нарисовал корабль Сокол.", paired_source_id="opaque-source-2", pair_source_id="opaque-source-1")
    assistant = message(2, "Расскажи про рисунок.", role="assistant", pair_source_id="opaque-source-1", paired_source_id="opaque-source-1")
    paired = retrieval.retrieve_conversation_context(Store([own, assistant]), Runtime(), "child", "account", "Как называлось судно?")
    assert {quote["role"] for quote in paired["sources"]} == {"user", "assistant"}
    assert paired["evidence_ids"] == ["opaque-source-1"]


def test_time_facet_filters_recorded_activity_day_before_search():
    rows = [message(0, "Корабль Сокол.", day="2026-09-01"), message(1, "Корабль Ястреб.", day="2026-10-01")]
    result = retrieval.retrieve_conversation_context(Store(rows), Runtime(), "child", "account", "Про судно?",
        time_filter={"from": "2026-10-01", "through": "2026-10-31"})
    assert result["source_ids"] == ["opaque-source-1"]
    assert result["time_filter"]["date_field"] == "activity_day"
    assert result["diagnostics"]["eligible_message_count"] == 1


def test_relative_today_uses_learning_activity_day_at_a_different_display_midnight(monkeypatch):
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            instant = cls(2026, 11, 7, 0, 0, tzinfo=timezone.utc)
            return instant.astimezone(tz) if tz is not None else instant

    monkeypatch.setattr(retrieval, "datetime", FixedClock)
    store = Store([message(0, "Я назвал корабль Сокол.", day="2026-11-07"),
                   message(1, "Корабль был вчера.", day="2026-11-06")])
    store.get_account_display_timezone = lambda actor: "Pacific/Honolulu"
    store.get_learner_learning_timezone = lambda learner: "Asia/Tomsk"
    result = retrieval.retrieve_conversation_context(store, Runtime(), "child", "account", "Что я рассказал про судно сегодня?")
    assert result["time_filter"]["reference_date"] == "2026-11-07"
    assert result["time_filter"]["learning_timezone"] == "Asia/Tomsk"
    assert result["source_ids"] == ["opaque-source-0"]


def test_teacher_author_role_is_rendered_alongside_literal_user_quote():
    row = message(0, "Я обсудила с родителями план корабля Сокол.", author_role="teacher", actor_role="teacher",
                  original_source_role="student")
    result = retrieval.retrieve_conversation_context(Store([row]), Runtime(), "child", "account", "Что я рассказывала про судно?")
    quote = json.loads(result["system_text"])["messages"][0]
    assert quote["role"] == "user" and quote["author_role"] == "teacher"
    assert quote["text"] == row["text"]
    assert result["sources"][0]["original_source_role"] == "student"
    assert "opaque-source" not in result["system_text"] and "T10:15" not in result["system_text"]


@pytest.mark.parametrize("change", ["actor", "learner", "observation"])
def test_foreign_actor_learner_or_private_observation_fails_before_embedding(change):
    row, runtime = message(0, "Мой корабль Сокол."), Runtime()
    row[{"actor": "actor_id", "learner": "learner_id", "observation": "usage"}[change]] = "foreign"
    with pytest.raises(PermissionError):
        retrieval.retrieve_conversation_context(Store([row]), runtime, "child", "account", "Про судно?")
    assert runtime.calls == []


def test_teacher_own_actual_chat_is_a_self_report_and_current_turn_is_excluded():
    row = message(0, "Мой корабль Сокол.", author_role="teacher", actor_role="teacher")
    assert retrieval.retrieve_conversation_context(Store([row]), Runtime(), "child", "account", "Про судно?")["sources"]
    runtime = Runtime()
    result = retrieval.retrieve_conversation_context(Store([row]), runtime, "child", "account", "Про судно?", exclude_source_id=row["source_id"])
    assert result["text"] == "" and runtime.calls == []


def test_scope_and_model_cache_reuse_sources_but_not_queries():
    store, runtime = Store([message(0, "Корабль Сокол.")]), Runtime()
    retrieval.retrieve_conversation_context(store, runtime, "child", "account", "Про судно?")
    runtime.calls.clear()
    retrieval.retrieve_conversation_context(store, runtime, "child", "account", "Про судно?")
    assert runtime.calls == ["Про судно?"]


def test_invalid_embedding_and_changed_digest_are_rejected():
    runtime = Runtime()
    runtime.embed = lambda texts: [[0.0] * 1024 for text in texts]
    with pytest.raises(ValueError, match="zero norm"):
        retrieval.retrieve_conversation_context(Store([message(0, "Корабль Сокол.")]), runtime, "child", "account", "Про судно?")
    runtime = Runtime()
    identities = iter(["bge-m3@" + "b" * 64, "bge-m3@" + "c" * 64])
    runtime.embedding_model_identity = lambda: next(identities)
    with pytest.raises(ValueError, match="changed"):
        retrieval.retrieve_conversation_context(Store([message(0, "Корабль Сокол.")]), runtime, "child", "account", "Про судно?")


def test_catalog_scores_are_similarities_and_uncertain_labels_do_not_remove_chunks():
    chunks = retrieval.chunk_conversation_messages([message(0, "Нет, я не люблю шахматы.")], "child", "account")
    embedder = retrieval.ConversationEmbedder(Store([]), Runtime(), "child", "account")
    vectors = embedder.source_vectors([chunk["text"] for chunk in chunks])
    labels, classifier = classify_chunks(chunks, vectors, embedder)
    assert classifier["score_kind"] == "cosine_similarity" and "confidence" not in classifier
    assert len(labels) == 1 and labels[0]["labels"] == [] and labels[0]["unknown"] is True
    catalog = summarize_catalog(labels, classifier, "a" * 64)
    assert catalog["unknown_chunk_count"] == 1 and catalog["topic_filter_is_mandatory"] is False
    assert catalog["days"][0]["activity_day"] == "2026-10-01"
