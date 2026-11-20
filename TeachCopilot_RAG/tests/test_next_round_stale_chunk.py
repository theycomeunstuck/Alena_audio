"""Recall stays scoped to the current source snapshot and date window."""
from datetime import date, datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from psycopg2.extras import Json

from local_app.conversation_retrieval import CHUNK_RECIPE, chunk_conversation_messages
from local_app.unified_memory_recall import recall_own_conversation


def test_recall_never_ranks_a_chunk_with_an_excluded_source(disposable_store):
    store = disposable_store
    learner = "stale-chunk-synthetic"
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card)
               VALUES (%s,'Synthetic',3,%s)""",
            (learner, Json({"learner_id": learner})),
        )
    student = store.create_account(uuid4().hex + "@synthetic.test", "Synthetic", "unused", "student")
    actor = student["id"]
    earlier = store.create_learning_session(learner, actor, "math.test")
    active = store.create_learning_session(learner, actor, "math.test")
    question = "Как назывался корабль Сокол?"
    historical = store.reserve_chat_message(earlier["id"], learner, "historical", question, actor)
    store.complete_chat_message(
        earlier["id"], "historical",
        {"choices": [{"message": {"content": "Корабль назывался Сокол."}}]},
    )
    excluded = store.reserve_chat_message(active["id"], learner, "current", question, actor)

    # Simulate another archive read that begins after the pending turn was
    # persisted and indexes it before this request searches the archive.
    first_search = recall_own_conversation(store, None, learner, actor, question, budget_bytes=8000)
    assert str(excluded["source_id"]) in first_search["retrieved_source_ids"]

    filtered_search = recall_own_conversation(
        store, None, learner, actor, question, budget_bytes=8000,
        exclude_source_id=str(excluded["source_id"]),
    )
    assert str(excluded["source_id"]) not in filtered_search["retrieved_source_ids"]
    assert str(excluded["source_id"]) not in {quote["source_id"] for quote in filtered_search["sources"]}
    assert str(historical["source_id"]) in filtered_search["evidence_ids"]


def test_date_fallback_indexes_the_widened_snapshot(disposable_store, monkeypatch):
    store = disposable_store
    learner = "date-fallback-synthetic"
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card)
               VALUES (%s,'Synthetic',3,%s)""",
            (learner, Json({"learner_id": learner})),
        )
    actor = store.create_account(uuid4().hex + "@synthetic.test", "Synthetic", "unused", "student")["id"]

    def source(source_id, day, text, order):
        return {"source_id": source_id, "source_order": order, "learner_id": learner,
                "actor_id": str(actor), "role": "user", "author_role": "student",
                "text": text, "activity_day": date.fromisoformat(day),
                "created_at": datetime.fromisoformat(day).replace(tzinfo=timezone.utc),
                "learning_session_id": str(uuid4()), "usage": "self_report"}

    messages = [source("older", "2026-09-01", "Корабль назывался Сокол.", 1),
                source("in-range", "2026-10-01", "Абрикосы и корабль.", 2)]
    monkeypatch.setattr(store, "get_own_conversation_messages", lambda *args, **kwargs: messages)

    class Runtime:
        settings = SimpleNamespace(embedding_model="bge-m3", ollama_url="http://127.0.0.1:11434")

        def embedding_model_identity(self):
            return "toy@" + "b" * 64

        def embed(self, texts):
            return [([1.0, 0.0] if "сокол" in text.casefold() else [0.0, 1.0])
                    + [0.0] * 1022 for text in texts]

    runtime = Runtime()
    # A day exists, but has no lexical or vector hit. The older source must be
    # indexed and embedded before the query-derived date can be widened.
    widened = recall_own_conversation(store, runtime, learner, actor, "Сокол 2026-10-01")
    assert widened["diagnostics"]["time_filter_fallback"] is True
    assert set(widened["eligible_source_ids"]) == {"older", "in-range"}
    assert widened["source_ids"] == ["older"]

    # The excluded, higher-scoring row must not consume the only ranked slot.
    in_range_key = chunk_conversation_messages([messages[1]], learner, actor)[0]["chunk_id"]
    search = lambda **kwargs: store.search_conversation_chunks(
        learner, actor, "корабль Сокол", chunk_recipe=CHUNK_RECIPE, limit=1, **kwargs)
    assert search()[0]["source_ids"] == ["older"]
    assert search(candidate_chunk_keys=[in_range_key])[0]["source_ids"] == ["in-range"]
    assert search(candidate_chunk_keys=[]) == []

    # The other fallback has no messages in range at all.
    empty_day = recall_own_conversation(store, runtime, learner, actor, "Сокол 2026-11-01")
    assert empty_day["diagnostics"]["time_filter_fallback"] is True
    assert empty_day["source_ids"] == ["older"]

    # An explicit range stays narrow, even when its indexed day has no hit.
    explicit = recall_own_conversation(
        store, runtime, learner, actor, "Сокол",
        time_filter={"from": "2026-10-01", "through": "2026-10-01"},
    )
    assert explicit["eligible_source_ids"] == ["in-range"]
    assert explicit["source_ids"] == []
    assert explicit["diagnostics"].get("time_filter_fallback") is not True

    matching = recall_own_conversation(store, runtime, learner, actor, "Сокол 2026-09-01")
    assert matching["source_ids"] == ["older"]
    assert matching["eligible_source_ids"] == ["older"]
    assert matching["diagnostics"].get("time_filter_fallback") is not True
