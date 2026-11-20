"""Round-3 follow-up: the request's own wording is not historical evidence.

The current message is reserved and persisted before the model generates, so
without an exclusion the recall could deliver the learner's own question as a
"quote" — the query would then look like proof of the very past event it
implies. The unified recall path now excludes the reserved source id (the
shipped fallback already did), while valid earlier repeated questions stay.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from local_app import unified_memory_recall as recall_module
from local_app.unified_memory_recall import recall_own_conversation

pytest.importorskip("psycopg2")


class ScriptedRuntime:
    """Deterministic embedder double for the vector channel."""

    context_tokens = 32768

    def __init__(self):
        from types import SimpleNamespace
        self.settings = SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                        ollama_url="http://127.0.0.1:11434")

    def model_identity(self, model_name=None):
        return "recording-gemma@sha256:" + "c" * 64

    def embedding_model_identity(self):
        return "hash-embedder@sha256:" + "d" * 64

    def embed(self, texts):
        import hashlib
        vectors = []
        for text in (texts if isinstance(texts, list) else [texts]):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [0.0] * 8
            for index, byte in enumerate(digest[:8]):
                vector[index] = byte / 255.0
            vectors.append(vector)
        return vectors

    def check_models(self):
        return {"ready": True, "chat": "scripted", "embedding": "scripted"}


def _row(source_id, text, *, role="user", order=1, session="session-1", day="2026-10-01"):
    return {"source_id": source_id, "source_order": order, "learner_id": "child",
            "actor_id": "account", "role": role,
            "author_role": "student" if role == "user" else "assistant",
            "text": text, "activity_day": date.fromisoformat(day),
            "activity_timezone": "Asia/Tomsk", "event_timezone": "Asia/Tomsk",
            "created_at": datetime.fromisoformat(day + "T10:15:00+00:00"),
            "learning_session_id": session, "topic_id": "general",
            "delivery_status": "completed",
            "usage": "self_report" if role == "user" else "context_only"}


class FakeOwnSourceStore:
    """read_own_messages' contract over an in-memory archive."""

    def __init__(self, rows):
        self.rows = rows
        self.excluded = []

    def get_own_conversation_messages(self, learner_id, actor_id, *, limit=None,
                                      active_day_limit=None, exclude_source_id=None):
        self.excluded.append(exclude_source_id)
        rows = [row for row in self.rows
                if exclude_source_id is None or row["source_id"] != exclude_source_id]
        return rows[:limit] if limit else rows

    def upsert_conversation_chunks(self, *args, **kwargs):
        return None

    def chunk_embedding_status(self, *args, **kwargs):
        return []

    def store_chunk_embedding(self, *args, **kwargs):
        return None

    def memory_source_correction_refs(self, learner_id, source_ids):
        return {}

    def search_conversation_chunks(self, learner_id, actor_id, query, *, chunk_recipe,
                                   embedding_identity=None, query_vector=None, limit=12,
                                   session_id=None, time_filter=None, rrf_k=60,
                                   min_similarity=0.20, candidate_chunk_keys=None,
                                   exclude_chunk_keys=None):
        # The real store searches chunk rows built from the filtered archive,
        # so whatever the source reader excluded never reaches the search.
        scored = []
        words = [word for word in query.casefold().split() if len(word) > 3]
        excluded = self.excluded[-1] if self.excluded else None
        for row in self.rows:
            if excluded is not None and row["source_id"] == excluded:
                continue
            score = sum(1 for word in words if word in str(row.get("text", "")).casefold())
            if score or row["source_id"] == "current":
                scored.append((score + (2 if row["source_id"] == "current" else 0),
                               row["source_id"]))
        scored.sort(reverse=True)
        return [{"chunk_key": f"chunk-{index}", "source_ids": [source_id], "spans": [],
                 "score": float(score)} for index, (score, source_id) in enumerate(scored[:limit])]

    def conversation_chunk_texts(self, learner_id, actor_id, chunk_recipe, chunk_keys):
        return {}

    def search_lesson_plan_context(self, learner_id, query, limit=3):
        return []

    def conversation_chunk_embeddings(self, learner_id, actor_id, chunk_recipe, chunk_keys):
        return {}


def test_current_question_is_excluded_but_earlier_repeats_stay():
    earlier = _row("earlier-question", "Какая собака была у семьи в прошлом году?",
                   order=1, day="2026-09-01")
    earlier_answer = _row("earlier-answer", "У нас был Байтик, австралийская овчарка.",
                          role="assistant", order=2, day="2026-09-01")
    earlier["paired_source_id"] = "earlier-answer"
    earlier_answer["pair_source_id"] = "earlier-question"
    current = _row("current", "Какая собака была у семьи в прошлом году?",
                   order=3, session="session-2", day="2026-11-01")
    store = FakeOwnSourceStore([earlier, earlier_answer, current])
    result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                     "Какая собака была у семьи в прошлом году?",
                                     exclude_source_id="current")
    quote_ids = {str(quote.get("source_id")) for quote in result.get("sources", [])}
    assert "current" not in quote_ids, "the request's own wording must not be delivered"
    assert "earlier-question" in quote_ids, "a valid earlier repeated question stays"
    assert "earlier-answer" in quote_ids
    assert store.excluded == ["current"], "the exclusion reached the source reader"


def test_without_exclusion_the_current_text_would_rank_first():
    # Demonstrates the pre-fix shape: the identical current text outranks the
    # real evidence, so the exclusion is what keeps the answer honest.
    earlier = _row("earlier-question", "Какая собака была у семьи в прошлом году?",
                   order=1, day="2026-09-01")
    current = _row("current", "Какая собака была у семьи в прошлом году?",
                   order=3, session="session-2", day="2026-11-01")
    store = FakeOwnSourceStore([earlier, current])
    result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                     "Какая собака была у семьи в прошлом году?",
                                     exclude_source_id="current")
    quote_ids = {str(quote.get("source_id")) for quote in result.get("sources", [])}
    assert quote_ids == {"earlier-question"}


class TestProvenanceStages:
    """Item 4: eligible/retrieved/selected from actual pipeline data."""

    def test_recall_exposes_eligible_and_retrieved_stages(self):
        rows = [_row("earlier-1", "Рисовал корабль Сокол в прошлом году.", order=1, day="2026-09-01"),
                _row("earlier-2", "У нас был Байтик, австралийская овчарка.", role="assistant",
                     order=2, day="2026-09-01"),
                _row("current", "Какой корабль я рисовал?", order=3, session="session-2",
                     day="2026-11-01")]
        store = FakeOwnSourceStore(rows)
        store.excluded.append("current")
        result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                         "Какой корабль я рисовал?",
                                         exclude_source_id="current")
        assert set(result["eligible_source_ids"]) == {"earlier-1", "earlier-2"}, \
            "the authorized candidate pool is the actual filtered archive"
        assert set(result["retrieved_source_ids"]) <= set(result["eligible_source_ids"]), \
            "what the search returned cannot exceed the authorized pool"
        delivered = {str(quote.get("source_id")) for quote in result.get("sources", [])}
        assert delivered <= set(result["retrieved_source_ids"]), \
            "delivered quotes are a subset of retrieved"

    def test_budget_pressure_omits_retrieved_sources_honestly(self):
        # A small budget: the pipeline retrieves candidates but the bounded
        # selection must omit some of them, and the delivered set stays a
        # subset of retrieved — never claiming a quote it could not fit.
        # (1700 fits roughly two quotes plus the evidence notice; the six
        # notes need ~2600 bytes complete.)
        rows = [_row(f"source-{index}", f"Записка номер {index} про корабль Сокол и море.",
                     order=index + 1, day="2026-09-01") for index in range(6)]
        store = FakeOwnSourceStore(rows)
        result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                         "корабль Сокол море",
                                         budget_bytes=1700)
        retrieved = set(result["retrieved_source_ids"])
        delivered = {str(quote.get("source_id")) for quote in result.get("sources", [])}
        assert retrieved and delivered, "both stages must carry actual data"
        assert delivered <= retrieved
        assert len(delivered) < len(retrieved), "budget pressure must omit sources"
