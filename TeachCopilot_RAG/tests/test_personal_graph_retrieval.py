import math
from uuid import uuid4

import pytest

from local_app import personal_graph_retrieval as retrieval
from tests.test_memory_api_flow import flow_store  # noqa: F401


def vector(index=0):
    values = [0.0] * 1024
    values[index] = 1.0
    return values


def lessons():
    return [
        {"lesson_id": "later", "observed_on": "2025-02-01", "turns": [
            {"turn_id": "s2", "role": "student", "text": "algebra fraction"},
            {"turn_id": "a2", "role": "assistant", "anchor_source_id": "s2", "text": "fraction explanation"},
            {"turn_id": "a_other", "role": "assistant", "anchor_source_id": "s1", "text": "secret needle"},
        ]},
        {"lesson_id": "earlier", "observed_on": "2025-01-01", "turns": [
            {"turn_id": "s1", "role": "teacher", "text": "geometry triangle"},
            {"turn_id": "a1", "role": "assistant", "anchor_source_id": "s1", "text": "triangle explanation"},
            {"turn_id": "unapproved", "role": "student", "text": "needle"},
        ]},
    ]


def test_chronological_lexical_modes_do_not_call_embedder_and_anchor_context_is_scoped():
    def forbidden(_):
        raise AssertionError("non-vector mode invoked embedder")

    chrono = retrieval.rank_source_candidates("triangle", lessons(), {"s1", "s2"},
                                              mode="chronological", embed_many=forbidden)
    lexical = retrieval.rank_source_candidates("triangle", lessons(), {"s1", "s2"},
                                               mode="lexical", embed_many=forbidden)
    assert chrono["ranked_source_ids"] == ["s2", "s1"]
    assert lexical["ranked_source_ids"] == ["s1"]
    # The word exists only in assistant context anchored to an unapproved ID.
    needle = retrieval.rank_source_candidates("needle", lessons(), {"s1", "s2"}, mode="lexical")
    assert needle["ranked_source_ids"] == []


@pytest.mark.parametrize("role", ["teacher", "student"])
def test_chronological_prefers_latest_same_day_lesson_and_turn(role):
    # Lessons are in source order, oldest to newest, as in full-archive retrieval.
    same_day = [
        {"lesson_id": "first", "observed_on": "2025-01-01", "turns": [
            {"turn_id": "old", "role": role, "text": "one"}]},
        {"lesson_id": "second", "observed_on": "2025-01-01", "turns": [
            {"turn_id": "new-first", "role": role, "text": "two"},
            {"turn_id": "new-last", "role": role, "text": "three"},
            {"turn_id": "assistant", "role": "assistant", "text": "not evidence"}]},
    ]
    allowed = {"old", "new-first", "new-last", "assistant"}
    result = retrieval.rank_source_candidates("nothing", same_day, allowed,
                                              mode="chronological", limit=1)
    assert result["ranked_source_ids"] == ["new-last"]
    assert result["channels"]["chronological"] == ["new-last", "new-first", "old"]
    scoped = retrieval.rank_source_candidates("nothing", same_day, {"old", "new-first"},
                                              mode="chronological", limit=1)
    assert scoped["ranked_source_ids"] == ["new-first"]


def test_vector_and_hybrid_rank_only_allowed_sources_and_graph_hits_are_explicit():
    embed_calls = []

    def embed_many(texts):
        embed_calls.append(list(texts))
        return [vector(0) if "geometry" in text or "triangle" in text else vector(1)
                for text in texts]

    hybrid = retrieval.rank_source_candidates("triangle", lessons(), {"s1", "s2"},
                                              mode="hybrid", embed_many=embed_many,
                                              limit=2)
    assert set(hybrid["ranked_source_ids"]) <= {"s1", "s2"}
    assert "unapproved" not in hybrid["ranked_source_ids"]
    assert all("needle" not in text for call in embed_calls for text in call[1:])

    with pytest.raises(ValueError, match="explicit graph"):
        retrieval.rank_source_candidates("triangle", lessons(), {"s1", "s2"}, mode="graph_hybrid",
                                         embed_many=embed_many)
    graph = retrieval.rank_source_candidates("triangle", lessons(), {"s1", "s2"},
                                             mode="graph_hybrid", embed_many=embed_many,
                                             graph_hit_source_ids=["unapproved", "s2", "s1"])
    assert graph["channels"]["graph"] == ["s2", "s1"]
    assert set(graph["ranked_source_ids"]) <= {"s1", "s2"}


def test_assistant_context_changes_source_embedding_text_but_unanchored_context_does_not():
    first = retrieval.rank_source_candidates("triangle", lessons(), {"s1"}, mode="lexical")
    changed_lessons = lessons()
    changed_lessons[1]["turns"][1]["text"] = "new triangle explanation"
    changed = retrieval.rank_source_candidates("triangle", changed_lessons, {"s1"}, mode="lexical")
    assert first["ranked_source_ids"] == ["s1"]
    assert changed["ranked_source_ids"] == ["s1"]
    assert "secret needle" not in str(changed["channels"])


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self.rows = []

    def execute(self, sql, params):
        sql = " ".join(sql.split())
        if sql.startswith("SELECT text_sha256"):
            learner, model, hashes = params
            self.rows = [(text_hash, value) for (child, identity, text_hash), value in self.store.items()
                         if child == learner and identity == model and text_hash in hashes]
        elif sql.startswith("DELETE"):
            learner, model, text_hash = params
            self.store.pop((learner, model, text_hash), None)
        elif sql.startswith("INSERT"):
            learner, model, text_hash, literal = params
            self.store[(learner, model, text_hash)] = literal
        else:
            raise AssertionError("Unexpected cache SQL: " + sql)

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeConnection:
    def __init__(self, store):
        self.store = store

    def cursor(self):
        return FakeCursor(self.store)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def install_fake_db(monkeypatch):
    store = {}
    monkeypatch.setattr(retrieval.psycopg2, "connect", lambda *args, **kwargs: FakeConnection(store))
    return store


def test_source_embedding_cache_reuses_exact_text_and_isolates_model_and_child(monkeypatch):
    install_fake_db(monkeypatch)
    cache = retrieval.SourceEmbeddingCache("postgresql://unused", "child-1", "model-a")
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return [vector(3) for _ in texts]

    first = cache.embed_many(["source text"], embed)
    second = cache.embed_many(["source text"], embed)
    assert first == second
    assert calls == [["source text"]]
    assert cache.embed_many(["changed source text"], embed) == [vector(3)]
    retrieval.SourceEmbeddingCache("postgresql://unused", "child-1", "model-b").embed_many(
        ["source text"], embed)
    retrieval.SourceEmbeddingCache("postgresql://unused", "child-2", "model-a").embed_many(
        ["source text"], embed)
    assert len(calls) == 4


def test_cache_uses_changed_anchored_context_as_new_exact_text(monkeypatch):
    install_fake_db(monkeypatch)
    cache = retrieval.SourceEmbeddingCache("postgresql://unused", "child", "model")
    calls = []

    def embed(texts):
        calls.append(texts)
        return [vector() for _ in texts]

    cache.embed_many(["student: x\nassistant context: first"], embed)
    cache.embed_many(["student: x\nassistant context: changed"], embed)
    assert len(calls) == 2


def test_cache_rejects_nonfinite_vectors_and_reembeds_invalid_cached_rows(monkeypatch):
    store = install_fake_db(monkeypatch)
    cache = retrieval.SourceEmbeddingCache("postgresql://unused", "child", "model")
    with pytest.raises(ValueError, match="non-finite"):
        cache.embed_many(["bad"], lambda texts: [[math.nan] + [0.0] * 1023])
    assert not store

    text_hash = cache._text_hash("stale")
    identity = cache.model_identity_digest
    store[("child", identity, text_hash)] = "[NaN," + ",".join(["0"] * 1023) + "]"
    calls = []

    def embed(texts):
        calls.append(texts)
        return [vector(8) for _ in texts]

    assert cache.embed_many(["stale"], embed) == [vector(8)]
    assert calls == [["stale"]]


def test_cache_rejects_zero_vectors_and_reembeds_zero_cached_rows(monkeypatch):
    store = install_fake_db(monkeypatch)
    cache = retrieval.SourceEmbeddingCache("postgresql://unused", "child", "model")
    with pytest.raises(ValueError, match="zero norm"):
        cache.embed_many(["zero"], lambda texts: [[0.0] * 1024])

    text_hash = cache._text_hash("stale zero")
    store[("child", cache.model_identity_digest, text_hash)] = "[" + ",".join(["0"] * 1024) + "]"
    assert cache.embed_many(["stale zero"], lambda texts: [vector(7)]) == [vector(7)]


def test_source_embedding_cache_roundtrips_in_postgres_and_isolates_scopes(flow_store):
    """Exercise migration 017 and pgvector persistence against the disposable flow DB."""
    store = flow_store
    learner = "cache-test-" + uuid4().hex
    model = "model-" + uuid4().hex
    texts = ["exact authorized source text", "source text with changed anchored context"]
    calls = []

    def embed(text_batch):
        calls.append(list(text_batch))
        return [vector(4) for _ in text_batch]

    cache = retrieval.SourceEmbeddingCache(store.dsn, learner, model)
    try:
        first = cache.embed_many([texts[0]], embed)
        repeated = cache.embed_many([texts[0]], embed)
        changed = cache.embed_many([texts[1]], embed)
        other_model = retrieval.SourceEmbeddingCache(store.dsn, learner, model + "-changed")
        other_model.embed_many([texts[0]], embed)
        other_learner = retrieval.SourceEmbeddingCache(store.dsn, learner + "-other", model)
        other_learner.embed_many([texts[0]], embed)
        assert first == repeated == [vector(4)]
        assert changed == [vector(4)]
        assert calls == [[texts[0]], [texts[1]], [texts[0]], [texts[0]]]

        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding))
                FROM teachcopilot.personal_graph_source_embedding_cache
                WHERE learner_id=%s""", (learner,))
            assert tuple(cur.fetchone().values()) == (3, 1024, 1024)

        # Rejected numeric payloads do not produce a persistent cache entry.
        with pytest.raises(ValueError, match="non-finite"):
            cache.embed_many(["invalid numeric payload"],
                             lambda batch: [[float("inf")] + [0.0] * 1023])
        with pytest.raises(ValueError, match="zero norm"):
            cache.embed_many(["zero numeric payload"], lambda batch: [[0.0] * 1024])
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT count(*) FROM teachcopilot.personal_graph_source_embedding_cache
                WHERE learner_id=%s AND text_sha256=ANY(%s)""",
                        (learner, [cache._text_hash("invalid numeric payload"),
                                   cache._text_hash("zero numeric payload")]))
            assert next(iter(cur.fetchone().values())) == 0
    finally:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM teachcopilot.personal_graph_source_embedding_cache WHERE learner_id = ANY(%s)",
                        ([learner, learner + "-other"],))
