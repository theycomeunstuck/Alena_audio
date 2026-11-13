"""Unified hybrid source retrieval and its pgvector cache on disposable PostgreSQL.

Real PostgreSQL plus pgvector only: every embedder below is a deterministic
fake, and no Neo4j, Ollama or graph service is contacted.
"""
from types import SimpleNamespace
from uuid import uuid4

import pytest

from local_app import personal_graph_retrieval as retrieval
from local_app.memory_store import get_learner_memory_snapshot
from local_app.personal_graph_memory import source_lessons
from local_app.personal_memory import PersonalMemoryProvider
from tests.test_memory_api_flow import flow_store  # noqa: F401


LEARNER = "demo-nika-04"
TOPIC = "math.g4.fractions.numerator_denominator"
VECTOR_DIMENSIONS = 1024


def unit_vector(index=0):
    values = [0.0] * VECTOR_DIMENSIONS
    values[index] = 1.0
    return values


def keyword_index(text):
    """Deterministic basis choice: "геометрия" wins over "дробь" on purpose."""
    if "геометрия" in text:
        return 1
    if "дробь" in text:
        return 0
    if "алгебра" in text:
        return 2
    return 3


def keyword_embed_many(calls):
    def embed(texts):
        calls.append(list(texts))
        return [unit_vector(keyword_index(text)) for text in texts]
    return embed


class RecordingModel:
    """Minimal SourceModel stand-in: deterministic vectors and counted checks."""

    context = 8192
    model_identities = {"chat": {"name": "gemma4:12b", "digest": "a" * 64},
                        "embedding": {"name": "bge-m3", "digest": "sha256:" + "b" * 64}}

    def __init__(self):
        self.identity_checks = 0
        self.embedded = []

    def check_model_identities(self):
        self.identity_checks += 1

    def embed_many(self, texts):
        self.embedded.append(list(texts))
        return [unit_vector(keyword_index(text)) for text in texts]


def canonical_snapshot(store, learner):
    snapshot = get_learner_memory_snapshot(store, learner, None, include_personal_graph_snapshot=True)
    return {**snapshot, "sources": snapshot["personal_graph_sources"]}


def drop_cache_rows(store, learners):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.personal_graph_source_embedding_cache WHERE learner_id = ANY(%s)",
                    ([str(value) for value in learners],))


def test_seeded_snapshot_rows_drive_a_cached_embedding_roundtrip(flow_store):
    store, learner = flow_store, LEARNER
    first = store.add_memory_source(learner, TOPIC, "Ученик спросил про числитель дроби.", "teacher",
                                    approved=True, observed_on="2026-02-10")
    second = store.add_memory_source(learner, TOPIC, "Ученик сравнил дроби на числовой прямой.", "teacher",
                                     approved=True, observed_on="2026-02-11")
    snapshot = get_learner_memory_snapshot(store, learner, None, include_personal_graph_snapshot=True)
    seeded = {row["source_id"] for row in snapshot["personal_graph_sources"]}
    assert {first["id"], second["id"]} <= seeded
    texts = [row["text"] for row in snapshot["personal_graph_sources"]]
    assert texts == [first["content"], second["content"]]

    identity = "cache-model-" + uuid4().hex
    calls = []
    cache = retrieval.SourceEmbeddingCache(store.dsn, learner, identity)
    try:
        vectors = cache.embed_many(texts, keyword_embed_many(calls))
        assert vectors == [unit_vector(keyword_index(text)) for text in texts]
        assert calls == [texts]

        # The exact same texts are served from PostgreSQL without re-invocation.
        repeated = cache.embed_many(texts, keyword_embed_many(calls))
        assert repeated == vectors
        assert calls == [texts]

        # A changed source text is a new exact text: only it is embedded again.
        extra = store.add_memory_source(learner, TOPIC, "Ученик исправил ответ про знаменатель дроби.",
                                        "teacher", approved=True, observed_on="2026-02-12")
        changed = [row["text"] for row in canonical_snapshot(store, learner)["sources"]]
        assert len(changed) == 3 and extra["content"] in changed
        cache.embed_many(changed, keyword_embed_many(calls))
        assert calls == [texts, [extra["content"]]]

        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding))
                FROM teachcopilot.personal_graph_source_embedding_cache WHERE learner_id=%s""", (learner,))
            assert tuple(cur.fetchone().values()) == (3, VECTOR_DIMENSIONS, VECTOR_DIMENSIONS)
    finally:
        drop_cache_rows(store, [learner])


def test_rank_source_candidates_modes_on_real_pg_seeded_lessons(flow_store):
    store, learner = flow_store, LEARNER
    seeded = {}
    for observed_on, text in (("2026-01-10", "алгебра уравнение"),
                              ("2026-01-15", "дробь числитель"),
                              ("2026-01-20", "дробь знаменатель"),
                              ("2026-01-25", "геометрия дробь смешанный пример")):
        seeded[observed_on] = store.add_memory_source(learner, TOPIC, text, "teacher",
                                                      approved=True, observed_on=observed_on)["id"]
    lessons, source_map = source_lessons(canonical_snapshot(store, learner), learner)
    allowed = set(source_map)
    assert len(source_map) == 4
    newest = [seeded["2026-01-25"], seeded["2026-01-20"], seeded["2026-01-15"], seeded["2026-01-10"]]

    calls = []
    embed = keyword_embed_many(calls)

    def forbid(_texts):
        raise AssertionError("non-vector mode invoked the embedder")

    chronological = retrieval.rank_source_candidates("занятие", lessons, allowed,
                                                      mode="chronological", embed_many=forbid)
    assert chronological["ranked_source_ids"] == newest
    lexical = retrieval.rank_source_candidates("дробь", lessons, allowed,
                                               mode="lexical", embed_many=forbid)
    assert lexical["ranked_source_ids"] == newest[:3]

    vector = retrieval.rank_source_candidates("дробь", lessons, allowed, mode="vector",
                                              embed_many=embed, limit=4)
    # The newest "дробь" lesson carries a "геометрия" vector, so the vector
    # channel ranks the exact-match older lessons first.
    assert vector["ranked_source_ids"] == [seeded["2026-01-20"], seeded["2026-01-15"],
                                           seeded["2026-01-25"], seeded["2026-01-10"]]
    assert calls[0] == ["дробь"] and len(calls[1]) == 4

    hybrid = retrieval.rank_source_candidates("дробь", lessons, allowed, mode="hybrid",
                                              embed_many=embed, limit=4)
    # Round-robin interleaves the channels: the lexical winner takes the second
    # slot even though the vector channel placed it third.
    assert hybrid["ranked_source_ids"] == [seeded["2026-01-20"], seeded["2026-01-25"],
                                           seeded["2026-01-15"], seeded["2026-01-10"]]
    assert hybrid["channels"]["vector"] == vector["ranked_source_ids"]
    assert hybrid["channels"]["lexical"] == lexical["ranked_source_ids"]
    assert calls[0] == ["дробь"] and len(calls[1]) == 4

    for mode, expected in (("chronological", newest[:2]), ("lexical", newest[:2]),
                           ("vector", vector["ranked_source_ids"][:2]),
                           ("hybrid", hybrid["ranked_source_ids"][:2])):
        limited = retrieval.rank_source_candidates("дробь", lessons, allowed, mode=mode,
                                                   embed_many=embed, limit=2)
        assert limited["ranked_source_ids"] == expected, mode


GRAPH_CHANNEL_DEGRADE_REASONS = {"no_ready_index", "password_file_unavailable"}


def documented_graph_channel_degrade(reason):
    """The channel records why it degraded; the exact reason may vary by host."""
    return reason in GRAPH_CHANNEL_DEGRADE_REASONS or str(reason).startswith("config_fingerprint:")


def test_graph_channel_degrades_to_plain_hybrid_without_neo4j(flow_store, tmp_path):
    store, learner = flow_store, LEARNER
    allowed_source = store.add_memory_source(learner, TOPIC, "Ученик спросил про числитель дроби.",
                                             "teacher", approved=True, observed_on="2026-02-10")
    other_source = store.add_memory_source(learner, TOPIC, "Ученик повторял геометрию.", "teacher",
                                           approved=True, observed_on="2026-02-11")
    snapshot = canonical_snapshot(store, learner)
    lessons, source_map = source_lessons(snapshot, learner)
    allowed = set(source_map)
    question = "дробь"
    direct = retrieval.rank_source_candidates(question, lessons, allowed, mode="hybrid",
                                              embed_many=RecordingModel().embed_many,
                                              limit=len(allowed))

    missing_password = tmp_path / "absent-personal-memory-password.txt"
    enabled = SimpleNamespace(settings=SimpleNamespace(
        personal_memory_retrieval_mode="hybrid", memory_graph_channel=True,
        personal_graph_password_file=str(missing_password)), context_tokens=8192)
    model = RecordingModel()
    try:
        degraded = PersonalMemoryProvider(store, enabled)._retrieve(
            snapshot, learner, TOPIC, question, allowed, model, "hybrid")
        channel = degraded["retrieval_diagnostics"]["graph_channel"]
        assert channel["source_ids"] is None and channel["run_id"] is None
        assert documented_graph_channel_degrade(channel["reason"])
        assert degraded["retrieval_diagnostics"]["mode"] == "hybrid"
        assert degraded["retrieval_diagnostics"]["channels"] == direct["channels"]
        assert degraded["retrieval_diagnostics"]["selection_order"] == direct["ranked_source_ids"]
        assert set(degraded["evidence_ids"]) == set(direct["ranked_source_ids"])
        assert "числитель дроби" in degraded["text"]

        # With a password file present the channel may reach the registry and
        # must still degrade cleanly without Neo4j and without building anything.
        password_file = tmp_path / "personal-memory-password.txt"
        password_file.write_text("synthetic loopback password", encoding="utf-8")
        prepared = SimpleNamespace(settings=SimpleNamespace(
            personal_memory_retrieval_mode="hybrid", memory_graph_channel=True,
            personal_graph_password_file=str(password_file)), context_tokens=8192)
        deep = PersonalMemoryProvider(store, prepared)._retrieve(
            snapshot, learner, TOPIC, question, allowed, RecordingModel(), "hybrid")
        deep_channel = deep["retrieval_diagnostics"]["graph_channel"]
        assert deep_channel["source_ids"] is None and deep_channel["run_id"] is None
        assert documented_graph_channel_degrade(deep_channel["reason"])
        assert deep["retrieval_diagnostics"]["channels"] == direct["channels"]
        assert deep["retrieval_diagnostics"]["selection_order"] == direct["ranked_source_ids"]
        assert allowed_source["id"] in deep["evidence_ids"] and other_source["id"] in deep["evidence_ids"]
    finally:
        drop_cache_rows(store, [learner])


def test_cache_rows_and_retrieval_are_isolated_per_learner(flow_store):
    store = flow_store
    learner_a = "hybrid-isolated-a-" + uuid4().hex[:8]
    learner_b = "hybrid-isolated-b-" + uuid4().hex[:8]
    from psycopg2.extras import Json
    with store._connect() as conn, conn.cursor() as cur:
        for learner in (learner_a, learner_b):
            cur.execute("""INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card)
                VALUES (%s,'Synthetic',3,%s)""", (learner, Json({"learner_id": learner})))
    shared_text = "Общая проверенная заметка про дробь."
    try:
        source_a = store.add_memory_source(learner_a, TOPIC, shared_text, "teacher",
                                           approved=True, observed_on="2026-03-05")["id"]
        source_b = store.add_memory_source(learner_b, TOPIC, shared_text, "teacher",
                                           approved=True, observed_on="2026-03-06")["id"]
        unique_b = store.add_memory_source(learner_b, TOPIC, "Отдельная заметка ученика B про дробь.",
                                           "teacher", approved=True, observed_on="2026-03-01")["id"]

        # The embedding cache is scoped by learner: identical text and identical
        # model identity still cost one embedder call per learner.
        identity = "isolated-model-" + uuid4().hex
        calls = []
        embed = keyword_embed_many(calls)
        cache_a = retrieval.SourceEmbeddingCache(store.dsn, learner_a, identity)
        cache_b = retrieval.SourceEmbeddingCache(store.dsn, learner_b, identity)
        assert cache_a.embed_many([shared_text], embed) == cache_b.embed_many([shared_text], embed)
        assert calls == [[shared_text], [shared_text]]
        assert cache_a.embed_many([shared_text], embed) == [unit_vector(keyword_index(shared_text))]
        assert len(calls) == 2  # the same learner still hits its own cache row
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT learner_id, count(*) FROM teachcopilot.personal_graph_source_embedding_cache
                WHERE learner_id = ANY(%s) GROUP BY learner_id ORDER BY learner_id""", ([learner_a, learner_b],))
            assert [tuple(row.values()) for row in cur.fetchall()] == [(learner_a, 1), (learner_b, 1)]

        # Retrieval never leaves the allowed learner scope, even when lessons
        # from both learners are visible to the ranking call.
        lessons_a, _ = source_lessons(canonical_snapshot(store, learner_a), learner_a)
        lessons_b, _ = source_lessons(canonical_snapshot(store, learner_b), learner_b)
        ranked = retrieval.rank_source_candidates("дробь", lessons_a + lessons_b, {source_a},
                                                  mode="hybrid", embed_many=embed)
        assert ranked["ranked_source_ids"] == [source_a]
        scoped_limited = retrieval.rank_source_candidates("дробь", lessons_a + lessons_b,
                                                          {source_a, unique_b}, mode="lexical")
        # Equal lexical overlap breaks the tie chronologically (newest first),
        # and the unallowed learner B source never surfaces.
        assert scoped_limited["ranked_source_ids"] == [source_a, unique_b]
        assert source_b not in scoped_limited["ranked_source_ids"]

        # The provider physically excludes sources outside the approved grant.
        disabled = SimpleNamespace(settings=SimpleNamespace(
            personal_memory_retrieval_mode="hybrid", memory_graph_channel=False), context_tokens=8192)
        context = PersonalMemoryProvider(store, disabled)._retrieve(
            canonical_snapshot(store, learner_a), learner_a, TOPIC, "дробь",
            {source_a}, RecordingModel(), "hybrid")
        assert context["retrieval_diagnostics"]["graph_channel"]["reason"] == "channel_disabled"
        assert set(context["evidence_ids"]) <= {source_a}
        assert unique_b not in context["evidence_ids"] and source_b not in context["evidence_ids"]
        assert learner_b not in context["text"]
    finally:
        drop_cache_rows(store, [learner_a, learner_b])
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM teachcopilot.memory_sources WHERE learner_id = ANY(%s)",
                        ([learner_a, learner_b],))
            cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id = ANY(%s)",
                        ([learner_a, learner_b],))
