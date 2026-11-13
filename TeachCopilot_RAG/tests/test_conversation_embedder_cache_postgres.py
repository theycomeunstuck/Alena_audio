"""Item 10 through the real persistent embedding cache: one unembeddable text must not poison the learner."""
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from psycopg2.extras import Json

from local_app import conversation_retrieval as retrieval
from local_app.store import Store

DSN = os.getenv("TEACHCOPILOT_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")


class PoisonRuntime:
    settings = SimpleNamespace(embedding_model="bge-m3", ollama_url="http://127.0.0.1:11435")

    def __init__(self):
        self.calls = []

    def embedding_model_identity(self):
        return "bge-m3@" + "c" * 64

    def embed(self, texts):
        self.calls.append(list(texts))
        if any("POISON" in text for text in texts):
            raise RuntimeError("Ollama /api/embed returned HTTP 500: unsupported value: NaN")
        return [[1.0, 0.0] + [0.0] * 1022 for _ in texts]


@pytest.fixture
def learner_store():
    store = Store(DSN)
    store.migrate()
    learner = "embed-cache-" + uuid4().hex
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,'Synthetic',3,%s)",
                    (learner, Json({"learner_id": learner})))
    yield store, learner
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.personal_graph_source_embedding_cache WHERE learner_id=%s", (learner,))
        cur.execute("DELETE FROM teachcopilot.learners WHERE learner_id=%s", (learner,))


def cached_rows(store, learner):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM teachcopilot.personal_graph_source_embedding_cache WHERE learner_id=%s",
                    (learner,))
        return cur.fetchone()["n"]


def test_unembeddable_text_is_isolated_and_healthy_texts_are_cached_persistently(learner_store):
    store, learner = learner_store
    runtime = PoisonRuntime()
    embedder = retrieval.ConversationEmbedder(store, runtime, learner, "account")
    assert embedder.cache is not None                      # the real persistent cache path, not the in-memory memo
    texts = ["Мой корабль Сокол.", "POISON Мой корабль Ястреб.", "Мы считали яблоки."]
    vectors = embedder.source_vectors(texts)
    assert vectors[0] is not None and vectors[1] is None and vectors[2] is not None
    assert cached_rows(store, learner) == 2                # the two healthy texts persisted, the poison text did not

    runtime.calls.clear()
    again = retrieval.ConversationEmbedder(store, runtime, learner, "account").source_vectors(texts)
    assert again[1] is None and again[0] is not None and again[2] is not None
    assert runtime.calls and all(any("POISON" in text for text in call) for call in runtime.calls)  # only the bad text retried


def test_real_outage_through_the_persistent_cache_is_still_raised(learner_store):
    store, learner = learner_store
    runtime = PoisonRuntime()
    runtime.embed = lambda texts: (_ for _ in ()).throw(RuntimeError("connection refused"))
    embedder = retrieval.ConversationEmbedder(store, runtime, learner, "account")
    with pytest.raises(RuntimeError, match="connection refused"):
        embedder.source_vectors(["один", "два", "три", "четыре", "пять"])
    assert cached_rows(store, learner) == 0
