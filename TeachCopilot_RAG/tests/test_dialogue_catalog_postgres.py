"""Three bounded current-catalog checks in an explicitly isolated PostgreSQL DB."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import os
from types import SimpleNamespace

from psycopg2.extras import Json
import pytest

from local_app import dialogue_catalog
from local_app.dialogue_catalog import DialogueCatalogConflict, get_dialogue_catalog, rebuild_dialogue_catalog
from tests.test_conversation_history_postgres import history_store, historical_turn


pytestmark = pytest.mark.skipif(not os.getenv("TEACHCOPILOT_TEST_DSN"),
                                reason="set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")
_TABLES = ("dialogue_catalog_classifier", "dialogue_catalog_scopes", "dialogue_catalog_chunks")
_JSON_FIELDS = {"classifier", "summary", "source_spans", "topic_scores"}


class FakeBGE:
    """All embedding calls stay inside this process; no GPU/model endpoint exists."""
    settings = SimpleNamespace(embedding_model="bge-m3", ollama_url="http://127.0.0.1:11435")

    def embedding_model_identity(self):
        return "bge-m3@" + "b" * 64

    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def classifier_for(recipe):
    def classify(chunks, vectors, embedder):
        records = [{key: value for key, value in chunk.items() if key != "text"} | {
            "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
            "labels": ["learning"], "topic_scores": {"learning": 0.9}, "unknown": False} for chunk in chunks]
        return records, {"recipe_id": recipe, "method": "synthetic-test-classifier", "score_kind": "cosine_similarity",
                         "model_identity": embedder.identity}
    return classify


def catalog_rows(store):
    with store._connect() as connection, connection.cursor() as cur:
        result = {}
        for table in _TABLES:
            order = ("singleton" if table == _TABLES[0] else "learner_id,actor_id" +
                     (",chunk_id" if table == _TABLES[2] else ""))
            cur.execute("SELECT * FROM teachcopilot." + table + " ORDER BY " + order)
            result[table] = [dict(row) for row in cur.fetchall()]
        return result


def source_digest(store, learners):
    with store._connect() as connection, connection.cursor() as cur:
        cur.execute("""SELECT id,learner_id,content,source_order,created_at,author_role FROM teachcopilot.memory_sources
            WHERE learner_id=ANY(%s) ORDER BY source_order""", (learners,))
        rows = cur.fetchall()
    return len(rows), hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


@pytest.fixture
def catalog_store(history_store):
    store, learners, accounts = history_store
    # A recipe switch is intentionally global. Restore all prior derived rows
    # after each isolated test so unrelated synthetic catalog scopes survive.
    previous = catalog_rows(store)
    learner = learners[0]
    for index, account in enumerate(accounts[:2]):
        session = store.create_learning_session(learner, account["id"], "math.test")
        historical_turn(store, learner, account["id"], session,
                        datetime.now(timezone.utc) - timedelta(days=index + 1))
    private = store.add_memory_source(learner, "math.test", "Private observation", accounts[2]["id"], approved=True)
    try:
        yield store, learners, accounts, private["source_id"]
    finally:
        with store._connect() as connection, connection.cursor() as cur:
            for table in reversed(_TABLES):
                cur.execute("DELETE FROM teachcopilot." + table)
            for table in _TABLES:
                for row in previous[table]:
                    columns = list(row)
                    values = [Json(value) if key in _JSON_FIELDS and value is not None else value
                              for key, value in row.items()]
                    cur.execute("INSERT INTO teachcopilot." + table + "(" + ",".join(columns) + ") VALUES(" +
                                ",".join(["%s"] * len(columns)) + ")", values)
            cur.execute("DELETE FROM teachcopilot.personal_graph_source_embedding_cache WHERE learner_id=ANY(%s)", (learners,))


def test_two_actor_scopes_share_one_current_recipe_and_keep_private_sources_out(catalog_store):
    store, learners, accounts, private_id = catalog_store
    learner, runtime, recipe = learners[0], FakeBGE(), "1" * 64
    before = source_digest(store, learners)
    results = [rebuild_dialogue_catalog(store, runtime, learner, account["id"], classifier=classifier_for(recipe))
               for account in accounts[:2]]
    assert results[0]["generation"] == results[1]["generation"]
    own_ids = []
    for account in accounts[:2]:
        catalog = get_dialogue_catalog(store, learner, account["id"])
        assert catalog["state"] == "ready" and catalog["classifier"]["recipe_id"] == recipe
        own_ids.append({row["source_id"] for row in store.get_own_conversation_messages(learner, account["id"], limit=None)})
    rows = catalog_rows(store)
    assert len(rows[_TABLES[0]]) == 1
    scoped = [row for row in rows[_TABLES[2]] if row["learner_id"] == learner]
    assert len(scoped) == 2 and all(private_id not in row["source_ids"] for row in scoped)
    for row in scoped:
        position = [str(account["id"]) for account in accounts[:2]].index(row["actor_id"])
        assert set(row["source_ids"]) <= own_ids[position]
        assert set(row["source_ids"]).isdisjoint(own_ids[1 - position])
    assert source_digest(store, learners) == before


def test_recipe_switch_replaces_only_derived_labels_and_other_scopes_become_pending(catalog_store):
    store, learners, accounts, _ = catalog_store
    learner, runtime = learners[0], FakeBGE()
    for account in accounts[:2]:
        rebuild_dialogue_catalog(store, runtime, learner, account["id"], classifier=classifier_for("2" * 64))
    before = source_digest(store, learners)
    old = get_dialogue_catalog(store, learner, accounts[0]["id"])
    result = rebuild_dialogue_catalog(store, runtime, learner, accounts[0]["id"], classifier=classifier_for("3" * 64))
    assert result["generation"] == old["generation"] + 1 and result["other_scopes_pending"] is True
    assert get_dialogue_catalog(store, learner, accounts[0]["id"])["state"] == "ready"
    pending = get_dialogue_catalog(store, learner, accounts[1]["id"])
    assert pending["state"] == "pending" and pending["topics"] == []
    rows = catalog_rows(store)
    assert rows[_TABLES[0]][0]["recipe_id"] == "3" * 64
    assert all(row["generation"] == result["generation"] for row in rows[_TABLES[2]])
    assert all(row["actor_id"] == str(accounts[0]["id"]) for row in rows[_TABLES[2]])
    assert source_digest(store, learners) == before


def test_failed_replacement_rolls_back_and_an_older_generation_cannot_replace_current(catalog_store, monkeypatch):
    store, learners, accounts, _ = catalog_store
    learner, actor, runtime = learners[0], accounts[0]["id"], FakeBGE()
    rebuild_dialogue_catalog(store, runtime, learner, actor, classifier=classifier_for("4" * 64))
    previous, sources_before = deepcopy(catalog_rows(store)), source_digest(store, learners)
    real_json = dialogue_catalog.Json

    def fail_after_global_delete(value):
        if isinstance(value, dict) and value.get("recipe_id") == "5" * 64:
            raise RuntimeError("synthetic replacement failure after derived rows were deleted")
        return real_json(value)

    with monkeypatch.context() as patch:
        patch.setattr(dialogue_catalog, "Json", fail_after_global_delete)
        with pytest.raises(RuntimeError, match="synthetic replacement failure"):
            rebuild_dialogue_catalog(store, runtime, learner, actor, classifier=classifier_for("5" * 64))
    assert catalog_rows(store) == previous
    nested_current = []

    def stale_worker(chunks, vectors, embedder):
        rebuild_dialogue_catalog(store, runtime, learner, actor, classifier=classifier_for("6" * 64))
        nested_current.append(catalog_rows(store))
        return classifier_for("7" * 64)(chunks, vectors, embedder)

    with pytest.raises(DialogueCatalogConflict, match="generation changed"):
        rebuild_dialogue_catalog(store, runtime, learner, actor, classifier=stale_worker)
    assert catalog_rows(store) == nested_current[0]
    assert get_dialogue_catalog(store, learner, actor)["classifier"]["recipe_id"] == "6" * 64
    assert source_digest(store, learners) == sources_before
