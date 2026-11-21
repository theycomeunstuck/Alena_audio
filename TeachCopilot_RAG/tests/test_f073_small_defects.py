"""Audit F-073: four small latent defects.

1. reindex rewrote an empty material's content before the snapshot check, so
   update_embeddings always saw a "changed" row (StaleMaterialSnapshotError).
2. Store connections had no connect timeout.
3. A completed example at the action limit advertised "switch", which the
   protocol rejects for completed states.
4. The em dash -> en dash rewrite also altered code inside Markdown fences
   and inline code spans.
"""
import local_app.store as store_module
from local_app import autonomous_division as division
from local_app import autonomous_division_store as division_store
from local_app.rag import RagService
from local_app.runtime import normalize_completion_dashes
from tests.test_local_rag import FakeRuntime, FakeStore, Settings


class SnapshotCheckingStore(FakeStore):
    def update_embeddings(self, chunks, embedding_model=None, embedding_dimension=None):
        by_id = {row["id"]: row for row in self.materials}
        for chunk in chunks:
            assert chunk["content"] == by_id[chunk["id"]]["content"], "snapshot content must stay the stored value"
        return super().update_embeddings(chunks, embedding_model, embedding_dimension)


def test_reindex_keeps_the_stored_content_of_an_empty_material():
    store, runtime = SnapshotCheckingStore(), FakeRuntime()
    store.materials = [{"id": "m1", "content": "", "title": "Луч", "description": "Луч имеет начало, но не имеет конца.",
                        "source_file": "geometry.md", "source_version": "sha256:1", "subject_id": "math"}]
    result = RagService(store, runtime, Settings()).reindex()
    assert result["materials"] == 1 and result["embedded"] == 1
    assert "Луч имеет начало" in runtime.embedded[0]
    assert "content:" not in runtime.embedded[0]


def test_reindex_skips_a_material_with_no_text_at_all():
    store, runtime = SnapshotCheckingStore(), FakeRuntime()
    store.materials = [{"id": "m1", "content": "  ", "source_file": "x.md", "source_version": "sha256:1"}]
    try:
        RagService(store, runtime, Settings()).reindex()
    except RuntimeError:
        pass  # the corpus stays not ready – reported, not a snapshot error
    assert runtime.embedded == []


def test_store_connections_have_a_connect_timeout(monkeypatch):
    seen = {}

    class Connection:
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    def fake_connect(dsn, **kwargs):
        seen.update(kwargs)
        return Connection()

    monkeypatch.setattr(store_module.psycopg2, "connect", fake_connect)
    with store_module.connect("postgresql://u@127.0.0.1:5432/db"):
        pass
    assert seen["connect_timeout"] == store_module.CONNECT_TIMEOUT_SECONDS


def test_explicit_connect_timeout_in_the_dsn_is_respected(monkeypatch):
    seen = {}

    class Connection:
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    monkeypatch.setattr(store_module.psycopg2, "connect", lambda dsn, **kwargs: seen.update(kwargs) or Connection())
    with store_module.connect("postgresql://u@127.0.0.1:5432/db?connect_timeout=30"):
        pass
    assert "connect_timeout" not in seen


def test_completed_example_at_the_action_limit_offers_next_not_switch():
    state = {"status": "completed", "actions_used": division.MAX_ACTIONS_PER_EXAMPLE}
    assert division_store.available_actions(state) == ["next"]


def test_running_example_at_the_action_limit_still_offers_switch():
    state = {"status": "active", "actions_used": division.MAX_ACTIONS_PER_EXAMPLE}
    assert division_store.available_actions(state) == ["switch"]


def test_dash_rewrite_leaves_code_untouched():
    text = ("Итак — смотри.\n```python\nprint(\"мало — много\")\n```\n"
            "А ещё `x — y` в строке — и всё.")
    assert normalize_completion_dashes(text) == (
        "Итак – смотри.\n```python\nprint(\"мало — много\")\n```\n"
        "А ещё `x — y` в строке – и всё.")
