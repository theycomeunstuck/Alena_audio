from copy import deepcopy
from types import SimpleNamespace

import pytest

from local_app import personal_graph_index, personal_graph_retrieval
from local_app.personal_graph_memory import ApplicationModel, PersonalGraphProvider


@pytest.fixture
def identities(monkeypatch):
    current = {
        "chat": {"name": "gemma4:12b", "digest": "a" * 64},
        "embedding": {"name": "bge-m3", "digest": "b" * 64},
    }
    monkeypatch.setattr(personal_graph_index, "resolve_model_identities", lambda model: deepcopy(current))
    return current


@pytest.fixture
def runtime():
    return SimpleNamespace(
        context_tokens=16384,
        settings=SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                 ollama_url="http://127.0.0.1:11434"),
        embed=lambda texts: [[1.0] + [0.0] * 1023 for _ in texts],
    )


def bound_model(runtime, identities, writer=None):
    model = ApplicationModel(runtime, writer)
    model.model_identities = deepcopy(identities)
    return model


@pytest.mark.parametrize("changed", ["chat", "embedding"])
def test_bound_identity_rejects_a_retargeted_model(runtime, identities, changed):
    model = bound_model(runtime, identities)
    model.check_model_identities()
    identities[changed]["digest"] = "c" * 64
    with pytest.raises(ValueError, match="model changed"):
        model.check_model_identities()


@pytest.mark.parametrize("when", ["before", "during"])
def test_embedding_rejects_model_change_before_return(runtime, identities, when):
    calls = []
    def embed(texts):
        calls.append(texts)
        if when == "during":
            identities["embedding"]["digest"] = "c" * 64
        return [[1.0] + [0.0] * 1023 for _ in texts]
    runtime.embed = embed
    model = bound_model(runtime, identities)
    if when == "before":
        identities["embedding"]["digest"] = "c" * 64
    with pytest.raises(ValueError, match="model changed"):
        model.embed_many(["source"])
    assert bool(calls) == (when == "during")


@pytest.mark.parametrize("when", ["before", "during"])
def test_writer_rejects_model_change_before_return(runtime, identities, when):
    calls = []
    def write(messages, purpose, schema):
        calls.append(messages)
        if when == "during":
            identities["chat"]["digest"] = "c" * 64
        return "generated"
    model = bound_model(runtime, identities, write)
    if when == "before":
        identities["chat"]["digest"] = "c" * 64
    with pytest.raises(ValueError, match="model changed"):
        model._post("/api/chat", {"messages": []}, "graphiti_llm")
    assert bool(calls) == (when == "during")


@pytest.mark.parametrize("vector", [[0.0] * 1024, [float("nan")] + [0.0] * 1023,
                                   [float("inf")] + [0.0] * 1023, [1.0] * 1023])
def test_invalid_embedding_is_rejected(runtime, identities, vector):
    runtime.embed = lambda texts: [vector]
    with pytest.raises(ValueError):
        bound_model(runtime, identities).embed_many(["source"])


def test_valid_embedding_is_returned(runtime, identities):
    assert bound_model(runtime, identities).embed_many(["source"])[0] == [1.0] + [0.0] * 1023


@pytest.mark.parametrize("when", ["before_creation", "before_hit", "during_hit"])
def test_cached_source_rejects_retargeted_embedding_model(runtime, identities, monkeypatch, when):
    calls = []
    class Cache:
        def __init__(self, dsn, learner_id, identity):
            calls.append("created")
        def embed_many(self, texts, embed):
            calls.append("hit")
            if when == "during_hit":
                identities["embedding"]["digest"] = "c" * 64
            # A cache hit deliberately does not call the supplied embedder.
            return [[1.0] + [0.0] * 1023 for _ in texts]
    monkeypatch.setattr(personal_graph_retrieval, "SourceEmbeddingCache", Cache)
    model = bound_model(runtime, identities)
    provider = PersonalGraphProvider(SimpleNamespace(dsn="fake"), runtime)
    if when == "before_creation":
        identities["embedding"]["digest"] = "c" * 64
        with pytest.raises(ValueError, match="model changed"):
            provider._source_embedder(model, "learner")
        assert calls == []
        return
    cached = provider._source_embedder(model, "learner")
    if when == "before_hit":
        identities["embedding"]["digest"] = "c" * 64
    with pytest.raises(ValueError, match="model changed"):
        cached(["source"])
    assert calls == (["created", "hit"] if when == "during_hit" else ["created"])


def test_cache_hit_with_bound_identity_succeeds_without_embedding(runtime, identities, monkeypatch):
    class Cache:
        def __init__(self, dsn, learner_id, identity):
            pass
        def embed_many(self, texts, embed):
            return [[1.0] + [0.0] * 1023 for _ in texts]
    monkeypatch.setattr(personal_graph_retrieval, "SourceEmbeddingCache", Cache)
    runtime.embed = lambda texts: pytest.fail("cache hit must not embed")
    cached = PersonalGraphProvider(SimpleNamespace(dsn="fake"), runtime)._source_embedder(
        bound_model(runtime, identities), "learner")
    assert cached(["source"])[0][0] == 1.0
