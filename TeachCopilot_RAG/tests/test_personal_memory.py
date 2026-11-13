"""Source-only production memory contracts, without services or model calls."""
from copy import deepcopy
from contextlib import contextmanager
import builtins
from types import SimpleNamespace

import pytest

from local_app import personal_memory
from local_app.config import Settings
from local_app.personal_graph_contract import personal_graph_snapshot_fingerprint
from local_app.personal_memory import (PersonalMemoryProvider, SourceModel, approved_personal_memory_version,
                                       select_ranked_source_context)
from local_app.personal_memory_contract import (PERSONAL_MEMORY_CONTEXT_PROTOCOL, PersonalMemoryStale,
                                               PERSONAL_MEMORY_SELECTION_VERSION, personal_memory_budgets)


TOPIC = "math.fractions"


@pytest.fixture
def identities(monkeypatch):
    current = {"chat": {"name": "gemma4:12b", "digest": "a" * 64},
               "embedding": {"name": "bge-m3", "digest": "b" * 64}}
    monkeypatch.setattr(personal_memory, "resolve_model_identities", lambda model: deepcopy(current))
    return current


@pytest.fixture
def runtime(identities):
    calls = []

    def embed(texts):
        calls.extend(texts)
        # The semantic candidate differs from the exact lexical match.
        return [([1.0, 0.0] if text == "маяк" or "семантика" in text else [0.0, 1.0])
                + [0.0] * 1022 for text in texts]

    return SimpleNamespace(context_tokens=16384, embed=embed, embedded_texts=calls,
        settings=SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                 ollama_url="http://127.0.0.1:11435", personal_memory_retrieval_mode="hybrid"))


def snapshot():
    sources = [
        {"source_id": "s1", "author_role": "student", "text": "Нарисовал красный маяк.",
         "topic_id": TOPIC, "observed_on": "2026-01-01", "session_id": "lesson1",
         "dialogue_context": {"role": "assistant", "text": "Какой цвет ты выбрал?"}},
        {"source_id": "s2", "author_role": "student", "text": "семантика: люблю рисовать.",
         "topic_id": "reading", "observed_on": "2026-01-02", "session_id": "lesson2"},
        {"source_id": "fix", "author_role": "teacher", "text": "Уточнение: маяк был синий.",
         "approved": True, "corrects_source_id": "s1", "topic_id": TOPIC,
         "observed_on": "2026-01-03"},
        {"source_id": "private", "author_role": "student", "text": "Частная история СЕКРЕТ.",
         "topic_id": "other", "observed_on": "2026-01-04"},
    ]
    return {"learner_id": "child", "revision": 2, "source_revision": 7,
            "sources": sources, "personal_graph_sources": sources,
            "personal_graph_snapshot_fingerprint": personal_graph_snapshot_fingerprint(sources), "versions": []}


def job():
    return {"job_id": "job", "learner_id": "child", "topic_id": TOPIC, "base_revision": 2,
            "source_revision": 7, "method": "personal_memory", "kind": "next_step", "provider": "local"}


def approve(provider, value, *, full_archive=False):
    state, context = provider.prepare(job(), value, "маяк")
    context["evidence_ids"] = ["s1", "fix"]
    context["source_access"] = {
        "mode": "full_archive" if full_archive else "selected_context",
        "source_ids": state["source_ids"] if full_archive else ["s1", "fix"],
        "snapshot_fingerprint": state["archive_snapshot_fingerprint"] if full_archive else None}
    version = {"id": "v1", "method": "personal_memory", "topic_id": TOPIC, "approved": True,
               "base_revision": 2, "source_revision": 7, "state": state, "context": context}
    value["versions"] = [version]
    return version


def test_ordinary_prepare_combines_lexical_and_vector_without_graph(runtime, monkeypatch):
    original_import = builtins.__import__

    def source_only_import(name, *args, **kwargs):
        assert not name.startswith(("research_memory.personal_memory_lab", "graphiti", "neo4j"))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", source_only_import)
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    state, context = provider.prepare(job(), snapshot(), "маяк")
    assert state["method"] == context["method"] == "personal_memory"
    assert state["source_context_protocol"] == PERSONAL_MEMORY_CONTEXT_PROTOCOL
    assert state["source_snapshot_id"].startswith("personal-memory-")
    assert "archive_run_id" not in state and "index_manifest" not in state
    channels = context["retrieval_diagnostics"]["channels"]
    assert channels["vector"][0] == "s2"
    assert "s1" in channels["lexical"] and "s1" in context["evidence_ids"]
    assert {"s1", "fix"} <= set(context["evidence_ids"])
    assert all(source["role"] != "assistant" for source in context["sources"])
    assert "Какой цвет ты выбрал?" in context["text"]


def test_selected_grant_never_embeds_or_selects_other_sources(runtime):
    value = snapshot()
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    approve(provider, value)
    runtime.embedded_texts.clear()
    context = provider.chat_context(value, "child", TOPIC, "маяк")
    assert context["method"] == "personal_memory" and context["version_id"] == "v1"
    assert context["stale"] is False and set(context["evidence_ids"]) == {"s1", "fix"}
    assert all("СЕКРЕТ" not in text and "семантика" not in text for text in runtime.embedded_texts)
    assert "private" not in context["retrieval_diagnostics"]["channels"]["vector"]
    assert context["retrieval_diagnostics"]["topic_recent_anchor"] is None


def test_generic_teacher_request_anchors_latest_source_of_exact_topic(runtime):
    state, context = PersonalMemoryProvider(SimpleNamespace(), runtime).prepare(
        job(), snapshot(), "Предложи следующий учебный шаг.")
    diagnostics = context["retrieval_diagnostics"]
    assert diagnostics["channels"]["vector"][0] == "private"
    assert diagnostics["selection_order"][0] == "fix"
    assert diagnostics["topic_recent_anchor"] == {"source_id": "fix", "source_ids": ["fix", "s1"]}
    assert diagnostics["selection_policy"] == state["selection_version"] == PERSONAL_MEMORY_SELECTION_VERSION


def test_teacher_topic_anchor_reserves_its_whole_correction_group_before_other_topics(runtime):
    runtime.context_tokens = 8192
    value = snapshot()
    value["sources"][-1]["text"] = "distractor " * 140
    value["personal_graph_snapshot_fingerprint"] = personal_graph_snapshot_fingerprint(value["sources"])
    _, context = PersonalMemoryProvider(SimpleNamespace(), runtime).prepare(
        job(), value, "Предложи следующий учебный шаг.")
    assert context["retrieval_diagnostics"]["channels"]["vector"][0] == "private"
    assert {"fix", "s1"} <= set(context["evidence_ids"])
    assert "private" not in context["evidence_ids"]
    assert "Какой цвет ты выбрал?" in context["text"] and "маяк был синий" in context["text"]


def test_teacher_without_history_of_requested_topic_keeps_query_ranking(runtime):
    request = {**job(), "topic_id": "math.topic.without.sessions"}
    _, context = PersonalMemoryProvider(SimpleNamespace(), runtime).prepare(request, snapshot(), "маяк")
    diagnostics = context["retrieval_diagnostics"]
    assert diagnostics["topic_recent_anchor"] is None
    assert diagnostics["selection_order"][0] == diagnostics["channels"]["vector"][0] == "s2"
    assert context["topic_id"] == request["topic_id"]


def test_full_archive_grant_retrieves_the_reviewed_snapshot_without_graph(runtime):
    value = snapshot()
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    approve(provider, value, full_archive=True)
    runtime.embedded_texts.clear()
    context = provider.chat_context(value, "child", TOPIC, "Частная история")
    assert "private" in context["evidence_ids"]
    assert any("СЕКРЕТ" in text for text in runtime.embedded_texts)


@pytest.mark.parametrize("change", ["digest", "recipe", "retrieval_mode"])
def test_changed_model_or_recipe_is_rejected_before_any_embedding(runtime, identities, change):
    value = snapshot()
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    version = approve(provider, value)
    runtime.embedded_texts.clear()
    if change == "digest":
        identities["embedding"]["digest"] = "c" * 64
    elif change == "recipe":
        version["state"]["selection_recipe_fingerprint"] = "c" * 64
    else:
        runtime.settings.personal_memory_retrieval_mode = "lexical"
    with pytest.raises(PersonalMemoryStale):
        provider.chat_context(value, "child", TOPIC, "маяк")
    assert runtime.embedded_texts == []


@pytest.mark.parametrize("change", ["source_text", "stored_source_revision", "protocol", "correction_scope", "topic", "grant"])
def test_invalid_or_stale_approvals_withhold_memory_before_model_access(runtime, change, monkeypatch):
    value = snapshot()
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    version = approve(provider, value)
    if change == "source_text":
        value["sources"][0]["text"] = "Новая версия исходника."
        value["personal_graph_snapshot_fingerprint"] = personal_graph_snapshot_fingerprint(value["sources"])
    elif change == "stored_source_revision":
        version["state"]["source_revision"] += 1
    elif change == "protocol":
        version["state"]["source_context_protocol"] = "old"
    elif change == "correction_scope":
        version["context"]["evidence_ids"] = ["s1"]
        version["context"]["source_access"]["source_ids"] = ["s1"]
    elif change == "topic":
        version["state"]["topic_id"] = "foreign"
    else:
        version["context"].pop("source_access")
    monkeypatch.setattr(provider, "_application_model", lambda: pytest.fail("invalid approval must not contact models"))
    result = provider.chat_context(value, "child", TOPIC, "маяк")
    assert result["text"] == "" and result["version_id"] is None and result["stale"] is True


def test_active_turn_revision_does_not_revoke_unchanged_approved_archive(runtime):
    value = snapshot()
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    approve(provider, value)
    # Active turns are excluded from personal_graph_sources until completion.
    value["source_revision"] += 2
    context = provider.chat_context(value, "child", TOPIC, "маяк")
    assert context["version_id"] == "v1" and context["stale"] is False
    assert set(context["evidence_ids"]) == {"s1", "fix"}


def test_new_invalid_approval_does_not_revive_an_old_grant(runtime):
    value = snapshot()
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    version = approve(provider, value)
    invalid = deepcopy(version)
    invalid["state"]["source_context_protocol"] = "old"
    value["versions"].insert(0, invalid)
    assert approved_personal_memory_version(value, "child", TOPIC) is None


@pytest.mark.parametrize("field", ["state", "context"])
def test_corrupt_latest_approval_does_not_revive_an_old_grant(runtime, field):
    value = snapshot()
    version = approve(PersonalMemoryProvider(SimpleNamespace(), runtime), value)
    invalid = deepcopy(version)
    invalid[field] = ["invalid"]
    value["versions"].insert(0, invalid)
    assert approved_personal_memory_version(value, "child", TOPIC) is None


def test_embeddings_use_request_specific_no_truncation_scope(runtime):
    active = []
    embed = runtime.embed

    @contextmanager
    def approved_memory_request():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def checked_embed(texts):
        assert active == [True]
        return embed(texts)

    runtime.approved_memory_request = approved_memory_request
    runtime.embed = checked_embed
    SourceModel(runtime).embed_many(["approved source"])
    assert active == []


def test_generation_commit_guard_rejects_model_changed_after_selection(runtime, identities):
    provider = PersonalMemoryProvider(SimpleNamespace(), runtime)
    state, _ = provider.prepare(job(), snapshot(), "маяк")
    runtime.embedded_texts.clear()
    provider.validate_state_identity(state)
    identities["chat"]["digest"] = "d" * 64
    with pytest.raises(PersonalMemoryStale):
        provider.validate_state_identity(state)
    assert runtime.embedded_texts == []


def test_cache_is_learner_and_model_scoped_and_rechecks_identity_on_hit(runtime, identities, monkeypatch):
    seen = []

    class Cache:
        def __init__(self, dsn, learner_id, model_identity):
            seen.append((learner_id, model_identity))

        def embed_many(self, texts, embed):
            identities["embedding"]["digest"] = "c" * 64
            return [[1.0] + [0.0] * 1023 for _ in texts]

    monkeypatch.setattr(personal_memory, "SourceEmbeddingCache", Cache)
    provider = PersonalMemoryProvider(SimpleNamespace(dsn="unused-test-dsn"), runtime)
    cached = provider._source_embedder(SourceModel(runtime), "child")
    with pytest.raises(PersonalMemoryStale):
        cached(["approved source"])
    assert seen[0][0] == "child" and len(seen[0][1]) == 64
    assert runtime.embedded_texts == []


def test_source_cache_identity_changes_with_embedding_recipe(runtime, monkeypatch):
    identities = []

    class Cache:
        def __init__(self, dsn, learner_id, model_identity):
            identities.append((learner_id, model_identity))

    monkeypatch.setattr(personal_memory, "SourceEmbeddingCache", Cache)
    provider = PersonalMemoryProvider(SimpleNamespace(dsn="unused-test-dsn"), runtime)
    model = SourceModel(runtime)
    provider._source_embedder(model, "child")
    monkeypatch.setattr(personal_memory, "_code_hashes", lambda files: {name: "changed" for name in files})
    provider._source_embedder(model, "child")
    provider._source_embedder(model, "other-child")
    assert identities[0][1] != identities[1][1]
    assert identities[1][1] == identities[2][1] and identities[1][0] != identities[2][0]


def test_budget_selection_skips_a_whole_oversized_correction_group(runtime):
    value = snapshot()
    value["sources"][0]["text"] *= 1500
    context = select_ranked_source_context(["s1", "fix", "s2"], value, "child", TOPIC,
                                           {"s1", "fix", "s2"}, 2500)
    assert context["evidence_ids"] == ["s2"]
    assert context["selection_warnings"] == [{"code": "memory_budget_exceeded", "count": 2}]
    assert "Уточнение" not in context["text"] and "Какой цвет" not in context["text"]
    assert len(context["text"].encode("utf-8")) <= 2500


@pytest.mark.parametrize("change", ["learner", "revision", "source_revision", "fingerprint", "provider"])
def test_prepare_rejects_mismatched_server_snapshot_before_embedding(runtime, change):
    value, request = snapshot(), job()
    if change == "learner":
        value["learner_id"] = "foreign"
    elif change in {"revision", "source_revision"}:
        value[change] += 1
    elif change == "fingerprint":
        value["personal_graph_snapshot_fingerprint"] = "f" * 64
    else:
        request["provider"] = "external"
    with pytest.raises(ValueError):
        PersonalMemoryProvider(SimpleNamespace(), runtime).prepare(request, value, "маяк")
    assert runtime.embedded_texts == []


def test_8k_memory_is_bounded_and_preserves_correction_context(runtime):
    runtime.context_tokens = 8192
    state, context = PersonalMemoryProvider(SimpleNamespace(), runtime).prepare(job(), snapshot(), "маяк")
    assert state["method"] == "personal_memory"
    assert len(context["text"].encode("utf-8")) <= 3000
    assert context["selection_budget"]["limit"] == 3000
    assert {"s1", "fix"} <= set(context["evidence_ids"])
    assert personal_memory_budgets(16384) == (10000, 12000)
    assert personal_memory_budgets(32768) == (20000, 24000)
    with pytest.raises(ValueError):
        personal_memory_budgets(4096)


def test_empty_history_remains_an_explicit_empty_draft(runtime):
    value = snapshot()
    value.update(sources=[], personal_graph_sources=[],
                 personal_graph_snapshot_fingerprint=personal_graph_snapshot_fingerprint([]))
    state, context = PersonalMemoryProvider(SimpleNamespace(), runtime).prepare(job(), value, "маяк")
    assert state["source_ids"] == [] and context["text"] == "" and context["evidence_ids"] == []
    assert runtime.embedded_texts == []


def test_default_settings_expose_hybrid_without_enabling_legacy_graph():
    settings = Settings()
    assert settings.personal_memory_enabled is True
    assert settings.personal_memory_retrieval_mode == "hybrid"
    assert settings.experimental_personal_graph_enabled is False
    with pytest.raises(ValueError, match="retrieval mode"):
        Settings(personal_memory_retrieval_mode="graph_hybrid")
