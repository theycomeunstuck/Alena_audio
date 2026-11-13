"""Reviewed canonical source memory with cached lexical/vector retrieval.

Callers authorize the learner before supplying a server-built snapshot. No
Graphiti index, graph database, graph password or research workflow is used.
Assistant turns stay anchored context and never become independent evidence.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re

from .config import require_local_url
from .personal_graph_contract import personal_graph_snapshot_fingerprint
from .personal_graph_index import resolve_model_identities
from .personal_graph_memory import (PersonalGraphSelectionBudgetExceeded, correction_closure,
                                    select_context, source_lessons, source_scoped_found,
                                    SOURCE_CONTEXT_NOTICE, SOURCE_CONTEXT_SKIPPED_NOTICE)
from .personal_graph_retrieval import SourceEmbeddingCache, _validated_vector, rank_source_candidates
from .personal_memory_contract import (PERSONAL_MEMORY_CONTEXT_PROTOCOL, PERSONAL_MEMORY_RETRIEVAL_MODES,
                                       PERSONAL_MEMORY_SELECTION_VERSION, PersonalMemoryStale,
                                       personal_memory_budgets)


METHOD = "personal_memory"
PersonalMemorySelectionBudgetExceeded = PersonalGraphSelectionBudgetExceeded


def _hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_hashes(files):
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def _valid_identities(identities):
    return isinstance(identities, dict) and all(
        isinstance(identities.get(kind), dict)
        and isinstance(identities[kind].get("name"), str) and bool(identities[kind]["name"])
        and isinstance(identities[kind].get("digest"), str)
        and re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", identities[kind]["digest"])
        for kind in ("chat", "embedding"))


def personal_memory_recipe_fingerprint(model, retrieval_mode):
    """Bind an approval to the models, context and source selection implementation."""
    if retrieval_mode not in PERSONAL_MEMORY_RETRIEVAL_MODES or not _valid_identities(model.model_identities):
        raise ValueError("Personal memory requires an explicit retrieval recipe and immutable model digests")
    files = ("local_app/personal_memory.py", "local_app/personal_memory_contract.py",
             "local_app/personal_graph_memory.py", "local_app/personal_graph_contract.py",
             "local_app/personal_graph_retrieval.py", "local_app/runtime.py", "local_app/memory_service.py",
             "local_app/session_summaries.py")
    return _hash({"protocol": PERSONAL_MEMORY_CONTEXT_PROTOCOL, "models": model.model_identities,
                  "context": model.context, "retrieval_mode": retrieval_mode,
                  "selection_version": PERSONAL_MEMORY_SELECTION_VERSION,
                  "code": _code_hashes(files)})


class SourceModel:
    """Bind retrieval and cache reads to immutable local chat/embedding models."""

    def __init__(self, runtime):
        settings = runtime.settings
        personal_memory_budgets(runtime.context_tokens)
        if settings.chat_model != "gemma4:12b" or settings.embedding_model.split(":")[0] != "bge-m3":
            raise ValueError("Personal memory requires local Gemma and BGE-M3")
        self.runtime = runtime
        self.context, self.model = runtime.context_tokens, settings.chat_model
        self.base_url = require_local_url(settings.ollama_url)
        self.model_identities = resolve_model_identities(self)
        if not _valid_identities(self.model_identities):
            raise ValueError("Personal memory requires immutable model digests")

    def check_model_identities(self):
        if resolve_model_identities(self) != self.model_identities:
            raise PersonalMemoryStale()

    def embed_many(self, texts):
        self.check_model_identities()
        request_scope = getattr(self.runtime, "approved_memory_request", nullcontext)
        with request_scope():
            vectors = self.runtime.embed(texts)
        self.check_model_identities()
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(texts):
            raise ValueError("BGE-M3 returned an invalid batch")
        return [_validated_vector(vector) for vector in vectors]


def _canonical_snapshot(snapshot, learner_id):
    if not isinstance(snapshot, dict) or snapshot.get("learner_id") != learner_id:
        raise ValueError("memory snapshot belongs to another learner")
    sources = snapshot.get("personal_graph_sources")
    if not isinstance(sources, list):
        raise ValueError("personal_memory_source_snapshot_unavailable")
    canonical = {**snapshot, "sources": sources}
    source_lessons(canonical, learner_id)
    fingerprint = personal_graph_snapshot_fingerprint(sources)
    if snapshot.get("personal_graph_snapshot_fingerprint") != fingerprint:
        raise ValueError("personal memory source snapshot fingerprint does not match")
    return canonical, fingerprint


def approved_personal_memory_version(snapshot, learner_id, topic_id):
    """Validate authorization scope without consulting model or graph services."""
    if snapshot.get("learner_id") != learner_id:
        raise ValueError("memory snapshot belongs to another learner")
    try:
        _, sources = source_lessons(snapshot, learner_id)
        fingerprint = personal_graph_snapshot_fingerprint(snapshot.get("sources", []))
    except (TypeError, ValueError):
        return None
    current_ids = set(sources)
    for version in snapshot.get("versions", []):
        if not isinstance(version, dict):
            return None
        state = version.get("state") or {}
        state_method = state.get("method") if isinstance(state, dict) else None
        if ((version.get("method") or state_method) != METHOD
                or not version.get("approved") or version.get("topic_id") != topic_id):
            continue
        from .memory_store import reviewed_snapshot_context
        if reviewed_snapshot_context(snapshot,version,learner_id,topic_id) is not None:
            return version
        # A newer invalid approval cannot silently revive an older grant.
        if (not isinstance(state, dict) or version.get("stale")
                or version.get("base_revision") != snapshot.get("revision")
                or state.get("source_revision") != version.get("source_revision")
                or state.get("method") != METHOD or state.get("learner_id") != learner_id
                or state.get("topic_id") != topic_id
                or state.get("source_context_protocol") != PERSONAL_MEMORY_CONTEXT_PROTOCOL
                or state.get("archive_snapshot_fingerprint") != fingerprint
                or state.get("source_snapshot_id") != "personal-memory-" + fingerprint
                or state.get("retrieval_mode") not in PERSONAL_MEMORY_RETRIEVAL_MODES
                or state.get("selection_version") != PERSONAL_MEMORY_SELECTION_VERSION
                or not _valid_identities(state.get("model_identities"))
                or not isinstance(state.get("selection_recipe_fingerprint"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", state["selection_recipe_fingerprint"])):
            return None
        ids = state.get("source_ids")
        context = version.get("context") or {}
        if not isinstance(context, dict):
            return None
        evidence = context.get("evidence_ids")
        if (not isinstance(ids, list) or any(not isinstance(item, str) or not item for item in ids)
                or len(ids) != len(set(ids)) or set(ids) != current_ids
                or not isinstance(evidence, list) or any(not isinstance(item, str) or not item for item in evidence)
                or len(evidence) != len(set(evidence)) or not set(evidence) <= current_ids
                or not correction_closure(sources, evidence) <= set(evidence)):
            return None
        access = context.get("source_access")
        if not isinstance(access, dict):
            return None
        access_ids = access.get("source_ids")
        if (not isinstance(access_ids, list) or any(not isinstance(item, str) or not item for item in access_ids)
                or len(access_ids) != len(set(access_ids))):
            return None
        if access.get("mode") == "full_archive":
            if set(access_ids) != current_ids or access.get("snapshot_fingerprint") != fingerprint:
                return None
        elif access.get("mode") == "selected_context":
            if set(access_ids) != set(evidence):
                return None
        else:
            return None
        return version
    return None


def _correction_components(sources):
    """Index each connected correction component once for this source snapshot."""
    links = {source_id: set() for source_id in sources}
    for source_id, source in sources.items():
        target = source.get("corrects_source_id")
        if target:
            links[source_id].add(target)
            links[target].add(source_id)
    components = {}
    for source_id in sources:
        if source_id in components:
            continue
        pending, members = [source_id], {source_id}
        while pending:
            for related in links[pending.pop()]:
                if related not in members:
                    members.add(related)
                    pending.append(related)
        component = frozenset(members)
        components.update((member, component) for member in component)
    return components


def _json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _json_list_bytes(item_sizes):
    return 2 + sum(item_sizes) + max(0, len(item_sizes) - 1)


class _SourceContextSizer:
    """Exact canonical payload costs without rendering the whole archive per trial."""

    def __init__(self, lessons, sources):
        self.lessons = {lesson["lesson_id"]: lesson for lesson in lessons}
        self.source_lesson, self.source_turns, self.turn_order = {}, {}, {}
        self.block_sizes = {}
        self.source_sizes = {source_id: _json_bytes(source) for source_id, source in sources.items()}
        self.empty_payload_size = _json_bytes({"notice": SOURCE_CONTEXT_NOTICE,
                                              "lessons": [], "related_corrections": []})
        for lesson in lessons:
            for position, turn in enumerate(lesson["turns"]):
                source_id = turn.get("anchor_source_id") if turn["role"] == "assistant" else turn["turn_id"]
                self.source_lesson[source_id] = lesson["lesson_id"]
                self.source_turns.setdefault(source_id, []).append({**turn, "topic_id": sources[source_id]["topic_id"]})
                self.turn_order[turn["turn_id"]] = position

    def _block_size(self, lesson_id, source_ids, *, context_only=False):
        key = (lesson_id, frozenset(source_ids), context_only)
        if key not in self.block_sizes:
            turns = [turn for source_id in source_ids for turn in self.source_turns[source_id]
                     if not context_only or turn["role"] == "assistant"]
            turns.sort(key=lambda turn: self.turn_order[turn["turn_id"]])
            if not turns:
                self.block_sizes[key] = 0
            else:
                lesson = self.lessons[lesson_id]
                self.block_sizes[key] = _json_bytes({"lesson_id": lesson_id, "observed_on": lesson["observed_on"],
                    "topic_ids": list(dict.fromkeys(turn["topic_id"] for turn in turns)), "turns": turns})
        return self.block_sizes[key]

    def _lesson_sizes(self, source_ids, *, context_only=False):
        by_lesson = {}
        for source_id in source_ids:
            by_lesson.setdefault(self.source_lesson[source_id], []).append(source_id)
        return [size for lesson_id, ids in by_lesson.items()
                if (size := self._block_size(lesson_id, ids, context_only=context_only))]

    def size(self, representatives, visible_ids, skipped_count):
        if not representatives:
            return 0
        linked_ids = visible_ids - set(representatives)
        size = (self.empty_payload_size
                + _json_list_bytes(self._lesson_sizes(representatives)) - 2
                + _json_list_bytes([self.source_sizes[source_id] for source_id in linked_ids]) - 2)
        related_sizes = self._lesson_sizes(linked_ids, context_only=True)
        if related_sizes:
            size += len(',"related_context":') + _json_list_bytes(related_sizes)
        if skipped_count:
            size += len(SOURCE_CONTEXT_SKIPPED_NOTICE.encode("utf-8"))
            size += len(',"selection_warnings":') + _json_bytes(
                [{"code": "memory_budget_exceeded", "count": skipped_count}])
        return size


def select_ranked_source_context(ranked_ids, snapshot, learner_id, topic_id, allowed_ids, memory_byte_budget):
    """Fit exact correction-group payload costs, then validate and render once."""
    lessons, sources = source_lessons(snapshot, learner_id)
    allowed = set(allowed_ids)
    components = _correction_components(sources)
    if not allowed <= set(sources):
        raise ValueError("source correction requires a new approved memory version")
    allowed_components = {components[source_id] for source_id in allowed}
    if any(not component <= allowed for component in allowed_components):
        raise ValueError("source correction requires a new approved memory version")
    if any(source_id not in allowed for source_id in ranked_ids):
        raise ValueError("retrieved source is outside the approved scope")
    groups, by_group, seen = [], {}, set()
    for source_id in ranked_ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        key = components[source_id]
        if key not in by_group:
            by_group[key] = []
            groups.append((key, by_group[key]))
        by_group[key].append(source_id)
    candidates = set().union(*(key for key, _ in groups)) if groups else set()
    selected, selected_ids = [], set()
    sizer = _SourceContextSizer(lessons, sources)
    for group_ids, representatives in groups:
        # Even the smallest representation must carry every other correction
        # member. Reject impossible components without quadratic retries.
        group_sizes = [sizer.source_sizes[source_id] for source_id in group_ids]
        if sum(group_sizes) - max(group_sizes) > memory_byte_budget:
            continue
        trial_ids = selected_ids | group_ids
        for source_id in representatives:
            trial = selected + [source_id]
            if sizer.size(trial, trial_ids, len(candidates) - len(trial_ids)) <= memory_byte_budget:
                selected, selected_ids = trial, trial_ids
                break
    found = source_scoped_found(lessons, selected, skipped_source_ids=candidates - selected_ids)
    context = select_context(found, snapshot, learner_id, topic_id,
                             allowed_ids=allowed, memory_byte_budget=memory_byte_budget)
    context.update(method=METHOD, selection_version=PERSONAL_MEMORY_SELECTION_VERSION)
    return context


class PersonalMemoryProvider:
    """The ordinary teacher draft and approved student recall path."""

    def __init__(self, store, runtime, *, model_call=None):
        self.store, self.runtime = store, runtime

    def _application_model(self):
        return SourceModel(self.runtime)

    def _source_embedder(self, model, learner_id):
        model.check_model_identities()
        if not getattr(self.store, "dsn", None):
            return model.embed_many
        # Cache identity excludes the query and every other learner's sources.
        identity = _hash({"model": model.model_identities["embedding"],
                          "source_recipe": PERSONAL_MEMORY_SELECTION_VERSION,
                          "code": _code_hashes(("local_app/runtime.py", "local_app/personal_memory.py",
                                               "local_app/personal_graph_memory.py",
                                               "local_app/personal_graph_retrieval.py"))})
        cache = SourceEmbeddingCache(self.store.dsn, learner_id, identity)

        def cached_sources(texts):
            model.check_model_identities()
            result = cache.embed_many(texts, model.embed_many)
            model.check_model_identities()
            return result
        return cached_sources

    def _retrieve(self, snapshot, learner_id, topic_id, question, allowed, model, mode, *, topic_recent_anchor=False):
        # Physically exclude unapproved texts before either embedding callback.
        scoped = {**snapshot, "sources": [row for row in snapshot["sources"]
                                         if str(row.get("source_id") or "") in allowed]}
        lessons, source_map = source_lessons(scoped, learner_id)
        cached = self._source_embedder(model, learner_id) if mode in {"vector", "hybrid"} else None
        ranked = rank_source_candidates(question, lessons, allowed, mode=mode,
            embed_many=model.embed_many, source_embed_many=cached, limit=max(1, len(source_map)))
        ranked_ids = ranked["ranked_source_ids"]
        anchor = None
        if topic_recent_anchor:
            # A generic next-step request still needs the latest observation of
            # its exact topic. Chat keeps the query-specific source order.
            anchor = next((source_id for source_id in ranked["channels"].get("chronological", [])
                           if source_map[source_id]["topic_id"] == topic_id), None)
            if anchor is not None:
                ranked_ids = list(dict.fromkeys([anchor] + ranked_ids))
        budget, _ = personal_memory_budgets(self.runtime.context_tokens)
        context = select_ranked_source_context(ranked_ids, scoped, learner_id,
                                              topic_id, allowed, budget)
        ranked["diagnostics"].update(selection_policy=PERSONAL_MEMORY_SELECTION_VERSION,
            selection_order=ranked_ids,
            topic_recent_anchor=({"source_id": anchor,
                                  "source_ids": sorted(correction_closure(source_map, {anchor}))}
                                 if anchor is not None else None))
        context["retrieval_diagnostics"] = ranked["diagnostics"]
        model.check_model_identities()
        return context

    def prepare(self, job, snapshot, question):
        learner_id, topic_id = job["learner_id"], job.get("topic_id") or "general"
        if job.get("method") != METHOD or job.get("kind") != "next_step" or job.get("provider", "local") != "local":
            raise ValueError("personal_memory_is_local_teacher_next_step_only")
        if (snapshot.get("revision") != job.get("base_revision")
                or snapshot.get("source_revision") != job.get("source_revision")):
            raise ValueError("source snapshot changed before personal memory selection")
        canonical, fingerprint = _canonical_snapshot(snapshot, learner_id)
        _, sources = source_lessons(canonical, learner_id)
        mode = getattr(self.runtime.settings, "personal_memory_retrieval_mode", "hybrid")
        model = self._application_model()
        recipe = personal_memory_recipe_fingerprint(model, mode)
        context = self._retrieve(canonical, learner_id, topic_id, question, set(sources), model, mode,
                                 topic_recent_anchor=True)
        state = {"method": METHOD, "topic_id": topic_id, "learner_id": learner_id,
                 "source_revision": snapshot["source_revision"], "source_ids": sorted(sources),
                 "source_context_protocol": PERSONAL_MEMORY_CONTEXT_PROTOCOL,
                 "archive_snapshot_fingerprint": fingerprint, "source_snapshot_id": "personal-memory-" + fingerprint,
                 "retrieval_mode": mode, "model_identities": deepcopy(model.model_identities),
                 "selection_recipe_fingerprint": recipe, "selection_version": PERSONAL_MEMORY_SELECTION_VERSION}
        if personal_memory_recipe_fingerprint(model, mode) != recipe:
            raise PersonalMemoryStale()
        return state, context

    def validate_state_identity(self, state):
        """Recheck a draft after generation, before the caller commits it."""
        model = self._application_model()
        mode = getattr(self.runtime.settings, "personal_memory_retrieval_mode", "hybrid")
        if (not isinstance(state, dict) or state.get("method") != METHOD
                or state.get("source_context_protocol") != PERSONAL_MEMORY_CONTEXT_PROTOCOL
                or state.get("selection_version") != PERSONAL_MEMORY_SELECTION_VERSION
                or state.get("model_identities") != model.model_identities
                or state.get("retrieval_mode") != mode
                or state.get("selection_recipe_fingerprint") != personal_memory_recipe_fingerprint(model, mode)):
            raise PersonalMemoryStale()
        model.check_model_identities()

    def chat_context(self, snapshot, learner_id, topic_id, question):
        canonical, _ = _canonical_snapshot(snapshot, learner_id)
        version = approved_personal_memory_version(canonical, learner_id, topic_id)
        if version is None:
            return {"text": "", "sources": [], "evidence_ids": [], "stale": True, "method": METHOD,
                    "version_id": None, "topic_id": topic_id, "withheld_reason": "personal_memory_not_approved_or_stale"}
        from .memory_store import reviewed_snapshot_context
        reviewed = reviewed_snapshot_context(canonical,version,learner_id,topic_id)
        if reviewed is not None:
            return reviewed
        state = version["state"]
        mode = getattr(self.runtime.settings, "personal_memory_retrieval_mode", "hybrid")
        model = self._application_model()
        recipe = personal_memory_recipe_fingerprint(model, mode)
        if (state["model_identities"] != model.model_identities
                or state["selection_recipe_fingerprint"] != recipe or state["retrieval_mode"] != mode):
            raise PersonalMemoryStale()
        allowed = set(version["context"]["source_access"]["source_ids"])
        context = self._retrieve(canonical, learner_id, topic_id, question, allowed, model, mode)
        if personal_memory_recipe_fingerprint(model, mode) != recipe:
            raise PersonalMemoryStale()
        context["version_id"] = version.get("version_id") or version.get("id")
        return context
