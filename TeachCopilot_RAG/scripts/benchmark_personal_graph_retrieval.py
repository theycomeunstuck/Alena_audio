"""Read-only retrieval benchmark for an existing synthetic personal graph run."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODES = ("chronological", "lexical", "vector", "hybrid", "graph_hybrid")
DB_NAME = re.compile(r"nirs_personal_graph_(?:it|demo)_[A-Za-z0-9_]+\Z")
SOURCE_ARCHIVE_CHANNELS = frozenset({"semantic", "lexical", "source_semantic",
                                     "source_lexical", "chronological"})


def validate_db_name_file(path):
    raw = Path(path).read_text(encoding="utf-8-sig")
    name = raw.strip()
    if not name or "\n" in name or "\r" in name or not DB_NAME.fullmatch(name):
        raise ValueError("database name file must contain one synthetic local database name")
    return name


def load_requests(path):
    rows = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid request JSON on line {line_number}") from exc
        if (not isinstance(row, dict) or not isinstance(row.get("query_id"), str) or not row["query_id"]
                or row["query_id"] in seen or not isinstance(row.get("question"), str)
                or not row["question"].strip() or not isinstance(row.get("expected_source_ids"), list)
                or any(not isinstance(item, str) or not item for item in row["expected_source_ids"])):
            raise ValueError(f"invalid or duplicate request record on line {line_number}")
        seen.add(row["query_id"])
        rows.append({"query_id": row["query_id"], "question": row["question"],
                     "expected_source_ids": list(dict.fromkeys(row["expected_source_ids"]))})
    if not rows:
        raise ValueError("requests file is empty")
    return rows


def percentile(values, percentile_value):
    """Linearly interpolated percentile for small descriptive timing samples."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def source_recall(selected_source_ids, expected_source_ids):
    expected = set(expected_source_ids)
    selected = set(selected_source_ids)
    missing = sorted(expected - selected)
    if not expected:
        return {"source_recall": None, "source_recall_status": "no_gold_sources",
                "missing_expected_source_ids": []}
    return {"source_recall": len(expected & selected) / len(expected),
            "source_recall_status": "gold_available",
            "missing_expected_source_ids": missing}


def _channel_counts(hits):
    counts = {}
    for hit in hits:
        kind = str(hit.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _candidate_source_ids(lessons):
    ids = []
    for lesson in lessons:
        for turn in lesson.get("turns", []):
            if turn.get("role") in {"teacher", "student"} and turn.get("turn_id"):
                ids.append(str(turn["turn_id"]))
    return list(dict.fromkeys(ids))


class _CountPolicy:
    """Delegate model operations while selecting app byte or measured token budget."""
    def __init__(self, model, budget_kind):
        self._model, self.budget_kind = model, budget_kind

    def __getattr__(self, name):
        return getattr(self._model, name)

    def count(self, text):
        if self.budget_kind == "bytes":
            return len(text.encode("utf-8")) if text else 0
        return self._model.count(text)


def _model_identities(runtime, resolver=None):
    if resolver is None:
        from local_app.personal_graph_index import resolve_model_identities
        resolver = resolve_model_identities
    return resolver(runtime)


def _code_provenance():
    root = Path(__file__).resolve().parents[1]
    relevant = [Path(__file__), root / "local_app" / "personal_graph_memory.py",
                root / "research_memory" / "personal_memory_lab" / "workflow.py",
                root / "research_memory" / "personal_memory_lab" / "archive.py",
                root / "research_memory" / "personal_memory_lab" / "runtime.py",
                root / "research_memory" / "personal_memory_lab" / "backends.py",
                root / "research_memory" / "personal_memory_lab" / "requirements.lock.txt"]
    files = {}
    for path in relevant:
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files_sha256": files, "combined_sha256": hashlib.sha256(canonical).hexdigest()}


def _library_provenance():
    packages = ("langgraph", "graphiti-core", "psycopg2-binary", "pgvector", "neo4j", "ollama")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    lock_path = (Path(__file__).resolve().parents[1] / "research_memory" /
                 "personal_memory_lab" / "requirements.lock.txt")
    return {"versions": versions, "requirements_lock_sha256": hashlib.sha256(
        lock_path.read_bytes()).hexdigest()}


def _json_sha256(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _limit_source_candidate_state(retrieved, source_top_k, recent_lesson_ids):
    """Cap unique archive chunks before hydration; recent lessons get their own lane."""
    if type(source_top_k) is not int or source_top_k <= 0:
        raise ValueError("source_top_k must be a positive integer")
    from local_app.personal_graph_memory import round_robin_source_hits

    all_hits = [hit for hit in retrieved.get("hits", []) if isinstance(hit, dict)]
    source_hits = [hit for hit in all_hits if hit.get("kind") in SOURCE_ARCHIVE_CHANNELS
                   and isinstance(hit.get("chunk_id"), str)]
    graph_hits = [hit for hit in all_hits if hit.get("kind") == "graphiti"]
    balanced = round_robin_source_hits(source_hits)
    raw_chunk_ids, selected, selected_ids = [], [], set()
    raw_seen = set()
    for hit in balanced:
        chunk_id = hit["chunk_id"]
        if chunk_id not in raw_seen:
            raw_seen.add(chunk_id)
            raw_chunk_ids.append(chunk_id)
        if chunk_id in selected_ids:
            continue
        if len(selected) >= source_top_k:
            continue
        selected_ids.add(chunk_id)
        selected.append(hit)

    recent = list(dict.fromkeys(str(item) for item in recent_lesson_ids))
    source_lesson_ids = list(dict.fromkeys(str(lesson_id) for hit in selected
                                            for lesson_id in hit.get("lesson_ids", [])
                                            if isinstance(lesson_id, str)))
    lesson_ids = recent + [lesson_id for lesson_id in source_lesson_ids if lesson_id not in set(recent)]
    channel_counts = _channel_counts(source_hits)
    diagnostics = {
        "source_top_k": source_top_k,
        "retrieved_hits_before_sha256": _json_sha256(all_hits),
        "archive_source_hits_after_sha256": _json_sha256(selected),
        "raw_archive_hit_count": len(source_hits),
        "raw_archive_channel_counts": channel_counts,
        "raw_unique_source_chunk_candidate_ids": raw_chunk_ids,
        "raw_unique_source_chunk_candidate_count": len(raw_chunk_ids),
        "final_source_chunk_candidate_ids": [hit["chunk_id"] for hit in selected],
        "final_source_chunk_candidate_count": len(selected),
        "recent_context_lesson_ids": recent,
        "recent_context_channel_separate_from_k": True,
        "final_hydration_lesson_ids": lesson_ids,
        "graph_fact_hit_count": len(graph_hits),
        "graph_hits_passed_to_hydration": False,
        "graph_hits_as_evidence": False,
        "graph_nodes_as_evidence": False,
        "dropped_graph_only_lesson_count": len(set(
            str(lesson_id) for hit in graph_hits for lesson_id in hit.get("lesson_ids", [])
            if isinstance(lesson_id, str)) - set(lesson_ids)),
    }
    return {**retrieved, "hits": selected, "unresolved": [], "lesson_ids": lesson_ids}, diagnostics


class _ArchiveSearchTopK:
    """Benchmark-only archive proxy limiting every source search invocation."""
    def __init__(self, archive, limit):
        if type(limit) is not int or limit <= 0:
            raise ValueError("archive source limit must be a positive integer")
        self.archive, self.limit = archive, limit
        self.reset()

    def __getattr__(self, name):
        return getattr(self.archive, name)

    def reset(self):
        self.search_calls = []

    def search(self, *args, **kwargs):
        if len(args) >= 5:
            values = list(args)
            values[4] = self.limit
            args = tuple(values)
        else:
            kwargs["limit"] = self.limit
        rows = self.archive.search(*args, **kwargs)
        counts = {}
        for row in rows:
            channel = str(row.get("channel") or row.get("kind") or "unknown")
            counts[channel] = counts.get(channel, 0) + 1
        self.search_calls.append({"limit": self.limit, "row_count": len(rows),
                                  "channel_counts": counts,
                                  "chunk_ids": [str(row["chunk_id"]) for row in rows
                                                if row.get("chunk_id") is not None]})
        return rows

    def diagnostics(self):
        counts = {}
        for call in self.search_calls:
            for channel, count in call["channel_counts"].items():
                counts[channel] = counts.get(channel, 0) + count
        return {"archive_search_call_count": len(self.search_calls),
                "archive_search_limit_each_call": self.limit,
                "archive_search_rows_by_channel": counts,
                "archive_search_calls": list(self.search_calls)}


def _install_source_top_k_graph(workflow, source_top_k, recent_lesson_ids, archive_proxy, mode):
    """Insert the common-K source gate between retrieve and hydrate in this benchmark only."""
    from langgraph.graph import END, START, StateGraph
    from research_memory.personal_memory_lab.workflow import State

    async def timed_call(stage, method, state):
        started = time.perf_counter()
        try:
            return await method(state)
        finally:
            timings = getattr(workflow, "source_top_k_stage_timings", None)
            if timings is not None:
                timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - started

    async def retrieve_node(state):
        return await timed_call("retrieve_seconds", workflow.retrieve, state)

    async def hydrate_node(state):
        return await timed_call("hydrate_seconds", workflow.hydrate, state)

    async def select_node(state):
        return await timed_call("select_seconds", workflow.select, state)

    async def limit_archive_sources(state):
        started = time.perf_counter()
        try:
            retrieved, diagnostics = _limit_source_candidate_state(
                state, source_top_k, recent_lesson_ids)
            diagnostics.update(archive_proxy.diagnostics())
            diagnostics["graph_search_calls"] = 1 if mode == "graph_hybrid" else 0
            diagnostics["graph_query_cost_note"] = (
                "graph_hybrid performs one graph search in addition to archive search calls; graph facts are retrieval hints only"
                if mode == "graph_hybrid" else "no graph search")
            workflow.source_top_k_diagnostics = diagnostics
            return {"hits": retrieved["hits"], "unresolved": retrieved["unresolved"],
                    "lesson_ids": retrieved["lesson_ids"]}
        finally:
            timings = getattr(workflow, "source_top_k_stage_timings", None)
            if timings is not None:
                timings["source_top_k_gate_seconds"] = (
                    timings.get("source_top_k_gate_seconds", 0.0) + time.perf_counter() - started)

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("source_top_k", limit_archive_sources)
    graph.add_node("hydrate", hydrate_node)
    graph.add_node("select", select_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "source_top_k")
    graph.add_edge("source_top_k", "hydrate")
    graph.add_edge("hydrate", "select")
    graph.add_edge("select", END)
    workflow.graph = graph.compile()
    workflow.source_top_k_diagnostics = {}
    workflow.source_top_k_stage_timings = {}
    workflow.source_top_k_gate_elapsed = 0.0
    workflow.source_top_k_archive_proxy = archive_proxy


def _validate_vector_compatibility_report(path, current_model_identities, archive, run_id, learner_id,
                                          selected_chunks):
    from research_memory.personal_memory_lab.backends import lesson_text
    from research_memory.personal_memory_lab.archive import canonical

    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise ValueError("vector compatibility report status is not pass")
    coverage = report.get("coverage") or {}
    records = report.get("chunks")
    if (not isinstance(records, list) or type(coverage.get("expected")) is not int
            or coverage.get("expected") != len(records)
            or coverage.get("found") != len(records) or coverage.get("processed") != len(records)):
        raise ValueError("vector compatibility report has incomplete chunk coverage")
    if report.get("model_identities_before") != current_model_identities or \
            report.get("model_identities_after") != current_model_identities:
        raise ValueError("vector compatibility report model identity differs from current runtime")
    expected_by_learner = report.get("scope_expected")
    if not isinstance(expected_by_learner, dict):
        raise ValueError("vector compatibility report is missing scope counts")
    report_by_key = {}
    by_learner = {}
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("invalid vector compatibility chunk row")
        key = (item.get("run_id"), item.get("learner_id"), item.get("chunk_id"))
        if not all(isinstance(value, str) and value for value in key) or key in report_by_key:
            raise ValueError("duplicate or invalid vector compatibility chunk identity")
        max_diff, cosine = item.get("max_abs_diff"), item.get("cosine")
        if (item.get("status") != "pass" or item.get("stored_vector_hash") != item.get("fresh_vector_hash")
                or not isinstance(item.get("content_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["content_hash"])
                or not isinstance(item.get("source_hash"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["source_hash"])
                or not isinstance(max_diff, (int, float)) or not math.isfinite(max_diff) or max_diff > 1e-6
                or not isinstance(cosine, (int, float)) or not math.isfinite(cosine) or cosine < 1 - 1e-6):
            raise ValueError("vector compatibility report contains a failed or malformed chunk")
        report_by_key[key] = item
        by_learner[item["learner_id"]] = by_learner.get(item["learner_id"], 0) + 1
    if by_learner != expected_by_learner:
        raise ValueError("vector compatibility report scope counts do not match chunk rows")
    rows_by_scope = {}
    for run, learner, _ in report_by_key:
        rows_by_scope.setdefault((run, learner), {})
    for scope in rows_by_scope:
        run, learner = scope
        expected_count = expected_by_learner.get(learner)
        report_rows = {chunk_id: item for (r, l, chunk_id), item in report_by_key.items()
                       if (r, l) == scope}
        if expected_count is None or len(report_rows) != expected_count:
            raise ValueError("vector compatibility report has mismatched per-scope coverage")
        archive_rows = {str(row["chunk_id"]): row for row in archive.chunks(run, learner)}
        if set(archive_rows) != set(report_rows):
            raise ValueError("vector compatibility report chunk IDs do not match archive scope")
        for chunk_id, row in archive_rows.items():
            item = report_rows[chunk_id]
            content_hash = hashlib.sha256(canonical(row["content"]).encode("utf-8")).hexdigest()
            source_hash = hashlib.sha256(lesson_text(row["content"]).encode("utf-8")).hexdigest()
            if item["content_hash"] != content_hash or item["source_hash"] != source_hash:
                raise ValueError("vector compatibility report chunk hashes do not match archive scope")
    selected_count = len({str(row["chunk_id"]) for row in selected_chunks})
    if (run_id, learner_id) not in rows_by_scope or selected_count != expected_by_learner.get(learner_id):
        raise ValueError("vector compatibility report does not cover selected archive scope")
    return {"status": "pass", "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "report_path": str(path), "validated_chunk_count": len(records),
            "selected_scope_chunk_count": selected_count,
            "report_chunk_count": len(records), "current_model_identities": current_model_identities,
            "index_embedding_identity": "unknown",
            "interpretation": "all report rows pass numerical comparison to the current model; historical ingest digest remains unknown"}


def _save_report(path, report, *, create=False):
    mode = "x" if create else "w"
    with Path(path).open(mode, encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _update_coverage(report, requests, repeats):
    planned_records = len(requests) * len(MODES)
    planned_attempts = planned_records * (repeats + 1)
    records = report["results"]
    attempts = [trial for row in records for trial in row.get("trials", [])]
    report["coverage"] = {
        "records_planned": planned_records,
        "records_written": sum(row.get("status") != "running" for row in records),
        "records_with_errors": sum(row.get("status") in {"setup_error", "partial_error", "error"}
                                    for row in records),
        "benchmark_setup_error_count": int(bool(report.get("setup_error_type"))),
        "query_attempts_planned": planned_attempts,
        "query_attempts_recorded": len(attempts),
        "query_attempts_succeeded": sum(trial.get("status") == "ok" for trial in attempts),
        "query_attempts_failed": sum(trial.get("status") == "error" for trial in attempts),
        "query_attempts_terminal_setup_failures": sum(
            repeats + 1 for row in records if row.get("status") == "setup_error"),
    }


def _finish_status(report):
    records = report["results"]
    coverage = report.get("coverage", {})
    if report.get("setup_error_type"):
        report["status"] = "setup_error"
    elif (not records or any(row.get("status") == "running" for row in records)
          or coverage.get("records_written", 0) < coverage.get("records_planned", 0)
          or coverage.get("query_attempts_recorded", 0)
          + coverage.get("query_attempts_terminal_setup_failures", 0)
          < coverage.get("query_attempts_planned", 0)):
        report["status"] = "running"
    elif any(row.get("status") in {"setup_error", "partial_error", "error"} for row in records):
        report["status"] = "partial_error"
    elif report.get("run_stability", {}).get("stable") is False:
        report["status"] = "partial_error"
    else:
        report["status"] = "complete"
    report["scientific_result"] = False


def _finalize_provenance(report, runtime, resolver):
    code_after = _code_provenance()
    report["code_provenance_before"] = report.get("code_provenance")
    report["code_provenance_after"] = code_after
    code_stable = report["code_provenance_before"] == code_after
    model_before = report.get("model", {}).get("current_model_identities_before")
    model_after = None
    model_error = None
    if model_before is not None:
        try:
            model_after = _model_identities(runtime, resolver)
        except Exception as exc:
            model_error = type(exc).__name__
    model_stable = (model_before == model_after) if model_before is not None else None
    stable = (code_stable and model_stable) if model_before is not None else None
    report["run_stability"] = {
        "code_hashes_stable": code_stable,
        "model_identities_stable": model_stable,
        "stable": stable,
        "model_identity_error_type": model_error,
    }
    report["model"]["current_model_identities_after"] = model_after
    if not code_stable or (model_before is not None and not model_stable):
        report["integrity_error"] = "code_or_model_identity_changed_during_run"


async def run_benchmark(*, database_url, run_id, learner_id, requests, output_path,
                        context=16384, repeats=3, budget_kind="bytes", budget=4096,
                        settings=None, runtime=None, archive=None, graph_factory=None,
                        workflow_factory=None, database_name=None,
                        requests_file=None, database_name_file=None, base_workflow=None,
                        model_identity_resolver=None, source_top_k=None,
                        vector_compatibility_report=None):
    """Run retrieval only; injected objects keep the benchmark unit-testable."""
    if context not in (16384, 32768) or type(repeats) is not int or repeats < 1:
        raise ValueError("context must be 16384 or 32768 and repeats must be positive")
    if budget_kind not in {"bytes", "tokens"} or type(budget) is not int or budget <= 0:
        raise ValueError("invalid benchmark budget")
    if budget > context - 2048:
        raise ValueError("budget exceeds the selected context working window")
    if source_top_k is not None and (type(source_top_k) is not int or source_top_k <= 0):
        raise ValueError("source_top_k must be a positive integer")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("benchmark output already exists")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("benchmark output directory does not exist")

    from research_memory.personal_memory_lab.archive import Archive
    from research_memory.personal_memory_lab.backends import GraphitiAdapter, make_graphiti
    from research_memory.personal_memory_lab.runtime import LocalModel
    from local_app.personal_graph_memory import application_memory_workflow, application_lesson_order
    if workflow_factory is None:
        from research_memory.personal_memory_lab.workflow import PersonalMemoryWorkflow
        base_workflow = base_workflow or PersonalMemoryWorkflow

    if runtime is None:
        if settings is None:
            raise ValueError("settings are required when a runtime is not injected")
        runtime = LocalModel(base_url=settings.ollama_url, model=settings.chat_model, context=context)
    if runtime.context != context:
        raise ValueError("runtime context does not match benchmark context")
    raw_archive = archive or Archive(database_url)
    model = _CountPolicy(runtime, budget_kind)
    lessons = raw_archive.lessons(run_id, learner_id)
    if not lessons:
        raise ValueError("selected synthetic run and learner have no archived lessons")
    chunks = raw_archive.chunks(run_id, learner_id)
    if not chunks:
        raise ValueError("selected synthetic run and learner have no archived source chunks")
    lesson_ids = {lesson["lesson_id"] for lesson in lessons}
    if any(row["lesson_id"] not in lesson_ids for row in chunks):
        raise ValueError("archive chunk metadata does not match archived lessons")
    chunk_ids = sorted(str(row["chunk_id"]) for row in chunks)
    chunk_manifest = [{"chunk_id": str(row["chunk_id"]), "lesson_id": str(row["lesson_id"]),
                       "position": row.get("position"), "content": row.get("content")} for row in chunks]
    chunk_manifest_hash = _json_sha256(chunk_manifest)
    lesson_content_hash = _json_sha256(lessons)

    references = raw_archive.index_references(run_id, learner_id, "graphiti")
    graph_lesson_ids = {row["lesson_id"] for row in references}
    missing_graph = sorted(lesson_ids - graph_lesson_ids)
    graph = graph_adapter = None
    password_path = None
    if settings is not None:
        password_path = Path(settings.personal_graph_password_file)
    graph_available = not missing_graph

    from local_app.personal_graph_memory import RETRIEVAL_MODES
    if set(MODES) != set(RETRIEVAL_MODES):
        raise RuntimeError("application retrieval modes differ from benchmark modes")

    if (settings is not None and (settings.chat_model != "gemma4:12b"
            or settings.embedding_model.split(":")[0] != "bge-m3")):
        raise ValueError("personal graph benchmark requires Gemma 4 12B and BGE-M3")

    request_hash = hashlib.sha256(Path(requests_file).read_bytes()).hexdigest() if requests_file else None
    db_name_hash = hashlib.sha256(Path(database_name_file).read_bytes()).hexdigest() if database_name_file else None
    compatibility_path = Path(vector_compatibility_report) if vector_compatibility_report else None
    if compatibility_path is not None and not compatibility_path.is_file():
        raise FileNotFoundError("vector compatibility report does not exist")
    compatibility_hash = (hashlib.sha256(compatibility_path.read_bytes()).hexdigest()
                          if compatibility_path else None)
    recent_lesson_ids = application_lesson_order(lessons)[-2:]
    model_metadata = {"chat": {"name": getattr(runtime, "model", None), "digest": None},
                      "embedding": {"name": getattr(settings, "embedding_model", "bge-m3"), "digest": None},
                      "context_tokens": context}
    document = {
        "schema_version": 1, "status": "running", "scientific_result": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "learner_id": learner_id, "database_name": database_name,
        "model": {**model_metadata, "identity_role": "current_runtime", "index_embedding_identity": "unknown"},
        "budget": {"kind": budget_kind, "value": budget,
                   "meaning": "UTF-8 bytes as used by the application workflow" if budget_kind == "bytes"
                   else "LocalModel.count exact token count"},
        "context_tokens": context, "repeats": repeats,
        "timing_semantics": {"first": "first sequential query in this process; not a cold service restart",
                             "warm": "subsequent sequential repeats in this process",
                             "stage_timings": ("with --source-top-k: retrieve includes embed, archive SQL and graph search; "
                                               "gate is source balancing/capping; hydrate and select are timed separately"),
                             "includes_token_count_probe": budget_kind == "tokens",
                             "token_probe_note": ("LocalModel.count calls the local model and its latency is included"
                                                  if budget_kind == "tokens" else None)},
        "application_context": {"shared_recent_lesson_ids": recent_lesson_ids,
                                 "shared_across_modes": True},
        "source_budget": ({"enabled": True, "k": source_top_k,
                           "per_archive_search_limit": source_top_k,
                           "unique_archive_chunk_cap_before_hydration": source_top_k,
                           "balancing": "round_robin_source_hits",
                           "recent_context_lesson_count": len(recent_lesson_ids),
                           "recent_context_outside_k": True,
                           "graph_facts_as_evidence": False,
                           "graph_search_is_additional_query_cost": True,
                           "comparability_scope": "matched archive source chunk candidates; not matched query topology"}
                          if source_top_k is not None else {"enabled": False, "pipeline": "default"}),
        "vector_compatibility": ({"status": "pending", "report_path": str(compatibility_path),
                                  "report_sha256": compatibility_hash,
                                  "index_embedding_identity": "unknown"}
                                 if compatibility_path else
                                 {"status": "not_supplied", "index_embedding_identity": "unknown"}),
        "archive_metadata": {"lesson_count": len(lessons), "lessons_content_sha256": lesson_content_hash,
                             "chunk_count": len(chunks), "chunk_ids": chunk_ids,
                             "chunk_manifest_sha256": chunk_manifest_hash,
                             "graphiti_lesson_count": len(graph_lesson_ids),
                             "missing_graphiti_lesson_ids": missing_graph},
        "inputs": {"requests_sha256": request_hash, "database_name_file_sha256": db_name_hash,
                   "vector_compatibility_report_sha256": compatibility_hash},
        "code_provenance": _code_provenance(), "libraries": _library_provenance(), "results": [],
        "coverage": {"records_planned": len(requests) * len(MODES), "records_written": 0,
                     "records_with_errors": 0,
                     "query_attempts_planned": len(requests) * len(MODES) * (repeats + 1),
                     "query_attempts_recorded": 0, "query_attempts_succeeded": 0,
                     "query_attempts_failed": 0},
        "interpretation": "Descriptive retrieval measurements only; no winner is selected.",
    }
    _save_report(output_path, document, create=True)

    # Metadata lookup failure is retained in the report before any query begins.
    try:
        if settings is not None:
            identities = _model_identities(runtime, model_identity_resolver)
            model_metadata["chat"] = identities["chat"]
            model_metadata["embedding"] = identities["embedding"]
            document["model"].update(model_metadata)
            document["model"]["current_model_identities_before"] = identities
            document["model"]["identity_role"] = "current_runtime"
            document["model"]["index_embedding_identity"] = "unknown"
            if compatibility_path is not None:
                document["vector_compatibility"] = _validate_vector_compatibility_report(
                    compatibility_path, identities, raw_archive, run_id, learner_id, chunks)
        elif compatibility_path is not None:
            raise ValueError("vector compatibility validation requires current model identities")
        else:
            identities = None
        if compatibility_path is None:
            document["vector_compatibility"] = {
                "status": "not_supplied", "index_embedding_identity": "unknown",
                "current_model_identities": identities}
        _save_report(output_path, document)
    except Exception as exc:
        document["status"] = "setup_error"
        document["setup_error_type"] = type(exc).__name__
        if compatibility_path is not None:
            document["vector_compatibility"] = {"status": "error", "error_type": type(exc).__name__,
                                                "report_path": str(compatibility_path),
                                                "report_sha256": compatibility_hash,
                                                "index_embedding_identity": "unknown"}
        _finalize_provenance(document, runtime, model_identity_resolver)
        _update_coverage(document, requests, repeats)
        _finish_status(document)
        _save_report(output_path, document)
        return document

    if graph_factory is None and settings is not None and password_path is not None and password_path.is_file():
        def graph_factory(model_arg):
            password = password_path.read_text(encoding="utf-8").strip()
            if not password:
                raise ValueError("configured personal graph passfile is empty")
            return make_graphiti(model_arg, password)

    async def make_workflow(mode):
        nonlocal graph, graph_adapter
        archive_for_workflow = (_ArchiveSearchTopK(raw_archive, source_top_k)
                                if source_top_k is not None else raw_archive)
        if mode == "graph_hybrid":
            if not graph_available:
                raise RuntimeError("selected run has no complete Graphiti index for every archived lesson")
            if graph_factory is None:
                raise ValueError("Graphiti client requires the configured personal graph passfile")
            graph = graph_factory(runtime)
            graph_adapter = GraphitiAdapter(graph, archive_for_workflow, model)
        factory = workflow_factory or application_memory_workflow
        workflow = factory(base_workflow, archive_for_workflow, model, graph_adapter,
                           source_order=application_lesson_order(lessons), retrieval_mode=mode)
        if source_top_k is not None:
            if not all(callable(getattr(workflow, name, None)) for name in ("retrieve", "hydrate", "select")):
                raise RuntimeError("source_top_k workflow must expose retrieve, hydrate, and select nodes")
            _install_source_top_k_graph(workflow, source_top_k, recent_lesson_ids,
                                        archive_for_workflow, mode)
        return workflow

    try:
        for request in requests:
            for mode in MODES:
                row = {"query_id": request["query_id"], "question": request["question"],
                       "expected_source_ids": request["expected_source_ids"], "mode": mode,
                       "scientific_result": False,
                       "budget_kind": budget_kind, "budget": budget, "context_tokens": context,
                       "attempts_planned": repeats + 1, "attempts_completed": 0,
                       "successful_attempt_count": 0, "failed_attempt_count": 0,
                       "candidate_source_ids": [], "selected_source_ids": [], "memory": None,
                       "channel_counts": {}, "trials": [], "status": "running"}
                document["results"].append(row)
                _update_coverage(document, requests, repeats)
                _save_report(output_path, document)
                try:
                    workflow = await make_workflow(mode)
                except Exception as exc:
                    row.update(status="setup_error", setup_error_type=type(exc).__name__,
                               attempts_planned=repeats + 1)
                    if graph is not None:
                        try:
                            await graph.close()
                        except Exception as close_exc:
                            row["graph_cleanup_error_type"] = type(close_exc).__name__
                        graph, graph_adapter = None, None
                    _update_coverage(document, requests, repeats)
                    _finish_status(document)
                    _save_report(output_path, document)
                    continue

                for trial_index in range(repeats + 1):
                    trial_name = "first_sequential" if trial_index == 0 else "warm_repeat"
                    archive_proxy = getattr(workflow, "source_top_k_archive_proxy", None)
                    if archive_proxy is not None:
                        archive_proxy.reset()
                        workflow.source_top_k_stage_timings = {}
                    started = time.perf_counter()
                    try:
                        result = await workflow.search(run_id, learner_id, request["question"],
                                                       mode="graphiti", budget_tokens=budget)
                        elapsed = time.perf_counter() - started
                        candidate_ids = _candidate_source_ids(result.get("lessons", []))
                        selected_ids = [str(item) for item in result.get("source_ids", [])]
                        trial = {"trial": trial_name, "status": "ok", "elapsed_seconds": elapsed,
                                 "candidate_source_ids": candidate_ids,
                                 "selected_source_ids": selected_ids,
                                 "memory": result.get("memory", ""),
                                 "channel_counts": _channel_counts(result.get("hits", [])),
                                 "skipped": result.get("skipped", [])}
                        if source_top_k is not None:
                            trial["source_top_k_diagnostics"] = getattr(
                                workflow, "source_top_k_diagnostics", {})
                            trial["stage_timings_seconds"] = {
                                **workflow.source_top_k_stage_timings,
                                "hydrate_plus_select_seconds": sum(
                                    workflow.source_top_k_stage_timings.get(key, 0.0)
                                    for key in ("hydrate_seconds", "select_seconds")),
                                "full_pipeline_seconds": elapsed,
                                "includes_token_count_probe": budget_kind == "tokens",
                            }
                    except Exception as exc:
                        elapsed = time.perf_counter() - started
                        trial = {"trial": trial_name, "status": "error", "elapsed_seconds": elapsed,
                                 "error_type": type(exc).__name__}
                        if source_top_k is not None:
                            trial["source_top_k_diagnostics"] = getattr(
                                workflow, "source_top_k_diagnostics", {})
                            trial["stage_timings_seconds"] = {
                                **getattr(workflow, "source_top_k_stage_timings", {}),
                                "hydrate_plus_select_seconds": sum(
                                    getattr(workflow, "source_top_k_stage_timings", {}).get(key, 0.0)
                                    for key in ("hydrate_seconds", "select_seconds")),
                                "full_pipeline_seconds": elapsed,
                                "includes_token_count_probe": budget_kind == "tokens",
                            }
                    row["trials"].append(trial)
                    row["attempts_completed"] += 1
                    row["successful_attempt_count"] += trial["status"] == "ok"
                    row["failed_attempt_count"] += trial["status"] == "error"
                    if trial_index == 0 and trial["status"] == "ok":
                        row.update(candidate_source_ids=trial["candidate_source_ids"],
                                   selected_source_ids=trial["selected_source_ids"],
                                   memory=trial["memory"], channel_counts=trial["channel_counts"],
                                   **source_recall(trial["selected_source_ids"], request["expected_source_ids"]))
                        if source_top_k is not None:
                            row["source_top_k_diagnostics"] = trial.get("source_top_k_diagnostics", {})
                        row["first_query_seconds"] = elapsed
                    elif trial_index == 0:
                        expected = request["expected_source_ids"]
                        row.update(source_recall=None,
                                   source_recall_status=("first_attempt_failed" if expected
                                                         else "no_gold_sources_first_attempt_failed"),
                                   missing_expected_source_ids=sorted(expected))
                        row["first_query_seconds"] = None
                    warm_successes = [item["elapsed_seconds"] for item in row["trials"][1:]
                                      if item["status"] == "ok"]
                    row["warm_repeat_seconds"] = {"planned": repeats,
                        "recorded": len(row["trials"]) - 1, "successful": len(warm_successes),
                        "failed": sum(item["status"] == "error" for item in row["trials"][1:]),
                        "p50": percentile(warm_successes, 50), "p95": percentile(warm_successes, 95)}
                    _update_coverage(document, requests, repeats)
                    _save_report(output_path, document)
                row["status"] = ("ok" if row["failed_attempt_count"] == 0 else
                                 "partial_error" if row["successful_attempt_count"] else "error")
                if graph is not None:
                    try:
                        await graph.close()
                    except Exception as exc:
                        row["graph_cleanup_error_type"] = type(exc).__name__
                        if row["status"] == "ok":
                            row["status"] = "partial_error"
                    graph, graph_adapter = None, None
                _update_coverage(document, requests, repeats)
                _finish_status(document)
                _save_report(output_path, document)
    finally:
        if graph is not None:
            try:
                await graph.close()
            except Exception:
                pass

    _finalize_provenance(document, runtime, model_identity_resolver)
    _update_coverage(document, requests, repeats)
    _finish_status(document)
    _save_report(output_path, document)
    return document


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name-file", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--learner-id", required=True)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context", required=True, type=int, choices=(16384, 32768))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget-kind", choices=("bytes", "tokens"), default="bytes")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--source-top-k", type=int,
                        help="Optional common cap on archive search limit and unique source chunks before hydration")
    parser.add_argument("--vector-compatibility-report", type=Path,
                        help="Optional read-only report validating archived chunk hashes and vectors against current model identities")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from psycopg2.extensions import make_dsn, parse_dsn
    from local_app.config import Settings

    settings = Settings.from_env()
    database_name = validate_db_name_file(args.db_name_file)
    params = parse_dsn(settings.database_url)
    addresses = [params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR")]
    if any(value and value not in {"localhost", "127.0.0.1", "::1"} for value in addresses):
        raise ValueError("benchmark only permits local PostgreSQL")
    database_url = make_dsn(**{**params, "dbname": database_name})
    requests = load_requests(args.requests)
    report = asyncio.run(run_benchmark(database_url=database_url, run_id=args.run_id, learner_id=args.learner_id,
                  requests=requests, output_path=args.output, context=args.context,
                  repeats=args.repeats, budget_kind=args.budget_kind, budget=args.budget,
                  settings=settings, database_name=database_name,
                  requests_file=args.requests, database_name_file=args.db_name_file,
                  source_top_k=args.source_top_k,
                  vector_compatibility_report=args.vector_compatibility_report))
    print(f"Benchmark written: {args.output}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
