"""Nine-condition local pilot; reads histories and questions, never gold answers."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request
from uuid import uuid4

from dotenv import load_dotenv

from local_app.personal_graph_index import resolve_model_identities

from research_memory.graph_extension.runner import _make_tail, _memory_context, _prepare_memory
from research_memory.graph_extension_v3.runner import _prepare_v2_memory
from research_memory.runner import CallJournal

from .archive import Archive
from .backends import GraphitiAdapter, Mem0Adapter, lesson_text, make_graphiti, make_mem0
from .chunks import lesson_chunks
from .structured_chunked_v1 import prepare_chunked_v1
from .runtime import AnswerValidationError, ContextWindowOverflow, LegacyClient, LocalModel, UnsafeMemoryProbe
from .workflow import PersonalMemoryWorkflow

METHODS = ("profile_only", "tail", "summary", "structured", "structured_v2", "graph",
           "mem0", "graphiti", "graphiti_mem0")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    stream.flush()


def gpu():
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                             "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return {"status": result.returncode, "value": result.stdout.strip()}


def model_digest(runtime):
    req = Request(runtime.base_url + "/api/tags")
    with runtime.opener.open(req, timeout=10) as response:
        models = json.load(response)["models"]
    found = [m["digest"] for m in models if runtime.model in (m.get("name"), m.get("model"))]
    if len(found) != 1:
        raise ValueError("Cannot identify the local model digest")
    return found[0]


def code_hashes():
    lab = Path(__file__).parent
    research = lab.parent
    paths = sorted(lab.glob("*.py")) + sorted((research / "prompts").glob("*.txt"))
    paths += [research / "methods.py", research / "contracts.py", research / "contracts_v2.py",
              research / "clients.py", research / "runner.py",
              research / "graph_extension/runner.py", research / "graph_extension_v3/runner.py",
              lab / "requirements.lock.txt"]
    hashes = {str(path.relative_to(research)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    helper = research.parent / "local_app" / "personal_graph_index.py"
    hashes["../local_app/personal_graph_index.py"] = hashlib.sha256(helper.read_bytes()).hexdigest()
    return hashes


def validate_data(histories, requests):
    by_id = {}
    all_turns = set()
    for h in histories:
        if not h.get("data_origin", "").startswith("synthetic") or h["history_id"] in by_id:
            raise ValueError("Expected unique synthetic histories")
        if not 8 <= len(h["lessons"]) <= 12:
            raise ValueError("Expected 8–12 lessons")
        by_id[h["history_id"]] = h
        for lesson in h["lessons"]:
            Archive._validate_lesson(lesson)
            for turn in lesson["turns"]:
                if turn["turn_id"] in all_turns:
                    raise ValueError("Turn IDs must be globally unique")
                all_turns.add(turn["turn_id"])
    seen = set()
    for request in requests:
        if request["request_id"] in seen or request["history_id"] not in by_id:
            raise ValueError("Invalid question identity")
        if any(key in request for key in ("expected", "gold", "source_turn_ids")):
            raise ValueError("Reference answers must not enter the generator")
        seen.add(request["request_id"])
    return by_id


async def prepare_indexes(archive, runtime, history, run_id, graph, mem, status_stream):
    child = history["learner_id"]
    errors = {"archive": None, "graphiti": None if graph else "backend_unavailable",
              "mem0": None if mem else "backend_unavailable"}
    for lesson in history["lessons"]:
        runtime.call_scope = {"history_id": history["history_id"], "lesson_id": lesson["lesson_id"], "stage": "ingest"}
        try:
            archive.put(run_id, child, lesson)
            parts = lesson_chunks(lesson)
            vectors = []
            for start in range(0, len(parts), 16):
                vectors.extend(await asyncio.to_thread(runtime.embed_many,
                    [lesson_text(part) for part in parts[start:start+16]]))
            archive.put_chunks(run_id, child, lesson, parts, vectors)
            append(status_stream, {"history_id": history["history_id"], "lesson_id": lesson["lesson_id"],
                                   "stage": "archive", "status": "ok", "source_fragments": len(parts)})
        except Exception as exc:
            errors["archive"] = type(exc).__name__
            append(status_stream, {"history_id": history["history_id"], "lesson_id": lesson["lesson_id"],
                                   "stage": "archive", "status": "error", "error": type(exc).__name__})
            break
        for name, adapter in (("graphiti", graph), ("mem0", mem)):
            if errors[name] is not None:
                continue
            try:
                if name == "graphiti":
                    outcome = await adapter.ingest(run_id, child, lesson)
                else:
                    outcome = await asyncio.to_thread(adapter.ingest, run_id, child, lesson)
                append(status_stream, {"history_id": history["history_id"], "lesson_id": lesson["lesson_id"],
                                       "stage": name, "status": "ok", "outcome": outcome})
            except Exception as exc:
                errors[name] = type(exc).__name__
                append(status_stream, {"history_id": history["history_id"], "lesson_id": lesson["lesson_id"],
                                       "stage": name, "status": "error", "error": type(exc).__name__})
    return errors


async def initialize_backends(methods, runtime, archive, db_url, root, extraction, status_stream):
    """Initialize only backends required by selected methods."""
    graph = graph_adapter = mem_adapter = None
    if {"graphiti", "graphiti_mem0"}.intersection(methods):
        try:
            password = (root / ".local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt").read_text().strip()
            graph = make_graphiti(runtime, password)
            graph_adapter = GraphitiAdapter(graph, archive, runtime)
            await graph_adapter.initialize()
        except Exception as exc:
            graph_adapter = None
            append(status_stream, {"stage": "graphiti_initialize", "status": "error", "error": type(exc).__name__})
    if {"mem0", "graphiti_mem0"}.intersection(methods):
        try:
            memory = make_mem0(runtime, db_url, root / ".local/personal-memory/mem0", extraction=extraction)
            mem_adapter = Mem0Adapter(memory, archive, extraction=extraction)
        except Exception as exc:
            append(status_stream, {"stage": "mem0_initialize", "status": "error", "error": type(exc).__name__})
    return graph, graph_adapter, mem_adapter


def legacy_context(method, history, question, legacy, v2, runtime, budget_tokens):
    if method == "profile_only":
        return "", set(), 0, []
    # Existing writers still receive a character budget; selection is then
    # measured with Gemma and reduced until the common token limit is met.
    char_budget = 12000 if budget_tokens == 4096 else 24000
    events = []
    while char_budget >= 128:
        if method == "structured_v2":
            v2.budget = char_budget
            v2.events.clear()
            selected = v2.context(question["topic_id"], include_incidental_facts=True)
            text, ids, events = selected.text, selected.evidence_ids, list(v2.events)
        else:
            tail = _make_tail(history, char_budget)
            text, ids, events = _memory_context(method, question, legacy["records"], legacy["graph"],
                                               legacy["sources"], tail, legacy["summary"], char_budget)
        try:
            tokens = runtime.count(text)
        except (ContextWindowOverflow, UnsafeMemoryProbe):
            tokens = budget_tokens + 1
        if tokens <= budget_tokens:
            return text, ids, tokens, events
        if method == "summary":
            raise ValueError("Summary exceeds measured memory budget")
        char_budget = int(char_budget * .75)
    raise ValueError("No safe legacy context fits the memory budget")


def validate_structured_output_limit(variant, context, limit, source_ids="canonical"):
    if source_ids not in {"canonical", "compact_v1"} or (source_ids != "canonical" and variant != "chunked_v1"):
        raise ValueError("Compact source IDs require chunked_v1")
    if type(limit) is not int or limit not in {2048, 4096, 8192}:
        raise ValueError("Unsupported structured V1 output limit")
    if limit != 2048 and (variant != "chunked_v1" or context != 32768):
        raise ValueError("Higher structured V1 output limits require chunked_v1 and 32k context")


async def run(args):
    validate_structured_output_limit(args.structured_ingest, args.context, args.structured_v1_output_tokens, args.structured_v1_source_ids)
    load_dotenv(Path(__file__).resolve().parents[2] / ".local.env", override=False)
    db_url = os.environ["DATABASE_URL"]
    histories, questions = read_jsonl(args.histories), read_jsonl(args.requests)
    by_id = validate_data(histories, questions)
    selected = args.history_id or list(by_id)
    if not selected or any(x not in by_id for x in selected):
        raise ValueError("Unknown history")
    methods = args.methods or list(METHODS)
    budgets = [("fixed", 4096), ("expanded", args.context - 4096)] if args.budget == "both" else [
        (args.budget, 4096 if args.budget == "fixed" else args.context - 4096)]
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = "personal-" + uuid4().hex[:16]
    manifest = {"run_id": run_id, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "gemma4:12b", "context_tokens": args.context, "budgets": budgets, "methods": methods,
        "history_ids": selected, "gold_loaded": False, "scientific_result": False,
        "structured_ingest": args.structured_ingest, "mem0_extraction": args.mem0_extraction,
        "structured_v1_max_output_tokens": args.structured_v1_output_tokens,
        "structured_v1_source_ids": args.structured_v1_source_ids,
        "model_identity_protocol": "dual_ollama_v1",
        "code_sha256_before": code_hashes(),
        "histories_sha256": hashlib.sha256(args.histories.read_bytes()).hexdigest(),
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(), "gpu_before": gpu()}
    manifest_path = args.output / "manifest.json"
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_manifest()
    root = Path(__file__).resolve().parents[4]
    archive = Archive(db_url)
    graph = None
    try:
        archive.initialize()
        with (args.output / "calls.jsonl").open("w", encoding="utf-8") as calls, \
             (args.output / "legacy_calls.jsonl").open("w", encoding="utf-8") as legacy_calls, \
             (args.output / "ingestion.jsonl").open("w", encoding="utf-8") as ingestion, \
             (args.output / "results.jsonl").open("w", encoding="utf-8") as results, \
             ((args.output / "compact_id_calls.jsonl").open("w", encoding="utf-8")
                if args.structured_v1_source_ids == "compact_v1" else nullcontext(None)) as compact_stream:
            runtime = LocalModel(context=args.context, call_stream=calls)
            manifest["model_identities_before"] = resolve_model_identities(runtime)
            manifest["model_digest_before"] = manifest["model_identities_before"]["chat"]["digest"].removeprefix("sha256:").lower()
            # A token-count probe is trustworthy only if oversized prompts are
            # rejected rather than silently shortened by this server version.
            manifest["truncation_guard"] = runtime.verify_no_truncation()
            save_manifest()
            graph, graph_adapter, mem_adapter = await initialize_backends(
                methods, runtime, archive, db_url, root, args.mem0_extraction, ingestion)
            workflow = PersonalMemoryWorkflow(archive, runtime, graphiti=graph_adapter, mem0=mem_adapter)
            journal = CallJournal(LegacyClient(runtime), legacy_calls, max_calls=args.max_calls)
            rows = error_rows = 0
            for history_id in selected:
                history = by_id[history_id]
                history_questions = [q for q in questions if q["history_id"] == history_id]
                if args.question_limit:
                    history_questions = history_questions[:args.question_limit]
                failures = await prepare_indexes(archive, runtime, history, run_id,
                                                 graph_adapter, mem_adapter, ingestion)
                runtime.call_scope = {"history_id": history_id, "stage": "legacy_ingest"}
                legacy_methods = tuple(m for m in methods
                    if m in {"tail", "summary", "structured", "graph", "profile_only"})
                if args.structured_ingest == "legacy" or not {"structured", "graph"}.intersection(methods):
                    legacy = await asyncio.to_thread(_prepare_memory, history, legacy_methods, journal, 12000)
                else:
                    # Summary and tail keep their original lesson boundaries.
                    legacy = await asyncio.to_thread(_prepare_memory, history,
                        tuple(m for m in legacy_methods if m not in {"structured", "graph"}), journal, 12000)
                    try:
                        chunked = await asyncio.to_thread(prepare_chunked_v1, history, journal, 12000,
                            extraction_client=LegacyClient(runtime, max_output_tokens=args.structured_v1_output_tokens),
                            source_id_mode=args.structured_v1_source_ids,
                            compact_trace=(lambda value: append(compact_stream, value)) if compact_stream else None)
                        for key in ("records", "graph", "extraction_errors", "extraction_warnings", "fragment_plan"):
                            legacy[key] = chunked[key]
                    except Exception as exc:
                        # Planning failures must remain visible as dependent result failures.
                        legacy["extraction_errors"].append({"type": type(exc).__name__,
                            "message": str(exc)[:300], "code": "fragment_planning_failed"})
                if "structured_v2" in methods:
                    v2, v2_errors = await asyncio.to_thread(_prepare_v2_memory, history, journal, 12000)
                else:
                    v2, v2_errors = None, []
                append(ingestion, {"history_id": history_id, "stage": "legacy_ingest", "status": "diagnostic",
                    "summary_errors": legacy["summary_errors"], "structured_errors": legacy["extraction_errors"],
                    "structured_warnings": legacy["extraction_warnings"],
                    "structured_fragment_plan": legacy.get("fragment_plan"),
                    "structured_partial_records": (legacy["records"] if legacy.get("fragment_plan")
                        and legacy["extraction_errors"] else None), "v2_errors": v2_errors,
                    "v2_warnings": list(v2.events) if v2 else []})
                for question in history_questions:
                    for label, budget in budgets:
                        for method in methods:
                            started = time.perf_counter()
                            runtime.call_scope = {"history_id": history_id, "request_id": question["request_id"],
                                                  "method": method, "budget": label, "stage": "answer"}
                            row = {"history_id": history_id, "request_id": question["request_id"],
                                   "question": question["text"], "category": question["category"],
                                   "method": method, "budget": label, "budget_tokens": budget,
                                   "status": "ok", "answer": None, "scientific_result": False}
                            try:
                                if method in {"mem0", "graphiti", "graphiti_mem0"}:
                                    required = ["archive"] + (["graphiti"] if "graphiti" in method else []) + (
                                        ["mem0"] if "mem0" in method else [])
                                    missing = {name: failures[name] for name in required if failures[name]}
                                    if missing:
                                        raise RuntimeError("index_failure:" + json.dumps(missing))
                                    found = await workflow.search(run_id, history["learner_id"], question["text"],
                                                                  mode=method, budget_tokens=budget)
                                    row.update({key: found[key] for key in ("hits", "unresolved", "lessons", "memory",
                                        "memory_tokens", "source_ids", "skipped")})
                                    ids = set(found["source_ids"])
                                else:
                                    if method == "summary" and legacy["summary_errors"]:
                                        raise RuntimeError("summary_ingest_failed")
                                    if method in {"structured", "graph"} and legacy["extraction_errors"]:
                                        raise RuntimeError("structured_ingest_failed")
                                    if method == "structured_v2" and v2_errors:
                                        raise RuntimeError("structured_v2_ingest_failed")
                                    text, ids, tokens, events = await asyncio.to_thread(
                                        legacy_context, method, history, question, legacy, v2, runtime, budget)
                                    row.update(memory=text, memory_tokens=tokens, source_ids=sorted(ids), events=events)
                                    if method in {"structured", "graph"}:
                                        row["inventory"] = legacy["records"]
                                    elif method == "structured_v2":
                                        row["inventory"] = v2.records
                                    elif method == "summary":
                                        row["inventory"] = legacy["summary"]
                                profile = {key: value for key, value in history["profile_as_of_cutoff"].items()
                                           if key not in {"topic_edges", "allowed_topic_ids"}}
                                row["answer"] = await asyncio.to_thread(runtime.answer, question["text"], row["memory"], ids, profile)
                            except Exception as exc:
                                row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:300])
                                if isinstance(exc, AnswerValidationError):
                                    row.update(invalid_answer=exc.answer, raw_model_answer=exc.raw_text)
                                error_rows += 1
                            row["elapsed_seconds"] = round(time.perf_counter() - started, 4)
                            append(results, row)
                            rows += 1
                manifest["completed_histories"] = manifest.get("completed_histories", []) + [history_id]
                manifest["result_rows"] = rows
                manifest["error_rows"] = error_rows
                save_manifest()
            if compact_stream is not None:
                manifest["compact_id_calls_sha256"] = hashlib.sha256(
                    (args.output / "compact_id_calls.jsonl").read_bytes()).hexdigest()
            manifest["model_identities_after"] = resolve_model_identities(runtime)
            manifest["model_identities_match"] = manifest["model_identities_after"] == manifest["model_identities_before"]
            manifest["model_digest_after"] = manifest["model_identities_after"]["chat"]["digest"].removeprefix("sha256:").lower()
            manifest["model_digest_matches"] = manifest["model_digest_after"] == manifest["model_digest_before"]
            manifest["code_sha256_after"] = code_hashes()
            manifest["code_matches"] = manifest["code_sha256_before"] == manifest["code_sha256_after"]
            manifest["status"] = ("model_or_code_changed" if not manifest["model_identities_match"] or not manifest["model_digest_matches"] or not manifest["code_matches"] else
                                  "complete_with_errors" if error_rows else "complete")
    except Exception as exc:
        manifest.update(status="interrupted", error_type=type(exc).__name__, error=str(exc)[:300])
        raise
    finally:
        if graph is not None:
            await graph.close()
        manifest["gpu_after"] = gpu()
        save_manifest()
    return manifest


def main():
    data = Path(__file__).parent / "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--histories", type=Path, default=data / "histories.dev.jsonl")
    parser.add_argument("--requests", type=Path, default=data / "requests.dev.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-id", action="append")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--context", type=int, choices=(16384, 32768), default=16384)
    parser.add_argument("--budget", choices=("fixed", "expanded", "both"), default="both")
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--mem0-extraction", choices=("legacy", "strict_v1"), default="legacy")
    parser.add_argument("--structured-ingest", choices=("legacy", "chunked_v1"), default="legacy")
    parser.add_argument("--structured-v1-source-ids", choices=("canonical", "compact_v1"), default="canonical")
    parser.add_argument("--structured-v1-output-tokens", type=int, choices=(2048, 4096, 8192), default=2048)
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args()
    try:
        validate_structured_output_limit(args.structured_ingest, args.context, args.structured_v1_output_tokens, args.structured_v1_source_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.question_limit is not None and args.question_limit <= 0:
        parser.error("question limit must be positive")
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=True))


if __name__ == "__main__":
    main()
