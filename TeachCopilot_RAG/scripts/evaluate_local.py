"""Evaluate local hybrid retrieval against a golden set of real source IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases", data.get("items", []))
    normalized = []
    for case in cases:
        expected = case.get("expected_source", case.get("expect_source"))
        normalized.append({
            **case,
            "expected_source": expected,
            "subject_id": case.get("subject_id"),
            "topic_id": case.get("topic_id"),
            "grade": case.get("grade"),
        })
    return normalized


def source_matches(source: dict[str, Any], expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(source_matches(source, item) for item in expected)
    candidates = [str(source.get("source_file", "")), str(source.get("id", ""))]
    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(str(metadata.get(key, "")) for key in ("source_id", "source_file", "filename"))
    return any(expected == candidate or expected in Path(candidate).name for candidate in candidates)


def case_source_matches(source: dict[str, Any], case: dict[str, Any]) -> bool:
    if case.get("expected_ids") is not None:
        expected_ids = case["expected_ids"]
        ids = expected_ids if isinstance(expected_ids, list) else [expected_ids]
        return str(source.get("id", "")) in {str(identifier) for identifier in ids}
    expected_source = case.get("expected_source")
    return bool(expected_source) and source_matches(source, expected_source)


def evaluation_subject(case: dict[str, Any]) -> str | None:
    """Pass only supported corpus subjects as hard filters."""
    value = case.get("subject_id")
    return value if value in {"math", "programming", "python", "scratch"} else None


def evaluate_retrieval(
    service: Any, cases: list[dict[str, Any]], k: int = 3, repeats: int = 2,
    use_learner_hints: bool = False,
) -> dict[str, Any]:
    ranked_cases = []
    reciprocal_ranks = []
    hits1 = 0
    hitsk = 0
    evaluable = 0
    abstention_cases = 0
    correct_abstentions = 0
    for repeat in range(repeats):
        for case in cases:
            query = str(case.get("query", "")).strip()
            if not query:
                continue
            started = time.perf_counter()
            results = service.search(
                query,
                subject_id=evaluation_subject(case),
                topic_id=case.get("topic_id") if use_learner_hints else None,
                grade=case.get("grade") if use_learner_hints else None,
                limit=k,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            expected = case.get("expected_ids", case.get("expected_source"))
            if expected is None and case.get("expect_no_match"):
                abstention_cases += 1
                if not results:
                    correct_abstentions += 1
            rank = None
            if expected:
                evaluable += 1
                for position, result in enumerate(results, start=1):
                    if case_source_matches(result, case):
                        rank = position
                        break
                if rank == 1:
                    hits1 += 1
                if rank is not None:
                    hitsk += 1
                    reciprocal_ranks.append(1 / rank)
                else:
                    reciprocal_ranks.append(0.0)
            ranked_cases.append({
                "query": query,
                "repeat": repeat + 1,
                "expected_source": expected,
                "rank": rank,
                "expect_no_match": bool(case.get("expect_no_match")),
                "abstained": not bool(results),
                "latency_ms": round(elapsed_ms, 2),
                "results": [
                    {"id": row.get("id"), "source_file": row.get("source_file"),
                     "score": row.get("score"), "similarity": row.get("similarity"),
                     "topic_id": row.get("topic_id")}
                    for row in results
                ],
            })
    return {
        "evaluated_cases": len(ranked_cases),
        "source_labeled_cases": evaluable,
        "recall_at_1": hits1 / evaluable if evaluable else None,
        f"recall_at_{k}": hitsk / evaluable if evaluable else None,
        "mrr": sum(reciprocal_ranks) / evaluable if evaluable else None,
        "abstention_cases": abstention_cases,
        "correct_abstentions": correct_abstentions,
        "abstention_accuracy": correct_abstentions / abstention_cases if abstention_cases else None,
        "unsupported_false_positive_rate": (
            (abstention_cases - correct_abstentions) / abstention_cases if abstention_cases else None
        ),
        "learner_hints_used": use_learner_hints,
        "mean_latency_ms": round(
            sum(row["latency_ms"] for row in ranked_cases) / len(ranked_cases), 2
        ) if ranked_cases else None,
        "cases": ranked_cases,
    }


def _replace_settings(settings: Any, **changes: Any) -> Any:
    try:
        from dataclasses import replace
        return replace(settings, **changes)
    except (TypeError, ValueError):
        clone = type("RuntimeSettings", (), {})()
        clone.__dict__.update(vars(settings))
        clone.__dict__.update(changes)
        return clone


def ollama_inventory(settings: Any, expected_models: list[str]) -> dict[str, Any]:
    import requests
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5, allow_redirects=False
        )
        if 300 <= response.status_code < 400:
            raise RuntimeError("Ollama model inventory attempted a redirect")
        response.raise_for_status()
        models = response.json().get("models", [])
    except Exception as exc:
        return {"error": str(exc), "models": {name: None for name in expected_models}}
    by_name = {}
    for model in models:
        name = model.get("name") or model.get("model")
        if name:
            by_name[name] = {"digest": model.get("digest"), "size_bytes": model.get("size")}
    resolved = {}
    for name in expected_models:
        model = by_name.get(name)
        if model is None and ":" not in name:
            model = by_name.get(f"{name}:latest")
        resolved[name] = model
    return {"models": resolved}


def gpu_memory() -> dict[str, Any] | None:
    command = shutil.which("nvidia-smi")
    if not command:
        return None
    result = subprocess.run(
        [command, "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode:
        return {"error": result.stderr.strip() or "nvidia-smi failed"}
    devices = []
    for line in result.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) == 3:
            devices.append({"name": pieces[0], "used_mb": pieces[1], "total_mb": pieces[2]})
    return {"devices": devices}


def live(args: argparse.Namespace) -> dict[str, Any]:
    from local_app.config import Settings
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime, SentenceTransformerRuntime
    from local_app.store import Store

    settings = Settings.from_env()
    store = Store(settings.database_url)
    cases = load_cases(args.fixture)
    reports = []
    embedding_models = args.embedding_models or [settings.embedding_model]
    chat_models = args.chat_models or [settings.chat_model]
    use_learner_hints = bool(getattr(args, "use_learner_hints", False))
    scope_gate_enabled = True
    inventory = ollama_inventory(settings, list(dict.fromkeys(embedding_models + chat_models)))
    material_snapshots = [
        (str(row.get("id")), str(row.get("source_version")),
         hashlib.sha256(str(row.get("content") or "").encode("utf-8")).hexdigest())
        for row in store.material_rows()
    ]
    fingerprint_spec = {
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(
            json.dumps(material_snapshots, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "embedding_models": embedding_models,
        "chat_models": chat_models,
        "k": args.k,
        "repeats": args.repeats,
        "generation_only": bool(args.generation_only),
        "retrieval_only": bool(args.retrieval_only),
        "use_learner_hints": use_learner_hints,
        "scope_gate_policy": "teachcopilot-curriculum-scope-v1",
        "model_inventory": inventory,
        "context_tokens": getattr(settings, "context_tokens", None),
        "max_output_tokens": getattr(settings, "max_output_tokens", None),
        "minimum_similarity": getattr(settings, "minimum_similarity", None),
        "evaluator_schema": 2,
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    in_progress = None
    if getattr(args, "resume", False):
        output_path = getattr(args, "output", None)
        if not output_path or not output_path.is_file():
            raise ValueError("--resume requires an existing --output checkpoint")
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Checkpoint fingerprint does not match this evaluation configuration")
        if previous.get("status") == "complete":
            raise ValueError("This evaluation checkpoint is already complete")
        reports = list(previous.get("reports", []))
        in_progress = previous.get("in_progress")

    def checkpoint(status: str = "running", progress: dict[str, Any] | None = None) -> None:
        output_path = getattr(args, "output", None)
        if not output_path:
            return
        try:
            import psutil
            memory_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
        except ImportError:
            memory_mb = None
        payload = {
            "status": status,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "process_rss_mb": memory_mb,
                "gpu_memory": gpu_memory(),
                "ollama": ollama_inventory(
                    settings, list(dict.fromkeys(embedding_models + chat_models))
                ),
            },
            "fixture": str(args.fixture),
            "run_fingerprint": run_fingerprint,
            "retrieval_mode": "learner_hints_with_scope_gate" if use_learner_hints else "blind_query_subject_with_scope_gate",
            "scope_gate": {"enabled": scope_gate_enabled, "model": settings.chat_model,
                           "policy_version": "teachcopilot-curriculum-scope-v1"},
            "reports": reports,
            "in_progress": progress,
            "limitations": [
                "expected-term checks are lexical proxies only and cannot establish factual correctness",
                "teacher_review fields are blank; a teacher must review factuality, age fit, and source grounding",
                "a validated citation identifies a retrieved source but does not prove every answer claim is supported",
                "model memory is process RSS and may omit Ollama server memory",
                "embedding models are reindexed sequentially; each report records its own model",
                "SentenceTransformers comparisons use cached local weights only and never download models",
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output_path)
    for embedding_model in embedding_models:
        if args.retrieval_only and any(
            row.get("embedding_model") == embedding_model and "retrieval" in row for row in reports
        ):
            continue
        if not args.retrieval_only and all(
            any(row.get("embedding_model") == embedding_model and row.get("chat_model") == chat_model
                for row in reports)
            for chat_model in chat_models
        ):
            continue
        model_settings = _replace_settings(settings, embedding_model=embedding_model)
        if embedding_model.startswith("sentence-transformers:"):
            runtime = SentenceTransformerRuntime(
                model_settings, embedding_model.split(":", 1)[1]
            )
        else:
            runtime = OllamaRuntime(model_settings)
        service = RagService(store, runtime, model_settings)
        index_status_method = getattr(service, "status", None)
        current_index = index_status_method() if callable(index_status_method) else None
        if current_index and current_index.get("ready"):
            index_counts = {
                "materials": current_index.get("total", 0),
                "embedded": 0,
                "index_status": current_index,
                "reused_existing_index": True,
            }
        else:
            index_counts = service.reindex()
        saved_progress = in_progress if isinstance(in_progress, dict) and in_progress.get(
            "embedding_model"
        ) == embedding_model else None
        retrieval = (saved_progress or {}).get("retrieval")
        if retrieval is None and not args.generation_only:
            retrieval = evaluate_retrieval(
                service, cases, args.k, args.repeats, use_learner_hints=use_learner_hints
            )
        if not args.retrieval_only:
            checkpoint(progress={
                "embedding_model": embedding_model,
                "index": index_counts,
                "retrieval": retrieval,
            })
        if args.retrieval_only:
            reports.append({
                "embedding_model": embedding_model,
                "index": index_counts,
                "retrieval": retrieval,
            })
            checkpoint()
            continue
        for chat_model in chat_models:
            if any(row.get("embedding_model") == embedding_model and row.get("chat_model") == chat_model
                   for row in reports):
                continue
            chat_settings = _replace_settings(model_settings, chat_model=chat_model)
            chat_runtime = OllamaRuntime(chat_settings)
            chat_service = RagService(store, chat_runtime, chat_settings)
            saved_chat_results = (
                saved_progress.get("chat", [])
                if saved_progress and saved_progress.get("chat_model") == chat_model else []
            )
            chat_results = list(saved_chat_results)
            completed_case_keys = {
                (int(row.get("repeat", 0)), int(row.get("case_index", -1)))
                for row in chat_results
            }
            pending_progress = {
                "embedding_model": embedding_model,
                "chat_model": chat_model,
                "index": index_counts,
                "retrieval": retrieval,
                "chat": chat_results,
            }
            new_results_since_checkpoint = 0
            for repeat in range(args.repeats):
                for case_index, case in enumerate(cases):
                    if (repeat + 1, case_index) in completed_case_keys:
                        continue
                    started = time.perf_counter()
                    answer = chat_service.answer(
                        [{"role": "user", "content": case["query"]}],
                        subject_id=evaluation_subject(case),
                        topic_id=case.get("topic_id") if use_learner_hints else None,
                        grade=case.get("grade") if use_learner_hints else None,
                        limit=args.k,
                    )
                    elapsed = (time.perf_counter() - started) * 1000
                    text = answer["choices"][0]["message"]["content"]
                    references = case.get("generation_reference", {}) or {}
                    expected_terms = case.get("expected_terms") or references.get("must_include") or []
                    if isinstance(expected_terms, str):
                        expected_terms = [expected_terms]
                    expected_terms = list(expected_terms)
                    found_terms = [term for term in expected_terms if str(term).casefold() in text.casefold()]
                    must_avoid = references.get("must_avoid") or []
                    if isinstance(must_avoid, str):
                        must_avoid = [must_avoid]
                    avoided_terms_present = [term for term in must_avoid if str(term).casefold() in text.casefold()]
                    expected_reference = case.get("expected_ids", case.get("expected_source"))
                    retrieved_expected_source = (
                        any(case_source_matches(source, case) for source in answer.get("sources", []))
                        if expected_reference else None
                    )
                    expected_citation_present = (
                        any(case_source_matches(citation, case) for citation in answer.get("citations", []))
                        if expected_reference else None
                    )
                    chat_results.append({
                        "query": case["query"],
                        "repeat": repeat + 1,
                        "case_index": case_index,
                        "latency_ms": round(elapsed, 2),
                        "schema_valid": isinstance(text, str) and bool(text.strip()),
                        "expected_terms_found": found_terms,
                        "expected_terms_missing": [term for term in expected_terms if term not in found_terms],
                        "expected_terms_coverage": len(found_terms) / len(expected_terms) if expected_terms else None,
                        "expected_terms": expected_terms,
                        "expected_answer": references.get("expected_answer", case.get("expected_answer")),
                        "must_avoid": must_avoid,
                        "must_avoid_found": avoided_terms_present,
                        "expected_source_retrieved": retrieved_expected_source,
                        "expected_source_cited": expected_citation_present,
                        "validated_citations": answer.get("citations", []),
                        "answer": text,
                        "citations": answer.get("citations", []),
                        "teacher_review": {
                            "correct_and_complete": None,
                            "age_appropriate": None,
                            "grounded_in_cited_sources": None,
                            "notes": "",
                        },
                    })
                    new_results_since_checkpoint += 1
                    if new_results_since_checkpoint >= 5:
                        pending_progress["chat"] = chat_results
                        checkpoint(progress=pending_progress)
                        new_results_since_checkpoint = 0
            scored_terms = [row for row in chat_results if row["expected_terms_coverage"] is not None]
            avoid_checks = [row for row in chat_results if row["must_avoid"]]
            source_retrieval_checks = [
                row for row in chat_results if row["expected_source_retrieved"] is not None
            ]
            source_citation_checks = [
                row for row in chat_results if row["expected_source_cited"] is not None
            ]
            reports.append({
                "embedding_model": embedding_model,
                "chat_model": chat_model,
                "learner_hints_used": use_learner_hints,
                "index": index_counts,
                "retrieval": retrieval,
                "generation_summary": {
                    "schema_valid_rate": sum(row["schema_valid"] for row in chat_results) / len(chat_results)
                    if chat_results else None,
                    "mean_expected_term_coverage": sum(row["expected_terms_coverage"] for row in scored_terms)
                    / len(scored_terms) if scored_terms else None,
                    "all_expected_terms_rate": sum(row["expected_terms_coverage"] == 1 for row in scored_terms)
                    / len(scored_terms) if scored_terms else None,
                    "must_avoid_violation_rate": sum(bool(row["must_avoid_found"]) for row in avoid_checks)
                    / len(avoid_checks) if avoid_checks else None,
                    "expected_source_retrieved_rate": (
                        sum(bool(row["expected_source_retrieved"]) for row in source_retrieval_checks)
                        / len(source_retrieval_checks) if source_retrieval_checks else None
                    ),
                    "expected_source_cited_rate": (
                        sum(bool(row["expected_source_cited"]) for row in source_citation_checks)
                        / len(source_citation_checks) if source_citation_checks else None
                    ),
                    "mean_latency_ms": round(sum(row["latency_ms"] for row in chat_results) / len(chat_results), 2)
                    if chat_results else None,
                },
                "chat": chat_results,
            })
            checkpoint()
    try:
        import psutil
        memory_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except ImportError:
        memory_mb = None
    return {
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "process_rss_mb": memory_mb,
                        "gpu_memory": gpu_memory(),
                        "ollama": ollama_inventory(
                            settings, list(dict.fromkeys(embedding_models + chat_models))
                        )},
        "fixture": str(args.fixture),
        "run_fingerprint": run_fingerprint,
        "retrieval_mode": "learner_hints_with_scope_gate" if use_learner_hints else "blind_query_subject_with_scope_gate",
        "scope_gate": {"enabled": scope_gate_enabled, "model": settings.chat_model,
                       "policy_version": "teachcopilot-curriculum-scope-v1"},
        "reports": reports,
        "in_progress": None,
        "limitations": [
            "expected-term checks are lexical proxies only and cannot establish factual correctness",
            "teacher_review fields are blank; a teacher must review factuality, age fit, and source grounding",
            "a validated citation identifies a retrieved source but does not prove every answer claim is supported",
            "model memory is process RSS and may omit Ollama server memory",
            "embedding models are reindexed sequentially; each report records its own model",
            "SentenceTransformers comparisons use cached local weights only and never download models",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path,
        default=ROOT / "tests" / "fixtures" / "local_rag_golden.json",
        help="Golden queries with expected source names",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--use-learner-hints", action="store_true",
        help="Pass fixture topic and grade hints to retrieval; default compares query plus subject only",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--retrieval-only", action="store_true", help="Compare embedding models without generation")
    modes.add_argument("--generation-only", action="store_true", help="Use one chosen embedding model to compare chat models")
    parser.add_argument("--embedding-models", nargs="*", help="Ollama embedding models to compare sequentially")
    parser.add_argument("--chat-models", nargs="*", help="Ollama chat models to compare")
    parser.add_argument("--output", type=Path, help="Write detailed JSON report to this path")
    parser.add_argument("--resume", action="store_true", help="Resume a matching checkpoint from --output")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.generation_only and len(args.embedding_models or []) > 1:
        parser.error("--generation-only accepts one winning --embedding-models value")
    if not args.retrieval_only and not args.generation_only and len(args.embedding_models or []) > 1:
        parser.error("Compare embedding models with --retrieval-only, then chat models with --generation-only")
    if not args.fixture.is_file():
        parser.error(f"fixture does not exist: {args.fixture}")
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    report = live(args)
    report["status"] = "complete"
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    rendered_for_stdout = rendered
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            rendered_for_stdout = json.dumps(report, ensure_ascii=True, indent=2)
    else:
        rendered_for_stdout = json.dumps(report, ensure_ascii=True, indent=2)
    print(rendered_for_stdout)


if __name__ == "__main__":
    main()
