"""One finite local application batch with a traced operation-selection contract.

Only ``case.public`` reaches RagService. Private expectations are used after the
single response attempt. A technical response or matching route does not verify
the model's interpretation of intent or the quality of its teaching. The old
strict path may return without operation selection; every real selection call is
traced separately from scope resolution and answer generation.
The elapsed budget is cooperative; HTTP timeouts bound socket inactivity and
do not promise an absolute wall deadline for a slow response or scheduler work.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_verified_learning_tools_v1 import (
    EXPECTED_MODEL_IDENTITY, FiniteBatchBudgetError, SourceIntegrityError,
    error_record, fixture_cases as _fixture_cases, fixture_path, git_commit,
    mechanical_diagnostic, source_hashes as _source_hashes, utc_now, write_json,
)

FIXTURE = "tests/fixtures/verified_learning_operations_v2.json"
EXPECTED_CASES = 24
MAX_GENERATION_CALLS = 84
MAX_SECONDS = 15 * 60
MAX_REQUEST_SECONDS = 120
GENERATION_PHASES = ("scope_resolution", "operation_selection", "answer_generation")


def source_hashes(fixture_name: str | Path = FIXTURE) -> dict[str, str]:
    """Freeze this runner, its imported helper, and all application modules/prompts."""
    hashes = _source_hashes(fixture_name)
    for path in (Path(__file__).resolve(), ROOT / "scripts/run_verified_learning_tools_v1.py"):
        hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def fixture_cases(fixture: object, expected_count: int = EXPECTED_CASES) -> list[tuple[dict, dict]]:
    """Validate exactly one fixed 24-case fixture and detach its public projection."""
    if expected_count != EXPECTED_CASES:
        raise ValueError("operation contract batch requires exactly 24 cases")
    return _fixture_cases(fixture, EXPECTED_CASES)


def new_counters() -> dict[str, int]:
    return {"generation_calls": 0, **{phase + "_calls": 0 for phase in GENERATION_PHASES},
            "embedding_calls": 0}


def counter_delta(after: dict, before: dict) -> dict[str, int]:
    """Reject missing or inconsistent accounting before publishing a case row."""
    delta = {key: after[key] - before[key] for key in new_counters()}
    if (any(type(value) is not int or value < 0 for value in delta.values())
            or delta["generation_calls"] != sum(delta[phase + "_calls"] for phase in GENERATION_PHASES)):
        raise RuntimeError("inconsistent_generation_phase_counters")
    return delta


def public_material_snapshot_hash(store, remaining) -> str:
    """Freeze every active retrieval row, including vector and full-text inputs."""
    remaining()
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id,content,source_file,topic_id,grade,subject_id,source_version,
            embedding_model,embedding_dimension,embedding::text AS embedding,
            search_document::text AS search_document,metadata,raw_snapshot
            FROM teachcopilot.materials WHERE active ORDER BY id""")
        rows = [dict(row) for row in cur.fetchall()]
    remaining()
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str,
                         separators=(",", ":")).encode("utf-8")
    result = hashlib.sha256(encoded).hexdigest()
    remaining()
    return result


def bounded_public_store_class(base_store, *, make_dsn, database_url: str,
                               base_options: str, remaining):
    """Bound only read-only public DB connections; scheduler writes stay unchanged."""
    class BoundedPublicStore(base_store):
        @contextmanager
        def _connect(self):
            statement_ms = max(1, min(MAX_SECONDS * 1000, int(remaining() * 1000)))
            options = (base_options + " -c default_transaction_read_only=on"
                       f" -c statement_timeout={statement_ms} -c lock_timeout=5000").strip()
            self.dsn = make_dsn(database_url, options=options, connect_timeout=5)
            try:
                with super()._connect() as conn:
                    yield conn
            finally:
                remaining()

    return BoundedPublicStore


def traced_runtime_class(base_runtime, *, current: dict, counters: dict, calls: list,
                         output_dir: Path, remaining, generation_enabled: bool = True):
    """Wrap the actual runtime, retaining failed selection even if RAG catches it.

    The base class is supplied to allow offline contract tests without importing
    the application runtime or opening a database or model connection.
    """
    class TracedRuntime(base_runtime):
        def resolve_chat_query(self, *args, **kwargs):
            previous = current["phase"]
            current["phase"] = "scope_resolution"
            try:
                result = super().resolve_chat_query(*args, **kwargs)
                current["resolution"] = deepcopy(result)
                return result
            finally:
                current["phase"] = previous

        def select_learning_operation(self, *args, **kwargs):
            previous = current["phase"]
            current["phase"] = "operation_selection"
            selection = {"status": "attempted", "decision": None, "error": None,
                         "intent_verified": False}
            current["operation_selection"] = selection
            try:
                result = super().select_learning_operation(*args, **kwargs)
                selection.update(status="returned", decision=deepcopy(result))
                return result
            except Exception as exc:
                selection.update(status="error", error=error_record(exc))
                if getattr(exc, "raw_response", None) is not None:
                    selection["raw_response"] = deepcopy(exc.raw_response)
                raise
            finally:
                current["phase"] = previous

        def _post(self, endpoint, payload):
            remaining()
            if endpoint != "/api/chat":
                if "embed" in endpoint:
                    counters["embedding_calls"] += 1
                return super()._post(endpoint, payload)
            if not generation_enabled:
                raise RuntimeError("preflight_generation_forbidden")
            phase = current["phase"]
            if phase not in GENERATION_PHASES:
                raise RuntimeError("unrecognized_generation_phase")
            if counters["generation_calls"] >= MAX_GENERATION_CALLS:
                raise FiniteBatchBudgetError("finite_learning_operations_generation_budget_exhausted")
            counters["generation_calls"] += 1
            counters[phase + "_calls"] += 1
            call_started = time.perf_counter()
            row = {"call_number": counters["generation_calls"], "case_id": current.get("case_id"),
                   "purpose": phase, "endpoint": endpoint, "request": deepcopy(payload),
                   "started_at": utc_now()}
            try:
                response = super()._post(endpoint, payload)
                row["response"] = deepcopy(response)
                return response
            except Exception as exc:
                row["error"] = error_record(exc)
                raise
            finally:
                row["http"] = deepcopy(self.session.last_post)
                row["elapsed_seconds"] = time.perf_counter() - call_started
                calls.append(row)
                with (output_dir / "ollama_http.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        def chat(self, *args, **kwargs):
            left = remaining()
            if self.generation_scheduler is not None:
                self.generation_scheduler.wait_timeout = min(MAX_REQUEST_SECONDS, left)
            return super().chat(*args, **kwargs)

    return TracedRuntime


def run(args: argparse.Namespace) -> dict:
    import requests
    from psycopg2.extensions import make_dsn
    from local_app.answer_adapter import AnswerAdapter
    from local_app.config import Settings
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from scripts.verify_memory import require_local_dsn

    selected_fixture = fixture_path(Path(getattr(args, "fixture", FIXTURE)))
    fixture_name = selected_fixture.relative_to(ROOT).as_posix()
    expected_count = getattr(args, "expected_cases", EXPECTED_CASES)
    if expected_count != EXPECTED_CASES:
        raise ValueError("operation contract batch requires exactly 24 cases")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    counters = new_counters()
    current: dict = {"phase": "answer_generation"}
    calls: list[dict] = []
    initial_hashes = initial_commit = initial_corpus = initial_embedding_identity = None
    runtime = store = None
    report = {
        "status": "initializing", "label": args.label, "created_at": utc_now(),
        "data_origin": "synthetic_unreviewed", "human_reviewed": False,
        "semantic_assessment": "not_human_reviewed", "paid_provider_calls": 0,
        "external_provider_disabled": True, "preflight_only": args.preflight_only,
        "expected_cases": expected_count, "fixture_path": fixture_name, "attempts_per_case": 1,
        "retries": 0, "max_generation_calls": MAX_GENERATION_CALLS, "max_elapsed_seconds": MAX_SECONDS,
        "max_request_timeout_seconds": MAX_REQUEST_SECONDS, "results": [],
        "technical_ok_does_not_mean_task_correct": True,
        "operation_selection_interpretation_verified": False,
        "elapsed_budget_is_cooperative": True,
        "absolute_hard_wall_deadline_guaranteed": False,
        "http_timeout_kind": "requests_socket_inactivity_not_total_wall_time",
        "public_database_statement_timeout": "remaining_elapsed_budget_milliseconds",
        "public_database_lock_timeout_milliseconds": 5000,
        "generation_accounting": "generation_calls = scope_resolution_calls + operation_selection_calls + answer_generation_calls; embedding_calls are separate",
    }

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise FiniteBatchBudgetError("finite_learning_operations_elapsed_budget_exhausted")
        return left

    def corpus_hash() -> str:
        return public_material_snapshot_hash(store, remaining)

    def assert_unchanged() -> None:
        if initial_hashes != source_hashes(fixture_name):
            raise SourceIntegrityError("source_or_fixture_changed_during_fixed_batch")
        if initial_commit != git_commit():
            raise SourceIntegrityError("git_head_changed_during_fixed_batch")
        if runtime.model_identity(runtime.settings.chat_model) != EXPECTED_MODEL_IDENTITY:
            raise SourceIntegrityError("model_identity_changed_during_fixed_batch")
        if runtime.embedding_model_identity() != initial_embedding_identity:
            raise SourceIntegrityError("embedding_model_identity_changed_during_fixed_batch")
        if initial_corpus != corpus_hash():
            raise SourceIntegrityError("public_materials_changed_during_fixed_batch")

    class RecordingSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.trust_env = False
            self.last_post = None

        def post(self, url, **kwargs):
            self.last_post = None
            kwargs["timeout"] = min(float(kwargs.get("timeout", MAX_REQUEST_SECONDS)), MAX_REQUEST_SECONDS, remaining())
            response = super().post(url, **kwargs)
            if url.endswith("/api/chat"):
                # Read the literal HTTP body before OllamaRuntime calls json() or
                # any answer normalization takes place. Base64 preserves bytes.
                self.last_post = {"status_code": response.status_code, "raw_body": response.text,
                                  "raw_body_base64": base64.b64encode(response.content).decode("ascii")}
            return response

        def get(self, url, **kwargs):
            kwargs["timeout"] = min(float(kwargs.get("timeout", MAX_REQUEST_SECONDS)), MAX_REQUEST_SECONDS, remaining())
            return super().get(url, **kwargs)

    TracedRuntime = traced_runtime_class(
        OllamaRuntime, current=current, counters=counters, calls=calls, output_dir=out,
        remaining=remaining, generation_enabled=not args.preflight_only)

    class PublicMaterials:
        """Only read-only public retrieval is exposed to RagService."""
        def embedding_index_status(self, model_identity):
            return store.embedding_index_status(model_identity)

        def hybrid_search(self, **kwargs):
            return store.hybrid_search(**kwargs)

    def save_progress() -> None:
        counter_delta(counters, new_counters())
        report.update(counters, elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        write_json(out / "PROCESSING.json", {
            "status": report["status"], "label": args.label, "updated_at": utc_now(),
            "recorded_cases": len(report["results"]), "expected_cases": expected_count,
            "semantic_assessment": "not_human_reviewed", "paid_provider_calls": 0, **counters,
        })

    try:
        initial_hashes = source_hashes(fixture_name)
        initial_commit = git_commit()
        fixture = json.loads(selected_fixture.read_text(encoding="utf-8"))
        prepared = fixture_cases(fixture, expected_count)
        if not callable(getattr(OllamaRuntime, "select_learning_operation", None)):
            raise RuntimeError("application_runtime_has_no_operation_selection_contract")
        settings = replace(Settings.from_env(), request_timeout=MAX_REQUEST_SECONDS)
        if settings.chat_model != "gemma4:12b" or settings.context_tokens != 8192:
            raise ValueError("batch requires preregistered gemma4:12b and 8192-token context")
        params = require_local_dsn(settings.database_url)
        BoundedPublicStore = bounded_public_store_class(
            Store, make_dsn=make_dsn, database_url=settings.database_url,
            base_options=params.get("options", ""), remaining=remaining)
        store = BoundedPublicStore(settings.database_url)
        runtime = TracedRuntime(settings, session=RecordingSession())
        # The explicit local adapter bypasses external-provider environment config.
        service = RagService(PublicMaterials(), runtime, settings, answer_adapter=AnswerAdapter(runtime))
        model_identity = runtime.model_identity(settings.chat_model)
        if model_identity != EXPECTED_MODEL_IDENTITY:
            raise ValueError("local generation model digest differs from the fixed model")
        initial_embedding_identity = runtime.embedding_model_identity()
        initial_corpus = corpus_hash()
        preregistered = {
            "kind": "finite_application_learning_operation_batch_not_memory_method_comparison",
            "created_at": utc_now(), "label": args.label, "git_commit": initial_commit,
            "fixture": fixture, "fixture_path": fixture_name, "expected_cases": expected_count,
            "sha256": initial_hashes, "public_materials_sha256": initial_corpus,
            "model_identity": model_identity, "model_manifest": runtime.manifest(),
            "embedding_model_identity": initial_embedding_identity,
            "public_materials_hash_scope": "all active content, source fields, metadata, raw snapshots, embedding vectors, dimensions, model identities and full-text search documents",
            "temperature": 0, "seed": 42, "think": False,
            "max_generation_calls": MAX_GENERATION_CALLS, "max_elapsed_seconds": MAX_SECONDS,
            "max_request_timeout_seconds": MAX_REQUEST_SECONDS, "attempts_per_case": 1, "retries": 0,
            "generation_phases": list(GENERATION_PHASES),
            "gold_policy": "Only each case.public is passed to RagService; identifiers, categories and private_gold never enter generation.",
            "material_database": "loopback PostgreSQL public materials; read-only transactions",
            "other_database_access": "normal global generation-admission scheduler only",
            "external_provider_disabled": True, "paid_provider_calls": 0,
            "human_reviewed": False, "semantic_assessment": "not_human_reviewed",
            "technical_ok_does_not_mean_task_correct": True,
            "operation_selection_interpretation_verified": False,
            "elapsed_budget_is_cooperative": True,
            "absolute_hard_wall_deadline_guaranteed": False,
            "http_timeout_kind": "requests_socket_inactivity_not_total_wall_time",
            "public_database_statement_timeout": "remaining_elapsed_budget_milliseconds",
            "public_database_lock_timeout_milliseconds": 5000,
        }
        write_json(out / "preregistered.json", preregistered)
        write_json(out / selected_fixture.name, fixture)
        report["index_status"] = service.status()
        if not report["index_status"]["ready"]:
            raise RuntimeError("real local material embedding index is not ready")
        assert_unchanged()
        if args.preflight_only:
            if counters["generation_calls"]:
                raise RuntimeError("preflight unexpectedly attempted generation")
            report["status"] = "preflight_passed"
            return report
        report["status"] = "running"
        save_progress()
        stop_error = None
        for case, public in prepared:
            case_id = case["case_id"]
            current.update(case_id=case_id, phase="answer_generation", resolution=None,
                           operation_selection={"status": "not_attempted", "decision": None,
                                                "error": None, "intent_verified": False})
            before = dict(counters)
            call_start = len(calls)
            case_started = time.perf_counter()
            row = {"case_id": case_id, "category": case.get("category"), "status": "error",
                   "public_input": deepcopy(public), "human_reviewed": False,
                   "semantic_assessment": "not_human_reviewed"}
            try:
                if stop_error is not None:
                    row["status"] = "not_attempted_after_batch_stop"
                    row["error"] = deepcopy(stop_error)
                else:
                    remaining()
                    assert_unchanged()
                    if counters["generation_calls"] >= MAX_GENERATION_CALLS:
                        raise FiniteBatchBudgetError("finite_learning_operations_generation_budget_exhausted")
                    row["response"] = service.answer(**public)
                    backend = (row["response"].get("answer_routing") or {}).get("backend", "local")
                    if backend not in {"local", "deterministic_local"}:
                        raise RuntimeError("nonlocal answer routing is forbidden in this batch")
                    row["status"] = "ok"
            except Exception as exc:
                row["error"] = error_record(exc)
                if getattr(exc, "raw_response", None) is not None:
                    row["raw_response"] = deepcopy(exc.raw_response)
                if isinstance(exc, (FiniteBatchBudgetError, SourceIntegrityError)):
                    stop_error = error_record(exc)
                    if counters == before:
                        row["status"] = "not_attempted_budget_exhausted" if isinstance(exc, FiniteBatchBudgetError) else "not_attempted_source_changed"
            row.update(counter_delta(counters, before))
            row["elapsed_seconds"] = time.perf_counter() - case_started
            row["scope_resolution"] = deepcopy(current.get("resolution"))
            row["operation_selection"] = deepcopy(current["operation_selection"])
            row["model_call_numbers"] = [call["call_number"] for call in calls[call_start:]]
            if row.get("response") is not None:
                row["tool_selection"] = deepcopy(row["response"].get("tool_selection"))
                try:
                    row["mechanical_diagnostic"] = mechanical_diagnostic(
                        row["response"], row["scope_resolution"], case["private_gold"],
                        {key: row[key] for key in counters})
                    row["mechanical_diagnostic"]["operation_selection_interpretation_verified"] = False
                except Exception as exc:
                    # Diagnosis must never erase or regenerate a real response.
                    row["mechanical_diagnostic_error"] = error_record(exc)
            report["results"].append(row)
            write_json(out / (case_id + ".json"), row)
            save_progress()
            print(f"{case_id}: {row['status']}", flush=True)
        report["status"] = "completed" if stop_error is None else "stopped_with_unattempted_cases"
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = error_record(exc)
        return report
    finally:
        if initial_hashes is not None:
            try:
                final_hashes = source_hashes(fixture_name)
                changed = sorted(name for name in set(initial_hashes) | set(final_hashes)
                                 if initial_hashes.get(name) != final_hashes.get(name))
                final_commit = git_commit()
                report["source_integrity"] = {
                    "unchanged": not changed and initial_commit == final_commit,
                    "changed_paths": changed, "start_git_commit": initial_commit, "end_git_commit": final_commit,
                    "start_sha256": initial_hashes, "end_sha256": final_hashes,
                }
                if changed or initial_commit != final_commit:
                    report["status"] = "invalid_source_changed_during_batch"
                if runtime is not None and initial_corpus is not None:
                    final_identity = runtime.model_identity(runtime.settings.chat_model)
                    final_embedding_identity = runtime.embedding_model_identity()
                    final_corpus = corpus_hash()
                    report["data_integrity"] = {
                        "model_identity_unchanged": final_identity == EXPECTED_MODEL_IDENTITY,
                        "end_model_identity": final_identity,
                        "embedding_model_identity_unchanged": final_embedding_identity == initial_embedding_identity,
                        "start_embedding_model_identity": initial_embedding_identity,
                        "end_embedding_model_identity": final_embedding_identity,
                        "public_materials_unchanged": initial_corpus == final_corpus,
                        "start_public_materials_sha256": initial_corpus, "end_public_materials_sha256": final_corpus,
                    }
                    if (final_identity != EXPECTED_MODEL_IDENTITY
                            or final_embedding_identity != initial_embedding_identity
                            or initial_corpus != final_corpus):
                        report["status"] = "invalid_model_or_public_materials_changed_during_batch"
            except Exception as exc:
                report["status"] = "invalid_source_integrity_check_failed"
                report["integrity_error"] = error_record(exc)
        rows = report["results"]
        deterministic = [row for row in rows if (row.get("response", {}).get("answer_routing") or {}).get("backend") == "deterministic_local"]
        report["technical_summary"] = {
            "denominator": expected_count, "recorded_case_outcomes": len(rows),
            "technical_responses_returned": sum(row["status"] == "ok" for row in rows),
            "error_outcomes": sum(row["status"] == "error" for row in rows),
            "not_attempted_outcomes": sum(row["status"].startswith("not_attempted") for row in rows),
            "deterministic_responses": len(deterministic),
            "deterministic_responses_with_zero_answer_generations": sum(row["answer_generation_calls"] == 0 for row in deterministic),
            "operation_selection_returned_decisions": sum(row["operation_selection"]["status"] == "returned" for row in rows),
            "operation_selection_errors": sum(row["operation_selection"]["status"] == "error" for row in rows),
            "mechanical_route_matches": sum((row.get("mechanical_diagnostic") or {}).get("route_matches_fixed_expectation") is True for row in rows),
            "failures_remain_in_denominator": True,
            "technical_ok_and_route_match_do_not_establish_semantic_correctness": True,
            "operation_selection_interpretation_verified": False,
        }
        token_rows = [call for call in calls if isinstance(call.get("response"), dict)]
        report["actual_generation_tokens"] = {
            "input_tokens": sum(int(call["response"].get("prompt_eval_count") or 0) for call in token_rows),
            "output_tokens": sum(int(call["response"].get("eval_count") or 0) for call in token_rows),
            "calls_with_both_counts": sum(type(call["response"].get("prompt_eval_count")) is int and type(call["response"].get("eval_count")) is int for call in token_rows),
            "generation_calls": counters["generation_calls"],
            "counts_are_partial_if_calls_with_both_counts_below_generation_calls": True,
        }
        report["generation_http_elapsed_seconds"] = sum(call["elapsed_seconds"] for call in calls)
        report["completed_at"] = utc_now()
        save_progress()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--label", default="verified-operations-v2", help="Immutable run label")
    parser.add_argument("--fixture", type=Path, default=Path(FIXTURE), help="Project-relative fixed JSON fixture; public fields only reach generation")
    parser.add_argument("--expected-cases", type=int, choices=(24,), default=EXPECTED_CASES, help="Exactly 24 cases, without padding or repetition")
    parser.add_argument("--preflight-only", action="store_true", help="Check inputs, local digests and index with generation forbidden")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report.get(key) for key in (
        "status", "label", "generation_calls", "scope_resolution_calls", "operation_selection_calls",
        "answer_generation_calls", "embedding_calls", "technical_summary")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in {"completed", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
