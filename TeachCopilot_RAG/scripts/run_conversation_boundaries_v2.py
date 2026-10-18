"""One finite, local application batch for preference and teaching boundaries.

Only student sources enter interest extraction; only public dialogue fields enter
RAG. Fixed expectations are saved for later diagnosis, never used as prompts.
All learner data is synthetic. Material retrieval is read-only. The application's
normal PostgreSQL generation admission remains active across processes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_GENERATION_CALLS = 64
MAX_SECONDS = 15 * 60
MAX_REQUEST_SECONDS = 120
INTEREST_OUTPUT_TOKENS = 1536
FIXTURE_NAMES = (
    "tests/fixtures/conversation_preference_boundaries_v2.json",
    "tests/fixtures/conversation_teaching_checks_v2.json",
)
TRACKED_NAMES = (
    *FIXTURE_NAMES,
    "scripts/run_conversation_boundaries_v2.py",
    "local_app/conversation_memory.py",
    "local_app/prompts/conversation_interests.txt",
    "local_app/runtime.py",
    "local_app/rag.py",
    "local_app/answer_adapter.py",
    "local_app/config.py",
    "local_app/memory_service.py",
    "local_app/generation_scheduler.py",
    "local_app/store.py",
    "research_memory/clients.py",
    "research_memory/contracts.py",
)
OPTIONAL_TRACKED_NAMES = ("local_app/teaching_policy.py",)


class FiniteBatchBudgetError(RuntimeError):
    """The fixed generation or elapsed-time limit was reached."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def error_record(exc: BaseException) -> dict:
    # Database and HTTP errors can include connection details. Do not persist
    # DSNs, even when a failed local connection emits them in an exception.
    message = re.sub(r"postgres(?:ql)?://[^\s'\"]+", "[local PostgreSQL DSN]", str(exc))
    message = re.sub(r"(?i)(password\s*=\s*)(?:'[^']*'|\"[^\"]*\"|[^\s]+)", r"\1[redacted]", message)
    return {"type": type(exc).__name__, "message": message}


def source_hashes() -> dict[str, str | None]:
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in TRACKED_NAMES}
    for name in OPTIONAL_TRACKED_NAMES:
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return hashes


def normalized(value: str, aliases: dict) -> str:
    plain = " ".join(value.casefold().split())
    for canonical, alternatives in aliases.items():
        if plain in {" ".join(item.casefold().split()) for item in [canonical, *alternatives]}:
            return " ".join(canonical.casefold().split())
    return plain


def pairs(values: list, aliases: dict) -> list[tuple[str, str]]:
    result = []
    for value in values:
        if isinstance(value, dict):
            operation, interest = value["operation"], value["interest"]
        else:
            operation, interest = value
        if operation not in {"add", "remove", "avoid", "allow"} or not isinstance(interest, str):
            raise ValueError("invalid preregistered preference pair")
        result.append((operation, normalized(interest, aliases)))
    return result


def fixture_cases(fixture: object, expected_count: int) -> list[dict]:
    result = fixture.get("cases") if isinstance(fixture, dict) else fixture
    if not isinstance(result, list) or len(result) != expected_count or any(not isinstance(row, dict) for row in result):
        raise ValueError(f"fixture must contain exactly {expected_count} case objects")
    return result


def preference_diagnostic(candidates: list[dict], case: dict, aliases: dict) -> dict:
    from collections import Counter

    actual_ordered = pairs(candidates, aliases)
    actual = set(actual_ordered)
    expected = set(pairs(case["expected_pairs"], aliases))
    allowed = set(pairs(case.get("allowed_additional_pairs", []), aliases))
    ordered = case.get("expected_ordered_operations")
    diagnostic = {
        "kind": "mechanical_pair_comparison_not_semantic_assessment",
        "actual_pairs": sorted(actual),
        "missing_expected_pairs": sorted(expected - actual),
        "additional_pairs": sorted(actual - expected),
        "additional_pairs_outside_preregistered_allowance": sorted(actual - expected - allowed),
        "exact_pair_set_match": actual == expected,
        "within_preregistered_pair_allowance": expected <= actual <= expected | allowed,
        "actual_ordered_operations": [
            {"operation": candidate["operation"], "interest": normalized(candidate["interest"], aliases),
             "source_id": candidate["source_id"]} for candidate in candidates
        ],
    }
    if ordered is not None:
        diagnostic["expected_ordered_operations"] = deepcopy(ordered)
        def grouped_operations(items):
            groups = []
            for item in items:
                source_id = item["source_id"]
                if not groups or groups[-1][0] != source_id:
                    groups.append((source_id, Counter()))
                groups[-1][1][(item["operation"], normalized(item["interest"], aliases))] += 1
            return groups
        # Source chronology matters; order of add/avoid within one source does
        # not. Multiplicity is retained for avoid -> allow -> avoid histories.
        diagnostic["ordered_operations_match"] = grouped_operations(candidates) == grouped_operations(ordered)
    return diagnostic


def run(args: argparse.Namespace) -> dict:
    import requests
    from psycopg2.extensions import make_dsn
    from local_app.answer_adapter import AnswerAdapter
    from local_app.config import Settings
    from local_app.conversation_memory import interest_messages, interest_schema, validate_interest_candidates
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from research_memory.clients import input_budget_diagnostic
    from scripts.verify_memory import require_local_dsn

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    counters = {"generation_calls": 0, "embedding_calls": 0}
    current: dict = {}
    calls: list[dict] = []
    report = {
        "status": "initializing", "label": args.label, "created_at": utc_now(),
        "data_origin": "synthetic_unreviewed", "human_reviewed": False,
        "semantic_assessment": "not_human_reviewed", "paid_provider_calls": 0,
        "external_provider_disabled": True, "preflight_only": args.preflight_only,
        "expected_interest_cases": 24, "expected_teaching_cases": 12,
        "results": [], "max_generation_calls": MAX_GENERATION_CALLS,
        "max_elapsed_seconds": MAX_SECONDS, "max_request_timeout_seconds": MAX_REQUEST_SECONDS,
        "attempts_per_case": 1,
    }
    initial_hashes = None
    fixtures = None

    def remaining() -> float:
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise FiniteBatchBudgetError("finite_boundary_elapsed_budget_exhausted")
        return budget

    class RecordingSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.trust_env = False
            self.last_post = None

        def post(self, url, **kwargs):
            self.last_post = None
            kwargs["timeout"] = min(float(kwargs.get("timeout", MAX_REQUEST_SECONDS)), remaining())
            response = super().post(url, **kwargs)
            if url.endswith("/api/chat"):
                # Read before OllamaRuntime normalizes display typography.
                self.last_post = {"status_code": response.status_code, "raw_body": response.text}
            return response

        def get(self, url, **kwargs):
            kwargs["timeout"] = min(float(kwargs.get("timeout", MAX_REQUEST_SECONDS)), remaining())
            return super().get(url, **kwargs)

    class TracedRuntime(OllamaRuntime):
        def _post(self, endpoint, payload):
            remaining()
            if endpoint == "/api/chat":
                if counters["generation_calls"] >= MAX_GENERATION_CALLS:
                    raise FiniteBatchBudgetError("finite_boundary_generation_budget_exhausted")
                counters["generation_calls"] += 1
            elif "embed" in endpoint:
                counters["embedding_calls"] += 1
                # No vector values or embedding input copies are persisted.
                return super()._post(endpoint, payload)
            else:
                return super()._post(endpoint, payload)
            row = {"call_number": counters["generation_calls"], "case_id": current.get("case_id"),
                   "kind": current.get("kind"), "endpoint": endpoint, "request": deepcopy(payload),
                   "started_at": utc_now()}
            call_started = time.perf_counter()
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
                with (out / "ollama_http.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        def chat(self, *chat_args, **chat_kwargs):
            # The production scheduler owns admission. Limit its wait to the
            # finite batch deadline as well as its normal 120-second ceiling.
            left = remaining()
            if self.generation_scheduler is not None:
                self.generation_scheduler.wait_timeout = min(MAX_REQUEST_SECONDS, left)
            return super().chat(*chat_args, **chat_kwargs)

    class PublicMaterials:
        """Expose only the two read operations used by RagService.answer."""
        def __init__(self, store):
            self._store = store

        def embedding_index_status(self, model_identity):
            return self._store.embedding_index_status(model_identity)

        def hybrid_search(self, **kwargs):
            return self._store.hybrid_search(**kwargs)

    def save_progress() -> None:
        report.update(counters, elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        write_json(out / "PROCESSING.json", {
            "status": report["status"], "label": args.label,
            "completed_cases": len(report["results"]), "expected_cases": 36,
            "semantic_assessment": "not_human_reviewed", "paid_provider_calls": 0,
            "updated_at": utc_now(), **counters,
        })

    try:
        fixtures = {name: json.loads((ROOT / name).read_text(encoding="utf-8")) for name in FIXTURE_NAMES}
        interest_fixture, teaching_fixture = (fixtures[name] for name in FIXTURE_NAMES)
        interest_cases = fixture_cases(interest_fixture, 24)
        teaching_cases = fixture_cases(teaching_fixture, 12)
        aliases = interest_fixture.get("canonical_aliases", interest_fixture.get("aliases", interest_fixture.get("expected_aliases", {}))) if isinstance(interest_fixture, dict) else {}
        case_ids = [str(case["case_id"]) for case in interest_cases] + [str(case["id"]) for case in teaching_cases]
        if len(case_ids) != len(set(case_ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", value) for value in case_ids):
            raise ValueError("case IDs must be unique safe file names")
        prepared_interests = []
        for case in interest_cases:
            source_rows = case["sources"]
            if not isinstance(source_rows, list) or any(not isinstance(source, dict) for source in source_rows):
                raise ValueError("interest sources must be a list of objects")
            # Same allowlisted student projection as MemoryWorker.
            sources = [{"source_id": str(source["source_id"]), "author_role": "student", "content": source["content"]}
                       for source in source_rows if source.get("author_role") == "student"]
            if not sources or any(not isinstance(source["content"], str) or not source["content"].strip() for source in sources):
                raise ValueError("each interest case requires nonempty literal student sources")
            if len({source["source_id"] for source in sources}) != len(sources):
                raise ValueError("duplicate student source IDs")
            pairs(case["expected_pairs"], aliases)
            pairs(case.get("allowed_additional_pairs", []), aliases)
            prepared_interests.append((case, sources, interest_messages(sources), interest_schema(sources)))
        public_teaching = []
        for case in teaching_cases:
            messages = case["messages"]
            if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str) or not message["content"].strip()
                for message in messages
            ) or messages[-1]["role"] != "user":
                raise ValueError("teaching cases require public user/assistant dialogue ending with a user message")
            public = {"messages": [{"role": message["role"], "content": message["content"]} for message in messages],
                      "subject_id": case["subject_id"], "topic_id": case["topic_id"]}
            if "preference_context" in case:
                preferences = case["preference_context"]
                if not isinstance(preferences, dict) or set(preferences) - {"interests", "avoided_interests"}:
                    raise ValueError("invalid public preference context")
                if any(not isinstance(values, list) or any(not isinstance(value, str) for value in values)
                       for values in preferences.values()):
                    raise ValueError("preference values must be lists of strings")
                public["preference_context"] = deepcopy(preferences)
            public_teaching.append((case, public))
        baseline = Settings.from_env()
        params = require_local_dsn(baseline.database_url)
        settings = replace(baseline, request_timeout=MAX_REQUEST_SECONDS)
        if settings.chat_model != "gemma4:12b" or settings.context_tokens != 8192:
            raise ValueError("batch requires preregistered gemma4:12b and 8192-token context")
        readonly_options = (params.get("options", "") + " -c default_transaction_read_only=on").strip()
        readonly_dsn = make_dsn(settings.database_url, options=readonly_options, connect_timeout=5)
        runtime = TracedRuntime(settings, session=RecordingSession())
        service = RagService(PublicMaterials(Store(readonly_dsn)), runtime, settings,
                             answer_adapter=AnswerAdapter(runtime))
        initial_hashes = source_hashes()
        for name, fixture in fixtures.items():
            write_json(out / Path(name).name, fixture)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True).stdout.strip()
        model_identity = runtime.model_identity(settings.chat_model)
        input_guards = {}
        for case, sources, messages, schema in prepared_interests:
            diagnostic = input_budget_diagnostic(messages, schema, context_tokens=runtime.context_tokens,
                                                max_output_tokens=INTEREST_OUTPUT_TOKENS, purpose="conversation_interests")
            diagnostic["enforced"] = True
            input_guards[case["case_id"]] = diagnostic
        preregistered = {
            "kind": "finite_application_boundary_batch_not_research_method_comparison",
            "label": args.label, "created_at": utc_now(), "git_commit": commit,
            "fixtures": fixtures, "sha256": initial_hashes, "model_identity": model_identity,
            "model_manifest": runtime.manifest(), "temperature": 0, "seed": 42, "think": False,
            "interest_output_tokens": INTEREST_OUTPUT_TOKENS,
            "interest_input_guard": input_guards, "interest_truncate_history": False,
            "max_generation_calls": MAX_GENERATION_CALLS, "max_elapsed_seconds": MAX_SECONDS,
            "max_request_timeout_seconds": MAX_REQUEST_SECONDS, "attempts_per_case": 1,
            "material_database": "loopback PostgreSQL public retrieval; read-only transactions",
            "other_database_access": "normal global generation-admission scheduler only",
            "external_provider_disabled": True, "paid_provider_calls": 0,
            "human_reviewed": False, "semantic_assessment": "not_human_reviewed",
            "gold_policy": "fixed expectations never enter interest or RAG generation inputs",
        }
        write_json(out / "preregistered.json", preregistered)
        report["index_status"] = service.status()
        if not report["index_status"]["ready"]:
            raise RuntimeError("real local material embedding index is not ready")
        report["input_guard_failures"] = [case_id for case_id, guard in input_guards.items() if not guard["within_guard"]]
        if args.preflight_only:
            if report["input_guard_failures"]:
                raise ValueError("one or more fixed interest inputs exceed the context guard")
            report["status"] = "preflight_passed"
            return report
        report["status"] = "running"
        save_progress()
        jobs = [("interest", case, (sources, messages, schema)) for case, sources, messages, schema in prepared_interests]
        jobs += [("teaching", case, public) for case, public in public_teaching]
        for kind, case, public in jobs:
            case_id = str(case["case_id"] if kind == "interest" else case["id"])
            current.update(case_id=case_id, kind=kind)
            case_started = time.perf_counter()
            before = dict(counters)
            row = {"case_id": case_id, "kind": kind, "split": case.get("split"),
                   "category": case.get("category"), "status": "error", "human_reviewed": False,
                   "semantic_assessment": "not_human_reviewed"}
            try:
                remaining()
                if counters["generation_calls"] >= MAX_GENERATION_CALLS:
                    raise FiniteBatchBudgetError("finite_boundary_generation_budget_exhausted")
                if kind == "interest":
                    sources, messages, schema = public
                    row["input_budget"] = input_guards[case_id]
                    if not row["input_budget"]["within_guard"]:
                        raise ValueError("memory_writer_input_exceeds_context")
                    response = runtime.chat(messages, response_format=schema,
                                            max_output_tokens=INTEREST_OUTPUT_TOKENS, truncate_history=False)
                    row["runtime_response"] = response
                    if int(response.get("prompt_eval_count") or 0) >= runtime.context_tokens - INTEREST_OUTPUT_TOKENS:
                        raise ValueError("context_saturation_requires_review")
                    candidates = validate_interest_candidates(response["message"]["content"], sources)
                    row["candidates"] = candidates
                    row["mechanical_diagnostic"] = preference_diagnostic(candidates, case, aliases)
                else:
                    row["public_input"] = deepcopy(public)
                    row["response"] = service.answer(**public)
                    backend = (row["response"].get("answer_routing") or {}).get("backend", "local")
                    if backend != "local":
                        raise RuntimeError("nonlocal answer routing is forbidden in this batch")
                row["status"] = "ok"
            except Exception as exc:
                row["error"] = error_record(exc)
                if isinstance(exc, FiniteBatchBudgetError) and counters == before:
                    row["status"] = "not_attempted_budget_exhausted"
                if getattr(exc, "raw_response", None) is not None:
                    row["raw_response"] = deepcopy(exc.raw_response)
            row.update(elapsed_seconds=time.perf_counter() - case_started,
                       generation_calls=counters["generation_calls"] - before["generation_calls"],
                       embedding_calls=counters["embedding_calls"] - before["embedding_calls"])
            report["results"].append(row)
            write_json(out / (case_id + ".json"), row)
            save_progress()
            print(f"{case_id}: {row['status']}", flush=True)
        report["status"] = "completed"
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = error_record(exc)
        return report
    finally:
        if initial_hashes is not None:
            try:
                final_hashes = source_hashes()
                changes = [name for name in initial_hashes if initial_hashes[name] != final_hashes[name]]
                report["source_integrity"] = {"unchanged": not changes, "changed_paths": changes,
                                              "start_sha256": initial_hashes, "end_sha256": final_hashes}
                if changes:
                    report["status"] = "invalid_source_changed_during_batch"
            except Exception as exc:
                report["status"] = "invalid_source_integrity_check_failed"
                report["source_integrity"] = {"unchanged": False, "error": error_record(exc)}
        rows = report["results"]
        report["technical_summary"] = {
            "expected_cases": 36, "recorded_case_outcomes": len(rows),
            "interest": {"denominator": 24, "technical_ok": sum(row["kind"] == "interest" and row["status"] == "ok" for row in rows)},
            "teaching": {"denominator": 12, "technical_ok": sum(row["kind"] == "teaching" and row["status"] == "ok" for row in rows)},
            "failures_remain_in_denominator": True,
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
    parser.add_argument("--label", required=True, help="Immutable run label, such as baseline or candidate")
    parser.add_argument("--preflight-only", action="store_true", help="Check fixed inputs, model digests and read-only index without generation")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ("status", "label", "generation_calls", "embedding_calls", "technical_summary")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in {"completed", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
