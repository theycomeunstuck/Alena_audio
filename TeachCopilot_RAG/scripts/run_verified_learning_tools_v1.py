"""One finite local batch of narrow learning tools and their boundaries.

Only ``case.public`` reaches RagService. Private expectations support mechanical
diagnosis after a response; they never choose, repair or regenerate an answer.
An ``ok`` outcome means the application returned a response, not that its teaching
content is correct. Raw model HTTP bodies are retained before display changes.
"""
from __future__ import annotations

import argparse
import base64
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

FIXTURE = "tests/fixtures/verified_learning_tools_v1.json"
EXPECTED_CASES = 24
MAX_GENERATION_CALLS = 60
MAX_SECONDS = 15 * 60
MAX_REQUEST_SECONDS = 120
EXPECTED_MODEL_IDENTITY = "gemma4:12b@6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b"
OPERATIONS = {"division_check", "division_places", "division_practice", "print_output"}


class FiniteBatchBudgetError(RuntimeError):
    """The fixed number of generations or elapsed-time budget was exhausted."""


class SourceIntegrityError(RuntimeError):
    """An input or implementation changed during the fixed batch."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def error_record(exc: BaseException) -> dict:
    message = re.sub(r"postgres(?:ql)?://[^\s'\"]+", "[local PostgreSQL DSN]", str(exc))
    message = re.sub(r"(?i)(password\s*=\s*)(?:'[^']*'|\"[^\"]*\"|[^\s]+)", r"\1[redacted]", message)
    return {"type": type(exc).__name__, "message": message}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()


def source_hashes() -> dict[str, str]:
    paths = {
        ROOT / FIXTURE, Path(__file__).resolve(),
        ROOT / "scripts/verify_memory.py",
        *ROOT.glob("local_app/**/*.py"),
        *ROOT.glob("local_app/prompts/**/*"),
        *ROOT.glob("research_memory/*.py"),
    }
    # New prompt files and modules are part of the guard, including all three
    # learning-tool implementations. Environment files and secrets are excluded.
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def fixture_cases(fixture: object) -> list[tuple[dict, dict]]:
    cases = fixture.get("cases") if isinstance(fixture, dict) else None
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError("fixture must contain exactly 24 case objects")
    prepared = []
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        identifier = case.get("case_id")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier)
                or identifier in identifiers):
            raise ValueError("case IDs must be unique safe file names")
        identifiers.add(identifier)
        public = case.get("public")
        if not isinstance(public, dict) or set(public) != {"messages", "subject_id", "topic_id"}:
            raise ValueError("public fields must be exactly messages, subject_id and topic_id")
        if any(not isinstance(public[key], str) or not public[key].strip()
               for key in ("subject_id", "topic_id")):
            raise ValueError("public subject and topic must be nonempty strings")
        messages = public["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("public dialogue must be a nonempty list")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"user", "assistant"}
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise ValueError("public dialogue requires literal user and assistant messages")
        if messages[-1]["role"] != "user":
            raise ValueError("public dialogue must end in a learner message")
        gold = case.get("private_gold")
        if not isinstance(gold, dict) or gold.get("expected_route") not in OPERATIONS | {"normal_model", "out_of_scope"}:
            raise ValueError("private route expectation must be explicit")
        if not isinstance(gold.get("mechanical_checks", []), list):
            raise ValueError("mechanical checks must be a list")
        for check in gold.get("mechanical_checks", []):
            if not isinstance(check, dict) or not isinstance(check.get("kind"), str):
                raise ValueError("mechanical checks must be named objects")
            if check["kind"] in {"division_fact", "division_places"}:
                dividend, divisor = check.get("dividend"), check.get("divisor")
                if type(dividend) is not int or type(divisor) is not int or divisor <= 0:
                    raise ValueError("invalid fixed division operands")
                quotient, remainder = divmod(dividend, divisor)
                if check.get("quotient") != quotient or check.get("remainder", remainder) != remainder:
                    raise ValueError("fixed division expectation is arithmetically inconsistent")
                if "proposed_product" in check and check["proposed_product"] != check["proposed_quotient"] * divisor:
                    raise ValueError("fixed product expectation is arithmetically inconsistent")
                if check["kind"] == "division_places" and any(
                    check.get(name) != value for name, value in (
                        ("hundreds", quotient // 100), ("tens", quotient // 10 % 10), ("ones", quotient % 10))):
                    raise ValueError("fixed place-value expectation is arithmetically inconsistent")
            elif check["kind"] == "python_stdout" and not isinstance(check.get("stdout"), str):
                raise ValueError("fixed stdout must be a literal string; it is never executed")
        # This separate object is the complete allowlist passed to generation.
        prepared.append((case, deepcopy(public)))
    return prepared


def visible_practice_pairs(answer: str) -> list[tuple[int, int]]:
    patterns = (
        r"(?<![\d.,])([0-9]{1,6})\s*[:÷/]\s*([0-9]{1,2})(?![0-9]|[.,][0-9])",
        r"(?:разделить|раздели)\s+([0-9]{1,6})\s+на\s+([0-9]{1,2})(?![0-9])",
    )
    return sorted({(int(match[1]), int(match[2])) for pattern in patterns
                   for match in re.finditer(pattern, answer, re.I)})


def mechanical_diagnostic(response: dict, resolution: dict | None, gold: dict, counts: dict) -> dict:
    """Compare bounded data properties; never assign semantic success labels."""
    tool = response.get("learning_tool") or {}
    backend = (response.get("answer_routing") or {}).get("backend", "local")
    operation = tool.get("operation")
    if backend == "deterministic_local":
        observed = operation if operation in OPERATIONS else "invalid_deterministic_operation"
    elif isinstance(resolution, dict) and resolution.get("in_scope") is False:
        observed = "out_of_scope"
    else:
        observed = "normal_model"
    diagnostic = {
        "kind": "mechanical_diagnosis_not_semantic_assessment",
        "expected_route": gold["expected_route"], "observed_route": observed,
        "route_matches_fixed_expectation": observed == gold["expected_route"],
        "backend": backend, "operation": operation,
        "reported_verification_scope": deepcopy(tool.get("verification_scope", [])),
        "actual_model_calls": deepcopy(counts),
        "deterministic_answer_has_zero_answer_generations": (
            counts["answer_generation_calls"] == 0 if backend == "deterministic_local" else None),
        "checks": [],
        "limits": "Checks do not establish teaching quality, source authorship, autonomy or mastery.",
    }
    facts = tool.get("facts") or {}
    answer = response["choices"][0]["message"]["content"]
    for check in gold.get("mechanical_checks", []):
        kind = check.get("kind")
        result = {"kind": kind}
        if kind == "python_stdout":
            stdout = facts.get("stdout")
            result.update(applicable=operation == "print_output", exact_stdout_matches_gold=stdout == check["stdout"],
                          stdout_character_count=len(stdout) if isinstance(stdout, str) else None)
            block = re.search(r"(?m)^(`{3,})text\n(.*?)^\1[ \t]*$", answer, re.S)
            expected_display = check["stdout"]
            if expected_display and not expected_display.endswith("\n"):
                expected_display += "\n"  # The closing Markdown fence requires its own line.
            result["display_block_matches_expected_characters"] = bool(block and block[2] == expected_display)
            result["scope"] = "Exact stdout metadata and rendered block; explanatory prose is not assessed."
        elif kind in {"division_fact", "division_places"}:
            fields = ("dividend", "divisor", "quotient", "remainder")
            result["applicable"] = operation in {"division_check", "division_places"}
            result["metadata_fields_match_gold"] = {
                field: facts.get(field) == check[field] for field in fields if field in check}
            result["scope"] = "Numeric metadata only; place-value prose requires a separate review."
        elif kind == "proposed_division_property":
            pairs = visible_practice_pairs(answer)
            result.update(applicable=operation == "division_practice", visible_operand_pairs=[list(pair) for pair in pairs])
            if len(pairs) == 1 and pairs[0][1] > 0:
                dividend, divisor = pairs[0]
                quotient, remainder = divmod(dividend, divisor)  # Independent integer arithmetic.
                digits = str(quotient)
                excluded = {(item["dividend"], item["divisor"]) for item in check.get("exclude_pairs", [])}
                result["visible_operands_have_requested_property"] = (
                    check["divisor_min"] <= divisor <= check["divisor_max"]
                    and remainder == 0 and len(digits) == check["quotient_digits"]
                    and digits[1] == str(check["middle_digit"])
                    and (not check["last_digit_nonzero"] or digits[-1] != "0")
                    and pairs[0] not in excluded)
                # The privately recomputed quotient is deliberately not added
                # to a response or report field for practice cases.
                result["numeric_quotient_not_in_student_text"] = not re.search(
                    rf"(?<![0-9]){quotient}(?![0-9])", answer)
                serialized_metadata = json.dumps(tool, ensure_ascii=False)
                result["numeric_quotient_not_in_tool_metadata"] = not re.search(
                    rf"(?<![0-9]){quotient}(?![0-9])", serialized_metadata)
            else:
                result["visible_operands_have_requested_property"] = None
                result["diagnostic"] = "A single complete visible division pair could not be read."
            result["private_derived_answer_fields_absent"] = not any(
                key in facts for key in ("quotient", "remainder", "product_check", "long_division_steps", "quotient_places", "facts"))
            result["scope"] = "Visible operand property and numeric answer leakage; spelled-out leakage and pedagogy require review."
        else:
            result.update(applicable=False, automatically_assessed=False,
                          scope="Boundary expectation retained for separate review; no code execution or semantic grading.")
        diagnostic["checks"].append(result)
    return diagnostic


def run(args: argparse.Namespace) -> dict:
    import requests
    from psycopg2.extensions import make_dsn
    from local_app.answer_adapter import AnswerAdapter
    from local_app.config import Settings
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from scripts.verify_memory import require_local_dsn

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    counters = {"generation_calls": 0, "scope_resolution_calls": 0, "answer_generation_calls": 0, "embedding_calls": 0}
    current: dict = {"phase": "answer_generation"}
    calls: list[dict] = []
    prepared: list[tuple[dict, dict]] = []
    initial_hashes = None
    initial_commit = None
    initial_corpus = None
    runtime = None
    store = None
    report = {
        "status": "initializing", "label": args.label, "created_at": utc_now(),
        "data_origin": "synthetic_unreviewed", "human_reviewed": False,
        "semantic_assessment": "not_human_reviewed", "paid_provider_calls": 0,
        "external_provider_disabled": True, "preflight_only": args.preflight_only,
        "expected_cases": EXPECTED_CASES, "attempts_per_case": 1,
        "max_generation_calls": MAX_GENERATION_CALLS, "max_elapsed_seconds": MAX_SECONDS,
        "max_request_timeout_seconds": MAX_REQUEST_SECONDS, "results": [],
        "technical_ok_does_not_mean_task_correct": True,
    }

    def remaining() -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise FiniteBatchBudgetError("finite_learning_tools_elapsed_budget_exhausted")
        return left

    def corpus_hash() -> str:
        rows = store.material_rows()
        encoded = json.dumps(sorted(rows, key=lambda item: str(item.get("id", ""))),
                             ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_unchanged() -> None:
        if initial_hashes != source_hashes():
            raise SourceIntegrityError("source_or_fixture_changed_during_fixed_batch")
        if initial_commit != git_commit():
            raise SourceIntegrityError("git_head_changed_during_fixed_batch")

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
                self.last_post = {"status_code": response.status_code, "raw_body": response.text,
                                  "raw_body_base64": base64.b64encode(response.content).decode("ascii")}
            return response

        def get(self, url, **kwargs):
            kwargs["timeout"] = min(float(kwargs.get("timeout", MAX_REQUEST_SECONDS)), MAX_REQUEST_SECONDS, remaining())
            return super().get(url, **kwargs)

    class TracedRuntime(OllamaRuntime):
        def resolve_chat_query(self, *resolve_args, **resolve_kwargs):
            previous = current["phase"]
            current["phase"] = "scope_resolution"
            try:
                result = super().resolve_chat_query(*resolve_args, **resolve_kwargs)
                current["resolution"] = deepcopy(result)
                return result
            finally:
                current["phase"] = previous

        def _post(self, endpoint, payload):
            remaining()
            if endpoint != "/api/chat":
                if "embed" in endpoint:
                    counters["embedding_calls"] += 1
                return super()._post(endpoint, payload)
            if counters["generation_calls"] >= MAX_GENERATION_CALLS:
                raise FiniteBatchBudgetError("finite_learning_tools_generation_budget_exhausted")
            counters["generation_calls"] += 1
            counters[current["phase"] + "_calls"] += 1
            call_started = time.perf_counter()
            row = {"call_number": counters["generation_calls"], "case_id": current.get("case_id"),
                   "purpose": current["phase"], "endpoint": endpoint, "request": deepcopy(payload),
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
                with (out / "ollama_http.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        def chat(self, *chat_args, **chat_kwargs):
            left = remaining()
            if self.generation_scheduler is not None:
                self.generation_scheduler.wait_timeout = min(MAX_REQUEST_SECONDS, left)
            return super().chat(*chat_args, **chat_kwargs)

    class PublicMaterials:
        """Only read-only public retrieval is exposed to RagService."""
        def embedding_index_status(self, model_identity):
            return store.embedding_index_status(model_identity)

        def hybrid_search(self, **kwargs):
            return store.hybrid_search(**kwargs)

    def save_progress() -> None:
        report.update(counters, elapsed_seconds=time.monotonic() - started)
        write_json(out / "report.json", report)
        write_json(out / "PROCESSING.json", {
            "status": report["status"], "label": args.label, "updated_at": utc_now(),
            "recorded_cases": len(report["results"]), "expected_cases": EXPECTED_CASES,
            "semantic_assessment": "not_human_reviewed", "paid_provider_calls": 0, **counters,
        })

    try:
        initial_hashes = source_hashes()
        initial_commit = git_commit()
        fixture = json.loads((ROOT / FIXTURE).read_text(encoding="utf-8"))
        prepared = fixture_cases(fixture)
        settings = replace(Settings.from_env(), request_timeout=MAX_REQUEST_SECONDS)
        if settings.chat_model != "gemma4:12b" or settings.context_tokens != 8192:
            raise ValueError("batch requires preregistered gemma4:12b and 8192-token context")
        params = require_local_dsn(settings.database_url)
        readonly_options = (params.get("options", "") + " -c default_transaction_read_only=on").strip()
        store = Store(make_dsn(settings.database_url, options=readonly_options, connect_timeout=5))
        runtime = TracedRuntime(settings, session=RecordingSession())
        # Passing the adapter explicitly bypasses every external-provider env config.
        service = RagService(PublicMaterials(), runtime, settings, answer_adapter=AnswerAdapter(runtime))
        model_identity = runtime.model_identity(settings.chat_model)
        if model_identity != EXPECTED_MODEL_IDENTITY:
            raise ValueError("local generation model digest differs from the fixed model")
        initial_corpus = corpus_hash()
        preregistered = {
            "kind": "finite_application_learning_tool_batch_not_memory_method_comparison",
            "created_at": utc_now(), "label": args.label, "git_commit": initial_commit,
            "fixture": fixture, "sha256": initial_hashes, "public_materials_sha256": initial_corpus,
            "model_identity": model_identity, "model_manifest": runtime.manifest(),
            "temperature": 0, "seed": 42, "think": False,
            "max_generation_calls": MAX_GENERATION_CALLS, "max_elapsed_seconds": MAX_SECONDS,
            "max_request_timeout_seconds": MAX_REQUEST_SECONDS, "attempts_per_case": 1,
            "gold_policy": "Only each case.public is passed to RagService; identifiers, categories and private_gold never enter generation.",
            "material_database": "loopback PostgreSQL public materials; read-only transactions",
            "other_database_access": "normal global generation-admission scheduler only",
            "external_provider_disabled": True, "paid_provider_calls": 0,
            "human_reviewed": False, "semantic_assessment": "not_human_reviewed",
            "technical_ok_does_not_mean_task_correct": True,
        }
        write_json(out / "preregistered.json", preregistered)
        write_json(out / Path(FIXTURE).name, fixture)
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
            current.update(case_id=case_id, phase="answer_generation", resolution=None)
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
                        raise FiniteBatchBudgetError("finite_learning_tools_generation_budget_exhausted")
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
            row.update({key: counters[key] - before[key] for key in counters})
            row["elapsed_seconds"] = time.perf_counter() - case_started
            row["scope_resolution"] = deepcopy(current.get("resolution"))
            row["model_call_numbers"] = [call["call_number"] for call in calls[call_start:]]
            if row.get("response") is not None:
                try:
                    row["mechanical_diagnostic"] = mechanical_diagnostic(
                        row["response"], row["scope_resolution"], case["private_gold"],
                        {key: row[key] for key in counters})
                except Exception as exc:
                    # A report/checker bug must not erase a real response or
                    # cause its generation to be repeated.
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
                final_hashes = source_hashes()
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
                    final_corpus = corpus_hash()
                    report["data_integrity"] = {
                        "model_identity_unchanged": final_identity == EXPECTED_MODEL_IDENTITY,
                        "end_model_identity": final_identity,
                        "public_materials_unchanged": initial_corpus == final_corpus,
                        "start_public_materials_sha256": initial_corpus, "end_public_materials_sha256": final_corpus,
                    }
                    if final_identity != EXPECTED_MODEL_IDENTITY or initial_corpus != final_corpus:
                        report["status"] = "invalid_model_or_public_materials_changed_during_batch"
            except Exception as exc:
                report["status"] = "invalid_source_integrity_check_failed"
                report["integrity_error"] = error_record(exc)
        rows = report["results"]
        deterministic = [row for row in rows if (row.get("response", {}).get("answer_routing") or {}).get("backend") == "deterministic_local"]
        report["technical_summary"] = {
            "denominator": EXPECTED_CASES, "recorded_case_outcomes": len(rows),
            "technical_responses_returned": sum(row["status"] == "ok" for row in rows),
            "error_outcomes": sum(row["status"] == "error" for row in rows),
            "not_attempted_outcomes": sum(row["status"].startswith("not_attempted") for row in rows),
            "deterministic_responses": len(deterministic),
            "deterministic_responses_with_zero_answer_generations": sum(row["answer_generation_calls"] == 0 for row in deterministic),
            "mechanical_route_matches": sum((row.get("mechanical_diagnostic") or {}).get("route_matches_fixed_expectation") is True for row in rows),
            "failures_remain_in_denominator": True,
            "technical_ok_and_route_match_do_not_establish_semantic_correctness": True,
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
    parser.add_argument("--label", default="verified-tools-v1", help="Immutable run label")
    parser.add_argument("--preflight-only", action="store_true", help="Check inputs, local digests and index with no generation")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ("status", "label", "generation_calls", "scope_resolution_calls", "answer_generation_calls", "embedding_calls", "technical_summary")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] in {"completed", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
