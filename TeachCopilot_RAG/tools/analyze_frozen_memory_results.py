"""Audit an existing memory experiment without models, grading or input edits.

Usage: python tools/analyze_frozen_memory_results.py
The generated files are a supplementary audit, not a replacement for frozen data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


METHODS = ("profile_only", "tail", "summary", "structured")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(app: Path, evidence: Path) -> tuple[dict, list[dict]]:
    source_hashes = read_json(evidence / "source_hashes.json")
    for name, source in source_hashes["files"].items():
        require(digest(evidence / name) == source["sha256"], f"Evidence checksum mismatch: {name}")
    report_path = evidence / "report.json"
    rows_path = evidence / "heldout_rows.jsonl"
    report, rows = read_json(report_path), read_jsonl(rows_path)
    freeze = read_json(evidence / "freeze.json")
    checks = []
    for group, relative in (("code", "research_memory"), ("prompts", "research_memory/prompts")):
        for name, expected in freeze["code_and_prompts"][group].items():
            path = app / relative / name
            checks.append({"path": path.relative_to(app).as_posix(), "sha256": digest(path), "expected": expected})
    for name, expected in freeze["inputs"].items():
        path = app / Path(name.replace("\\", "/"))
        checks.append({"path": path.relative_to(app).as_posix(), "sha256": digest(path), "expected": expected})
    require(len(checks) == 29, "Expected all 29 frozen code, prompt and input hashes")
    require(all(x["sha256"] == x["expected"] for x in checks), "Frozen hash mismatch")
    require(len(rows) == 128, "Expected 128 rows including failures")
    require(len({x["result_id"] for x in rows}) == 128, "Duplicate result_id")
    require(all(x["assessment_status"] == "not_assessed" and x["judgement"] is None for x in rows),
            "This audit expects the unassessed experiment and must not introduce grades")
    require(report["assessment"]["calibration_qualified"] is False, "Unexpected calibration state")

    design_path = evidence / "design.private.json"
    designs = {x["history_id"]: x for x in read_json(design_path)["design"]}
    require(len(designs) == 16, "Expected 16 histories")
    methods = {}
    history_pairs = []
    for history_id, design in sorted(designs.items()):
        group = [x for x in rows if x["history_id"] == history_id]
        require(len(group) == 8, f"Expected two requests and four methods for {history_id}")
        paired = {key: design[key] for key in ("history_id", "situation", "length", "lesson_count",
                                              "source_text_chars", "full_tail_chars", "selected_tail_chars")}
        paired["key_turns"] = len(design["key_turn_ids"])
        paired["key_turns_visible_in_tail"] = len(design["key_turn_ids_visible_in_tail"])
        for method in METHODS:
            method_rows = [x for x in group if x["method"] == method]
            require(len(method_rows) == 2 and len({x["request_id"] for x in method_rows}) == 2,
                    f"Invalid request pairing for {history_id}/{method}")
            paired[f"{method}_ok_of_2"] = sum(x["status"] == "ok" for x in method_rows)
        history_pairs.append(paired)
    for method in METHODS:
        group = [x for x in rows if x["method"] == method]
        methods[method] = {"planned": len(group), "technical_ok": sum(x["status"] == "ok" for x in group),
                           "statuses": dict(Counter(x["status"] for x in group)), "assessed": 0}
        require(methods[method]["technical_ok"] == report["heldout"]["methods"][method]["technical_ok"],
                "Report and row counts differ")
        for length in ("short", "long"):
            subset = [x for x in group if designs[x["history_id"]]["length"] == length]
            methods[method][length] = {"planned": len(subset), "technical_ok": sum(x["status"] == "ok" for x in subset)}

    # The portable SQL projection was extracted read-only. Attempt logs include
    # two locally rejected wrapper calls that incorrectly carry performed=true.
    accounting = read_json(evidence / "call_accounting.json")
    ledger_groups = defaultdict(lambda: {"actual_calls": 0, "failed_calls": 0, "input_tokens": 0,
                                        "output_tokens": 0, "elapsed_seconds": 0.0, "incomplete_usage_calls": 0})
    ledger_keys = set()
    for call in accounting["durable_calls"]:
        ledger_keys.add((call["job_id"], call["ordinal"]))
        purpose = call["purpose"]
        require(purpose in {"action", "summary", "extract", "shorten"}, "Unrecognized ledger purpose")
        stage = "answer" if purpose == "action" else "memory_update"
        out = ledger_groups[(call["method"], stage)]
        out["actual_calls"] += 1
        out["failed_calls"] += call["status"] == "failed"
        usage = call["usage"]
        known = all(usage.get(k) is not None for k in ("input_tokens", "output_tokens"))
        out["incomplete_usage_calls"] += not known
        for key in ("input_tokens", "output_tokens"):
            out[key] += usage.get(key) or 0
        out["elapsed_seconds"] += call["elapsed_seconds"] or 0

    unmatched = []
    for call in accounting["attempt_log_records"]:
        if (call["job_id"], call["call_id"]) not in ledger_keys:
            unmatched.append(call)
    require(len(unmatched) == 2 and all(x["error"] == "RuntimeError" and x["usage"]["input_tokens"] is None
                                       for x in unmatched), "Unexpected call-log discrepancy")
    stages = []
    for observation in report["generation_resources"]["heldout"]["active_attempts"]:
        method, stage = observation["method"], observation["stage"]
        actual = ledger_groups[(method, stage)]
        stages.append({"method": method, "stage": stage,
                       "observed_recorded_calls": observation["recorded_calls"],
                       "observed_performed_true_records": observation["performed_calls"],
                       "observed_failed_records": observation["failed_calls"],
                       "observed_known_input_tokens": observation["input_tokens"],
                       "observed_known_output_tokens": observation["output_tokens"],
                       "observed_known_elapsed_seconds": observation["elapsed_seconds"],
                       "message_content_chars": observation["message_content_chars"]["known_sum"],
                       "response_schema_chars": observation["response_schema_chars"]["known_sum"],
                       "durable": dict(actual)})
    ledger_total = {key: sum(x[key] for x in ledger_groups.values()) for key in
                    ("actual_calls", "failed_calls", "input_tokens", "output_tokens", "elapsed_seconds", "incomplete_usage_calls")}
    authoritative = report["generation_resources"]["heldout"]["durable_call_ledger"]
    require(ledger_total["actual_calls"] == authoritative["calls"] == 236, "Ledger call count mismatch")
    require(ledger_total["input_tokens"] == authoritative["known_input_tokens"] == 275825, "Input token mismatch")
    require(ledger_total["output_tokens"] == authoritative["known_output_tokens"] == 75750, "Output token mismatch")

    # Each writer error is repeated in both request rows. Count the same event
    # once per history, method and lesson, retaining distinct error diagnostics.
    unique_errors = {}
    for row in rows:
        for error in row["memory_errors"]:
            key = (row["history_id"], row["method"], error.get("lesson_id"), error.get("error_code"), error.get("error"))
            unique_errors[key] = {"history_id": row["history_id"], "method": row["method"], **error}
    metrics = {
        "audit_version": 1, "kind": "supplementary_read_only_technical_audit", "semantic_assessment_added": False,
        "inputs": {"report_sha256": digest(report_path), "rows_sha256": digest(rows_path),
                   "freeze_sha256": digest(evidence / "freeze.json"), "design_sha256": digest(design_path),
                   "source_hashes": source_hashes},
        "validation": {"frozen_hashes_verified": len(checks), "unique_result_ids": len(rows),
                       "all_not_assessed": True, "calibration_qualified": False},
        "frozen_hashes": checks, "planned": 128, "distinct_histories": 16,
        "technical_ok": sum(x["status"] == "ok" for x in rows), "methods": methods,
        "technical_failures_by_stage": dict(Counter(x["observed_stage"] for x in report["heldout"]["failures_by_stage"]["technical_failures"])),
        "unique_writer_error_events": list(unique_errors.values()), "resources": stages,
        "durable_call_ledger_total": ledger_total, "unmatched_attempt_log_records": unmatched,
        "call_count_explanation": "238 attempt-log records include two locally rejected summary wrapper calls: t08-l08 and t10-l08. "
                                  "After l07 truncated_completion, _CachedClient.stop_reason raises before ledger INSERT and network request. "
                                  "These records incorrectly carry performed=true; durable ledger contains 236 actual calls, not a regeneration.",
        "notes": ["128 rows are dependent within 16 histories, with eight related short/long scenario pairs.",
                  "All technical failures remain in planned denominators. No semantic winner is established.",
                  "Writer costs belong to a history/method and are shared across its two requests.",
                  "Token totals are full-call model usage. Schema character counts are separate; schema token counts are unavailable.",
                  "Time is summed call elapsed time, not end-to-end batch latency or a throughput estimate.",
                  "Local hardware, electricity and monetary costs were not measured; missing cost does not mean zero."]}
    return metrics, history_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence", type=Path, default=Path("docs/research-report-2026-10-20/evidence"))
    parser.add_argument("--output", type=Path, default=Path("docs/research-report-2026-10-20"))
    args = parser.parse_args()
    app = args.app_root.resolve()
    evidence = args.evidence if args.evidence.is_absolute() else app / args.evidence
    output = args.output if args.output.is_absolute() else app / args.output
    frozen = (app / ".local/research_memory/final_v4_20261018").resolve()
    require(output.resolve() != frozen and frozen not in output.resolve().parents,
            "Supplementary output must not be written inside the frozen experiment")
    metrics, pairs = audit(app, evidence)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "history_pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps({"validation": metrics["validation"], "technical_ok": metrics["technical_ok"],
                      "actual_calls": metrics["durable_call_ledger_total"]["actual_calls"],
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
