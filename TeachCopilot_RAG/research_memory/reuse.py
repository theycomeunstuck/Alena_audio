"""One recovery lineage: reuse every valid calibration result, retry transport only.

This module never calls a provider, touches its ledger, or changes source files.
The target workflow counts only its new calls; source accounting stays in lineage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .dataset import require
from .judge import (CONTEXT_PROTOCOL, CONTEXT_JUDGEMENT_SCHEMA, ROOT, _validate_case,
                    prompt_path, summarize_calibration, validate_judgement)


SEMANTIC_CODE = ("judge.py", "runner.py", "clients.py", "spiderssense.py", "contracts.py", "dataset.py")
STEP_FILES = ("step.json", "input.jsonl", "output/manifest.json", "output/judgements.jsonl",
              "output/calls.jsonl", "output/trace.jsonl", "output/report.md")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _inside(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe reuse path")
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()) and not result.is_symlink(), "Unsafe reuse path")
    return result


def _identity(definition):
    return {"batch_dir": str(Path(definition["batch_dir"]).resolve()),
            "judge_config": definition["judge_config"], "sha256": definition["sha256"],
            "versions": {"code": {name: definition["versions"]["code"][name] for name in SEMANTIC_CODE},
                         "prompts": definition["versions"]["prompts"]}}


def describe_calibration_reuse(source_dir, target_definition) -> dict:
    """Validate an unchanged terminal source and freeze complete recovery lineage."""
    source = Path(source_dir).resolve()
    definition, state, report = (_json(source / name) for name in ("workflow.json", "state.json", "report.json"))
    require("calibration_reuse" not in definition, "Chained calibration reuse is forbidden")
    identity = _identity(definition)
    require(identity == _identity(target_definition), "Reuse configuration, input or semantic code mismatch")
    config = identity["judge_config"]
    require(config.get("backend") == "spiderssense" and config.get("protocol") == CONTEXT_PROTOCOL
            and config.get("provider_configuration_hash") and config.get("provider_ledger_path"),
            "Recovery requires the same explicit provider configuration and ledger")
    require(state.get("status") == report.get("status") == "calibration_not_qualified"
            and report.get("calibration_qualified") is False and report.get("contract_only") is False,
            "Reuse source must be a terminal real calibration failure")
    for name, digest in identity["versions"]["code"].items():
        require(_sha(ROOT / name) == digest, "Current semantic code differs from the source")
    require({path.name: _sha(path) for path in (ROOT / "prompts").glob("*.txt")}
            == identity["versions"]["prompts"], "Current prompts differ from the source")
    required = ["workflow.json", "report.json", "state.json", "calibration/judgements.jsonl",
                "calibration/summary.json", "inputs/calibration.cases.jsonl", "inputs/calibration.expected.jsonl"]
    for name, key in (("calibration.cases.jsonl", "calibration_cases"),
                      ("calibration.expected.jsonl", "calibration_expected")):
        require(_sha(source / "inputs" / name) == identity["sha256"][key], "Frozen calibration input mismatch")
    require(_sha(Path(identity["batch_dir"]) / "batch.json") == identity["sha256"]["batch_definition"]
            and _sha(prompt_path(CONTEXT_PROTOCOL)) == identity["sha256"]["judge_prompt"],
            "Batch or judge prompt hash mismatch")
    cases, expected, rows = (_rows(source / name) for name in
        ("inputs/calibration.cases.jsonl", "inputs/calibration.expected.jsonl", "calibration/judgements.jsonl"))
    ids = [case["answer_code"] for case in cases]
    require(len(set(ids)) == len(ids) and len(rows) == len(expected) == len(cases)
            and {row["answer_code"] for row in rows} == {row["answer_code"] for row in expected} == set(ids),
            "Calibration coverage or answer codes mismatch")
    require(state.get("judge_calls_reserved") == state.get("real_client_attempts") == len(cases)
            and state.get("fixture_call_reservations") == 0, "Source attempt accounting is incomplete")
    with tempfile.TemporaryDirectory() as directory:
        summary = summarize_calibration(source / "calibration/judgements.jsonl",
            source / "inputs/calibration.expected.jsonl", Path(directory) / "summary.json", protocol=CONTEXT_PROTOCOL)
    require(summary == _json(source / "calibration/summary.json") and not summary["qualified"]
            and summary["gate"]["not_contract_only"] and summary["gate"]["has_critical_and_acceptable_checks"],
            "Stored calibration gate is stale or not eligible for transport recovery")
    by_code, checks = ({row["answer_code"]: row for row in group} for group in (rows, summary["cases"]))
    copied, reused, retry = [], [], []
    for ordinal, case in enumerate(cases, 1):
        sources = _validate_case(case, protocol=CONTEXT_PROTOCOL)
        code, relative = case["answer_code"], f"steps/calibration/{ordinal:05d}"
        step = source / relative
        stamp, manifest = _json(step / "step.json"), _json(step / "output/manifest.json")
        require((step / "input.jsonl").read_text(encoding="utf-8") == json.dumps(case, ensure_ascii=False) + "\n",
                "Calibration step input differs from the frozen case")
        require(stamp.get("status") == "completed" and stamp.get("input_sha256") == _sha(step / "input.jsonl")
                and stamp.get("judgements_sha256") == _sha(step / "output/judgements.jsonl"), "Step stamp mismatch")
        require(manifest.get("status") in {"completed", "completed_with_errors"}
                and manifest.get("judge_protocol") == CONTEXT_PROTOCOL and manifest.get("backend") == config["backend"]
                and manifest.get("model") == config["model"] and manifest.get("contract_only") is False
                and manifest.get("real_model_calls") is True and manifest.get("case_count") == 1
                and manifest.get("input_sha256", {}).get("cases") == stamp["input_sha256"]
                and manifest.get("prompt_sha256") == identity["sha256"]["judge_prompt"]
                and manifest.get("code_sha256") == identity["versions"]["code"]["judge.py"], "Step manifest mismatch")
        require(_rows(step / "output/judgements.jsonl") == [by_code[code]], "Aggregate and step judgement differ")
        calls, traces = _rows(step / "output/calls.jsonl"), _rows(step / "output/trace.jsonl")
        require(len(calls) == len(traces) == 1 and calls[0].get("performed") is True
                and calls[0].get("answer_code") == traces[0].get("answer_code") == code, "Invalid call lineage")
        require(traces[0].get("messages") == [{"role": "system", "content": prompt_path(CONTEXT_PROTOCOL).read_text(encoding="utf-8")},
                {"role": "user", "content": json.dumps(case["payload"], ensure_ascii=False)}]
                and traces[0].get("response_schema") == CONTEXT_JUDGEMENT_SCHEMA, "Trace input differs from the case")
        files = [f"{relative}/{name}" for name in STEP_FILES]
        require({str(path.relative_to(step)).replace("\\", "/") for path in step.rglob("*") if path.is_file()}
                == set(STEP_FILES), "Unexpected calibration step files")
        required.extend(files)
        row, check = by_code[code], checks[code]
        require(row.get("contract_only") is False, "Fixture results cannot be reused")
        if row.get("status") == "assessed":
            validate_judgement(json.dumps(row["judgement"]), sources, protocol=CONTEXT_PROTOCOL)
            require(check["satisfied"] and check["utility_satisfied"] and calls[0].get("error") is None
                    and json.loads(traces[0]["completion_text"]) == row["judgement"], "Assessed calibration result failed its checks")
            copied.extend(files)
            reused.append({"answer_code": code, "ordinal": ordinal, "step": relative})
        else:
            call = calls[0]
            require(row.get("status") == "judge_error" and row.get("judgement") is None
                    and call.get("error") == "ProviderRequestError" and type(call.get("ledger_call_id")) is int
                    and call["ledger_call_id"] > 0 and (call.get("error_code") == "provider_connection_error"
                    or call.get("error_code") == "provider_http_error" and call.get("http_status") in {429, 500, 502, 503, 504}),
                    "Only recorded provider transport failures may be retried")
            retry.append({"answer_code": code, "ordinal": ordinal, "step": relative,
                          **{key: call[key] for key in ("error_code", "http_status", "ledger_call_id") if key in call}})
    require(reused and retry and summary["not_assessed_count"] == len(retry), "Recovery needs assessed results and transport failures")
    return {"reuse_version": 1, "source_dir": str(source), "target_identity": identity,
            "frozen_sha256": {name: _sha(_inside(source, name)) for name in sorted(required)},
            "copied_files": sorted(copied), "reused_cases": reused, "retry_cases": retry,
            "counts": {"reused_calibration_cases": len(reused), "transport_retry_cases": len(retry)},
            "source_accounting": {key: state[key] for key in
                ("judge_calls_reserved", "real_client_attempts", "fixture_call_reservations", "provider_accounting") if key in state}}


def import_calibration_reuse(descriptor, target_output) -> dict:
    """Copy only proven assessed steps; leave failed steps for one fresh call."""
    source, target = Path(descriptor["source_dir"]).resolve(), Path(target_output).resolve()
    require(not target.is_relative_to(source), "Recovery target must be separate from its source")
    for name, digest in descriptor["frozen_sha256"].items():
        require(_sha(_inside(source, name)) == digest, "Frozen reuse source changed")
    definition = _json(target / "workflow.json")
    require(describe_calibration_reuse(source, definition) == descriptor, "Reuse descriptor or target definition changed")
    for name, key in (("calibration.cases.jsonl", "calibration_cases"), ("calibration.expected.jsonl", "calibration_expected")):
        require(_sha(target / "inputs" / name) == descriptor["target_identity"]["sha256"][key], "Target calibration input mismatch")
    for name in descriptor["copied_files"]:
        path = _inside(target, name)
        require(not path.exists() or path.is_file() and _sha(path) == descriptor["frozen_sha256"][name],
                "Existing target calibration step differs from its source")
    for name in descriptor["copied_files"]:
        path = _inside(target, name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_inside(source, name), path)
        require(_sha(path) == descriptor["frozen_sha256"][name], "Copied calibration file hash mismatch")
    return dict(descriptor["counts"])
