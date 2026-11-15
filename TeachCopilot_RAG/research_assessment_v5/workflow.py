"""Finite no-retry paid phases; fresh outputs and an irreversible phase claim."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import random
import shutil
from urllib.error import HTTPError, URLError

from research_memory.clients import CompletionValidationError
from research_memory.spiderssense import BudgetError, ProviderRequestError
from .protocol import (CRITERIA, ROOT, SCHEMA, calibration_gate, digest, lines,
                       read, require, rows, sha, sources, validate, validate_expected, write)

PROJECT = ROOT.parent
FROZEN = PROJECT / ".local/research_memory/final_v4_20261018"
MODEL = "openai/gpt-6.1-sol"
EXPECTED_LABEL_FILES = ("calibration.expected.private.jsonl", "dev.expected.private.jsonl")


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_checks(frozen=FROZEN):
    freeze = read(frozen / "freeze.json")
    checks = {}
    for group in ("code", "prompts"):
        base = PROJECT / "research_memory" / ("prompts" if group == "prompts" else "")
        for name, expected in freeze["code_and_prompts"][group].items():
            path = base / name
            require(sha(path) == expected, f"Frozen implementation changed: {path.name}")
            checks[str(path.relative_to(PROJECT))] = expected
    for name, expected in freeze["inputs"].items():
        path = PROJECT / name
        require(sha(path) == expected, f"Frozen input changed: {path.name}")
        checks[name] = expected
    for path in (frozen / "freeze.json", frozen / "heldout/results.jsonl",
                 frozen / "heldout/batch.json", frozen / "heldout/batch_manifest.json"):
        checks[str(path.relative_to(PROJECT))] = sha(path)
    return checks


def _history(value):
    return {"profile_as_of_cutoff": {key: value["profile_as_of_cutoff"][key]
                                    for key in ("grade", "topic_id", "status")},
            "lessons": [{"lesson_id": lesson["lesson_id"], "observed_on": lesson["observed_on"],
                         "turns": [{key: turn[key] for key in ("turn_id", "role", "text")}
                                   for turn in lesson["turns"]]} for lesson in value["lessons"]]}


def prepare(output, *, frozen=FROZEN):
    output, frozen = Path(output), Path(frozen)
    integrity = frozen_checks(frozen)
    result_rows = rows(frozen / "heldout/results.jsonl")
    histories = rows(frozen / "heldout/snapshots/histories.jsonl")
    history_by_id = {row["history_id"]: row for row in histories}
    require(len(result_rows) == 128 and sum(row["status"] == "ok" for row in result_rows) == 94,
            "Expected the frozen 128 rows with 94 technical successes")
    require(len(history_by_id) == 16 and all(row["data_origin"].startswith("synthetic") for row in histories),
            "Expected 16 synthetic histories")
    expected_histories = read(frozen / "heldout/batch.json")["snapshot_sha256"]["histories.jsonl"]
    require(sha(frozen / "heldout/snapshots/histories.jsonl") == expected_histories, "Frozen history hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    for path in sorted((ROOT / "data").glob("*.jsonl")):
        shutil.copyfile(path, output / "inputs" / path.name)
    for name in EXPECTED_LABEL_FILES:
        validate_expected(rows(output / "inputs" / name))
    result_rows = list(result_rows)
    random.Random(20261022).shuffle(result_rows)
    cases, mapping = [], []
    for number, row in enumerate(result_rows, 1):
        code = f"A{number:03d}"
        mapping.append({"answer_code": code, **{key: row.get(key) for key in
                        ("result_id", "method", "history_id", "request_id", "status", "error")}})
        if row["status"] != "ok":
            continue
        history = history_by_id[row["history_id"]]
        case = {"answer_code": code, "data_origin": history["data_origin"],
                "execution_status": "ok", "contract_only": False, "payload": {
                    "history": _history(history), "question": row["request_text"], "answer": row["action"],
                    "generator_context": {"profile_as_of_cutoff": row["profile_as_of_cutoff"],
                                          "selected_memory": row["memory_context"]}}}
        sources(case)
        cases.append(case)
    lines(output / "inputs/main.cases.jsonl", cases)
    lines(output / "mapping.private.jsonl", mapping)
    selection = PROJECT / "research_memory/data/v4/review.selection.json"
    shutil.copyfile(selection, output / "review.selection.private.json")
    write(output / "source_integrity.json", integrity)
    write(output / "preparation.json", {"protocol": "additional-memory-assessment-v5", "created_at_utc": now(),
          "frozen_rows": 128, "accepted_rows": 94, "technical_failures": 34,
          "input_sha256": {path.name: sha(path) for path in sorted((output / "inputs").glob("*.jsonl"))},
          "frozen_sha256": integrity, "status": "prepared", "method_blinding_seed": 20261022,
          "labels_origin": "agent authored synthetic; not human validation",
          "heldout_policy": "new cases and expected labels fixed before dev; never used for protocol tuning"})
    return read(output / "preparation.json")


def check_inputs(workspace):
    workspace = Path(workspace)
    stamp = read(workspace / "preparation.json")
    for name, expected in stamp["input_sha256"].items():
        require(sha(workspace / "inputs" / name) == expected, f"Prepared case file changed: {name}")
    require(frozen_checks() == stamp["frozen_sha256"], "Frozen source changed since preparation")
    for name in EXPECTED_LABEL_FILES:
        validate_expected(rows(workspace / "inputs" / name))


def _settings(config):
    require(config["model"] == MODEL and config["reasoning_effort"] == "medium"
            and config["max_output_tokens"] == 8192, "Judge model, reasoning and output must match authorization")
    require(Decimal(str(config["max_cost_usd"])) == Decimal("16") and config["max_calls"] == 126,
            "Evaluation must use the separate $16 / 126-call authorization")


def _tree_hashes(directory):
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def _completion_proof(workspace, config_path, authorization_path):
    """Only the two never attempted tail cases may consume the remaining dev calls."""
    workspace = Path(workspace)
    require(not (workspace / "frozen_protocol.json").exists(), "Dev cannot be completed after protocol freeze")
    original = workspace / "dev"
    manifest = read(original / "manifest.json")
    cases = rows(workspace / "inputs/dev.cases.jsonl")
    outcomes = rows(original / "judgements.jsonl")
    require(len(cases) == len(outcomes) == 8 and manifest["attempted_calls"] == 6
            and manifest["status"] == "completed_with_errors", "Only the original six-call interrupted dev is eligible")
    require([row["answer_code"] for row in outcomes] == [case["answer_code"] for case in cases], "Original dev IDs changed")
    require(manifest["configuration_sha256"] == sha(config_path)
            and manifest["authorization_sha256"] == sha(authorization_path), "Original dev settings/authorization changed")
    require(manifest["cases_sha256"] == sha(workspace / "inputs/dev.cases.jsonl")
            and read(original / "settings.json") == read(config_path)
            and read(original / "schema.json") == SCHEMA, "Original dev cases/schema/settings changed")
    for name in ("prompt.txt", "rubric.json", "protocol.py"):
        require(manifest["protocol_sha256"][str((ROOT / name).relative_to(PROJECT))] == sha(ROOT / name),
                "Completion cannot change prompt, rubric, validator or schema")
    claims = Path(authorization_path).resolve().parent / "assessment_v5_execution_claims"
    claim = read(claims / "dev.json")
    require(Path(claim["workspace"]).resolve() == workspace.resolve(), "Completion cannot reuse another workspace's dev claim")
    selected = []
    for number, (case, outcome) in enumerate(zip(cases, outcomes), 1):
        step = original / "calls" / f"{number:03d}"
        if number <= 6:
            require((step / "attempt.json").is_file() and outcome.get("new_reserved_calls") == 1
                    and outcome["status"] in {"assessed", "judge_error"}, "An attempted original dev case lacks durable proof")
            continue
        require(outcome["status"] == "not_assessed" and outcome.get("error", {}).get("code") == "batch_stopped_no_retry"
                and outcome.get("new_reserved_calls", 0) == 0 and outcome.get("judgement") is None,
                "Completion only accepts never attempted outcomes")
        require(step.is_dir() and {path.name for path in step.iterdir()} == {"request.json"},
                "Never attempted proof failed: original case has attempt, HTTP, completion or outcome artifacts")
        expected_messages = [{"role": "system", "content": (ROOT / "prompt.txt").read_text(encoding="utf-8")},
                             {"role": "user", "content": json.dumps(case["payload"], ensure_ascii=False)}]
        original_request = read(step / "request.json")
        require(original_request["messages"] == expected_messages and original_request["schema"] == SCHEMA
                and original_request["settings"] == read(config_path), "Unsent original request differs from completion request")
        selected.append(case)
    require(len(selected) == 2 and outcomes[5].get("error", {}).get("code") == "provider_connection_error",
            "Completion scope must retain the sixth unknown outcome and only call cases seven/eight")
    return selected, {"original_dev_sha256": _tree_hashes(original), "original_attempted_calls": 6,
                      "never_attempted_answer_codes": [case["answer_code"] for case in selected],
                      "retained_unknown_answer_code": outcomes[5]["answer_code"], "maximum_total_dev_calls": 8}


def _dev_receipt(workspace):
    workspace = Path(workspace)
    original = read(workspace / "dev/manifest.json")
    merged_path = workspace / "dev_merged/receipt.json"
    if not merged_path.exists():
        return original, workspace / "dev/judgements.jsonl", None
    merged = read(merged_path)
    require(_tree_hashes(workspace / "dev") == merged["original_dev_sha256"]
            and _tree_hashes(workspace / "dev_completion") == merged["completion_sha256"], "Dev source bytes changed after merging")
    result_path = workspace / "dev_merged/judgements.jsonl"
    require(sha(result_path) == merged["merged_judgements_sha256"], "Merged dev outcomes changed")
    require(len(rows(result_path)) == len({row["answer_code"] for row in rows(result_path)}) == 8,
            "Merged dev must retain eight unique outcomes")
    return {**original, "attempted_calls": merged["attempted_calls"], "status": merged["status"]}, result_path, merged


def _merge_dev(workspace, proof):
    workspace = Path(workspace)
    require(_tree_hashes(workspace / "dev") == proof["original_dev_sha256"], "Original dev bytes changed during completion")
    completion = read(workspace / "dev_completion/manifest.json")
    require(completion["attempted_calls"] == 2 and completion["status"] in {"completed", "completed_with_errors"},
            "Both never attempted dev cases must have a single terminal attempt")
    original = rows(workspace / "dev/judgements.jsonl")
    new = rows(workspace / "dev_completion/judgements.jsonl")
    require([row["answer_code"] for row in new] == proof["never_attempted_answer_codes"], "Completion IDs changed")
    require(all(row.get("new_reserved_calls") == 1 for row in new)
            and completion["before_accounting"]["calls_reserved"] == 6
            and completion["after_accounting"]["calls_reserved"] == 8, "Completion must retain exactly eight total dev reservations")
    merged_rows = original[:6] + new
    require(len({row["answer_code"] for row in merged_rows}) == 8, "Dev completion duplicates an attempted case")
    target = workspace / "dev_merged"
    target.mkdir(exist_ok=False)
    lines(target / "judgements.jsonl", merged_rows)
    receipt = {**proof, "completion_sha256": _tree_hashes(workspace / "dev_completion"), "attempted_calls": 8,
               "status": "completed_with_errors", "paid_retries": 0,
               "merged_judgements_sha256": sha(target / "judgements.jsonl"),
               "cost_policy": "latest shared ledger total; original/completion ledger snapshots are not added"}
    write(target / "receipt.json", receipt)
    return receipt


def freeze(workspace, config_path, authorization_path, *, correction_note=None):
    workspace = Path(workspace)
    check_inputs(workspace)
    dev, dev_path, dev_receipt = _dev_receipt(workspace)
    require(dev["phase"] == "dev" and dev["attempted_calls"] == 8 and dev["status"] in {"completed", "completed_with_errors"},
            "Freeze requires the one complete eight-call development run")
    require(not (workspace / "frozen_protocol.json").exists(), "Protocol already frozen; calibration cannot be retuned")
    config = read(config_path)
    _settings(config)
    protocol_files = _protocol_files()
    current = {str(path.relative_to(PROJECT)): sha(path) for path in protocol_files}
    dev_hashes = dev["protocol_sha256"]
    changed = {name for name in current if current[name] != dev_hashes.get(name)}
    corrected = bool(changed or dev_receipt)
    require(not corrected or bool(correction_note and correction_note.strip()), "The single dev correction needs an explicit note")
    snapshot = workspace / "protocol_snapshot"
    snapshot.mkdir(exist_ok=False)
    for path in protocol_files:
        target = snapshot / path.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(config_path, snapshot / "provider.json")
    shutil.copyfile(authorization_path, snapshot / "authorization.json")
    write(snapshot / "schema.json", SCHEMA)
    result = {"created_at_utc": now(), "protocol_sha256": current,
              "schema_sha256": digest(SCHEMA), "configuration_sha256": sha(config_path),
              "authorization_sha256": sha(authorization_path), "configuration": config,
              "prepared_inputs": read(workspace / "preparation.json")["input_sha256"],
              "dev_sha256": sha(dev_path), "dev_receipt": dev_receipt,
              "correction_count": int(corrected), "correction_note": correction_note,
              "calibration_attempts_allowed": 1, "post_calibration_tuning": False}
    write(workspace / "frozen_protocol.json", result)
    return result


def check_freeze(workspace, config_path, authorization_path):
    frozen = read(Path(workspace) / "frozen_protocol.json")
    _, dev_path, dev_receipt = _dev_receipt(workspace)
    require(sha(dev_path) == frozen["dev_sha256"] and dev_receipt == frozen.get("dev_receipt"), "Frozen dev receipt changed")
    for name, expected in frozen["protocol_sha256"].items():
        require(sha(PROJECT / name) == expected, "Protocol changed after freeze; no calibration retuning allowed")
    require(sha(config_path) == frozen["configuration_sha256"] and
            sha(authorization_path) == frozen["authorization_sha256"] and digest(SCHEMA) == frozen["schema_sha256"],
            "Frozen provider configuration, authorization or schema changed")
    return frozen


def _transport_diagnostic(directory, exc, key, stage):
    reason = getattr(exc, "reason", exc)
    write(Path(directory) / "http.transport.error.json", {
        "type": type(exc).__name__, "reason_type": type(reason).__name__, "stage": stage,
        "reason": str(reason).replace(key.decode(), "[REDACTED_CREDENTIAL]")[:2048],
        "errno": getattr(reason, "errno", None), "reservation_policy": "retain unknown; no retry"})


class _RecordedResponse:
    def __init__(self, response, path, key):
        self.response, self.path, self.key = response, path, key

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def read(self, count=-1):
        try:
            value = self.response.read(count)
        except (URLError, TimeoutError, OSError) as exc:
            _transport_diagnostic(self.path.parent, exc, self.key, "read")
            raise
        self.path.write_bytes(value.replace(self.key, b"[REDACTED_CREDENTIAL]") if self.key else value)
        write(self.path.with_suffix(".sha256.json"), {"original_body_sha256": __import__("hashlib").sha256(value).hexdigest(),
              "credential_redacted": bool(self.key and self.key in value)})
        return value


class _RecordingOpener:
    """Records the actual HTTP body, including client's schema projection."""
    def __init__(self, client):
        self.opener, self.client, self.directory = client._opener, client, None

    def open(self, request, **kwargs):
        require(self.directory is not None, "Missing request recording directory")
        (self.directory / "http.request.body.json").write_bytes(request.data)
        key = self.client._key().encode()
        try:
            response = self.opener.open(request, **kwargs)
        except HTTPError as exc:
            body = exc.read(8 * 1024 * 1024 + 1)
            (self.directory / "http.error.body.raw").write_bytes(body.replace(key, b"[REDACTED_CREDENTIAL]"))
            write(self.directory / "http.error.body.sha256.json", {
                "original_body_sha256": __import__("hashlib").sha256(body).hexdigest(),
                "credential_redacted": key in body, "http_status": exc.code})
            raise
        except (URLError, TimeoutError, OSError) as exc:
            _transport_diagnostic(self.directory, exc, key, "open")
            raise
        return _RecordedResponse(response, self.directory / "http.response.body.raw", key)


def _protocol_files():
    return [ROOT / "prompt.txt", ROOT / "rubric.json", *sorted(ROOT.rglob("*.py")),
            PROJECT / "scripts/run_assessment_v5.py", PROJECT / "local_app/budget_authorization.py"]


def _protocol_hashes():
    return {str(path.relative_to(PROJECT)): sha(path) for path in _protocol_files()}


def run_phase(workspace, phase, config_path, authorization_path, *, execute=False, client=None):
    require(phase in {"dev", "dev_completion", "calibration", "main"}, "Unknown phase")
    workspace = Path(workspace)
    check_inputs(workspace)
    config = read(config_path)
    _settings(config)
    path = workspace / "inputs" / ("dev.cases.jsonl" if phase == "dev_completion" else f"{phase}.cases.jsonl")
    cases = rows(path)
    completion_proof = None
    if phase == "dev_completion":
        cases, completion_proof = _completion_proof(workspace, config_path, authorization_path)
    require(len(cases) == {"dev": 8, "dev_completion": 2, "calibration": 24, "main": 94}[phase], "Phase case count changed")
    for case in cases:
        sources(case)
    if phase not in {"dev", "dev_completion"}:
        check_freeze(workspace, config_path, authorization_path)
    if phase == "main":
        gate = read(workspace / "calibration/qualification.json")
        persisted_rows = rows(workspace / "calibration/judgements.jsonl")
        fresh_gate = calibration_gate(rows(workspace / "inputs/calibration.cases.jsonl"),
                     rows(workspace / "inputs/calibration.expected.private.jsonl"), persisted_rows)
        require(gate == fresh_gate and gate["qualified"], "Calibration is not qualified; main evaluation is blocked")
    plan = {"phase": phase, "case_count": len(cases), "max_new_paid_calls": len(cases),
            "model": config["model"], "reasoning_effort": config["reasoning_effort"],
            "max_output_tokens": config["max_output_tokens"], "network_used": False,
            "dry_run": not execute, "paid_retries": 0}
    if not execute:
        return plan
    if client is None:
        from local_app.budget_authorization import build_authorized_client
        client = build_authorized_client(config_path, authorization_path, purpose="evaluation")
    preflight = client.preflight()
    require(not preflight["blocked"], "Authorized budget ledger is blocked")
    if phase == "dev_completion":
        require(preflight["calls_reserved"] == 6, "Dev completion requires exactly six existing reserved calls, including unknown case six")
    target = workspace / phase
    target.mkdir(exist_ok=False)
    # Exclusive phase creation is deliberately irreversible, even after unknown outcome.
    write(target / "claim.json", {"created_at_utc": now(), "phase": phase, "paid_retries": 0,
          "status": "claimed", "policy": "never resume or repeat this phase"})
    for name in _protocol_hashes():
        snapshot_path = target / "protocol_snapshot" / name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, snapshot_path)
    write(target / "schema.json", SCHEMA)
    write(target / "settings.json", config)
    recorder = _RecordingOpener(client) if hasattr(client, "_opener") else None
    if recorder:
        client._opener = recorder
    manifest = {**plan, "dry_run": False, "status": "running", "created_at_utc": now(),
                "attempted_calls": 0, "assessed": 0, "errors": 0, "network_used": True,
                "protocol_sha256": _protocol_hashes(), "cases_sha256": sha(path),
                "configuration_sha256": sha(config_path), "authorization_sha256": sha(authorization_path),
                "before_accounting": preflight, "assessment_origin": "machine_only; not human expertise"}
    if completion_proof:
        manifest["completion_proof"] = completion_proof
    write(target / "manifest.json", manifest)
    # The cross-workspace claim is written last, after the durable setup above has fully
    # succeeded and immediately before the first paid call – a transient setup failure
    # must not permanently consume the paid phase with zero calls.
    claims = Path(authorization_path).resolve().parent / "assessment_v5_execution_claims"
    claims.mkdir(exist_ok=True)
    global_claim = claims / f"{phase}.json"
    with global_claim.open("x", encoding="utf-8") as stream:
        json.dump({"phase": phase, "workspace": str(workspace.resolve()), "created_at_utc": now(),
                   "policy": "one attempt across all workspaces; unknown outcome cannot be retried"}, stream, ensure_ascii=False)
    answers = []
    stopped = False
    try:
        with (target / "judgements.jsonl").open("x", encoding="utf-8") as output:
            for number, case in enumerate(cases, 1):
                step = target / "calls" / f"{number:03d}"
                step.mkdir(parents=True, exist_ok=False)
                messages = [{"role": "system", "content": (ROOT / "prompt.txt").read_text(encoding="utf-8")},
                            {"role": "user", "content": json.dumps(case["payload"], ensure_ascii=False)}]
                write(step / "request.json", {"messages": messages, "schema": SCHEMA, "settings": config,
                      "answer_code": case["answer_code"], "expected_labels_sent": False})
                row = {"answer_code": case["answer_code"], "status": "not_assessed", "judgement": None,
                       "contract_only": False, "error": None, "assessment_origin": "machine_only"}
                if stopped:
                    row["error"] = {"code": "batch_stopped_no_retry"}
                else:
                    manifest["attempted_calls"] += 1
                    write(target / "manifest.json", manifest)
                    write(step / "attempt.json", {"status": "attempting_unknown_until_persisted", "created_at_utc": now()})
                    if recorder:
                        recorder.directory = step
                    before_call = client.preflight()
                    try:
                        completion = client.complete(messages, "judge", SCHEMA)
                        write(step / "completion.json", {"text": completion.text, "usage": completion.usage,
                              "elapsed_seconds": completion.elapsed_seconds, "metadata": completion.metadata})
                        judgement = validate(completion.text, case)
                        row.update(status="assessed", judgement=judgement)
                        manifest["assessed"] += 1
                    except Exception as exc:
                        if isinstance(exc, CompletionValidationError):
                            completion = exc.completion
                            write(step / "completion.json", {"text": completion.text, "usage": completion.usage,
                                  "elapsed_seconds": completion.elapsed_seconds, "metadata": completion.metadata})
                        row.update(status="judge_error", error={"code": getattr(exc, "code", "invalid_judgement"),
                                   "type": type(exc).__name__, "http_status": getattr(exc, "http_status", None),
                                   "ledger_call_id": getattr(exc, "ledger_call_id", None)})
                        manifest["errors"] += 1
                        # Each independent case has one attempt. Unknown outcomes retain their reservation.
                        # The original dev behavior is retained to reproduce its one documented completion path.
                        stopped = isinstance(exc, BudgetError) or phase == "dev" and isinstance(exc, ProviderRequestError)
                    after_call = client.preflight()
                    row["charged_or_reserved_usd"] = str(Decimal(after_call["charged_or_reserved_usd"])
                                                         - Decimal(before_call["charged_or_reserved_usd"]))
                    row["new_reserved_calls"] = after_call["calls_reserved"] - before_call["calls_reserved"]
                    write(step / "outcome.json", row)
                answers.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                write(target / "manifest.json", manifest)
        manifest["status"] = "completed_with_errors" if any(row["status"] != "assessed" for row in answers) else "completed"
    finally:
        manifest["finished_at_utc"] = now()
        manifest["after_accounting"] = client.preflight()
        manifest["unprocessed_count"] = len(cases) - len(answers)
        if manifest["status"] == "running":
            manifest["status"] = "interrupted_unknown_no_retry"
        write(target / "manifest.json", manifest)
        write(target / "terminal_outcome.json", {"status": manifest["status"], "paid_retries": 0,
              "attempted_calls": manifest["attempted_calls"], "assessed": manifest["assessed"],
              "unprocessed_count": manifest["unprocessed_count"], "unknown_reservations_retained": True})
        if recorder:
            client._opener = recorder.opener
    if phase == "calibration":
        qualification = calibration_gate(cases, rows(workspace / "inputs/calibration.expected.private.jsonl"), answers)
        write(target / "qualification.json", qualification)
        if not qualification["qualified"]:
            human_packet(workspace)
    elif phase in {"dev", "dev_completion"}:
        expected = {row["answer_code"]: row for row in rows(workspace / "inputs/dev.expected.private.jsonl")}
        write(target / "review.private.json", {"origin": "agent authored expected labels; not human review",
              "tuning_scope": "only these eight development results; no repeat calls",
              "cases": [{"answer_code": row["answer_code"], "status": row["status"],
                         "expected": expected[row["answer_code"]], "actual": row["judgement"],
                         "error": row["error"]} for row in answers]})
        if phase == "dev_completion":
            _merge_dev(workspace, completion_proof)
    report(workspace)
    return manifest


def human_packet(workspace):
    workspace = Path(workspace)
    target = workspace / "human_review"
    target.mkdir(exist_ok=False)
    selection = read(workspace / "review.selection.private.json")
    mapping = rows(workspace / "mapping.private.jsonl")
    eligible = [row for row in mapping if row["request_id"] in selection["requests"]]
    require(len(eligible) == 20, "Human review must preserve 20 preselected places")
    cases = {case["answer_code"]: case for case in rows(workspace / "inputs/main.cases.jsonl")}
    random.Random(20261023).shuffle(eligible)
    packet, private = [], []
    text = ["# Пакет для человеческого просмотра\n",
            "Человеческий просмотр ещё не выполнен. Это 20 заранее выбранных мест из пяти историй; методы закрыты. "
            "Агентские метки не являются экспертной проверкой.\n"]
    histories = {row["history_id"]: row for row in rows(FROZEN / "heldout/snapshots/histories.jsonl")}
    frozen_rows = {row["result_id"]: row for row in rows(FROZEN / "heldout/results.jsonl")}
    for number, meta in enumerate(eligible, 1):
        code = f"R{number:02d}"
        original = frozen_rows[meta["result_id"]]
        case = cases.get(meta["answer_code"])
        item = {"review_code": code, "history": _history(histories[meta["history_id"]]),
                "question": original["request_text"], "answer": case["payload"]["answer"] if case else None,
                "technical_status": meta["status"], "failure": None if case else meta.get("error"),
                "human_review_status": "not_reviewed"}
        packet.append(item)
        private.append({"review_code": code, **meta})
        text.append(f"\n## Место {code}\n\n```json\n{json.dumps(item, ensure_ascii=False, indent=2)}\n```\n")
    lines(target / "packet.jsonl", packet)
    lines(target / "mapping.private.jsonl", private)
    (target / "packet.md").write_text("".join(text), encoding="utf-8")
    write(target / "manifest.json", {"places": 20, "histories": 5, "methods_blinded": True,
          "selection_sha256": sha(workspace / "review.selection.private.json"), "human_review_performed": False})


def report(workspace):
    workspace = Path(workspace)
    mapping = rows(workspace / "mapping.private.jsonl")
    calibration = read(workspace / "calibration/qualification.json") if (workspace / "calibration/qualification.json").exists() else None
    main_rows = rows(workspace / "main/judgements.jsonl") if (workspace / "main/judgements.jsonl").exists() else []
    by_code = {row["answer_code"]: row for row in main_rows}
    by_method = defaultdict(lambda: {"total": 0, "technical_ok": 0, "technical_failure": 0,
                    "assessed": 0, "substantial_distortion": 0, "verdicts": Counter(), "utilities": Counter(),
                    "charged_or_reserved_usd": "0", "criteria": {key: Counter() for key in CRITERIA}})
    history_comparison = defaultdict(dict)
    for meta in mapping:
        method = by_method[meta["method"]]
        method["total"] += 1
        method["technical_ok" if meta["status"] == "ok" else "technical_failure"] += 1
        row = by_code.get(meta["answer_code"], {})
        method["charged_or_reserved_usd"] = str(Decimal(method["charged_or_reserved_usd"])
                                                + Decimal(row.get("charged_or_reserved_usd", "0")))
        judgement = row.get("judgement") if row.get("status") == "assessed" else None
        if judgement:
            method["assessed"] += 1
            method["substantial_distortion"] += int(judgement["substantial_distortion"])
            method["verdicts"][judgement["verdict"]] += 1
            method["utilities"][judgement["utility"]] += 1
            for key in CRITERIA:
                method["criteria"][key][str(judgement["criteria"][key]["score"])] += 1
        history_comparison[meta["history_id"]].setdefault(meta["method"], []).append({
            "request_id": meta["request_id"], "technical_status": meta["status"],
            "assessment_status": row.get("status", "not_assessed"),
            "verdict": judgement["verdict"] if judgement else None,
            "utility": judgement["utility"] if judgement else None,
            "substantial_distortion": judgement["substantial_distortion"] if judgement else None})
    phases = {name: read(workspace / name / "manifest.json") for name in ("dev", "dev_completion", "calibration", "main")
              if (workspace / name / "manifest.json").exists()}
    after = list(phases.values())[-1].get("after_accounting") if phases else None
    merged_dev = _dev_receipt(workspace)[2] if (workspace / "dev_merged/receipt.json").exists() else None
    result = {"protocol": "additional-memory-assessment-v5", "assessment_origin": "additional automatic evaluation; no human expertise",
              "frozen_rows": 128, "technical_ok": 94, "technical_failures": 34,
              "calibration_qualified": calibration["qualified"] if calibration else None,
              "assessed": sum(row["status"] == "assessed" for row in main_rows),
              "coverage_denominator": 128, "technical_eligible_denominator": 94,
              "methods": dict(by_method), "within_history": dict(history_comparison),
              "cost_accounting": after, "dev_completion_receipt": merged_dev,
              "phase_statuses": {name: value["status"] for name, value in phases.items()},
              "paid_retries": 0, "winner_claimed": False, "independent_observations": "16 histories, not 128 rows"}
    write(workspace / "report.json", result)
    text = ["# Дополнительная автоматическая оценка сохранённых ответов\n\n",
            "Исходная генерация сохранена. Синтетические данные и агентские метки; человеческой экспертизы нет.\n\n",
            f"Калибровка qualified={result['calibration_qualified']}. Основное покрытие: {result['assessed']}/128; "
            "технически допустимы 94/128. Остальные 34 строки сохранены как технические отказы.\n\n",
            "| Метод | Технически ok | Отказы | Оценено | Существенные искажения | useful |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"]
    for name, method in sorted(by_method.items()):
        text.append(f"| {name} | {method['technical_ok']}/32 | {method['technical_failure']} | {method['assessed']}/32 | "
                    f"{method['substantial_distortion']} | {method['utilities']['useful']} |\n")
    text.append("\nПри нулевом покрытии нулевые содержательные счётчики не означают отсутствие ошибок. "
                "Пять критериев, все verdict/utility и сопоставление внутри 16 историй сохранены раздельно в report.json. "
                "Два запроса и четыре метода внутри истории зависимы; победитель не объявляется.\n\n")
    if after:
        text.append(f"Новый журнал оценки: {after['calls_reserved']} зарезервированных вызовов; "
                    f"консервативно списано или зарезервировано ${after['charged_or_reserved_usd']}. "
                    "Старые расходы и резервы учитываются отдельным переходом, здесь повторно не суммируются.\n")
    (workspace / "report.md").write_text("".join(text), encoding="utf-8")
    return result
