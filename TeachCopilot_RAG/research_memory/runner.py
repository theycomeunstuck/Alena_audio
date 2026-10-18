"""Recorded pilot runs and a separate, shuffled human-review export."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

from .clients import CompletionValidationError, FixtureClient, OllamaClient
from .contracts import action_schema, validate_action
from .dataset import load_dataset, require
from .methods import METHODS, PROMPT_DIR, Memory, messages


ROOT = Path(__file__).parent


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _line(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CallJournal:
    def __init__(self, client, stream, max_calls: int | None = None, trace_stream=None):
        self.client, self.stream, self.max_calls = client, stream, max_calls
        self.trace_stream = trace_stream
        self.performed = self.attempted = 0
        self.context: dict = {}

    def __call__(self, messages, purpose: str, response_schema: dict | None = None) -> str:
        self.attempted += 1
        entry = {**self.context, "call_id": self.attempted, "purpose": purpose,
                 "input_chars": sum(len(m["content"]) for m in messages),
                 "response_schema_chars": len(json.dumps(response_schema, ensure_ascii=False))
                    if response_schema is not None else 0,
                 "response_chars": None, "usage": {"input_tokens": None, "output_tokens": None},
                 "elapsed_seconds": None, "error": None, "performed": False}
        trace = {"call_id": self.attempted, **self.context, "purpose": purpose,
                 "messages": messages, "response_schema": response_schema, "completion_text": None,
                 "completion_metadata": {}}

        def persist():
            _line(self.stream, entry)
            if self.trace_stream is not None:
                _line(self.trace_stream, {**trace, "error": entry["error"], "performed": entry["performed"]})
        if self.max_calls is not None and self.performed >= self.max_calls:
            entry.update(error="max_calls_exceeded", elapsed_seconds=0)
            persist()
            raise RuntimeError("max_calls_exceeded")
        started = time.perf_counter()
        self.performed += 1
        entry["performed"] = True

        def capture(completion):
            entry.update(usage=completion.usage, elapsed_seconds=completion.elapsed_seconds,
                         response_chars=len(completion.text))
            trace.update(completion_text=completion.text, completion_metadata={
                key: completion.metadata[key] for key in ("backend", "model", "purpose", "total_duration",
                    "load_duration", "request_timeout_seconds", "scientific_result", "input_budget",
                    "network_request_performed") if key in completion.metadata})
            if completion.metadata.get("network_request_performed") is False:
                entry["performed"] = False
                self.performed -= 1

        try:
            completion = self.client.complete(messages, purpose, response_schema)
            capture(completion)
        except Exception as exc:
            entry.update(error=type(exc).__name__, elapsed_seconds=time.perf_counter() - started)
            from .spiderssense import ProviderRequestError

            if isinstance(exc, ProviderRequestError) and exc.code in {
                "provider_http_error", "provider_connection_error", "provider_transport_error"
            }:
                entry["error_code"] = exc.code
                safe_failure = {"provider_error_code": exc.code}
                if type(exc.http_status) is int and 100 <= exc.http_status <= 599:
                    entry["http_status"] = exc.http_status
                    safe_failure["http_status"] = exc.http_status
                if type(exc.ledger_call_id) is int and exc.ledger_call_id > 0:
                    entry["ledger_call_id"] = exc.ledger_call_id
                    safe_failure["ledger_call_id"] = exc.ledger_call_id
                trace["completion_metadata"].update(safe_failure)
            if isinstance(exc, CompletionValidationError):
                capture(exc.completion)
                entry["error_code"] = exc.code
                trace["completion_metadata"]["validation_error_code"] = exc.code
            persist()
            # Never persist arbitrary provider bodies, URLs with credentials, or prompts in errors.
            failure = RuntimeError(f"Model call failed ({type(exc).__name__})")
            if entry.get("error_code"):
                failure.code = entry["error_code"]
            raise failure from None
        persist()
        return completion.text


def run(histories_path, requests_path, output, *, backend="fixture", history_ids=None,
        methods=METHODS, budget_chars=4000, max_calls=None, config_path=None,
        model=None, base_url=None, client=None) -> dict:
    """Run each memory once per history, then answer its cutoff requests."""
    dataset = load_dataset(histories_path, requests_path)
    require(backend in {"fixture", "ollama"}, "Unsupported backend")
    require(isinstance(budget_chars, int) and budget_chars >= 128, "budget_chars must be at least 128")
    require(max_calls is None or isinstance(max_calls, int) and max_calls > 0, "max_calls must be positive")
    methods = tuple(methods)
    require(bool(methods) and len(methods) == len(set(methods)) and set(methods) <= set(METHODS), "Invalid methods")
    selected_ids = set(history_ids or dataset.histories)
    require(selected_ids <= set(dataset.histories) and bool(selected_ids), "Unknown or empty history selection")
    require(all(h["data_origin"].startswith("synthetic") for h in dataset.histories.values()),
            "This pilot accepts synthetic data only; traces include full model inputs and outputs")
    config_path = Path(config_path or ROOT / "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if model is not None:
        config["model"] = model
    if base_url is not None:
        config["base_url"] = base_url
    config["budget_chars"] = budget_chars
    allowed_config = {"protocol_version", "model", "base_url", "timeout_seconds", "context_tokens",
                      "max_output_tokens", "budget_chars"}
    require(set(config) == allowed_config, "Unexpected or missing configuration fields")
    local_client = OllamaClient(base_url=config["base_url"], model=config["model"], timeout=config["timeout_seconds"],
                                context_tokens=config["context_tokens"], max_output_tokens=config["max_output_tokens"])
    # Configuration is explicit and local; this stand has no provider/API-key settings.
    if client is None:
        client = FixtureClient() if backend == "fixture" else local_client
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "protocol_version": config["protocol_version"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend, "scientific_result": False,
        "run_kind": "contract_only" if backend == "fixture" else "unreviewed_development_pilot",
        "result_type": "contract_only" if backend == "fixture" else "pilot_unreviewed", "real_model_calls": False,
        "split": dataset.split, "data_origins": sorted({h["data_origin"] for h in dataset.histories.values()}),
        "history_ids": sorted(selected_ids), "methods": list(methods), "budget_chars": budget_chars,
        "max_calls": max_calls, "resolved_config": config,
        "input_sha256": {"histories": _hash(Path(histories_path)), "requests": _hash(Path(requests_path))},
        "config_sha256": _hash(config_path),
        "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "prompt_sha256": {p.name: _hash(p) for p in sorted(PROMPT_DIR.glob("*.txt"))},
        "code_sha256": {p.name: _hash(p) for p in sorted(ROOT.glob("*.py"))},
        "validation_scope": "format_and_source_membership_only; no learning outcomes or automatic quality scores",
        "status": "running", "result_count": 0, "error_result_count": 0,
    }
    _write(output / "manifest.json", manifest)
    journal = None
    try:
        with (output / "calls.jsonl").open("w", encoding="utf-8") as calls, \
                (output / "results.jsonl").open("w", encoding="utf-8") as results, \
                (output / "trace.jsonl").open("w", encoding="utf-8") as trace:
            journal = CallJournal(client, calls, max_calls, trace)
            for history_id, history in dataset.histories.items():
                if history_id not in selected_ids:
                    continue
                requests = [q for q in dataset.requests if q["history_id"] == history_id]
                for method in methods:
                    memory = Memory(method, journal, budget_chars, history["profile_as_of_cutoff"]["topic_id"])
                    update_errors = []
                    for lesson in history["lessons"]:
                        journal.context = {"history_id": history_id, "method": method, "lesson_id": lesson["lesson_id"]}
                        try:
                            memory.update(lesson)
                        except Exception as exc:
                            update_errors.append({"lesson_id": lesson["lesson_id"], "error": str(exc),
                                                  "error_type": type(exc).__name__, "error_code": getattr(exc, "code", None)})
                    for request in requests:
                        context = memory.context(request["topic_id"])
                        row = {"result_id": f"{history_id}:{request['request_id']}:{method}",
                               "history_id": history_id, "request_id": request["request_id"],
                               "method": method, "backend": backend, "scientific_result": False,
                               "result_type": manifest["result_type"],
                               "run_kind": manifest["run_kind"], "request_text": request["text"],
                               "profile_as_of_cutoff": history["profile_as_of_cutoff"],
                               "memory_context": context.text, "memory_chars": len(context.text),
                               "memory_events": memory.events, "memory_errors": update_errors,
                               "action": None, "source_turns": [], "status": "memory_error" if update_errors else "ok",
                               "error": None}
                        if not update_errors:
                            journal.context = {"history_id": history_id, "method": method,
                                               "request_id": request["request_id"]}
                            try:
                                text = journal(messages("action", {
                                    "profile_as_of_cutoff": history["profile_as_of_cutoff"],
                                    "request": {"topic_id": request["topic_id"], "text": request["text"]},
                                    "selected_memory": context.text}), "action", action_schema(context.evidence_ids))
                                row["action"] = validate_action(text, context.evidence_ids)
                                row["source_turns"] = [memory.sources[x] for x in row["action"]["evidence_turn_ids"]]
                            except Exception as exc:
                                row.update(status="action_error", error={"type": type(exc).__name__, "message": str(exc),
                                                                       "code": getattr(exc, "code", None)})
                        _line(results, row)
                        manifest["result_count"] += 1
                        manifest["error_result_count"] += row["status"] != "ok"
        manifest["status"] = "completed_with_errors" if manifest["error_result_count"] else "completed"
    finally:
        manifest["performed_calls"] = journal.performed if journal else 0
        manifest["attempted_calls"] = journal.attempted if journal else 0
        manifest["real_model_calls"] = backend == "ollama" and manifest["performed_calls"] > 0
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        _write(output / "manifest.json", manifest)
    return manifest


def export_review(run_directory, output) -> int:
    """Export blind answer codes; keep mapping.json private from reviewers."""
    run_directory, output = Path(run_directory), Path(output)
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["status"] in {"completed", "completed_with_errors"}, "Run must finish before export")
    rows = [json.loads(line) for line in (run_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    random.SystemRandom().shuffle(rows)
    output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    with (output / "review.jsonl").open("w", encoding="utf-8") as review:
        for number, row in enumerate(rows, 1):
            code = f"P{number:03d}"
            mapping[code] = {key: row[key] for key in ("result_id", "method", "history_id", "request_id")}
            _line(review, {"answer_code": code, "history_id": row["history_id"],
                           "request_text": row["request_text"], "answer": row["action"],
                           "source_turns": row["source_turns"], "execution_status": row["status"],
                           "run_kind": row["run_kind"], "scientific_result": False,
                           "result_type": row["result_type"],
                           "human_review": {"factual_accuracy": "", "help_attribution": "",
                                            "action_appropriateness": "", "source_support": "", "comment": ""}})
    _write(output / "mapping.private.json", mapping)
    return len(rows)
