"""Finite local dialogue batch; a dry run is the default, with no service calls.

Only case.public reaches RAG. Real materials are read through a PostgreSQL
read-only connection; normal generation admission is the sole database writer.
Synthetic memory is passed in process, never saved to learner tables.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_conversation_boundaries_v2 import error_record, utc_now, write_json

FIXTURE = ROOT / "tests/fixtures/short_division_dialogue_v1.json"
EXPECTED_CASES = 12
MAX_CHAT_CALLS = 32
MAX_SECONDS = 1200
MAX_REQUEST_SECONDS = 120
EXPECTED_MODEL_IDENTITY = "gemma4:12b@6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b"
PUBLIC_FIELDS = frozenset({"messages", "subject_id", "topic_id", "memory_context", "preference_context"})


class BatchLimitError(RuntimeError):
    """A fixed elapsed-time or generation limit was reached before a request."""


class SourceIntegrityError(RuntimeError):
    """Code, fixture or material corpus changed within the fixed batch."""


class BatchBudget:
    def __init__(self, *, now=time.monotonic, max_chat_calls=MAX_CHAT_CALLS, max_seconds=MAX_SECONDS):
        self.now = now
        self.started = now()
        self.deadline = self.started + max_seconds
        self.max_chat_calls = max_chat_calls
        self.chat_calls = 0
        self.embedding_calls = 0
        self.http_calls = 0

    def remaining(self):
        left = self.deadline - self.now()
        if left <= 0:
            raise BatchLimitError("short_division_elapsed_budget_exhausted")
        return left

    def admit(self, path):
        self.remaining()
        if path == "/api/chat":
            if self.chat_calls >= self.max_chat_calls:
                raise BatchLimitError("short_division_chat_budget_exhausted")
            self.chat_calls += 1
        elif "embed" in path:
            self.embedding_calls += 1
        self.http_calls += 1


def public_arguments(case):
    """An explicit allowlist; no case ID, rubric or expected answer reaches RAG."""
    value = case.get("public")
    if not isinstance(value, dict) or set(value) - PUBLIC_FIELDS:
        raise ValueError("public input contains an unsupported field")
    if not {"messages", "subject_id", "topic_id"} <= set(value):
        raise ValueError("public input requires messages, subject_id and topic_id")
    messages = value["messages"]
    if (not isinstance(messages, list) or not messages or len(messages) > 16
            or any(not isinstance(row, dict) or set(row) != {"role", "content"}
                   or row["role"] not in {"user", "assistant"}
                   or not isinstance(row["content"], str) or not row["content"].strip()
                   for row in messages)
            or messages[-1]["role"] != "user"):
        raise ValueError("public messages must end with a literal learner message")
    if value["subject_id"] != "math" or value["topic_id"] != "math.g4.numbers.division_by_1_2_digit":
        raise ValueError("batch requires the fixed division curriculum topic")
    for key in ("memory_context", "preference_context"):
        if key in value and not isinstance(value[key], dict):
            raise ValueError(f"{key} must be an object")
    return {key: deepcopy(value[key]) for key in PUBLIC_FIELDS if key in value}


def load_cases(path=FIXTURE):
    fixture = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = fixture.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASES:
        raise ValueError("fixture must contain exactly twelve cases")
    ids = [case.get("case_id") for case in cases]
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not re.fullmatch(r"D[0-9]{2}", value) for value in ids):
        raise ValueError("case IDs must be distinct safe Dxx names")
    for case in cases:
        public_arguments(case)
        gold = case.get("private_gold")
        if not isinstance(gold, dict) or any(not isinstance(gold.get(key), list) or not gold[key] for key in ("facts", "allowed", "forbidden")):
            raise ValueError("each case requires facts, allowed and forbidden private expectations")
    return fixture, cases


def source_hashes():
    files = {FIXTURE, Path(__file__).resolve(), ROOT / "scripts/run_conversation_boundaries_v2.py"}
    files.update((ROOT / "local_app").rglob("*.py"))
    files.update((ROOT / "local_app/prompts").glob("*.txt"))
    files.update((ROOT / "research_memory").glob("*.py"))
    files.update((ROOT / "research_memory/prompts").glob("*.txt"))
    files.update((ROOT / "db/migrations").glob("*.sql"))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def plan():
    fixture, cases = load_cases()
    return {"protocol": fixture["version"], "dry_run": True, "case_ids": [case["case_id"] for case in cases],
            "expected_cases": len(cases), "max_chat_calls": MAX_CHAT_CALLS, "max_elapsed_seconds": MAX_SECONDS,
            "max_request_seconds": MAX_REQUEST_SECONDS, "automatic_retries": False, "local_only": True,
            "learner_writes": False, "private_gold_enters_generation": False,
            "note": "Future-session cases use synthetic in-process approved-memory projections; persistence is not measured here.",
            "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}


def generate_case(service, case):
    return service.answer(**public_arguments(case))


def execute(output_dir):
    import requests
    from psycopg2.extensions import make_dsn
    from local_app.answer_adapter import AnswerAdapter
    from local_app.config import Settings
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from scripts.verify_memory import require_local_dsn

    fixture, cases = load_cases()
    frozen_hashes, frozen_commit = source_hashes(), git_commit()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "http").mkdir()
    budget = BatchBudget()
    current = {"case_id": None}
    report = {**plan(), "dry_run": False, "status": "initializing", "created_at": utc_now(),
              "data_origin": "synthetic_unreviewed", "human_reviewed": False,
              "semantic_assessment": "not_assessed", "technical_ok_does_not_mean_correct": True,
              "paid_provider_calls": 0, "results": []}
    runtime = store = None
    initial_corpus = None
    # Persist the fixed data/code boundary even when service preflight fails.
    write_json(out / "STARTED.json", {"created_at": utc_now(), "fixture": fixture,
        "sha256": frozen_hashes, "git_commit": frozen_commit, "limits": plan(),
        "external_provider_disabled": True, "automatic_retries": False})

    def save():
        report.update(chat_calls=budget.chat_calls, embedding_calls=budget.embedding_calls,
                      http_calls=budget.http_calls, elapsed_seconds=budget.now() - budget.started)
        write_json(out / "results.json", report)
        write_json(out / "PROCESSING.json", {key: report[key] for key in (
            "status", "expected_cases", "chat_calls", "embedding_calls", "http_calls", "elapsed_seconds")}
            | {"recorded_cases": len(report["results"]), "updated_at": utc_now(),
               "semantic_assessment": "not_assessed", "paid_provider_calls": 0})
        lines = ["Короткий диалог по делению – фактические ответы", "Синтетические случаи; содержательная оценка не выполнена.", ""]
        for row in report["results"]:
            case = next(case for case in cases if case["case_id"] == row["case_id"])
            lines.extend([row["case_id"] + " – " + row["status"], "Последняя реплика: " + case["public"]["messages"][-1]["content"],
                          row.get("answer") or json.dumps(row.get("error") or {}, ensure_ascii=False), ""])
        (out / "responses.txt").write_text("\n".join(lines), encoding="utf-8")

    def corpus_hash():
        rows = sorted(store.material_rows(), key=lambda row: str(row.get("id", "")))
        return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def assert_unchanged():
        if source_hashes() != frozen_hashes or git_commit() != frozen_commit:
            raise SourceIntegrityError("source_fixture_or_git_head_changed_during_batch")

    class RecordingSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.trust_env = False
            self.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))

        def send(self, request, **kwargs):
            parsed = urlsplit(request.url)
            if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
                raise ValueError("batch transport allows credential-free HTTP loopback only")
            budget.admit(parsed.path)
            kwargs["timeout"] = min(float(kwargs.get("timeout") or MAX_REQUEST_SECONDS), MAX_REQUEST_SECONDS, budget.remaining())
            kwargs["allow_redirects"] = False
            body = request.body or b""
            if isinstance(body, str):
                body = body.encode("utf-8")
            if not isinstance(body, bytes):
                raise ValueError("streaming request bodies are not supported")
            stem = f"{budget.http_calls:03d}"
            record = {"case_id": current["case_id"], "started_at": utc_now(), "method": request.method,
                      "url": request.url, "request_body_base64": base64.b64encode(body).decode("ascii"),
                      "request_body_utf8": body.decode("utf-8"), "automatic_retries": False}
            write_json(out / "http" / (stem + "-request.json"), record)
            started = time.monotonic()
            try:
                response = super().send(request, **kwargs)
                # Persist exact bytes before runtime parsing or typography changes.
                record.update(status_code=response.status_code,
                              response_body_base64=base64.b64encode(response.content).decode("ascii"),
                              response_body_utf8=response.content.decode("utf-8", errors="replace"))
                return response
            except Exception as exc:
                record["error"] = error_record(exc)
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
                write_json(out / "http" / (stem + "-result.json"), record)

    class FiniteRuntime(OllamaRuntime):
        def chat(self, *args, **kwargs):
            left = budget.remaining()
            if self.generation_scheduler is not None:
                self.generation_scheduler.wait_timeout = min(MAX_REQUEST_SECONDS, left)
            return super().chat(*args, **kwargs)

    class PublicMaterials:
        def embedding_index_status(self, identity):
            return store.embedding_index_status(identity)

        def hybrid_search(self, **kwargs):
            return store.hybrid_search(**kwargs)

    try:
        settings = replace(Settings.from_env(), request_timeout=MAX_REQUEST_SECONDS,
                           learning_operation_selection_enabled=False)
        if settings.chat_model != "gemma4:12b" or settings.context_tokens != 8192:
            raise ValueError("batch requires gemma4:12b and an 8192-token context")
        params = require_local_dsn(settings.database_url)
        options = (params.get("options", "") + " -c default_transaction_read_only=on -c statement_timeout=120000 -c lock_timeout=5000").strip()
        store = Store(make_dsn(settings.database_url, options=options, connect_timeout=5))
        runtime = FiniteRuntime(settings, session=RecordingSession())
        service = RagService(PublicMaterials(), runtime, settings, answer_adapter=AnswerAdapter(runtime))
        identity = runtime.model_identity(settings.chat_model)
        if identity != EXPECTED_MODEL_IDENTITY:
            raise ValueError("generation model digest differs from the fixed model")
        initial_corpus = corpus_hash()
        write_json(out / "preregistered.json", {"created_at": utc_now(), "fixture": fixture,
            "sha256": frozen_hashes, "git_commit": frozen_commit, "public_materials_sha256": initial_corpus,
            "model_identity": identity, "model_manifest": runtime.manifest(), "limits": plan(),
            "settings_override": {"learning_operation_selection_enabled": False},
            "database_access": "public materials read-only; normal generation-admission scheduler writes only",
            "memory_origin": "synthetic in-process projections; no learner sessions or sources are written"})
        report["index_status"] = service.status()
        if not report["index_status"].get("ready"):
            raise RuntimeError("real material embedding index is not ready")
        assert_unchanged()
        report["status"] = "running"
        save()
        stop_error = None
        for case in cases:
            current["case_id"] = case["case_id"]
            row = {"case_id": case["case_id"], "status": "not_attempted", "semantic_assessment": "not_assessed"}
            if stop_error:
                row["error"] = stop_error
            else:
                started = time.monotonic()
                before_calls = budget.chat_calls
                try:
                    budget.remaining()
                    assert_unchanged()
                    row["status"] = "attempted"
                    response = generate_case(service, case)
                    row.update(status="technical_ok", response=response,
                               answer=response["choices"][0]["message"]["content"])
                except Exception as exc:
                    row.update(status="technical_failure", error=error_record(exc))
                    if isinstance(exc, (BatchLimitError, SourceIntegrityError)):
                        stop_error = error_record(exc)
                finally:
                    row.update(elapsed_seconds=time.monotonic() - started, chat_calls=budget.chat_calls - before_calls)
            report["results"].append(row)
            write_json(out / (case["case_id"] + ".json"), row)
            save()
            print(case["case_id"] + ": " + row["status"], flush=True)
        report["status"] = "completed" if all(row["status"] == "technical_ok" for row in report["results"]) else "completed_with_failures"
    except Exception as exc:
        report.update(status="stopped_without_retry", error=error_record(exc))
    finally:
        recorded = {row["case_id"] for row in report["results"]}
        for case in cases:
            if case["case_id"] not in recorded:
                row = {"case_id": case["case_id"], "status": "not_attempted", "chat_calls": 0,
                       "semantic_assessment": "not_assessed",
                       "error": report.get("error") or {"type": "BatchStopped", "message": "batch stopped before case"}}
                report["results"].append(row)
                write_json(out / (case["case_id"] + ".json"), row)
        try:
            assert_unchanged()
            if initial_corpus is not None and corpus_hash() != initial_corpus:
                raise SourceIntegrityError("public_material_corpus_changed_during_batch")
            report["source_integrity"] = {"unchanged": True}
        except Exception as exc:
            report.update(status="source_integrity_failure", source_integrity={"unchanged": False, "error": error_record(exc)})
        if runtime is not None:
            runtime.session.close()
        report["finished_at"] = utc_now()
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run one finite local batch; default is a service-free dry run")
    parser.add_argument("--out", type=Path, help="New output directory; existing paths are rejected")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return 0
    if args.out is None:
        parser.error("--execute requires a new --out directory")
    result = execute(args.out)
    print(json.dumps({"status": result["status"], "chat_calls": result["chat_calls"]}, ensure_ascii=False))
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
