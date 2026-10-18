"""Finite local-only conversational-interest pilot with immutable raw results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from local_app.config import Settings
    from local_app.conversation_memory import interest_messages, interest_schema, validate_interest_candidates
    from local_app.runtime import OllamaRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "tests/fixtures/conversation_interests_v1.json"
    fixture = json.loads(source.read_text(encoding="utf-8"))
    settings = Settings.from_env()
    runtime = OllamaRuntime(settings)
    # Expected results stay in this local receipt and never enter model prompts.
    cases = []
    for case in fixture["cases"]:
        student_sources = [row for row in case["sources"] if row["author_role"] == "student"]
        cases.append({**case, "messages": interest_messages(student_sources),
                      "schema": interest_schema(student_sources), "student_sources": student_sources})
    tracked = [source, ROOT / "local_app/conversation_memory.py",
               ROOT / "local_app/prompts/conversation_interests.txt", ROOT / "local_app/runtime.py"]
    prereg = {"kind": "application_pilot_not_method_comparison", "created_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
              "model": settings.chat_model, "ollama_url": settings.ollama_url,
              "context_tokens": settings.context_tokens, "max_output_tokens": 1536,
              "temperature": 0, "seed": 42, "think": False,
              "model_identity": runtime.model_identity(settings.chat_model),
              "attempts_per_case": 1, "paid_provider_calls": 0,
              "semantic_review": "not_human_reviewed", "model_manifest": runtime.manifest()}
    (out / "preregistered.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for case in cases:
        started = time.perf_counter()
        row = {"case_id": case["case_id"], "status": "failed"}
        try:
            response = runtime.chat(case["messages"], response_format=case["schema"],
                                    max_output_tokens=1536, truncate_history=False)
            row["raw_response"] = response
            row["candidates"] = validate_interest_candidates(response["message"]["content"], case["student_sources"])
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if getattr(exc, "raw_response", None) is not None:
                row["raw_response"] = exc.raw_response
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
        (out / (case["case_id"] + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(case["case_id"] + ": " + row["status"], flush=True)
    report = {"status": "completed", "cases": len(rows), "technical_ok": sum(r["status"] == "ok" for r in rows),
              "semantic_assessment": "pending_separate_review", "paid_provider_calls": 0,
              "completed_at": datetime.now(timezone.utc).isoformat(), "results": rows}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["technical_ok"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
