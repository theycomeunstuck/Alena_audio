"""Finite synthetic integration pass through the real local privacy gate.

Default mode only describes the pass. --execute performs up to five answers;
Spiderssense calls use the existing USD 0.40 / 20-call integration ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AuditedOpener:
    """Inspect actual HTTP bytes, never capture Authorization or other headers."""
    def __init__(self, delegate, forbidden):
        self.delegate = delegate
        self.forbidden = tuple(value.casefold() for value in forbidden if value)
        self.requests = []

    def open(self, request, **kwargs):
        body = request.data.decode("utf-8")
        if any(value in body.casefold() for value in self.forbidden):
            raise ValueError("Known identifier survived into the actual outbound body")
        self.requests.append({"sha256": hashlib.sha256(request.data).hexdigest(),
                              "body_bytes": len(request.data), "body": json.loads(body),
                              "known_identifiers_absent": True})
        return self.delegate.open(request, **kwargs)


def run_check(config_path: Path, output: Path, *, execute=False, requests=5) -> dict:
    from local_app.answer_adapter import AnswerAdapter, EducationalProjection
    from local_app.config import Settings
    from local_app.runtime import OllamaRuntime
    from local_app.memory_ported.contracts import action_schema, validate_action

    if not 1 <= requests <= 5:
        raise ValueError("finite integration pass supports 1..5 requests")
    plain = json.loads(config_path.read_text(encoding="utf-8"))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "executed": execute,
              "data": "synthetic educational projection only", "requested_answers": requests,
              "public_api_throughput_measured": False,
              "budget": {"cap_usd": plain.get("max_cost_usd"), "cap_calls": plain.get("max_calls")},
              "answers": [], "outbound_requests": []}
    if not execute:
        report.update(status="prepared_not_executed", passed=None)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report

    local = OllamaRuntime(Settings.from_env())
    wrapper = {"kind": "local_app_external_answers_v1", "enabled": True, "spiderssense": plain}
    with tempfile.TemporaryDirectory(prefix="teachcopilot-external-check-") as temporary:
        wrapper_path = Path(temporary) / "answers.json"
        wrapper_path.write_text(json.dumps(wrapper), encoding="utf-8")
        adapter = AnswerAdapter.from_config(local, wrapper_path)
    identifiers = ("synthetic-learner-731", "Synthetic Student QXZ", "synthetic-source-731", "synthetic-material-731")
    audited = AuditedOpener(adapter.external_client._opener, identifiers)
    adapter.external_client._opener = audited
    before = adapter.external_client.preflight()
    report["budget_before"] = {key: before[key] for key in ("calls_reserved", "charged_or_reserved_usd", "available_budget_usd", "blocked")}
    errors = []
    try:
        for index in range(requests):
            query = ("Synthetic Student QXZ: предложи одно следующее учебное действие по делению "
                     f"с нулём в частном. Номер синтетического примера {index + 1}.")
            projection = EducationalProjection(
                topic="Деление с нулём в частном", current_message=query,
                memory=({"source_id": identifiers[2], "claim": "Условия помощи не указаны.",
                         "quote": "Synthetic Student QXZ записал 408 : 4 = 102. Условия помощи не указаны."},),
                materials=({"source_id": identifiers[3], "title": "Разряды частного",
                            "content": "Сравни запись 408 : 4 и 408 : 3; объясни место нуля."},),
                identity_context_complete=True)
            started = time.perf_counter()
            try:
                answer = adapter.answer([{"role": "user", "content": query}], projection=projection,
                    known_identifiers=identifiers, purpose="next_step", response_schema=action_schema({identifiers[2], identifiers[3]}))
                action = validate_action(answer["message"]["content"], {identifiers[2], identifiers[3]})
                report["answers"].append({"index": index + 1, "elapsed_seconds": time.perf_counter() - started,
                    "routing": answer.get("answer_routing"), "usage": answer.get("provider_usage"),
                    "provider_metadata": answer.get("provider_metadata"), "action": action})
            except Exception as exc:
                # Unknown paid-call outcomes remain in the provider ledger. No
                # regeneration, response repair or local duplicate is attempted.
                errors.append({"index": index + 1, "error_type": type(exc).__name__,
                               "diagnostic_code": getattr(exc, "code", "integration_answer_failed")})
                break
    finally:
        after = adapter.external_client.preflight()
        report["budget_after"] = {key: after[key] for key in ("calls_reserved", "charged_or_reserved_usd", "available_budget_usd", "blocked")}
        report["new_ledger_calls"] = after["calls_reserved"] - before["calls_reserved"]
        report["new_charged_or_reserved_usd"] = str(Decimal(after["charged_or_reserved_usd"]) - Decimal(before["charged_or_reserved_usd"]))
        report["outbound_requests"] = audited.requests
        report["errors"] = errors
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        external = sum(row.get("routing", {}).get("backend") == "spiderssense" for row in report["answers"])
        report["external_answers"] = external
        report["local_fallback_answers"] = len(report["answers"]) - external
        report["human_content_review"] = "not performed"
        report["passed"] = not errors and external > 0 and report["new_ledger_calls"] == len(audited.requests) <= requests
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".local/research_memory/budget_v4_20261017/integration.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/memory-verification/external-integration.json")
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = run_check(args.config, args.output, execute=args.execute, requests=args.requests)
    except Exception as exc:
        print(f"External integration check could not start ({type(exc).__name__}).", file=sys.stderr)
        return 2
    print(json.dumps({key: report.get(key) for key in ("status", "executed", "passed", "external_answers", "local_fallback_answers", "new_ledger_calls", "new_charged_or_reserved_usd")}, ensure_ascii=False))
    return 0 if report.get("passed") is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
