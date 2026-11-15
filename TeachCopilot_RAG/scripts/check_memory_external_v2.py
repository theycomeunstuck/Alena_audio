"""One finite synthetic pass: six real external answers and six local fallbacks.

Dry-run is the default. Execution uses the real local privacy checker, the
existing shared generation run budget, the main database admission scheduler,
and the newly authorized USD 4 / six-call provider ledger. No retries occur.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IDENTITIES = ("Синтетический Ученик АБВ", "synthetic-learner-731", "synthetic-source-731", "synthetic-material-731")
RESPONSE_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"},
    "citations": {"type": "array", "items": {"type": "string", "enum": list(IDENTITIES[2:])}}},
    "required": ["answer", "citations"], "additionalProperties": False}


def scenarios() -> list[dict]:
    """Frozen before execution; expected routes are never supplied to the gate."""
    base = {"topic": "Пошаговое деление", "current_message": "Синтетический Ученик АБВ: как проверить 408 : 4?",
        "memory": [{"source_id": IDENTITIES[2], "role": "student", "claim": "Одна запись примера; условия помощи неизвестны.",
                    "quote": "Синтетический Ученик АБВ: 408 : 4 = 102. Условия помощи не указаны."}],
        "materials": [{"source_id": IDENTITIES[3], "title": "Проверка деления", "content": "Проверь частное умножением."}],
        "identity_context_complete": True}
    safe = [
        ("E01", "quotient_zero", {}, "Синтетический Ученик АБВ: как сохранить ноль в частном у 408 : 4?"),
        ("E02", "short_uncertain_answer", {"dialogue": [{"role": "assistant", "content": "Как проверить 408 : 4 умножением?"}]}, "хз"),
        ("E03", "generic_interest_and_avoidance", {"interests": ["конструкторы"], "avoided_interests": ["гонки"]}, "Вернёмся к заданию 408 : 4, без гонок."),
        ("E04", "unknown_help_conditions", {}, "Получилось 102. Не помню, была ли подсказка. Что проверить дальше?"),
        ("E05", "one_success_not_mastery", {"topic": "Дроби", "memory": [{"source_id": IDENTITIES[2], "role": "student",
            "claim": "Один ответ на вопрос о дробях.", "quote": "Синтетический Ученик АБВ: 1/2 больше 1/4."}]}, "Один раз сравнил 1/2 и 1/4. Дай похожую проверку."),
        ("E06", "quoted_current_context", {"topic": "Вывод Python", "memory": [{"source_id": IDENTITIES[2], "role": "student",
            "claim": "Один прогноз вывода Python.", "quote": "Синтетический Ученик АБВ: print(2 + 2) выведет 4."}]}, "4? Объясни проверку вывода print(2 + 2)."),
    ]
    unsafe = [
        ("L01", "email_in_current_message", {}, "Напиши мне person731@example.test, и вернёмся к 408 : 4."),
        ("L02", "phone_in_quote", {"memory": [{"source_id": IDENTITIES[2], "role": "student", "claim": "Контакт в цитате.",
             "quote": "Для связи +7 (999) 123-45-67; ответ 102."}]}, "ага, вернёмся к делению"),
        ("L03", "url_in_material", {"materials": [{"source_id": IDENTITIES[3], "title": "Небезопасный синтетический фрагмент",
             "content": "Личная страница https://private731.example.test/profile. Проверь 408 : 4."}]}, "Как проверить ответ?"),
        ("L04", "credential_in_dialogue", {"dialogue": [{"role": "user", "content": "api_key = synthetic-secret-731"},
             {"role": "assistant", "content": "Не присылай ключи. Вернёмся к математике."}]}, "вернёмся к 408 : 4"),
        ("L05", "private_autobiographical_text", {}, "У меня болезнь, про которую знают только родители; из-за лечения пропускаю занятия в школе 731. Давай теперь 408 : 4."),
        ("L06", "ambiguous_identifying_story", {}, "В прошлый раз это было у нашего подъезда рядом со школой 731. Не знаю, можно ли такое пересказывать. Вернись к примеру."),
    ]
    result = []
    # Check all local-route cases before spending on allowed requests.
    for route, entries in (("local", unsafe), ("spiderssense", safe)):
        for case_id, feature, changes, message in entries:
            projection = {**deepcopy(base), **deepcopy(changes), "current_message": message}
            result.append({"case_id": case_id, "feature": feature, "expected_backend": route,
                "projection": projection, "expected_current_message": message.replace(IDENTITIES[0], "[REDACTED]"),
                "expected_quote": projection["memory"][0]["quote"].replace(IDENTITIES[0], "[REDACTED]")})
    return result


class UnexpectedExternalRoute(RuntimeError):
    pass


class _RecordedResponse:
    def __init__(self, response, record, secrets):
        self.response, self.record, self.secrets = response, record, secrets

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def read(self, *args):
        raw = self.response.read(*args)
        _record_raw_response(self.record, raw, self.secrets)
        return raw


def _record_raw_response(record: dict, raw: bytes, secrets: tuple[str, ...]) -> None:
    record["response_sha256"] = hashlib.sha256(raw).hexdigest()
    record["response_bytes"] = len(raw)
    text = raw.decode("utf-8", errors="replace")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED_CREDENTIAL]")
    record["raw_response_text"] = text


class AuditedOpener:
    """Assert exact HTTP content before transport; never capture headers."""
    def __init__(self, delegate, *, secret=""):
        self.delegate = delegate
        self.secret = secret
        self.case = None
        self.requests = []

    def open(self, request, **kwargs):
        if self.case is None or self.case["expected_backend"] != "spiderssense":
            raise AssertionError("No external body is allowed for a local-route scenario")
        body = request.data.decode("utf-8")
        for identifier in (*IDENTITIES, "АБВ"):
            if identifier.casefold() in body.casefold():
                raise AssertionError("Known synthetic identifier survived into actual HTTP bytes")
        parsed = json.loads(body)
        payload = json.loads(parsed["messages"][1]["content"])
        if payload["current_message"] != self.case["expected_current_message"]:
            raise AssertionError("Actual HTTP body omitted or changed the current message")
        if payload["memory"][0]["quote"] != self.case["expected_quote"]:
            raise AssertionError("Actual HTTP body omitted or changed the evidence quotation")
        if payload["memory"][0]["source_id"] != "S1" or payload["materials"][0]["source_id"] != "S2":
            raise AssertionError("Actual HTTP body failed source aliasing")
        record = {"case_id": self.case["case_id"], "sha256": hashlib.sha256(request.data).hexdigest(),
            "body_bytes": len(request.data), "body": parsed, "current_message_present": True,
            "quote_present": True, "identifiers_absent": True, "source_aliases_verified": True}
        self.requests.append(record)
        try:
            response = self.delegate.open(request, **kwargs)
        except HTTPError as exc:
            from local_app.memory_ported.clients import MAX_RESPONSE_BYTES
            record.update(transport_error_type=type(exc).__name__, http_status=exc.code)
            try:
                raw = exc.read(MAX_RESPONSE_BYTES + 1)
                _record_raw_response(record, raw, (self.secret,))
                record["error_body_truncated"] = len(raw) > MAX_RESPONSE_BYTES
            except Exception as read_exc:
                record["error_body_read_error_type"] = type(read_exc).__name__
            # The original client retains the unknown reservation and exposes a
            # sanitized error. Preserve that behavior without response repair.
            raise
        except Exception as exc:
            record["transport_error_type"] = type(exc).__name__
            raise
        return _RecordedResponse(response, record, (self.secret,))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _state(client):
    value = client.preflight()
    return {key: value[key] for key in ("calls_reserved", "charged_or_reserved_usd", "available_budget_usd", "blocked")}


def run_check(config_path: Path, output: Path, *, execute=False, budget_ledger: Path | None = None,
              scheduler_dsn: str | None = None) -> dict:
    cases = scenarios()
    if output.exists():
        raise FileExistsError("Integration package already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=False)
    config_bytes = config_path.read_bytes() if config_path.exists() else None
    report = {"kind": "external_answers_synthetic_v2", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "executed": execute, "data": "synthetic only", "human_review": "not performed",
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest() if config_bytes else None,
        "expected_external_answers": 6, "expected_local_answers": 6, "automatic_paid_retries": False,
        "public_api_throughput_measured": False, "cases": [], "outbound_requests": []}
    code_files = ("scripts/check_memory_external_v2.py", "local_app/runtime.py", "local_app/answer_adapter.py",
        "local_app/budget_authorization.py", "local_app/config.py", "local_app/generation_run_budget.py",
        "research_memory/spiderssense.py")
    report["code_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files}
    _write(output / "preregistered.json", {"cases": cases, "response_schema": RESPONSE_SCHEMA,
        "settings": report, "case_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()})
    if not execute:
        report.update(status="prepared_not_executed", passed=None)
        _write(output / "report.json", report)
        return report
    if not budget_ledger or not budget_ledger.is_file():
        raise ValueError("Execution requires the existing shared product generation budget ledger")
    os.environ["LOCAL_GENERATION_RUN_BUDGET_PATH"] = str(budget_ledger.resolve())
    from local_app.answer_adapter import AnswerAdapter, EducationalProjection
    from local_app.config import Settings
    from local_app.runtime import OllamaRuntime
    settings = Settings.from_env()
    main_dsn = scheduler_dsn or os.environ.get("LOCAL_GENERATION_SCHEDULER_DSN") or settings.database_url
    os.environ["LOCAL_GENERATION_SCHEDULER_DSN"] = main_dsn
    # Settings may own scheduler/run-budget fields; load again after explicit env.
    settings = Settings.from_env()
    local = OllamaRuntime(settings)
    if settings.chat_model != "gemma4:12b":
        raise ValueError("This integration package requires the real local gemma4:12b")
    if getattr(local, "generation_run_budget", None) is None:
        raise ValueError("Runtime did not bind the shared durable generation budget")
    local.check_models()
    adapter = AnswerAdapter.from_config(local, config_path)
    if adapter.external_client is None:
        raise ValueError("Explicit v2 external answer mode must be enabled for this finite run")
    wrapper = json.loads(config_bytes)
    if wrapper.get("kind") != "local_app_external_answers_v2":
        raise ValueError("This integration package requires the authorized v2 configuration")
    client = adapter.external_client
    before = _state(client)
    if before["calls_reserved"] != 0:
        raise ValueError("The finite six-call integration ledger was already used; automatic repetitions are prohibited")
    audited = AuditedOpener(client._opener, secret=client._key())
    client._opener = audited
    original_complete = client.complete

    def guarded_complete(*args, **kwargs):
        if audited.case["expected_backend"] != "spiderssense":
            # This is a failed privacy route, recorded as an error. Stop before
            # reservation/HTTP; never coerce a failed gate into a passing result.
            raise UnexpectedExternalRoute("Unsafe or ambiguous case escaped the local gate")
        return original_complete(*args, **kwargs)

    client.complete = guarded_complete
    report["budget_before"] = before
    report["local_generation_budget_before"] = local.generation_run_budget.snapshot()
    report["generation_budget_path"] = str(budget_ledger.resolve())
    report["local_model"] = settings.chat_model
    report["local_runtime_manifest"] = local.manifest()
    _write(output / "configuration.json", wrapper)
    _write(output / "report.json", report)
    errors = []
    try:
        for case in cases:
            audited.case = case
            row = {"case_id": case["case_id"], "feature": case["feature"], "expected_backend": case["expected_backend"],
                "request": case["projection"], "budget_before": _state(client)}
            started = time.perf_counter()
            try:
                projection = EducationalProjection(**case["projection"])
                messages = [*projection.dialogue, {"role": "user", "content": projection.current_message}]
                row["answer"] = adapter.answer(messages, projection=projection, known_identifiers=IDENTITIES,
                    purpose="chat", response_schema=RESPONSE_SCHEMA)
                row["budget_after"] = _state(client)
                delta = row["budget_after"]["calls_reserved"] - row["budget_before"]["calls_reserved"]
                row["new_provider_calls"] = delta
                routing = row["answer"].get("answer_routing", {})
                row["passed"] = (routing.get("backend") == case["expected_backend"]
                    and delta == (1 if case["expected_backend"] == "spiderssense" else 0))
                if case["expected_backend"] == "spiderssense":
                    result = json.loads(row["answer"]["message"]["content"])
                    row["response_contract_valid"] = (set(result) == {"answer", "citations"}
                        and isinstance(result["answer"], str) and bool(result["answer"].strip())
                        and isinstance(result["citations"], list)
                        and all(isinstance(value, str) and value in IDENTITIES[2:] for value in result["citations"]))
                    row["passed"] = row["passed"] and row["response_contract_valid"]
                if not row["passed"]:
                    errors.append({"case_id": case["case_id"], "error_type": "ScenarioAssertionFailed"})
            except Exception as exc:
                row.update(passed=False, error_type=type(exc).__name__,
                    diagnostic_code=getattr(exc, "code", "integration_scenario_failed"), budget_after=_state(client))
                errors.append({"case_id": case["case_id"], "error_type": type(exc).__name__})
            finally:
                row["elapsed_seconds"] = time.perf_counter() - started
                report["cases"].append(row)
                report["outbound_requests"] = audited.requests
                _write(output / f"{case['case_id']}.json", row)
                _write(output / "http.json", audited.requests)
                _write(output / "report.json", report)
            # No automatic retry, response repair or replacement after an error.
            # The remaining finite scenarios are independent. Unknown billing
            # remains reserved by SpiderssenseClient and counts toward both caps.
    finally:
        after = _state(client)
        report.update(budget_after=after, new_provider_calls=after["calls_reserved"] - before["calls_reserved"],
            new_charged_or_reserved_usd=str(Decimal(after["charged_or_reserved_usd"]) - Decimal(before["charged_or_reserved_usd"])),
            errors=errors, completed_at_utc=datetime.now(timezone.utc).isoformat())
        report["local_generation_budget_after"] = local.generation_run_budget.snapshot()
        report["external_answers"] = sum(row.get("answer", {}).get("answer_routing", {}).get("backend") == "spiderssense" for row in report["cases"])
        report["local_fallback_answers"] = sum(row.get("answer", {}).get("answer_routing", {}).get("backend") == "local" for row in report["cases"])
        report["passed"] = (not errors and len(report["cases"]) == 12 and report["external_answers"] == 6
            and report["local_fallback_answers"] == 6 and report["new_provider_calls"] == len(audited.requests) == 6)
        report["status"] = "completed" if report["passed"] else "completed_with_failures"
        _write(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / ".local/research_memory/budget_v5_20261022/external_answers_v2.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = run_check(args.config, args.output, execute=args.execute, budget_ledger=args.budget_ledger)
    except Exception as exc:
        if args.output.is_dir():
            _write(args.output / "startup_failure.json", {"status": "start_failed", "error_type": type(exc).__name__,
                "paid_retry_attempted": False, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"status": "start_failed", "error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({key: report.get(key) for key in ("status", "passed", "external_answers", "local_fallback_answers", "new_provider_calls", "new_charged_or_reserved_usd")}))
    return 1 if report.get("passed") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
