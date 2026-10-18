"""Blind, source-grounded machine judging of synthetic development answers.

Citation checks verify membership and literal text, not educational truth. No
function here loads reference answers while generating or judging a response.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from .clients import Completion, CompletionValidationError, OllamaClient
from .contracts import normalized, parse_object
from .dataset import identifier, nonempty, require
from .runner import CallJournal


ROOT = Path(__file__).parent
PROMPT = ROOT / "prompts" / "judge.txt"
CONTEXT_PROTOCOL = "context-v2"
HELDOUT_PROTOCOL = "context-v3"
CONTEXT_PROTOCOLS = {CONTEXT_PROTOCOL, HELDOUT_PROTOCOL}
UTILITIES = ("useful", "limited", "uncertain")
CRITERIA = ("factual_accuracy", "help_attribution", "action_appropriateness",
            "source_support", "unknowns_handling")
VERDICTS = ("acceptable", "critical_error", "uncertain")
STR = {"type": "string"}
FINDING_FIELDS = {"criterion": {"type": "string", "enum": list(CRITERIA)},
                  "claim": STR, "evidence_turn_ids": {"type": "array", "items": STR,
                                                     "uniqueItems": True},
                  "quote": STR, "explanation": STR}
JUDGEMENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "scores", "findings", "rationale"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "scores": {"type": "object", "additionalProperties": False,
                   "required": list(CRITERIA), "properties": {
                       key: {"type": ["integer", "null"], "enum": [0, 1, 2, None]}
                       for key in CRITERIA}},
        "findings": {"type": "array", "maxItems": 30, "items": {
            "type": "object", "additionalProperties": False,
            "required": list(FINDING_FIELDS), "properties": FINDING_FIELDS}},
        "rationale": STR,
    },
}
CONTEXT_JUDGEMENT_SCHEMA = {
    **JUDGEMENT_SCHEMA,
    "required": ["verdict", "utility", "scores", "findings", "rationale"],
    "properties": {**JUDGEMENT_SCHEMA["properties"],
                   "utility": {"type": "string", "enum": list(UTILITIES)}},
}
CASE_FIELDS = {"answer_code", "data_origin", "execution_status", "contract_only", "payload"}


def prompt_path(protocol="legacy") -> Path:
    """Resolve an explicit judge protocol; unknown versions never fall back."""
    require(protocol in {"legacy", *CONTEXT_PROTOCOLS}, "Unsupported judge protocol")
    if protocol == "legacy":
        return PROMPT
    return ROOT / "prompts" / ("judge_context_v3.txt" if protocol == HELDOUT_PROTOCOL else "judge_context.txt")


def _profile_payload(profile: dict) -> dict:
    require(isinstance(profile, dict), "Missing cutoff profile")
    return {key: profile[key] for key in ("grade", "topic_id", "status") if key in profile}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _line(stream, value: dict) -> None:
    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    stream.flush()


def _rows(path: Path, *, allow_empty=False) -> list[dict]:
    require("gold" not in path.name.lower(), "Judge inputs must not be reference-answer files")
    rows = []
    for number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from None
        require(isinstance(row, dict), f"{path.name}:{number}: expected object")
        rows.append(row)
    require(allow_empty or bool(rows), f"{path.name}: empty input")
    return rows


def _history_payload(history: dict) -> dict:
    """Allowlist source fields; do not copy run/method metadata into the judge."""
    require(isinstance(history.get("profile_as_of_cutoff"), dict), "Missing cutoff profile")
    lessons = history.get("lessons")
    require(isinstance(lessons, list) and bool(lessons), "History needs lessons")
    selected = []
    seen = set()
    for lesson in lessons:
        require(isinstance(lesson, dict) and identifier(lesson.get("lesson_id")), "Invalid lesson")
        require(nonempty(lesson.get("observed_on")), "Missing observation date")
        turns = lesson.get("turns")
        require(isinstance(turns, list) and bool(turns), "Lesson needs turns")
        selected_turns = []
        for turn in turns:
            require(isinstance(turn, dict) and identifier(turn.get("turn_id")), "Invalid turn")
            require(turn["turn_id"] not in seen, "Duplicate history turn ID")
            seen.add(turn["turn_id"])
            require(turn.get("role") in {"teacher", "student", "assistant"}
                    and nonempty(turn.get("text")), "Invalid source turn")
            selected_turns.append({key: turn[key] for key in ("turn_id", "role", "text")})
        selected.append({"lesson_id": lesson["lesson_id"], "observed_on": lesson["observed_on"],
                         "turns": selected_turns})
    # Profiles are source data; run metadata must be supplied outside this object.
    profile = _profile_payload(history["profile_as_of_cutoff"])
    return {"profile_as_of_cutoff": profile, "lessons": selected}


def _validate_case(case: dict, *, protocol="legacy") -> dict[str, dict]:
    prompt_path(protocol)
    require(set(case) == CASE_FIELDS and identifier(case.get("answer_code")), "Invalid blind case fields")
    require(nonempty(case.get("data_origin")) and case["data_origin"].startswith("synthetic"),
            "Judging accepts synthetic data only; traces contain full histories")
    require(type(case.get("contract_only")) is bool and nonempty(case.get("execution_status")),
            "Invalid case provenance")
    payload = case.get("payload")
    payload_fields = {"history", "question", "answer"}
    if protocol in CONTEXT_PROTOCOLS:
        payload_fields.add("generator_context")
    require(isinstance(payload, dict) and set(payload) == payload_fields,
            "Invalid judge payload fields for the selected protocol")
    require(nonempty(payload["question"]) and (payload["answer"] is None
            or isinstance(payload["answer"], dict)), "Invalid question/answer")
    history = payload["history"]
    require(isinstance(history, dict) and set(history) == {"profile_as_of_cutoff", "lessons"},
            "Invalid judge history envelope")
    require(_history_payload(history) == history, "Unexpected metadata in judge history")
    if protocol in CONTEXT_PROTOCOLS:
        context = payload["generator_context"]
        require(isinstance(context, dict) and set(context) == {"profile_as_of_cutoff", "selected_memory"},
                "Invalid generator_context fields")
        profile, memory = context["profile_as_of_cutoff"], context["selected_memory"]
        require(profile is None or isinstance(profile, dict)
                and profile == _profile_payload(profile)
                and profile == history["profile_as_of_cutoff"],
                "Actual generator profile differs from the full-history profile")
        require(memory is None or isinstance(memory, str), "Actual selected memory must be text or null")
        require(case["execution_status"] != "ok" or isinstance(profile, dict) and isinstance(memory, str),
                "Successful answers need actual generator profile and selected memory")
    return {turn["turn_id"]: turn for lesson in history["lessons"] for turn in lesson["turns"]}


def prepare_judging(run_directory, histories_path, output, *, seed=42, protocol="legacy") -> dict:
    """Export all completed-run results blindly; failures remain in coverage."""
    prompt_path(protocol)
    run_directory, histories_path, output = map(Path, (run_directory, histories_path, output))
    manifest_path = run_directory / "manifest.json"
    is_batch = not manifest_path.exists()
    if is_batch:
        manifest_path = run_directory / "batch_manifest.json"
    results_path = run_directory / "results.jsonl"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if is_batch:
        require(isinstance(source_manifest.get("job_counts"), dict)
                and type(source_manifest.get("expected_results")) is int,
                "Batch report needs job counts and expected result count")
    else:
        require(source_manifest.get("status") in {"completed", "completed_with_errors"},
                "Run must finish before judging preparation")
    histories = _rows(histories_path)
    require(all(nonempty(h.get("data_origin")) and h["data_origin"].startswith("synthetic")
                for h in histories), "Judging accepts synthetic histories only")
    by_id = {h.get("history_id"): h for h in histories}
    require(len(by_id) == len(histories) and all(identifier(x) for x in by_id), "Invalid history IDs")
    recorded_hash = source_manifest.get("input_sha256", {}).get("histories")
    definition_path = run_directory / "batch.json"
    if is_batch:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        recorded_hash = definition.get("snapshot_sha256", {}).get("histories.jsonl")
        require(nonempty(recorded_hash), "Batch definition needs frozen history hash")
    require(recorded_hash is None or recorded_hash == _hash(histories_path), "History version mismatch")
    rows = _rows(results_path)
    require(all(nonempty(r.get("result_id")) for r in rows)
            and len({r["result_id"] for r in rows}) == len(rows), "Invalid or duplicate result IDs")
    expected_count = source_manifest.get("expected_results" if is_batch else "result_count", len(rows))
    require(expected_count == len(rows), "Result file does not cover the source manifest")
    random.Random(seed).shuffle(rows)
    contract_only = (source_manifest.get("backend") == "fixture"
                     or source_manifest.get("result_type") == "contract_only"
                     or source_manifest.get("run_kind") == "contract_only")
    cases, mapping = [], {}
    for number, row in enumerate(rows, 1):
        require(row.get("history_id") in by_id, "Result refers to an unknown history")
        history = by_id[row["history_id"]]
        code = f"J{number:04d}"
        case = {"answer_code": code, "data_origin": history["data_origin"],
                "execution_status": row.get("status"), "contract_only": contract_only
                or row.get("backend") == "fixture" or row.get("result_type") == "contract_only"
                or row.get("run_kind") == "contract_only",
                "payload": {"history": _history_payload(history), "question": row.get("request_text"),
                            "answer": row.get("action")}}
        if protocol in CONTEXT_PROTOCOLS:
            if row.get("status") == "ok":
                require("profile_as_of_cutoff" in row and isinstance(row["profile_as_of_cutoff"], dict)
                        and "memory_context" in row and isinstance(row["memory_context"], str),
                        "Successful answers need recorded actual generator context")
            actual_profile = row.get("profile_as_of_cutoff")
            actual_memory = row.get("memory_context")
            require(actual_profile is None or isinstance(actual_profile, dict), "Invalid recorded actual profile")
            require(actual_memory is None or isinstance(actual_memory, str), "Invalid recorded actual memory")
            case["payload"]["generator_context"] = {
                "profile_as_of_cutoff": _profile_payload(actual_profile) if actual_profile is not None else None,
                "selected_memory": actual_memory}
        _validate_case(case, protocol=protocol)
        cases.append(case)
        mapping[code] = {key: row.get(key) for key in
                         ("result_id", "method", "history_id", "request_id", "status")}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            _line(stream, case)
    _json(output / "mapping.private.json", mapping)
    source_partial = is_batch and (any(source_manifest["job_counts"].get(key, 0) for key in
                                      ("pending", "running", "blocked"))
                                    or any(row.get("status") == "not_executed" for row in rows))
    report = {"protocol_version": "memory-judge-v1" if protocol == "legacy" else f"memory-judge-{protocol}",
              "status": "completed",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "case_count": len(cases),
              "generation_error_count": sum(c["execution_status"] != "ok" for c in cases),
              "source_kind": "batch" if is_batch else "run", "source_partial": bool(source_partial),
              "seed": seed, "contract_only": any(c["contract_only"] for c in cases), "scientific_result": False,
              "input_sha256": {"source_manifest": _hash(manifest_path), "results": _hash(results_path),
                               "histories": _hash(histories_path)},
              "cases_sha256": _hash(output / "cases.jsonl"),
              "validation_scope": "blind export; no semantic evaluation"}
    if is_batch:
        report["input_sha256"]["batch_definition"] = _hash(definition_path)
    if protocol in CONTEXT_PROTOCOLS:
        report["protocol"] = protocol
        report["judge_protocol"] = protocol
    _json(output / "manifest.json", report)
    return report


def validate_judgement(text: str, sources: dict[str, dict], *, protocol="legacy") -> dict:
    """Strict output and citation validation, without treating quotes as proof."""
    prompt_path(protocol)
    obj = parse_object(text)
    fields = {"verdict", "scores", "findings", "rationale"}
    if protocol in CONTEXT_PROTOCOLS:
        fields.add("utility")
    require(set(obj) == fields, "Invalid judgement fields")
    require(obj["verdict"] in VERDICTS and nonempty(obj["rationale"]), "Invalid verdict/rationale")
    require(protocol not in CONTEXT_PROTOCOLS or obj["utility"] in UTILITIES, "Invalid judgement utility")
    scores = obj["scores"]
    require(isinstance(scores, dict) and set(scores) == set(CRITERIA), "Invalid score fields")
    require(all(value is None or type(value) is int and value in {0, 1, 2} for value in scores.values()),
            "Scores must be 0, 1, 2 or null")
    findings = obj["findings"]
    require(isinstance(findings, list) and len(findings) <= 30, "Invalid findings")
    require(obj["verdict"] != "critical_error" or bool(findings), "Critical verdict needs findings")
    for finding in findings:
        require(isinstance(finding, dict) and set(finding) == set(FINDING_FIELDS), "Invalid finding fields")
        require(finding["criterion"] in CRITERIA and nonempty(finding["claim"])
                and nonempty(finding["explanation"]), "Invalid finding content")
        ids, quote = finding["evidence_turn_ids"], finding["quote"]
        require(isinstance(ids, list) and all(isinstance(x, str) for x in ids)
                and len(ids) == len(set(ids)) and set(ids) <= set(sources), "Unknown or duplicate citation")
        require(isinstance(quote, str), "Finding quote must be text")
        require(not ids and not quote or bool(ids) and nonempty(quote)
                and any(normalized(quote) in normalized(sources[x]["text"]) for x in ids),
                "Finding quote must be a literal excerpt from a cited turn")
    return obj


class _JudgeFixture:
    def __init__(self, protocol="legacy"):
        prompt_path(protocol)
        self.protocol = protocol

    def complete(self, messages, purpose, response_schema=None):
        require(purpose == "judge", "Unexpected judge fixture purpose")
        result = {"verdict": "uncertain", "scores": {key: None for key in CRITERIA},
                  "findings": [], "rationale": "Проверка контракта без смысловой оценки моделью."}
        if self.protocol in CONTEXT_PROTOCOLS:
            result["utility"] = "uncertain"
        return Completion(json.dumps(result, ensure_ascii=False),
                          {"input_tokens": None, "output_tokens": None}, 0,
                          {"backend": "fixture", "purpose": purpose, "scientific_result": False})


class _ValidatedClient:
    def __init__(self, client, protocol="legacy"):
        self.client, self.sources, self.protocol = client, {}, protocol

    def complete(self, messages, purpose, response_schema=None):
        completion = self.client.complete(messages, purpose, response_schema)
        try:
            validate_judgement(completion.text, self.sources, protocol=self.protocol)
        except (ValueError, TypeError, KeyError):
            raise CompletionValidationError("Judge output failed schema or source checks", completion,
                                            "invalid_judgement") from None
        return completion


def judge_cases(cases_path, output, *, backend="fixture", model=None, base_url=None,
                max_calls=128, timeout_seconds=180, max_output_tokens=1536, client=None,
                protocol="legacy") -> dict:
    """Judge one blind answer per call; fixture never performs semantic grading.

    Output is exclusive: an existing or interrupted directory is never silently
    resumed or repeated. A caller must inspect it before scheduling more calls.
    """
    selected_prompt = prompt_path(protocol)
    cases_path, output = Path(cases_path), Path(output)
    cases = _rows(cases_path)
    require(len({c.get("answer_code") for c in cases}) == len(cases), "Duplicate answer codes")
    sources = {case["answer_code"]: _validate_case(case, protocol=protocol) for case in cases}
    require(backend in {"fixture", "ollama", "spiderssense"}, "Unsupported judge backend")
    require(type(max_calls) is int and max_calls >= 0, "max_calls must be nonnegative")
    require(backend == "fixture" or nonempty(model), "An explicit judge model is required")
    require(backend != "spiderssense" or client is not None, "Spiderssense requires an explicit configured client")
    require(backend != "fixture" or client is None, "Fixture judging uses its built-in stub only")
    require(timeout_seconds > 0 and type(max_output_tokens) is int and 0 < max_output_tokens < 16384,
            "Invalid judge time/output limit")
    if client is None:
        client = _JudgeFixture(protocol) if backend == "fixture" else OllamaClient(
            base_url=base_url or "http://127.0.0.1:11435", model=model, timeout=timeout_seconds,
            context_tokens=16384, max_output_tokens=max_output_tokens)
    wrapped = _ValidatedClient(client, protocol)
    prompt = selected_prompt.read_text(encoding="utf-8")
    response_schema = JUDGEMENT_SCHEMA if protocol == "legacy" else CONTEXT_JUDGEMENT_SCHEMA
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol_version": "memory-judge-v1" if protocol == "legacy" else f"memory-judge-{protocol}",
                "status": "running", "backend": backend,
                "model": model if backend != "fixture" else None,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "scientific_result": False,
                "contract_only": backend == "fixture" or any(c["contract_only"] for c in cases),
                "real_model_calls": False, "case_count": len(cases), "max_calls": max_calls,
                "timeout_seconds": timeout_seconds, "max_output_tokens": max_output_tokens,
                "context_tokens": 16384 if backend == "ollama" else None,
                "input_sha256": {"cases": _hash(cases_path)},
                "prompt_sha256": _hash(selected_prompt), "code_sha256": _hash(Path(__file__)),
                "assessment_origin": "machine_only; not pedagogical expert review"}
    if protocol in CONTEXT_PROTOCOLS:
        manifest["protocol"] = protocol
        manifest["judge_protocol"] = protocol
    _json(output / "manifest.json", manifest)
    counts = Counter()
    verdicts = Counter()
    utilities = Counter()
    journal = None
    try:
        with (output / "calls.jsonl").open("w", encoding="utf-8") as calls_stream, \
                (output / "trace.jsonl").open("w", encoding="utf-8") as trace_stream, \
                (output / "judgements.jsonl").open("w", encoding="utf-8") as results:
            journal = CallJournal(wrapped, calls_stream, max_calls, trace_stream)
            for case in cases:
                code = case["answer_code"]
                row = {"answer_code": code, "status": "not_assessed", "judgement": None,
                       "contract_only": backend == "fixture" or case["contract_only"],
                       "assessment_origin": "machine_only", "error": None}
                if protocol in CONTEXT_PROTOCOLS:
                    row["protocol"] = protocol
                if case["execution_status"] != "ok" or case["payload"]["answer"] is None:
                    row["error"] = {"code": "generation_not_successful"}
                else:
                    journal.context = {"answer_code": code}
                    wrapped.sources = sources[code]
                    try:
                        response = journal([{"role": "system", "content": prompt},
                                            {"role": "user", "content": json.dumps(case["payload"],
                                                                                   ensure_ascii=False)}],
                                           "judge", response_schema)
                        row.update(status="assessed", judgement=validate_judgement(
                            response, wrapped.sources, protocol=protocol))
                        verdicts[row["judgement"]["verdict"]] += 1
                        if protocol in CONTEXT_PROTOCOLS:
                            utilities[row["judgement"]["utility"]] += 1
                    except Exception as exc:
                        row.update(status="judge_error", error={"code": "judge_call_failed",
                                                                "type": type(exc).__name__})
                _line(results, row)
                counts[row["status"]] += 1
        manifest["status"] = "completed_with_errors" if counts["judge_error"] or counts["not_assessed"] else "completed"
    finally:
        manifest.update(counts=dict(counts), verdict_counts=dict(verdicts),
                        unprocessed_count=len(cases) - sum(counts.values()),
                        performed_calls=journal.performed if journal else 0,
                        attempted_calls=journal.attempted if journal else 0)
        manifest["real_model_calls"] = backend != "fixture" and manifest["performed_calls"] > 0
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        if protocol in CONTEXT_PROTOCOLS:
            manifest["utility_counts"] = dict(utilities)
        _json(output / "manifest.json", manifest)
        report = ("# Машинная оценка ответов\n\n"
                  + ("Проверка контракта: смысловая пригодность не измерялась.\n\n"
                     if manifest["contract_only"] else
                     "Один автоматический судья; педагогической экспертной проверки нет.\n\n")
                  + "| Показатель | Число |\n| --- | ---: |\n"
                  + f"| Запланировано | {len(cases)} |\n"
                  + f"| Технически оценено | {counts['assessed']} |\n"
                  + f"| Не оценено из-за генерации | {counts['not_assessed']} |\n"
                  + f"| Сбой судьи | {counts['judge_error']} |\n"
                  + f"| Не обработано при остановке | {manifest['unprocessed_count']} |\n"
                  + f"| acceptable | {verdicts['acceptable']} |\n"
                  + f"| critical_error | {verdicts['critical_error']} |\n"
                  + f"| uncertain | {verdicts['uncertain']} |\n"
                  + ("" if protocol == "legacy" else
                     "".join(f"| utility: {label} | {utilities[label]} |\n" for label in UTILITIES))
                  + "\n"
                  + "Сбои сохранены в общем числе заданий. Метки относятся к реально оценённым ответам; "
                  + "это не измерение точности судьи. Баллы по осям не суммируются. "
                  + "См. judgements.jsonl, calls.jsonl и manifest.json; полные синтетические входы "
                  + "и ответы судьи сохранены в trace.jsonl. Второй судья этим запуском не выполнялся.\n")
        (output / "report.md").write_text(report, encoding="utf-8")
    return manifest


def summarize_calibration(judgements_path, expected_path, output, *, protocol="legacy") -> dict:
    """Apply an explicit known-case gate; expected labels never go to the judge.

    Expected rows: answer_code, expected_verdict, optional critical (boolean),
    explanation (text), evidence_turn_ids (unique source ID list). Informational
    explanation/source fields remain outside the judge's model payload.
    In context protocols required_utility optionally adds an independent utility gate.
    Every acceptable answer must be accepted; every designated critical error
    must be found. Uncertain expected examples are diagnostic and do not require
    a particular verdict. All expected cases must be technically assessed.
    """
    prompt_path(protocol)
    judgements_path, expected_path, output = map(Path, (judgements_path, expected_path, output))
    expected, judgements = _rows(expected_path), _rows(judgements_path, allow_empty=True)
    require(len({r.get("answer_code") for r in expected}) == len(expected), "Duplicate expected answer codes")
    require(len({r.get("answer_code") for r in judgements}) == len(judgements), "Duplicate judgement answer codes")
    expected_fields = {"answer_code", "expected_verdict", "critical", "explanation", "evidence_turn_ids"}
    if protocol in CONTEXT_PROTOCOLS:
        expected_fields.add("required_utility")
    for row in expected:
        require(set(row) <= expected_fields
                and identifier(row.get("answer_code")) and row.get("expected_verdict") in VERDICTS,
                "Invalid calibration expectation")
        require("critical" not in row or type(row["critical"]) is bool, "Invalid critical designation")
        require("explanation" not in row or nonempty(row["explanation"]), "Invalid expected explanation")
        ids = row.get("evidence_turn_ids", [])
        require(isinstance(ids, list) and all(identifier(value) for value in ids)
                and len(ids) == len(set(ids)), "Invalid expected evidence IDs")
        require(not row.get("critical") or row["expected_verdict"] == "critical_error",
                "Critical designation needs critical_error expectation")
        require("required_utility" not in row or row["required_utility"] in UTILITIES,
                "Invalid required calibration utility")
    expected_ids = {r["answer_code"] for r in expected}
    by_code = {r.get("answer_code"): r for r in judgements}
    require(set(by_code) <= expected_ids, "Judgements contain cases absent from calibration expectations")
    confusion = {label: {actual: 0 for actual in (*VERDICTS, "not_assessed")} for label in VERDICTS}
    detail = []
    utilities = Counter()
    contract_only = any(r.get("contract_only") is not False for r in judgements)
    for expectation in expected:
        code, want = expectation["answer_code"], expectation["expected_verdict"]
        row = by_code.get(code)
        judgement = row.get("judgement") if row else None
        got = (judgement.get("verdict") if row and row.get("status") == "assessed"
               and isinstance(judgement, dict) else "not_assessed")
        require(got in {*VERDICTS, "not_assessed"}, "Invalid persisted judgement verdict")
        utility = "not_assessed"
        if isinstance(judgement, dict):
            if protocol in CONTEXT_PROTOCOLS and got != "not_assessed":
                require(judgement.get("utility") in UTILITIES, "Invalid persisted judgement utility")
                utility = judgement["utility"]
                utilities[utility] += 1
            elif protocol == "legacy":
                require("utility" not in judgement, "Context judgements require an explicit context protocol")
        require(row is None or row.get("protocol", protocol) == protocol,
                "Persisted judgement uses a different protocol")
        confusion[want][got] += 1
        critical = expectation.get("critical", want == "critical_error")
        required = "critical_error" if critical else "acceptable" if want == "acceptable" else None
        detail.append({"answer_code": code, "expected_verdict": want, "actual_verdict": got,
                       "required_verdict": required, "satisfied": got != "not_assessed"
                       and (required is None or got == required)})
        if protocol in CONTEXT_PROTOCOLS:
            required_utility = expectation.get("required_utility")
            detail[-1].update(required_utility=required_utility, actual_utility=utility,
                              utility_satisfied=got != "not_assessed"
                              and (required_utility is None or utility == required_utility))
    assessed = sum(r["actual_verdict"] != "not_assessed" for r in detail)
    unknown = sum(r["actual_verdict"] == "uncertain" for r in detail)
    gate = {"all_cases_assessed": assessed == len(expected),
            "all_designated_critical_errors_found": all(r["satisfied"] for r in detail
                                                       if r["required_verdict"] == "critical_error"),
            "all_acceptable_answers_accepted": all(r["satisfied"] for r in detail
                                                   if r["required_verdict"] == "acceptable"),
            "not_contract_only": not contract_only,
            "has_critical_and_acceptable_checks": any(r["required_verdict"] == "critical_error" for r in detail)
            and any(r["required_verdict"] == "acceptable" for r in detail)}
    if protocol in CONTEXT_PROTOCOLS:
        gate["all_required_utilities_matched"] = all(r["utility_satisfied"] for r in detail
                                                   if r["required_utility"] is not None)
    report = {"protocol_version": ("memory-judge-calibration-v1" if protocol == "legacy"
                                   else f"memory-judge-calibration-{protocol}"), "expected_count": len(expected),
              "assessed_count": assessed, "not_assessed_count": len(expected) - assessed,
              "uncertain_count": unknown, "determinate_count": assessed - unknown,
              "determinate_fraction_of_expected": (assessed - unknown) / len(expected),
              "confusion": confusion, "contract_only": contract_only, "gate": gate,
              "qualified": all(gate.values()), "scientific_result": False,
              "scope": "known synthetic cases; machine evaluation, not expert validation",
              "input_sha256": {"judgements": _hash(judgements_path), "expected": _hash(expected_path)},
              "cases": detail}
    if protocol in CONTEXT_PROTOCOLS:
        report.update(protocol=protocol, judge_protocol=protocol, utility_counts=dict(utilities))
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "Calibration report already exists")
    _json(output, report)
    return report
