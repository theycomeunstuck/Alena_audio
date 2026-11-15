"""Strict source-grounded contracts, independent of the frozen judge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CRITERIA = ("factual_accuracy", "help_attribution", "action_appropriateness",
            "source_support", "unknowns_handling")
VERDICTS = ("acceptable", "critical_error", "uncertain")
UTILITIES = ("useful", "limited", "uncertain")
ROOT = Path(__file__).resolve().parent
STR = {"type": "string", "minLength": 1}
EVIDENCE = {"type": "array", "minItems": 1, "maxItems": 8, "items": {
    "type": "object", "additionalProperties": False,
    "required": ["source_turn_id", "quote"],
    "properties": {"source_turn_id": STR, "quote": STR}}}
SCORE = {"type": "object", "additionalProperties": False,
         "required": ["score", "explanation", "evidence"], "properties": {
             "score": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
             "explanation": STR, "evidence": EVIDENCE}}
FINDING = {"type": "object", "additionalProperties": False,
           "required": ["criterion", "claim", "explanation", "evidence"],
           "properties": {"criterion": {"type": "string", "enum": list(CRITERIA)},
                          "claim": STR, "explanation": STR, "evidence": EVIDENCE}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["verdict", "utility", "substantial_distortion", "criteria", "findings", "rationale"],
          "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)},
                         "utility": {"type": "string", "enum": list(UTILITIES)},
                         "substantial_distortion": {"type": "boolean"},
                         "criteria": {"type": "object", "additionalProperties": False,
                                      "required": list(CRITERIA),
                                      "properties": {key: SCORE for key in CRITERIA}},
                         "findings": {"type": "array", "maxItems": 20, "items": FINDING},
                         "rationale": STR}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def lines(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def sources(case):
    require(set(case) == {"answer_code", "data_origin", "execution_status", "contract_only", "payload"},
            "Unexpected blind case metadata")
    require(str(case["data_origin"]).startswith("synthetic"), "Only synthetic histories are allowed")
    payload = case["payload"]
    require(set(payload) == {"history", "question", "answer", "generator_context"}, "Unexpected judge payload")
    require(set(payload["history"]) == {"profile_as_of_cutoff", "lessons"}, "Unexpected source metadata")
    turns = [turn for lesson in payload["history"]["lessons"] for turn in lesson["turns"]]
    result = {turn["turn_id"]: turn for turn in turns}
    require(len(result) == len(turns) and bool(turns), "Sources must have unique IDs")
    require(all(set(turn) == {"turn_id", "role", "text"} and turn["role"] in {"teacher", "student", "assistant"}
                and isinstance(turn["text"], str) and turn["text"].strip() for turn in turns), "Invalid source turns")
    context = payload["generator_context"]
    require(set(context) == {"profile_as_of_cutoff", "selected_memory"}, "Unexpected generator metadata")
    require(context["profile_as_of_cutoff"] == payload["history"]["profile_as_of_cutoff"], "Profile mismatch")
    require(isinstance(context["selected_memory"], str), "Missing recorded memory")
    require(isinstance(payload["answer"], dict) and isinstance(payload["question"], str), "Invalid answer/question")
    return result


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _evidence(value, available):
    require(isinstance(value, list) and 0 < len(value) <= 8, "Each assessed criterion/finding needs evidence")
    seen = set()
    for item in value:
        require(isinstance(item, dict) and set(item) == {"source_turn_id", "quote"}, "Invalid evidence fields")
        source_id, quote = item["source_turn_id"], item["quote"]
        require(isinstance(source_id, str) and source_id in available, "Unknown evidence ID")
        require(_text(quote) and quote in available[source_id]["text"], "Quote is not a literal source substring")
        require((source_id, quote) not in seen, "Duplicate evidence")
        seen.add((source_id, quote))


def validate(text, case):
    """No whitespace normalization, duplicate-key tolerance, fences or repairs."""
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Non-finite JSON value")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(value, dict) and set(value) == set(SCHEMA["required"]), "Incomplete or extra judgement fields")
    available = sources(case)
    require(value["verdict"] in VERDICTS and value["utility"] in UTILITIES and _text(value["rationale"]),
            "Invalid judgement labels/rationale")
    require(type(value["substantial_distortion"]) is bool, "Invalid distortion label")
    criteria = value["criteria"]
    require(isinstance(criteria, dict) and set(criteria) == set(CRITERIA), "All five criteria are required")
    for item in criteria.values():
        require(isinstance(item, dict) and set(item) == {"score", "explanation", "evidence"}, "Incomplete criterion")
        require(item["score"] is None or type(item["score"]) is int and item["score"] in {0, 1, 2}, "Invalid criterion score")
        require(_text(item["explanation"]), "Empty criterion explanation")
        _evidence(item["evidence"], available)
    findings = value["findings"]
    require(isinstance(findings, list) and len(findings) <= 20, "Invalid findings")
    for finding in findings:
        require(isinstance(finding, dict) and set(finding) == {"criterion", "claim", "explanation", "evidence"},
                "Incomplete finding")
        require(finding["criterion"] in CRITERIA and _text(finding["claim"]) and _text(finding["explanation"]),
                "Invalid finding")
        claim = finding["claim"]
        answer = case["payload"]["answer"]
        require(claim in answer or any(claim in text for text in (answer.get("reason", ""), answer.get("action", ""))),
                "Finding claim must identify an answer field or literal answer substring")
        _evidence(finding["evidence"], available)
    require(value["verdict"] != "critical_error" or bool(findings) and value["substantial_distortion"],
            "Critical error requires a grounded substantial distortion")
    require(not value["substantial_distortion"] or bool(findings), "Distortion requires findings")
    require(value["verdict"] != "acceptable" or not value["substantial_distortion"], "Inconsistent acceptable verdict")
    require(not any(item["score"] == 0 for item in criteria.values()) or bool(findings), "Zero scores require findings")
    return value


def validate_expected(expected):
    """Schema-check expected-label rows; must hold before any paid call depends on them."""
    problems = []
    seen = set()
    for number, row in enumerate(expected, 1):
        where = f"row {number}"
        if not isinstance(row, dict):
            problems.append(f"{where}: row is not a JSON object")
            continue
        code = row.get("answer_code")
        if _text(code):
            where = f"row {number} ({code})"
            if code in seen:
                problems.append(f"{where}: duplicate answer_code")
            seen.add(code)
        missing = [field for field in ("answer_code", "expected_verdict", "critical",
                                       "required_utility", "evidence_turn_ids", "situation_type")
                   if field not in row]
        if missing:
            problems.append(f"{where}: missing required fields {missing}")
        if "answer_code" in row and not _text(row["answer_code"]):
            problems.append(f"{where}: answer_code must be a non-empty string")
        if "expected_verdict" in row and row["expected_verdict"] not in VERDICTS:
            problems.append(f"{where}: expected_verdict must be one of " + ", ".join(VERDICTS))
        if "critical" in row and type(row["critical"]) is not bool:
            problems.append(f"{where}: critical must be a boolean")
        if "required_utility" in row and row["required_utility"] not in UTILITIES:
            problems.append(f"{where}: required_utility must be one of " + ", ".join(UTILITIES))
        ids = row.get("evidence_turn_ids")
        if "evidence_turn_ids" in row and not (isinstance(ids, list) and bool(ids)
                and all(_text(item) for item in ids)):
            problems.append(f"{where}: evidence_turn_ids must be a non-empty list of non-empty strings")
        if "situation_type" in row and not _text(row["situation_type"]):
            problems.append(f"{where}: situation_type must be a non-empty string")
    require(not problems, "Malformed expected labels (checked before any paid call); offending rows: "
            + "; ".join(problems))


def calibration_gate(cases, expected, judgements):
    validate_expected(expected)
    require(len(cases) == len(expected) == 24, "Calibration must cover 24 new cases")
    by_code = {case["answer_code"]: case for case in cases}
    want = {row["answer_code"]: row for row in expected}
    got = {row["answer_code"]: row for row in judgements}
    require(len(by_code) == len(want) == 24 and len(got) == len(judgements), "Duplicate calibration ID")
    require(set(by_code) == set(want) and set(got) <= set(want), "Calibration ID mismatch")
    details = []
    for code, expectation in want.items():
        row = got.get(code, {})
        judgement = row.get("judgement")
        assessed = row.get("status") == "assessed" and isinstance(judgement, dict)
        if assessed:
            try:
                validate(json.dumps(judgement, ensure_ascii=False), by_code[code])
            except (ValueError, TypeError, KeyError):
                assessed = False
        cited = {item["source_turn_id"] for criterion in judgement["criteria"].values()
                 for item in criterion["evidence"]} if assessed else set()
        if assessed:
            cited.update(item["source_turn_id"] for finding in judgement["findings"] for item in finding["evidence"])
        details.append({"answer_code": code, "assessed": assessed,
                        "expected_verdict": expectation["expected_verdict"],
                        "actual_verdict": judgement.get("verdict") if assessed else None,
                        "critical_found": assessed and judgement["verdict"] == "critical_error" if expectation["critical"] else None,
                        "acceptable_accepted": assessed and judgement["verdict"] == "acceptable" if expectation["expected_verdict"] == "acceptable" else None,
                        "required_utility": expectation["required_utility"],
                        "actual_utility": judgement.get("utility") if assessed else None,
                        "utility_matched": assessed and judgement["utility"] == expectation["required_utility"],
                        "required_evidence_ids_covered": assessed and set(expectation["evidence_turn_ids"]) <= cited})
    checks = {"all_cases_assessed_and_evidence_valid": all(row["assessed"] for row in details),
              "all_designated_critical_errors_found": all(row["critical_found"] for row in details if row["critical_found"] is not None),
              "all_acceptable_answers_accepted": all(row["acceptable_accepted"] for row in details if row["acceptable_accepted"] is not None),
              "all_required_utilities_matched": all(row["utility_matched"] for row in details),
              "all_required_grounds_cited": all(row["required_evidence_ids_covered"] for row in details),
              "eight_types_three_variants": len({row["situation_type"] for row in expected}) == 8
                  and all(sum(row["situation_type"] == typ for row in expected) == 3 for typ in {row["situation_type"] for row in expected}),
              "eight_critical_and_eight_acceptable": sum(row["critical"] for row in expected) == 8
                  and sum(row["expected_verdict"] == "acceptable" for row in expected) == 8,
              "real_judge_only": bool(judgements) and all(row.get("contract_only") is False for row in judgements)}
    return {"qualified": all(checks.values()), "gate": checks, "cases": details,
            "assessment_origin": "additional automatic evaluation; not human expertise"}
