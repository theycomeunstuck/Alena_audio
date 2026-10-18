"""Syntactic, provenance and literal-quote checks, never educational truth checks."""

from __future__ import annotations

from copy import deepcopy
import json
import re

from .dataset import nonempty


ACTIONS = ["enough_data_next_task", "independent_check", "continue_work", "clarify_missing_observation"]
KINDS = ["learning_episode", "unfinished_work", "preference", "source_correction"]
SUPPORT = ["none", "hint_question", "visual", "joint", "unknown"]
RECHECK = ["not_observed", "success", "failure", "unknown"]
STR = {"type": "string"}
IDS = {"type": "array", "items": STR, "uniqueItems": True}
CONDITION_EVIDENCE = {
    "anyOf": [{"type": "null"}, {
        "type": "object", "additionalProperties": False,
        "properties": {"source_turn_id": STR, "quote": STR},
        "required": ["source_turn_id", "quote"],
    }],
}
RECORD_FIELDS = {
    "kind": {"type": "string", "enum": KINDS}, "topic_id": STR, "event": STR,
    "support_level": {"type": "string", "enum": SUPPORT},
    "independent_recheck": {"type": "string", "enum": RECHECK},
    "evidence_turn_ids": IDS, "source_quote": STR,
    "support_evidence": deepcopy(CONDITION_EVIDENCE), "recheck_evidence": deepcopy(CONDITION_EVIDENCE),
    "corrects_record_id": {"type": ["string", "null"]},
}
EXTRACT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["records"],
                  "properties": {"records": {"type": "array", "maxItems": 100, "items": {
                      "type": "object", "additionalProperties": False,
                      "properties": RECORD_FIELDS, "required": list(RECORD_FIELDS)}}}}
ACTION_FIELDS = {"action": {"type": "string", "enum": ACTIONS}, "reason": STR,
                 "evidence_turn_ids": IDS, "unknowns": {"type": "array", "items": STR},
                 "review_required": {"type": "boolean", "const": True}}
ACTION_SCHEMA = {"type": "object", "additionalProperties": False,
                 "properties": ACTION_FIELDS, "required": list(ACTION_FIELDS)}


class ContractValidationError(ValueError):
    """A safe, stable diagnostic; the original completion must remain unmodified."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def _check(condition: bool, message: str, code: str) -> None:
    if not condition:
        raise ContractValidationError(message, code)


def _citation_ids_schema(allowed_ids: set[str]) -> dict:
    schema = deepcopy(IDS)
    schema["maxItems"] = len(allowed_ids)
    if allowed_ids:
        schema["items"]["enum"] = sorted(allowed_ids)
    return schema


def extract_schema(source_ids: set[str], topic_id: str, record_ids: set[str] | None = None) -> dict:
    """Constrain source and correction IDs without claiming semantic correctness."""
    schema = deepcopy(EXTRACT_SCHEMA)
    fields = schema["properties"]["records"]["items"]["properties"]
    fields["topic_id"] = {"type": "string", "enum": [topic_id]}
    fields["evidence_turn_ids"] = _citation_ids_schema(source_ids)
    fields["corrects_record_id"] = {"type": ["string", "null"], "enum": [None, *sorted(record_ids or ())]}
    for name in ("support_evidence", "recheck_evidence"):
        source_field = fields[name]["anyOf"][1]["properties"]["source_turn_id"]
        source_field["enum"] = sorted(source_ids)
        if not source_ids:
            fields[name] = {"type": "null"}
    return schema


def action_schema(visible_ids: set[str]) -> dict:
    """Constrain citation syntax to the IDs present in the selected context."""
    schema = deepcopy(ACTION_SCHEMA)
    schema["properties"]["evidence_turn_ids"] = _citation_ids_schema(visible_ids)
    return schema


def parse_object(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            _check(key not in result, "Completion contains a duplicate JSON key", "duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs)
    except (TypeError, json.JSONDecodeError):
        raise ContractValidationError("Completion is not a JSON object", "invalid_json") from None
    _check(isinstance(value, dict), "Completion must be a JSON object", "invalid_envelope")
    return value


def _validate_ids(values: object, allowed: set[str]) -> None:
    _check(isinstance(values, list) and all(isinstance(x, str) for x in values),
           "Citation IDs must be a list of strings", "invalid_citations")
    _check(len(values) == len(set(values)), "Duplicate citation IDs are not allowed", "duplicate_citation")
    _check(set(values) <= allowed, "Citation refers to a source absent from its context", "missing_source")


def normalized(text: str) -> str:
    """Whitespace normalization for callers; literal contract checks do not use it."""
    return " ".join(text.split())


def _condition_evidence(value: str, evidence: object, sources: dict[str, dict], cited: set[str], name: str) -> None:
    if value == "unknown":
        _check(evidence is None, f"Unknown {name} must have null evidence", "unknown_condition_has_evidence")
        return
    _check(isinstance(evidence, dict) and set(evidence) == {"source_turn_id", "quote"},
           f"Known {name} needs a source ID and literal quote; otherwise use unknown",
           "missing_condition_evidence")
    source_id = evidence["source_turn_id"]
    _check(isinstance(source_id, str) and source_id in sources and source_id in cited,
           f"{name} evidence refers to an absent or uncited source", "missing_source")
    quote = evidence["quote"]
    _check(nonempty(quote) and quote in sources[source_id]["text"],
           f"{name} quote must be a literal source excerpt", "invalid_quote")
    if name == "recheck" and value in {"success", "failure"}:
        _check(sources[source_id].get("role") != "assistant",
               "Assistant output cannot establish a learner's independent result", "assistant_as_learner_evidence")


def validate_records(text: str, sources: dict[str, dict], new_ids: set[str], topic_id: str,
                     previous_records: list[dict] | tuple = ()) -> list[dict]:
    obj = parse_object(text)
    _check(set(obj) == {"records"} and isinstance(obj["records"], list),
           "Invalid records envelope", "invalid_envelope")
    _check(len(obj["records"]) <= 100, "Too many records in one update", "too_many_records")
    previous = {r["record_id"]: r for r in previous_records}
    seen = set()
    for record in obj["records"]:
        _check(isinstance(record, dict) and set(record) == set(RECORD_FIELDS),
               "Invalid record fields", "invalid_record_fields")
        fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        _check(fingerprint not in seen, "Completion contains a duplicate memory record", "duplicate_record")
        seen.add(fingerprint)
        _check(record["kind"] in KINDS and record["topic_id"] == topic_id,
               "Invalid record kind/topic", "invalid_record_topic")
        _check(nonempty(record["event"]) and len(record["event"]) <= 2000,
               "Invalid event text", "invalid_event")
        _check(record["support_level"] in SUPPORT and record["independent_recheck"] in RECHECK,
               "Invalid support/recheck value", "invalid_condition")
        ids = record["evidence_turn_ids"]
        _validate_ids(ids, set(sources))
        _check(bool(set(ids) & new_ids), "Record needs processed and new sources", "missing_new_source")
        quote = record["source_quote"]
        _check(nonempty(quote) and any(quote in sources[x]["text"] for x in set(ids) & new_ids),
               "source_quote must be a literal excerpt from one cited new turn", "invalid_quote")
        _condition_evidence(record["support_level"], record["support_evidence"], sources, set(ids), "support")
        _condition_evidence(record["independent_recheck"], record["recheck_evidence"], sources, set(ids), "recheck")
        target = record["corrects_record_id"]
        if record["kind"] == "source_correction":
            _check(isinstance(target, str) and target in previous,
                   "Source correction needs a concrete existing record ID", "invalid_correction_target")
            _check(previous[target]["topic_id"] == topic_id,
                   "Source correction cannot target another topic", "invalid_correction_target")
        else:
            _check(target is None, "Ordinary progress cannot correct a past source record", "invalid_correction_kind")
    return obj["records"]


def validate_action(text: str, visible_ids: set[str]) -> dict:
    action = parse_object(text)
    _check(set(action) == set(ACTION_FIELDS), "Invalid action fields", "invalid_action_fields")
    _check(action["action"] in ACTIONS and nonempty(action["reason"]), "Invalid action/reason", "invalid_action")
    _validate_ids(action["evidence_turn_ids"], visible_ids)
    _check(isinstance(action["unknowns"], list) and all(nonempty(x) for x in action["unknowns"]),
           "Invalid unknowns", "invalid_unknowns")
    _check(action["review_required"] is True, "Human review must remain required", "review_required")
    return action


def visible_references(text: str, known_ids: set[str]) -> set[str]:
    return {x for x in known_ids if re.search(r"(?<![\w.:-])" + re.escape(x) + r"(?![\w.:-])", text)}
