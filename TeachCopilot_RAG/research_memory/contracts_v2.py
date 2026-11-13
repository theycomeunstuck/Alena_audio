"""Strict, versioned learner-memory contract used by the opt-in V2 writer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from .contracts import ContractValidationError, parse_object
from .dataset import nonempty


SCHEMA_PATH = Path(__file__).parent / "data" / "graph_extension_v2" / "memory_record_schema.design.json"
PERSISTED_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
RECORD_TYPES = tuple(PERSISTED_SCHEMA["properties"]["record_type"]["enum"])
SUPPORT = tuple(PERSISTED_SCHEMA["properties"]["support_level"]["enum"])
RECHECK = tuple(PERSISTED_SCHEMA["properties"]["independent_recheck"]["enum"])
INTERPRETATION_STATUS = tuple(PERSISTED_SCHEMA["properties"]["interpretation_status"]["enum"])
LINK_RELATIONS = tuple(PERSISTED_SCHEMA["$defs"]["memory_link"]["properties"]["relation"]["enum"])
INTEREST_OPERATIONS = ("add", "remove", "avoid", "allow")
RETENTION_CLASSES = ("session_only", "profile_candidate", "episodic_personal_fact")

SERVER_FIELDS = {
    "record_id", "occurred_on", "last_seen", "evidence_count", "use_policy", "corrected_by_record_ids",
}
EXTRACT_FIELDS = tuple(key for key in PERSISTED_SCHEMA["properties"] if key not in SERVER_FIELDS)
LEARNING_TYPES = {
    "child_error", "child_correction", "supported_attempt", "uncertain_observation",
    "learning_success", "unfinished_work", "source_correction",
}
PROFILE_TYPES = {"interest_candidate", "session_preference_candidate", "incidental_fact"}

USE_POLICY = {
    "child_error": "Одна наблюдавшаяся ошибка. Сначала проверьте, повторяется ли она.",
    "child_correction": "Сохранить рядом с прежним ответом, не стирать ошибку.",
    "supported_attempt": "Сохранить вид помощи; не называть попытку самостоятельной.",
    "uncertain_observation": "Считать навык неизвестным и уточнить коротким вопросом.",
    "learning_success": "Один наблюдавшийся успех не доказывает устойчивое освоение.",
    "unfinished_work": "Перед продолжением уточнить, осталась ли задача незавершённой.",
    "interest_candidate": "Кандидат интереса, не подтверждённое предпочтение.",
    "session_preference_candidate": "Применять к текущему занятию; для постоянного предпочтения нужно подтверждение.",
    "incidental_fact": "Хранить отдельно от учебного навыка; упоминать только по делу.",
    "source_correction": "Исправляет только связанную запись и сохраняет её историю.",
}


def _check(condition: bool, message: str, code: str) -> None:
    if not condition:
        raise ContractValidationError(message, code)


def extract_schema_v2(source_ids: set[str], allowed_topic_ids: set[str],
                      record_ids: set[str] | None = None) -> dict:
    """Build a model-facing schema from the persisted V2 design schema.

    IDs and dates assigned by the server are removed from the extraction contract.
    The shared V1 contract remains unchanged.
    """
    _check(isinstance(allowed_topic_ids, set), "V2 extraction needs a topic set", "invalid_topic_set")
    persisted = deepcopy(PERSISTED_SCHEMA)
    properties = persisted["properties"]
    record_properties = {key: deepcopy(properties[key]) for key in EXTRACT_FIELDS}
    if not allowed_topic_ids:
        record_properties["record_type"]["enum"] = [kind for kind in RECORD_TYPES if kind in PROFILE_TYPES]
    record_properties["topic_id"] = {"type": ["string", "null"],
                                     "enum": [None, *sorted(allowed_topic_ids)]}
    evidence_schema = deepcopy(persisted["$defs"]["evidence"])
    evidence_schema["properties"]["turn_id"]["enum"] = sorted(source_ids)
    record_properties["evidence"] = {
        "type": "array", "minItems": 1, "maxItems": 20, "items": evidence_schema,
    }
    for field in ("support_evidence", "recheck_evidence"):
        record_properties[field] = {
            "anyOf": [{"type": "null"}, deepcopy(evidence_schema)]
        }
    link_schema = deepcopy(persisted["$defs"]["memory_link"])
    link = link_schema
    link["properties"]["target_kind"]["enum"] = ["record", "turn"]
    link["properties"]["target_id"]["enum"] = sorted(source_ids | set(record_ids or ()))
    record_properties["links"] = {"type": "array", "maxItems": 20, "items": link_schema}
    record_properties["corrects_record_id"] = {
        "type": ["string", "null"], "enum": [None, *sorted(record_ids or ())]
    }
    record_schema = {
        "type": "object", "additionalProperties": False,
        "properties": record_properties, "required": list(EXTRACT_FIELDS),
        "allOf": [],
    }
    allowed_record_types = set(RECORD_TYPES) if allowed_topic_ids else PROFILE_TYPES
    for condition in persisted.get("allOf", []):
        condition_type = (condition.get("if", {}).get("properties", {})
                          .get("record_type", {}).get("const"))
        if condition_type is not None and condition_type not in allowed_record_types:
            continue
        record_schema["allOf"].append(deepcopy(condition))
    return {
        "type": "object", "additionalProperties": False, "required": ["records"],
        "properties": {"records": {"type": "array", "maxItems": 8, "items": record_schema}},
    }


def _role(source: dict) -> str | None:
    return source.get("role") or source.get("author_role")


def _whitespace_equivalent_source_excerpt(quote: str, source_text: str) -> str | None:
    """Return the exact source span when a quote differs only in whitespace."""
    if quote in source_text:
        return quote
    tokens = quote.split()
    if not tokens:
        return None
    pattern = [re.escape(tokens[0])]
    for left, right in zip(tokens, tokens[1:]):
        separator = r"\s+" if left[-1].isalnum() and right[0].isalnum() else r"\s*"
        pattern.extend((separator, re.escape(right)))
    match = re.search("".join(pattern), source_text)
    return match.group(0) if match else None


def _session_preference_evidence_quote(record: dict, sources: dict[str, dict],
                                       new_ids: set[str]) -> dict | None:
    """Recover one direct, request-only clause from a cited learner quote."""
    if (record.get("record_type") != "session_preference_candidate"
            or record.get("topic_id") is not None
            or record.get("claim_status") != "candidate"
            or record.get("retention_class") != "session_only"
            or any(record.get(field) is not None
                   for field in ("interest_operation", "interest", "fact", "support_level",
                                 "support_evidence", "independent_recheck", "recheck_evidence"))):
        return None
    request_marker = re.compile(
        r"(?<!\w)(?:давайте|давай|можно|хочу|хотелось\s+бы|предпочитаю|лучше|удобнее)(?!\w)",
        re.IGNORECASE,
    )
    temporary_prefix = re.compile(
        r"(?:сегодня|сейчас|пока|на\s+этот\s+раз|в\s+этот\s+раз)\s*",
        re.IGNORECASE,
    )
    candidates = []
    for item in record.get("evidence", []):
        if not isinstance(item, dict):
            continue
        turn_id, quote = item.get("turn_id"), item.get("quote")
        source = sources.get(turn_id) if isinstance(turn_id, str) else None
        if (turn_id in new_ids and isinstance(source, dict) and _role(source) == "student"
                and isinstance(quote, str) and nonempty(quote) and len(quote) <= 300
                and quote in str(source.get("text", ""))):
            for sentence in re.finditer(r"[^.!?;\n]+(?:[.!?;]+|$)", quote):
                text = sentence.group(0).strip()
                matches = list(request_marker.finditer(text))
                if len(matches) != 1:
                    continue
                marker = matches[0]
                prefix = text[:marker.start()].strip(" \t,:")
                start = marker.start()
                if prefix and temporary_prefix.fullmatch(prefix):
                    start = text.index(prefix)
                clause = text[start:].strip()
                if len(clause.split()) < 2 or len(clause) > 300:
                    continue
                # A clause must itself be a literal excerpt of this exact cited
                # quote. Returning the source substring keeps strict validation intact.
                offset = quote.find(clause)
                if offset >= 0:
                    candidates.append({"turn_id": turn_id, "quote": quote[offset:offset + len(clause)]})
    return candidates[0] if len(candidates) == 1 else None


def _adjacent_teacher_context_evidence(record: dict, sources: dict[str, dict], new_ids: set[str],
                                       new_source_order: list[str] | tuple) -> list[dict]:
    """Recover the teacher turns around cited learner evidence for an uncertain observation."""
    if (record.get("record_type") != "uncertain_observation"
            or not isinstance(record.get("evidence"), list)):
        return []
    cited = {item.get("turn_id") for item in record["evidence"] if isinstance(item, dict)}
    if any(turn_id in sources and _role(sources[turn_id]) == "teacher" for turn_id in cited):
        return []
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    student_positions = sorted(
        positions[turn_id] for turn_id in cited
        if turn_id in new_ids and turn_id in sources and _role(sources[turn_id]) == "student")
    if not student_positions:
        return []
    candidate_positions: set[int] = set()
    first = student_positions[0]
    if first > 0 and _role(sources.get(new_source_order[first - 1], {})) == "teacher":
        candidate_positions.add(first - 1)
    if len(student_positions) == 1:
        after_last = student_positions[0] + 1
        if (after_last < len(new_source_order)
                and _role(sources.get(new_source_order[after_last], {})) == "teacher"):
            candidate_positions.add(after_last)
    else:
        for left, right in zip(student_positions, student_positions[1:]):
            candidate_positions.update(
                index for index in range(left + 1, right)
                if _role(sources.get(new_source_order[index], {})) == "teacher")
    recovered = []
    for index in sorted(candidate_positions):
        turn_id = new_source_order[index]
        if turn_id in cited:
            continue
        source = sources.get(turn_id, {})
        text = source.get("text")
        if turn_id in new_ids and isinstance(text, str) and nonempty(text):
            recovered.append({"turn_id": turn_id, "quote": text[:2000]})
    if len(record["evidence"]) + len(recovered) > 20:
        return []
    return recovered


def _insert_evidence_in_source_order(record: dict, recovered: list[dict],
                                     new_source_order: list[str] | tuple) -> None:
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    evidence = record["evidence"]
    present = {item.get("turn_id") for item in evidence if isinstance(item, dict)}
    for item in sorted(recovered, key=lambda value: positions.get(value["turn_id"], len(positions))):
        turn_id = item["turn_id"]
        if turn_id in present:
            continue
        position = positions.get(turn_id)
        insert_at = next((index for index, current in enumerate(evidence)
                          if isinstance(current, dict)
                          and positions.get(current.get("turn_id"), -1) > position), len(evidence))
        evidence.insert(insert_at, deepcopy(item))
        present.add(turn_id)


def _validate_evidence(value: object, sources: dict[str, dict], cited: set[str], name: str) -> None:
    if value is None:
        return
    _check(isinstance(value, dict) and set(value) == {"turn_id", "quote"},
           f"{name} needs exactly turn_id and quote", "invalid_condition_evidence")
    turn_id, quote = value["turn_id"], value["quote"]
    _check(isinstance(turn_id, str) and turn_id in sources and turn_id in cited,
           f"{name} source is absent or not cited", "missing_source")
    _check(nonempty(quote) and len(quote) <= 2000 and quote in sources[turn_id]["text"],
           f"{name} quote must be a literal source excerpt", "invalid_quote")
    _check(_role(sources[turn_id]) != "assistant",
           "Assistant output cannot establish learner evidence", "assistant_as_learner_evidence")


def _condition_evidence(value: str | None, evidence: object, sources: dict[str, dict], cited: set[str],
                        name: str) -> None:
    if value in {None, "unknown", "unknown_without_support"}:
        _check(evidence is None, f"Unknown or inapplicable {name} must have null evidence",
               "unknown_condition_has_evidence")
        return
    _validate_evidence(evidence, sources, cited, name)
    _check(isinstance(evidence, dict), f"Known {name} needs separate evidence", "missing_condition_evidence")
    if name == "recheck" and value in {"success", "failure", "one_success_only"}:
        _check(_role(sources[evidence["turn_id"]]) == "student",
               "A learner result must cite a student turn", "assistant_as_learner_evidence")


def _is_learning_type(record_type: str) -> bool:
    return record_type in LEARNING_TYPES


def _is_profile_type(record_type: str) -> bool:
    return record_type in PROFILE_TYPES


_PROFILE_LOW_INFORMATION = frozenset({
    "\u0434\u0430", "\u043d\u0435\u0442", "\u0430\u0433\u0430", "\u0443\u0433\u0443", "\u0445\u0437",
    "\u043e\u043a", "\u043e\u043a\u0435\u0439", "\u043d\u0435", "\u043d\u0443", "\u043b\u0430\u0434\u043d\u043e",
    "\u0445\u043e\u0440\u043e\u0448\u043e", "\u0434\u0443\u043c\u0430\u044e", "\u0441\u0447\u0438\u0442\u0430\u044e",
    "\u043a\u0430\u0436\u0435\u0442\u0441\u044f", "\u043d\u0430\u0432\u0435\u0440\u043d\u043e\u0435", "\u0432\u0440\u043e\u0434\u0435",
    "\u0432\u043e\u0437\u043c\u043e\u0436\u043d\u043e", "\u043c\u043e\u0436\u0435\u0442", "\u043f\u043e\u0436\u0430\u043b\u0443\u0439\u0441\u0442\u0430",
    "\u044f", "\u043c\u043d\u0435", "\u043c\u0435\u043d\u044f", "\u043c\u043e\u0439", "\u043c\u043e\u044f", "\u043c\u043e\u0435", "\u043c\u043e\u0438",
    "\u0438\u0445", "\u044d\u0442\u043e", "\u0438", "\u0430", "\u043d\u043e", "\u0432", "\u043d\u0430", "\u043f\u043e", "\u0437\u0430",
    "\u0438\u0437", "\u043a", "\u0441", "\u043e", "\u0443", "\u0440\u0430\u0437",
})
_RUSSIAN_INFLECTION_ENDINGS = (
    "\u0438\u044f\u043c\u0438", "\u0438\u044f\u0445", "\u0438\u044f\u043c", "\u0430\u043c\u0438", "\u044f\u043c\u0438",
    "\u043e\u0433\u043e", "\u0435\u0433\u043e", "\u043e\u043c\u0443", "\u0435\u043c\u0443", "\u044b\u043c\u0438", "\u0438\u043c\u0438",
    "\u043e\u0439", "\u0435\u0439", "\u043e\u043c", "\u0435\u043c", "\u044b\u043c", "\u0438\u043c", "\u0443\u044e", "\u044e\u044e",
    "\u0430\u044f", "\u044f\u044f", "\u043e\u0435", "\u0435\u0435", "\u044b\u0435", "\u0438\u0435", "\u044b\u0445", "\u0438\u0445",
    "\u0430\u0445", "\u044f\u0445", "\u0430\u043c", "\u044f\u043c", "\u0443", "\u044e", "\u0430", "\u044f", "\u044b", "\u0438",
    "\u0435", "\u043e", "\u044c", "\u0439",
)


def _profile_terms(value: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[^\W\d_]+", value.casefold(), flags=re.UNICODE):
        if token in _PROFILE_LOW_INFORMATION:
            continue
        if re.search(r"[\u0430-\u044f\u0451]", token):
            for ending in _RUSSIAN_INFLECTION_ENDINGS:
                if token.endswith(ending) and len(token) - len(ending) >= 3:
                    token = token[:-len(ending)]
                    break
        terms.add(token)
    return terms


def _profile_value_overlaps_student_quote(value: str, evidence: list[dict], sources: dict[str, dict]) -> bool:
    """Require meaningful overlap with the exact cited learner excerpt."""
    terms = _profile_terms(value)
    quoted_terms = set()
    for item in evidence:
        turn_id = item.get("turn_id")
        if turn_id in sources and _role(sources[turn_id]) == "student":
            quoted_terms.update(_profile_terms(item.get("quote", "")))
    return bool(terms & quoted_terms)


def _validate_link(record: dict, link: dict, previous: dict[str, dict],
                   sources: dict[str, dict], cited: set[str], source_order: list[str] | tuple = ()) -> None:
    relation, target_kind, target_id = link["relation"], link["target_kind"], link["target_id"]
    source_type = record["record_type"]
    if target_kind == "turn":
        _check(target_id in sources and target_id in cited,
               "V2 turn link must target cited source", "invalid_link_target")
        role = _role(sources[target_id])
        if relation == "corrects_answer":
            _check(source_type == "child_correction" and role == "student",
                   "corrects_answer must point from a child correction to a student turn",
                   "incompatible_link")
            positions = {source_id: index for index, source_id in enumerate(source_order)}
            if target_id in positions:
                later_student_evidence = [
                    item["turn_id"] for item in record["evidence"]
                    if item["turn_id"] in positions and item["turn_id"] != target_id
                    and _role(sources[item["turn_id"]]) == "student"
                    and positions[item["turn_id"]] > positions[target_id]
                ]
                _check(bool(later_student_evidence),
                       "corrects_answer must point to an earlier learner turn",
                       "invalid_child_correction")
        elif relation == "supported_by":
            _check(_is_learning_type(source_type) and role == "teacher",
                   "supported_by turn links need a learning record and teacher evidence",
                   "incompatible_link")
        elif relation == "response_to":
            pass
        elif relation == "followed_by":
            _check(_is_learning_type(source_type) and role == "student",
                   "followed_by turn links need a learning record and student evidence",
                   "incompatible_link")
        else:
            _check(False, "This V2 relation cannot target a turn", "incompatible_link")
        return

    _check(target_kind == "record" and target_id in previous,
           "V2 record link must target an existing record", "invalid_link_target")
    target = previous[target_id]
    target_type = target.get("record_type")
    same_topic = record.get("topic_id") == target.get("topic_id")
    if relation == "corrects_answer":
        compatible = (source_type == "child_correction" and target_type == "child_error" and same_topic)
    elif relation == "same_pattern_as":
        compatible = source_type == target_type and same_topic
    elif relation in {"supported_by", "rechecks"}:
        compatible = (_is_learning_type(source_type) and _is_learning_type(target_type) and same_topic)
    elif relation == "mentioned_again":
        compatible = (_is_profile_type(source_type) and source_type == target_type)
    elif relation == "related_to":
        compatible = source_type == target_type and same_topic
    else:
        compatible = False
    _check(compatible, "V2 link relation is incompatible with its record types or topic",
           "incompatible_link")


def validate_records_v2(text: str, sources: dict[str, dict], new_ids: set[str],
                        allowed_topic_ids: set[str], previous_records: list[dict] | tuple = (),
                        source_order: list[str] | tuple = ()) -> list[dict]:
    """Validate V2 records without deciding whether their educational meaning is true."""
    obj = parse_object(text)
    _check(set(obj) == {"records"} and isinstance(obj["records"], list),
           "Invalid V2 records envelope", "invalid_envelope")
    _check(len(obj["records"]) <= 100, "Too many V2 records in one update", "too_many_records")
    previous = {record["record_id"]: record for record in previous_records}
    seen: set[str] = set()
    validated: list[dict] = []

    for record in obj["records"]:
        _check(isinstance(record, dict) and set(record) == set(EXTRACT_FIELDS),
               "Invalid V2 record fields", "invalid_record_fields")
        fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        _check(fingerprint not in seen, "Completion contains a duplicate V2 memory record", "duplicate_record")
        seen.add(fingerprint)

        kind = record["record_type"]
        topic_id = record["topic_id"]
        _check(kind in RECORD_TYPES, "Unknown V2 record type", "invalid_record_type")
        _check(topic_id is None or topic_id in allowed_topic_ids, "V2 record topic is not allowed", "invalid_record_topic")
        for field in ("context", "interpretation"):
            _check(nonempty(record[field]) and len(record[field]) <= 2000,
                   f"Invalid {field}", "invalid_record_text")
        _check(record["interpretation_status"] in INTERPRETATION_STATUS,
               "Invalid interpretation status", "invalid_interpretation_status")
        _check(record["support_level"] in SUPPORT and record["independent_recheck"] in RECHECK,
               "Invalid support or recheck value", "invalid_condition")

        evidence = record["evidence"]
        _check(isinstance(evidence, list) and 1 <= len(evidence) <= 20,
               "V2 record needs one to twenty evidence excerpts", "invalid_evidence")
        cited: set[str] = set()
        for item in evidence:
            _check(isinstance(item, dict) and set(item) == {"turn_id", "quote"},
                   "Evidence needs exactly turn_id and quote", "invalid_evidence")
            turn_id, quote = item["turn_id"], item["quote"]
            _check(isinstance(turn_id, str) and turn_id in sources,
                   "Evidence refers to a source absent from memory", "missing_source")
            _check(nonempty(quote) and len(quote) <= 2000 and quote in sources[turn_id]["text"],
                   "Evidence quote must be a literal source excerpt", "invalid_quote")
            cited.add(turn_id)
        _check(len(cited) == len(evidence), "Repeated evidence turn IDs are not allowed", "duplicate_citation")
        _check(bool(cited & new_ids), "V2 record needs evidence from the new lesson", "missing_new_source")

        # Separate condition evidence is authoritative too. If the model omitted
        # that same turn from the general citation list, add the already-validated
        # literal excerpt there so downstream citation checks cannot lose it.
        for field in ("support_evidence", "recheck_evidence"):
            condition = record[field]
            if isinstance(condition, dict) and condition.get("turn_id") in sources \
                    and condition["turn_id"] not in cited:
                evidence.append(deepcopy(condition))
                cited.add(condition["turn_id"])
        _check(len(evidence) <= 20, "Too many V2 evidence excerpts", "invalid_evidence")
        _condition_evidence(record["support_level"], record["support_evidence"], sources, cited, "support")
        _condition_evidence(record["independent_recheck"], record["recheck_evidence"], sources, cited, "recheck")

        links = record["links"]
        _check(isinstance(links, list) and len(links) <= 20, "Invalid V2 links", "invalid_links")
        for link in links:
            _check(isinstance(link, dict) and set(link) == {"relation", "target_kind", "target_id"},
                   "Invalid V2 link fields", "invalid_links")
            _check(link["relation"] in LINK_RELATIONS, "Unknown V2 link relation", "invalid_link_relation")
            _validate_link(record, link, previous, sources, cited, source_order or tuple(sources))

        target = record["corrects_record_id"]
        if kind == "source_correction":
            _check(isinstance(target, str) and target in previous,
                   "Source correction needs an existing record ID", "invalid_correction_target")
            _check(previous[target].get("topic_id") == topic_id,
                   "Source correction cannot target another topic", "invalid_correction_target")
            _check(any(_role(sources[x]) == "teacher" for x in cited),
                   "Memory-source correction needs teacher evidence", "invalid_correction_evidence")
            _check(record["interpretation_status"] == "explicit",
                   "Memory-source correction must be explicit", "invalid_interpretation_status")
            _check(record["support_level"] is None and record["support_evidence"] is None
                   and record["independent_recheck"] is None and record["recheck_evidence"] is None,
                   "Memory-source correction cannot describe learner support or recheck",
                   "mixed_record_fields")
        else:
            _check(target is None, "Only source_correction can set corrects_record_id", "invalid_correction_kind")

        for field in ("interest_operation", "interest", "preference", "fact", "claim_status", "retention_class"):
            value = record[field]
            _check(value is None or isinstance(value, str) and nonempty(value),
                   f"Invalid {field}", "invalid_profile_value")
        _check(record["interest_operation"] is None or record["interest_operation"] in INTEREST_OPERATIONS,
               "Invalid interest operation", "invalid_profile_value")
        _check(record["claim_status"] in {None, "candidate", "explicit"},
               "Invalid claim status", "invalid_profile_value")
        _check(record["retention_class"] in {None, *RETENTION_CLASSES},
               "Invalid retention class", "invalid_profile_value")
        for field, limit in (("interest", 300), ("preference", 300), ("fact", 500)):
            value = record[field]
            _check(value is None or len(value) <= limit,
                   f"{field} exceeds its schema limit", "invalid_profile_value")

        student_cited = any(_role(sources[x]) == "student" for x in cited)
        if kind in LEARNING_TYPES:
            _check(topic_id is not None, "Learning memory needs a topic", "missing_record_topic")
            _check(all(record[field] is None for field in (
                "interest_operation", "interest", "preference", "fact", "claim_status", "retention_class")),
                "Learning memory cannot contain profile fields", "mixed_record_fields")
            if kind != "source_correction":
                _check(student_cited, "Learner event needs student-source evidence", "missing_learner_evidence")
        elif kind == "interest_candidate":
            _check(topic_id is None and record["interest"] is not None
                   and record["interest_operation"] in INTEREST_OPERATIONS
                   and record["claim_status"] == "candidate"
                   and record["retention_class"] == "profile_candidate"
                   and record["preference"] is None and record["fact"] is None,
                   "Interest candidate fields are incomplete or mixed", "mixed_record_fields")
            _check(record["interpretation_status"] == "candidate" and student_cited,
                   "Interest candidates need a student statement and candidate status", "invalid_interest_candidate")
            _check(record["support_level"] is None and record["independent_recheck"] is None
                   and record["support_evidence"] is None and record["recheck_evidence"] is None,
                   "Interest cannot establish learning support or skill", "mixed_record_fields")
        elif kind == "session_preference_candidate":
            _check(topic_id is None and record["claim_status"] == "candidate"
                   and record["retention_class"] == "session_only"
                   and record["interest_operation"] is None and record["interest"] is None and record["fact"] is None,
                   "Session preference fields are incomplete or mixed", "mixed_record_fields")
            _check(record["preference"] is not None,
                   "Session preference candidate is missing its preference value", "missing_preference_field")
            _check(record["interpretation_status"] == "candidate" and student_cited,
                   "Session preference needs a student statement and candidate status", "invalid_preference_candidate")
            _check(record["support_level"] is None and record["independent_recheck"] is None
                   and record["support_evidence"] is None and record["recheck_evidence"] is None,
                   "Preference cannot establish learning support or skill", "mixed_record_fields")
        elif kind == "incidental_fact":
            _check(topic_id is None and record["fact"] is not None
                   and record["claim_status"] == "explicit"
                   and record["retention_class"] == "episodic_personal_fact"
                   and record["interest_operation"] is None and record["interest"] is None and record["preference"] is None,
                   "Incidental fact fields are incomplete or mixed", "mixed_record_fields")
            _check(record["interpretation_status"] == "explicit" and student_cited,
                   "Incidental facts need a student statement", "invalid_incidental_fact")
            _check(record["support_level"] is None and record["independent_recheck"] is None
                   and record["support_evidence"] is None and record["recheck_evidence"] is None,
                   "Incidental facts cannot establish learning support or skill", "mixed_record_fields")

        if kind == "uncertain_observation":
            roles = {_role(sources[x]) for x in cited}
            support_unknown = (record["support_level"] in {None, "unknown"}
                               and record["support_evidence"] is None)
            support_known = (record["support_level"] in {"hint_question", "visual", "joint"}
                             and isinstance(record["support_evidence"], dict))
            _check(record["interpretation_status"] == "ambiguous"
                   and record["independent_recheck"] in {None, "unknown"}
                   and (support_unknown or support_known)
                   and record["recheck_evidence"] is None
                   and "student" in roles and "teacher" in roles,
                   "Uncertain observations need teacher and learner evidence with unknown outcome conditions",
                   "invalid_uncertain_observation")
        if kind in {"interest_candidate", "session_preference_candidate", "incidental_fact"}:
            field = {"interest_candidate": "interest", "session_preference_candidate": "preference",
                     "incidental_fact": "fact"}[kind]
            _check(_profile_value_overlaps_student_quote(record[field], evidence, sources),
                   "Profile value must be directly represented in a student source quote",
                   "invalid_profile_evidence")
        elif kind == "supported_attempt":
            _check(record["interpretation_status"] == "explicit_with_assistance"
                   and record["support_level"] in {"hint_question", "visual", "joint"}
                   and isinstance(record["support_evidence"], dict),
                   "Supported attempt needs a named, evidenced form of help", "invalid_supported_attempt")
        elif kind == "child_correction":
            _check(record["interpretation_status"] in {
                       "explicit", "explicit_with_assistance", "single_independent_observation"}
                   and any(link["relation"] == "corrects_answer" for link in links),
                   "Child correction needs an explicit link to the earlier answer", "invalid_child_correction")
            if record["interpretation_status"] == "explicit":
                _check(record["support_level"] in {None, "unknown"}
                       and record["support_evidence"] is None,
                       "Neutral child correction status requires unknown support", "invalid_child_correction")
            # Older stored observations can pair single_independent_observation
            # with unknown support. Keep strict reads compatible; normalize only
            # new extraction candidates in validate_records_v2_partial.
            for link in links:
                if link["relation"] != "corrects_answer":
                    continue
                if link["target_kind"] == "record":
                    target_record = previous[link["target_id"]]
                    _check(target_record.get("record_type") == "child_error"
                           and target_record.get("topic_id") == topic_id,
                           "Child correction must link to an earlier error on the same topic",
                           "invalid_child_correction")
                else:
                    _check(_role(sources[link["target_id"]]) == "student",
                           "Child correction must link to a learner answer", "invalid_child_correction")
        elif kind == "child_error":
            _check(record["interpretation_status"] in {
                "explicit", "explicit_with_assistance", "single_independent_observation"},
                   "Use uncertain_observation when the answer is ambiguous", "invalid_child_error")
        elif kind == "learning_success":
            _check(record["interpretation_status"] in {
                "explicit", "explicit_with_assistance", "single_independent_observation"},
                "Learning success needs a concrete observed outcome", "invalid_learning_success")

        validated.append(record)
    return validated


def _adjacent_task_prompt_evidence(record: dict, sources: dict[str, dict], new_ids: set[str],
                                   new_source_order: list[str] | tuple) -> dict | None:
    """Recover only the immediately preceding teacher question or task prompt for a short reply."""
    if any(item.get("turn_id") in sources and _role(sources[item["turn_id"]]) == "teacher"
           for item in record.get("evidence", []) if isinstance(item, dict)):
        return None
    position = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    student_ids = [
        item.get("turn_id") for item in record.get("evidence", [])
        if isinstance(item, dict) and item.get("turn_id") in new_ids
        and item.get("turn_id") in sources and _role(sources[item["turn_id"]]) == "student"
    ]
    task_markers = (
        "реши", "решите", "вычисли", "запиши", "проверь", "проверьте", "разложи",
        "покажи", "покажите", "объясни", "объясните", "определи", "сравни", "назови",
        "посчитай", "найди", "отметь", "укажи", "продолжи", "выполни", "ответь",
        "что ", "как ", "почему ", "где ", "сколько ", "какая ", "какой ", "какое ",
        "можно ли ",
    )
    for student_id in sorted(student_ids, key=lambda turn_id: position.get(turn_id, -1)):
        index = position.get(student_id)
        if index is None or index == 0:
            continue
        candidate_id = new_source_order[index - 1]
        source = sources.get(candidate_id, {})
        text = source.get("text", "")
        lowered = text.casefold()
        is_question = any(mark in text for mark in ("?", "\uff1f"))
        is_task_prompt = any(marker in lowered for marker in task_markers)
        if _role(source) == "teacher" and (is_question or is_task_prompt):
            return {"turn_id": candidate_id, "quote": text}
    return None


def _immediate_teacher_context_evidence(record: dict, sources: dict[str, dict], new_ids: set[str],
                                        new_source_order: list[str] | tuple) -> list[dict]:
    """Keep only the teacher turn immediately preceding each cited learning-success answer."""
    if record.get("record_type") != "learning_success" or not isinstance(record.get("evidence"), list):
        return []
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    recovered = []
    for item in record["evidence"]:
        if not isinstance(item, dict):
            continue
        student_id, quote = item.get("turn_id"), item.get("quote")
        index = positions.get(student_id)
        student = sources.get(student_id, {}) if isinstance(student_id, str) else {}
        if (index is None or index == 0 or student_id not in new_ids
                or _role(student) != "student" or not isinstance(quote, str)
                or quote not in str(student.get("text", ""))):
            continue
        teacher_id = new_source_order[index - 1]
        teacher = sources.get(teacher_id, {})
        teacher_text = teacher.get("text")
        if teacher_id in new_ids and _role(teacher) == "teacher" and isinstance(teacher_text, str):
            recovered.append({"turn_id": teacher_id, "quote": teacher_text})
    return recovered


def _explicit_no_followup_pair(record: dict, sources: dict[str, dict], new_ids: set[str],
                               new_source_order: list[str] | tuple) -> tuple[dict, dict] | None:
    """Find a directly cited teacher question and unambiguous negative answer about later practice."""
    if record.get("record_type") != "uncertain_observation" or not new_source_order:
        return None
    order = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    cited = {
        item.get("turn_id") for item in record.get("evidence", [])
        if isinstance(item, dict) and isinstance(item.get("turn_id"), str)
        and item.get("turn_id") in new_ids
    }
    practice_subject_markers = ("задач", "пример", "упражнен", "карточк")
    direct_occurrence_question_patterns = (
        r"^(?:после\b.{0,80}\b)?(?:ещё|еще|снова)\s+(?:встречал\w*|попадал\w*|попадавш\w*|появлял\w*|случал\w*)\b",
        r"^(?:(?:ты|вы)\s+)?(?:встречал\w*|попадал\w*|попадавш\w*|появлял\w*|случал\w*)\b",
        r"^потом\s+(?:встречал\w*|попадал\w*|попадавш\w*|появлял\w*|случал\w*)\b",
    )
    related_detail_markers = (
        "похож", "подобн", "друг", "где", "в котором", "в которой", "такой", "такая", "такое",
        "с пустым", "с нулевым",
    )
    prior_practice_markers = ("ещё", "еще", "снова", "после той работы", "после этой работы", "после этого")
    property_markers = ("трудн", "сложн", "легк", "правильн", "неправильн", "решил", "получил", "ошибк",
                        "понрав", "интерес", "люб")
    same_work_markers = ("этой строк", "эту строк", "к этой строк", "эту же задач", "к этой задаче",
                         "после урока", "дома", "возвращал", "продолжал", "так и оставил")
    autonomy_markers = ("без подсказки", "без помощи", "самостоятельно", "сам решил", "сама решила")
    uncertainty_markers = ("не помню", "не знаю", "не уверен", "не уверена",
                           "думаю", "наверное", "может быть", "кажется")
    explicit_absence_patterns = (
        r"\bдальше\b.{0,120}\bбыли\b.{0,100}\b(?:примеры|задач\w*|упражнен\w*)\b.{0,100}\bбез\b.{0,80}\b(?:такого места|такого задания|такой задачи|этой особенности)\b",
        r"\b(?:похож\w*|друг\w*)\b.{0,80}\b(?:не было|не встречал\w*|не решал\w*)\b",
        r"\bне (?:встречал\w*|решал\w*)\b.{0,80}\b(?:похож\w*|друг\w*|задач\w*|пример\w*|упражнен\w*)\b",
        r"\bне было\b.{0,80}\b(?:похож\w*|друг\w*|задач\w*|пример\w*|упражнен\w*)\b",
    )
    for student_id in sorted(cited, key=lambda turn_id: order.get(turn_id, -1)):
        student = sources.get(student_id, {})
        student_text = student.get("text", "")
        if _role(student) != "student" or not isinstance(student_text, str):
            continue
        answer = " ".join(student_text.casefold().split())
        if not re.match(r"^нет(?:\b|[,!.?…])", answer):
            continue
        if any(marker in answer for marker in uncertainty_markers):
            continue
        exact_no = re.fullmatch(r"нет[.!?…]*", answer) is not None
        if not exact_no and not any(re.search(pattern, answer) for pattern in explicit_absence_patterns):
            continue
        preceding_questions = []
        for teacher_id in cited:
            teacher = sources.get(teacher_id, {})
            teacher_text = teacher.get("text", "")
            teacher_lower = " ".join(teacher_text.casefold().split()) if isinstance(teacher_text, str) else ""
            is_direct_pair = order.get(teacher_id, len(order)) + 1 == order.get(student_id, -1)
            asks_about_practice = (
                any(marker in teacher_lower for marker in practice_subject_markers)
                and any(re.search(pattern, teacher_lower) for pattern in direct_occurrence_question_patterns)
                and any(marker in teacher_lower for marker in related_detail_markers)
                and (any(marker in teacher_lower for marker in prior_practice_markers)
                     or re.match(r"^(?:(?:ты|вы)\s+)?(?:встречал|встречала|встречались)", teacher_lower))
                and not any(marker in teacher_lower for marker in property_markers)
                and not any(marker in teacher_lower for marker in same_work_markers)
                and not any(marker in teacher_lower for marker in autonomy_markers)
            )
            if (_role(teacher) == "teacher" and is_direct_pair
                    and ("?" in teacher_text or "？" in teacher_text)
                    and asks_about_practice):
                preceding_questions.append(teacher_id)
        if not preceding_questions:
            continue
        teacher_id = max(preceding_questions, key=lambda turn_id: order.get(turn_id, -1))
        return (
            {"turn_id": teacher_id, "quote": sources[teacher_id]["text"]},
            {"turn_id": student_id, "quote": student_text},
        )
    return None


def _has_explicit_unfinished_cue(text: str) -> bool:
    lowered = " ".join(text.casefold().split())
    completion_verbs = r"\b(?:закончил\w*|завершил\w*)\b"
    for intro in ("потом", "позже", "в итоге", "но"):
        for match in re.finditer(rf"\b{intro}\b", lowered):
            tail = re.split(r"[.!?;]", lowered[match.end():], maxsplit=1)[0][:140]
            completion = re.search(completion_verbs, tail)
            partial_action = re.search(
                r"\b(?:закончил\w*|завершил\w*|решил\w*|доделал\w*|выполнил\w*)\s+"
                r"(?:(?:только|лишь)\s+)?(?:перв\w*|следующ\w*|промежуточн\w*)\s+"
                r"(?:шаг\w*|этап\w*|част\w*)\b",
                tail,
            )
            if completion and not partial_action and not re.search(r"\bне\b", tail[:completion.start()]):
                return False
            task_completion = re.search(
                r"\b(?:решил\w*|доделал\w*|выполнил\w*)\b.{0,60}\b(?:задач\w*|пример\w*|упражнен\w*|работ\w*|решени\w*)\b",
                tail,
            )
            if task_completion and not partial_action and not re.search(r"\bне\b", tail[:task_completion.start()]):
                return False
    explicitly_missing_result = re.search(
        r"\b(?:ответ|итог|решение)\b.{0,40}\bне\s+записал\w*\b"
        r"|\bне\s+записал\w*\b.{0,40}\b(?:ответ|итог|решение)\b",
        lowered,
    )
    if explicitly_missing_result:
        return True
    if any(marker in lowered for marker in (
        "не закончил", "не закончила", "не завершил", "не завершила",
        "не дописал", "не дописала",
        "дальше в тетради пусто", "в тетради пусто", "дальше пусто",
        "тетрадь пустая",
    )):
        return True
    return re.search(
        r"\bне успел\w*\b.{0,60}\b(?:закончить|завершить|дописать|записать|продолжить|доделать)\b"
        r"|\bне продолжил\w*\b.{0,50}\b(?:работ\w*|задач\w*|решени\w*|пример\w*|выполнени\w*)\b",
        lowered,
    ) is not None


def _unfinished_interpretation(stop_text: str) -> str:
    lowered = " ".join(stop_text.casefold().split())
    if re.search(r"\b(?:ответ|результат)\b.{0,40}\b\d", lowered):
        return "Ученик сообщает, что работа не завершена, и называет ответ или результат; по этим репликам неясно, является ли он окончательным."
    blank_markers = ("дальше в тетради пусто", "в тетради пусто", "дальше пусто", "тетрадь пустая")
    if any(marker in lowered for marker in blank_markers):
        return "Ученик сообщает, что после указанного шага в тетради пусто; итоговый ответ в этой реплике не приведён."
    if re.search(r"\b(?:ответ|итог|решение)\b.{0,40}\bне\s+записал\w*\b"
                 r"|\bне\s+записал\w*\b.{0,40}\b(?:ответ|итог|решение)\b",
                 lowered):
        return "Ученик сообщает, что остановился до записи ответа; итоговый ответ в этой реплике не приведён."
    if any(marker in lowered for marker in ("не дописал", "не дописала")):
        return "Ученик сообщает, что продолжение после указанного шага не дописано; итоговый ответ в этой реплике не приведён."
    if any(marker in lowered for marker in ("не успел", "не успела", "остановился", "остановилась", "не продолжил", "не продолжила")):
        return "Ученик сообщает, что не продолжил работу после указанного шага; итоговый ответ в этой реплике не приведён."
    return "Ученик сообщает, что не закончил работу после указанного шага; итоговый ответ в этой реплике не приведён."


def _unfinished_record_from_direct_stop(success: dict, sources: dict[str, dict], new_ids: set[str],
                                        new_source_order: list[str] | tuple) -> tuple[dict, list[str]] | None:
    """Recover an omitted unfinished state only from an adjacent continuation prompt and direct stop report."""
    topic_id = success.get("topic_id")
    evidence = success.get("evidence")
    if (success.get("record_type") != "learning_success" or not isinstance(topic_id, str)
            or not isinstance(evidence, list) or not new_source_order):
        return None
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    continuation_markers = ("какой шаг", "какую цифру", "какой разряд", "следующий шаг",
                            "следующую цифру", "следующий разряд", "что записать", "что записывается",
                            "продолж", "дальше в тетрад", "дальше в примере", "дальше в задаче")
    new_task_markers = ("новый пример", "новую задачу", "следующий пример", "следующую задачу",
                        "другой пример", "другая задача", "теперь реши", "а теперь")
    student_ids = [
        item.get("turn_id") for item in evidence
        if isinstance(item, dict) and item.get("turn_id") in new_ids
        and item.get("turn_id") in positions and item.get("turn_id") in sources
        and _role(sources[item["turn_id"]]) == "student"
    ]
    for student_id in sorted(student_ids, key=lambda turn_id: positions[turn_id]):
        next_position = positions[student_id] + 1
        if next_position + 1 >= len(new_source_order):
            continue
        prompt_id, stop_id = new_source_order[next_position:next_position + 2]
        prompt, stop = sources.get(prompt_id, {}), sources.get(stop_id, {})
        prompt_text, stop_text = prompt.get("text", ""), stop.get("text", "")
        if (prompt_id not in new_ids or stop_id not in new_ids
                or _role(prompt) != "teacher" or _role(stop) != "student"
                or not isinstance(prompt_text, str) or not isinstance(stop_text, str)
                or not any(marker in prompt_text.casefold() for marker in continuation_markers)
                or any(marker in prompt_text.casefold() for marker in new_task_markers)
                or not _has_explicit_unfinished_cue(stop_text)):
            continue
        related_evidence = [
            deepcopy(item) for item in evidence
            if isinstance(item, dict) and item.get("turn_id") in new_ids
            and item.get("turn_id") in sources and isinstance(item.get("quote"), str)
            and item["quote"] in str(sources[item["turn_id"]].get("text", ""))
        ]
        related_evidence.extend([
            {"turn_id": prompt_id, "quote": prompt_text},
            {"turn_id": stop_id, "quote": stop_text},
        ])
        recovered = {
            "record_type": "unfinished_work",
            "topic_id": topic_id,
            "context": "Учебная попытка осталась незавершённой после промежуточного шага.",
            "interpretation": _unfinished_interpretation(stop_text),
            "interpretation_status": "explicit",
            "support_level": "unknown",
            "support_evidence": None,
            "independent_recheck": "unknown",
            "recheck_evidence": None,
            "evidence": related_evidence,
            "links": [],
            "interest_operation": None,
            "interest": None,
            "preference": None,
            "fact": None,
            "claim_status": None,
            "retention_class": None,
            "corrects_record_id": None,
        }
        # The evidence list is built from the success plus an adjacent teacher/student pair.
        recovered["evidence"] = sorted(
            {item["turn_id"]: item for item in related_evidence}.values(),
            key=lambda item: positions.get(item["turn_id"], len(positions)))
        return recovered, [prompt_id, stop_id]
    return None


def _adjacent_visual_support_evidence(record: dict, sources: dict[str, dict], new_ids: set[str],
                                     new_source_order: list[str] | tuple) -> dict | None:
    """Recover only direct visual-material evidence before a learner response."""
    if not new_source_order or not isinstance(record.get("evidence"), list):
        return None
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    student_ids = [
        item.get("turn_id") for item in record["evidence"]
        if isinstance(item, dict) and item.get("turn_id") in new_ids
        and item.get("turn_id") in sources and _role(sources[item["turn_id"]]) == "student"
    ]
    visual_markers = ("рисунок", "рисунке", "рисунка", "картин", "полоск", "схем", "диаграмм", "наглядн")
    negative_markers = ("без рисунка", "без изображения", "не было рисунка", "не показывал")
    for student_id in sorted(student_ids, key=lambda turn_id: positions.get(turn_id, -1)):
        index = positions.get(student_id)
        if index is None or index == 0:
            continue
        candidate_id = new_source_order[index - 1]
        source = sources.get(candidate_id, {})
        text = source.get("text", "")
        lowered = text.casefold()
        if (_role(source) == "teacher" and not any(marker in lowered for marker in negative_markers)
                and any(marker in lowered for marker in visual_markers)):
            return {"turn_id": candidate_id, "quote": text[:2000]}
    return None


def _adjacent_prompted_recheck_evidence(record: dict, sources: dict[str, dict], new_ids: set[str],
                                        new_source_order: list[str] | tuple) -> list[dict]:
    """Recover a teacher-requested check without claiming it was independent."""
    if (record.get("record_type") != "child_correction"
            or record.get("independent_recheck") not in (None, "unknown", "unknown_without_support")
            or not isinstance(record.get("evidence"), list)
            or not isinstance(record.get("links"), list)
            or not new_source_order):
        return []
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    target_ids = [link.get("target_id") for link in record.get("links", [])
                  if isinstance(link, dict) and link.get("relation") == "corrects_answer"
                  and link.get("target_kind") == "turn" and isinstance(link.get("target_id"), str)
                  and link.get("target_id") in positions
                  and link.get("target_id") in sources and _role(sources[link["target_id"]]) == "student"]
    target_position = max((positions[turn_id] for turn_id in target_ids), default=-1)
    correction_positions = [
        positions[item["turn_id"]] for item in record["evidence"]
        if isinstance(item, dict) and isinstance(item.get("turn_id"), str)
        and item.get("turn_id") in positions
        and item.get("turn_id") in new_ids and item.get("turn_id") in sources
        and _role(sources[item["turn_id"]]) == "student"
        and positions[item["turn_id"]] > target_position
    ]
    if not correction_positions:
        return []
    correction_position = min(correction_positions)
    check_directives = ("проверь", "проверьте", "проверить", "проверим", "сделай проверку", "выполни проверку")
    check_cues = ("обратн", "умнож", "делен", "подстанов", "ответ", "решени", "вычислен", "результат")
    new_task_markers = ("новый пример", "новую задачу", "следующий пример", "следующую задачу",
                        "другой пример", "другая задача", "теперь реши", "а теперь")

    def is_recheck_prompt(text: str) -> bool:
        lowered = text.casefold()
        return (any(marker in lowered for marker in check_directives)
                and any(marker in lowered for marker in check_cues)
                and not any(marker in lowered for marker in new_task_markers))

    acknowledgement_markers = ("вопрос помог", "подсказка помог", "рисунок помог",
                                "после вопроса", "после подсказки")
    prompt_id = None
    for turn_id in new_source_order[correction_position + 1:]:
        if turn_id not in new_ids or turn_id not in sources:
            return []
        source = sources[turn_id]
        if _role(source) == "student":
            text = str(source.get("text", "")).casefold()
            if any(marker in text for marker in acknowledgement_markers):
                continue
            return []
        if _role(source) == "teacher":
            if is_recheck_prompt(str(source.get("text", ""))):
                prompt_id = turn_id
            break
        return []
    if prompt_id is None:
        return []
    next_position = positions[prompt_id] + 1
    answer_position = next_position
    if answer_position >= len(new_source_order):
        return []
    answer_id = new_source_order[answer_position]
    answer = sources.get(answer_id, {})
    answer_text = answer.get("text")
    prompt_text = sources[prompt_id].get("text")
    if (answer_id in new_ids and _role(answer) == "student"
            and isinstance(prompt_text, str) and nonempty(prompt_text)
            and isinstance(answer_text, str) and nonempty(answer_text)):
        return [
            {"turn_id": prompt_id, "quote": prompt_text[:2000]},
            {"turn_id": answer_id, "quote": answer_text[:2000]},
        ]
    return []


def _reported_old_answer(text: str) -> bool:
    lowered = text.casefold()
    old_markers = ("старой строк", "старой запис", "старом ответ", "предыдущей запис",
                   "предыдущем ответ", "прошлой запис", "прошлом ответ")
    return any(marker in lowered for marker in old_markers) and any(
        marker in lowered for marker in ("ответ", "запис", "строк", "остаток", "стоит"))


def _reported_correction(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in ("исправ", "перепровер", "пересчитал", "пересчитала"))


def _reported_correction_sequence(record: dict, sources: dict[str, dict], new_ids: set[str],
                                  new_source_order: list[str] | tuple) -> list[dict]:
    """Keep intervening turns when a learner reports changing a teacher-reported old answer."""
    if record.get("record_type") != "uncertain_observation" or not isinstance(record.get("evidence"), list):
        return []
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    cited = {item.get("turn_id") for item in record["evidence"] if isinstance(item, dict)}
    reports = [turn_id for turn_id in new_source_order
               if turn_id in new_ids and turn_id in cited and turn_id in sources
               and _role(sources[turn_id]) == "teacher"
               and _reported_old_answer(str(sources[turn_id].get("text", "")))]
    corrections = [turn_id for turn_id in new_source_order
                   if turn_id in new_ids and turn_id in cited and turn_id in sources
                   and _role(sources[turn_id]) == "student"
                   and _reported_correction(str(sources[turn_id].get("text", "")))]
    pair = next(((report, correction) for report in reports for correction in corrections
                 if positions[report] < positions[correction]), None)
    if pair is None:
        return []
    start, end = positions[pair[0]], positions[pair[1]]
    return [{"turn_id": turn_id, "quote": str(sources[turn_id].get("text", ""))[:2000]}
            for turn_id in new_source_order[start:end + 1]
            if turn_id in new_ids and turn_id in sources
            and _role(sources[turn_id]) in {"teacher", "student"}
            and isinstance(sources[turn_id].get("text"), str)
            and nonempty(sources[turn_id]["text"])]


def _is_reported_hint_quote(source: dict, quote: str) -> bool:
    lowered = quote.casefold()
    if _role(source) == "student":
        return any(marker in lowered for marker in (
            "учитель спросил", "учитель спросила", "учитель спрашивал", "учитель спрашивала",
            "учитель попросил", "учитель попросила", "учитель задал вопрос", "после вопроса учителя"))
    if _role(source) != "teacher":
        return False
    if "похожую задачу" in lowered and "без вопроса" in lowered:
        return False
    if "после вопроса" in lowered and any(marker in lowered for marker in ("что сделал", "что сделала")):
        return False
    hint_markers = ("попробуй", "попробуйте", "проверь", "проверьте", "посмотри", "посмотрите",
                    "вспомни", "вспомните", "сколько", "почему", "какая дробь", "какое действие")
    return "?" in quote or "？" in quote or any(marker in lowered for marker in hint_markers)


def _reported_hint_evidence(record: dict, sources: dict[str, dict], new_ids: set[str],
                            new_source_order: list[str] | tuple) -> dict | None:
    """Find a cited question reported to precede a learner's reported correction."""
    if record.get("record_type") != "uncertain_observation" or not isinstance(record.get("evidence"), list):
        return None
    positions = {turn_id: index for index, turn_id in enumerate(new_source_order)}
    cited_ids = {item.get("turn_id") for item in record["evidence"] if isinstance(item, dict)}
    has_teacher_report = any(turn_id in sources and _role(sources[turn_id]) == "teacher"
                             and _reported_old_answer(str(sources[turn_id].get("text", "")))
                             for turn_id in cited_ids)
    correction_positions = [positions[turn_id] for turn_id in cited_ids
                            if turn_id in positions and turn_id in sources
                            and _role(sources[turn_id]) == "student"
                            and _reported_correction(str(sources[turn_id].get("text", "")))]
    if not has_teacher_report or not correction_positions:
        return None
    candidates = []
    for item in record.get("evidence", []):
        if not isinstance(item, dict):
            continue
        turn_id, quote = item.get("turn_id"), item.get("quote")
        source = sources.get(turn_id) if isinstance(turn_id, str) else None
        if (turn_id in new_ids and isinstance(source, dict) and isinstance(quote, str)
                and quote in str(source.get("text", "")) and _is_reported_hint_quote(source, quote)):
            position = positions.get(turn_id, -1)
            if not any(position < correction_position for correction_position in correction_positions):
                continue
            rank = 0 if _role(source) == "student" else 1
            candidates.append((rank, position, item))
    if not candidates:
        return None
    return deepcopy(min(candidates, key=lambda item: (item[0], item[1]))[2])


def _mark_reported_answer_authorship_unknown(record: dict, sources: dict[str, dict]) -> bool:
    """Use neutral wording when only the teacher identifies the old answer."""
    if record.get("record_type") != "uncertain_observation" or not isinstance(record.get("evidence"), list):
        return False
    cited = [item.get("turn_id") for item in record["evidence"] if isinstance(item, dict)]
    report_ids = [turn_id for turn_id in cited if turn_id in sources
                  and _role(sources[turn_id]) == "teacher"
                  and _reported_old_answer(str(sources[turn_id].get("text", "")))]
    correction_ids = [turn_id for turn_id in cited if turn_id in sources
                      and _role(sources[turn_id]) == "student"
                      and _reported_correction(str(sources[turn_id].get("text", "")))]
    if not report_ids or not correction_ids:
        return False
    student_text = " ".join(str(sources[turn_id].get("text", "")).casefold()
                            for turn_id in cited if turn_id in sources and _role(sources[turn_id]) == "student")
    explicit_authorship = ("я записал" in student_text or "я записала" in student_text
                           or "я написал" in student_text or "я написала" in student_text
                           or "моя запись" in student_text or "мой ответ" in student_text)
    context = str(record.get("context", "")).casefold()
    already_uncertain = any(marker in context for marker in ("автор", "кто запис", "не подтвержд", "неизвест"))
    if explicit_authorship or already_uncertain:
        return False
    record["context"] = "Ребёнок сообщает об изменении прежнего ответа после вопроса; авторство исходной записи не подтверждено."
    return True


def validate_records_v2_partial(text: str, sources: dict[str, dict], new_ids: set[str],
                allowed_topic_ids: set[str],
                previous_records: list[dict] | tuple = (),
                new_source_order: list[str] | tuple = ()) -> tuple[list[dict], list[dict]]:
    """Validate candidates independently so one bad record cannot erase good ones.

    Envelope, parse and size failures remain fatal. A candidate rejected by the
    strict validator is omitted with a stable code and its one-based position.
    """
    obj = parse_object(text)
    _check(set(obj) == {"records"} and isinstance(obj["records"], list),
           "Invalid V2 records envelope", "invalid_envelope")
    _check(len(obj["records"]) <= 100, "Too many V2 records in one update", "too_many_records")
    accepted: list[tuple[int, dict]] = []
    warnings: list[dict] = []
    seen: set[str] = set()
    for index, record in enumerate(obj["records"], start=1):
        try:
            if isinstance(record, dict):
                record = deepcopy(record)
                for field in ("evidence", "support_evidence", "recheck_evidence"):
                    value = record.get(field)
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        turn_id, quote = item.get("turn_id"), item.get("quote")
                        source = sources.get(turn_id) if isinstance(turn_id, str) else None
                        source_text = source.get("text") if isinstance(source, dict) else None
                        if isinstance(quote, str) and isinstance(source_text, str) and quote not in source_text:
                            exact_quote = _whitespace_equivalent_source_excerpt(quote, source_text)
                            if exact_quote is not None:
                                item["quote"] = exact_quote
                                warnings.append({"event": "record_normalized", "record_index": index,
                                                 "field": f"{field}.quote",
                                                 "code": "quote_whitespace_normalized",
                                                 "turn_id": turn_id})
                if record.get("record_type") in PROFILE_TYPES:
                    cleared_conditions = []
                    for value_field, evidence_field in (
                            ("support_level", "support_evidence"),
                            ("independent_recheck", "recheck_evidence")):
                        if record.get(value_field) == "unknown" and record.get(evidence_field) is None:
                            record[value_field] = None
                            cleared_conditions.append(value_field)
                    if cleared_conditions:
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "fields": cleared_conditions,
                                         "code": "profile_condition_fields_cleared"})
                preference_quote = _session_preference_evidence_quote(record, sources, new_ids)
                if preference_quote is not None:
                    normalized_fields = []
                    existing_preference = record.get("preference")
                    if (not nonempty(existing_preference)
                            or not _profile_value_overlaps_student_quote(
                                existing_preference,
                                [{"turn_id": preference_quote["turn_id"],
                                  "quote": preference_quote["quote"]}], sources)):
                        record["preference"] = preference_quote["quote"]
                        normalized_fields.append("preference")
                    if record.get("interpretation_status") != "candidate":
                        record["interpretation_status"] = "candidate"
                        normalized_fields.append("interpretation_status")
                    for item in record.get("evidence", []):
                        if isinstance(item, dict) and item.get("turn_id") == preference_quote["turn_id"]:
                            if item.get("quote") != preference_quote["quote"]:
                                item["quote"] = preference_quote["quote"]
                                normalized_fields.append("evidence.quote")
                            break
                    if normalized_fields:
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "fields": normalized_fields,
                                         "code": "session_preference_recovered_from_quote",
                                         "turn_id": preference_quote["turn_id"]})
                if (record.get("record_type") == "incidental_fact" and record.get("fact") is None
                        and record.get("topic_id") is None and record.get("interest") is None
                        and record.get("preference") is None and record.get("claim_status") == "explicit"
                        and record.get("retention_class") == "episodic_personal_fact"
                        and record.get("interpretation_status") == "explicit"
                        and isinstance(record.get("context"), str)
                        and any(
                            isinstance(item, dict)
                            and isinstance(item.get("turn_id"), str)
                            and item["turn_id"] in new_ids
                            and isinstance(sources.get(item["turn_id"]), dict)
                            and _role(sources[item["turn_id"]]) == "student"
                            and isinstance(item.get("quote"), str)
                            and item["quote"] in str(sources[item["turn_id"]].get("text", ""))
                            and record["context"] in item["quote"]
                            for item in record.get("evidence", []))):
                    record["fact"] = record["context"][:500]
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "field": "fact", "code": "fact_recovered_from_exact_context_quote"})
                if record.get("record_type") == "uncertain_observation" \
                        and isinstance(record.get("evidence"), list) and new_source_order:
                    old_support = record.get("support_evidence")
                    if isinstance(old_support, dict):
                        turn_id, quote = old_support.get("turn_id"), old_support.get("quote")
                        source = sources.get(turn_id) if isinstance(turn_id, str) else None
                        present_ids = {item.get("turn_id") for item in record["evidence"]
                                       if isinstance(item, dict)}
                        if (turn_id in new_ids and isinstance(source, dict) and isinstance(quote, str)
                                and quote in str(source.get("text", "")) and turn_id not in present_ids):
                            record["evidence"].append(deepcopy(old_support))
                    completed_sequence = _reported_correction_sequence(
                        record, sources, new_ids, new_source_order)
                    present_ids = {item.get("turn_id") for item in record["evidence"]
                                   if isinstance(item, dict)}
                    missing_sequence = [item for item in completed_sequence
                                        if item["turn_id"] not in present_ids]
                    if missing_sequence and len(record["evidence"]) + len(missing_sequence) <= 20:
                        record["evidence"].extend(deepcopy(missing_sequence))
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "evidence", "code": "reported_correction_sequence_completed",
                                         "turn_ids": [item["turn_id"] for item in missing_sequence]})
                    present_ids = {item.get("turn_id") for item in record["evidence"]
                                   if isinstance(item, dict)}
                    if record.get("support_level") in {None, "unknown", "hint_question"}:
                        reported_hint = _reported_hint_evidence(
                            record, sources, new_ids, new_source_order)
                        if reported_hint is not None:
                            current_support = record.get("support_evidence")
                            if (record.get("support_level") != "hint_question"
                                    or current_support != reported_hint):
                                record["support_level"] = "hint_question"
                                record["support_evidence"] = deepcopy(reported_hint)
                                if reported_hint["turn_id"] not in present_ids:
                                    record["evidence"].append(deepcopy(reported_hint))
                                warnings.append({"event": "record_normalized", "record_index": index,
                                                 "fields": ["support_level", "support_evidence"],
                                                 "code": "reported_hint_evidence_recovered",
                                                 "turn_id": reported_hint["turn_id"]})
                        elif record.get("support_level") == "hint_question" \
                                or record.get("support_evidence") is not None:
                            record["support_level"] = "unknown"
                            record["support_evidence"] = None
                            warnings.append({"event": "record_normalized", "record_index": index,
                                             "field": "support_level",
                                             "code": "support_evidence_unverified"})
                    if _mark_reported_answer_authorship_unknown(record, sources):
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "context", "code": "reported_answer_authorship_marked"})
                if record.get("record_type") in LEARNING_TYPES \
                        and record.get("support_level") in {"none", "none_before_attempt"} \
                        and record.get("support_evidence") is None:
                    record["support_level"] = "unknown"
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "field": "support_level", "code": "missing_support_evidence"})
                visual_support = None
                if record.get("record_type") in {"child_error", "child_correction", "supported_attempt", "learning_success"} \
                        and record.get("support_evidence") is None:
                    visual_support = _adjacent_visual_support_evidence(
                        record, sources, new_ids, new_source_order)
                if visual_support is not None:
                    old_type = record["record_type"]
                    if old_type == "learning_success":
                        record["record_type"] = "supported_attempt"
                    record["support_level"] = "visual"
                    record["support_evidence"] = visual_support
                    if not any(isinstance(item, dict) and item.get("turn_id") == visual_support["turn_id"]
                               for item in record["evidence"]):
                        record["evidence"].append(deepcopy(visual_support))
                    fields = ["support_level", "support_evidence", "evidence"]
                    if old_type != record["record_type"]:
                        fields.insert(0, "record_type")
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "fields": fields, "code": "adjacent_visual_support"})
                if record.get("record_type") == "learning_success" \
                        and record.get("support_level") in {"hint_question", "visual", "joint"} \
                        and record.get("support_evidence") is not None:
                    record["record_type"] = "supported_attempt"
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "fields": ["record_type"], "code": "learning_success_with_support"})
                if record.get("record_type") == "supported_attempt" \
                        and record.get("support_evidence") is not None \
                        and record.get("interpretation_status") != "explicit_with_assistance":
                    record["interpretation_status"] = "explicit_with_assistance"
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "field": "interpretation_status", "code": "supported_attempt"})
                if (record.get("record_type") in {"child_error", "child_correction"}
                        and record.get("interpretation_status") in {
                            "explicit", "explicit_with_assistance", "single_independent_observation"}
                        and isinstance(record.get("evidence"), list)
                        and any(isinstance(item, dict) and item.get("turn_id") in sources
                                and _role(sources[item["turn_id"]]) == "student"
                                for item in record["evidence"])):
                    status_prefix = record["record_type"]
                    required_status = None
                    status_code = None
                    if record.get("support_evidence") is not None \
                            and record.get("support_level") in {"hint_question", "visual", "joint"}:
                        required_status = "explicit_with_assistance"
                        status_code = f"{status_prefix}_support"
                    elif record.get("support_evidence") is not None \
                            and record.get("support_level") in {"none", "none_before_attempt"}:
                        required_status = "single_independent_observation"
                        status_code = f"{status_prefix}_independent"
                    elif record.get("support_level") in {None, "unknown"} \
                            and record.get("support_evidence") is None \
                            and record.get("interpretation_status") in {
                                "explicit", "explicit_with_assistance", "single_independent_observation"}:
                        required_status = "explicit"
                        status_code = f"{status_prefix}_unknown_support"
                    if required_status is not None \
                            and record.get("interpretation_status") != required_status:
                        record["interpretation_status"] = required_status
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "interpretation_status", "code": status_code})
                prompted_recheck = _adjacent_prompted_recheck_evidence(
                    record, sources, new_ids, new_source_order)
                present_turn_ids = {item.get("turn_id") for item in record.get("evidence", [])
                                    if isinstance(item, dict) and isinstance(item.get("turn_id"), str)}
                missing_recheck = [item for item in prompted_recheck
                                   if item["turn_id"] not in present_turn_ids]
                if (missing_recheck and isinstance(record.get("evidence"), list)
                        and len(record["evidence"]) + len(missing_recheck) <= 20):
                    record["evidence"].extend(deepcopy(missing_recheck))
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "field": "evidence", "code": "prompted_recheck_evidence_preserved",
                                     "turn_ids": [item["turn_id"] for item in missing_recheck]})
                if record.get("record_type") in LEARNING_TYPES \
                        and record.get("independent_recheck") in (None, "unknown", "unknown_without_support") \
                        and isinstance(record.get("recheck_evidence"), dict) \
                        and isinstance(record.get("evidence"), list):
                    condition = record["recheck_evidence"]
                    turn_id, quote = condition.get("turn_id"), condition.get("quote")
                    source = sources.get(turn_id) if isinstance(turn_id, str) else None
                    source_text = source.get("text") if isinstance(source, dict) else None
                    existing = next((item for item in record["evidence"]
                                     if isinstance(item, dict) and item.get("turn_id") == turn_id), None)
                    recovered_answer_ids = {item["turn_id"] for item in prompted_recheck
                                            if _role(sources.get(item["turn_id"], {})) == "student"}
                    recheck_matches_prompted_answer = (
                        record.get("record_type") != "child_correction" or turn_id in recovered_answer_ids
                    )
                    quote_is_preserved = existing is None or (
                        isinstance(existing.get("quote"), str)
                        and isinstance(quote, str)
                        and (quote in existing["quote"] or existing["quote"] in quote)
                    )
                    if (isinstance(source, dict) and _role(source) != "assistant"
                            and isinstance(source_text, str)
                            and isinstance(quote, str) and nonempty(quote) and quote in source_text
                            and quote_is_preserved and recheck_matches_prompted_answer):
                        if existing is None:
                            record["evidence"].append(deepcopy(condition))
                        elif quote not in existing["quote"]:
                            existing["quote"] = quote
                        record["recheck_evidence"] = None
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "evidence", "code": "unknown_recheck_evidence_preserved",
                                         "turn_id": turn_id})
                if record.get("record_type") == "uncertain_observation" and new_source_order:
                    task_prompt = _adjacent_task_prompt_evidence(record, sources, new_ids, new_source_order)
                    if task_prompt is not None:
                        record["evidence"].append(task_prompt)
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "question_evidence", "code": "adjacent_teacher_prompt"})
                    teacher_context = _adjacent_teacher_context_evidence(
                        record, sources, new_ids, new_source_order)
                    if teacher_context:
                        _insert_evidence_in_source_order(record, teacher_context, new_source_order)
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "evidence",
                                         "code": "adjacent_teacher_context_recovered",
                                         "turn_ids": [item["turn_id"] for item in teacher_context]})
                no_followup_pair = _explicit_no_followup_pair(
                    record, sources, new_ids, new_source_order)
                if no_followup_pair is not None:
                    _insert_evidence_in_source_order(record, list(no_followup_pair), new_source_order)
                    record["context"] = "Ученик сообщает, что после этой работы похожих заданий не встречал."
                    record["interpretation"] = (
                        "Повторная практика с этой особенностью не подтверждена; навык остаётся непроверенным.")
                    record["interpretation_status"] = "ambiguous"
                    record["independent_recheck"] = "unknown"
                    record["recheck_evidence"] = None
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "field": "interpretation",
                                     "code": "explicit_no_followup_preserved"})
                immediate_teacher_context = _immediate_teacher_context_evidence(
                    record, sources, new_ids, new_source_order)
                present_turn_ids = {item.get("turn_id") for item in record.get("evidence", [])
                                    if isinstance(item, dict)}
                missing_teacher_context = [item for item in immediate_teacher_context
                                          if item["turn_id"] not in present_turn_ids]
                if (missing_teacher_context and isinstance(record.get("evidence"), list)
                        and len(record["evidence"]) + len(missing_teacher_context) <= 20):
                    _insert_evidence_in_source_order(record, missing_teacher_context, new_source_order)
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "field": "evidence", "code": "immediate_teacher_context_preserved",
                                     "turn_ids": [item["turn_id"] for item in missing_teacher_context]})
                if record.get("record_type") == "child_correction" \
                        and isinstance(record.get("links"), list) and isinstance(record.get("evidence"), list):
                    cited = {item.get("turn_id") for item in record["evidence"] if isinstance(item, dict)}
                    added_turns = []
                    for link in record["links"]:
                        if not isinstance(link, dict) or link.get("relation") != "corrects_answer" \
                                or link.get("target_kind") != "turn":
                            continue
                        target_id = link.get("target_id")
                        source = sources.get(target_id, {})
                        if (target_id not in cited and _role(source) == "student"
                                and isinstance(source.get("text"), str) and source["text"].strip()):
                            record["evidence"].append({"turn_id": target_id,
                                                       "quote": source["text"][:2000]})
                            cited.add(target_id)
                            added_turns.append(target_id)
                    if added_turns:
                        warnings.append({"event": "record_normalized", "record_index": index,
                                         "field": "evidence", "code": "linked_turn_citation",
                                         "turn_ids": added_turns})
                fixed_fields = {
                    "interest_candidate": {"interpretation_status": "candidate",
                                           "claim_status": "candidate", "retention_class": "profile_candidate"},
                    "session_preference_candidate": {"interpretation_status": "candidate",
                                                     "claim_status": "candidate", "retention_class": "session_only"},
                    "incidental_fact": {"interpretation_status": "explicit",
                                        "claim_status": "explicit",
                                        "retention_class": "episodic_personal_fact"},
                }.get(record.get("record_type"), {})
                if any(record.get(field) != value for field, value in fixed_fields.items()):
                    for field, value in fixed_fields.items():
                        record[field] = value
                    warnings.append({"event": "record_normalized", "record_index": index,
                                     "fields": sorted(fixed_fields), "code": "record_type_invariants"})
            if isinstance(record, dict) and record.get("record_type") in LEARNING_TYPES \
                    and len(allowed_topic_ids) == 1 and record.get("topic_id") is None:
                record = deepcopy(record)
                record["topic_id"] = next(iter(allowed_topic_ids))
                warnings.append({"event": "record_normalized", "record_index": index,
                                 "field": "topic_id", "code": "single_allowed_topic"})
            fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            _check(fingerprint not in seen, "Completion contains a duplicate V2 memory record", "duplicate_record")
            seen.add(fingerprint)
            accepted.extend((index, item) for item in validate_records_v2(
                json.dumps({"records": [record]}, ensure_ascii=False), sources, new_ids,
                allowed_topic_ids, previous_records, new_source_order))
        except ContractValidationError as exc:
            warnings.append({"event": "record_rejected", "record_index": index,
                             "code": exc.code or "invalid_record"})

    recovered_stop_turn_ids: set[str] = set()
    next_recovered_index = len(obj["records"]) + 1
    for _, success in list(accepted):
        recovered = _unfinished_record_from_direct_stop(
            success, sources, new_ids, new_source_order)
        if recovered is None:
            continue
        unfinished_record, related_turn_ids = recovered
        stop_turn_id = related_turn_ids[-1]
        existing_unfinished_turns = {
            item.get("turn_id") for _, prior in accepted
            if prior.get("record_type") == "unfinished_work"
            for item in prior.get("evidence", []) if isinstance(item, dict)
        }
        if stop_turn_id in recovered_stop_turn_ids or stop_turn_id in existing_unfinished_turns:
            continue
        try:
            validated = validate_records_v2(
                json.dumps({"records": [unfinished_record]}, ensure_ascii=False),
                sources, new_ids, allowed_topic_ids, previous_records, new_source_order)
        except ContractValidationError as exc:
            warnings.append({"event": "record_rejected", "record_index": next_recovered_index,
                             "code": exc.code or "invalid_record"})
            next_recovered_index += 1
            continue
        accepted.extend((next_recovered_index, item) for item in validated)
        warnings.append({"event": "record_normalized", "record_index": next_recovered_index,
                         "field": "record_type", "code": "unfinished_work_recovered_from_direct_stop",
                         "turn_ids": related_turn_ids})
        recovered_stop_turn_ids.add(stop_turn_id)
        next_recovered_index += 1

    interest_evidence = {
        (item.get("turn_id"), item.get("quote"))
        for _, record in accepted if record.get("record_type") == "interest_candidate"
        for item in record.get("evidence", []) if isinstance(item, dict)
    }
    profile_filtered: list[tuple[int, dict]] = []
    for index, record in accepted:
        if record.get("record_type") == "incidental_fact":
            shared_evidence = [
                item for item in record.get("evidence", [])
                if isinstance(item, dict)
                and (item.get("turn_id"), item.get("quote")) in interest_evidence
            ]
            if shared_evidence:
                warnings.append({"event": "record_rejected", "record_index": index,
                                 "code": "duplicate_interest_fact_source",
                                 "turn_ids": sorted({item["turn_id"] for item in shared_evidence
                                                     if isinstance(item.get("turn_id"), str)})})
                continue
        profile_filtered.append((index, record))
    accepted = profile_filtered

    priority = {"child_correction": 0, "child_error": 1, "supported_attempt": 2,
                "learning_success": 3, "uncertain_observation": 4, "unfinished_work": 5,
                "source_correction": 6}

    def student_turns(row: dict) -> set[str]:
        return {item["turn_id"] for item in row.get("evidence", [])
                if item.get("turn_id") in sources and _role(sources[item["turn_id"]]) == "student"}

    def linked_pair(left: dict, right: dict, shared: set[str]) -> bool:
        for source, target in ((left, right), (right, left)):
            for link in source.get("links", []):
                if link.get("relation") == "response_to" and link.get("target_kind") == "turn" \
                        and link.get("target_id") in shared:
                    return True
                if link.get("relation") == "corrects_answer":
                    if link.get("target_kind") == "turn" and link.get("target_id") in shared:
                        return True
                    if link.get("target_kind") == "record" and link.get("target_id") == target.get("record_id"):
                        return True
        return False

    def complementary_unfinished_pair(left: dict, right: dict) -> bool:
        if {left.get("record_type"), right.get("record_type")} != {
                "learning_success", "unfinished_work"}:
            return False
        unfinished = left if left.get("record_type") == "unfinished_work" else right
        success = right if unfinished is left else left
        success_turns = {item.get("turn_id") for item in success.get("evidence", [])
                         if isinstance(item, dict)}
        for item in unfinished.get("evidence", []):
            if not isinstance(item, dict):
                continue
            turn_id, quote = item.get("turn_id"), item.get("quote")
            if (turn_id not in success_turns and isinstance(turn_id, str)
                    and _role(sources.get(turn_id, {})) == "student"
                    and isinstance(quote, str) and _has_explicit_unfinished_cue(quote)):
                return True
        return False

    filtered: list[tuple[int, dict]] = []
    for index, record in accepted:
        current_turns = student_turns(record)
        conflicts = []
        if _is_learning_type(record["record_type"]):
            for prior_index, prior in filtered:
                shared = current_turns & student_turns(prior)
                if (not shared or record.get("topic_id") != prior.get("topic_id")
                        or not _is_learning_type(prior["record_type"]) or linked_pair(record, prior, shared)
                        or complementary_unfinished_pair(record, prior)):
                    continue
                conflicts.append((prior_index, prior))
        if not conflicts:
            filtered.append((index, record))
            continue
        if all(priority[record["record_type"]] < priority[prior["record_type"]]
               for _, prior in conflicts):
            conflicting_indices = {prior_index for prior_index, _ in conflicts}
            filtered = [(prior_index2, prior2) for prior_index2, prior2 in filtered
                        if prior_index2 not in conflicting_indices]
            warnings.extend({"event": "record_rejected", "record_index": prior_index,
                             "code": "duplicate_learning_source"}
                            for prior_index, _ in conflicts)
            filtered.append((index, record))
        else:
            warnings.append({"event": "record_rejected", "record_index": index,
                             "code": "duplicate_learning_source"})
    filtered.sort(key=lambda item: item[0])
    return [record for _, record in filtered], warnings


def record_use_policy(record_type: str) -> str:
    try:
        return USE_POLICY[record_type]
    except KeyError:
        raise ContractValidationError("Unknown V2 record type", "invalid_record_type") from None
