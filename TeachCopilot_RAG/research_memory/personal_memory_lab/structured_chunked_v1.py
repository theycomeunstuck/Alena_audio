"""Opt-in V1 extraction over complete ordered source fragments.

The V1 writer, accumulated prior records, validators, and topic graph are
unchanged. Fragmenting changes the number of model updates and is a separate
experimental condition. Output or context overflow is a failure, never a
reason to discard previous records or silently retry a modified request.
"""
from copy import deepcopy
import hashlib
import json

from research_memory.contracts import parse_object

from research_memory.graph_extension.runner import _prepare_memory
from .chunks import lesson_chunks


def fragment_history(history, *, chunk_bytes=16000):
    if not 0 < chunk_bytes <= 16000:
        raise ValueError("Invalid V1 fragment size")
    output = deepcopy(history)
    output["lessons"] = []
    plan, seen, lesson_ids = [], set(), set()
    for lesson in history["lessons"]:
        lesson_id = lesson.get("lesson_id")
        if not isinstance(lesson_id, str) or not lesson_id or lesson_id in lesson_ids:
            raise ValueError("Lesson IDs must be unique")
        lesson_ids.add(lesson_id)
        parts = lesson_chunks(lesson, max_bytes=chunk_bytes)
        expected = [turn["turn_id"] for turn in lesson["turns"]]
        combined = [turn["turn_id"] for part in parts for turn in part["turns"]]
        if not expected or combined != expected or len(set(expected)) != len(expected) or set(expected) & seen:
            raise ValueError("V1 fragments must cover unique source turns in order")
        seen.update(expected)
        for index, part in enumerate(parts, 1):
            output["lessons"].append({**deepcopy(lesson), "turns": deepcopy(part["turns"])})
            plan.append({"lesson_id": lesson_id, "fragment": index, "fragment_count": len(parts),
                         "source_ids": [turn["turn_id"] for turn in part["turns"]]})
    return output, plan


def compact_request(messages, schema):
    """Map only typed turn references; leave all natural text and record IDs intact."""
    messages, schema = deepcopy(messages), deepcopy(schema)
    payload = parse_object(messages[-1]["content"])
    fields = schema["properties"]["records"]["items"]["properties"]
    enum = fields["evidence_turn_ids"]["items"].get("enum", [])
    if len(enum) != len(set(enum)) or not all(isinstance(value, str) for value in enum):
        raise ValueError("Invalid source ID schema")
    mapping = {source: f"t{index}" for index, source in enumerate(sorted(enum), 1)}
    for turn in payload["new_lesson"]["turns"]:
        turn["turn_id"] = mapping[turn["turn_id"]]
    for record in payload["previous_records"]:
        record["evidence_turn_ids"] = [mapping[value] for value in record["evidence_turn_ids"]]
        for field in ("support_evidence", "recheck_evidence"):
            if record.get(field) is not None:
                record[field]["source_turn_id"] = mapping[record[field]["source_turn_id"]]
    fields["evidence_turn_ids"]["items"]["enum"] = [mapping[value] for value in enum]
    for field in ("support_evidence", "recheck_evidence"):
        branches = fields[field].get("anyOf", [])
        for branch in branches:
            if branch.get("type") == "object":
                target = branch["properties"]["source_turn_id"]
                if set(target["enum"]) != set(enum):
                    raise ValueError("Inconsistent source ID schema")
                target["enum"] = [mapping[value] for value in target["enum"]]
    messages[-1]["content"] = json.dumps(payload, ensure_ascii=False)
    return messages, schema, mapping


def decode_compact_response(text, mapping):
    """Decode only typed response IDs; unknown aliases fail before V1 validation."""
    inverse = {alias: source for source, alias in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Non-unique source aliases")
    response = parse_object(text)
    records = response.get("records")
    if not isinstance(records, list):
        raise ValueError("Invalid records envelope")
    def decode(value):
        if not isinstance(value, str) or value not in inverse:
            raise ValueError("Unknown source alias")
        return inverse[value]
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Invalid record")
        if not isinstance(row.get("evidence_turn_ids"), list):
            raise ValueError("Invalid source aliases")
        row["evidence_turn_ids"] = [decode(value) for value in row["evidence_turn_ids"]]
        for field in ("support_evidence", "recheck_evidence"):
            item = row.get(field)
            if item is not None:
                if not isinstance(item, dict) or "source_turn_id" not in item:
                    raise ValueError("Invalid condition evidence")
                item["source_turn_id"] = decode(item["source_turn_id"])
    return json.dumps(response, ensure_ascii=False)


def prepare_chunked_v1(history, journal, budget, *, chunk_bytes=16000, extraction_client=None,
                       source_id_mode="canonical", compact_trace=None):
    if source_id_mode not in {"canonical", "compact_v1"}:
        raise ValueError("Invalid source ID mode")
    fragmented, plan = fragment_history(history, chunk_bytes=chunk_bytes)
    by_lesson = {}
    for item in plan:
        by_lesson.setdefault(item["lesson_id"], []).append(item)

    class FragmentJournal:
        def __init__(self):
            self.positions = {}

        @property
        def context(self):
            return journal.context

        @context.setter
        def context(self, value):
            journal.context = value

        def __call__(self, messages, purpose, response_schema=None):
            if purpose == "extract":
                lesson_id = journal.context["lesson_id"]
                position = self.positions.get(lesson_id, 0)
                fragment = by_lesson[lesson_id][position]
                self.positions[lesson_id] = position + 1
                journal.context.update(fragment=fragment["fragment"],
                                       fragment_count=fragment["fragment_count"])
            mapping = None
            if purpose == "extract" and source_id_mode == "compact_v1":
                messages, response_schema, mapping = compact_request(messages, response_schema)
                encoded = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if compact_trace is None:
                    raise ValueError("Compact source ID trace is required")
                compact_trace({**journal.context, "call_id": journal.attempted + 1,
                               "source_id_mode": source_id_mode, "mapping": mapping,
                               "mapping_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                               "messages": messages, "response_schema": response_schema})
            # This shared journal is sequential; keep its call counter and budget.
            original_client = journal.client if extraction_client is not None else None
            if extraction_client is not None:
                journal.client = extraction_client
            try:
                result = journal(messages, purpose, response_schema)
            finally:
                if extraction_client is not None:
                    journal.client = original_client
            return decode_compact_response(result, mapping) if mapping is not None else result

    result = _prepare_memory(fragmented, ("structured", "graph"), FragmentJournal(), budget)
    result["fragment_plan"] = plan
    return result
