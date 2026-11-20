"""Session processing: source-linked summaries and validated memory extraction.

The model is used through the injected ``model_call(messages, purpose, schema)``
hook (or the runtime's local admission path). Every proposal is validated by
code against the session snapshot: schema, provenance and literal evidence.
The model never writes numbers and never applies academic state directly.
"""
from __future__ import annotations

import hashlib
import json


SUMMARY_CHUNK_PROMPT = "session_summary_chunk_v1"
SUMMARY_REDUCE_PROMPT = "session_summary_reduce_v1"
EXTRACT_PROMPT = "personal_extract_v1"
PROMPT_DIR_RECIPE = "unified_memory_v1"

SUMMARY_HEADING_STUDIED = "Что изучали"
SUMMARY_HEADING_ACHIEVED = "Что получилось"
SUMMARY_HEADING_REPEAT = "Что стоит повторить"
SUMMARY_HEADING_NEXT = "Следующий шаг"

SUMMARY_CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "studied": {"type": "array", "items": {"type": "string"}},
        "achieved": {"type": "array", "items": {"type": "string"}},
        "repeat": {"type": "array", "items": {"type": "string"}},
        "next_step": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["studied", "achieved", "repeat", "next_step"],
    "additionalProperties": False,
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["interest", "avoided_interest", "preference"]},
                    "operation": {"type": "string", "enum": ["add", "update", "remove"]},
                    "value": {"type": "string"},
                    "scope": {"type": "string", "enum": ["durable", "session"]},
                    "explicit_durable": {"type": "boolean"},
                    "source_id": {"type": "string"},
                    "evidence_text": {"type": "string"},
                },
                "required": ["category", "operation", "value", "scope", "explicit_durable",
                             "source_id", "evidence_text"],
                "additionalProperties": False,
            },
        },
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string",
                                 "enum": ["knowledge_observation", "error_pattern", "learning_goal"]},
                    "proposed_value": {"type": "object"},
                    "source_id": {"type": "string"},
                    "evidence_text": {"type": "string"},
                },
                "required": ["category", "proposed_value", "source_id", "evidence_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["changes", "candidates"],
    "additionalProperties": False,
}

MAX_LIST_ITEMS = 30
MAX_ITEM_CHARS = 400


def _warning(code, **detail):
    return {"code": code, **detail}


def source_fingerprint(sources):
    """Order-bound fingerprint of the exact session snapshot."""
    canonical = [{"source_id": source["source_id"], "author_role": source["author_role"],
                  "text": source["text"], "delivery_status": source.get("delivery_status")}
                 for source in sources]
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prompt_recipe(hashes):
    """Recipe identity binds the exact prompt files in one stable string."""
    joined = "|".join(f"{name}:{hashes[name]}" for name in sorted(hashes))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{PROMPT_DIR_RECIPE}:{digest}"


def session_model_messages(sources):
    """Map snapshot sources onto role-tagged messages for chunking."""
    messages = []
    for source in sources:
        role = "assistant" if source["author_role"] == "assistant" else "user"
        text = source["text"]
        if not isinstance(text, str) or not text.strip():
            continue
        messages.append({"source_id": source["source_id"], "role": role, "text": text,
                         "learning_session_id": source.get("session_id") or source.get("learning_session_id")})
    return messages


def summarize_text(result):
    """Render the four lesson-agent headings from a reduce result."""
    lists = []
    for heading, key in ((SUMMARY_HEADING_STUDIED, "studied"), (SUMMARY_HEADING_ACHIEVED, "achieved"),
                         (SUMMARY_HEADING_REPEAT, "repeat"), (SUMMARY_HEADING_NEXT, "next_step")):
        items = [str(item).strip() for item in (result.get(key) or []) if str(item).strip()][:MAX_LIST_ITEMS]
        lines = [f"- {item}" for item in items] or ["- (нет данных)"]
        lists.append("\n".join([heading, *lines]))
    return "\n\n".join(lists)


def _bounded_list(values):
    if not isinstance(values, list):
        return []
    items = []
    for value in values[:MAX_LIST_ITEMS]:
        text = str(value).strip()
        if text:
            items.append(text[:MAX_ITEM_CHARS])
    return items


def normalize_chunk_summary(payload):
    if not isinstance(payload, dict):
        raise ValueError("summary chunk must be an object")
    return {key: _bounded_list(payload.get(key)) for key in ("studied", "achieved", "repeat", "next_step")}


def reduce_chunk_summaries(summaries):
    """Deterministic merge of chunk summaries without another model round trip.

    The reduce model call is still made for quality; this merge is the
    deterministic fallback used when the reduce call itself fails, and the
    de-duplication basis applied to the model result.
    """
    merged = {key: [] for key in ("studied", "achieved", "repeat", "next_step")}
    for summary in summaries:
        for key in merged:
            for item in summary.get(key) or []:
                if item not in merged[key]:
                    merged[key].append(item)
    return {key: values[:MAX_LIST_ITEMS] for key, values in merged.items()}


def _find_literal_evidence(evidence_text, source_text):
    """Require the quoted evidence to be a literal substring of the source."""
    if not isinstance(evidence_text, str) or not evidence_text.strip():
        return False
    needle = " ".join(evidence_text.split()).casefold()
    haystack = " ".join(source_text.split()).casefold()
    return bool(needle) and needle in haystack


def _numeric_fields(value):
    found = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_numeric_fields(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_numeric_fields(item))
    return found


def validate_extraction(payload, chunk, sources_by_id):
    """Validate one extraction result; keep only code-verified items.

    The prompt chooses semantic candidates; this function validates schema,
    provenance and literal evidence, then decides applicability. Items that
    fail validation never reach storage; each failure becomes a warning.
    """
    if not isinstance(payload, dict):
        raise ValueError("extraction result must be an object")
    changes, candidates, warnings = [], [], []
    chunk_source_ids = set(chunk.get("source_ids") or [])
    seen_changes = set()
    for raw in payload.get("changes") or []:
        if not isinstance(raw, dict):
            warnings.append(_warning("malformed_change"))
            continue
        category = raw.get("category")
        operation = raw.get("operation")
        value = raw.get("value")
        scope = raw.get("scope")
        explicit = raw.get("explicit_durable")
        source_id = raw.get("source_id")
        evidence_text = raw.get("evidence_text")
        source = sources_by_id.get(source_id)
        if category not in ("interest", "avoided_interest", "preference"):
            warnings.append(_warning("invalid_category", category=category))
            continue
        if operation not in ("add", "update", "remove"):
            warnings.append(_warning("invalid_operation", operation=operation))
            continue
        if source is None or source_id not in chunk_source_ids:
            warnings.append(_warning("unavailable_source", source_id=str(source_id)[:64]))
            continue
        if not _find_literal_evidence(evidence_text, source["text"]):
            warnings.append(_warning("evidence_not_literal", source_id=source_id))
            continue
        # A-report F2: a remove also names the fact being withdrawn, so the
        # value must be a nonempty literal for every operation.
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            warnings.append(_warning("invalid_value", source_id=source_id))
            continue
        if scope not in ("durable", "session"):
            warnings.append(_warning("invalid_scope", scope=scope))
            continue
        if explicit is not True and scope == "durable":
            # Non-explicit durable claims stay proposals for teacher review.
            status = "proposed"
        elif scope == "session":
            # Temporary requests belong to session state, not durable facts.
            warnings.append(_warning("session_scope_not_durable", source_id=source_id))
            continue
        else:
            status = "applied"
        key = (category, operation, value, status)
        if key in seen_changes:
            continue
        seen_changes.add(key)
        changes.append({"category": category, "operation": operation,
                        "value": value,
                        "scope": scope, "status": status, "explicit_durable": explicit is True,
                        "source_id": source_id,
                        "evidence": [{"source_id": source_id, "text": str(evidence_text)[:300]}],
                        "source_refs": [{"kind": "conversation_source", "source_id": source_id,
                                         "author_role": source["author_role"],
                                         "session_id": source.get("session_id")}]})
    seen_candidates = set()
    for raw in payload.get("candidates") or []:
        if not isinstance(raw, dict):
            warnings.append(_warning("malformed_candidate"))
            continue
        category = raw.get("category")
        proposed_value = raw.get("proposed_value")
        source_id = raw.get("source_id")
        evidence_text = raw.get("evidence_text")
        source = sources_by_id.get(source_id)
        if category not in ("knowledge_observation", "error_pattern", "learning_goal"):
            warnings.append(_warning("invalid_category", category=category))
            continue
        if source is None or source_id not in chunk_source_ids:
            warnings.append(_warning("unavailable_source", source_id=str(source_id)[:64]))
            continue
        if not _find_literal_evidence(evidence_text, source["text"]):
            warnings.append(_warning("evidence_not_literal", source_id=source_id))
            continue
        if not isinstance(proposed_value, dict) or not proposed_value:
            warnings.append(_warning("invalid_proposed_value", source_id=source_id))
            continue
        numbers = _numeric_fields(proposed_value)
        if numbers:
            # The model must never author a mastery delta; discard and warn.
            warnings.append(_warning("numeric_proposal_discarded", source_id=source_id))
            continue
        key = json.dumps({"category": category, "value": proposed_value}, ensure_ascii=False,
                         sort_keys=True)
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        candidates.append({"category": category,
                           "proposed_value": {str(k): str(v)[:500] for k, v in proposed_value.items()},
                           "source_id": source_id,
                           "source_refs": [{"kind": "conversation_source", "source_id": source_id,
                                            "author_role": source["author_role"],
                                            "session_id": source.get("session_id")}]})
    return changes, candidates, warnings
