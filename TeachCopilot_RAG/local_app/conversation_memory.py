"""Application-only drafts for conversational preferences, with literal sources.

These rules do not modify the frozen academic-memory research experiment.
Extraction suggests a draft; only an explicit teacher choice changes the card.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


AVOID_PREFIX = "Не использовать сюжет: "
MAX_CANDIDATES = 16
MAX_INTEREST_CHARS = 120
OPERATIONS = ("add", "remove", "avoid", "allow")


def interest_schema(sources: list[dict]) -> dict:
    source_ids = [str(source["source_id"]) for source in sources]
    return {"type": "object", "additionalProperties": False, "required": ["candidates"],
            "properties": {"candidates": {"type": "array", "maxItems": MAX_CANDIDATES,
                "items": {"type": "object", "additionalProperties": False,
                    "required": ["operation", "interest", "source_id", "quote"],
                    "properties": {"operation": {"type": "string", "enum": list(OPERATIONS)},
                        "interest": {"type": "string", "minLength": 1, "maxLength": MAX_INTEREST_CHARS},
                        "source_id": {"type": "string", "enum": source_ids},
                        "quote": {"type": "string", "minLength": 1, "maxLength": 2000}}}}}}


def interest_messages(sources: list[dict]) -> list[dict]:
    prompt = (Path(__file__).parent / "prompts" / "conversation_interests.txt").read_text(encoding="utf-8")
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps({"sources": sources}, ensure_ascii=False)}]


def validate_interest_candidates(raw: str, sources: list[dict]) -> list[dict]:
    """Check syntax, author, identifiers and literal quotes; not semantic truth."""
    from local_app.memory_ported.contracts import parse_object
    data = parse_object(raw)
    if set(data) != {"candidates"} or not isinstance(data["candidates"], list):
        raise ValueError("invalid interest candidate envelope")
    if len(data["candidates"]) > MAX_CANDIDATES:
        raise ValueError("too many interest candidates")
    by_id = {str(source["source_id"]): source for source in sources}
    result, seen = [], set()
    for candidate in data["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != {"operation", "interest", "source_id", "quote"}:
            raise ValueError("invalid interest candidate fields")
        if not all(isinstance(item, str) and item.strip() for item in candidate.values()):
            raise ValueError("interest candidate fields must be nonempty strings")
        if candidate["operation"] not in OPERATIONS or len(candidate["interest"]) > MAX_INTEREST_CHARS or len(candidate["quote"]) > 2000:
            raise ValueError("invalid interest operation or length")
        if candidate["source_id"] not in by_id:
            raise ValueError("interest refers to an unavailable student source")
        source = by_id[candidate["source_id"]]
        if source.get("author_role", source.get("role")) != "student":
            raise ValueError("interest evidence must be a student source")
        text = source.get("content", source.get("text", ""))
        if not candidate["quote"].strip() or candidate["quote"] not in text:
            raise ValueError("interest quote must be a literal source substring")
        if candidate["interest"] != candidate["interest"].strip():
            raise ValueError("interest has surrounding whitespace")
        if any(ord(char) < 32 for char in candidate["interest"]):
            raise ValueError("interest contains control characters")
        digest = hashlib.sha256(json.dumps(candidate, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
        candidate_id = "interest-" + digest
        if candidate_id in seen:
            raise ValueError("duplicate interest candidate")
        seen.add(candidate_id)
        result.append({"candidate_id": candidate_id, **candidate})
    # Applying contradictory updates follows the recorded conversation order,
    # independent of the order in which a model emitted its suggestions.
    source_order = {str(source["source_id"]): index for index, source in enumerate(sources)}
    return sorted(result, key=lambda item: source_order[item["source_id"]])


def preference_projection(card: dict) -> dict:
    """Bounded generic preferences for all topics; no raw notes or provenance."""
    interests = [item for item in card.get("interests", []) if isinstance(item, str)
                 and 0 < len(item) <= MAX_INTEREST_CHARS][:8]
    avoided = [item[len(AVOID_PREFIX):] for item in (card.get("rag_context") or {}).get("avoid", [])
               if isinstance(item, str) and item.startswith(AVOID_PREFIX)
               and 0 < len(item[len(AVOID_PREFIX):]) <= MAX_INTEREST_CHARS][:8]
    denied = {item.casefold() for item in avoided}
    return {"interests": [item for item in interests if item.casefold() not in denied],
            "avoided_interests": avoided}


def apply_interest_choices(card: dict, candidates: list[dict]) -> dict:
    """Apply a teacher's already validated choices without deleting history."""
    updated = deepcopy(card)
    interests = list(updated.get("interests") or [])
    rag_context = updated.setdefault("rag_context", {})
    avoid = list(rag_context.get("avoid") or [])
    for candidate in candidates:
        interest = candidate["interest"]
        key = interest.casefold()
        operation = candidate["operation"]
        if operation in {"add", "remove"}:
            interests = [item for item in interests if item.casefold() != key]
        if operation in {"remove", "avoid", "allow"}:
            avoid = [item for item in avoid if not (
                isinstance(item, str) and item.startswith(AVOID_PREFIX)
                and item[len(AVOID_PREFIX):].casefold() == key)]
        if operation == "add":
            interests.append(interest)
        elif operation in {"remove", "avoid"}:
            avoid.append(AVOID_PREFIX + interest)
    updated["interests"] = interests
    rag_context["avoid"] = avoid
    return updated
