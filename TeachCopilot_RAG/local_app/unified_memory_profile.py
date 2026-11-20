"""Effective personal profile: manual values plus the durable change replay.

The effective projection starts from the manual/profile (learner card) values
and replays applied durable change events in source order. Keys are compared
consistently (whitespace-collapsed, case-folded), a removal suppresses the
matching original card value as well as conversation-added values, and an
avoidance never removes an interest: interest and avoided story theme are
separate categories. Reverted events and their revert markers are skipped, so
a rollback restores the previous effective value without touching history.

Pure functions only: no store, no model, no I/O. The facade feeds this module
from the learner card and the store's ordered durable change events.
"""
from __future__ import annotations

from datetime import date, datetime


def normalize_fact_key(value):
    """Consistent comparison key for personal fact values."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).casefold()


def _iso(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _item(value, origin, *, source_refs=None, evidence=None, event=None, confirmations=None):
    item = {"value": value, "origin": origin,
            "source_refs": list(source_refs or [])}
    if evidence:
        item["evidence"] = [dict(entry) for entry in evidence]
    if event is not None:
        # Literal evidence, actual source ids, status, event/date and scope of
        # the applied change that produced this effective value.
        item["change_id"] = str(event["id"])
        item["event_at"] = _iso(event.get("created_at"))
        item["scope"] = event.get("scope")
        item["status"] = event.get("status")
        item["explicit_durable"] = bool(event.get("explicit_durable"))
    if confirmations:
        item["confirmations"] = list(confirmations)
    return item


def replay_personal_changes(profile, events):
    """Replay ordered durable change events over the manual profile items.

    ``profile`` maps category to the manual items:
    ``{"interests": [...], "avoided_interests": [...], "preferences": [...]}``.
    Interest/avoided items are ``{"value", "source_refs"}``; preference items
    are ``{"name", "value", "source_refs"}``.

    ``events`` are durable personal change events in source order (the store
    guarantees the order). Each event is a dict with ``id``, ``category``,
    ``operation``, ``value``, ``scope``, ``status``, ``explicit_durable``,
    ``source_refs``, ``evidence``, ``reverts_id`` and ``created_at``.

    Returns the same three buckets with effective items only.
    """
    buckets = {"interest": [], "avoided_interest": [], "preference": []}
    profile_map = {
        "interest": profile.get("interests") or [],
        "avoided_interest": profile.get("avoided_interests") or [],
        "preference": profile.get("preferences") or [],
    }
    effective = {
        "interest": [_item(str(item["value"]), "profile", source_refs=item.get("source_refs"))
                     for item in profile_map["interest"] if str(item.get("value") or "").strip()],
        "avoided_interest": [_item(str(item["value"]), "profile", source_refs=item.get("source_refs"))
                             for item in profile_map["avoided_interest"] if str(item.get("value") or "").strip()],
        "preference": [_item(str(item["value"]), "profile",
                             source_refs=item.get("source_refs"),
                             ) for item in profile_map["preference"] if str(item.get("value") or "").strip()],
    }
    for index, item in enumerate(effective["preference"]):
        name = str(profile_map["preference"][index].get("name") or "")
        if name:
            item["name"] = name

    reverted = {event["reverts_id"] for event in events if event.get("reverts_id")}
    for event in events:
        if event.get("reverts_id") or event["id"] in reverted:
            continue
        if event.get("status") != "applied" or event.get("scope") != "durable":
            # Session-scoped and undecided events never reach the effective
            # projection; the store normally filters them already.
            continue
        category = event.get("category")
        if category not in buckets:
            continue
        items = effective[category]
        key = normalize_fact_key(event.get("value"))
        if not key:
            continue
        operation = event.get("operation")
        if operation == "remove":
            effective[category] = [item for item in items
                                   if normalize_fact_key(item.get("value")) != key]
            continue
        if operation not in ("add", "update"):
            continue
        existing = next((item for item in effective[category]
                         if normalize_fact_key(item.get("value")) == key), None)
        if existing is not None:
            # The same fact stated again confirms the existing value; the
            # manual value stays authoritative for its provenance.
            confirmations = list(existing.get("confirmations") or [])
            confirmations.append({"change_id": str(event["id"]),
                                  "event_at": _iso(event.get("created_at")),
                                  "source_refs": list(event.get("source_refs") or [])})
            existing["confirmations"] = confirmations
            continue
        refs = list(event.get("source_refs") or [])
        if not refs:
            refs = [{"kind": "unified_personal_change", "change_id": str(event["id"])}]
        item = _item(str(event.get("value")).strip(), "conversation",
                     source_refs=refs, evidence=event.get("evidence"), event=event)
        if category == "preference":
            item["name"] = "stated_preference"
        effective[category].append(item)
    return {"interests": effective["interest"],
            "avoided_interests": effective["avoided_interest"],
            "preferences": effective["preference"]}
