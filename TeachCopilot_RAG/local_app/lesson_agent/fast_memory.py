"""Projection of the unified facade's fast-memory shape into prompt blocks.

The merged facade returns effective personal and learning facts as records
with ``value`` (or ``name``+``value``) and ``source_refs``. This module is the
single projection point for lesson generation and guided responses. It never
concatenates stale card data onto effective records, keeps avoidance separate
from interests, and budgets whole fact/source units with an explicit
omitted-data diagnostic instead of truncating JSON mid-structure.

An empty effective field is not a service failure: it is reported in the
diagnostics and stays absent from the prompt, never refilled from an old card.
"""
from __future__ import annotations

import json

DEFAULT_BUDGET_CHARS = 3600
GUIDED_BUDGET_CHARS = 1500

# Per-section item caps; the character budget applies on top of these.
SECTION_LIMITS = {
    "interests": 8,
    "avoided_themes": 4,
    "preferences": 6,
    "learning_goals": 3,
    "confirmed_observations": 6,
    "known_errors": 6,
    "helpful_strategies": 6,
}

_VALUE_KEY_LIMIT = 200


def _bounded(value, limit: int = _VALUE_KEY_LIMIT) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _record_value(item) -> str | None:
    """A usable value from a facade record (or a legacy plain string)."""
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        value = item.get("value")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, bool) or isinstance(value, (int, float)):
            return str(value)
    return None


def _map_section(section: str, items) -> tuple[list[dict], list[dict]]:
    """Map one facade field to prompt units; report unusable records."""
    units: list[dict] = []
    problems: list[dict] = []
    for index, item in enumerate(items or []):
        if section == "interests" or section == "avoided_themes":
            value = _record_value(item)
            if value is None:
                problems.append({"section": section, "index": index, "problem": "missing_value"})
                continue
            units.append({"value": _bounded(value)})
        elif section == "preferences":
            if not isinstance(item, dict) or _record_value(item) is None:
                problems.append({"section": section, "index": index, "problem": "missing_value"})
                continue
            name = str(item.get("name") or "preference")
            units.append({"name": _bounded(name, 80), "value": _bounded(item.get("value"))})
        elif section == "learning_goals":
            if not isinstance(item, dict) or not str(item.get("goal") or "").strip():
                problems.append({"section": section, "index": index, "problem": "missing_goal"})
                continue
            units.append({"topic_id": _bounded(item.get("topic_id"), 80),
                          "goal": _bounded(item.get("goal"))})
        elif section == "confirmed_observations":
            if not isinstance(item, dict) or not item.get("topic_id"):
                problems.append({"section": section, "index": index, "problem": "missing_topic"})
                continue
            unit = {"topic_id": _bounded(item.get("topic_id"), 80)}
            if item.get("status"):
                unit["status"] = _bounded(item.get("status"), 80)
            units.append(unit)
        elif section == "known_errors":
            if not isinstance(item, dict) or not item.get("error_tag"):
                problems.append({"section": section, "index": index, "problem": "missing_tag"})
                continue
            unit = {"error_tag": _bounded(item.get("error_tag"), 80)}
            if item.get("topic_id"):
                unit["topic_id"] = _bounded(item.get("topic_id"), 80)
            if item.get("count") is not None:
                unit["count"] = item.get("count")
            units.append(unit)
        elif section == "helpful_strategies":
            if not isinstance(item, dict) or not item.get("topic_id"):
                problems.append({"section": section, "index": index, "problem": "missing_topic"})
                continue
            strategies = item.get("strategies")
            if isinstance(strategies, list):
                strategies = "; ".join(_bounded(s, 120) for s in strategies[:4] if str(s).strip())
            unit = {"topic_id": _bounded(item.get("topic_id"), 80)}
            if strategies:
                unit["strategies"] = _bounded(strategies)
            units.append(unit)
    return units, problems


def _unfinished_unit(unfinished) -> dict | None:
    if not isinstance(unfinished, dict) or not unfinished.get("lesson_id"):
        return None
    unit = {"lesson_id": _bounded(unfinished.get("lesson_id"), 80),
            "stage": _bounded(unfinished.get("stage"), 80)}
    if unfinished.get("pending_task_id"):
        unit["pending_task_id"] = _bounded(unfinished.get("pending_task_id"), 120)
    if unfinished.get("completed_task_ids"):
        completed = unfinished.get("completed_task_ids")
        if isinstance(completed, list):
            unit["completed_task_ids"] = [_bounded(t, 120) for t in completed[:10]]
    return unit


def fast_memory_block(fast_memory, *, budget_chars: int = DEFAULT_BUDGET_CHARS) -> tuple[dict, dict]:
    """Project the facade's fast-memory records into a bounded prompt block.

    Returns ``(block, diagnostics)``. Budgets apply to whole units: a fact
    with its serialized representation either fits entirely or is omitted and
    reported. The serialized block is always complete valid JSON.
    """
    diagnostics = {"budget_chars": budget_chars, "chars_used": 0,
                   "included": {}, "omitted": [], "empty_sections": [],
                   "shape_problems": []}
    if not isinstance(fast_memory, dict):
        diagnostics["shape_problems"].append({"where": "root", "problem": "not_an_object"})
        return {}, diagnostics
    personal = fast_memory.get("personal")
    learning = fast_memory.get("learning")
    personal = personal if isinstance(personal, dict) else {}
    learning = learning if isinstance(learning, dict) else {}

    interest_units, problems = _map_section("interests", personal.get("interests"))
    diagnostics["shape_problems"].extend(problems)
    avoided_units, problems = _map_section("avoided_themes", personal.get("avoided_interests"))
    diagnostics["shape_problems"].extend(problems)
    preference_units, problems = _map_section("preferences", personal.get("preferences"))
    diagnostics["shape_problems"].extend(problems)
    goal_units, problems = _map_section("learning_goals", learning.get("goals"))
    diagnostics["shape_problems"].extend(problems)
    confirmed_units, problems = _map_section("confirmed_observations",
                                             learning.get("confirmed_observations"))
    diagnostics["shape_problems"].extend(problems)
    error_units, problems = _map_section("known_errors", learning.get("errors"))
    diagnostics["shape_problems"].extend(problems)
    help_units, problems = _map_section("helpful_strategies", learning.get("help"))
    diagnostics["shape_problems"].extend(problems)

    sections = {
        "interests": interest_units,
        "avoided_themes": avoided_units,
        "preferences": preference_units,
        "learning_goals": goal_units,
        "confirmed_observations": confirmed_units,
        "known_errors": error_units,
        "helpful_strategies": help_units,
    }
    unfinished = _unfinished_unit(learning.get("unfinished"))
    if unfinished is not None:
        sections["unfinished_step"] = [unfinished]

    block: dict = {}
    for section, units in sections.items():
        limit = SECTION_LIMITS.get(section)
        if limit is not None and len(units) > limit:
            diagnostics["omitted"].append({"section": section, "count": len(units) - limit,
                                           "reason": "section_limit"})
            units = units[:limit]
        kept: list = []
        omitted_for_budget = False
        for unit in units:
            candidate = dict(block)
            candidate[section] = kept + [unit]
            if len(json.dumps(candidate, ensure_ascii=False)) > budget_chars:
                diagnostics["omitted"].append({"section": section,
                                               "count": len(units) - len(kept),
                                               "reason": "budget"})
                omitted_for_budget = True
                break
            kept.append(unit)
        if kept:
            block[section] = kept if section != "unfinished_step" else kept[0]
            diagnostics["included"][section] = len(kept)
        elif not units:
            diagnostics["empty_sections"].append(section)
        elif omitted_for_budget:
            pass  # already reported above
        else:
            # Units existed but every record was unusable; shape problems say why.
            diagnostics["empty_sections"].append(section)
    diagnostics["chars_used"] = len(json.dumps(block, ensure_ascii=False))
    return block, diagnostics


def bounded_fast_memory_text(fast, *, budget_chars: int = GUIDED_BUDGET_CHARS) -> str | None:
    """A bounded, valid-JSON block of current effective facts for responses.

    ``None`` when the effective block is empty: absence is honest, not a
    fallback to older card fields.
    """
    block, _diagnostics = fast_memory_block(fast, budget_chars=budget_chars)
    if not block:
        return None
    return json.dumps(block, ensure_ascii=False)
