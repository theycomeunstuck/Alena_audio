"""App-only literal source notes for reviewing a generated recommendation."""
from __future__ import annotations

from copy import deepcopy
import re


_UNKNOWN = r"(?:неизвест\w*|unknown|не\s+(?:указан\w*|установлен\w*|подтвержд[её]н\w*|уточнял\w*|уточн[её]н\w*))"
_LINK = r"\s*(?:[:=–-]\s*)?(?:(?:пока|сейчас|сегодня|тоже|остаются|осталась|были|была|было|ещё|еще)\s+){0,2}"
_CONDITIONS = {
    "support": re.compile(
        r"\b(?:условия\s+помощи|помощ[ьи]|уровень\s+поддержки|support_level)"
        + _LINK + _UNKNOWN + r"\b", re.I),
    "independence": re.compile(
        r"\b(?:самостоятельность|самостоятельн\w*\s+(?:решени\w*|перепроверк\w*|работ\w*)"
        r"|independent_work|independent_recheck)"
        r"(?:\s+и\s+освоени\w*(?:\s+(?!не\b)\w+){0,2})?" + _LINK + _UNKNOWN + r"\b", re.I),
}

# This flags words to inspect, not unsupported claims. It intentionally covers
# only a few common formulations and is not a semantic fact checker.
_TEACHER_ATTRIBUTION = re.compile(
    r"\b(?:вопрос\w*|подсказк\w*)\s+(?:учител\w*|преподавател\w*)\b"
    r"|\b(?:учител\w*|преподавател\w*)\s+(?:спросил\w*|задал\w*|зада[её]т\w*|подсказал\w*)\b",
    re.I,
)
_SESSION_END_DISPLAY_FIELDS = ("role", "usage", "text", "excerpt_mode", "truncated")


def recommendation_source_review(action: dict, context: dict) -> dict:
    """Copy literal grounds and explicit unknown phrases, without grading meaning.

    Original model fields and quotations stay untouched. A quoted unknown is an
    observation about that particular source, not an inferred current condition.
    Corrections travel with their original sources even when the model omitted
    one side from its references. Assistant words remain labelled context only.
    """
    sources = context.get("sources", [])
    by_id, positions = {}, {}
    for index, source in enumerate(sources):
        source_id = source.get("source_id")
        if (not isinstance(source_id, str) or not source_id or source_id in by_id
                or source.get("role") not in {"student", "teacher"}
                or not isinstance(source.get("text"), str)):
            raise ValueError("invalid_recommendation_source")
        by_id[source_id], positions[source_id] = source, index
    cited = set(action["evidence_turn_ids"])
    if not by_id.keys() <= set(context.get("evidence_ids", [])) or not cited <= by_id.keys():
        raise ValueError("recommendation_source_unavailable")
    corrected_by, links = {}, {source_id: set() for source_id in by_id}
    for source_id, source in by_id.items():
        target = source.get("corrects_source_id")
        if not target:
            continue
        if (source["role"] != "teacher" or target not in by_id
                or positions[target] >= positions[source_id]):
            raise ValueError("invalid_recommendation_correction")
        corrected_by.setdefault(target, []).append(source_id)
        links[source_id].add(target)
        links[target].add(source_id)
    selected, pending = set(cited), list(cited)
    while pending:
        source_id = pending.pop()
        for linked in links[source_id] - selected:
            selected.add(linked)
            pending.append(linked)
    dialogue = {row["source_id"]: row for row in context.get("dialogue_context", [])
                if row.get("source_id") in by_id and by_id[row["source_id"]]["role"] == "student"
                and row.get("role") == "assistant" and row.get("usage") == "context_only"
                and isinstance(row.get("text"), str)}
    endings = {row["source_id"]: row for row in context.get("session_end_context", [])
               if row.get("source_id") in by_id and by_id[row["source_id"]]["role"] == "student"
               and row.get("role") == "assistant" and row.get("usage") == "context_only"
               and isinstance(row.get("text"), str)}

    def with_context(source_id):
        row = deepcopy(by_id[source_id])
        if source_id in dialogue:
            row["dialogue_context"] = deepcopy(dialogue[source_id])
        if source_id in endings:
            # This is display-only. The assistant reference remains solely in
            # the canonical context.session_end_context checked by the store.
            row["session_end_context"] = {
                key: deepcopy(endings[source_id][key])
                for key in _SESSION_END_DISPLAY_FIELDS if key in endings[source_id]
            }
        return row

    evidence = []
    for source_id, source in by_id.items():
        if source_id not in selected:
            continue
        row = {**with_context(source_id), "cited": source_id in cited,
               "corrected_by": corrected_by.get(source_id, [])}
        evidence.append(row)
    unknowns = []
    for source_id, source in by_id.items():
        if source_id in corrected_by:
            continue
        for condition, pattern in _CONDITIONS.items():
            for match in pattern.finditer(source["text"]):
                unknowns.append({"source_id": source_id, "condition": condition,
                                 "quote": match.group(), "cited": source_id in cited})
    review = {"version": "recommendation-source-review-v1", "sources": evidence,
              "source_unknowns": unknowns, "unknowns_empty": not action["unknowns"],
              "meaning_verified": False}
    if context.get("method") == "personal_graph":
        reason = action.get("reason", "")
        if isinstance(reason, str):
            review["attribution_checks"] = [
                {"quote": match.group()} for match in _TEACHER_ATTRIBUTION.finditer(reason)
            ]
            if review["attribution_checks"]:
                additional = {source_id for source_id in by_id
                              if source_id not in selected and (source_id in dialogue or source_id in endings)}
                pending = list(additional)
                while pending:
                    source_id = pending.pop()
                    for linked in links[source_id] - selected - additional:
                        additional.add(linked)
                        pending.append(linked)
                review["attribution_contexts"] = [
                    {**with_context(source_id), "cited": False,
                     "corrected_by": corrected_by.get(source_id, [])}
                    for source_id in by_id
                    if source_id in additional
                ]
    return review
