"""Independent sequential memories and deterministic, budgeted selection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re

from .contracts import (ContractValidationError, extract_schema, validate_records_partial,
                        visible_references)
from .contracts_v2 import extract_schema_v2, record_use_policy, validate_records_v2_partial
from .dataset import require


METHODS = ("profile_only", "tail", "summary", "structured")
APPLICATION_METHODS = METHODS + ("structured_v2",)
V2_RECORD_TYPES = {
    "child_error", "child_correction", "supported_attempt", "uncertain_observation",
    "learning_success", "unfinished_work", "source_correction",
}
V2_PROFILE_BUCKETS = {
    "interest_candidate": "interest_candidates",
    "session_preference_candidate": "session_preferences",
    "incidental_fact": "incidental_facts",
}


def _v2_bucket(record: dict) -> str | None:
    kind = record.get("record_type")
    if kind in V2_RECORD_TYPES:
        return "topic_memory"
    return V2_PROFILE_BUCKETS.get(kind)


def _v2_record_link_is_selectable(source: dict, target: dict, relation: str) -> bool:
    """Keep retrieval groups inside one semantic family, even for stale states."""
    source_type, target_type = source.get("record_type"), target.get("record_type")
    same_topic = source.get("topic_id") == target.get("topic_id")
    if _v2_bucket(source) != _v2_bucket(target) or _v2_bucket(source) is None:
        return False
    if relation == "corrects_answer":
        return source_type == "child_correction" and target_type == "child_error" and same_topic
    if relation == "source_correction":
        return source_type == "source_correction" and target_type in V2_RECORD_TYPES and same_topic
    if relation == "same_pattern_as":
        return source_type == target_type and same_topic
    if relation in {"supported_by", "rechecks"}:
        return source_type in V2_RECORD_TYPES and target_type in V2_RECORD_TYPES and same_topic
    if relation == "mentioned_again":
        return source_type == target_type and source_type in V2_PROFILE_BUCKETS
    if relation == "related_to":
        return source_type == target_type and same_topic
    return False


def _v2_context_projection(record: dict) -> dict:
    """Compact only the next-step view; keep complete records in persisted memory."""
    fields = (
        "record_id", "record_type", "topic_id", "context", "interpretation",
        "interpretation_status", "support_level", "support_evidence",
        "independent_recheck", "recheck_evidence", "evidence", "links",
        "corrects_record_id", "corrected_by_record_ids", "interest_operation",
        "interest", "preference", "fact", "claim_status", "retention_class",
        "occurred_on", "last_seen",
    )

    def compact_text(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        prefix = value[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
        return prefix + "…"

    def compact_evidence(value: dict | None) -> dict | None:
        if not isinstance(value, dict):
            return None
        result = {"turn_id": value["turn_id"]}
        if "quote" in value:
            result["quote"] = compact_text(str(value["quote"]), 96)
        return result

    projected = {key: record[key] for key in fields if key in record
                 and (record[key] is not None or key == "topic_id")}
    if record.get("record_type") in V2_PROFILE_BUCKETS:
        # Profile records may cite a turn that also contains a temporary
        # session preference. Keep the typed value and source ID, but do not
        # leak unrelated source text or free-form interpretation across buckets.
        profile_fields = {"record_id", "record_type", "topic_id", "interpretation_status",
                          "interest_operation", "interest", "preference", "fact",
                          "claim_status", "retention_class", "occurred_on", "last_seen", "evidence"}
        projected = {key: value for key, value in projected.items() if key in profile_fields}
        projected["evidence"] = [{"turn_id": item["turn_id"]} for item in record.get("evidence", [])
                                 if isinstance(item, dict) and item.get("turn_id")]
    for field, limit in (("context", 90), ("interpretation", 110)):
        if field in projected:
            projected[field] = compact_text(str(projected[field]), limit)
    for field in ("support_evidence", "recheck_evidence"):
        if field in projected:
            projected[field] = compact_evidence(projected[field])
    if "evidence" in projected:
        condition_turn_ids = {
            value.get("turn_id") for value in (record.get("support_evidence"), record.get("recheck_evidence"))
            if isinstance(value, dict)
        }
        main_evidence = [item for item in projected["evidence"]
                         if isinstance(item, dict) and item.get("turn_id") not in condition_turn_ids]
        if not main_evidence:
            main_evidence = [item for item in projected["evidence"] if isinstance(item, dict)][:1]
        projected["evidence"] = [compact_evidence(item) for item in main_evidence]
        if not projected["evidence"]:
            projected.pop("evidence", None)
    if not projected.get("links"):
        projected.pop("links", None)
    if not projected.get("corrected_by_record_ids"):
        projected.pop("corrected_by_record_ids", None)
    return projected


def _v2_previous_projection(records: list[dict]) -> list[dict]:
    """Keep prior state useful for updates without resending every stored quote."""
    result = []
    for record in records:
        row = {
            "record_id": record.get("record_id"),
            "record_type": record.get("record_type"),
            "topic_id": record.get("topic_id"),
            "last_seen": record.get("last_seen"),
        }
        if record.get("record_type") not in V2_PROFILE_BUCKETS:
            row.update({
                "context": str(record.get("context") or "")[:80],
                "interpretation": str(record.get("interpretation") or "")[:100],
                "interpretation_status": record.get("interpretation_status"),
                "support_level": record.get("support_level"),
                "independent_recheck": record.get("independent_recheck"),
            })
        for field, limit in (("interest_operation", 24), ("interest", 100), ("preference", 100),
                             ("fact", 130), ("claim_status", 16), ("retention_class", 32),
                             ("corrects_record_id", 64)):
            if record.get(field):
                value = record[field]
                row[field] = str(value)[:limit] if isinstance(value, str) else value
        result.append(row)
    return result
PROMPT_DIR = Path(__file__).parent / "prompts"


def prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def messages(name: str, payload: dict) -> list[dict[str, str]]:
    return [{"role": "system", "content": prompt(name)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def source_turns(lesson: dict) -> dict[str, dict]:
    return {turn["turn_id"]: {**turn, "observed_on": lesson["observed_on"]} for turn in lesson["turns"]}


def _chunks(text: str, width: int) -> list[str]:
    # Split before selection: one long message must not empty the tail baseline.
    result = []
    for paragraph in re.split(r"\n\s*\n", text):
        current = ""
        for word in paragraph.split():
            while len(word) > width:
                if current:
                    result.append(current)
                    current = ""
                result.append(word[:width])
                word = word[width:]
            if len(current) + bool(current) + len(word) > width:
                result.append(current)
                current = ""
            current += (" " if current else "") + word
        if current:
            result.append(current)
    return result


def _whole_sentences(text: str, budget: int) -> str:
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())
    kept = []
    for sentence in sentences:
        candidate = " ".join(kept + [sentence])
        if len(candidate) <= budget:
            kept.append(sentence)
        else:
            break
    return " ".join(kept)


@dataclass
class Context:
    text: str
    evidence_ids: set[str]
    action_evidence_ids: set[str] | None = None


class Memory:
    def __init__(self, method: str, call, budget: int, topic_id: str):
        require(method in APPLICATION_METHODS and budget >= 128, "Invalid method or budget below 128 characters")
        self.method, self.call, self.budget = method, call, budget
        self.topic_id = topic_id
        self.sources: dict[str, dict] = {}
        self.summary = ""
        self.records: list[dict] = []
        self.blocks: list[tuple[str, str]] = []
        self.events: list[dict] = []

    def update(self, lesson: dict) -> None:
        """Accept one lesson only. No request or reference answer enters updates."""
        previous_sources = self.sources.copy()
        previous_summary = self.summary
        previous_records = deepcopy(self.records)
        previous_blocks = list(self.blocks)
        try:
            self._update(lesson)
        except BaseException:
            # A rejected or interrupted update must not reserve its source IDs
            # or partially modify existing correction links. Keep diagnostic
            # events so the failure remains visible to the caller.
            self.sources.clear()
            self.sources.update(previous_sources)
            self.summary = previous_summary
            self.records[:] = previous_records
            self.blocks[:] = previous_blocks
            raise

    def _update(self, lesson: dict) -> None:
        new_sources = source_turns(lesson)
        require(not set(new_sources) & set(self.sources), "Source IDs must be immutable and unique")
        self.sources.update(new_sources)
        if self.method == "tail":
            for turn_id, turn in new_sources.items():
                header = f"[{turn_id} | {turn['observed_on']} | {turn['role']}] "
                width = max(1, min(320, self.budget - len(header)))
                for chunk in _chunks(turn["text"], width):
                    self.blocks.append((turn_id, header + chunk))
        elif self.method == "summary":
            text = self.call(messages("summary", {"previous_summary": self.summary,
                             "new_lesson": lesson, "budget_chars": self.budget}), "summary", None)
            require(isinstance(text, str) and bool(text.strip()), "Empty summary")
            if len(text) > self.budget:
                self.events.append({"lesson_id": lesson["lesson_id"], "event": "summary_exceeds_budget",
                                    "response_chars": len(text), "budget_chars": self.budget})
                raise ContractValidationError("Summary exceeds the selected memory budget", "summary_exceeds_budget")
            self.summary = text
        elif self.method == "structured":
            text = self.call(messages("extract", {"allowed_topic_id": self.topic_id, "previous_records": self.records,
                             "new_lesson": lesson}), "extract", extract_schema(
                                 set(self.sources), self.topic_id, {r["record_id"] for r in self.records}))
            new_records, validation_events = validate_records_partial(
                text, self.sources, set(new_sources), self.topic_id, self.records)
            self.events.extend({"lesson_id": lesson["lesson_id"], **event} for event in validation_events)
            for raw in new_records:
                record = {**raw, "record_id": f"r{len(self.records) + 1:04d}",
                          "observed_on": lesson["observed_on"], "corrected_by_record_ids": []}
                # Ordinary learning progress never changes an earlier observation.
                if record["kind"] == "source_correction":
                    old = next(r for r in self.records if r["record_id"] == record["corrects_record_id"])
                    old["corrected_by_record_ids"].append(record["record_id"])
                self.records.append(record)
        elif self.method == "structured_v2":
            allowed_topics = (set(lesson["allowed_topic_ids"])
                              if "allowed_topic_ids" in lesson else {self.topic_id})
            previous_records = (self.records if allowed_topics else [
                record for record in self.records if record["record_type"] in V2_PROFILE_BUCKETS])
            available_source_ids = set(new_sources)
            for previous in previous_records:
                for field in ("evidence", "support_evidence", "recheck_evidence"):
                    evidence = previous.get(field)
                    if isinstance(evidence, list):
                        available_source_ids.update(item["turn_id"] for item in evidence
                                                    if isinstance(item, dict) and item.get("turn_id"))
                    elif isinstance(evidence, dict) and evidence.get("turn_id"):
                        available_source_ids.add(evidence["turn_id"])
            available_sources = {source_id: self.sources[source_id] for source_id in available_source_ids
                                 if source_id in self.sources}
            text = self.call(messages("extract_v2", {
                "allowed_topic_ids": sorted(allowed_topics),
                "previous_records": _v2_previous_projection(previous_records),
                "new_lesson": lesson,
            }), "extract_v2", extract_schema_v2(
                available_source_ids, allowed_topics, {r["record_id"] for r in previous_records}))
            new_records, warnings = validate_records_v2_partial(
                text, available_sources, set(new_sources), allowed_topics, previous_records,
                new_source_order=list(new_sources))
            self.events.extend({"lesson_id": lesson["lesson_id"], **warning} for warning in warnings)
            for raw in new_records:
                record_id = f"r{len(self.records) + 1:04d}"
                evidence_dates = [self.sources[item["turn_id"]].get("observed_on")
                                  for item in raw["evidence"]]
                evidence_dates = [value for value in evidence_dates if isinstance(value, str) and value]
                record = {
                    **raw,
                    "record_id": record_id,
                    "occurred_on": min(evidence_dates) if evidence_dates else lesson["observed_on"],
                    "last_seen": max(evidence_dates) if evidence_dates else lesson["observed_on"],
                    "evidence_count": len({item["turn_id"] for item in raw["evidence"]}),
                    "corrected_by_record_ids": [],
                    "use_policy": record_use_policy(raw["record_type"]),
                }
                if record["record_type"] == "source_correction":
                    old = next(r for r in self.records if r["record_id"] == record["corrects_record_id"])
                    old["corrected_by_record_ids"].append(record_id)
                self.records.append(record)

    def context(self, topic_id: str, *, include_incidental_facts: bool = False) -> Context:
        if self.method == "profile_only":
            return Context("", set())
        if self.method == "summary":
            return Context(self.summary, visible_references(self.summary, set(self.sources)))
        if self.method == "tail":
            selected, size = [], 0
            for turn_id, block in reversed(self.blocks):
                added = len(block) + bool(selected)
                if size + added <= self.budget:
                    selected.append((turn_id, block))
                    size += added
                else:
                    break
            selected.reverse()
            return Context("\n".join(x[1] for x in selected), {x[0] for x in selected})
        if self.method == "structured_v2":
            return self._context_v2(topic_id, include_incidental_facts=include_incidental_facts)
        ranked = sorted(self.records, key=lambda r: (
            r["topic_id"] == topic_id, r["observed_on"], r["record_id"]), reverse=True)
        # Corrections and every related original form one indivisible selection group.
        # A correction chain is safe only when all of its records fit together.
        by_id = {r["record_id"]: r for r in self.records}
        links = {record_id: set() for record_id in by_id}
        for record in self.records:
            target = record.get("corrects_record_id")
            if target is not None:
                require(target in by_id, "Correction target is absent from memory")
                links[record["record_id"]].add(target)
                links[target].add(record["record_id"])
        selected, visited = [], set()
        for record in ranked:
            if record["record_id"] in visited:
                continue
            pending, group_ids = [record["record_id"]], set()
            while pending:
                record_id = pending.pop()
                if record_id not in group_ids:
                    group_ids.add(record_id)
                    pending.extend(links[record_id] - group_ids)
            visited.update(group_ids)
            group = sorted((by_id[record_id] for record_id in group_ids),
                           key=lambda r: (r["observed_on"], r["record_id"]))
            candidate = selected + group
            if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= self.budget:
                selected.extend(group)
            else:
                record_chars = len(json.dumps(group, ensure_ascii=False, separators=(",", ":")))
                event = {"event": "record_skipped", "record_id": record["record_id"],
                         "reason": "record_exceeds_budget" if record_chars > self.budget else "remaining_budget_exceeded",
                         "topic_id": topic_id, "budget_chars": self.budget, "record_chars": record_chars,
                         "group_record_ids": [r["record_id"] for r in group]}
                if event not in self.events:
                    self.events.append(event)
        text = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
        ids = {x for record in selected for x in record["evidence_turn_ids"]}
        return Context(text, ids)

    def _context_v2(self, topic_id: str, *, include_incidental_facts: bool = False) -> Context:
        """Keep academic events and personal candidates visibly separated."""
        bucket_order = ("topic_memory", "interest_candidates", "session_preferences", "incidental_facts")
        payload = {"record_schema_version": 2, "topic_id": topic_id,
                   "session_only_records_excluded": sum(
                       1 for record in self.records if record.get("retention_class") == "session_only"),
                   **{key: [] for key in bucket_order}}
        relevant = [record for record in self.records
                    if ((record["record_type"] in V2_RECORD_TYPES and record["topic_id"] == topic_id)
                        or (record["record_type"] in V2_PROFILE_BUCKETS
                            and (include_incidental_facts or record["record_type"] != "incidental_fact")))
                    and record.get("retention_class") != "session_only"]

        by_id = {record["record_id"]: record for record in relevant}
        linked: dict[str, set[str]] = {record_id: set() for record_id in by_id}
        by_evidence_turn: dict[str, list[str]] = {}
        for record in relevant:
            for item in record["evidence"]:
                by_evidence_turn.setdefault(item["turn_id"], []).append(record["record_id"])
        for record in relevant:
            record_id = record["record_id"]
            target = record.get("corrects_record_id")
            if (target in by_id and record.get("record_type") == "source_correction"
                    and _v2_record_link_is_selectable(record, by_id[target], "source_correction")):
                linked[record_id].add(target)
                linked[target].add(record_id)
            for edge in record["links"]:
                if edge["target_kind"] == "record" and edge["target_id"] in by_id:
                    other = edge["target_id"]
                    if _v2_record_link_is_selectable(record, by_id[other], edge["relation"]):
                        linked[record_id].add(other)
                        linked[other].add(record_id)
                elif edge["target_kind"] == "turn" and edge["relation"] == "corrects_answer":
                    for other in by_evidence_turn.get(edge["target_id"], []):
                        if _v2_record_link_is_selectable(
                                record, by_id[other], "corrects_answer"):
                            linked[record_id].add(other)
                            linked[other].add(record_id)

        groups, visited = [], set()
        for record in relevant:
            record_id = record["record_id"]
            if record_id in visited:
                continue
            pending, group_ids = [record_id], set()
            while pending:
                current = pending.pop()
                if current not in group_ids:
                    group_ids.add(current)
                    pending.extend(linked[current] - group_ids)
            visited.update(group_ids)
            group = sorted((by_id[item] for item in group_ids),
                           key=lambda row: (row["occurred_on"], row["record_id"]))
            groups.append(group)

        queues = {key: [] for key in bucket_order}
        for group in groups:
            kind = group[0]["record_type"]
            bucket = "topic_memory" if kind in V2_RECORD_TYPES else V2_PROFILE_BUCKETS[kind]
            queues[bucket].append(group)
        for key in bucket_order:
            queues[key].sort(key=lambda group: max(
                (record["last_seen"], record["record_id"]) for record in group), reverse=True)
        # Repeated mentions of the same candidate do not need duplicate slots in
        # the next-step prompt. The complete dated records remain in persisted state.
        seen_interests = set()
        unique_interests = []
        for group in queues["interest_candidates"]:
            if len(group) != 1:
                unique_interests.append(group)
                continue
            record = group[0]
            key = (record.get("interest_operation"),
                   " ".join(str(record.get("interest") or "").casefold().split()))
            if key[1] and key in seen_interests:
                self.events.append({"event": "record_skipped",
                                    "record_ids": [record["record_id"]],
                                    "record_types": [record["record_type"]],
                                    "reason": "duplicate_interest_in_context",
                                    "topic_id": topic_id, "budget_chars": self.budget})
                continue
            seen_interests.add(key)
            unique_interests.append(group)
        queues["interest_candidates"] = unique_interests

        selected_ids: set[str] = set()
        max_groups = max((len(queue) for queue in queues.values()), default=0)
        for index in range(max_groups):
            for bucket in bucket_order:
                queue = queues[bucket]
                if index >= len(queue):
                    continue
                group = queue[index]
                candidate_ids = selected_ids | {record["record_id"] for record in group}
                candidate = {"record_schema_version": 2, "topic_id": topic_id,
                             "session_only_records_excluded": payload["session_only_records_excluded"],
                             **{key: [] for key in bucket_order}}
                for record_id in sorted(candidate_ids, key=lambda value: (
                        by_id[value]["occurred_on"], value)):
                    record = by_id[record_id]
                    group_kind = record["record_type"]
                    group_bucket = "topic_memory" if group_kind in V2_RECORD_TYPES else V2_PROFILE_BUCKETS[group_kind]
                    candidate[group_bucket].append(_v2_context_projection(record))
                if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= self.budget:
                    selected_ids = candidate_ids
                else:
                    event = {"event": "record_skipped", "record_ids": [r["record_id"] for r in group],
                             "record_types": [r["record_type"] for r in group],
                             "reason": "record_exceeds_budget" if not selected_ids else "remaining_budget_exceeded",
                             "topic_id": topic_id, "budget_chars": self.budget}
                    if event not in self.events:
                        self.events.append(event)

        selected = [by_id[record_id] for record_id in selected_ids]
        for record in selected:
            bucket = "topic_memory" if record["record_type"] in V2_RECORD_TYPES else V2_PROFILE_BUCKETS[record["record_type"]]
            payload[bucket].append(_v2_context_projection(record))
        payload["topic_memory"].sort(key=lambda row: (row["occurred_on"], row["record_id"]))
        for bucket in ("interest_candidates", "session_preferences", "incidental_facts"):
            payload[bucket].sort(key=lambda row: (row["last_seen"], row["record_id"]))
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        require(len(text) <= self.budget, "Selected V2 memory context exceeds its character budget")
        evidence_ids = {item["turn_id"] for record in selected for item in record["evidence"]}
        action_evidence_ids = {
            item["turn_id"] for record in selected if record["record_type"] in V2_RECORD_TYPES
            for item in record["evidence"]
        }
        return Context(text, evidence_ids, action_evidence_ids)
