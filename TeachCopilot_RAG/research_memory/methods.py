"""Independent sequential memories and deterministic, budgeted selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .contracts import ContractValidationError, extract_schema, validate_records, visible_references
from .dataset import require


METHODS = ("profile_only", "tail", "summary", "structured")
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


class Memory:
    def __init__(self, method: str, call, budget: int, topic_id: str):
        require(method in METHODS and budget >= 128, "Invalid method or budget below 128 characters")
        self.method, self.call, self.budget = method, call, budget
        self.topic_id = topic_id
        self.sources: dict[str, dict] = {}
        self.summary = ""
        self.records: list[dict] = []
        self.blocks: list[tuple[str, str]] = []
        self.events: list[dict] = []

    def update(self, lesson: dict) -> None:
        """Accept one lesson only. No request or reference answer enters updates."""
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
            new_records = validate_records(text, self.sources, set(new_sources), self.topic_id, self.records)
            for raw in new_records:
                record = {**raw, "record_id": f"r{len(self.records) + 1:04d}",
                          "observed_on": lesson["observed_on"], "corrected_by_record_ids": []}
                # Ordinary learning progress never changes an earlier observation.
                if record["kind"] == "source_correction":
                    old = next(r for r in self.records if r["record_id"] == record["corrects_record_id"])
                    old["corrected_by_record_ids"].append(record["record_id"])
                self.records.append(record)

    def context(self, topic_id: str) -> Context:
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
