"""Validate only histories and requests; reference answers are never inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identifier(value: object) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value))


def nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read(path: Path) -> list[dict]:
    require("gold" not in path.name.lower(), "Reference-answer files cannot be loaded by this stand")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from None
        require(isinstance(row, dict), f"{path.name}:{number}: expected object")
        rows.append(row)
    require(bool(rows), f"{path.name}: empty input")
    return rows


@dataclass
class Dataset:
    histories: dict[str, dict]
    requests: list[dict]
    split: str


def load_dataset(histories_path: str | Path, requests_path: str | Path) -> Dataset:
    histories, requests = _read(Path(histories_path)), _read(Path(requests_path))
    seen: set[str] = set()

    def unique(value: object) -> None:
        require(identifier(value) and value not in seen, "Invalid or duplicate identifier")
        seen.add(value)

    by_id = {}
    splits = set()
    for history in histories:
        unique(history.get("history_id"))
        split = history.get("split")
        require(split in {"dev", "test"}, "History split must be dev or test")
        splits.add(split)
        require(nonempty(history.get("data_origin")), "History data_origin is required")
        require(identifier(history.get("learner_id")), "Invalid learner_id")
        profile = history.get("profile_as_of_cutoff")
        require(isinstance(profile, dict) and identifier(profile.get("topic_id")), "Invalid cutoff profile")
        lessons = history.get("lessons")
        require(isinstance(lessons, list) and bool(lessons), "History needs lessons")
        previous = None
        for lesson in lessons:
            require(isinstance(lesson, dict), "Lesson must be an object")
            unique(lesson.get("lesson_id"))
            raw_date = lesson.get("observed_on")
            try:
                observed = date.fromisoformat(raw_date)
            except (TypeError, ValueError):
                raise ValueError("Invalid lesson date") from None
            require(observed.isoformat() == raw_date, "Lesson date must use YYYY-MM-DD")
            require(previous is None or observed >= previous, "Lessons are out of chronological order")
            previous = observed
            turns = lesson.get("turns")
            require(isinstance(turns, list) and bool(turns), "Lesson needs turns")
            for turn in turns:
                require(isinstance(turn, dict), "Turn must be an object")
                unique(turn.get("turn_id"))
                require(turn.get("role") in {"teacher", "student", "assistant"}, "Invalid turn role")
                require(nonempty(turn.get("text")), "Turn text must be nonempty")
        by_id[history["history_id"]] = history
    require(len(splits) == 1, "One run must not mix dev and test splits")
    for request in requests:
        unique(request.get("request_id"))
        require(request.get("history_id") in by_id, "Request refers to an unknown history")
        history = by_id[request["history_id"]]
        require(request.get("split", history["split"]) == history["split"], "Request split mismatch")
        require(identifier(request.get("topic_id")) and nonempty(request.get("text")), "Invalid request")
        require(request["topic_id"] == history["profile_as_of_cutoff"]["topic_id"], "Request topic mismatch")
    return Dataset(by_id, requests, splits.pop())
