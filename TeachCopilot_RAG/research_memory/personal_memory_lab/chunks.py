"""Make source fragments without splitting a teacher question and child answer."""
from __future__ import annotations

from .archive import canonical


def lesson_chunks(lesson, max_bytes=6000, hard_max_bytes=None):
    hard_max_bytes = max_bytes if hard_max_bytes is None else hard_max_bytes
    if not 0 < max_bytes <= hard_max_bytes:
        raise ValueError("Invalid source-fragment limits")
    turns = lesson["turns"]
    groups, index = [], 0
    while index < len(turns):
        group = [turns[index]]
        if (turns[index]["role"] in {"teacher", "assistant"} and index + 1 < len(turns)
                and turns[index + 1]["role"] == "student"):
            group.append(turns[index + 1])
            index += 1
        groups.append(group)
        index += 1
    result, current = [], []
    def block(items):
        return {"lesson_id": lesson["lesson_id"], "observed_on": lesson["observed_on"],
                "topic_ids": lesson.get("topic_ids", []), "turns": items}
    for group in groups:
        if len(canonical(block(group)).encode("utf-8")) > hard_max_bytes:
            raise ValueError("A question-answer pair exceeds the source-fragment limit")
        if current and len(canonical(block(current + group)).encode("utf-8")) > max_bytes:
            result.append(block(current))
            current = []
        current.extend(group)
    if current:
        result.append(block(current))
    return result
