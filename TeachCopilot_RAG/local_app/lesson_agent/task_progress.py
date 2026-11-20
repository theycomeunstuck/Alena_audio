"""Task-level lesson progress driven by the facade's task catalog.

The catalog comes from ``UnifiedMemoryService.lesson_tasks`` (peer-owned
backend). This module only consumes it: it resolves the pending task, verifies
supported arithmetic attempts deterministically, and never invents progress.
A model stage marker, silence or a short non-answer such as «хз» never
completes a task; an unverifiable attempt keeps the work pending. This is
exercise progress, not a new numerical mastery model.
"""
from __future__ import annotations

import re

_STAGE_ORDER = ("warmup", "explanation", "practice", "review", "completed")

# A multiplication attempt like «24 × 3 = 72» (×, x/X or Cyrillic х/Х, *).
_ATTEMPT_PATTERN = re.compile(r"(\d{1,4})\s*[×xXхХ*]\s*(\d{1,4})\s*=\s*(\d{1,6})")
# The task prompt's left side, e.g. «24 × 3 = ?» or «Сколько будет 24 × 3?».
_TASK_EXPRESSION = re.compile(r"(\d{1,4})\s*[×xXхХ*]\s*(\d{1,4})")
# A bare-number answer («12») to the task's expression.
_BARE_NUMBER = re.compile(r"\s*[=:]?\s*(\d{1,6})\s*\.?\s*")

_NON_ANSWERS = {"хз", "хз.", "хз,", "не знаю", "незнаю", "не знаю...", "?", "??",
                "...", "—", "-", "–", "idk", "dunno", "хз :)", "хз)", "ничего"}

_MAX_ATTEMPT_TEXT = 4000


def normalize_non_answer(text: str) -> bool:
    """A silence or short non-answer («хз», «?», «...») is never an attempt."""
    stripped = (text or "").strip().lower()
    if not stripped or stripped in _NON_ANSWERS:
        return True
    return len(stripped) <= 2


def parse_task_expression(task_prompt: str) -> tuple[int, int] | None:
    match = _TASK_EXPRESSION.search(str(task_prompt or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_attempt(text: str) -> tuple[int, int, int] | None:
    """The first multiplication equality in the learner's message."""
    match = _ATTEMPT_PATTERN.search(str(text or "")[:_MAX_ATTEMPT_TEXT])
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def task_attempt_outcome(task_prompt: str, message_text: str) -> str:
    """``completed`` | ``pending`` | ``uncertain`` for the student's latest turn.

    ``completed`` requires a deterministic arithmetic check that matches the
    current task's factors. A bare number («12») is a valid attempt at the
    task's expression. Anything else — wrong result, different exercise,
    unverifiable wording — keeps the task pending; nothing is fabricated.
    """
    expression = parse_task_expression(task_prompt)
    bare = _BARE_NUMBER.fullmatch(str(message_text or ""))
    if expression is not None and bare is not None:
        return "completed" if expression[0] * expression[1] == int(bare.group(1)) else "pending"
    if normalize_non_answer(message_text):
        return "pending"
    attempt = parse_attempt(message_text)
    if expression is None or attempt is None:
        return "uncertain"
    left, right = expression
    factor_a, factor_b, result = attempt
    if {factor_a, factor_b} != {left, right}:
        # A real answer to a different exercise: not our task's completion.
        return "uncertain"
    return "completed" if factor_a * factor_b == result else "pending"


def stage_rank(stage: str | None) -> int:
    try:
        return _STAGE_ORDER.index(str(stage or ""))
    except ValueError:
        return len(_STAGE_ORDER)


def resolve_pending_task(items: list[dict], completed_task_ids, pending_task_id=None) -> dict | None:
    """The current task: the persisted one, else the next incomplete item.

    Catalog order decides the fallback; already completed or unknown task
    ids in the state are filtered against the catalog. ``None`` means the
    catalog has no remaining resolvable task — never a fabricated one.
    """
    valid = {item.get("task_id") for item in items if item.get("task_id")}
    completed = {task_id for task_id in (completed_task_ids or []) if task_id in valid}
    if pending_task_id and pending_task_id in valid and pending_task_id not in completed:
        return next(item for item in items if item.get("task_id") == pending_task_id)
    remaining = [item for item in items if item.get("task_id") not in completed]
    if not remaining:
        return None
    return sorted(remaining, key=lambda item: (stage_rank(item.get("stage")),
                                               item.get("position") or 0))[0]


def task_prompt_line(task: dict | None) -> str:
    """The original catalog wording for the guided continuation prompt."""
    if not task:
        return ""
    position = task.get("position")
    prompt = str(task.get("prompt_text") or "").strip()
    prefix = f"Задание {position}: " if position else "Текущее задание: "
    return prefix + prompt if prompt else ""
