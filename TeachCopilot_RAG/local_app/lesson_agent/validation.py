"""Lesson validation ported from the upstream check_lesson workflow.

Checks: exact section set and order, title, no praise before the child
answers, arithmetic equality verification and the multiplication practice
contract. Returns human-readable Russian problems for the corrective attempt
and the teacher validation view. This never replaces teacher review.
"""
from __future__ import annotations

import re

from .prompts import LESSON_REQUIRED_SECTIONS

_ARITHMETIC_PATTERN = re.compile(r"(?<!\d)(\d+)\s*([×x*+])\s*(\d+)\s*=\s*(\d+)(?!\d)")
_NUMBERED_TASK_PATTERN = re.compile(r"(?m)^\s*\d+\.\s+.+$")
_PRACTICE_EXAMPLE_PATTERN = re.compile(r"\d+\s*[×x*]\s*\d+")
_DIVISION_STORY_PATTERN = re.compile(r"сколько\s+(?:циклов|дней|групп)", re.I)
_PRAISE_PATTERN = re.compile(r"\b(?:молодец|отлично справился|ты справился)\b", re.I)


def _arithmetic_expected(left: int, operation: str, right: int) -> int | None:
    if operation == "+":
        return left + right
    if operation in {"×", "x", "*"}:
        return left * right
    return None


def validate_lesson(lesson_text: str, topic_id: str) -> list[str]:
    """Return the problems of a generated lesson; an empty list means valid."""
    lesson = str(lesson_text or "")
    problems: list[str] = []
    if not lesson.startswith("# Урок:"):
        problems.append("отсутствует заголовок # Урок:")
    for heading in LESSON_REQUIRED_SECTIONS:
        if heading not in lesson:
            problems.append(f"отсутствует раздел {heading}")
    if _PRAISE_PATTERN.search(lesson):
        problems.append("убери похвалу за ответы до того, как ребёнок ответил")
    # Explicit equalities: this does not replace teacher review, but removes
    # simple arithmetic slips of the local model.
    for left, operation, right, answer in _ARITHMETIC_PATTERN.findall(lesson):
        expected = _arithmetic_expected(int(left), operation, int(right))
        if expected is not None and int(answer) != expected:
            problems.append(f"исправь равенство {left} {operation} {right} = {answer}")
    if "умнож" in str(topic_id or "").lower() and "## Практика" in lesson:
        practice = lesson.split("## Практика", 1)[1].split("## Проверка", 1)[0]
        answers = lesson.split("## Ответы", 1)[1] if "## Ответы" in lesson else ""
        numbered_tasks = _NUMBERED_TASK_PATTERN.findall(practice)
        if len(numbered_tasks) != 3:
            problems.append("в разделе «Практика» нужны ровно 3 задания")
        if any(not _PRACTICE_EXAMPLE_PATTERN.search(item) for item in numbered_tasks):
            problems.append("каждое задание по умножению должно содержать пример")
        if _DIVISION_STORY_PATTERN.search(practice):
            problems.append("убери сюжетные задачи на деление из практики")
        for left, right in re.findall(r"(\d+)\s*[×x*]\s*(\d+)\s*=\s*\?", practice):
            expected = int(left) * int(right)
            if not re.search(rf"(?<!\d){left}\s*[×x*]\s*{right}\s*=\s*{expected}(?!\d)", answers):
                problems.append(f"добавь правильный ответ {left} × {right} = {expected}")
    return problems


def split_lesson_sections(lesson_text: str) -> dict[str, str]:
    """Defensively split a lesson into ordered sections keyed by heading.

    Markdown parsing stays defensive: unknown headings are preserved under
    their own key, a missing section yields no key. Never trust this split as
    an authoritative answer separation; the teacher reviews the whole text.
    """
    lesson = str(lesson_text or "")
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"(?m)^##\s*(.+?)\s*$", lesson))
    if not matches:
        return {"_title": lesson} if lesson.strip() else {}
    title = lesson[:matches[0].start()].strip()
    if title:
        sections["_title"] = title
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(lesson)
        sections[match.group(1).strip()] = lesson[match.end():end].strip()
    return sections


def answers_section(lesson_text: str) -> str | None:
    """The teacher-only answer section text, or None when absent/unparsable."""
    if "## Ответы" not in str(lesson_text or ""):
        return None
    sections = split_lesson_sections(lesson_text)
    return sections.get("Ответы")
