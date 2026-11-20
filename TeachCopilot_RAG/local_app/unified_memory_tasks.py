"""Deterministic task catalog for saved unified lesson versions (client A).

The catalog is derived once from explicit exercises and questions inside the
validated lesson sections and stored on the lesson row itself. Task ids are
qualified by the saved lesson version and the section/position, so they are
stable across reads, retries and re-attachments. Parsing is conservative: a
line that is not an explicit numbered exercise or a standalone question never
becomes a task, and an unparsable section yields no tasks at all.

Answers never enter the catalog: answer keys stay in the lesson's teacher-only
``answers_text``/answer sections where the existing redaction rules apply.
"""
from __future__ import annotations

import re

TASK_CATALOG_RECIPE = "unified_task_catalog_v1"

# Lesson sections that can carry tasks, mapped to guided-flow stages.
STAGE_BY_HEADING = {
    "Разминка": "warmup",
    "Объяснение": "explanation",
    "Практика": "practice",
    "Проверка": "check",
}

_HEADING_RE = re.compile(r"(?m)^##\s*(?P<title>.+?)\s*$")
_NUMBERED_RE = re.compile(r"^[ \t]*(?P<number>\d+)[.)][ \t]+(?P<body>\S.*?)[ \t]*$", re.MULTILINE)
_QUESTION_RE = re.compile(r"^[ \t]*(?P<body>[^\-\*\d#>\s].*?\?)[ \t]*$", re.MULTILINE)

MAX_PROMPT_CHARS = 500


def split_sections(lesson_text: str):
    """Defensive split into ordered ``## `` sections (title, body, offsets)."""
    sections = []
    matches = list(_HEADING_RE.finditer(lesson_text or ""))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(lesson_text)
        sections.append({"title": match.group("title").strip(),
                         "body": lesson_text[match.end():end],
                         "start": start, "end": end,
                         "body_start": match.end()})
    return sections


def _task_candidates(section):
    """Explicit exercises/questions of one section, with lesson-text offsets."""
    body = section["body"]
    base = section["body_start"]
    found = []
    for match in _NUMBERED_RE.finditer(body):
        found.append({"number": int(match.group("number")),
                      "prompt_text": match.group("body").strip(),
                      "start": base + match.start("body"), "end": base + match.end("body")})
    for match in _QUESTION_RE.finditer(body):
        # A numbered line was already captured; a bare question line becomes a
        # task only when it is a real standalone question.
        if any(match.start() >= task["start"] and match.end() <= task["end"]
               for task in found):
            continue
        prompt_text = match.group("body").strip()
        if len(prompt_text) < 8:
            continue
        found.append({"number": None, "prompt_text": prompt_text,
                      "start": base + match.start("body"), "end": base + match.end("body")})
    found.sort(key=lambda item: item["start"])
    return found


def build_task_catalog(lesson_id, lesson_text):
    """Build the immutable catalog for one saved lesson version.

    Returns ``{"recipe": ..., "items": [...]}``. Empty ``items`` is honest:
    the lesson version simply has no parseable explicit exercise or question.
    """
    items = []
    for section in split_sections(lesson_text or ""):
        stage = STAGE_BY_HEADING.get(section["title"])
        if stage is None:
            continue
        for candidate in _task_candidates(section):
            prompt_text = candidate["prompt_text"][:MAX_PROMPT_CHARS]
            items.append({
                "task_id": f"task:{lesson_id}:{stage}:{len(items) + 1}",
                "stage": stage,
                "position": len(items) + 1,
                "prompt_text": prompt_text,
                "section": section["title"],
                "number": candidate["number"],
                "offsets": {"start": candidate["start"], "end": candidate["end"]},
                "source_refs": [{"kind": "lesson_document", "lesson_id": str(lesson_id),
                                 "section": section["title"],
                                 "offsets": {"start": candidate["start"],
                                             "end": candidate["end"]}}],
            })
    return {"recipe": TASK_CATALOG_RECIPE, "items": items}


def catalog_task_ids(catalog):
    if not isinstance(catalog, dict):
        return set()
    return {item["task_id"] for item in (catalog.get("items") or [])
            if isinstance(item, dict) and isinstance(item.get("task_id"), str)}
