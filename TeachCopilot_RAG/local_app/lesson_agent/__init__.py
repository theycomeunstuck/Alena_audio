"""Lesson workflows ported from D3strovich/child-lesson-agent (43e6e36).

The package keeps the upstream generate → validate → one corrective attempt
lesson workflow, the guided lesson chat and the four-heading session summary
presentation. Model access is always an injected ``model_call(messages,
purpose, schema)`` callable bound to the existing local runtime, so LM Studio
never becomes a second mandatory server.
"""
from .guided import (GUIDED_STAGE_LABELS, GUIDED_STAGES, LESSON_START_REQUEST,
                     guided_history, guided_messages, guided_reply,
                     parse_stage_marker, student_lesson_content)
from .generation import generate_lesson
from .prompts import (LESSON_GENERATE_PURPOSE, GUIDED_CHAT_PURPOSE,
                      LESSON_REQUIRED_SECTIONS, SUMMARY_HEADINGS,
                      lesson_messages, summary_messages)
from .summary import summary_sections, validate_summary
from .validation import validate_lesson

__all__ = [
    "GUIDED_CHAT_PURPOSE",
    "GUIDED_STAGES",
    "GUIDED_STAGE_LABELS",
    "LESSON_GENERATE_PURPOSE",
    "LESSON_REQUIRED_SECTIONS",
    "LESSON_START_REQUEST",
    "SUMMARY_HEADINGS",
    "generate_lesson",
    "guided_history",
    "guided_messages",
    "guided_reply",
    "lesson_messages",
    "parse_stage_marker",
    "student_lesson_content",
    "summary_messages",
    "summary_sections",
    "validate_lesson",
    "validate_summary",
]
