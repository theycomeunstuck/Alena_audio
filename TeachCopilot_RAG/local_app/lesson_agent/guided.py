"""Guided lesson chat: one question at a time, server-owned lesson selection.

The student-facing reply is model-driven like the upstream guided mode; the
server keeps the stage authoritative by asking the model to end each reply
with a ``[Этап: <название>]`` marker line. The marker is parsed defensively,
stripped from the child-visible text, and recorded through the memory
backend's lesson-state interface with a compare-and-set revision. A missing
or invalid marker never advances the stage.
"""
from __future__ import annotations

from .prompts import GUIDED_CHAT_PURPOSE, LESSON_START_REQUEST, guided_messages

GUIDED_STAGES = ("not_started", "warmup", "explanation", "practice", "check", "completed")

# Russian display labels map one-to-one onto the stage enum; the model writes
# only the Russian label, the server records the enum value.
GUIDED_STAGE_LABELS = {
    "разминка": "warmup",
    "объяснение": "explanation",
    "практика": "practice",
    "проверка": "check",
    "завершено": "completed",
}

_STAGE_MARKER_MAX_LENGTH = 120
ANSWERS_HEADING = "## Ответы"


def student_lesson_content(lesson_text: str) -> str:
    """The student-facing lesson text: the answer section stays teacher-only."""
    lesson = str(lesson_text or "")
    if ANSWERS_HEADING not in lesson:
        return lesson
    return lesson.split(ANSWERS_HEADING, 1)[0].rstrip()


def guided_history(messages: list[dict[str, str]], *, guided_mode: bool = False,
                   max_turns: int = 12, max_chars: int = 4000) -> list[dict[str, str]]:
    """Upstream history handling: last-12 window, consecutive turns merged.

    The first stored assistant turn (the lesson opening) is replaced by an
    explicit continuation request so a restart re-anchors the guided lesson.
    The latest user message is not part of the returned history; the caller
    passes it separately as ``question`` so the pairing stays exact.
    """
    history: list[dict[str, str]] = []
    for turn in messages[-max_turns:]:
        content = str(turn.get("content", "")).strip()
        role = turn.get("role")
        if role not in {"user", "assistant"} or not content:
            continue
        if not history and role == "assistant":
            history.append({"role": "user",
                            "content": LESSON_START_REQUEST if guided_mode else "Продолжим учебный диалог."})
        if history and history[-1]["role"] == role:
            history[-1] = {"role": role,
                           "content": (history[-1]["content"] + "\n\n" + content)[-max_chars:]}
        else:
            history.append({"role": role, "content": content[:max_chars]})
    return history


def parse_stage_marker(text: str) -> tuple[str, str | None]:
    """Split the trailing ``[Этап: ...]`` marker from the child-visible text.

    Returns ``(visible_text, stage_enum_or_None)``. Any malformed, ambiguous
    or out-of-vocabulary marker returns the whole text unchanged and None, so
    a broken marker never advances or rewinds the stage.
    """
    reply = str(text or "").strip()
    if not reply.endswith("]"):
        return reply, None
    start = reply.rfind("[Этап:")
    if start < 0:
        return reply, None
    marker = reply[start:]
    if len(marker) > _STAGE_MARKER_MAX_LENGTH or "\n" in marker:
        return reply, None
    inner = marker[1:-1].strip()
    if not inner.startswith("Этап:") or ":" not in inner:
        return reply, None
    label = inner.split(":", 1)[1].strip().strip("«»\"'").lower()
    stage = GUIDED_STAGE_LABELS.get(label)
    if stage is None:
        return reply, None
    visible = reply[:start].rstrip()
    return visible, stage


def guided_reply(model_call, *, profile: dict, lesson_text: str, guided_mode: bool,
                 history: list[dict[str, str]], student_message: str) -> tuple[str, str | None]:
    """Generate one guided reply; returns (visible reply, stage or None).

    ``history`` and ``lesson_text`` are server-built; a browser cannot inject
    them. The stage marker line is stripped from the visible reply.
    """
    if not callable(model_call):
        raise ValueError("model_call is required for the guided lesson chat")
    question = str(student_message or "").strip()
    if not question:
        raise ValueError("Напишите сообщение перед отправкой.")
    if len(question) > 4000:
        raise ValueError("Сообщение слишком длинное: максимум 4000 символов.")
    bounded_history = guided_history(history, guided_mode=guided_mode)
    if not guided_mode:
        lesson_text = ""
    # Upstream merges the trailing history turn into the question when both
    # come from the student, so one exchange stays one prompt turn.
    if bounded_history and bounded_history[-1]["role"] == "user":
        question = bounded_history.pop()["content"] + "\n\n" + question
    messages = guided_messages(profile, str(lesson_text or ""), guided_mode,
                               bounded_history, question)
    text = model_call(messages, GUIDED_CHAT_PURPOSE, None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Модель вернула пустой или нетекстовый ответ.")
    visible, stage = parse_stage_marker(text.strip())
    return visible, stage


__all__ = ["GUIDED_STAGES", "GUIDED_STAGE_LABELS", "LESSON_START_REQUEST",
           "guided_history", "guided_messages", "guided_reply",
           "parse_stage_marker", "student_lesson_content"]
