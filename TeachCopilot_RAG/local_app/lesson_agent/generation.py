"""The lesson generation workflow: generate → validate → one corrective attempt.

Ported from the upstream LangGraph workflow. The model is reached only through
the injected ``model_call(messages, purpose, schema)`` callable, which the
worker binds to the existing local runtime with its journal, heartbeat and
admission. A lesson that still fails validation after the corrective attempt
is never stored as a draft; its diagnostics stay in the job record.
"""
from __future__ import annotations

from copy import deepcopy

from .fast_memory import fast_memory_block
from .prompts import LESSON_GENERATE_PURPOSE, lesson_messages
from .validation import validate_lesson

MAX_ATTEMPTS = 2  # the first generation plus one corrective attempt


def _invoke(model_call, messages: list[dict[str, str]]) -> str:
    text = model_call(messages, LESSON_GENERATE_PURPOSE, None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Модель вернула пустой или нетекстовый ответ.")
    return text.strip()


def _profile_payload(profile: dict, *, bounded: bool = True) -> dict:
    """A bounded, label-safe profile projection for the generation prompt."""
    if not isinstance(profile, dict):
        profile = {}
    payload = {
        "grade": profile.get("grade"),
        "language": profile.get("language"),
        "interests": [str(item) for item in (profile.get("interests") or [])[:8]
                      if isinstance(item, (str, int))],
        "learning_goal": str(profile.get("learning_goal") or profile.get("goal") or "")[:600],
        "preferred_subject": str(profile.get("preferred_subject") or profile.get("topic_title") or "")[:300],
    }
    if not bounded:
        payload.update({key: profile[key] for key in ("pseudonym",) if profile.get(key)})
    # Unknown facts stay absent: the prompt must not invite fabrication.
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def generate_lesson(model_call, *, profile: dict, topic_id: str, topic: str,
                    minutes: int, fast_memory: dict | None = None,
                    previous_lesson: dict | None = None) -> dict:
    """Run the generate → validate → one corrective attempt workflow.

    Returns ``lesson_text`` (or None), the per-attempt ``validation`` results,
    and ``attempts``. When the facade's fast memory is available its effective
    records replace the card interests/goal entirely: an empty effective field
    stays empty instead of silently resurrecting a stale card value. With no
    fast memory the legacy card projection still applies, so old callers keep
    working. Avoidance stays separate from interests.
    """
    if not callable(model_call):
        raise ValueError("model_call is required for lesson generation")
    if not 5 <= int(minutes) <= 90:
        raise ValueError("minutes must be between 5 and 90")
    prompt_profile = _profile_payload(profile)
    fast_diagnostics = None
    if isinstance(fast_memory, dict):
        # The facade already starts from profile values and replays applied
        # durable changes, so its effective fields supersede the card's.
        prompt_profile.pop("interests", None)
        prompt_profile.pop("learning_goal", None)
        block, fast_diagnostics = fast_memory_block(fast_memory)
        prompt_profile.update(block)
    elif fast_memory is not None:
        fast_diagnostics = {"shape_problems": [{"where": "root", "problem": "not_an_object"}]}
    feedback = "нет"
    attempts, attempts_log = 0, []
    lesson_text = None
    problems: list[str] = []
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        lesson_text = _invoke(model_call, lesson_messages(
            prompt_profile, topic, minutes, feedback, previous_lesson=previous_lesson))
        problems = validate_lesson(lesson_text, topic_id)
        attempts_log.append({"attempt": attempts, "problems": list(problems)})
        if not problems:
            break
        if attempts < MAX_ATTEMPTS:
            feedback = "; ".join(problems)
    return {
        "lesson_text": lesson_text if not problems else None,
        # `status`/`passed` follow the memory backend's draft contract;
        # `ok`/`problems`/`attempts_log` stay the workflow's own diagnostics.
        "validation": {"ok": not problems, "passed": not problems,
                       "status": "passed" if not problems else "failed",
                       "problems": problems,
                       "attempts_log": attempts_log, "attempts": attempts},
        "attempts": attempts,
        "profile_payload": deepcopy(prompt_profile),
        "fast_memory_diagnostics": fast_diagnostics,
    }
