# pipeline/profile_agent.py
"""Dynamic child profile updates based on LLM conversation analysis.
Calls LM Studio directly (not via Open WebUI) to avoid proxy loops."""
import json
import logging
import re
from contextlib import closing

import requests
from pipeline.config import (
    PROFILE_LLM_API_URL, PROFILE_LLM_MODEL, PROFILE_UPDATE_INTERVAL
)
from pipeline.db import get_child_full_profile, get_connection
from pipeline.event_logger import log_event

logger = logging.getLogger(__name__)

# Event type of the persisted assessment marker (events table). Written in the
# same transaction as the score deltas so the outlet can skip an
# already-assessed user-message boundary (regenerate/retry must not re-apply
# relative deltas).
ASSESSMENT_MARKER_EVENT = "profile_assessed"

# The assessor prompt asks for "short snake_case labels"; this is the contract
# error tags must match after normalization before they may be persisted.
ERROR_TAG_RE = re.compile(r"^[a-z0-9_]{1,64}$")

ASSESS_PROMPT = """You are a learning assessment system. Analyze this conversation
between a tutor and a child student.

Child profile:
{profile_json}

Last {n} messages:
{messages}

For each topic discussed, assess:
1. Did the child demonstrate understanding? (score_delta: -10 to +15)
2. What is the appropriate status? (knows / learning / struggling)
3. Brief reasoning (1 sentence)
4. List specific error tags if the child made mistakes (e.g. "ray_vs_segment", "figure_vs_length")

Respond ONLY with valid JSON array, no other text:
[
  {{"topic": "geometry: segments", "score_delta": 10, "new_status": "learning",
    "reasoning": "Child correctly identified segment endpoints after hint",
    "error_tags": ["ray_vs_segment"]}}
]

Rules:
- score_delta is RELATIVE to current score (positive = improved, negative = regressed)
- Only assess topics that were actually discussed
- If unsure, use score_delta: 0 and keep current status
- Maximum +15 per assessment (avoid score inflation)
- Minimum -10 per assessment (avoid discouragement from one bad session)
- error_tags: short snake_case labels for specific mistakes; empty list if no errors"""


def should_update(message_count: int) -> bool:
    """Update every PROFILE_UPDATE_INTERVAL user messages."""
    return message_count > 0 and message_count % PROFILE_UPDATE_INTERVAL == 0


def assess_interaction(child_id: str, messages: list[dict]) -> list[dict]:
    """Ask LLM to assess child's understanding. Returns list of assessments.

    An empty list can mean "nothing to assess" (model answered, no findings)
    OR a failure. Every failure path logs an assessment_failed event with a
    clear reason code, so a silently skipped window stays visible in
    diagnostics; the chat itself keeps flowing.
    """
    if not PROFILE_LLM_MODEL:
        logger.warning("PROFILE_LLM_MODEL is not configured; skipping profile assessment")
        log_event(child_id, "assessment_failed", {"reason": "model_not_configured"})
        return []

    profile = get_child_full_profile(child_id)
    if not profile:
        logger.warning("Profile unavailable for child=%s; skipping assessment", child_id)
        log_event(child_id, "assessment_failed", {"reason": "profile_unavailable"})
        return []

    # Take last N user+assistant messages
    recent = [m for m in messages if m.get("role") in ("user", "assistant")]
    recent = recent[-(PROFILE_UPDATE_INTERVAL * 2):]

    prompt = ASSESS_PROMPT.format(
        profile_json=json.dumps(profile, default=str, ensure_ascii=False),
        n=len(recent),
        messages="\n".join(f"{m['role']}: {m['content']}" for m in recent)
    )

    try:
        resp = requests.post(
            f"{PROFILE_LLM_API_URL}/chat/completions",
            json={
                "model": PROFILE_LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            },
            timeout=30
        )
        text = resp.json()["choices"][0]["message"]["content"]
        # Strip markdown fences if present
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(text)
    except Exception as exc:
        logger.exception("Assessment LLM call failed")
        log_event(child_id, "assessment_failed", {
            "reason": "llm_error",
            "error": type(exc).__name__,
        })
        return []


def _normalize_error_tag(value) -> str | None:
    """Normalize an LLM-proposed error tag to the contract the assessor prompt
    declares (short snake_case labels).

    Non-strings — and strings that still do not match ^[a-z0-9_]{1,64}$ after
    normalization — return None, so the tag is dropped instead of being
    persisted verbatim into error_patterns and re-rendered into later system
    prompts. This mirrors the discipline card_updater applies to model strings
    before anything is written.
    """
    if not isinstance(value, str):
        return None
    tag = " ".join(value.split()).strip().lower().replace(" ", "_").strip("_")
    return tag if tag and ERROR_TAG_RE.fullmatch(tag) else None


def already_assessed(child_id: str, message_count: int) -> bool:
    """Return True if a profile assessment marker exists for this child at
    this user-message boundary.

    Fail-safe: on any DB error return False — a broken marker read must not
    block assessment entirely; the worst case is a re-applied delta.
    """
    try:
        # closing(): psycopg2's context manager commits/rolls back but does not
        # close the connection, so plain `with get_connection()` leaked one per call.
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM events
                    WHERE child_id = %s AND event_type = %s
                      AND payload->>'message_count' = %s
                    LIMIT 1
                """, (child_id, ASSESSMENT_MARKER_EVENT, str(int(message_count))))
                return cur.fetchone() is not None
    except Exception:
        logger.exception("already_assessed check failed for child=%s", child_id)
        return False


def update_profile(child_id: str, assessments: list[dict],
                   message_count: int | None = None) -> None:
    """Apply score deltas with clamping, status transitions, and error pattern tracking.

    When message_count is given, an assessment marker row (events table,
    ASSESSMENT_MARKER_EVENT) is written in the SAME transaction, and a marker
    already present for this boundary skips application entirely —
    regenerate/retry of the same conversation must not re-apply relative deltas.
    """
    try:
        # closing(): psycopg2's context manager commits/rolls back but does not
        # close the connection, so plain `with get_connection()` leaked one per call.
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                # Idempotency guard: a marker row committed by a previous
                # invocation at this boundary means these relative deltas were
                # already applied; applying them again would double-count.
                if message_count is not None:
                    cur.execute("""
                        SELECT 1 FROM events
                        WHERE child_id = %s AND event_type = %s
                          AND payload->>'message_count' = %s
                        LIMIT 1
                    """, (child_id, ASSESSMENT_MARKER_EVENT, str(int(message_count))))
                    if cur.fetchone() is not None:
                        logger.info("Boundary %s already assessed for child=%s — skipping re-apply",
                                    message_count, child_id)
                        return

                for a in assessments:
                    topic = a.get("topic", "")
                    score_delta = a.get("score_delta", 0)
                    new_status = a.get("new_status", "learning")

                    # Clamp delta
                    score_delta = max(-10, min(15, score_delta))

                    # Get current score
                    cur.execute("""
                        SELECT score, status FROM child_knowledge
                        WHERE child_id = %s AND topic = %s
                    """, (child_id, topic))
                    row = cur.fetchone()

                    if row is None:
                        # New topic discovered — create at base 50 + delta
                        new_score = max(0, min(100, 50 + score_delta))
                        cur.execute("""
                            INSERT INTO child_knowledge
                                (child_id, subject, topic, status, score)
                            VALUES (%s, 'math', %s, %s, %s)
                            ON CONFLICT (child_id, subject, topic) DO NOTHING
                        """, (child_id, topic, new_status, new_score))
                        logger.info("New topic created: %s = %s (%d)",
                                   topic, new_status, new_score)
                    else:
                        current_score = row["score"] or 0
                        new_score = max(0, min(100, current_score + score_delta))

                        # Status transition rules (prevent oscillation)
                        if new_status == "learning" and new_score < 40:
                            new_status = "struggling"
                        elif new_status == "knows" and new_score < 75:
                            new_status = "learning"

                        cur.execute("""
                            UPDATE child_knowledge
                            SET score = %s, status = %s, updated_at = NOW()
                            WHERE child_id = %s AND topic = %s
                        """, (new_score, new_status, child_id, topic))
                        logger.info("Topic updated: %s %d→%d (%s)",
                                   topic, current_score, new_score, new_status)

                    # Upsert error patterns — tags normalized/validated first:
                    # only labels matching the declared snake_case contract
                    # persist; anything else the model emits is dropped.
                    for raw_tag in a.get("error_tags") or []:
                        error_tag = _normalize_error_tag(raw_tag)
                        if not error_tag:
                            if raw_tag:
                                logger.warning("Dropped non-conforming error tag %r for child=%s topic=%s",
                                               str(raw_tag)[:80], child_id, topic)
                            continue
                        cur.execute("""
                            INSERT INTO error_patterns
                                (child_id, subject, topic, error_tag, count, last_seen)
                            VALUES (%s, 'math', %s, %s, 1, NOW())
                            ON CONFLICT (child_id, subject, error_tag)
                            DO UPDATE SET count = error_patterns.count + 1,
                                          topic = EXCLUDED.topic,
                                          last_seen = NOW()
                        """, (child_id, topic, error_tag))
                        logger.info("Error pattern: %s/%s count+1", topic, error_tag)

                # Persist the assessment marker in the SAME transaction as the
                # deltas, so a retried outlet call at this boundary is a no-op.
                if message_count is not None:
                    cur.execute("""
                        INSERT INTO events (child_id, event_type, payload)
                        VALUES (%s, %s, %s)
                    """, (child_id, ASSESSMENT_MARKER_EVENT, json.dumps(
                        {"message_count": int(message_count),
                         "assessment_count": len(assessments)},
                        default=str, ensure_ascii=False)))

            conn.commit()
    except Exception:
        logger.exception("update_profile failed for child=%s", child_id)
