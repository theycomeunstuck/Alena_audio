# pipeline/profile_agent.py
"""Dynamic child profile updates based on LLM conversation analysis.
Calls LM Studio directly (not via Open WebUI) to avoid proxy loops."""
import json
import logging
import math
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

# The prompt builder (pipeline/prompt_builder.py) buckets child knowledge on
# exactly these lowercase literals; any other status would be stored but render
# in no prompt bucket, hiding the topic from every prompt section. The assessor
# prompt (ASSESS_PROMPT) declares the same vocabulary.
VALID_STATUSES = ("knows", "learning", "struggling")

# Declared assessment bounds (ASSESS_PROMPT: "Maximum +15", "Minimum -10").
SCORE_DELTA_MIN = -10
SCORE_DELTA_MAX = 15

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


def _coerce_score_delta(value) -> int | float | None:
    """Coerce an LLM-proposed score_delta to a clamped number, or None.

    json.loads already yields int/float for well-formed payloads; bools,
    strings (the old clamp raised TypeError on max(-10, min(15, "10"))) and
    non-finite floats (json accepts NaN/Infinity, and the old min/max clamp
    silently turned them into +15) are shape violations and return None, so
    the item is skipped instead of corrupting a score.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    delta = float(value)
    if not math.isfinite(delta):
        return None
    if delta.is_integer():
        delta = int(delta)  # keep integral deltas integral for storage
    return max(SCORE_DELTA_MIN, min(SCORE_DELTA_MAX, delta))


def _validate_assessment(item, child_id: str) -> tuple[str, int | float, str] | None:
    """Validate one LLM assessment item before anything touches the database.

    Returns (topic, clamped score_delta, lowercase status from the vocabulary
    the prompt builder buckets on) — or None when the item is malformed, so a
    single bad model answer skips itself with a logged warning instead of
    raising inside the open transaction and rolling back the whole batch.
    """
    if not isinstance(item, dict):
        logger.warning("Skipping non-dict assessment item %r for child=%s",
                       str(item)[:80], child_id)
        return None

    topic = item.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        logger.warning("Skipping assessment with invalid topic %r for child=%s",
                       str(topic)[:80], child_id)
        return None

    raw_delta = item.get("score_delta", 0)
    score_delta = _coerce_score_delta(raw_delta)
    if score_delta is None:
        logger.warning("Skipping assessment for topic=%r: score_delta %r is not "
                       "a number in [%d, %d] (child=%s)",
                       topic, str(raw_delta)[:80], SCORE_DELTA_MIN, SCORE_DELTA_MAX, child_id)
        return None

    raw_status = item.get("new_status", "learning")
    if not isinstance(raw_status, str):
        logger.warning("Skipping assessment for topic=%r: new_status %r is not a "
                       "string (child=%s)", topic, str(raw_status)[:80], child_id)
        return None
    new_status = raw_status.strip().lower()
    if new_status not in VALID_STATUSES:
        # No code maps foreign statuses to a prompt bucket, so storing them
        # verbatim would make the topic invisible in every prompt section —
        # skip the item instead.
        logger.warning("Skipping assessment for topic=%r: unknown new_status %r "
                       "(expected one of: %s) for child=%s",
                       topic, str(raw_status)[:80], "/".join(VALID_STATUSES), child_id)
        return None

    return topic, score_delta, new_status


def _apply_status_rules(new_status: str, new_score) -> str:
    """Status transition rules (prevent oscillation), shared by the UPDATE and
    the INSERT branch so a new topic cannot enter outside the same thresholds
    an existing topic must obey ("knows" needs >= 75; "learning" below 40 is
    "struggling")."""
    if new_status == "learning" and new_score < 40:
        return "struggling"
    if new_status == "knows" and new_score < 75:
        return "learning"
    return new_status


def already_assessed(child_id: str, message_count: int,
                     conversation_key: str | None = None) -> bool:
    """Return True if a profile assessment marker exists for this child at
    this user-message boundary of THIS conversation.

    The marker is keyed by (child_id, conversation_key, message_count):
    message counts are transient per-conversation positions, so a boundary
    reached in one chat must never suppress assessment in another chat of
    the same child (audit regression REG-1).

    Fail-safe: on any DB error return False — a broken marker read must not
    block assessment entirely; the worst case is a re-applied delta.
    """
    key = str(conversation_key or "unknown")
    try:
        # closing(): psycopg2's context manager commits/rolls back but does not
        # close the connection, so plain `with get_connection()` leaked one per call.
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM events
                    WHERE child_id = %s AND event_type = %s
                      AND payload->>'message_count' = %s
                      AND payload->>'conversation_key' = %s
                    LIMIT 1
                """, (child_id, ASSESSMENT_MARKER_EVENT, str(int(message_count)), key))
                return cur.fetchone() is not None
    except Exception:
        logger.exception("already_assessed check failed for child=%s", child_id)
        return False


def update_profile(child_id: str, assessments: list[dict],
                   message_count: int | None = None,
                   conversation_key: str | None = None) -> None:
    """Apply score deltas with clamping, status transitions, and error pattern tracking.

    The payload comes straight from the model's JSON reply, so every item is
    validated first: malformed items are skipped individually with a logged
    warning (the rest of the batch still applies), deltas are clamped to
    -10..+15, statuses are coerced into the lowercase vocabulary the prompt
    builder buckets on, and new topics go through the same 40/75 transition
    rules as updates.

    When message_count is given, an assessment marker row (events table,
    ASSESSMENT_MARKER_EVENT) is written in the SAME transaction, keyed by
    conversation (conversation_key), and a marker already present for this
    boundary of this conversation skips application entirely —
    regenerate/retry must not re-apply relative deltas. When validation
    dropped EVERY item, no marker is written (the boundary stays retryable)
    and an assessment_failed event records why (audit regression REG-4).
    """
    try:
        # closing(): psycopg2's context manager commits/rolls back but does not
        # close the connection, so plain `with get_connection()` leaked one per call.
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                # Idempotency guard: a marker row committed by a previous
                # invocation at this boundary of this conversation means these
                # relative deltas were already applied; applying them again
                # would double-count.
                key = str(conversation_key or "unknown")
                if message_count is not None:
                    cur.execute("""
                        SELECT 1 FROM events
                        WHERE child_id = %s AND event_type = %s
                          AND payload->>'message_count' = %s
                          AND payload->>'conversation_key' = %s
                        LIMIT 1
                    """, (child_id, ASSESSMENT_MARKER_EVENT, str(int(message_count)), key))
                    if cur.fetchone() is not None:
                        logger.info("Boundary %s already assessed for child=%s (conversation %s) — skipping re-apply",
                                    message_count, child_id, key)
                        return 0

                if not isinstance(assessments, (list, tuple)):
                    # The payload comes straight from json.loads of the model
                    # reply; a dict/None here used to crash the loop (or make
                    # it iterate the dict's keys) and abort the transaction.
                    logger.warning("Assessment payload is not a list (%s) for "
                                   "child=%s; nothing to apply",
                                   type(assessments).__name__, child_id)
                    assessments = []

                applied = 0
                for a in assessments:
                    valid = _validate_assessment(a, child_id)
                    if valid is None:
                        continue  # malformed item skipped individually — batch continues
                    applied += 1
                    topic, score_delta, new_status = valid

                    # Get current score
                    cur.execute("""
                        SELECT score, status FROM child_knowledge
                        WHERE child_id = %s AND topic = %s
                    """, (child_id, topic))
                    row = cur.fetchone()

                    if row is None:
                        # New topic discovered — create at base 50 + delta,
                        # then apply the SAME transition rules as the UPDATE
                        # branch (previously the INSERT branch stored
                        # new_status verbatim, bypassing the rules).
                        new_score = max(0, min(100, 50 + score_delta))
                        new_status = _apply_status_rules(new_status, new_score)
                        cur.execute("""
                            INSERT INTO child_knowledge
                                (child_id, subject, topic, status, score)
                            VALUES (%s, 'math', %s, %s, %s)
                            ON CONFLICT (child_id, subject, topic) DO NOTHING
                        """, (child_id, topic, new_status, new_score))
                        logger.info("New topic created: %s = %s (%s)",
                                   topic, new_status, new_score)
                    else:
                        current_score = row["score"] or 0
                        new_score = max(0, min(100, current_score + score_delta))

                        # Status transition rules (prevent oscillation) — the
                        # same helper the INSERT branch uses.
                        new_status = _apply_status_rules(new_status, new_score)

                        cur.execute("""
                            UPDATE child_knowledge
                            SET score = %s, status = %s, updated_at = NOW()
                            WHERE child_id = %s AND topic = %s
                        """, (new_score, new_status, child_id, topic))
                        logger.info("Topic updated: %s %s→%s (%s)",
                                   topic, current_score, new_score, new_status)

                    # Upsert error patterns — tags normalized/validated first:
                    # only labels matching the declared snake_case contract
                    # persist; anything else the model emits is dropped. The
                    # tags must also be list-shaped: a string would otherwise
                    # be iterated per character.
                    raw_tags = a.get("error_tags") or []
                    if not isinstance(raw_tags, (list, tuple)):
                        if raw_tags:
                            logger.warning("Dropped non-list error_tags %r for "
                                           "child=%s topic=%s",
                                           str(raw_tags)[:80], child_id, topic)
                        raw_tags = []
                    for raw_tag in raw_tags:
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
                # REG-4: when validation dropped every item, keep the boundary
                # retryable instead of burning it - record why it was skipped.
                if message_count is not None:
                    if applied == 0:
                        log_event(child_id, "assessment_failed", {
                            "reason": "all_items_invalid",
                            "message_count": int(message_count),
                            "conversation_key": key,
                        })
                        logger.warning("No valid assessment items for child=%s at "
                                       "boundary %s - marker not written", child_id, message_count)
                        conn.commit()  # persist the diagnostic event, keep the boundary retryable
                        return 0
                    cur.execute("""
                        INSERT INTO events (child_id, event_type, payload)
                        VALUES (%s, %s, %s)
                    """, (child_id, ASSESSMENT_MARKER_EVENT, json.dumps(
                        {"message_count": int(message_count),
                         "conversation_key": key,
                         "assessment_count": applied},
                        default=str, ensure_ascii=False)))

            conn.commit()
            return applied
    except Exception:
        logger.exception("update_profile failed for child=%s", child_id)
