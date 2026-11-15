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
# F-012: chat-side observations are advisory - recorded as events, never scores.
ADVISORY_EVENT = "profile_assessment_advisory"

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
                   conversation_key: str | None = None) -> int:
    """Record chat-side LLM observations as an ADVISORY event (F-012).

    User decision 2026-11-15: the EMA zone assessment (learner cards via
    card_updater/zpd) is the authoritative scoring system. The profile agent
    no longer writes scores, statuses or error patterns - it validates the
    model reply (malformed items are skipped individually with a logged
    warning, deltas are clamped to -10..+15, statuses are coerced into the
    lowercase vocabulary the prompt builder buckets on) and records the
    validated observations as a profile_assessment_advisory event for
    diagnostics. The assessment marker (conversation-keyed, REG-1) still
    guards against re-running the assessor on regenerate/retry; a reply
    where validation dropped EVERY item writes no marker and logs
    assessment_failed (REG-4)."""
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

                # F-012 (user decision 2026-11-15): advisory mode. The items
                # are validated and recorded as ONE advisory event; nothing is
                # written to child_knowledge or error_patterns - the EMA zone
                # assessment (learner cards) is the authoritative scorer.
                recorded = []
                for a in assessments:
                    valid = _validate_assessment(a, child_id)
                    if valid is None:
                        continue  # malformed item skipped individually — batch continues
                    topic, score_delta, new_status = valid

                    # Error tags are normalized/validated first: only labels
                    # matching the declared snake_case contract are recorded;
                    # anything else the model emits is dropped. The tags must
                    # also be list-shaped: a string would otherwise be
                    # iterated per character.
                    raw_tags = a.get("error_tags") or []
                    if not isinstance(raw_tags, (list, tuple)):
                        if raw_tags:
                            logger.warning("Dropped non-list error_tags %r for "
                                           "child=%s topic=%s",
                                           str(raw_tags)[:80], child_id, topic)
                        raw_tags = []
                    tags = []
                    for raw_tag in raw_tags:
                        error_tag = _normalize_error_tag(raw_tag)
                        if not error_tag:
                            if raw_tag:
                                logger.warning("Dropped non-conforming error tag %r for child=%s topic=%s",
                                               str(raw_tag)[:80], child_id, topic)
                            continue
                        tags.append(error_tag)
                    recorded.append({"topic": topic, "score_delta": score_delta,
                                     "new_status": new_status, "error_tags": tags})
                applied = len(recorded)

                # Persist the assessment marker in the SAME transaction, so a
                # retried outlet call at this boundary does not re-run the
                # assessor. REG-4: when validation dropped every item, keep
                # the boundary retryable instead of burning it.
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

                # Advisory record: validated observations, never scores.
                cur.execute("""
                    INSERT INTO events (child_id, event_type, payload)
                    VALUES (%s, %s, %s)
                """, (child_id, ADVISORY_EVENT, json.dumps(
                    {"message_count": int(message_count) if message_count is not None else None,
                     "conversation_key": key,
                     "items": recorded},
                    default=str, ensure_ascii=False)))

            conn.commit()
            return applied
    except Exception:
        logger.exception("update_profile failed for child=%s", child_id)
