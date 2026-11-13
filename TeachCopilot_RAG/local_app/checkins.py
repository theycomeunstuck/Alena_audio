"""Voluntary opening questions with durable, account-scoped daily state."""
from __future__ import annotations

from uuid import UUID
from zoneinfo import ZoneInfo

from .timezones import utc_datetime, validate_timezone


QUESTION_VERSION = "voluntary-opening-v1"
QUESTION = "Как ты сегодня? Есть что-то, о чём хочешь рассказать перед занятием? Можно сразу перейти к заданиям."
TERMINAL_STATUSES = frozenset({"answered", "skipped"})


class CheckinConflict(ValueError):
    """A daily question has already been resolved or belongs to another day."""


def _local_day(value, zone):
    return utc_datetime(value).astimezone(ZoneInfo(zone)).date()


def _public(row):
    def timestamp(name):
        return utc_datetime(row[name]).isoformat() if row.get(name) is not None else None

    return {
        "checkin_id": str(row["id"]),
        "status": row["status"],
        "text": row["question"],
        "question_version": row["question_version"],
        "activity_date": row["activity_date"].isoformat(),
        "timezone": row["learning_timezone"],
        "offered_at": timestamp("offered_at"),
        "shown_at": timestamp("shown_at"),
        "resolved_at": timestamp("resolved_at"),
        "learning_session_id": str(row["learning_session_id"]) if row["learning_session_id"] is not None else None,
        "assistant_source_id": row.get("assistant_source_id"),
        "answered_source_id": row.get("answered_source_id"),
        "available": row["status"] not in TERMINAL_STATUSES,
    }


def _session(cur, learner_id, actor_id, learning_session_id):
    cur.execute("""SELECT l.learning_timezone,CURRENT_TIMESTAMP AS server_now
        FROM teachcopilot.learners l
        JOIN teachcopilot.account_learners al ON al.learner_id=l.learner_id
        JOIN teachcopilot.accounts a ON a.id=al.account_id
        WHERE l.learner_id=%s AND a.id::text=%s AND a.active AND a.role='student'""",
        (learner_id, str(actor_id)))
    scope = cur.fetchone()
    if scope is None:
        raise PermissionError("daily check-in requires this learner's active student account")
    scope["learning_timezone"] = validate_timezone(scope["learning_timezone"])
    if learning_session_id is None:
        return {**scope, "id": None, "status": "active", "topic_id": "general"}
    session_id = str(UUID(str(learning_session_id)))
    cur.execute("""SELECT * FROM teachcopilot.learning_sessions
        WHERE id=%s AND learner_id=%s FOR UPDATE""", (session_id, learner_id))
    session = cur.fetchone()
    if session is None:
        raise KeyError(session_id)
    if str(session["actor_id"]) != str(actor_id):
        raise PermissionError("check-in conversation belongs to another account")
    return {**session, **scope}


def _require_active(session):
    if session["status"] != "active":
        raise CheckinConflict("check-in conversation is already completed")


def _load(cur, learner_id, actor_id, checkin_id):
    checkin_id = str(UUID(str(checkin_id)))
    cur.execute("""SELECT * FROM teachcopilot.daily_checkins
        WHERE id=%s AND learner_id=%s FOR UPDATE""", (checkin_id, learner_id))
    row = cur.fetchone()
    if row is None:
        raise KeyError(checkin_id)
    if str(row["actor_id"]) != str(actor_id):
        raise PermissionError("daily check-in belongs to another account")
    return row


def _event(cur, row, event_type, source_id=None):
    cur.execute("""INSERT INTO teachcopilot.daily_checkin_events
        (checkin_id,learner_id,event_type,source_id) VALUES (%s,%s,%s,%s)
        ON CONFLICT(checkin_id,event_type) DO NOTHING""",
        (row["id"], row["learner_id"], event_type, source_id))


def _require_current_day(row, session):
    if _local_day(session["server_now"], row["learning_timezone"]) != row["activity_date"]:
        raise CheckinConflict("daily check-in belongs to an earlier activity day")


def get_or_create_checkin(store, learner_id, actor_id, learning_session_id=None):
    """Offer at most one question across this account's conversations today.

    The authenticated API must check access before calling. This function also
    checks the actual conversation owner. Offering a card does not add a chat
    source until the client acknowledges its display.
    """
    with store._connect() as conn, conn.cursor() as cur:
        session = _session(cur, learner_id, actor_id, learning_session_id)
        _require_active(session)
        zone = session["learning_timezone"]
        day = _local_day(session["server_now"], zone)
        cur.execute("""INSERT INTO teachcopilot.daily_checkins
            (learner_id,actor_id,activity_date,learning_timezone,
             origin_learning_session_id,learning_session_id,question,question_version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(learner_id,actor_id,activity_date) DO NOTHING RETURNING *""",
            (learner_id, str(actor_id), day, zone, session["id"], session["id"], QUESTION, QUESTION_VERSION))
        row = cur.fetchone()
        if row is None:
            cur.execute("""SELECT * FROM teachcopilot.daily_checkins
                WHERE learner_id=%s AND actor_id=%s AND activity_date=%s""", (learner_id, str(actor_id), day))
            row = cur.fetchone()
        else:
            _event(cur, row, "offered")
        return _public(row)


def _show(store, cur, row, session):
    if row["shown_at"] is not None:
        return row
    _require_active(session)
    _require_current_day(row, session)
    session_id = str(session["id"]) if session["id"] is not None else None
    source = store._memory_source(cur, row["learner_id"], session["topic_id"], "assistant", row["question"],
        author_role="assistant", session_id=session_id, source_key="checkin:" + str(row["id"]),
        event_timezone=row["learning_timezone"], activity_timezone=row["learning_timezone"])
    cur.execute("""UPDATE teachcopilot.daily_checkins
        SET status='shown',shown_at=now(),assistant_source_id=%s,learning_session_id=%s
        WHERE id=%s RETURNING *""", (source["id"], session["id"], row["id"]))
    row = cur.fetchone()
    _event(cur, row, "shown", source["id"])
    return row


def mark_checkin_shown(store, learner_id, actor_id, checkin_id, learning_session_id=None):
    """Archive the displayed question once, retaining its literal wording."""
    with store._connect() as conn, conn.cursor() as cur:
        session = _session(cur, learner_id, actor_id, learning_session_id)
        row = _load(cur, learner_id, actor_id, checkin_id)
        if row["status"] in TERMINAL_STATUSES:
            return _public(row)
        return _public(_show(store, cur, row, session))


def skip_checkin(store, learner_id, actor_id, checkin_id, learning_session_id=None):
    """Retain a voluntary skip so another conversation will not ask again."""
    with store._connect() as conn, conn.cursor() as cur:
        session = _session(cur, learner_id, actor_id, learning_session_id)
        row = _load(cur, learner_id, actor_id, checkin_id)
        if row["status"] == "skipped":
            return _public(row)
        if row["status"] == "answered":
            raise CheckinConflict("daily check-in is already answered")
        _require_active(session)
        _require_current_day(row, session)
        row = _show(store, cur, row, session)
        cur.execute("""UPDATE teachcopilot.daily_checkins SET status='skipped',resolved_at=now()
            WHERE id=%s RETURNING *""", (row["id"],))
        row = cur.fetchone()
        _event(cur, row, "skipped")
        return _public(row)


def mark_checkin_answered(store, learner_id, actor_id, checkin_id, source_id):
    """Link the child's existing submitted message without duplicating it.

    Call after reserving the ordinary chat message. A failed model response
    still leaves the child's actual reply and this event available in history.
    """
    with store._connect() as conn, conn.cursor() as cur:
        _session(cur, learner_id, actor_id, None)
        cur.execute("""SELECT s.*,ls.actor_id AS session_actor_id,
            ls.id AS answer_session_id,m.status AS message_status
            FROM teachcopilot.memory_sources s
            JOIN teachcopilot.learning_messages m ON m.source_id=s.id AND m.learner_id=s.learner_id
                AND m.learning_session_id=s.learning_session_id
            JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
            WHERE s.id=%s AND s.learner_id=%s""", (source_id, learner_id))
        source = cur.fetchone()
        if source is None:
            raise KeyError(source_id)
        if (source["author_role"] != "student" or str(source["actor_id"]) != str(actor_id)
                or str(source["session_actor_id"]) != str(actor_id)):
            raise PermissionError("check-in answer must be this account's student message")
        row = _load(cur, learner_id, actor_id, checkin_id)
        if row["status"] == "answered":
            if row["answered_source_id"] != source_id:
                raise CheckinConflict("daily check-in already has another answer")
            return _public(row)
        if row["status"] == "skipped":
            raise CheckinConflict("daily check-in was skipped")
        if row["shown_at"] is None:
            raise CheckinConflict("daily check-in must be displayed before an answer")
        if (_local_day(source["created_at"], row["learning_timezone"]) != row["activity_date"]
                or utc_datetime(source["created_at"]) < utc_datetime(row["shown_at"])):
            raise CheckinConflict("check-in answer must follow the question in the same activity day")
        cur.execute("""UPDATE teachcopilot.daily_checkins
            SET status='answered',answered_source_id=%s,answer_learning_session_id=%s,resolved_at=now()
            WHERE id=%s RETURNING *""", (source_id, source["answer_session_id"], row["id"]))
        row = cur.fetchone()
        _event(cur, row, "answered", source_id)
        return _public(row)
