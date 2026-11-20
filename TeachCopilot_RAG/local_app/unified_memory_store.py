"""Durable storage for the unified memory backend (client A).

All rows are learner-scoped with composite foreign keys, mirroring
``memory_store.py`` conventions. Authorization belongs to the service layer;
these methods only enforce identifiers, revisions and idempotency.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from psycopg2.extras import Json

from .memory_store import MemoryIdempotencyConflict, MemoryStoreMixin, _revision_error, _row
from .timezones import utc_datetime

# Store-level integrity failures inside savepoint-wrapped persist artifacts are
# component degradations, never contract exceptions (A-report F2 pattern).
try:
    from psycopg2 import Error as _Psycopg2Error
    _INTEGRITY_ERRORS = (_Psycopg2Error,)
except ImportError:  # pragma: no cover - deployments without the DB driver
    _INTEGRITY_ERRORS = ()

ANSWER_HEADING = re.compile(r"^\s*#{1,6}\s*((?:ответ\w*|answer\w*|ключ\w*|key\w*|решени\w*).*)$", re.IGNORECASE | re.MULTILINE)
ANY_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
STAGE_VALUES = ("not_started", "warmup", "explanation", "practice", "check", "completed")


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat()
    return str(value)


def parse_answer_sections(lesson_text: str):
    """Defensively locate Markdown answer sections in generated lesson text.

    The client never decides where answers live. A section starts at a heading
    that names answers/keys/solutions and runs to the next heading of any
    level. Unmatched text stays untouched; the projection only excludes the
    exact spans found here.
    """
    sections = []
    matches = list(ANSWER_HEADING.finditer(lesson_text or ""))
    heading_positions = [match.start() for match in ANY_HEADING.finditer(lesson_text or "")]
    for index, match in enumerate(matches):
        start = match.start()
        end = next((position for position in heading_positions if position > start), len(lesson_text))
        sections.append({"heading": match.group(1).strip(), "start": start, "end": end})
    return sections


def redact_lesson_text(lesson_text: str, sections):
    """Remove the stored answer spans; everything else stays byte-identical."""
    if not sections:
        return lesson_text, False
    kept, position, removed = [], 0, False
    for section in sorted(sections, key=lambda item: item["start"]):
        if (section["start"] < position or section["end"] > len(lesson_text)
                or section["end"] < section["start"]):
            continue
        kept.append(lesson_text[position:section["start"]])
        position = section["end"]
        removed = True
    kept.append(lesson_text[position:])
    return "".join(kept), removed


class UnifiedMemoryStoreMixin:
    # ---------------------------------------------------------------- lessons
    @staticmethod
    def _validate_lesson_payload(learner_id, topic_id, minutes, lesson_text, validation, request_id):
        if not isinstance(learner_id, str) or not learner_id:
            raise ValueError("learner_id is required")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError("topic_id is required")
        if type(minutes) is not int or not 1 <= minutes <= 480:
            raise ValueError("minutes must be an integer between 1 and 480")
        if not isinstance(lesson_text, str) or not lesson_text.strip():
            raise ValueError("lesson_text is required")
        if not isinstance(validation, dict) or not validation:
            raise ValueError("validation record is required for a lesson draft")
        if not (validation.get("status") in {"passed", "valid"} or validation.get("passed") is True):
            raise ValueError("only validated lesson drafts can be saved")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 180:
            raise ValueError("request_id must contain 1 to 180 characters")

    def save_unified_lesson(self, learner_id, topic_id, minutes, lesson_text, answers_text,
                            validation, source_refs, memory_revision, request_id, actor_id):
        self._validate_lesson_payload(learner_id, topic_id, minutes, lesson_text, validation, request_id)
        if answers_text is None:
            answers_text = ""
        if not isinstance(answers_text, str):
            raise ValueError("answers_text must be a string")
        if source_refs is not None and not isinstance(source_refs, list):
            raise ValueError("source_refs must be a list")
        sections = parse_answer_sections(lesson_text)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO teachcopilot.unified_lessons
                (learner_id,topic_id,minutes,lesson_text,answers_text,validation,source_refs,
                 answer_sections,memory_revision,request_id,created_by)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT(learner_id,request_id) DO NOTHING RETURNING *""",
                (learner_id, topic_id, minutes, lesson_text, answers_text, Json(validation),
                 Json(source_refs or []), Json(sections), memory_revision, request_id, str(actor_id)))
            row = cur.fetchone()
            if row is not None:
                # The deterministic task catalog is derived once for this saved
                # version from the validated lesson text (P1.1).
                from .unified_memory_tasks import build_task_catalog
                catalog = build_task_catalog(row["id"], lesson_text)
                cur.execute("""UPDATE teachcopilot.unified_lessons SET task_catalog=%s
                    WHERE id=%s AND learner_id=%s""", (Json(catalog), row["id"], learner_id))
                return dict(_row(row), created=True, task_catalog=catalog)
            cur.execute("SELECT * FROM teachcopilot.unified_lessons WHERE learner_id=%s AND request_id=%s",
                        (learner_id, request_id))
            existing = cur.fetchone()
            if existing is None:
                raise KeyError(request_id)
            same = (existing["topic_id"] == topic_id and existing["minutes"] == minutes
                    and existing["lesson_text"] == lesson_text and existing["answers_text"] == answers_text
                    and _json(existing["validation"]) == validation
                    and _json(existing["source_refs"]) == (source_refs or [])
                    and existing["memory_revision"] == memory_revision)
            if not same:
                raise MemoryIdempotencyConflict("lesson request_id was reused for different input")
            return dict(_row(existing), created=False)

    def list_unified_lessons(self, learner_id, *, approved_only=False):
        with self._connect() as conn, conn.cursor() as cur:
            query = """SELECT id,learner_id,topic_id,minutes,revision,status,memory_revision,
                created_at,confirmed_at FROM teachcopilot.unified_lessons WHERE learner_id=%s"""
            if approved_only:
                query += " AND status='approved'"
            query += " ORDER BY created_at DESC,id DESC LIMIT 200"
            cur.execute(query, (learner_id,))
            return [_row(row) for row in cur.fetchall()]

    def get_unified_lesson(self, lesson_id, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM teachcopilot.unified_lessons WHERE id=%s AND learner_id=%s",
                        (lesson_id, learner_id))
            row = cur.fetchone()
            if row is None:
                raise KeyError(lesson_id)
            return _row(row)

    def get_lesson_task_catalog(self, lesson_id, learner_id):
        """Return ``(lesson_row, catalog_or_None)``; ``None`` marks a lesson
        saved before the catalog existed (explicit limitation, never a write).
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id,revision,status,task_catalog FROM teachcopilot.unified_lessons
                WHERE id=%s AND learner_id=%s""", (lesson_id, learner_id))
            row = cur.fetchone()
            if row is None:
                raise KeyError(lesson_id)
            return _row(row)

    def search_lesson_plan_context(self, learner_id, query, *, limit=3):
        """Approved lesson plans of THIS learner matched by full text (P2).

        Context only, never student evidence: plans are ranked by ts_rank over
        the lesson text (unindexed scan over the learner's own ≤200 lesson
        versions), approved versions only, learner-scoped by construction.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("lesson plan context limit must be between 1 and 10")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id,revision,topic_id,minutes,lesson_text,answer_sections,
                    answers_text,ts_rank(to_tsvector('russian',lesson_text),q1) AS rank
                FROM teachcopilot.unified_lessons, plainto_tsquery('russian',%s) q1
                WHERE learner_id=%s AND status='approved'
                    AND to_tsvector('russian',lesson_text) @@ q1
                ORDER BY rank DESC,updated_at DESC,id DESC LIMIT %s""",
                (query, learner_id, limit))
            return [_row(row) for row in cur.fetchall()]

    def confirm_unified_lesson(self, lesson_id, learner_id, expected_revision, actor_id, request_id):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 180:
            raise ValueError("request_id must contain 1 to 180 characters")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_lessons
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (lesson_id, learner_id))
            lesson = cur.fetchone()
            if lesson is None:
                raise KeyError(lesson_id)
            if lesson["status"] == "approved":
                if lesson["revision"] != expected_revision:
                    _revision_error(f"lesson revision is {lesson['revision']}, expected {expected_revision}")
                return dict(_row(lesson), confirmed=False)
            if lesson["revision"] != expected_revision:
                _revision_error(f"lesson revision is {lesson['revision']}, expected {expected_revision}")
            cur.execute("""UPDATE teachcopilot.unified_lessons
                SET status='approved',confirmed_by=%s,confirmed_at=now(),updated_at=now()
                WHERE id=%s AND learner_id=%s RETURNING *""", (str(actor_id), lesson_id, learner_id))
            return dict(_row(cur.fetchone()), confirmed=True)

    # ----------------------------------------------------------- lesson state
    def attach_unified_lesson(self, session_id, learner_id, lesson_id, actor_id, request_id):
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 180:
            raise ValueError("request_id must contain 1 to 180 characters")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_lessons
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (lesson_id, learner_id))
            lesson = cur.fetchone()
            if lesson is None:
                raise KeyError(lesson_id)
            if lesson["status"] != "approved":
                raise ValueError("only an approved lesson can be attached to a session")
            cur.execute("""SELECT * FROM teachcopilot.learning_sessions
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (session_id, learner_id))
            session = cur.fetchone()
            if session is None:
                raise KeyError(session_id)
            cur.execute("""SELECT * FROM teachcopilot.unified_lesson_states
                WHERE learning_session_id=%s FOR UPDATE""", (session_id,))
            state = cur.fetchone()
            if state is not None:
                if str(state["lesson_id"]) == str(lesson_id):
                    return dict(_row(state), created=False)
                # Re-attach replaces the lesson and restarts the guided flow.
                cur.execute("""UPDATE teachcopilot.unified_lesson_states
                    SET lesson_id=%s,revision=revision+1,stage='not_started',pending_task_id=NULL,
                        last_question_source_id=NULL,completed_task_ids='[]'::jsonb,request_id=%s,
                        updated_by=%s,updated_at=now()
                    WHERE learning_session_id=%s RETURNING *""",
                    (lesson_id, request_id, str(actor_id), session_id))
                return dict(_row(cur.fetchone()), created=False)
            cur.execute("""INSERT INTO teachcopilot.unified_lesson_states
                (learning_session_id,learner_id,lesson_id,revision,stage,request_id,updated_by)
                VALUES (%s,%s,%s,1,'not_started',%s,%s) RETURNING *""",
                (session_id, learner_id, lesson_id, request_id, str(actor_id)))
            return dict(_row(cur.fetchone()), created=True)

    def get_unified_lesson_state(self, session_id, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT s.*,l.topic_id,l.minutes FROM teachcopilot.unified_lesson_states s
                JOIN teachcopilot.unified_lessons l ON l.id=s.lesson_id AND l.learner_id=s.learner_id
                WHERE s.learning_session_id=%s AND s.learner_id=%s""", (session_id, learner_id))
            row = cur.fetchone()
            return _row(row) if row else None

    def update_unified_lesson_state(self, session_id, learner_id, expected_revision, *, stage=None,
                                    pending_task_id=..., last_question_source_id=...,
                                    completed_task_ids=None, request_id=None, actor_id=None):
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if stage is not None and stage not in STAGE_VALUES:
            raise ValueError("unknown lesson stage")
        if completed_task_ids is not None:
            if (not isinstance(completed_task_ids, list)
                    or any(not isinstance(item, str) or not item or len(item) > 200 for item in completed_task_ids)
                    or len(completed_task_ids) != len(set(completed_task_ids))):
                raise ValueError("completed_task_ids must be distinct nonempty strings")
        for name, value in (("pending_task_id", pending_task_id), ("last_question_source_id", last_question_source_id)):
            if value is not ... and value is not None and (not isinstance(value, str) or not value or len(value) > 200):
                raise ValueError(f"{name} must be a nonempty string or null")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_lesson_states
                WHERE learning_session_id=%s AND learner_id=%s FOR UPDATE""", (session_id, learner_id))
            state = cur.fetchone()
            if state is None:
                raise KeyError(session_id)
            if request_id is not None and state["request_id"] == request_id:
                # Accepted requests are idempotent: the same request returns the
                # already persisted state.
                return dict(_row(state), applied=False)
            if state["revision"] != expected_revision:
                _revision_error(f"lesson state revision is {state['revision']}, expected {expected_revision}")
            self._validate_state_references(cur, learner_id, state,
                                            pending_task_id=pending_task_id,
                                            completed_task_ids=completed_task_ids,
                                            last_question_source_id=last_question_source_id)
            assignments, params = [], []
            if stage is not None:
                assignments.append("stage=%s")
                params.append(stage)
            if pending_task_id is not ...:
                assignments.append("pending_task_id=%s")
                params.append(pending_task_id)
            if last_question_source_id is not ...:
                assignments.append("last_question_source_id=%s")
                params.append(last_question_source_id)
            if completed_task_ids is not None:
                assignments.append("completed_task_ids=%s")
                params.append(Json(completed_task_ids))
            if request_id is not None:
                assignments.append("request_id=%s")
                params.append(request_id)
            assignments.append("updated_by=%s")
            params.append(str(actor_id))
            assignments.append("updated_at=now()")
            assignments.append("revision=revision+1")
            params.extend([session_id, learner_id])
            cur.execute("UPDATE teachcopilot.unified_lesson_states SET " + ",".join(assignments) +
                        " WHERE learning_session_id=%s AND learner_id=%s RETURNING *", params)
            row = dict(_row(cur.fetchone()), applied=True)
            if stage == "completed":
                # P0.2: completing the guided lesson state completes the
                # corresponding pre-login assignment in the same transaction.
                cur.execute("""UPDATE teachcopilot.unified_lesson_assignments
                    SET status='completed',completed_at=now(),updated_at=now()
                    WHERE learner_id=%s AND learning_session_id=%s AND status='started'
                    RETURNING id""", (learner_id, session_id))
                completed = cur.fetchone()
                if completed is not None:
                    row["completed_assignment_id"] = str(completed["id"])
            return row

    @staticmethod
    def _validate_state_references(cur, learner_id, state, *, pending_task_id,
                                   completed_task_ids, last_question_source_id):
        """P1.1: task and question references are validated against the
        attached lesson catalog and the actual session sources.

        Lessons saved before the catalog existed stay permissive (explicit
        limitation); no task of a foreign lesson can enter the state.
        """
        cur.execute("SELECT task_catalog FROM teachcopilot.unified_lessons WHERE id=%s AND learner_id=%s",
                    (state["lesson_id"], learner_id))
        lesson = cur.fetchone()
        if lesson is not None and lesson["task_catalog"] is not None:
            from .unified_memory_tasks import catalog_task_ids
            allowed = catalog_task_ids(lesson["task_catalog"])
            for name, value in (("pending_task_id", pending_task_id),):
                if value is not ... and value is not None and value not in allowed:
                    raise ValueError("pending_task_id does not belong to the attached lesson")
            if completed_task_ids:
                unknown = [task_id for task_id in completed_task_ids if task_id not in allowed]
                if unknown:
                    raise ValueError("completed_task_ids contains tasks of another lesson or unknown ids")
        for name, value in (("last_question_source_id", last_question_source_id),):
            if value is ... or value is None:
                continue
            cur.execute("""SELECT 1 FROM teachcopilot.memory_sources
                WHERE id=%s AND learner_id=%s AND learning_session_id=%s AND author_role='assistant'""",
                (value, learner_id, state["learning_session_id"]))
            if cur.fetchone() is None:
                raise ValueError("last_question_source_id must reference an assistant source of this session")

    # -------------------------------------------------- personal change events
    @staticmethod
    def _validate_change_fields(category, operation, value, scope):
        if category not in ("interest", "avoided_interest", "preference"):
            raise ValueError("unknown personal change category")
        if operation not in ("add", "update", "remove"):
            raise ValueError("unknown personal change operation")
        # A-report F2: the remove value is the fact being withdrawn and stays
        # a nonempty literal like add/update.
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ValueError("personal change value must contain 1 to 200 characters")
        if scope not in ("durable", "session"):
            raise ValueError("unknown personal change scope")

    def append_personal_change(self, learner_id, *, category, operation, value, scope, status,
                               evidence=None, source_refs=None, session_id=None, run_id=None,
                               explicit_durable=False, request_id=None, decided_by=None, actor_id=None):
        self._validate_change_fields(category, operation, value, scope)
        if status not in ("proposed", "applied", "rejected"):
            raise ValueError("unknown personal change status")
        with self._connect() as conn, conn.cursor() as cur:
            return self._append_personal_change_cur(
                cur, learner_id, category=category, operation=operation, value=value,
                scope=scope, status=status, evidence=evidence, source_refs=source_refs,
                session_id=session_id, run_id=run_id, explicit_durable=explicit_durable,
                request_id=request_id, decided_by=decided_by)

    @classmethod
    def _append_personal_change_cur(cls, cur, learner_id, *, category, operation, value, scope,
                                    status, evidence=None, source_refs=None, session_id=None,
                                    run_id=None, explicit_durable=False, request_id=None,
                                    decided_by=None):
        """Insert one personal change event on the caller's transaction."""
        cls._validate_change_fields(category, operation, value, scope)
        if status not in ("proposed", "applied", "rejected"):
            raise ValueError("unknown personal change status")
        if request_id:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE learner_id=%s AND request_id=%s""", (learner_id, request_id))
            existing = cur.fetchone()
            if existing is not None:
                same = (existing["category"] == category and existing["operation"] == operation
                        and (operation == "remove" or existing["value"] == value)
                        and existing["scope"] == scope and existing["status"] == status)
                if not same:
                    raise MemoryIdempotencyConflict("personal change request_id was reused for different input")
                return dict(_row(existing), created=False)
        cur.execute("""INSERT INTO teachcopilot.unified_personal_changes
            (learner_id,session_id,run_id,category,operation,value,scope,status,explicit_durable,
             evidence,source_refs,decided_by,decided_at,request_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (learner_id, session_id, run_id, category, operation,
             value, scope, status, explicit_durable,
             Json(evidence or []), Json(source_refs or []), decided_by,
             None if decided_by is None else datetime.now(timezone.utc), request_id))
        row = _row(cur.fetchone())
        cur.execute("""UPDATE teachcopilot.learners SET unified_changes_revision=unified_changes_revision+1
            WHERE learner_id=%s""", (learner_id,))
        return dict(row, created=True)

    def revert_personal_change(self, learner_id, change_id, actor_id, request_id=None):
        """Append a reverting event; the applied event is never rewritten."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (change_id, learner_id))
            target = cur.fetchone()
            if target is None:
                raise KeyError(change_id)
            if target["status"] != "applied" or target["scope"] != "durable":
                raise ValueError("only an applied durable personal change can be reverted")
            cur.execute("SELECT 1 FROM teachcopilot.unified_personal_changes WHERE reverts_id=%s", (target["id"],))
            if cur.fetchone():
                return {"reverted": False, "change_id": str(target["id"])}
            cur.execute("""INSERT INTO teachcopilot.unified_personal_changes
                (learner_id,session_id,run_id,category,operation,value,scope,status,explicit_durable,
                 evidence,source_refs,decided_by,decided_at,request_id,reverts_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'reverted',%s,%s,%s,%s,now(),%s,%s) RETURNING id""",
                (learner_id, target["session_id"], target["run_id"], target["category"], target["operation"],
                 target["value"], target["scope"], target["explicit_durable"], target["evidence"],
                 target["source_refs"], str(actor_id), request_id, target["id"]))
            revert_id = cur.fetchone()["id"]
            cur.execute("""UPDATE teachcopilot.learners SET unified_changes_revision=unified_changes_revision+1
                WHERE learner_id=%s""", (learner_id,))
            return {"reverted": True, "change_id": str(target["id"]), "revert_id": str(revert_id)}

    def durable_personal_change_events(self, learner_id):
        """All durable change events of one learner in replay (source) order.

        Migration 027 gives every event a stable ``event_seq``; before it
        existed, ``(created_at, id)`` order is used. The facade replays these
        over the manual profile to build the effective projection.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE learner_id=%s AND scope='durable'
                ORDER BY event_seq""", (learner_id,))
            return [_row(row) for row in cur.fetchall()]

    def project_personal_changes(self, learner_id):
        """Current durable projection replayed from append-only events."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE learner_id=%s AND scope='durable' ORDER BY event_seq""", (learner_id,))
            events = [_row(row) for row in cur.fetchall()]
        reverted = {event["reverts_id"] for event in events if event["reverts_id"]}
        values = {"interest": [], "avoided_interest": [], "preference": []}
        for event in events:
            # Revert markers carry the superseded fact but never replay it; a
            # reverted event stays in history with its original status.
            if event["reverts_id"] or event["id"] in reverted or event["status"] != "applied":
                continue
            bucket = values[event["category"]]
            if event["operation"] in ("add", "update"):
                if event["value"] not in bucket:
                    bucket.append(event["value"])
            elif event["operation"] == "remove" and event["value"] in bucket:
                bucket.remove(event["value"])
        return values

    def list_personal_changes(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE learner_id=%s ORDER BY created_at DESC,id DESC""", (learner_id,))
            return [_row(row) for row in cur.fetchall()]

    def get_unified_changes_revision(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT unified_changes_revision FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(learner_id)
            return row["unified_changes_revision"]

    # ---------------------------------------------------- academic candidates
    def insert_academic_candidate(self, learner_id, *, category, proposed_value, source_refs=None,
                                  session_id=None, run_id=None, request_id=None):
        with self._connect() as conn, conn.cursor() as cur:
            return self._insert_academic_candidate_cur(
                cur, learner_id, category=category, proposed_value=proposed_value,
                source_refs=source_refs, session_id=session_id, run_id=run_id,
                request_id=request_id)

    @staticmethod
    def _insert_academic_candidate_cur(cur, learner_id, *, category, proposed_value, source_refs=None,
                                       session_id=None, run_id=None, request_id=None):
        if category not in ("knowledge_observation", "error_pattern", "learning_goal"):
            raise ValueError("unknown academic candidate category")
        if not isinstance(proposed_value, dict) or not proposed_value:
            raise ValueError("proposed_value must be a nonempty object")
        if request_id:
            cur.execute("""SELECT * FROM teachcopilot.unified_academic_candidates
                WHERE learner_id=%s AND request_id=%s""", (learner_id, request_id))
            existing = cur.fetchone()
            if existing is not None:
                if (existing["category"] != category or _json(existing["proposed_value"]) != proposed_value):
                    raise MemoryIdempotencyConflict("candidate request_id was reused for different input")
                return dict(_row(existing), created=False)
        cur.execute("""INSERT INTO teachcopilot.unified_academic_candidates
            (learner_id,session_id,run_id,category,proposed_value,source_refs,request_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (learner_id, session_id, run_id, category, Json(proposed_value),
             Json(source_refs or []), request_id))
        row = _row(cur.fetchone())
        cur.execute("""UPDATE teachcopilot.learners SET unified_changes_revision=unified_changes_revision+1
            WHERE learner_id=%s""", (learner_id,))
        return dict(row, created=True)

    def list_academic_candidates(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_academic_candidates
                WHERE learner_id=%s ORDER BY created_at DESC,id DESC""", (learner_id,))
            return [_row(row) for row in cur.fetchall()]

    def decide_academic_candidate(self, learner_id, candidate_id, decision, actor_id, note=None):
        if decision not in ("accepted", "rejected"):
            raise ValueError("unknown candidate decision")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_academic_candidates
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (candidate_id, learner_id))
            candidate = cur.fetchone()
            if candidate is None:
                raise KeyError(candidate_id)
            if candidate["status"] != "pending":
                return dict(_row(candidate), applied=False)
            cur.execute("""UPDATE teachcopilot.unified_academic_candidates
                SET status=%s,decided_by=%s,decided_at=now(),decision_note=%s
                WHERE id=%s RETURNING *""", (decision, str(actor_id), note, candidate_id))
            row = _row(cur.fetchone())
            cur.execute("""UPDATE teachcopilot.learners SET unified_changes_revision=unified_changes_revision+1
                WHERE learner_id=%s""", (learner_id,))
            return dict(row, applied=True)

    def accept_academic_candidate_canonically(self, learner_id, candidate_id, actor_id, *,
                                              topic_id, note_text):
        """Accept an academic candidate and apply it canonically in ONE
        transaction (A-report F6).

        Decision, the approved teacher-note source and its durable
        memory_update job either all commit or none do; a queue-full failure
        rolls the acceptance back instead of leaving an accepted candidate
        without its canonical note. The advisory queue lock is taken first
        (reentrant inside _enqueue_memory_job), keeping the SP-1 ordering.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            cur.execute("""SELECT * FROM teachcopilot.unified_academic_candidates
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (candidate_id, learner_id))
            candidate = cur.fetchone()
            if candidate is None:
                raise KeyError(candidate_id)
            if candidate["status"] != "pending":
                return dict(_row(candidate), applied=False, note_source_id=None)
            cur.execute("""UPDATE teachcopilot.unified_academic_candidates
                SET status='accepted',decided_by=%s,decided_at=now()
                WHERE id=%s RETURNING *""", (str(actor_id), candidate_id))
            row = _row(cur.fetchone())
            cur.execute("""UPDATE teachcopilot.learners SET unified_changes_revision=unified_changes_revision+1
                WHERE learner_id=%s""", (learner_id,))
            source = self._memory_source(cur, learner_id, topic_id, actor_id, note_text,
                                         author_role="teacher", approved=True,
                                         source_key=f"unified-candidate:{candidate_id}")
            job = self._enqueue_memory_job(cur, learner_id, topic_id, "memory_update",
                                           {"source_id": source["id"]}, actor_id,
                                           method="tail", idempotency_key=f"source:{source['id']}")
            return dict(row, applied=True, note_source_id=source["id"], job_id=job["id"])

    def get_personal_change(self, learner_id, change_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE id=%s AND learner_id=%s""", (change_id, learner_id))
            row = cur.fetchone()
            return _row(row) if row else None

    def decide_personal_change(self, learner_id, change_id, decision, actor_id):
        if decision not in ("applied", "rejected"):
            raise ValueError("unknown personal change decision")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_personal_changes
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (change_id, learner_id))
            change = cur.fetchone()
            if change is None:
                raise KeyError(change_id)
            if change["status"] != "proposed":
                return dict(_row(change), applied=False)
            cur.execute("""UPDATE teachcopilot.unified_personal_changes
                SET status=%s,decided_by=%s,decided_at=now()
                WHERE id=%s RETURNING *""", (decision, str(actor_id), change_id))
            row = _row(cur.fetchone())
            cur.execute("""UPDATE teachcopilot.learners SET unified_changes_revision=unified_changes_revision+1
                WHERE learner_id=%s""", (learner_id,))
            return dict(row, applied=True)

    def get_academic_candidate(self, learner_id, candidate_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_academic_candidates
                WHERE id=%s AND learner_id=%s""", (candidate_id, learner_id))
            row = cur.fetchone()
            return _row(row) if row else None

    def latest_open_lesson_state(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT s.*,l.topic_id,l.minutes FROM teachcopilot.unified_lesson_states s
                JOIN teachcopilot.unified_lessons l ON l.id=s.lesson_id AND l.learner_id=s.learner_id
                WHERE s.learner_id=%s AND s.stage<>'completed'
                ORDER BY s.updated_at DESC LIMIT 1""", (learner_id,))
            row = cur.fetchone()
            return _row(row) if row else None

    def latest_lesson_state(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT s.*,l.topic_id,l.minutes FROM teachcopilot.unified_lesson_states s
                JOIN teachcopilot.unified_lessons l ON l.id=s.lesson_id AND l.learner_id=s.learner_id
                WHERE s.learner_id=%s ORDER BY s.updated_at DESC LIMIT 1""", (learner_id,))
            row = cur.fetchone()
            return _row(row) if row else None

    # ------------------------------------------------------------ assignments
    @staticmethod
    def _validated_assignment_request(request_id):
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 180:
            raise ValueError("request_id must contain 1 to 180 characters")
        return request_id

    def create_lesson_assignment(self, learner_id, lesson_id, lesson_revision, actor_id, request_id):
        """Idempotent per payload: same (lesson, revision) returns the existing
        assignment; the same request id with a different payload conflicts."""
        self._validated_assignment_request(request_id)
        if type(lesson_revision) is not int or lesson_revision < 1:
            raise ValueError("lesson_revision must be a positive integer")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_lesson_assignments
                WHERE learner_id=%s AND request_id=%s""", (learner_id, request_id))
            existing = cur.fetchone()
            if existing is not None:
                if (str(existing["lesson_id"]) != str(lesson_id)
                        or existing["lesson_revision"] != lesson_revision):
                    raise MemoryIdempotencyConflict(
                        "assignment request_id was reused for a different lesson payload")
                return dict(_row(existing), created=False)
            cur.execute("""SELECT * FROM teachcopilot.unified_lesson_assignments
                WHERE learner_id=%s AND lesson_id=%s FOR UPDATE""", (learner_id, lesson_id))
            same_payload = cur.fetchone()
            if same_payload is not None:
                if same_payload["lesson_revision"] != lesson_revision:
                    raise MemoryIdempotencyConflict(
                        "lesson already has an assignment with a different revision")
                return dict(_row(same_payload), created=False)
            cur.execute("""INSERT INTO teachcopilot.unified_lesson_assignments
                (learner_id,lesson_id,lesson_revision,status,assigned_by,request_id)
                VALUES (%s,%s,%s,'pending',%s,%s) RETURNING *""",
                (learner_id, lesson_id, lesson_revision, str(actor_id), request_id))
            return dict(_row(cur.fetchone()), created=True)

    def get_lesson_assignment(self, learner_id, assignment_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT a.*,l.topic_id,l.minutes,l.status AS lesson_status
                FROM teachcopilot.unified_lesson_assignments a
                JOIN teachcopilot.unified_lessons l ON l.id=a.lesson_id AND l.learner_id=a.learner_id
                WHERE a.id=%s AND a.learner_id=%s""", (assignment_id, learner_id))
            row = cur.fetchone()
            return _row(row) if row else None

    def list_lesson_assignments(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT a.*,l.topic_id,l.minutes
                FROM teachcopilot.unified_lesson_assignments a
                JOIN teachcopilot.unified_lessons l ON l.id=a.lesson_id AND l.learner_id=a.learner_id
                WHERE a.learner_id=%s ORDER BY a.assigned_at DESC,a.id DESC LIMIT 200""",
                (learner_id,))
            return [_row(row) for row in cur.fetchall()]

    def start_lesson_assignment(self, learner_id, assignment_id, actor_id, request_id):
        """Claim a pending assignment into a NEW standard session owned by the
        linked student, in ONE transaction.

        The approved lesson revision is re-validated; repeats return the
        persisted session; completed assignments never create a fresh session.
        The advisory queue lock is taken first, matching the SP-1 order used by
        every other transaction that touches learning_sessions rows.
        """
        self._validated_assignment_request(request_id)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            self._memory_learner(cur, learner_id)
            cur.execute("""SELECT * FROM teachcopilot.unified_lesson_assignments
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (assignment_id, learner_id))
            assignment = cur.fetchone()
            if assignment is None:
                raise KeyError(assignment_id)
            if assignment["status"] != "pending":
                # Started/completed assignments keep their persisted session.
                return dict(_row(assignment), created=False, state=None)
            cur.execute("""SELECT 1 FROM teachcopilot.accounts a
                JOIN teachcopilot.account_learners al ON al.account_id=a.id
                WHERE a.id=%s AND a.role='student' AND a.active
                    AND al.learner_id=%s AND al.relationship='student'""",
                (str(actor_id), learner_id))
            if cur.fetchone() is None:
                raise PermissionError("only the linked student can start the assignment")
            cur.execute("""SELECT * FROM teachcopilot.unified_lessons
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (assignment["lesson_id"], learner_id))
            lesson = cur.fetchone()
            if lesson is None:
                raise KeyError(assignment["lesson_id"])
            if lesson["status"] != "approved":
                raise ValueError("only an approved lesson can be started")
            if lesson["revision"] != assignment["lesson_revision"]:
                _revision_error(f"approved lesson revision is {lesson['revision']}, "
                                f"assignment recorded {assignment['lesson_revision']}")
            cur.execute("""INSERT INTO teachcopilot.learning_sessions
                (learner_id,actor_id,topic_id,session_mode,event_timezone)
                VALUES (%s,%s,%s,'standard',
                    (SELECT learning_timezone FROM teachcopilot.learners WHERE learner_id=%s))
                RETURNING *""", (learner_id, str(actor_id), lesson["topic_id"], learner_id))
            session = _row(cur.fetchone())
            cur.execute("""INSERT INTO teachcopilot.unified_lesson_states
                (learning_session_id,learner_id,lesson_id,revision,stage,request_id,updated_by)
                VALUES (%s,%s,%s,1,'not_started',%s,%s) RETURNING *""",
                (session["id"], learner_id, lesson["id"], request_id, str(actor_id)))
            state = _row(cur.fetchone())
            cur.execute("""UPDATE teachcopilot.unified_lesson_assignments
                SET status='started',learning_session_id=%s,started_by=%s,started_at=now(),
                    request_id=%s,updated_at=now()
                WHERE id=%s AND learner_id=%s RETURNING *""",
                (session["id"], str(actor_id), request_id, assignment_id, learner_id))
            assignment = _row(cur.fetchone())
            return dict(assignment, created=True,
                        state=dict(state, topic_id=lesson["topic_id"], minutes=lesson["minutes"]))

    def rebuild_lesson_task_catalog(self, learner_id, lesson_id, actor_id, request_id):
        """Explicit worker/admin backfill of the catalog for an existing
        lesson version. Never called by a read path."""
        self._validated_assignment_request(request_id)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_lessons
                WHERE id=%s AND learner_id=%s FOR UPDATE""", (lesson_id, learner_id))
            lesson = cur.fetchone()
            if lesson is None:
                raise KeyError(lesson_id)
            from .unified_memory_tasks import build_task_catalog
            catalog = build_task_catalog(lesson["id"], lesson["lesson_text"])
            if lesson["task_catalog"] is not None and _json(lesson["task_catalog"]) == catalog:
                return dict(_row(lesson), rebuilt=False)
            cur.execute("""UPDATE teachcopilot.unified_lessons SET task_catalog=%s,updated_at=now()
                WHERE id=%s AND learner_id=%s""", (Json(catalog), lesson_id, learner_id))
            return dict(_row(lesson), rebuilt=True, task_catalog=catalog)

    # ------------------------------------------------------- processing runs
    def find_unified_run(self, learner_id, session_id, source_fingerprint, prompt_recipe, model_identity):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_processing_runs
                WHERE learner_id=%s AND session_id=%s AND source_fingerprint=%s
                    AND prompt_recipe=%s AND model_identity=%s
                ORDER BY created_at DESC,id DESC LIMIT 1""",
                (learner_id, session_id, source_fingerprint, prompt_recipe, model_identity))
            row = cur.fetchone()
            return _row(row) if row else None

    def get_unified_run_by_request(self, learner_id, request_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_processing_runs
                WHERE learner_id=%s AND request_id=%s""", (learner_id, request_id))
            row = cur.fetchone()
            return _row(row) if row else None

    def latest_unified_run(self, learner_id, session_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_processing_runs
                WHERE learner_id=%s AND session_id=%s ORDER BY created_at DESC,id DESC LIMIT 1""",
                (learner_id, session_id))
            row = cur.fetchone()
            return _row(row) if row else None

    def pending_session_process_job(self, learner_id, session_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id,status FROM teachcopilot.memory_jobs
                WHERE learner_id=%s AND kind='session_process'
                    AND payload->>'learning_session_id'=%s
                    AND status IN ('queued','running')
                ORDER BY created_at DESC LIMIT 1""", (learner_id, str(session_id)))
            row = cur.fetchone()
            return _row(row) if row else None

    def save_unified_run(self, learner_id, session_id, *, status, provisional, source_fingerprint,
                         prompt_recipe, model_identity, summary, coverage, warnings, request_id):
        if status not in ("completed", "partial", "failed"):
            raise ValueError("unknown processing run status")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO teachcopilot.unified_processing_runs
                (learner_id,session_id,status,provisional,source_fingerprint,prompt_recipe,
                 model_identity,summary,coverage,warnings,request_id,completed_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()) RETURNING *""",
                (learner_id, session_id, status, provisional, source_fingerprint, prompt_recipe,
                 model_identity, Json(summary) if summary is not None else None,
                 Json(coverage or {}), Json(warnings or []), request_id))
            return _row(cur.fetchone())

    def persist_unified_run(self, learner_id, session_id, *, status, provisional,
                            source_fingerprint, prompt_recipe, model_identity, summary,
                            coverage, warnings, request_id, summary_version=None, changes=(),
                            candidates=(), release_claim=None):
        """Persist the run, its summary version and every side effect in ONE
        transaction (P1.2 personal-change atomicity), then release the
        processing claim in the same commit.

        Independent artifacts are wrapped in per-artifact savepoints: one
        rejected change or a failed summary write becomes a warning and never
        loses the other components. There are no model calls inside this
        transaction. ``release_claim`` is ``(fingerprint, recipe, identity,
        claim_token)``.
        """
        if status not in ("completed", "partial", "failed"):
            raise ValueError("unknown processing run status")
        warnings = list(warnings or [])
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO teachcopilot.unified_processing_runs
                (learner_id,session_id,status,provisional,source_fingerprint,prompt_recipe,
                 model_identity,summary,coverage,warnings,request_id,completed_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now()) RETURNING *""",
                (learner_id, session_id, status, provisional, source_fingerprint, prompt_recipe,
                 model_identity, Json(summary) if summary is not None else None,
                 Json(coverage or {}), Json(warnings or []), request_id))
            run = _row(cur.fetchone())
            summary_block = None
            if summary_version is not None:
                cur.execute("SAVEPOINT summary_version")
                try:
                    row = self._save_unified_summary_version_cur(
                        cur, learner_id, session_id, run_id=run["id"],
                        input_fingerprint=summary_version["input_fingerprint"],
                        text=summary_version["text"],
                        source_refs=summary_version["source_refs"],
                        coverage=summary_version["coverage"],
                        provisional=summary_version["provisional"])
                except Exception:
                    # A summary write failure degrades that component only
                    # (P1.2); the run and the personal changes still commit.
                    cur.execute("ROLLBACK TO SAVEPOINT summary_version")
                    warnings.append({"code": "summary_write_failed"})
                    summary_block = None
                else:
                    summary_block = {"summary_id": str(row["id"]),
                                     "text": summary_version["text"],
                                     "source_refs": summary_version["source_refs"],
                                     "coverage": summary_version["coverage"],
                                     "provisional": bool(summary_version["provisional"])}
            for change in changes:
                cur.execute("SAVEPOINT personal_change")
                try:
                    self._append_personal_change_cur(
                        cur, learner_id, category=change["category"], operation=change["operation"],
                        value=change["value"], scope=change["scope"], status=change["status"],
                        evidence=change.get("evidence"), source_refs=change.get("source_refs"),
                        session_id=session_id, run_id=run["id"],
                        explicit_durable=change.get("explicit_durable", False),
                        request_id=change.get("request_id"))
                except (ValueError,) + _INTEGRITY_ERRORS:
                    cur.execute("ROLLBACK TO SAVEPOINT personal_change")
                    warnings.append({"code": "personal_change_rejected",
                                     "category": change["category"],
                                     "value": str(change.get("value", ""))[:80]})
            for candidate in candidates:
                cur.execute("SAVEPOINT academic_candidate")
                try:
                    self._insert_academic_candidate_cur(
                        cur, learner_id, category=candidate["category"],
                        proposed_value=candidate["proposed_value"],
                        source_refs=candidate.get("source_refs"), session_id=session_id,
                        run_id=run["id"], request_id=candidate.get("request_id"))
                except (ValueError,) + _INTEGRITY_ERRORS:
                    cur.execute("ROLLBACK TO SAVEPOINT academic_candidate")
                    warnings.append({"code": "academic_candidate_rejected",
                                     "category": candidate["category"]})
            if release_claim is not None:
                fingerprint, recipe, identity, claim_token = release_claim
                cur.execute("""DELETE FROM teachcopilot.unified_processing_claims
                    WHERE learner_id=%s AND session_id=%s AND source_fingerprint=%s
                        AND prompt_recipe=%s AND model_identity=%s AND claim_token=%s""",
                    (learner_id, session_id, fingerprint, recipe, identity, claim_token))
            final_status, final_provisional = status, provisional
            if summary_version is not None and summary_block is None:
                # A run without its summary is not a completed run: the status
                # degrades honestly instead of masquerading as fresh.
                if status == "completed":
                    final_status = "partial"
                final_provisional = True
            cur.execute("""UPDATE teachcopilot.unified_processing_runs
                SET summary=%s,warnings=%s,status=%s,provisional=%s
                WHERE id=%s RETURNING *""",
                (Json(summary_block) if summary_block is not None else None,
                 Json(warnings), final_status, final_provisional, run["id"]))
            run = dict(_row(cur.fetchone()))
        return run

    # ------------------------------------------------------- processing claims
    @staticmethod
    def _validate_claim_lease(lease_seconds):
        if type(lease_seconds) is not int or not 60 <= lease_seconds <= 14400:
            raise ValueError("claim lease must be between 60 and 14400 seconds")

    def acquire_processing_claim(self, learner_id, session_id, source_fingerprint, prompt_recipe,
                                 model_identity, *, claimed_by, lease_seconds=1800):
        """Durable identity claim for one exact processing run.

        Returns ``{"claimed": True, "claim_token": ...}`` when this caller owns
        the claim, or ``{"claimed": False, holder, lease_expires_at}`` when an
        unexpired claim exists. An expired lease is taken over atomically.
        """
        self._validate_claim_lease(lease_seconds)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO teachcopilot.unified_processing_claims
                (learner_id,session_id,source_fingerprint,prompt_recipe,model_identity,
                 claimed_by,lease_expires_at)
                VALUES (%s,%s,%s,%s,%s,%s,now()+(%s*interval '1 second'))
                ON CONFLICT (learner_id,session_id,source_fingerprint,prompt_recipe,model_identity)
                DO UPDATE SET claim_token=gen_random_uuid(),claimed_by=EXCLUDED.claimed_by,
                    claimed_at=now(),lease_expires_at=now()+(%s*interval '1 second')
                    WHERE teachcopilot.unified_processing_claims.lease_expires_at < now()
                RETURNING claim_token""",
                (learner_id, str(session_id), source_fingerprint, prompt_recipe, model_identity,
                 str(claimed_by), lease_seconds, lease_seconds))
            row = cur.fetchone()
            if row is not None:
                return {"claimed": True, "claim_token": str(row["claim_token"])}
            cur.execute("""SELECT claimed_by,lease_expires_at FROM teachcopilot.unified_processing_claims
                WHERE learner_id=%s AND session_id=%s AND source_fingerprint=%s
                    AND prompt_recipe=%s AND model_identity=%s""",
                (learner_id, str(session_id), source_fingerprint, prompt_recipe, model_identity))
            held = cur.fetchone()
            return {"claimed": False,
                    "holder": str(held["claimed_by"]) if held else None,
                    "lease_expires_at": _iso(held["lease_expires_at"]) if held else None}

    def extend_processing_claim(self, learner_id, session_id, source_fingerprint, prompt_recipe,
                                model_identity, claim_token, *, lease_seconds=1800):
        self._validate_claim_lease(lease_seconds)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.unified_processing_claims
                SET lease_expires_at=now()+(%s*interval '1 second')
                WHERE learner_id=%s AND session_id=%s AND source_fingerprint=%s
                    AND prompt_recipe=%s AND model_identity=%s AND claim_token=%s
                RETURNING claim_token""",
                (lease_seconds, learner_id, str(session_id), source_fingerprint, prompt_recipe,
                 model_identity, claim_token))
            return cur.fetchone() is not None

    def release_processing_claim(self, learner_id, session_id, source_fingerprint, prompt_recipe,
                                 model_identity, claim_token):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""DELETE FROM teachcopilot.unified_processing_claims
                WHERE learner_id=%s AND session_id=%s AND source_fingerprint=%s
                    AND prompt_recipe=%s AND model_identity=%s AND claim_token=%s""",
                (learner_id, str(session_id), source_fingerprint, prompt_recipe, model_identity,
                 claim_token))
            return cur.rowcount > 0

    def save_unified_summary_version(self, learner_id, session_id, *, run_id, input_fingerprint, text,
                                     source_refs, coverage, provisional):
        with self._connect() as conn, conn.cursor() as cur:
            return self._save_unified_summary_version_cur(
                cur, learner_id, session_id, run_id=run_id, input_fingerprint=input_fingerprint,
                text=text, source_refs=source_refs, coverage=coverage, provisional=provisional)

    @staticmethod
    def _save_unified_summary_version_cur(cur, learner_id, session_id, *, run_id, input_fingerprint,
                                          text, source_refs, coverage, provisional):
        cur.execute("""SELECT id,version_index FROM teachcopilot.unified_summary_versions
            WHERE learner_id=%s AND session_id=%s AND input_fingerprint=%s
            ORDER BY version_index DESC LIMIT 1""", (learner_id, session_id, input_fingerprint))
        row = cur.fetchone()
        if row is not None:
            return _row(row)
        cur.execute("""INSERT INTO teachcopilot.unified_summary_versions
            (learner_id,session_id,run_id,version_index,input_fingerprint,text,source_refs,
             coverage,provisional)
            SELECT %s,%s,%s,COALESCE((SELECT max(version_index)
                FROM teachcopilot.unified_summary_versions
                WHERE learner_id=%s AND session_id=%s),0)+1,%s,%s,%s,%s,%s RETURNING *""",
            (learner_id, session_id, run_id, learner_id, session_id, input_fingerprint, text,
             Json(source_refs or []), Json(coverage or {}), provisional))
        return _row(cur.fetchone())

    def list_unified_summary_versions(self, learner_id, session_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT * FROM teachcopilot.unified_summary_versions
                WHERE learner_id=%s AND session_id=%s ORDER BY version_index DESC,id DESC""",
                (learner_id, session_id))
            return [_row(row) for row in cur.fetchall()]

    def latest_completed_lesson_summary(self, learner_id):
        """Pick the session by completion time, even when its summary is pending."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT previous.session_id,previous.lesson_id,previous.topic_id,
                    previous.completed_on,summary.id AS summary_id,summary.text
                FROM (
                    SELECT s.id AS session_id,s.learner_id,l.id AS lesson_id,l.topic_id,
                        (COALESCE(s.completed_at,s.created_at) AT TIME ZONE
                            COALESCE(s.event_timezone,learner.learning_timezone))::date AS completed_on
                    FROM teachcopilot.learning_sessions s
                    JOIN teachcopilot.unified_lesson_states state
                        ON state.learning_session_id=s.id AND state.learner_id=s.learner_id
                    JOIN teachcopilot.unified_lessons l
                        ON l.id=state.lesson_id AND l.learner_id=state.learner_id
                    JOIN teachcopilot.learners learner ON learner.learner_id=s.learner_id
                    WHERE s.learner_id=%s AND s.status='completed'
                    ORDER BY COALESCE(s.completed_at,s.created_at) DESC,s.id DESC LIMIT 1
                ) previous
                JOIN LATERAL (
                    SELECT v.id,v.text FROM teachcopilot.unified_summary_versions v
                    WHERE v.learner_id=previous.learner_id AND v.session_id=previous.session_id
                        AND NOT v.provisional
                    ORDER BY v.version_index DESC,v.id DESC LIMIT 1
                ) summary ON TRUE""", (learner_id,))
            row = cur.fetchone()
            return _row(row) if row else None

    # ---------------------------------------------- session snapshot (source)
    def unified_session_snapshot(self, learner_id, session_id):
        """Ordered, learner-scoped session transcript for processing.

        Student and teacher sources are evidence; assistant sources are
        context with their delivery status preserved. Failed user turns stay.
        """
        with self._connect() as conn, conn.cursor() as cur:
            connection = conn
            connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
            cur.execute("""SELECT id,topic_id,actor_id,status,session_mode,created_at,completed_at
                FROM teachcopilot.learning_sessions WHERE id=%s AND learner_id=%s""",
                (session_id, learner_id))
            session = cur.fetchone()
            if session is None:
                raise KeyError(session_id)
            cur.execute("""WITH RECURSIVE session_sources AS (
                    SELECT s.id, s.learner_id
                    FROM teachcopilot.memory_sources s
                    WHERE s.learning_session_id=%s AND s.learner_id=%s
                    UNION
                    SELECT c.id, c.learner_id
                    FROM session_sources r JOIN teachcopilot.memory_sources c
                        ON c.corrects_source_id=r.id AND c.learner_id=r.learner_id
                )
                SELECT s.id AS source_id,s.source_order,s.author_role,s.content AS text,
                    s.created_at,s.topic_id,s.approved,s.corrects_source_id,
                    s.learning_session_id AS session_id,
                    COALESCE(m.status,'standalone') AS delivery_status,
                    corr.id AS superseded_by
                FROM teachcopilot.memory_sources s
                JOIN session_sources r ON r.id=s.id
                LEFT JOIN teachcopilot.learning_messages m
                    ON m.source_id=s.id AND m.learner_id=s.learner_id
                    AND m.learning_session_id=s.learning_session_id
                LEFT JOIN LATERAL (
                    SELECT c.id FROM teachcopilot.memory_sources c
                    WHERE c.corrects_source_id=s.id AND c.learner_id=s.learner_id
                    ORDER BY c.created_at,c.id LIMIT 1) corr ON TRUE
                ORDER BY s.source_order""", (session_id, learner_id))
            sources = [_row(row) for row in cur.fetchall()]
        return {"session": dict(_row(session)), "sources": sources}

    def memory_source_correction_refs(self, learner_id, source_ids):
        if not source_ids:
            return {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id,corrects_source_id FROM teachcopilot.memory_sources
                WHERE learner_id=%s AND id=ANY(%s) AND corrects_source_id IS NOT NULL""",
                (learner_id, list(source_ids)))
            return {row["id"]: row["corrects_source_id"] for row in cur.fetchall()}

    # ------------------------------------------- persisted conversation chunks
    def upsert_conversation_chunks(self, learner_id, actor_id, chunks):
        """Persist chunk rows for search; returns the stored chunk keys."""
        if not chunks:
            return []
        with self._connect() as conn, conn.cursor() as cur:
            stored = []
            for index, chunk in enumerate(chunks):
                session_uuid = None
                raw_session = chunk.get("session_id")
                if isinstance(raw_session, str) and re.fullmatch(
                        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", raw_session):
                    session_uuid = raw_session
                cur.execute("""INSERT INTO teachcopilot.unified_conversation_chunks
                    (learner_id,actor_id,chunk_key,chunk_recipe,session_id,chunk_index,chunk_start,
                     chunk_end,text,source_ids,spans,activity_days,first_source_order,
                     last_source_order,search_document)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        to_tsvector('russian',%s)||to_tsvector('simple',%s))
                    ON CONFLICT(learner_id,actor_id,chunk_key,chunk_recipe) DO UPDATE
                    SET text=EXCLUDED.text,source_ids=EXCLUDED.source_ids,spans=EXCLUDED.spans,
                        activity_days=EXCLUDED.activity_days,
                        first_source_order=EXCLUDED.first_source_order,
                        last_source_order=EXCLUDED.last_source_order,
                        search_document=EXCLUDED.search_document,updated_at=now()""",
                    (learner_id, str(actor_id), chunk["chunk_id"],
                     chunk.get("recipe", "own-conversation-fixed900-overlap120-long240-v1"),
                     session_uuid, index, chunk["start"], chunk["end"], chunk["text"],
                     Json(chunk["source_ids"]), Json(chunk["spans"]),
                     list(chunk.get("activity_days") or []),
                     chunk.get("first_source_order", 0), chunk.get("last_source_order", 0),
                     chunk["text"], chunk["text"]))
                stored.append(chunk["chunk_id"])
            return stored

    def chunk_embedding_status(self, learner_id, actor_id, chunk_recipe, embedding_identity):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT chunk_key FROM teachcopilot.unified_conversation_chunks
                WHERE learner_id=%s AND actor_id=%s AND chunk_recipe=%s
                    AND (embedding IS NULL OR embedding_identity IS DISTINCT FROM %s)""",
                (learner_id, str(actor_id), chunk_recipe, embedding_identity))
            return [row["chunk_key"] for row in cur.fetchall()]

    def store_chunk_embedding(self, learner_id, actor_id, chunk_recipe, embedding_identity, chunk_key, vector):
        from .personal_graph_retrieval import _vector_literal, _validated_vector
        checked = _validated_vector(vector)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.unified_conversation_chunks
                SET embedding=%s::vector,embedding_identity=%s,updated_at=now()
                WHERE learner_id=%s AND actor_id=%s AND chunk_recipe=%s AND chunk_key=%s""",
                (_vector_literal(checked), embedding_identity, learner_id, str(actor_id),
                 chunk_recipe, chunk_key))

    def search_conversation_chunks(self, learner_id, actor_id, query, *, chunk_recipe,
                                   embedding_identity=None, query_vector=None, limit=12,
                                   session_id=None, time_filter=None, rrf_k=60,
                                   min_similarity=0.20, candidate_chunk_keys=None,
                                   exclude_chunk_keys=None):
        """Deterministic RRF over two learner- and actor-scoped channels.

        The vector and full-text candidate sets are both filtered by the same
        scope predicates before ranking; k=60 matches the existing curriculum
        RRF constant and the vector floor reuses the shipped engineering gate
        (cosine similarity >= min_similarity). Ties break by chunk key, so the
        order is stable. ``candidate_chunk_keys`` restricts the search to the
        caller's current authorized snapshot; ``exclude_chunk_keys`` remains
        for narrow callers, but the recall now scans the snapshot in ONE call
        instead: excluding already-seen keys would shrink the ``scoped`` set
        before the IDF statistics and ranks are computed, so every expansion
        sample would score its rows against a different corpus and the order
        could drift between samples. A single call over the fixed snapshot
        keeps IDF, both rank channels and the fused order computed on exactly
        the same set, and it costs one round trip instead of one per batch.
        The scan cap stays bounded (<= 1000 rows) and cannot exceed the
        authorized snapshot because ``candidate_chunk_keys`` filters it.
        """
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("search limit must be between 1 and 1000")
        if not 0 <= min_similarity < 1:
            raise ValueError("min_similarity must be in [0, 1)")
        days = None
        if time_filter and (time_filter.get("from") or time_filter.get("through")):
            days = [time_filter.get("from") or "0001-01-01", time_filter.get("through") or "9999-12-31"]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""WITH scoped AS (
                    SELECT chunk_key,chunk_index,text,source_ids,spans,activity_days,session_id,
                           first_source_order,last_source_order,search_document,embedding
                    FROM teachcopilot.unified_conversation_chunks
                    WHERE learner_id=%(learner)s AND actor_id=%(actor)s AND chunk_recipe=%(recipe)s
                        AND (%(chunk_keys)s::text[] IS NULL OR chunk_key=ANY(%(chunk_keys)s::text[]))
                        AND (%(excluded_keys)s::text[] IS NULL OR NOT (chunk_key=ANY(%(excluded_keys)s::text[])))
                        AND (%(session)s::uuid IS NULL OR session_id=%(session)s::uuid)
                        AND (%(days)s::text[] IS NULL OR activity_days && %(days)s::text[])
                ),
                raw_words AS (
                    SELECT DISTINCT w AS word FROM unnest(string_to_array(
                        regexp_replace(trim(%(query)s), '\\s+', ' ', 'g'), ' ')) AS w
                    WHERE length(w) >= 2
                ),
                words AS (
                    SELECT w.word,
                        (SELECT count(*) FROM scoped) AS total,
                        (SELECT count(*) FROM scoped s2
                         WHERE s2.search_document @@ (plainto_tsquery('russian', w.word)
                                                      || plainto_tsquery('simple', w.word))) AS df
                    FROM raw_words w
                ),
                weighted AS (
                    SELECT word,
                        CASE WHEN df = 0 OR total = 0 THEN 0.0
                             ELSE 1.0 + ln(total::numeric / df) END AS weight
                    FROM words
                ),
                per_word AS (
                    SELECT s.chunk_key,
                        sum(w.weight) AS score,
                        max(ts_rank(s.search_document, plainto_tsquery('russian', w.word)
                                    || plainto_tsquery('simple', w.word))) AS best_rank,
                        count(*) AS matched_words
                    FROM scoped s CROSS JOIN weighted w
                    WHERE s.search_document @@ (plainto_tsquery('russian', w.word)
                                                || plainto_tsquery('simple', w.word))
                    GROUP BY s.chunk_key
                ),
                tr AS (
                    SELECT chunk_key,row_number() OVER
                        (ORDER BY score DESC,best_rank DESC,chunk_key) AS rank
                    FROM per_word
                ),
                vr AS (
                    SELECT chunk_key,row_number() OVER (ORDER BY embedding <=> %(vector)s::vector,chunk_key) AS rank
                    FROM scoped
                    WHERE embedding IS NOT NULL AND %(identity)s::text IS NOT NULL
                        AND %(vector)s::vector IS NOT NULL
                        AND (embedding <=> %(vector)s::vector) <= %(max_distance)s
                ),
                fused AS (
                    SELECT COALESCE(vr.chunk_key,tr.chunk_key) AS chunk_key,
                        COALESCE(1.0/(%(k)s+vr.rank),0)+COALESCE(1.0/(%(k)s+tr.rank),0) AS score,
                        vr.rank AS vector_rank,tr.rank AS text_rank
                    FROM vr FULL OUTER JOIN tr USING(chunk_key)
                )
                SELECT s.chunk_key,s.chunk_index,s.text,s.source_ids,s.spans,s.activity_days,
                       s.session_id,s.first_source_order,s.last_source_order,
                       f.score::float8 AS score,f.vector_rank,f.text_rank
                FROM fused f JOIN scoped s USING(chunk_key)
                ORDER BY f.score DESC,s.chunk_key LIMIT %(limit)s""",
                {"learner": learner_id, "actor": str(actor_id), "recipe": chunk_recipe,
                 "session": str(session_id) if session_id else None, "days": days,
                 "chunk_keys": list(candidate_chunk_keys) if candidate_chunk_keys is not None else None,
                 "excluded_keys": list(exclude_chunk_keys) if exclude_chunk_keys else None,
                 "query": query,
                 "vector": ("[" + ",".join(format(float(v), ".17g") for v in query_vector) + "]")
                           if query_vector is not None else None,
                 "identity": embedding_identity, "k": rrf_k, "limit": limit,
                 "max_distance": 1.0 - min_similarity})
            return [_row(row) for row in cur.fetchall()]

    def chunk_text_by_key(self, learner_id, actor_id, chunk_recipe, chunk_key):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT text FROM teachcopilot.unified_conversation_chunks
                WHERE learner_id=%s AND actor_id=%s AND chunk_recipe=%s AND chunk_key=%s""",
                (learner_id, str(actor_id), chunk_recipe, chunk_key))
            row = cur.fetchone()
            return row["text"] if row else None

    def conversation_chunk_texts(self, learner_id, actor_id, chunk_recipe, chunk_keys):
        """Texts of the named chunks (learner/actor-scoped), key -> text."""
        keys = [str(key) for key in chunk_keys]
        if not keys:
            return {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT chunk_key, text FROM teachcopilot.unified_conversation_chunks
                WHERE learner_id=%s AND actor_id=%s AND chunk_recipe=%s AND chunk_key = ANY(%s)""",
                (learner_id, str(actor_id), chunk_recipe, keys))
            return {row["chunk_key"]: row["text"] for row in cur.fetchall()}

    def conversation_chunk_embeddings(self, learner_id, actor_id, chunk_recipe, chunk_keys):
        """Stored chunk embedding literals (key -> pgvector text or None)."""
        keys = [str(key) for key in chunk_keys]
        if not keys:
            return {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT chunk_key, embedding::text AS vector
                FROM teachcopilot.unified_conversation_chunks
                WHERE learner_id=%s AND actor_id=%s AND chunk_recipe=%s AND chunk_key = ANY(%s)""",
                (learner_id, str(actor_id), chunk_recipe, keys))
            return {row["chunk_key"]: row["vector"] for row in cur.fetchall()}

    def get_learner_card_revision(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT revision,memory_source_revision,memory_chat_source_revision,
                unified_changes_revision FROM teachcopilot.learners WHERE learner_id=%s""", (learner_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(learner_id)
            return dict(row)
