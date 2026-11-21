"""Durable learning memory storage, idempotency and bounded worker queue.

API authorization belongs to the existing account layer. Composite foreign keys
and learner predicates additionally prevent mixing sources across learners.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import base64
import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from psycopg2.extras import Json

from .timezones import DEFAULT_DISPLAY_TIMEZONE, utc_datetime, validate_timezone


DIALOGUE_CONTEXT_CHARS = 600
SESSION_END_CONTEXT_CHARS = 1200
ARCHIVE_MEMORY_METHODS = ("personal_graph", "personal_memory")


class MemoryQueueFullError(RuntimeError):
    """The durable queue has forty waiting jobs; retry later."""


class MemoryIdempotencyConflict(ValueError):
    """An existing client identifier was reused for different input."""


class LearningSessionNotCompletedError(RuntimeError):
    """An active conversation cannot be opened as a completed archive."""


def _archive_cursor(kind, learner_id, actor_id, **values):
    payload = {"kind":kind,"learner":learner_id,"actor":str(actor_id),**values}
    return base64.urlsafe_b64encode(json.dumps(payload,separators=(',',':')).encode()).decode().rstrip('=')


def _read_archive_cursor(value, kind, learner_id, actor_id, *, session_id=None):
    if value is None:
        return None
    try:
        if not isinstance(value,str) or not 0 < len(value) <= 1024:
            raise ValueError()
        raw = base64.b64decode(value + '=' * (-len(value) % 4),altchars=b'-_',validate=True)
        if base64.urlsafe_b64encode(raw).decode().rstrip('=') != value:
            raise ValueError()
        payload = json.loads(raw)
        keys = {'kind','learner','actor','time','id'} if kind == 'sessions' else {'kind','learner','actor','session','order'}
        if (not isinstance(payload,dict) or set(payload) != keys or payload['kind'] != kind
                or payload['learner'] != learner_id or payload['actor'] != str(actor_id)):
            raise ValueError()
        if kind == 'sessions':
            if (not isinstance(payload['time'],str) or not isinstance(payload['id'],str)
                    or str(UUID(payload['id'])) != payload['id']):
                raise ValueError()
            timestamp = datetime.fromisoformat(payload['time'])
            if timestamp.tzinfo is None:
                raise ValueError()
            return timestamp.astimezone(timezone.utc),payload['id']
        if (payload['session'] != str(session_id) or type(payload['order']) is not int
                or not 0 < payload['order'] <= 9223372036854775807):
            raise ValueError()
        return payload['order']
    except (ValueError,TypeError,KeyError,UnicodeError,OverflowError) as exc:
        raise ValueError('invalid archive cursor for this conversation') from exc


def _row(value):
    if value is None:
        return None
    result = dict(value)
    for key, item in result.items():
        if key == "id" or key.endswith("_id"):
            if item is not None:
                result[key] = str(item)
    if "id" in result:
        if "kind" in result and "provider" in result:
            result["job_id"] = result["id"]
        elif "state" in result:
            result["version_id"] = result["id"]
        elif "memory_version_id" in result:
            result["next_step_id"] = result["id"]
        elif "client_session_id" in result:
            result["learning_session_id"] = result["id"]
        elif "author_role" in result:
            result["source_id"] = result["id"]
    return result


def _history_message(turn, role):
    prefix = "user" if role == "user" else "assistant"
    message = {"role": role, "content": turn[prefix + "_content"]}
    if role == "user":
        message["delivery_status"] = turn["status"]
    for field in ("source_id", "created_at", "event_timezone", "activity_timezone", "activity_day"):
        key = prefix + "_" + field
        if key not in turn:
            continue
        value = turn[key]
        if value is not None and field == "created_at":
            value = utc_datetime(value).isoformat()
        elif value is not None and field == "activity_day":
            value = value.isoformat()
        message[field] = value
    return message


def get_learner_memory_snapshot(store, learner_id, topic_id=None, *, include_personal_graph_snapshot=False):
    """Use the optional learner-wide read only for archive workflows and review."""
    reader = store.get_learner_memory
    if include_personal_graph_snapshot:
        try:
            parameters = inspect.signature(reader).parameters
        except (TypeError, ValueError):
            parameters = {}
        if ("include_personal_graph_snapshot" not in parameters
                and not any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values())):
            raise RuntimeError("personal_graph_source_snapshot_unavailable")
        snapshot = reader(learner_id, topic_id, include_personal_graph_snapshot=True)
        if (not isinstance(snapshot, dict) or not isinstance(snapshot.get("personal_graph_sources"), list)
                or not isinstance(snapshot.get("personal_graph_snapshot_fingerprint"), str)):
            raise RuntimeError("personal_graph_source_snapshot_unavailable")
        return snapshot
    return reader(learner_id, topic_id)


def _revision_error(message):
    # Store imports this mixin before declaring its public exception.
    from .store import RevisionConflict
    raise RevisionConflict(message)


def _child_chat_only_revision_change(learner, version):
    """Prove that every source revision since a draft was a child chat turn."""
    current = learner.get("memory_source_revision")
    current_chat = learner.get("memory_chat_source_revision")
    original = version.get("source_revision")
    original_chat = version.get("chat_source_revision")
    if any(type(value) is not int for value in (current, current_chat, original, original_chat)):
        return False
    difference = current - original
    return difference > 0 and current_chat - original_chat == difference


def _approved_memory_source_ids(version):
    """Recognize the complete source snapshot written by the native memory worker.

    Selected context contains only a budgeted subset and cannot determine
    freshness. Legacy versions without the native snapshot retain revision
    checks instead of being silently treated as current.
    """
    state = version.get("state")
    if version.get("approved") and isinstance(state,dict) and state.get("method") in ARCHIVE_MEMORY_METHODS:
        ids = state.get("source_ids")
        if (isinstance(ids,list) and all(isinstance(item,str) and item for item in ids)
                and len(ids) == len(set(ids))):
            return set(ids)
        return None
    if (not version.get("approved") or not isinstance(state,dict)
            or state.get("method") not in {"profile_only", "tail", "summary", "structured"}
            or not isinstance(state.get("topic_id"),str) or not state["topic_id"]
            or state["topic_id"] != version.get("topic_id")
            or not isinstance(state.get("sources"),dict)):
        return None
    sources = state["sources"]
    if any(not isinstance(source_id,str) or not source_id or not isinstance(source,dict)
           or source.get("turn_id") != source_id
           or source.get("role") not in {"student", "teacher"}
           or not isinstance(source.get("text"),str)
           or not isinstance(source.get("observed_on"),str)
           for source_id,source in sources.items()):
        return None
    return set(sources)


def _personal_memory_validation_problem(state, context, learner_id, topic_id, source_revision,
                                        sources, snapshot_fingerprint, *, approved=False, allow_snapshot_superset=False):
    """Validate a personal-memory grant against the current canonical archive.

    Selected IDs include both ends of every correction chain. The fingerprint
    covers ordered source content and attached dialogue, not just identifiers.
    The return value distinguishes changed snapshots from malformed drafts.
    """
    from .personal_memory_contract import (PERSONAL_MEMORY_CONTEXT_PROTOCOL,
        PERSONAL_MEMORY_RETRIEVAL_MODES, PERSONAL_MEMORY_SELECTION_VERSION)

    if (not isinstance(state,dict) or state.get("method") != "personal_memory"
            or state.get("learner_id") != learner_id or state.get("topic_id") != topic_id):
        return "invalid", "personal memory identity does not match the learner and topic"
    if state.get("source_context_protocol") != PERSONAL_MEMORY_CONTEXT_PROTOCOL:
        return "stale", "personal memory protocol changed"
    if state.get("selection_version") != PERSONAL_MEMORY_SELECTION_VERSION:
        return "stale", "personal memory selection version changed"
    if (not isinstance(state.get("retrieval_mode"),str)
            or state["retrieval_mode"] not in PERSONAL_MEMORY_RETRIEVAL_MODES):
        return "invalid", "personal memory retrieval mode is invalid"
    identities = state.get("model_identities")
    if not isinstance(identities,dict) or not all(
            isinstance(identities.get(kind),dict)
            and isinstance(identities[kind].get("name"),str) and identities[kind]["name"]
            and isinstance(identities[kind].get("digest"),str)
            and re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}",identities[kind]["digest"])
            for kind in ("chat","embedding")):
        return "invalid", "personal memory model identities are invalid"
    recipe = state.get("selection_recipe_fingerprint")
    if not isinstance(recipe,str) or not re.fullmatch(r"[0-9a-f]{64}",recipe):
        return "invalid", "personal memory selection recipe is invalid"
    if type(state.get("source_revision")) is not int or state["source_revision"] != source_revision:
        return "stale", "personal memory source revision changed"
    fingerprint = state.get("archive_snapshot_fingerprint")
    if (not isinstance(fingerprint,str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or state.get("source_snapshot_id") != "personal-memory-" + fingerprint):
        return "invalid", "personal memory archive fingerprint is invalid"
    if fingerprint != snapshot_fingerprint and not allow_snapshot_superset:
        return "stale", "personal memory archive changed"

    state_ids = state.get("source_ids")
    if (not isinstance(state_ids,list) or not all(isinstance(item,str) and item for item in state_ids)
            or len(state_ids) != len(set(state_ids))):
        return "invalid", "personal memory archive source IDs are invalid"
    source_map = {source["source_id"]:source for source in sources}
    if allow_snapshot_superset and not set(state_ids) <= set(source_map):
        return "stale", "personal memory archive source IDs changed"
    if not allow_snapshot_superset and set(state_ids) != set(source_map):
        return "stale", "personal memory archive source IDs changed"
    if (not isinstance(context,dict) or context.get("method") != "personal_memory"
            or context.get("topic_id") != topic_id):
        return "invalid", "personal memory context does not match the topic"
    evidence_ids = context.get("evidence_ids")
    if (not isinstance(evidence_ids,list) or not all(isinstance(item,str) and item for item in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids) <= set(state_ids)):
        return "invalid", "personal memory selected source IDs are invalid"

    projected = context.get("sources")
    if not isinstance(projected,list) or any(not isinstance(item,dict) for item in projected):
        return "invalid", "personal memory selected sources are invalid"
    projected_ids = [item.get("source_id") for item in projected]
    if (not all(isinstance(item,str) and item for item in projected_ids)
            or len(projected_ids) != len(set(projected_ids)) or set(projected_ids) != set(evidence_ids)):
        return "invalid", "personal memory selected sources do not match the evidence IDs"
    for item in projected:
        source = source_map[item["source_id"]]
        if (item.get("role") != source.get("author_role") or item.get("text") != source.get("text")
                or item.get("topic_id") != (source.get("topic_id") or "general")
                or item.get("corrects_source_id") != source.get("corrects_source_id")):
            return "invalid", "personal memory selected source differs from the archive"

    selected = set(evidence_ids)
    for source_id,source in source_map.items():
        target = source.get("corrects_source_id")
        if target is not None and ((source_id in selected) != (target in selected)):
            return "invalid", "personal memory selected sources omit a linked correction"

    if approved:
        access = context.get("source_access")
        if not isinstance(access,dict):
            return "invalid", "personal memory source approval is missing"
        access_ids = access.get("source_ids")
        if (not isinstance(access_ids,list) or not all(isinstance(item,str) and item for item in access_ids)
                or len(access_ids) != len(set(access_ids))):
            return "invalid", "personal memory approved source IDs are invalid"
        if access.get("mode") == "selected_context":
            if set(access_ids) != selected or access.get("snapshot_fingerprint") is not None:
                return "invalid", "personal memory selected-context approval is invalid"
        elif access.get("mode") == "full_archive":
            if allow_snapshot_superset or set(access_ids) != set(state_ids) or access.get("snapshot_fingerprint") != fingerprint:
                return "invalid", "personal memory full-archive approval is invalid"
        else:
            return "invalid", "personal memory source approval mode is invalid"
    return None


def _selected_snapshot_matches(state, context, sources):
    """Check the saved selected evidence against live, learner-scoped originals."""
    if not isinstance(state,dict) or not isinstance(context,dict) or not isinstance(sources,list):
        return False
    ids, evidence, projected = state.get("source_ids"), context.get("evidence_ids"), context.get("sources")
    if (not isinstance(ids,list) or not isinstance(evidence,list) or not isinstance(projected,list)
            or any(not isinstance(item,str) or not item for item in ids + evidence)
            or len(ids) != len(set(ids)) or len(evidence) != len(set(evidence))):
        return False
    if any(not isinstance(row,dict) or not isinstance(row.get("source_id"),str) or not row["source_id"] for row in sources):
        return False
    source_map = {row["source_id"]: row for row in sources}
    if len(source_map) != len(sources) or not set(ids) <= set(source_map) or not set(evidence) <= set(ids):
        return False
    if any(not isinstance(item,dict) for item in projected):
        return False
    projected_ids = [item.get("source_id") for item in projected]
    if (any(not isinstance(item,str) or not item for item in projected_ids)
            or len(projected_ids) != len(set(projected_ids)) or set(projected_ids) != set(evidence)):
        return False
    for item in projected:
        source = source_map[item["source_id"]]
        observed = str(source.get("observed_on") or source.get("created_at") or "")[:10]
        if (item.get("role") != (source.get("author_role") or source.get("role"))
                or item.get("text") != (source.get("text") or source.get("content"))
                or (item.get("topic_id") is not None and item["topic_id"] != (source.get("topic_id") or "general"))
                or item.get("corrects_source_id") != source.get("corrects_source_id")
                or (item.get("observed_on") is not None and str(item["observed_on"]) != observed)):
            return False
    selected = set(evidence)
    for source_id,source in source_map.items():
        target = source.get("corrects_source_id")
        if target is not None and ((source_id in selected) != (target in selected)):
            return False
    return isinstance(context.get("text"),str)


def reviewed_snapshot_context(snapshot, version, learner_id, topic_id):
    """Return only an explicitly confirmed historical selection, or fail closed."""
    state, context = version.get("state"), version.get("context")
    if not isinstance(state,dict) or not isinstance(context,dict):
        return None
    review, access = context.get("snapshot_review"), context.get("source_access")
    if (not version.get("approved") or snapshot.get("learner_id") != learner_id
            or version.get("topic_id") != topic_id or version.get("base_revision") != snapshot.get("revision")
            or not isinstance(review,dict) or review.get("new_chat_since_snapshot") is not True
            or review.get("refresh_required") is not True
            or review.get("snapshot_source_revision") != version.get("source_revision")
            or type(review.get("confirmed_source_revision")) is not int
            or type(version.get("source_revision")) is not int
            or type(snapshot.get("source_revision")) is not int
            or review["confirmed_source_revision"] <= version["source_revision"]
            or review["confirmed_source_revision"] > snapshot["source_revision"]
            or not isinstance(access,dict) or access.get("mode") != "selected_context"
            or access.get("snapshot_fingerprint") is not None):
        return None
    learner = {"memory_source_revision": snapshot.get("source_revision"),
               "memory_chat_source_revision": snapshot.get("chat_source_revision")}
    if not _child_chat_only_revision_change(learner,version):
        return None
    evidence, allowed = context.get("evidence_ids"), access.get("source_ids")
    if (not isinstance(evidence,list) or not isinstance(allowed,list)
            or any(not isinstance(item,str) or not item for item in evidence + allowed)
            or len(allowed) != len(set(allowed)) or set(allowed) != set(evidence)):
        return None
    sources = snapshot.get("sources")
    if not _selected_snapshot_matches(state,context,sources):
        return None
    method = state.get("method")
    if method == "personal_memory":
        from .personal_graph_contract import personal_graph_snapshot_fingerprint
        try:
            fingerprint = personal_graph_snapshot_fingerprint(sources)
        except (TypeError,ValueError):
            return None
        if _personal_memory_validation_problem(state,context,learner_id,topic_id,version["source_revision"],
                sources,fingerprint,approved=True,allow_snapshot_superset=True) is not None:
            return None
    elif method == "personal_graph":
        from .personal_graph_contract import PERSONAL_GRAPH_CONTEXT_PROTOCOL
        if (state.get("learner_id") != learner_id or state.get("topic_id") != topic_id
                or state.get("source_revision") != version.get("source_revision")
                or state.get("source_context_protocol") != PERSONAL_GRAPH_CONTEXT_PROTOCOL
                or not state.get("archive_run_id")):
            return None
    else:
        return None
    result = deepcopy(context)
    result.update(method=method,topic_id=topic_id,version_id=version.get("version_id") or version.get("id"),
                  stale=True,snapshot_is_frozen=True)
    return result


class MemoryStoreMixin:
    @staticmethod
    def _student_session_actor(cur, session):
        cur.execute("""SELECT 1 FROM teachcopilot.accounts a
            JOIN teachcopilot.account_learners al ON al.account_id=a.id
            WHERE a.id::text=%s AND a.role='student' AND a.active
                AND al.learner_id=%s AND al.relationship='student'""",
                    (str(session["actor_id"]), session["learner_id"]))
        return cur.fetchone() is not None

    def guided_division_context(self, learner_id, topic_id, actor_id, **kwargs):
        from .guided_division_store import guided_context
        return guided_context(self, learner_id, topic_id, actor_id, **kwargs)

    def autonomous_division_context(self, learner_id, topic_id, actor_id, **kwargs):
        from .autonomous_division_store import autonomous_context
        return autonomous_context(self, learner_id, topic_id, actor_id, **kwargs)

    @staticmethod
    def _memory_learner(cur, learner_id, expected_revision=None, lock="UPDATE"):
        if lock not in ("UPDATE", "SHARE"):
            raise ValueError("invalid lock")
        cur.execute("SELECT revision,memory_source_revision,memory_chat_source_revision FROM teachcopilot.learners WHERE learner_id=%s FOR " + lock, (learner_id,))
        learner = cur.fetchone()
        if learner is None:
            raise KeyError(learner_id)
        if expected_revision is not None and learner["revision"] != expected_revision:
            _revision_error(f"learner revision is {learner['revision']}, expected {expected_revision}")
        return learner

    @staticmethod
    def _memory_source(cur, learner_id, topic_id, actor_id, content, *, author_role="teacher", session_id=None, approved=False, observed_on=None, source_key=None, corrects_source_id=None, event_timezone=None, activity_timezone=None, child_chat=False):
        if author_role not in ("student", "teacher", "assistant"):
            raise ValueError("invalid source author")
        if child_chat and (author_role not in ("student", "assistant") or session_id is None):
            raise ValueError("child chat source requires a student or assistant in a learning session")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("source content is required")
        event_timezone = validate_timezone(DEFAULT_DISPLAY_TIMEZONE if event_timezone is None else event_timezone)
        if activity_timezone is not None:
            activity_timezone = validate_timezone(activity_timezone)
        if corrects_source_id is not None:
            if author_role != "teacher" or approved is not True:
                raise ValueError("a source correction must be an approved teacher note")
            MemoryStoreMixin._correction_target(cur,learner_id,corrects_source_id,topic_id)
        cur.execute("""INSERT INTO teachcopilot.memory_sources
            (learner_id,topic_id,actor_id,content,author_role,learning_session_id,approved,observed_on,source_key,corrects_source_id,event_timezone,activity_timezone)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                COALESCE(%s,(SELECT learning_timezone FROM teachcopilot.learners WHERE learner_id=%s)))
            ON CONFLICT(learner_id,source_key) DO NOTHING RETURNING *""",
            (learner_id,topic_id,str(actor_id),content,author_role,session_id,approved,observed_on,source_key,corrects_source_id,event_timezone,activity_timezone,learner_id))
        source = cur.fetchone()
        if source is None:
            cur.execute("SELECT * FROM teachcopilot.memory_sources WHERE learner_id=%s AND source_key=%s", (learner_id,source_key))
            source = cur.fetchone()
            if (source["content"] != content or source["author_role"] != author_role or source["topic_id"] != topic_id
                    or str(source["actor_id"]) != str(actor_id) or source["corrects_source_id"] != corrects_source_id):
                raise MemoryIdempotencyConflict("source key was reused for different content")
        else:
            cur.execute("""UPDATE teachcopilot.learners SET memory_source_revision=memory_source_revision+1,
                memory_chat_source_revision=memory_chat_source_revision+%s WHERE learner_id=%s""",
                (1 if child_chat else 0, learner_id))
        return _row(source)

    def add_memory_source(self, learner_id, topic_id, content, actor_id, *, author_role="teacher", approved=False, observed_on=None, source_key=None, expected_revision=None, corrects_source_id=None, event_timezone=None):
        with self._connect() as conn, conn.cursor() as cur:
            self._memory_learner(cur,learner_id,expected_revision)
            return self._memory_source(cur,learner_id,topic_id,actor_id,content,author_role=author_role,approved=approved,observed_on=observed_on,source_key=source_key,corrects_source_id=corrects_source_id,event_timezone=event_timezone)

    @staticmethod
    def _correction_target(cur, learner_id, source_id, topic_id=None):
        if not isinstance(source_id,str) or not source_id:
            raise ValueError("correction target source is required")
        cur.execute("""SELECT s.*,ls.status AS session_status FROM teachcopilot.memory_sources s
            LEFT JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
            WHERE s.id=%s AND s.learner_id=%s""", (source_id,learner_id))
        source = cur.fetchone()
        if source is None:
            raise KeyError(source_id)
        if topic_id is not None and source["topic_id"] != topic_id:
            raise ValueError("correction target must have the same topic")
        if not ((source["author_role"] == "teacher" and source["approved"]
                 and (source["learning_session_id"] is None or source["session_status"] == "completed"))
                or (source["author_role"] == "student" and source["session_status"] == "completed")):
            raise ValueError("correction target must be an eligible student or approved teacher source")
        return source

    def correct_memory_source(self, learner_id, source_id, text, actor_id, *, expected_revision,
                              expected_source_revision, client_correction_id):
        """Append one reviewed correction and its durable update job atomically."""
        if not isinstance(text,str) or not text.strip() or len(text)>2000:
            raise ValueError("correction text must contain 1 to 2000 characters")
        if (not isinstance(client_correction_id,str) or not client_correction_id.strip()
                or len(client_correction_id)>180):
            raise ValueError("client correction identifier must contain 1 to 180 characters")
        with self._connect() as conn, conn.cursor() as cur:
            # SP-1: take the queue advisory lock before the learners row lock
            # (this transaction enqueues via _enqueue_memory_job further down).
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            learner = self._memory_learner(cur,learner_id)
            source_key = "correction:" + client_correction_id
            cur.execute("SELECT * FROM teachcopilot.memory_sources WHERE learner_id=%s AND source_key=%s",
                        (learner_id,source_key))
            existing = cur.fetchone()
            if existing is not None:
                if (existing["content"] != text or existing["corrects_source_id"] != source_id
                        or str(existing["actor_id"]) != str(actor_id) or existing["author_role"] != "teacher"
                        or not existing["approved"]):
                    raise MemoryIdempotencyConflict("correction identifier was reused for different input")
                source = _row(existing)
                cur.execute("""SELECT * FROM teachcopilot.memory_jobs
                    WHERE learner_id=%s AND kind='memory_update' AND idempotency_key=%s""",
                    (learner_id,f"source:{source['id']}"))
                job = cur.fetchone()
                if job is None:
                    raise MemoryIdempotencyConflict("correction has no original update job")
                return {"source":source,"job_id":str(job["id"]),"learner_id":learner_id,
                        "topic_id":source["topic_id"]}
            if learner["revision"] != expected_revision:
                _revision_error("learner revision changed before source correction")
            if learner["memory_source_revision"] != expected_source_revision:
                _revision_error("learning sources changed before source correction")
            target = self._correction_target(cur,learner_id,source_id)
            source = self._memory_source(cur,learner_id,target["topic_id"],actor_id,text,
                approved=True,source_key=source_key,corrects_source_id=source_id)
            job = self._enqueue_memory_job(cur,learner_id,target["topic_id"],"memory_update",
                {"source_id":source["id"]},actor_id,expected_revision=expected_revision,
                method="tail",idempotency_key=f"source:{source['id']}")
            return {"source":source,"job_id":job["id"],"learner_id":learner_id,"topic_id":target["topic_id"]}

    def create_learning_session(self, learner_id, actor_id, topic_id, client_session_id=None, mode="standard", event_timezone=None):
        if not topic_id or not str(topic_id).strip():
            raise ValueError("topic_id is required")
        if mode not in {"standard", "autonomous_division"}:
            raise ValueError("unknown learning session mode")
        event_timezone = validate_timezone(DEFAULT_DISPLAY_TIMEZONE if event_timezone is None else event_timezone)
        if mode == "autonomous_division":
            from .autonomous_division import TOPIC as AUTONOMOUS_DIVISION_TOPIC
            if topic_id != AUTONOMOUS_DIVISION_TOPIC:
                raise ValueError("autonomous division requires its supported math topic")
        with self._connect() as conn, conn.cursor() as cur:
            self._memory_learner(cur,learner_id)
            cur.execute("""INSERT INTO teachcopilot.learning_sessions(learner_id,actor_id,topic_id,client_session_id,session_mode,event_timezone)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(learner_id,actor_id,client_session_id) DO NOTHING RETURNING *""",
                (learner_id,str(actor_id),topic_id,client_session_id,mode,event_timezone))
            session = cur.fetchone()
            if session is None:
                cur.execute("SELECT * FROM teachcopilot.learning_sessions WHERE learner_id=%s AND actor_id=%s AND client_session_id=%s", (learner_id,str(actor_id),client_session_id))
                session = cur.fetchone()
                if session["topic_id"] != topic_id or session["session_mode"] != mode:
                    raise MemoryIdempotencyConflict("session identifier belongs to another topic or mode")
            return _row(session)

    def ensure_conversation_session(self, learner_id, actor_id, topic_id, event_timezone=None):
        """Reuse an actor's standard dialogue channel for the ordinary chat API."""
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise ValueError("topic_id is required")
        zone = validate_timezone(DEFAULT_DISPLAY_TIMEZONE if event_timezone is None else event_timezone)
        with self._connect() as conn, conn.cursor() as cur:
            # Session creation takes this same lock, so concurrent ordinary
            # requests cannot create duplicate active channels for this scope.
            self._memory_learner(cur, learner_id)
            cur.execute("""SELECT * FROM teachcopilot.learning_sessions
                WHERE learner_id=%s AND actor_id=%s AND topic_id=%s
                    AND status='active' AND session_mode='standard'
                ORDER BY created_at DESC,id DESC LIMIT 1""", (learner_id, str(actor_id), topic_id))
            existing = cur.fetchone()
            if existing is not None:
                return _row(existing)
            cur.execute("""INSERT INTO teachcopilot.learning_sessions
                (learner_id,actor_id,topic_id,session_mode,event_timezone)
                VALUES (%s,%s,%s,'standard',%s) RETURNING *""", (learner_id, str(actor_id), topic_id, zone))
            return _row(cur.fetchone())

    def get_learning_session(self, session_id, learner_id=None):
        # Audit F-074: a malformed id used to reach the uuid cast and surface as
        # a misleading 502 from the chat endpoint; no such session exists.
        try:
            UUID(str(session_id))
        except (ValueError, TypeError, AttributeError):
            return None
        with self._connect() as conn, conn.cursor() as cur:
            query = "SELECT * FROM teachcopilot.learning_sessions WHERE id=%s"
            values = [session_id]
            if learner_id is not None:
                query += " AND learner_id=%s"
                values.append(learner_id)
            cur.execute(query,values)
            return _row(cur.fetchone())

    def current_learning_session(self, learner_id, actor_id):
        """Read one actor's active dialogue without replaying or repairing calls.

        The bounded window contains whole message turns. Source text is the
        only public content; cached responses can contain private diagnostics
        and typed-action provenance and never enter this projection.
        """
        with self._connect() as connection, connection.cursor() as cur:
            connection.set_session(readonly=True, isolation_level='REPEATABLE READ')
            cur.execute('''SELECT id,topic_id,session_mode FROM teachcopilot.learning_sessions
                WHERE learner_id=%s AND actor_id=%s AND status='active'
                ORDER BY created_at DESC,id DESC LIMIT 1''', (learner_id,str(actor_id)))
            session = cur.fetchone()
            if session is None:
                return {'available': False}
            session_id, topic_id = session['id'], session['topic_id']
            cur.execute('''SELECT bool_or(status='pending') AS pending,
                bool_or(status='unknown_outcome') AS unknown_outcome
                FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND learner_id=%s''',
                (session_id,learner_id))
            unresolved = cur.fetchone()
            unresolved_status = ('pending' if unresolved['pending'] else
                                 'unknown_outcome' if unresolved['unknown_outcome'] else None)
            cur.execute('''SELECT m.status,s.content AS user_content,a.content AS assistant_content,
                s.id AS user_source_id,s.created_at AS user_created_at,s.event_timezone AS user_event_timezone,
                s.activity_timezone AS user_activity_timezone,s.activity_day AS user_activity_day,
                a.id AS assistant_source_id,a.created_at AS assistant_created_at,a.event_timezone AS assistant_event_timezone,
                a.activity_timezone AS assistant_activity_timezone,a.activity_day AS assistant_activity_day
                FROM teachcopilot.learning_messages m
                JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
                    AND s.learning_session_id=m.learning_session_id AND s.topic_id=%s
                    AND s.actor_id=%s AND s.author_role IN ('student','teacher')
                LEFT JOIN teachcopilot.memory_sources a ON m.status='completed'
                    AND a.id=m.assistant_source_id AND a.learner_id=m.learner_id
                    AND a.learning_session_id=m.learning_session_id AND a.topic_id=s.topic_id
                    AND a.author_role='assistant'
                WHERE m.learning_session_id=%s AND m.learner_id=%s
                ORDER BY s.source_order DESC LIMIT 21''',
                (topic_id,str(actor_id),session_id,learner_id))
            turns = cur.fetchall()
            messages = []
            for turn in reversed(turns[:20]):
                messages.append(_history_message(turn, "user"))
                if turn['status'] == 'completed' and turn['assistant_content'] is not None:
                    messages.append(_history_message(turn, "assistant"))
            return {'available':True,'learning_session_id':str(session_id),'topic_id':topic_id,
                    'session_mode':session['session_mode'],'status':'active','messages':messages,'history_truncated':len(turns) > 20,
                    'unresolved_status':unresolved_status}

    def list_completed_learning_sessions(self, learner_id, actor_id, *, before=None):
        """List only the actor's completed conversations using stable keyset pages."""
        boundary = _read_archive_cursor(before,'sessions',learner_id,actor_id)
        with self._connect() as connection, connection.cursor() as cur:
            connection.set_session(readonly=True,isolation_level='REPEATABLE READ')
            query = '''SELECT id,topic_id,session_mode,created_at,event_timezone,COALESCE(completed_at,created_at) AS ended_at
                FROM teachcopilot.learning_sessions
                WHERE learner_id=%s AND actor_id=%s AND status='completed' '''
            values = [learner_id,str(actor_id)]
            if boundary is not None:
                query += 'AND (COALESCE(completed_at,created_at),id)<(%s,%s::uuid) '
                values.extend(boundary)
            cur.execute(query+'ORDER BY COALESCE(completed_at,created_at) DESC,id DESC LIMIT 11',values)
            rows = cur.fetchall()
            sessions = [{'learning_session_id':str(row['id']),'topic_id':row['topic_id'],
                         'session_mode':row['session_mode'],
                         'completed_at':row['ended_at'].isoformat(),
                         **({'created_at':utc_datetime(row['created_at']).isoformat()} if row.get('created_at') else {}),
                         **({'event_timezone':row['event_timezone']} if 'event_timezone' in row else {})} for row in rows[:10]]
            next_cursor = None
            if len(rows) > 10:
                last = rows[9]
                next_cursor = _archive_cursor('sessions',learner_id,actor_id,
                    time=last['ended_at'].isoformat(),id=str(last['id']))
            return {'sessions':sessions,'next_cursor':next_cursor}

    def completed_learning_session_history(self, learner_id, session_id, actor_id, *, before=None):
        """Read canonical archived turns, never a cached model response or a replay."""
        boundary = _read_archive_cursor(before,'history',learner_id,actor_id,session_id=session_id)
        with self._connect() as connection, connection.cursor() as cur:
            connection.set_session(readonly=True,isolation_level='REPEATABLE READ')
            cur.execute('''SELECT actor_id,topic_id,status,session_mode FROM teachcopilot.learning_sessions
                WHERE id=%s AND learner_id=%s''',(session_id,learner_id))
            session = cur.fetchone()
            if session is None:
                raise KeyError(session_id)
            if str(session['actor_id']) != str(actor_id):
                raise PermissionError('archived conversation belongs to another actor')
            if session['status'] != 'completed':
                raise LearningSessionNotCompletedError('only completed sessions have archived history')
            query = '''SELECT m.status,s.source_order,s.content AS user_content,a.content AS assistant_content,
                s.id AS user_source_id,s.created_at AS user_created_at,s.event_timezone AS user_event_timezone,
                s.activity_timezone AS user_activity_timezone,s.activity_day AS user_activity_day,
                a.id AS assistant_source_id,a.created_at AS assistant_created_at,a.event_timezone AS assistant_event_timezone,
                a.activity_timezone AS assistant_activity_timezone,a.activity_day AS assistant_activity_day
                FROM teachcopilot.learning_messages m
                JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
                    AND s.learning_session_id=m.learning_session_id AND s.topic_id=%s
                    AND s.actor_id=%s AND s.author_role IN ('student','teacher')
                LEFT JOIN teachcopilot.memory_sources a ON m.status='completed'
                    AND a.id=m.assistant_source_id AND a.learner_id=m.learner_id
                    AND a.learning_session_id=m.learning_session_id AND a.topic_id=s.topic_id
                    AND a.author_role='assistant'
                WHERE m.learning_session_id=%s AND m.learner_id=%s '''
            values = [session['topic_id'],str(actor_id),session_id,learner_id]
            if boundary is not None:
                query += 'AND s.source_order<%s '
                values.append(boundary)
            cur.execute(query+'ORDER BY s.source_order DESC LIMIT 21',values)
            rows = cur.fetchall()
            messages = []
            for turn in reversed(rows[:20]):
                messages.append(_history_message(turn, "user"))
                if turn['status'] == 'completed' and turn['assistant_content'] is not None:
                    messages.append(_history_message(turn, "assistant"))
            next_cursor = (_archive_cursor('history',learner_id,actor_id,session=str(session_id),
                            order=rows[19]['source_order']) if len(rows) > 20 else None)
            return {'learning_session_id':str(session_id),'topic_id':session['topic_id'],
                    'session_mode':session.get('session_mode','standard'),
                    'status':'completed','messages':messages,'next_cursor':next_cursor}

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, *, author_role="student", event_timezone=None):
        if author_role not in ("student", "teacher"):
            raise ValueError("chat source must be student or teacher")
        if not client_message_id or len(client_message_id) > 200:
            raise ValueError("client_message_id must contain 1..200 characters")
        if not isinstance(content,str) or not content.strip():
            raise ValueError("message content is required")
        if event_timezone is not None:
            event_timezone = validate_timezone(event_timezone)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM teachcopilot.learning_sessions WHERE id=%s AND learner_id=%s FOR UPDATE", (session_id,learner_id))
            session = cur.fetchone()
            if session is None:
                raise KeyError(session_id)
            if str(session["actor_id"]) != str(actor_id):
                raise PermissionError("learning session belongs to another actor")
            self._recover_chat_messages(cur,600,session_id)
            cur.execute("SELECT * FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND client_message_id=%s", (session_id,client_message_id))
            existing = cur.fetchone()
            if existing is not None:
                if existing["content_hash"] != digest:
                    raise MemoryIdempotencyConflict("client_message_id was reused for different text")
                return dict(_row(existing),created=False)
            if session["status"] != "active":
                raise ValueError("learning session is already completed")
            # Serialize turns within a session so server-built dialogue is complete.
            cur.execute("SELECT 1 FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND status='pending'", (session_id,))
            if cur.fetchone():
                raise ValueError("another message in this session is still pending")
            child_chat = author_role == "student" and self._student_session_actor(cur,session)
            source = self._memory_source(cur,learner_id,session["topic_id"],actor_id,content,author_role=author_role,session_id=session_id,approved=author_role=="teacher",source_key=f"chat:{session_id}:{client_message_id}:student",event_timezone=session.get("event_timezone") or event_timezone,child_chat=child_chat)
            cur.execute("""INSERT INTO teachcopilot.learning_messages(learning_session_id,learner_id,client_message_id,source_id,content_hash)
                VALUES (%s,%s,%s,%s,%s) RETURNING *""", (session_id,learner_id,client_message_id,source["id"],digest))
            return dict(_row(cur.fetchone()),created=True)

    def complete_chat_message(self, session_id, client_message_id, response):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND client_message_id=%s FOR UPDATE", (session_id,client_message_id))
            message = cur.fetchone()
            if message is None:
                raise KeyError(client_message_id)
            if message["status"] == "completed":
                return _row(message)
            if message["status"] != "pending":
                raise ValueError("message outcome cannot be automatically retried")
            cur.execute("SELECT * FROM teachcopilot.learning_sessions WHERE id=%s", (session_id,))
            session = cur.fetchone()
            if isinstance(response,str):
                text = response
            elif isinstance(response,dict) and response.get("choices"):
                text = str(response["choices"][0].get("message",{}).get("content") or "")
            else:
                text = next((str(response[key]) for key in ("answer","response","content","text") if isinstance(response,dict) and response.get(key)),json.dumps(response,ensure_ascii=False))
            cur.execute("SELECT author_role,actor_id,event_timezone FROM teachcopilot.memory_sources WHERE id=%s AND learner_id=%s",
                        (message["source_id"], message["learner_id"]))
            original = cur.fetchone()
            if original is None:
                raise ValueError("chat message source is unavailable")
            event_timezone = session.get("event_timezone") or original.get("event_timezone")
            child_chat = (original["author_role"] == "student" and str(original["actor_id"]) == str(session["actor_id"])
                          and self._student_session_actor(cur,session))
            source = self._memory_source(cur,message["learner_id"],session["topic_id"],"assistant",text,author_role="assistant",session_id=session_id,source_key=f"chat:{session_id}:{client_message_id}:assistant",event_timezone=event_timezone,child_chat=child_chat)
            cur.execute("""UPDATE teachcopilot.learning_messages SET status='completed',response=%s,assistant_source_id=%s,completed_at=now()
                WHERE id=%s RETURNING *""", (Json(response),source["id"],message["id"]))
            return _row(cur.fetchone())

    def fail_chat_message(self, session_id, client_message_id, error=None, unknown_outcome=True):
        status = "unknown_outcome" if unknown_outcome else "failed"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.learning_messages SET status=%s,error=%s,completed_at=now()
                WHERE learning_session_id=%s AND client_message_id=%s AND status='pending' RETURNING *""", (status,str(error or status),session_id,client_message_id))
            row = cur.fetchone()
            if row is None:
                cur.execute("SELECT * FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND client_message_id=%s", (session_id,client_message_id))
                row = cur.fetchone()
            if row is None:
                raise KeyError(client_message_id)
            return _row(row)

    @staticmethod
    def _recover_chat_messages(cur, older_than_seconds=600, session_id=None):
        values = [max(1,int(older_than_seconds))]
        session_filter = ""
        if session_id is not None:
            session_filter = " AND m.learning_session_id=%s"
            values.append(session_id)
        cur.execute("""UPDATE teachcopilot.learning_messages m SET status='unknown_outcome',
            error='pending chat reservation expired; outcome requires review',completed_at=now()
            WHERE m.status='pending' AND m.created_at < now()-(%s * interval '1 second')""" + session_filter + """
            AND NOT EXISTS (SELECT 1 FROM teachcopilot.memory_jobs j WHERE j.learner_id=m.learner_id
                AND j.kind='chat' AND j.idempotency_key='chat:'||m.learning_session_id::text||':'||m.client_message_id
                AND j.status='running' AND j.lease_until>now())""", values)
        recovered = cur.rowcount
        # A synchronous request owns queued chats. After its admission deadline
        # and reservation recovery there is nobody who can execute this entry.
        # Retire it so crashed clients cannot permanently consume the queue cap.
        cur.execute("""UPDATE teachcopilot.memory_jobs j SET status='failed',
            error='orphaned chat admission expired before provider call',completed_at=now()
            WHERE j.kind='chat' AND j.status='queued' AND (
                j.created_at < now()-(%s * interval '1 second') OR EXISTS (
                    SELECT 1 FROM teachcopilot.learning_messages m WHERE m.learner_id=j.learner_id
                    AND j.idempotency_key='chat:'||m.learning_session_id::text||':'||m.client_message_id
                    AND m.status IN ('failed','unknown_outcome')))""", (max(1,int(older_than_seconds)),))
        return recovered

    def recover_chat_messages(self, older_than_seconds=600):
        with self._connect() as conn, conn.cursor() as cur:
            return self._recover_chat_messages(cur,older_than_seconds)

    def list_learning_messages(self, session_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT id AS source_id,learning_session_id AS session_id,author_role,
                CASE WHEN author_role IN ('student','teacher') THEN 'user' ELSE author_role END AS role,content,created_at,event_timezone
                FROM teachcopilot.memory_sources WHERE learning_session_id=%s ORDER BY source_order""", (session_id,))
            return [_row(row) for row in cur.fetchall()]

    def get_own_conversation_messages(self, learner_id, actor_id, *, limit=20,
                                      active_day_limit=None, learning_timezone=None,
                                      exclude_source_id=None, before_source_order=None, as_of=None,
                                      _cursor=None):
        """Read one actor's own messages and associated assistant context.

        Every actual student message remains eligible even if generation failed.
        Assistant text requires its completed pair or a recorded shown check-in.
        The limit counts individual messages. Day windows use immutable activity
        dates captured at insertion, independently of display preferences.
        """
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 10000):
            raise ValueError("message limit must be between 1 and 10000")
        if active_day_limit is not None and (type(active_day_limit) is not int or not 1 <= active_day_limit <= 366):
            raise ValueError("active day limit must be between 1 and 366")
        if before_source_order is not None and (type(before_source_order) is not int
                or not 1 <= before_source_order <= 9223372036854775807):
            raise ValueError("message boundary must be a positive source order")
        if exclude_source_id is not None and (not isinstance(exclude_source_id, str) or not exclude_source_id):
            raise ValueError("excluded source ID must be nonempty")
        if learning_timezone is not None:
            validate_timezone(learning_timezone)
        cutoff = utc_datetime(as_of) if as_of is not None else datetime.now(timezone.utc)
        values = {"learner": learner_id, "actor": str(actor_id), "limit": limit,
                  "day_limit": active_day_limit, "excluded": exclude_source_id,
                  "before": before_source_order, "cutoff": cutoff}
        def query(cur):
            cur.execute("""WITH own_pairs AS (
                SELECT m.*,s.topic_id,s.actor_id,account.role AS actor_role
                FROM teachcopilot.learning_messages m
                JOIN teachcopilot.learning_sessions ls ON ls.id=m.learning_session_id
                    AND ls.learner_id=m.learner_id AND ls.actor_id=%(actor)s
                JOIN teachcopilot.accounts account ON account.id::text=ls.actor_id AND account.role IN ('student','teacher')
                JOIN teachcopilot.memory_sources s ON s.id=m.source_id AND s.learner_id=m.learner_id
                    AND s.learning_session_id=m.learning_session_id AND s.topic_id=ls.topic_id
                    AND s.actor_id=ls.actor_id
                    AND (s.author_role=account.role OR (account.role='teacher' AND s.author_role='student'))
                WHERE m.learner_id=%(learner)s
                    AND (%(excluded)s::text IS NULL OR m.source_id<>%(excluded)s)
            ), own_messages AS (
                SELECT s.id AS source_id,s.source_order,m.actor_role AS author_role,'user'::text AS role,
                    s.content AS text,s.created_at,s.learning_session_id,s.topic_id,
                    s.event_timezone,s.activity_timezone,s.activity_day,
                    'self_report'::text AS usage,m.status AS delivery_status,
                    m.source_id AS pair_source_id,
                    CASE WHEN m.status='completed' AND m.completed_at<=%(cutoff)s
                        THEN m.assistant_source_id END AS paired_source_id,
                    m.actor_role,s.author_role AS original_source_role,
                    s.learner_id AS learner_id,s.actor_id AS actor_id,
                    corr.id AS superseded_by
                FROM own_pairs m JOIN teachcopilot.memory_sources s
                    ON s.id=m.source_id AND s.learner_id=m.learner_id
                LEFT JOIN LATERAL (
                    SELECT c.id FROM teachcopilot.memory_sources c
                    WHERE c.corrects_source_id=s.id AND c.learner_id=s.learner_id
                    ORDER BY c.created_at,c.id LIMIT 1) corr ON TRUE
                UNION ALL
                SELECT a.id,a.source_order,a.author_role,'assistant',a.content,a.created_at,
                    a.learning_session_id,a.topic_id,a.event_timezone,a.activity_timezone,a.activity_day,
                    'context_only',m.status,m.source_id,m.source_id,m.actor_role,a.author_role,
                    a.learner_id AS learner_id,m.actor_id AS actor_id,
                    NULL::text AS superseded_by
                FROM own_pairs m JOIN teachcopilot.memory_sources a
                    ON a.id=m.assistant_source_id AND a.learner_id=m.learner_id
                    AND a.learning_session_id=m.learning_session_id AND a.topic_id=m.topic_id
                    AND a.author_role='assistant'
                WHERE m.status='completed' AND m.completed_at<=%(cutoff)s
                UNION ALL
                SELECT a.id,a.source_order,a.author_role,'assistant',a.content,a.created_at,
                    a.learning_session_id,a.topic_id,a.event_timezone,a.activity_timezone,a.activity_day,
                    'context_only','shown',NULL::text,NULL::text,account.role,a.author_role,
                    a.learner_id AS learner_id,c.actor_id AS actor_id,
                    NULL::text AS superseded_by
                FROM teachcopilot.daily_checkins c
                JOIN teachcopilot.accounts account ON account.id::text=c.actor_id AND account.role='student'
                JOIN teachcopilot.memory_sources a ON a.id=c.assistant_source_id
                    AND a.learner_id=c.learner_id
                    AND a.learning_session_id IS NOT DISTINCT FROM c.learning_session_id
                    AND a.author_role='assistant'
                WHERE c.learner_id=%(learner)s AND c.actor_id=%(actor)s
                    AND c.shown_at IS NOT NULL AND c.shown_at<=%(cutoff)s
            ), eligible AS (
                SELECT * FROM own_messages
                WHERE created_at<=%(cutoff)s
                    AND (%(before)s::bigint IS NULL OR source_order<%(before)s)
            ), selected_days AS (
                SELECT DISTINCT activity_day FROM eligible
                ORDER BY activity_day DESC LIMIT %(day_limit)s
            ) SELECT * FROM eligible
                WHERE %(day_limit)s::integer IS NULL OR activity_day IN (SELECT activity_day FROM selected_days)
                ORDER BY source_order DESC LIMIT %(limit)s""", values)
            return cur.fetchall()

        if _cursor is not None:
            # A caller-held read-only snapshot (get_own_conversation_message_windows).
            rows = query(_cursor)
        else:
            with self._connect() as connection, connection.cursor() as cur:
                connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
                rows = query(cur)
        messages = []
        for row in reversed(rows):
            message = _row(row)
            message["created_at"] = utc_datetime(message["created_at"])
            message["content"] = message["text"]
            messages.append(message)
        return messages

    def get_own_conversation_message_windows(self, learner_id, actor_id, windows, **options):
        """Audit F-060: read several message windows (for example the active-day
        window and the recent tail) from ONE read-only REPEATABLE READ snapshot,
        so a message committed between two separate reads cannot appear in one
        view and be missing from the other."""
        with self._connect() as connection, connection.cursor() as cur:
            connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
            return [self.get_own_conversation_messages(learner_id, actor_id, _cursor=cur, **window, **options)
                    for window in windows]

    def complete_learning_session(self, session_id, learner_id, actor_id):
        with self._connect() as conn, conn.cursor() as cur:
            # Take the queue advisory lock before any session or message row
            # lock, matching the claim/enqueue order (advisory lock first, row
            # locks after). Row locks first invert the order against
            # claim_memory_job, which holds the advisory lock while row-locking
            # the same pending messages, and PostgreSQL aborts one transaction
            # with a 40P01 deadlock error.
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            cur.execute("SELECT * FROM teachcopilot.learning_sessions WHERE id=%s AND learner_id=%s FOR UPDATE", (session_id,learner_id))
            session = cur.fetchone()
            if session is None:
                raise KeyError(session_id)
            if str(session["actor_id"]) != str(actor_id):
                raise PermissionError("learning session belongs to another actor")
            self._recover_chat_messages(cur,600,session_id)
            cur.execute("SELECT 1 FROM teachcopilot.learning_messages WHERE learning_session_id=%s AND status='pending'", (session_id,))
            if cur.fetchone():
                raise ValueError("cannot complete a session with a pending message")
            if session["status"] != "completed":
                cur.execute("UPDATE teachcopilot.learning_sessions SET status='completed',completed_at=now() WHERE id=%s RETURNING *", (session_id,))
                session = cur.fetchone()
                child_completion = False
                if self._student_session_actor(cur,session):
                    cur.execute("""SELECT EXISTS (
                            SELECT 1 FROM teachcopilot.learning_messages m
                            JOIN teachcopilot.memory_sources first ON first.id=m.source_id AND first.learner_id=m.learner_id
                            WHERE m.learning_session_id=%s AND m.learner_id=%s AND first.author_role='student'
                                AND first.actor_id=%s
                        ) AS has_student_turn,
                        NOT EXISTS (
                            SELECT 1 FROM teachcopilot.memory_sources s
                            WHERE s.learning_session_id=%s AND s.learner_id=%s AND NOT EXISTS (
                                SELECT 1 FROM teachcopilot.learning_messages m
                                JOIN teachcopilot.memory_sources first ON first.id=m.source_id AND first.learner_id=m.learner_id
                                WHERE m.learning_session_id=s.learning_session_id AND m.learner_id=s.learner_id
                                    AND first.author_role='student' AND first.actor_id=%s
                                    AND (s.id=m.source_id OR s.id=m.assistant_source_id)
                            )
                        ) AS only_student_chat""",
                        (session_id,learner_id,str(session["actor_id"]),session_id,learner_id,str(session["actor_id"])))
                    classification = cur.fetchone()
                    child_completion = bool(classification["has_student_turn"] and classification["only_student_chat"])
                cur.execute("""UPDATE teachcopilot.learners SET memory_source_revision=memory_source_revision+1,
                    memory_chat_source_revision=memory_chat_source_revision+%s WHERE learner_id=%s""",
                    (1 if child_completion else 0,learner_id))
            if session.get("session_mode") == "autonomous_division":
                return dict(_row(session),job=None,job_id=None)
            job = self._enqueue_memory_job(cur,learner_id,session["topic_id"],"memory_update",{"learning_session_id":str(session_id)},actor_id,idempotency_key=f"session:{session_id}",method="tail")
            return dict(_row(session),job=job,job_id=job["id"])

    def _enqueue_memory_job(self, cur, learner_id, topic_id, kind, payload=None, actor_id="system", expected_revision=None, method="tail", idempotency_key=None, provider="local"):
        # Unified memory backend (migration 025) adds lesson_generate and
        # session_process; session processing uses the local personal_memory
        # method. The older method restrictions keep their original scope.
        if kind not in ("memory_update","next_step","chat","lesson_generate","session_process") or method not in ("profile_only","tail","summary","structured","structured_v2","personal_graph","personal_memory") or provider not in ("local","external"):
            raise ValueError("invalid memory job options")
        if method == "structured_v2" and kind != "next_step":
            raise ValueError("structured_v2_is_teacher_next_step_only")
        if method == "personal_graph" and (kind != "next_step" or provider != "local"):
            raise ValueError("personal_graph_is_local_teacher_next_step_only")
        if method == "personal_memory" and (kind not in ("next_step", "session_process") or provider != "local"):
            raise ValueError("personal_memory_method_is_local_next_step_or_session_process_only")
        if kind in ("lesson_generate", "session_process"):
            required = {"topic_id","minutes","client_request_id"} if kind == "lesson_generate" else {"learning_session_id","client_request_id"}
            if not isinstance(payload,dict) or not required <= set(payload):
                raise ValueError(f"{kind} payload is missing required fields")
        # SP-1: the advisory lock must come before the learners row lock in
        # EVERY transaction that enqueues, otherwise this order inverts
        # against complete_learning_session (advisory first, learners UPDATE
        # later) and re-creates the F-031 deadlock on a different pair. The
        # xact-scoped lock is reentrant, so callers that already hold it are
        # unaffected.
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
        learner = self._memory_learner(cur,learner_id,expected_revision)
        self._recover_chat_messages(cur)
        if idempotency_key is not None:
            cur.execute("SELECT * FROM teachcopilot.memory_jobs WHERE learner_id=%s AND kind=%s AND idempotency_key=%s", (learner_id,kind,idempotency_key))
            existing = cur.fetchone()
            if existing is not None:
                if existing["topic_id"] != topic_id or existing["payload"] != (payload or {}) or existing["method"] != method or str(existing["actor_id"]) != str(actor_id):
                    raise MemoryIdempotencyConflict("job key was reused for different input")
                return _row(existing)
        cur.execute("DELETE FROM teachcopilot.generation_admission WHERE status='waiting' AND lease_until<clock_timestamp()")
        cur.execute("""SELECT (SELECT count(*) FROM teachcopilot.memory_jobs WHERE status='queued') +
            (SELECT count(*) FROM teachcopilot.generation_admission WHERE status='waiting') AS count""")
        if cur.fetchone()["count"] >= 40:
            raise MemoryQueueFullError("memory queue contains 40 waiting requests; retry later")
        cur.execute("""INSERT INTO teachcopilot.memory_jobs
            (learner_id,topic_id,actor_id,kind,method,provider,priority,base_revision,source_revision,idempotency_key,payload)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (learner_id,topic_id,str(actor_id),kind,method,provider,10 if kind in ("next_step","chat") else 0,learner["revision"],learner["memory_source_revision"],idempotency_key,Json(payload or {})))
        return _row(cur.fetchone())

    def enqueue_memory_job(self, learner_id, topic_id, kind, payload=None, actor_id="system", expected_revision=None, method="tail", idempotency_key=None, provider="local"):
        with self._connect() as conn, conn.cursor() as cur:
            return self._enqueue_memory_job(cur,learner_id,topic_id,kind,payload,actor_id,expected_revision,method,idempotency_key,provider)

    def get_memory_job(self, job_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM teachcopilot.memory_jobs WHERE id=%s", (job_id,))
            return _row(cur.fetchone())

    def get_learner_source_revision(self, learner_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT memory_source_revision FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
            learner = cur.fetchone()
            if learner is None:
                raise KeyError(learner_id)
            return learner["memory_source_revision"]

    def cancel_memory_job(self, job_id, error="cancelled before provider call"):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.memory_jobs SET status='failed',error=%s,completed_at=now()
                WHERE id=%s AND status='queued' RETURNING *""", (str(error),job_id))
            job = cur.fetchone()
            if job is None:
                cur.execute("SELECT * FROM teachcopilot.memory_jobs WHERE id=%s", (job_id,))
                job = cur.fetchone()
                if job is None:
                    raise KeyError(job_id)
                if job["status"] == "running":
                    raise ValueError("a running provider call cannot be cancelled as unstarted")
            return _row(job)

    @staticmethod
    def _recover_memory_jobs(cur):
        # A call that may have reached a provider is never automatically replayed.
        cur.execute("""UPDATE teachcopilot.memory_jobs SET status='unknown_outcome',error='worker lease expired after call started',completed_at=now(),lease_until=NULL
            WHERE status='running' AND lease_until<now() AND call_started_at IS NOT NULL""")
        unknown = cur.rowcount
        cur.execute("""UPDATE teachcopilot.memory_jobs SET status='queued',worker_id=NULL,lease_until=NULL,started_at=NULL
            WHERE status='running' AND lease_until<now() AND call_started_at IS NULL""")
        requeued = cur.rowcount
        MemoryStoreMixin._recover_chat_messages(cur)
        return {"unknown_outcome":unknown,"requeued":requeued}

    def recover_memory_jobs(self):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            return self._recover_memory_jobs(cur)

    def claim_memory_job(self, worker_id, lease_seconds=120):
        lease_seconds = max(1,min(int(lease_seconds),3600))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            self._recover_memory_jobs(cur)
            cur.execute("""SELECT j.* FROM teachcopilot.memory_jobs j WHERE j.status='queued' AND j.kind <> 'chat'
                AND NOT EXISTS (SELECT 1 FROM teachcopilot.memory_jobs busy WHERE busy.learner_id=j.learner_id AND busy.status='running')
                AND NOT EXISTS (SELECT 1 FROM teachcopilot.memory_jobs waiting
                    WHERE waiting.status='queued' AND waiting.provider=j.provider AND waiting.priority>j.priority
                    AND NOT EXISTS (SELECT 1 FROM teachcopilot.memory_jobs busy
                        WHERE busy.learner_id=waiting.learner_id AND busy.status='running'))
                AND (SELECT count(*) FROM teachcopilot.memory_jobs busy WHERE busy.provider=j.provider AND busy.status='running') < CASE WHEN j.provider='local' THEN 1 ELSE 4 END
                ORDER BY j.priority DESC,j.created_at,j.id FOR UPDATE OF j SKIP LOCKED LIMIT 1""")
            job = cur.fetchone()
            if job is None:
                return None
            cur.execute("""UPDATE teachcopilot.memory_jobs SET status='running',worker_id=%s,started_at=now(),lease_until=now()+(%s * interval '1 second')
                WHERE id=%s RETURNING *""", (str(worker_id),lease_seconds,job["id"]))
            return _row(cur.fetchone())

    def claim_interactive_chat_job(self, job_id, worker_id, lease_seconds=600):
        """Acquire a synchronous chat slot under the same durable queue limits."""
        lease_seconds = max(1,min(int(lease_seconds),3600))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
            self._recover_memory_jobs(cur)
            cur.execute("""SELECT j.* FROM teachcopilot.memory_jobs j WHERE j.id=%s AND j.status='queued' AND j.kind='chat'
                AND NOT EXISTS (SELECT 1 FROM teachcopilot.memory_jobs busy WHERE busy.learner_id=j.learner_id AND busy.status='running')
                AND (SELECT count(*) FROM teachcopilot.memory_jobs busy WHERE busy.provider=j.provider AND busy.status='running') < CASE WHEN j.provider='local' THEN 1 ELSE 4 END
                FOR UPDATE OF j""", (job_id,))
            job = cur.fetchone()
            if job is None:
                return None
            cur.execute("""UPDATE teachcopilot.memory_jobs SET status='running',worker_id=%s,started_at=now(),lease_until=now()+(%s * interval '1 second')
                WHERE id=%s RETURNING *""", (str(worker_id),lease_seconds,job_id))
            return _row(cur.fetchone())

    def start_memory_job_call(self, job_id, worker_id):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.memory_jobs SET call_started_at=COALESCE(call_started_at,now())
                WHERE id=%s AND status='running' AND worker_id=%s AND lease_until>now() RETURNING *""", (job_id,str(worker_id)))
            job = cur.fetchone()
            if job is None:
                raise ValueError("worker no longer owns a live job lease")
            return _row(job)

    def heartbeat_memory_job(self, job_id, worker_id, lease_seconds=120):
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.memory_jobs SET lease_until=now()+(%s * interval '1 second')
                WHERE id=%s AND status='running' AND worker_id=%s AND lease_until>now() RETURNING id""", (max(1,min(int(lease_seconds),3600)),job_id,str(worker_id)))
            return cur.fetchone() is not None

    @staticmethod
    def _latest_session_end_context(cur, learner_id, topic_id=None):
        """Return the final assistant turn of the latest completed topic session.

        The preceding student source owns the context link. Looking at the two
        actual last sources avoids skipping over teacher notes, another topic,
        or a failed turn to find a more convenient historical instruction.
        """
        topic_filter = " AND topic_id=%s" if topic_id is not None else ""
        values = [learner_id] + ([topic_id] if topic_id is not None else [])
        cur.execute("""WITH latest AS (
                SELECT DISTINCT ON (topic_id) id,learner_id,topic_id
                FROM teachcopilot.learning_sessions
                WHERE learner_id=%s AND status='completed'""" + topic_filter + """
                ORDER BY topic_id,completed_at DESC,id DESC
            )
            SELECT preceding.id AS source_id,ending.id AS assistant_source_id,
                latest.id AS session_id,ending.content AS text
            FROM latest
            JOIN LATERAL (
                SELECT s.id,s.content,s.author_role,s.topic_id,s.source_order
                FROM teachcopilot.memory_sources s
                WHERE s.learning_session_id=latest.id AND s.learner_id=latest.learner_id
                ORDER BY s.source_order DESC LIMIT 1
            ) ending ON TRUE
            JOIN LATERAL (
                SELECT s.id,s.author_role,s.topic_id
                FROM teachcopilot.memory_sources s
                WHERE s.learning_session_id=latest.id AND s.learner_id=latest.learner_id
                    AND s.source_order<ending.source_order
                ORDER BY s.source_order DESC LIMIT 1
            ) preceding ON TRUE
            WHERE ending.author_role='assistant' AND preceding.author_role='student'
                AND ending.topic_id=latest.topic_id AND preceding.topic_id=latest.topic_id""", values)
        contexts = {}
        for raw in cur.fetchall():
            row = _row(raw)
            if len(row["text"]) > SESSION_END_CONTEXT_CHARS:
                contexts[row["source_id"]] = {"session_end_context_omitted": {
                    "reason":"exceeds_context_excerpt_limit", "limit_chars":SESSION_END_CONTEXT_CHARS}}
            else:
                contexts[row["source_id"]] = {"session_end_context": {
                    "assistant_source_id":row["assistant_source_id"], "session_id":row["session_id"],
                    "role":"assistant", "usage":"context_only", "text":row["text"],
                    "excerpt_mode":"whole", "truncated":False}}
        return contexts

    @staticmethod
    def _remove_orphaned_memory_corrections(sources):
        """Drop correction chains whose target was excluded from the source snapshot."""
        source_ids = {source["source_id"] for source in sources}
        while True:
            orphaned = {source["source_id"] for source in sources
                if source["source_id"] in source_ids and source.get("corrects_source_id") is not None
                and source["corrects_source_id"] not in source_ids}
            if not orphaned:
                break
            source_ids.difference_update(orphaned)
        return [source for source in sources if source["source_id"] in source_ids]

    @staticmethod
    def _current_personal_graph_sources(cur, learner_id):
        """Read only graph-eligible archived sources under the caller's transaction."""
        cur.execute("""SELECT s.id AS source_id,s.learning_session_id AS session_id,s.author_role,s.content AS text,
                s.observed_on,s.created_at,s.topic_id,s.approved,s.corrects_source_id,
                CASE WHEN preceding.author_role='assistant' AND preceding.topic_id=s.topic_id AND s.topic_id=ls.topic_id
                    THEN right(preceding.content,%s) END AS preceding_assistant_text,
                CASE WHEN preceding.author_role='assistant' AND preceding.topic_id=s.topic_id AND s.topic_id=ls.topic_id
                    THEN length(preceding.content)>%s ELSE FALSE END AS preceding_assistant_truncated
            FROM teachcopilot.memory_sources s
            LEFT JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
            LEFT JOIN LATERAL (
                SELECT previous.author_role,previous.content,previous.topic_id
                FROM teachcopilot.memory_sources previous
                WHERE s.author_role='student' AND previous.learner_id=s.learner_id
                    AND previous.learning_session_id=s.learning_session_id AND previous.source_order<s.source_order
                ORDER BY previous.source_order DESC LIMIT 1
            ) preceding ON TRUE
            WHERE s.learner_id=%s AND s.author_role <> 'assistant'
                AND ((s.author_role='teacher' AND s.approved AND (s.learning_session_id IS NULL OR ls.status='completed'))
                     OR (s.author_role='student' AND ls.status='completed'))
                AND NOT EXISTS (SELECT 1 FROM teachcopilot.learning_messages m
                    WHERE m.source_id=s.id AND m.learner_id=s.learner_id AND m.status <> 'completed')
            ORDER BY s.source_order""", (DIALOGUE_CONTEXT_CHARS,DIALOGUE_CONTEXT_CHARS,learner_id))
        candidate_sources = [_row(row) for row in cur.fetchall()]
        sources = MemoryStoreMixin._remove_orphaned_memory_corrections(candidate_sources)
        visible_source_ids = {source["source_id"] for source in sources}
        hidden_correction_count = sum(1 for source in candidate_sources
            if source.get("corrects_source_id") is not None
            and source["source_id"] not in visible_source_ids)
        ending_contexts = MemoryStoreMixin._latest_session_end_context(cur,learner_id,None)
        for source in sources:
            preceding_text = source.pop("preceding_assistant_text")
            truncated = source.pop("preceding_assistant_truncated")
            if preceding_text:
                source["dialogue_context"] = {"role":"assistant", "usage":"context_only", "text":preceding_text,
                    "excerpt_mode":"tail", "truncated":bool(truncated)}
            source.update(ending_contexts.get(source["source_id"],{}))
        return sources, hidden_correction_count

    @staticmethod
    def _current_personal_graph_snapshot(cur, learner_id):
        """Read the current eligible archive under the caller's transaction."""
        from .personal_graph_contract import personal_graph_snapshot_fingerprint
        sources, _ = MemoryStoreMixin._current_personal_graph_sources(cur,learner_id)
        source_ids = [source["source_id"] for source in sources]
        return source_ids, personal_graph_snapshot_fingerprint(sources)

    @staticmethod
    def _assert_memory_references(cur, learner_id, value, *, topic_id=None, all_topics=False):
        """Reject cross-learner source links in persisted context and memory."""
        ids = set()
        def visit(node, path=()):
            if isinstance(node,dict):
                for key,item in node.items():
                    if key == "assistant_source_id" and not (
                            len(path) == 3 and path[:2] == ("context","session_end_context")
                            and isinstance(path[2],int)):
                        raise ValueError("assistant context reference is only allowed in session_end_context")
                    if key == "corrects_source_id" and item is not None and (not isinstance(item,str) or not item):
                        raise ValueError("memory correction reference must be a nonempty source identifier")
                    if key in ("source_id","evidence_source_id","turn_id","source_turn_id","corrects_source_id") and isinstance(item,str):
                        ids.add(item)
                    elif key in ("source_ids","evidence_ids","evidence_source_ids","evidence_turn_ids") and isinstance(item,list):
                        ids.update(str(x) for x in item)
                    else:
                        visit(item,(*path,key))
            elif isinstance(node,list):
                for index,item in enumerate(node):
                    visit(item,(*path,index))
        visit(value)
        if ids:
            cur.execute("SELECT id FROM teachcopilot.memory_sources WHERE learner_id=%s AND id=ANY(%s) AND author_role <> 'assistant'", (learner_id,list(ids)))
            found = {row["id"] for row in cur.fetchall()}
            if found != ids:
                raise ValueError("memory contains unavailable or assistant evidence sources")
        context = value.get("context") if isinstance(value,dict) else None
        if isinstance(context,dict) and "session_end_context" in context:
            endings = context["session_end_context"]
            if not isinstance(endings,list):
                raise ValueError("session_end_context must be a list")
            required = {"source_id","assistant_source_id","session_id","role","usage","text","excerpt_mode","truncated"}
            anchors, assistants = set(), set()
            expected = MemoryStoreMixin._latest_session_end_context(
                cur, learner_id, None if all_topics else topic_id) if endings else {}
            for item in endings:
                if (not isinstance(item,dict) or set(item) != required
                        or any(not isinstance(item.get(key),str) or not item[key]
                               for key in ("source_id","assistant_source_id","session_id","text"))
                        or item["role"] != "assistant" or item["usage"] != "context_only"
                        or item["excerpt_mode"] != "whole" or item["truncated"] is not False
                        or len(item["text"]) > SESSION_END_CONTEXT_CHARS
                        or item["source_id"] in anchors or item["assistant_source_id"] in assistants):
                    raise ValueError("invalid session_end_context metadata")
                actual = expected.get(item["source_id"],{}).get("session_end_context")
                if actual != {key:val for key,val in item.items() if key != "source_id"}:
                    raise ValueError("session_end_context must match the final assistant of the latest completed session")
                anchors.add(item["source_id"])
                assistants.add(item["assistant_source_id"])

    def _save_memory_version(self, cur, learner_id, topic_id, state, *, base_revision, source_revision=None, approved=False, context=None, job_id=None):
        if isinstance(context,dict) and "snapshot_review" in context:
            raise ValueError("snapshot review metadata is reserved for teacher confirmation")
        learner = self._memory_learner(cur,learner_id,base_revision)
        if source_revision is not None and learner["memory_source_revision"] != source_revision:
            _revision_error("learning sources changed while memory was generated")
        if isinstance(state,dict) and state.get("method") == "personal_memory":
            if approved:
                raise ValueError("personal_memory_requires_explicit_approval")
            sources, _ = self._current_personal_graph_sources(cur,learner_id)
            from .personal_graph_contract import personal_graph_snapshot_fingerprint
            problem = _personal_memory_validation_problem(state,context or {},learner_id,topic_id,
                learner["memory_source_revision"],sources,personal_graph_snapshot_fingerprint(sources))
            if problem:
                if problem[0] == "stale":
                    _revision_error(problem[1])
                raise ValueError(problem[1])
        self._assert_memory_references(cur,learner_id,{"state":state,"context":context or {}},topic_id=topic_id,
                                       all_topics=isinstance(state,dict) and state.get("method") in ARCHIVE_MEMORY_METHODS)
        cur.execute("""INSERT INTO teachcopilot.memory_versions(learner_id,topic_id,base_revision,source_revision,chat_source_revision,state,context,approved,job_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (learner_id,topic_id,base_revision,learner["memory_source_revision"],learner["memory_chat_source_revision"],Json(state),Json(context or {}),approved,job_id))
        return _row(cur.fetchone())

    def save_memory_version(self, learner_id, topic_id, state, *, base_revision, approved=False, context=None, job_id=None, source_revision=None):
        with self._connect() as conn, conn.cursor() as cur:
            return self._save_memory_version(cur,learner_id,topic_id,state,base_revision=base_revision,source_revision=source_revision,approved=approved,context=context,job_id=job_id)

    def _save_next_step(self, cur, learner_id, topic_id, result, *, base_revision, source_revision=None, context=None, memory_version_id=None, job_id=None, method="tail"):
        learner = self._memory_learner(cur,learner_id,base_revision)
        if source_revision is not None and learner["memory_source_revision"] != source_revision:
            _revision_error("learning sources changed while next step was generated")
        self._assert_memory_references(cur,learner_id,{"result":result,"context":context or {}},topic_id=topic_id,
                                       all_topics=method in ARCHIVE_MEMORY_METHODS)
        cur.execute("""INSERT INTO teachcopilot.next_steps(learner_id,topic_id,base_revision,source_revision,result,context,memory_version_id,job_id,method)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (learner_id,topic_id,base_revision,learner["memory_source_revision"],Json(result),Json(context or {}),memory_version_id,job_id,method))
        return _row(cur.fetchone())

    def save_next_step(self, learner_id, topic_id, result, *, base_revision, context=None, memory_version_id=None, job_id=None, method="tail", source_revision=None):
        with self._connect() as conn, conn.cursor() as cur:
            return self._save_next_step(cur,learner_id,topic_id,result,base_revision=base_revision,source_revision=source_revision,context=context,memory_version_id=memory_version_id,job_id=job_id,method=method)

    def get_next_step(self, next_step_id, learner_id=None):
        with self._connect() as conn, conn.cursor() as cur:
            query = "SELECT * FROM teachcopilot.next_steps WHERE id=%s"
            values = [next_step_id]
            if learner_id is not None:
                query += " AND learner_id=%s"
                values.append(learner_id)
            cur.execute(query,values)
            return _row(cur.fetchone())

    def finish_memory_job(self, job_id, *, result=None, error=None, unknown_outcome=False, worker_id=None, memory_state=None, next_step=None, context=None, approved=False):
        with self._connect() as conn, conn.cursor() as cur:
            # Enqueue and proposal confirmation already hold learner locks
            # before touching jobs. Never hold a job row while waiting for the
            # learner: queue recovery takes the queue lock and then job rows.
            cur.execute("SELECT learner_id FROM teachcopilot.memory_jobs WHERE id=%s", (job_id,))
            initial = cur.fetchone()
            if initial is None:
                raise KeyError(job_id)
            learner = self._memory_learner(cur,initial["learner_id"])
            cur.execute("SELECT *,lease_until>now() AS live_lease FROM teachcopilot.memory_jobs WHERE id=%s FOR UPDATE", (job_id,))
            job = cur.fetchone()
            if job is None:
                raise KeyError(job_id)
            if job["status"] in ("completed","failed","unknown_outcome"):
                return _row(job)
            if job["status"] != "running" or (worker_id is not None and job["worker_id"] != str(worker_id)):
                raise ValueError("worker does not own this running job")
            if not job["live_lease"]:
                unknown_outcome = job["call_started_at"] is not None
                error = "worker_lease_expired_after_call" if unknown_outcome else "worker_lease_expired_before_call"
            stale = learner["revision"] != job["base_revision"] or learner["memory_source_revision"] != job["source_revision"]
            if stale and error is None:
                error = "stale_revision: learner card or learning sources changed during generation"
            status = "unknown_outcome" if unknown_outcome else "failed" if error else "completed"
            output = dict(result or {})
            if status == "completed":
                version = None
                if memory_state is not None:
                    version = self._save_memory_version(cur,job["learner_id"],job["topic_id"],memory_state,base_revision=job["base_revision"],source_revision=job["source_revision"],approved=approved,context=context,job_id=job_id)
                    output["memory_version_id"] = version["id"]
                if next_step is not None:
                    step = self._save_next_step(cur,job["learner_id"],job["topic_id"],next_step,base_revision=job["base_revision"],source_revision=job["source_revision"],context=context,memory_version_id=version["id"] if version else (context or {}).get("memory_version_id"),job_id=job_id,method=job["method"])
                    output.update(next_step)
                    output["next_step_id"] = step["id"]
            cur.execute("""UPDATE teachcopilot.memory_jobs SET status=%s,result=%s,error=%s,completed_at=now(),lease_until=NULL
                WHERE id=%s RETURNING *""", (status,Json(output),str(error) if error else None,job_id))
            return _row(cur.fetchone())

    def get_learner_memory(self, learner_id, topic_id=None, *, include_personal_graph_snapshot=False):
        from .personal_graph_contract import PERSONAL_GRAPH_CONTEXT_PROTOCOL
        with self._connect() as conn, conn.cursor() as cur:
            learner = self._memory_learner(cur,learner_id,lock="SHARE")
            topic_filter = " AND s.topic_id=%s" if topic_id is not None else ""
            values = [learner_id] + ([topic_id] if topic_id is not None else [])
            cur.execute("""SELECT s.id AS source_id,s.learning_session_id AS session_id,s.author_role,s.content AS text,
                s.content,s.observed_on,s.topic_id,s.approved,s.created_at,s.corrects_source_id,
                CASE WHEN preceding.author_role='assistant' AND preceding.topic_id=s.topic_id AND s.topic_id=ls.topic_id
                    THEN right(preceding.content,%s) END AS preceding_assistant_text,
                CASE WHEN preceding.author_role='assistant' AND preceding.topic_id=s.topic_id AND s.topic_id=ls.topic_id
                    THEN length(preceding.content)>%s ELSE FALSE END AS preceding_assistant_truncated
                FROM teachcopilot.memory_sources s
                LEFT JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
                LEFT JOIN LATERAL (
                    SELECT previous.author_role,previous.content,previous.topic_id
                    FROM teachcopilot.memory_sources previous
                    WHERE s.author_role='student' AND previous.learner_id=s.learner_id
                        AND previous.learning_session_id=s.learning_session_id AND previous.source_order<s.source_order
                    ORDER BY previous.source_order DESC LIMIT 1
                ) preceding ON TRUE
                WHERE s.learner_id=%s AND s.author_role <> 'assistant'
                AND ((s.author_role='teacher' AND s.approved AND (s.learning_session_id IS NULL OR ls.status='completed'))
                     OR (s.author_role='student' AND ls.status='completed'))""" + topic_filter + " ORDER BY s.source_order", [DIALOGUE_CONTEXT_CHARS,DIALOGUE_CONTEXT_CHARS,*values])
            sources = [_row(row) for row in cur.fetchall()]
            ending_contexts = self._latest_session_end_context(cur,learner_id,topic_id)
            for source in sources:
                preceding_text = source.pop("preceding_assistant_text")
                truncated = source.pop("preceding_assistant_truncated")
                if preceding_text:
                    # Context belongs to the following student's literal source.
                    # It is never an assistant evidence source or an achievement.
                    source["dialogue_context"] = {"role":"assistant", "usage":"context_only", "text":preceding_text, "excerpt_mode":"tail",
                                                  "truncated":bool(truncated)}
                source.update(ending_contexts.get(source["source_id"],{}))
            cur.execute("SELECT * FROM teachcopilot.memory_versions WHERE learner_id=%s" + (" AND topic_id=%s" if topic_id is not None else "") + " ORDER BY created_at DESC,id", values)
            versions = [_row(row) for row in cur.fetchall()]
            has_personal_memory = any(isinstance(version.get("state"),dict)
                and version["state"].get("method") == "personal_memory" for version in versions)
            graph_ending_contexts = ending_contexts
            if (include_personal_graph_snapshot or has_personal_memory) and topic_id is not None and any(
                    isinstance(version.get("state"),dict)
                    and version["state"].get("method") in ARCHIVE_MEMORY_METHODS for version in versions):
                graph_ending_contexts = self._latest_session_end_context(cur,learner_id,None)
            eligible_by_topic = {}
            for source in sources:
                eligible_by_topic.setdefault(source["topic_id"],set()).add(source["source_id"])
            all_graph_source_ids = None
            graph_sources = []
            hidden_graph_correction_count = 0
            current_graph_fingerprint = None
            # Archive methods use the same learner-wide eligible snapshot. Keep
            # it out of the response unless explicitly requested, even though
            # personal-memory freshness always needs to verify its fingerprint.
            if include_personal_graph_snapshot or has_personal_memory:
                graph_sources, hidden_graph_correction_count = self._current_personal_graph_sources(cur,learner_id)
                all_graph_source_ids = {source["source_id"] for source in graph_sources}
                from .personal_graph_contract import personal_graph_snapshot_fingerprint
                current_graph_fingerprint = personal_graph_snapshot_fingerprint(graph_sources)
            for version in versions:
                version["method"] = version["state"].get("method") if isinstance(version["state"],dict) else None
                source_ids = _approved_memory_source_ids(version)
                if source_ids is None:
                    sources_changed = version["source_revision"] != learner["memory_source_revision"]
                else:
                    # Active turns and another topic change the global revision
                    # for concurrency guards, but do not change this approved
                    # long-term memory's immutable eligible source snapshot.
                    if version.get("method") in ARCHIVE_MEMORY_METHODS:
                        eligible = (all_graph_source_ids if include_personal_graph_snapshot
                                    or version.get("method") == "personal_memory" else None)
                    else:
                        eligible = eligible_by_topic.get(version["topic_id"],set())
                    if eligible is None:
                        sources_changed = version["source_revision"] != learner["memory_source_revision"]
                    else:
                        sources_changed = source_ids != eligible
                version_context = version.get("context")
                invalid_context = not isinstance(version_context,dict)
                if invalid_context:
                    version_context = {}
                stored_endings = version_context.get("session_end_context",[])
                version_endings = graph_ending_contexts if version.get("method") in ARCHIVE_MEMORY_METHODS else ending_contexts
                endings_changed = (include_personal_graph_snapshot or version.get("method") != "personal_graph") and (
                    not isinstance(stored_endings,list) or any(
                        not isinstance(item,dict) or not isinstance(item.get("source_id"),str)
                        or version_endings.get(item["source_id"],{}).get("session_end_context")
                            != {key:val for key,val in item.items() if key != "source_id"}
                        for item in stored_endings))
                # A completed empty lesson can replace the closing context
                # without adding student evidence. Only an ending actually
                # stored in this version participates in this freshness check.
                protocol_changed = (version.get("method") == "personal_graph" and
                    version["state"].get("source_context_protocol") != PERSONAL_GRAPH_CONTEXT_PROTOCOL)
                source_access = version_context.get("source_access") or {}
                if not isinstance(source_access,dict):
                    source_access = {}
                archive_permission_changed = (include_personal_graph_snapshot and current_graph_fingerprint is not None
                    and source_access.get("mode") == "full_archive" and (
                    source_access.get("source_ids") != version["state"].get("source_ids")
                    or source_access.get("snapshot_fingerprint") != version["state"].get("archive_snapshot_fingerprint")
                    or current_graph_fingerprint != version["state"].get("archive_snapshot_fingerprint")))
                personal_memory_changed = False
                if version.get("method") == "personal_memory":
                    approved = bool(version.get("approved"))
                    # A live chat increments the global revision before its
                    # unfinished turns enter the canonical archive. Approved
                    # grants remain valid while their exact archive is unchanged;
                    # drafts retain the stricter concurrency revision guard.
                    expected_source_revision = (version["source_revision"] if approved
                                                else learner["memory_source_revision"])
                    personal_memory_changed = ((not approved and version["source_revision"] != expected_source_revision)
                        or invalid_context or _personal_memory_validation_problem(version["state"],version_context,
                            learner_id,version["topic_id"],expected_source_revision,graph_sources,
                            current_graph_fingerprint,approved=approved) is not None)
                version["stale"] = (version["base_revision"] != learner["revision"]
                                    or sources_changed or endings_changed or protocol_changed or archive_permission_changed
                                    or personal_memory_changed)
                chat_only = (not version.get("approved") and version["base_revision"] == learner["revision"]
                             and _child_chat_only_revision_change(learner,version))
                snapshot_intact = False
                if chat_only and version.get("method") == "tail":
                    native_ids = _approved_memory_source_ids({**version, "approved": True})
                    snapshot_intact = native_ids is not None and native_ids <= eligible_by_topic.get(version["topic_id"],set())
                elif chat_only and version.get("method") in ARCHIVE_MEMORY_METHODS:
                    state_ids = version["state"].get("source_ids")
                    snapshot_intact = (all_graph_source_ids is not None and isinstance(state_ids,list)
                        and all(isinstance(item,str) and item for item in state_ids)
                        and len(state_ids) == len(set(state_ids)) and set(state_ids) <= all_graph_source_ids
                        and _selected_snapshot_matches(version["state"],version_context,graph_sources))
                    if snapshot_intact and version.get("method") == "personal_memory":
                        snapshot_intact = _personal_memory_validation_problem(version["state"],version_context,
                            learner_id,version["topic_id"],version["source_revision"],graph_sources,
                            current_graph_fingerprint,allow_snapshot_superset=True) is None
                version["new_chat_since_snapshot"] = bool(chat_only)
                version["can_confirm_previous_snapshot"] = bool(chat_only and snapshot_intact
                    and not invalid_context and not protocol_changed and not archive_permission_changed)
            from .personal_graph_contract import personal_graph_snapshot_fingerprint
            result = {"learner_id":learner_id,"revision":learner["revision"],"source_revision":learner["memory_source_revision"],
                    "chat_source_revision":learner.get("memory_chat_source_revision"),
                    "source_snapshot_fingerprint":personal_graph_snapshot_fingerprint(sources),
                    "sources":sources,"versions":versions}
            if include_personal_graph_snapshot:
                result.update(personal_graph_sources=graph_sources,
                    personal_graph_snapshot_fingerprint=current_graph_fingerprint,
                    personal_graph_hidden_correction_count=hidden_graph_correction_count)
            return result

    def approve_memory_version(self, learner_id, version_id, actor_id, expected_revision, interest_candidate_ids=None,
                               full_archive_access=False, expected_archive_fingerprint=None,
                               acknowledge_new_messages=False, expected_source_revision=None):
        from .conversation_memory import apply_interest_choices, validate_interest_candidates
        from .personal_graph_contract import PERSONAL_GRAPH_CONTEXT_PROTOCOL
        selected_ids = [] if interest_candidate_ids is None else interest_candidate_ids
        if (not isinstance(selected_ids,list) or not all(isinstance(item,str) for item in selected_ids)
                or len(selected_ids) != len(set(selected_ids))):
            raise ValueError("interest candidate IDs must be a list of distinct strings")
        if not isinstance(full_archive_access,bool):
            raise ValueError("full archive access must be an explicit boolean")
        if not isinstance(acknowledge_new_messages,bool):
            raise ValueError("new-message acknowledgement must be an explicit boolean")
        if expected_source_revision is not None and (type(expected_source_revision) is not int or expected_source_revision < 0):
            raise ValueError("expected source revision must be a nonnegative integer")
        if expected_archive_fingerprint is not None and (not isinstance(expected_archive_fingerprint,str)
                or len(expected_archive_fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in expected_archive_fingerprint)):
            raise ValueError("invalid personal graph archive fingerprint")
        with self._connect() as conn, conn.cursor() as cur:
            learner = self._memory_learner(cur,learner_id)
            cur.execute("SELECT * FROM teachcopilot.memory_versions WHERE id=%s AND learner_id=%s FOR UPDATE", (version_id,learner_id))
            version = cur.fetchone()
            if version is None:
                raise KeyError(version_id)
            state = version.get("state") or {}
            state_method = state.get("method") if isinstance(state,dict) else None
            if (isinstance(state,dict) and state.get("method") == "personal_graph"
                    and state.get("source_context_protocol") != PERSONAL_GRAPH_CONTEXT_PROTOCOL):
                _revision_error("Версия персональной памяти устарела. Создайте и проверьте новую версию.")
            if state_method == "personal_memory":
                from .personal_memory_contract import PERSONAL_MEMORY_CONTEXT_PROTOCOL
                if state.get("source_context_protocol") != PERSONAL_MEMORY_CONTEXT_PROTOCOL:
                    _revision_error("Версия персональной памяти устарела. Создайте и проверьте новую версию.")
            if version.get("method") == "structured_v2":
                raise ValueError("structured_v2_versions_are_review_only")
            stored_context = version.get("context")
            if state_method == "personal_memory" and not isinstance(stored_context,dict):
                raise ValueError("personal memory context is invalid")
            context = dict(stored_context or {})
            if full_archive_access and state_method not in ARCHIVE_MEMORY_METHODS:
                raise ValueError("full archive access is only available for personal memory versions")
            if version["approved"]:
                reviewed_ids = (context.get("interest_review") or {}).get("candidate_ids") or []
                prior_access = context.get("source_access") or {}
                prior_full_archive = prior_access.get("mode") == "full_archive"
                if (set(reviewed_ids) != set(selected_ids) or prior_full_archive != full_archive_access
                        or (full_archive_access and expected_archive_fingerprint != prior_access.get("snapshot_fingerprint"))):
                    raise MemoryIdempotencyConflict("memory version was already approved with different selections")
                return _row(version)
            if learner["revision"] != expected_revision:
                _revision_error(f"learner revision is {learner['revision']}, expected {expected_revision}")
            if version["base_revision"] != learner["revision"]:
                _revision_error("memory version no longer matches the learner and sources")
            chat_override = version["source_revision"] != learner["memory_source_revision"]
            if chat_override:
                if (not acknowledge_new_messages or expected_source_revision != learner["memory_source_revision"]
                        or full_archive_access or not _child_chat_only_revision_change(learner,version)):
                    _revision_error("memory version no longer matches the learner and sources")
                if state_method == "tail":
                    native_ids = _approved_memory_source_ids({**version, "approved": True, "method": state_method})
                    if native_ids is None:
                        _revision_error("memory version has no verifiable source snapshot")
                    cur.execute("""SELECT s.id FROM teachcopilot.memory_sources s
                        LEFT JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
                        WHERE s.learner_id=%s AND s.topic_id=%s AND s.author_role <> 'assistant'
                        AND ((s.author_role='teacher' AND s.approved AND (s.learning_session_id IS NULL OR ls.status='completed'))
                             OR (s.author_role='student' AND ls.status='completed'))""", (learner_id,version["topic_id"]))
                    if not native_ids <= {str(row["id"]) for row in cur.fetchall()}:
                        _revision_error("reviewed memory sources changed")
                elif state_method in ARCHIVE_MEMORY_METHODS:
                    current_sources, _ = self._current_personal_graph_sources(cur,learner_id)
                    state_ids = state.get("source_ids")
                    if (not isinstance(state_ids,list) or any(not isinstance(item,str) or not item for item in state_ids)
                            or len(state_ids) != len(set(state_ids))
                            or not _selected_snapshot_matches(state,context,current_sources)):
                        _revision_error("reviewed archive snapshot changed")
                else:
                    _revision_error("memory version has no verifiable source snapshot")
                context["snapshot_review"] = {"new_chat_since_snapshot": True, "refresh_required": True,
                    "snapshot_source_revision": version["source_revision"],
                    "confirmed_source_revision": learner["memory_source_revision"]}
            revision = learner["revision"]
            if state_method == "personal_graph":
                evidence_ids = context.get("evidence_ids")
                state_ids = state.get("source_ids")
                if (not isinstance(evidence_ids,list) or not all(isinstance(item,str) and item for item in evidence_ids)
                        or len(evidence_ids) != len(set(evidence_ids))):
                    raise ValueError("personal graph context has invalid selected source IDs")
                if (not isinstance(state_ids,list) or not all(isinstance(item,str) and item for item in state_ids)
                        or len(state_ids) != len(set(state_ids)) or not set(evidence_ids) <= set(state_ids)):
                    raise ValueError("personal graph archive source snapshot is invalid")
                archive_fingerprint = state.get("archive_snapshot_fingerprint")
                if full_archive_access:
                    current_ids, current_fingerprint = self._current_personal_graph_snapshot(cur,learner_id)
                    if (not isinstance(archive_fingerprint,str) or len(archive_fingerprint) != 64
                            or expected_archive_fingerprint != archive_fingerprint
                            or current_fingerprint != archive_fingerprint
                            or set(current_ids) != set(state_ids)):
                        _revision_error("Архив изменился после просмотра. Создайте и проверьте новую версию памяти.")
                    access_ids = list(state_ids)
                    access_mode = "full_archive"
                else:
                    access_ids = list(evidence_ids)
                    access_mode = "selected_context"
                context["source_access"] = {"mode":access_mode,"source_ids":access_ids,
                    "snapshot_fingerprint":archive_fingerprint if full_archive_access else None,
                    "selected_by":str(actor_id)}
            elif state_method == "personal_memory":
                sources, _ = self._current_personal_graph_sources(cur,learner_id)
                from .personal_graph_contract import personal_graph_snapshot_fingerprint
                fingerprint = personal_graph_snapshot_fingerprint(sources)
                problem = _personal_memory_validation_problem(state,context,learner_id,version["topic_id"],
                    version["source_revision"] if chat_override else learner["memory_source_revision"],sources,fingerprint,
                    allow_snapshot_superset=chat_override)
                if problem:
                    if problem[0] == "stale":
                        _revision_error("Архив изменился после просмотра. Создайте и проверьте новую версию памяти.")
                    raise ValueError(problem[1])
                if full_archive_access and expected_archive_fingerprint != fingerprint:
                    _revision_error("Архив изменился после просмотра. Создайте и проверьте новую версию памяти.")
                context["source_access"] = {
                    "mode":"full_archive" if full_archive_access else "selected_context",
                    "source_ids":list(state["source_ids"] if full_archive_access else context["evidence_ids"]),
                    "snapshot_fingerprint":fingerprint if full_archive_access else None,
                    "selected_by":str(actor_id),
                }
            candidates = context.get("interest_candidates") or []
            if selected_ids:
                if not isinstance(candidates,list) or any(not isinstance(item,dict) for item in candidates):
                    raise ValueError("invalid stored interest candidates")
                candidate_ids = [item.get("candidate_id") for item in candidates]
                if (not all(isinstance(item,str) and item for item in candidate_ids)
                        or len(candidate_ids) != len(set(candidate_ids)) or not set(selected_ids) <= set(candidate_ids)):
                    raise ValueError("selected interest candidate is unavailable in this memory version")
                source_ids = [item.get("source_id") for item in candidates]
                cur.execute("""SELECT s.id AS source_id,s.author_role,s.content FROM teachcopilot.memory_sources s
                    JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id
                    WHERE s.learner_id=%s AND s.id=ANY(%s) AND ls.status='completed'
                    ORDER BY s.source_order""", (learner_id,source_ids))
                sources = [_row(row) for row in cur.fetchall()]
                validated = validate_interest_candidates(json.dumps({"candidates":[
                    {key:value for key,value in candidate.items() if key != "candidate_id"}
                    for candidate in candidates]},ensure_ascii=False),sources)
                if {item["candidate_id"] for item in validated} != set(candidate_ids):
                    raise ValueError("stored interest candidate identifier does not match its source")
                chosen = [item for item in validated if item["candidate_id"] in selected_ids]
                cur.execute("SELECT card FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
                card = apply_interest_choices(cur.fetchone()["card"],chosen)
                if len((card.get("rag_context") or {}).get("avoid") or []) > 3:
                    raise ValueError("В карточке уже три ограничения. Сначала пересмотрите нежелательные сюжеты.")
                saved = self._save_card(cur,card,str(actor_id),expected_revision)
                revision = saved["revision"]
                selected_ids = [item["candidate_id"] for item in chosen]
            context["interest_review"] = {"candidate_ids":selected_ids,"actor_id":str(actor_id),
                                            "card_revision":revision}
            cur.execute("""UPDATE teachcopilot.memory_versions SET approved=TRUE,approved_by=COALESCE(approved_by,%s),approved_at=COALESCE(approved_at,now()),
                context=%s,base_revision=%s WHERE id=%s RETURNING *""", (str(actor_id),Json(context),revision,version_id))
            return _row(cur.fetchone())

    def _confirm_memory_proposal(self, cur, metadata, actor_id, new_revisions):
        """Called from Store.confirm_proposal inside its existing transaction."""
        metadata = metadata or {}
        if isinstance(new_revisions,list):
            new_revisions = {card["learner_id"]:card["revision"] for card in new_revisions}
        sources = metadata.get("memory_sources") or {}
        if not isinstance(sources,dict) or set(sources) - set(new_revisions):
            raise ValueError("proposal memory must refer only to learners in this proposal")
        jobs = []
        for learner_id, value in sources.items():
            entries = value if isinstance(value,list) else [value]
            for index,entry in enumerate(entries):
                if not isinstance(entry,dict) or not entry.get("topic_id") or not entry.get("text"):
                    raise ValueError("proposal memory source requires topic_id and text")
                digest = hashlib.sha256(json.dumps(entry,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                text = entry["text"]
                if metadata.get("group_note") or entry.get("group_note"):
                    text = "Групповая заметка преподавателя – авторство отдельных результатов не установлено.\n" + text
                source = self._memory_source(cur,learner_id,entry["topic_id"],actor_id,text,author_role="teacher",approved=True,observed_on=entry.get("observed_on"),source_key=f"proposal:{new_revisions[learner_id]}:{index}:{digest}",corrects_source_id=entry.get("corrects_source_id"))
                jobs.append(self._enqueue_memory_job(cur,learner_id,entry["topic_id"],"memory_update",{"source_id":source["id"]},actor_id,expected_revision=new_revisions[learner_id],method="tail",idempotency_key=f"source:{source['id']}"))
        return jobs

    def learner_known_identifiers(self, learner_id):
        """Collect local identity terms before constructing an external projection."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT learner_id,pseudonym,card FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
            learner = cur.fetchone()
            if learner is None:
                raise KeyError(learner_id)
            values = {str(learner["learner_id"]),str(learner["pseudonym"])}
            identity_keys = {"name","full_name","first_name","last_name","middle_name","surname","email","phone","address","birth_date","birthdate","date_of_birth","school","school_name","pseudonym","learner_id","account_id","student_id","telegram_id","telegram_username","username"}
            def collect(node):
                if isinstance(node,dict):
                    for key,value in node.items():
                        if key.lower() in identity_keys and isinstance(value,(str,int)) and not isinstance(value,bool):
                            values.add(str(value))
                        elif isinstance(value,(dict,list)):
                            collect(value)
                elif isinstance(node,list):
                    for item in node:
                        collect(item)
            collect(learner["card"])
            cur.execute("""SELECT a.id,a.email,a.display_name,a.payload FROM teachcopilot.accounts a
                JOIN teachcopilot.account_learners al ON al.account_id=a.id WHERE al.learner_id=%s""", (learner_id,))
            for account in cur.fetchall():
                values.update((str(account["id"]),str(account["email"]),str(account["display_name"])))
                collect(account["payload"])
            return sorted(value for value in values if value and value != "None")
