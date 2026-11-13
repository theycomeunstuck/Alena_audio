"""Versioned, source-backed extracts from the last completed learning sessions.

The caller must authorize the learner first. ``approved_source_ids=None`` is
reserved for the authorized teacher view. For chat the caller must pass the
exact source IDs already granted by a reviewed memory version and a topic ID;
the summary is a projection of those grants, never a grant of its own.

Every read checks the current source and correction snapshot. A correction
outside the allowed subset withholds its entire chain, including older claims.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date, datetime

from psycopg2.extras import Json


ALGORITHM = "extractive_v1"
SUMMARY_FORMAT_VERSION = 2
FACT_LIMIT = 3
EXCERPT_CHARS = 120

_INTEREST = re.compile(r"\b(?:люблю|нравится|увлекаюсь|увлекается|интересуюсь|интересуется|обожаю)\b", re.I)
_UNCERTAINTY = re.compile(r"\b(?:не знаю|не понимаю|не уверен\w*|сомневаюсь|хз|забыл\w*)\b", re.I)


def _hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _time(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value) if value is not None else None


def _excerpt(value):
    """Return an explicitly marked, bounded literal excerpt."""
    text = str(value).strip()
    if len(text) <= EXCERPT_CHARS:
        return text, False
    return text[:EXCERPT_CHARS].rstrip() + "…", True


def _source_kind(source):
    if source.get("corrects_source_id"):
        return "correction"
    text = source["content"]
    if _UNCERTAINTY.search(text):
        return "uncertainty"
    if _INTEREST.search(text):
        return "interest"
    return "observation"


def _canonical_source(source):
    return {
        "source_id": str(source["source_id"]),
        "session_id": str(source["source_session_id"]) if source.get("source_session_id") else None,
        "order": source["source_order"],
        "topic_id": source["topic_id"],
        "role": source["author_role"],
        "text": source["content"],
        "approved": bool(source["approved"]),
        "corrects_source_id": str(source["corrects_source_id"]) if source.get("corrects_source_id") else None,
        "observed_on": _time(source.get("observed_on")),
        "created_at": _time(source.get("created_at")),
    }


def _session_payload(session, sources, ending, allowed_ids):
    """Build one bounded extract; return None when a restricted view has no grant.

    Each source row belongs to the same rooted correction tree. A component is
    visible only if every member is granted. The original and intermediate
    versions remain identified in ``correction_chain`` as superseded, but their
    withdrawn text is never presented as a current observation.
    """
    ordered = sorted((_canonical_source(item) for item in sources), key=lambda item: (item["order"], item["source_id"]))
    by_id = {item["source_id"]: item for item in ordered}
    children = defaultdict(list)
    for item in ordered:
        target = item["corrects_source_id"]
        if target in by_id:
            children[target].append(item["source_id"])

    visible, visible_ids = [], set()
    for root in (item for item in ordered if not item["corrects_source_id"]):
        stack, component = [root["source_id"]], []
        while stack:
            source_id = stack.pop()
            component.append(source_id)
            stack.extend(children[source_id])
        if allowed_ids is not None and not set(component).issubset(allowed_ids):
            continue
        visible_ids.update(component)
        parallel_corrections = any(len(children[source_id]) > 1 for source_id in component)
        for source_id in component:
            if children[source_id]:
                continue
            leaf = by_id[source_id]
            chain, current = [], leaf
            while current:
                chain.append({"source_id": current["source_id"], "role": current["role"],
                              "status": "current" if current["source_id"] == source_id else "superseded"})
                current = by_id.get(current["corrects_source_id"])
            chain.reverse()
            visible.append((leaf, chain, parallel_corrections))

    if allowed_ids is not None and not visible:
        return None

    priority = {"correction": 3, "uncertainty": 2, "interest": 1, "observation": 0}
    ranked = sorted(visible, key=lambda pair: (-priority[_source_kind({
        "corrects_source_id": pair[0]["corrects_source_id"], "content": pair[0]["text"]})],
        -pair[0]["order"], pair[0]["source_id"]))
    selected = sorted(ranked[:FACT_LIMIT], key=lambda pair: (pair[0]["order"], pair[0]["source_id"]))
    facts = []
    for source, chain, parallel_corrections in selected:
        excerpt, truncated = _excerpt(source["text"])
        kind = _source_kind({"corrects_source_id": source["corrects_source_id"], "content": source["text"]})
        kind_origin = ("lexical_heuristic" if kind in {"interest", "uncertainty"} else
                       "source_link" if kind == "correction" else "default")
        facts.append({
            "source_id": source["source_id"], "session_id": str(session["learning_session_id"]),
            "source_session_id": source["session_id"], "role": source["role"],
            "kind": kind, "kind_origin": kind_origin,
            "text": excerpt, "truncated": truncated, "observed_on": source["observed_on"],
            "source_created_at": source["created_at"], "corrects_source_id": source["corrects_source_id"],
            "correction_chain": chain if len(chain) > 1 else [], "parallel_corrections": parallel_corrections,
        })

    ending_payload = None
    if ending is not None and allowed_ids is None:
        text, truncated = _excerpt(ending["content"])
        ending_payload = {"source_id": str(ending["source_id"]),
                          "session_id": str(session["learning_session_id"]),
                          "role": "assistant", "usage": "context_only", "text": text,
                          "truncated": truncated}

    canonical_ending = ({"source_id": str(ending["source_id"]), "content": ending["content"]}
                        if ending is not None and allowed_ids is None else None)
    fingerprint_sources = ordered if allowed_ids is None else [item for item in ordered if item["source_id"] in visible_ids]
    fingerprint = _hash({"algorithm": ALGORITHM, "summary_format_version": SUMMARY_FORMAT_VERSION,
                         "fact_limit": FACT_LIMIT, "excerpt_chars": EXCERPT_CHARS,
                         "session": {"id": str(session["learning_session_id"]), "topic_id": session["topic_id"],
                                     "completed_at": _time(session["completed_at"])},
                         "sources": fingerprint_sources, "ending": canonical_ending,
                         "allowed_ids": sorted(allowed_ids) if allowed_ids is not None else None})
    return {
        "summary_format_version": SUMMARY_FORMAT_VERSION,
        "learning_session_id": str(session["learning_session_id"]), "topic_id": session["topic_id"],
        "completed_at": _time(session["completed_at"]), "facts": facts, "ending": ending_payload,
        "selected_source_count": len(facts), "omitted_source_count": len(visible) - len(facts),
        "truncated_source_count": sum(item["truncated"] for item in facts),
        "truncated_context_count": int(bool(ending_payload and ending_payload["truncated"])),
        "input_fingerprint": fingerprint, "fingerprint": fingerprint,
    }


def _session_text(session):
    lines = [f"Занятие завершено {session['completed_at']}; тема: {session['topic_id']}."]
    roles = {"student": "Реплика ученика", "teacher": "Заметка педагога"}
    kinds = {"interest": "автоматическая отметка: упоминание интереса",
             "uncertainty": "автоматическая отметка: упоминание неуверенности",
             "correction": "исправление"}
    for fact in session["facts"]:
        kind = f" ({kinds[fact['kind']]})" if fact["kind"] in kinds else ""
        lines.append(f"{roles[fact['role']]}{kind}: «{fact['text']}» [источник: {fact['source_id']}].")
    if not session["facts"]:
        lines.append("Отобранных наблюдений нет.")
    if any(fact["parallel_corrections"] for fact in session["facts"]):
        lines.append("Есть параллельные исправления источника; требуется проверка.")
    if session["ending"]:
        ending = session["ending"]
        lines.append(f"Последний сохранённый ответ помощника, только контекст: «{ending['text']}» "
                     f"[источник: {ending['source_id']}].")
    return "\n".join(lines)


def _save_version(cur, learner_id, kind, session_id, scope, fingerprint, payload):
    args = (learner_id, kind, session_id, scope)
    cur.execute("""SELECT id,input_fingerprint,payload,supersedes_version_id
        FROM teachcopilot.session_summary_versions
        WHERE learner_id=%s AND kind=%s AND learning_session_id IS NOT DISTINCT FROM %s::uuid
          AND scope_fingerprint=%s ORDER BY created_at DESC,id DESC LIMIT 1""", args)
    previous = cur.fetchone()
    if previous and previous["input_fingerprint"] == fingerprint:
        return previous
    cur.execute("""INSERT INTO teachcopilot.session_summary_versions
        (learner_id,kind,learning_session_id,scope_fingerprint,input_fingerprint,algorithm,payload,supersedes_version_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
        RETURNING id,input_fingerprint,payload,supersedes_version_id""",
        (learner_id, kind, session_id, scope, fingerprint, ALGORITHM, Json(payload),
         previous["id"] if previous else None))
    inserted = cur.fetchone()
    if inserted is not None:
        return inserted
    # Concurrent readers can compute the same immutable version.
    cur.execute("""SELECT id,input_fingerprint,payload,supersedes_version_id
        FROM teachcopilot.session_summary_versions
        WHERE learner_id=%s AND kind=%s AND learning_session_id IS NOT DISTINCT FROM %s::uuid
          AND scope_fingerprint=%s AND input_fingerprint=%s""", (*args, fingerprint))
    return cur.fetchone()


def _stored_payload(row):
    value = dict(row["payload"])
    value.update(version_id=str(row["id"]), stale=False,
                 supersedes_version_id=str(row["supersedes_version_id"]) if row["supersedes_version_id"] else None)
    return value


class _SnapshotChanged(Exception):
    pass


def get_or_build_summaries(store, learner_id, *, limit=1, actor_id=None, approved_source_ids=None, topic_id=None):
    """Read or persist the current summary for one or seven completed sessions.

    ``approved_source_ids`` must come from the server's already reviewed source
    grant, never from a request body. An empty grant returns no chat summaries.
    Teacher access and any actor restrictions are enforced by the caller.
    """
    if approved_source_ids is not None and not isinstance(approved_source_ids, (str, bytes)):
        try:
            approved_source_ids = tuple(approved_source_ids)
        except TypeError as exc:
            raise ValueError("approved_source_ids must be a collection of source IDs") from exc
    for _ in range(3):
        try:
            return _get_or_build_summaries_once(store, learner_id, limit=limit, actor_id=actor_id,
                                                approved_source_ids=approved_source_ids, topic_id=topic_id)
        except _SnapshotChanged:
            continue
    raise RuntimeError("session_summary_snapshot_changed; retry the request")


def _get_or_build_summaries_once(store, learner_id, *, limit, actor_id, approved_source_ids, topic_id):
    if type(limit) is not int or limit not in (1, 7):
        raise ValueError("summary limit must be one or seven completed sessions")
    if not isinstance(learner_id, str) or not learner_id:
        raise ValueError("learner_id is required")
    if topic_id is not None and (not isinstance(topic_id, str) or not topic_id):
        raise ValueError("topic_id must be a nonempty string")
    if actor_id is not None and not str(actor_id):
        raise ValueError("actor_id must be nonempty")
    allowed = None
    if approved_source_ids is not None:
        if isinstance(approved_source_ids, (str, bytes)):
            raise ValueError("approved_source_ids must be a collection of source IDs")
        try:
            values = list(approved_source_ids)
        except TypeError as exc:
            raise ValueError("approved_source_ids must be a collection of source IDs") from exc
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("approved_source_ids must contain nonempty source IDs")
        if not topic_id:
            raise ValueError("restricted summaries require an exact topic_id")
        allowed = set(values)

    scope = _hash({"mode": "teacher" if allowed is None else "approved_subset",
                   "actor_id": str(actor_id) if actor_id is not None else None,
                   "topic_id": topic_id, "source_ids": sorted(allowed) if allowed is not None else None})
    with store._connect() as conn, conn.cursor() as cur:
        # Optimistically validate the snapshot instead of holding the learner
        # lock while inserting a session FK. Completion takes the session lock
        # first, so taking those two locks in reverse order could deadlock.
        cur.execute("SELECT memory_source_revision FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
        learner = cur.fetchone()
        if learner is None:
            raise KeyError(learner_id)
        revision = learner["memory_source_revision"]
        where, params = ["learner_id=%s", "status='completed'"], [learner_id]
        if actor_id is not None:
            where.append("actor_id=%s")
            params.append(str(actor_id))
        if topic_id is not None:
            where.append("topic_id=%s")
            params.append(topic_id)
        params.append(limit)
        cur.execute("""SELECT id AS learning_session_id,topic_id,COALESCE(completed_at,created_at) AS completed_at
            FROM teachcopilot.learning_sessions WHERE """ + " AND ".join(where) + """
            ORDER BY COALESCE(completed_at,created_at) DESC,id DESC LIMIT %s""", params)
        sessions = [dict(row) for row in cur.fetchall()]
        ids = [str(row["learning_session_id"]) for row in sessions]
        sources, endings = defaultdict(list), {}
        if ids:
            # Corrections usually have no session ID; recursively attach them
            # only to the original session, retaining their actual source date.
            cur.execute("""WITH RECURSIVE relevant AS (
                SELECT s.id AS source_id,s.learner_id,s.learning_session_id AS source_session_id,
                    s.learning_session_id AS summary_session_id,s.source_order,s.topic_id,s.author_role,
                    s.content,s.approved,s.corrects_source_id,s.observed_on,s.created_at
                FROM teachcopilot.memory_sources s
                JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id AND ls.learner_id=s.learner_id
                WHERE s.learner_id=%s AND s.learning_session_id=ANY(%s::uuid[])
                    AND ls.status='completed' AND s.topic_id=ls.topic_id AND s.corrects_source_id IS NULL
                    AND (s.author_role='student' OR (s.author_role='teacher' AND s.approved))
                    AND NOT EXISTS (SELECT 1 FROM teachcopilot.learning_messages m
                        WHERE m.source_id=s.id AND m.learner_id=s.learner_id AND m.status<>'completed')
                UNION ALL
                SELECT c.id,c.learner_id,c.learning_session_id,r.summary_session_id,c.source_order,c.topic_id,
                    c.author_role,c.content,c.approved,c.corrects_source_id,c.observed_on,c.created_at
                FROM relevant r JOIN teachcopilot.memory_sources c
                    ON c.corrects_source_id=r.source_id AND c.learner_id=r.learner_id AND c.topic_id=r.topic_id
                LEFT JOIN teachcopilot.learning_sessions ls ON ls.id=c.learning_session_id AND ls.learner_id=c.learner_id
                WHERE c.author_role='teacher' AND c.approved
                    AND (c.learning_session_id IS NULL OR ls.status='completed')
                    AND NOT EXISTS (SELECT 1 FROM teachcopilot.learning_messages m
                        WHERE m.source_id=c.id AND m.learner_id=c.learner_id AND m.status<>'completed')
            ) SELECT * FROM relevant ORDER BY summary_session_id,source_order,source_id""", (learner_id, ids))
            for row in cur.fetchall():
                sources[str(row["summary_session_id"])].append(dict(row))
            if allowed is None:
                cur.execute("""SELECT DISTINCT ON (m.learning_session_id)
                    m.learning_session_id,s.id AS source_id,s.content
                    FROM teachcopilot.learning_messages m
                    JOIN teachcopilot.learning_sessions ls ON ls.id=m.learning_session_id AND ls.learner_id=m.learner_id
                    JOIN teachcopilot.memory_sources s ON s.id=m.assistant_source_id AND s.learner_id=m.learner_id
                        AND s.learning_session_id=m.learning_session_id AND s.topic_id=ls.topic_id AND s.author_role='assistant'
                    WHERE m.learner_id=%s AND m.learning_session_id=ANY(%s::uuid[]) AND m.status='completed'
                    ORDER BY m.learning_session_id,s.source_order DESC,s.id DESC""", (learner_id, ids))
                endings = {str(row["learning_session_id"]): dict(row) for row in cur.fetchall()}

        cur.execute("SELECT memory_source_revision FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
        learner = cur.fetchone()
        if learner is None:
            raise KeyError(learner_id)
        if learner["memory_source_revision"] != revision:
            raise _SnapshotChanged()

        saved = []
        for session in sessions:
            session_id = str(session["learning_session_id"])
            payload = _session_payload(session, sources[session_id], endings.get(session_id), allowed)
            if payload is None:
                continue
            payload["text"] = _session_text(payload)
            row = _save_version(cur, learner_id, "session", session_id, scope, payload["input_fingerprint"], payload)
            saved.append(_stored_payload(row))

        result = {"algorithm": ALGORITHM, "summary_format_version": SUMMARY_FORMAT_VERSION,
                  "available": bool(saved), "limit": limit, "count": len(saved),
                  "sessions": saved, "text": "\n\n".join(item["text"] for item in saved),
                  "date_span": {"start": saved[-1]["completed_at"], "end": saved[0]["completed_at"]} if saved else None,
                  "selected_source_count": sum(item["selected_source_count"] for item in saved),
                  "omitted_source_count": sum(item["omitted_source_count"] for item in saved),
                  "truncated_source_count": sum(item["truncated_source_count"] for item in saved),
                  "truncated_context_count": sum(item["truncated_context_count"] for item in saved),
                  "selection_limit_per_session": FACT_LIMIT, "excerpt_limit_chars": EXCERPT_CHARS,
                  "selection_scope": "current_visible_sources",
                  "version_id": None, "input_fingerprint": None, "fingerprint": None,
                  "stale": False, "supersedes_version_id": None}
        if saved and limit == 1:
            result.update(version_id=saved[0]["version_id"], input_fingerprint=saved[0]["input_fingerprint"],
                          fingerprint=saved[0]["fingerprint"], supersedes_version_id=saved[0]["supersedes_version_id"])
        elif saved:
            fingerprint = _hash({"algorithm": ALGORITHM, "scope": scope,
                                 "sessions": [{"id": item["learning_session_id"], "version_id": item["version_id"],
                                               "input_fingerprint": item["input_fingerprint"]} for item in saved]})
            result["input_fingerprint"] = fingerprint
            result["fingerprint"] = fingerprint
            row = _save_version(cur, learner_id, "rolling7", None, scope, fingerprint, result)
            result = _stored_payload(row)
        cur.execute("SELECT memory_source_revision FROM teachcopilot.learners WHERE learner_id=%s", (learner_id,))
        learner = cur.fetchone()
        if learner is None:
            raise KeyError(learner_id)
        if learner["memory_source_revision"] != revision:
            raise _SnapshotChanged()
        return result
