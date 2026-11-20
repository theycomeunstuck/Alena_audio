"""Load a frozen synthetic history into a disposable Store (client A).

Chat message text and provenance are immutable once written (the storage
layer's own trigger), so the loader inserts the synthetic sources directly
with their explicit historical instants, mirroring the real persistence
invariants (learner/session scoping, author roles, content hash, completed
pairs, source revision counter). Sessions go through the standard
create/complete path. Corrections go through the standard store correction
path and keep their real (load-time) instant – a correction made later is
newer, exactly like a real one. Returns the immutable source-key → actual
source-id mapping the gold manifest is written against.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

HISTORIES_DIR = Path(__file__).resolve().parent / "histories"


def load_history_file(history_id):
    return json.loads((HISTORIES_DIR / f"{history_id}.json").read_text(encoding="utf-8"))


def load_corrections():
    return json.loads((HISTORIES_DIR / "corrections.json").read_text(encoding="utf-8"))


def _instant(base_day, start_hour, minute_offset):
    day = date.today() + timedelta(days=base_day)
    return datetime(day.year, day.month, day.day, start_hour, 0,
                    tzinfo=timezone.utc) + timedelta(minutes=minute_offset)


def load_history(store, history, *, learner_id, student_actor_id, teacher_actor_id):
    """Insert one frozen history; returns {source_key: actual source id}.

    The mapping carries the concrete frozen turn keys (``<session_key>-t<n>``,
    plus ``:a`` for the assistant source of the same exchange) and the
    semantic keys from ``key_block_keys`` (e.g. ``final.brother_age``).
    """
    mapping = {}
    semantic_keys = {}
    for session in history["sessions"]:
        semantic_keys.update(session.get("key_block_keys") or {})
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT learning_timezone FROM teachcopilot.learners WHERE learner_id=%s",
                    (learner_id,))
        learning_timezone = cur.fetchone()["learning_timezone"]
    for session in history["sessions"]:
        session_key = session["session_key"]
        session_row = store.create_learning_session(
            learner_id, student_actor_id, session["topic_id"], client_session_id=session_key)
        session_id = session_row["id"]
        mapping[f"{session_key}:session"] = str(session_id)
        start = _instant(session["day_offset"], session["start_hour"], 0)
        with store._connect() as conn, conn.cursor() as cur:
            for index, block in enumerate(session["turns"]):
                block_instant = start + timedelta(minutes=2 * index)
                user_id = uuid4().hex
                assistant_id = None
                cur.execute("""INSERT INTO teachcopilot.memory_sources
                    (id,learner_id,topic_id,learning_session_id,actor_id,author_role,content,
                     source_key,event_timezone,activity_timezone,created_at)
                    VALUES (%s,%s,%s,%s,%s,'student',%s,%s,%s,%s,%s) RETURNING source_order""",
                    (user_id, learner_id, session["topic_id"], session_id, str(student_actor_id),
                     block["text"], f"{block['key']}:u", learning_timezone, learning_timezone,
                     block_instant))
                user_order = cur.fetchone()["source_order"]
                if block.get("assistant"):
                    assistant_id = uuid4().hex
                    cur.execute("""INSERT INTO teachcopilot.memory_sources
                        (id,learner_id,topic_id,learning_session_id,actor_id,author_role,content,
                         source_key,event_timezone,activity_timezone,created_at)
                        VALUES (%s,%s,%s,%s,%s,'assistant',%s,%s,%s,%s,%s)""",
                        (assistant_id, learner_id, session["topic_id"], session_id,
                         str(student_actor_id), block["assistant"], f"{block['key']}:a",
                         learning_timezone, learning_timezone,
                         block_instant + timedelta(seconds=75)))
                cur.execute("""INSERT INTO teachcopilot.learning_messages
                    (id,learning_session_id,learner_id,client_message_id,source_id,
                     assistant_source_id,content_hash,status,response,created_at,completed_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,'completed',NULL,%s,%s)""",
                    (str(uuid4()), session_id, learner_id, block["key"], user_id, assistant_id,
                     hashlib.sha256(block["text"].encode("utf-8")).hexdigest(),
                     block_instant, block_instant + timedelta(seconds=75)))
                mapping[block["key"]] = user_id
                if assistant_id:
                    mapping[f"{block['key']}:a"] = assistant_id
                mapping[f"{block['key']}:u"] = user_id
            cur.execute("""UPDATE teachcopilot.learners
                SET memory_source_revision=memory_source_revision+1
                WHERE learner_id=%s""", (learner_id,))
        store.complete_learning_session(session_id, learner_id, student_actor_id)
        session_end = start + timedelta(minutes=2 * len(session["turns"]) + 5)
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE teachcopilot.learning_sessions
                SET created_at=%s,completed_at=%s WHERE id=%s""",
                (start, session_end, session_id))
            cur.execute("""DELETE FROM teachcopilot.memory_jobs
                WHERE learner_id=%s AND kind='memory_update' AND status='queued'""",
                (learner_id,))
    for correction in load_corrections():
        if correction["history_id"] != history["history_id"]:
            continue
        target_id = mapping[semantic_keys[correction["corrects_key"]]]
        result = store.correct_memory_source(
            learner_id, target_id, correction["text"], teacher_actor_id,
            expected_revision=store.get_revision(learner_id),
            expected_source_revision=store.get_learner_source_revision(learner_id),
            client_correction_id=correction["key"])
        mapping[correction["key"]] = result["source"]["id"]
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""DELETE FROM teachcopilot.memory_jobs
                WHERE learner_id=%s AND kind='memory_update' AND status='queued'""",
                (learner_id,))
    missing = [key for key, value in mapping.items() if value is None]
    if missing:
        raise RuntimeError(f"loader lost source mapping for keys: {missing[:5]}")
    for semantic, concrete in semantic_keys.items():
        mapping[semantic] = mapping[concrete]
        if f"{concrete}:a" in mapping:
            mapping[f"{semantic}:a"] = mapping[f"{concrete}:a"]
    return mapping
