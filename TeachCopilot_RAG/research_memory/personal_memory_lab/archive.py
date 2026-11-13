"""Research-only canonical lesson archive. Search indexes are disposable."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _vector(values):
    if len(values) != 1024:
        raise ValueError("BGE-M3 vector must have 1024 dimensions")
    import math
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError("Invalid vector")
    return "[" + ",".join(str(float(x)) for x in values) + "]"


class Archive:
    """All reads and writes require both run and learner identifiers."""

    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self):
        return psycopg2.connect(self.database_url, connect_timeout=10)

    def initialize(self):
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            if not cur.fetchone()[0]:
                raise RuntimeError("pgvector extension is not installed")
            cur.execute("CREATE SCHEMA IF NOT EXISTS nirs_personal_memory")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nirs_personal_memory.lessons (
                    run_id TEXT NOT NULL, learner_id TEXT NOT NULL, lesson_id TEXT NOT NULL,
                    observed_on DATE NOT NULL, content JSONB NOT NULL, content_hash TEXT NOT NULL,
                    embedding vector(1024), created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (run_id, learner_id, lesson_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nirs_personal_memory.turns (
                    run_id TEXT NOT NULL, learner_id TEXT NOT NULL, lesson_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL, position INTEGER NOT NULL,
                    PRIMARY KEY (run_id, learner_id, turn_id),
                    FOREIGN KEY (run_id, learner_id, lesson_id)
                        REFERENCES nirs_personal_memory.lessons(run_id, learner_id, lesson_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nirs_personal_memory.source_chunks (
                    run_id TEXT NOT NULL, learner_id TEXT NOT NULL, lesson_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL, position INTEGER NOT NULL, content JSONB NOT NULL,
                    content_hash TEXT NOT NULL, embedding vector(1024) NOT NULL,
                    PRIMARY KEY (run_id,learner_id,chunk_id),
                    FOREIGN KEY (run_id,learner_id,lesson_id)
                        REFERENCES nirs_personal_memory.lessons(run_id,learner_id,lesson_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS nirs_personal_memory.index_jobs (
                    run_id TEXT NOT NULL, learner_id TEXT NOT NULL, lesson_id TEXT NOT NULL,
                    backend TEXT NOT NULL, state TEXT NOT NULL CHECK (state IN
                    ('pending','running','complete','failed','uncertain')),
                    error_type TEXT, metadata JSONB NOT NULL DEFAULT '{}',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (run_id, learner_id, lesson_id, backend),
                    FOREIGN KEY (run_id, learner_id, lesson_id)
                        REFERENCES nirs_personal_memory.lessons(run_id, learner_id, lesson_id)
                )
            """)

    @staticmethod
    def _validate_lesson(lesson):
        if not all(isinstance(lesson.get(k), str) and lesson[k] for k in ("lesson_id", "observed_on")):
            raise ValueError("Lesson ID and date are required")
        turns = lesson.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError("Lesson needs turns")
        ids = set()
        for t in turns:
            if not isinstance(t, dict) or not isinstance(t.get("turn_id"), str) or not t["turn_id"]:
                raise ValueError("Every turn requires an ID")
            if t["turn_id"] in ids or t.get("role") not in {"teacher", "student", "assistant"}:
                raise ValueError("Invalid turn role or duplicate ID")
            if t.get("role") == "assistant" and t.get("usage") != "context_only":
                raise ValueError("Assistant turns are context only, never evidence")
            if not isinstance(t.get("text"), str) or not t["text"].strip():
                raise ValueError("Every turn requires text")
            ids.add(t["turn_id"])

    def put(self, run_id: str, learner_id: str, lesson: dict, embedding=None):
        self._validate_lesson(lesson)
        if not run_id or not learner_id:
            raise ValueError("Both scope IDs are required")
        digest = hashlib.sha256(canonical(lesson).encode("utf-8")).hexdigest()
        vector = _vector(embedding) if embedding is not None else None
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO nirs_personal_memory.lessons
                    (run_id, learner_id, lesson_id, observed_on, content, content_hash, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s::vector)
                ON CONFLICT (run_id,learner_id,lesson_id) DO NOTHING
            """, (run_id, learner_id, lesson["lesson_id"], lesson["observed_on"], Json(lesson), digest, vector))
            inserted = cur.rowcount == 1
            if inserted:
                for position, turn in enumerate(lesson["turns"]):
                    cur.execute("""INSERT INTO nirs_personal_memory.turns
                        (run_id,learner_id,lesson_id,turn_id,position) VALUES (%s,%s,%s,%s,%s)""",
                                (run_id, learner_id, lesson["lesson_id"], turn["turn_id"], position))
            if not inserted:
                cur.execute("SELECT content_hash FROM nirs_personal_memory.lessons WHERE run_id=%s AND learner_id=%s AND lesson_id=%s FOR UPDATE",
                            (run_id, learner_id, lesson["lesson_id"]))
                if cur.fetchone()[0] != digest:
                    raise ValueError("Lesson ID already exists with different content")
                if vector is not None:
                    cur.execute("UPDATE nirs_personal_memory.lessons SET embedding=COALESCE(embedding,%s::vector) WHERE run_id=%s AND learner_id=%s AND lesson_id=%s",
                                (vector, run_id, learner_id, lesson["lesson_id"]))
        return {"inserted": inserted, "sha256": digest}

    def put_chunks(self, run_id, learner_id, lesson, chunks, embeddings):
        archived = self.lessons(run_id, learner_id, [lesson["lesson_id"]])
        if len(archived) != 1 or canonical(archived[0]) != canonical(lesson):
            raise ValueError("Source fragments must match the canonical lesson")
        if len(chunks) != len(embeddings) or not chunks:
            raise ValueError("Source fragments and vectors must have the same nonzero length")
        expected = [turn for part in chunks for turn in part["turns"]]
        if expected != lesson["turns"] or any(
            part["lesson_id"] != lesson["lesson_id"] or part["observed_on"] != lesson["observed_on"]
            or part.get("topic_ids", []) != lesson.get("topic_ids", []) for part in chunks
        ):
            raise ValueError("Source fragments do not exactly cover the archived turns")
        with self.connect() as conn, conn.cursor() as cur:
            for position, (part, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = lesson["lesson_id"] + ":c" + str(position + 1).zfill(4)
                digest = hashlib.sha256(canonical(part).encode("utf-8")).hexdigest()
                cur.execute("""INSERT INTO nirs_personal_memory.source_chunks
                    (run_id,learner_id,lesson_id,chunk_id,position,content,content_hash,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector) ON CONFLICT DO NOTHING""",
                            (run_id, learner_id, lesson["lesson_id"], chunk_id, position,
                             Json(part), digest, _vector(embedding)))
                if cur.rowcount == 0:
                    cur.execute("""SELECT content_hash FROM nirs_personal_memory.source_chunks
                        WHERE run_id=%s AND learner_id=%s AND chunk_id=%s""", (run_id, learner_id, chunk_id))
                    if cur.fetchone()[0] != digest:
                        raise ValueError("Source fragment conflicts with existing index")
        return len(chunks)

    def chunks(self, run_id, learner_id, lesson_ids=None):
        with self.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = "SELECT chunk_id,lesson_id,position,content FROM nirs_personal_memory.source_chunks WHERE run_id=%s AND learner_id=%s"
            params = [run_id, learner_id]
            if lesson_ids is not None:
                sql += " AND lesson_id=ANY(%s)"
                params.append(list(lesson_ids))
            cur.execute(sql + " ORDER BY lesson_id,position", params)
            return [dict(row) for row in cur.fetchall()]

    def lessons(self, run_id: str, learner_id: str, ids=None):
        with self.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = "SELECT content FROM nirs_personal_memory.lessons WHERE run_id=%s AND learner_id=%s"
            params = [run_id, learner_id]
            if ids is not None:
                sql += " AND lesson_id = ANY(%s)"
                params.append(list(ids))
            cur.execute(sql + " ORDER BY observed_on,lesson_id", params)
            return [dict(row["content"]) for row in cur.fetchall()]

    def search(self, run_id: str, learner_id: str, query: str, embedding, limit=8, lesson_ids=None,
               source_ids=None, *, channels=("semantic", "lexical")):
        # Separate rankings avoid adding incomparable similarity scales.
        if not channels or not set(channels) <= {"semantic", "lexical"}:
            raise ValueError("Unknown archive search channel")
        vector = _vector(embedding) if "semantic" in channels else None
        semantic, lexical = [], []
        with self.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            restriction = " AND lesson_id=ANY(%s)" if lesson_ids is not None else ""
            if source_ids is not None:
                restriction += (" AND EXISTS (SELECT 1 FROM jsonb_array_elements(content->'turns') AS turn(value)"
                                " WHERE turn.value->>'turn_id'=ANY(%s))")
            values = [vector, run_id, learner_id]
            if lesson_ids is not None:
                values.append(list(lesson_ids))
            if source_ids is not None:
                values.append(list(source_ids))
            values.extend([vector, limit])
            if "semantic" in channels:
                cur.execute("""SELECT lesson_id,chunk_id, 1 - (embedding <=> %s::vector) AS score
                    FROM nirs_personal_memory.source_chunks WHERE run_id=%s AND learner_id=%s"""
                    + restriction + " ORDER BY embedding <=> %s::vector,lesson_id,chunk_id LIMIT %s", values)
                semantic = [dict(row, channel="semantic") for row in cur.fetchall()]
            values = [query, run_id, learner_id, query]
            if lesson_ids is not None:
                values.append(list(lesson_ids))
            if source_ids is not None:
                values.append(list(source_ids))
            values.append(limit)
            if "lexical" in channels:
                cur.execute("""
                    SELECT lesson_id,chunk_id, ts_rank_cd(to_tsvector('russian', content::text),
                                                  plainto_tsquery('russian', %s)) AS score
                    FROM nirs_personal_memory.source_chunks WHERE run_id=%s AND learner_id=%s
                      AND to_tsvector('russian',content::text) @@ plainto_tsquery('russian',%s)"""
                    + restriction + " ORDER BY score DESC,lesson_id,chunk_id LIMIT %s", values)
                lexical = [dict(row, channel="lexical") for row in cur.fetchall()]
        return semantic + lexical

    def claim(self, run_id, learner_id, lesson_id, backend):
        """Claim a single job; interrupted or failed writes need explicit repair."""
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""INSERT INTO nirs_personal_memory.index_jobs
                (run_id,learner_id,lesson_id,backend,state) VALUES (%s,%s,%s,%s,'pending')
                ON CONFLICT DO NOTHING""", (run_id, learner_id, lesson_id, backend))
            cur.execute("""SELECT state FROM nirs_personal_memory.index_jobs
                WHERE run_id=%s AND learner_id=%s AND lesson_id=%s AND backend=%s FOR UPDATE""",
                        (run_id, learner_id, lesson_id, backend))
            state = cur.fetchone()[0]
            if state == "complete":
                return False
            if state != "pending":
                raise RuntimeError("Index state is uncertain or failed; rebuild the scoped index before retry")
            cur.execute("""UPDATE nirs_personal_memory.index_jobs SET state='running',updated_at=now()
                WHERE run_id=%s AND learner_id=%s AND lesson_id=%s AND backend=%s""",
                        (run_id, learner_id, lesson_id, backend))
        return True

    def finish(self, run_id, learner_id, lesson_id, backend, error_type=None, metadata=None):
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""UPDATE nirs_personal_memory.index_jobs SET state=%s,error_type=%s,metadata=%s,updated_at=now()
                WHERE run_id=%s AND learner_id=%s AND lesson_id=%s AND backend=%s AND state='running'""",
                        ("uncertain" if error_type else "complete", error_type, Json(metadata or {}),
                         run_id, learner_id, lesson_id, backend))
            if cur.rowcount != 1:
                raise RuntimeError("Index job was not running")

    def index_references(self, run_id, learner_id, backend):
        with self.connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT lesson_id, metadata FROM nirs_personal_memory.index_jobs
                WHERE run_id=%s AND learner_id=%s AND backend=%s AND state='complete'""",
                        (run_id, learner_id, backend))
            return [dict(row) for row in cur.fetchall()]

    def clone_for_rebuild(self, old_run_id, new_run_id, learner_id):
        """Copy originals and source vectors to a new scope, leaving uncertain indexes untouched."""
        if not old_run_id or not new_run_id or old_run_id == new_run_id or not learner_id:
            raise ValueError("Rebuild needs a different, nonempty run ID and learner ID")
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT count(*) FROM nirs_personal_memory.lessons
                WHERE run_id=%s AND learner_id=%s""", (new_run_id, learner_id))
            if cur.fetchone()[0]:
                raise ValueError("Destination research scope is not empty")
            cur.execute("""INSERT INTO nirs_personal_memory.lessons
                (run_id,learner_id,lesson_id,observed_on,content,content_hash,embedding)
                SELECT %s,learner_id,lesson_id,observed_on,content,content_hash,embedding
                FROM nirs_personal_memory.lessons WHERE run_id=%s AND learner_id=%s""",
                        (new_run_id, old_run_id, learner_id))
            copied = cur.rowcount
            if not copied:
                raise ValueError("Source research scope is empty")
            cur.execute("""INSERT INTO nirs_personal_memory.turns
                (run_id,learner_id,lesson_id,turn_id,position)
                SELECT %s,learner_id,lesson_id,turn_id,position FROM nirs_personal_memory.turns
                WHERE run_id=%s AND learner_id=%s""", (new_run_id, old_run_id, learner_id))
            cur.execute("""INSERT INTO nirs_personal_memory.source_chunks
                (run_id,learner_id,lesson_id,chunk_id,position,content,content_hash,embedding)
                SELECT %s,learner_id,lesson_id,chunk_id,position,content,content_hash,embedding
                FROM nirs_personal_memory.source_chunks WHERE run_id=%s AND learner_id=%s""",
                        (new_run_id, old_run_id, learner_id))
        return copied

