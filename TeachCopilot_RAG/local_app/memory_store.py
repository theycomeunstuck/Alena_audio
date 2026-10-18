"""Durable learning memory storage, idempotency and bounded worker queue.

API authorization belongs to the existing account layer. Composite foreign keys
and learner predicates additionally prevent mixing sources across learners.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from psycopg2.extras import Json


class MemoryQueueFullError(RuntimeError):
    """The durable queue has forty waiting jobs; retry later."""


class MemoryIdempotencyConflict(ValueError):
    """An existing client identifier was reused for different input."""


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


def _revision_error(message):
    # Store imports this mixin before declaring its public exception.
    from .store import RevisionConflict
    raise RevisionConflict(message)


class MemoryStoreMixin:
    @staticmethod
    def _memory_learner(cur, learner_id, expected_revision=None, lock="UPDATE"):
        if lock not in ("UPDATE", "SHARE"):
            raise ValueError("invalid lock")
        cur.execute("SELECT revision,memory_source_revision FROM teachcopilot.learners WHERE learner_id=%s FOR " + lock, (learner_id,))
        learner = cur.fetchone()
        if learner is None:
            raise KeyError(learner_id)
        if expected_revision is not None and learner["revision"] != expected_revision:
            _revision_error(f"learner revision is {learner['revision']}, expected {expected_revision}")
        return learner

    @staticmethod
    def _memory_source(cur, learner_id, topic_id, actor_id, content, *, author_role="teacher", session_id=None, approved=False, observed_on=None, source_key=None):
        if author_role not in ("student", "teacher", "assistant"):
            raise ValueError("invalid source author")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("source content is required")
        cur.execute("""INSERT INTO teachcopilot.memory_sources
            (learner_id,topic_id,actor_id,content,author_role,learning_session_id,approved,observed_on,source_key)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT(learner_id,source_key) DO NOTHING RETURNING *""",
            (learner_id,topic_id,str(actor_id),content,author_role,session_id,approved,observed_on,source_key))
        source = cur.fetchone()
        if source is None:
            cur.execute("SELECT * FROM teachcopilot.memory_sources WHERE learner_id=%s AND source_key=%s", (learner_id,source_key))
            source = cur.fetchone()
            if source["content"] != content or source["author_role"] != author_role or source["topic_id"] != topic_id:
                raise MemoryIdempotencyConflict("source key was reused for different content")
        else:
            cur.execute("UPDATE teachcopilot.learners SET memory_source_revision=memory_source_revision+1 WHERE learner_id=%s", (learner_id,))
        return _row(source)

    def add_memory_source(self, learner_id, topic_id, content, actor_id, *, author_role="teacher", approved=False, observed_on=None, source_key=None, expected_revision=None):
        with self._connect() as conn, conn.cursor() as cur:
            self._memory_learner(cur,learner_id,expected_revision)
            return self._memory_source(cur,learner_id,topic_id,actor_id,content,author_role=author_role,approved=approved,observed_on=observed_on,source_key=source_key)

    def create_learning_session(self, learner_id, actor_id, topic_id, client_session_id=None):
        if not topic_id or not str(topic_id).strip():
            raise ValueError("topic_id is required")
        with self._connect() as conn, conn.cursor() as cur:
            self._memory_learner(cur,learner_id)
            cur.execute("""INSERT INTO teachcopilot.learning_sessions(learner_id,actor_id,topic_id,client_session_id)
                VALUES (%s,%s,%s,%s) ON CONFLICT(learner_id,actor_id,client_session_id) DO NOTHING RETURNING *""",
                (learner_id,str(actor_id),topic_id,client_session_id))
            session = cur.fetchone()
            if session is None:
                cur.execute("SELECT * FROM teachcopilot.learning_sessions WHERE learner_id=%s AND actor_id=%s AND client_session_id=%s", (learner_id,str(actor_id),client_session_id))
                session = cur.fetchone()
                if session["topic_id"] != topic_id:
                    raise MemoryIdempotencyConflict("session identifier belongs to another topic")
            return _row(session)

    def get_learning_session(self, session_id, learner_id=None):
        with self._connect() as conn, conn.cursor() as cur:
            query = "SELECT * FROM teachcopilot.learning_sessions WHERE id=%s"
            values = [session_id]
            if learner_id is not None:
                query += " AND learner_id=%s"
                values.append(learner_id)
            cur.execute(query,values)
            return _row(cur.fetchone())

    def reserve_chat_message(self, session_id, learner_id, client_message_id, content, actor_id, *, author_role="student"):
        if author_role not in ("student", "teacher"):
            raise ValueError("chat source must be student or teacher")
        if not client_message_id or len(client_message_id) > 200:
            raise ValueError("client_message_id must contain 1..200 characters")
        if not isinstance(content,str) or not content.strip():
            raise ValueError("message content is required")
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
            source = self._memory_source(cur,learner_id,session["topic_id"],actor_id,content,author_role=author_role,session_id=session_id,approved=author_role=="teacher",source_key=f"chat:{session_id}:{client_message_id}:student")
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
            source = self._memory_source(cur,message["learner_id"],session["topic_id"],"assistant",text,author_role="assistant",session_id=session_id,source_key=f"chat:{session_id}:{client_message_id}:assistant")
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
                CASE WHEN author_role IN ('student','teacher') THEN 'user' ELSE author_role END AS role,content,created_at
                FROM teachcopilot.memory_sources WHERE learning_session_id=%s ORDER BY source_order""", (session_id,))
            return [_row(row) for row in cur.fetchall()]

    def complete_learning_session(self, session_id, learner_id, actor_id):
        with self._connect() as conn, conn.cursor() as cur:
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
                cur.execute("UPDATE teachcopilot.learners SET memory_source_revision=memory_source_revision+1 WHERE learner_id=%s", (learner_id,))
            job = self._enqueue_memory_job(cur,learner_id,session["topic_id"],"memory_update",{"learning_session_id":str(session_id)},actor_id,idempotency_key=f"session:{session_id}",method="tail")
            return dict(_row(session),job=job,job_id=job["id"])

    def _enqueue_memory_job(self, cur, learner_id, topic_id, kind, payload=None, actor_id="system", expected_revision=None, method="tail", idempotency_key=None, provider="local"):
        if kind not in ("memory_update","next_step","chat") or method not in ("profile_only","tail","summary","structured") or provider not in ("local","external"):
            raise ValueError("invalid memory job options")
        learner = self._memory_learner(cur,learner_id,expected_revision)
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
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
    def _assert_memory_references(cur, learner_id, value):
        """Reject cross-learner source links in persisted context and memory."""
        ids = set()
        def visit(node):
            if isinstance(node,dict):
                for key,item in node.items():
                    if key in ("source_id","evidence_source_id","turn_id","source_turn_id") and isinstance(item,str):
                        ids.add(item)
                    elif key in ("source_ids","evidence_ids","evidence_source_ids","evidence_turn_ids") and isinstance(item,list):
                        ids.update(str(x) for x in item)
                    else:
                        visit(item)
            elif isinstance(node,list):
                for item in node:
                    visit(item)
        visit(value)
        if ids:
            cur.execute("SELECT id FROM teachcopilot.memory_sources WHERE learner_id=%s AND id=ANY(%s) AND author_role <> 'assistant'", (learner_id,list(ids)))
            found = {row["id"] for row in cur.fetchall()}
            if found != ids:
                raise ValueError("memory contains unavailable or assistant evidence sources")

    def _save_memory_version(self, cur, learner_id, topic_id, state, *, base_revision, source_revision=None, approved=False, context=None, job_id=None):
        learner = self._memory_learner(cur,learner_id,base_revision)
        if source_revision is not None and learner["memory_source_revision"] != source_revision:
            _revision_error("learning sources changed while memory was generated")
        self._assert_memory_references(cur,learner_id,{"state":state,"context":context or {}})
        cur.execute("""INSERT INTO teachcopilot.memory_versions(learner_id,topic_id,base_revision,source_revision,state,context,approved,job_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (learner_id,topic_id,base_revision,learner["memory_source_revision"],Json(state),Json(context or {}),approved,job_id))
        return _row(cur.fetchone())

    def save_memory_version(self, learner_id, topic_id, state, *, base_revision, approved=False, context=None, job_id=None, source_revision=None):
        with self._connect() as conn, conn.cursor() as cur:
            return self._save_memory_version(cur,learner_id,topic_id,state,base_revision=base_revision,source_revision=source_revision,approved=approved,context=context,job_id=job_id)

    def _save_next_step(self, cur, learner_id, topic_id, result, *, base_revision, source_revision=None, context=None, memory_version_id=None, job_id=None, method="tail"):
        learner = self._memory_learner(cur,learner_id,base_revision)
        if source_revision is not None and learner["memory_source_revision"] != source_revision:
            _revision_error("learning sources changed while next step was generated")
        self._assert_memory_references(cur,learner_id,{"result":result,"context":context or {}})
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

    def get_learner_memory(self, learner_id, topic_id=None):
        with self._connect() as conn, conn.cursor() as cur:
            learner = self._memory_learner(cur,learner_id,lock="SHARE")
            topic_filter = " AND s.topic_id=%s" if topic_id is not None else ""
            values = [learner_id] + ([topic_id] if topic_id is not None else [])
            cur.execute("""SELECT s.id AS source_id,s.learning_session_id AS session_id,s.author_role,s.content AS text,
                s.content,s.observed_on,s.topic_id,s.approved,s.created_at FROM teachcopilot.memory_sources s
                LEFT JOIN teachcopilot.learning_sessions ls ON ls.id=s.learning_session_id
                WHERE s.learner_id=%s AND s.author_role <> 'assistant'
                AND ((s.author_role='teacher' AND s.approved AND (s.learning_session_id IS NULL OR ls.status='completed')) OR (s.author_role='student' AND ls.status='completed'))""" + topic_filter + " ORDER BY s.source_order", values)
            sources = [_row(row) for row in cur.fetchall()]
            cur.execute("SELECT * FROM teachcopilot.memory_versions WHERE learner_id=%s" + (" AND topic_id=%s" if topic_id is not None else "") + " ORDER BY created_at DESC,id", values)
            versions = [_row(row) for row in cur.fetchall()]
            for version in versions:
                version["method"] = version["state"].get("method") if isinstance(version["state"],dict) else None
                version["stale"] = version["base_revision"] != learner["revision"] or version["source_revision"] != learner["memory_source_revision"]
            return {"learner_id":learner_id,"revision":learner["revision"],"source_revision":learner["memory_source_revision"],"sources":sources,"versions":versions}

    def approve_memory_version(self, learner_id, version_id, actor_id, expected_revision, interest_candidate_ids=None):
        from .conversation_memory import apply_interest_choices, validate_interest_candidates
        selected_ids = [] if interest_candidate_ids is None else interest_candidate_ids
        if (not isinstance(selected_ids,list) or not all(isinstance(item,str) for item in selected_ids)
                or len(selected_ids) != len(set(selected_ids))):
            raise ValueError("interest candidate IDs must be a list of distinct strings")
        with self._connect() as conn, conn.cursor() as cur:
            learner = self._memory_learner(cur,learner_id)
            cur.execute("SELECT * FROM teachcopilot.memory_versions WHERE id=%s AND learner_id=%s FOR UPDATE", (version_id,learner_id))
            version = cur.fetchone()
            if version is None:
                raise KeyError(version_id)
            context = dict(version.get("context") or {})
            if version["approved"]:
                reviewed_ids = (context.get("interest_review") or {}).get("candidate_ids") or []
                if set(reviewed_ids) != set(selected_ids):
                    raise MemoryIdempotencyConflict("memory version was already approved with different interest choices")
                return _row(version)
            if learner["revision"] != expected_revision:
                _revision_error(f"learner revision is {learner['revision']}, expected {expected_revision}")
            if version["base_revision"] != learner["revision"] or version["source_revision"] != learner["memory_source_revision"]:
                _revision_error("memory version no longer matches the learner and sources")
            revision = learner["revision"]
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
                source = self._memory_source(cur,learner_id,entry["topic_id"],actor_id,text,author_role="teacher",approved=True,observed_on=entry.get("observed_on"),source_key=f"proposal:{new_revisions[learner_id]}:{index}:{digest}")
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
