"""Bound actual model calls across processes without holding row transactions.

The HTTP call owns a PostgreSQL session advisory lock. An active-row lease also
keeps a conservative reservation if its DB connection disappears while a model
request may still be running. Waiting rows heartbeat and expire after a crash.
No provider call is retried here.
"""
from __future__ import annotations

from contextlib import contextmanager
import time
import uuid
import math

import psycopg2
from psycopg2.extras import RealDictCursor


class GenerationCapacityError(RuntimeError):
    """The shared model queue is full or its admission wait expired."""


class GenerationScheduler:
    SLOTS = {"local": 1, "external": 4}
    QUEUE_LIMIT = 40

    def __init__(self, dsn: str, *, wait_timeout: float = 180.0,
                 request_timeout: float = 180.0, poll_seconds: float = 0.05):
        self.dsn = dsn
        if not all(math.isfinite(float(value)) and float(value)>0
                   for value in (wait_timeout, request_timeout, poll_seconds)):
            raise ValueError("Generation timing limits must be finite and positive")
        self.wait_timeout = max(0.01, float(wait_timeout))
        self.request_timeout = max(0.01, float(request_timeout))
        self.poll_seconds = max(0.005, min(float(poll_seconds), 1.0))
        self.wait_lease_seconds = 30.0
        # A dead connection must not immediately admit another request while
        # the previous HTTP request's outcome is still unknown.
        self.active_lease_seconds = max(30.0, self.request_timeout + 30.0)

    def _connect(self):
        return psycopg2.connect(self.dsn, cursor_factory=RealDictCursor,
                                connect_timeout=5, application_name="teachcopilot-generation")

    @staticmethod
    def _slot_key(lane: str, slot: int) -> str:
        return f"teachcopilot_generation:{lane}:{slot}"

    @staticmethod
    def _lock_queue(cur):
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('teachcopilot_memory_queue',0))")
        # A live session lock still protects any call whose lease expires.
        cur.execute("DELETE FROM teachcopilot.generation_admission WHERE lease_until<clock_timestamp()")

    @contextmanager
    def acquire(self, lane: str = "local", *, priority: str = "interactive", request_timeout: float | None = None):
        if lane not in self.SLOTS or priority not in {"interactive", "background"}:
            raise ValueError("Invalid generation lane or priority")
        entry_id = str(uuid.uuid4())
        active_lease_seconds = self.active_lease_seconds
        if request_timeout is not None:
            if not math.isfinite(float(request_timeout)) or float(request_timeout)<=0:
                raise ValueError("Generation request timeout must be finite and positive")
            active_lease_seconds = max(30.0,float(request_timeout)+30.0)
        conn = self._connect()
        admitted_slot = None
        registered = False
        unknown_outcome = False
        deadline = time.monotonic() + self.wait_timeout
        try:
            with conn, conn.cursor() as cur:
                self._lock_queue(cur)
                cur.execute("""SELECT (SELECT count(*) FROM teachcopilot.generation_admission WHERE status='waiting')
                    +(SELECT count(*) FROM teachcopilot.memory_jobs WHERE status='queued') AS count""")
                if cur.fetchone()["count"] >= self.QUEUE_LIMIT:
                    raise GenerationCapacityError("Shared model queue contains 40 waiting requests; retry later")
                cur.execute("""INSERT INTO teachcopilot.generation_admission(id,lane,priority,lease_until)
                    VALUES (%s,%s,%s,clock_timestamp()+(%s * interval '1 second'))""",
                    (entry_id, lane, 10 if priority == "interactive" else 0, self.wait_lease_seconds))
                registered = True
            while admitted_slot is None:
                with conn, conn.cursor() as cur:
                    self._lock_queue(cur)
                    cur.execute("""UPDATE teachcopilot.generation_admission
                        SET lease_until=clock_timestamp()+(%s * interval '1 second')
                        WHERE id=%s AND status='waiting' RETURNING id""", (self.wait_lease_seconds, entry_id))
                    if cur.fetchone() is None:
                        raise GenerationCapacityError("Shared generation reservation expired; retry explicitly later")
                    cur.execute("""SELECT id FROM teachcopilot.generation_admission
                        WHERE lane=%s AND status='waiting' ORDER BY priority DESC,created_at,id LIMIT 1""", (lane,))
                    first = cur.fetchone()
                    if first and str(first["id"]) == entry_id:
                        cur.execute("SELECT slot FROM teachcopilot.generation_admission WHERE lane=%s AND status IN ('active','unknown_outcome')", (lane,))
                        occupied = {row["slot"] for row in cur.fetchall()}
                        for slot in range(self.SLOTS[lane]):
                            if slot in occupied:
                                continue
                            key = self._slot_key(lane, slot)
                            cur.execute("SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS acquired", (key,))
                            if cur.fetchone()["acquired"]:
                                admitted_slot = slot
                                cur.execute("""UPDATE teachcopilot.generation_admission
                                    SET status='active',slot=%s,lease_until=clock_timestamp()+(%s * interval '1 second')
                                    WHERE id=%s""", (slot, active_lease_seconds, entry_id))
                                break
                if admitted_slot is None:
                    if time.monotonic() >= deadline:
                        raise GenerationCapacityError("Shared model admission wait expired; retry explicitly later")
                    time.sleep(min(self.poll_seconds, max(0, deadline-time.monotonic())))
            try:
                yield {"admission_id": entry_id, "lane": lane, "slot": admitted_slot}
            except BaseException:
                # A network/provider exception does not prove that the remote
                # generator stopped. Retain its slot through the safe lease.
                unknown_outcome = True
                raise
        finally:
            try:
                if registered:
                    with conn, conn.cursor() as cur:
                        if unknown_outcome:
                            cur.execute("""INSERT INTO teachcopilot.generation_admission
                                (id,lane,priority,status,slot,lease_until)
                                VALUES (%s,%s,%s,'unknown_outcome',%s,clock_timestamp()+(%s * interval '1 second'))
                                ON CONFLICT(id) DO UPDATE SET status='unknown_outcome',lease_until=EXCLUDED.lease_until""",
                                (entry_id,lane,10 if priority=="interactive" else 0,admitted_slot,active_lease_seconds))
                        else:
                            cur.execute("DELETE FROM teachcopilot.generation_admission WHERE id=%s", (entry_id,))
                        if admitted_slot is not None:
                            cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s,0))", (self._slot_key(lane, admitted_slot),))
            except psycopg2.Error:
                # A disappeared connection retains its durable reservation until
                # expiry. Preserve the original generation result/exception.
                pass
            finally:
                conn.close()
