# pipeline/event_logger.py
"""Event logging to PostgreSQL for analytics, profile updates, debugging."""
import json
import logging
from contextlib import closing
from pipeline.db import get_connection

logger = logging.getLogger(__name__)


def log_event(child_id: str, event_type: str, payload: dict) -> None:
    """Insert an event record. Never raises — logs and returns."""
    try:
        # closing() guarantees the connection is closed. psycopg2's own context
        # manager commits/rolls back but does NOT close, so `with get_connection()`
        # leaked a connection on every event on the hot serving path.
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (child_id, event_type, payload)
                    VALUES (%s, %s, %s)
                """, (child_id, event_type, json.dumps(payload, default=str, ensure_ascii=False)))
            conn.commit()
        logger.debug("Event logged: child=%s type=%s", child_id, event_type)
    except Exception:
        logger.exception("Failed to log event: child=%s type=%s", child_id, event_type)
