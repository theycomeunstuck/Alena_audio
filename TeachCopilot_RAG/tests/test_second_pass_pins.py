"""Pins for the second-pass findings SP-1..SP-5 (audit branch review).

Source-level pins where a DB is unavailable, unit checks where cheap. Each
pin failed before its corresponding fix.
"""
import inspect
import re
from pathlib import Path

import pytest

import pipeline.profile_agent as pa

ROOT = Path(__file__).resolve().parents[1]


def test_sp1_enqueue_takes_advisory_lock_before_learner_row():
    """SP-1: _enqueue_memory_job must take the queue advisory lock BEFORE the
    learners row lock, or it re-creates the F-031 deadlock against
    complete_learning_session (which holds the advisory and later locks the
    learners row)."""
    import local_app.memory_store as ms

    src = inspect.getsource(ms.MemoryStoreMixin._enqueue_memory_job)
    advisory = src.find("pg_advisory_xact_lock")
    learner = src.find("_memory_learner")
    assert advisory != -1 and learner != -1 and advisory < learner, (
        "the advisory lock must be acquired before the learners row lock in "
        "_enqueue_memory_job")

    src_corr = inspect.getsource(ms.MemoryStoreMixin.correct_memory_source)
    advisory = src_corr.find("pg_advisory_xact_lock")
    learner = src_corr.find("_memory_learner")
    assert advisory != -1 and learner != -1 and advisory < learner, (
        "correct_memory_source row-locks the learner and then enqueues - it "
        "must take the advisory lock first")


def test_sp2_marker_not_keyed_by_message_content():
    """SP-2: without chat_id the outlet must not key the marker by a content
    hash (different conversations collide; edits change the key). The marker
    is used only when a real conversation id is present."""
    src = (ROOT / "pipeline" / "filter_function.py").read_text(encoding="utf-8")
    assert "sha1" not in src, "the content-hash fallback must be gone"
    assert 'body.get("chat_id")' in src
    assert "chat_id" in src and "message_count=user_msg_count if chat_id" in src, (
        "the outlet must pass message_count only when a chat_id exists")


def test_sp3_link_student_maps_race_loss_to_409():
    """SP-3: the store-level race guard raises PermissionError; the route must
    map it to the documented 409 instead of an opaque 500."""
    src = (ROOT / "local_app" / "api.py").read_text(encoding="utf-8")
    start = src.find('def link_student')
    end = src.find('@app.post', start)
    body = src[start:end]
    assert "except PermissionError" in body, \
        "link_student must catch PermissionError and return 409"


def test_sp4_outlet_logs_success_only_for_applied_items():
    """SP-4: update_profile returns the applied count and the outlet logs the
    success line only when something actually applied. F-012: the outlet no
    longer writes a profile_update event - the advisory event is written by
    update_profile itself."""
    src = (ROOT / "pipeline" / "filter_function.py").read_text(encoding="utf-8")
    assert "applied > 0" in src, "the success line must be gated on the applied count"
    assert '"profile_update"' not in src and "'profile_update'" not in src, (
        "the outlet must not write a profile_update event (advisory mode)")

    # Unit check with the fake-connection pattern: valid item -> 1.
    from contextlib import closing

    class FakeCursor:
        def __init__(self):
            self.script = [None, None]

        def execute(self, sql, params=None):
            pass

        def fetchone(self):
            return self.script.pop(0) if self.script else None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeConn:
        def __init__(self):
            self.cursor_obj = FakeCursor()

        def cursor(self):
            return self.cursor_obj

        def commit(self):
            pass

        def close(self):
            pass

    conn = FakeConn()
    orig = pa.get_connection
    pa.get_connection = lambda: conn
    try:
        item = {"topic": "fractions", "score_delta": 5, "new_status": "learning"}
        assert pa.update_profile("child-1", [item], message_count=5,
                                 conversation_key="chat-1") == 1
        assert pa.update_profile("child-1", ["garbage"], message_count=6,
                                 conversation_key="chat-1") == 0
    finally:
        pa.get_connection = orig


def test_sp5_write_card_is_atomic():
    """SP-5: card_updater.write_card must not truncate the only copy of the
    learner card in place - write to a temp file and os.replace."""
    src = (ROOT / "pipeline" / "card_updater.py").read_text(encoding="utf-8")
    start = src.find("def write_card")
    end = src.find("\ndef ", start + 1)
    body = src[start:end]
    assert "os.replace" in body, "write_card must write atomically"
    assert re.search(r"write_text", body) and "tmp" in body
