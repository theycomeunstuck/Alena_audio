"""Regression tests for the audit-branch review findings REG-1 and REG-4.

REG-1: the profile_assessed marker must be keyed by conversation, not only by
(child_id, message_count) - otherwise the first conversation of a child that
reaches a 5-message boundary permanently disables assessment in every later
conversation of the same child.

REG-4: the marker must not be written when payload validation dropped every
assessment item (an all-invalid reply must not burn the boundary) - and under
the advisory contract (F-012) it must not write a profile_assessment_advisory
row either, only the assessment_failed diagnostic. The advisory row is written
in the same transaction as the marker and keyed with the same conversation
identity.

Pure-Python fakes; no database, no network.
"""
import json

import pytest

import pipeline.profile_agent as pa


class FakeCursor:
    def __init__(self, script):
        self._script = list(script or [])
        self.executes = []

    def execute(self, sql, params=None):
        self.executes.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._script.pop(0) if self._script else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self, script=None):
        self.cursor_obj = FakeCursor(script)
        self.committed = 0

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed += 1

    def close(self):
        pass


def marker_inserts(cursor):
    out = []
    for sql, params in cursor.executes:
        if sql.startswith("INSERT INTO events") and len(params) == 3 and params[1] == pa.ASSESSMENT_MARKER_EVENT:
            out.append(json.loads(params[2]))
    return out


def event_inserts(cursor, event_type):
    out = []
    for sql, params in cursor.executes:
        if sql.startswith("INSERT INTO events") and len(params) == 3 and params[1] == event_type:
            out.append(json.loads(params[2]))
    return out


VALID_ITEM = {"topic": "fractions", "score_delta": 5, "new_status": "learning",
              "error_tags": ["sign_error"]}


@pytest.fixture()
def fake_db(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(pa, "get_connection", lambda: conn)
    return conn


def test_marker_payload_carries_conversation_key(fake_db):
    result = pa.update_profile("child-1", [VALID_ITEM], message_count=5, conversation_key="chat-abc")
    markers = marker_inserts(fake_db.cursor_obj)
    assert result == 1, "update_profile returns the number of valid items"
    assert markers, "marker must be written for an applied assessment"
    assert markers[0]["conversation_key"] == "chat-abc"
    # F-012: the advisory row shares the transaction and the conversation key.
    advisories = event_inserts(fake_db.cursor_obj, pa.ADVISORY_EVENT)
    assert len(advisories) == 1, "exactly one advisory row alongside the marker"
    assert advisories[0]["conversation_key"] == "chat-abc"
    assert advisories[0]["message_count"] == 5
    assert advisories[0]["items"] == [
        {"topic": "fractions", "score_delta": 5, "new_status": "learning",
         "error_tags": ["sign_error"]}]


def test_already_assessed_filters_by_conversation_key(fake_db):
    fake_db.cursor_obj._script = [None]  # no marker row
    pa.already_assessed("child-1", 5, conversation_key="chat-abc")
    selects = [p for sql, p in fake_db.cursor_obj.executes
               if sql.startswith("SELECT 1 FROM events")]
    assert selects, "already_assessed must query the events table"
    assert selects[0][-1] == "chat-abc", "the conversation key must be part of the marker lookup"


def test_marker_lookup_filters_on_conversation_key_in_sql():
    # The cross-conversation guarantee lives in the SQL itself: the marker
    # lookup must filter on payload->>'conversation_key', so a marker from
    # chat-1 can never suppress chat-2 at the same boundary.
    import inspect

    src = inspect.getsource(pa.already_assessed)
    assert "conversation_key" in src
    assert "payload->>'conversation_key' = %s" in " ".join(src.split())


def test_no_marker_when_every_item_is_invalid(fake_db, monkeypatch):
    events = []
    monkeypatch.setattr(pa, "log_event", lambda child, etype, payload: events.append((child, etype, payload)))
    bad_items = [{"topic": ""}, {"topic": "t", "score_delta": "10"}, "garbage"]
    result = pa.update_profile("child-1", bad_items, message_count=5, conversation_key="chat-abc")
    assert result == 0
    assert marker_inserts(fake_db.cursor_obj) == [], "an all-invalid reply must not burn the boundary"
    assert event_inserts(fake_db.cursor_obj, pa.ADVISORY_EVENT) == [], \
        "an all-invalid reply must not write an advisory row either"
    failures = [p for _, etype, p in events if etype == "assessment_failed"]
    assert failures == [{"reason": "all_items_invalid", "message_count": 5,
                         "conversation_key": "chat-abc"}]


def test_mixed_batch_still_writes_marker(fake_db):
    pa.update_profile("child-1", [VALID_ITEM, {"topic": ""}], message_count=10,
                      conversation_key="chat-abc")
    markers = marker_inserts(fake_db.cursor_obj)
    assert markers and markers[0]["assessment_count"] == 1, "only applied items are counted"


def test_outlet_passes_conversation_key():
    # Source-level pin: the outlet must derive a conversation key and hand it
    # to both already_assessed and update_profile.
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "pipeline" / "filter_function.py").read_text(encoding="utf-8")
    assert "conversation_key" in src
    assert "already_assessed(" in src and "update_profile(" in src
