"""Payload-shape validation tests for the profile-agent hot path (audit finding
F-015), run against pure-Python fakes — no DB server, no model download.

Finding covered — F-015: update_profile trusted the LLM payload shape:
- json.loads output with no shape check: a dict payload made
  `for a in assessments` iterate the dict's string keys (AttributeError) and a
  string score_delta made max(-10, min(15, "10")) raise TypeError; each abort
  inside the single transaction, so the broad except rolled back the whole
  batch (including the valid items that preceded the bad one) while the caller
  still logged "Profile updated".
- the INSERT branch wrote new_status verbatim at 50+delta, bypassing the 40/75
  transition rules the UPDATE branch enforces;
- unknown status strings ("Knows", "excellent") were stored verbatim and then
  matched no prompt bucket (pipeline/prompt_builder.py:84–86 buckets on the
  lowercase literals "knows"/"learning"/"struggling" only), making the topic
  invisible in every prompt section.

Expected behaviour after the fix: malformed items are skipped individually with
a logged warning (the rest of the batch still applies and commits), score
deltas must be plain finite numbers clamped to -10..+15, statuses are coerced
into the lowercase vocabulary, and the INSERT branch derives the initial
status through the same 40/75 rules as the UPDATE branch.
"""
import copy
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import profile_agent as pa  # noqa: E402

CHILD = "00000000-0000-0000-0000-000000000001"
TOPIC = "geometry: segments"


def _assessment(topic=TOPIC, score_delta=10, status="learning", error_tags=None):
    a = {"topic": topic, "score_delta": score_delta, "new_status": status}
    if error_tags is not None:
        a["error_tags"] = list(error_tags)
    return a


def _warnings_for(caplog, *needles):
    """Warning (and up) records whose message contains all needles (case-insensitive)."""
    out = []
    for rec in caplog.records:
        if rec.levelno < logging.WARNING:
            continue
        msg = rec.getMessage().lower()
        if all(n.lower() in msg for n in needles):
            out.append(rec)
    return out


# ---------------------------------------------------------------------------
# In-memory transactional fake for pipeline.db.get_connection() (the V4
# repro fake_db.py pattern): staged writes apply only on commit(); rollback
# discards them, mirroring psycopg2's context-manager semantics.
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn
        self.db = conn.db
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        up = s.upper()
        if "FROM EVENTS" in up:
            # marker lookup: (child_id, event_type, message_count, conversation_key)
            child_id, event_type, message_count, conversation_key = params
            want = int(message_count)
            self._row = None
            for ev in self.db._staged_events:
                if ev["child_id"] != child_id or ev["event_type"] != event_type:
                    continue
                payload = json.loads(ev["payload"])
                if payload.get("message_count") == want and \
                        payload.get("conversation_key", "unknown") == conversation_key:
                    self._row = {"marker": 1}
                    break
        elif up.startswith("SELECT SCORE"):
            child_id, topic = params
            row = self.db._staged_knowledge.get((child_id, topic))
            self._row = {"score": row["score"], "status": row["status"]} if row else None
        elif "INSERT INTO CHILD_KNOWLEDGE" in up:
            child_id, topic, status, score = params
            key = (child_id, topic)
            if key not in self.db._staged_knowledge:  # ON CONFLICT DO NOTHING
                self.db._staged_knowledge[key] = {"score": score, "status": status}
        elif "UPDATE CHILD_KNOWLEDGE" in up:
            score, status, child_id, topic = params
            key = (child_id, topic)
            if key in self.db._staged_knowledge:
                self.db._staged_knowledge[key] = {"score": score, "status": status}
        elif "INSERT INTO ERROR_PATTERNS" in up:
            child_id, topic, error_tag = params
            key = (child_id, "math", error_tag)
            entry = self.db._staged_errors.get(key)
            if entry is None:
                self.db._staged_errors[key] = {"count": 1, "topic": topic}
            else:
                entry["count"] += 1
                entry["topic"] = topic
        elif "INSERT INTO EVENTS" in up:
            child_id, event_type, payload = params
            self.db._staged_events.append(
                {"child_id": child_id, "event_type": event_type, "payload": payload})
        else:
            raise AssertionError(f"unhandled SQL in fake cursor: {s[:120]}")

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, db):
        self.db = db

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.db.knowledge = dict(self.db._staged_knowledge)
        self.db.errors = {k: dict(v) for k, v in self.db._staged_errors.items()}
        self.db.events = [dict(e) for e in self.db._staged_events]
        self.db.commit_count += 1

    def rollback(self):
        self.db._resync_staged()
        self.db.rollback_count += 1

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False


class FakeDB:
    """Transactional in-memory child_knowledge/error_patterns/events store."""

    def __init__(self):
        self.knowledge = {}   # (child_id, topic) -> {score, status}
        self.errors = {}      # (child_id, subject, error_tag) -> {count, topic}
        self.events = []      # committed event rows
        self.commit_count = 0
        self.rollback_count = 0
        self._resync_staged()

    def _resync_staged(self):
        self._staged_knowledge = dict(self.knowledge)
        self._staged_errors = {k: dict(v) for k, v in self.errors.items()}
        self._staged_events = [dict(e) for e in self.events]

    def get_connection(self):
        return _FakeConn(self)

    def seed(self, child_id, topic, score, status):
        self.knowledge[(child_id, topic)] = {"score": score, "status": status}
        self._staged_knowledge = dict(self.knowledge)

    def markers(self):
        return [json.loads(e["payload"]) for e in self.events
                if e["event_type"] == "profile_assessed"]


@pytest.fixture
def patched_db(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(pa, "get_connection", db.get_connection)
    return db


# ---------------------------------------------------------------------------
# F-015 — payload shape: malformed items skip individually, batch survives
# ---------------------------------------------------------------------------

class TestF015PayloadShapeValidated:
    def test_dict_shaped_payload_does_not_abort_and_marks_boundary(self, patched_db, caplog, monkeypatch):
        """V4 CL-3a(i): a dict payload used to make `for a in assessments`
        iterate the dict's string keys (AttributeError), abort the transaction
        and roll everything back. It must be rejected without touching rows and
        without an abort. REG-4: nothing valid was applied, so the boundary
        marker is NOT written (the boundary stays retryable) and an
        assessment_failed event records why."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        before = copy.deepcopy(patched_db.knowledge)
        logged = []
        monkeypatch.setattr(pa, "log_event",
                            lambda child, etype, payload: logged.append((child, etype, payload)))

        pa.update_profile(CHILD, {"assessments": [dict(_assessment())]}, message_count=3)

        assert patched_db.knowledge == before, "dict payload must not touch rows"
        assert patched_db.commit_count == 1 and patched_db.rollback_count == 0, \
            f"dict payload must not abort the transaction " \
            f"(commits={patched_db.commit_count}, rollbacks={patched_db.rollback_count})"
        assert patched_db.markers() == [], \
            "an all-invalid reply must not burn the boundary (REG-4)"
        assert len(logged) == 1 and logged[0][1] == "assessment_failed" \
            and logged[0][2]["reason"] == "all_items_invalid"
        assert _warnings_for(caplog, "assessment")

    def test_non_dict_items_skipped_valid_ones_still_written(self, patched_db, caplog):
        """Strings/numbers/None among the items are skipped individually; the
        valid dict item in the same batch is still applied and committed."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, "topic: valid", 50, "learning")

        pa.update_profile(CHILD, [
            "garbage string",
            42,
            None,
            _assessment(topic="topic: valid", score_delta=7, status="learning"),
        ])

        assert patched_db.knowledge[(CHILD, "topic: valid")]["score"] == 57, \
            "the valid item must still be applied"
        assert len(patched_db.knowledge) == 1, "skipped items must not create rows"
        assert patched_db.commit_count == 1 and patched_db.rollback_count == 0
        assert len(_warnings_for(caplog, "assessment")) >= 3, \
            "each malformed item must produce a warning"

    def test_string_score_delta_skips_item_without_typeerror(self, patched_db, caplog):
        """V4 CL-3a(ii): max(-10, min(15, "10")) used to raise TypeError and
        the broad except rolled back the whole batch, including the valid item
        before it. Now the bad item is skipped and the valid one still applies."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, "topic: valid-first", 50, "learning")

        pa.update_profile(CHILD, [
            _assessment(topic="topic: valid-first", score_delta=10, status="learning"),
            _assessment(topic="topic: string-delta", score_delta="10", status="learning"),
        ])

        assert patched_db.knowledge[(CHILD, "topic: valid-first")]["score"] == 60, \
            "the valid item must not be lost to the malformed one"
        assert (CHILD, "topic: string-delta") not in patched_db.knowledge, \
            "a non-numeric score_delta must skip the item, not write a row"
        assert patched_db.commit_count == 1 and patched_db.rollback_count == 0
        assert _warnings_for(caplog, "topic: string-delta")

    def test_bool_score_delta_skipped(self, patched_db):
        """True/False are ints in Python; the old clamp silently turned them
        into deltas of 1/0. They are not numbers the model may send."""
        patched_db.seed(CHILD, "topic: bool", 50, "learning")
        pa.update_profile(CHILD, [_assessment(topic="topic: bool", score_delta=True)])
        assert patched_db.knowledge[(CHILD, "topic: bool")] == {"score": 50, "status": "learning"}

    def test_non_finite_score_delta_skipped(self, patched_db):
        """json.loads accepts NaN/Infinity; the old clamp turned them into +15
        (min/max comparisons with NaN/inf fall through). Scores must not move."""
        patched_db.seed(CHILD, "topic: nan", 50, "learning")
        patched_db.seed(CHILD, "topic: inf", 50, "learning")
        pa.update_profile(CHILD, [
            _assessment(topic="topic: nan", score_delta=float("nan")),
            _assessment(topic="topic: inf", score_delta=float("inf")),
        ])
        assert patched_db.knowledge[(CHILD, "topic: nan")]["score"] == 50
        assert patched_db.knowledge[(CHILD, "topic: inf")]["score"] == 50

    def test_empty_topic_skipped(self, patched_db, caplog):
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        pa.update_profile(CHILD, [_assessment(topic="", score_delta=10, status="learning"),
                                  _assessment(topic="   ", score_delta=10, status="learning")])
        assert patched_db.knowledge == {}, "empty-topic items must not create rows"
        assert len(_warnings_for(caplog, "topic")) >= 2


# ---------------------------------------------------------------------------
# F-015 — status vocabulary + 40/75 rules on both branches
# ---------------------------------------------------------------------------

class TestF015StatusVocabularyAndRules:
    def test_insert_branch_applies_40_75_rules(self, patched_db):
        """V4 CL-3b: the INSERT branch used to write new_status verbatim at
        50+delta. A new topic must be derived through the same rules the
        UPDATE branch enforces ('knows' needs >= 75)."""
        pa.update_profile(CHILD, [_assessment(topic="brand new topic", score_delta=5,
                                              status="knows", error_tags=[])])
        row = patched_db.knowledge[(CHILD, "brand new topic")]
        assert row["score"] == 55
        assert row["status"] == "learning", "'knows' at 55 must be demoted, same as UPDATE"

    def test_update_branch_rules_unchanged(self, patched_db):
        patched_db.seed(CHILD, "topic: rules", 70, "learning")
        pa.update_profile(CHILD, [_assessment(topic="topic: rules", score_delta=0,
                                              status="knows", error_tags=[])])
        assert patched_db.knowledge[(CHILD, "topic: rules")]["status"] == "learning"

    def test_status_lowercased_on_update(self, patched_db):
        """'Knows' used to be stored verbatim; prompt_builder buckets on the
        lowercase literals only, so the topic vanished from every section."""
        patched_db.seed(CHILD, TOPIC, 80, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=0, status="Knows", error_tags=[])])
        assert patched_db.knowledge[(CHILD, TOPIC)]["status"] == "knows"

    def test_status_lowercased_before_insert_rules(self, patched_db):
        pa.update_profile(CHILD, [_assessment(topic="topic: cased-insert", score_delta=15,
                                              status="Knows", error_tags=[])])
        row = patched_db.knowledge[(CHILD, "topic: cased-insert")]
        assert row["score"] == 65
        assert row["status"] == "learning", "'Knows' → 'knows', then 65 < 75 demotes"

    def test_unknown_status_skipped_with_warning(self, patched_db, caplog):
        """No code maps 'excellent' (or similar) to a prompt bucket — storing
        it would hide the topic from every section, so the item is skipped."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=10, status="excellent",
                                              error_tags=[])])
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 50, "status": "learning"}, \
            "unknown status must not be stored"
        assert _warnings_for(caplog, "excellent")

    def test_unknown_status_skipped_on_insert(self, patched_db):
        pa.update_profile(CHILD, [_assessment(topic="topic: excellent", score_delta=10,
                                              status="excellent", error_tags=[])])
        assert (CHILD, "topic: excellent") not in patched_db.knowledge

    def test_non_string_status_skipped(self, patched_db, caplog):
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=10, status=3, error_tags=[])])
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 50, "status": "learning"}
        assert _warnings_for(caplog, "new_status")


# ---------------------------------------------------------------------------
# F-015 — well-formed payloads behave exactly as before (and prior audit
# fixes F-013/F-041 stay intact)
# ---------------------------------------------------------------------------

class TestF015ValidPayloadsUnchanged:
    def test_update_delta_clamped_to_plus15(self, patched_db):
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=99, status="learning", error_tags=[])])
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 65, "status": "learning"}

    def test_update_score_clamped_to_100(self, patched_db):
        patched_db.seed(CHILD, TOPIC, 95, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=99, status="learning", error_tags=[])])
        assert patched_db.knowledge[(CHILD, TOPIC)]["score"] == 100

    def test_insert_with_declared_defaults(self, patched_db):
        """Items that only carry a topic keep the documented defaults:
        score_delta 0, status learning."""
        pa.update_profile(CHILD, [{"topic": "topic: only-topic", "error_tags": []}])
        assert patched_db.knowledge[(CHILD, "topic: only-topic")] == \
            {"score": 50, "status": "learning"}

    def test_error_tags_still_normalized(self, patched_db):
        """F-041 stays intact: tags are normalized/dropped as before."""
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=0, status="learning",
                                              error_tags=["Ray vs Segment", "!!!"])])
        assert set(patched_db.errors) == {(CHILD, "math", "ray_vs_segment")}

    def test_non_list_error_tags_dropped_item_still_applied(self, patched_db, caplog):
        """A string error_tags must not become per-character tags (each single
        letter passes the snake_case contract); the item's score still applies."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        a = _assessment(score_delta=5, status="learning")
        a["error_tags"] = "not a list"
        pa.update_profile(CHILD, [a])
        assert patched_db.errors == {}, "non-list error_tags must be dropped, not iterated"
        assert patched_db.knowledge[(CHILD, TOPIC)]["score"] == 55
        assert _warnings_for(caplog, "error_tags")

    def test_marker_still_written_for_valid_payload(self, patched_db):
        """F-013 stays intact: marker in the same transaction, correct count."""
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        pa.update_profile(CHILD, [_assessment(score_delta=10, status="learning",
                                              error_tags=[])], message_count=5)
        assert patched_db.markers() == [{"message_count": 5, "conversation_key": "unknown",
                                         "assessment_count": 1}]
        assert patched_db.knowledge[(CHILD, TOPIC)]["score"] == 60
