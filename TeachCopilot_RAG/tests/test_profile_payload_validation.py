"""Payload-shape validation tests for the profile-agent hot path (audit finding
F-015 under the F-012 advisory contract), run against pure-Python fakes – no DB
server, no model download.

Finding covered – F-015: update_profile trusted the LLM payload shape:
- json.loads output with no shape check: a dict payload made
  `for a in assessments` iterate the dict's string keys (AttributeError) and a
  string score_delta made max(-10, min(15, "10")) raise TypeError; each abort
  inside the single transaction rolled back the whole batch;
- unknown status strings ("Knows", "excellent") were stored verbatim and then
  matched no prompt bucket (pipeline/prompt_builder.py buckets on the lowercase
  literals "knows"/"learning"/"struggling" only), making the topic invisible in
  every prompt section.

Expected behaviour now: malformed items are skipped individually with a logged
warning (the rest of the batch is still recorded), score deltas must be plain
finite numbers clamped to -10..+15, and statuses are coerced into the lowercase
vocabulary. Under F-012 (user decision 2026-11-15) update_profile is ADVISORY:
nothing is written to child_knowledge or error_patterns at all. A reply with at
least one valid item records ONE profile_assessed marker row (at a message
boundary) and ONE profile_assessment_advisory row carrying the validated items,
in the same transaction; a reply where validation dropped every item records
only an assessment_failed event (reason all_items_invalid), writes no marker
and keeps the boundary retryable (REG-4). The old 40/75 status transition rules
apply nowhere any more – nothing is written – so the advisory item preserves
the model's validated, lowercased status as recorded data.
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
# In-memory fake for pipeline.db.get_connection(). Staged writes apply only on
# commit(); rollback discards them, mirroring psycopg2's semantics. The
# child_knowledge/error_patterns branches still exist so a regression to score
# writes fails with a clear state diff, and every executed statement is
# recorded for the zero-writes assertions (the defining F-012 property).
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
        self.db.statements.append(s)
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


class FakeDB:
    """Transactional in-memory child_knowledge/error_patterns/events store."""

    def __init__(self):
        self.knowledge = {}   # (child_id, topic) -> {score, status}
        self.errors = {}      # (child_id, subject, error_tag) -> {count, topic}
        self.events = []      # committed event rows
        self.statements = []  # every SQL statement executed against the fake
        self.commit_count = 0
        self.rollback_count = 0
        self._resync_staged()

    def _resync_staged(self):
        self._staged_knowledge = dict(self.knowledge)
        self._staged_errors = {k: dict(v) for k, v in self.errors.items()}
        self._staged_events = [dict(e) for e in self.events]

    def get_connection(self):
        self._resync_staged()  # a fresh transaction sees committed state
        return _FakeConn(self)

    def seed(self, child_id, topic, score, status):
        self.knowledge[(child_id, topic)] = {"score": score, "status": status}
        self._staged_knowledge = dict(self.knowledge)

    def markers(self):
        return [json.loads(e["payload"]) for e in self.events
                if e["event_type"] == "profile_assessed"]

    def advisories(self):
        return [json.loads(e["payload"]) for e in self.events
                if e["event_type"] == "profile_assessment_advisory"]

    def failures(self):
        return [json.loads(e["payload"]) for e in self.events
                if e["event_type"] == "assessment_failed"]

    def profile_writes(self):
        """Statements touching the score tables update_profile must never use."""
        return [s for s in self.statements
                if "child_knowledge" in s or "error_patterns" in s]


@pytest.fixture
def patched_db(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(pa, "get_connection", db.get_connection)

    # update_profile reaches log_event through this module's namespace; route
    # assessment_failed events into the fake's staged rows exactly like a real
    # INSERT INTO events + commit would, so assertions stay on one surface
    # (the committed events list).
    def fake_log_event(child_id, event_type, payload):
        db._staged_events.append(
            {"child_id": child_id, "event_type": event_type,
             "payload": json.dumps(payload, default=str, ensure_ascii=False)})

    monkeypatch.setattr(pa, "log_event", fake_log_event)
    return db


# ---------------------------------------------------------------------------
# F-015 – payload shape: malformed items skip individually, batch survives
# ---------------------------------------------------------------------------

class TestF015PayloadShapeValidated:
    def test_dict_shaped_payload_does_not_abort_or_mark_boundary(self, patched_db, caplog):
        """V4 CL-3a(i): a dict payload used to make `for a in assessments`
        iterate the dict's string keys (AttributeError), abort the transaction
        and roll everything back. It must be rejected without writing anything:
        no marker (REG-4 – the boundary stays retryable, already_assessed stays
        False), NO advisory row, and one assessment_failed event with reason
        all_items_invalid."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        before = copy.deepcopy(patched_db.knowledge)

        result = pa.update_profile(CHILD, {"assessments": [dict(_assessment())]}, message_count=3)

        assert result == 0, "update_profile returns the number of valid items"
        assert patched_db.knowledge == before and patched_db.errors == {}, \
            "a dict payload must not touch any rows"
        assert patched_db.commit_count == 1 and patched_db.rollback_count == 0, \
            "a dict payload must not abort the transaction"
        assert patched_db.markers() == [], \
            "an all-invalid reply must not burn the boundary (REG-4)"
        assert patched_db.advisories() == [], "no advisory row for an all-invalid reply"
        assert patched_db.failures() == [{"reason": "all_items_invalid",
                                          "message_count": 3,
                                          "conversation_key": "unknown"}]
        assert pa.already_assessed(CHILD, 3) is False, "the boundary stays retryable"
        assert patched_db.profile_writes() == []
        assert _warnings_for(caplog, "assessment")

    def test_non_dict_items_skipped_valid_one_still_recorded(self, patched_db, caplog):
        """Strings/numbers/None among the items are skipped individually; the
        valid dict item in the same batch is still recorded in the advisory."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")

        result = pa.update_profile(CHILD, [
            "garbage string",
            42,
            None,
            _assessment(topic="topic: valid", score_delta=7),
        ])

        assert result == 1
        advisories = patched_db.advisories()
        assert len(advisories) == 1
        assert advisories[0]["message_count"] is None, "no boundary passed"
        assert advisories[0]["conversation_key"] == "unknown"
        assert advisories[0]["items"] == [{"topic": "topic: valid", "score_delta": 7,
                                           "new_status": "learning", "error_tags": []}]
        assert patched_db.markers() == [], "without a boundary no marker is written"
        assert patched_db.commit_count == 1 and patched_db.rollback_count == 0
        assert len(_warnings_for(caplog, "assessment")) >= 3, \
            "each malformed item must produce a warning"

    def test_string_score_delta_skips_item_without_typeerror(self, patched_db, caplog):
        """V4 CL-3a(ii): max(-10, min(15, "10")) used to raise TypeError and the
        broad except rolled back the whole batch. Now the bad item is skipped
        and the valid one is still recorded."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")

        result = pa.update_profile(CHILD, [
            _assessment(topic="topic: valid-first", score_delta=10),
            _assessment(topic="topic: string-delta", score_delta="10"),
        ])

        assert result == 1
        items = patched_db.advisories()[0]["items"]
        assert items == [{"topic": "topic: valid-first", "score_delta": 10,
                          "new_status": "learning", "error_tags": []}], \
            "the valid item must not be lost to the malformed one"
        assert patched_db.commit_count == 1 and patched_db.rollback_count == 0
        assert _warnings_for(caplog, "topic: string-delta")

    def test_bool_score_delta_skipped(self, patched_db):
        """True/False are ints in Python; the old clamp silently turned them
        into deltas of 1/0. They are not numbers the model may send – the item
        is skipped, so no advisory item carries a bool-derived delta."""
        patched_db.seed(CHILD, "topic: bool", 50, "learning")

        result = pa.update_profile(CHILD, [_assessment(topic="topic: bool", score_delta=True)],
                                   message_count=4)

        assert result == 0
        assert patched_db.advisories() == [] and patched_db.markers() == []
        assert patched_db.failures()[0]["reason"] == "all_items_invalid"
        assert patched_db.knowledge[(CHILD, "topic: bool")] == {"score": 50, "status": "learning"}

    def test_non_finite_score_delta_skipped(self, patched_db):
        """json.loads accepts NaN/Infinity; the old clamp turned them into +15
        (min/max comparisons with NaN/inf fall through). Neither may be
        recorded as a delta."""
        patched_db.seed(CHILD, "topic: nan", 50, "learning")
        patched_db.seed(CHILD, "topic: inf", 50, "learning")

        result = pa.update_profile(CHILD, [
            _assessment(topic="topic: nan", score_delta=float("nan")),
            _assessment(topic="topic: inf", score_delta=float("inf")),
        ], message_count=4)

        assert result == 0
        assert patched_db.advisories() == [] and patched_db.markers() == []
        assert patched_db.knowledge[(CHILD, "topic: nan")] == {"score": 50, "status": "learning"}
        assert patched_db.knowledge[(CHILD, "topic: inf")] == {"score": 50, "status": "learning"}

    def test_empty_topic_skipped(self, patched_db, caplog):
        """Empty/blank topics skip the item; with every item dropped only the
        assessment_failed diagnostic is recorded – no advisory row, no marker."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")

        result = pa.update_profile(CHILD, [_assessment(topic="", score_delta=10),
                                           _assessment(topic="   ", score_delta=10)],
                                   message_count=8)

        assert result == 0
        assert patched_db.advisories() == [] and patched_db.markers() == []
        assert patched_db.failures() == [{"reason": "all_items_invalid",
                                          "message_count": 8,
                                          "conversation_key": "unknown"}]
        assert len(_warnings_for(caplog, "topic")) >= 2


# ---------------------------------------------------------------------------
# F-015 – status vocabulary; the 40/75 transition rules are gone (advisory)
# ---------------------------------------------------------------------------

class TestF015StatusVocabularyAdvisory:
    def test_insert_case_advisory_preserves_status_without_transition_rules(self, patched_db):
        """Was V4 CL-3b: the INSERT branch had to derive the initial status
        through the same 40/75 rules the UPDATE branch enforced ('knows' needs
        >= 75), else a verbatim 'knows' at score 55 contradicted it. Under
        F-012 nothing is written to child_knowledge any more, so NO transition
        rules apply anywhere (the rules helper was removed): the validated,
        lowercased model status is preserved verbatim as advisory data."""
        result = pa.update_profile(CHILD, [_assessment(topic="brand new topic", score_delta=5,
                                                       status="knows", error_tags=[])],
                                   message_count=4)
        assert result == 1
        assert patched_db.advisories()[0]["items"] == [
            {"topic": "brand new topic", "score_delta": 5, "new_status": "knows",
             "error_tags": []}], \
            "the validated status is recorded as-is – no 40/75 demotion"
        assert [m["assessment_count"] for m in patched_db.markers()] == [1]
        assert patched_db.knowledge == {} and patched_db.profile_writes() == []

    def test_update_case_advisory_preserves_status_without_transition_rules(self, patched_db):
        """Was: an existing topic at 70/'learning' with status 'knows' was
        demoted back to 'learning' by the 40/75 rules. Advisory mode records
        the validated status as data; nothing is written, so no rule applies."""
        patched_db.seed(CHILD, "topic: rules", 70, "learning")
        result = pa.update_profile(CHILD, [_assessment(topic="topic: rules", score_delta=0,
                                                       status="knows", error_tags=[])],
                                   message_count=4)
        assert result == 1
        assert patched_db.advisories()[0]["items"][0]["new_status"] == "knows"
        assert patched_db.knowledge[(CHILD, "topic: rules")] == {"score": 70, "status": "learning"}, \
            "the seeded knowledge row is untouched"
        assert patched_db.profile_writes() == []

    def test_status_lowercased_in_advisory_item(self, patched_db):
        """'Knows' used to be stored verbatim; prompt_builder buckets on the
        lowercase literals only, so the topic vanished from every section.
        The advisory item must carry the lowercased status."""
        patched_db.seed(CHILD, TOPIC, 80, "learning")
        result = pa.update_profile(CHILD, [_assessment(score_delta=0, status="Knows",
                                                       error_tags=[])])
        assert result == 1
        assert patched_db.advisories()[0]["items"][0]["new_status"] == "knows"
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 80, "status": "learning"}

    def test_status_lowercased_before_any_rule_check(self, patched_db):
        """Was: 'Knows' was lowercased and then run through the 40/75 INSERT
        rules (65 < 75 demoted it). Now lowercasing is the only transformation –
        the validated status is preserved as recorded data and nothing is
        written."""
        result = pa.update_profile(CHILD, [_assessment(topic="topic: cased-insert", score_delta=15,
                                                       status="Knows", error_tags=[])],
                                   message_count=4)
        assert result == 1
        assert patched_db.advisories()[0]["items"][0] == {
            "topic": "topic: cased-insert", "score_delta": 15, "new_status": "knows",
            "error_tags": []}
        assert patched_db.profile_writes() == []

    def test_unknown_status_skipped_with_warning(self, patched_db, caplog):
        """No code maps 'excellent' (or similar) to a prompt bucket – recording
        it would emit advisory data no consumer understands, so the item is
        skipped like any other malformed one."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        result = pa.update_profile(CHILD, [_assessment(score_delta=10, status="excellent",
                                                       error_tags=[])], message_count=7)
        assert result == 0
        assert patched_db.advisories() == [] and patched_db.markers() == []
        assert patched_db.failures() == [{"reason": "all_items_invalid",
                                          "message_count": 7,
                                          "conversation_key": "unknown"}]
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 50, "status": "learning"}
        assert _warnings_for(caplog, "excellent")

    def test_unknown_status_recorded_nowhere(self, patched_db):
        """The unknown-status item contributes no advisory item and no marker;
        nothing about it reaches the database."""
        patched_db.seed(CHILD, "topic: excellent", 40, "struggling")
        result = pa.update_profile(CHILD, [_assessment(topic="topic: excellent", score_delta=10,
                                                       status="excellent", error_tags=[])],
                                   message_count=9)
        assert result == 0
        assert patched_db.advisories() == [] and patched_db.markers() == []
        assert patched_db.knowledge[(CHILD, "topic: excellent")] == {"score": 40,
                                                                     "status": "struggling"}

    def test_non_string_status_skipped(self, patched_db, caplog):
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        result = pa.update_profile(CHILD, [_assessment(score_delta=10, status=3, error_tags=[])],
                                   message_count=7)
        assert result == 0
        assert patched_db.advisories() == [] and patched_db.markers() == []
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 50, "status": "learning"}
        assert _warnings_for(caplog, "new_status")


# ---------------------------------------------------------------------------
# F-015 – well-formed payloads are recorded as advisory data (prior audit
# fixes F-013/F-041 stay intact)
# ---------------------------------------------------------------------------

class TestF015ValidPayloadsAdvisory:
    def test_delta_clamped_to_plus15(self, patched_db):
        """The declared maximum (+15) clamps an inflated delta before it is
        recorded; nothing is written to child_knowledge."""
        result = pa.update_profile(CHILD, [_assessment(score_delta=99, status="learning",
                                                       error_tags=[])])
        assert result == 1
        assert patched_db.advisories()[0]["items"][0]["score_delta"] == 15

    def test_delta_clamped_to_minus10(self, patched_db):
        """The declared minimum (-10) clamps a regressed delta. This replaces
        the old absolute score cap at 100 – that cap is gone with the score
        writes; the clamp now lives entirely on the recorded delta."""
        result = pa.update_profile(CHILD, [_assessment(score_delta=-99, status="struggling",
                                                       error_tags=[])])
        assert result == 1
        assert patched_db.advisories()[0]["items"][0]["score_delta"] == -10
        assert patched_db.knowledge == {}, "no knowledge row may appear"

    def test_integral_delta_stays_integral(self, patched_db):
        """7.5 stays 7.5, but 10.0 is recorded as the int 10 so integral deltas
        stay integral for storage."""
        result = pa.update_profile(CHILD, [
            _assessment(topic="topic: float", score_delta=7.5),
            _assessment(topic="topic: integral", score_delta=10.0),
        ])
        assert result == 2
        items = patched_db.advisories()[0]["items"]
        assert items[0]["score_delta"] == 7.5
        assert items[1]["score_delta"] == 10
        assert isinstance(items[1]["score_delta"], int)

    def test_items_with_only_topic_get_declared_defaults(self, patched_db):
        """Items that only carry a topic keep the documented defaults:
        score_delta 0, status learning, no tags."""
        result = pa.update_profile(CHILD, [{"topic": "topic: only-topic", "error_tags": []}])
        assert result == 1
        assert patched_db.advisories()[0]["items"] == [
            {"topic": "topic: only-topic", "score_delta": 0, "new_status": "learning",
             "error_tags": []}]

    def test_error_tags_still_normalized(self, patched_db):
        """F-041 stays intact: tags are normalized/dropped as before – but they
        land in the advisory item, not in error_patterns."""
        result = pa.update_profile(CHILD, [_assessment(score_delta=0, status="learning",
                                                       error_tags=["Ray vs Segment", "!!!"])])
        assert result == 1
        assert patched_db.advisories()[0]["items"][0]["error_tags"] == ["ray_vs_segment"]
        assert patched_db.errors == {}

    def test_non_list_error_tags_dropped_item_still_recorded(self, patched_db, caplog):
        """A string error_tags must not become per-character tags (each single
        letter passes the snake_case contract); the item itself is still
        recorded with an empty tag list."""
        caplog.set_level(logging.WARNING, logger="pipeline.profile_agent")
        a = _assessment(score_delta=5, status="learning")
        a["error_tags"] = "not a list"
        result = pa.update_profile(CHILD, [a])
        assert result == 1
        items = patched_db.advisories()[0]["items"]
        assert items[0]["error_tags"] == [], "non-list error_tags must be dropped, not iterated"
        assert items[0]["score_delta"] == 5, "the item itself is still recorded"
        assert patched_db.errors == {}
        assert _warnings_for(caplog, "error_tags")

    def test_marker_still_written_for_valid_payload(self, patched_db):
        """F-013 stays intact: marker in the same transaction as the advisory
        row, correct count, and update_profile returns the valid-item count."""
        result = pa.update_profile(CHILD, [_assessment(score_delta=10, status="learning",
                                                       error_tags=[])], message_count=5)
        assert result == 1
        assert patched_db.markers() == [{"message_count": 5, "conversation_key": "unknown",
                                         "assessment_count": 1}]
        assert len(patched_db.advisories()) == 1, "the advisory row is written alongside the marker"
        assert patched_db.commit_count == 1, "marker and advisory share one transaction"


# ---------------------------------------------------------------------------
# F-012 – the defining advisory property: zero writes to the score tables
# ---------------------------------------------------------------------------

class TestF012AdvisoryModeWritesNothing:
    def test_valid_multi_item_batch_never_touches_score_tables(self, patched_db):
        """A fully valid multi-item batch records ONE marker and ONE advisory
        row in a single transaction and NOTHING else: zero SELECT/INSERT/
        UPDATE statements against child_knowledge or error_patterns, and a
        pre-seeded knowledge row is left byte-identical. This is the defining
        F-012 property – the EMA zone assessment (learner cards) is the
        authoritative scorer."""
        patched_db.seed(CHILD, TOPIC, 50, "learning")
        batch = [
            _assessment(topic=TOPIC, score_delta=10, status="learning",
                        error_tags=["ray_vs_segment"]),
            _assessment(topic="topic: second", score_delta=-3, status="struggling",
                        error_tags=["sign_error"]),
            _assessment(topic="topic: third", score_delta=0, status="knows"),
        ]

        result = pa.update_profile(CHILD, batch, message_count=6, conversation_key="chat-1")

        assert result == 3
        assert patched_db.commit_count == 1, "marker and advisory share one transaction"
        assert patched_db.markers() == [{"message_count": 6, "conversation_key": "chat-1",
                                         "assessment_count": 3}]
        advisories = patched_db.advisories()
        assert len(advisories) == 1
        assert advisories[0]["message_count"] == 6
        assert advisories[0]["conversation_key"] == "chat-1"
        assert advisories[0]["items"] == [
            {"topic": TOPIC, "score_delta": 10, "new_status": "learning",
             "error_tags": ["ray_vs_segment"]},
            {"topic": "topic: second", "score_delta": -3, "new_status": "struggling",
             "error_tags": ["sign_error"]},
            {"topic": "topic: third", "score_delta": 0, "new_status": "knows",
             "error_tags": []},
        ]
        assert patched_db.knowledge[(CHILD, TOPIC)] == {"score": 50, "status": "learning"}, \
            "the pre-seeded knowledge row is byte-identical"
        assert patched_db.errors == {}
        assert patched_db.profile_writes() == [], \
            "no SELECT/INSERT/UPDATE may touch child_knowledge or error_patterns"
