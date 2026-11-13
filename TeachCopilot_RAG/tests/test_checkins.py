"""Daily optional questions retain exact messages and terminal decisions."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from local_app.checkins import (
    CheckinConflict, QUESTION, _local_day, get_or_create_checkin,
    mark_checkin_answered, mark_checkin_shown, skip_checkin,
)


LEARNER = "synthetic-child"
ACTOR = "student-account"
NOW = datetime(2026, 11, 7, 16, 58, 34, tzinfo=timezone.utc)


class FakeCursor:
    def __init__(self, store):
        self.store = store
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def fetchone(self):
        return deepcopy(self.rows.pop(0)) if self.rows else None

    def execute(self, query, params):
        sql = " ".join(query.split())
        store = self.store
        self.rows = []
        if sql.startswith("SELECT l.learning_timezone"):
            if params == (LEARNER, ACTOR) and store.student_active:
                self.rows = [{"learning_timezone": store.zone, "server_now": store.now}]
        elif sql.startswith("SELECT * FROM teachcopilot.learning_sessions"):
            session_id, learner = params
            row = store.sessions.get(session_id)
            if row and learner == row["learner_id"]:
                self.rows = [row]
        elif sql.startswith("INSERT INTO teachcopilot.daily_checkins"):
            learner, actor, day, zone, origin, session, question, version = params
            existing = next((row for row in store.checkins.values()
                             if (row["learner_id"], row["actor_id"], row["activity_date"]) == (learner, actor, day)), None)
            if existing is None:
                row = {"id": str(UUID(int=100 + len(store.checkins))), "learner_id": learner,
                       "actor_id": actor, "activity_date": day, "learning_timezone": zone,
                       "origin_learning_session_id": origin, "learning_session_id": session,
                       "answer_learning_session_id": None, "question": question, "question_version": version,
                       "status": "offered", "offered_at": store.now, "shown_at": None, "resolved_at": None,
                       "assistant_source_id": None, "answered_source_id": None}
                store.checkins[row["id"]] = row
                self.rows = [row]
        elif sql.startswith("SELECT * FROM teachcopilot.daily_checkins"):
            if "WHERE id=%s" in sql:
                checkin_id, learner = params
                row = store.checkins.get(checkin_id)
                if row and row["learner_id"] == learner:
                    self.rows = [row]
            else:
                self.rows = [row for row in store.checkins.values()
                             if (row["learner_id"], row["actor_id"], row["activity_date"]) == params]
        elif sql.startswith("INSERT INTO teachcopilot.daily_checkin_events"):
            checkin_id, learner, kind, source_id = params
            store.events.setdefault((checkin_id, kind), {"learner_id": learner, "source_id": source_id,
                                                        "created_at": store.now})
        elif sql.startswith("UPDATE teachcopilot.daily_checkins"):
            row = store.checkins[params[-1]]
            if "SET status='shown'" in sql:
                source_id, session_id, _ = params
                row.update(status="shown", shown_at=store.now, assistant_source_id=source_id,
                           learning_session_id=session_id)
            elif "SET status='skipped'" in sql:
                row.update(status="skipped", resolved_at=store.now)
            elif "SET status='answered'" in sql:
                source_id, session_id, _ = params
                row.update(status="answered", answered_source_id=source_id,
                           answer_learning_session_id=session_id, resolved_at=store.now)
            else:
                raise AssertionError(sql)
            self.rows = [row]
        elif sql.startswith("SELECT s.*,ls.actor_id AS session_actor_id"):
            source_id, learner = params
            row = store.sources.get(source_id)
            if row and row["learner_id"] == learner and row.get("message_status"):
                self.rows = [row]
        else:
            raise AssertionError(sql)


class FakeStore:
    def __init__(self):
        self.now = NOW
        self.zone = "Asia/Tomsk"
        self.student_active = True
        self.checkins = {}
        self.events = {}
        self.sources = {}
        self.sessions = {}

    @contextmanager
    def _connect(self):
        saved = deepcopy((self.checkins, self.events, self.sources))
        try:
            yield self
        except Exception:
            self.checkins, self.events, self.sources = saved
            raise

    def cursor(self):
        return FakeCursor(self)

    def _memory_source(self, cur, learner_id, topic_id, actor_id, content, **kwargs):
        existing = next((row for row in self.sources.values() if row.get("source_key") == kwargs["source_key"]), None)
        if existing:
            return existing
        source_id = "source-" + str(len(self.sources))
        row = {"id": source_id, "learner_id": learner_id, "topic_id": topic_id,
               "actor_id": actor_id, "content": content, "created_at": self.now, **kwargs}
        self.sources[source_id] = row
        return row

    def session(self, number=1, *, actor=ACTOR, status="active"):
        session_id = str(UUID(int=number))
        self.sessions[session_id] = {"id": session_id, "learner_id": LEARNER, "actor_id": actor,
                                     "status": status, "topic_id": "topic-" + str(number)}
        return session_id

    def answer(self, number=1, **overrides):
        source_id = "answer-" + str(number)
        session_id = self.session(number)
        self.sources[source_id] = {"id": source_id, "learner_id": LEARNER, "actor_id": ACTOR,
            "author_role": "student", "content": "Я сегодня придумал корабль.",
            "created_at": self.now, "answer_session_id": session_id, "session_actor_id": ACTOR,
            "message_status": "pending", **overrides}
        return source_id


def shown(store, session_id=None):
    item = get_or_create_checkin(store, LEARNER, ACTOR, session_id)
    return mark_checkin_shown(store, LEARNER, ACTOR, item["checkin_id"], session_id)


def test_offer_without_session_does_not_create_chat_or_source():
    store = FakeStore()
    item = get_or_create_checkin(store, LEARNER, ACTOR)
    assert item["status"] == "offered" and item["available"]
    assert item["learning_session_id"] is None
    assert item["text"] == QUESTION
    assert item["activity_date"] == "2026-11-07"
    assert item["offered_at"] == NOW.isoformat()
    assert not store.sessions and not store.sources
    assert list(store.events) == [(item["checkin_id"], "offered")]


def test_offer_is_once_per_activity_day_across_topics_sessions_and_reload():
    store = FakeStore()
    first = get_or_create_checkin(store, LEARNER, ACTOR, store.session(1))
    second = get_or_create_checkin(store, LEARNER, ACTOR, store.session(2))
    assert first == second
    assert get_or_create_checkin(store, LEARNER, ACTOR) == first
    assert len(store.checkins) == len(store.events) == 1


def test_shown_question_is_one_literal_standalone_assistant_source():
    store = FakeStore()
    item = shown(store)
    assert item["status"] == "shown"
    source = store.sources[item["assistant_source_id"]]
    assert source["content"] == QUESTION and source["author_role"] == "assistant"
    assert source["session_id"] is None and source["activity_timezone"] == "Asia/Tomsk"
    assert mark_checkin_shown(store, LEARNER, ACTOR, item["checkin_id"], store.session()) == item
    assert len(store.sources) == 1 and len(store.events) == 2


def test_show_can_bind_a_new_session_after_offer_session_completed():
    store = FakeStore()
    old_id, new_id = store.session(1), store.session(2)
    item = get_or_create_checkin(store, LEARNER, ACTOR, old_id)
    store.sessions[old_id]["status"] = "completed"
    displayed = mark_checkin_shown(store, LEARNER, ACTOR, item["checkin_id"], new_id)
    row = store.checkins[item["checkin_id"]]
    assert row["origin_learning_session_id"] == old_id
    assert displayed["learning_session_id"] == new_id
    assert store.sources[displayed["assistant_source_id"]]["session_id"] == new_id


def test_skip_archives_question_and_remains_terminal_across_conversations():
    store = FakeStore()
    item = get_or_create_checkin(store, LEARNER, ACTOR)
    skipped = skip_checkin(store, LEARNER, ACTOR, item["checkin_id"])
    assert skipped["status"] == "skipped" and not skipped["available"]
    assert skipped["shown_at"] and skipped["assistant_source_id"]
    assert skip_checkin(store, LEARNER, ACTOR, item["checkin_id"], store.session(2)) == skipped
    assert get_or_create_checkin(store, LEARNER, ACTOR, store.session(3)) == skipped
    assert len(store.sources) == 1 and len(store.events) == 3
    with pytest.raises(CheckinConflict, match="skipped"):
        mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"], store.answer())


def test_answer_links_existing_student_source_without_promoting_or_rewriting_it():
    store = FakeStore()
    item = shown(store)
    store.now += timedelta(seconds=1)
    source_id = store.answer()
    source_before = deepcopy(store.sources[source_id])
    answer = mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"], source_id)
    assert answer["status"] == "answered" and not answer["available"]
    assert answer["learning_session_id"] is None
    assert answer["answered_source_id"] == source_id
    assert store.sources[source_id] == source_before
    assert mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"], source_id) == answer
    assert len(store.sources) == 2 and len(store.events) == 3
    with pytest.raises(CheckinConflict, match="another answer"):
        mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"], store.answer(2))
    with pytest.raises(CheckinConflict, match="already answered"):
        skip_checkin(store, LEARNER, ACTOR, item["checkin_id"])


@pytest.mark.parametrize("overrides", [
    {"actor_id": "foreign"}, {"session_actor_id": "foreign"}, {"author_role": "teacher"},
    {"author_role": "assistant"},
])
def test_answer_rejects_foreign_or_nonstudent_source(overrides):
    store = FakeStore()
    item = shown(store)
    with pytest.raises(PermissionError):
        mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"], store.answer(**overrides))
    assert store.checkins[item["checkin_id"]]["status"] == "shown"


def test_answer_requires_displayed_question_before_source_on_same_activity_day():
    store = FakeStore()
    item = get_or_create_checkin(store, LEARNER, ACTOR)
    with pytest.raises(CheckinConflict, match="displayed"):
        mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"], store.answer())
    mark_checkin_shown(store, LEARNER, ACTOR, item["checkin_id"])
    for index, instant in enumerate((NOW - timedelta(seconds=1), NOW + timedelta(days=1)), 2):
        with pytest.raises(CheckinConflict, match="same activity day"):
            mark_checkin_answered(store, LEARNER, ACTOR, item["checkin_id"],
                                 store.answer(index, created_at=instant))


def test_next_tomsk_day_gets_one_new_offer_without_erasing_previous_history():
    store = FakeStore()
    old = shown(store)
    skip_checkin(store, LEARNER, ACTOR, old["checkin_id"])
    old_snapshot = deepcopy(store.checkins[old["checkin_id"]])
    store.now = datetime(2026, 11, 7, 17, 0, tzinfo=timezone.utc)
    new = get_or_create_checkin(store, LEARNER, ACTOR)
    assert new["checkin_id"] != old["checkin_id"]
    assert new["activity_date"] == "2026-11-08"
    assert store.checkins[old["checkin_id"]] == old_snapshot
    assert len(store.events) == 4


def test_old_unresolved_offer_cannot_be_shown_or_skipped_on_new_day():
    store = FakeStore()
    old = get_or_create_checkin(store, LEARNER, ACTOR)
    store.now += timedelta(days=1)
    for action in (mark_checkin_shown, skip_checkin):
        with pytest.raises(CheckinConflict, match="earlier activity day"):
            action(store, LEARNER, ACTOR, old["checkin_id"])
    assert not store.sources and len(store.events) == 1


@pytest.mark.parametrize("learner,actor", [("foreign-child", ACTOR), (LEARNER, "foreign-account")])
def test_missing_account_scope_cannot_offer_or_read_questions(learner, actor):
    store = FakeStore()
    with pytest.raises(PermissionError):
        get_or_create_checkin(store, learner, actor)
    assert not store.checkins


def test_foreign_session_and_inactive_account_are_rejected():
    store = FakeStore()
    with pytest.raises(PermissionError, match="another account"):
        get_or_create_checkin(store, LEARNER, ACTOR, store.session(actor="foreign"))
    store.student_active = False
    with pytest.raises(PermissionError, match="active student"):
        get_or_create_checkin(store, LEARNER, ACTOR)


def test_day_uses_learning_timezone_and_handles_dst_offsets():
    boundary = datetime(2026, 11, 7, 17, 0, tzinfo=timezone.utc)
    assert _local_day(boundary, "Asia/Tomsk").isoformat() == "2026-11-08"
    assert _local_day(boundary, "Europe/Moscow").isoformat() == "2026-11-07"
    dst_fold_early = datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)
    dst_fold_late = datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc)
    assert _local_day(dst_fold_early, "America/New_York") == _local_day(dst_fold_late, "America/New_York")
    with pytest.raises(ValueError, match="UTC offset"):
        _local_day(boundary.replace(tzinfo=None), "Asia/Tomsk")
