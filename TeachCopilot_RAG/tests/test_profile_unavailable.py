"""Audit F-059: get_child_full_profile turned a database failure into None, so
"the database is down" looked exactly like "this child has no profile" and
personalization was stripped silently. A failure is now ProfileUnavailable;
callers degrade explicitly and say why."""
import logging

import pytest

from pipeline import db
from pipeline import filter_function as ff
from pipeline import profile_agent as pa
from tests.test_filter_prefix_authorization import MY_CHILD, OWUI_USER, filter_env  # noqa: F401


class _MissingRowCursor:
    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _MissingRowConnection:
    def cursor(self):
        return _MissingRowCursor()

    def close(self):
        pass


def test_database_failure_raises_profile_unavailable(monkeypatch):
    def down():
        raise db.psycopg2.OperationalError("connection refused")
    monkeypatch.setattr(db, "get_connection", down)
    with pytest.raises(db.ProfileUnavailable):
        db.get_child_full_profile(MY_CHILD)


def test_missing_child_is_still_none(monkeypatch):
    monkeypatch.setattr(db, "get_connection", lambda: _MissingRowConnection())
    assert db.get_child_full_profile(MY_CHILD) is None


def test_inlet_marks_the_body_and_logs_when_the_profile_store_is_down(filter_env, monkeypatch, caplog):
    def down(child_id):
        raise db.ProfileUnavailable("profile store unavailable")
    monkeypatch.setattr(ff, "get_child_full_profile", down)
    body = {"messages": [{"role": "user", "content": "что такое луч?"}]}
    with caplog.at_level(logging.ERROR):
        body = ff.Filter().inlet(body, {"id": OWUI_USER})
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}  # base prompt, no profile
    assert filter_env["prompts"][-1][0] is None
    assert body[ff.PROFILE_STATUS_KEY] == "unavailable"
    assert any("without personalization" in record.getMessage() for record in caplog.records)


def test_assessment_records_why_it_was_skipped(monkeypatch):
    events = []
    monkeypatch.setattr(pa, "PROFILE_LLM_MODEL", "local-model")
    monkeypatch.setattr(pa, "log_event", lambda child_id, kind, payload=None, **kw: events.append((kind, payload)))

    def down(child_id):
        raise db.ProfileUnavailable("profile store unavailable")
    monkeypatch.setattr(pa, "get_child_full_profile", down)
    assert pa.assess_interaction(MY_CHILD, [{"role": "user", "content": "вопрос"}]) == []
    assert events == [("assessment_failed", {"reason": "profile_store_unavailable"})]

    events.clear()
    monkeypatch.setattr(pa, "get_child_full_profile", lambda child_id: None)
    assert pa.assess_interaction(MY_CHILD, [{"role": "user", "content": "вопрос"}]) == []
    assert events == [("assessment_failed", {"reason": "profile_unavailable"})]
