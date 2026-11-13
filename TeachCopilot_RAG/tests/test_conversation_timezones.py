from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pytest

from local_app.memory_store import MemoryStoreMixin, _history_message
from local_app.store import connect
from local_app.timezones import utc_datetime, validate_timezone


@pytest.mark.parametrize("name", ["UTC", "Asia/Tomsk", "Europe/Moscow", "Asia/Qyzylorda", "America/New_York"])
def test_available_iana_identifiers(name):
    assert validate_timezone(name) == name


@pytest.mark.parametrize("name", [None, "", " UTC", "UTC\n", "../UTC", "+03:00", "No/Such_Zone", "x" * 129])
def test_invalid_timezone_identifiers(name):
    with pytest.raises(ValueError, match="IANA"):
        validate_timezone(name)


def test_utc_normalization_preserves_seconds_and_microseconds():
    local = datetime(2026, 11, 7, 0, 1, 39, 123456, tzinfo=timezone(timedelta(hours=7)))
    expected = datetime(2026, 11, 6, 17, 1, 39, 123456, tzinfo=timezone.utc)
    assert utc_datetime(local) == expected
    assert utc_datetime(local.isoformat()) == expected
    assert utc_datetime("2026-11-06T17:01:39.123456Z") == expected


@pytest.mark.parametrize("value", [None, date(2026, 11, 7), datetime(2026, 11, 7), "2026-11-07", "broken"])
def test_naive_or_invalid_event_time_is_rejected(value):
    with pytest.raises(ValueError, match="UTC offset"):
        utc_datetime(value)


def test_database_startup_timezone_preserves_other_connection_options(monkeypatch):
    observed = {}

    class Connection:
        def commit(self):
            observed["committed"] = True

        def close(self):
            observed["closed"] = True

    def open_connection(dsn, **kwargs):
        observed.update(kwargs)
        return Connection()

    monkeypatch.setattr("local_app.store.psycopg2.connect", open_connection)
    with connect("dbname=test options='-c statement_timeout=1000'"):
        pass
    assert observed["options"] == "-c statement_timeout=1000 -c timezone=UTC"
    assert observed["committed"] and observed["closed"]


def test_public_history_has_precise_utc_time_and_explicit_unknown_legacy_zone():
    turn = {"status": "failed", "user_content": "My actual words", "user_source_id": "source-1",
            "user_created_at": datetime(2026, 11, 7, 0, 1, 39, 5, tzinfo=timezone(timedelta(hours=7))),
            "user_event_timezone": None, "user_activity_timezone": "Asia/Tomsk",
            "user_activity_day": date(2026, 11, 7)}
    message = _history_message(turn, "user")
    assert message["created_at"] == "2026-11-06T17:01:39.000005+00:00"
    assert message["activity_day"] == "2026-11-07"
    assert message["event_timezone"] is None
    assert message["delivery_status"] == "failed"


class ReaderStore(MemoryStoreMixin):
    def __init__(self):
        self.values = None
        self.sql = None

    @contextmanager
    def _connect(self):
        owner = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def execute(self, sql, values):
                owner.sql, owner.values = sql, values

            def fetchall(self):
                return []

        class Connection:
            def cursor(self):
                return Cursor()

            def set_session(self, **options):
                assert options == {"readonly": True, "isolation_level": "REPEATABLE READ"}

        yield Connection()


def test_own_reader_accepts_exact_individual_limit_and_stable_cutoff():
    store = ReaderStore()
    cutoff = datetime(2026, 11, 7, 10, 23, tzinfo=timezone.utc)
    assert store.get_own_conversation_messages("learner", "actor", limit=20,
        active_day_limit=7, exclude_source_id="current", as_of=cutoff) == []
    assert store.values["limit"] == 20
    assert store.values["cutoff"] == cutoff
    assert store.values["excluded"] == "current"
    assert "SELECT DISTINCT activity_day" in store.sql
    assert "AT TIME ZONE" not in store.sql
    assert "IS NOT DISTINCT FROM c.learning_session_id" in store.sql


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": True}, {"limit": 10001},
    {"active_day_limit": 0}, {"active_day_limit": True}, {"before_source_order": 0}, {"before_source_order": 2 ** 64},
    {"exclude_source_id": ""}, {"as_of": datetime(2026, 11, 7)}, {"learning_timezone": "Invalid/Zone"}])
def test_own_reader_rejects_invalid_window_before_query(options):
    store = ReaderStore()
    with pytest.raises(ValueError):
        store.get_own_conversation_messages("learner", "actor", **options)
    assert store.sql is None
