"""Calendar hierarchy, provenance, bounded context and immutable cache checks."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

from local_app.hierarchical_memory import (
    ALGORITHM, _canonical_messages, _day_payload, _model_context, build_conversation_memory,
)


NOW = datetime(2026, 11, 7, 12, 34, 56, tzinfo=timezone.utc)
LEARNER = "calendar-learner"
ACTOR = "student"


def message(number, *, text="Мой корабль называется Искра.", created_at=NOW, role="user",
            session="session-a", topic="math", activity_timezone="Asia/Tomsk", activity_day=None,
            actor=ACTOR, learner=LEARNER, status="completed"):
    return {"source_id": str(UUID(int=number)), "source_order": number,
        "role": role, "author_role": "student" if role == "user" else "assistant",
        "content": text, "created_at": created_at, "activity_timezone": activity_timezone,
        "activity_day": activity_day or created_at.astimezone(ZoneInfo(activity_timezone)).date(),
        "event_timezone": activity_timezone, "learning_session_id": session, "topic_id": topic,
        "actor_id": actor, "learner_id": learner, "delivery_status": status}


class Cursor:
    def __init__(self, store):
        self.store, self.rows = store, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def execute(self, sql, params):
        sql = " ".join(sql.split())
        if sql.startswith("INSERT INTO teachcopilot.conversation_summary_versions"):
            learner, actor, kind, day, scope, fingerprint, algorithm, watermark, payload, supersedes = params
            found = [row for row in self.store.versions if row["key"] == (learner, actor, kind, day, scope)
                     and row["input_fingerprint"] == fingerprint]
            if found:
                self.rows = []
                return
            row = {"id": str(UUID(int=10000 + len(self.store.versions))),
                "key": (learner, actor, kind, day, scope), "input_fingerprint": fingerprint,
                "algorithm": algorithm, "source_watermark": watermark,
                "payload": deepcopy(payload.adapted), "supersedes_version_id": supersedes}
            self.store.versions.append(row)
            self.rows = [row]
            return
        key, last = tuple(params[:5]), params[5]
        rows = [row for row in self.store.versions if row["key"] == key]
        if "input_fingerprint=%s" in sql:
            rows = [row for row in rows if row["input_fingerprint"] == last]
        elif "source_watermark<=%s" in sql:
            rows = [row for row in rows if row["source_watermark"] <= last]
            rows.sort(key=lambda row: row["source_watermark"], reverse=True)
        else:
            raise AssertionError(sql)
        self.rows = rows[:1]


class Connection:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return Cursor(self.store)


class Store:
    def __init__(self, rows, learning_timezone="Asia/Tomsk"):
        self.rows, self.versions, self.learning_timezone = rows, [], learning_timezone
        self.reads = []

    def _connect(self):
        return Connection(self)

    def get_learner_learning_timezone(self, learner):
        assert learner == LEARNER
        return self.learning_timezone

    def get_own_conversation_messages(self, learner, actor, *, limit=20, active_day_limit=None,
            learning_timezone=None, exclude_source_id=None, as_of=None):
        self.reads.append({"limit": limit, "active_day_limit": active_day_limit, "as_of": as_of})
        rows = [row for row in self.rows if row["learner_id"] == learner and row["actor_id"] == actor
                and row["source_id"] != exclude_source_id and row["created_at"] <= as_of
                and row["author_role"] in {"student", "assistant"}]
        if active_day_limit:
            days = set(sorted({row["activity_day"] for row in rows})[-active_day_limit:])
            rows = [row for row in rows if row["activity_day"] in days]
        rows.sort(key=lambda row: row["source_order"])
        return deepcopy(rows if limit is None else rows[-limit:])


def build(store, **kwargs):
    return build_conversation_memory(store, LEARNER, ACTOR, as_of=NOW + timedelta(hours=1), **kwargs)


def test_empty_history_is_available_without_approval_and_does_not_write_empty_versions():
    store = Store([])
    result = build(store)
    assert result["method"] == "conversation_history"
    assert not result["available"]
    assert result["system_text"] == ""
    assert result["recent_messages"] == []
    assert result["last_seven_active_days"]["count"] == 0
    assert store.versions == []


def test_calendar_day_merges_topics_and_sessions_and_preserves_role_provenance():
    store = Store([message(1, text="Я придумал корабль.", topic="math"),
                   message(2, role="assistant", text="Как ты его назвал?"),
                   message(3, session="session-b", topic="reading")])
    result = build(store)
    assert result["last_day"]["day"] == "2026-11-07"
    assert result["last_day"]["source_count"] == 3
    assert result["last_seven_active_days"]["count"] == 1
    assert {item["usage"] for item in result["recent_messages"]} == {"self_report", "context_only"}
    assert [item["source_id"] for item in result["last_day"]["facts"]] == [str(UUID(int=1)), str(UUID(int=3))]
    assert result["last_day"]["assistant_context"]["usage"] == "context_only"


def test_seven_last_active_days_can_span_two_months_and_include_active_today():
    rows = [message(i + 1, created_at=NOW - timedelta(days=(7 - i) * 8)) for i in range(8)]
    result = build(Store(rows), context_budget_bytes=20000)
    days = result["last_seven_active_days"]["days"]
    assert len(days) == 7
    assert days[0]["day"] == (NOW - timedelta(days=48)).date().isoformat()
    assert days[-1]["day"] == NOW.date().isoformat()
    assert str(UUID(int=1)) not in result["last_seven_active_days"]["text"]
    assert result["last_seven_active_days"]["source_count"] == 7


def test_recent_window_counts_twenty_individual_messages():
    rows = [message(i + 1, role="user" if i % 2 == 0 else "assistant", text=str(i)) for i in range(44)]
    result = build(Store(rows), context_budget_bytes=10000)
    assert len(result["recent_messages"]) == 20
    assert result["recent_messages"][0]["source_id"] == str(UUID(int=25))
    assert result["last_day"]["source_count"] == 44
    assert result["budget"]["available_recent_message_count"] == 20


def test_current_source_exclusion_does_not_drop_previous_failed_words():
    store = Store([message(1, text="Мне нравятся корабли.", status="failed"),
                   message(2, text="Текущий вопрос", status="pending")])
    result = build(store, exclude_source_id=str(UUID(int=2)))
    assert [item["source_id"] for item in result["recent_messages"]] == [str(UUID(int=1))]
    assert "без завершённого ответа" in result["system_text"]
    assert "Текущий вопрос" not in result["system_text"]


def test_owned_reader_excludes_other_actor_learner_and_teacher_notes():
    teacher = message(4, text="PRIVATE TEACHER NOTE")
    teacher["author_role"] = "teacher"
    store = Store([message(1, text="Own"), message(2, text="OTHER ACTOR", actor="someone"),
                   message(3, text="OTHER LEARNER", learner="someone"), teacher])
    result = build(store)
    assert [item["text"] for item in result["recent_messages"]] == ["Own"]
    assert result["provenance"]["source_ids"] == [str(UUID(int=1))]


def test_repeated_read_reuses_day_and_rollup_versions_and_model_prefix():
    store = Store([message(1)])
    first = build(store)
    second = build_conversation_memory(store, LEARNER, ACTOR, as_of=NOW + timedelta(hours=3))
    assert len(store.versions) == 2
    assert first["last_day"]["version_id"] == second["last_day"]["version_id"]
    assert first["last_seven_active_days"]["version_id"] == second["last_seven_active_days"]["version_id"]
    assert first["system_text"] == second["system_text"]
    assert first["input_fingerprint"] == second["input_fingerprint"]
    assert first["last_day"]["version_id"] not in first["system_text"]
    assert first["recent_messages"][0]["source_id"] not in first["system_text"]
    assert "12:34" not in first["system_text"]
    assert "19 ч. Asia/Tomsk" in first["system_text"]
    assert "12:34:56" in first["recent_messages"][0]["created_at"]


def test_new_message_creates_new_immutable_versions_and_retains_original_payload():
    store = Store([message(1, text="Сокол")])
    first = build(store)
    preserved = deepcopy(store.versions)
    store.rows.append(message(2, text="Исправляю имя, теперь Искра."))
    second = build(store)
    assert len(store.versions) == 4
    assert store.versions[:2] == preserved
    assert second["last_day"]["supersedes_version_id"] == first["last_day"]["version_id"]
    assert second["last_seven_active_days"]["supersedes_version_id"] == first["last_seven_active_days"]["version_id"]
    assert second["system_text"] != first["system_text"]


def test_historical_bucket_survives_learning_timezone_preference_change():
    at = datetime(2026, 11, 7, 18, tzinfo=timezone.utc)
    store = Store([message(1, created_at=at)])
    first = build_conversation_memory(store, LEARNER, ACTOR, as_of=at)
    store.learning_timezone = "Europe/Moscow"
    second = build_conversation_memory(store, LEARNER, ACTOR, as_of=at)
    assert first["last_day"]["day"] == second["last_day"]["day"] == "2026-11-08"
    assert first["last_day"]["version_id"] == second["last_day"]["version_id"]
    assert first["system_text"] == second["system_text"]
    assert second["learning_timezone"] == "Europe/Moscow"
    assert second["last_day"]["timezone"] == "Asia/Tomsk"


def test_midnight_and_dst_hours_are_derived_from_each_stored_activity_timezone():
    rows = [message(1, created_at=datetime(2026, 11, 7, 16, 59, tzinfo=timezone.utc)),
            message(2, created_at=datetime(2026, 11, 7, 17, 0, tzinfo=timezone.utc)),
            message(3, created_at=datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc), activity_timezone="America/New_York"),
            message(4, created_at=datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc), activity_timezone="America/New_York")]
    canonical = _canonical_messages(rows, "Asia/Tomsk")
    assert canonical[0]["activity_day"] == "2026-11-07"
    assert canonical[1]["activity_day"] == "2026-11-08"
    assert canonical[2]["hour_label"] == canonical[3]["hour_label"] == "2026-11-01 01 ч. America/New_York"
    assert canonical[2]["created_at"] != canonical[3]["created_at"]


def test_summary_lexical_selection_preserves_negative_quote_without_asserting_interest():
    rows = [message(1, text="Привет"), message(2, text="Я не люблю шахматы"),
            message(3, text="4"), message(4, text="До завтра")]
    day = _day_payload("2026-11-07", _canonical_messages(rows, "Asia/Tomsk"))
    assert [item["text"] for item in day["facts"]] == ["Привет", "Я не люблю шахматы", "До завтра"]
    assert day["kind_origin"] == "literal_extraction_with_lexical_selection"
    assert "устойчивый интерес" not in day["text"]


def test_teacher_chat_is_labelled_as_its_actual_author():
    row = message(1, text="Я готовлю упражнение для ученика")
    row["author_role"] = "teacher"
    canonical = _canonical_messages([row], "Asia/Tomsk")
    day = _day_payload("2026-11-07", canonical)
    assert day["facts"][0]["author_role"] == "teacher"
    assert "Преподаватель, собственная реплика" in day["text"]
    assert "Ученик, самоотчёт" not in day["text"]


def test_shown_assistant_checkin_is_context_without_invented_child_facts():
    result = build(Store([message(1, role="assistant", text="Что интересного получилось сегодня?")]))
    assert result["last_day"]["facts"] == []
    assert result["last_day"]["assistant_context"]["usage"] == "context_only"
    assert "Помощник, контекст разговора" in result["system_text"]
    assert "Ученик, самоотчёт" not in result["system_text"]


def test_compact_rollup_keeps_seven_verbose_days_with_default_budget():
    days = []
    for offset in range(7):
        rows = [message(offset * 4 + index + 1, text="Я люблю придумывать истории и персонажей. " * 7,
                        created_at=NOW - timedelta(days=6 - offset),
                        role="assistant" if index == 3 else "user") for index in range(4)]
        canonical = _canonical_messages(rows, "Asia/Tomsk")
        days.append(_day_payload(canonical[0]["activity_day"], canonical))
    text, selected, diagnostic = _model_context(days, [], 6000)
    assert diagnostic["included_day_count"] == 7
    assert diagnostic["omitted_day_count"] == 0
    assert len(text.encode("utf-8")) <= 3000


@pytest.mark.parametrize("budget", [0, 50, 800, 2000, 6000])
def test_full_content_utf8_budget_omits_whole_messages_and_reports_counts(budget):
    recent = _canonical_messages([message(1, text="А" * 1600), message(2, text="Б" * 1600)], "Asia/Tomsk")
    days = [_day_payload("2026-11-07", recent)]
    text, selected, diagnostic = _model_context(days, recent, budget)
    used = len(text.encode("utf-8")) + sum(len(item["content"].encode("utf-8")) for item in selected)
    assert used <= budget
    assert diagnostic["used"] == used
    assert diagnostic["included_recent_message_count"] + diagnostic["omitted_recent_message_count"] == 2
    assert all(len(item["content"]) == 1600 for item in selected)


def test_naive_timestamps_and_invalid_messages_fail_closed():
    with pytest.raises(ValueError, match="include a timezone"):
        _canonical_messages([message(1, created_at=NOW.replace(tzinfo=None), activity_day=date(2026, 11, 7))], "Asia/Tomsk")
    wrong = message(1)
    wrong["role"] = "system"
    with pytest.raises(ValueError, match="invalid own conversation"):
        _canonical_messages([wrong], "Asia/Tomsk")


def test_older_snapshot_cannot_supersede_a_newer_version():
    store = Store([message(1, created_at=NOW), message(2, created_at=NOW + timedelta(minutes=5))])
    current = build(store)
    historical = build_conversation_memory(store, LEARNER, ACTOR, as_of=NOW)
    assert historical["last_day"]["supersedes_version_id"] is None
    newest_again = build(store)
    assert newest_again["last_day"]["version_id"] == current["last_day"]["version_id"]


def test_same_cutoff_is_used_for_window_and_recent_reads():
    store = Store([message(1)])
    result = build(store)
    assert result["algorithm"] == ALGORITHM
    assert store.reads[0]["as_of"] == store.reads[1]["as_of"]
    assert store.reads[0]["active_day_limit"] == 7
    assert store.reads[1]["limit"] == 20
