"""Synthetic completed-session summary and authorization regressions."""
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from uuid import UUID

import pytest

import local_app.session_summaries as summaries
from local_app.session_summaries import EXCERPT_CHARS, FACT_LIMIT, _session_payload, get_or_build_summaries


NOW = datetime(2026, 11, 7, 10, tzinfo=timezone.utc)
LEARNER = "learner"
TOPIC = "math.synthetic"


def session(number, *, status="completed", topic=TOPIC, actor="student", completed_at=NOW, learner=LEARNER):
    return {"learning_session_id": str(UUID(int=number)), "learner_id": learner, "topic_id": topic,
            "actor_id": actor, "status": status, "created_at": NOW - timedelta(hours=1),
            "completed_at": completed_at}


def source(name, owner, text, *, role="student", corrects=None, order=1, learner=LEARNER,
           topic=TOPIC, approved=None, message_status="completed"):
    return {"source_id": name, "learner_id": learner, "source_session_id": owner,
            "summary_session_id": owner, "source_order": order, "topic_id": topic, "author_role": role,
            "content": text, "approved": (role == "teacher" if approved is None else approved),
            "corrects_source_id": corrects, "observed_on": None, "created_at": NOW,
            "message_status": message_status}


class FakeCursor:
    def __init__(self, store):
        self.store, self.rows = store, []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows, self.rows = self.rows, []
        return rows

    def execute(self, query, params):
        sql = " ".join(query.split())
        if sql.startswith("SELECT memory_source_revision FROM teachcopilot.learners"):
            self.store.revision_reads += 1
            if self.store.change_revision_on_read == self.store.revision_reads:
                self.store.revision += 1
            self.rows = [{"memory_source_revision": self.store.revision}] if params[0] == LEARNER else []
        elif sql.startswith("SELECT id AS learning_session_id"):
            learner, *rest = params
            limit = rest.pop()
            actor = rest.pop(0) if "actor_id=%s" in sql else None
            topic = rest.pop(0) if "topic_id=%s" in sql else None
            rows = [row for row in self.store.sessions if row["learner_id"] == learner
                    and row["status"] == "completed" and (actor is None or row["actor_id"] == actor)
                    and (topic is None or row["topic_id"] == topic)]
            rows.sort(key=lambda row: (row["completed_at"] or row["created_at"], row["learning_session_id"]), reverse=True)
            self.rows = [{**row, "completed_at": row["completed_at"] or row["created_at"]} for row in rows[:limit]]
        elif sql.startswith("WITH RECURSIVE relevant"):
            learner, session_ids = params
            candidates = [row for row in self.store.sources if row["learner_id"] == learner
                          and row["message_status"] in (None, "completed")]
            rows = [deepcopy(row) for row in candidates if row["source_session_id"] in session_ids
                    and not row["corrects_source_id"] and
                    (row["author_role"] == "student" or (row["author_role"] == "teacher" and row["approved"]))]
            pending = list(rows)
            while pending:
                parent = pending.pop()
                for candidate in candidates:
                    if (candidate["corrects_source_id"] == parent["source_id"] and candidate["author_role"] == "teacher"
                            and candidate["approved"] and candidate["topic_id"] == parent["topic_id"]):
                        child = deepcopy(candidate)
                        child["summary_session_id"] = parent["summary_session_id"]
                        rows.append(child)
                        pending.append(child)
            self.rows = rows
        elif sql.startswith("SELECT DISTINCT ON (m.learning_session_id)"):
            learner, session_ids = params
            self.rows = [row for row in self.store.endings if row["learner_id"] == learner
                         and row["learning_session_id"] in session_ids]
        elif sql.startswith("SELECT id,input_fingerprint,payload,supersedes_version_id"):
            learner, kind, session_id, scope, *fingerprint = params
            matching = [row for row in self.store.versions if row["learner_id"] == learner and row["kind"] == kind
                        and row["learning_session_id"] == session_id and row["scope_fingerprint"] == scope
                        and (not fingerprint or row["input_fingerprint"] == fingerprint[0])]
            self.rows = list(reversed(matching))[:1]
        elif sql.startswith("INSERT INTO teachcopilot.session_summary_versions"):
            learner, kind, session_id, scope, fingerprint, algorithm, payload, supersedes = params
            existing = [row for row in self.store.versions if row["learner_id"] == learner and row["kind"] == kind
                        and row["learning_session_id"] == session_id and row["scope_fingerprint"] == scope
                        and row["input_fingerprint"] == fingerprint]
            if existing:
                self.rows = []
                return
            row = {"id": str(UUID(int=1000 + len(self.store.versions))), "learner_id": learner, "kind": kind,
                   "learning_session_id": session_id, "scope_fingerprint": scope, "input_fingerprint": fingerprint,
                   "algorithm": algorithm, "payload": deepcopy(payload.adapted), "supersedes_version_id": supersedes}
            self.store.versions.append(row)
            self.rows = [row]
        else:
            raise AssertionError(f"unexpected query: {sql}")


class FakeStore:
    def __init__(self, sessions=(), sources=(), endings=()):
        self.sessions, self.sources, self.endings = list(sessions), list(sources), list(endings)
        self.versions = []
        self.revision, self.revision_reads, self.change_revision_on_read = 0, 0, None

    def _connect(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return FakeCursor(self)

def test_latest_seven_are_completed_sessions_with_stable_ties_and_dates():
    sessions = [session(i) for i in range(1, 10)]
    sessions.extend([session(20, status="active"), session(30, learner="other")])
    store = FakeStore(sessions)
    seven = get_or_build_summaries(store, LEARNER, limit=7)
    assert seven["algorithm"] == "extractive_v1" and seven["limit"] == seven["count"] == 7
    assert [row["learning_session_id"] for row in seven["sessions"]] == [str(UUID(int=i)) for i in range(9, 2, -1)]
    assert seven["date_span"] == {"start": NOW.isoformat(), "end": NOW.isoformat()}
    assert len(store.versions) == 8 and store.versions[-1]["kind"] == "rolling7"
    again = get_or_build_summaries(store, LEARNER, limit=7)
    assert again == seven and len(store.versions) == 8
    one = get_or_build_summaries(store, LEARNER, limit=1)
    assert one["count"] == 1 and one["version_id"] == seven["sessions"][0]["version_id"]


def test_new_completion_creates_new_rolling_version_without_reusing_active_turns():
    old, new = session(1), session(2, status="active")
    store = FakeStore([old, new], [source("old", old["learning_session_id"], "Старое"),
                                   source("new", new["learning_session_id"], "Новое", order=2)])
    before = get_or_build_summaries(store, LEARNER, limit=7)
    assert before["count"] == 1 and "Новое" not in before["text"]
    new["status"] = "completed"
    store.revision += 1
    after = get_or_build_summaries(store, LEARNER, limit=7)
    assert after["count"] == 2 and after["version_id"] != before["version_id"]
    assert after["supersedes_version_id"] == before["version_id"] and "Новое" in after["text"]


def test_scope_and_latest_one_never_fall_back_to_older_authorized_session():
    old, new = session(1), session(2)
    old_source = source("old-approved", old["learning_session_id"], "Старое занятие")
    secret = source("new-secret", new["learning_session_id"], "Секрет нового занятия", order=2)
    store = FakeStore([old, new], [old_source, secret])
    result = get_or_build_summaries(store, LEARNER, limit=1, topic_id=TOPIC, approved_source_ids=["old-approved"])
    assert result["count"] == 0 and result["sessions"] == [] and result["text"] == ""
    restricted = get_or_build_summaries(store, LEARNER, limit=7, topic_id=TOPIC, approved_source_ids=["old-approved"])
    assert restricted["count"] == 1 and restricted["sessions"][0]["learning_session_id"] == old["learning_session_id"]
    assert "new-secret" not in str(restricted) and "Секрет" not in str(restricted)
    teacher = get_or_build_summaries(store, LEARNER, limit=7)
    assert teacher["count"] == 2 and "Секрет" in teacher["text"]
    assert teacher["version_id"] != restricted["version_id"]


def test_unapproved_independent_sources_do_not_change_chat_output_or_fingerprint():
    lesson = session(1)
    sid = lesson["learning_session_id"]
    store = FakeStore([lesson], [source("approved", sid, "Разрешённое наблюдение")])
    before = get_or_build_summaries(store, LEARNER, topic_id=TOPIC, approved_source_ids=["approved"])
    store.sources.append(source("secret", sid, "Частная заметка", role="teacher", order=2))
    store.revision += 1
    after = get_or_build_summaries(store, LEARNER, topic_id=TOPIC, approved_source_ids=["approved"])
    assert after == before and "secret" not in str(after) and "Частная заметка" not in str(after)


def test_correction_refreshes_versions_with_roles_and_withholds_unapproved_chain():
    lesson = session(1)
    root = source("original", lesson["learning_session_id"], "Я решил сам")
    store = FakeStore([lesson], [root])
    before = get_or_build_summaries(store, LEARNER, limit=7)
    assert "Я решил сам" in before["text"]
    correction = source("correction", None, "Исправление: была помощь", role="teacher", corrects="original", order=2)
    store.sources.append(correction)
    after = get_or_build_summaries(store, LEARNER, limit=7)
    assert after["version_id"] != before["version_id"] and after["supersedes_version_id"] == before["version_id"]
    fact = after["sessions"][0]["facts"][0]
    assert "Я решил сам" not in after["text"] and "была помощь" in after["text"]
    assert fact["role"] == "teacher" and fact["source_session_id"] is None and fact["corrects_source_id"] == "original"
    assert fact["kind_origin"] == "source_link"
    assert fact["correction_chain"] == [{"source_id": "original", "role": "student", "status": "superseded"},
                                         {"source_id": "correction", "role": "teacher", "status": "current"}]
    withheld = get_or_build_summaries(store, LEARNER, topic_id=TOPIC, approved_source_ids=["original"])
    assert withheld["count"] == 0 and "correction" not in str(withheld) and "Я решил сам" not in str(withheld)
    granted = get_or_build_summaries(store, LEARNER, topic_id=TOPIC,
                                     approved_source_ids=["original", "correction"])
    assert granted["count"] == 1 and "была помощь" in granted["text"]
    assert "Я решил сам" not in granted["text"]


def test_parallel_corrections_are_not_collapsed_into_a_single_confirmed_claim():
    lesson = session(1)
    sid = lesson["learning_session_id"]
    store = FakeStore([lesson], [source("old", sid, "Исходное утверждение"),
        source("first", None, "Первое исправление", role="teacher", corrects="old", order=2),
        source("second", None, "Другое исправление", role="teacher", corrects="old", order=3)])
    result = get_or_build_summaries(store, LEARNER)
    assert {item["source_id"] for item in result["sessions"][0]["facts"]} == {"first", "second"}
    assert all(item["parallel_corrections"] for item in result["sessions"][0]["facts"])
    assert "параллельные исправления" in result["text"]
    restricted = get_or_build_summaries(store, LEARNER, topic_id=TOPIC, approved_source_ids=["old", "first"])
    assert restricted["count"] == 0


def test_selection_is_bounded_extractive_and_labels_author_and_uncertainty():
    lesson = session(1)
    sid = lesson["learning_session_id"]
    sources = [source(f"s{i}", sid, f"Обычный ответ {i}", order=i) for i in range(1, 6)]
    sources.extend([source("interest", sid, "Я люблю шахматы", order=6),
                    source("unsure", sid, "Не знаю " + "слово " * 30, order=7),
                    source("teacher", sid, "Уточнение педагога", role="teacher", order=8)])
    result = _session_payload(lesson, sources, None, None)
    assert result["selected_source_count"] == FACT_LIMIT and result["omitted_source_count"] == 5
    facts = {item["source_id"]: item for item in result["facts"]}
    assert facts["interest"]["kind"] == "interest" and facts["interest"]["role"] == "student"
    assert facts["unsure"]["kind"] == "uncertainty" and facts["unsure"]["truncated"]
    assert facts["interest"]["kind_origin"] == facts["unsure"]["kind_origin"] == "lexical_heuristic"
    assert len(facts["unsure"]["text"]) <= EXCERPT_CHARS + 1
    assert facts["teacher"]["role"] == "teacher" and result["truncated_source_count"] == 1


def test_negated_interest_and_uncertainty_are_labeled_as_automatic_mentions():
    lesson = session(1)
    sid = lesson["learning_session_id"]
    store = FakeStore([lesson], [source("no-chess", sid, "Я не люблю шахматы."),
                                source("quote", sid, "Мой друг сказал: не знаю.", order=2)])
    result = get_or_build_summaries(store, LEARNER)
    facts = {item["source_id"]: item for item in result["sessions"][0]["facts"]}
    assert facts["no-chess"]["text"] == "Я не люблю шахматы."
    assert facts["quote"]["text"] == "Мой друг сказал: не знаю."
    assert facts["no-chess"]["kind_origin"] == facts["quote"]["kind_origin"] == "lexical_heuristic"
    assert "автоматическая отметка: упоминание интереса" in result["text"]
    assert "автоматическая отметка: упоминание неуверенности" in result["text"]
    assert "[источник: no-chess]" in result["text"] and "[источник: quote]" in result["text"]


def test_format_version_change_rebuilds_cached_summary(monkeypatch):
    lesson = session(1)
    store = FakeStore([lesson], [source("mention", lesson["learning_session_id"], "Я не люблю шахматы.")])
    monkeypatch.setattr(summaries, "SUMMARY_FORMAT_VERSION", 1)
    old = get_or_build_summaries(store, LEARNER)
    monkeypatch.setattr(summaries, "SUMMARY_FORMAT_VERSION", 2)
    current = get_or_build_summaries(store, LEARNER)
    assert current["summary_format_version"] == current["sessions"][0]["summary_format_version"] == 2
    assert current["fingerprint"] != old["fingerprint"] and current["version_id"] != old["version_id"]
    assert current["supersedes_version_id"] == old["version_id"]


def test_date_observation_is_json_safe_and_matches_iso_text_fingerprint():
    lesson = session(1)
    dated = source("dated", lesson["learning_session_id"], "Наблюдение с датой")
    dated["observed_on"] = date(2026, 11, 7)
    text_dated = {**dated, "observed_on": "2026-11-07"}
    date_summary = _session_payload(lesson, [dated], None, None)
    text_summary = _session_payload(lesson, [text_dated], None, None)
    assert date_summary["facts"][0]["observed_on"] == "2026-11-07"
    assert date_summary["fingerprint"] == text_summary["fingerprint"]
    stored = get_or_build_summaries(FakeStore([lesson], [dated]), LEARNER)
    assert stored["sessions"][0]["facts"][0]["observed_on"] == "2026-11-07"


def test_non_session_sources_failed_messages_and_assistant_context_are_not_evidence():
    lesson = session(1)
    sid = lesson["learning_session_id"]
    orphan = source("old-without-session", None, "Чужая прежняя запись")
    failed = source("failed", sid, "Не завершено", message_status="failed", order=2)
    ok = source("student", sid, "Я не знаю", order=3)
    assistant = {"learner_id": LEARNER, "learning_session_id": sid, "source_id": "assistant",
                 "content": "Покажи следующий шаг"}
    store = FakeStore([lesson], [orphan, failed, ok], [assistant])
    teacher = get_or_build_summaries(store, LEARNER)
    assert [fact["source_id"] for fact in teacher["sessions"][0]["facts"]] == ["student"]
    assert teacher["sessions"][0]["ending"]["role"] == "assistant"
    assert teacher["sessions"][0]["ending"]["usage"] == "context_only"
    chat = get_or_build_summaries(store, LEARNER, topic_id=TOPIC, approved_source_ids=["student"])
    assert chat["sessions"][0]["ending"] is None and "assistant" not in str(chat)


def test_snapshot_change_retries_before_persisting_and_preserves_a_generator_grant():
    lesson = session(1)
    sid = lesson["learning_session_id"]
    store = FakeStore([lesson], [source("student", sid, "Есть наблюдение")])
    store.change_revision_on_read = 2
    result = get_or_build_summaries(store, LEARNER, topic_id=TOPIC,
                                    approved_source_ids=(value for value in ["student"]))
    assert store.revision_reads == 5 and result["count"] == 1 and len(store.versions) == 1
    assert result["sessions"][0]["facts"][0]["source_id"] == "student"


def test_actor_and_topic_filters_prevent_other_archives_from_entering_summary():
    selected, other_actor, other_topic = session(1), session(2, actor="other"), session(3, topic="other.topic")
    store = FakeStore([selected, other_actor, other_topic], [
        source("mine", selected["learning_session_id"], "Моё"),
        source("foreign", other_actor["learning_session_id"], "Другой участник"),
        source("other-topic", other_topic["learning_session_id"], "Другая тема", topic="other.topic")])
    result = get_or_build_summaries(store, LEARNER, limit=7, actor_id="student", topic_id=TOPIC)
    assert result["count"] == 1 and result["sessions"][0]["learning_session_id"] == selected["learning_session_id"]
    assert "Другой участник" not in str(result) and "Другая тема" not in str(result)


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": True}, {"approved_source_ids": ["a"]},
    {"topic_id": TOPIC, "approved_source_ids": "a"}, {"topic_id": TOPIC, "approved_source_ids": [""]},
])
def test_invalid_or_unscoped_chat_inputs_fail_closed(kwargs):
    with pytest.raises(ValueError):
        get_or_build_summaries(FakeStore(), LEARNER, **kwargs)
