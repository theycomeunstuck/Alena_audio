"""Validation and opt-in disposable-PostgreSQL tests for synthetic demo records."""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.seed_rich_demo import (
    FIXTURES, LEARNER_DATA, LIVE_DEMO_TOPICS, _prepare_demo_card, _validator,
    cards_from_fixtures, seed_demo, validate_demo_cards,
)
from local_app.auth import verify_password
from local_app.store import Store

DSN = os.getenv("TEACHCOPILOT_TEST_DSN")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if not DSN:
        pytest.skip("set TEACHCOPILOT_TEST_DSN to an expendable local database")
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn

    # The seed uses fixed demo identities and a new temporary credentials file.
    # Reusing a populated test database would make its first-run assertions fail.
    require_local_dsn(DSN)
    name = DATABASE_PREFIX + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(DSN, dbname="postgres", connect_timeout=5))
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            instance = Store(make_dsn(DSN, dbname=name))
            instance.migrate()
            repo = tmp_path_factory.mktemp("rich-demo-repository")
            shutil.copytree(LEARNER_DATA / "catalog", repo / "learner-data" / "catalog")
            instance.import_repository(repo)
            yield instance
        finally:
            # Only the exact database created above is eligible for removal.
            assert name.startswith(DATABASE_PREFIX) and len(name) == len(DATABASE_PREFIX) + 12
            with admin.cursor() as cur:
                cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
                cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
    finally:
        admin.close()


def catalog_from_files():
    catalog = {}
    for path in (LEARNER_DATA / "catalog").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        catalog.update({item["id"]: item for item in payload.get("topics", [])})
    return catalog


def test_rich_demo_fixtures_are_valid_and_cover_zpd_cases():
    cards = cards_from_fixtures(FIXTURES)
    validate_demo_cards(cards, catalog_from_files())
    validate, zpd = _validator()
    zones = {verdict.zone for card in cards for verdict in zpd.classify_card(card)}
    assert {zpd.ZONE_MASTERED, zpd.ZONE_ZPD, zpd.ZONE_OUTSIDE}.issubset(zones)
    luna = next(card for card in cards if card["learner_id"] == "demo-luna-03")
    transition = next(item for item in luna["learner_model"]["competencies"] if item["topic_id"].endswith("times_table"))
    assert transition["history"][0]["help_level"] == "joint"
    assert transition["history"][-1]["help_level"] == "independent"
    assert zpd.classify(transition).zone == zpd.ZONE_MASTERED
    unknown = next(card for card in cards if card["learner_id"] == "demo-rita-04")
    unknown_topic = "programming.python.lists"
    assert unknown_topic not in {item["topic_id"] for item in unknown["learner_model"]["competencies"]}
    assert next(item for item in unknown["knowledge"] if item["topic_id"] == unknown_topic)["status"] == "not_started"


def test_each_demo_has_explicit_fixed_date_zpd_evidence():
    _, zpd = _validator()
    expectation = json.loads((LEARNER_DATA / "demo" / "zpd_expected.json").read_text(encoding="utf-8"))
    assert expectation["as_of"] == "2026-10-13"
    expected_outcomes = {item["learner_id"]: item for item in expectation["profiles"]}
    cards = {card["learner_id"]: card for card in cards_from_fixtures()}
    for learner_id, expected in expected_outcomes.items():
        topic_id = expected["topic_id"]
        card = cards[learner_id]
        assert card["language"] == "russian"
        assert len(card["learner_model"]["competencies"]) >= 6
        assert len(card["mvp"]["journal"]) == 5
        assert len(card["mvp"]["lessons"]) == 5
        assert len(card["mvp"]["points_ledger"]) == 5
        assert len(card["learner_model"]["projects"]) == 2
        assert "completed" in {item["status"] for item in card["learner_model"]["projects"]}
        assert any(item["status"] in {"planning", "testing", "building"} for item in card["learner_model"]["projects"])
        competency = next(
            row for row in card["learner_model"]["competencies"]
            if row["topic_id"] == topic_id
        )
        verdict = zpd.confirmed_zone(competency, "2026-10-13")
        assert (verdict["zone"], verdict["evidence_status"]) == (
            expected["confirmed_zone"], expected["evidence_status"]
        )
        assert verdict["evidence"]["paired"]["tasks"] == expected["paired_tasks"]
        if expected["confirmed_zone"] == "outside":
            assisted_failures = [
                row for row in competency["history"] if row["support_level"] != "independent"
            ]
            assert len(assisted_failures) == 2
            assert all(row["support_level"] == "joint" and row["solved"] is False for row in assisted_failures)
        history = competency["history"]
        assert all(
            row.get("support_level") in {"independent", "hint_question", "visual", "joint"}
            and type(row.get("solved")) is bool
            and row.get("task_id")
            and row.get("assessment_pair_id")
            and row.get("difficulty") in {"easy", "medium", "hard"}
            and row.get("date") <= "2026-10-13"
            for row in history
        )
    for learner_id in ("demo-luna-03", "demo-oren-03"):
        competency = next(
            row for row in cards[learner_id]["learner_model"]["competencies"]
            if row["topic_id"] == expected_outcomes[learner_id]["topic_id"]
        )
        support_order = [row["support_level"] for row in competency["history"]]
        assert support_order == ["joint", "independent", "hint_question", "independent"]


def test_demo_events_and_accounts_are_distinct_and_russian():
    cards = cards_from_fixtures()
    assert len({card["learner_id"] for card in cards}) == 6
    expected_goals = {
        "demo-luna-03": "math.g3.word_problems.one_step",
        "demo-mika-03": "programming.scratch.loops",
        "demo-nika-04": "math.g4.fractions.numerator_denominator",
        "demo-ilya-04": "programming.python.variables",
        "demo-rita-04": "math.g4.quantities.area_units",
        "demo-oren-03": "math.g3.numbers.multiplication_table.division_from_table",
    }
    for card in cards:
        assert card["pseudonym"].startswith("ДЕМО – ")
        goal = card["rag_context"]["current_goal"]
        assert goal["topic_id"] == expected_goals[card["learner_id"]]
        assert "Успех" in goal["goal"]
        assert not re.search(r"\b(Practice|matched|strategy|Current evidence|Recent evidence|No structured|Do not)\b", goal["goal"], re.I)
        assert not re.search(r"\b(Practice|matched to the learner|Current evidence|Recent evidence|No structured|Do not)\b", json.dumps(card, ensure_ascii=False), re.I)
        assert len(card["learner_model"]["projects"]) == 2
        assert card["learner_model"]["projects"][0]["status"] in {"building", "planning", "testing"}
        assert card["learner_model"]["projects"][1]["status"] == "completed"
        assert sum(item["points"] for item in card["mvp"]["points_ledger"]) == card["learner_model"]["gamification"]["xp"]
        assert card["profile"]["pace"]
        assert card["learner_model"]["ai_usage"]["reflection"]


def test_recent_demo_observations_validate_and_fill_all_three_confirmed_zones(monkeypatch):
    validate, zpd = _validator()
    catalog = catalog_from_files()
    today = date(2026, 11, 22)
    refresh_day = today + timedelta(days=zpd.STALE_AFTER_DAYS)

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls.fromisoformat(refresh_day.isoformat())

    monkeypatch.setattr(zpd.datetime, "date", FixedDate)
    for fixture in cards_from_fixtures():
        learner_id = fixture["learner_id"]
        card = _prepare_demo_card(fixture, fixture, today)
        assert not validate(card, catalog, learner_id)
        rows = zpd.zone_report(card, catalog, today.isoformat())
        confirmed = {row["confirmed_zone"]: row["topic_id"] for row in rows
                     if row["evidence_status"] == "confirmed" and row["evidence"]["sufficient"]}
        assert confirmed == LIVE_DEMO_TOPICS[learner_id]
        assert Counter(row["confirmed_zone"] for row in rows) == {
            "mastered": 1, "zpd": 1, "outside": 1, "unknown": len(rows) - 3,
        }
        assert _prepare_demo_card(fixture, card, refresh_day - timedelta(days=1)) == card
        refreshed = _prepare_demo_card(fixture, card, refresh_day)
        assert refreshed != card
        for item in refreshed["learner_model"]["competencies"]:
            if item["topic_id"] in confirmed.values():
                observations = [entry for entry in item["history"]
                                if entry["task_id"].startswith("synthetic-demo:")]
                assert len(observations) == 4
                assert len({entry["task_id"] for entry in observations}) == 4
                assert all(entry["date"] == (refresh_day - timedelta(days=1)).isoformat() and "синтетическое" in entry["note"]
                           for entry in observations)


def test_seed_preserves_observations_on_topics_outside_the_demo_pairs():
    learner_id = "demo-nika-04"
    fixture = next(card for card in cards_from_fixtures() if card["learner_id"] == learner_id)
    _, zpd = _validator()
    topic = "math.g4.fractions.numerator_denominator"
    today = date.today()
    edited = zpd.record_observation(fixture, topic_id=topic, support_level="visual", solved=True)
    seeded = _prepare_demo_card(fixture, edited, today)

    def competency(card):
        return next(item for item in card["learner_model"]["competencies"] if item["topic_id"] == topic)

    assert competency(seeded) == competency(edited)
    assert _prepare_demo_card(fixture, seeded, today) == seeded


@pytest.mark.skipif(not DSN, reason="set TEACHCOPILOT_TEST_DSN to an expendable local database")
def test_seed_is_idempotent_preserves_edits_and_assigns_only_demo_roster(store, tmp_path):
    credentials_file = tmp_path / "private" / "rich-demo-credentials.json"
    first = seed_demo(store, FIXTURES, credentials_file)
    assert first["conflicts"] == []
    assert len(first["cards_created"]) == 6
    assert len(first["student_accounts_created"]) == 6
    assert first["teacher_created"] is True
    credentials = json.loads(credentials_file.read_text(encoding="utf-8"))
    teacher = store.get_account(email=credentials["teacher"]["email"])
    assert teacher["role"] == "teacher"
    assert verify_password(credentials["teacher"]["password"], teacher["password_hash"])
    for item in credentials["students"]:
        account = store.get_account(email=item["email"])
        assert account["role"] == "student"
        assert verify_password(item["password"], account["password_hash"])
        assert store.authorize_learner_access(account["id"], item["learner_id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT learner_id FROM teachcopilot.account_learners WHERE account_id=%s", (teacher["id"],))
        assert {row["learner_id"] for row in cur.fetchall()} == {card["learner_id"] for card in cards_from_fixtures()}

    second = seed_demo(store, FIXTURES, credentials_file)
    assert second["conflicts"] == []
    assert second["cards_created"] == [] and second["student_accounts_created"] == []
    assert second["cards_evidence_refreshed"] == []

    learner_id = "demo-luna-03"
    seeded = store.get_card(learner_id)
    fixture = next(card for card in cards_from_fixtures() if card["learner_id"] == learner_id)
    store.save_card(fixture, str(teacher["id"]), expected_revision=store.get_revision(learner_id))
    third = seed_demo(store, FIXTURES, credentials_file)
    assert third["conflicts"] == []
    assert third["cards_evidence_refreshed"] == [learner_id]
    assert store.get_card(learner_id) == seeded
    revision = store.get_revision(learner_id)
    assert seed_demo(store, FIXTURES, credentials_file)["cards_evidence_refreshed"] == []
    assert store.get_revision(learner_id) == revision

    edited = store.get_card(learner_id)
    edited["profile"]["pace"] = "manually edited demo profile"
    store.save_card(edited, str(teacher["id"]), expected_revision=revision)
    result = seed_demo(store, FIXTURES, credentials_file)
    assert result["conflicts"] == [f"learner card was edited: {learner_id}"]
    assert store.get_card(learner_id) == edited
