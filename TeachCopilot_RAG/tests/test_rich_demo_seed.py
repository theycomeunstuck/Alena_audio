"""Validation and opt-in disposable-PostgreSQL tests for synthetic demo records."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.seed_rich_demo import FIXTURES, LEARNER_DATA, _validator, cards_from_fixtures, seed_demo, validate_demo_cards
from local_app.auth import verify_password
from local_app.store import Store

DSN = os.getenv("TEACHCOPILOT_TEST_DSN")


@pytest.fixture(scope="module")
def store():
    if not DSN:
        pytest.skip("set TEACHCOPILOT_TEST_DSN to an expendable local database")
    instance = Store(DSN)
    instance.migrate()
    instance.import_repository(LEARNER_DATA.parent)
    return instance


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

    learner_id = "demo-luna-03"
    edited = store.get_card(learner_id)
    edited["profile"]["pace"] = "manually edited demo profile"
    store.save_card(edited, str(teacher["id"]), expected_revision=1)
    third = seed_demo(store, FIXTURES, credentials_file)
    assert third["conflicts"] == [f"learner card was edited: {learner_id}"]
    assert store.get_card(learner_id)["profile"]["pace"] == "manually edited demo profile"
