"""Preregistered finite orchestration contracts, without PostgreSQL or models."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_product_sessions_v1 import (FIXTURE, load_fixture, load_regression_fixture,
                                           make_card, require_owned_database, run)
from local_app.guided_division import start_state, transition


def test_preregistered_six_sessions_exactly_fit_the_finite_package():
    fixture, counts = load_fixture()
    assert counts == {"ordinary_messages": 24, "explicit_actions": 12, "teacher_recommendations": 2}
    assert fixture["limits"]["generation_calls"] == 160
    assert fixture["limits"]["seconds"] == 90 * 60
    assert {e["text"] for l in fixture["learners"] for s in l["sessions"] for e in s["events"] if e["kind"] == "message"} >= {"хз", "ага", "4?"}
    assert all(s["facts"] and s["allowed_results"] and s["forbidden_claims"]
               for l in fixture["learners"] for s in l["sessions"])


def test_free_conversation_requires_an_explicit_resume_before_further_actions():
    fixture, _ = load_fixture()
    state = None
    for session in fixture["learners"][0]["sessions"][:2]:
        barrier = False
        for index, event in enumerate(session["events"]):
            mid = session["code"] + str(index)
            if event["kind"] == "message":
                barrier = bool(state)
                continue
            operation = event["operation"]
            assert not barrier or operation == "division_guided_resume"
            if operation == "division_guided_start":
                state = start_state(event["dividend"], event["divisor"], state_id=mid)
            else:
                state, _ = transition(state, operation, state_id=mid, value=event.get("value"))
            barrier = False
        assert all(state[k] == v for k, v in session["expected_guided"].items())


def test_regression_is_one_dependent_bounded_iteration():
    fixture, counts = load_regression_fixture()
    assert counts == {"ordinary_messages": 8, "explicit_actions": 4, "teacher_recommendations": 0}
    assert fixture["independent_experiment"] is False
    assert fixture["max_engineering_iterations"] == 1
    assert [s["learner_code"] for s in fixture["sessions"]] == ["L01", "L02"]


@pytest.mark.parametrize("name", ["teachcopilot_local", "teachcopilot_product_sessions_123", "teachcopilot_product_sessions_" + "a" * 13,
                                  "other_teachcopilot_product_sessions_" + "a" * 12, "teachcopilot_product_sessions_" + "G" * 12])
def test_database_teardown_cannot_target_any_other_database(name):
    with pytest.raises(ValueError):
        require_owned_database(name)


def test_generated_database_name_is_accepted():
    name = "teachcopilot_product_sessions_0123456789ab"
    assert require_owned_database(name) == name


def test_dry_run_has_no_database_models_processes_or_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run performed a side effect")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    args = argparse.Namespace(execute=False, output_dir=tmp_path / "unused", budget_ledger=None,
                              preflight_only=False, keep_database=False, regression_from=None)
    result = run(args)
    assert result["status"] == "dry_run"
    assert result["model_calls"] == 0 and result["database_created"] is False
    assert result["sequence"] == ["L01S1", "L02S1", "L01S2", "L02S2", "L01S3", "L02S3"]
    assert not list(tmp_path.iterdir())


def test_fixture_mutation_cannot_expand_ordinary_message_budget(tmp_path):
    fixture, _ = load_fixture()
    fixture["learners"][0]["sessions"][0]["events"].append({"kind": "message", "text": "unexpected"})
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="finite counts"):
        load_fixture(path)


def test_cards_are_synthetic_empty_and_do_not_copy_learning_history():
    fixture, _ = load_fixture()
    first, second = [make_card(learner) for learner in fixture["learners"]]
    assert first["learner_id"] != second["learner_id"]
    for card in (first, second):
        assert card["knowledge"] == card["interests"] == card["error_patterns"] == []
        assert all(value == [] for value in card["mvp"].values())
        assert "legal_name" not in card and "learner_model" not in card
        assert card["profile"]["autonomy_level"] == "Условия помощи неизвестны"


def test_synthetic_cards_pass_the_actual_learner_data_contract():
    from pipeline.learner_context import load_learner_data_tools
    fixture, _ = load_fixture()
    tools = load_learner_data_tools()
    catalog = {s["topic_id"]: {"id": s["topic_id"], "grade": "4", "deprecated": False}
               for learner in fixture["learners"] for s in learner["sessions"]}
    for learner in fixture["learners"]:
        findings = tools.validate_card(make_card(learner), catalog, learner["learner_id"] + ".json")
        assert not [finding.format() for finding in findings if finding.is_error()]


def test_preregistered_sources_use_requested_dash():
    for path in (FIXTURE, Path(__file__).parents[1] / "tests/fixtures/product_sessions_v1_regression.json"):
        assert "\u2014" not in path.read_text(encoding="utf-8")
