"""Offline checks of the finite runner's projection and mechanical diagnostics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.run_verified_learning_tools_v1 import fixture_cases, fixture_path, mechanical_diagnostic
from scripts.replay_verified_tool_selection import validate_saved_resolution


FIXTURE = Path(__file__).parent / "fixtures" / "verified_learning_tools_v1.json"
COUNTS = {"generation_calls": 1, "scope_resolution_calls": 1,
          "answer_generation_calls": 0, "embedding_calls": 0}


def response(operation, answer, facts):
    return {
        "choices": [{"message": {"content": answer}}],
        "answer_routing": {"backend": "deterministic_local"},
        "learning_tool": {"operation": operation, "facts": facts,
                          "verification_scope": ["bounded_fixture_check"]},
    }


def test_public_projection_excludes_private_fields_and_is_a_separate_object():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    prepared = fixture_cases(fixture)
    case, public = prepared[0]
    assert set(public) == {"messages", "subject_id", "topic_id"}
    assert "private_gold" not in public and "case_id" not in public and "category" not in public
    public["messages"][0]["content"] = "changed only in a detached test projection"
    assert case["public"]["messages"][0]["content"] != public["messages"][0]["content"]


def test_extra_public_input_or_invalid_fixed_gold_is_rejected_before_a_call():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    extra = deepcopy(fixture)
    extra["cases"][0]["public"]["expected_route"] = "division_check"
    with pytest.raises(ValueError, match="public fields"):
        fixture_cases(extra)
    fixture["cases"][0]["private_gold"]["mechanical_checks"][0]["quotient"] += 1
    with pytest.raises(ValueError, match="arithmetically inconsistent"):
        fixture_cases(fixture)


def test_exact_stdout_and_rendered_block_are_separate_from_route_and_prose():
    actual = response("print_output", "Проверенный вывод:\n\n```text\nДом\n```\n\nWrong prose is not graded here.",
                      {"stdout": "Дом\n"})
    gold = {"expected_route": "print_output", "mechanical_checks": [{"kind": "python_stdout", "stdout": "Дом\n"}]}
    diagnostic = mechanical_diagnostic(actual, {"in_scope": True}, gold, COUNTS)
    assert diagnostic["route_matches_fixed_expectation"]
    assert diagnostic["checks"][0]["exact_stdout_matches_gold"]
    assert diagnostic["checks"][0]["display_block_matches_expected_characters"]
    actual["learning_tool"]["facts"]["stdout"] = "'Дом'\n"
    wrong = mechanical_diagnostic(actual, {"in_scope": True}, gold, COUNTS)
    assert not wrong["checks"][0]["exact_stdout_matches_gold"]


def test_practice_checks_visible_operands_independently_and_never_records_answer():
    gold = {"expected_route": "division_practice", "mechanical_checks": [{
        "kind": "proposed_division_property", "divisor_min": 2, "divisor_max": 9,
        "quotient_digits": 3, "middle_digit": 0, "last_digit_nonzero": True}]}
    actual = response("division_practice", "Попробуй разделить 1212 на 4.", {"dividend": 1212, "divisor": 4})
    diagnostic = mechanical_diagnostic(actual, {"in_scope": True}, gold, COUNTS)
    check = diagnostic["checks"][0]
    assert check["visible_operands_have_requested_property"]
    assert check["numeric_quotient_not_in_student_text"]
    assert check["numeric_quotient_not_in_tool_metadata"]
    assert check["private_derived_answer_fields_absent"]
    assert "303" not in json.dumps(diagnostic, ensure_ascii=False)
    actual["learning_tool"]["facts"]["quotient"] = 303
    leaked = mechanical_diagnostic(actual, {"in_scope": True}, gold, COUNTS)["checks"][0]
    assert not leaked["numeric_quotient_not_in_tool_metadata"]
    assert not leaked["private_derived_answer_fields_absent"]


def test_unsupported_boundary_has_no_automatic_semantic_grade():
    actual = response("print_output", "7", {"stdout": "7\n"})
    gold = {"expected_route": "normal_model", "mechanical_checks": [{"kind": "unsupported_syntax"}]}
    diagnostic = mechanical_diagnostic(actual, {"in_scope": True}, gold, COUNTS)
    assert not diagnostic["route_matches_fixed_expectation"]
    assert diagnostic["checks"][0]["automatically_assessed"] is False


def test_explicit_eight_case_fixture_is_strict_and_project_relative():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["cases"] = fixture["cases"][:8]
    assert len(fixture_cases(fixture, 8)) == 8
    with pytest.raises(ValueError, match="exactly 24"):
        fixture_cases(fixture)
    with pytest.raises(ValueError, match="8 or 24"):
        fixture_cases(fixture, 7)
    assert fixture_path(Path("tests/fixtures/verified_learning_tools_v1.json")) == FIXTURE.resolve()
    with pytest.raises(ValueError, match="project-relative"):
        fixture_path(FIXTURE.resolve())
    with pytest.raises(ValueError, match="inside the project"):
        fixture_path(Path("../fixture.json"))


def test_saved_scope_replay_requires_an_actual_learner_anchor_before_selection():
    public = {"subject_id": "python", "topic_id": "programming.python.output", "messages": [
        {"role": "user", "content": "What is print output?"},
        {"role": "assistant", "content": "Assistant-only code is not learner evidence."},
        {"role": "user", "content": "Please return to my question."},
    ]}
    decision = {"in_scope": True, "uses_prior_context": True,
                "search_query": "literal print output", "prior_learner_question_index": 0}
    assert validate_saved_resolution(decision, public) == decision
    decision["prior_learner_question_index"] = 1
    with pytest.raises(ValueError, match="visible_prior_learner"):
        validate_saved_resolution(decision, public)
    del decision["prior_learner_question_index"]
    with pytest.raises(ValueError, match="explicit_learner_anchor"):
        validate_saved_resolution(decision, public)
    with pytest.raises(ValueError, match="absent_or_malformed"):
        validate_saved_resolution(None, public)
