import pytest

from local_app.autonomous_division import (
    KINDS, MAX_ACTIONS_PER_EXAMPLE, VERSION, action_response, choose_example,
    public_state, start_state, transition, validate_action, validate_operands,
)


@pytest.mark.parametrize("dividend,divisor,quotient", [
    (1, 1, 1), (3636, 12, 303), (1440, 12, 120), (999999, 1, 999999),
    (9999, 99, 101),
])
def test_supported_arithmetic_boundaries_use_the_shared_kernel(dividend, divisor, quotient):
    facts = validate_operands(dividend, divisor)
    assert facts["quotient"] == quotient
    assert facts["remainder"] == 0
    assert facts["product_check"]["reconstructed_dividend"] == dividend


@pytest.mark.parametrize("dividend,divisor", [(0, 1), (-1, 1), (10, 0), (1000000, 1), (200, 100), (17, 4)])
def test_out_of_bounds_and_non_exact_examples_are_rejected(dividend, divisor):
    with pytest.raises(ValueError):
        validate_operands(dividend, divisor)


@pytest.mark.parametrize("digits", [1, 2])
@pytest.mark.parametrize("kind", KINDS)
def test_selector_reproducibly_covers_all_requested_quotient_patterns(digits, kind):
    first = choose_example(divisor_digits=digits, kind=kind, seed="stable-action-id")
    assert choose_example(divisor_digits=digits, kind=kind, seed="stable-action-id") == first
    dividend, divisor = first
    facts = validate_operands(dividend, divisor)
    assert len(str(divisor)) == digits
    quotient = str(facts["quotient"])
    if kind == "ordinary":
        assert "0" not in quotient
    elif kind == "internal_zero":
        assert "0" in quotient[1:-1] and quotient[-1] != "0"
    else:
        assert quotient.endswith("0") and quotient[-2] != "0"


def _answer(state, value, action_id):
    return transition(state, {"operation": "autonomous_division_answer",
        "expected_state_id": state["state_id"], "value": value}, state_id=action_id)


def test_wrong_answer_help_and_show_step_never_advance_or_reveal_full_quotient():
    state = start_state(3636, 12, state_id="start", kind="internal_zero", divisor_digits=2)
    wrong, assessment = _answer(state, 4, "wrong")
    assert assessment["status"] == "incorrect"
    assert wrong["step_index"] == state["step_index"] and wrong["phase"] == state["phase"]
    shown, assessment = transition(wrong, {"operation": "autonomous_division_show_step",
        "expected_state_id": "wrong"}, state_id="shown")
    response = action_response(shown, assessment, "autonomous_division_show_step", "Покажи только текущий шаг.")
    text = response["choices"][0]["message"]["content"]
    assert "12 × 3 = 36" in text
    assert shown["step_index"] == wrong["step_index"] and shown["phase"] == wrong["phase"]
    assert response["learning_tool"]["facts"] == {}
    assert "303" not in str(response)
    assert response["autonomous_division"] == public_state(shown)
    assert response["autonomous_assessment"]["competence_assessed"] is False
    assert response["autonomous_assessment"]["independent_work"] == "unknown"
    assert "_autonomous_division_state" not in response


def test_explain_pause_resume_preserve_the_position():
    state = start_state(3636, 12, state_id="start", kind="internal_zero", divisor_digits=2)
    explained, _ = transition(state, {"operation": "autonomous_division_explain",
        "expected_state_id": "start"}, state_id="explain")
    assert (explained["step_index"], explained["phase"]) == (0, "quotient_digit")
    paused, _ = transition(explained, {"operation": "autonomous_division_pause",
        "expected_state_id": "explain"}, state_id="pause")
    assert paused["status"] == "paused"
    with pytest.raises(ValueError, match="на паузе"):
        _answer(paused, 3, "blocked")
    resumed, _ = transition(paused, {"operation": "autonomous_division_resume",
        "expected_state_id": "pause"}, state_id="resume")
    assert resumed["status"] == "active"
    assert (resumed["step_index"], resumed["phase"]) == (0, "quotient_digit")


def test_internal_zero_and_terminal_zero_are_completed_only_at_the_last_remainder():
    state = start_state(3636, 12, state_id="start", kind="internal_zero", divisor_digits=2)
    for index, value in enumerate((3, 0, 3, 0, 3, 6, 3, 0)):
        state, assessment = _answer(state, value, f"answer-{index}")
        assert assessment["status"] == "correct"
    assert state["status"] == "completed"
    assert state["phase"] == "remainder"
    assert "303" not in str(action_response(state, assessment, "autonomous_division_answer", "Мой ответ на шаг: 0."))

    terminal = start_state(1440, 12, state_id="terminal", kind="terminal_zero", divisor_digits=2)
    for index, value in enumerate((1, 2, 4, 2, 0, 0, 0, 0)):
        terminal, assessment = _answer(terminal, value, f"terminal-{index}")
    assert terminal["status"] == "completed"


def test_switch_is_explicit_and_action_limit_does_not_advance_the_cursor():
    state = start_state(3636, 12, state_id="start", kind="internal_zero", divisor_digits=2,
                        actions_used=MAX_ACTIONS_PER_EXAMPLE)
    limited, assessment = _answer(state, 3, "limit")
    assert assessment["status"] == "limit_reached"
    assert limited["step_index"] == state["step_index"] and limited["phase"] == state["phase"]
    with pytest.raises(ValueError, match="смену"):
        transition(state, {"operation": "autonomous_division_start", "divisor_digits": 1}, state_id="implicit")
    switched, metadata = transition(state, {"operation": "autonomous_division_switch",
        "expected_state_id": "start", "divisor_digits": 1}, state_id="switch")
    assert metadata["interrupted"] is True
    assert switched["status"] == "active" and switched["actions_used"] == 1


@pytest.mark.parametrize("value", ["4", "4?", "ага", "хз", True, None, -1, 1.25])
def test_answer_actions_accept_only_a_typed_integer(value):
    with pytest.raises(ValueError):
        validate_action({"operation": "autonomous_division_answer",
            "expected_state_id": "step", "value": value})


def test_each_problem_and_response_uses_a_distinct_protocol_without_private_trace():
    action, text = validate_action({"operation": "autonomous_division_start"})
    assert action == {"operation": "autonomous_division_start", "divisor_digits": 1}
    assert VERSION == "autonomous-division-v1"
    assert "двузначным" not in text
