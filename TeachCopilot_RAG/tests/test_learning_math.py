from __future__ import annotations

import re

import pytest

from local_app.learning_math import division_facts, format_verified_division, practice_division


@pytest.mark.parametrize(
    "dividend,divisor,quotient,remainder,partials,quotient_digits",
    [
        (1236, 12, 103, 0, [12, 3, 36], [1, 0, 3]),
        (2842, 14, 203, 0, [28, 4, 42], [2, 0, 3]),
        (1206, 12, 100, 6, [12, 0, 6], [1, 0, 0]),
        (1005, 5, 201, 0, [10, 0, 5], [2, 0, 1]),
        (3456, 12, 288, 0, [34, 105, 96], [2, 8, 8]),
        (12, 12, 1, 0, [12], [1]),
        (5, 12, 0, 5, [5], [0]),
        (0, 12, 0, 0, [0], [0]),
        (999999, 99, 10101, 0, [99, 9, 99, 9, 99], [1, 0, 1, 0, 1]),
        (999999, 1, 999999, 0, [9, 9, 9, 9, 9, 9], [9, 9, 9, 9, 9, 9]),
    ],
)
def test_division_table(dividend, divisor, quotient, remainder, partials, quotient_digits):
    facts = division_facts(dividend, divisor)
    assert facts["quotient"] == quotient
    assert facts["remainder"] == remainder
    assert facts["integer"] is (remainder == 0)
    assert [step["partial_dividend"] for step in facts["long_division_steps"]] == partials
    assert [step["quotient_digit"] for step in facts["long_division_steps"]] == quotient_digits


def test_trace_and_places_reconstruct_original_numbers_across_bounded_range():
    for dividend in range(256):
        for divisor in range(1, 21):
            facts = division_facts(dividend, divisor)
            digits = str(dividend)
            steps = facts["long_division_steps"]
            first_end = steps[0]["digit_end_index"]
            assert steps[0]["digit_start_index"] == 0
            assert steps[0]["consumed_digits"] == digits[: first_end + 1]
            assert steps[0]["partial_dividend"] == int(digits[: first_end + 1])
            assert steps[0]["brought_down_digit"] is None
            assert steps[0]["brought_down_digit_index"] is None
            if first_end:
                assert int(digits[:first_end]) < divisor
            assert [step["digit_end_index"] for step in steps] == list(range(first_end, len(digits)))
            previous_remainder = None
            for index, step in enumerate(steps):
                assert step["quotient_digit_index"] == index
                assert 0 <= step["quotient_digit"] <= 9
                assert 0 <= step["remainder"] < divisor
                assert step["partial_dividend"] == step["quotient_digit"] * divisor + step["remainder"]
                assert step["subtracted_product"] == step["quotient_digit"] * divisor
                assert step["quotient_place_value"] == 10 ** (len(digits) - step["digit_end_index"] - 1)
                if index:
                    digit_index = first_end + index
                    assert step["digit_start_index"] == digit_index
                    assert step["consumed_digits"] == digits[digit_index]
                    assert step["brought_down_digit_index"] == digit_index
                    assert step["brought_down_digit"] == int(digits[digit_index])
                    assert step["partial_dividend"] == previous_remainder * 10 + int(digits[digit_index])
                previous_remainder = step["remainder"]
            assert int("".join(str(step["quotient_digit"]) for step in steps)) == facts["quotient"]
            assert sum(step["quotient_digit"] * step["quotient_place_value"] for step in steps) == facts["quotient"]
            assert previous_remainder == facts["remainder"]
            assert sum(place["value"] for place in facts["quotient_places"]) == facts["quotient"]
            assert facts["product_check"]["valid"] is True
            assert facts["product_check"]["reconstructed_dividend"] == dividend


def test_internal_zero_preserves_original_digit_positions_and_place_names():
    facts = division_facts(2842, 14)
    assert [step["digit_end_index"] for step in facts["long_division_steps"]] == [1, 2, 3]
    assert [step["brought_down_digit"] for step in facts["long_division_steps"]] == [None, 4, 2]
    assert [step["quotient_place_value"] for step in facts["long_division_steps"]] == [100, 10, 1]
    assert [(item["digit"], item["place_name"], item["value"]) for item in facts["quotient_places"]] == [
        (2, "сотен", 200), (0, "десятков", 0), (3, "единиц", 3),
    ]


@pytest.mark.parametrize("dividend,divisor", [(-1, 2), (1000000, 2), (1, 0), (1, 100), (True, 2), (2, False), (2.0, 2), (2, "2")])
def test_invalid_operands_fail_without_coercion(dividend, divisor):
    with pytest.raises(ValueError):
        division_facts(dividend, divisor)


def test_check_formatter_recomputes_facts_and_handles_remainders():
    facts = division_facts(1237, 12)
    facts["quotient"] = 999
    assert format_verified_division(facts) == (
        "1237 : 12 = 103 (остаток 1). Проверка: 103 × 12 + 1 = 1237; 1 < 12."
    )
    assert format_verified_division(division_facts(0, 12)) == "0 : 12 = 0. Проверка: 0 × 12 = 0."


def test_places_formatter_does_not_call_two_hundreds_two_thousands():
    answer = format_verified_division(division_facts(2842, 14), mode="places")
    assert "Цифра 2 стоит в разряде сотен: 2 × 100 = 200." in answer
    assert "Цифра 0 стоит в разряде десятков: 0 × 10 = 0." in answer
    assert "тысяч" not in answer
    with pytest.raises(ValueError):
        format_verified_division(division_facts(1, 1), mode="unknown")


@pytest.mark.parametrize("divisor_digits", [None, 1, 2])
def test_practice_examples_have_verified_internal_zero_and_exclude_previous_pairs(divisor_digits):
    excluded = []
    for _ in range(12):
        example = practice_division(divisor_digits=divisor_digits, exclude_pairs=excluded)
        pair = example["dividend"], example["divisor"]
        assert pair not in excluded
        excluded.append(pair)
        facts = example["facts"]
        quotient_digits = str(facts["quotient"])
        assert len(quotient_digits) == 3
        assert quotient_digits[0] != "0" and quotient_digits[1] == "0" and quotient_digits[2] != "0"
        assert facts["integer"] is True
        assert facts["quotient"] * example["divisor"] == example["dividend"]
        assert "=" not in example["student_prompt"]
        assert str(facts["quotient"]) not in re.findall(r"\d+", example["student_prompt"])
        if divisor_digits:
            assert len(str(example["divisor"])) == divisor_digits


def test_practice_is_deterministic_and_exhaustion_is_explicit():
    assert practice_division() == practice_division()
    all_one_digit_pairs = [(quotient * divisor, divisor) for quotient in range(101, 910) if str(quotient)[1] == "0" and quotient % 10 for divisor in range(2, 10)]
    with pytest.raises(ValueError, match="no unused"):
        practice_division(divisor_digits=1, exclude_pairs=all_one_digit_pairs)


@pytest.mark.parametrize("kwargs", [{"divisor_digits": True}, {"divisor_digits": 3}, {"exclude_pairs": "1:2"}, {"exclude_pairs": [(1, 0)]}, {"exclude_pairs": [(True, 2)]}, {"exclude_pairs": [(1, 2, 3)]}])
def test_invalid_practice_constraints_fail(kwargs):
    with pytest.raises(ValueError):
        practice_division(**kwargs)
