"""Bounded, deterministic arithmetic facts for educational responses.

This module never interprets learner text. Callers decide whether a complete
answer is appropriate before displaying either formatter. Digit indices are
zero based and refer to the ordinary decimal spelling of the dividend.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


_PLACE_NAMES = {
    1: "единиц",
    10: "десятков",
    100: "сотен",
    1000: "тысяч",
    10000: "десятков тысяч",
    100000: "сотен тысяч",
}


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def division_facts(dividend: int, divisor: int) -> dict[str, Any]:
    """Return exact Euclidean division and a position preserving school trace.

    The first step consumes the shortest prefix at least as large as the
    divisor, or the entire dividend if no such prefix exists. Its
    ``brought_down_digit`` is ``None`` because no previous remainder exists.
    Every subsequent step consumes exactly one original digit, including any
    step whose quotient digit is zero. ``digit_end_index`` is inclusive.
    """
    dividend = _integer(dividend, "dividend", 0, 999999)
    divisor = _integer(divisor, "divisor", 1, 99)
    quotient, remainder = divmod(dividend, divisor)
    digits = str(dividend)

    prefix_end = 0
    while prefix_end < len(digits) - 1 and int(digits[: prefix_end + 1]) < divisor:
        prefix_end += 1

    steps: list[dict[str, Any]] = []
    previous_remainder = 0
    for end_index in range(prefix_end, len(digits)):
        first = not steps
        partial = (
            int(digits[: end_index + 1])
            if first
            else previous_remainder * 10 + int(digits[end_index])
        )
        quotient_digit, step_remainder = divmod(partial, divisor)
        steps.append(
            {
                "digit_start_index": 0 if first else end_index,
                "digit_end_index": end_index,
                "consumed_digits": digits[: end_index + 1] if first else digits[end_index],
                "partial_dividend": partial,
                "quotient_digit": quotient_digit,
                "quotient_place_value": 10 ** (len(digits) - end_index - 1),
                "quotient_digit_index": len(steps),
                "subtracted_product": quotient_digit * divisor,
                "remainder": step_remainder,
                "brought_down_digit": None if first else int(digits[end_index]),
                "brought_down_digit_index": None if first else end_index,
            }
        )
        previous_remainder = step_remainder

    quotient_digits = str(quotient)
    places = []
    for index, digit_text in enumerate(quotient_digits):
        place_value = 10 ** (len(quotient_digits) - index - 1)
        digit = int(digit_text)
        places.append(
            {
                "digit": digit,
                "place_name": _PLACE_NAMES[place_value],
                "place_value": place_value,
                "value": digit * place_value,
                "quotient_digit_index": index,
            }
        )

    product = quotient * divisor
    return {
        "dividend": dividend,
        "divisor": divisor,
        "quotient": quotient,
        "remainder": remainder,
        "integer": remainder == 0,
        "product_check": {
            "quotient": quotient,
            "divisor": divisor,
            "product": product,
            "remainder": remainder,
            "reconstructed_dividend": product + remainder,
            "valid": product + remainder == dividend and 0 <= remainder < divisor,
        },
        "quotient_places": places,
        "long_division_steps": steps,
    }


def format_verified_division(facts: dict[str, Any], mode: str = "check") -> str:
    """Render a full answer only after a caller has chosen to reveal it.

    Recompute from the operands, so changed derived fields cannot introduce
    false arithmetic into the deterministic answer.
    """
    if mode not in {"check", "places"}:
        raise ValueError("mode must be check or places")
    verified = division_facts(facts["dividend"], facts["divisor"])
    dividend = verified["dividend"]
    divisor = verified["divisor"]
    quotient = verified["quotient"]
    remainder = verified["remainder"]
    equation = f"{dividend} : {divisor} = {quotient}"
    if remainder:
        equation += f" (остаток {remainder})"
    if mode == "places":
        places = " ".join(
            f"Цифра {place['digit']} стоит в разряде {place['place_name']}: "
            f"{place['digit']} × {place['place_value']} = {place['value']}."
            for place in verified["quotient_places"]
        )
        return f"{equation}. {places}"
    check = f"Проверка: {quotient} × {divisor}"
    if remainder:
        check += f" + {remainder} = {dividend}; {remainder} < {divisor}."
    else:
        check += f" = {dividend}."
    return f"{equation}. {check}"


def practice_division(
    *,
    divisor_digits: int | None = None,
    exclude_pairs: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Choose a new exact example with a three digit quotient of form a0b.

    The finite scan is ordered by quotient, then divisor; a and b are nonzero.
    Only ``student_prompt`` is intended for immediate display. ``facts``
    contains the answer for later verification and must stay out of a practice
    question until the learner requests a check or a complete solution.
    """
    if divisor_digits is not None:
        divisor_digits = _integer(divisor_digits, "divisor_digits", 1, 2)
    if (
        isinstance(exclude_pairs, (str, bytes))
        or not isinstance(exclude_pairs, Sequence)
        or len(exclude_pairs) > 10000
    ):
        raise ValueError("exclude_pairs must be a sequence of at most 10000 pairs")
    excluded: set[tuple[int, int]] = set()
    for pair in exclude_pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each excluded pair must contain dividend and divisor")
        excluded.add(
            (
                _integer(pair[0], "excluded dividend", 0, 999999),
                _integer(pair[1], "excluded divisor", 1, 99),
            )
        )

    divisors = range(10, 100) if divisor_digits == 2 else range(2, 10)
    if divisor_digits is None:
        divisors = range(2, 100)
    for hundreds in range(1, 10):
        for units in range(1, 10):
            quotient = hundreds * 100 + units
            for divisor in divisors:
                dividend = quotient * divisor
                if (dividend, divisor) in excluded:
                    continue
                return {
                    "dividend": dividend,
                    "divisor": divisor,
                    "student_prompt": (
                        f"Попробуй разделить {dividend} на {divisor}. "
                        "Обрати внимание, нужно ли записать ноль внутри частного."
                    ),
                    "facts": division_facts(dividend, divisor),
                }
    raise ValueError("no unused practice example is available within the bounds")
