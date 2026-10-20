"""Bounded arithmetic context for a model, never a learner assessment.

The caller supplies only the literal anchors it has already selected. This
module does not search dialogue, model rewrites, materials, or learner memory.
"""
from __future__ import annotations

import re
from typing import Any

from local_app.learning_math import division_facts


VERSION = "checked-division-first-group-v1"
DIVISION_TOPIC = "math.g4.numbers.division_by_1_2_digit"
CHAT_CONTEXT_INSTRUCTIONS = (
    "checked_division_context содержит локально проверенные числовые факты только о первой "
    "группе разрядов в данном делении. digits – её начальные цифры, amount – количество "
    "единиц разряда place_name, place_value – значение одной такой единицы, real_value – "
    "полное значение группы. Это не выбор текущего "
    "этапа решения: не повторяй уже пройденный шаг, учитывай текущую реплику и диалог. "
    "Если первая группа уже обработана и текущая частичная группа меньше делителя, "
    "в этом разряде нужен ноль перед "
    "сносом следующей цифры; не занимай из соседнего разряда. Уважай просьбу о паузе. "
    "Эти факты не доказывают, что ребёнок выполнил действие, ответил самостоятельно или "
    "освоил тему; короткое число, «ага» и «хз» сами по себе этого не доказывают. "
    "При просьбе о подсказке предложи один уместный шаг или вопрос без полного решения "
    "и без раскрытия частного. Если этап неясен, сначала уточни его. Не дополняй проверенные "
    "факты выдуманными разрядами, оценкой ответа или предположением о незавершённости задания."
)

_MAX_TEXT = 4000
_MAX_ANCHORS = 20
_NUMBER = r"(?:[-+−–]\s*)?[0-9]+(?:[.,][0-9]+)?"
_TOKEN_END = r"(?!\w|[.,][0-9])"
_SYMBOL = re.compile(rf"(?<!\w)(?P<a>{_NUMBER})\s*(?P<op>[:÷/])\s*(?P<b>{_NUMBER}){_TOKEN_END}")
_VERBAL = re.compile(
    rf"\b(?:раздел(?:и|ить|им)|подел(?:и|ить|им)|делени[ея])\s+"
    rf"(?P<a>{_NUMBER})\s+на\s+(?P<b>{_NUMBER}){_TOKEN_END}", re.I)
_DIVISION_WORD = re.compile(r"\b(?:раздел\w*|подел\w*|делени\w*)\b", re.I)
_OPERAND_REFERENCE = re.compile(r"\b(?:делител\w*|делим(?:ое|ого|ому|ым|ом))\b", re.I)
_FRACTION_WORD = re.compile(r"\b(?:дроб\w*|числител\w*|знаменател\w*|сократ\w*|сокращ\w*)\b", re.I)
_NEGATIVE_WORD = re.compile(r"\b(?:минус|отрицательн\w*)\b", re.I)
_COMPOUND_WORD = re.compile(r"\b(?:плюс|прибав\w*|вычт\w*|вычит\w*|умножь|умножить|возвед\w*|степен\w*)\b", re.I)
_MATH_ATTEMPT = re.compile(r"\d\s*[:÷/]|[:÷/]\s*[-+−–]?\d")
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_EQUALITY = re.compile(r"^\s*=\s*(?:0|[1-9][0-9]*)(?![0-9]|[.,][0-9])(?=$|[\s.,!?;])")
_OUTSIDE_MATH = re.compile(r"[=+*/×÷^%<>−\\\[\](){}]|(?<![A-Za-zА-Яа-яЁё])[-–]")
_OUTSIDE_DECIMAL = re.compile(r"[0-9][.,][0-9]")
_PLACES = {1: "единиц", 10: "десятков", 100: "сотен", 1000: "тысяч",
           10000: "десятков тысяч", 100000: "сотен тысяч"}


def _explicit_attempt(text: str) -> bool:
    # Without a complete current literal, an operand reference may change the
    # old task even when the replacement is written in words. Do not guess it
    # from an older anchor. A bare numeric reply remains unassessed context.
    return bool(_MATH_ATTEMPT.search(text)
                or _OPERAND_REFERENCE.search(text)
                or (_DIVISION_WORD.search(text) and re.search(r"\d", text)))


def _division_request(text: str) -> bool:
    # A colon or division sign is an explicit operation; a generic slash can
    # describe a fraction and needs a selected division topic or clear wording.
    return bool(_DIVISION_WORD.search(text) or re.search(r"[0-9]\s*[:÷]", text))


def extract_literal_division_pair(text: str, *, division_context: bool = False) -> tuple[int, int] | None:
    """Extract one complete supported literal operation, never a prefix.

    ``division_context`` may only reflect an already selected division topic.
    It disambiguates a slash; it does not override explicit fraction wording.
    This first package supports exact positive integer division only.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_TEXT:
        return None
    if type(division_context) is not bool or _FRACTION_WORD.search(text) or _NEGATIVE_WORD.search(text):
        return None
    matches = [*list(_SYMBOL.finditer(text)), *list(_VERBAL.finditer(text))]
    if len(matches) != 1:
        return None
    match = matches[0]
    if match.re is _SYMBOL and match["op"] == "/" and not (division_context or _DIVISION_WORD.search(text)):
        return None
    left, right = match["a"].strip(), match["b"].strip()
    if not _INTEGER.fullmatch(left) or not _INTEGER.fullmatch(right):
        return None
    if len(left) > 6 or len(right) > 2:
        return None
    dividend, divisor = int(left), int(right)
    if not 1 <= dividend <= 999999 or not 1 <= divisor <= 99 or dividend % divisor:
        return None
    # A digit group separated by spaces is not a shorter integer operation.
    if re.search(r"[0-9]\s+$", text[:match.start()]):
        return None
    # An optional literal equality is context only, never a verified learner
    # answer. Narrative integers may describe earlier work or constraints; the
    # first-group facts neither interpret nor verify those parts of the text.
    after = _EQUALITY.sub(" ", text[match.end():], count=1)
    outside = text[:match.start()] + " " + after
    if (_OUTSIDE_MATH.search(outside) or _COMPOUND_WORD.search(outside)
            or _OUTSIDE_DECIMAL.search(outside)):
        return None
    return dividend, divisor


def _context(dividend: int, divisor: int) -> dict[str, Any]:
    facts = division_facts(dividend, divisor)
    if dividend < 1 or facts["remainder"] or facts["quotient"] < 1:
        raise ValueError("Checked division context requires exact positive integer division")
    first = facts["long_division_steps"][0]
    place = first["quotient_place_value"]
    return {"version": VERSION, "dividend": dividend, "divisor": divisor,
            "first_group": {"digits": first["consumed_digits"], "place_value": place,
                            "place_name": _PLACES[place], "amount": first["partial_dividend"],
                            "real_value": first["partial_dividend"] * place},
            "complete_answer_included": False, "student_work_assessed": False}


def checked_division_context(current_message: str, subject_id: str | None,
                             topic_id: str | None, *, anchors=()) -> dict[str, Any] | None:
    """Return first-group facts for the current task or explicit active anchors.

    A current expression wins over older anchors. An unsupported current
    expression blocks fallback, so changing the task cannot revive an old one.
    Conflicting or malformed mathematical anchors do not produce checked facts.
    """
    if (not isinstance(current_message, str) or len(current_message) > _MAX_TEXT
            or subject_id not in (None, "math") or not (topic_id is None or isinstance(topic_id, str))):
        return None
    selected_division = topic_id == DIVISION_TOPIC
    pair = extract_literal_division_pair(current_message, division_context=selected_division)
    if pair is not None:
        return _context(*pair) if selected_division or (subject_id == "math" and _division_request(current_message)) else None
    if _explicit_attempt(current_message):
        return None
    if (not isinstance(anchors, (tuple, list)) or len(anchors) > _MAX_ANCHORS
            or any(not isinstance(text, str) or len(text) > _MAX_TEXT for text in anchors)):
        return None
    pairs = set()
    for text in anchors:
        pair = extract_literal_division_pair(text, division_context=selected_division)
        if pair is None:
            if _explicit_attempt(text):
                return None
            continue
        if not (selected_division or (subject_id == "math" and _division_request(text))):
            return None
        pairs.add(pair)
    return _context(*next(iter(pairs))) if len(pairs) == 1 else None


def validate_checked_division_context(value: Any) -> dict[str, Any]:
    """Validate strict keys/types and return a freshly recomputed mapping."""
    if type(value) is not dict or set(value) != {
        "version", "dividend", "divisor", "first_group", "complete_answer_included", "student_work_assessed"
    }:
        raise ValueError("Invalid checked division context fields")
    if (type(value["version"]) is not str or value["version"] != VERSION or type(value["dividend"]) is not int
            or type(value["divisor"]) is not int or value["complete_answer_included"] is not False
            or value["student_work_assessed"] is not False):
        raise ValueError("Invalid checked division context types or disclosure")
    first = value["first_group"]
    if (type(first) is not dict or set(first) != {"digits", "place_value", "place_name", "amount", "real_value"}
            or type(first["digits"]) is not str or type(first["place_name"]) is not str
            or any(type(first[key]) is not int for key in ("place_value", "amount", "real_value"))):
        raise ValueError("Invalid checked division first-group fields or types")
    expected = _context(value["dividend"], value["divisor"])
    if value != expected:
        raise ValueError("Checked division facts do not match recomputed operands")
    return expected
