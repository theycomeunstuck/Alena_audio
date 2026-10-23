"""Deterministic self-guided division practice.

The protocol stores one opaque cursor and typed learner actions. It never uses
dialogue inference, retrieval, or a model, and its public projection withholds
the arithmetic trace and answer.
"""
from __future__ import annotations

import hashlib

from local_app.learning_math import division_facts
from local_app.learning_tools import _completion

VERSION = "autonomous-division-v1"
TOPIC = "math.g4.numbers.division_by_1_2_digit"
OPERATIONS = frozenset({
    "autonomous_division_start", "autonomous_division_answer",
    "autonomous_division_explain", "autonomous_division_show_step",
    "autonomous_division_pause", "autonomous_division_resume",
    "autonomous_division_next", "autonomous_division_switch",
})
KINDS = ("ordinary", "internal_zero", "terminal_zero")
MAX_ACTIONS_PER_EXAMPLE = 60
MAX_DIVIDEND = 999999
_PHASES = {"quotient_digit", "remainder", "bring_down"}
_STATUSES = {"active", "paused", "completed"}
_PLACE_NAMES = {
    1: "единиц", 10: "десятков", 100: "сотен", 1000: "тысяч",
    10000: "десятков тысяч", 100000: "сотен тысяч",
}


def validate_identifier(value: object, label: str = "Идентификатор шага") -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 180
            or not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f"{label} должен быть непустой строкой до 180 символов без управляющих знаков.")
    return value


def validate_operands(dividend: object, divisor: object) -> dict:
    facts = division_facts(dividend, divisor)
    if facts["dividend"] <= 0 or facts["quotient"] <= 0 or facts["remainder"] != 0:
        raise ValueError("Режим поддерживает только положительное деление без остатка.")
    return facts


def validate_action(value: object) -> tuple[dict, str]:
    if not isinstance(value, dict) or not isinstance(value.get("operation"), str):
        raise ValueError("Выберите учебное действие.")
    operation = value["operation"]
    if operation not in OPERATIONS:
        raise ValueError("Неизвестное действие автономного режима.")
    allowed = {"operation"}
    if operation in {"autonomous_division_start", "autonomous_division_next", "autonomous_division_switch"}:
        if operation != "autonomous_division_start":
            allowed.add("expected_state_id")
            validate_identifier(value.get("expected_state_id"))
        digits = value.get("divisor_digits", 1)
        if type(digits) is not int or digits not in (1, 2):
            raise ValueError("Выберите однозначный или двузначный делитель.")
        allowed.add("divisor_digits")
        text = "Начать пример с однозначным делителем." if digits == 1 else "Начать пример с двузначным делителем."
        if operation == "autonomous_division_next":
            text = "Перейти к следующему примеру."
        elif operation == "autonomous_division_switch":
            text = "Явно прервать пример и выбрать новый."
    else:
        allowed.add("expected_state_id")
        validate_identifier(value.get("expected_state_id"))
        if operation == "autonomous_division_answer":
            allowed.add("value")
            answer = value.get("value")
            if type(answer) is not int or not 0 <= answer <= 999999:
                raise ValueError("Ответ на шаг должен быть целым числом от 0 до 999999.")
            text = f"Мой ответ на текущий шаг: {answer}."
        else:
            text = {
                "autonomous_division_explain": "Объяснить правило текущего шага.",
                "autonomous_division_show_step": "Показать только текущий шаг.",
                "autonomous_division_pause": "Сделать паузу.",
                "autonomous_division_resume": "Продолжить сохранённый пример.",
            }[operation]
    if set(value) - allowed:
        raise ValueError("В учебном действии есть лишние поля.")
    canonical = dict(value)
    if "divisor_digits" not in canonical and operation in {
            "autonomous_division_start", "autonomous_division_next", "autonomous_division_switch"}:
        canonical["divisor_digits"] = 1
    return canonical, text


def _kind_for(state: dict | None, state_id: str) -> str:
    if state and state.get("example_kind") in KINDS:
        return KINDS[(KINDS.index(state["example_kind"]) + 1) % len(KINDS)]
    seed = hashlib.sha256(state_id.encode("utf-8")).digest()
    return KINDS[int.from_bytes(seed[:4], "big") % len(KINDS)]


def _matches_kind(quotient: int, kind: str) -> bool:
    digits = str(quotient)
    if kind == "ordinary":
        return "0" not in digits
    if kind == "internal_zero":
        return len(digits) >= 3 and "0" in digits[1:-1] and digits[-1] != "0"
    return len(digits) >= 2 and digits.endswith("0") and digits[-2] != "0"


def choose_example(*, divisor_digits: int, kind: str, seed: str) -> tuple[int, int]:
    """Select reproducibly from exact examples generated with the math kernel."""
    if type(divisor_digits) is not int or divisor_digits not in (1, 2) or kind not in KINDS:
        raise ValueError("Unsupported example selection.")
    divisors = range(1, 10) if divisor_digits == 1 else range(10, 100)
    candidates = []
    # A three digit quotient gives all three patterns a useful long-division
    # path while remaining comfortably inside the existing arithmetic kernel.
    for quotient in range(100, 1000):
        if not _matches_kind(quotient, kind):
            continue
        for divisor in divisors:
            dividend = quotient * divisor
            if dividend <= MAX_DIVIDEND:
                facts = division_facts(dividend, divisor)
                if facts["remainder"] == 0 and facts["quotient"] == quotient:
                    candidates.append((dividend, divisor))
    if not candidates:
        raise ValueError("Не удалось выбрать пример в заданных границах.")
    digest = hashlib.sha256(f"{VERSION}:{seed}:{divisor_digits}:{kind}".encode("utf-8")).digest()
    return candidates[int.from_bytes(digest[:8], "big") % len(candidates)]


def start_state(dividend: int, divisor: int, *, state_id: str, kind: str, divisor_digits: int,
                actions_used: int = 1) -> dict:
    facts = validate_operands(dividend, divisor)
    if divisor_digits not in (1, 2) or len(str(facts["divisor"])) != divisor_digits:
        raise ValueError("Делитель не соответствует выбранному числу разрядов.")
    if not _matches_kind(facts["quotient"], kind):
        raise ValueError("Частное не соответствует типу примера.")
    return {"version": VERSION, "state_id": validate_identifier(state_id),
            "dividend": facts["dividend"], "divisor": facts["divisor"],
            "step_index": 0, "phase": "quotient_digit", "status": "active",
            "example_kind": kind, "divisor_digits": divisor_digits,
            "actions_used": actions_used}


def validate_state(state: object) -> dict:
    keys = {"version", "state_id", "dividend", "divisor", "step_index", "phase", "status",
            "example_kind", "divisor_digits", "actions_used"}
    if not isinstance(state, dict) or set(state) != keys or state.get("version") != VERSION:
        raise ValueError("Сохранённое состояние автономного примера повреждено.")
    validate_identifier(state["state_id"])
    facts = validate_operands(state["dividend"], state["divisor"])
    if type(state["divisor_digits"]) is not int or state["divisor_digits"] not in (1, 2):
        raise ValueError("Неверное число разрядов делителя.")
    if len(str(state["divisor"])) != state["divisor_digits"]:
        raise ValueError("Делитель не соответствует выбранному числу разрядов.")
    if state["example_kind"] not in KINDS or not _matches_kind(facts["quotient"], state["example_kind"]):
        raise ValueError("Неверный тип сохранённого примера.")
    index = state["step_index"]
    if type(index) is not int or not 0 <= index < len(facts["long_division_steps"]):
        raise ValueError("Номер шага не соответствует примеру.")
    if state["phase"] not in _PHASES or state["status"] not in _STATUSES:
        raise ValueError("Неизвестный шаг или статус примера.")
    if state["phase"] == "bring_down" and index + 1 >= len(facts["long_division_steps"]):
        raise ValueError("После последней цифры нечего сносить.")
    if state["status"] == "completed" and (index != len(facts["long_division_steps"]) - 1
                                                or state["phase"] != "remainder"):
        raise ValueError("Завершённое состояние должно быть последним остатком.")
    if type(state["actions_used"]) is not int or not 1 <= state["actions_used"] <= MAX_ACTIONS_PER_EXAMPLE:
        raise ValueError("Неверное число действий по примеру.")
    return dict(state)


def _new_example(previous: dict | None, *, state_id: str, divisor_digits: int) -> dict:
    kind = _kind_for(previous, state_id)
    dividend, divisor = choose_example(divisor_digits=divisor_digits, kind=kind, seed=state_id)
    return start_state(dividend, divisor, state_id=state_id, kind=kind,
                       divisor_digits=divisor_digits, actions_used=1)


def transition(state: dict | None, action: dict, *, state_id: str) -> tuple[dict | None, dict]:
    """Apply one typed action to the latest reproducible cursor."""
    current = validate_state(state) if state is not None else None
    if isinstance(action, dict) and action.get("operation") == "autonomous_division_clarify":
        if set(action) != ({"operation"} if current is None else {"operation", "expected_state_id"}):
            raise ValueError("Некорректная запись уточнения.")
        if current is not None and action.get("expected_state_id") != current["state_id"]:
            raise ValueError("Состояние примера изменилось. Обновите его.")
        if current is None:
            return None, {"status": "not_assessed"}
        next_state = dict(current, state_id=validate_identifier(state_id),
                          actions_used=min(MAX_ACTIONS_PER_EXAMPLE, current["actions_used"] + 1))
        return validate_state(next_state), {"status": "not_assessed"}
    action, _ = validate_action(action)
    operation = action["operation"]
    new_id = validate_identifier(state_id)
    if current and current["state_id"] == new_id:
        raise ValueError("Нужен новый идентификатор действия.")
    if operation == "autonomous_division_start":
        if current is not None:
            raise ValueError("Для незавершённого примера выберите продолжение или явную смену.")
        return _new_example(None, state_id=new_id, divisor_digits=action["divisor_digits"]), {"status": "not_assessed"}
    if current is None:
        raise ValueError("Сначала начните пример.")
    if action.get("expected_state_id") != current["state_id"]:
        raise ValueError("Состояние примера изменилось. Обновите его.")
    if operation in {"autonomous_division_next", "autonomous_division_switch"}:
        if operation == "autonomous_division_next" and current["status"] != "completed":
            raise ValueError("Сначала завершите пример или выберите явное прерывание.")
        if operation == "autonomous_division_switch" and current["status"] == "completed":
            raise ValueError("Завершённый пример можно сменить обычным переходом.")
        return _new_example(current, state_id=new_id, divisor_digits=action["divisor_digits"]), {
            "status": "not_assessed", "interrupted": operation == "autonomous_division_switch"}
    if current["status"] == "completed":
        raise ValueError("Пример завершён. Выберите следующий пример.")
    if operation == "autonomous_division_resume":
        next_state = dict(current, state_id=new_id, status="active")
        return validate_state(next_state), {"status": "not_assessed"}
    if current["status"] == "paused":
        raise ValueError("Пример на паузе. Сначала продолжите его.")
    # Once the action budget is used, only switching to a new example is safe.
    if current["actions_used"] >= MAX_ACTIONS_PER_EXAMPLE:
        next_state = dict(current, state_id=new_id)
        return validate_state(next_state), {"status": "limit_reached"}
    facts = validate_operands(current["dividend"], current["divisor"])
    steps = facts["long_division_steps"]
    next_state = dict(current, state_id=new_id, actions_used=current["actions_used"] + 1)
    assessment = {"status": "not_assessed"}
    if operation == "autonomous_division_pause":
        next_state["status"] = "paused"
    elif operation == "autonomous_division_explain" or operation == "autonomous_division_show_step":
        pass
    elif operation == "autonomous_division_answer":
        step = steps[current["step_index"]]
        expected = (step["quotient_digit"] if current["phase"] == "quotient_digit" else
                    step["remainder"] if current["phase"] == "remainder" else
                    steps[current["step_index"] + 1]["brought_down_digit"])
        assessment = {"status": "correct" if action["value"] == expected else "incorrect"}
        if assessment["status"] == "correct":
            if current["phase"] == "quotient_digit":
                next_state["phase"] = "remainder"
            elif current["phase"] == "remainder":
                if current["step_index"] + 1 < len(steps):
                    next_state["phase"] = "bring_down"
                else:
                    next_state["status"] = "completed"
            else:
                next_state.update(step_index=current["step_index"] + 1, phase="quotient_digit")
    else:
        raise ValueError("Неизвестное действие для текущего шага.")
    return validate_state(next_state), assessment


def _current_step(state: dict) -> dict:
    facts = validate_operands(state["dividend"], state["divisor"])
    return facts["long_division_steps"][state["step_index"]]


def render_prompt(state: dict) -> str:
    current = validate_state(state)
    if current["status"] == "completed":
        return "Пример решён. Можно перейти к следующему примеру."
    if current["status"] == "paused":
        return "Пример на паузе. Можно продолжить с того же шага."
    step = _current_step(current)
    partial, divisor = step["partial_dividend"], current["divisor"]
    if current["phase"] == "quotient_digit":
        place = _PLACE_NAMES[step["quotient_place_value"]]
        return f"Сейчас делим {partial} на {divisor}. Какую цифру запишешь в разряде {place}?"
    if current["phase"] == "remainder":
        return f"Вычти {step['subtracted_product']} из {partial}. Какой остаток получился?"
    return f"Остаток этого шага – {step['remainder']}. Какую следующую цифру делимого нужно снести?"


def explain_rule(state: dict) -> str:
    current = validate_state(state)
    if current["phase"] == "quotient_digit":
        return "Правило: выбери наибольшую цифру от 0 до 9, чтобы её произведение на делитель не превышало текущее неполное делимое. Запиши эту цифру в нужный разряд частного."
    if current["phase"] == "remainder":
        return "Правило: умножь записанную цифру частного на делитель и вычти произведение из неполного делимого. Получится остаток этого шага."
    return "Правило: снеси следующую цифру исходного делимого рядом с остатком. Не объединяй остаток с уже использованными цифрами."


def show_current_step(state: dict) -> str:
    current = validate_state(state)
    step = _current_step(current)
    if current["phase"] == "quotient_digit":
        digit = step["quotient_digit"]
        return f"Текущий шаг: {current['divisor']} × {digit} = {step['subtracted_product']}. В разряде { _PLACE_NAMES[step['quotient_place_value']] } записывается {digit}."
    if current["phase"] == "remainder":
        return (f"Текущий шаг: {step['partial_dividend']} − {step['subtracted_product']} "
                f"= {step['remainder']}. Это только ответ на текущий шаг.")
    following = _current_step(dict(current, step_index=current["step_index"] + 1, phase="quotient_digit"))
    return f"Текущий шаг: нужно снести цифру {following['brought_down_digit']} из делимого."


def public_state(state: dict | None) -> dict | None:
    if state is None:
        return None
    current = validate_state(state)
    total = len(validate_operands(current["dividend"], current["divisor"])["long_division_steps"])
    return {"state_id": current["state_id"], "dividend": current["dividend"],
            "divisor": current["divisor"], "step_index": current["step_index"],
            "phase": current["phase"], "status": current["status"],
            "question_number": min(total, current["step_index"] + 1),
            "question_count": total}


def action_response(state: dict | None, assessment: dict, operation: str, request_text: str,
                    *, interrupted: bool = False) -> dict:
    if operation not in OPERATIONS | {"autonomous_division_clarify"}:
        raise ValueError("Неизвестное действие автономного режима.")
    if state is None:
        text = ("В автономном режиме я не разбираю свободные сообщения. Начни пример, введи ответ числом "
                "или выбери доступное учебное действие: объяснить правило, показать шаг, пауза или смена примера.")
        public = None
    else:
        current = validate_state(state)
        prompt = render_prompt(current)
        if operation == "autonomous_division_clarify":
            text = ("Я не могу проверить этот текст как числовой ответ. Введи одно целое число в поле ответа "
                    "или выбери подсказку. Текущий шаг сохранён. ") + prompt
        elif assessment.get("status") == "incorrect":
            text = "Это число не подходит к текущему шагу. Попробуй ещё раз. " + prompt
        elif assessment.get("status") == "correct":
            text = "Верно. " + prompt
        elif assessment.get("status") == "limit_reached":
            text = (f"Для этого примера использовано {MAX_ACTIONS_PER_EXAMPLE} действий. "
                    "Можно явно прервать его и выбрать новый пример.")
        elif operation == "autonomous_division_explain":
            text = explain_rule(current) + " " + prompt
        elif operation == "autonomous_division_show_step":
            text = show_current_step(current) + " " + prompt
        elif operation == "autonomous_division_pause":
            text = "Сохранил место остановки. " + render_prompt(current)
        elif operation == "autonomous_division_resume":
            text = "Продолжаем с сохранённого шага. " + prompt
        elif interrupted:
            text = "Незавершённый пример отмечен как прерванный. " + prompt
        elif current["status"] == "completed":
            facts = validate_operands(current["dividend"], current["divisor"])
            text = (f"Пример завершён: {current['dividend']} : {current['divisor']}. "
                    f"Можешь перейти к следующему примеру.")
            # Keep the answer out of the student response and public state.
            assert facts["remainder"] == 0
        else:
            text = prompt
        public = public_state(current)
    result = _completion(text, operation, {}, ["autonomous_division_action", "answer_withheld"])
    result["usage_scope"] = "deterministic_action_no_model_calls"
    result["learning_protocol"] = VERSION
    result["autonomous_division"] = public
    result["limit_reached"] = bool(state and state.get("actions_used", 0) >= MAX_ACTIONS_PER_EXAMPLE)
    result["autonomous_assessment"] = {"status": assessment.get("status", "not_assessed"),
        "competence_assessed": False, "independent_work": "unknown"}
    result["learning_action"] = {"operation": operation, "request_text": request_text, "version": VERSION}
    result["answer_routing"]["policy"] = VERSION
    result["learning_tool"]["version"] = VERSION
    result["learning_tool"]["facts"] = {}
    return result
