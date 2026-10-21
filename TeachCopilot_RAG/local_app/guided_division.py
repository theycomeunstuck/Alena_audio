"""An explicit, bounded division exercise with no model or text inference.

Only a typed learner answer can advance the cursor. This module checks the
arithmetic coordinates, not their provenance: the storage layer must verify
the saved action chain, learner, session and review before resuming a state.
The serialized state deliberately contains no expected answer or full trace.
"""
from __future__ import annotations

from local_app.learning_math import division_facts
from local_app.learning_tools import _completion


GUIDED_VERSION = 'guided-division-v1'
VERSION = GUIDED_VERSION
GUIDED_OPERATIONS = frozenset({
    'division_guided_start', 'division_guided_answer', 'division_guided_hint',
    'division_guided_pause', 'division_guided_resume',
})
_STATE_KEYS = {'version', 'state_id', 'dividend', 'divisor', 'step_index', 'phase', 'status'}
_PHASES = {'quotient_digit', 'remainder', 'bring_down'}
_STATUSES = {'active', 'paused', 'completed'}
_PLACE_NAMES = {1: 'единиц', 10: 'десятков', 100: 'сотен', 1000: 'тысяч',
                10000: 'десятков тысяч', 100000: 'сотен тысяч'}


def validate_identifier(value: object, label: str = 'Идентификатор шага') -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= 180
            or not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f'{label} должен быть непустой строкой до 180 символов без управляющих знаков.')
    return value


def validate_operands(dividend: object, divisor: object) -> dict:
    """Return internal facts only for positive, exact division in our bounds."""
    facts = division_facts(dividend, divisor)
    if facts['dividend'] <= 0 or facts['quotient'] <= 0 or facts['remainder'] != 0:
        raise ValueError('Пошаговый режим поддерживает только положительное деление без остатка.')
    return facts


def validate_answer(value: object) -> int:
    if type(value) is not int or not 0 <= value <= 999999:
        raise ValueError('Ответ шага должен быть целым числом от 0 до 999999.')
    return value


def start_state(dividend: int, divisor: int, *, state_id: str) -> dict:
    facts = validate_operands(dividend, divisor)
    return {
        'version': GUIDED_VERSION, 'state_id': validate_identifier(state_id),
        'dividend': facts['dividend'], 'divisor': facts['divisor'],
        'step_index': 0, 'phase': 'quotient_digit', 'status': 'active',
    }


def validate_state(state: object) -> dict:
    if not isinstance(state, dict) or set(state) != _STATE_KEYS:
        raise ValueError('Сохранённое состояние шага имеет неверные поля.')
    if state['version'] != GUIDED_VERSION:
        raise ValueError('Версия сохранённого шага не поддерживается.')
    validate_identifier(state['state_id'])
    steps = validate_operands(state['dividend'], state['divisor'])['long_division_steps']
    index = state['step_index']
    if type(index) is not int or not 0 <= index < len(steps):
        raise ValueError('Номер сохранённого шага не соответствует примеру.')
    if not isinstance(state['phase'], str) or state['phase'] not in _PHASES:
        raise ValueError('Неизвестная фаза сохранённого шага.')
    if not isinstance(state['status'], str) or state['status'] not in _STATUSES:
        raise ValueError('Неизвестное состояние занятия с примером.')
    if state['phase'] == 'bring_down' and index + 1 >= len(steps):
        raise ValueError('После последней цифры нечего сносить.')
    if state['status'] == 'completed' and (index != len(steps) - 1 or state['phase'] != 'remainder'):
        raise ValueError('Завершённое состояние должно быть последним остатком.')
    # All values are immutable scalars. A new dict also prevents caller mutation.
    return dict(state)


def _assessment(status: str = 'not_assessed') -> dict:
    return {'status': status, 'competence_assessed': False, 'independent_work': 'unknown'}


def transition(state: dict, operation: str, *, state_id: str, value: int | None = None) -> tuple[dict, dict]:
    current = validate_state(state)
    if operation not in GUIDED_OPERATIONS - {'division_guided_start'}:
        raise ValueError('Неизвестное действие с сохранённым шагом.')
    new_id = validate_identifier(state_id)
    if new_id == current['state_id']:
        raise ValueError('Новому действию требуется новый идентификатор шага.')
    if operation == 'division_guided_answer':
        validate_answer(value)
    elif value is not None:
        raise ValueError('Это действие не принимает числовой ответ.')
    if current['status'] == 'completed':
        raise ValueError('Этот пример завершён. Начните новый пример.')
    if operation != 'division_guided_resume' and current['status'] != 'active':
        raise ValueError('Пример приостановлен. Сначала явно продолжите его.')

    next_state = dict(current, state_id=new_id)
    result = _assessment()
    if operation == 'division_guided_pause':
        next_state['status'] = 'paused'
    elif operation == 'division_guided_resume':
        next_state['status'] = 'active'
    elif operation == 'division_guided_answer':
        steps = validate_operands(current['dividend'], current['divisor'])['long_division_steps']
        index, phase = current['step_index'], current['phase']
        step = steps[index]
        expected = (step['quotient_digit'] if phase == 'quotient_digit' else
                    step['remainder'] if phase == 'remainder' else
                    steps[index + 1]['brought_down_digit'])
        if value != expected:
            result = _assessment('incorrect')
        else:
            result = _assessment('correct')
            if phase == 'quotient_digit':
                next_state['phase'] = 'remainder'
            elif phase == 'remainder':
                if index + 1 < len(steps):
                    next_state['phase'] = 'bring_down'
                else:
                    next_state['status'] = 'completed'
            else:
                next_state.update(step_index=index + 1, phase='quotient_digit')
    return validate_state(next_state), result


def render_prompt(state: dict, *, hint: bool = False) -> str:
    current = validate_state(state)
    if type(hint) is not bool:
        raise ValueError('Признак подсказки должен быть логическим.')
    if current['status'] == 'completed':
        return 'Все шаги этого примера завершены. Можно начать новый пример.'
    if current['status'] == 'paused':
        return 'Пример приостановлен. Когда захочешь, можно продолжить с того же шага.'
    steps = validate_operands(current['dividend'], current['divisor'])['long_division_steps']
    step = steps[current['step_index']]
    partial, divisor = step['partial_dividend'], current['divisor']
    if current['phase'] == 'quotient_digit':
        place = _PLACE_NAMES[step['quotient_place_value']]
        lead = f'Сейчас делим {partial} на {divisor}. '
        if hint:
            if partial < divisor:
                lead += f'{partial} меньше {divisor}. Этот разряд нельзя пропускать. '
            else:
                lead += f'Найди цифру, при умножении которой на {divisor} получится самое большое число, не превышающее {partial}. '
        return lead + f'Какую цифру запишешь в разряде {place}?'
    if current['phase'] == 'remainder':
        lead = f'Сейчас нужно вычесть {step["subtracted_product"]} из {partial}. '
        if hint:
            lead += 'Вычитай только эти два числа; следующую цифру пока не сносим. '
        return lead + 'Какой остаток получился?'
    lead = f'Остаток этого шага – {step["remainder"]}. Уже использованы первые {step["digit_end_index"] + 1} цифры делимого {current["dividend"]}. '
    if hint:
        lead += 'Нужна только следующая цифра исходного делимого, без объединения с остатком. '
    return lead + 'Какую цифру снесёшь следующей?'


def action_response(state: dict, assessment: dict, request_text: str, operation: str) -> dict:
    """Render public metadata without the internally computed answer or trace."""
    current = validate_state(state)
    if operation not in GUIDED_OPERATIONS:
        raise ValueError('Неизвестное пошаговое действие.')
    if (not isinstance(assessment, dict) or set(assessment) != {'status', 'competence_assessed', 'independent_work'}
            or assessment.get('status') not in {'correct', 'incorrect', 'not_assessed'}
            or assessment.get('competence_assessed') is not False or assessment.get('independent_work') != 'unknown'):
        raise ValueError('Некорректная оценка одного шага.')
    if (operation != 'division_guided_answer' and assessment['status'] != 'not_assessed'
            or operation == 'division_guided_answer' and assessment['status'] == 'not_assessed'):
        raise ValueError('Оценка не соответствует пошаговому действию.')
    if not isinstance(request_text, str) or not request_text.strip() or len(request_text) > 12000:
        raise ValueError('Нужен исходный текст явного действия.')
    text = render_prompt(current, hint=operation == 'division_guided_hint')
    if assessment['status'] == 'incorrect':
        text = 'Это число не подходит к текущему шагу. Попробуй ещё раз. ' + text
    elif assessment['status'] == 'correct':
        text = 'Этот шаг верный. ' + text
    result = _completion(text, operation, {}, ['typed_division_step', 'complete_quotient_withheld'])
    result['usage_scope'] = 'deterministic_action_no_model_calls'
    result['guided_division'] = current
    result['guided_assessment'] = dict(assessment)
    result['learning_action'] = {'operation': operation, 'request_text': request_text, 'version': GUIDED_VERSION}
    result['answer_routing']['policy'] = GUIDED_VERSION
    result['learning_tool']['version'] = GUIDED_VERSION
    return result
