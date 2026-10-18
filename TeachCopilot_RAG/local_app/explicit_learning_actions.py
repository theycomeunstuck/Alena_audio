"""Typed, bounded learning actions. No model, retrieval or code execution."""
from __future__ import annotations

import re

from local_app.learning_math import division_facts, format_verified_division, practice_division
from local_app.learning_python import evaluate_literal_print, render_print_result
from local_app.learning_tools import _completion

VERSION = 'explicit-learning-action-v1'
MATH_OPERATIONS = {'division_check', 'division_places', 'division_practice'}


def validate_action(value: object) -> tuple[dict, str]:
    if not isinstance(value, dict):
        raise ValueError('Выберите учебное действие и заполните его поля.')
    operation = value.get('operation')
    if not isinstance(operation, str) or operation not in MATH_OPERATIONS | {'print_output'}:
        raise ValueError('Неизвестное учебное действие.')
    allowed = {'operation'}
    if operation in {'division_check', 'division_places'}:
        allowed |= {'dividend', 'divisor'}
        if operation == 'division_check':
            allowed.add('stated_quotient')
        division_facts(value.get('dividend'), value.get('divisor'))
        if 'stated_quotient' in value:
            quotient = value['stated_quotient']
            if type(quotient) is not int or not 0 <= quotient <= 999999:
                raise ValueError('Мой ответ должен быть целым числом от 0 до 999999.')
        label = 'Проверь деление' if operation == 'division_check' else 'Объясни разряды частного'
        request_text = f"{label}: {value['dividend']} : {value['divisor']}"
        if 'stated_quotient' in value:
            request_text += f" = {value['stated_quotient']}"
        request_text += '.'
    elif operation == 'division_practice':
        allowed.add('divisor_digits')
        digits = value.get('divisor_digits', 0)
        if type(digits) is not int or digits not in (0, 1, 2):
            raise ValueError('Выберите делитель: любой, однозначный или двузначный.')
        value = dict(value, divisor_digits=digits)
        width = {0: 'однозначным или двузначным', 1: 'однозначным', 2: 'двузначным'}[digits]
        request_text = f'Дай пример на деление без остатка с {width} делителем и трёхзначным частным с нулём в десятках и ненулевыми единицами. Ответ не показывай.'
    else:
        allowed.add('code')
        code = value.get('code')
        evaluate_literal_print(code)
        # Longer than every input backtick run; preserve the source verbatim.
        fences = [character * max(3, 1 + max((len(run) for run in re.findall(re.escape(character) + '+', code)), default=0)) for character in ('`', '~')]
        fence = min(fences, key=len)
        request_text = f'Покажи точный вывод Python:\n{fence}python\n{code}\n{fence}'
    if set(value) - allowed:
        raise ValueError('В действии есть лишние поля. Условия задаются только полями формы.')
    return dict(value), request_text


def validate_subject(action: dict, topic_id: str, requested: object, catalog: dict) -> None:
    subject = topic_id if topic_id in {'math', 'python', 'programming', 'scratch'} else None
    if subject is None and topic_id in catalog:
        if topic_id.startswith('math.'):
            subject = 'math'
        elif topic_id.startswith('programming.python.'):
            subject = 'python'
        elif topic_id.startswith('programming.scratch.'):
            subject = 'scratch'
    allowed = {'math'} if action['operation'] in MATH_OPERATIONS else {'python', 'programming'}
    if subject not in allowed or (requested is not None and requested not in allowed):
        raise ValueError('Действие не соответствует предмету занятия. Завершите занятие и выберите подходящий предмет.')


def action_response(action: dict, request_text: str, previous_messages=()) -> dict:
    operation = action['operation']
    if operation == 'print_output':
        result = evaluate_literal_print(action['code'])
        text, facts, scope = render_print_result(result), {'stdout': result['stdout']}, ['literal_print_stdout']
    elif operation == 'division_practice':
        # History only prevents repeated examples. It never proves competence.
        pairs = []
        for row in list(previous_messages)[-40:]:
            content = str(row.get('content', ''))[:4000]
            for match in re.finditer(r'(?<![0-9])([0-9]{1,6})\s*(?::|÷|/|на)\s*([0-9]{1,2})(?![0-9])', content):
                if 1 <= int(match[2]) <= 99:
                    pairs.append((int(match[1]), int(match[2])))
        result = practice_division(divisor_digits=action['divisor_digits'] or None, exclude_pairs=pairs[:10000])
        text = result['student_prompt']
        facts = {key: result[key] for key in ('dividend', 'divisor')}
        scope = ['exact_division_practice_properties', 'answer_withheld']
    else:
        facts = division_facts(action['dividend'], action['divisor'])
        text = format_verified_division(facts, 'places' if operation == 'division_places' else 'check')
        scope = ['integer_division', 'product_check']
        if operation == 'division_places':
            scope.append('quotient_place_values')
        if 'stated_quotient' in action:
            correct = action['stated_quotient'] == facts['quotient'] and facts['remainder'] == 0
            facts.update(stated_quotient=action['stated_quotient'], stated_equality_correct=correct)
            text = ('Результат верный. ' if correct else 'Записанное равенство неверно. ') + text
    response = _completion(text, operation, facts, scope)
    response['usage_scope'] = 'deterministic_action_no_model_calls'
    response['learning_action'] = {'operation': operation, 'request_text': request_text, 'version': VERSION}
    response['answer_routing']['policy'] = VERSION
    response['learning_tool']['version'] = VERSION
    return response
