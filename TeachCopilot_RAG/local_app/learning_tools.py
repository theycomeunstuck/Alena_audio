"""Narrow current-message tools, after the normal curriculum scope decision."""
from __future__ import annotations

import re

from local_app.learning_math import division_facts, format_verified_division, practice_division
from local_app.learning_python import evaluate_literal_print, extract_print_example, render_print_result


VERSION = 'verified-learning-v1'
_DIVISION = re.compile(r'(?<![\w.,+\-−])([0-9]{1,6})\s*[:÷/]\s*([0-9]{1,2})(?!\w|[.,][0-9])(?:\s*=\s*([0-9]{1,6})(?!\w|[.,][0-9]))?')
_HINT = re.compile(r'подсказ|нам[её]к|только.{0,30}(?:шаг|цифр)|не\s+(?:решай|пиши|показывай|сообщай|выдавай|раскрывай|называй)|без.{0,15}(?:ответ|решени)|оставь.{0,35}(?:мне|самому|самой)|перв\w*\s+шаг', re.I)
_CHECK_WORDS = re.compile(r'(?:проверь|проверить|проверка|пожалуйста|верно|правильно|ли|это|получилось|получил|получила|у|меня|я|в|записи|как|поясни|объясни|почему|что|означает|значит|разряд(?:ы|е|ов)?|частн(?:ое|ом|ого)|сот(?:ни|ен|нях)|десятк(?:и|ов|ах)|единиц(?:ы|ах)?|цифр(?:а|ы)?|нол[ья]|ноль|и|а|где)', re.I)
_PRACTICE_WORDS = re.compile(r'(?:дай|подбери|предложи|пожалуйста|мне|один|ещ[её]|новый|другой|похожий|пример|на|деление|с|в|частн(?:ое|ом|ым)|три|цифр(?:ы|а)|тр[её]хзначн(?:ое|ым)|нул[её]м|ноль|внутренн(?:ий|им)|внутри|посередине|середине|делител(?:ь|я|ем)|однозначн(?:ый|ым)|двузначн(?:ый|ым)|от|до|предыдущий|прошлый|был|уже|а|и)', re.I)


def _known_words(text: str, vocabulary: re.Pattern) -> bool:
    # Deliberately closed wording: an unknown constraint goes to normal chat.
    return all(vocabulary.fullmatch(word) is not None for word in re.findall(r'[A-Za-zА-Яа-яЁё]+', text))


def _completion(text: str, operation: str, facts: dict, scope: list[str]) -> dict:
    return {
        'id': None, 'object': 'chat.completion', 'created': None,
        'model': 'teachcopilot-verified-tools-v1',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
        'citations': [], 'sources': [],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        'usage_scope': 'deterministic_answer_only_excludes_scope_resolution',
        'answer_routing': {'backend': 'deterministic_local', 'reason': 'bounded_literal_operation', 'policy': VERSION},
        'learning_tool': {'version': VERSION, 'operation': operation, 'status': 'verified',
                          'verification_scope': scope, 'facts': facts,
                          'memory_used': False, 'retrieved_sources_used': False,
                          'student_competence_assessed': False},
    }


def verified_learning_response(current_message: str, subject_id: str | None = None, *, previous_messages=()) -> dict | None:
    """Return None whenever the complete narrow operation is not understood."""
    if not isinstance(current_message, str) or len(current_message) > 4000:
        return None
    if re.search(r'\bprint\s*\(', current_message):
        if subject_id not in (None, 'python', 'programming') or _HINT.search(current_message):
            return None
        if not re.search(r'что.{0,25}(?:вывед|покаж)|како[йв].{0,20}вывод|точн\w*\s+вывод', current_message, re.I):
            return None
        code = extract_print_example(current_message)
        if code is None:
            return None
        result = evaluate_literal_print(code)
        return _completion(render_print_result(result), 'print_output',
                           {'stdout': result['stdout']}, ['literal_print_stdout'])
    if subject_id not in (None, 'math'):
        return None
    expressions = list(_DIVISION.finditer(current_message))
    if (re.search(r'подбери|предложи|дай', current_message, re.I)
            and re.search(r'нов\w*\s+пример|друг\w*\s+пример|ещ[её]\s+один\s+пример', current_message, re.I)
            and re.search(r'частн', current_message, re.I)
            and re.search(r'три\s+цифр|тр[её]хзначн', current_message, re.I)
            and re.search(r'(?:внутренн\w*\s+(?:нул|нол)|(?:внутри|посередине|середине).{0,15}(?:нул|нол)|(?:нул|нол).{0,20}(?:внутри|посередине|середине))', current_message, re.I)):
        two_digits = bool(re.search(r'двузначн', current_message, re.I))
        one_digit = bool(re.search(r'однозначн', current_message, re.I))
        if (two_digits and one_digit or len(expressions) > 1
                or re.search(r'тр[её]хзначн\w*\s+делител|делител\w*\s+тр[её]хзначн', current_message, re.I)):
            return None
        envelope = _DIVISION.sub(' ', current_message)
        envelope = re.sub(r'(?:ответ\s+не\s+(?:показывай|называй|сообщай|давай)|не\s+(?:показывай|называй|сообщай|давай)\s+ответ|без\s+ответа|не\s+повторяй)', ' ', envelope, flags=re.I)
        if one_digit:
            envelope = re.sub(r'от\s+2\s+до\s+9', ' ', envelope, flags=re.I)
        if two_digits:
            envelope = re.sub(r'от\s+10\s+до\s+99', ' ', envelope, flags=re.I)
        if re.search(r'[0-9=+*/×÷^<>−-]', envelope) or not _known_words(envelope, _PRACTICE_WORDS):
            return None
        excluded = [(int(item[1]), int(item[2])) for item in expressions]
        if not isinstance(previous_messages, (list, tuple)) or len(previous_messages) > 40:
            return None
        for previous in previous_messages:
            if (not isinstance(previous, dict) or previous.get('role') not in {'user', 'assistant'}
                    or not isinstance(previous.get('content'), str) or len(previous['content']) > 4000):
                return None
            # Prior assistant examples are excluded as examples only, never
            # interpreted as evidence of a learner's actions or competence.
            excluded.extend((int(item[1]), int(item[2])) for item in _DIVISION.finditer(previous['content'])
                            if 1 <= int(item[2]) <= 99)
        excluded = list(dict.fromkeys(excluded))
        try:
            exercise = practice_division(divisor_digits=2 if two_digits else 1 if one_digit else None,
                                         exclude_pairs=excluded)
        except ValueError:
            return None
        result = _completion(f"Попробуй вычислить {exercise['dividend']} : {exercise['divisor']}. "
                             'Обрати внимание, нужно ли записать ноль внутри частного.', 'division_practice',
                           {'dividend': exercise['dividend'], 'divisor': exercise['divisor']},
                           ['integer_division', 'three_digit_quotient', 'internal_zero', 'quotient_withheld'])
        result['learning_tool']['dialogue_used_for_example_exclusion'] = bool(previous_messages)
        return result
    if _HINT.search(current_message):
        return None
    places = bool(re.search(r'разряд|сотн|десятк|единиц', current_message, re.I))
    if len(expressions) != 1 or not (places or re.search(r'провер|верно|правильн|сходится', current_message, re.I)):
        return None
    expression = expressions[0]
    outside = current_message[:expression.start()] + ' ' + current_message[expression.end():]
    if re.search(r'[0-9=+*/×÷^<>−-]', outside) or not _known_words(outside, _CHECK_WORDS):
        return None
    dividend, divisor = int(expression[1]), int(expression[2])
    try:
        facts = division_facts(dividend, divisor)
    except ValueError:
        return None
    if facts['remainder'] or 'остат' in current_message.casefold():
        return None
    if places and expression[3] is None:
        return None
    text = format_verified_division(facts, mode='places' if places else 'check')
    if expression[3] is not None:
        stated = int(expression[3])
        text = ('Результат верный. ' if stated == facts['quotient'] else
                f"В записи есть ошибка: частное равно {facts['quotient']}, а не {stated}. ") + text
    return _completion(text, 'division_places' if places else 'division_check',
                       {'dividend': dividend, 'divisor': divisor, 'quotient': facts['quotient'], 'remainder': 0},
                       ['integer_division_identity', 'quotient_place_values'] if places else ['integer_division_identity'])
