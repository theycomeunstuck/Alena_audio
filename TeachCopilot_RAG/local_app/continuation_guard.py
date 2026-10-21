"""A narrow guard against explicitly relabelling a remembered division task.

It is not a general math judge. Ordinary intermediate calculations are allowed;
only a claim naming a different exercise during a clear same-task return fails.
"""
import re

from local_app.checked_division_context import DIVISION_TOPIC, extract_literal_division_pair

_RESUME = re.compile(r'\b(?:продолжим|продолжить|верн[её]мся|где\s+остановил\w*)\b', re.I)
_NEW = re.compile(r'\b(?:нов\w*|друг\w*|не\s+(?:продолж\w*|возвращ\w*))\b', re.I)
_PAIR = r'(?P<a>[0-9]{1,6})\s*(?:[:÷/]|на)\s*(?P<b>[0-9]{1,2})(?!\w|[.,][0-9])'
_CLAIM = re.compile(
    r'\b(?:пример(?:у|ом|е|а)?|задани(?:е|ю|я|ем))\s*:?\s*(?:с\s+)?(?:делени(?:ем|е|я)\s+)?' + _PAIR, re.I)
_RETURN = re.compile(r'\b(?:продолжим|верн[её]мся\s+к)\s*(?:делени(?:ю|е)\s+)?' + _PAIR, re.I)


def foreign_exercise_guard(current_message: str, context: dict, answer: str) -> dict | None:
    if (not isinstance(context, dict) or context.get('topic_id') != DIVISION_TOPIC
            or not context.get('version_id') or not _RESUME.search(current_message) or _NEW.search(current_message)):
        return None
    rows = context.get('sources', [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) or row.get('corrects_source_id') for row in rows):
        return None
    selected, pairs = set(), set()
    for row in rows:
        if (row.get('role') or row.get('author_role')) not in {'student','teacher'}:
            continue
        if row.get('source_id'):
            selected.add(row['source_id'])
        text = row.get('text')
        pair = extract_literal_division_pair(text, division_context=True) if isinstance(text, str) else None
        if pair:
            pairs.add(pair)
    for key in ('dialogue_context','session_end_context'):
        for row in context.get(key, ()):
            if (not isinstance(row, dict) or row.get('source_id') not in selected
                    or row.get('role') != 'assistant' or row.get('usage') != 'context_only'
                    or row.get('truncated') is not False or not isinstance(row.get('text'), str)):
                continue
            pair = extract_literal_division_pair(row['text'], division_context=True)
            if pair:
                pairs.add(pair)
    if len(pairs) != 1:
        return None
    expected = next(iter(pairs))
    explicit = extract_literal_division_pair(current_message, division_context=True)
    if explicit and explicit != expected:
        return None
    for pattern in (_CLAIM, _RETURN):
        for match in pattern.finditer(answer):
            referenced = int(match['a']), int(match['b'])
            if referenced != expected:
                return {'answer': f'В памяти занятия есть пример {expected[0]} : {expected[1]}. Какой шаг в нём сейчас непонятен?',
                        'expected_pair': list(expected), 'referenced_pair': list(referenced)}
    return None
