"""Literal retrieval anchors from an already selected, approved topic context.

This does not identify an active exercise or assess a learner. Correction groups
and whole assistant task context remain labelled, inseparable retrieval data.
"""
from __future__ import annotations

ANCHOR_BUDGET = 4000


def memory_retrieval_query(current_message: str, topic_id: str, context: dict,
                           *, topic_title: str | None = None) -> dict:
    if not isinstance(current_message, str) or not current_message.strip() or len(current_message) > 12000:
        raise ValueError('Invalid current message for memory retrieval')
    if not isinstance(topic_id, str) or not topic_id or len(topic_id) > 200:
        raise ValueError('Invalid retrieval topic')
    if not isinstance(context, dict) or context.get('topic_id') != topic_id or not context.get('version_id'):
        raise ValueError('Retrieval requires attributed memory for the exact topic')
    rows = context.get('sources', [])
    if not isinstance(rows, list):
        raise ValueError('Invalid selected memory sources')
    # No parsing of summary/prose or model rewrites. Duplicate or broken source
    # groups are omitted explicitly, never repaired into a different assertion.
    indexed, invalid, links = {}, set(), {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            invalid.add(f'invalid-{index}')
            continue
        key = row.get('source_id')
        if not isinstance(key, str) or not key:
            invalid.add(f'invalid-{index}')
            continue
        if key in indexed:
            invalid.add(key)
        else:
            indexed[key] = (index, row)
        links.setdefault(key, set())
        if ((row.get('role') or row.get('author_role')) not in {'student', 'teacher'}
                or not isinstance(row.get('text'), str) or not row['text'].strip()):
            invalid.add(key)
    for key, (index, row) in indexed.items():
        target = row.get('corrects_source_id')
        if target is None:
            continue
        if (not isinstance(target, str) or target not in indexed
                or indexed[target][0] >= index
                or (row.get('role') or row.get('author_role')) != 'teacher'):
            invalid.add(key)
        if isinstance(target, str) and target in indexed:
            links[key].add(target)
            links[target].add(key)
    contexts = {}
    omitted_contexts = 0
    for name in ('dialogue_context', 'session_end_context'):
        for item in context.get(name, ()):
            if (not isinstance(item, dict) or item.get('source_id') not in indexed
                    or (indexed[item['source_id']][1].get('role') or indexed[item['source_id']][1].get('author_role')) != 'student'
                    or item.get('role') != 'assistant' or item.get('usage') != 'context_only'
                    or item.get('truncated') is not False or not isinstance(item.get('text'), str)
                    or not item['text'].strip() or len(item['text']) > 1200):
                omitted_contexts += 1
                continue
            contexts.setdefault(item['source_id'], []).append(item['text'])
    groups, seen, invalid_groups = [], set(), 0
    for key in indexed:
        if key in seen:
            continue
        pending, group = [key], set()
        while pending:
            source = pending.pop()
            if source not in group:
                group.add(source)
                pending.extend(links[source] - group)
        seen.update(group)
        if group & invalid:
            invalid_groups += 1
            continue
        ordered = sorted(group, key=lambda source: indexed[source][0])
        corrected = {indexed[source][1].get('corrects_source_id') for source in ordered}
        parts, context_count = [], 0
        for source in ordered:
            row = indexed[source][1]
            role = row.get('role') or row.get('author_role')
            label = ('Исправление преподавателя' if row.get('corrects_source_id') else
                     'Исправленная исходная запись' if source in corrected else
                     'Реплика ученика' if role == 'student' else 'Заметка преподавателя')
            parts.append(label + ': ' + row['text'])
            for text in contexts.get(source, ()):
                parts.append('Условие/вопрос ассистента, только контекст, не действие ученика: ' + text)
                context_count += 1
        groups.append((ordered, '\n'.join(parts), context_count))
    selected, used, omitted_groups = [], 0, 0
    for group in reversed(groups):
        size = len(group[1]) + 1
        if used + size > ANCHOR_BUDGET:
            omitted_groups += 1
            continue
        selected.append(group)
        used += size
    selected.reverse()
    title = topic_title if isinstance(topic_title, str) and 0 < len(topic_title) <= 200 else topic_id
    query = f'Тема занятия: {title}.\nТекущая реплика ученика: {current_message}'
    if selected:
        query += '\nВыбранные источники прошлого занятия (не устанавливают текущий этап):\n'
        query += '\n'.join(group[1] for group in selected)
    return {'query': query, 'diagnostics': {
        'mode': 'approved_memory_exact_topic', 'topic_id': topic_id,
        'literal_source_count': sum(len(group[0]) for group in selected),
        'context_only_count': sum(group[2] for group in selected),
        'anchor_status': 'selected_literal_sources' if selected else 'literal_anchor_unavailable',
        'omitted_source_groups': omitted_groups, 'invalid_source_groups': invalid_groups,
        'omitted_contexts': omitted_contexts,
    }}
