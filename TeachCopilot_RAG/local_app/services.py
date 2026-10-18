"""Small deterministic services used by the local teacher API."""
from __future__ import annotations

import json
import re


def _snapshot(row: dict) -> dict:
    value = row.get("raw_snapshot")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _editorial_text(value: str) -> str:
    return value.replace("\u2014", "\u2013")


def _general_scaffolds(topic_id: str) -> tuple[str, str, str]:
    if topic_id.startswith("math."):
        return (
            "\u041a\u0430\u043a\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0443\u0436\u0435 \u0435\u0441\u0442\u044c \u0432 \u0443\u0441\u043b\u043e\u0432\u0438\u0438 \u0438 \u0447\u0442\u043e \u043d\u0443\u0436\u043d\u043e \u043d\u0430\u0439\u0442\u0438?",
            "\u0421\u043e\u0441\u0442\u0430\u0432\u044c \u043d\u0430 \u0431\u0443\u043c\u0430\u0433\u0435 \u0441\u0445\u0435\u043c\u0443: \u0441\u043d\u0430\u0447\u0430\u043b\u0430 \u043e\u0442\u043c\u0435\u0442\u044c \u0438\u0437\u0432\u0435\u0441\u0442\u043d\u043e\u0435, \u0437\u0430\u0442\u0435\u043c \u043f\u043e\u043a\u0430\u0436\u0438 \u043f\u0435\u0440\u0432\u044b\u0439 \u0448\u0430\u0433 \u0440\u0435\u0448\u0435\u043d\u0438\u044f.",
            "\u0421\u0434\u0435\u043b\u0430\u0439 \u043f\u0435\u0440\u0432\u044b\u0439 \u0448\u0430\u0433 \u0432\u043c\u0435\u0441\u0442\u0435 \u0441\u043e \u0432\u0437\u0440\u043e\u0441\u043b\u044b\u043c, \u0430 \u0437\u0430\u0442\u0435\u043c \u043f\u043e\u043f\u0440\u043e\u0431\u0443\u0439 \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u0442\u044c \u0441\u0430\u043c\u043e\u0441\u0442\u043e\u044f\u0442\u0435\u043b\u044c\u043d\u043e.",
        )
    return (
        "\u041a\u0430\u043a\u043e\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 \u0434\u043e\u043b\u0436\u043d\u0430 \u043f\u043e\u043b\u0443\u0447\u0438\u0442\u044c \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0430 \u0438 \u043a\u0430\u043a\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0435\u0439 \u0434\u0430\u043d\u044b?",
        "\u041f\u0440\u043e\u0441\u043b\u0435\u0434\u0438 \u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d\u0438\u0435 \u043f\u043e \u0448\u0430\u0433\u0430\u043c \u043d\u0430 \u0431\u0443\u043c\u0430\u0433\u0435: \u0437\u0430\u043f\u0438\u0448\u0438 \u0432\u0445\u043e\u0434\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435 \u0438 \u043f\u0440\u043e\u043c\u0435\u0436\u0443\u0442\u043e\u0447\u043d\u044b\u0439 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442.",
        "\u0421\u043e\u0441\u0442\u0430\u0432\u044c \u043f\u0435\u0440\u0432\u0443\u044e \u043a\u043e\u043c\u0430\u043d\u0434\u0443 \u0432\u043c\u0435\u0441\u0442\u0435 \u0441\u043e \u0432\u0437\u0440\u043e\u0441\u043b\u044b\u043c, \u0430 \u0437\u0430\u0442\u0435\u043c \u0434\u043e\u0431\u0430\u0432\u044c \u0441\u043b\u0435\u0434\u0443\u044e\u0449\u0438\u0439 \u0448\u0430\u0433 \u0441\u0430\u043c\u043e\u0441\u0442\u043e\u044f\u0442\u0435\u043b\u044c\u043d\u043e.",
    )


def design_source_template(learner_id: str, topic_id: str | None, *, store, count: int = 1):
    """Build a source-grounded card from canonical active curriculum tasks.

    Questions and answer keys come from exact-topic source snapshots. Authored
    hints are used when available. General scaffolding fills missing rungs and
    is clearly marked for tutor review.
    """
    from pipeline.learner_context import load_task_design_context
    from pipeline.task_designer import Hint, Task, TaskCard, TaskDesignError

    context = load_task_design_context(learner_id, topic_id, store=store)
    requested = max(1, int(count))
    rows = store.material_rows({"topic_id": context.topic_id})
    eligible: list[tuple[dict, dict]] = []
    for row in rows:
        source = _snapshot(row)
        if source.get("topic_id") != context.topic_id:
            continue
        question = source.get("content")
        answer = source.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        if source.get("difficulty") != context.difficulty:
            continue
        eligible.append((row, source))

    eligible.sort(key=lambda pair: str(pair[1].get("task_id") or pair[0].get("id") or ""))
    unique: list[tuple[dict, dict]] = []
    seen = set()
    for pair in eligible:
        key = _normalized(pair[1]["content"])
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    if len(unique) < requested:
        raise TaskDesignError(
            f"Only {len(unique)} distinct source tasks are available for difficulty {context.difficulty}; requested {requested}"
        )

    tasks = []
    warnings = []
    sources = []
    hint_depths = []
    used_general_scaffolding = False
    hint_types = ("hint_question", "visual", "joint")
    general_scaffolds = _general_scaffolds(context.topic_id)
    for row, source in unique[:requested]:
        raw_hints = source.get("hints")
        answer_text = _editorial_text(source["answer"].strip())
        answer_normalized = _normalized(answer_text)
        authored_hints = []
        if isinstance(raw_hints, list):
            for hint in raw_hints:
                if not isinstance(hint, str) or not hint.strip():
                    continue
                safe_hint = _editorial_text(hint.strip())
                hint_normalized = _normalized(safe_hint)
                if answer_normalized and answer_normalized in hint_normalized:
                    warnings.append(f"Source task {source.get('task_id', '')} contained a hint with the complete answer; it was omitted.")
                    continue
                authored_hints.append(safe_hint)

        depth = min(max(0, int(context.hint_depth)), len(hint_types))
        hints = []
        for index in range(depth):
            if index < len(authored_hints):
                hint_text = authored_hints[index]
            else:
                hint_text = general_scaffolds[index]
                used_general_scaffolding = True
            hints.append(Hint(level=index + 1, type=hint_types[index], text=hint_text))
        hint_depths.append(depth)
        tasks.append(Task(
            statement=_editorial_text(source["content"].strip()),
            expected_answer=answer_text,
            checks_error="",
            hints=tuple(hints),
            materials=(str(row.get("source_file") or "curriculum source"),),
        ))
        source_name = row.get("source_file")
        if source_name:
            sources.append(str(source_name))

    actual_depth = min(hint_depths, default=0)
    if used_general_scaffolding:
        warnings.append("General scaffolding was added; it is not source-authored. Tutor should check applicability before use.")
    return TaskCard(
        learner_id=learner_id,
        pseudonym=context.pseudonym,
        topic_id=context.topic_id,
        topic_title=context.topic_title,
        zone=context.zone,
        zone_ru=context.zone_ru,
        difficulty=context.difficulty,
        hint_depth=actual_depth,
        story_title="\u0417\u0430\u0434\u0430\u043d\u0438\u0435 \u0438\u0437 \u043c\u0430\u0442\u0435\u0440\u0438\u0430\u043b\u0430 \u043a\u0443\u0440\u0441\u0430",
        story_intro="\u0420\u0435\u0448\u0438 \u0437\u0430\u0434\u0430\u043d\u0438\u0435 \u043f\u043e \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u043e\u0439 \u0442\u0435\u043c\u0435 \u0438 \u043e\u0431\u044a\u044f\u0441\u043d\u0438 \u0445\u043e\u0434 \u0440\u0435\u0448\u0435\u043d\u0438\u044f.",
        tasks=tuple(tasks),
        reflection_question="\u041a\u0430\u043a\u043e\u0439 \u0448\u0430\u0433 \u043f\u043e\u043c\u043e\u0433 \u0442\u0435\u0431\u0435 \u0440\u0435\u0448\u0438\u0442\u044c \u0437\u0430\u0434\u0430\u043d\u0438\u0435?",
        tutor_notes="\u0421\u0432\u0435\u0440\u044c \u043e\u0442\u0432\u0435\u0442 \u0441 \u043a\u043b\u044e\u0447\u043e\u043c \u0438\u0437 \u0438\u0441\u0442\u043e\u0447\u043d\u0438\u043a\u0430 \u0438 \u043f\u0440\u043e\u0432\u0435\u0440\u044c, \u0447\u0442\u043e \u0443\u0440\u043e\u0432\u0435\u043d\u044c \u043f\u043e\u0434\u0445\u043e\u0434\u0438\u0442 \u0443\u0447\u0435\u043d\u0438\u043a\u0443.",
        sources=tuple(dict.fromkeys(sources)),
        warnings=tuple(dict.fromkeys(warnings)),
    )
