"""ЗБР stage 4: build one individual task card for one child.

This is the deliverable of the fourth (individual) lesson stage from the lesson
deck: after the group solves the detective case, every child gets a personal
card whose storyline is built from what that child loves, whose difficulty comes
from the competence map, and whose hints form a ladder the adult walks down only
when needed.

Division of labour, on purpose:

* :mod:`zpd` (in learner-data) decides the topic, the difficulty and how many
  hint rungs to prepare — deterministic, explainable, no model involved;
* :mod:`pipeline.retrieval` supplies the mathematics the card may rely on;
* the LLM only writes the storyline and the wording, then its output is checked
  against the contract here.

Nothing in the generated card is trusted blindly: :func:`parse_task_card`
re-imposes the topic, the hint count and the ladder order, and reports what the
model got wrong instead of silently passing it to a child.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from pipeline.config import (
    MAX_CONTRACT_RETRIES,
    TASK_CARD_TASK_COUNT,
    TASK_DESIGN_TEMPERATURE,
    TASK_DESIGN_TIMEOUT_SEC,
)
from pipeline.learner_context import TaskDesignContext, load_task_design_context
from pipeline.llm_json import CompleteFn, LlmError, complete_json
from pipeline.retrieval import Material, retrieve_material
from pipeline.task_checks import check_arithmetic, check_not_copied, check_personalization

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "task_designer_prompt.txt"

_FALLBACK_PROMPT = (
    "Ты собираешь индивидуальную карточку-задание для ребёнка. Верни один JSON-объект "
    "с полями topic_id, story{title,intro}, tasks[{statement,expected_answer,checks_error,"
    "hints[{level,type,text}],materials}], reflection_question, tutor_notes. "
    "Подсказки не содержат готового ответа."
)

# The ladder rung order the contract requires. Kept here (not imported from zpd)
# because it describes the *output format*, while zpd owns the pedagogy.
HINT_TYPES = ("hint_question", "visual", "joint")

# Strings that must never reach a child: internal vocabulary and scores.
_LEAK_MARKERS = ("mastery", "independence", "збр", "зона ближайшего", "topic_id", "math.g3", "math.g4")


class TaskDesignError(RuntimeError):
    """Raised when no usable task card could be produced."""


@dataclass(frozen=True)
class Hint:
    level: int
    type: str
    text: str

    def as_dict(self) -> dict:
        return {"level": self.level, "type": self.type, "text": self.text}


@dataclass(frozen=True)
class Task:
    statement: str
    expected_answer: str
    checks_error: str = ""
    hints: tuple[Hint, ...] = field(default_factory=tuple)
    materials: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "statement": self.statement,
            "expected_answer": self.expected_answer,
            "checks_error": self.checks_error,
            "hints": [hint.as_dict() for hint in self.hints],
            "materials": list(self.materials),
        }


@dataclass(frozen=True)
class TaskCard:
    """One printable individual card plus the metadata the tutor needs."""

    learner_id: str
    pseudonym: str
    topic_id: str
    topic_title: str
    zone: str
    zone_ru: str
    difficulty: str
    hint_depth: int
    story_title: str
    story_intro: str
    tasks: tuple[Task, ...]
    reflection_question: str = ""
    tutor_notes: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    retries: int = 0

    def as_dict(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "pseudonym": self.pseudonym,
            "topic_id": self.topic_id,
            "topic_title": self.topic_title,
            "zone": self.zone,
            "zone_ru": self.zone_ru,
            "difficulty": self.difficulty,
            "hint_depth": self.hint_depth,
            "story": {"title": self.story_title, "intro": self.story_intro},
            "tasks": [task.as_dict() for task in self.tasks],
            "reflection_question": self.reflection_question,
            "tutor_notes": self.tutor_notes,
            "sources": list(self.sources),
            "warnings": list(self.warnings),
            "retries": self.retries,
        }


def load_designer_prompt() -> str:
    """Read the stage-4 pedagogical prompt, which the methodologist owns."""
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Prompt file not found: %s — using fallback", _PROMPT_FILE)
        return _FALLBACK_PROMPT


def _render_material(materials: Sequence[Material]) -> str:
    if not materials:
        return (
            "<curriculum_material>\n"
            "Материал не найден. Составь задание только по школьной программе класса, "
            "не вводя новых правил и терминов.\n"
            "</curriculum_material>"
        )
    parts = [
        "<curriculum_material>",
        "Правила_использования: математику берём отсюда. Сюжет придумываешь сам, "
        "правила и термины — нет.",
    ]
    for index, material in enumerate(materials, start=1):
        parts.append(f"[Фрагмент {index}] {material.topic} (источник: {material.source})")
        parts.append(material.content)
        if material.answer:
            parts.append(f"Разбор для взрослого: {material.answer}")
    parts.append("</curriculum_material>")
    return "\n".join(parts)


def _plural(count: int, one: str, few: str, many: str) -> str:
    """Russian plural agreement — the prompt is read by a Russian-speaking model."""
    tail_100 = abs(count) % 100
    tail_10 = abs(count) % 10
    if 11 <= tail_100 <= 14:
        return many
    if tail_10 == 1:
        return one
    if 2 <= tail_10 <= 4:
        return few
    return many


def build_task_prompt(context: TaskDesignContext, materials: Sequence[Material], count: int) -> str:
    """Assemble the user prompt: learner projection + material + the concrete order."""
    request = "\n".join([
        "<request>",
        f"Собери индивидуальную карточку-задание: ровно {count} "
        f"{_plural(count, 'задание', 'задания', 'заданий')} по теме «{context.topic_title}».",
        f"В каждом задании ровно {context.hint_depth} "
        f"{_plural(context.hint_depth, 'ступень', 'ступени', 'ступеней')} подсказок.",
        f"topic_id в ответе: {context.topic_id}",
        "Верни только JSON по контракту.",
        "</request>",
    ])
    return "\n\n".join([context.prompt_text, _render_material(materials), request])


def _clean_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalise_hints(raw: object, hint_depth: int, warnings: list[str], task_number: int) -> tuple[Hint, ...]:
    """Force the model's hints onto the ladder: right count, right order, right types.

    Levels and types are re-imposed rather than trusted, because the ladder order
    is pedagogy (нельзя выдать частичное решение вместо наводящего вопроса), not
    something the model may improvise.
    """
    items = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    texts = [_clean_text(item.get("text")) for item in items]
    texts = [text for text in texts if text]

    if len(texts) > hint_depth:
        warnings.append(
            f"задание {task_number}: модель прислала {len(texts)} подсказок вместо {hint_depth} — лишние отброшены"
        )
        texts = texts[:hint_depth]
    elif len(texts) < hint_depth:
        warnings.append(
            f"задание {task_number}: подсказок {len(texts)} вместо {hint_depth} — лестница короче правила"
        )

    return tuple(
        Hint(level=index, type=HINT_TYPES[min(index - 1, len(HINT_TYPES) - 1)], text=text)
        for index, text in enumerate(texts, start=1)
    )


def _check_leaks(label: str, text: str, warnings: list[str]) -> None:
    lowered = text.lower()
    for marker in _LEAK_MARKERS:
        if marker in lowered:
            warnings.append(f"{label}: в тексте для ребёнка встречается служебное «{marker}»")
            return


def parse_task_card(
    payload: dict,
    context: TaskDesignContext,
    count: int,
    material_texts: list[str] | None = None,
) -> TaskCard:
    """Validate and normalise the model's JSON into a :class:`TaskCard`.

    Raises :class:`TaskDesignError` only for damage that cannot be repaired
    (no tasks, empty statements). Everything repairable becomes a warning
    attached to the card, so the tutor sees what the model got wrong.
    """
    warnings: list[str] = []

    story = payload.get("story") if isinstance(payload.get("story"), dict) else {}
    story_title = _clean_text(story.get("title")) or f"Задание для {context.pseudonym}"
    story_intro = _clean_text(story.get("intro"))
    if not story_intro:
        warnings.append("сюжетное вступление пустое — карточка без истории")

    returned_topic = _clean_text(payload.get("topic_id"))
    if returned_topic and returned_topic != context.topic_id:
        warnings.append(
            f"модель вернула topic_id «{returned_topic}» вместо «{context.topic_id}» — исправлено"
        )

    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise TaskDesignError("модель не вернула ни одного задания в поле tasks")

    if len(raw_tasks) > count:
        warnings.append(f"заданий пришло {len(raw_tasks)} вместо {count} — лишние отброшены")
        raw_tasks = raw_tasks[:count]
    elif len(raw_tasks) < count:
        warnings.append(f"заданий пришло {len(raw_tasks)} вместо {count}")

    tasks: list[Task] = []
    for number, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, dict):
            warnings.append(f"задание {number}: не объект, пропущено")
            continue
        statement = _clean_text(raw_task.get("statement"))
        if not statement:
            warnings.append(f"задание {number}: пустой текст задания, пропущено")
            continue

        expected_answer = _clean_text(raw_task.get("expected_answer"))
        if not expected_answer:
            warnings.append(f"задание {number}: нет expected_answer — взрослому нечем проверять")

        hints = _normalise_hints(raw_task.get("hints"), context.hint_depth, warnings, number)

        _check_leaks(f"задание {number}", statement, warnings)
        for hint in hints:
            _check_leaks(f"задание {number}, подсказка {hint.level}", hint.text, warnings)
            if expected_answer and len(expected_answer) > 6 and expected_answer.lower() in hint.text.lower():
                warnings.append(
                    f"задание {number}, подсказка {hint.level}: содержит готовый ответ — опора не должна его выдавать"
                )

        # Арифметика — единственная ошибка, которую нельзя заметить по форме
        # карточки: неверный ответ выглядит ровно так же, как верный.
        for problem in check_arithmetic(expected_answer) + check_arithmetic(statement):
            warnings.append(f"задание {number}: проверьте счёт — {problem}")

        materials = raw_task.get("materials")
        tasks.append(
            Task(
                statement=statement,
                expected_answer=expected_answer,
                checks_error=_clean_text(raw_task.get("checks_error")),
                hints=hints,
                materials=tuple(
                    _clean_text(item) for item in materials if _clean_text(item)
                ) if isinstance(materials, list) else (),
            )
        )

    if not tasks:
        raise TaskDesignError("ни одно задание из ответа модели не прошло проверку")

    _check_leaks("сюжет", story_intro, warnings)

    if context.error_tags and not any(task.checks_error for task in tasks):
        warnings.append(
            "ни одно задание не отмечено как проверка типичной ошибки, хотя ошибки в профиле есть"
        )

    # Персонализация — это весь смысл 4 этапа. Формально карточка может пройти
    # все проверки и при этом не иметь никакого отношения к тому, что ребёнок любит.
    if not check_personalization(
        [story_title, story_intro, *(task.statement for task in tasks)],
        context.interests,
        context.story_preferences,
    ):
        warnings.append(
            "сюжет не опирается на интересы ребёнка: "
            f"ни один из них ({', '.join(context.interests + context.story_preferences)}) не упомянут"
        )
    elif not any(
        check_personalization([task.statement], context.interests, context.story_preferences)
        for task in tasks
    ):
        # Сюжет только во вступлении — ребёнок читает задание и снова видит
        # обычный учебник. По сценарию урока тема должна быть в самих заданиях.
        warnings.append("сюжет есть только во вступлении: ни одно задание не связано с интересами")

    for number, task in enumerate(tasks, start=1):
        if check_not_copied(task.statement, material_texts or []):
            warnings.append(
                f"задание {number} почти дословно повторяет учебный материал — "
                "материал нужен как источник математики, а не как готовая формулировка"
            )

    return TaskCard(
        learner_id=context.learner_id,
        pseudonym=context.pseudonym,
        topic_id=context.topic_id,
        topic_title=context.topic_title,
        zone=context.zone,
        zone_ru=context.zone_ru,
        difficulty=context.difficulty,
        hint_depth=context.hint_depth,
        story_title=story_title,
        story_intro=story_intro,
        tasks=tuple(tasks),
        reflection_question=_clean_text(payload.get("reflection_question")),
        tutor_notes=_clean_text(payload.get("tutor_notes")),
        warnings=tuple(warnings),
    )


# Замечания, ради которых имеет смысл переспросить модель. Это те, где она
# поняла задачу неверно, а не те, где код уже всё починил без потерь.
RETRYABLE_MARKERS = (
    "проверьте счёт",
    "дословно повторяет",
    "только во вступлении",
    "не опирается на интересы",
    "готовый ответ",
    "служебное",
    "подсказок",
    "лестница короче",
    "заданий пришло",
    "проверка типичной ошибки",
)


def build_correction(warnings: Sequence[str]) -> str:
    """Собрать претензию к прошлому ответу — конкретную, а не «сделай лучше»."""
    complaints = [
        warning for warning in warnings
        if any(marker in warning.lower() for marker in RETRYABLE_MARKERS)
    ]
    if not complaints:
        return ""
    lines = [
        "<correction>",
        "Твой прошлый ответ нарушил контракт. Что именно не так:",
        *(f"- {complaint}" for complaint in complaints),
        "",
        "Собери карточку заново, исправив ровно это. Остальное менять не нужно.",
        "Верни только JSON по контракту.",
        "</correction>",
    ]
    return "\n".join(lines)


def design_task_card(
    learner_id: str,
    topic_id: str | None = None,
    *,
    count: int | None = None,
    complete: CompleteFn | None = None,
    retriever: Callable[[TaskDesignContext], list[Material]] | None = None,
    index_path: Path | None = None,
) -> TaskCard:
    """Run the whole stage-4 pipeline for one learner.

    ``complete`` and ``retriever`` are injection points: tests (and offline
    demos) replace them without touching the pedagogy.
    """
    task_count = max(1, int(count or TASK_CARD_TASK_COUNT))
    context = load_task_design_context(learner_id, topic_id)

    if retriever is not None:
        materials = list(retriever(context))
    else:
        materials = retrieve_material(
            context.retrieval_query,
            topic_id=context.topic_id,
            grade=context.grade,
            limit=3,
            index_path=index_path,
            difficulty=context.difficulty,
            error_tags=context.error_tags,
        )

    prompt = build_task_prompt(context, materials, task_count)
    system_prompt = load_designer_prompt()
    material_texts = [item.content for item in materials]

    def ask(user_prompt: str) -> TaskCard:
        payload = complete_json(
            user_prompt,
            system_prompt,
            complete=complete,
            temperature=TASK_DESIGN_TEMPERATURE,
            timeout=TASK_DESIGN_TIMEOUT_SEC,
        )
        return parse_task_card(payload, context, task_count, material_texts)

    card = ask(prompt)

    # Один повторный запрос с конкретной претензией. Слабые модели со второй
    # попытки обычно попадают, а стоит это одну лишнюю генерацию — заметно
    # дешевле, чем карточка, которую придётся переделывать руками.
    for _ in range(max(0, MAX_CONTRACT_RETRIES)):
        correction = build_correction(card.warnings)
        if not correction:
            break
        try:
            candidate = ask(f"{prompt}\n\n{correction}")
        except (LlmError, TaskDesignError) as exc:
            logger.warning("Повторный запрос не удался, оставляю первый вариант: %s", exc)
            break
        card = replace(card, retries=card.retries + 1)
        if len(candidate.warnings) < len(card.warnings):
            card = replace(candidate, retries=card.retries)
    # Retrieval provenance is attached after validation so the tutor can see
    # which material the wording was grounded in (or that there was none).
    sources = tuple(dict.fromkeys(material.source for material in materials if material.source))
    return replace(card, sources=sources)


def render_task_card(card: TaskCard, *, show_answers: bool = True) -> str:
    """Render the card as printable text.

    ``show_answers=False`` produces the child's copy: statements and nothing
    else. The hints stay out of the child's copy on purpose — the adult opens
    them one rung at a time, and only if the child is stuck.
    """
    lines = [f"КАРТОЧКА: {card.story_title}", f"Для: {card.pseudonym}", ""]
    if card.story_intro:
        lines += [card.story_intro, ""]

    for number, task in enumerate(card.tasks, start=1):
        lines.append(f"{number}. {task.statement}")
        if task.materials:
            lines.append(f"   Понадобится: {', '.join(task.materials)}")
        if show_answers:
            if task.expected_answer:
                lines.append(f"   Ответ (для взрослого): {task.expected_answer}")
            if task.checks_error:
                lines.append(f"   Проверяет ошибку: {task.checks_error}")
            for hint in task.hints:
                lines.append(f"   Подсказка {hint.level} ({hint.type}): {hint.text}")
        lines.append("")

    if card.reflection_question:
        lines += [f"Вопрос напоследок: {card.reflection_question}", ""]

    if show_answers:
        lines += [
            f"Тема: {card.topic_title} [{card.topic_id}]",
            f"Зона: {card.zone_ru}; сложность: {card.difficulty}; ступеней подсказок: {card.hint_depth}",
        ]
        if card.tutor_notes:
            lines.append(f"Педагогу: {card.tutor_notes}")
        if card.sources:
            lines.append(f"Материал: {', '.join(card.sources)}")
        if card.warnings:
            lines.append("")
            lines.append("Замечания к сгенерированной карточке:")
            lines.extend(f"- {warning}" for warning in card.warnings)

    return "\n".join(lines).rstrip() + "\n"


def card_to_json(card: TaskCard) -> str:
    return json.dumps(card.as_dict(), ensure_ascii=False, indent=2)
