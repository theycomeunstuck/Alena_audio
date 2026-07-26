"""Turn free-text lesson notes into validated updates of the learner JSON cards.

The tutor writes what happened in the lesson — one text for the whole group or
one per child. The LLM extracts *facts* (which topic, how much help was needed,
which mistake reappeared); this module turns those facts into numbers and writes
them back into ``learner-data/learners/*.json``.

Two guarantees make that safe enough to run on real data:

* the model never writes ``mastery``. It reports the help level; the EMA in
  :mod:`zpd` computes the number. Re-running the same notes gives the same card.
* nothing is written before the whole card passes learner-data validation. A
  malformed proposal fails the update instead of corrupting a card.

Modes (``TEACHCOPILOT_MASTERY_MODE``):

* ``auto_ema`` — observations are applied straight away;
* ``tutor_confirmed`` — the same numbers are computed and shown as a proposal;
  only the topics the tutor confirms are written. Journal, error patterns and
  progress notes apply in both modes: they are facts, not assessments.
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pipeline.config import LEARNER_DATA_DIR, MASTERY_MODE, MASTERY_MODES, ROSTER_BATCH_SIZE
from pipeline.learner_context import (
    LearnerContextError,
    load_learner_data_tools,
    load_validated_card,
)
from pipeline.llm_json import CompleteFn, complete_json

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "card_update_prompt.txt"

_FALLBACK_PROMPT = (
    "Извлеки из заметок об уроке факты по каждому ученику и верни JSON "
    '{"learners":[{"learner_id":"...","observations":[{"topic_id":"...",'
    '"help_level":"independent|hint_question|visual|joint","solved":true,"note":"..."}]}]}. '
    "Числа освоения не выставляй."
)

# Free-text fields that end up in rag_context are capped by the learner-data
# contract; trim before validation so a chatty model fails softly.
RAG_TEXT_MAX = 280
MAX_SCAFFOLDING_PER_TOPIC = 4

# Zone → knowledge status mapping lives in learner-data/tools/zpd.py
# (zpd.STATUS_BY_ZONE / zpd.status_conflicts) so the validator and the updater
# cannot disagree about what a status means.

# In tutor_confirmed mode the tutor lists the topics they accept. This sentinel
# accepts all of them at once without having to type every topic_id.
CONFIRM_ALL = "all"


class CardUpdateError(RuntimeError):
    """Raised when an update could not be produced or would corrupt a card."""


@dataclass(frozen=True)
class ObservedAttempt:
    """One attempt as reported by the model: which topic, how much help."""

    topic_id: str
    help_level: str
    solved: bool
    note: str = ""


@dataclass(frozen=True)
class CardUpdateProposal:
    """What the model extracted for one learner. Numbers are not part of it."""

    learner_id: str
    observations: tuple[ObservedAttempt, ...] = field(default_factory=tuple)
    new_error_patterns: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (topic_id, error_tag)
    effective_scaffolding: tuple[tuple[str, str], ...] = field(default_factory=tuple)  # (topic_id, strategy)
    journal_note: str = ""
    progress_note: str = ""
    new_interests: tuple[str, ...] = field(default_factory=tuple)
    new_story_preferences: tuple[str, ...] = field(default_factory=tuple)
    new_goal: str = ""
    lesson_minutes: int | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not any((
            self.observations,
            self.new_error_patterns,
            self.effective_scaffolding,
            self.journal_note,
            self.progress_note,
            self.new_interests,
            self.new_story_preferences,
            self.new_goal,
        ))


@dataclass(frozen=True)
class AppliedUpdate:
    """Result of applying one proposal: the new card plus a readable diff."""

    learner_id: str
    card: dict
    changes: tuple[str, ...]
    pending: tuple[str, ...]
    warnings: tuple[str, ...]

    def changed(self) -> bool:
        return bool(self.changes)


def load_update_prompt() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Prompt file not found: %s — using fallback", _PROMPT_FILE)
        return _FALLBACK_PROMPT


def _today() -> str:
    return datetime.date.today().isoformat()


def _clean(value: object, limit: int = RAG_TEXT_MAX) -> str:
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:limit].rstrip()


# --- Roster: what the model is allowed to talk about --------------------------

def build_roster(learner_ids: Sequence[str]) -> tuple[str, dict[str, dict], dict]:
    """Render the group roster; also return the loaded cards and the catalog.

    The roster is the model's whitelist: every ``topic_id`` and every existing
    ``error_tag`` it may reuse appears here. That is what keeps it from inventing
    curriculum ids or duplicating an error under slightly different wording.
    """
    tools = load_learner_data_tools()
    zpd = tools.zpd

    cards: dict[str, dict] = {}
    catalog: dict = {}
    parts = ["<roster>"]
    for learner_id in learner_ids:
        card, catalog = load_validated_card(learner_id)
        cards[learner_id] = card


        parts.append(f"- learner_id: {learner_id}")
        parts.append(f"  имя: {card['pseudonym']}, {card['grade']} класс")

        verdicts = {verdict.topic_id: verdict for verdict in zpd.classify_card(card)}
        topic_ids: list[str] = list(verdicts)
        rag_context = card.get("rag_context") or {}
        for topic_id in [
            (rag_context.get("current_goal") or {}).get("topic_id"),
            *(rag_context.get("current_topics") or []),
        ]:
            if isinstance(topic_id, str) and topic_id and topic_id not in topic_ids:
                topic_ids.append(topic_id)

        parts.append("  темы:")
        for topic_id in topic_ids:
            entry = catalog.get(topic_id) or {}
            verdict = verdicts.get(topic_id)
            zone = f", зона: {verdict.label_ru()}" if verdict else ", зона: не оценивалась"
            parts.append(f"    - {topic_id} — {entry.get('title_ru', topic_id)}{zone}")

        existing_errors = [
            f"{item['error_tag']} ({item['topic_id']})"
            for item in card.get("error_patterns") or []
            if isinstance(item.get("error_tag"), str) and isinstance(item.get("topic_id"), str)
        ]
        if existing_errors:
            parts.append("  уже известные ошибки (используй дословно, если повторились):")
            parts.extend(f"    - {item}" for item in existing_errors)
    parts.append("</roster>")
    return "\n".join(parts), cards, catalog


def build_update_prompt(notes: str, roster_text: str) -> str:
    lesson_notes = "\n".join(["<lesson_notes>", notes.strip(), "</lesson_notes>"])
    request = "\n".join(["<request>", "Разложи заметки по контракту. Верни только JSON.", "</request>"])
    return "\n\n".join([roster_text, lesson_notes, request])


# --- Parsing the model reply -------------------------------------------------

def _lesson_minutes(value: object, warnings: list[str]) -> int | None:
    """Длительность занятия: только правдоподобные значения, иначе — ничего."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        warnings.append(f"длительность занятия «{value}» не распознана — пропущена")
        return None
    minutes = int(value)
    if not 1 <= minutes <= 600:
        warnings.append(f"длительность занятия {minutes} мин выглядит ошибкой — пропущена")
        return None
    return minutes


def _parse_learner_block(block: dict, cards: dict[str, dict], catalog: dict) -> CardUpdateProposal | None:
    tools = load_learner_data_tools()
    zpd = tools.zpd

    learner_id = _clean(block.get("learner_id"), 80)
    if learner_id not in cards:
        logger.warning("Skipping unknown learner_id in model reply: %r", learner_id)
        return None

    warnings: list[str] = []
    allowed_topics = set(catalog)
    # Тема чужого класса — почти всегда выдумка модели: во всех карточках темы
    # совпадают с классом ребёнка. Молча записанная, она испортит карту компетенций.
    grade = str(cards[learner_id].get("grade", ""))
    grade_marker = f".g{grade}." if grade else ""

    def _wrong_grade(topic_id: str) -> bool:
        return bool(grade_marker) and grade_marker not in topic_id

    observations: list[ObservedAttempt] = []
    for raw in block.get("observations") or []:
        if not isinstance(raw, dict):
            continue
        topic_id = _clean(raw.get("topic_id"), 120)
        help_level = _clean(raw.get("help_level"), 40)
        if topic_id not in allowed_topics:
            warnings.append(f"наблюдение отброшено: тема «{topic_id}» не из каталога")
            continue
        if _wrong_grade(topic_id):
            warnings.append(
                f"наблюдение отброшено: тема «{topic_id}» не для {grade} класса"
            )
            continue
        if help_level not in zpd.HELP_BY_KEY:
            warnings.append(f"наблюдение по «{topic_id}» отброшено: неизвестный уровень помощи «{help_level}»")
            continue
        solved = raw.get("solved")
        if not isinstance(solved, bool):
            warnings.append(f"наблюдение по «{topic_id}»: solved не булево, считаем «справился»")
            solved = True
        observations.append(
            ObservedAttempt(topic_id=topic_id, help_level=help_level, solved=solved, note=_clean(raw.get("note")))
        )

    error_patterns: list[tuple[str, str]] = []
    for raw in block.get("new_error_patterns") or []:
        if not isinstance(raw, dict):
            continue
        topic_id = _clean(raw.get("topic_id"), 120)
        error_tag = _clean(raw.get("error_tag"))
        if topic_id in allowed_topics and not _wrong_grade(topic_id) and error_tag:
            error_patterns.append((topic_id, error_tag))
        elif error_tag:
            warnings.append(f"ошибка «{error_tag}» отброшена: тема «{topic_id}» не подходит ученику")

    scaffolding: list[tuple[str, str]] = []
    for raw in block.get("effective_scaffolding") or []:
        if not isinstance(raw, dict):
            continue
        topic_id = _clean(raw.get("topic_id"), 120)
        strategy = _clean(raw.get("strategy"))
        if topic_id in allowed_topics and not _wrong_grade(topic_id) and strategy:
            scaffolding.append((topic_id, strategy))

    return CardUpdateProposal(
        learner_id=learner_id,
        observations=tuple(observations),
        new_error_patterns=tuple(error_patterns),
        effective_scaffolding=tuple(scaffolding),
        journal_note=_clean(block.get("journal_note"), 1000),
        progress_note=_clean(block.get("progress_note")),
        new_interests=tuple(
            item for item in (_clean(value, 60) for value in block.get("new_interests") or []) if item
        ),
        new_story_preferences=tuple(
            item for item in (_clean(value, 60) for value in block.get("new_story_preferences") or []) if item
        ),
        new_goal=_clean(block.get("new_goal")),
        lesson_minutes=_lesson_minutes(block.get("lesson_minutes"), warnings),
        warnings=tuple(warnings),
    )


def propose_updates(
    notes: str,
    learner_ids: Sequence[str],
    *,
    complete: CompleteFn | None = None,
    batch_size: int | None = None,
) -> dict[str, CardUpdateProposal]:
    """Ask the model to split lesson notes into per-learner facts.

    Большие группы разбиваются на пачки по :data:`ROSTER_BATCH_SIZE` учеников, и
    на каждую делается свой запрос с полным текстом заметок. Причина не в
    производительности: чем больше детей в одном ростере, тем выше шанс, что
    модель припишет ошибку одного ребёнка другому, и тем ближе промпт к пределу
    контекста локальной модели (17 учеников — это уже около 4000 токенов входа).
    """
    if not notes or not notes.strip():
        raise CardUpdateError("заметки об уроке пустые — нечего разбирать")
    if not learner_ids:
        raise CardUpdateError("не указан ни один ученик")

    size = max(1, batch_size if batch_size is not None else ROSTER_BATCH_SIZE)
    batches = [list(learner_ids[start:start + size]) for start in range(0, len(learner_ids), size)]
    if len(batches) > 1:
        logger.info("Группа из %d учеников разбита на %d запросов по %d",
                    len(learner_ids), len(batches), size)

    proposals: dict[str, CardUpdateProposal] = {}
    for batch in batches:
        roster_text, cards, catalog = build_roster(batch)
        payload = complete_json(
            build_update_prompt(notes, roster_text),
            load_update_prompt(),
            complete=complete,
            temperature=0.1,
        )

        raw_learners = payload.get("learners")
        if not isinstance(raw_learners, list):
            raise CardUpdateError("модель не вернула список learners")

        for block in raw_learners:
            if not isinstance(block, dict):
                continue
            proposal = _parse_learner_block(block, cards, catalog)
            if proposal is not None and not proposal.is_empty():
                proposals[proposal.learner_id] = proposal
    return proposals


# --- Пошаговое согласование ---------------------------------------------------

# Решение по одному пункту разбора: (вид, человекочитаемое описание) -> брать ли.
DecideFn = Callable[[str, str], bool]


def filter_proposal(proposal: CardUpdateProposal, decide: DecideFn) -> CardUpdateProposal:
    """Оставить в разборе только то, что подтвердил человек.

    Фильтруем именно разбор, а не готовые изменения: так подтверждение остаётся
    про факты («Тёма справился с наводящим вопросом»), а не про их последствия
    («mastery 0.62 → 0.65»), считать которые всё равно должен код.
    """
    observations = tuple(
        item for item in proposal.observations
        if decide("observation", f"{item.topic_id}: {zpd_help_title(item.help_level)}"
                                 f"{'' if item.solved else ', не справился'}"
                                 f"{f' — {item.note}' if item.note else ''}")
    )
    errors = tuple(
        item for item in proposal.new_error_patterns
        if decide("error", f"ошибка «{item[1]}» по теме {item[0]}")
    )
    scaffolding = tuple(
        item for item in proposal.effective_scaffolding
        if decide("scaffolding", f"сработавшая опора по {item[0]}: {item[1]}")
    )
    journal = proposal.journal_note if (
        not proposal.journal_note or decide("journal", f"запись в журнал: {proposal.journal_note}")
    ) else ""
    progress = proposal.progress_note if (
        not proposal.progress_note or decide("progress", f"последний прогресс: {proposal.progress_note}")
    ) else ""
    goal = proposal.new_goal if (
        not proposal.new_goal or decide("goal", f"новая цель: {proposal.new_goal}")
    ) else ""
    interests = tuple(
        item for item in proposal.new_interests if decide("interest", f"новый интерес: {item}")
    )
    stories = tuple(
        item for item in proposal.new_story_preferences if decide("story", f"новый сюжет: {item}")
    )

    return CardUpdateProposal(
        learner_id=proposal.learner_id,
        observations=observations,
        new_error_patterns=errors,
        effective_scaffolding=scaffolding,
        journal_note=journal,
        progress_note=progress,
        new_interests=interests,
        new_story_preferences=stories,
        new_goal=goal,
        lesson_minutes=proposal.lesson_minutes,
        warnings=proposal.warnings,
    )


def zpd_help_title(help_key: str) -> str:
    """Человеческое название ступени помощи для вопроса репетитору."""
    level = load_learner_data_tools().zpd.HELP_BY_KEY.get(help_key)
    return level.title_ru if level else help_key


# --- Applying a proposal to a card ------------------------------------------

def _competency_index(card: dict) -> dict[str, int]:
    competencies = ((card.get("learner_model") or {}).get("competencies")) or []
    return {
        item["topic_id"]: index
        for index, item in enumerate(competencies)
        if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
    }


def _ensure_learner_model(card: dict) -> dict:
    """Create the learner_model skeleton so a short card can start collecting data."""
    learner_model = card.get("learner_model")
    if not isinstance(learner_model, dict):
        learner_model = {}
        card["learner_model"] = learner_model
    learner_model.setdefault("competencies", [])
    learner_model.setdefault("zpd", {"current": [], "outside": []})
    for key in ("learning_preferences", "engagement", "ai_usage", "gamification"):
        learner_model.setdefault(key, {})
    for key in ("projects", "strengths", "support_needs", "scaffolding_by_topic"):
        learner_model.setdefault(key, [])
    return learner_model


def _apply_observations(
    card: dict,
    proposal: CardUpdateProposal,
    *,
    mode: str,
    today: str,
    confirmed_topics: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    tools = load_learner_data_tools()
    zpd = tools.zpd

    changes: list[str] = []
    pending: list[str] = []
    confirmed = None if confirmed_topics is None else set(confirmed_topics)

    learner_model = _ensure_learner_model(card)
    competencies = learner_model["competencies"]

    for observation in proposal.observations:
        index = _competency_index(card).get(observation.topic_id)
        before = competencies[index] if index is not None else None
        updated = zpd.update_competency(
            before,
            zpd.Observation(
                topic_id=observation.topic_id,
                help_key=observation.help_level,
                solved=observation.solved,
                date=today,
                note=observation.note,
            ),
        )
        summary = zpd.explain_update(before, updated)

        write_it = mode == "auto_ema" or (
            confirmed is not None and (CONFIRM_ALL in confirmed or observation.topic_id in confirmed)
        )
        if not write_it:
            pending.append(f"{summary} — ожидает подтверждения репетитора")
            continue

        if index is None:
            competencies.append(updated)
        else:
            competencies[index] = updated
        changes.append(summary)

    return changes, pending


def _apply_error_patterns(card: dict, proposal: CardUpdateProposal, today: str) -> list[str]:
    changes: list[str] = []
    patterns = card.setdefault("error_patterns", [])

    for topic_id, error_tag in proposal.new_error_patterns:
        existing = next(
            (
                item
                for item in patterns
                if isinstance(item, dict)
                and item.get("topic_id") == topic_id
                and item.get("error_tag") == error_tag
            ),
            None,
        )
        if existing is None:
            patterns.append({
                "subject": "math",
                "topic_id": topic_id,
                "error_tag": error_tag,
                "count": 1,
                "last_seen": today,
            })
            changes.append(f"новая типичная ошибка: «{error_tag}» ({topic_id})")
        else:
            existing["count"] = int(existing.get("count", 1)) + 1
            existing["last_seen"] = today
            changes.append(f"ошибка «{error_tag}» повторилась, счётчик: {existing['count']}")
    return changes


def _apply_scaffolding(card: dict, proposal: CardUpdateProposal) -> list[str]:
    """Record what worked, on the competency itself so task design can reuse it."""
    changes: list[str] = []
    if not proposal.effective_scaffolding:
        return changes

    learner_model = _ensure_learner_model(card)
    competencies = learner_model["competencies"]
    index = _competency_index(card)

    for topic_id, strategy in proposal.effective_scaffolding:
        position = index.get(topic_id)
        if position is None:
            continue  # no assessment for this topic yet — nothing to attach it to
        competency = competencies[position]
        strategies = competency.setdefault("effective_scaffolding", [])
        if strategy in strategies:
            continue
        strategies.append(strategy)
        del strategies[:-MAX_SCAFFOLDING_PER_TOPIC]
        changes.append(f"сработавшая опора по «{topic_id}»: {strategy}")
    return changes


def _sync_derived(card: dict) -> list[str]:
    """Recompute everything that must follow from the competence map.

    ``learner_model.zpd`` and ``knowledge[].status`` are projections of the map
    (concept section 2 is authoritative). Recomputing them here is what stops a
    card from telling two different stories after an update.
    """
    tools = load_learner_data_tools()
    zpd = tools.zpd

    changes: list[str] = []
    learner_model = card.get("learner_model")
    if not isinstance(learner_model, dict):
        return changes

    derived = zpd.derive_zpd(card)
    if learner_model.get("zpd") != derived:
        learner_model["zpd"] = derived
        changes.append(
            f"ЗБР пересчитана: сейчас {derived['current'] or '—'}, за пределами {derived['outside'] or '—'}"
        )

    knowledge = card.setdefault("knowledge", [])
    by_topic = {
        item["topic_id"]: item
        for item in knowledge
        if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
    }
    for verdict in zpd.classify_card(card):
        status = zpd.STATUS_BY_ZONE.get(verdict.zone)
        if status is None:
            continue
        entry = by_topic.get(verdict.topic_id)
        if entry is None:
            knowledge.append({"topic_id": verdict.topic_id, "status": status})
            changes.append(f"тема «{verdict.topic_id}» добавлена в knowledge со статусом {status}")
        elif zpd.status_conflicts(verdict.zone, entry.get("status")):
            # Only contradictions are rewritten. "learning" vs "needs_support"
            # inside the zone is the tutor's judgement and is left alone.
            changes.append(f"статус «{verdict.topic_id}»: {entry.get('status')} → {status}")
            entry["status"] = status
    return changes


def already_applied(card: dict, proposal: CardUpdateProposal, today: str) -> str | None:
    """Return a reason when this proposal looks like it was already written.

    Re-running the same lesson notes must not move mastery twice or inflate the
    error counters — a tutor who repeats the command (or presses it after a
    typo) would silently corrupt the child's history.

    Two independent signals, because the model does not word the journal note
    identically every time:

    * an identical journal entry (same date, same text) already exists;
    * every observation in the proposal already has a history entry with the
      same date and the same help level.
    """
    journal = ((card.get("mvp") or {}).get("journal")) or []
    if proposal.journal_note and any(
        isinstance(entry, dict)
        and entry.get("date") == today
        and entry.get("note") == proposal.journal_note
        for entry in journal
    ):
        return "в журнале уже есть эта запись за сегодня"

    if not proposal.observations:
        return None

    competencies = {
        item["topic_id"]: item
        for item in ((card.get("learner_model") or {}).get("competencies") or [])
        if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
    }
    for observation in proposal.observations:
        history = (competencies.get(observation.topic_id) or {}).get("history") or []
        if not any(
            isinstance(entry, dict)
            and entry.get("date") == today
            and entry.get("help_level") == observation.help_level
            for entry in history
        ):
            return None
    return "все наблюдения уже записаны сегодня с тем же уровнем помощи"


def apply_proposal(
    card: dict,
    proposal: CardUpdateProposal,
    *,
    mode: str | None = None,
    today: str | None = None,
    confirmed_topics: Iterable[str] | None = None,
    force: bool = False,
) -> AppliedUpdate:
    """Apply one proposal to a copy of ``card`` and return the new card plus a diff.

    The input card is never mutated: callers compare, print and only then write.
    ``force`` applies the update even when it looks like a repeat.
    """
    resolved_mode = (mode or MASTERY_MODE).strip().lower()
    if resolved_mode not in MASTERY_MODES:
        raise CardUpdateError(f"неизвестный режим оценки «{resolved_mode}»; допустимы: {', '.join(MASTERY_MODES)}")

    stamp = today or _today()
    updated = json.loads(json.dumps(card, ensure_ascii=False))  # deep copy, JSON-safe by construction

    if not force:
        repeat_reason = already_applied(card, proposal, stamp)
        if repeat_reason is not None:
            return AppliedUpdate(
                learner_id=proposal.learner_id,
                card=updated,
                changes=(),
                pending=(),
                warnings=proposal.warnings + (
                    f"похоже, эти итоги уже применены: {repeat_reason}. "
                    "Повторите с --force, если запись нужна ещё раз.",
                ),
            )

    changes, pending = _apply_observations(
        updated, proposal, mode=resolved_mode, today=stamp, confirmed_topics=confirmed_topics
    )
    changes += _apply_error_patterns(updated, proposal, stamp)
    changes += _apply_scaffolding(updated, proposal)

    if proposal.journal_note:
        journal = updated.setdefault("mvp", {}).setdefault("journal", [])
        journal.append({"date": stamp, "note": proposal.journal_note})
        changes.append(f"запись в журнал: {proposal.journal_note}")

    # Занятие как отдельная запись (пункт 4 концепта): по каким темам работали и
    # сколько это заняло. Журнал остаётся свободным текстом, а это — данные,
    # по которым потом считается статистика.
    covered = list(dict.fromkeys(item.topic_id for item in proposal.observations))
    if covered or proposal.lesson_minutes:
        lesson = {"date": stamp}
        if covered:
            lesson["topic_ids"] = covered
        if proposal.lesson_minutes:
            lesson["minutes"] = proposal.lesson_minutes
        if proposal.journal_note:
            lesson["note"] = proposal.journal_note
        lessons = updated.setdefault("mvp", {}).setdefault("lessons", [])
        if not any(entry.get("date") == stamp and entry.get("topic_ids") == covered
                   for entry in lessons if isinstance(entry, dict)):
            lessons.append(lesson)
            duration = f", {proposal.lesson_minutes} мин" if proposal.lesson_minutes else ""
            changes.append(f"занятие записано: тем {len(covered)}{duration}")

    if proposal.progress_note:
        rag_context = updated.setdefault("rag_context", {})
        rag_context["recent_progress"] = {"date": stamp, "note": proposal.progress_note}
        changes.append(f"последний прогресс: {proposal.progress_note}")

    if proposal.new_goal:
        goal_block = updated.setdefault("rag_context", {}).setdefault("current_goal", {})
        if goal_block.get("goal") != proposal.new_goal:
            changes.append(f"цель: «{goal_block.get('goal', '—')}» → «{proposal.new_goal}»")
            goal_block["goal"] = proposal.new_goal
            goal_block["updated_at"] = stamp

    interests = updated.setdefault("interests", [])
    for interest in proposal.new_interests:
        if interest not in interests:
            interests.append(interest)
            changes.append(f"новый интерес: {interest}")

    story_preferences = updated.setdefault("mvp", {}).setdefault("story_preferences", [])
    for story in proposal.new_story_preferences:
        if story not in story_preferences:
            story_preferences.append(story)
            changes.append(f"новый любимый сюжет: {story}")

    changes += _sync_derived(updated)

    return AppliedUpdate(
        learner_id=proposal.learner_id,
        card=updated,
        changes=tuple(changes),
        pending=tuple(pending),
        warnings=proposal.warnings,
    )


# --- Persisting --------------------------------------------------------------

def card_path(learner_id: str) -> Path:
    return LEARNER_DATA_DIR / "learners" / f"{learner_id}.json"


# Копии лежат отдельной папкой, а не рядом с карточками: файлы учеников
# перечисляются глобом learners/*.json, и резервная копия попала бы в список
# детей со всеми вытекающими.
BACKUP_DIR_NAME = ".backups"
BACKUPS_KEPT = 10


def backup_dir() -> Path:
    return LEARNER_DATA_DIR / BACKUP_DIR_NAME


def backup_card(learner_id: str, *, stamp: str | None = None) -> Path | None:
    """Сохранить текущую версию карточки перед перезаписью.

    Git спасает только того, кто коммитит. Репетитор — не обязан, поэтому
    откатить неудачное обновление должно быть можно и без него.
    """
    source = card_path(learner_id)
    if not source.is_file():
        return None

    backups = backup_dir()
    backups.mkdir(parents=True, exist_ok=True)
    moment = stamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backups / f"{learner_id}-{moment}.json"
    target.write_text(source.read_text(encoding="utf-8-sig"), encoding="utf-8")

    previous = sorted(backups.glob(f"{learner_id}-*.json"))
    for stale in previous[:-BACKUPS_KEPT]:
        stale.unlink(missing_ok=True)
    return target


def validate_or_raise(card: dict, learner_id: str) -> None:
    """Fail closed: an update that breaks the card contract is never written."""
    tools = load_learner_data_tools()
    catalog = tools.load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")
    findings = tools.validate_card(card, catalog, f"{learner_id}.json")
    errors = [finding.format() for finding in findings if finding.is_error()]
    if errors:
        raise CardUpdateError(
            "обновление не записано, карточка не прошла валидацию: " + "; ".join(errors)
        )


def write_card(learner_id: str, card: dict) -> Path:
    """Validate and write one card. The file is normalised to 2-space JSON."""
    validate_or_raise(card, learner_id)
    path = card_path(learner_id)
    if not path.is_file():
        raise CardUpdateError(f"карточка «{learner_id}» не найдена по пути {path}")

    backup = backup_card(learner_id)
    if backup is not None:
        logger.info("Backup saved: %s", backup)
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Learner card written: %s", path)
    return path


def update_from_notes(
    notes: str,
    learner_ids: Sequence[str],
    *,
    apply: bool = False,
    mode: str | None = None,
    confirmed_topics: Iterable[str] | None = None,
    complete: CompleteFn | None = None,
    today: str | None = None,
    force: bool = False,
    batch_size: int | None = None,
    decide: DecideFn | None = None,
) -> list[AppliedUpdate]:
    """Full pipeline: notes → proposals → applied updates (optionally written).

    ``apply=False`` is the default on purpose: the tutor reads the diff first.
    """
    proposals = propose_updates(notes, learner_ids, complete=complete, batch_size=batch_size)
    if not proposals:
        return []

    results: list[AppliedUpdate] = []
    for learner_id, proposal in proposals.items():
        if decide is not None:
            proposal = filter_proposal(proposal, decide)
            if proposal.is_empty():
                continue
        try:
            card, _catalog = load_validated_card(learner_id)
        except LearnerContextError as exc:
            raise CardUpdateError(str(exc)) from exc
        result = apply_proposal(
            card, proposal, mode=mode, today=today, confirmed_topics=confirmed_topics, force=force
        )
        if apply and result.changed():
            write_card(learner_id, result.card)
        results.append(result)
    return results


def render_update(result: AppliedUpdate, *, applied: bool) -> str:
    """Tutor-facing summary of one card update."""
    header = "ЗАПИСАНО" if applied else "ПРЕДПРОСМОТР (не записано)"
    lines = [f"[{header}] {result.learner_id}"]
    if result.changes:
        lines.extend(f"  + {change}" for change in result.changes)
    else:
        lines.append("  изменений нет")
    if result.pending:
        lines.append("  ожидает подтверждения (--confirm):")
        lines.extend(f"    ? {item}" for item in result.pending)
    if result.warnings:
        lines.append("  замечания к ответу модели:")
        lines.extend(f"    ! {item}" for item in result.warnings)
    return "\n".join(lines)


def ask_tutor(kind: str, description: str) -> bool:
    """Спросить репетитора про один пункт разбора. y — принять, n — отклонить.

    Пустой ответ означает «принять»: репетитор уже прочитал предпросмотр, и
    прощёлкивать Enter по согласованным пунктам должно быть быстро.
    """
    labels = {
        "observation": "Наблюдение",
        "error": "Ошибка",
        "scaffolding": "Опора",
        "journal": "Журнал",
        "progress": "Прогресс",
        "goal": "Цель",
        "interest": "Интерес",
        "story": "Сюжет",
    }
    prompt = f"{labels.get(kind, kind)}: {description}\n  принять? [Y/n] "
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return True
    return answer not in ("n", "no", "н", "нет")
