#!/usr/bin/env python3
"""ЗБР policy: the zone of proximal development derived from the competence map.

Concept mapping. Section 2 of the concept document ("Карта компетенций") is the
single source of truth for the zone. Section 3 ("Модель ЗБР") is therefore a
*derived mirror*: :func:`derive_zpd` recomputes ``learner_model.zpd`` from
``learner_model.competencies``, so the two can never drift apart and nobody has
to maintain the zone by hand.

Everything here is deliberately deterministic and stdlib-only:

* the tutor can be told exactly why a topic landed in the zone (``reason``);
* the LLM never invents the numbers. It reports *observations* ("solved after a
  leading question"), and :func:`update_competency` turns observations into
  numbers. That keeps mastery reproducible between runs.

Vygotsky/scaffolding vocabulary used below comes from the lesson deck
("ЗБР_Сценарий урока"): the hint ladder is наводящий вопрос → визуальная опора →
частичное решение, and scaffolds are removed as soon as the child copes.

CLI: ``python learner-data/tools/zpd.py <learner_id> [--format text|json]``
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learner_common import (
        LEARNER_ID_RE,
        configure_utf8_streams,
        find_package_root,
        load_catalog,
        load_learner,
    )
else:
    from .learner_common import (
        LEARNER_ID_RE,
        configure_utf8_streams,
        find_package_root,
        load_catalog,
        load_learner,
    )


# --- Zones ------------------------------------------------------------------

ZONE_MASTERED = "mastered"   # уже делает сам — опоры снимаем
ZONE_ZPD = "zpd"             # зона ближайшего развития — делает вместе со взрослым
ZONE_OUTSIDE = "outside"     # пока за пределами ЗБР — нужна предыдущая тема
ZONE_UNKNOWN = "unknown"     # нет данных для решения

ZONE_LABELS_RU = {
    ZONE_MASTERED: "освоено (делает сам)",
    ZONE_ZPD: "в ЗБР (делает с опорой)",
    ZONE_OUTSIDE: "пока за пределами ЗБР",
    ZONE_UNKNOWN: "нет данных",
}

# ``knowledge[].status`` values that do not contradict a derived zone.
# The tutor's nuance is kept on purpose: a topic inside the zone may reasonably
# be marked either "learning" or "needs_support". Only contradictions are
# flagged by the validator and repaired by the card updater — for example
# "not_started" for a topic that already has an assessment, or "confident" for
# a topic the child cannot do alone.
COMPATIBLE_STATUSES = {
    ZONE_MASTERED: ("confident",),
    ZONE_ZPD: ("learning", "needs_support"),
    ZONE_OUTSIDE: ("needs_support",),
}
STATUS_BY_ZONE = {
    ZONE_MASTERED: "confident",
    ZONE_ZPD: "learning",
    ZONE_OUTSIDE: "needs_support",
}


def status_conflicts(zone: str, status: object) -> bool:
    """True when a knowledge status contradicts the zone derived from the map."""
    allowed = COMPATIBLE_STATUSES.get(zone)
    if allowed is None:  # unknown zone: nothing was assessed, nothing to contradict
        return False
    return status not in allowed

# Thresholds are module constants, not env vars, on purpose: the same numbers
# must mean the same thing in the validator, in the RAG service and in the
# tutor's report. Tuned for grade 3-4 math with a human tutor in the room.
MASTERED_MASTERY = 0.85
MASTERED_INDEPENDENCE = 0.80
OUTSIDE_MASTERY = 0.30

# Weight of one fresh observation in the mastery/independence EMA. 0.4 means a
# single lesson can move mastery noticeably but cannot overwrite the history.
DEFAULT_ALPHA = 0.4

# Where a never-assessed topic starts. Deliberately neutral (0.5, "не знаем"),
# not the first observation's credit: one unaided success is not mastery. With
# alpha 0.4 it takes three clean solo runs to cross MASTERED_MASTERY, which is
# the rule we want ("завтра сам" must be demonstrated more than once).
PRIOR_MASTERY = 0.5
PRIOR_INDEPENDENCE = 0.5

# One bad attempt must not erase three good lessons. Without this cap a single
# unsolved task drops mastery from 0.94 to 0.56, and the child gets demoted from
# "делает сам" to the middle of the zone because of one distracted afternoon.
MAX_MASTERY_DROP = 0.15

# How many past assessments one competency keeps. Older entries are dropped so
# a card stays readable and small.
HISTORY_LIMIT = 10

# У оценки нет забывания: среднее не убывает само по себе, поэтому июльское
# число в октябре весит столько же, сколько вчерашнее. Автоматически занижать
# его нельзя — это было бы выдуманным наблюдением. Но молчать тоже нельзя,
# поэтому старая оценка помечается как требующая перепроверки.
STALE_AFTER_DAYS = 30


# --- Hint ladder (цифровой скаффолдинг) -------------------------------------

@dataclass(frozen=True)
class HelpLevel:
    """One rung of the scaffolding ladder.

    ``credit`` is how much a solved task counts towards mastery. Solving alone
    is worth 1.0; the more support the child needed, the less the success says
    about independent mastery.
    """

    level: int
    key: str
    title_ru: str
    credit: float


HELP_LADDER: tuple[HelpLevel, ...] = (
    HelpLevel(0, "independent", "решил сам, без опор", 1.0),
    HelpLevel(1, "hint_question", "наводящий вопрос", 0.7),
    HelpLevel(2, "visual", "визуальная опора или банк слов", 0.5),
    HelpLevel(3, "joint", "частичное решение вместе со взрослым", 0.25),
)
HELP_BY_KEY = {level.key: level for level in HELP_LADDER}
HELP_BY_LEVEL = {level.level: level for level in HELP_LADDER}

# Rungs 1..3 are the ladder the AI is allowed to walk down (slide "Цифровой
# скаффолдинг"). Rung 0 is not a hint — it is the child working unaided.
OFFERABLE_HELP = tuple(level for level in HELP_LADDER if level.level > 0)

UNSOLVED_CREDIT = 0.0


class ZpdPolicyError(ValueError):
    """Raised when an observation cannot be mapped onto the hint ladder."""


def help_level(key_or_level: str | int) -> HelpLevel:
    """Resolve a ladder rung from its key ("hint_question") or number (1)."""
    if isinstance(key_or_level, bool):  # bool is an int in Python — reject early
        raise ZpdPolicyError(f"некорректный уровень помощи: {key_or_level!r}")
    if isinstance(key_or_level, int):
        found = HELP_BY_LEVEL.get(key_or_level)
    elif isinstance(key_or_level, str):
        found = HELP_BY_KEY.get(key_or_level.strip())
    else:
        found = None
    if found is None:
        allowed = ", ".join(level.key for level in HELP_LADDER)
        raise ZpdPolicyError(f"неизвестный уровень помощи {key_or_level!r}; допустимы: {allowed}")
    return found


def credit_for(key_or_level: str | int, solved: bool = True) -> float:
    """Return the mastery credit of one attempt.

    A task the child did not finish even with support gives 0.0 — it is
    evidence, not a punishment: the EMA moves down slowly.
    """
    level = help_level(key_or_level)
    return level.credit if solved else UNSOLVED_CREDIT


# --- Classification ---------------------------------------------------------

@dataclass(frozen=True)
class ZoneVerdict:
    """Why exactly one topic is in (or out of) the zone, in tutor-readable form."""

    topic_id: str
    zone: str
    reason: str
    mastery: float | None
    independence: float | None
    needs_scaffolding: bool
    hint_depth: int          # how many ladder rungs to prepare, 0..3
    difficulty: str          # "easy" | "medium" | "hard" — matches the task bank
    recommended_support: str | None
    last_assessed: str | None = None

    def label_ru(self) -> str:
        return ZONE_LABELS_RU.get(self.zone, self.zone)

    def stale_days(self, today: str | None = None) -> int | None:
        """Сколько дней прошло с последней оценки. None, если даты нет."""
        return days_since(self.last_assessed, today)

    def is_stale(self, today: str | None = None, limit: int = STALE_AFTER_DAYS) -> bool:
        days = self.stale_days(today)
        return days is not None and days > limit


def days_since(date_text: object, today: str | None = None) -> int | None:
    """Дней между датой оценки и сегодняшним днём; None при некорректной дате."""
    if not isinstance(date_text, str):
        return None
    try:
        assessed = datetime.date.fromisoformat(date_text)
        now = datetime.date.fromisoformat(today) if today else datetime.date.today()
    except ValueError:
        return None
    return (now - assessed).days


def _unit_number(value: object) -> float | None:
    """Return a 0..1 float, or None when the field is absent/unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0.0 or number > 1.0:
        return None
    return number


def _clamp_unit(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def classify(competency: object) -> ZoneVerdict:
    """Decide the zone of one competency-map entry.

    Rules (deliberately explainable to a tutor):

    * no usable ``mastery`` → ``unknown``: guessing a zone from nothing is worse
      than admitting the topic was never assessed;
    * ``mastery >= 0.85`` **and** the child works unaided (``independence >=
      0.80`` when recorded, and ``needs_scaffolding`` is not set) → ``mastered``;
    * ``mastery < 0.30`` → ``outside``: the gap is too wide for one lesson, the
      prerequisite topic comes first;
    * everything in between → ``zpd``. High mastery with low independence stays
      in the zone: "can do it with help, not yet alone" *is* the zone.
    """
    if not isinstance(competency, dict):
        raise ZpdPolicyError("запись карты компетенций должна быть объектом")

    topic_id = competency.get("topic_id")
    if not isinstance(topic_id, str) or not topic_id:
        raise ZpdPolicyError("в записи карты компетенций нет topic_id")

    mastery = _unit_number(competency.get("mastery"))
    independence = _unit_number(competency.get("independence"))
    stored_needs_scaffolding = competency.get("needs_scaffolding")
    needs_scaffolding_flag = stored_needs_scaffolding is True
    support = competency.get("recommended_scaffolding")
    support = support if isinstance(support, str) and support else None
    last_assessed = competency.get("last_assessed")
    last_assessed = last_assessed if isinstance(last_assessed, str) and last_assessed else None

    if mastery is None:
        return ZoneVerdict(
            topic_id=topic_id,
            zone=ZONE_UNKNOWN,
            reason="нет оценки mastery — тему ещё не диагностировали",
            mastery=None,
            independence=independence,
            needs_scaffolding=True,
            hint_depth=len(OFFERABLE_HELP),
            difficulty="easy",
            recommended_support=support,
            last_assessed=last_assessed,
        )

    works_alone = independence is None or independence >= MASTERED_INDEPENDENCE
    if mastery >= MASTERED_MASTERY and works_alone and not needs_scaffolding_flag:
        return ZoneVerdict(
            topic_id=topic_id,
            zone=ZONE_MASTERED,
            reason=(
                f"mastery {mastery:.2f} >= {MASTERED_MASTERY} и ребёнок работает без опор"
                " — опоры снимаем, тему можно закреплять в фоне"
            ),
            mastery=mastery,
            independence=independence,
            needs_scaffolding=False,
            hint_depth=1,
            difficulty="hard",
            recommended_support=support,
            last_assessed=last_assessed,
        )

    if mastery < OUTSIDE_MASTERY:
        return ZoneVerdict(
            topic_id=topic_id,
            zone=ZONE_OUTSIDE,
            reason=(
                f"mastery {mastery:.2f} < {OUTSIDE_MASTERY} — разрыв слишком велик,"
                " сначала предыдущая тема"
            ),
            mastery=mastery,
            independence=independence,
            needs_scaffolding=True,
            hint_depth=len(OFFERABLE_HELP),
            difficulty="easy",
            recommended_support=support,
            last_assessed=last_assessed,
        )

    if mastery >= MASTERED_MASTERY:
        reason = (
            f"mastery {mastery:.2f} высокая, но самостоятельность"
            f" {'не подтверждена' if independence is None else f'{independence:.2f} < {MASTERED_INDEPENDENCE}'}"
            " — делает с опорой, значит это ЗБР"
        )
    else:
        reason = (
            f"mastery {mastery:.2f} между {OUTSIDE_MASTERY} и {MASTERED_MASTERY}"
            " — сегодня делает вместе со взрослым, завтра сам"
        )

    return ZoneVerdict(
        topic_id=topic_id,
        zone=ZONE_ZPD,
        reason=reason,
        mastery=mastery,
        independence=independence,
        needs_scaffolding=True,
        hint_depth=2 if mastery >= 0.60 else len(OFFERABLE_HELP),
        difficulty="medium" if mastery >= 0.60 else "easy",
        recommended_support=support,
        last_assessed=last_assessed,
    )


def classify_card(card: dict) -> list[ZoneVerdict]:
    """Classify every competency of one learner card, keeping card order."""
    competencies = (card.get("learner_model") or {}).get("competencies") or []
    verdicts: list[ZoneVerdict] = []
    for competency in competencies:
        try:
            verdicts.append(classify(competency))
        except ZpdPolicyError:
            continue  # validate.py reports malformed entries; policy stays silent
    return verdicts


def derive_zpd(card: dict) -> dict[str, list[str]]:
    """Recompute ``learner_model.zpd`` (concept section 3) from the competence map.

    Card order is preserved instead of sorting by mastery so the tutor sees the
    same order they typed, and diffs stay small.
    """
    zpd: dict[str, list[str]] = {"current": [], "outside": []}
    for verdict in classify_card(card):
        if verdict.zone == ZONE_ZPD and verdict.topic_id not in zpd["current"]:
            zpd["current"].append(verdict.topic_id)
        elif verdict.zone == ZONE_OUTSIDE and verdict.topic_id not in zpd["outside"]:
            zpd["outside"].append(verdict.topic_id)
    return zpd


def find_verdict(card: dict, topic_id: str) -> ZoneVerdict | None:
    """Return the verdict for one topic, or None when it is not in the map."""
    for verdict in classify_card(card):
        if verdict.topic_id == topic_id:
            return verdict
    return None


# --- Support preferences (карта компетенций → «как объяснять») ---------------

# learner_model.learning_preferences keys mapped onto the ladder's rung 2, i.e.
# what kind of visual/verbal support to build for this child.
SUPPORT_FROM_PREFERENCES = (
    ("visual", "визуальная опора: схема, рисунок, чертёж"),
    ("hands_on", "предметная опора: сделать руками, измерить, вырезать"),
    ("storyline", "сюжетная опора: задача внутри истории"),
    ("audio", "проговорить вслух: объяснить шаг словами"),
    ("collaboration", "работа в паре: объяснить товарищу"),
    ("competition", "соревновательный формат: время, счёт, рекорд"),
)
PREFERENCE_MIN = 0.6

# Two support hints count as duplicates when one normalised form is a prefix of
# the other and the shared part is at least this long. Short strings are never
# merged: «рисунок» and «рисунок пути» are genuinely different hints.
DEDUPE_MIN_PREFIX = 15


def preferred_support(card: dict, topic_id: str | None = None) -> list[str]:
    """Return support types for this child, strongest first.

    Order of evidence: what already worked on this exact topic beats a general
    preference, and an explicit ``profile.prefers_visual`` beats nothing at all.
    """
    support: list[str] = []
    seen_keys: list[str] = []

    def add(item: object) -> None:
        """Add one support hint, skipping near-duplicates.

        The same trick is usually written twice in a card — once in the
        competency's ``effective_scaffolding`` and once in
        ``scaffolding_by_topic``. Comparing a normalised prefix keeps the prompt
        from repeating «сравнить луч с фонариком» three times in a row.
        """
        if not isinstance(item, str):
            return
        text = item.strip()
        if not text:
            return
        key = " ".join(
            "".join(char for char in text.lower().replace("ё", "е") if char.isalnum() or char.isspace()).split()
        )
        if not key:
            return
        for seen in seen_keys:
            shorter, longer = sorted((key, seen), key=len)
            if key == seen or (len(shorter) >= DEDUPE_MIN_PREFIX and longer.startswith(shorter)):
                return
        seen_keys.append(key)
        support.append(text)

    learner_model = card.get("learner_model") or {}

    if topic_id:
        for competency in learner_model.get("competencies") or []:
            if isinstance(competency, dict) and competency.get("topic_id") == topic_id:
                for strategy in competency.get("effective_scaffolding") or []:
                    add(strategy)
        for entry in learner_model.get("scaffolding_by_topic") or []:
            if isinstance(entry, dict) and entry.get("topic_id") == topic_id:
                for strategy in entry.get("strategies") or []:
                    add(strategy)

    preferences = learner_model.get("learning_preferences") or {}
    if isinstance(preferences, dict):
        scored = []
        for key, description in SUPPORT_FROM_PREFERENCES:
            value = _unit_number(preferences.get(key))
            if value is not None and value >= PREFERENCE_MIN:
                scored.append((value, description))
        for _value, description in sorted(scored, key=lambda pair: pair[0], reverse=True):
            add(description)

    profile = card.get("profile") or {}
    if profile.get("prefers_visual") is True:
        add("визуальная опора: схема, рисунок, чертёж")

    for strategy in (card.get("mvp") or {}).get("help_strategies") or []:
        add(strategy)

    return support


# --- Target selection -------------------------------------------------------

def select_target(card: dict, topic_id: str | None = None) -> ZoneVerdict | None:
    """Pick the topic an individual task should train.

    Priority: an explicitly requested topic (the tutor sets the lesson theme) →
    the card's ``current_goal`` → the first topic in the zone. Returns None only
    when the card has no competence map at all and no goal.
    """
    verdicts = classify_card(card)
    by_topic = {verdict.topic_id: verdict for verdict in verdicts}

    if topic_id:
        found = by_topic.get(topic_id)
        if found is not None:
            return found
        # Requested topic is not in the competence map yet: it has never been
        # assessed, so treat it as unknown rather than silently retargeting.
        return classify({"topic_id": topic_id})

    goal_topic = ((card.get("rag_context") or {}).get("current_goal") or {}).get("topic_id")
    if isinstance(goal_topic, str) and goal_topic:
        found = by_topic.get(goal_topic)
        return found if found is not None else classify({"topic_id": goal_topic})

    for verdict in verdicts:
        if verdict.zone == ZONE_ZPD:
            return verdict
    return verdicts[0] if verdicts else None


# --- Observation → numbers --------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One assessed attempt, as reported by the tutor or extracted by the LLM."""

    topic_id: str
    help_key: str
    solved: bool
    date: str
    note: str = ""

    def credit(self) -> float:
        return credit_for(self.help_key, self.solved)


def _ema(
    previous: float | None,
    evidence: float,
    alpha: float,
    prior: float,
    max_drop: float | None = None,
) -> float:
    """Exponential moving average, starting from ``prior`` when there is no value.

    Using a prior instead of the raw first observation is what keeps a single
    lucky answer from reading as full mastery. ``max_drop`` limits how far one
    observation may push the value down (see :data:`MAX_MASTERY_DROP`).
    """
    base = prior if previous is None else previous
    value = _clamp_unit((1.0 - alpha) * base + alpha * evidence)
    if max_drop is not None and previous is not None:
        value = max(value, previous - max_drop)
    return round(value, 3)


def update_competency(
    competency: dict | None,
    observation: Observation,
    *,
    alpha: float = DEFAULT_ALPHA,
    history_limit: int = HISTORY_LIMIT,
) -> dict:
    """Apply one observation to one competency entry and return a new dict.

    ``mastery`` is the EMA of per-attempt credit (see :data:`HELP_LADDER`), and
    ``independence`` is the EMA of the "solved with no help at all" indicator.
    Both are recomputed here rather than taken from the LLM, which is what makes
    the number reproducible and explainable ("0.55 → 0.67 после подсказки").

    ``needs_scaffolding`` is derived, never copied: it is simply "not yet
    mastered", so scaffolds disappear from the card the moment the child works
    unaided (slide "Цифровой скаффолдинг": опоры убираются).
    """
    if alpha <= 0.0 or alpha > 1.0:
        raise ZpdPolicyError(f"alpha должна быть в диапазоне (0, 1], получено {alpha}")

    base = dict(competency) if isinstance(competency, dict) else {"topic_id": observation.topic_id}
    if base.get("topic_id") != observation.topic_id:
        base["topic_id"] = observation.topic_id

    credit = observation.credit()
    solved_alone = 1.0 if (observation.solved and help_level(observation.help_key).level == 0) else 0.0

    mastery = _ema(_unit_number(base.get("mastery")), credit, alpha, PRIOR_MASTERY, MAX_MASTERY_DROP)
    independence = _ema(_unit_number(base.get("independence")), solved_alone, alpha, PRIOR_INDEPENDENCE)
    # Confidence is the same EMA walking towards 1.0 from zero, i.e. 1-(1-alpha)^n:
    # it says how many observations stand behind the numbers above, so a tutor can
    # tell "0.9 after one lesson" from "0.9 after six".
    confidence = _ema(_unit_number(base.get("confidence")), 1.0, alpha, 0.0)

    updated = dict(base)
    updated["mastery"] = mastery
    updated["independence"] = independence
    updated["confidence"] = confidence
    updated["last_assessed"] = observation.date

    # Classify on a copy WITHOUT the old needs_scaffolding: it is this function's
    # own previous output, and feeding it back in would latch the flag on forever
    # (a topic could never leave the zone no matter how well the child did).
    probe = dict(updated)
    probe.pop("needs_scaffolding", None)
    verdict = classify(probe)
    updated["needs_scaffolding"] = verdict.zone != ZONE_MASTERED

    history = [item for item in (base.get("history") or []) if isinstance(item, dict)]
    history.append({
        "date": observation.date,
        "mastery": mastery,
        "independence": independence,
        "help_level": help_level(observation.help_key).key,
    })
    if history_limit > 0:
        history = history[-history_limit:]
    updated["history"] = history

    return updated


def explain_update(before: dict | None, after: dict) -> str:
    """One tutor-facing Russian line describing what a card update changed."""
    old_mastery = _unit_number((before or {}).get("mastery"))
    new_mastery = _unit_number(after.get("mastery"))
    # Three decimals: a "joint help" success moves mastery by ~0.005, and a diff
    # line reading "0.24 → 0.24" would look like nothing happened.
    old_text = "—" if old_mastery is None else f"{old_mastery:.3f}"
    new_text = "—" if new_mastery is None else f"{new_mastery:.3f}"
    verdict = classify(after)
    return (
        f"{after.get('topic_id')}: mastery {old_text} → {new_text}, "
        f"зона: {verdict.label_ru()}"
    )


# --- Reporting / CLI --------------------------------------------------------

def zone_report(card: dict, catalog: dict | None = None, today: str | None = None) -> list[dict]:
    """Serialisable zone table for one card (used by the CLI and the RAG service)."""
    catalog = catalog or {}
    rows = []
    for verdict in classify_card(card):
        entry = catalog.get(verdict.topic_id) or {}
        rows.append({
            "topic_id": verdict.topic_id,
            "title_ru": entry.get("title_ru", verdict.topic_id),
            "zone": verdict.zone,
            "zone_ru": verdict.label_ru(),
            "mastery": verdict.mastery,
            "independence": verdict.independence,
            "needs_scaffolding": verdict.needs_scaffolding,
            "hint_depth": verdict.hint_depth,
            "difficulty": verdict.difficulty,
            "reason": verdict.reason,
            "last_assessed": verdict.last_assessed,
            "stale_days": verdict.stale_days(today),
            "stale": verdict.is_stale(today),
        })
    return rows


def _render_report(card: dict, rows: list[dict], derived: dict[str, list[str]]) -> str:
    lines = [f"Ученик: {card.get('pseudonym', '?')} ({card.get('learner_id', '?')}), {card.get('grade', '?')} класс", ""]
    if not rows:
        lines.append("Карта компетенций пуста — ЗБР вывести нельзя.")
        return "\n".join(lines)
    for row in rows:
        mastery = "—" if row["mastery"] is None else f"{row['mastery']:.2f}"
        independence = "—" if row["independence"] is None else f"{row['independence']:.2f}"
        lines.append(f"[{row['zone_ru']}] {row['title_ru']} [{row['topic_id']}]")
        lines.append(f"    mastery {mastery}, самостоятельность {independence}, подсказок готовим: {row['hint_depth']}")
        lines.append(f"    почему: {row['reason']}")
        if row["stale"]:
            lines.append(
                f"    ВНИМАНИЕ: последняя оценка {row['last_assessed']}, это {row['stale_days']} дней назад "
                "— число могло устареть, стоит перепроверить"
            )
    lines.append("")
    lines.append("ЗБР_сейчас: " + (", ".join(derived["current"]) or "—"))
    lines.append("Пока_за_пределами_ЗБР: " + (", ".join(derived["outside"]) or "—"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_streams()

    package_root = find_package_root()
    parser = argparse.ArgumentParser(description="Показать ЗБР ученика, выведенную из карты компетенций.")
    parser.add_argument("learner_id")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--learners-dir", type=Path, default=package_root / "learners")
    parser.add_argument("--catalog", type=Path, default=package_root / "catalog" / "math_g3_g4.json")
    parser.add_argument(
        "--write",
        action="store_true",
        help="перезаписать learner_model.zpd в карточке по карте компетенций (файл будет переформатирован в отступ 2)",
    )
    args = parser.parse_args(argv)

    if not LEARNER_ID_RE.fullmatch(args.learner_id):
        print(f"Некорректный learner_id: '{args.learner_id}'", file=sys.stderr)
        return 2

    try:
        catalog = load_catalog(args.catalog)
        card, _had_bom = load_learner(args.learners_dir, args.learner_id)
    except FileNotFoundError:
        print(f"Ученик '{args.learner_id}' не найден", file=sys.stderr)
        return 2
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Не удалось прочитать данные: {exc}", file=sys.stderr)
        return 1

    if not isinstance(card, dict):
        print("Некорректная карточка ученика: ожидается JSON-объект", file=sys.stderr)
        return 1

    rows = zone_report(card, catalog)
    derived = derive_zpd(card)

    if args.write:
        learner_model = card.get("learner_model")
        if not isinstance(learner_model, dict):
            print("В карточке нет learner_model — записывать ЗБР некуда", file=sys.stderr)
            return 1
        if learner_model.get("zpd") == derived:
            print("ЗБР в карточке уже совпадает с картой компетенций — файл не изменён")
        else:
            learner_model["zpd"] = derived
            path = (args.learners_dir / f"{args.learner_id}.json")
            path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Записано в {path}: ЗБР_сейчас {derived['current'] or '—'}, за_пределами {derived['outside'] or '—'}")

    if args.format == "json":
        print(json.dumps({"learner_id": args.learner_id, "zones": rows, "zpd": derived}, ensure_ascii=False, indent=2))
    else:
        print(_render_report(card, rows, derived))
    return 0


if __name__ == "__main__":
    sys.exit(main())
