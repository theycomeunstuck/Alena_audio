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
import copy
import datetime
import json
import sys
from dataclasses import dataclass, replace
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

# The ZPD needs a comparison between independent performance and performance
# with assistance. These are product evidence requirements, not research-derived
# diagnostic cutoffs.
MIN_PAIRED_TASKS = 2

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
    evidence: dict | None = None

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
    """Return the backward-compatible planning zone from the score map.

    This is a provisional planning projection. Use :func:`zone_report` for a
    strict evidence-backed verdict; a missing independence score is never read
    as evidence of independent work.
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

    evidence = competency_evidence(competency)
    if mastery is None:
        zone = ZONE_UNKNOWN
        reason = "нет оценки освоения; сначала нужно провести оценивание"
    elif mastery >= MASTERED_MASTERY and independence is None:
        zone = ZONE_UNKNOWN
        reason = "самостоятельность не записана; высокий балл освоения не подтверждает работу без помощи"
    elif mastery >= MASTERED_MASTERY and independence >= MASTERED_INDEPENDENCE and not needs_scaffolding_flag:
        zone = ZONE_MASTERED
        reason = f"в карте высокие оценки (освоение {mastery:.2f}, самостоятельность {independence:.2f}); предварительная зона планирования"
    elif mastery < OUTSIDE_MASTERY:
        zone = ZONE_OUTSIDE
        reason = f"оценка освоения низкая ({mastery:.2f}); повтор предыдущей темы пока только предположение, нужна проверка наблюдениями"
    else:
        zone = ZONE_ZPD
        independence_text = "не записана" if independence is None else f"{independence:.2f}"
        reason = f"оценки предполагают пользу опоры (освоение {mastery:.2f}, самостоятельность {independence_text}); предварительная зона планирования"

    return ZoneVerdict(
        topic_id=topic_id,
        zone=zone,
        reason=reason,
        mastery=mastery,
        independence=independence,
        needs_scaffolding=zone != ZONE_MASTERED,
        hint_depth=(1 if mastery is not None and mastery >= MASTERED_MASTERY
                    else 2 if mastery is not None and mastery >= 0.60 else len(OFFERABLE_HELP)),
        difficulty="hard" if zone == ZONE_MASTERED else "medium" if zone == ZONE_ZPD else "easy",
        recommended_support=support,
        last_assessed=last_assessed,
        evidence=evidence,
    )


def competency_evidence(competency: dict, today: str | None = None) -> dict:
    """Summarize support-level outcomes from the retained history window.

    Legacy history entries with a ``help_level`` but no ``solved`` field mean
    success, matching the original schema description that the child "справился"
    at that rung. New entries always store the outcome explicitly.
    """
    counts = {
        "independent": {"attempts": 0, "successes": 0},
        "assisted": {"attempts": 0, "successes": 0},
    }
    help_levels: dict[str, int] = {}
    explicit_counts = {
        "independent": {"attempts": 0, "successes": 0},
        "assisted": {"attempts": 0, "successes": 0},
    }
    legacy_observations = 0
    pair_groups: dict[tuple[str, str | None], dict[str, list[tuple[str, int, bool]]]] = {}
    now = datetime.date.fromisoformat(today) if today else datetime.date.today()
    history = competency.get("history") if isinstance(competency, dict) else None
    for event_index, item in enumerate(history if isinstance(history, list) else []):
        if not isinstance(item, dict):
            continue
        level = item.get("support_level")
        if level in ("independent", "hint_question", "visual", "joint"):
            bucket = "independent" if level == "independent" else "assisted"
        else:
            level = item.get("help_level")
            if level not in HELP_BY_KEY:
                continue
            bucket = "independent" if level == "independent" else "assisted"
        solved = item.get("solved", True)
        if type(solved) is not bool:
            continue
        counts[bucket]["attempts"] += 1
        counts[bucket]["successes"] += int(solved)
        help_levels[level] = help_levels.get(level, 0) + 1
        if "solved" not in item or "support_level" not in item:
            legacy_observations += 1
            continue
        explicit_counts[bucket]["attempts"] += 1
        explicit_counts[bucket]["successes"] += int(solved)
        pair_id = item.get("assessment_pair_id") or item.get("task_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            continue
        event_date = item.get("date")
        try:
            parsed_date = datetime.date.fromisoformat(event_date)
        except (TypeError, ValueError):
            continue
        age = (now - parsed_date).days
        if age < 0 or age > STALE_AFTER_DAYS:
            continue
        difficulty = item.get("difficulty")
        if difficulty not in (None, "easy", "medium", "hard"):
            continue
        pair_key = (pair_id, difficulty)
        pair_groups.setdefault(pair_key, {}).setdefault(level, []).append(
            (event_date, event_index, solved)
        )

    independent_count = counts["independent"]["attempts"]
    assisted_count = counts["assisted"]["attempts"]
    paired = {"independent": {"attempts": 0, "successes": 0},
              "assisted": {"attempts": 0, "successes": 0}}
    pair_records: list[tuple[tuple[str, int], dict[str, Any]]] = []
    for support_pair in pair_groups.values():
        outcomes: dict[str, Any] = {}
        latest_events = []
        rung_outcomes: dict[str, bool] = {}
        for level, observations in support_pair.items():
            event_date, event_index, solved = max(
                observations, key=lambda value: (value[0], value[1])
            )
            rung_outcomes[level] = solved
            latest_events.append((event_date, event_index))
        if "independent" not in rung_outcomes:
            continue
        assisted_rungs = [level for level in rung_outcomes if level != "independent"]
        if not assisted_rungs:
            continue
        highest_assisted = max(assisted_rungs, key=lambda level: HELP_BY_KEY[level].level)
        outcomes["independent"] = rung_outcomes["independent"]
        outcomes["assisted"] = rung_outcomes[highest_assisted]
        outcomes["assisted_level"] = highest_assisted
        outcomes["joint_failed"] = (
            rung_outcomes.get("joint") is False
            and all(not rung_outcomes[level] for level in assisted_rungs)
        )
        for bucket in ("independent", "assisted"):
            paired[bucket]["attempts"] += 1
            paired[bucket]["successes"] += int(outcomes[bucket])
        pair_records.append((max(latest_events), outcomes))
    pair_records.sort(key=lambda value: value[0])
    paired_tasks = len(pair_records)
    decision_records = pair_records[-MIN_PAIRED_TASKS:]
    decision = {
        "independent": {"attempts": 0, "successes": 0},
        "assisted": {"attempts": 0, "successes": 0},
        "highest_assisted_levels": {},
        "joint_failures": 0,
    }
    for _date, outcomes in decision_records:
        for bucket in ("independent", "assisted"):
            decision[bucket]["attempts"] += 1
            decision[bucket]["successes"] += int(outcomes[bucket])
        level = outcomes["assisted_level"]
        decision["highest_assisted_levels"][level] = (
            decision["highest_assisted_levels"].get(level, 0) + 1
        )
        decision["joint_failures"] += int(outcomes["joint_failed"])
    sufficient = paired_tasks >= MIN_PAIRED_TASKS
    recent_date = None
    if isinstance(competency, dict):
        recent_date = competency.get("last_assessed")
    age = days_since(recent_date, today)
    return {
        **counts,
        "explicit": explicit_counts,
        "paired": {"tasks": paired_tasks, **paired,
                   "decision_window": {"tasks": len(decision_records), **decision}},
        "paired_observations": paired_tasks,
        "sufficient": sufficient,
        "confidence": "sufficient" if sufficient else "insufficient",
        "help_levels": help_levels,
        "legacy_observations": legacy_observations,
        "last_assessed": recent_date if isinstance(recent_date, str) else None,
        "stale_days": age,
        "stale": age is not None and age > STALE_AFTER_DAYS,
    }


def confirmed_zone(competency: dict, today: str | None = None) -> dict:
    """Return a strict, recent zone decision with an explicit evidence trail."""
    evidence = competency_evidence(competency, today)
    if evidence["stale"]:
        zone, reason, status = ZONE_UNKNOWN, "наблюдения устарели; перед выводом зоны нужно повторное оценивание", "stale"
    elif not evidence["sufficient"]:
        zone, reason, status = ZONE_UNKNOWN, (
            "недостаточно сопоставимых пар наблюдений; нужны как минимум "
            f"{MIN_PAIRED_TASKS} пары заданий с попытками самостоятельно и с помощью"
        ), "insufficient"
    elif evidence["paired"]["decision_window"]["independent"]["successes"] == MIN_PAIRED_TASKS:
        zone, reason, status = ZONE_MASTERED, "в обеих последних сопоставимых задачах ученик справился самостоятельно", "confirmed"
    elif evidence["paired"]["decision_window"]["assisted"]["successes"] == MIN_PAIRED_TASKS:
        zone, reason, status = ZONE_ZPD, "в обеих последних задачах помощь привела к успеху, самостоятельные попытки ещё не всегда успешны", "confirmed"
    elif (
        evidence["paired"]["decision_window"]["independent"]["successes"] == 0
        and evidence["paired"]["decision_window"]["assisted"]["successes"] == 0
        and evidence["paired"]["decision_window"]["joint_failures"] == MIN_PAIRED_TASKS
    ):
        zone, reason, status = ZONE_OUTSIDE, (
            "в обеих последних сопоставимых задачах не получилось самостоятельно и при совместном решении со взрослым; "
            "это описывает только эти попытки"
        ), "confirmed"
    elif (
        evidence["paired"]["decision_window"]["independent"]["successes"] == 0
        and evidence["paired"]["decision_window"]["assisted"]["successes"] == 0
    ):
        zone, reason, status = ZONE_UNKNOWN, (
            "в последних парных задачах попытки не удались, но совместное решение со взрослым ещё не проверено "
            "в обеих парах; добавьте попытку с этой помощью"
        ), "inconclusive"
    else:
        zone, reason, status = ZONE_UNKNOWN, "результаты последних сопоставимых задач различаются; нужно ещё одно наблюдение", "inconclusive"
    return {"zone": zone, "reason": reason, "evidence_status": status, "evidence": evidence}


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
            return _task_selection_verdict(card, found)
        # Requested topic is not in the competence map yet: it has never been
        # assessed, so treat it as unknown rather than silently retargeting.
        return _task_selection_verdict(card, classify({"topic_id": topic_id}))

    goal_topic = ((card.get("rag_context") or {}).get("current_goal") or {}).get("topic_id")
    if isinstance(goal_topic, str) and goal_topic:
        found = by_topic.get(goal_topic)
        found = found if found is not None else classify({"topic_id": goal_topic})
        return _task_selection_verdict(card, found)

    for verdict in verdicts:
        if verdict.zone == ZONE_ZPD:
            return _task_selection_verdict(card, verdict)
    return _task_selection_verdict(card, verdicts[0]) if verdicts else None


def _task_selection_verdict(card: dict, planning: ZoneVerdict) -> ZoneVerdict:
    """Prefer a strict recent evidence result when choosing the task level.

    The numeric classifier remains the compatibility planning projection. Cards
    with only legacy observations continue to use that provisional projection.
    Once any explicit outcomes exist, insufficient or mixed evidence schedules
    an easy diagnostic task instead of presenting an unverified zone as settled.
    """
    model = card.get("learner_model") if isinstance(card, dict) else None
    competencies = model.get("competencies") if isinstance(model, dict) else None
    competency = next(
        (
            item for item in competencies or []
            if isinstance(item, dict) and item.get("topic_id") == planning.topic_id
        ),
        None,
    )
    if competency is None:
        return planning
    evidence = competency_evidence(competency)
    explicit = evidence["explicit"]["independent"]["attempts"] + evidence["explicit"]["assisted"]["attempts"]
    if explicit == 0:
        return planning

    confirmed = confirmed_zone(competency)
    support_evidence = {
        "planning": planning.evidence,
        "confirmed_zone": confirmed["zone"],
        "evidence_status": confirmed["evidence_status"],
        "confirmed": confirmed["evidence"],
    }
    if confirmed["evidence_status"] != "confirmed":
        return replace(
            planning,
            zone=ZONE_UNKNOWN,
            reason=(
                "Подтверждённая зона пока не установлена; для задания выбран короткий "
                f"диагностический шаг. {confirmed['reason']}"
            ),
            needs_scaffolding=True,
            hint_depth=len(OFFERABLE_HELP),
            difficulty="easy",
            evidence=support_evidence,
        )

    zone = confirmed["zone"]
    if zone == ZONE_MASTERED:
        difficulty = "hard"
        hint_depth = 1
        needs_scaffolding = False
    elif zone == ZONE_ZPD:
        difficulty = "medium"
        levels = confirmed["evidence"]["paired"]["decision_window"]["highest_assisted_levels"]
        hint_depth = max(
            (HELP_BY_KEY[level].level for level, count in levels.items() if count),
            default=1,
        )
        needs_scaffolding = True
    else:
        difficulty = "easy"
        hint_depth = len(OFFERABLE_HELP)
        needs_scaffolding = True

    return replace(
        planning,
        zone=zone,
        reason=f"Уровень задания выбран по подтверждённым парным наблюдениям. {confirmed['reason']}",
        needs_scaffolding=needs_scaffolding,
        hint_depth=hint_depth,
        difficulty=difficulty,
        evidence=support_evidence,
    )


# --- Observation → numbers --------------------------------------------------

@dataclass(frozen=True)
class Observation:
    """One assessed attempt, as reported by the tutor or extracted by the LLM."""

    topic_id: str
    help_key: str
    solved: bool
    date: str
    note: str = ""
    task_id: str | None = None
    assessment_pair_id: str | None = None
    difficulty: str | None = None

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
        "support_level": help_level(observation.help_key).key,
        "solved": observation.solved,
    })
    if observation.task_id:
        history[-1]["task_id"] = observation.task_id
    if observation.assessment_pair_id:
        history[-1]["assessment_pair_id"] = observation.assessment_pair_id
    if observation.difficulty:
        history[-1]["difficulty"] = observation.difficulty
    if observation.note:
        history[-1]["note"] = observation.note
    if history_limit > 0:
        history = history[-history_limit:]
    updated["history"] = history

    return updated


def record_observation(
    card: dict,
    *,
    topic_id: str,
    support_level: str,
    solved: bool,
    observed_on: str | None = None,
    task_id: str | None = None,
    assessment_pair_id: str | None = None,
    difficulty: str | None = None,
    notes: str | None = None,
) -> dict:
    """Append one explicit observation and return a validated-shape card copy.

    ``support_level`` is one of the existing hint rungs. Outcome is independent
    from support, so a failed unaided attempt differs from failure after help.
    This helper never accepts model-proposed mastery or zone labels.
    """
    if not isinstance(card, dict):
        raise ZpdPolicyError("card must be an object")
    if not isinstance(topic_id, str) or not topic_id.strip():
        raise ZpdPolicyError("topic_id is required")
    if support_level not in HELP_BY_KEY:
        raise ZpdPolicyError(f"unknown support_level: {support_level!r}")
    if type(solved) is not bool:
        raise ZpdPolicyError("solved must be a boolean")
    date = observed_on or datetime.date.today().isoformat()
    try:
        parsed_date = datetime.date.fromisoformat(date)
    except (TypeError, ValueError) as exc:
        raise ZpdPolicyError("observed_on must be an ISO date") from exc
    if parsed_date > datetime.date.today():
        raise ZpdPolicyError("observed_on cannot be in the future")
    if task_id is not None and (not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 200):
        raise ZpdPolicyError("task_id must contain 1 to 200 characters")
    if assessment_pair_id is not None and (
        not isinstance(assessment_pair_id, str) or not assessment_pair_id.strip() or len(assessment_pair_id) > 200
    ):
        raise ZpdPolicyError("assessment_pair_id must contain 1 to 200 characters")
    if difficulty is not None and difficulty not in {"easy", "medium", "hard"}:
        raise ZpdPolicyError("difficulty must be easy, medium, or hard")
    if notes is not None and (not isinstance(notes, str) or len(notes) > 1000):
        raise ZpdPolicyError("notes must be a string of at most 1000 characters")

    result = copy.deepcopy(card)
    model = result.setdefault("learner_model", {})
    if not isinstance(model, dict):
        raise ZpdPolicyError("learner_model must be an object")
    competencies = model.setdefault("competencies", [])
    if not isinstance(competencies, list):
        raise ZpdPolicyError("learner_model.competencies must be a list")
    matches = [i for i, item in enumerate(competencies)
               if isinstance(item, dict) and item.get("topic_id") == topic_id]
    if len(matches) > 1:
        raise ZpdPolicyError(f"duplicate competency for topic_id: {topic_id}")
    before = competencies[matches[0]] if matches else None
    updated = update_competency(
        before,
        Observation(topic_id, support_level, solved, date, notes or "", task_id,
                    assessment_pair_id, difficulty),
    )
    if matches:
        competencies[matches[0]] = updated
    else:
        competencies.append(updated)

    model["zpd"] = derive_zpd(result)
    knowledge = result.setdefault("knowledge", [])
    if isinstance(knowledge, list):
        by_topic = {item.get("topic_id"): item for item in knowledge if isinstance(item, dict)}
        for verdict in classify_card(result):
            status = STATUS_BY_ZONE.get(verdict.zone)
            if status is None:
                continue
            entry = by_topic.get(verdict.topic_id)
            if entry is None:
                entry = {"topic_id": verdict.topic_id, "status": status}
                knowledge.append(entry)
                by_topic[verdict.topic_id] = entry
            elif status_conflicts(verdict.zone, entry.get("status")):
                entry["status"] = status
    return result


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
    competencies = ((card.get("learner_model") or {}).get("competencies") or [])
    by_topic = {
        item.get("topic_id"): item for item in competencies if isinstance(item, dict)
    }
    for verdict in classify_card(card):
        entry = catalog.get(verdict.topic_id) or {}
        verified = confirmed_zone(by_topic.get(verdict.topic_id) or {}, today)
        rows.append({
            "topic_id": verdict.topic_id,
            "title_ru": entry.get("title_ru", verdict.topic_id),
            "zone": verdict.zone,
            "planning_zone": verdict.zone,
            "confirmed_zone": verified["zone"],
            "evidence_status": verified["evidence_status"],
            "confirmed_reason": verified["reason"],
            "evidence": verified["evidence"],
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
