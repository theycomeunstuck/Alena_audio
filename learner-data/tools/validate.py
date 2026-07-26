#!/usr/bin/env python3
"""Validator for learner-data/: checks catalog/ and learners/*.json.

CLI: python learner-data/tools/validate.py [--learners-dir DIR] [--catalog FILE] [--strict]

Exit codes:
  0 - clean (warnings are allowed unless --strict)
  1 - errors found (or warnings found with --strict)
  2 - IO failure (missing directory/catalog, unreadable file)

All findings/messages meant for the tutor are in Russian; identifiers and
comments in the code are in English, per package convention.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learner_common import (
        DATE_RE,
        KNOWLEDGE_STATUSES,
        LEARNER_ID_RE,
        TOPIC_ID_RE,
        configure_utf8_streams,
        find_package_root,
        iter_learner_files,
        load_json,
    )
    import zpd
else:
    from .learner_common import (
        DATE_RE,
        KNOWLEDGE_STATUSES,
        LEARNER_ID_RE,
        TOPIC_ID_RE,
        configure_utf8_streams,
        find_package_root,
        iter_learner_files,
        load_json,
    )
    from . import zpd


class Finding:
    """One validator finding (a single output line)."""

    __slots__ = ("level", "file", "json_path", "message")

    def __init__(self, level: str, file: str, json_path: str, message: str):
        assert level in ("ERROR", "WARN")
        self.level = level
        self.file = file
        self.json_path = json_path
        self.message = message

    def is_error(self) -> bool:
        return self.level == "ERROR"

    def format(self) -> str:
        return f"{self.level}  {self.file}  {self.json_path}  {self.message}"


def _err(file: str, json_path: str, message: str) -> Finding:
    return Finding("ERROR", file, json_path, message)


def _warn(file: str, json_path: str, message: str) -> Finding:
    return Finding("WARN", file, json_path, message)


# --- generic structural helpers -------------------------------------------------

def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_strict_int(value: object) -> bool:
    """True int, not bool (isinstance(True, int) is True in Python, must reject)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and len(value) > 0


def _find_unknown_keys(obj: dict, allowed: set[str], file: str, json_path: str) -> list[Finding]:
    findings = []
    for key in obj.keys():
        if key not in allowed:
            findings.append(
                _err(file, f"{json_path}/{key}", f'неизвестный ключ "{key}" (проверьте опечатку)')
            )
    return findings


# --- catalog validation -----------------------------------------------------

CATALOG_TOP_KEYS = {"schema_version", "subject", "grades", "topics"}
CATALOG_TOPIC_KEYS = {"id", "title_ru", "synonyms", "parent", "grade", "deprecated"}


def validate_catalog(catalog_data: object, catalog_file: str) -> tuple[list[Finding], dict]:
    """Validate the skills catalog structure. Returns (findings, id_to_entry)."""
    findings: list[Finding] = []
    catalog_index: dict = {}

    if not isinstance(catalog_data, dict):
        findings.append(_err(catalog_file, "/", "каталог должен быть JSON-объектом"))
        return findings, catalog_index

    findings.extend(_find_unknown_keys(catalog_data, CATALOG_TOP_KEYS, catalog_file, ""))

    for required_key in CATALOG_TOP_KEYS:
        if required_key not in catalog_data:
            findings.append(_err(catalog_file, f"/{required_key}", f'отсутствует обязательный ключ "{required_key}"'))

    topics = catalog_data.get("topics")
    if not isinstance(topics, list):
        findings.append(_err(catalog_file, "/topics", 'поле "topics" должно быть списком'))
        return findings, catalog_index

    seen_ids: dict[str, int] = {}
    entries: list[tuple[int, dict]] = []

    for i, entry in enumerate(topics):
        json_path = f"/topics/{i}"
        if not isinstance(entry, dict):
            findings.append(_err(catalog_file, json_path, "запись каталога должна быть объектом"))
            continue

        findings.extend(_find_unknown_keys(entry, CATALOG_TOPIC_KEYS, catalog_file, json_path))
        for required_key in CATALOG_TOPIC_KEYS:
            if required_key not in entry:
                findings.append(_err(catalog_file, f"{json_path}/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

        entries.append((i, entry))

        topic_id = entry.get("id")
        if not isinstance(topic_id, str):
            findings.append(_err(catalog_file, f"{json_path}/id", 'поле "id" должно быть строкой'))
            continue

        if not TOPIC_ID_RE.match(topic_id):
            findings.append(_err(catalog_file, f"{json_path}/id", f'id "{topic_id}" не соответствует грамматике topic_id'))

        if topic_id in seen_ids:
            findings.append(_err(catalog_file, f"{json_path}/id", f'дублирующийся id "{topic_id}"'))
        else:
            seen_ids[topic_id] = i
            catalog_index[topic_id] = entry

        title_ru = entry.get("title_ru")
        if not _is_nonempty_str(title_ru):
            findings.append(_err(catalog_file, f"{json_path}/title_ru", 'поле "title_ru" должно быть непустой строкой'))

        grade = entry.get("grade")
        if not isinstance(grade, int) or isinstance(grade, bool):
            findings.append(_err(catalog_file, f"{json_path}/grade", 'поле "grade" должно быть целым числом'))

        deprecated = entry.get("deprecated")
        if not _is_bool(deprecated):
            findings.append(_err(catalog_file, f"{json_path}/deprecated", 'поле "deprecated" должно быть булевым значением'))

        synonyms = entry.get("synonyms")
        if not isinstance(synonyms, list):
            findings.append(_err(catalog_file, f"{json_path}/synonyms", 'поле "synonyms" должно быть списком'))

    # Cross-checks that need the full id set: parent grammar/existence, grade-vs-id segment.
    for i, entry in entries:
        json_path = f"/topics/{i}"
        topic_id = entry.get("id")
        if not isinstance(topic_id, str) or not TOPIC_ID_RE.match(topic_id):
            continue  # already reported above

        segments = topic_id.split(".")
        expected_parent = ".".join(segments[:-1]) if len(segments) > 3 else None

        parent = entry.get("parent")
        if parent != expected_parent:
            findings.append(
                _err(
                    catalog_file,
                    f"{json_path}/parent",
                    f'parent "{parent}" не совпадает с ожидаемым "{expected_parent}" (id без последнего сегмента)',
                )
            )
        elif parent is not None and parent not in seen_ids:
            findings.append(_err(catalog_file, f"{json_path}/parent", f'parent "{parent}" не найден в каталоге'))

        grade = entry.get("grade")
        if isinstance(grade, int) and not isinstance(grade, bool):
            expected_grade_segment = f"g{grade}"
            actual_grade_segment = segments[1] if len(segments) > 1 else ""
            if actual_grade_segment != expected_grade_segment:
                findings.append(
                    _err(
                        catalog_file,
                        f"{json_path}/grade",
                        f'grade {grade} не совпадает с сегментом id "{actual_grade_segment}" (ожидался "{expected_grade_segment}")',
                    )
                )

    return findings, catalog_index


# --- learner card validation -------------------------------------------------

CARD_TOP_KEYS = {
    "schema_version", "learner_id", "pseudonym", "legal_name", "age_group", "grade",
    "language", "rag_context", "profile", "interests", "knowledge", "error_patterns", "mvp",
    "learner_model",
}
PROFILE_KEYS = {"explanation_style", "pace", "autonomy_level", "motivation", "prefers_visual"}
# "lessons" — необязательный журнал занятий (пункт 4 концепта): дата, темы,
# длительность. Старые карточки без него остаются валидными.
MVP_KEYS = {"story_preferences", "learning_preferences_observed", "help_strategies",
            "journal", "points_ledger", "lessons"}
MVP_REQUIRED_KEYS = MVP_KEYS - {"lessons"}
LESSON_ITEM_KEYS = {"date", "topic_ids", "minutes", "note"}
KNOWLEDGE_ITEM_KEYS = {"topic_id", "status", "notes"}
ERROR_PATTERN_ITEM_KEYS = {"subject", "topic_id", "error_tag", "count", "last_seen"}
JOURNAL_ITEM_KEYS = {"date", "note"}
POINTS_ITEM_KEYS = {"date", "points", "reason", "role", "topic_id"}
RAG_CONTEXT_KEYS = {
    "current_goal", "current_topics", "priority_difficulties",
    "effective_strategies", "avoid", "recent_progress",
}
RAG_GOAL_KEYS = {"topic_id", "goal", "updated_at"}
RAG_DIFFICULTY_KEYS = {"topic_id", "description"}
RAG_PROGRESS_KEYS = {"date", "note"}
LEARNER_MODEL_KEYS = {
    "competencies", "zpd", "learning_preferences", "engagement", "ai_usage",
    "projects", "strengths", "support_needs", "scaffolding_by_topic", "gamification",
}
# Competence map (concept section 2). Only topic_id and mastery are required so
# that short demo cards stay valid; the ZPD policy handles missing independence.
COMPETENCY_ITEM_KEYS = {
    "topic_id", "mastery", "confidence", "independence", "needs_scaffolding",
    "effective_scaffolding", "recommended_scaffolding", "last_assessed", "history",
}
COMPETENCY_REQUIRED_KEYS = ("topic_id", "mastery")
COMPETENCY_UNIT_KEYS = ("mastery", "confidence", "independence")
COMPETENCY_HISTORY_KEYS = {"date", "mastery", "independence", "help_level"}
RAG_TEXT_MAX = 280
LEGAL_NAME_KEYS = {"first_name", "last_name", "patronymic"}


def _contains_balance_key(obj: object, json_path: str, file: str, findings: list[Finding]) -> None:
    """Recursively flag any key containing 'balance' at any nesting level."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "balance" in key.lower():
                findings.append(
                    _err(file, f"{json_path}/{key}", 'ключ, содержащий "balance", не должен храниться — баланс всегда вычисляется')
                )
            _contains_balance_key(value, f"{json_path}/{key}", file, findings)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _contains_balance_key(item, f"{json_path}/{i}", file, findings)


def _check_bounded_text(value: object, filename: str, json_path: str, label: str) -> list[Finding]:
    """Validate text destined for the compact RAG projection."""
    findings: list[Finding] = []
    if not _is_nonempty_str(value):
        findings.append(_err(filename, json_path, f'{label} должен быть непустой строкой'))
    elif len(value) > RAG_TEXT_MAX:
        findings.append(_err(filename, json_path, f'{label} длиннее {RAG_TEXT_MAX} символов и не поместится в компактный RAG-контекст'))
    return findings


def _validate_rag_context(rag: object, catalog: dict, filename: str) -> list[Finding]:
    """Validate the small explicit projection that may reach the AI tutor."""
    findings: list[Finding] = []
    if not isinstance(rag, dict):
        return [_err(filename, "/rag_context", 'поле "rag_context" должно быть объектом')]

    findings.extend(_find_unknown_keys(rag, RAG_CONTEXT_KEYS, filename, "/rag_context"))
    for key in RAG_CONTEXT_KEYS:
        if key not in rag:
            findings.append(_err(filename, f"/rag_context/{key}", f'отсутствует обязательное поле "{key}"'))

    goal = rag.get("current_goal")
    if not isinstance(goal, dict):
        findings.append(_err(filename, "/rag_context/current_goal", 'поле "current_goal" должно быть объектом'))
    else:
        findings.extend(_find_unknown_keys(goal, RAG_GOAL_KEYS, filename, "/rag_context/current_goal"))
        for key in RAG_GOAL_KEYS:
            if key not in goal:
                findings.append(_err(filename, f"/rag_context/current_goal/{key}", f'отсутствует обязательное поле "{key}"'))
        if isinstance(goal.get("topic_id"), str):
            findings.extend(_check_topic_ref(goal["topic_id"], catalog, filename, "/rag_context/current_goal/topic_id"))
        elif "topic_id" in goal:
            findings.append(_err(filename, "/rag_context/current_goal/topic_id", 'поле "topic_id" должно быть строкой'))
        if "goal" in goal:
            findings.extend(_check_bounded_text(goal["goal"], filename, "/rag_context/current_goal/goal", "цель"))
        if "updated_at" in goal:
            findings.extend(_check_date(goal["updated_at"], filename, "/rag_context/current_goal/updated_at"))

    current_topics = rag.get("current_topics")
    if not isinstance(current_topics, list):
        findings.append(_err(filename, "/rag_context/current_topics", 'поле "current_topics" должно быть списком'))
    else:
        if len(current_topics) > 3:
            findings.append(_err(filename, "/rag_context/current_topics", "в RAG-контексте может быть максимум 3 текущие темы"))
        seen: set[str] = set()
        for i, topic_id in enumerate(current_topics):
            path = f"/rag_context/current_topics/{i}"
            if not isinstance(topic_id, str):
                findings.append(_err(filename, path, "topic_id должен быть строкой"))
            else:
                if topic_id in seen:
                    findings.append(_err(filename, path, f'дублирующийся topic_id "{topic_id}"'))
                seen.add(topic_id)
                findings.extend(_check_topic_ref(topic_id, catalog, filename, path))

    difficulties = rag.get("priority_difficulties")
    if not isinstance(difficulties, list):
        findings.append(_err(filename, "/rag_context/priority_difficulties", 'поле "priority_difficulties" должно быть списком'))
    else:
        if len(difficulties) > 3:
            findings.append(_err(filename, "/rag_context/priority_difficulties", "в RAG-контексте может быть максимум 3 трудности"))
        for i, item in enumerate(difficulties):
            path = f"/rag_context/priority_difficulties/{i}"
            if not isinstance(item, dict):
                findings.append(_err(filename, path, "запись трудности должна быть объектом"))
                continue
            findings.extend(_find_unknown_keys(item, RAG_DIFFICULTY_KEYS, filename, path))
            for key in RAG_DIFFICULTY_KEYS:
                if key not in item:
                    findings.append(_err(filename, f"{path}/{key}", f'отсутствует обязательное поле "{key}"'))
            if isinstance(item.get("topic_id"), str):
                findings.extend(_check_topic_ref(item["topic_id"], catalog, filename, f"{path}/topic_id"))
            elif "topic_id" in item:
                findings.append(_err(filename, f"{path}/topic_id", 'поле "topic_id" должно быть строкой'))
            if "description" in item:
                findings.extend(_check_bounded_text(item["description"], filename, f"{path}/description", "описание трудности"))

    for key, label in (("effective_strategies", "стратегия помощи"), ("avoid", "пункт avoid")):
        items = rag.get(key)
        path = f"/rag_context/{key}"
        if not isinstance(items, list):
            findings.append(_err(filename, path, f'поле "{key}" должно быть списком'))
            continue
        if len(items) > 3:
            findings.append(_err(filename, path, f'в RAG-контексте может быть максимум 3 элементов "{key}"'))
        for i, item in enumerate(items):
            findings.extend(_check_bounded_text(item, filename, f"{path}/{i}", label))

    progress = rag.get("recent_progress")
    if not isinstance(progress, dict):
        findings.append(_err(filename, "/rag_context/recent_progress", 'поле "recent_progress" должно быть объектом'))
    else:
        findings.extend(_find_unknown_keys(progress, RAG_PROGRESS_KEYS, filename, "/rag_context/recent_progress"))
        for key in RAG_PROGRESS_KEYS:
            if key not in progress:
                findings.append(_err(filename, f"/rag_context/recent_progress/{key}", f'отсутствует обязательное поле "{key}"'))
        if "date" in progress:
            findings.extend(_check_date(progress["date"], filename, "/rag_context/recent_progress/date"))
        if "note" in progress:
            findings.extend(_check_bounded_text(progress["note"], filename, "/rag_context/recent_progress/note", "заметка о прогрессе"))
    return findings


def _check_unit_number(value: object, filename: str, json_path: str, label: str) -> list[Finding]:
    """Validate a 0..1 pedagogical score (mastery, independence, confidence)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [_err(filename, json_path, f"{label} должен быть числом от 0 до 1")]
    if float(value) < 0.0 or float(value) > 1.0:
        return [_err(filename, json_path, f"{label} должен быть в диапазоне от 0 до 1, получено {value}")]
    return []


def _validate_competencies(competencies: list, catalog: dict, filename: str) -> list[Finding]:
    """Validate the competence map — the source of truth for the ZPD policy.

    These checks matter more than usual because ``card_updater`` lets an LLM
    propose writes here: a malformed mastery must fail validation, not quietly
    become a zone verdict.
    """
    findings: list[Finding] = []
    seen_topic_ids: set[str] = set()

    for i, item in enumerate(competencies):
        json_path = f"/learner_model/competencies/{i}"
        if not isinstance(item, dict):
            findings.append(_err(filename, json_path, "запись карты компетенций должна быть объектом"))
            continue

        findings.extend(_find_unknown_keys(item, COMPETENCY_ITEM_KEYS, filename, json_path))
        for required_key in COMPETENCY_REQUIRED_KEYS:
            if required_key not in item:
                findings.append(_err(filename, f"{json_path}/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

        topic_id = item.get("topic_id")
        if isinstance(topic_id, str):
            if topic_id in seen_topic_ids:
                findings.append(_err(filename, f"{json_path}/topic_id", f'дублирующийся topic_id "{topic_id}" в карте компетенций'))
            seen_topic_ids.add(topic_id)
            findings.extend(_check_topic_ref(topic_id, catalog, filename, f"{json_path}/topic_id"))
        elif "topic_id" in item:
            findings.append(_err(filename, f"{json_path}/topic_id", 'поле "topic_id" должно быть строкой'))

        for key in COMPETENCY_UNIT_KEYS:
            if key in item:
                findings.extend(_check_unit_number(item[key], filename, f"{json_path}/{key}", f'поле "{key}"'))

        if "needs_scaffolding" in item and not _is_bool(item.get("needs_scaffolding")):
            findings.append(_err(filename, f"{json_path}/needs_scaffolding", 'поле "needs_scaffolding" должно быть булевым значением'))

        if "effective_scaffolding" in item:
            strategies = item.get("effective_scaffolding")
            if not isinstance(strategies, list):
                findings.append(_err(filename, f"{json_path}/effective_scaffolding", 'поле "effective_scaffolding" должно быть списком'))
            else:
                for j, strategy in enumerate(strategies):
                    if not _is_nonempty_str(strategy):
                        findings.append(_err(filename, f"{json_path}/effective_scaffolding/{j}", "стратегия должна быть непустой строкой"))

        if "recommended_scaffolding" in item and not _is_nonempty_str(item.get("recommended_scaffolding")):
            findings.append(_err(filename, f"{json_path}/recommended_scaffolding", 'поле "recommended_scaffolding" должно быть непустой строкой'))

        if "last_assessed" in item:
            findings.extend(_check_date(item.get("last_assessed"), filename, f"{json_path}/last_assessed"))

        if "history" in item:
            history = item.get("history")
            if not isinstance(history, list):
                findings.append(_err(filename, f"{json_path}/history", 'поле "history" должно быть списком'))
            else:
                if len(history) > zpd.HISTORY_LIMIT:
                    findings.append(
                        _warn(
                            filename,
                            f"{json_path}/history",
                            f"в истории {len(history)} записей — храним последние {zpd.HISTORY_LIMIT}",
                        )
                    )
                for j, entry in enumerate(history):
                    entry_path = f"{json_path}/history/{j}"
                    if not isinstance(entry, dict):
                        findings.append(_err(filename, entry_path, "запись истории должна быть объектом"))
                        continue
                    findings.extend(_find_unknown_keys(entry, COMPETENCY_HISTORY_KEYS, filename, entry_path))
                    if "date" not in entry:
                        findings.append(_err(filename, f"{entry_path}/date", 'отсутствует обязательное поле "date"'))
                    else:
                        findings.extend(_check_date(entry.get("date"), filename, f"{entry_path}/date"))
                    for key in ("mastery", "independence"):
                        if key in entry:
                            findings.extend(_check_unit_number(entry[key], filename, f"{entry_path}/{key}", f'поле "{key}"'))
                    help_key = entry.get("help_level")
                    if "help_level" in entry and (not isinstance(help_key, str) or help_key not in zpd.HELP_BY_KEY):
                        allowed = ", ".join(zpd.HELP_BY_KEY)
                        findings.append(
                            _err(filename, f"{entry_path}/help_level", f'неизвестный уровень помощи "{help_key}"; допустимы: {allowed}')
                        )

    return findings


def _check_zpd_matches_competencies(model: dict, filename: str) -> list[Finding]:
    """Warn when the stored ZPD block contradicts the competence map.

    Concept section 2 (the competence map) is authoritative and section 3 is a
    derived mirror, so a mismatch means the card was hand-edited and is now
    stale. It is a warning, not an error: reading a slightly stale card is
    better than refusing to run a lesson.
    """
    stored = model.get("zpd")
    if not isinstance(stored, dict):
        return []
    if not isinstance(model.get("competencies"), list) or not model["competencies"]:
        return []

    mapped_topics = {
        item["topic_id"]
        for item in model["competencies"]
        if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
    }
    derived = zpd.derive_zpd({"learner_model": model})
    findings: list[Finding] = []
    for key in ("current", "outside"):
        stored_list = stored.get(key)
        if not isinstance(stored_list, list):
            continue
        stored_topics = [item for item in stored_list if isinstance(item, str)]

        for i, topic_id in enumerate(stored_topics):
            if topic_id not in mapped_topics:
                findings.append(
                    _warn(
                        filename,
                        f"/learner_model/zpd/{key}/{i}",
                        f'тема "{topic_id}" указана в zpd, но её нет в карте компетенций — '
                        "добавьте оценку mastery, иначе зона не выводится",
                    )
                )

        assessed = sorted(topic_id for topic_id in stored_topics if topic_id in mapped_topics)
        if assessed != sorted(derived[key]):
            findings.append(
                _warn(
                    filename,
                    f"/learner_model/zpd/{key}",
                    "не совпадает с ЗБР, выведенной из карты компетенций "
                    f"(ожидалось: {', '.join(derived[key]) or '—'}); "
                    "перезаписать: python learner-data/tools/zpd.py <learner_id> --write",
                )
            )
    return findings


def _check_knowledge_matches_competencies(card: dict, knowledge: list, filename: str) -> list[Finding]:
    """Warn when a topic's status contradicts its zone in the competence map.

    Catches the two unambiguous cases: a topic that already has an assessment
    still marked ``not_started``, and a topic marked ``confident`` that the child
    cannot yet do alone. Compatible variations are left to the tutor.
    """
    findings: list[Finding] = []
    by_topic = {
        item["topic_id"]: index
        for index, item in enumerate(knowledge)
        if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
    }
    for verdict in zpd.classify_card(card):
        index = by_topic.get(verdict.topic_id)
        if index is None:
            continue
        status = knowledge[index].get("status")
        if zpd.status_conflicts(verdict.zone, status):
            findings.append(
                _warn(
                    filename,
                    f"/knowledge/{index}/status",
                    f'статус "{status}" противоречит карте компетенций '
                    f'(зона: {verdict.label_ru()}, ожидается один из '
                    f'{", ".join(zpd.COMPATIBLE_STATUSES[verdict.zone])})',
                )
            )
    return findings


def _validate_learner_model(model: object, catalog: dict, filename: str) -> list[Finding]:
    """Lightweight structural guard for the rich benchmark-only learner model."""
    findings: list[Finding] = []
    if not isinstance(model, dict):
        return [_err(filename, "/learner_model", 'поле "learner_model" должно быть объектом')]
    findings.extend(_find_unknown_keys(model, LEARNER_MODEL_KEYS, filename, "/learner_model"))
    for key in LEARNER_MODEL_KEYS:
        if key not in model:
            findings.append(_err(filename, f"/learner_model/{key}", f'отсутствует обязательное поле "{key}"'))

    for key in ("competencies", "projects", "strengths", "support_needs", "scaffolding_by_topic"):
        if key in model and not isinstance(model[key], list):
            findings.append(_err(filename, f"/learner_model/{key}", f'поле "{key}" должно быть списком'))
    for key in ("zpd", "learning_preferences", "engagement", "ai_usage", "gamification"):
        if key in model and not isinstance(model[key], dict):
            findings.append(_err(filename, f"/learner_model/{key}", f'поле "{key}" должно быть объектом'))

    for key in ("strengths", "support_needs"):
        values = model.get(key)
        if isinstance(values, list):
            for i, value in enumerate(values):
                if not _is_nonempty_str(value):
                    findings.append(_err(filename, f"/learner_model/{key}/{i}", "элемент должен быть непустой строкой"))

    # Local name is zpd_block, not zpd: the module-level `zpd` import is the ZPD
    # policy and must stay reachable from this function.
    zpd_block = model.get("zpd")
    if isinstance(zpd_block, dict):
        findings.extend(_find_unknown_keys(zpd_block, {"current", "outside"}, filename, "/learner_model/zpd"))
        for key in ("current", "outside"):
            topics = zpd_block.get(key)
            if not isinstance(topics, list):
                findings.append(_err(filename, f"/learner_model/zpd/{key}", "поле должно быть списком topic_id"))
                continue
            for i, topic_id in enumerate(topics):
                if isinstance(topic_id, str):
                    findings.extend(_check_topic_ref(topic_id, catalog, filename, f"/learner_model/zpd/{key}/{i}"))
                else:
                    findings.append(_err(filename, f"/learner_model/zpd/{key}/{i}", "topic_id должен быть строкой"))

    competencies = model.get("competencies")
    if isinstance(competencies, list):
        findings.extend(_validate_competencies(competencies, catalog, filename))
        findings.extend(_check_zpd_matches_competencies(model, filename))

    return findings


def _suggest_topic_id(bad_id: str, catalog: dict) -> str:
    """Suggest close matches by id and by synonym, for an unknown topic_id."""
    candidates: dict[str, str] = {}  # display string -> sort key (unused, kept simple)
    all_ids = list(catalog.keys())
    id_matches = difflib.get_close_matches(bad_id, all_ids, n=3, cutoff=0.5)

    synonym_pool: list[tuple[str, str]] = []  # (synonym, owning id)
    for tid, entry in catalog.items():
        for syn in entry.get("synonyms") or []:
            if isinstance(syn, str):
                synonym_pool.append((syn, tid))
    synonym_texts = [s for s, _ in synonym_pool]
    synonym_matches = difflib.get_close_matches(bad_id, synonym_texts, n=3, cutoff=0.5)
    synonym_to_id = dict(synonym_pool)

    suggestions: list[str] = []
    for m in id_matches:
        if m not in suggestions:
            suggestions.append(m)
    for m in synonym_matches:
        owning_id = synonym_to_id.get(m)
        if owning_id and owning_id not in suggestions:
            suggestions.append(owning_id)

    if not suggestions:
        return ""
    return " (возможно, вы имели в виду: " + ", ".join(suggestions) + ")"


def validate_card(card: object, catalog: dict, filename: str) -> list[Finding]:
    """Validate a single learner card (dict) against structural/business rules.

    ``filename`` is used only as the file label in findings (it need not exist
    on disk — this function is also used directly against in-memory fixtures).
    Returns a list of Finding objects; does not touch the filesystem.
    """
    findings: list[Finding] = []

    if not isinstance(card, dict):
        findings.append(_err(filename, "/", "карточка должна быть JSON-объектом"))
        return findings

    findings.extend(_find_unknown_keys(card, CARD_TOP_KEYS, filename, ""))

    for required_key in CARD_TOP_KEYS:
        if required_key not in {"learner_model", "legal_name"} and required_key not in card:
            findings.append(_err(filename, f"/{required_key}", f'отсутствует обязательный ключ "{required_key}"'))

    if card.get("schema_version") != 2:
        findings.append(_err(filename, "/schema_version", 'schema_version должен быть целым числом 2'))

    # learner_id / filename stem check
    learner_id = card.get("learner_id")
    if not isinstance(learner_id, str):
        findings.append(_err(filename, "/learner_id", 'поле "learner_id" должно быть строкой'))
    else:
        if not LEARNER_ID_RE.match(learner_id):
            findings.append(_err(filename, "/learner_id", f'learner_id "{learner_id}" не соответствует шаблону ^[a-z0-9][a-z0-9-]*$'))
        stem = Path(filename).stem
        if stem != learner_id:
            findings.append(
                _err(filename, "/learner_id", f'learner_id "{learner_id}" не совпадает с именем файла "{stem}"')
            )

    pseudonym = card.get("pseudonym")
    if not _is_nonempty_str(pseudonym):
        findings.append(_err(filename, "/pseudonym", 'поле "pseudonym" должно быть непустой строкой'))

    legal_name = card.get("legal_name")
    if legal_name is not None:
        if not isinstance(legal_name, dict):
            findings.append(_err(filename, "/legal_name", 'поле должно быть объектом с first_name, last_name и patronymic'))
        else:
            findings.extend(_find_unknown_keys(legal_name, LEGAL_NAME_KEYS, filename, "/legal_name"))
            for key in LEGAL_NAME_KEYS:
                if key not in legal_name:
                    findings.append(_err(filename, f"/legal_name/{key}", f'отсутствует обязательное поле "{key}"'))
            for key in ("first_name", "last_name"):
                value = legal_name.get(key)
                if not _is_nonempty_str(value) or len(value) > 80:
                    findings.append(_err(filename, f"/legal_name/{key}", 'поле должно быть непустой строкой до 80 символов'))
            patronymic = legal_name.get("patronymic")
            if not isinstance(patronymic, str) or len(patronymic) > 80:
                findings.append(_err(filename, "/legal_name/patronymic", 'отчество должно быть строкой до 80 символов; допустимы "" и "-"'))

    age_group = card.get("age_group")
    if not _is_nonempty_str(age_group):
        findings.append(_err(filename, "/age_group", 'поле "age_group" должно быть непустой строкой'))

    grade = card.get("grade")
    if grade not in ("3", "4"):
        findings.append(_err(filename, "/grade", f'grade "{grade}" — допустимы только "3" или "4"'))

    language = card.get("language")
    if not _is_nonempty_str(language):
        findings.append(_err(filename, "/language", 'поле "language" должно быть непустой строкой'))

    findings.extend(_validate_rag_context(card.get("rag_context"), catalog, filename))

    # profile
    profile = card.get("profile")
    if not isinstance(profile, dict):
        findings.append(_err(filename, "/profile", 'поле "profile" должно быть объектом'))
        profile = {}
    else:
        findings.extend(_find_unknown_keys(profile, PROFILE_KEYS, filename, "/profile"))
        for required_key in PROFILE_KEYS:
            if required_key not in profile:
                findings.append(_err(filename, f"/profile/{required_key}", f'отсутствует обязательное поле "{required_key}"'))
        if "prefers_visual" in profile and not _is_bool(profile.get("prefers_visual")):
            findings.append(_err(filename, "/profile/prefers_visual", 'поле "prefers_visual" должно быть булевым значением'))

    # interests
    interests = card.get("interests")
    if not isinstance(interests, list):
        findings.append(_err(filename, "/interests", 'поле "interests" должно быть списком'))
    else:
        for i, el in enumerate(interests):
            if not isinstance(el, str):
                findings.append(_err(filename, f"/interests/{i}", "элемент должен быть строкой"))

    # knowledge
    knowledge = card.get("knowledge")
    if not isinstance(knowledge, list):
        findings.append(_err(filename, "/knowledge", 'поле "knowledge" должно быть списком'))
        knowledge = []

    seen_topic_ids: set[str] = set()
    for i, item in enumerate(knowledge):
        json_path = f"/knowledge/{i}"
        if not isinstance(item, dict):
            findings.append(_err(filename, json_path, "запись knowledge должна быть объектом"))
            continue
        findings.extend(_find_unknown_keys(item, KNOWLEDGE_ITEM_KEYS, filename, json_path))
        for required_key in ("topic_id", "status"):
            if required_key not in item:
                findings.append(_err(filename, f"{json_path}/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

        topic_id = item.get("topic_id")
        if isinstance(topic_id, str):
            if topic_id in seen_topic_ids:
                findings.append(_err(filename, f"{json_path}/topic_id", f'дублирующийся topic_id "{topic_id}" в knowledge'))
            seen_topic_ids.add(topic_id)
            findings.extend(_check_topic_ref(topic_id, catalog, filename, f"{json_path}/topic_id"))
        elif "topic_id" in item:
            findings.append(_err(filename, f"{json_path}/topic_id", 'поле "topic_id" должно быть строкой'))

        status = item.get("status")
        if status is not None and status not in KNOWLEDGE_STATUSES:
            findings.append(
                _err(filename, f"{json_path}/status", f'"{status}" нет в списке допустимых статусов {KNOWLEDGE_STATUSES}')
            )

        if "notes" in item and not isinstance(item.get("notes"), str):
            findings.append(_err(filename, f"{json_path}/notes", 'поле "notes" должно быть строкой'))

    # error_patterns
    error_patterns = card.get("error_patterns")
    if not isinstance(error_patterns, list):
        findings.append(_err(filename, "/error_patterns", 'поле "error_patterns" должно быть списком'))
        error_patterns = []

    for i, item in enumerate(error_patterns):
        json_path = f"/error_patterns/{i}"
        if not isinstance(item, dict):
            findings.append(_err(filename, json_path, "запись error_patterns должна быть объектом"))
            continue
        findings.extend(_find_unknown_keys(item, ERROR_PATTERN_ITEM_KEYS, filename, json_path))
        for required_key in ERROR_PATTERN_ITEM_KEYS:
            if required_key not in item:
                findings.append(_err(filename, f"{json_path}/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

        topic_id = item.get("topic_id")
        if isinstance(topic_id, str):
            findings.extend(_check_topic_ref(topic_id, catalog, filename, f"{json_path}/topic_id"))
        elif "topic_id" in item:
            findings.append(_err(filename, f"{json_path}/topic_id", 'поле "topic_id" должно быть строкой'))

        subject = item.get("subject")
        if "subject" in item and not _is_nonempty_str(subject):
            findings.append(_err(filename, f"{json_path}/subject", 'поле "subject" должно быть непустой строкой'))

        error_tag = item.get("error_tag")
        if "error_tag" in item and not _is_nonempty_str(error_tag):
            findings.append(_err(filename, f"{json_path}/error_tag", 'поле "error_tag" должно быть непустой строкой'))

        count = item.get("count")
        if "count" in item:
            if not _is_strict_int(count) or count < 1:
                findings.append(_err(filename, f"{json_path}/count", 'поле "count" должно быть целым числом >= 1 (не bool)'))

        last_seen = item.get("last_seen")
        if "last_seen" in item:
            findings.extend(_check_date(last_seen, filename, f"{json_path}/last_seen"))

    # mvp
    mvp = card.get("mvp")
    if not isinstance(mvp, dict):
        findings.append(_err(filename, "/mvp", 'поле "mvp" должно быть объектом'))
        mvp = {}
    else:
        findings.extend(_find_unknown_keys(mvp, MVP_KEYS, filename, "/mvp"))
        for required_key in MVP_REQUIRED_KEYS:
            if required_key not in mvp:
                findings.append(_err(filename, f"/mvp/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

    for list_key in ("story_preferences", "learning_preferences_observed", "help_strategies"):
        if list_key in mvp:
            list_value = mvp.get(list_key)
            if not isinstance(list_value, list):
                findings.append(_err(filename, f"/mvp/{list_key}", f'поле "{list_key}" должно быть списком'))
            else:
                for i, el in enumerate(list_value):
                    if not isinstance(el, str):
                        findings.append(_err(filename, f"/mvp/{list_key}/{i}", "элемент должен быть строкой"))

    # journal
    journal = mvp.get("journal")
    if "journal" in mvp and not isinstance(journal, list):
        findings.append(_err(filename, "/mvp/journal", 'поле "journal" должно быть списком'))
        journal = []
    journal = journal if isinstance(journal, list) else []

    for i, item in enumerate(journal):
        json_path = f"/mvp/journal/{i}"
        if not isinstance(item, dict):
            findings.append(_err(filename, json_path, "запись journal должна быть объектом"))
            continue
        findings.extend(_find_unknown_keys(item, JOURNAL_ITEM_KEYS, filename, json_path))
        for required_key in JOURNAL_ITEM_KEYS:
            if required_key not in item:
                findings.append(_err(filename, f"{json_path}/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

        if "date" in item:
            findings.extend(_check_date(item.get("date"), filename, f"{json_path}/date"))
        if "note" in item and not _is_nonempty_str(item.get("note")):
            findings.append(_err(filename, f"{json_path}/note", 'поле "note" должно быть непустой строкой'))

    # lessons — необязательный журнал занятий
    lessons = mvp.get("lessons")
    if "lessons" in mvp and not isinstance(lessons, list):
        findings.append(_err(filename, "/mvp/lessons", 'поле "lessons" должно быть списком'))
        lessons = []
    for i, item in enumerate(lessons if isinstance(lessons, list) else []):
        json_path = f"/mvp/lessons/{i}"
        if not isinstance(item, dict):
            findings.append(_err(filename, json_path, "запись занятия должна быть объектом"))
            continue
        findings.extend(_find_unknown_keys(item, LESSON_ITEM_KEYS, filename, json_path))
        if "date" not in item:
            findings.append(_err(filename, f"{json_path}/date", 'отсутствует обязательное поле "date"'))
        else:
            findings.extend(_check_date(item.get("date"), filename, f"{json_path}/date"))
        topic_ids = item.get("topic_ids")
        if topic_ids is not None:
            if not isinstance(topic_ids, list):
                findings.append(_err(filename, f"{json_path}/topic_ids", 'поле "topic_ids" должно быть списком'))
            else:
                for j, topic_id in enumerate(topic_ids):
                    if isinstance(topic_id, str):
                        findings.extend(_check_topic_ref(topic_id, catalog, filename, f"{json_path}/topic_ids/{j}"))
                    else:
                        findings.append(_err(filename, f"{json_path}/topic_ids/{j}", "topic_id должен быть строкой"))
        minutes = item.get("minutes")
        if minutes is not None and (not _is_strict_int(minutes) or minutes <= 0 or minutes > 600):
            findings.append(_err(filename, f"{json_path}/minutes", 'поле "minutes" должно быть целым числом от 1 до 600'))
        if "note" in item and not isinstance(item.get("note"), str):
            findings.append(_err(filename, f"{json_path}/note", 'поле "note" должно быть строкой'))

    # points_ledger
    points_ledger = mvp.get("points_ledger")
    if "points_ledger" in mvp and not isinstance(points_ledger, list):
        findings.append(_err(filename, "/mvp/points_ledger", 'поле "points_ledger" должно быть списком'))
        points_ledger = []
    points_ledger = points_ledger if isinstance(points_ledger, list) else []

    prev_date: str | None = None
    running_balance = 0
    balance_went_negative = False
    for i, item in enumerate(points_ledger):
        json_path = f"/mvp/points_ledger/{i}"
        if not isinstance(item, dict):
            findings.append(_err(filename, json_path, "запись points_ledger должна быть объектом"))
            continue
        findings.extend(_find_unknown_keys(item, POINTS_ITEM_KEYS, filename, json_path))
        for required_key in ("date", "points", "reason", "role"):
            if required_key not in item:
                findings.append(_err(filename, f"{json_path}/{required_key}", f'отсутствует обязательное поле "{required_key}"'))

        date_val = item.get("date")
        if "date" in item:
            findings.extend(_check_date(date_val, filename, f"{json_path}/date"))
            if isinstance(date_val, str) and DATE_RE.match(date_val):
                if prev_date is not None and date_val < prev_date:
                    findings.append(_warn(filename, f"{json_path}/date", "даты в points_ledger идут не по возрастанию"))
                prev_date = date_val

        points = item.get("points")
        if "points" in item:
            if not _is_strict_int(points):
                findings.append(_err(filename, f"{json_path}/points", 'поле "points" должно быть целым числом (не bool)'))
            else:
                running_balance += points
                if running_balance < 0:
                    balance_went_negative = True

        reason = item.get("reason")
        if "reason" in item and not _is_nonempty_str(reason):
            findings.append(_err(filename, f"{json_path}/reason", 'поле "reason" должно быть непустой строкой'))

        role = item.get("role")
        if "role" in item and role != "tutor":
            findings.append(_err(filename, f"{json_path}/role", f'поле "role" должно быть "tutor", получено "{role}"'))

        if "topic_id" in item:
            if isinstance(item.get("topic_id"), str):
                findings.extend(_check_topic_ref(item["topic_id"], catalog, filename, f"{json_path}/topic_id"))
            else:
                findings.append(_err(filename, f"{json_path}/topic_id", 'поле "topic_id" должно быть строкой'))

    if balance_went_negative:
        findings.append(_warn(filename, "/mvp/points_ledger", "накопительный баланс баллов уходит в отрицательные значения в процессе хронологии"))

    if "learner_model" in card:
        findings.extend(_validate_learner_model(card["learner_model"], catalog, filename))
        findings.extend(_check_knowledge_matches_competencies(card, knowledge, filename))

    # rule 11: any key containing "balance" anywhere in the card -> error
    _contains_balance_key(card, "", filename, findings)

    return findings


def _check_topic_ref(topic_id: str, catalog: dict, filename: str, json_path: str) -> list[Finding]:
    findings: list[Finding] = []
    entry = catalog.get(topic_id)
    if entry is None:
        suggestion = _suggest_topic_id(topic_id, catalog)
        findings.append(_err(filename, json_path, f'неизвестный topic_id "{topic_id}"{suggestion}'))
        return findings
    if entry.get("deprecated"):
        findings.append(_warn(filename, json_path, f'topic_id "{topic_id}" помечен как deprecated'))
    return findings


def _check_date(value: object, filename: str, json_path: str) -> list[Finding]:
    findings: list[Finding] = []
    if not isinstance(value, str) or not DATE_RE.match(value):
        findings.append(_err(filename, json_path, f'дата "{value}" не соответствует формату YYYY-MM-DD'))
        return findings
    try:
        parsed = datetime.date.fromisoformat(value)
    except ValueError:
        findings.append(_err(filename, json_path, f'дата "{value}" некорректна (не существует такого календарного дня)'))
        return findings
    if parsed > datetime.date.today():
        findings.append(_warn(filename, json_path, f'дата "{value}" в будущем'))
    return findings


# --- top-level orchestration --------------------------------------------------

def validate_all(learners_dir: Path, catalog_file: Path) -> tuple[list[Finding], int]:
    """Run full validation over the catalog and all learner cards.

    Returns (findings, file_count). Raises OSError/FileNotFoundError-family
    exceptions for missing directories, caught by main() as exit code 2.
    """
    findings: list[Finding] = []

    if not learners_dir.is_dir():
        raise FileNotFoundError(f'папка учеников не найдена: "{learners_dir}"')
    if not catalog_file.is_file():
        raise FileNotFoundError(f'файл каталога не найден: "{catalog_file}"')

    package_root = find_package_root()

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(package_root))
        except ValueError:
            return str(p)

    catalog_label = rel(catalog_file)
    try:
        catalog_data, catalog_bom = load_json(catalog_file)
    except json.JSONDecodeError as e:
        findings.append(_err(catalog_label, "/", f"ошибка разбора JSON: строка {e.lineno}, колонка {e.colno}: {e.msg}"))
        return findings, 0
    except UnicodeDecodeError as e:
        findings.append(_err(catalog_label, "/", f"ошибка кодировки файла: {e}"))
        return findings, 0

    if catalog_bom:
        findings.append(_warn(catalog_label, "/", "файл начинается с BOM (byte order mark)"))

    catalog_findings, catalog_index = validate_catalog(catalog_data, catalog_label)
    findings.extend(catalog_findings)

    learner_files = iter_learner_files(learners_dir)
    seen_learner_ids: dict[str, str] = {}
    seen_pseudonyms: dict[str, str] = {}

    for path in learner_files:
        label = rel(path)
        try:
            card, had_bom = load_json(path)
        except json.JSONDecodeError as e:
            findings.append(_err(label, "/", f"ошибка разбора JSON: строка {e.lineno}, колонка {e.colno}: {e.msg}"))
            continue
        except UnicodeDecodeError as e:
            findings.append(_err(label, "/", f"ошибка кодировки файла: {e}"))
            continue

        if had_bom:
            findings.append(_warn(label, "/", "файл начинается с BOM (byte order mark)"))

        card_findings = validate_card(card, catalog_index, label)
        findings.extend(card_findings)

        if isinstance(card, dict):
            learner_id = card.get("learner_id")
            if isinstance(learner_id, str):
                if learner_id in seen_learner_ids:
                    findings.append(
                        _err(label, "/learner_id", f'learner_id "{learner_id}" уже используется в файле "{seen_learner_ids[learner_id]}"')
                    )
                else:
                    seen_learner_ids[learner_id] = label

            pseudonym = card.get("pseudonym")
            if isinstance(pseudonym, str):
                if pseudonym in seen_pseudonyms:
                    findings.append(
                        _err(label, "/pseudonym", f'pseudonym "{pseudonym}" уже используется в файле "{seen_pseudonyms[pseudonym]}"')
                    )
                else:
                    seen_pseudonyms[pseudonym] = label

    return findings, len(learner_files)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_streams()

    package_root = find_package_root()
    parser = argparse.ArgumentParser(description="Validate learner-data catalog and learner cards.")
    parser.add_argument("--learners-dir", type=Path, default=package_root / "learners")
    parser.add_argument("--catalog", type=Path, default=package_root / "catalog" / "math_g3_g4.json")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors for exit code purposes")
    args = parser.parse_args(argv)

    try:
        findings, file_count = validate_all(args.learners_dir, args.catalog)
    except OSError as e:
        print(str(e), file=sys.stderr)
        return 2

    errors = [f for f in findings if f.is_error()]
    warnings = [f for f in findings if not f.is_error()]

    for finding in findings:
        print(finding.format())

    print(f"--- {file_count} files, {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        return 1
    if warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
