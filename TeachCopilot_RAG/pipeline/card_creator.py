"""Свободное описание ученика от репетитора → черновик карточки.

Раньше нового ребёнка заводили руками из ``_TEMPLATE.json``: два десятка полей,
половина из которых обязательные, и любая опечатка в ``topic_id`` роняла
валидацию. Здесь репетитор пишет как говорит — «Миша, четвёртый класс, любит
динозавров, деление столбиком не идёт» — а код собирает из этого валидную
карточку.

Разделение ответственности то же, что и везде в проекте:

* модель извлекает факты и формулировки, но не выдумывает числа освоения и не
  придумывает темы вне каталога;
* обязательный каркас (schema_version, пустые блоки mvp, структура rag_context)
  ставит код — модели незачем помнить контракт целиком;
* результат проходит полную валидацию learner-data до того, как его покажут
  человеку, и записывается только по явному подтверждению.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import LEARNER_DATA_DIR
from pipeline.learner_context import (
    LearnerContextError,
    list_learner_ids,
    load_learner_data_tools,
)
from pipeline.llm_json import CompleteFn, complete_json

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "card_create_prompt.txt"

_FALLBACK_PROMPT = (
    "Разложи описание ученика по JSON: learner_id, pseudonym, age_group, grade, "
    "interests, story_preferences, profile, knowledge, error_patterns, current_goal. "
    "Числа освоения не выставляй, темы бери только из каталога."
)

# Значения по умолчанию для полей, которые обязательны по контракту, но которых
# в описании обычно нет. Пустая строка карточку не пройдёт, а выдумывать за
# репетитора нельзя — поэтому честная заглушка «пока не наблюдали».
_UNKNOWN = "пока не наблюдали"
MAX_TEXT = 280


class CardCreateError(RuntimeError):
    """Черновик карточки не удалось собрать."""


@dataclass
class CardDraft:
    """Черновик плюс всё, что человек должен проверить перед записью."""

    learner_id: str
    card: dict
    warnings: tuple[str, ...] = field(default_factory=tuple)
    model_notes: str = ""

    def path(self) -> Path:
        return LEARNER_DATA_DIR / "learners" / f"{self.learner_id}.json"


def load_create_prompt() -> str:
    try:
        return _PROMPT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Prompt file not found: %s — using fallback", _PROMPT_FILE)
        return _FALLBACK_PROMPT


def _clean(value: object, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit].rstrip()


def _clean_list(value: object, limit: int, item_limit: int = 80) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [_clean(item, item_limit) for item in value]
    return [item for item in items if item][:limit]


def build_catalog_block(catalog: dict, grade: str | None = None) -> str:
    """Список тем для промпта: модель не должна выдумывать идентификаторы."""
    lines = ["<catalog>"]
    for topic_id, entry in catalog.items():
        if grade and f".g{grade}." not in topic_id:
            continue
        if entry.get("deprecated"):
            continue
        lines.append(f"{topic_id} — {entry.get('title_ru', topic_id)}")
    lines.append("</catalog>")
    return "\n".join(lines)


def build_create_prompt(description: str, catalog: dict, grade: str | None = None) -> str:
    taken = ", ".join(list_learner_ids()) or "—"
    return "\n\n".join([
        build_catalog_block(catalog, grade),
        "\n".join(["<taken_ids>", f"Уже заняты, повторять нельзя: {taken}", "</taken_ids>"]),
        "\n".join(["<description>", description.strip(), "</description>"]),
        "\n".join(["<request>", "Разложи описание по контракту. Верни только JSON.", "</request>"]),
    ])


def _unique_learner_id(proposed: str, pseudonym: str) -> str:
    """Привести идентификатор к правилам и развести с уже существующими."""
    candidate = re.sub(r"[^a-z0-9-]+", "-", (proposed or "").lower()).strip("-")
    if not candidate:
        candidate = re.sub(r"[^a-z0-9-]+", "-", pseudonym.lower()).strip("-") or "uchenik"
    if not re.match(r"^[a-z0-9]", candidate):
        candidate = f"u-{candidate}"

    existing = set(list_learner_ids())
    if candidate not in existing:
        return candidate
    for suffix in range(2, 100):
        variant = f"{candidate}-{suffix}"
        if variant not in existing:
            return variant
    raise CardCreateError(f"не удалось подобрать свободный learner_id для «{candidate}»")


def _known_topic(topic_id: object, catalog: dict, grade: str) -> bool:
    return (
        isinstance(topic_id, str)
        and topic_id in catalog
        and f".g{grade}." in topic_id
    )


def build_card(payload: dict, catalog: dict, today: str, learner_id: str | None = None) -> CardDraft:
    """Собрать полную карточку из того, что вернула модель.

    Каркас ставит код: модель отвечает только за содержательные поля, а всё
    обязательное по контракту (пустые блоки mvp, структура rag_context,
    schema_version) добавляется здесь.
    """
    warnings: list[str] = []

    pseudonym = _clean(payload.get("pseudonym"), 60)
    if not pseudonym:
        raise CardCreateError("модель не назвала имя для обращения — уточните описание")

    taken_pseudonyms = set()
    tools = load_learner_data_tools()
    for existing_id in list_learner_ids():
        try:
            card, _ = tools.load_learner(LEARNER_DATA_DIR / "learners", existing_id)
            if isinstance(card, dict) and isinstance(card.get("pseudonym"), str):
                taken_pseudonyms.add(card["pseudonym"])
        except (OSError, ValueError):
            continue
    if pseudonym in taken_pseudonyms:
        warnings.append(
            f"псевдоним «{pseudonym}» уже занят другим учеником — придумайте другой, "
            "иначе карточку не примет валидатор"
        )

    grade = _clean(payload.get("grade"), 4)
    if grade not in ("3", "4"):
        # Класс можно восстановить по темам: если репетитор перечислил
        # программу четвёртого класса, то и ребёнок в четвёртом.
        mentioned = [
            item.get("topic_id") for item in (payload.get("knowledge") or [])
            if isinstance(item, dict) and isinstance(item.get("topic_id"), str)
        ]
        votes = {"3": sum(".g3." in t for t in mentioned), "4": sum(".g4." in t for t in mentioned)}
        inferred = "4" if votes["4"] > votes["3"] else "3"
        warnings.append(
            f"класс «{grade or '—'}» не распознан, поставил {inferred} по перечисленным темам — проверьте"
        )
        grade = inferred

    resolved_id = learner_id or _unique_learner_id(_clean(payload.get("learner_id"), 60), pseudonym)

    knowledge: list[dict] = []
    seen_topics: set[str] = set()
    for item in payload.get("knowledge") or []:
        if not isinstance(item, dict):
            continue
        topic_id = item.get("topic_id")
        status = _clean(item.get("status"), 20)
        if not _known_topic(topic_id, catalog, grade):
            warnings.append(f"тема «{topic_id}» пропущена: не из каталога {grade} класса")
            continue
        if topic_id in seen_topics:
            continue
        if status not in ("not_started", "learning", "confident", "needs_support"):
            warnings.append(f"у темы «{topic_id}» непонятный статус «{status}», поставил learning")
            status = "learning"
        seen_topics.add(topic_id)
        entry = {"topic_id": topic_id, "status": status}
        note = _clean(item.get("notes"))
        if note:
            entry["notes"] = note
        knowledge.append(entry)

    error_patterns = []
    for item in payload.get("error_patterns") or []:
        if not isinstance(item, dict):
            continue
        topic_id = item.get("topic_id")
        error_tag = _clean(item.get("error_tag"))
        if not error_tag:
            continue
        if not _known_topic(topic_id, catalog, grade):
            warnings.append(f"ошибка «{error_tag}» пропущена: тема не из каталога {grade} класса")
            continue
        error_patterns.append({
            "subject": "math",
            "topic_id": topic_id,
            "error_tag": error_tag,
            "count": 1,
            "last_seen": today,
        })

    goal_block = payload.get("current_goal") if isinstance(payload.get("current_goal"), dict) else {}
    goal_topic = goal_block.get("topic_id")
    if not _known_topic(goal_topic, catalog, grade):
        # Цель обязательна по контракту. Берём первую тему, которая не идёт,
        # затем ту, что в работе: это и есть ближайшая работа с ребёнком.
        fallback = next(
            (item["topic_id"] for item in knowledge if item["status"] == "needs_support"),
            next((item["topic_id"] for item in knowledge if item["status"] == "learning"), None),
        )
        if fallback is None:
            raise CardCreateError(
                "не удалось определить учебную цель: в описании нет ни одной темы из каталога"
            )
        warnings.append(f"учебная цель не названа явно — поставил «{fallback}», проверьте")
        goal_topic = fallback

    goal_text = _clean(goal_block.get("goal")) or f"продвинуться по теме {goal_topic}"

    profile_payload = payload.get("profile") if isinstance(payload.get("profile"), dict) else {}
    profile = {
        "explanation_style": _clean(profile_payload.get("explanation_style")) or _UNKNOWN,
        "pace": _clean(profile_payload.get("pace")) or _UNKNOWN,
        "autonomy_level": _clean(profile_payload.get("autonomy_level")) or _UNKNOWN,
        "motivation": _clean(profile_payload.get("motivation")) or _UNKNOWN,
        "prefers_visual": bool(profile_payload.get("prefers_visual")),
    }
    for key, value in profile.items():
        if value == _UNKNOWN:
            warnings.append(f"в описании нет данных про «{key}» — заполните после первого занятия")

    card = {
        "schema_version": 2,
        "learner_id": resolved_id,
        "pseudonym": pseudonym,
        "age_group": _clean(payload.get("age_group"), 20) or ("9-10" if grade == "3" else "10-11"),
        "grade": grade,
        "language": "russian",
        "rag_context": {
            "current_goal": {"topic_id": goal_topic, "goal": goal_text, "updated_at": today},
            "current_topics": [goal_topic],
            "priority_difficulties": [
                {"topic_id": item["topic_id"], "description": item["error_tag"]}
                for item in error_patterns[:3]
            ],
            "effective_strategies": _clean_list(payload.get("effective_strategies"), 3, MAX_TEXT)
                                    or ["подобрать опору на первом занятии"],
            "avoid": _clean_list(payload.get("avoid"), 3, MAX_TEXT)
                     or ["не давать длинную инструкцию без проверки"],
            "recent_progress": {"date": today, "note": "Карточка заведена, наблюдений ещё нет."},
        },
        "profile": profile,
        "interests": _clean_list(payload.get("interests"), 6),
        "knowledge": knowledge,
        "error_patterns": error_patterns,
        "mvp": {
            "story_preferences": _clean_list(payload.get("story_preferences"), 3),
            "learning_preferences_observed": [],
            "help_strategies": [],
            "journal": [{"date": today, "note": _clean(payload.get("notes")) or "Карточка заведена по описанию репетитора."}],
            "points_ledger": [],
        },
    }

    if not card["interests"]:
        warnings.append("интересы не названы — без них 4 этап не сможет построить сюжет")
    if not knowledge:
        warnings.append("ни одной темы из каталога не распознано — карточка почти пустая")

    return CardDraft(
        learner_id=resolved_id,
        card=card,
        warnings=tuple(warnings),
        model_notes=_clean(payload.get("notes")),
    )


def validate_draft(draft: CardDraft) -> list[str]:
    """Прогнать черновик через валидатор learner-data. Пустой список — можно писать."""
    tools = load_learner_data_tools()
    catalog = tools.load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")
    findings = tools.validate_card(draft.card, catalog, f"{draft.learner_id}.json")
    return [finding.format() for finding in findings if finding.is_error()]


def create_from_description(
    description: str,
    *,
    learner_id: str | None = None,
    grade: str | None = None,
    complete: CompleteFn | None = None,
    today: str | None = None,
) -> CardDraft:
    """Полный путь: текст репетитора → проверенный черновик карточки."""
    if not description or not description.strip():
        raise CardCreateError("описание пустое — нечего разбирать")

    tools = load_learner_data_tools()
    catalog = tools.load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")
    stamp = today or datetime.date.today().isoformat()

    payload = complete_json(
        build_create_prompt(description, catalog, grade),
        load_create_prompt(),
        complete=complete,
        temperature=0.1,
    )
    draft = build_card(payload, catalog, stamp, learner_id)

    errors = validate_draft(draft)
    if errors:
        raise CardCreateError(
            "черновик не прошёл валидацию: " + "; ".join(errors)
        )
    return draft


def write_draft(draft: CardDraft, *, overwrite: bool = False) -> Path:
    """Записать карточку. Существующий файл не перезаписывается без явного разрешения."""
    errors = validate_draft(draft)
    if errors:
        raise CardCreateError("карточка не прошла валидацию: " + "; ".join(errors))

    path = draft.path()
    if path.exists() and not overwrite:
        raise CardCreateError(f"файл {path.name} уже существует — задайте другой --learner-id")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft.card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Learner card created: %s", path)
    return path


def render_draft(draft: CardDraft) -> str:
    """Человекочитаемый черновик: что распозналось и что надо проверить."""
    card = draft.card
    lines = [
        f"Черновик карточки: {card['pseudonym']} ({draft.learner_id}), {card['grade']} класс",
        f"Возрастная группа: {card['age_group']}",
        "",
        f"Интересы: {', '.join(card['interests']) or '—'}",
        f"Любимые сюжеты: {', '.join(card['mvp']['story_preferences']) or '—'}",
        "",
        "Как работать:",
        f"  объяснять: {card['profile']['explanation_style']}",
        f"  темп: {card['profile']['pace']}",
        f"  самостоятельность: {card['profile']['autonomy_level']}",
        f"  мотивация: {card['profile']['motivation']}",
        f"  визуал: {'да' if card['profile']['prefers_visual'] else 'нет'}",
        "",
        f"Учебная цель: {card['rag_context']['current_goal']['goal']}",
        f"  тема: {card['rag_context']['current_goal']['topic_id']}",
    ]

    if card["knowledge"]:
        lines.append("")
        lines.append("Темы:")
        for item in card["knowledge"]:
            note = f" — {item['notes']}" if item.get("notes") else ""
            lines.append(f"  [{item['status']}] {item['topic_id']}{note}")

    if card["error_patterns"]:
        lines.append("")
        lines.append("Типичные ошибки:")
        lines.extend(f"  - {item['error_tag']} ({item['topic_id']})" for item in card["error_patterns"])

    if draft.warnings:
        lines.append("")
        lines.append("Проверьте перед записью:")
        lines.extend(f"  ! {item}" for item in draft.warnings)

    lines.append("")
    lines.append("Оценки освоения не заполнены намеренно: их посчитает код после "
                 "первого занятия по уровню помощи.")
    return "\n".join(lines)
