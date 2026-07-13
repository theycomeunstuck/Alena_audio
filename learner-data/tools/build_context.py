#!/usr/bin/env python3
"""Build the AI-tutor context for one learner (text or JSON).

CLI: python learner-data/tools/build_context.py <learner_id> [--format text|json]
                                                 [--learners-dir DIR] [--catalog FILE]

Text format mirrors the colleague's pipeline/prompt_builder.py (read-only
reference, outside this repo) so the RAG system can consume it verbatim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learner_common import (
        configure_utf8_streams,
        find_package_root,
        load_catalog,
        load_learner,
    )
else:
    from .learner_common import (
        configure_utf8_streams,
        find_package_root,
        load_catalog,
        load_learner,
    )

REQUIRED_TOP_KEYS = (
    "schema_version", "learner_id", "pseudonym", "age_group", "grade",
    "language", "profile", "interests", "knowledge", "error_patterns", "mvp",
)
REQUIRED_PROFILE_KEYS = ("explanation_style", "pace", "autonomy_level", "motivation", "prefers_visual")
REQUIRED_MVP_KEYS = ("story_preferences", "learning_preferences_observed", "help_strategies", "journal", "points_ledger")


class CardShapeError(ValueError):
    """Raised when a loaded card is missing required keys for context building."""


def _check_card_shape(card: object) -> None:
    if not isinstance(card, dict):
        raise CardShapeError("карточка должна быть JSON-объектом")
    for key in REQUIRED_TOP_KEYS:
        if key not in card:
            raise CardShapeError(f'отсутствует обязательный ключ "{key}"')
    profile = card.get("profile")
    if not isinstance(profile, dict):
        raise CardShapeError('поле "profile" должно быть объектом')
    for key in REQUIRED_PROFILE_KEYS:
        if key not in profile:
            raise CardShapeError(f'отсутствует обязательное поле "profile/{key}"')
    mvp = card.get("mvp")
    if not isinstance(mvp, dict):
        raise CardShapeError('поле "mvp" должно быть объектом')
    for key in REQUIRED_MVP_KEYS:
        if key not in mvp:
            raise CardShapeError(f'отсутствует обязательное поле "mvp/{key}"')


def _topic_title(topic_id: str, catalog: dict) -> str:
    """title_ru for a topic_id via catalog; falls back to the id itself if unknown."""
    entry = catalog.get(topic_id)
    if entry is None:
        return topic_id
    return entry.get("title_ru", topic_id)


def build_context_text(card: dict, catalog: dict) -> str:
    """Render the text context block for one learner card. See task brief template."""
    parts: list[str] = []

    # --- child_safe_profile ---
    parts.append("\n<child_safe_profile>")
    parts.append(f"Имя_для_обращения: {card['pseudonym']}")
    parts.append(f"Возрастная_группа: {card['age_group']}")
    parts.append("</child_safe_profile>")

    # --- child_learning_profile ---
    profile = card["profile"]
    parts.append("\n<child_learning_profile>")
    parts.append(f"Школьный_уровень: {card['grade']} класс")
    parts.append(f"Язык: {card['language']}")
    parts.append(f"Предпочтительный_стиль_объяснения: {profile['explanation_style']}")
    parts.append(f"Темп: {profile['pace']}")
    parts.append(f"Уровень_самостоятельности: {profile['autonomy_level']}")
    parts.append(f"Мотивация: {profile['motivation']}")
    parts.append(f"Предпочитает_визуальное: {bool(profile['prefers_visual'])}")

    interests = card.get("interests") or []
    if interests:
        parts.append(f"\nИнтересы: {', '.join(interests)}")

    mvp = card["mvp"]
    story_preferences = mvp.get("story_preferences") or []
    if story_preferences:
        parts.append(f"\nПредпочитаемые_сюжеты: {', '.join(story_preferences)}")

    learning_prefs_observed = mvp.get("learning_preferences_observed") or []
    if learning_prefs_observed:
        parts.append(f"\nНаблюдения_о_стиле_обучения: {'; '.join(learning_prefs_observed)}")

    knowledge = card.get("knowledge") or []
    known = [k for k in knowledge if k.get("status") == "confident"]
    learning = [k for k in knowledge if k.get("status") == "learning"]
    struggling = [k for k in knowledge if k.get("status") == "needs_support"]
    # not_started skills are never rendered anywhere.

    if known:
        parts.append("\nЧто_уже_знает:")
        for k in known:
            parts.append(f"- {_topic_title(k['topic_id'], catalog)} [{k['topic_id']}]")

    if learning:
        parts.append("\nЧто_изучает_сейчас:")
        for k in learning:
            parts.append(f"- {_topic_title(k['topic_id'], catalog)} [{k['topic_id']}]")

    if struggling:
        parts.append("\nТипичные_трудности:")
        for k in struggling:
            line = f"- {_topic_title(k['topic_id'], catalog)} [{k['topic_id']}]"
            if k.get("notes"):
                line += f" — {k['notes']}"
            parts.append(line)

    parts.append("</child_learning_profile>")

    # --- error_patterns ---
    error_patterns = card.get("error_patterns") or []
    if error_patterns:
        sorted_errors = sorted(error_patterns, key=lambda e: e.get("count", 0), reverse=True)
        parts.append("\n<error_patterns>")
        parts.append("Типичные_ошибки:")
        for ep in sorted_errors[:10]:
            title = _topic_title(ep["topic_id"], catalog)
            parts.append(
                f"- {ep['error_tag']} (предмет: {ep['subject']}, тема: {title}, повторений: {ep['count']})"
            )
        parts.append("</error_patterns>")

    # --- help_strategies ---
    help_strategies = mvp.get("help_strategies") or []
    if help_strategies:
        parts.append("\n<help_strategies>")
        parts.append("Эффективные_приёмы_помощи:")
        for strategy in help_strategies:
            parts.append(f"- {strategy}")
        parts.append("</help_strategies>")

    # --- session_journal ---
    journal = mvp.get("journal") or []
    if journal:
        sorted_journal = sorted(journal, key=lambda j: j.get("date", ""), reverse=True)
        parts.append("\n<session_journal>")
        parts.append("Последние_наблюдения:")
        for entry in sorted_journal[:5]:
            parts.append(f"- {entry['date']}: {entry['note']}")
        parts.append("</session_journal>")

    # --- points_balance (always present) ---
    points_ledger = mvp.get("points_ledger") or []
    balance = sum(p.get("points", 0) for p in points_ledger)
    parts.append("\n<points_balance>")
    parts.append(f"Баллы: {balance}")
    parts.append("</points_balance>")

    return "\n".join(parts)


def build_context_json(card: dict) -> str:
    """Render the card as-is (no derived fields) as pretty JSON."""
    return json.dumps(card, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_streams()

    package_root = find_package_root()
    parser = argparse.ArgumentParser(description="Build AI-tutor context for one learner.")
    parser.add_argument("learner_id")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--learners-dir", type=Path, default=package_root / "learners")
    parser.add_argument("--catalog", type=Path, default=package_root / "catalog" / "math_g3_g4.json")
    args = parser.parse_args(argv)

    expected_path = args.learners_dir / f"{args.learner_id}.json"
    try:
        card, _had_bom = load_learner(args.learners_dir, args.learner_id)
    except FileNotFoundError:
        print(f"Ученик '{args.learner_id}' не найден: {expected_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"Ошибка разбора JSON в карточке ученика: строка {e.lineno}, колонка {e.colno}: {e.msg}", file=sys.stderr)
        return 1

    try:
        _check_card_shape(card)
    except CardShapeError as e:
        print(f"Некорректная карточка ученика: {e}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(build_context_json(card))
        return 0

    try:
        catalog = load_catalog(args.catalog)
    except (OSError, json.JSONDecodeError):
        catalog = {}

    print(build_context_text(card, catalog))
    return 0


if __name__ == "__main__":
    sys.exit(main())
