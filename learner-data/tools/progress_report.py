#!/usr/bin/env python3
"""Отчёт о динамике одного ученика за период — для репетитора и родителя.

Карточка хранит историю оценок (``learner_model.competencies[].history``), но
читать её глазами неудобно. Этот отчёт отвечает на вопрос «что изменилось»:
насколько выросло освоение, стало ли меньше нужно помощи, какие ошибки ещё
повторяются.

Отчёт намеренно говорит про наблюдаемое поведение, а не про способности
ребёнка: «стало хватать наводящего вопроса вместо разбора вместе», а не
«стал умнее».

    python learner-data/tools/progress_report.py zvezdin-artem
    python learner-data/tools/progress_report.py zvezdin-artem --since 2026-07-01
    python learner-data/tools/progress_report.py zvezdin-artem --format json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learner_common import (
        DATE_RE,
        LEARNER_ID_RE,
        configure_utf8_streams,
        find_package_root,
        load_catalog,
        load_learner,
    )
    import zpd
else:
    from .learner_common import (
        DATE_RE,
        LEARNER_ID_RE,
        configure_utf8_streams,
        find_package_root,
        load_catalog,
        load_learner,
    )
    from . import zpd


def _in_period(date: object, since: str | None, until: str | None) -> bool:
    """Даты в формате YYYY-MM-DD сравниваются как строки — порядок сохраняется."""
    if not isinstance(date, str) or not DATE_RE.match(date):
        return False
    if since and date < since:
        return False
    if until and date > until:
        return False
    return True


def _help_title(key: object) -> str:
    level = zpd.HELP_BY_KEY.get(key) if isinstance(key, str) else None
    return level.title_ru if level else "—"


def _topic_dynamics(competency: dict, since: str | None, until: str | None) -> dict | None:
    """Что случилось с одной темой за период: первая и последняя точки истории."""
    history = [
        entry for entry in (competency.get("history") or [])
        if isinstance(entry, dict) and _in_period(entry.get("date"), since, until)
    ]
    if not history:
        return None

    first, last = history[0], history[-1]
    mastery_before = first.get("mastery")
    mastery_after = last.get("mastery")
    delta = None
    if isinstance(mastery_before, (int, float)) and isinstance(mastery_after, (int, float)):
        delta = round(float(mastery_after) - float(mastery_before), 3)

    verdict = zpd.classify(competency)
    return {
        "topic_id": competency.get("topic_id"),
        "last_assessed": verdict.last_assessed,
        "stale_days": verdict.stale_days(until),
        "stale": verdict.is_stale(until),
        "observations": len(history),
        "from_date": first.get("date"),
        "to_date": last.get("date"),
        "mastery_before": mastery_before,
        "mastery_after": mastery_after,
        "mastery_delta": delta,
        "help_before": first.get("help_level"),
        "help_after": last.get("help_level"),
        "zone": verdict.zone,
        "zone_ru": verdict.label_ru(),
    }


def build_report(card: dict, catalog: dict, since: str | None = None, until: str | None = None) -> dict:
    """Собрать структуру отчёта. Персональные идентификаторы сюда не попадают."""
    learner_model = card.get("learner_model") or {}
    topics = [
        dynamics
        for competency in (learner_model.get("competencies") or [])
        if isinstance(competency, dict)
        for dynamics in [_topic_dynamics(competency, since, until)]
        if dynamics is not None
    ]

    journal = [
        entry for entry in ((card.get("mvp") or {}).get("journal") or [])
        if isinstance(entry, dict) and _in_period(entry.get("date"), since, until)
    ]
    points = [
        entry for entry in ((card.get("mvp") or {}).get("points_ledger") or [])
        if isinstance(entry, dict) and _in_period(entry.get("date"), since, until)
    ]
    lessons = [
        entry for entry in ((card.get("mvp") or {}).get("lessons") or [])
        if isinstance(entry, dict) and _in_period(entry.get("date"), since, until)
    ]
    errors = sorted(
        (item for item in (card.get("error_patterns") or []) if isinstance(item, dict)),
        key=lambda item: item.get("count", 0),
        reverse=True,
    )

    return {
        "learner_id": card.get("learner_id"),
        "pseudonym": card.get("pseudonym"),
        "grade": card.get("grade"),
        "since": since,
        "until": until,
        "topics": topics,
        "journal": journal,
        "lessons": len(lessons),
        "minutes": sum(entry.get("minutes", 0) or 0 for entry in lessons),
        "points_earned": sum(entry.get("points", 0) for entry in points),
        "error_patterns": [
            {
                "topic_id": item.get("topic_id"),
                "error_tag": item.get("error_tag"),
                "count": item.get("count"),
                "last_seen": item.get("last_seen"),
            }
            for item in errors
        ],
        "zpd_now": zpd.derive_zpd(card),
    }


def render_text(report: dict, catalog: dict) -> str:
    def title(topic_id: object) -> str:
        return (catalog.get(topic_id) or {}).get("title_ru", str(topic_id))

    period = " — ".join(filter(None, [report.get("since"), report.get("until")])) or "вся история"
    lines = [
        f"Отчёт: {report['pseudonym']} ({report['learner_id']}), {report['grade']} класс",
        f"Период: {period}",
        "",
    ]

    if not report["topics"]:
        lines.append("За этот период оценок по темам не записывали.")
    else:
        lines.append("Динамика по темам:")
        for topic in report["topics"]:
            before = "—" if topic["mastery_before"] is None else f"{topic['mastery_before']:.3f}"
            after = "—" if topic["mastery_after"] is None else f"{topic['mastery_after']:.3f}"
            delta = ""
            if topic["mastery_delta"] is not None:
                sign = "+" if topic["mastery_delta"] >= 0 else ""
                delta = f" ({sign}{topic['mastery_delta']:.3f})"
            lines.append(f"  {title(topic['topic_id'])}")
            lines.append(f"    освоение: {before} → {after}{delta}; сейчас {topic['zone_ru']}")
            if topic["help_before"] != topic["help_after"]:
                lines.append(
                    f"    помощь: было «{_help_title(topic['help_before'])}», "
                    f"стало «{_help_title(topic['help_after'])}»"
                )
            else:
                lines.append(f"    помощь: по-прежнему «{_help_title(topic['help_after'])}»")
            lines.append(
                f"    наблюдений: {topic['observations']} "
                f"({topic['from_date']} — {topic['to_date']})"
            )
            if topic["stale"]:
                lines.append(
                    f"    тему не проверяли {topic['stale_days']} дней — оценка могла устареть"
                )
    lines.append("")

    if report["error_patterns"]:
        lines.append("Повторяющиеся ошибки (всего за всё время):")
        for item in report["error_patterns"]:
            lines.append(
                f"  - {item['error_tag']} — {item['count']} набл., последнее {item['last_seen']}"
            )
        lines.append("")

    if report["journal"]:
        lines.append("Журнал занятий:")
        lines.extend(f"  {entry['date']}: {entry['note']}" for entry in report["journal"])
        lines.append("")

    if report["lessons"]:
        duration = f", всего {report['minutes']} мин" if report["minutes"] else ""
        lines.append(f"Занятий за период: {report['lessons']}{duration}")
        lines.append("")

    if report["points_earned"]:
        lines.append(f"Баллов за период: {report['points_earned']}")
        lines.append("")

    zpd_now = report["zpd_now"]
    lines.append("Сейчас в зоне ближайшего развития:")
    lines.extend(f"  - {title(topic_id)}" for topic_id in zpd_now["current"] or [])
    if not zpd_now["current"]:
        lines.append("  —")
    if zpd_now["outside"]:
        lines.append("Пока за пределами зоны:")
        lines.extend(f"  - {title(topic_id)}" for topic_id in zpd_now["outside"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_utf8_streams()

    package_root = find_package_root()
    parser = argparse.ArgumentParser(description="Отчёт о динамике ученика за период.")
    parser.add_argument("learner_id")
    parser.add_argument("--since", help="начало периода, YYYY-MM-DD")
    parser.add_argument("--until", help="конец периода, YYYY-MM-DD")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--learners-dir", type=Path, default=package_root / "learners")
    parser.add_argument("--catalog", type=Path, default=package_root / "catalog" / "math_g3_g4.json")
    args = parser.parse_args(argv)

    if not LEARNER_ID_RE.fullmatch(args.learner_id):
        print(f"Некорректный learner_id: '{args.learner_id}'", file=sys.stderr)
        return 2
    for value, name in ((args.since, "--since"), (args.until, "--until")):
        if value and not DATE_RE.match(value):
            print(f"{name} должен быть в формате YYYY-MM-DD", file=sys.stderr)
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

    report = build_report(card, catalog, args.since, args.until)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report, catalog))
    return 0


if __name__ == "__main__":
    sys.exit(main())
