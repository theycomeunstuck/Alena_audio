#!/usr/bin/env python3
"""Проверка покрытия: есть ли учебный материал под темы, на которых стоят ученики.

Без материала генератор 4 этапа собирает задание вслепую: в промпт уходит
«материал не найден, новых правил не выдумывай». Поэтому разрыв между темами
учеников и содержимым ``knowledge-data/`` — это не мелочь, а прямой ограничитель
качества, и он должен быть виден одной командой.

    python scripts/check_material_coverage.py
    python scripts/check_material_coverage.py --format json
    python scripts/check_material_coverage.py --warn-only   # не падать на пробелах

Код возврата: 0 — покрытие полное, 1 — есть пробелы или битые topic_id, 2 — не
удалось прочитать данные.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import LEARNER_DATA_DIR  # noqa: E402
from pipeline.learner_context import load_learner_data_tools  # noqa: E402
from pipeline.retrieval import KNOWLEDGE_DIR, _load_task_bank  # noqa: E402


@dataclass
class Coverage:
    """Кто на какой теме стоит и чем эта тема обеспечена."""

    needed: dict[str, list[str]] = field(default_factory=dict)   # topic_id -> learner_ids
    material: dict[str, int] = field(default_factory=dict)       # topic_id -> число задач
    unknown_topics: list[tuple[str, str]] = field(default_factory=list)  # (topic_id, файл)
    grade_mismatch: list[tuple[str, str, str]] = field(default_factory=list)  # (task_id, topic_id, grade)

    @property
    def gaps(self) -> dict[str, list[str]]:
        return {topic: who for topic, who in self.needed.items() if topic not in self.material}

    @property
    def covered(self) -> dict[str, list[str]]:
        return {topic: who for topic, who in self.needed.items() if topic in self.material}

    def is_clean(self) -> bool:
        return not self.gaps and not self.unknown_topics and not self.grade_mismatch


def collect_needed_topics() -> dict[str, list[str]]:
    """Темы, по которым ученикам реально могут понадобиться задания.

    Берём то, что заявлено педагогом (цель, текущие темы, приоритетные
    трудности) плюс то, что выводит правило ЗБР. Освоенные темы и темы за
    пределами зоны сюда не попадают: задания по ним сейчас не собираются.
    """
    tools = load_learner_data_tools()
    learners_dir = LEARNER_DATA_DIR / "learners"

    needed: dict[str, list[str]] = {}
    for path in sorted(learners_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        card, _had_bom = tools.load_learner(learners_dir, path.stem)
        if not isinstance(card, dict):
            continue

        rag_context = card.get("rag_context") or {}
        topics = {(rag_context.get("current_goal") or {}).get("topic_id")}
        topics |= set(rag_context.get("current_topics") or [])
        topics |= {item.get("topic_id") for item in rag_context.get("priority_difficulties") or []}
        topics |= {
            verdict.topic_id
            for verdict in tools.zpd.classify_card(card)
            if verdict.zone == tools.zpd.ZONE_ZPD
        }
        for topic_id in topics:
            if isinstance(topic_id, str) and topic_id:
                needed.setdefault(topic_id, []).append(path.stem)
    return needed


def collect_material(knowledge_dir: Path, catalog: dict) -> tuple[dict[str, int], list, list]:
    """Что лежит в банке заданий и не битые ли у него ссылки на каталог."""
    material: dict[str, int] = {}
    unknown: list[tuple[str, str]] = []
    grade_mismatch: list[tuple[str, str, str]] = []

    for item in _load_task_bank(knowledge_dir):
        if not item.topic_id:
            continue  # тексты из books/ без темы — не считаем покрытием
        if item.topic_id not in catalog:
            unknown.append((item.topic_id, item.source))
            continue
        material[item.topic_id] = material.get(item.topic_id, 0) + 1

        expected_grade = catalog[item.topic_id].get("grade")
        if expected_grade is not None and f".g{expected_grade}." not in item.topic_id:
            grade_mismatch.append((item.topic, item.topic_id, str(expected_grade)))
    return material, unknown, grade_mismatch


def build_coverage(knowledge_dir: Path | None = None) -> Coverage:
    tools = load_learner_data_tools()
    catalog = tools.load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")
    material, unknown, grade_mismatch = collect_material(knowledge_dir or KNOWLEDGE_DIR, catalog)
    return Coverage(
        needed=collect_needed_topics(),
        material=material,
        unknown_topics=unknown,
        grade_mismatch=grade_mismatch,
    )


def render_text(coverage: Coverage, catalog: dict) -> str:
    def title(topic_id: str) -> str:
        return (catalog.get(topic_id) or {}).get("title_ru", topic_id)

    lines = [
        f"Тем, на которых стоят ученики: {len(coverage.needed)}",
        f"Из них с учебным материалом:   {len(coverage.covered)}",
        f"Всего задач в банке:           {sum(coverage.material.values())}",
        "",
    ]

    if coverage.gaps:
        lines.append("НЕТ МАТЕРИАЛА:")
        for topic_id, who in sorted(coverage.gaps.items()):
            lines.append(f"  {title(topic_id)} [{topic_id}]")
            lines.append(f"      ученики: {', '.join(sorted(who))}")
        lines.append("")
    else:
        lines.append("Пробелов нет: под каждую тему учеников есть материал.")
        lines.append("")

    if coverage.unknown_topics:
        lines.append("TOPIC_ID НЕ ИЗ КАТАЛОГА:")
        lines.extend(f"  {topic_id} (файл {source})" for topic_id, source in coverage.unknown_topics)
        lines.append("")

    if coverage.grade_mismatch:
        lines.append("КЛАСС НЕ СОВПАДАЕТ С TOPIC_ID:")
        lines.extend(f"  {task} — {topic_id}, ожидался класс {grade}" for task, topic_id, grade in coverage.grade_mismatch)
        lines.append("")

    lines.append("Покрыто (тема — задач в банке — учеников):")
    for topic_id, who in sorted(coverage.covered.items()):
        lines.append(f"  {title(topic_id)} — {coverage.material[topic_id]} — {len(who)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Проверить покрытие тем учеников учебным материалом.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--knowledge-dir", type=Path, default=KNOWLEDGE_DIR)
    parser.add_argument("--warn-only", action="store_true", help="всегда возвращать 0, даже если есть пробелы")
    args = parser.parse_args(argv)

    try:
        coverage = build_coverage(args.knowledge_dir)
        catalog = load_learner_data_tools().load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")
    except Exception as exc:  # noqa: BLE001 - CLI boundary, любая проблема чтения данных
        print(f"Не удалось собрать отчёт: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps({
            "needed": coverage.needed,
            "material": coverage.material,
            "gaps": coverage.gaps,
            "unknown_topics": coverage.unknown_topics,
            "grade_mismatch": coverage.grade_mismatch,
        }, ensure_ascii=False, indent=2))
    else:
        print(render_text(coverage, catalog))

    if coverage.is_clean() or args.warn_only:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
