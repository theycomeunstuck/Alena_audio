#!/usr/bin/env python3
"""Новый ученик из свободного описания репетитора.

    # предпросмотр, файл не создаётся
    python scripts/create_learner_card.py --description "Миша, 4 класс, любит динозавров
        и Minecraft. Деление столбиком не идёт — теряет ноль. Счёт до 1000 уверенный."

    # то же из файла и с записью
    python scripts/create_learner_card.py --description-file misha.txt --apply
    python scripts/create_learner_card.py --description-file misha.txt --learner-id petrov-misha --apply

Без ``--apply`` команда только показывает черновик. Оценки освоения не
заполняются намеренно: их посчитает код после первого занятия по уровню помощи,
который понадобился ребёнку.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.card_creator import (  # noqa: E402
    CardCreateError,
    create_from_description,
    render_draft,
    write_draft,
)
from pipeline.learner_context import LearnerContextError  # noqa: E402
from pipeline.llm_json import LlmError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Завести карточку ученика по описанию.")
    parser.add_argument("--description", help="описание ребёнка прямо в аргументе")
    parser.add_argument("--description-file", type=Path, help="файл с описанием")
    parser.add_argument("--learner-id", help="задать идентификатор вручную")
    parser.add_argument("--grade", choices=("3", "4"), help="подсказать класс, если в тексте его нет")
    parser.add_argument("--apply", action="store_true", help="создать файл карточки")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if args.description:
        description = args.description
    elif args.description_file:
        description = args.description_file.read_text(encoding="utf-8")
    elif sys.stdin is not None and not sys.stdin.isatty():
        description = sys.stdin.read()
    else:
        print("Нужно описание: --description, --description-file или текст на stdin", file=sys.stderr)
        return 2

    try:
        draft = create_from_description(
            description, learner_id=args.learner_id, grade=args.grade
        )
    except (CardCreateError, LearnerContextError) as exc:
        print(f"Карточка не собрана: {exc}", file=sys.stderr)
        return 1
    except LlmError as exc:
        print(f"Проблема с LLM: {exc}", file=sys.stderr)
        return 3

    if args.format == "json":
        print(json.dumps(draft.card, ensure_ascii=False, indent=2))
    else:
        print(render_draft(draft))

    if not args.apply:
        print()
        print("Ничего не создано. Проверьте черновик и повторите с --apply.")
        return 0

    try:
        path = write_draft(draft)
    except CardCreateError as exc:
        print(f"Не записано: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Создано: {path}")
    print("Дальше: python learner-data/tools/validate.py --strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
