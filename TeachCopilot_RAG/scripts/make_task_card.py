#!/usr/bin/env python3
"""ЗБР stage 4: индивидуальные карточки-задания для одного ребёнка или всей группы.

    # один ребёнок
    python scripts/make_task_card.py zvezdin-artem
    python scripts/make_task_card.py zvezdin-artem --topic math.g3.geometry.point_line_ray_segment
    python scripts/make_task_card.py zvezdin-artem --format child      # копия для ребёнка

    # вся группа сразу, на печать
    python scripts/make_task_card.py --learners zvezdin-artem,belka-06 --format html --out cards.html
    python scripts/make_task_card.py --all --format html --out cards.html

    # что уйдёт в модель — без вызова модели, сервисы не нужны
    python scripts/make_task_card.py zvezdin-artem --prompt-only

Печатная форма (``--format html``) кладёт сначала все детские карточки, потом
листы для взрослого с ответами и лестницей подсказок: стопку не нужно разбирать.

Если карточка не собралась для кого-то одного, остальные всё равно делаются, а
в конце печатается список неудач. Код возврата 1 означает «часть не собралась».
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

from pipeline.config import TASK_CARD_TASK_COUNT  # noqa: E402
from pipeline.learner_context import (  # noqa: E402
    LearnerContextError,
    list_learner_ids,
    load_group,
    load_task_design_context,
)
from pipeline.llm_json import LlmError  # noqa: E402
from pipeline.retrieval import retrieve_material  # noqa: E402
from pipeline.task_card_html import render_cards_html  # noqa: E402
from pipeline.task_designer import (  # noqa: E402
    TaskDesignError,
    build_task_prompt,
    card_to_json,
    design_task_card,
    load_designer_prompt,
    render_task_card,
)


def _resolve_learner_ids(args: argparse.Namespace) -> list[str]:
    if getattr(args, "group", None):
        return load_group(args.group)
    if args.all:
        return list_learner_ids()
    if args.learners:
        return [item.strip() for item in args.learners.split(",") if item.strip()]
    if args.learner_id:
        return [args.learner_id]
    raise LearnerContextError("укажите learner_id, --learners или --all")


def _print_prompt(learner_id: str, args: argparse.Namespace) -> None:
    context = load_task_design_context(learner_id, args.topic)
    materials = retrieve_material(
        context.retrieval_query,
        topic_id=context.topic_id,
        grade=context.grade,
        limit=3,
        index_path=args.index,
    )
    print("=" * 70)
    print(f"SYSTEM PROMPT (prompts/task_designer_prompt.txt) — {learner_id}")
    print("=" * 70)
    print(load_designer_prompt())
    print()
    print("=" * 70)
    print(f"USER PROMPT — {learner_id}")
    print("=" * 70)
    print(build_task_prompt(context, materials, max(1, args.count)))
    print()
    print(f"[источник материала: {materials[0].origin if materials else 'ничего не найдено'};"
          f" фрагментов: {len(materials)}]")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Предупреждения пайплайна (не найден материал, слишком длинный промпт,
    # пустой content у думающей модели) должны доходить до репетитора.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Собрать индивидуальные карточки-задания (4 этап ЗБР).")
    parser.add_argument("learner_id", nargs="?", help="например zvezdin-artem")
    parser.add_argument("--learners", help="несколько learner_id через запятую")
    parser.add_argument("--all", action="store_true", help="все карточки из learner-data/learners")
    parser.add_argument("--group", help="состав группы из learner-data/groups, например detektivy-3")
    parser.add_argument("--topic", help="topic_id темы урока; по умолчанию выбирает политика ЗБР")
    parser.add_argument("--count", type=int, default=TASK_CARD_TASK_COUNT, help="сколько заданий в карточке")
    parser.add_argument("--format", choices=("text", "json", "child", "html"), default="text",
                        help="text — с ответами и подсказками, child — только задания, "
                             "html — печатная форма, json — машинный вид")
    parser.add_argument("--out", type=Path, help="записать результат в файл вместо вывода на экран")
    parser.add_argument("--prompt-only", action="store_true", help="показать промпт и выйти, LLM не вызывается")
    parser.add_argument("--index", type=Path, help="путь к JSON-индексу из scripts/json_rag.py")
    args = parser.parse_args(argv)

    try:
        learner_ids = _resolve_learner_ids(args)
    except LearnerContextError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.format == "html" and not args.out and not args.prompt_only:
        # Перенаправление `> cards.html` в Windows PowerShell может записать файл
        # в UTF-16, и кириллица превратится в мусор. --out пишет UTF-8 сам.
        print("Для --format html укажите --out cards.html: перенаправление вывода "
              "в PowerShell портит кодировку файла.", file=sys.stderr)
        return 2

    if args.topic and len(learner_ids) > 1:
        print("Замечание: --topic задаёт одну тему всей группе; без него тему каждому "
              "выбирает правило ЗБР.", file=sys.stderr)

    if args.prompt_only:
        for learner_id in learner_ids:
            try:
                _print_prompt(learner_id, args)
            except LearnerContextError as exc:
                print(f"{learner_id}: {exc}", file=sys.stderr)
                return 2
        return 0

    cards = []
    failures: list[tuple[str, str]] = []
    interrupted = False
    for number, learner_id in enumerate(learner_ids, start=1):
        if len(learner_ids) > 1:
            print(f"[{number}/{len(learner_ids)}] {learner_id}...", file=sys.stderr)
        try:
            cards.append(design_task_card(learner_id, args.topic, count=args.count, index_path=args.index))
        except (LearnerContextError, TaskDesignError, LlmError) as exc:
            failures.append((learner_id, str(exc)))
        except KeyboardInterrupt:
            # Прогон на группу — это десятки минут работы модели. Прерывание не
            # должно выбрасывать уже собранные карточки.
            interrupted = True
            print(f"\nПрервано на {learner_id}. Сохраняю то, что уже собрано.", file=sys.stderr)
            break

    if not cards:
        for learner_id, reason in failures:
            print(f"{learner_id}: {reason}", file=sys.stderr)
        print("Не собрано ни одной карточки.", file=sys.stderr)
        return 1

    if args.format == "html":
        output = render_cards_html(cards)
    elif args.format == "json":
        if len(cards) > 1:
            output = "[\n" + ",\n".join(card_to_json(card) for card in cards) + "\n]"
        else:
            output = card_to_json(cards[0])
    else:
        show_answers = args.format == "text"
        output = ("\n" + "-" * 70 + "\n\n").join(
            render_task_card(card, show_answers=show_answers) for card in cards
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
        print(f"Записано: {args.out} ({len(cards)} карточек)", file=sys.stderr)
    else:
        print(output)

    total_warnings = sum(len(card.warnings) for card in cards)
    if total_warnings and args.format != "text":
        print(f"Замечаний к сгенерированному тексту: {total_warnings}. "
              f"Смотрите поле warnings или --format text.", file=sys.stderr)

    if failures:
        print(f"\nНе собрано карточек: {len(failures)}", file=sys.stderr)
        for learner_id, reason in failures:
            print(f"  {learner_id}: {reason}", file=sys.stderr)
    if interrupted:
        remaining = len(learner_ids) - len(cards) - len(failures)
        print(f"Осталось не обработано: {remaining}", file=sys.stderr)
    return 1 if (failures or interrupted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
