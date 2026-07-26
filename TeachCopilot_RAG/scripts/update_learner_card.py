#!/usr/bin/env python3
"""Итоги урока текстом → обновление JSON-карточек учеников (LLM + правила ЗБР).

    # предпросмотр, ничего не пишется
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt

    # одна заметка на всю группу
    python scripts/update_learner_card.py --learners zvezdin-artem,belka-06 --notes-file lesson.txt

    # записать
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt --apply

    # режим «модель предлагает, репетитор подтверждает»
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt \
        --mode tutor_confirmed --confirm math.g3.geometry.point_line_ray_segment --apply

    # что уйдёт в модель, без её вызова
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt --prompt-only

Без ``--apply`` команда только показывает разбор. Это намеренно: карточка — это
педагогические данные о ребёнке, и репетитор читает диф до записи.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.card_updater import (  # noqa: E402
    CardUpdateError,
    ask_tutor,
    build_roster,
    build_update_prompt,
    load_update_prompt,
    render_update,
    update_from_notes,
)
from pipeline.config import MASTERY_MODE, MASTERY_MODES, ROSTER_BATCH_SIZE  # noqa: E402
from pipeline.learner_context import LearnerContextError, load_group  # noqa: E402
from pipeline.llm_json import LlmError  # noqa: E402


def _read_notes(args: argparse.Namespace) -> str:
    if args.notes:
        return args.notes
    if args.notes_file:
        return args.notes_file.read_text(encoding="utf-8")
    if sys.stdin is not None and not sys.stdin.isatty():
        return sys.stdin.read()
    raise CardUpdateError("нет текста заметок: передайте --notes, --notes-file или подайте текст на stdin")


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Предупреждения пайплайна (слишком длинный промпт, пустой content у
    # думающей модели, не найден материал) должны быть видны репетитору.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Обновить карточки учеников по итогам урока.")
    parser.add_argument("--learners", help="learner_id через запятую")
    parser.add_argument("--group", help="состав группы из learner-data/groups, например detektivy-3")
    parser.add_argument("--notes", help="текст заметок прямо в аргументе")
    parser.add_argument("--notes-file", type=Path, help="файл с заметками об уроке")
    parser.add_argument("--apply", action="store_true", help="записать изменения в файлы карточек")
    parser.add_argument("--mode", choices=MASTERY_MODES, default=MASTERY_MODE,
                        help="auto_ema — считать сразу; tutor_confirmed — только подтверждённые темы")
    parser.add_argument("--confirm", default="",
                        help="topic_id через запятую, которые репетитор подтверждает (для tutor_confirmed); "
                             "значение all подтверждает всё")
    parser.add_argument("--prompt-only", action="store_true", help="показать промпт и выйти, LLM не вызывается")
    parser.add_argument("--force", action="store_true",
                        help="применить, даже если похоже, что эти итоги уже записаны сегодня")
    parser.add_argument("--interactive", action="store_true",
                        help="спросить по каждому пункту разбора: принять или отклонить")
    parser.add_argument("--batch", type=int, default=None,
                        help=f"сколько учеников уходит в модель за один запрос (по умолчанию {ROSTER_BATCH_SIZE})")
    args = parser.parse_args(argv)

    try:
        if args.group:
            learner_ids = load_group(args.group)
        else:
            learner_ids = [item.strip() for item in (args.learners or "").split(",") if item.strip()]
    except LearnerContextError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not learner_ids:
        print("Укажите --learners или --group", file=sys.stderr)
        return 2

    try:
        notes = _read_notes(args)
    except CardUpdateError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.prompt_only:
            roster_text, _cards, _catalog = build_roster(learner_ids)
            print("=" * 70)
            print("SYSTEM PROMPT (prompts/card_update_prompt.txt)")
            print("=" * 70)
            print(load_update_prompt())
            print()
            print("=" * 70)
            print("USER PROMPT")
            print("=" * 70)
            print(build_update_prompt(notes, roster_text))
            return 0

        confirmed = [item.strip() for item in args.confirm.split(",") if item.strip()] or None

        results = update_from_notes(
            notes,
            learner_ids,
            apply=args.apply,
            mode=args.mode,
            confirmed_topics=confirmed,
            force=args.force,
            batch_size=args.batch,
            decide=ask_tutor if args.interactive else None,
        )
    except (CardUpdateError, LearnerContextError) as exc:
        print(f"Обновление не выполнено: {exc}", file=sys.stderr)
        return 1
    except LlmError as exc:
        print(f"Проблема с LLM: {exc}", file=sys.stderr)
        return 3

    if not results:
        print("Модель не нашла в заметках ничего, что можно записать в карточки.")
        return 0

    for result in results:
        print(render_update(result, applied=args.apply and result.changed()))
        print()

    if not args.apply:
        print("Ничего не записано. Повторите с --apply, чтобы сохранить изменения.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
