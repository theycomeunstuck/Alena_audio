#!/usr/bin/env python3
"""Обновление карточек учеников по итогам урока (LLM + правила ЗБР).

Карточки читаются из PostgreSQL (``teachcopilot.learners``) – источника истины.
``--apply`` сохраняет изменения в БД. Без него команда показывает разбор,
чтобы репетитор мог проверить изменения перед записью.

``--json-seed`` переключает чтение и запись на файлы начальных данных
``learner-data/learners/*.json``. Для CLI-отчётов по актуальным карточкам
выгрузите их из БД через ``scripts/export_learner_cards.py``.

    # предпросмотр, ничего не пишется
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt

    # одна заметка на всю группу
    python scripts/update_learner_card.py --learners zvezdin-artem,belka-06 --notes-file lesson.txt

    # записать в БД
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt --apply

    # режим «модель предлагает, репетитор подтверждает»
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt \
        --mode tutor_confirmed --confirm math.g3.geometry.point_line_ray_segment --apply

    # что уйдёт в модель, без её вызова
    python scripts/update_learner_card.py --learners zvezdin-artem --notes-file lesson.txt --prompt-only
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
from pipeline.learner_context import LearnerContextError, load_group, load_learner_data_tools  # noqa: E402
from pipeline.llm_json import LlmError  # noqa: E402

CLI_ACTOR_ID = "cli:update_learner_card"
TEACHER_ZONE_TITLES = {
    "mastered": "Самостоятельно",
    "zpd": "С поддержкой",
    "outside": "Пока не получилось с совместной помощью",
    "unknown": "Нужно проверить",
}


def open_card_store():
    from local_app.config import Settings
    from local_app.store import Store

    return Store(Settings.from_env().database_url)


def _read_notes(args: argparse.Namespace) -> str:
    if args.notes:
        return args.notes
    if args.notes_file:
        return args.notes_file.read_text(encoding="utf-8")
    if sys.stdin is not None and not sys.stdin.isatty():
        return sys.stdin.read()
    raise CardUpdateError("нет текста заметок: передайте --notes, --notes-file или подайте текст на stdin")


def _teacher_map_notes(card: dict) -> dict[str, str]:
    zpd = load_learner_data_tools().zpd
    return {
        row["topic_id"]: (
            f"{TEACHER_ZONE_TITLES[row['confirmed_zone']]} – {row['confirmed_reason']}"
        )
        for row in zpd.zone_report(card)
    }


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    # Предупреждения пайплайна (слишком длинный промпт, пустой content у
    # думающей модели, не найден материал) должны быть видны репетитору.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Обновить карточки учеников по итогам урока.",
        epilog="Для CLI-отчётов выгрузите актуальные карточки через scripts/export_learner_cards.py.",
    )
    parser.add_argument("--learners", help="learner_id через запятую")
    parser.add_argument("--group", help="состав группы из learner-data/groups, например detektivy-3")
    parser.add_argument("--notes", help="текст заметок прямо в аргументе")
    parser.add_argument("--notes-file", type=Path, help="файл с заметками об уроке")
    parser.add_argument("--apply", action="store_true",
                        help="записать изменения в PostgreSQL (источник истины), "
                             "а с --json-seed – в файлы начальных данных")
    parser.add_argument("--json-seed", action="store_true",
                        help="работать с файлами начальных данных learner-data/learners")
    parser.add_argument("--mode", choices=MASTERY_MODES, default=MASTERY_MODE,
                        help="auto_ema – считать сразу, tutor_confirmed – только подтверждённые темы")
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
            learner_ids = load_group(args.group, require_cards=args.json_seed)
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

    from psycopg2 import Error as PostgreSQLError

    try:
        store = None if args.json_seed else open_card_store()
        if args.prompt_only:
            roster_text, _cards, _catalog = build_roster(learner_ids, store=store)
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
            store=store,
            actor_id=CLI_ACTOR_ID,
        )
        for result in results:
            applied = args.apply and result.changed()
            if args.json_seed:
                teacher_map = {}
                missing_topic = "не вычисляется в режиме --json-seed"
            else:
                # The preview's EMA is hypothetical. The map shows the saved card.
                card = result.card if applied else store.get_card(result.learner_id)
                teacher_map = _teacher_map_notes(card or {})
                missing_topic = "тема пока отсутствует на карте"
            print(render_update(
                result, applied=applied, teacher_map=lambda topic: teacher_map.get(topic, missing_topic),
            ))
            print()
    except PostgreSQLError:
        print("Не удалось обратиться к PostgreSQL. Проверьте доступность БД, "
              "настройки подключения (DATABASE_URL) и применение миграций.", file=sys.stderr)
        return 1
    except (CardUpdateError, LearnerContextError) as exc:
        print(f"Обновление не выполнено: {exc}", file=sys.stderr)
        return 1
    except LlmError as exc:
        print(f"Проблема с LLM: {exc}", file=sys.stderr)
        return 3

    if not results:
        print("Модель не нашла в заметках ничего, что можно записать в карточки.")
        return 0

    if not args.apply:
        print("Ничего не записано. Повторите с --apply, чтобы сохранить изменения.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
