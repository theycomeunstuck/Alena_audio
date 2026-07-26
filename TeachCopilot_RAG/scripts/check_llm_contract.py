#!/usr/bin/env python3
"""Проверка: насколько конкретная модель соблюдает контракт 4 этапа.

Генератор не доверяет ответу модели и чинит его сам, но чинить приходится не
бесплатно: каждое исправление — это либо потерянная подсказка, либо задание не
той сложности. Этот скрипт прогоняет генерацию N раз и показывает, что именно
модель нарушает чаще всего. По результатам понятно, стоит ли менять модель,
температуру или формулировки в ``prompts/task_designer_prompt.txt``.

Нужна работающая LLM (LM Studio или другой OpenAI-совместимый рантайм).

    python scripts/check_llm_contract.py --runs 5
    python scripts/check_llm_contract.py --learners zvezdin-artem,ivanov-ivan --runs 3
    python scripts/check_llm_contract.py --runs 10 --model qwen/qwen3-4b --min-pass 0.7
    python scripts/check_llm_contract.py --runs 5 --format json

Код возврата: 0 — доля чистых прогонов не ниже --min-pass, 1 — ниже, 2 — не
удалось запустить ни одного прогона.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.config import TASK_CARD_TASK_COUNT, TASK_DESIGN_TEMPERATURE, TASK_DESIGN_TIMEOUT_SEC  # noqa: E402
from pipeline.learner_context import LearnerContextError  # noqa: E402
from pipeline.llm_json import LlmError, chat_text, json_response_format  # noqa: E402
from pipeline import task_designer  # noqa: E402
from pipeline.task_designer import TaskDesignError, design_task_card  # noqa: E402

# Категории нарушений. Порядок важен: первое совпадение выигрывает.
WARNING_CATEGORIES = (
    ("answer_in_hint", ("готовый ответ",), "подсказка выдаёт ответ"),
    ("internal_leak", ("служебное",), "служебная лексика в тексте ребёнку"),
    ("wrong_topic", ("topic_id",), "чужой topic_id"),
    ("hint_count", ("подсказ", "лестница короче"), "не то число ступеней подсказок"),
    ("task_count", ("заданий пришло",), "не то число заданий"),
    ("no_answer", ("expected_answer",), "нет ответа для взрослого"),
    ("no_story", ("сюжетное вступление",), "пустой сюжет"),
    ("no_error_check", ("проверка типичной ошибки",), "ни одно задание не бьёт по ошибке"),
    ("bad_arithmetic", ("проверьте счёт",), "неверный счёт в задании"),
    ("no_personalization", ("не опирается на интересы",), "сюжет мимо интересов ребёнка"),
    ("story_only_in_intro", ("только во вступлении",), "сюжет лишь во вступлении, не в заданиях"),
    ("copied_material", ("дословно повторяет",), "задание списано из материала"),
    ("empty_task", ("пустой текст задания", "не объект"), "пустое или битое задание"),
)


@dataclass
class RunResult:
    """Один прогон генерации."""

    learner_id: str
    ok: bool
    seconds: float = 0.0
    error: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return self.ok and not self.warnings


def classify_warning(warning: str) -> str:
    """Свести текст замечания к категории нарушения контракта."""
    lowered = warning.lower()
    for slug, markers, _title in WARNING_CATEGORIES:
        if any(marker in lowered for marker in markers):
            return slug
    return "other"


def category_title(slug: str) -> str:
    for candidate, _markers, title in WARNING_CATEGORIES:
        if candidate == slug:
            return title
    return "прочее"


def summarise(results: list[RunResult]) -> dict:
    """Свести прогоны в отчёт. Чистая функция — тестируется без модели."""
    total = len(results)
    failed = [item for item in results if not item.ok]
    clean = [item for item in results if item.clean]

    categories: dict[str, int] = {}
    for result in results:
        for slug in {classify_warning(warning) for warning in result.warnings}:
            categories[slug] = categories.get(slug, 0) + 1

    durations = [item.seconds for item in results if item.ok and item.seconds > 0]
    return {
        "runs": total,
        "failed": len(failed),
        "clean": len(clean),
        "pass_rate": round(len(clean) / total, 3) if total else 0.0,
        "violations": dict(sorted(categories.items(), key=lambda pair: pair[1], reverse=True)),
        "errors": [item.error for item in failed],
        "seconds_avg": round(sum(durations) / len(durations), 1) if durations else None,
        "seconds_max": round(max(durations), 1) if durations else None,
    }


def render_text(summary: dict) -> str:
    lines = [
        f"Прогонов: {summary['runs']}",
        f"Без единого замечания: {summary['clean']} ({summary['pass_rate'] * 100:.0f}%)",
        f"Совсем не собралось: {summary['failed']}",
    ]
    if summary["seconds_avg"] is not None:
        lines.append(f"Время генерации: в среднем {summary['seconds_avg']} с, максимум {summary['seconds_max']} с")
    lines.append("")

    if summary["violations"]:
        lines.append("Что модель нарушает (в скольких прогонах):")
        for slug, count in summary["violations"].items():
            lines.append(f"  {count:3d}  {category_title(slug)}  [{slug}]")
    else:
        lines.append("Нарушений контракта не было.")

    if summary["errors"]:
        lines.append("")
        lines.append("Ошибки прогонов:")
        for error in summary["errors"][:10]:
            lines.append(f"  - {error}")
    return "\n".join(lines)


def _make_completer(model: str | None, base_url: str | None, temperature: float, timeout: float,
                    json_mode: bool = True):
    """Тот же вызов, что в бою: иначе стенд меряет не тот путь, что работает.

    ``json_mode=False`` полезен отдельно — чтобы увидеть, сколько именно даёт
    строгий режим ответа на конкретном рантайме.
    """
    response_format = json_response_format() if json_mode else None

    def complete(prompt: str, system_prompt: str) -> str:
        return chat_text(
            prompt,
            system_prompt,
            model=model,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format,
        )

    return complete


def use_prompt_file(path: Path | None):
    """Подменить педагогический промпт, чтобы сравнить редакции на одних данных."""
    if path is None:
        return
    text = path.read_text(encoding="utf-8").strip()
    task_designer.load_designer_prompt = lambda: text


def run_checks(
    learner_ids: list[str],
    runs: int,
    *,
    topic: str | None = None,
    count: int = TASK_CARD_TASK_COUNT,
    complete=None,
    progress=None,
) -> list[RunResult]:
    """Прогнать генерацию runs раз на каждого ученика."""
    results: list[RunResult] = []
    for learner_id in learner_ids:
        for attempt in range(1, runs + 1):
            if progress:
                progress(learner_id, attempt, runs)
            started = time.perf_counter()
            try:
                card = design_task_card(learner_id, topic, count=count, complete=complete)
            except (TaskDesignError, LlmError, LearnerContextError) as exc:
                results.append(
                    RunResult(learner_id=learner_id, ok=False, seconds=time.perf_counter() - started,
                              error=f"{learner_id}: {exc}")
                )
                continue
            results.append(
                RunResult(
                    learner_id=learner_id,
                    ok=True,
                    seconds=time.perf_counter() - started,
                    warnings=card.warnings,
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Проверить, как модель соблюдает контракт 4 этапа.")
    parser.add_argument("--learners", default="zvezdin-artem", help="learner_id через запятую")
    parser.add_argument("--topic", help="одна тема для всех прогонов")
    parser.add_argument("--runs", type=int, default=5, help="прогонов на каждого ученика")
    parser.add_argument("--count", type=int, default=TASK_CARD_TASK_COUNT, help="заданий в карточке")
    parser.add_argument("--model", help="перекрыть модель рантайма")
    parser.add_argument("--base-url", help="перекрыть адрес OpenAI-совместимого рантайма")
    parser.add_argument("--temperature", type=float, default=TASK_DESIGN_TEMPERATURE)
    parser.add_argument("--timeout", type=float, default=TASK_DESIGN_TIMEOUT_SEC)
    parser.add_argument("--min-pass", type=float, default=0.8, help="минимальная доля чистых прогонов")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--prompt-file", type=Path,
                        help="взять педагогический промпт из этого файла (для сравнения редакций)")
    parser.add_argument("--save", type=Path, help="сохранить сводку в JSON-файл")
    parser.add_argument("--no-json-mode", action="store_true",
                        help="не просить строгий JSON — чтобы измерить, сколько даёт этот режим")
    args = parser.parse_args(argv)

    use_prompt_file(args.prompt_file)

    learner_ids = [item.strip() for item in args.learners.split(",") if item.strip()]
    if not learner_ids or args.runs < 1:
        print("Нужен хотя бы один learner_id и --runs >= 1", file=sys.stderr)
        return 2

    def progress(learner_id: str, attempt: int, total: int) -> None:
        print(f"[{learner_id} {attempt}/{total}]", file=sys.stderr)

    results = run_checks(
        learner_ids,
        args.runs,
        topic=args.topic,
        count=args.count,
        complete=_make_completer(args.model, args.base_url, args.temperature, args.timeout,
                                 json_mode=not args.no_json_mode),
        progress=progress,
    )
    if not results:
        print("Не выполнено ни одного прогона", file=sys.stderr)
        return 2

    summary = summarise(results)
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_text(summary))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary["pass_rate"] < args.min_pass:
        print(
            f"\nДоля чистых прогонов {summary['pass_rate']:.0%} ниже порога {args.min_pass:.0%}.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
