"""Агрегация контракт-теста проверяется без модели: сводку считает чистый код.

Сам прогон требует запущенной LLM, поэтому здесь проверяется то, что можно
проверить всегда: классификация нарушений, подсчёт доли чистых прогонов и то,
что упавший прогон не роняет весь отчёт.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import check_llm_contract as checker  # noqa: E402


def test_warning_texts_map_to_categories():
    cases = {
        "задание 1, подсказка 1: содержит готовый ответ — опора не должна его выдавать": "answer_in_hint",
        "задание 2: в тексте для ребёнка встречается служебное «збр»": "internal_leak",
        "модель вернула topic_id «math.g4.geometry.angles» вместо «math.g3...» — исправлено": "wrong_topic",
        "задание 1: подсказок 1 вместо 2 — лестница короче правила": "hint_count",
        "заданий пришло 5 вместо 3 — лишние отброшены": "task_count",
        "задание 3: нет expected_answer — взрослому нечем проверять": "no_answer",
        "сюжетное вступление пустое — карточка без истории": "no_story",
        "совершенно новое замечание": "other",
    }
    for warning, expected in cases.items():
        assert checker.classify_warning(warning) == expected, warning


def test_summary_counts_clean_runs_and_pass_rate():
    results = [
        checker.RunResult("a", ok=True, seconds=2.0),
        checker.RunResult("a", ok=True, seconds=4.0, warnings=("заданий пришло 5 вместо 3",)),
        checker.RunResult("a", ok=False, seconds=1.0, error="a: LLM недоступна"),
        checker.RunResult("a", ok=True, seconds=3.0),
    ]
    summary = checker.summarise(results)

    assert summary["runs"] == 4
    assert summary["clean"] == 2
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["violations"] == {"task_count": 1}
    assert summary["seconds_avg"] == 3.0
    assert summary["seconds_max"] == 4.0


def test_repeated_category_in_one_run_counts_once():
    """Отчёт отвечает «в скольких прогонах», а не «сколько строк замечаний»."""
    results = [
        checker.RunResult(
            "a",
            ok=True,
            warnings=("задание 1: подсказок 1 вместо 2", "задание 2: подсказок 1 вместо 2"),
        )
    ]
    assert checker.summarise(results)["violations"] == {"hint_count": 1}


def test_summary_of_empty_input_does_not_divide_by_zero():
    assert checker.summarise([])["pass_rate"] == 0.0


def test_render_mentions_every_violation_category():
    summary = checker.summarise([
        checker.RunResult("a", ok=True, warnings=("подсказка 1 содержит готовый ответ",)),
    ])
    text = checker.render_text(summary)

    assert "подсказка выдаёт ответ" in text
    assert "answer_in_hint" in text


def test_failed_run_is_reported_not_swallowed():
    def explode(_prompt, _system):
        raise checker.LlmError("LLM недоступна (127.0.0.1:1234)")

    results = checker.run_checks(["zvezdin-artem"], runs=2, complete=explode)
    summary = checker.summarise(results)

    assert summary["runs"] == 2 and summary["failed"] == 2
    assert all("LLM недоступна" in error for error in summary["errors"])
