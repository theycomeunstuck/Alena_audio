"""Проверки готового задания: считается ли арифметика и есть ли персонализация.

Детектор арифметики должен быть осторожным: ложная тревога на каждой второй
карточке хуже, чем пропуск. Поэтому тестов на «должен молчать» здесь больше,
чем на «должен ловить».
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.task_checks import check_arithmetic, check_personalization, evaluate_expression  # noqa: E402

CATCHES = [
    "8 · 5 = 45.",
    "P = (8 + 3) · 2 = 20 см",
    "20 : 4 = 6, потом 6 · 3 = 18",
    "1) 5 · 12 = 60 кг. 2) 3 · 10 = 35 кг.",
    "Всего: 12 + 16 + 28 = 54 км",
    "56 : 8 = 8",
]

STAYS_SILENT = [
    # верные вычисления
    "56 : 8 = 7",
    "(2 + 4) · 5 = 30",
    "1) 5 · 12 = 60 кг. 2) 3 · 10 = 30 кг. 3) 60 + 30 = 90 кг",
    "Всего: 12 + 16 + 28 = 56 км",
    # перевод единиц: числа и не должны совпадать
    "1 кг = 1000 г",
    "3 т 4 ц = 3400 кг",
    "1 т = 10 ц",
    # деление с остатком верно только с оговоркой
    "85 : 4 = 21 (остаток 1)",
    # задание «найди ошибку» содержит неверное равенство по замыслу
    "Ученик записал 68 + 25 = 83. Найди ошибку.",
    "Ученик посчитал 20 − 4 · 3 = 48. Прав ли он?",
    # время считается не по десятичным правилам
    "9 ч 40 мин + 45 мин = 10 ч 25 мин",
    # уравнение с неизвестным: левый операнд не число
    "Реши уравнение x + 7 = 15.",
    "Найди пропущенное число: 7 · ? = 63",
    # текст без вычислений
    "Начерти луч от точки A.",
    "",
]


@pytest.mark.parametrize("text", CATCHES)
def test_broken_arithmetic_is_caught(text):
    assert check_arithmetic(text), f"не поймана ошибка: {text!r}"


@pytest.mark.parametrize("text", STAYS_SILENT)
def test_correct_or_unverifiable_text_is_not_flagged(text):
    assert check_arithmetic(text) == [], f"ложная тревога: {text!r}"


def test_message_shows_what_was_computed():
    problem = check_arithmetic("8 · 5 = 45")[0]
    assert "8 · 5 = 45" in problem and "40" in problem


def test_expression_evaluation_rejects_anything_but_arithmetic():
    assert evaluate_expression("(6 + 4) · 2") == 20
    for bad in ("__import__('os')", "2 ** 8", "x + 1", ""):
        with pytest.raises(ValueError):
            evaluate_expression(bad)


def test_personalization_matches_word_forms():
    assert check_personalization(
        ["Тёма, вор спрятал чертежи космической ракеты"], ("космос", "ракеты"), ("детектив",)
    )


def test_personalization_fails_when_story_ignores_interests():
    assert not check_personalization(["Реши пример 2 + 2."], ("космос", "ракеты"), ("детектив",))


def test_personalization_passes_when_there_is_nothing_to_match():
    assert check_personalization(["Реши пример 2 + 2."], (), ())
