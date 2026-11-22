#!/usr/bin/env python3
"""Create and seed six clearly fictional, schema-valid rich demo learners."""
from __future__ import annotations

import argparse
import copy
import json
import os
import secrets
import stat
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEARNER_DATA = ROOT.parent / "learner-data"
FIXTURES = LEARNER_DATA / "demo" / "learners"
sys.path.insert(0, str(ROOT))

from local_app.auth import hash_password, verify_password
from local_app.config import Settings
from local_app.store import Store

ROSTER = [
    {"id": "demo-luna-03", "pseudonym": "\u0414\u0415\u041c\u041e \u2013 \u041b\u0443\u043d\u0430 \u041b\u0438\u0441", "grade": "3", "age": "8-9",
     "email": "demo.luna03@local.test", "interests": ["space maps", "baking", "fox stories"],
     "profile": {"explanation_style": "picture first, then one short rule", "pace": "steady with pauses to reread", "autonomy_level": "likes to try before asking", "motivation": "making a useful plan for a story", "prefers_visual": True},
     "skills": [
         ("math.g3.numbers.multiplication_table.times_table", 0.96, 0.91, [("2026-09-25", .68, .42, "joint"), ("2026-10-02", .82, .70, "hint_question"), ("2026-10-09", .96, .91, "independent")], ["array groups", "say the fact aloud"]),
         ("math.g3.word_problems.one_step", .72, .55, [("2026-10-01", .64, .46, "visual"), ("2026-10-10", .72, .55, "hint_question")], ["underline the question", "draw a bar model"]),
         ("programming.scratch.sequence", .91, .86, [("2026-09-27", .69, .53, "visual"), ("2026-10-11", .91, .86, "independent")], ["predict each block before running"]),
         ("math.g3.numbers.numeration_1000", .88, .82, [("2026-10-04", .75, .66, "visual"), ("2026-10-12", .88, .82, "hint_question")], ["сгруппировать сотни, десятки и единицы", "проверить разряд по таблице"]),
         ("math.g3.geometry.area_rectangle", .79, .62, [("2026-10-03", .61, .48, "joint"), ("2026-10-11", .79, .62, "visual")], ["покрыть прямоугольник квадратами", "сравнить площадь с периметром"]),
         ("programming.python.output", .68, .54, [("2026-10-05", .55, .39, "visual"), ("2026-10-12", .68, .54, "hint_question")], ["прочитать print по частям", "сначала предсказать строку вывода"]),
     ], "unknown": ["programming.python.debugging"],
     "errors": [("math.g3.word_problems.one_step", "chooses an operation before identifying the question", 3), ("programming.scratch.sequence", "skips a block while retelling a sequence", 1)],
     "strategies": ["draw equal groups", "ask what the question wants", "let Luna explain a prediction before checking"],
     "project": {"id": "luna-space-market", "title": "Space market price board", "description": "A Scratch market where each planet sells fruit in equal groups.", "status": "building", "skills": ["math.g3.numbers.multiplication_table.times_table", "programming.scratch.sequence"]},
     "ai": {"asks_clarifying_questions": True, "uses_hints": True, "copies_code_without_reading": False, "reflection": "Explains why a picture helped after solving."},
     "reading": "Reads short word problems confidently and benefits from a quiet reread of longer prompts."},
    {"id": "demo-mika-03", "pseudonym": "\u0414\u0415\u041c\u041e \u2013 \u041c\u0438\u043a\u0430 \u0421\u043e\u0441\u043d\u0430", "grade": "3", "age": "8-9",
     "email": "demo.mika03@local.test", "interests": ["skate parks", "building kits", "weather"],
     "profile": {"explanation_style": "hands-on example followed by a quick sketch", "pace": "fast on familiar tasks, slower on new words", "autonomy_level": "starts independently and checks in", "motivation": "designing a route that works", "prefers_visual": True},
     "skills": [
         ("math.g3.geometry.perimeter_polygon", .93, .88, [("2026-09-24", .74, .61, "visual"), ("2026-10-08", .93, .88, "independent")], ["trace the outside edge", "label each side once"]),
         ("math.g3.numbers.division_with_remainder", .24, .19, [("2026-10-03", .25, .22, "joint"), ("2026-10-12", .24, .19, "joint")], ["make equal piles with counters", "check what is left over"]),
         ("programming.scratch.loops", .69, .55, [("2026-10-02", .58, .40, "visual"), ("2026-10-10", .69, .55, "hint_question")], ["predict one repeat at a time"]),
         ("math.g3.word_problems.one_step", .61, .48, [("2026-10-04", .49, .36, "visual"), ("2026-10-12", .61, .48, "hint_question")], ["подчеркнуть вопрос", "показать данные на схеме"]),
         ("programming.scratch.sequence", .91, .83, [("2026-10-02", .72, .61, "visual"), ("2026-10-11", .91, .83, "independent")], ["проговорить блоки по порядку", "проверить стартовый блок"]),
         ("programming.python.output", .66, .52, [("2026-10-05", .54, .38, "visual"), ("2026-10-12", .66, .52, "hint_question")], ["отследить значение в кавычках", "прочитать вывод до запуска"]),
     ], "unknown": ["programming.python.variables"],
     "errors": [("math.g3.numbers.division_with_remainder", "shares the remainder as if it were another full group", 4), ("programming.scratch.loops", "counts the start block as a repeated action", 2)],
     "strategies": ["use counters before writing a number sentence", "ask Mika to point to one repeat", "compare the total before and after the loop"],
     "project": {"id": "mika-skate-route", "title": "Skate park route planner", "description": "A Scratch character follows a route with repeated ramps and turns.", "status": "planning", "skills": ["math.g3.geometry.perimeter_polygon", "programming.scratch.loops"]},
     "ai": {"asks_clarifying_questions": False, "uses_hints": True, "copies_code_without_reading": False, "reflection": "Stays engaged when a task has a visible build-and-test cycle."},
     "reading": "Understands short directions; unfamiliar comparison words need a concrete example."},
    {"id": "demo-nika-04", "pseudonym": "\u0414\u0415\u041c\u041e \u2013 \u041d\u0438\u043a\u0430 \u041a\u043e\u043c\u0435\u0442\u0430", "grade": "4", "age": "9-10",
     "email": "demo.nika04@local.test", "interests": ["graphic novels", "aquariums", "puzzles"],
     "profile": {"explanation_style": "worked example with labels and a chance to explain back", "pace": "careful and reflective", "autonomy_level": "works alone after the first example", "motivation": "solving a puzzle and teaching the method", "prefers_visual": True},
     "skills": [
         ("math.g4.fractions.numerator_denominator", .91, .58, [("2026-09-26", .83, .49, "joint"), ("2026-10-11", .91, .58, "hint_question")], ["shade the whole first", "name numerator and denominator from the picture"]),
         ("programming.scratch.conditions", .64, .49, [("2026-10-01", .57, .38, "visual"), ("2026-10-12", .64, .49, "hint_question")], ["act out the condition", "test both true and false cases"]),
         ("programming.python.output", .92, .84, [("2026-09-23", .73, .62, "hint_question"), ("2026-10-07", .92, .84, "independent")], ["predict the printed line before running"]),
         ("math.g4.geometry.angles", .87, .82, [("2026-10-03", .71, .60, "visual"), ("2026-10-12", .87, .82, "independent")], ["сравнить угол с прямым углом", "отметить вершину угла"]),
         ("math.g4.fractions.comparing_same_denominator", .83, .68, [("2026-10-04", .71, .52, "joint"), ("2026-10-11", .83, .68, "hint_question")], ["сравнить закрашенные части целого", "проверить, что знаменатели равны"]),
         ("programming.python.conditions", .71, .59, [("2026-10-02", .59, .42, "visual"), ("2026-10-12", .71, .59, "hint_question")], ["проверить истинный и ложный случаи", "прочитать условие вслух"]),
     ], "unknown": ["math.g4.data.bar_charts"],
     "errors": [("math.g4.fractions.numerator_denominator", "names the denominator before identifying the whole", 3), ("programming.scratch.conditions", "checks only the true branch", 2)],
     "strategies": ["mark the whole before shading a fraction", "ask what must be true for this block to run", "invite Nika to explain the example in her own words"],
     "project": {"id": "nika-aquarium-guide", "title": "Aquarium facts kiosk", "description": "A bilingual kiosk prints a short fact after the visitor chooses a fish.", "status": "testing", "skills": ["math.g4.fractions.numerator_denominator", "programming.scratch.conditions", "programming.python.output"]},
     "ai": {"asks_clarifying_questions": True, "uses_hints": True, "copies_code_without_reading": False, "reflection": "Checks examples carefully and notices when the output differs from a prediction."},
     "reading": "Reads grade-level text well; diagrams help separate vocabulary from the calculation."},
    {"id": "demo-ilya-04", "pseudonym": "\u0414\u0415\u041c\u041e \u2013 \u0418\u043b\u044c\u044f \u041e\u0440\u0431\u0438\u0442\u0430", "grade": "4", "age": "9-10",
     "email": "demo.ilya04@local.test", "interests": ["football stats", "model rockets", "riddles"],
     "profile": {"explanation_style": "brief verbal steps, then a numbered checklist", "pace": "quick start with a pause between steps", "autonomy_level": "confident on routines, asks for a first step on unfamiliar work", "motivation": "beating a personal score and comparing strategies", "prefers_visual": False},
     "skills": [
         ("math.g4.numbers.multidigit_addition_subtraction", .94, .89, [("2026-09-22", .76, .59, "hint_question"), ("2026-10-08", .94, .89, "independent")], ["estimate before calculating", "check each column"]),
         ("math.g4.word_problems.multi_step", .24, .23, [("2026-10-04", .30, .27, "joint"), ("2026-10-12", .24, .23, "joint")], ["mark the known quantities", "solve one question at a time"]),
         ("programming.python.variables", .68, .51, [("2026-10-02", .55, .38, "visual"), ("2026-10-11", .68, .51, "hint_question")], ["trace the value after each line"]),
         ("math.g4.quantities.unit_conversion", .62, .48, [("2026-10-04", .50, .36, "visual"), ("2026-10-12", .62, .48, "hint_question")], ["записать исходную и нужную единицу", "проверить направление перевода"]),
         ("math.g4.geometry.angles", .88, .81, [("2026-10-03", .73, .60, "visual"), ("2026-10-11", .88, .81, "independent")], ["сверить угол с прямым", "подписать вершину и стороны"]),
         ("programming.scratch.loops", .73, .61, [("2026-10-05", .58, .42, "visual"), ("2026-10-12", .73, .61, "hint_question")], ["записать состояние после каждого повтора", "сверить число повторений"]),
     ], "unknown": ["programming.scratch.events"],
     "errors": [("math.g4.word_problems.multi_step", "uses the first numbers without checking which step they belong to", 5), ("programming.python.variables", "uses a variable before tracing its latest value", 2)],
     "strategies": ["number the steps in the question", "write the variable value beside each line", "ask Ilya to estimate the answer before calculating"],
     "project": {"id": "ilya-match-stats", "title": "Match statistics tracker", "description": "A Python program stores team scores and prints a short match summary.", "status": "building", "skills": ["math.g4.numbers.multidigit_addition_subtraction", "programming.python.variables"]},
     "ai": {"asks_clarifying_questions": False, "uses_hints": True, "copies_code_without_reading": True, "reflection": "Benefits from asking what each variable means before changing code."},
     "reading": "Reads quickly and may skip a condition word such as each, left, or altogether."},
    {"id": "demo-rita-04", "pseudonym": "\u0414\u0415\u041c\u041e \u2013 \u0420\u0438\u0442\u0430 \u041f\u0438\u043a\u0441\u0435\u043b\u044c", "grade": "4", "age": "9-10",
     "email": "demo.rita04@local.test", "interests": ["pixel art", "cats", "craft recipes"],
     "profile": {"explanation_style": "one instruction at a time with a concrete worked model", "pace": "unhurried, especially when reading a dense prompt", "autonomy_level": "prefers checking each step together", "motivation": "making a design that can be shared", "prefers_visual": True},
     "skills": [
         ("math.g4.quantities.area_units", .72, .52, [("2026-10-01", .61, .41, "visual"), ("2026-10-10", .72, .52, "hint_question")], ["draw unit squares", "label the unit on the answer"]),
         ("math.g4.numbers.multiplication_by_1_2_digit", .48, .37, [("2026-10-03", .43, .32, "visual"), ("2026-10-12", .48, .37, "joint")], ["split the number into tens and ones"]),
         ("programming.scratch.variables", .76, .61, [("2026-09-28", .62, .46, "visual"), ("2026-10-09", .76, .61, "hint_question")], ["show the score changing one point at a time"]),
         ("math.g4.geometry.composite_area_perimeter", .58, .44, [("2026-10-03", .46, .33, "joint"), ("2026-10-12", .58, .44, "visual")], ["разбить фигуру на знакомые части", "подписать единицы площади и длины отдельно"]),
         ("math.g4.fractions.comparing_same_denominator", .71, .58, [("2026-10-04", .55, .41, "visual"), ("2026-10-11", .71, .58, "hint_question")], ["сравнить части на одной полоске", "назвать, что именно осталось одинаковым"]),
         ("programming.python.debugging", .63, .48, [("2026-10-05", .50, .36, "visual"), ("2026-10-12", .63, .48, "hint_question")], ["прочитать сообщение об ошибке до конца", "изменять только одну строку за раз"]),
     ], "unknown": ["programming.python.lists"],
     "errors": [("math.g4.quantities.area_units", "reports a perimeter unit for an area answer", 3), ("math.g4.numbers.multiplication_by_1_2_digit", "forgets to combine the partial products", 3), ("programming.scratch.variables", "sets a score instead of changing its current value", 2)],
     "strategies": ["cover all but the current instruction", "draw or shade each square", "ask Rita to predict how the score will change"],
     "project": {"id": "rita-cat-gallery", "title": "Pixel cat gallery", "description": "A Scratch gallery tracks visitor votes and shows the winning cat.", "status": "planning", "skills": ["math.g4.quantities.area_units", "programming.scratch.variables"]},
     "ai": {"asks_clarifying_questions": True, "uses_hints": True, "copies_code_without_reading": False, "reflection": "Confidence rises when the first task is short and the next step is visible."},
     "reading": "Long prompts can overload working memory; split text into short chunks and reread the final question."},
    {"id": "demo-oren-03", "pseudonym": "\u0414\u0415\u041c\u041e \u2013 \u041e\u0440\u0435\u043d \u041a\u0430\u0440\u0442\u0430", "grade": "3", "age": "8-9",
     "email": "demo.oren03@local.test", "interests": ["city maps", "train timetables", "board games"],
     "profile": {"explanation_style": "map or table first, then a spoken summary", "pace": "steady and consistent", "autonomy_level": "tries a plan and asks for a check at the end", "motivation": "finding the shortest route and explaining why", "prefers_visual": True},
     "skills": [
         ("math.g3.numbers.simple_equations", .93, .88, [("2026-09-20", .62, .39, "joint"), ("2026-09-30", .77, .64, "hint_question"), ("2026-10-11", .93, .88, "independent")], ["balance both sides with a drawing", "check by substitution"]),
         ("math.g3.numbers.multiplication_table.division_from_table", .82, .74, [("2026-10-02", .68, .51, "visual"), ("2026-10-12", .82, .74, "hint_question")], ["turn a fact family around"]),
         ("programming.scratch.events", .90, .85, [("2026-09-26", .72, .59, "hint_question"), ("2026-10-08", .90, .85, "independent")], ["name the event that starts the script"]),
         ("programming.python.loops", .71, .57, [("2026-10-01", .61, .45, "visual"), ("2026-10-12", .71, .57, "hint_question")], ["trace one loop pass before predicting the total"]),
         ("math.g3.data.reading_tables", .91, .87, [("2026-10-03", .74, .62, "visual"), ("2026-10-11", .91, .87, "independent")], ["найти заголовок нужного столбца", "сверить единицы в таблице"]),
         ("math.g3.quantities.time", .78, .64, [("2026-10-04", .62, .48, "visual"), ("2026-10-12", .78, .64, "hint_question")], ["разложить время на часы и минуты", "проверить переход через час"]),
         ("programming.python.variables", .69, .56, [("2026-10-02", .56, .39, "visual"), ("2026-10-10", .69, .56, "hint_question")], ["вести таблицу значения переменной", "проверить обновление после строки"]),
     ], "unknown": ["math.g3.word_problems.two_step"],
     "errors": [("math.g3.numbers.multiplication_table.division_from_table", "reads the related fact in the wrong direction", 2), ("programming.python.loops", "counts the initial value as an extra loop pass", 2)],
     "strategies": ["use a fact-family triangle", "mark the start event", "write one row per loop pass"],
     "project": {"id": "oren-route-map", "title": "Train route helper", "description": "A Scratch map and Python route summary compare stops and travel time.", "status": "building", "skills": ["math.g3.numbers.simple_equations", "programming.scratch.events", "programming.python.loops"]},
     "ai": {"asks_clarifying_questions": True, "uses_hints": False, "copies_code_without_reading": False, "reflection": "Can now solve simple equations independently after earlier joint modeling."},
     "reading": "Comfortable with tables and maps; word problems need the question highlighted before choosing operations."},
]

RUSSIAN_DETAILS = {
    "demo-luna-03": {
        "interests": ["карты космоса", "выпечка", "истории о лисах"],
        "profile": {"explanation_style": "сначала рисунок, затем одно короткое правило", "pace": "ровный темп с паузами на перечитывание", "autonomy_level": "любит сначала попробовать самостоятельно", "motivation": "составить полезный план для истории", "prefers_visual": True},
        "reading": "Короткие задачи читает уверенно; длинное условие полезно перечитать в спокойном темпе.",
        "strategies": ["нарисовать равные группы", "уточнить, что именно спрашивают", "попросить сначала предсказать результат"],
        "ai_reflection": "После решения объясняет, почему рисунок помог увидеть равные группы.",
        "error_tags": {"math.g3.word_problems.one_step": "выбирает действие до того, как определит вопрос задачи", "programming.scratch.sequence": "пропускает один блок, когда пересказывает последовательность"},
        "project": {"title": "Табло космического рынка", "description": "В Scratch планеты продают фрукты равными группами; таблица показывает цену нескольких покупок."},
    },
    "demo-mika-03": {
        "interests": ["скейт-парки", "конструкторы", "погода"],
        "profile": {"explanation_style": "сначала действие с предметами, затем короткий рисунок", "pace": "быстро на знакомом материале, медленнее при новых словах", "autonomy_level": "начинает сам и просит проверить ход решения", "motivation": "спроектировать маршрут, который работает", "prefers_visual": True},
        "reading": "Короткие инструкции понимает; новые слова сравнения лучше объяснить на конкретном примере.",
        "strategies": ["сначала разложить предметы на равные группы", "попросить показать один повтор", "сравнить результат до и после цикла"],
        "ai_reflection": "Видимый цикл проб и проверки помогает сохранять интерес к задаче.",
        "error_tags": {"math.g3.numbers.division_with_remainder": "считает остаток ещё одной полной группой", "programming.scratch.loops": "считает стартовый блок повторяемым действием"},
        "project": {"title": "Маршрут по скейт-парку", "description": "Персонаж в Scratch проезжает серию рамп и поворотов; Мика проверяет порядок блоков и число повторов."},
    },
    "demo-nika-04": {
        "interests": ["графические романы", "аквариумы", "головоломки"],
        "profile": {"explanation_style": "разобранный пример с подписями и просьбой объяснить метод своими словами", "pace": "внимательный, с паузой для проверки", "autonomy_level": "после первого примера обычно работает сама", "motivation": "решить головоломку и научить другого способу", "prefers_visual": True},
        "reading": "Текст по возрасту читает уверенно; схема помогает отделить новые слова от вычислений.",
        "strategies": ["сначала отметить целое", "проверить оба исхода условия", "попросить объяснить пример своими словами"],
        "ai_reflection": "Сравнивает прогноз с результатом и замечает, когда они не совпали.",
        "error_tags": {"math.g4.fractions.numerator_denominator": "называет знаменатель до того, как определит целое", "programming.scratch.conditions": "проверяет только истинную ветку условия"},
        "project": {"title": "Справочник обитателей аквариума", "description": "Интерактивный стенд печатает короткий факт о выбранной рыбе и проверяет оба исхода выбора."},
    },
    "demo-ilya-04": {
        "interests": ["спортивная статистика", "модельные ракеты", "загадки"],
        "profile": {"explanation_style": "короткие устные шаги, затем нумерованный список", "pace": "быстро начинает, затем делает паузу между этапами", "autonomy_level": "уверен в знакомых правилах, на новой теме просит показать первый шаг", "motivation": "улучшить личный результат и сравнить способы", "prefers_visual": False},
        "reading": "Читает быстро и может пропустить слово-условие, например «каждый», «осталось» или «всего».",
        "strategies": ["пронумеровать шаги условия", "записывать значение переменной рядом со строкой", "сначала оценить ответ, затем вычислять"],
        "ai_reflection": "Перед изменением кода полезно спросить, что хранит каждая переменная.",
        "error_tags": {"math.g4.word_problems.multi_step": "использует первые числа, не проверив, к какому шагу они относятся", "programming.python.variables": "использует переменную, не проследив её последнее значение"},
        "project": {"title": "Табло результатов матча", "description": "Программа на Python хранит счёт команд и печатает короткую сводку после матча."},
    },
    "demo-rita-04": {
        "interests": ["пиксельная графика", "кошки", "рецепты поделок"],
        "profile": {"explanation_style": "одна инструкция за раз и конкретный разобранный пример", "pace": "неторопливый, особенно при чтении длинного условия", "autonomy_level": "предпочитает проверять каждый шаг вместе", "motivation": "создать работу, которой можно поделиться", "prefers_visual": True},
        "reading": "Длинное условие удобнее делить на короткие части и отдельно перечитывать вопрос в конце.",
        "strategies": ["закрывать текст кроме текущей инструкции", "рисовать или закрашивать единичные квадраты", "сначала предсказывать изменение счёта"],
        "ai_reflection": "Уверенность растёт, когда первая задача короткая и следующий шаг виден заранее.",
        "error_tags": {"math.g4.quantities.area_units": "указывает единицу периметра в ответе о площади", "math.g4.numbers.multiplication_by_1_2_digit": "забывает сложить промежуточные произведения", "programming.scratch.variables": "заменяет счёт новым значением вместо изменения текущего"},
        "project": {"title": "Галерея пиксельных котов", "description": "Галерея в Scratch учитывает голоса посетителей и показывает кота, набравшего больше голосов."},
    },
    "demo-oren-03": {
        "interests": ["карты города", "расписания поездов", "настольные игры"],
        "profile": {"explanation_style": "сначала карта или таблица, затем устное объяснение", "pace": "ровный и последовательный", "autonomy_level": "составляет план и просит проверить его в конце", "motivation": "найти короткий маршрут и объяснить выбор", "prefers_visual": True},
        "reading": "Уверенно работает с картами и таблицами; в задаче полезно сначала выделить вопрос.",
        "strategies": ["использовать треугольник связанных фактов", "отметить событие, которое запускает скрипт", "записывать отдельную строку для каждого прохода цикла"],
        "ai_reflection": "После совместного разбора простого уравнения теперь объясняет самостоятельное решение подстановкой.",
        "error_tags": {"math.g3.numbers.multiplication_table.division_from_table": "читает связанный факт в обратном направлении", "programming.python.loops": "считает начальное значение дополнительным проходом цикла"},
        "project": {"title": "Помощник для поездки на поезде", "description": "Карта в Scratch и сводка в Python сравнивают остановки и время в пути."},
    },
}

RUSSIAN_CURRENT_GOALS = {
    "demo-luna-03": {
        "topic_id": "math.g3.word_problems.one_step",
        "goal": "Научиться выбирать действие в короткой задаче на один шаг: сначала подчеркнуть вопрос и нарисовать схему, затем решить две новые задачи без подсказки. Успех – оба ответа верны; после точной попытки убрать схему и попросить объяснить выбор действия.",
    },
    "demo-mika-03": {
        "topic_id": "programming.scratch.loops",
        "goal": "Предсказывать итог короткого цикла: сначала пройти один повтор наглядно, затем проверить два сценария с разным числом повторов. Успех – оба прогноза верны; после первой точной попытки убрать счётчик-подсказку.",
    },
    "demo-nika-04": {
        "topic_id": "math.g4.fractions.numerator_denominator",
        "goal": "Определять целое перед записью числителя и знаменателя: сначала подписать целое на рисунке, затем самостоятельно разобрать две полоски. Успех – обе записи верны без вопроса-подсказки; после первой точной попытки убрать подписи.",
    },
    "demo-ilya-04": {
        "topic_id": "programming.python.variables",
        "goal": "Отслеживать переменную по строкам: заполнить таблицу значений, затем предсказать вывод двух фрагментов. Успех – два верных прогноза подряд; после первого убрать таблицу. Многозначное сложение проверить заново: данные устарели.",
    },
    "demo-rita-04": {
        "topic_id": "math.g4.quantities.area_units",
        "goal": "Отличать площадь от периметра: покрыть прямоугольник единичными квадратами и указать квадратную единицу. Успех – два верных ответа на разных фигурах; сначала оставить сетку, после точного ответа убрать её и попросить объяснить выбор единицы.",
    },
    "demo-oren-03": {
        "topic_id": "math.g3.numbers.multiplication_table.division_from_table",
        "goal": "Связывать деление с известным умножением: решить два новых примера по треугольнику фактов, затем убрать схему. Успех – верно назвать делимое и делитель в обоих примерах. Уравнения Орен уже решает самостоятельно, деление пока – с подсказкой.",
    },
}

STRATEGY_TRANSLATIONS = {
    "array groups": "Нарисовать множители как равные группы предметов.",
    "say the fact aloud": "Произнести факт умножения и объяснить его рисунком.",
    "underline the question": "Подчеркнуть вопрос задачи до выбора действия.",
    "draw a bar model": "Показать известные и неизвестные величины на полосовой схеме.",
    "predict each block before running": "Предсказать действие каждого блока до запуска проекта.",
    "trace the outside edge": "Обвести внешний контур и посчитать каждую сторону один раз.",
    "label each side once": "Подписать длину каждой стороны, не повторяя её.",
    "make equal piles with counters": "Разложить предметы на равные группы и отдельно оставить остаток.",
    "check what is left over": "Проверить, сколько предметов осталось после деления на группы.",
    "predict one repeat at a time": "Предсказать результат одного повтора цикла, затем следующего.",
    "shade the whole first": "Сначала отметить целое, затем закрасить его части.",
    "name numerator and denominator from the picture": "Назвать числитель и знаменатель по рисунку целого.",
    "act out the condition": "Проверить условие на конкретном примере из жизни.",
    "test both true and false cases": "Проверить, что происходит при истинном и ложном условии.",
    "predict the printed line before running": "Предсказать строку вывода до запуска программы.",
    "estimate before calculating": "Сначала прикинуть ответ, затем выполнить точное вычисление.",
    "check each column": "Проверить вычисления отдельно в каждом разряде.",
    "mark the known quantities": "Отметить известные величины и вопрос задачи разными знаками.",
    "solve one question at a time": "Решать вопросы задачи по порядку и записывать промежуточный ответ.",
    "trace the value after each line": "Записывать значение переменной после каждой строки кода.",
    "balance both sides with a drawing": "Показать равенство как две одинаковые части на рисунке.",
    "check by substitution": "Подставить найденное число и проверить равенство.",
    "turn a fact family around": "Записать умножение и связанные с ним примеры деления.",
    "name the event that starts the script": "Назвать событие, которое запускает скрипт.",
    "trace one loop pass before predicting the total": "Проследить один проход цикла перед прогнозом общего результата.",
    "draw unit squares": "Покрыть фигуру единичными квадратами и посчитать их.",
    "label the unit on the answer": "Указать единицу измерения рядом с ответом.",
    "split the number into tens and ones": "Разложить число на десятки и единицы перед умножением.",
    "show the score changing one point at a time": "Показать, как счёт меняется на один балл после каждого события.",
}

LESSON_NOTES = {
    "independent": "Решил самостоятельно, проговорил план и проверил ответ.",
    "hint_question": "Решил после одного наводящего вопроса о следующем шаге.",
    "visual": "Решил, сверяясь с рисунком или наглядной моделью.",
    "joint": "Потребовалось вместе разобрать первый шаг и продолжить с учителем.",
}

# Demo evidence is deliberately independent of the score-map projection above.
# The strict zone report checks these dated, explicit attempts as of this fixed
# date so the fixture stays reproducible when the calendar advances.
DEMO_AS_OF = "2026-10-13"
CONFIRMED_EVIDENCE = {
    "demo-luna-03": {
        "topic": "math.g3.numbers.multiplication_table.times_table",
        "expected": "mastered",
        "attempts": [
            ("2026-10-07", "luna-times-01", "joint", True),
            ("2026-10-08", "luna-times-01", "independent", True),
            ("2026-10-09", "luna-times-02", "hint_question", True),
            ("2026-10-10", "luna-times-02", "independent", True),
        ],
        "explanation": "Сначала Луна решала вместе с учителем, затем по наводящему вопросу, а две сопоставимые задачи выполнила самостоятельно.",
    },
    "demo-mika-03": {
        "topic": "math.g3.numbers.division_with_remainder",
        "expected": "outside",
        "attempts": [
            ("2026-10-08", "mika-remainder-01", "joint", False),
            ("2026-10-09", "mika-remainder-01", "independent", False),
            ("2026-10-10", "mika-remainder-02", "joint", False),
            ("2026-10-11", "mika-remainder-02", "independent", False),
        ],
        "explanation": "В двух задачах одинаковой сложности Мика пока не справился ни самостоятельно, ни при максимальной помощи – совместном решении с учителем. Сначала стоит вернуться к делению на предметах и проверить необходимые предпосылки.",
    },
    "demo-nika-04": {
        "topic": "math.g4.fractions.numerator_denominator",
        "expected": "zpd",
        "attempts": [
            ("2026-10-08", "nika-fraction-01", "hint_question", True),
            ("2026-10-09", "nika-fraction-01", "independent", False),
            ("2026-10-10", "nika-fraction-02", "visual", True),
            ("2026-10-11", "nika-fraction-02", "independent", False),
        ],
        "explanation": "Ника выполнила обе задачи с наводящим вопросом или рисунком, но пока не решила их самостоятельно.",
    },
    "demo-ilya-04": {
        "topic": "math.g4.numbers.multidigit_addition_subtraction",
        "expected": "unknown",
        "evidence_status": "stale",
        "attempts": [
            ("2026-08-19", "ilya-addition-01", "hint_question", True),
            ("2026-08-20", "ilya-addition-01", "independent", True),
            ("2026-08-21", "ilya-addition-02", "visual", True),
            ("2026-08-22", "ilya-addition-02", "independent", True),
        ],
        "explanation": "Раньше Илья справлялся, но последние сравнимые наблюдения устарели; перед планированием нужно проверить навык снова.",
    },
    "demo-rita-04": {
        "topic": "math.g4.quantities.area_units",
        "expected": "unknown",
        "evidence_status": "insufficient",
        "attempts": [
            ("2026-10-11", "rita-area-01", "visual", True),
            ("2026-10-12", "rita-area-01", "independent", False),
        ],
        "explanation": "Есть только одна пара сопоставимых попыток; данных пока недостаточно, чтобы подтвердить зону.",
    },
    "demo-oren-03": {
        "topic": "math.g3.numbers.simple_equations",
        "expected": "mastered",
        "attempts": [
            ("2026-10-07", "oren-equation-01", "joint", True),
            ("2026-10-08", "oren-equation-01", "independent", True),
            ("2026-10-09", "oren-equation-02", "hint_question", True),
            ("2026-10-10", "oren-equation-02", "independent", True),
        ],
        "explanation": "Поддержка постепенно сократилась от совместного решения до вопроса-подсказки; обе сопоставимые задачи Орен решил самостоятельно.",
    },
}

# A recent, synthetic assessment is added to these topics when seeding PostgreSQL.
# The static cards retain their historical scenarios for reproducible reports.
LIVE_DEMO_TOPICS = {
    "demo-luna-03": {
        "mastered": "math.g3.numbers.multiplication_table.times_table",
        "zpd": "math.g3.word_problems.one_step",
        "outside": "programming.python.output",
    },
    "demo-mika-03": {
        "mastered": "math.g3.geometry.perimeter_polygon",
        "zpd": "programming.scratch.loops",
        "outside": "math.g3.numbers.division_with_remainder",
    },
    "demo-nika-04": {
        "mastered": "programming.python.output",
        "zpd": "math.g4.fractions.comparing_same_denominator",
        "outside": "programming.python.conditions",
    },
    "demo-ilya-04": {
        "mastered": "math.g4.geometry.angles",
        "zpd": "programming.python.variables",
        "outside": "math.g4.word_problems.multi_step",
    },
    "demo-rita-04": {
        "mastered": "programming.scratch.variables",
        "zpd": "math.g4.fractions.comparing_same_denominator",
        "outside": "math.g4.geometry.composite_area_perimeter",
    },
    "demo-oren-03": {
        "mastered": "math.g3.numbers.simple_equations",
        "zpd": "math.g3.numbers.multiplication_table.division_from_table",
        "outside": "programming.python.variables",
    },
}
LIVE_DEMO_ATTEMPTS = {
    "mastered": (("hint_question", True), ("independent", True)),
    "zpd": (("independent", False), ("visual", True)),
    "outside": (("independent", False), ("joint", False)),
}
LIVE_DEMO_DATE_KEY = "demo_map_assessed_on"


def build_card(config: dict[str, Any]) -> dict[str, Any]:
    details = RUSSIAN_DETAILS[config["id"]]
    profile = details["profile"]
    interests = details["interests"]
    reading = details["reading"]
    strategies = details["strategies"]
    ai_usage = {**config["ai"], "reflection": details["ai_reflection"]}
    active_project = {**config["project"], **details["project"]}
    competencies = []
    knowledge = []
    scaffolding = []
    current = []
    outside = []
    evidence_override = CONFIRMED_EVIDENCE.get(config["id"])
    for topic, mastery, independence, history, topic_strategies in config["skills"]:
        topic_strategies = [STRATEGY_TRANSLATIONS.get(item, item) for item in topic_strategies]
        attempts = None
        if evidence_override and topic == evidence_override["topic"]:
            attempts = evidence_override["attempts"]
        else:
            assistance = history[-1][3]
            if assistance == "independent":
                assistance = "hint_question"
            slug = topic.replace(".", "-")
            assisted_solved = mastery >= .30
            independent_solved = mastery >= .85 and independence >= .80
            attempts = [
                ("2026-10-08", f"{config['id']}-{slug}-pair01", assistance, assisted_solved),
                ("2026-10-09", f"{config['id']}-{slug}-pair01", "independent", independent_solved),
                ("2026-10-10", f"{config['id']}-{slug}-pair02", assistance, assisted_solved),
                ("2026-10-11", f"{config['id']}-{slug}-pair02", "independent", independent_solved),
            ]
        evidence_history = [(date, mastery, independence, support) for date, _pair_id, support, _solved in attempts]
        needs = mastery < .85 or independence < .80
        competency = {
            "topic_id": topic,
            "mastery": mastery,
            "confidence": min(.96, round(.36 + .17 * len(history), 2)),
            "independence": independence,
            "needs_scaffolding": needs,
            "effective_scaffolding": topic_strategies,
            "recommended_scaffolding": "visual" if needs and profile["prefers_visual"] else "hint_question",
            "last_assessed": evidence_history[-1][0],
            "history": [
                {"date": date, "mastery": score, "independence": independent, "help_level": help_level}
                for date, score, independent, help_level in evidence_history
            ],
        }
        competency["history"] = [
            {
                "date": date,
                "mastery": mastery,
                "independence": independence,
                "help_level": support,
                "support_level": support,
                "solved": solved,
                "task_id": pair_id,
                "assessment_pair_id": pair_id,
                "difficulty": "medium",
                "note": (
                    evidence_override["explanation"] if evidence_override and topic == evidence_override["topic"]
                    else ("Задача выполнена с опорой; затем проверили тот же тип задания без подсказки." if support != "independent"
                          else ("Решение выполнено самостоятельно." if solved else "Самостоятельная попытка пока не завершена верно."))
                ),
            }
            for date, pair_id, support, solved in attempts
        ]
        competencies.append(competency)
        if mastery < .30:
            zone, status = "outside", "needs_support"
            outside.append(topic)
        elif mastery >= .85 and independence >= .80:
            zone, status = "mastered", "confident"
        else:
            zone, status = "current", "learning"
            current.append(topic)
        knowledge.append({"topic_id": topic, "status": status, "notes": f"По последнему наблюдению: {LESSON_NOTES[history[-1][3]]}"})
        if needs:
            scaffolding.append({"topic_id": topic, "strategies": topic_strategies})

    for topic in config["unknown"]:
        knowledge.append({"topic_id": topic, "status": "not_started", "notes": "Структурированную проверку этой темы ещё не проводили; навык пока неизвестен."})

    topics = [item[0] for item in config["skills"]]
    goal = RUSSIAN_CURRENT_GOALS[config["id"]]
    goal_topic = goal["topic_id"]
    if goal_topic not in topics:
        raise ValueError(f"current goal topic is absent from {config['id']}")
    difficulties = [item for item in config["skills"] if item[1] < .85 or item[2] < .80]
    errors = [
        {"subject": "programming" if topic.startswith("programming.") else "math", "topic_id": topic, "error_tag": details["error_tags"].get(topic, tag), "count": count, "last_seen": "2026-10-12"}
        for topic, tag, count in config["errors"]
    ]
    ledger_events = [
        ("2026-10-08", topics[0], 3, "Сформулировал план решения и объяснил первый шаг."),
        ("2026-10-09", topics[1], 2, "Проверил ответ по рисунку или обратному действию."),
        ("2026-10-10", topics[2], 4, "После подсказки сам выбрал следующий шаг."),
        ("2026-10-11", topics[3], 3, "Нашёл и исправил одну ошибку при проверке."),
        ("2026-10-12", topics[4], 2, "Сравнил начальную попытку с новым способом."),
    ]
    ledger = [
        {"date": date, "points": points, "reason": reason, "role": "tutor", "topic_id": topic}
        for date, topic, points, reason in ledger_events
    ]
    lesson_notes = [
        f"Разобрали задание по теме {topics[0]}: ученик выбрал способ решения и объяснил, почему он подходит.",
        f"По теме {topics[1]} сравнили ответ с наглядной моделью; при проверке нашли место возможной ошибки.",
        f"В теме {topics[2]} ученик получил одну подсказку и затем самостоятельно продолжил решение.",
        f"По теме {topics[3]} выполнили короткую проверку: сначала прогноз, затем запуск или вычисление.",
        f"В теме {topics[4]} ученик назвал, какая опора помогла и какую часть уже может выполнить сам.",
    ]
    lesson_dates = ["2026-10-08", "2026-10-09", "2026-10-10", "2026-10-11", "2026-10-12"]
    lessons = [
        {"date": date, "topic_ids": [topics[index], topics[(index + 1) % len(topics)]], "minutes": 18 + index * 2, "note": note}
        for index, (date, note) in enumerate(zip(lesson_dates, lesson_notes))
    ]
    journal = [
        {"date": date, "note": note}
        for date, note in zip(lesson_dates, [
            f"Начали с короткого задания по теме {topics[0]}, чтобы увидеть выбранный способ.",
        f"В теме {topics[1]} помогла опора: {strategies[0]}.",
            f"Для темы {topics[2]} сначала использовали вопрос-подсказку, потом ученик продолжил сам.",
            f"В теме {topics[3]} проверили предположение на новом примере и обсудили исправление.",
            f"После задания по теме {topics[4]} ученик назвал следующий шаг для самостоятельной тренировки.",
        ])
    ]
    completed_project = {
        "id": f"{config['id']}-completed-project",
        "title": f"Завершённый мини-проект: {interests[0]}",
        "description": f"Ученик собрал небольшой результат по теме {topics[0]} и показал, как в нём используется навык из темы {topics[-1]}.",
        "status": "completed",
        "skills": [topics[0], topics[-1]],
        "reflection": "Ученик объяснил один удачный выбор и назвал, что хочет улучшить в следующей версии.",
        "completed_at": "2026-10-06",
    }

    return {
        "schema_version": 2,
        "learner_id": config["id"],
        "pseudonym": config["pseudonym"],
        "age_group": config["age"],
        "grade": config["grade"],
        "language": "russian",
        "rag_context": {
            "current_goal": {"topic_id": goal_topic, "goal": goal["goal"], "updated_at": "2026-10-12"},
            "current_topics": topics[:3],
            "priority_difficulties": [
                {
                    "topic_id": item[0],
                    "description": (
                        f"По наблюдениям: освоение {item[1]:.0%}, самостоятельность {item[2]:.0%}. "
                        f"Начать с опоры: {strategies[index % len(strategies)]}; после точной попытки проверить, можно ли её убрать."
                    ),
                }
                for index, item in enumerate(difficulties[:3])
            ],
            "effective_strategies": strategies[:3],
            "avoid": ["Не убирать опору, пока самостоятельное решение не подтверждено наблюдениями.", "Не считать непроверенную тему неуспехом: сначала провести диагностику."],
            "recent_progress": {"date": "2026-10-12", "note": reading},
        },
        "profile": profile,
        "interests": interests,
        "knowledge": knowledge,
        "error_patterns": errors,
        "mvp": {
            "story_preferences": interests[:2],
            "learning_preferences_observed": [profile["explanation_style"], reading],
            "help_strategies": strategies,
            "journal": journal,
            "lessons": lessons,
            "points_ledger": ledger,
        },
        "learner_model": {
            "competencies": competencies,
            "zpd": {"current": current, "outside": outside},
            "learning_preferences": {"visual": .88 if profile["prefers_visual"] else .35, "hands_on": .72, "storyline": .64, "audio": .48, "collaboration": .57, "competition": .42},
            "engagement": {"record_type": "synthetic_demo", "strongest_conditions": interests[:2], "reading_note": reading, "preferred_session_minutes": 20 if config["grade"] == "3" else 25},
            "ai_usage": ai_usage,
            "projects": [active_project, completed_project],
            "strengths": [f"Связывает задания с интересом к теме «{interests[0]}».", "Может назвать способ, который помог решить задачу."],
            "support_needs": [reading, "Давать по одной подсказке и отмечать, какой шаг ученик затем выполнил самостоятельно."],
            "scaffolding_by_topic": scaffolding,
            "gamification": {"xp": sum(item[2] for item in ledger_events), "level": 1, "badges": ["ДЕМО – ученик", "Замечает стратегию"], "missions_completed": 5, "next_mission": goal["goal"].split(":", 1)[0]},
        },
    }


def cards_from_fixtures(directory: Path = FIXTURES) -> list[dict[str, Any]]:
    cards = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("demo-*.json"))]
    expected = {item["id"] for item in ROSTER}
    if {card.get("learner_id") for card in cards} != expected or len(cards) != len(expected):
        raise ValueError("demo fixture directory must contain exactly the six expected demo learner cards")
    return cards


def write_fixtures(directory: Path = FIXTURES) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for config in ROSTER:
        path = directory / f"{config['id']}.json"
        path.write_text(json.dumps(build_card(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.append(path)
    return outputs


def _validator():
    tools = LEARNER_DATA / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from validate import validate_card
    import zpd
    return validate_card, zpd


def _validate_card(card, catalog, validate_card) -> None:
    learner_id = card["learner_id"]
    findings = validate_card(card, catalog, f"{learner_id}.json")
    if findings:
        details = "; ".join(f"{item.json_path}: {item.message}" for item in findings[:8])
        raise ValueError(f"demo card has validation findings {learner_id}: {details}")


def validate_demo_cards(cards: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    validate_card, zpd = _validator()
    seen = set()
    for card in cards:
        learner_id = card.get("learner_id")
        if learner_id in seen:
            raise ValueError(f"duplicate demo learner id: {learner_id}")
        seen.add(learner_id)
        if not str(card.get("pseudonym", "")).startswith("\u0414\u0415\u041c\u041e \u2013 "):
            raise ValueError(f"demo card lacks a clear fictional label: {learner_id}")
        _validate_card(card, catalog, validate_card)
        if card["learner_model"]["zpd"] != zpd.derive_zpd(card):
            raise ValueError(f"stored ZPD projection differs from competency evidence: {learner_id}")
        expected = CONFIRMED_EVIDENCE.get(learner_id)
        if expected:
            competency = next(
                item for item in card["learner_model"]["competencies"]
                if item["topic_id"] == expected["topic"]
            )
            actual = zpd.confirmed_zone(competency, DEMO_AS_OF)
            if actual["zone"] != expected["expected"]:
                raise ValueError(f"unexpected confirmed ZPD zone for {learner_id}: {actual['zone']}")
            if expected.get("evidence_status", "confirmed") != actual["evidence_status"]:
                raise ValueError(f"unexpected evidence status for {learner_id}: {actual['evidence_status']}")


def _live_demo_card(fixture: dict[str, Any], assessed_on: date) -> dict[str, Any]:
    _, zpd = _validator()
    learner_id = fixture["learner_id"]
    day = assessed_on.isoformat()
    card = fixture
    for zone, topic_id in LIVE_DEMO_TOPICS[learner_id].items():
        for pair in (1, 2):
            pair_id = f"synthetic-demo:{learner_id}:{topic_id}:{day}:{pair}"
            for support, solved in LIVE_DEMO_ATTEMPTS[zone]:
                card = zpd.record_observation(
                    card, topic_id=topic_id, support_level=support, solved=solved, observed_on=day,
                    task_id=f"{pair_id}:{support}", assessment_pair_id=pair_id, difficulty="medium",
                    notes="ДЕМО – синтетическое наблюдение для карты понимания.",
                )
    card["learner_model"]["engagement"][LIVE_DEMO_DATE_KEY] = day
    return card


def _prepare_demo_card(fixture: dict[str, Any], existing: dict[str, Any] | None, today: date) -> dict[str, Any]:
    # Keep the day of the live demo available for new lesson notes in the CLI.
    new_assessment = today - timedelta(days=1)
    if existing is None:
        return _live_demo_card(fixture, new_assessment)

    learner_id = fixture["learner_id"]
    conflict = f"learner card was edited: {learner_id}"
    if any(existing.get(key) != fixture.get(key) for key in ("pseudonym", "grade", "profile")):
        raise ValueError(conflict)
    topics = set(LIVE_DEMO_TOPICS[learner_id].values())
    fixture_topics = {item["topic_id"]: item for item in fixture["learner_model"]["competencies"]
                      if item["topic_id"] in topics}
    competencies = existing["learner_model"]["competencies"]
    existing_topics = [item for item in competencies if item["topic_id"] in topics]
    actual = {item["topic_id"]: item for item in existing_topics}
    if len(existing_topics) != len(topics):
        raise ValueError(conflict)
    marker = existing["learner_model"].get("engagement", {}).get(LIVE_DEMO_DATE_KEY)
    if marker is None:
        if actual != fixture_topics:
            raise ValueError(conflict)
        return _live_demo_card(existing, new_assessment)
    try:
        assessed_on = date.fromisoformat(marker)
    except (TypeError, ValueError) as exc:
        raise ValueError(conflict) from exc
    if assessed_on > today:
        raise ValueError(conflict)
    _, zpd = _validator()
    if (today - assessed_on).days <= zpd.STALE_AFTER_DAYS:
        return existing

    # Refresh only the original demo competencies plus their unchanged seed evidence.
    # Other lesson history and product observations stay in the live card.
    seeded = _live_demo_card(fixture, assessed_on)
    expected = {item["topic_id"]: item for item in seeded["learner_model"]["competencies"]
                if item["topic_id"] in topics}
    if actual != expected:
        raise ValueError(conflict)
    base = copy.deepcopy(existing)
    base["learner_model"]["competencies"] = [
        fixture_topics.get(item["topic_id"], item) for item in base["learner_model"]["competencies"]
    ]
    return _live_demo_card(base, new_assessment)


def _load_credentials(path: Path, emails: dict[str, str]) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("teacher", {}).get("email") != emails["teacher"]:
            raise ValueError("existing demo credential file has an unexpected teacher account")
        students = {item.get("learner_id"): item for item in data.get("students", [])}
        if set(students) != {item["id"] for item in ROSTER}:
            raise ValueError("existing demo credential file does not cover the expected six students")
        for learner in ROSTER:
            if students[learner["id"]].get("email") != emails[learner["id"]]:
                raise ValueError(f"existing demo credential email does not match {learner['id']}")
        return data
    return {
        "notice": "Synthetic demonstration accounts only. Keep this local file private.",
        "teacher": {"email": emails["teacher"], "password": secrets.token_urlsafe(24)},
        "students": [{"learner_id": item["id"], "email": emails[item["id"]], "password": secrets.token_urlsafe(24)} for item in ROSTER],
    }


def _write_credentials(path: Path, credentials: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
        json.dump(credentials, output, ensure_ascii=True, indent=2)
        output.write("\n")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def seed_demo(store: Store, fixture_dir: Path = FIXTURES, credentials_file: Path | None = None) -> dict[str, Any]:
    cards = cards_from_fixtures(fixture_dir)
    catalog = store.catalog()
    if not catalog:
        raise RuntimeError("curriculum catalog is empty; run scripts/manage_local.py import first")
    validate_demo_cards(cards, catalog)
    credentials_file = credentials_file or ROOT / ".local" / "rich-demo-credentials.json"
    teacher_email = "demo.rich.teacher@local.test"
    emails = {"teacher": teacher_email, **{item["id"]: item["email"] for item in ROSTER}}

    existing_teacher = store.get_account(email=teacher_email)
    existing_students = {item["id"]: store.get_account(email=item["email"]) for item in ROSTER}
    # Read revisions before the cards so a concurrent update cannot be overwritten.
    revisions = {card["learner_id"]: store.get_revision(card["learner_id"]) for card in cards}
    existing_cards = {card["learner_id"]: store.get_card(card["learner_id"]) for card in cards}
    if not credentials_file.exists() and (existing_teacher or any(existing_students.values())):
        return {"created": [], "existing": [], "conflicts": ["demo account exists but the private credential file is missing"]}

    credentials = _load_credentials(credentials_file, emails)
    teacher_secret = credentials["teacher"]["password"]
    student_secrets = {item["learner_id"]: item["password"] for item in credentials["students"]}
    conflicts = []
    today = date.today()
    validate_card, _ = _validator()
    prepared_cards = []
    for fixture in cards:
        learner_id = fixture["learner_id"]
        existing = existing_cards[learner_id]
        if (revisions[learner_id] is None) != (existing is None):
            conflicts.append(f"learner card changed while preparing the demo seed: {learner_id}")
            continue
        try:
            card = _prepare_demo_card(fixture, existing, today)
        except ValueError as exc:
            conflicts.append(str(exc))
            continue
        if card != existing:
            _validate_card(card, catalog, validate_card)
        prepared_cards.append(card)

    if existing_teacher:
        if existing_teacher["role"] != "teacher" or not existing_teacher["active"] or not verify_password(teacher_secret, existing_teacher["password_hash"]):
            conflicts.append("demo teacher account has a different role, is inactive, or does not match the credential file")
    for learner in ROSTER:
        account = existing_students[learner["id"]]
        if account:
            if account["role"] != "student" or not account["active"] or not verify_password(student_secrets[learner["id"]], account["password_hash"]):
                conflicts.append(f"student account conflicts with demo credential file: {learner['id']}")
            else:
                with store._connect() as conn, conn.cursor() as cur:
                    cur.execute("SELECT learner_id FROM teachcopilot.account_learners WHERE account_id=%s", (account["id"],))
                    assigned = {row["learner_id"] for row in cur.fetchall()}
                if assigned and assigned != {learner["id"]}:
                    conflicts.append(f"student account is assigned outside its demo learner: {learner['id']}")
    if existing_teacher:
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT learner_id FROM teachcopilot.account_learners WHERE account_id=%s", (existing_teacher["id"],))
            assigned = {row["learner_id"] for row in cur.fetchall()}
        allowed = {card["learner_id"] for card in cards}
        if not assigned.issubset(allowed):
            conflicts.append("demo teacher account is assigned to learners outside the synthetic roster")
    if conflicts:
        return {"created": [], "existing": [], "conflicts": conflicts}

    if not credentials_file.exists():
        _write_credentials(credentials_file, credentials)
    teacher = existing_teacher or store.create_account(teacher_email, "DEMO \u2013 Fictional Teacher", hash_password(teacher_secret), "teacher", {"synthetic_demo": True})
    refreshed = [
        card for card in prepared_cards
        if existing_cards[card["learner_id"]] is not None and card != existing_cards[card["learner_id"]]
    ]
    if refreshed:
        store.save_cards(refreshed, teacher["id"], expected_revisions=revisions)
    created_cards = []
    existing_learners = []
    for card in prepared_cards:
        learner_id = card["learner_id"]
        if existing_cards[learner_id] is None:
            store.create_assigned_learner(card, teacher["id"])
            created_cards.append(learner_id)
        else:
            store.assign_learner(teacher["id"], learner_id, "teacher")
            existing_learners.append(learner_id)

    created_students = []
    for learner in ROSTER:
        existing = existing_students[learner["id"]]
        if not existing:
            store.create_student_account(learner["email"], learner["pseudonym"], hash_password(student_secrets[learner["id"]]), learner["id"], teacher["id"])
            created_students.append(learner["id"])
        else:
            store.assign_learner(existing["id"], learner["id"], "student")
    return {
        "teacher_email": teacher_email,
        "teacher_created": existing_teacher is None,
        "cards_created": created_cards,
        "cards_already_present": existing_learners,
        "cards_evidence_refreshed": [card["learner_id"] for card in refreshed],
        "student_accounts_created": created_students,
        "student_accounts_already_present": [item["id"] for item in ROSTER if existing_students[item["id"]]],
        "credentials_file": str(credentials_file),
        "passwords_printed": False,
        "conflicts": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed six schema-validated synthetic learners and isolated demo accounts.")
    parser.add_argument("--dsn", default=Settings.from_env().database_url, help="local PostgreSQL DSN; defaults to DATABASE_URL from .local.env")
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES)
    parser.add_argument("--credentials-file", type=Path, default=ROOT / ".local" / "rich-demo-credentials.json")
    parser.add_argument("--write-fixtures", action="store_true", help="regenerate only the six static JSON fixture cards")
    args = parser.parse_args(argv)
    if args.write_fixtures:
        paths = write_fixtures(args.fixtures_dir)
        print(json.dumps({"fixtures_written": [path.name for path in paths]}, ensure_ascii=False))
        return 0
    try:
        store = Store(args.dsn)
        result = seed_demo(store, args.fixtures_dir, args.credentials_file)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("conflicts") else 0
    except Exception as exc:
        print(f"seed_rich_demo: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
