"""Детерминированные проверки готового задания: арифметика и персонализация.

Генератор проверяет форму карточки — тему, число ступеней подсказок, отсутствие
ответа в подсказке. Но самое опасное, что может сделать модель, — выдать задачу
с неверным ответом: карточка выглядит безупречно и уходит ребёнку.

Здесь два вида проверок, обе без модели и без сети:

* :func:`check_arithmetic` считает равенства в тексте и ловит «8 · 5 = 45»;
* :func:`check_personalization` смотрит, опирается ли сюжет на интересы ребёнка
  — иначе персонализация формально сработала, а фактически нет.

Проверки намеренно осторожные: лучше пропустить сомнительный случай, чем
завалить репетитора ложными тревогами. Всё, что нашлось, попадает в warnings
карточки, а не блокирует её.
"""
from __future__ import annotations

import ast
import re

# Символы, из которых может состоять арифметическое выражение в русском тексте.
# Двоеточие — деление, средняя точка и звёздочка — умножение, минусы бывают трёх
# видов (дефис, минус U+2212, тире).
# Запятая намеренно НЕ входит: она обрывает выражение. Иначе «20 : 4 = 6, потом
# 6 · 3 = 18» разбирается как одно неразбираемое целое и ошибка теряется.
_EXPRESSION_CHARS = r"0-9\s+\-−–—·⋅*:/=()."
_EXPRESSION_RUN = re.compile(f"[{_EXPRESSION_CHARS}]*=[{_EXPRESSION_CHARS}]*")
_HAS_OPERATOR = re.compile(r"[+\-−–—·⋅*:/]")
_HAS_DIGIT = re.compile(r"\d")

# Если в тексте есть деление с остатком, равенство «85 : 4 = 21» верно только с
# оговоркой про остаток. Такие фрагменты не проверяем.
_REMAINDER_MARKERS = ("остат", "неполное частное")

# Задание «найди ошибку» содержит заведомо неверное равенство — это его суть.
# Проверять такой текст нельзя, иначе каждая подобная задача будет ложной тревогой.
_DELIBERATE_MISTAKE_MARKERS = (
    "найди ошибку", "найти ошибку", "найди и исправь", "в чём ошибка",
    "прав ли", "почему это неверно", "не подходит", "верно ли",
)

# Двоеточие в русском тексте бывает знаком препинания («осталось 4: 4 − 2 = 2»)
# и знаком деления («30 : 6»). Отличаем по пробелу слева: у деления он есть.
_PUNCTUATION_COLON = re.compile(r"(?<=\S):(?=\s)")

# Выражение, у которого срезали левый операнд («x + 7 = 15» → «+ 7 = 15»),
# посчитается как унарный плюс и даст ложную тревогу.
_LEADING_OPERATOR = re.compile(r"^[+\-−–—·⋅*:/]")

# Маркер шага разбора в начале фрагмента: «2) 3 · 10».
_STEP_MARKER = re.compile(r"^\s*\d+\)\s*")

# Узлы, разрешённые в выражении. Никакого eval: разбираем ast и считаем сами.
_ALLOWED_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_TOLERANCE = 1e-9


class _Unevaluatable(ValueError):
    """Фрагмент не является арифметическим выражением — просто пропускаем."""


def _to_python(expression: str) -> str:
    # Точка от конца предыдущего предложения и нумерация шагов в разборе
    # («1) 5 · 12 = 60 кг. 2) 3 · 10 = 30 кг.») иначе оставляют висящие символы,
    # выражение не разбирается — и ошибка в нём теряется молча.
    expression = _STEP_MARKER.sub("", expression.strip(" .\t\n"))
    replacements = {
        "·": "*", "⋅": "*", ":": "/",
        "−": "-", "–": "-", "—": "-",
    }
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    # «1 000» и «1,5» встречаются в текстах; запятая как разделитель разрядов
    # или дробей одинаково мешает разбору, поэтому такие фрагменты пропускаем.
    if "," in expression:
        raise _Unevaluatable("запятая")
    return expression.strip()


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _Unevaluatable("не число")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_OPERATORS):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise _Unevaluatable("деление на ноль")
        return left / right
    raise _Unevaluatable("недопустимая конструкция")


def evaluate_expression(expression: str) -> float:
    """Посчитать простое арифметическое выражение. Бросает при чём угодно ином."""
    text = _to_python(expression)
    if not text or not _HAS_DIGIT.search(text):
        raise _Unevaluatable("нет чисел")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise _Unevaluatable("не разбирается") from exc
    return _evaluate(tree)


def check_arithmetic(text: str) -> list[str]:
    """Найти равенства, которые не сходятся.

    Проверяются только цепочки с действием: «12 + 3 = 15» проверяем, а
    «1 кг = 1000 г» нет — это перевод единиц, там числа и не должны совпадать.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    lowered = text.lower()
    if any(marker in lowered for marker in _REMAINDER_MARKERS + _DELIBERATE_MISTAKE_MARKERS):
        return []

    problems: list[str] = []
    for match in _EXPRESSION_RUN.finditer(_PUNCTUATION_COLON.sub("\n", text)):
        run = match.group(0)
        if "=" not in run:
            continue
        if _LEADING_OPERATOR.match(run.strip()):
            continue  # левый операнд остался за пределами выражения

        parts = [part.strip() for part in run.split("=")]
        values: list[float] = []
        has_operator = False
        usable = True
        for part in parts:
            if not part:
                continue
            if _HAS_OPERATOR.search(part):
                has_operator = True
            try:
                values.append(evaluate_expression(part))
            except _Unevaluatable:
                usable = False
                break

        if not usable or not has_operator or len(values) < 2:
            continue
        if any(abs(value - values[0]) > _TOLERANCE for value in values[1:]):
            computed = " = ".join(
                f"{value:g}" for value in values
            )
            problems.append(f"«{run.strip()}» не сходится ({computed})")
    return problems


def _stems(text: str, length: int = 5) -> set[str]:
    normalised = text.lower().replace("ё", "е")
    return {token[:length] for token in re.findall(r"[a-zа-я0-9]+", normalised)}


# Доля общих основ, при которой задание считается переписанным из материала.
# Материал — источник математики, а не готовые формулировки: если задание
# совпадает с ним дословно, персонализации не произошло.
COPY_SIMILARITY = 0.8


def check_not_copied(statement: str, material_texts: list[str]) -> str | None:
    """Вернуть текст материала, из которого задание списано почти дословно."""
    statement_stems = _stems(statement)
    if len(statement_stems) < 4:
        return None
    for text in material_texts:
        if not isinstance(text, str) or not text.strip():
            continue
        material_stems = _stems(text)
        if not material_stems:
            continue
        overlap = len(statement_stems & material_stems) / len(statement_stems)
        if overlap >= COPY_SIMILARITY:
            return text
    return None


def check_personalization(texts: list[str], interests: tuple[str, ...],
                          story_preferences: tuple[str, ...]) -> bool:
    """True, если сюжет действительно опирается на интересы или любимые сюжеты.

    Сравниваем по основам слов: «космос» засчитывается и как «космическая», и
    как «космонавт». Если у ребёнка интересы не заполнены, проверять нечего.
    """
    wanted: set[str] = set()
    for phrase in tuple(interests) + tuple(story_preferences):
        wanted |= _stems(phrase)
    if not wanted:
        return True

    present: set[str] = set()
    for text in texts:
        if isinstance(text, str):
            present |= _stems(text)
    return bool(wanted & present)
