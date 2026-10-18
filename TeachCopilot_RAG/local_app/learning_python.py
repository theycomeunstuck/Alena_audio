"""Локальная проверка вывода узкого подмножества учебных примеров Python."""
from __future__ import annotations

import ast
import io
import re
import tokenize


PRINT_EVALUATOR_VERSION = "literal-print-v1"
MAX_SOURCE_CHARACTERS = 4000
MAX_PRINT_CALLS = 8
MAX_ARGUMENTS = 16
MAX_STDOUT_CHARACTERS = 8192
MAX_PLAIN_QUESTION_CHARACTERS = 800


_OUTPUT_WRAPPER_WORDS = frozenset("""
    что какой какие точно точный точного выведет напечатает напечатано будет
    будут получится получатся результат вывод вывода результатом код кода
    этот этого этих эта программы программа программой вызов вызова вызовов
    print python у для в на экране после с и а ли есть стоит стоят
    покажи покажите проверь проверьте определи определите скажи скажите
    поясни поясните объясни объясните отдельно важно вокруг разделитель
    разделителя разделители разделителем сохрани сохраните пробел пробелы
    пробелов пустая пустую пустые пустых строка строку строки строк строке
    строках строками новой новые новая новую первой последней последнего
    последний символ символы символа символов появится появятся перевод
    перевода переводы конце конца финальный начнется начинается закончится
    заканчивается данные данных снаружи внутри внешние внешних внутренние внутренних
    двойные двойных одинарные одинарных кавычки кавычек кавычками оформления
    одного двух трех четырех пяти шести семи восьми один два три четыре
    пять шесть семь восемь 1 2 3 4 5 6 7 8
""".split())
_SAFE_FORMATTING_QUALIFIERS = tuple(re.compile(pattern) for pattern in (
    r"\bбез (?:внешних )?кавычек(?: оформления строки программы)?\b",
    r"\bне потеряй пустую строку(?:\s*,?\s*если она появится)?\b",
    r"\bне добавляй внешние кавычки\b",
    r"\bне убирай пробелы\b",
))
_FENCE_LINE = re.compile(r" {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*\Z")


class UnsupportedPythonExample(ValueError):
    """Пример не принадлежит поддерживаемому подмножеству Python."""


def _require_unicode_scalars(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise UnsupportedPythonExample("Пример содержит неподдерживаемые суррогатные символы Unicode.")


def _literal(node: ast.AST):
    if not isinstance(node, ast.Constant) or type(node.value) not in (str, int, bool, type(None)):
        raise UnsupportedPythonExample("Поддерживаются только литералы строк, целых чисел, bool и None.")
    if isinstance(node.value, str):
        _require_unicode_scalars(node.value)
    return node.value


def evaluate_literal_print(code: str) -> dict:
    """Разобрать прямые print с литералами без исполнения исходного кода.

    Все вызовы проверяются и суммарная длина считается до сборки вывода.
    Имена, выражения, обращения к объектам и любые другие инструкции запрещены.
    """
    if not isinstance(code, str) or not code.strip():
        raise UnsupportedPythonExample("Нужен непустой исходный код.")
    if len(code) > MAX_SOURCE_CHARACTERS:
        raise UnsupportedPythonExample("Исходный код превышает 4000 символов.")
    _require_unicode_scalars(code)
    try:
        tree = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, RecursionError, UnicodeError) as exc:
        raise UnsupportedPythonExample("Не удалось разобрать пример Python.") from exc
    if not 1 <= len(tree.body) <= MAX_PRINT_CALLS:
        raise UnsupportedPythonExample("Поддерживаются от 1 до 8 прямых вызовов print.")

    calls = []
    rendered_calls = []
    output_length = 0
    for statement in tree.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print"):
            raise UnsupportedPythonExample("Поддерживаются только прямые вызовы print.")
        if len(call.args) > MAX_ARGUMENTS:
            raise UnsupportedPythonExample("В одном print допускается не более 16 аргументов.")
        args = [_literal(arg) for arg in call.args]
        options = {}
        for keyword in call.keywords:
            if keyword.arg not in {"sep", "end"} or keyword.arg in options:
                raise UnsupportedPythonExample("Разрешены только неповторяющиеся параметры sep и end.")
            value = _literal(keyword.value)
            if value is not None and not isinstance(value, str):
                raise UnsupportedPythonExample("Параметры sep и end должны быть строками или None.")
            options[keyword.arg] = value
        # Встроенный print трактует None как значение параметра по умолчанию.
        sep = options.get("sep")
        sep = " " if sep is None else sep
        end = options.get("end")
        end = "\n" if end is None else end
        parts = [str(value) for value in args]
        output_length += sum(map(len, parts)) + max(len(parts) - 1, 0) * len(sep) + len(end)
        if output_length > MAX_STDOUT_CHARACTERS:
            raise UnsupportedPythonExample("Вывод превышает 8192 символа.")
        calls.append({"args": args, "sep": sep, "end": end})
        rendered_calls.append((parts, sep, end))
    stdout = "".join(sep.join(parts) + end for parts, sep, end in rendered_calls)
    return {"version": PRINT_EVALUATOR_VERSION, "code": code, "stdout": stdout, "calls": calls}


def _question_wrapper(text: str) -> bool:
    """Разрешить замкнутый словарь вопросов о выводе и его форматировании.

    Это проверка окружения, а не интерпретация естественного языка. Отрицание
    самого вывода, подсказки, условия и неизвестные команды не поддерживаются.
    """
    phrase = " ".join(text.split()).lower().replace("ё", "е")
    if len(phrase) > 1000:
        return False
    for qualifier in _SAFE_FORMATTING_QUALIFIERS:
        phrase = qualifier.sub(" ", phrase)
    if re.search(r"\b(?:не|нет|без|если|подсказ\w*|подскаж\w*|намек\w*)\b", phrase):
        return False
    for command in re.finditer(r"\b(?:покажи|покажите|проверь|проверьте|определи|определите|скажи|скажите)\b", phrase):
        if not re.match(r"\s+(?:(?:точный|точно)\s+)?(?:вывод|результат|что)\b", phrase[command.end():]):
            return False
    for command in re.finditer(r"\b(?:поясни|поясните|объясни|объясните)\b", phrase):
        if not re.match(r"\s+(?:вывод|результат|перевод|внешние|внутренние|кавычки)\b", phrase[command.end():]):
            return False
    for command in re.finditer(r"\b(?:сохрани|сохраните)\b", phrase):
        clause = re.split(r"[.,:;!?]", phrase[command.end():], maxsplit=1)[0]
        if not re.match(r"\s+(?:пробелы|пустые|переводы)\b", clause):
            return False
        if re.search(r"\b(?:код|кода|программа|программы|print|python|вызов|вызова|вызовов)\b", clause):
            return False
    # Одно слово в типографских кавычках может обозначать фрагмент результата.
    # Скобки, код и команды из нескольких слов таким способом не скрываются.
    phrase = re.sub(r"«[а-яa-z0-9_]{1,40}»", " данные ", phrase)
    if re.search(r"[^а-яa-z0-9_\s.,:;!?：？！/]", phrase):
        return False
    words = re.findall(r"[а-яa-z0-9_]+", phrase)
    return len(words) <= 100 and all(word in _OUTPUT_WRAPPER_WORDS for word in words)


def _checked_fragment(text: str, start: int, end: int) -> str | None:
    if not _question_wrapper(text[:start]) or not _question_wrapper(text[end:]):
        return None
    return _accepted_code(text[start:end])


def _accepted_code(code: str) -> str | None:
    try:
        evaluate_literal_print(code)
    except UnsupportedPythonExample:
        return None
    return code


def extract_print_example(text: str) -> str | None:
    """Выделить один полный безопасный пример из текущей реплики.

    Произвольный окружающий текст, второй фрагмент и неподдерживаемый хвост
    приводят к отказу. Код не исправляется и не сокращается до первого print.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_SOURCE_CHARACTERS:
        return None

    # Блок должен занимать отдельные строки по правилам Markdown. Внутренние
    # более короткие последовательности fence остаются частью исходного кода.
    lines = text.splitlines(keepends=True)
    offset = 0
    for line_index, line in enumerate(lines):
        opening = _FENCE_LINE.fullmatch(line)
        if opening:
            delimiter, language = opening.groups()
            if language.strip().lower() not in {"", "python", "py"}:
                return None
            start = offset + len(line)
            end = start
            closing = re.compile(r" {0,3}" + re.escape(delimiter[0]) + "{" + str(len(delimiter)) + r",}[ \t]*[\r\n]*\Z")
            for following_line in lines[line_index + 1:]:
                if closing.fullmatch(following_line):
                    if not _question_wrapper(text[:offset]) or not _question_wrapper(text[end + len(following_line):]):
                        return None
                    return _accepted_code(text[start:end])
                end += len(following_line)
            return None
        offset += len(line)

    # Inline delimiter может быть длиннее одной обратной кавычки, если сама
    # строковая константа содержит обратные кавычки.
    opening = re.search(r"`+", text)
    if opening:
        delimiter = opening.group()
        closing = next((match for match in re.finditer(r"`+", text[opening.end():]) if match.group() == delimiter), None)
        if closing is None:
            return None
        start = opening.end()
        end = start + closing.start()
        suffix_start = start + closing.end()
        if "\n" in text[start:end] or "\r" in text[start:end] or "`" in text[suffix_start:]:
            return None
        if not _question_wrapper(text[:opening.start()]) or not _question_wrapper(text[suffix_start:]):
            return None
        return _accepted_code(text[start:end])

    # Для plain разрешён один print в коротком однострочном вопросе. Tokenize
    # отличает скобки внутри строк от настоящих скобок вызова.
    if len(text) > MAX_PLAIN_QUESTION_CHARACTERS or "\n" in text or "\r" in text:
        return None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeError):
        return None
    print_indices = [index for index, token in enumerate(tokens) if token.type == tokenize.NAME and token.string == "print"]
    if len(print_indices) != 1:
        return None
    index = print_indices[0]
    if index + 1 >= len(tokens) or tokens[index + 1].type != tokenize.OP or tokens[index + 1].string != "(":
        return None
    stack = []
    matching = {")": "(", "]": "[", "}": "{"}
    for token in tokens[index + 1:]:
        if token.type != tokenize.OP:
            continue
        if token.string in {"(", "[", "{"}:
            stack.append(token.string)
        elif token.string in matching:
            if not stack or stack.pop() != matching[token.string]:
                return None
            if not stack:
                return _checked_fragment(text, tokens[index].start[1], token.end[1])
    return None


def render_print_result(data: dict) -> str:
    """Показать вычисленный вывод как данные, сохранив пробелы и кавычки."""
    # Проверяем происхождение результата, чтобы renderer не доверял подменённому stdout.
    if not isinstance(data, dict):
        raise UnsupportedPythonExample("Нужен результат проверки print.")
    verified = evaluate_literal_print(data.get("code"))
    if data != verified:
        raise UnsupportedPythonExample("Результат не соответствует проверенному исходному коду.")
    stdout = verified["stdout"]
    longest_run = max((len(match.group()) for match in re.finditer(r"`+", stdout)), default=0)
    fence = "`" * max(3, longest_run + 1)
    block = f"{fence}text\n{stdout}"
    if not stdout.endswith("\n") and stdout:
        block += "\n"
    block += fence
    if not stdout:
        ending = "Вывод пустой: не напечатано ни одного символа."
    elif stdout.endswith("\n"):
        ending = "После последней строки есть перевод строки."
    else:
        ending = "После последнего символа перевода строки нет."
    return ("Проверенный вывод:\n\n" + block + "\n\n" + ending
            + " Кавычки, ограничивающие строку в коде, не печатаются; кавычки внутри строки печатаются.")
