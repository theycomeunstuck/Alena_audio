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


_QUESTION_PHRASE = re.compile(
    r"(?:что (?:выведет|напечатает)(?: (?:этот код|код|эта программа|программа|этот вызов|вызов))?"
    r"|что будет напечатано"
    r"|какой (?:(?:будет|получится) )?(?:точный )?вывод(?: (?:у этого кода|этого кода|программы|у программы|кода))?"
    r"|(?:покажи|покажите|проверь|проверьте|определи|определите) (?:точный )?вывод(?: (?:этого кода|программы|кода))?"
    r"|точный вывод|в python)",
    re.IGNORECASE,
)
_FENCE_LINE = re.compile(r" {0,3}(`{3,}|~{3,})([^\r\n]*)[\r\n]*\Z")


class UnsupportedPythonExample(ValueError):
    """Пример не принадлежит поддерживаемому подмножеству Python."""


def _literal(node: ast.AST):
    if not isinstance(node, ast.Constant) or type(node.value) not in (str, int, bool, type(None)):
        raise UnsupportedPythonExample("Поддерживаются только литералы строк, целых чисел, bool и None.")
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
    """Разрешить только пустое окружение или короткую явную просьбу о выводе."""
    phrase = " ".join(text.split()).strip(" :：?？.!！")
    return not phrase or _QUESTION_PHRASE.fullmatch(phrase) is not None


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
