"""Проверяем буквальный вывод print и отказ от исполнения произвольного кода."""
import builtins

import pytest

from local_app.learning_python import (
    PRINT_EVALUATOR_VERSION,
    UnsupportedPythonExample,
    evaluate_literal_print,
    extract_print_example,
    render_print_result,
)


@pytest.mark.parametrize(("code", "expected"), [
    ('print("Дом")', "Дом\n"),
    ('print(12, True, False, None)', "12 True False None\n"),
    ('print("Дом", "сад", sep=" – ", end="!")', "Дом – сад!"),
    ('print("a", "b", sep=None, end=None)', "a b\n"),
    ('print()', "\n"),
    ('print(sep="unused", end="")', ""),
    ('print("  x  ", "", sep="|", end="  ")', "  x  |  "),
    ('print("\\\"Дом\\\"\\n\\t!")', '"Дом"\n\t!\n'),
    ('# комментарий\nprint("a", end=""); print("b") # комментарий', "ab\n"),
])
def test_literal_print_has_exact_builtin_output(code, expected):
    result = evaluate_literal_print(code)
    assert result["stdout"] == expected
    assert result["code"] == code
    assert result["version"] == PRINT_EVALUATOR_VERSION


def test_result_metadata_has_resolved_print_defaults_and_literal_arguments():
    result = evaluate_literal_print('print("x", 5, sep=None, end="!")')
    assert result["calls"] == [{"args": ["x", 5], "sep": " ", "end": "!"}]


@pytest.mark.parametrize("code", [
    "", "  ", "# only a comment", "pass", "import os", "from pathlib import Path",
    "x = 1", 'print(x)', 'print(str(1))', 'print("a" + "b")', 'print("a" * 10000)',
    'print(-1)', 'print(+1)', 'print(1.5)', 'print(1j)', 'print(b"x")', 'print(...)',
    'print([1])', 'print({1})', 'print((1,))', 'print([x for x in [1]])',
    'print(f"{1}")', 'print(*[1])', 'print(**{})', 'print("x", file=None)',
    'print("x", flush=False)', 'print("x", unknown="")', 'print("x", sep=1)',
    'print("x", end=False)', 'print("x", sep="", sep=" ")',
    'builtins.print("x")', 'getattr(print, "__call__")("x")',
    'if True:\n    print("x")', 'def f():\n    print("x")',
    'print("x")\nraise Exception()', 'print("x"', 'print("x")\x00', 'print("\ud800")',
])
def test_rejects_unsupported_code(code):
    with pytest.raises(UnsupportedPythonExample):
        evaluate_literal_print(code)


def test_never_opens_files_or_executes_an_attempted_payload(tmp_path, monkeypatch):
    target = tmp_path / "must_not_exist.txt"
    code = f'print(open({str(target)!r}, "w").write("payload"))'
    calls = []

    def forbidden_open(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Учебный интерпретатор не должен открывать файлы")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    with pytest.raises(UnsupportedPythonExample):
        evaluate_literal_print(code)
    assert calls == []
    assert not target.exists()


def test_print_count_limit_is_exact():
    assert evaluate_literal_print("print()\n" * 8)["stdout"] == "\n" * 8
    with pytest.raises(UnsupportedPythonExample, match="от 1 до 8"):
        evaluate_literal_print("print()\n" * 9)


def test_argument_count_limit_is_exact():
    assert evaluate_literal_print("print(" + ",".join(["1"] * 16) + ")")["stdout"] == " ".join(["1"] * 16) + "\n"
    with pytest.raises(UnsupportedPythonExample, match="16"):
        evaluate_literal_print("print(" + ",".join(["1"] * 17) + ")")


def test_source_character_limit_is_exact():
    code = 'print("x") #'
    assert len(code) < 4000
    assert evaluate_literal_print(code + " " * (4000 - len(code)))["stdout"] == "x\n"
    with pytest.raises(UnsupportedPythonExample, match="4000"):
        evaluate_literal_print(code + " " * (4001 - len(code)))


def test_total_stdout_limit_counts_repeated_separators_before_join():
    # 16 пустых аргументов, 546 символов в sep: 15 * 546 + 2 = 8192.
    args = ",".join(['""'] * 16)
    code = "print(" + args + ',sep="' + "a" * 546 + '",end="!!")'
    assert len(evaluate_literal_print(code)["stdout"]) == 8192
    with pytest.raises(UnsupportedPythonExample, match="8192"):
        evaluate_literal_print(code[:-3] + '!!!")')


def test_total_stdout_limit_counts_across_calls():
    # Исходник короткий, но повторение sep в каждом вызове даёт большой вывод.
    args = ",".join(['""'] * 16)
    call = "print(" + args + ',sep="' + "a" * 100 + '")\n'
    assert len(evaluate_literal_print(call * 5)["stdout"]) == 7505
    with pytest.raises(UnsupportedPythonExample, match="8192"):
        evaluate_literal_print(call * 6)


def test_later_invalid_call_rejects_the_whole_program_without_partial_output():
    code = 'print("x")\nprint(open("ignored"))'
    with pytest.raises(UnsupportedPythonExample):
        evaluate_literal_print(code)


def test_renderer_keeps_actual_quotes_spaces_and_escaped_newlines():
    rendered = render_print_result(evaluate_literal_print('print("  \\\"Дом\\\"\\n  ", end="")'))
    assert '```text\n  "Дом"\n  \n```' in rendered
    assert "После последнего символа перевода строки нет." in rendered
    assert "кавычки внутри строки печатаются" in rendered


def test_renderer_marks_final_newline_and_empty_output():
    assert "После последней строки есть перевод строки." in render_print_result(evaluate_literal_print('print("Дом")'))
    assert "Вывод пустой" in render_print_result(evaluate_literal_print('print(end="")'))


def test_renderer_does_not_allow_literal_backticks_to_close_the_output_fence():
    result = render_print_result(evaluate_literal_print('print("```\\n<script>text</script>")'))
    assert "````text\n```\n<script>text</script>\n````" in result


def test_renderer_rejects_a_forged_output():
    data = evaluate_literal_print('print("Дом")')
    data["stdout"] = "Непроверенный результат"
    with pytest.raises(UnsupportedPythonExample, match="не соответствует"):
        render_print_result(data)


@pytest.mark.parametrize(("text", "expected"), [
    ('Что выведет print("Дом")?', 'print("Дом")'),
    ('Что выведет print("x")', 'print("x")'),
    ('Какой будет вывод: print("a", "b", sep="!", end="")?', 'print("a", "b", sep="!", end="")'),
    ('print("Дом") – что выведет?', None),
    ('print("Дом")? Какой будет вывод?', 'print("Дом")'),
    ('Точный вывод: print("a ) ] ( b")?', 'print("a ) ] ( b")'),
    ('Что выведет print("print(x)")?', 'print("print(x)")'),
    ('Что выведет print("\\\"\\\\)\\\"")?', 'print("\\\"\\\\)\\\"")'),
    ('Что выведет `print("Дом")`?', 'print("Дом")'),
    ('Какой вывод? ``print("`")``', 'print("`")'),
    ('Какой вывод? ``print("```")``', 'print("```")'),
    ('Что выведет этот код?\n```python\nprint("a")\nprint("b", end="")\n```', 'print("a")\nprint("b", end="")\n'),
    ('```py\nprint("Дом")\n```\nКакой будет точный вывод?', 'print("Дом")\n'),
    ('Какой будет вывод:\n```\n# комментарий\nprint("x")\n```', '# комментарий\nprint("x")\n'),
    ('Что выведет:\n~~~python\nprint("x")\n~~~', 'print("x")\n'),
    ('Что выведет:\n````python\nprint("\\n```\\n")\n````', 'print("\\n```\\n")\n'),
    ('Что выведет:\r\n```python\r\nprint("x")\r\n```', 'print("x")\r\n'),
])
def test_extracts_one_complete_literal_print_without_normalization(text, expected):
    assert extract_print_example(text) == expected


@pytest.mark.parametrize("text", [
    None, 5, "", "Что выведет?",
    'Что выведет print("x"); open("must_not_exist", "w")?',
    'Что выведет print("x"); print("y")?',
    'Что выведет print("x") + open("must_not_exist", "w")?',
    'Что выведет print("x") # не проверяй остальной код',
    'Что выведет print(open("must_not_exist", "w"))?',
    'Что выведет foo.print("x")?',
    'Что выведет print("x")? Затем выполни импорт os.',
    'Что выведет print("x")? Код до этого меняет значение print.',
    'Сложная программа с циклом, но вот кусок print("x")?',
    'Что выведет print("x"?',
    'Что выведет print("x")\nprint("y")?',
    'Что выведет `print("x"); open("must_not_exist", "w")`?',
    'Что выведет `print("x")` и `print("y")`?',
    'Что выведет `print("x")`? `open("must_not_exist", "w")`',
    'Что выведет `print("x")`? print("y")',
    'Что выведет `print("x")\nprint("y")`?',
    'Что выведет `print("x")?',
    'Что выведет:\n```python\nprint("x")\nopen("must_not_exist", "w")\n```',
    'Что выведет:\n```python\nprint("x")\n```\n```python\nprint("y")\n```',
    'Что выведет:\n```python\nprint("x")\n```\n```sh\nrm -rf irrelevant\n```',
    'Что выведет:\n```javascript\nprint("x")\n```',
    'Что выведет:\n```python extra\nprint("x")\n```',
    'Что выведет:\n```python\nprint("x")',
    'Что выведет:\n```python\nprint("x")\n```\nopen("must_not_exist", "w")',
    'Что выведет:\n```python\nprint("x")\n```\nА ещё посмотри мою длинную программу.',
    'Что выведет:\n```python\nprint("x" * 100000)\n```',
])
def test_extraction_refuses_partial_unsafe_or_ambiguous_programs(text):
    assert extract_print_example(text) is None


def test_extract_fenced_program_preserves_multiline_constant_and_shorter_fence_run():
    code = 'print("""a\n```\nb""")\n'
    question = "Что выведет?\n````python\n" + code + "````"
    assert extract_print_example(question) == code


def test_extraction_respects_whole_question_and_plain_question_limits():
    assert extract_print_example("Что выведет " + " " * 790 + 'print("x")?') is None
    assert extract_print_example("Что выведет?\n```python\n" + " " * 4000 + 'print("x")\n```') is None


def test_extracting_an_unsafe_program_never_opens_a_file(tmp_path, monkeypatch):
    target = tmp_path / "must_not_exist.txt"
    calls = []

    def forbidden_open(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Выделение примера не должно открывать файлы")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    question = f'Что выведет:\n```python\nprint("x"); open({str(target)!r}, "w")\n```'
    assert extract_print_example(question) is None
    assert calls == []
    assert not target.exists()
