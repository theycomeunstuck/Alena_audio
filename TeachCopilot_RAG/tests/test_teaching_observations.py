"""Untrusted code examples must remain data in the offline checker."""
import pytest

from scripts.check_teaching_observations_v2 import divisions, literal_print_stdout


def test_literal_print_observation_preserves_content_and_separators():
    assert literal_print_stdout("print('\"Луч\"', 'заря', sep='|')") == '"Луч"|заря\n'
    assert literal_print_stdout("print('a', end=''); print('b')") == 'ab\n'


@pytest.mark.parametrize("template", [
    "print(open({target}, 'w'))",
    "print('x', file=open({target}, 'w'))",
    "__import__('pathlib').Path({target}).write_text('changed')",
    "print(str(1))",
    "print(*['x'])",
    "print('x', **{{'sep': '|'}})",
    "print('x', sep=42)",
    "value = 1\nprint(value)",
    "print('x' * 1000000000)",
])
def test_untrusted_print_example_cannot_call_code_or_write_files(template, tmp_path):
    target = tmp_path / 'must-not-exist.txt'
    code = template.format(target=repr(str(target)))
    with pytest.raises(ValueError):
        literal_print_stdout(code)
    assert not target.exists()


def test_division_observations_distinguish_wrong_property_from_absent_match():
    constraint = {"quotient_digits": 3, "middle_digit": 0, "last_digit_nonzero": True,
                  "exclude_pairs": []}
    assert divisions('408 : 4', constraint)[0]['numeric_property_status'] == 'satisfied'
    assert divisions('4080 : 12', constraint)[0]['numeric_property_status'] == 'wrong'
    assert divisions('No symbolic division here.', constraint) == []
