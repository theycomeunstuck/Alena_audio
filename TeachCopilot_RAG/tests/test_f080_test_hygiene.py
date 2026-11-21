"""Audit F-080: the live-model guard and the reloaded-config leak."""
import pytest

from tests.live_guard import live_explicitly_selected


@pytest.mark.parametrize("expression,selected", [
    ("", False), (None, False), ("not slow", False), ("sanity", False),
    ("not live", False), ("postgres and not live", False),
    ("live", True), ("live and postgres", True), ("postgres or live", True),
])
def test_live_tests_run_only_when_live_is_explicitly_selected(expression, selected):
    assert live_explicitly_selected(expression) is selected


def test_live_modules_use_the_shared_guard():
    from pathlib import Path
    root = Path(__file__).resolve().parent
    for name in ("test_memory_live_ollama.py", "test_personal_graph_live_postgres.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "skip_unless_live_selected" in source, name
        assert 'getoption("markexpr") or "").strip()' not in source, name
