"""Shared guard for live-model tests (audit F-080).

The previous guard skipped only when no -m expression was given, so any
unrelated marker filter ("-m 'not slow'", "-m sanity") let live tests make real
model calls. Live tests now run only when the marker expression selects
"live" explicitly and positively.
"""
import re

import pytest


def live_explicitly_selected(markexpr: str | None) -> bool:
    expression = str(markexpr or "")
    if not re.search(r"\blive\b", expression):
        return False
    return not re.search(r"\bnot\s+live\b", expression)


def skip_unless_live_selected(request) -> None:
    if not live_explicitly_selected(request.config.getoption("markexpr")):
        pytest.skip("live checks are deselected by default; run with -m live")
