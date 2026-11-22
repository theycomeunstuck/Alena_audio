"""Exercise assistant chat formatting with Node.js and the production renderer."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_assistant_chat_markdown_renderer():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the UI renderer check")
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, "tests/test_chat_markdown_render.js"],
        cwd=project, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
