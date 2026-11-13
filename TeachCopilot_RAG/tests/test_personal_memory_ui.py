"""Exercise practical memory rendering and dispatch without a browser or model."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("script", ["test_personal_memory_ui_render.js", "test_personal_memory_ui_alena.js"])
def test_personal_memory_production_render_and_dispatch(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the UI renderer check")
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, f"tests/{script}"],
        cwd=project, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
