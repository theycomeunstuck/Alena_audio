"""Keep the local app independent of the frozen Open WebUI Filter path."""

import ast
from collections.abc import Iterator
from pathlib import Path
from tokenize import open as open_python_source


ROOT = Path(__file__).resolve().parents[1]
LEGACY_MODULES = {
    "pipeline.filter_function",
    "pipeline.profile_agent",
    "pipeline.prompt_builder",
    "pipeline.event_logger",
    "pipeline.db",
}


def _legacy_imports(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == "pipeline":
                modules = [f"pipeline.{alias.name}" for alias in node.names]
            else:
                modules = [node.module]
        else:
            continue

        for module in modules:
            legacy_module = ".".join(module.split(".", 2)[:2])
            if legacy_module in LEGACY_MODULES:
                yield node.lineno, legacy_module


def test_local_app_does_not_import_legacy_filter_modules():
    """Scan the local application and its startup entry points."""
    sources = sorted((ROOT / "local_app").rglob("*.py")) + [
        ROOT / "server.py",
        ROOT / "scripts" / "memory_worker.py",
        ROOT / "scripts" / "manage_local.py",
    ]

    violations = []
    for path in sources:
        filename = path.relative_to(ROOT).as_posix()
        with open_python_source(path) as source:
            tree = ast.parse(source.read(), filename=filename)
        violations.extend(
            f"{filename}:{line}: {module}" for line, module in _legacy_imports(tree)
        )

    assert not violations, (
        "F-012 froze the Open WebUI MVP path. The product must not use stale "
        "child_knowledge scores. Forbidden imports:\n" + "\n".join(violations)
    )


def test_legacy_import_scanner_recognizes_all_three_forms():
    """Detect forbidden absolute imports and ignore unrelated and relative imports."""
    tree = ast.parse(
        "import logging, pipeline.filter_function as legacy_filter\n"
        "from pipeline.profile_agent import assess\n"
        "from pipeline import prompt_builder as builder, event_logger\n"
        "from pipeline.db import get_connection\n"
        "from .pipeline import db\n"
        "from .pipeline.db import get_connection\n"
        "from . import db\n"
        "from pipeline import rag\n"
        "import pipeline.db_tools\n"
    )

    assert list(_legacy_imports(tree)) == [
        (1, "pipeline.filter_function"),
        (2, "pipeline.profile_agent"),
        (3, "pipeline.prompt_builder"),
        (3, "pipeline.event_logger"),
        (4, "pipeline.db"),
    ]
