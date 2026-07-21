"""Shared helpers for learner-data stdlib tools (validate.py, build_context.py).

Only Python stdlib is used anywhere in this package (json, re, sys, argparse,
pathlib, datetime, difflib, unittest, subprocess) — no pip installs, Python >= 3.10.
All paths are derived from ``Path(__file__)`` so the tools work from any cwd.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# --- Regexes / constants (single source of truth for validate.py and build_context.py) ---

LEARNER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TOPIC_ID_RE = re.compile(r"^math\.g[34]\.[a-z_]+(\.[a-z0-9_]+){0,2}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

KNOWLEDGE_STATUSES = ("not_started", "learning", "confident", "needs_support")


def configure_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8 with replacement on encode errors.

    Needed so Cyrillic output does not crash on a non-UTF-8 console
    (e.g. legacy Windows code pages). Call this at the top of any CLI script,
    before any output is produced.
    """
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def find_package_root() -> Path:
    """Return the path to the ``learner-data/`` package root.

    Derived from this file's location (``learner-data/tools/learner_common.py``),
    not from cwd, so tools work no matter where they are invoked from.
    """
    return Path(__file__).resolve().parent.parent


def load_json(path: Path) -> tuple[object, bool]:
    """Read and parse a JSON file.

    Uses ``encoding="utf-8-sig"`` so a UTF-8 BOM (common after editing in
    Notepad-like tools) is stripped transparently rather than becoming an
    illegal leading character.

    Returns a ``(data, had_bom)`` tuple. ``had_bom`` is True if the raw bytes
    started with the UTF-8 BOM sequence. Raises ``json.JSONDecodeError`` if the
    content is not valid JSON, and ``OSError`` subclasses (e.g.
    ``FileNotFoundError``) if the file cannot be read.
    """
    raw_bytes = path.read_bytes()
    had_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    text = raw_bytes.decode("utf-8-sig")
    data = json.loads(text)
    return data, had_bom


def load_catalog(path: Path) -> dict:
    """Load the skills catalog and index its topics by id.

    Returns a dict mapping topic id -> topic entry (the raw dict from the
    catalog's ``topics`` list). Does not validate structure — that is
    validate.py's job; this is just the load-and-index convenience used by
    both validate.py and build_context.py.
    """
    data, _had_bom = load_json(path)
    topics = data.get("topics", []) if isinstance(data, dict) else []
    catalog: dict = {}
    for entry in topics:
        if isinstance(entry, dict) and "id" in entry:
            catalog[entry["id"]] = entry
    return catalog


def iter_learner_files(learners_dir: Path) -> list[Path]:
    """Return a sorted list of learner card files in ``learners_dir``.

    Excludes any file whose name starts with ``_`` (e.g. ``_TEMPLATE.json``),
    which is a template, not a real learner card.
    """
    return sorted(
        p for p in learners_dir.glob("*.json") if not p.name.startswith("_")
    )


def load_learner(learners_dir: Path, learner_id: str) -> tuple[object, bool]:
    """Load exactly one learner card by id.

    Opens exactly one non-symlink file directly inside ``learners_dir``. The
    id is validated before a path is built, so ``..`` and path separators can
    never escape the learner directory. Callers must not weaken this to a
    search/glob.

    Returns the same ``(data, had_bom)`` tuple as ``load_json``. Raises
    ``FileNotFoundError`` if the file does not exist, and
    ``json.JSONDecodeError`` if it is not valid JSON.
    """
    if not isinstance(learner_id, str) or not LEARNER_ID_RE.fullmatch(learner_id):
        raise ValueError(f'некорректный learner_id: "{learner_id}"')

    base_dir = learners_dir.resolve()
    path = base_dir / f"{learner_id}.json"
    if path.is_symlink() or path.resolve().parent != base_dir:
        raise ValueError(f'некорректный путь к карточке ученика: "{path}"')
    return load_json(path)
