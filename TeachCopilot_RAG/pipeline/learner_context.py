"""Safe adapter from ``learner-data`` JSON cards to the RAG runtime.

The learner-data package deliberately keeps a compact ``rag_context`` separate
from the complete learner card.  This module is the only place in the RAG
service that reads those cards, so full cards cannot accidentally be added to
an LLM prompt.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from pipeline.config import LEARNER_DATA_DIR


@dataclass(frozen=True)
class LearnerContext:
    """The safe, prompt-ready projection and retrieval hints for one learner."""

    learner_id: str
    prompt_text: str
    topic_ids: tuple[str, ...]
    grade: str


class LearnerContextError(ValueError):
    """Raised when a requested learner context is unavailable or invalid."""


def _load_learner_data_tools():
    """Import the existing stdlib-only learner-data contract implementation.

    ``learner-data`` contains a hyphen and therefore cannot be imported as a
    normal Python package.  Adding its ``tools`` directory to ``sys.path`` lets
    us reuse its ID validation, schema validation and compact renderer instead
    of creating a second, drifting implementation in the RAG service.
    """
    tools_dir = LEARNER_DATA_DIR / "tools"
    if not tools_dir.is_dir():
        raise LearnerContextError(
            "learner-data is unavailable: set TEACHCOPILOT_LEARNER_DATA_DIR "
            f"to a directory containing tools/, learners/ and catalog/ (got {LEARNER_DATA_DIR})"
        )
    tools_dir_text = str(tools_dir)
    if tools_dir_text not in sys.path:
        sys.path.insert(0, tools_dir_text)

    try:
        from build_context import build_context_text  # type: ignore[import-not-found]
        from learner_common import LEARNER_ID_RE, load_catalog, load_learner  # type: ignore[import-not-found]
        from validate import validate_card  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LearnerContextError(f"learner-data tools could not be imported: {exc}") from exc
    return LEARNER_ID_RE, build_context_text, load_catalog, load_learner, validate_card


def load_learner_context(learner_id: str) -> LearnerContext:
    """Load exactly one validated card and return only its compact projection.

    The ID is checked before a file path is constructed by learner-data's
    loader, which prevents traversal.  Invalid cards fail closed: the chat is
    not sent with unvalidated personalisation data.
    """
    LEARNER_ID_RE, build_context_text, load_catalog, load_learner, validate_card = _load_learner_data_tools()

    if not isinstance(learner_id, str) or not LEARNER_ID_RE.fullmatch(learner_id):
        raise LearnerContextError("learner_id must use lowercase letters, digits and hyphens")

    try:
        catalog = load_catalog(LEARNER_DATA_DIR / "catalog" / "math_g3_g4.json")
        card, _had_bom = load_learner(LEARNER_DATA_DIR / "learners", learner_id)
    except FileNotFoundError as exc:
        raise LearnerContextError(f"learner '{learner_id}' was not found") from exc
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise LearnerContextError(f"could not load learner '{learner_id}': {exc}") from exc

    if not isinstance(card, dict):
        raise LearnerContextError(f"learner '{learner_id}' is not a JSON object")

    findings = validate_card(card, catalog, f"{learner_id}.json")
    errors = [finding.format() for finding in findings if finding.is_error()]
    if errors:
        raise LearnerContextError(
            f"learner '{learner_id}' did not pass validation: {'; '.join(errors)}"
        )

    rag_context = card["rag_context"]
    topic_ids = tuple(dict.fromkeys([
        rag_context["current_goal"]["topic_id"],
        *rag_context.get("current_topics", []),
        *(item["topic_id"] for item in rag_context.get("priority_difficulties", [])),
    ]))
    return LearnerContext(
        learner_id=learner_id,
        prompt_text=build_context_text(card, catalog),
        topic_ids=topic_ids,
        grade=str(card["grade"]),
    )
