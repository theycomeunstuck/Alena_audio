"""Session summary presentation helpers.

Full summary generation belongs to the memory backend's ``process_session``
(chunks and reduce). This module only validates the four-heading summary
contract and splits it for UI presentation, so a summary that omits the
required honesty headings is never shown as a finished artifact.
"""
from __future__ import annotations

from .prompts import SUMMARY_HEADINGS

_HEADING_ORDER_WARNING = "нарушен порядок разделов"


def validate_summary(text: str) -> list[str]:
    """Return problems; an empty list means the summary follows the contract."""
    summary = str(text or "")
    problems: list[str] = []
    if not summary.strip():
        return ["итоги пусты"]
    positions = []
    for heading in SUMMARY_HEADINGS:
        marker = "## " + heading
        position = summary.find(marker)
        if position < 0:
            problems.append(f"отсутствует раздел {marker}")
        else:
            positions.append((position, marker))
    if len(positions) == len(SUMMARY_HEADINGS) and positions != sorted(positions):
        problems.append(_HEADING_ORDER_WARNING)
    return problems


def summary_sections(text: str) -> dict[str, str]:
    """Split a four-heading summary for UI rendering (defensive)."""
    summary = str(text or "")
    sections: dict[str, str] = {}
    found = [(summary.find("## " + heading), heading) for heading in SUMMARY_HEADINGS]
    present = sorted((position, heading) for position, heading in found if position >= 0)
    for index, (position, heading) in enumerate(present):
        end = present[index + 1][0] if index + 1 < len(present) else len(summary)
        sections[heading] = summary[position: end].strip()
    return sections


__all__ = ["summary_sections", "validate_summary"]
