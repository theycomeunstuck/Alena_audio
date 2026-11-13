"""Contract for reviewed source memory with retrieval independent of Neo4j."""
from __future__ import annotations


PERSONAL_MEMORY_CONTEXT_PROTOCOL = "personal-memory-context-v1"
PERSONAL_MEMORY_SELECTION_VERSION = "source-hybrid-topic-anchor-correction-groups-v2"
PERSONAL_MEMORY_RETRIEVAL_MODES = frozenset({"chronological", "lexical", "vector", "hybrid"})


def personal_memory_budgets(context_tokens):
    """Conservative UTF-8 limits; the complete prompt still needs its own guard."""
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens < 8192:
        raise ValueError("Personal memory requires at least an 8k model context")
    if context_tokens >= 32768:
        return 20000, 24000
    if context_tokens >= 16384:
        return 10000, 12000
    return 3000, 4000


class PersonalMemoryStale(ValueError):
    """The reviewed sources or model recipe changed and require a new review."""

    def __init__(self):
        super().__init__("Персональная память изменилась. Преподавателю нужно обновить и заново утвердить память.")
