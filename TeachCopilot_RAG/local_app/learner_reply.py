"""Keep a brief agreement distinct from an answer to an open lesson question."""
from __future__ import annotations

import re
import unicodedata


_AGREEMENTS = frozenset({
    "ага", "угу", "да", "ну да", "ну вроде", "вроде", "вроде да", "кажется да",
    "понятно", "кажется понятно", "ясно", "ок", "окей", "хорошо",
})
_OPEN_QUESTION = re.compile(
    r"\b(?:сколько|чему|какой|какая|какое|какие|какую|каких|каким|где|почему|зачем|как|что)\b", re.I)
_READINESS_QUESTION = re.compile(
    r"^(?:(?:а|теперь|тогда)\s+)?(?:хочешь|готов[аы]?|продолжим|разбер[её]м|могу|можно|давай)\b", re.I)


def unanswered_question_followup(reply: str, assistant_text: str) -> str | None:
    """Return a fixed invitation only for a bare agreement to an open question.

    This is not an answer grader or a general intent classifier. The caller must
    first validate the curriculum scope and dialogue references. A number,
    explanation, new request, yes/no question or readiness agreement keeps its
    normal route. Assistant text is inspected, never repeated as verified fact.
    """
    if not isinstance(reply, str) or not isinstance(assistant_text, str) or len(reply) > 80:
        return None
    normalized = unicodedata.normalize("NFKC", reply).casefold()
    normalized = re.sub(r"[\s.,!?…]+", " ", normalized).strip()
    if normalized not in _AGREEMENTS or not assistant_text or len(assistant_text) > 1200:
        return None
    # Only the last question matters. An offer of help after an exercise is a
    # legitimate yes/no prompt, even when an earlier sentence asks for a number.
    parts = re.split(r"(?<=[!?])\s*|(?<=\.)\s+|[\r\n]+", assistant_text)
    question = next((part.strip() for part in reversed(parts) if part.strip().endswith("?")), "")
    question = re.sub(r"[*_`]", "", question)
    if not question or _READINESS_QUESTION.search(question) or not _OPEN_QUESTION.search(question):
        return None
    return "Напиши свой ответ на этот вопрос. Если нужна подсказка, разберём вместе."
