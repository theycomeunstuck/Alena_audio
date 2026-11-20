"""Check literal recall against the conversation excerpts already delivered to the model."""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Mapping, Sequence

from local_app.conversation_retrieval import _stem_list


_MARKERS = ("дословн", "процитир", "цитат", "слово в слово", "теми же словами", "точными словами")
_ASSISTANT = re.compile(
    r"\bты\s+(мне\s+)?(сказал|написал|ответил|объяснял|спрашивал|показывал)"
)
_STOP_WORDS = ("вспомни процитируй дословно цитата цитату фраза фразу первый первую урок уроке "
               "слово словами точно точными получилось сказала сказал написала написал")
_STOP_STEMS = frozenset(_stem_list(_STOP_WORDS))
_UNAVAILABLE = ("В найденных репликах нет фразы, которую можно привести дословно. "
                "Подскажи, когда это было или о чём шла речь?")


def _normalized_question(text: str) -> str:
    return text.casefold().replace("ё", "е")


def is_verbatim_request(question: str) -> bool:
    question = _normalized_question(question)
    return any(marker in question for marker in _MARKERS)


def is_assistant_quote_request(question: str) -> bool:
    question = _normalized_question(question)
    return bool(_ASSISTANT.search(question) or "ассистент" in question or "помощник" in question)


def quote_features(text: str) -> set[str]:
    return (set(_stem_list(text)) - _STOP_STEMS) | set(re.findall(r"\d+", text))


def eligible_sources(sources: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [source for source in sources if isinstance(source, Mapping)
            and source.get("role") == "user" and source.get("usage") == "self_report"
            and isinstance(source.get("text"), str) and source["text"].strip()]


def select_quote_sources(question: str, sources: Sequence[Mapping[str, Any]]) -> tuple[str, list[Mapping[str, Any]]]:
    if not sources:
        return "unavailable", []
    features = quote_features(question)
    if not features:
        return ("unique" if len(sources) == 1 else "ambiguous"), list(sources)
    scores = [len(features & quote_features(source["text"])) for source in sources]
    highest = max(scores)
    if highest == 0:
        return "unavailable", []
    leaders = [source for source, score in zip(sources, scores) if score == highest]
    return ("unique" if len(leaders) == 1 else "ambiguous"), leaders


def contains_full_quote(answer: str, quote: str) -> bool:
    return " ".join(quote.split()) in " ".join(answer.split())


def truncate_quote(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    prefix = text[:limit - 1]
    last_boundary = max((prefix.rfind(mark) for mark in ".!?…"), default=-1)
    if last_boundary >= 0:
        prefix = prefix[:last_boundary + 1]
    return prefix + "…"


def _activity_day(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return date.fromisoformat(value).isoformat()
        if len(value) > 10 and value[10] in "T ":
            parsed = datetime.fromisoformat(value)
            return parsed.date().isoformat() if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value[:10]) else None
    except ValueError:
        pass
    return None


def format_inserted(source: Mapping[str, Any]) -> str:
    day = _activity_day(source.get("activity_day"))
    date_text = f" (запись от {day})" if day else ""
    return f"Вот твоя фраза дословно{date_text}: «{truncate_quote(source['text'], 700)}»"


def format_ambiguous(sources: Sequence[Mapping[str, Any]]) -> str:
    lines = ["Нашлось несколько похожих фраз:"]
    for index, source in enumerate(sources[:2], 1):
        day = _activity_day(source.get("activity_day"))
        date_text = f"({day}) " if day else ""
        lines.append(f"{index}. {date_text}«{truncate_quote(source['text'], 300)}»")
    return "\n".join([*lines, "Какую из них ты имеешь в виду?"])


def recall_quote_guard(question: str, sources: Sequence[Mapping[str, Any]], answer: str) -> dict[str, Any] | None:
    if not is_verbatim_request(question):
        return None
    candidates = eligible_sources(sources)
    candidate_ids = [str(source["source_id"]) for source in candidates if source.get("source_id") is not None]
    check: dict[str, Any] = {"status": "not_applicable", "selected_source_id": None,
                             "candidate_source_ids": candidate_ids}
    if is_assistant_quote_request(question):
        return check
    selection, leaders = select_quote_sources(question, candidates)
    if selection == "unique":
        source = leaders[0]
        source_id = source.get("source_id")
        check["selected_source_id"] = str(source_id) if source_id is not None else None
        if source.get("superseded_by") or source.get("corrects_source_id"):
            check["status"] = "skipped_correction"
            return check
    if leaders and any(contains_full_quote(answer, source["text"]) for source in leaders):
        check["status"] = "model_verbatim"
    elif selection == "unique":
        check.update(status="inserted", answer=format_inserted(leaders[0]))
    elif selection == "ambiguous":
        check.update(status="ambiguous", answer=format_ambiguous(leaders))
    else:
        check.update(status="unavailable", answer=_UNAVAILABLE)
    return check
