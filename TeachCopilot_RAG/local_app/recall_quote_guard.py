"""Check literal recall against the conversation excerpts already delivered to the model."""
from __future__ import annotations

from datetime import date, datetime
import re
from typing import Any, Literal, Mapping, Sequence

from local_app.conversation_retrieval import _stem_list


_MARKERS = ("дословн", "процитир", "цитат", "слово в слово", "теми же словами", "точными словами")
_VERBS = r"(?:сказал|написал|ответил|объяснял|спрашивал|показывал)\w*"
_ASSISTANT_PATTERNS = tuple(re.compile(pattern) for pattern in (
    rf"\bты\s+(?:мне\s+)?{_VERBS}",
    rf"\b(?:ассистент|помощник)\s+(?:мне\s+)?{_VERBS}",
    rf"{_VERBS}\s+(?:мне\s+)?(?:ассистент|помощник)\b",
    r"\b(?:слова|фраз[ауы]?|ответ[ы]?)\s+(?:ассистента|помощника)\b",
    r"\bтво(?:й|ю|и|им|его|ей)\s+(?:ответ|слов|фраз)",
))
_STOP_WORDS = ("вспомни процитируй дословно цитата цитату фраза фразу первый первую урок уроке "
               "слово словами точно точными получилось сказала сказал написала написал "
               "ответил объяснял спрашивал показывал ассистент помощник твой твою твои")
_STOP_STEMS = frozenset(_stem_list(_STOP_WORDS))
_UNAVAILABLE = ("В найденных репликах нет фразы, которую можно привести дословно. "
                "Подскажи, когда это было или о чём шла речь?")
_ASSISTANT_UNAVAILABLE = ("В найденных репликах нет моего ответа, который можно привести дословно. "
                          "Подскажи, когда это было или о чём шла речь?")
_INSERTED_HEADINGS = {"user": "Вот твоя фраза дословно", "assistant": "Вот мой ответ дословно"}
_AMBIGUOUS_HEADINGS = {"user": "Нашлось несколько похожих фраз:",
                       "assistant": "Нашлось несколько похожих моих ответов:"}
# Replies this guard produced itself are stored as assistant messages; they are not earlier answers to quote.
_GUARD_REPLY_PREFIXES = (*_INSERTED_HEADINGS.values(), *_AMBIGUOUS_HEADINGS.values(),
                         _UNAVAILABLE, _ASSISTANT_UNAVAILABLE)


def _normalized_question(text: str) -> str:
    return text.casefold().replace("ё", "е")


def is_verbatim_request(question: str) -> bool:
    question = _normalized_question(question)
    return any(marker in question for marker in _MARKERS)


def is_assistant_quote_request(question: str) -> bool:
    question = _normalized_question(question)
    return any(pattern.search(question) for pattern in _ASSISTANT_PATTERNS)


def quote_features(text: str) -> set[str]:
    # Known limitation (user decision 2026-11-22: keep as is for now): features
    # are 5-letter prefixes from _stem_list, so a short form and its inflections
    # do not meet ("маяк" -> "маяк", "маяком" -> "маяко"). 4-letter prefixes were
    # tried and changed two pinned selection outcomes in the guard tests.
    # TODO: if pilot data shows missed quotes, switch to a lemmatizer such as
    # pymorphy3 (new dependency; re-pin tests/test_recall_quote_guard.py and
    # tests/test_recall_verbatim_behavior.py deliberately).
    return (set(_stem_list(text)) - _STOP_STEMS) | set(re.findall(r"\d+", text))


def eligible_sources(sources: Sequence[Mapping[str, Any]],
                     author: Literal["user", "assistant"] = "user") -> list[Mapping[str, Any]]:
    usage = "self_report" if author == "user" else "context_only"
    return [source for source in sources if isinstance(source, Mapping)
            and source.get("role") == author and source.get("usage") == usage
            and isinstance(source.get("text"), str) and source["text"].strip()
            and not _is_quote_machinery(source["text"], author)]


def _is_quote_machinery(text: str, author: Literal["user", "assistant"]) -> bool:
    """A verbatim request or a guard reply is not quotable content."""
    if author == "user":
        return is_verbatim_request(text)
    return text.lstrip().startswith(_GUARD_REPLY_PREFIXES)


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


def best_sentences(question: str, text: str) -> list[str]:
    sentences = [" ".join(part.split()).rstrip(".!?…")
                 for part in re.split(r"(?<=[.!?…])\s+", text)]
    sentences = [sentence for sentence in sentences if len(re.findall(r"\w+", sentence)) >= 4]
    if not sentences:
        return []
    features = quote_features(question)
    if not features:
        return sentences
    scores = [len(features & quote_features(sentence)) for sentence in sentences]
    highest = max(scores)
    return [sentence for sentence, score in zip(sentences, scores) if score == highest]


def answer_quote_match(question: str, answer: str,
                       leaders: Sequence[Mapping[str, Any]]) -> Literal["full", "sentence"] | None:
    if any(contains_full_quote(answer, source["text"]) for source in leaders):
        return "full"
    normalized_answer = " ".join(answer.split())
    if any(re.search(r"(?<!\w)" + re.escape(sentence) + r"(?!\w)", normalized_answer)
           for source in leaders for sentence in best_sentences(question, source["text"])):
        return "sentence"
    return None


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


def format_inserted(source: Mapping[str, Any], author: Literal["user", "assistant"] = "user") -> str:
    day = _activity_day(source.get("activity_day"))
    date_text = f" (запись от {day})" if day else ""
    heading = _INSERTED_HEADINGS[author]
    return f"{heading}{date_text}: «{truncate_quote(source['text'], 700)}»"


def format_ambiguous(sources: Sequence[Mapping[str, Any]], author: Literal["user", "assistant"] = "user") -> str:
    lines = [_AMBIGUOUS_HEADINGS[author]]
    for index, source in enumerate(sources[:2], 1):
        day = _activity_day(source.get("activity_day"))
        date_text = f"({day}) " if day else ""
        lines.append(f"{index}. {date_text}«{truncate_quote(source['text'], 300)}»")
    question = "Какую из них ты имеешь в виду?" if author == "user" else "Какой из них ты имеешь в виду?"
    return "\n".join([*lines, question])


def recall_quote_guard(question: str, sources: Sequence[Mapping[str, Any]], answer: str) -> dict[str, Any] | None:
    if not is_verbatim_request(question):
        return None
    author = "assistant" if is_assistant_quote_request(question) else "user"
    candidates = eligible_sources(sources, author)
    candidate_ids = [str(source["source_id"]) for source in candidates if source.get("source_id") is not None]
    check: dict[str, Any] = {"status": "unavailable", "selected_source_id": None,
                             "candidate_source_ids": candidate_ids, "match": None, "author": author}
    selection, leaders = select_quote_sources(question, candidates)
    if selection == "unique":
        source = leaders[0]
        source_id = source.get("source_id")
        check["selected_source_id"] = str(source_id) if source_id is not None else None
        if source.get("superseded_by") or source.get("corrects_source_id"):
            check["status"] = "skipped_correction"
            return check
    match = answer_quote_match(question, answer, leaders)
    if match is not None:
        check["status"] = "model_verbatim"
        check["match"] = match
    elif selection == "unique":
        check.update(status="inserted", answer=format_inserted(leaders[0], author))
    elif selection == "ambiguous":
        check.update(status="ambiguous", answer=format_ambiguous(leaders, author))
    else:
        check.update(status="unavailable", answer=_UNAVAILABLE if author == "user" else _ASSISTANT_UNAVAILABLE)
    return check
