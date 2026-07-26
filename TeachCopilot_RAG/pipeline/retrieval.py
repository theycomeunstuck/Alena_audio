"""Task-bank retrieval for ЗБР stage 4, with a fallback chain down to stdlib.

Stage 4 must produce a *grounded* task: the storyline is invented, the
mathematics is not. This module finds the curriculum fragments the generator is
allowed to lean on, trying three sources in order:

1. ``pgvector`` through :func:`pipeline.rag.search_knowledge` — the production path;
2. a local JSON vector index built by ``scripts/json_rag.py`` — no PostgreSQL;
3. a stdlib lexical scan of ``knowledge-data/*.json`` and ``books/*.md``.

Step 3 exists so a tutor can generate a card on a laptop with nothing installed
and no services running. Every result carries ``origin``, so the CLI can say
which path actually answered instead of pretending they are equivalent.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.config import DATABASE_URL

logger = logging.getLogger(__name__)

_RAG_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = _RAG_ROOT / "knowledge-data"
BOOKS_DIR = _RAG_ROOT / "books"

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_STEM_LENGTH = 5


@dataclass(frozen=True)
class Material:
    """One curriculum fragment offered to the task generator."""

    topic_id: str | None
    topic: str
    content: str
    answer: str = ""
    difficulty: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    source: str = ""
    score: float = 0.0
    origin: str = "lexical"

    def as_dict(self) -> dict:
        return {
            "topic_id": self.topic_id,
            "topic": self.topic,
            "content": self.content,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "tags": list(self.tags),
            "source": self.source,
            "score": round(self.score, 4),
            "origin": self.origin,
        }


def _stems(text: str) -> list[str]:
    """Crude Russian-friendly stemming: lowercase, ё→е, clip to a 5-char prefix.

    Good enough to match «луч» against «лучи» and «отрезок» against «отрезка»
    without pulling in a morphology dependency.
    """
    normalised = text.lower().replace("ё", "е")
    return [token[:_STEM_LENGTH] for token in _TOKEN_RE.findall(normalised)]


def _lexical_score(query_stems: list[str], document: str) -> float:
    """Token-overlap score, length-normalised so long fragments do not win by size."""
    if not query_stems:
        return 0.0
    document_stems = _stems(document)
    if not document_stems:
        return 0.0
    document_set = set(document_stems)
    hits = sum(1 for stem in set(query_stems) if stem in document_set)
    if not hits:
        return 0.0
    return hits / (len(set(query_stems)) * (1.0 + math.log(1 + len(document_stems) / 40)))


def _load_task_bank(knowledge_dir: Path) -> list[Material]:
    """Read every JSON task file. Malformed files are skipped, not fatal."""
    materials: list[Material] = []
    if not knowledge_dir.is_dir():
        return materials

    for path in sorted(knowledge_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            logger.warning("Skipping unreadable task bank file: %s", path.name)
            continue

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = next(
                (payload[key] for key in ("tasks", "records", "items", "data") if isinstance(payload.get(key), list)),
                [],
            )
        else:
            records = []

        for record in records:
            if not isinstance(record, dict):
                continue
            content = record.get("content") or record.get("question") or record.get("text") or ""
            if not str(content).strip():
                continue
            tags = record.get("tags") or []
            materials.append(
                Material(
                    topic_id=record.get("topic_id"),
                    topic=str(record.get("task_id") or record.get("topic") or path.stem),
                    content=str(content),
                    answer=str(record.get("answer") or record.get("solution") or ""),
                    difficulty=record.get("difficulty"),
                    tags=tuple(str(tag) for tag in tags if isinstance(tag, str)),
                    source=path.name,
                )
            )
    return materials


def _load_books(books_dir: Path) -> list[Material]:
    """Read the plain-text explanations shipped in ``books/`` as extra material."""
    materials: list[Material] = []
    if not books_dir.is_dir():
        return materials

    for path in sorted(books_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        topic = path.stem
        body_lines = []
        for line in text.splitlines():
            if line.startswith("TOPIC:"):
                topic = line.split(":", 1)[1].strip() or topic
            elif line.startswith("SUBJECT:"):
                continue
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if body:
            materials.append(Material(topic_id=None, topic=topic, content=body, source=path.name))
    return materials


# Насколько поднимать материал нужной сложности и материал про ту самую ошибку
# ребёнка. Это предпочтения, а не фильтры: подходящего может просто не быть,
# и тогда лучше отдать что-то по теме, чем ничего.
DIFFICULTY_BOOST = 1.4
ERROR_TAG_BOOST = 1.6


def _search_lexically(
    query: str,
    topic_id: str | None,
    grade: str | None,
    limit: int,
    knowledge_dir: Path,
    books_dir: Path,
    difficulty: str | None = None,
    error_tags: tuple[str, ...] = (),
) -> list[Material]:
    pool = _load_task_bank(knowledge_dir) + _load_books(books_dir)
    if not pool:
        return []

    # Prefer material tagged with exactly this curriculum topic; only widen the
    # search when the topic has no material at all, so a geometry request never
    # gets answered with division tasks while geometry material exists.
    on_topic = [item for item in pool if topic_id and item.topic_id == topic_id]
    if not on_topic and topic_id:
        parent = topic_id.rsplit(".", 1)[0]
        on_topic = [item for item in pool if item.topic_id and item.topic_id.startswith(parent)]
    candidates = on_topic or pool

    query_stems = _stems(f"{query} {topic_id or ''}")
    error_stems = set(_stems(" ".join(error_tags))) if error_tags else set()

    scored: list[Material] = []
    for item in candidates:
        haystack = " ".join([item.topic, item.content, item.answer, " ".join(item.tags)])
        score = _lexical_score(query_stems, haystack)
        if grade and item.topic_id and f".g{grade}." not in item.topic_id:
            score *= 0.5  # wrong grade is a penalty, not a hard filter
        if difficulty and item.difficulty == difficulty:
            # Сложность вычислена правилом ЗБР — материал той же ступени ближе
            # к тому, что ребёнок сейчас потянет.
            score *= DIFFICULTY_BOOST
        if error_stems and set(_stems(" ".join(item.tags) + " " + item.content)) & error_stems:
            # Материал, разбирающий именно эту ошибку, ценнее любого другого:
            # задание должно бить по ней, а не по теме вообще.
            score *= ERROR_TAG_BOOST
        if score <= 0.0:
            continue
        scored.append(
            Material(
                topic_id=item.topic_id,
                topic=item.topic,
                content=item.content,
                answer=item.answer,
                difficulty=item.difficulty,
                tags=item.tags,
                source=item.source,
                score=score,
                origin="lexical",
            )
        )
    if not scored and on_topic:
        # Everything in `on_topic` is on the requested curriculum topic by
        # construction, so a zero lexical overlap means the wording differed —
        # not that the topic has no material. Returning the topic's own tasks
        # beats returning nothing and generating a task with no grounding.
        logger.debug("Lexical overlap was zero for topic_id=%s; falling back to topic order", topic_id)
        return [
            Material(
                topic_id=item.topic_id,
                topic=item.topic,
                content=item.content,
                answer=item.answer,
                difficulty=item.difficulty,
                tags=item.tags,
                source=item.source,
                score=0.0,
                origin="lexical",
            )
            for item in on_topic[:limit]
        ]

    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]


def _search_pgvector(query: str, topic_id: str | None, grade: str | None, limit: int) -> list[Material]:
    if not DATABASE_URL:
        return []
    try:
        from pipeline.rag import search_knowledge  # noqa: PLC0415 - heavy optional import
    except ImportError:
        logger.debug("pgvector retrieval unavailable: embedding/DB dependencies are not installed")
        return []

    try:
        rows = search_knowledge(query=query, topic_ids=[topic_id] if topic_id else None, grade=grade, limit=limit)
    except Exception:  # pragma: no cover - search_knowledge already logs details
        logger.exception("pgvector retrieval failed")
        return []

    materials = []
    for row in rows or []:
        content = row.get("content") or row.get("image_descriptions") or ""
        if not str(content).strip():
            continue
        materials.append(
            Material(
                topic_id=row.get("topic_id"),
                topic=str(row.get("topic") or "материал"),
                content=str(content),
                difficulty=row.get("difficulty"),
                tags=tuple(row.get("tags") or ()),
                source=str(row.get("source_file") or "knowledge_base"),
                score=float(row.get("score") or 0.0),
                origin="pgvector",
            )
        )
    return materials


def _search_json_index(
    index_path: Path, query: str, topic_id: str | None, grade: str | None, limit: int
) -> list[Material]:
    if not index_path.is_file():
        return []
    import sys  # noqa: PLC0415 - scripts/ is not a package

    scripts_dir = str(_RAG_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        import json_rag  # noqa: PLC0415 - optional, needs sentence-transformers
    except ImportError:
        logger.debug("JSON index retrieval unavailable: embedding dependencies are not installed")
        return []

    try:
        rows = json_rag.search_index(index_path, query, top_k=limit, topic_id=topic_id, grade=grade)
    except Exception:
        logger.exception("JSON index retrieval failed for %s", index_path)
        return []

    materials = []
    for row in rows:
        metadata = row.get("metadata") or {}
        materials.append(
            Material(
                topic_id=metadata.get("topic_id"),
                topic=str(metadata.get("task_id") or metadata.get("topic") or "материал"),
                content=str(row.get("content") or ""),
                answer=str(metadata.get("answer") or ""),
                difficulty=metadata.get("difficulty"),
                tags=tuple(metadata.get("tags") or ()),
                source=index_path.name,
                score=float(row.get("score") or 0.0),
                origin="json_index",
            )
        )
    return materials


def retrieve_material(
    query: str,
    *,
    topic_id: str | None = None,
    grade: str | None = None,
    limit: int = 3,
    index_path: Path | None = None,
    knowledge_dir: Path | None = None,
    books_dir: Path | None = None,
    difficulty: str | None = None,
    error_tags: tuple[str, ...] = (),
) -> list[Material]:
    """Return up to ``limit`` curriculum fragments for one task-design request.

    The first source that returns anything wins; an empty list means no source
    had material, and the caller must decide whether to generate ungrounded.

    ``difficulty`` (из правила ЗБР) и ``error_tags`` (типичные ошибки ребёнка)
    — предпочтения ранжирования, а не фильтры: подходящего материала может не
    быть, и тогда лучше отдать что-то по теме, чем ничего.
    """
    for finder in (
        lambda: _search_pgvector(query, topic_id, grade, limit),
        lambda: _search_json_index(index_path, query, topic_id, grade, limit) if index_path else [],
        lambda: _search_lexically(
            query,
            topic_id,
            grade,
            limit,
            knowledge_dir or KNOWLEDGE_DIR,
            books_dir or BOOKS_DIR,
            difficulty,
            tuple(error_tags),
        ),
    ):
        results = finder()
        if results:
            logger.info("Retrieved %d fragments via %s", len(results), results[0].origin)
            return results
    logger.warning("No curriculum material found for topic_id=%s query=%r", topic_id, query[:80])
    return []
