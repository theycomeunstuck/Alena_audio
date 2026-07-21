#!/usr/bin/env python3
"""Small file-backed vector RAG for JSON learning materials.

The index is JSON too: each source record keeps its metadata and a generated
embedding. It is intentionally a simple local MVP, with no PostgreSQL service.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.rag import embed_text


def _records(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = next(
            (payload[key] for key in ("records", "items", "tasks", "data") if isinstance(payload.get(key), list)),
            [payload],
        )
    else:
        raise ValueError("JSON source must be an object or list of objects")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every JSON record must be an object")
    return records


def _content(record: dict[str, Any]) -> str:
    question = record.get("content") or record.get("question") or record.get("text") or ""
    answer = record.get("answer") or record.get("solution") or ""
    tags = record.get("tags") or []
    return "\n".join(part for part in (str(question), f"Теги: {', '.join(tags)}" if tags else "", str(answer)) if part)


def build_index(source: Path, index: Path) -> int:
    """Embed a task-bank JSON file and persist a portable JSON vector index."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for number, record in enumerate(_records(payload), start=1):
        content = _content(record)
        if not content.strip():
            raise ValueError(f"record {number} has no content/question/text")
        rows.append({"content": content, "metadata": record, "embedding": embed_text(content)})
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"version": 1, "source": source.name, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def search_index(index: Path, query: str, top_k: int = 3, topic_id: str | None = None, grade: str | None = None) -> list[dict[str, Any]]:
    """Search the JSON index, optionally using exact curriculum metadata filters."""
    data = json.loads(index.read_text(encoding="utf-8"))
    query_embedding = embed_text(query)
    results = []
    for row in data.get("rows", []):
        metadata = row["metadata"]
        if topic_id and metadata.get("topic_id") != topic_id:
            continue
        if grade and str(metadata.get("grade")) != grade:
            continue
        results.append({"score": _cosine(query_embedding, row["embedding"]), "content": row["content"], "metadata": metadata})
    return sorted(results, key=lambda result: result["score"], reverse=True)[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and search a local JSON vector RAG index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--index", type=Path, required=True)
    search = subparsers.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--topic-id")
    search.add_argument("--grade")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps({"indexed": build_index(args.source, args.index), "index": str(args.index)}, ensure_ascii=False))
    else:
        print(json.dumps(search_index(args.index, args.query, args.top_k, args.topic_id, args.grade), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
