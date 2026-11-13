"""Short, actual-adapter smoke. Creates only new synthetic research scopes."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from .archive import Archive
from .backends import GraphitiAdapter, Mem0Adapter, lesson_text, make_graphiti, make_mem0
from .chunks import lesson_chunks
from .runtime import LocalModel


async def run(output, count, history_id=None):
    load_dotenv(Path(__file__).resolve().parents[2] / ".local.env", override=False)
    database_url = os.environ["DATABASE_URL"]
    run_id = "smoke-" + uuid4().hex[:12]
    path = (Path(__file__).parent / "data/histories.dev.jsonl" if history_id else
            Path(__file__).resolve().parents[1] / "data/graph_extension_structured_v2_v1/histories.candidate.jsonl")
    histories = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    history = next(item for item in histories if item["history_id"] == history_id) if history_id else histories[0]
    learner_id = history["learner_id"]
    lessons = history["lessons"][-count:]
    question = "Какую ошибку допустил ребёнок и что помогло её исправить?"
    archive, runtime = Archive(database_url), LocalModel(context=16384)
    archive.initialize()
    report = {"run_id": run_id, "learner_id": learner_id, "lesson_count": len(lessons),
              "question": question, "scientific_result": False, "steps": {}}
    for lesson in lessons:
        try:
            first = archive.put(run_id, learner_id, lesson)
            chunks = lesson_chunks(lesson)
            vectors = runtime.embed_many([lesson_text(chunk) for chunk in chunks])
            archive.put_chunks(run_id, learner_id, lesson, chunks, vectors)
            second = archive.put(run_id, learner_id, lesson)
            report["steps"].setdefault("archive", []).append({"lesson_id": lesson["lesson_id"],
                "first": first, "duplicate": second})
        except Exception as exc:
            report["steps"].setdefault("archive", []).append({"error": type(exc).__name__, "detail": str(exc)})
    try:
        report["steps"]["archive_search"] = archive.search(run_id, learner_id, question, runtime.embed(question))
        report["steps"]["isolation"] = archive.lessons(run_id, "different-synthetic-child") == []
    except Exception as exc:
        report["steps"]["archive_search"] = {"error": type(exc).__name__, "detail": str(exc)}

    password = (Path(__file__).resolve().parents[4] /
                ".local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt").read_text().strip()
    graph = None
    try:
        graph = make_graphiti(runtime, password)
        adapter = GraphitiAdapter(graph, archive, runtime)
        await adapter.initialize()
        ingested = [await adapter.ingest(run_id, learner_id, lesson) for lesson in lessons]
        duplicate = await adapter.ingest(run_id, learner_id, lessons[-1])
        hits = await adapter.search(run_id, learner_id, question)
        isolated = await adapter.search(run_id, "different-synthetic-child", question)
        report["steps"]["graphiti"] = {"ingested": ingested, "duplicate": duplicate,
                                      "hits": hits, "other_child_hits": isolated}
    except Exception as exc:
        report["steps"]["graphiti"] = {"error": type(exc).__name__, "detail": str(exc)[:500]}
    finally:
        if graph is not None:
            await graph.close()

    try:
        work_dir = Path(__file__).resolve().parents[4] / ".local/personal-memory/mem0"
        memory = make_mem0(runtime, database_url, work_dir)
        adapter = Mem0Adapter(memory, archive)
        ingested = [adapter.ingest(run_id, learner_id, lesson) for lesson in lessons]
        duplicate = adapter.ingest(run_id, learner_id, lessons[-1])
        hits = adapter.search(run_id, learner_id, question)
        isolated = adapter.search(run_id, "different-synthetic-child", question)
        report["steps"]["mem0"] = {"ingested": ingested, "duplicate": duplicate,
                                   "hits": hits, "other_child_hits": isolated}
    except Exception as exc:
        report["steps"]["mem0"] = {"error": type(exc).__name__, "detail": str(exc)[:500]}
    report["calls"] = runtime.calls
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "steps": {key: (value.get("error", "ok") if isinstance(value, dict)
        else "ok") for key, value in report["steps"].items()}, "calls": len(runtime.calls)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--lessons", type=int, choices=(1, 3, 10), default=1)
    parser.add_argument("--history-id", choices=("pm-h01", "pm-h02", "pm-h03"))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    asyncio.run(run(output, args.lessons, args.history_id))


if __name__ == "__main__":
    main()
