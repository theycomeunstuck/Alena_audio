"""Simulate an interrupted real Neo4j write and rebuild in a fresh test scope."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4

from psycopg2.extensions import make_dsn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.config import Settings
from research_memory.personal_memory_lab.archive import Archive
from research_memory.personal_memory_lab.backends import GraphitiAdapter, lesson_text, make_graphiti
from research_memory.personal_memory_lab.chunks import lesson_chunks
from research_memory.personal_memory_lab.runtime import LocalModel
from research_memory.personal_memory_lab.workflow import PersonalMemoryWorkflow


async def run(args):
    dbname = args.db_name_file.read_text(encoding="ascii").strip()
    if not dbname.startswith("nirs_personal_graph_it_") or not dbname.replace("_", "").isalnum():
        raise ValueError("Only an isolated synthetic database is allowed")
    settings = Settings.from_env()
    dsn = make_dsn(settings.database_url, dbname=dbname)
    archive, runtime = Archive(dsn), LocalModel(base_url=settings.ollama_url, context=16384)
    archive.initialize()
    token = uuid4().hex[:12]
    old_run, new_run, child = "recovery-old-" + token, "recovery-new-" + token, "recovery-child-" + token
    lesson = {"lesson_id": "lesson-recovery", "observed_on": "2026-01-01", "topic_ids": ["math"],
              "turns": [{"turn_id": "q", "role": "teacher", "text": "Сколько равных частей?"},
                        {"turn_id": "a", "role": "student", "text": "Четыре, значит три четверти."}]}
    archive.put(old_run, child, lesson)
    chunks = lesson_chunks(lesson, max_bytes=1800, hard_max_bytes=6000)
    archive.put_chunks(old_run, child, lesson, chunks, runtime.embed_many([lesson_text(c) for c in chunks]))
    graph = make_graphiti(runtime, Path(settings.personal_graph_password_file).read_text(encoding="utf-8").strip())
    report = {"database": dbname, "old_run": old_run, "new_run": new_run, "scientific_result": False}
    try:
        adapter = GraphitiAdapter(graph, archive, runtime)
        await adapter.initialize()
        class InterruptAfterWrite:
            calls = 0
            episode_id = None
            async def add_episode(self, **kwargs):
                self.calls += 1
                result = await graph.add_episode(**kwargs)
                self.episode_id = result.episode.uuid
                raise ConnectionError("simulated loss after Neo4j acknowledged the write")
        interrupted = InterruptAfterWrite()
        try:
            await GraphitiAdapter(interrupted, archive, runtime).ingest(old_run, child, lesson)
        except ConnectionError:
            report["interrupted_after_graph_write"] = bool(interrupted.episode_id)
        try:
            await GraphitiAdapter(interrupted, archive, runtime).ingest(old_run, child, lesson)
        except RuntimeError:
            report["uncertain_retry_blocked"] = interrupted.calls == 1
        report["old_scope_has_no_trusted_references"] = not archive.index_references(old_run, child, "graphiti")
        report["copied_lessons"] = archive.clone_for_rebuild(old_run, new_run, child)
        report["canonical_original_preserved"] = archive.lessons(new_run, child) == [lesson]
        rebuilt = await adapter.ingest(new_run, child, lesson)
        report["rebuilt"] = rebuilt["status"] == "indexed"
        report["duplicate_is_noop"] = (await adapter.ingest(new_run, child, lesson))["status"] == "already_indexed"
        result = await PersonalMemoryWorkflow(archive, runtime, graphiti=adapter).search(
            new_run, child, "Что помогло исправить ответ?", mode="graphiti", budget_tokens=4096)
        report["retrieved_original_sources"] = set(result["source_ids"]) == {"q", "a"}
        report["other_child_isolated"] = not archive.lessons(new_run, "another-child") and not (
            await adapter.search(new_run, "another-child", "Что помогло?"))
        report["model_calls"] = len(runtime.calls)
        checks = ("interrupted_after_graph_write", "uncertain_retry_blocked", "old_scope_has_no_trusted_references",
                  "canonical_original_preserved", "rebuilt", "duplicate_is_noop", "retrieved_original_sources",
                  "other_child_isolated")
        report["ok"] = report["copied_lessons"] == 1 and all(report.get(key) for key in checks)
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:300])
    finally:
        await graph.close()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("ok", "model_calls", "error_type", "error")}, ensure_ascii=True))
    if not report.get("ok"):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-name-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
