"""Rebuild a real Mem0 index after an acknowledged write with a lost response."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

from psycopg2.extensions import make_dsn, parse_dsn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.config import Settings
from local_app.memory_ported.personal_memory_lab.archive import Archive
from local_app.memory_ported.personal_memory_lab.backends import Mem0Adapter, lesson_text, make_mem0, scope
from local_app.memory_ported.personal_memory_lab.chunks import lesson_chunks
from local_app.memory_ported.personal_memory_lab.runtime import LocalModel
from local_app.memory_ported.personal_memory_lab.workflow import PersonalMemoryWorkflow


async def run(args):
    dbname = args.db_name_file.read_text(encoding="ascii").strip()
    if not dbname.startswith("nirs_personal_graph_it_") or not dbname.replace("_", "").isalnum():
        raise ValueError("Only an isolated synthetic database is allowed")
    settings = Settings.from_env()
    # Audit F-049: only a loopback PostgreSQL endpoint is allowed (same guard
    # as smoke_personal_graph_pilot.py / smoke_personal_memory_pilot.py).
    params = parse_dsn(settings.database_url)
    hosts = (params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR"))
    if not (any(hosts) and all(not host or host in {"localhost", "127.0.0.1", "::1"} for host in hosts)):
        raise SystemExit("Refusing to run: only a loopback PostgreSQL endpoint is allowed")
    dsn = make_dsn(settings.database_url, dbname=dbname)
    archive, runtime = Archive(dsn), LocalModel(base_url=settings.ollama_url, context=16384)
    archive.initialize()
    token = uuid4().hex[:12]
    old_run, new_run, child = "mem0-old-" + token, "mem0-new-" + token, "mem0-child-" + token
    lesson = {"lesson_id": "lesson-mem0-recovery", "observed_on": "2026-01-01", "topic_ids": ["art"],
              "turns": [{"turn_id": "q", "role": "teacher", "text": "Что тебе нравится рисовать?"},
                        {"turn_id": "a", "role": "student", "text": "Мне нравится рисовать космических котов."}]}
    complete_lesson = {"lesson_id": "lesson-mem0-complete", "observed_on": "2026-01-02", "topic_ids": ["art"],
              "turns": [{"turn_id": "q2", "role": "teacher", "text": "Каких животных ты рисовал?"},
                        {"turn_id": "a2", "role": "student", "text": "Я рисовал лесных мишек."}]}
    report = {"database": dbname, "old_run": old_run, "new_run": new_run, "scientific_result": False}
    try:
        for item in (lesson, complete_lesson):
            archive.put(old_run, child, item)
            chunks = lesson_chunks(item, max_bytes=1800, hard_max_bytes=6000)
            archive.put_chunks(old_run, child, item, chunks, runtime.embed_many([lesson_text(c) for c in chunks]))
        memory = make_mem0(runtime, dsn, args.output.parent / ("mem0-recovery-runtime-" + token))
        adapter = Mem0Adapter(memory, archive)
        complete = adapter.ingest(old_run, child, complete_lesson)
        report["initial_complete_lesson_indexed"] = complete["status"] == "indexed" and bool(complete["results"])

        class InterruptAfterWrite:
            calls = 0
            emitted = []

            def add(self, text, **kwargs):
                self.calls += 1
                result = memory.add(text, **kwargs)
                self.emitted = result.get("results", [])
                raise ConnectionError("simulated loss after Mem0 acknowledged the write")

        interrupted = InterruptAfterWrite()
        try:
            Mem0Adapter(interrupted, archive).ingest(old_run, child, lesson)
        except ConnectionError:
            report["interrupted_after_mem0_write"] = bool(interrupted.emitted)
        try:
            Mem0Adapter(interrupted, archive).ingest(old_run, child, lesson)
        except RuntimeError:
            report["uncertain_retry_blocked"] = interrupted.calls == 1

        question = "Что ребёнок любит рисовать?"
        raw_old = memory.search(question, filters={"user_id": scope(old_run, child)}, top_k=8, threshold=0)
        report["raw_uncertain_index_contains_memory"] = any(
            (item.get("metadata") or {}).get("lesson_id") == lesson["lesson_id"]
            for item in raw_old.get("results", []))
        limited_hits = adapter.search(old_run, child, question, limit=1)
        report["uncertain_memory_not_trusted"] = all(
            lesson["lesson_id"] not in hit.get("lesson_ids", []) for hit in limited_hits)
        report["completed_memory_not_crowded_out"] = any(
            hit.get("lesson_ids") == [complete_lesson["lesson_id"]] for hit in limited_hits)
        report["copied_lessons"] = archive.clone_for_rebuild(old_run, new_run, child)
        report["canonical_original_preserved"] = archive.lessons(new_run, child) == [lesson, complete_lesson]
        rebuilt = [adapter.ingest(new_run, child, item) for item in (lesson, complete_lesson)]
        report["rebuilt"] = all(item["status"] == "indexed" and item["results"] for item in rebuilt)
        report["duplicate_is_noop"] = all(adapter.ingest(new_run, child, item)["status"] == "already_indexed"
                                           for item in (lesson, complete_lesson))
        result = await PersonalMemoryWorkflow(archive, runtime, mem0=adapter).search(
            new_run, child, question, mode="mem0", budget_tokens=4096)
        report["retrieved_original_sources"] = {"q", "a"} <= set(result["source_ids"])
        raw_other = memory.search(question, filters={"user_id": scope(new_run, "another-child")},
                                  top_k=8, threshold=0)
        report["other_child_isolated"] = not raw_other.get("results") and not adapter.search(
            new_run, "another-child", question)
        report["model_calls"] = len(runtime.calls)
        checks = ("initial_complete_lesson_indexed", "interrupted_after_mem0_write", "uncertain_retry_blocked",
                  "raw_uncertain_index_contains_memory", "uncertain_memory_not_trusted", "completed_memory_not_crowded_out",
                  "canonical_original_preserved", "rebuilt", "duplicate_is_noop",
                  "retrieved_original_sources", "other_child_isolated")
        report["ok"] = report["copied_lessons"] == 2 and all(report.get(key) for key in checks)
    except Exception as exc:
        report.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:300])
    finally:
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
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
