"""Exercise all three retrieval workflows on an already indexed smoke scope."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .archive import Archive
from .backends import GraphitiAdapter, Mem0Adapter, make_graphiti, make_mem0
from .runtime import LocalModel
from .workflow import PersonalMemoryWorkflow


async def run(smoke_path, output):
    load_dotenv(Path(__file__).resolve().parents[2] / ".local.env", override=False)
    seed = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[4]
    password = (root / ".local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt").read_text().strip()
    runtime, archive = LocalModel(), Archive(os.environ["DATABASE_URL"])
    graph = make_graphiti(runtime, password)
    mem = make_mem0(runtime, os.environ["DATABASE_URL"], root / ".local/personal-memory/mem0")
    workflow = PersonalMemoryWorkflow(archive, runtime,
        graphiti=GraphitiAdapter(graph, archive, runtime), mem0=Mem0Adapter(mem, archive))
    report = {"source_smoke": str(smoke_path), "scientific_result": False, "results": {}}
    try:
        for mode in ("mem0", "graphiti", "graphiti_mem0"):
            try:
                report["results"][mode] = await workflow.answer(seed["run_id"], seed["learner_id"],
                    seed["question"], mode=mode, budget_tokens=4096)
            except Exception as exc:
                report["results"][mode] = {"error": type(exc).__name__, "detail": str(exc)[:500]}
    finally:
        await graph.close()
    report["calls"] = runtime.calls
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: {"error": v.get("error"), "memory_tokens": v.get("memory_tokens"),
                          "source_count": len(v.get("source_ids", [])), "answer": v.get("answer")}
                      for k, v in report["results"].items()}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.smoke, output))


if __name__ == "__main__":
    main()
