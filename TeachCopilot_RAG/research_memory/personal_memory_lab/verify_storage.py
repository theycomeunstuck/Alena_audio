"""Live PostgreSQL boundary and recovery check on isolated synthetic rows."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from .archive import Archive
from .chunks import lesson_chunks


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".local.env", override=False)
    db = Archive(os.environ["DATABASE_URL"])
    db.initialize()
    run = "storage-qa-" + uuid4().hex[:12]
    clone = run + "-rebuild"
    first = {"lesson_id": "lesson-a", "observed_on": "2026-01-01", "topic_ids": ["math"],
             "turns": [{"turn_id": "turn-1", "role": "teacher", "text": "Как получилось?"},
                       {"turn_id": "turn-2", "role": "student", "text": "Не знаю."}]}
    other = {**first, "turns": [{**first["turns"][0]}, {**first["turns"][1], "text": "Проверил."}]}
    vector = [1.0] + [0.0] * 1023
    report = {"run_id": run, "scientific_result": False}
    db.put(run, "child-a", first)
    report["duplicate_is_noop"] = not db.put(run, "child-a", first)["inserted"]
    db.put(run, "child-b", other)
    for child, value in (("child-a", first), ("child-b", other)):
        parts = lesson_chunks(value)
        db.put_chunks(run, child, value, parts, [vector] * len(parts))
    report["isolated_originals"] = db.lessons(run, "child-a") == [first] and db.lessons(run, "child-b") == [other]
    search_rows = db.search(run, "child-a", "получилось", vector)
    report["isolated_search"] = (bool(search_rows) and all(x["lesson_id"] == "lesson-a" for x in search_rows)
                                and db.chunks(run, "child-a")[0]["content"]["turns"] == first["turns"])
    try:
        db.put(run, "child-a", other)
    except ValueError:
        report["immutable_original"] = True
    else:
        report["immutable_original"] = False
    db.claim(run, "child-a", "lesson-a", "fake-index")
    db.finish(run, "child-a", "lesson-a", "fake-index", "SimulatedCrash")
    try:
        db.claim(run, "child-a", "lesson-a", "fake-index")
    except RuntimeError:
        report["uncertain_write_blocks_retry"] = True
    else:
        report["uncertain_write_blocks_retry"] = False
    db.clone_for_rebuild(run, clone, "child-a")
    report["rebuild_preserves_source"] = db.lessons(clone, "child-a") == [first]
    report["rebuild_copies_fragments"] = len(db.chunks(clone, "child-a")) == 1
    report["rebuild_has_no_old_index_jobs"] = db.index_references(clone, "child-a", "fake-index") == []
    db.claim(clone, "child-a", "lesson-a", "fake-index")
    db.finish(clone, "child-a", "lesson-a", "fake-index")
    report["rebuild_can_index_once"] = not db.claim(clone, "child-a", "lesson-a", "fake-index")
    report["ok"] = all(v for k, v in report.items() if k not in {"run_id", "scientific_result"})
    print(json.dumps(report, ensure_ascii=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
