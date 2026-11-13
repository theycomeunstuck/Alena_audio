"""Try source-preserving fragments with the unchanged legacy memory writers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_memory.graph_extension.runner import _prepare_memory
from research_memory.graph_extension_v3.runner import _prepare_v2_memory
from research_memory.runner import CallJournal
from research_memory.personal_memory_lab.chunks import lesson_chunks
from research_memory.personal_memory_lab.experiment import model_digest
from research_memory.personal_memory_lab.runtime import LegacyClient, LocalModel


class CompactSummaryJournal(CallJournal):
    """Diagnostic variant only: avoid enumerating repetitive routine exercises."""

    def __call__(self, messages, purpose, response_schema=None):
        if purpose == "summary":
            messages = [dict(message) for message in messages]
            messages[0]["content"] += (
                "\n\nДополнение для длинных занятий: не перечисляй однотипные правильные "
                "упражнения. Ошибки, уточнения, помощь, исправления, короткие неоднозначные "
                "ответы, просьбы, интересы, случайные факты, незавершённые дела и противоречия "
                "сохраняй с контекстом и источниками. Для повторяющихся успешных упражнений "
                "достаточно одного подтверждённого примера. Пиши очень кратко: ориентир для "
                "всего резюме – не более 2000 символов, без потери важных условий.")
        return super().__call__(messages, purpose, response_schema)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lessons", type=int, default=3)
    parser.add_argument("--fragment-bytes", type=int, default=3000)
    parser.add_argument("--whole-lessons", action="store_true", help="Test unchanged writers on whole lessons")
    parser.add_argument("--context", type=int, choices=(16384, 32768), default=16384)
    parser.add_argument("--compact-summary", action="store_true", help="Try a separate compact-summary prompt")
    parser.add_argument("--memory-budget-chars", type=int, default=12000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--methods", nargs="+", choices=("summary", "structured", "graph", "structured_v2"),
                        default=["summary", "structured", "graph", "structured_v2"])
    args = parser.parse_args()
    if (not 1 <= args.lessons <= 10 or not 1000 <= args.fragment_bytes <= 6000
            or not 1000 <= args.memory_budget_chars <= 12000 or not 128 <= args.max_output_tokens <= 3072):
        parser.error("invalid lesson count, fragment size or memory/output limit")
    args.output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "research_memory/personal_memory_lab/data/histories.dev.jsonl"
    histories = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    history = next(item for item in histories if item["history_id"] == "pm-h02")
    original = history["lessons"][:args.lessons]
    fragments = []
    for lesson in original:
        parts = [lesson] if args.whole_lessons else lesson_chunks(
            lesson, max_bytes=args.fragment_bytes, hard_max_bytes=6000)
        for index, part in enumerate(parts, 1):
            fragments.append(part if args.whole_lessons else {
                **part, "lesson_id": f"{lesson['lesson_id']}#part-{index:03d}",
                "original_lesson_id": lesson["lesson_id"]})
    adapted = {**history, "lessons": fragments}
    original_ids = [t["turn_id"] for lesson in original for t in lesson["turns"]]
    fragment_ids = [t["turn_id"] for lesson in fragments for t in lesson["turns"]]
    if original_ids != fragment_ids:
        raise ValueError("Fragments do not preserve the original source order")
    started = time.perf_counter()
    with (args.output / "calls.jsonl").open("w", encoding="utf-8") as calls, \
         (args.output / "legacy_calls.jsonl").open("w", encoding="utf-8") as legacy_calls:
        runtime = LocalModel(context=args.context, call_stream=calls)
        digest = model_digest(runtime)
        client = LegacyClient(runtime)
        client.max_output_tokens = args.max_output_tokens
        journal = (CompactSummaryJournal if args.compact_summary else CallJournal)(client, legacy_calls)
        methods = tuple(m for m in args.methods if m != "structured_v2")
        legacy = _prepare_memory(adapted, methods, journal, args.memory_budget_chars)
        v2, v2_errors = (_prepare_v2_memory(adapted, journal, args.memory_budget_chars)
                         if "structured_v2" in args.methods else (None, []))
        report = {"history_id": history["history_id"], "lessons": len(original), "fragments": len(fragments),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "histories_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "mode": "whole_lessons" if args.whole_lessons else "fragments",
            "compact_summary_variant": args.compact_summary,
            "fragment_bytes": None if args.whole_lessons else args.fragment_bytes,
            "context_tokens": args.context, "model_digest": digest,
            "memory_budget_chars": args.memory_budget_chars, "max_output_tokens": args.max_output_tokens,
            "methods": args.methods,
            "model_digest_matches": digest == model_digest(runtime), "scientific_result": False,
            "source_order_preserved": original_ids == fragment_ids,
            "all_legacy_sources_preserved": set(original_ids) == set(legacy["sources"]),
            "summary_errors": legacy["summary_errors"], "structured_errors": legacy["extraction_errors"],
            "v2_errors": v2_errors, "legacy_warnings": len(legacy["extraction_warnings"]),
            "v2_warnings": len(v2.events) if v2 else 0, "structured_records": len(legacy["records"]),
            "v2_records": len(v2.records) if v2 else 0, "call_purposes": dict(Counter(c["purpose"] for c in runtime.calls)),
            "calls": len(runtime.calls), "elapsed_seconds": round(time.perf_counter() - started, 1)}
        if "summary" in args.methods:
            (args.output / "summary.txt").write_text(legacy["summary"] + "\n", encoding="utf-8")
        (args.output / "inventory.json").write_text(json.dumps({"structured": legacy["records"],
            "structured_v2": v2.records if v2 else [], "v2_warnings": list(v2.events) if v2 else [],
            "legacy_warnings": legacy["extraction_warnings"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    if report["summary_errors"] or report["structured_errors"] or report["v2_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
