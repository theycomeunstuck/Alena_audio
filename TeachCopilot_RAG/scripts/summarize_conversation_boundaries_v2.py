"""Unblind sealed AI reviews; keep coverage and factual errors separate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FLAGS = ("technical_answer_present", "material_factual_error", "meets_task",
         "respects_requested_help", "unsupported_mastery_or_assistance_claim")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(review_dir: Path, output_dir: Path):
    root, out = review_dir.resolve(), output_dir.resolve()
    if out.exists() or out == root or root in out.parents:
        raise ValueError("Use a new output directory outside the sealed reviews")
    seal = read(root / "REVIEW_SEAL.json")
    for name, digest in seal["sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("A sealed review changed before unblinding")
    # Reviews must already be sealed before the mapping is read.
    mapping_path = root / "restricted_mapping.json"
    mapping = {row["case_id"]: row["slots"] for row in read(mapping_path)["cases"]}
    joined = []
    sources = [root / "REVIEW_SEAL.json", mapping_path, root / "paired_machine_summary.json"]
    for part in ("T01_T06", "T07_T12"):
        path = root / f"ai_review_{part}.json"
        sources.append(path)
        for case in read(path)["cases"]:
            case_id = case["case_id"]
            row = {"case_id": case_id, "variants": {}}
            for slot in case["slots"]:
                variant = mapping[case_id][slot["slot"]]
                if variant in row["variants"]:
                    raise ValueError("Duplicate reviewed variant")
                row["variants"][variant] = dict(slot)
            if set(row["variants"]) != {"baseline", "candidate"}:
                raise ValueError("Each case requires two reviewed variants")
            joined.append(row)
    if {row["case_id"] for row in joined} != {f"T{i:02d}" for i in range(1, 13)} or len(joined) != 12:
        raise ValueError("All twelve teaching cases must be reviewed exactly once")
    counts = {}
    for variant in ("baseline", "candidate"):
        counts[variant] = {}
        for flag in FLAGS:
            values = [row["variants"][variant][flag] for row in joined]
            counts[variant][flag] = {
                "denominator": 12, "true": sum(value is True for value in values),
                "false": sum(value is False for value in values),
                "unknown": sum(type(value) is not bool for value in values),
            }
    paired = {"baseline_only_meets": [], "candidate_only_meets": [], "both_meet": [], "neither_meets": []}
    for row in joined:
        left, right = [row["variants"][name]["meets_task"] is True for name in ("baseline", "candidate")]
        key = "both_meet" if left and right else "baseline_only_meets" if left else "candidate_only_meets" if right else "neither_meets"
        paired[key].append(row["case_id"])
    machine = read(root / "paired_machine_summary.json")
    preference = {split: machine[split] for split in ("all", "dev", "heldout")}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed_unblinded_comparison",
        "human_reviewed": False, "semantic_assessment": "uncalibrated_ai_review_not_formal_judge",
        "review_sealed_before_mapping_opened_at": seal["sealed_at_before_unblinding"],
        "preference_machine_diagnostics": preference,
        "teaching": {"denominator": 12, "counts": counts, "paired": paired, "cases": joined},
        "input_sha256": {path.name: sha(path) for path in sources},
        "summarizer_sha256": sha(Path(__file__)), "model_calls": 0, "paid_calls": 0,
        "limits": ["AI reviews are uncalibrated and human review is missing.",
                   "Paired responses are not independent observations.",
                   "A mechanical or AI label is not an overall production-readiness verdict."],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2).replace("\u2014", "–") + "\n", encoding="utf-8")
    lines = ["# Раскрытое сравнение учебных ответов", "",
             "Некалиброванный AI-разбор синтетического набора, человеческого просмотра нет.", "",
             "| Признак | Исходная версия | Изменённая версия |", "| --- | ---: | ---: |"]
    for flag in FLAGS:
        left, right = [counts[name][flag] for name in ("baseline", "candidate")]
        lines.append(f"| {flag} | {left['true']}/12, неизвестно {left['unknown']} | {right['true']}/12, неизвестно {right['unknown']} |")
    lines.extend(["", "Значение true у material_factual_error означает ошибку, а не успех.",
                  "Правильность, объём помощи и неподтверждённые выводы считаются раздельно.", "",
                  "| Случай | Выполняет запрос: исходная | Выполняет запрос: изменённая |", "| --- | --- | --- |"])
    for row in joined:
        lines.append(f"| {row['case_id']} | {row['variants']['baseline']['meets_task']} | {row['variants']['candidate']['meets_task']} |")
    lines.extend(["", "Обзоры зафиксированы SHA256 до раскрытия соответствия A/B.",
                  "Новых генераций, программной переоценки смысла и платных вызовов нет.", ""])
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.review_dir, args.output_dir)
    print(json.dumps({"status": report["status"], "teaching_counts": report["teaching"]["counts"]}))
