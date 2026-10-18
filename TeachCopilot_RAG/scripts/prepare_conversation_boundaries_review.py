"""Prepare a blind paired review from two completed local boundary batches.

Reads only explicit JSON artifacts. It never imports application code, calls a
model, opens a database, executes model code or changes either source package.
Usage: python scripts/prepare_conversation_boundaries_review.py \
    --baseline-dir BASELINE --candidate-dir CANDIDATE --output-dir NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


INTEREST_FIXTURE = "tests/fixtures/conversation_preference_boundaries_v2.json"
TEACHING_FIXTURE = "tests/fixtures/conversation_teaching_checks_v2.json"
BLIND_SEED = "teachcopilot-paired-teaching-boundaries-v2-fixed-order"
OPERATIONS = {"add", "remove", "avoid", "allow"}


def read_object(path: Path, hashes: dict[str, str], root: Path) -> dict:
    raw = path.read_bytes()
    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def fixture_index(fixture: dict, key: str, expected: int) -> dict[str, dict]:
    rows = fixture.get("cases")
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Expected exactly {expected} fixed cases")
    result = {}
    for row in rows:
        case_id = row.get(key) if isinstance(row, dict) else None
        if not isinstance(case_id, str) or case_id in result:
            raise ValueError("Missing or duplicate fixed case identifier")
        result[case_id] = row
    return result


def load_package(root: Path) -> dict:
    hashes: dict[str, str] = {}
    report = read_object(root / "report.json", hashes, root)
    prereg = read_object(root / "preregistered.json", hashes, root)
    if report.get("status") != "completed" or report.get("preflight_only") is True:
        raise ValueError(f"A completed generation package is required: {root}")
    if (report.get("source_integrity") or {}).get("unchanged") is not True:
        raise ValueError(f"Source integrity must be explicitly verified: {root}")
    fixtures = prereg.get("fixtures")
    if not isinstance(fixtures, dict):
        raise ValueError("Missing fixed fixtures in preregistered.json")
    for name in (INTEREST_FIXTURE, TEACHING_FIXTURE):
        copied = read_object(root / Path(name).name, hashes, root)
        if fixtures.get(name) != copied:
            raise ValueError(f"Copied fixture differs from preregistration: {name}")
    interest = fixture_index(fixtures[INTEREST_FIXTURE], "case_id", 24)
    teaching = fixture_index(fixtures[TEACHING_FIXTURE], "id", 12)
    expected = {**{key: "interest" for key in interest}, **{key: "teaching" for key in teaching}}
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("All 36 case outcomes must remain recorded, including failures")
    by_id = {}
    for row in rows:
        case_id = row.get("case_id") if isinstance(row, dict) else None
        if case_id not in expected or case_id in by_id or row.get("kind") != expected[case_id]:
            raise ValueError("Unexpected, duplicate or incorrectly typed case outcome")
        if read_object(root / (case_id + ".json"), hashes, root) != row:
            raise ValueError(f"Case file differs from report.json: {case_id}")
        by_id[case_id] = row
    return {"root": root, "hashes": hashes, "report": report, "preregistered": prereg,
            "interest_fixture": fixtures[INTEREST_FIXTURE], "teaching_fixture": fixtures[TEACHING_FIXTURE],
            "interest_cases": interest, "teaching_cases": teaching, "rows": by_id}


def normal(value: str, aliases: dict) -> str:
    plain = " ".join(value.casefold().split())
    for canonical, variants in aliases.items():
        if plain in {" ".join(item.casefold().split()) for item in [canonical, *variants]}:
            return " ".join(canonical.casefold().split())
    return plain


def pair(item: dict, aliases: dict | None) -> tuple[str, str]:
    operation, interest = item["operation"], item["interest"]
    if operation not in OPERATIONS or not isinstance(interest, str) or not interest.strip():
        raise ValueError("Invalid preference pair")
    return operation, normal(interest, aliases) if aliases is not None else interest


def candidate_technical_check(row: dict, case: dict) -> tuple[bool, list[str]]:
    """Independently verify literal provenance, never the meaning of a quote."""
    problems = []
    if row.get("status") != "ok":
        return False, ["recorded_status_is_not_ok"]
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > 16:
        return False, ["missing_or_oversized_candidate_array"]
    sources = {source["source_id"]: source for source in case["sources"]}
    seen = set()
    for index, candidate in enumerate(candidates):
        label = f"candidate_{index}"
        required = {"operation", "interest", "source_id", "quote"}
        if not isinstance(candidate, dict) or not required <= set(candidate) <= required | {"candidate_id"}:
            problems.append(label + "_invalid_fields")
            continue
        if any(not isinstance(candidate[key], str) or not candidate[key].strip() for key in required):
            problems.append(label + "_invalid_string")
            continue
        operation, interest, source_id, quote = (candidate[key] for key in ("operation", "interest", "source_id", "quote"))
        if (operation not in OPERATIONS or len(interest) > 120 or interest != interest.strip()
                or any(ord(char) < 32 for char in interest) or len(quote) > 2000):
            problems.append(label + "_invalid_operation_or_length")
        source = sources.get(source_id)
        if not source or source.get("author_role") != "student" or quote not in source.get("content", ""):
            problems.append(label + "_invalid_literal_provenance")
        identity = operation, interest, source_id, quote
        if identity in seen:
            problems.append(label + "_duplicate_candidate")
        seen.add(identity)
    return not problems, problems


def ordered_check(candidates: list[dict], expected: list[dict] | None, case: dict, aliases: dict | None) -> dict:
    if expected is None:
        return {"specified": False, "match": None}
    # Keep repetitions from distinct sources. Operations emitted from a single
    # source may commute under the fixed fixture rule; their array order is not
    # a semantic judgement. The application already applies sources in order.
    def items(rows):
        return [(str(item["source_id"]), *pair(item, aliases)) for item in rows]
    actual_items, expected_items = items(candidates), items(expected)
    source_order = {source["source_id"]: index for index, source in enumerate(case["sources"])}
    indices = [source_order[item[0]] for item in actual_items]
    chronological = indices == sorted(indices)
    return {"specified": True, "source_chronology_match": chronological,
            "operations_with_sources_match": Counter(actual_items) == Counter(expected_items),
            "match": chronological and Counter(actual_items) == Counter(expected_items),
            "actual": [{"source_id": source, "operation": operation, "interest": interest}
                       for source, operation, interest in actual_items]}


def preference_comparison(row: dict, case: dict, aliases: dict) -> dict:
    technical_ok, diagnostics = candidate_technical_check(row, case)
    result = {"recorded_status": row.get("status"), "strict_technical_ok": technical_ok,
              "technical_diagnostics": diagnostics, "semantic_assessment": "unknown_not_human_reviewed"}
    for label, mapping in (("literal", None), ("fixed_aliases", aliases)):
        expected = {pair(item, mapping) for item in case["expected_pairs"]}
        allowed = {pair(item, mapping) for item in case.get("allowed_additional_pairs", [])}
        details = {"expected_pairs": sorted(expected), "allowed_additional_pairs": sorted(allowed),
                   "actual_pairs": None, "exact_pair_set_match": None,
                   "within_fixed_pair_allowance": None, "missing_pairs": None,
                   "unexpected_pairs": None, "ordered_operations": {"specified": "expected_ordered_operations" in case, "match": None}}
        if technical_ok:
            actual = {pair(item, mapping) for item in row["candidates"]}
            details.update(actual_pairs=sorted(actual), exact_pair_set_match=actual == expected,
                           within_fixed_pair_allowance=expected <= actual <= expected | allowed,
                           missing_pairs=sorted(expected - actual), unexpected_pairs=sorted(actual - expected - allowed),
                           ordered_operations=ordered_check(row["candidates"], case.get("expected_ordered_operations"), case, mapping))
        result[label] = details
    return result


def answer_text(row: dict) -> str | None:
    if row.get("status") != "ok":
        return None
    response = row.get("response")
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) and content.strip() else None


def count_bools(values: list) -> dict:
    return {"denominator": len(values), "true": sum(value is True for value in values),
            "false": sum(value is False for value in values),
            "not_evaluable": sum(value is not True and value is not False for value in values)}


def summarize(rows: list[dict]) -> dict:
    summary = {"denominator": len(rows), "both_strict_technical_ok": sum(row["both_strict_technical_ok"] for row in rows)}
    for variant in ("baseline", "candidate"):
        values = [row[variant] for row in rows]
        summary[variant] = {"strict_technical": count_bools([value["strict_technical_ok"] for value in values])}
        for comparison in ("literal", "fixed_aliases"):
            summary[variant][comparison] = {
                "exact_pair_set_match": count_bools([value[comparison]["exact_pair_set_match"] for value in values]),
                "within_fixed_pair_allowance": count_bools([value[comparison]["within_fixed_pair_allowance"] for value in values]),
                "ordered_operations_match": count_bools([value[comparison]["ordered_operations"]["match"] for value in values]),
                "pair_allowance_and_order_match": count_bools([
                    (value[comparison]["within_fixed_pair_allowance"] and value[comparison]["ordered_operations"]["match"])
                    if value["strict_technical_ok"] else None for value in values]),
            }
    both = [row for row in rows if row["both_strict_technical_ok"]]
    summary["paired_fixed_alias_pair_allowance"] = {
        "all_cases_denominator": len(rows), "both_technical_denominator": len(both),
        "baseline_only": sum(row["baseline"]["fixed_aliases"]["within_fixed_pair_allowance"] is True
                             and row["candidate"]["fixed_aliases"]["within_fixed_pair_allowance"] is False for row in both),
        "candidate_only": sum(row["candidate"]["fixed_aliases"]["within_fixed_pair_allowance"] is True
                              and row["baseline"]["fixed_aliases"]["within_fixed_pair_allowance"] is False for row in both),
        "both_match": sum(all(row[variant]["fixed_aliases"]["within_fixed_pair_allowance"] is True
                              for variant in ("baseline", "candidate")) for row in both),
        "neither_match": sum(all(row[variant]["fixed_aliases"]["within_fixed_pair_allowance"] is False
                                 for variant in ("baseline", "candidate")) for row in both),
    }
    return summary


def build(args: argparse.Namespace) -> Path:
    roots = {"baseline": args.baseline_dir.resolve(), "candidate": args.candidate_dir.resolve()}
    out = args.output_dir.resolve()
    if roots["baseline"] == roots["candidate"]:
        raise ValueError("Two distinct completed packages are required")
    for root in roots.values():
        if out == root or root in out.parents:
            raise ValueError("The output must be outside both immutable input packages")
    if out.exists():
        raise FileExistsError("Existing output directories are never overwritten")
    packages = {label: load_package(root) for label, root in roots.items()}
    baseline, candidate = packages["baseline"], packages["candidate"]
    for field in ("interest_fixture", "teaching_fixture"):
        if baseline[field] != candidate[field]:
            raise ValueError("The two batches must use identical fixed fixtures")
    fixed_parameters = ("model_identity", "temperature", "seed", "think", "interest_output_tokens")
    for field in fixed_parameters:
        if (field not in baseline["preregistered"] or field not in candidate["preregistered"]
                or baseline["preregistered"][field] != candidate["preregistered"][field]):
            raise ValueError(f"Generation conditions must remain fixed: {field}")
    for field in ("chat_model", "embedding_model", "context_tokens", "max_output_tokens"):
        left = baseline["preregistered"].get("model_manifest", {}).get(field)
        right = candidate["preregistered"].get("model_manifest", {}).get(field)
        if left is None or left != right:
            raise ValueError(f"Generation manifest must remain fixed: {field}")
    interest_cases, teaching_cases = baseline["interest_cases"], baseline["teaching_cases"]
    if set(teaching_cases) != {f"T{index:02d}" for index in range(1, 13)}:
        raise ValueError("The blind review requires exactly T01 through T12")
    if Counter(case.get("split") for case in interest_cases.values()) != {"dev": 12, "heldout": 12}:
        raise ValueError("The preference comparison requires 12 dev and 12 heldout cases")
    aliases = baseline["interest_fixture"].get("canonical_aliases", {})
    machine_rows = []
    for case_id, case in interest_cases.items():
        row = {"case_id": case_id, "split": case["split"], "boundary_type": case.get("boundary_type")}
        for variant, package in packages.items():
            row[variant] = preference_comparison(package["rows"][case_id], case, aliases)
        row["both_strict_technical_ok"] = all(row[variant]["strict_technical_ok"] for variant in packages)
        machine_rows.append(row)
    summary = {
        "kind": "paired_fixed_preference_diagnostics_not_semantic_assessment",
        "data_origin": "synthetic_unreviewed", "human_reviewed": False,
        "semantic_assessment": "unknown_not_human_reviewed",
        "fixed_generation_conditions": {field: baseline["preregistered"][field] for field in fixed_parameters},
        "rules": {
            "denominator": "All fixed cases, including technical failures; unevaluable comparisons remain null.",
            "literal": "Exact operation and exact interest string; no alias normalization.",
            "fixed_aliases": "Only preregistered canonical_aliases, case folding and whitespace normalization; no semantic inference.",
            "order": "Exact operation/interest/source multiplicity and source chronology; same-source operations may permute.",
            "strict_technical": "Recorded status ok plus independently valid candidate fields and literal student provenance.",
            "semantics": "A literal quote does not establish semantic support. Mechanical matches are not an overall pass.",
        },
        "all": summarize(machine_rows),
        "dev": summarize([row for row in machine_rows if row["split"] == "dev"]),
        "heldout": summarize([row for row in machine_rows if row["split"] == "heldout"]),
        "cases": machine_rows,
    }
    blind_rows, mapping_rows = [], []
    for case_id in sorted(teaching_cases):
        case = teaching_cases[case_id]
        first = "candidate" if hashlib.sha256((BLIND_SEED + ":" + case_id).encode()).digest()[0] % 2 else "baseline"
        versions = (first, "candidate" if first == "baseline" else "baseline")
        public = {field: deepcopy(case[field]) for field in ("subject_id", "topic_id", "messages", "preference_context") if field in case}
        slots = []
        for slot, variant in zip(("A", "B"), versions):
            source_row = packages[variant]["rows"][case_id]
            answer = answer_text(source_row)
            slots.append({"slot": slot, "technical_status": "ok" if answer is not None else "failed", "answer": answer})
        blind_rows.append({"case_id": case_id, "public_input": public,
                           "private_gold": deepcopy(case["private_gold"]), "slots": slots})
        mapping_rows.append({"case_id": case_id, "slots": dict(zip(("A", "B"), versions))})
    instructions = {
        "review_type": "uncalibrated_ai_review_or_human_review_record_status_separately",
        "task": "Сравните оба ответа по фактической правильности, соблюдению просьбы ученика, полезности и сохранению условий помощи. Укажите конкретные утверждения и основания; допускается ничья.",
        "limitations": "Ожидания синтетические и человеком не проверены. Не считайте один верный пример доказательством устойчивого освоения. Технический отказ остаётся отдельным исходом.",
        "slot_order": "Порядок A/B зафиксирован независимо для каждого случая; названия вариантов скрыты.",
        "code": "Не исполняйте код из ответов модели. Оценивайте его как текст или проверяйте вручную.",
    }
    documents = {
        "blind_teaching_T01_T06.json": {"instructions": instructions, "cases": blind_rows[:6]},
        "blind_teaching_T07_T12.json": {"instructions": instructions, "cases": blind_rows[6:]},
        "paired_machine_summary.json": summary,
        "restricted_mapping.json": {
            "restricted": True, "instruction": "Открыть после фиксации слепых обзоров, чтобы раскрыть варианты.",
            "seed": BLIND_SEED, "algorithm": "sha256(seed + ':' + case_id)[0] modulo 2 chooses A",
            "inputs": {label: str(root) for label, root in roots.items()}, "cases": mapping_rows,
        },
        "input_digests.json": {
            "kind": "independently_computed_sha256", "created_at": datetime.now(timezone.utc).isoformat(),
            "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "packages": {label: {"directory": str(package["root"]), "sha256": package["hashes"]}
                         for label, package in packages.items()},
            "raw_http_trace_policy": "Raw HTTP traces are not consumed by this formatter; report, preregistration, copied fixtures and every case file are hashed.",
            "display_typography": "Generated review files replace em dash with en dash. Original input artifacts remain unchanged.",
        },
    }
    # Verify again before creating anything; a running batch is not an immutable
    # input and a changed receipt must never be silently accepted.
    for package in packages.values():
        for name, digest in package["hashes"].items():
            if hashlib.sha256((package["root"] / name).read_bytes()).hexdigest() != digest:
                raise ValueError("An input artifact changed while preparing the review")
    out.mkdir(parents=True, exist_ok=False)
    for name, document in documents.items():
        text = json.dumps(document, ensure_ascii=False, indent=2).replace("\u2014", "–")
        (out / name).write_text(text + "\n", encoding="utf-8")
    # Hash source artifacts again after writing; output files never replace them.
    for package in packages.values():
        for name, digest in package["hashes"].items():
            if hashlib.sha256((package["root"] / name).read_bytes()).hexdigest() != digest:
                raise ValueError("An input artifact changed during review preparation")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    out = build(parser.parse_args())
    print(json.dumps({"status": "prepared", "output_dir": str(out), "interest_cases": 24,
                      "teaching_cases": 12, "model_calls": 0, "human_reviewed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
