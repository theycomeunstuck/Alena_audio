"""Read-only research reporting, kept outside the frozen research implementation.

This command never calls a model or changes a batch. Missing and failed answers
remain in the planned denominator. Machine labels require a qualified, real
workflow; no inference in this script establishes educational truth.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sqlite3
from statistics import mean


METHODS = ("profile_only", "tail", "summary", "structured")
AXES = ("factual_accuracy", "help_attribution", "action_appropriateness",
        "source_support", "unknowns_handling")
VERDICTS = ("acceptable", "critical_error", "uncertain")
UTILITIES = ("useful", "limited", "uncertain")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    values = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            values.append(row)
    return values


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique(rows, key):
    result = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"Invalid or duplicate {key}: {value!r}")
        result[value] = row
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_rows(path, values):
    Path(path).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")


class Batch:
    def __init__(self, directory):
        self.path = Path(directory).resolve()
        self.definition = read_json(self.path / "batch.json")
        self.histories = unique(read_rows(self.path / "snapshots/histories.jsonl"), "history_id")
        self.requests = unique(read_rows(self.path / "snapshots/requests.jsonl"), "request_id")
        self.input_paths = [self.path / p for p in
                            ("batch.json", "snapshots/histories.jsonl", "snapshots/requests.jsonl")]
        for filename, expected in self.definition.get("snapshot_sha256", {}).items():
            if sha(self.path / "snapshots" / filename) != expected:
                raise ValueError(f"Frozen snapshot hash mismatch: {filename}")
        actual = unique(read_rows(self.path / "results.jsonl"), "result_id")
        self.input_paths.append(self.path / "results.jsonl")
        self.rows = []
        methods = self.definition["methods"]
        if len(set(methods)) != len(methods) or set(methods) - set(METHODS):
            raise ValueError("Invalid batch methods")
        for history_id in self.definition["history_ids"]:
            if history_id not in self.histories:
                raise ValueError("Batch refers to missing history")
            for request in self.requests.values():
                if request["history_id"] != history_id:
                    continue
                for method in methods:
                    identity = f"{history_id}:{request['request_id']}:{method}"
                    row = actual.pop(identity, None)
                    if row is None:
                        row = {"result_id": identity, "history_id": history_id,
                               "request_id": request["request_id"], "method": method,
                               "request_text": request["text"], "status": "missing_result",
                               "action": None}
                    if any(row.get(k) != v for k, v in
                           (("history_id", history_id), ("request_id", request["request_id"]), ("method", method))):
                        raise ValueError(f"Result identity mismatch: {identity}")
                    self.rows.append(row)
        if actual:
            raise ValueError("Results include unplanned identities")
        self.by_id = unique(self.rows, "result_id")
        self.manifest = {}
        if (self.path / "batch_manifest.json").exists():
            self.manifest = read_json(self.path / "batch_manifest.json")
            self.input_paths.append(self.path / "batch_manifest.json")


def dev_comparison(before, after):
    if set(before.by_id) != set(after.by_id):
        raise ValueError("Dev runs have different planned identities")
    rows = []
    for identity, new in after.by_id.items():
        old = before.by_id[identity]
        old_ok, new_ok = old.get("status") == "ok", new.get("status") == "ok"
        transition = ("unchanged_ok" if old_ok and new_ok else "new_failure" if old_ok
                      else "recovered" if new_ok else "failure_in_both")
        rows.append({"result_id": identity, **{k: new[k] for k in ("history_id", "request_id", "method")},
                     "transition": transition, "old_status": old.get("status"),
                     "new_status": new.get("status"), "old_error": old.get("error"),
                     "new_error": new.get("error"), "old_memory_errors": old.get("memory_errors", []),
                     "new_memory_errors": new.get("memory_errors", []),
                     "old_action": old.get("action"), "new_action": new.get("action"),
                     "semantic_status": "human_unreviewed"})
    return rows


def workflow_labels(directory, batch):
    """Do not open the private mapping before checking the calibration gate."""
    empty = {"status": "not_provided", "calibration_qualified": False, "semantic_labels_used": False}
    if directory is None:
        return {}, empty, []
    directory = Path(directory).resolve()
    definition = read_json(directory / "workflow.json")
    report = read_json(directory / "report.json")
    calibration_path = directory / "calibration/summary.json"
    calibration = read_json(calibration_path) if calibration_path.exists() else {}
    paths = [directory / "workflow.json", directory / "report.json"]
    if calibration_path.exists():
        paths.append(calibration_path)
    if Path(definition["batch_dir"]).resolve() != batch.path:
        raise ValueError("Workflow belongs to another batch")
    if definition.get("sha256", {}).get("batch_definition") not in {None, sha(batch.path / "batch.json")}:
        raise ValueError("Workflow batch hash mismatch")
    gate = {"status": report.get("status"), "calibration_qualified": report.get("calibration_qualified") is True,
            "calibration_summary": calibration, "semantic_labels_used": False,
            "provider_accounting": report.get("provider_accounting"),
            "judge_protocol": definition.get("judge_config", {}).get("protocol", "legacy")}
    eligible = (str(report.get("status", "")).startswith("completed")
                and report.get("calibration_qualified") is True and calibration.get("qualified") is True
                and report.get("contract_only") is not True
                and definition.get("judge_config", {}).get("backend") != "fixture"
                and gate["judge_protocol"] == "context-v3"
                and batch.definition.get("backend") != "fixture")
    if not eligible:
        gate["reason"] = "Semantic assessment stopped or unavailable; all planned answers remain not_assessed."
        return {}, gate, paths
    prepared = directory / "prepared"
    manifest = read_json(prepared / "manifest.json")
    if (manifest.get("input_sha256", {}).get("results") != sha(batch.path / "results.jsonl")
            or manifest.get("input_sha256", {}).get("histories") != sha(batch.path / "snapshots/histories.jsonl")
            or manifest.get("cases_sha256") != sha(prepared / "cases.jsonl")):
        raise ValueError("Prepared assessment provenance mismatch")
    cases = unique(read_rows(prepared / "cases.jsonl"), "answer_code")
    mapping = read_json(prepared / "mapping.private.json")
    judgements = unique(read_rows(directory / "answers/judgements.jsonl"), "answer_code")
    if set(cases) != set(mapping) or set(judgements) - set(cases):
        raise ValueError("Assessment mapping/case coverage mismatch")
    if {link.get("result_id") for link in mapping.values()} != set(batch.by_id):
        raise ValueError("Assessment mapping does not cover planned results")
    labels = {}
    for code, link in mapping.items():
        source = batch.by_id[link["result_id"]]
        if any(link.get(k) != source.get(k) for k in ("method", "history_id", "request_id")):
            raise ValueError("Assessment mapping identity mismatch")
        case, answer = cases[code], judgements.get(code, {})
        if (case.get("execution_status") != source.get("status") or case.get("contract_only") is True
                or case.get("payload", {}).get("answer") != source.get("action")):
            raise ValueError("Assessment case/result mismatch")
        if answer.get("status") != "assessed":
            labels[source["result_id"]] = {"answer_code": code, "status": answer.get("status", "missing_judgement"),
                                           "judgement": None, "error": answer.get("error")}
            continue
        judgement = answer.get("judgement")
        if (source.get("status") != "ok" or not isinstance(judgement, dict)
                or judgement.get("verdict") not in VERDICTS or judgement.get("utility") not in UTILITIES
                or set(judgement.get("scores", {})) != set(AXES)
                or any(v is not None and (type(v) is not int or v not in {0, 1, 2})
                       for v in judgement["scores"].values())):
            raise ValueError("Invalid assessed judgement")
        labels[source["result_id"]] = {"answer_code": code, "status": "assessed", "judgement": judgement}
    gate["semantic_labels_used"] = any(label["status"] == "assessed" for label in labels.values())
    paths += [prepared / name for name in ("manifest.json", "cases.jsonl", "mapping.private.json")]
    paths.append(directory / "answers/judgements.jsonl")
    return labels, gate, paths


def method_counts(rows):
    counts = {}
    for method in METHODS:
        selected = [x for x in rows if x["method"] == method]
        assessed = [x for x in selected if x["assessment_status"] == "assessed"]
        axes = {}
        for axis in AXES:
            values = [x["judgement"]["scores"][axis] for x in assessed]
            axes[axis] = {"distribution": dict(Counter("null" if v is None else str(v) for v in values)),
                          "scored": sum(v is not None for v in values), "not_assessed": len(selected) - len(values)}
        counts[method] = {"planned": len(selected), "technical_ok": sum(x["status"] == "ok" for x in selected),
                          "technical_statuses": dict(Counter(x["status"] for x in selected)),
                          "assessed": len(assessed), "not_assessed": len(selected) - len(assessed),
                          "factual_verdict": {v: sum(x["judgement"]["verdict"] == v for x in assessed) for v in VERDICTS},
                          "utility": {v: sum(x["judgement"]["utility"] == v for x in assessed) for v in UTILITIES},
                          "score_axes": axes}
    return counts


def paired_history_analysis(rows):
    """Aggregate requests within histories before comparing methods."""
    histories = sorted({x["history_id"] for x in rows})
    histories_out = []
    for history in histories:
        for method in METHODS:
            subset = [x for x in rows if x["history_id"] == history and x["method"] == method]
            if not subset:
                continue
            assessed = [x for x in subset if x["assessment_status"] == "assessed"]
            full = len(assessed) == len(subset)
            histories_out.append({"history_id": history, "method": method, "planned": len(subset),
                "technical_ok": sum(x["status"] == "ok" for x in subset), "assessed": len(assessed),
                "complete_assessment": full,
                "critical_fraction": mean(x["judgement"]["verdict"] == "critical_error" for x in assessed) if full else None,
                "useful_fraction": mean(x["judgement"]["utility"] == "useful" for x in assessed) if full else None})
    index = {(x["history_id"], x["method"]): x for x in histories_out}
    pairs = []
    for first, second in combinations(METHODS, 2):
        available = [(index[(h, first)], index[(h, second)]) for h in histories
                     if (h, first) in index and (h, second) in index]
        for metric in ("technical_fraction", "critical_fraction", "useful_fraction"):
            deltas = []
            for a, b in available:
                av = a["technical_ok"] / a["planned"] if metric == "technical_fraction" else a[metric]
                bv = b["technical_ok"] / b["planned"] if metric == "technical_fraction" else b[metric]
                if av is not None and bv is not None:
                    deltas.append({"history_id": a["history_id"], "first_minus_second": av - bv})
            pairs.append({"first": first, "second": second, "metric": metric,
                          "paired_histories": len(deltas), "planned_shared_histories": len(available),
                          "mean_history_delta": mean(x["first_minus_second"] for x in deltas) if deltas else None,
                          "deltas": deltas})
    return {"unit": "history", "independent_observation_claim": False,
            "note": "Requests of one history are dependent; short/long variants may also share a scenario. No significance test or winner is declared.",
            "history_method": histories_out, "pairs": pairs}


def generation_usage(batch):
    """Call logs describe the active attempts; durable call ledger detects gaps."""
    jobs = batch.manifest.get("jobs", [])
    paths = []
    groups = {(method, stage): [] for method in batch.definition["methods"]
              for stage in ("memory_update", "answer")}
    seen = set()
    for job in jobs:
        if not job.get("artifact_path"):
            continue
        artifact = (batch.path / job["artifact_path"]).resolve()
        if not artifact.is_relative_to(batch.path):
            raise ValueError("Artifact path escapes batch")
        calls_path, trace_path = artifact / "calls.jsonl", artifact / "trace.jsonl"
        if not calls_path.exists():
            continue
        paths.append(calls_path)
        traces = {}
        if trace_path.exists():
            paths.append(trace_path)
            for trace in read_rows(trace_path):
                if trace.get("call_id") in traces:
                    raise ValueError("Duplicate trace call ID")
                traces[trace.get("call_id")] = trace
        for call in read_rows(calls_path):
            identity = (job.get("id"), call.get("call_id"))
            if identity in seen:
                raise ValueError("Duplicate recorded call ID")
            seen.add(identity)
            stage = "answer" if call.get("purpose") == "action" else "memory_update"
            group = groups.setdefault((job["method"], stage), [])
            trace = traces.get(call.get("call_id"), {})
            group.append({**call, "trace_available": bool(trace)})
    metrics = []
    for (method, stage), calls in sorted(groups.items()):
        performed = [x for x in calls if x.get("performed") is True]
        def known_sum(field, nested=None):
            values = [(x.get(nested) or {}).get(field) if nested else x.get(field) for x in performed]
            valid = [v for v in values if type(v) in {int, float} and v >= 0]
            return {"known_sum": sum(valid), "known_calls": len(valid), "unknown_calls": len(values) - len(valid)}
        combined_input = [x["input_chars"] + x["response_schema_chars"] for x in performed
                          if type(x.get("input_chars")) is int and type(x.get("response_schema_chars")) is int]
        metrics.append({"method": method, "stage": stage, "recorded_calls": len(calls),
            "performed_calls": len(performed), "failed_calls": sum(bool(x.get("error")) for x in performed),
            "input_tokens": known_sum("input_tokens", "usage"),
            "output_tokens": known_sum("output_tokens", "usage"),
            "elapsed_seconds": known_sum("elapsed_seconds"),
            "message_content_chars": known_sum("input_chars"),
            "response_schema_chars": known_sum("response_schema_chars"),
            "message_content_plus_schema_chars": {"known_sum": sum(combined_input),
                "known_calls": len(combined_input), "unknown_calls": len(performed) - len(combined_input)},
            "response_chars": known_sum("response_chars"),
            "trace_missing_calls": sum(not x["trace_available"] for x in performed),
            "monetary_cost_usd": None})
    ledger = None
    if (batch.path / "queue.sqlite3").exists():
        uri = (batch.path / "queue.sqlite3").as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            records = connection.execute("SELECT status,completion FROM calls").fetchall()
        totals = {"calls": len(records), "states": dict(Counter(x[0] for x in records)),
                  "known_input_tokens": 0, "known_output_tokens": 0, "incomplete_usage_calls": 0}
        for _, completion in records:
            usage = (json.loads(completion).get("usage") or {}) if completion else {}
            incomplete = False
            for key in ("input_tokens", "output_tokens"):
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    totals["known_" + key] += value
                else:
                    incomplete = True
            totals["incomplete_usage_calls"] += incomplete
        ledger = totals
    return {"active_attempts": metrics, "durable_call_ledger": ledger,
        "reported_batch_usage": batch.manifest.get("usage"),
        "notes": ["Update costs belong to a history/method and are shared by its requests; they are not multiplied per answer.",
                  "Input tokens are reported by the model for the full call. Separate schema tokenization is unavailable.",
                  "Message-content and schema character counts are separate; transport JSON, roles and tokenizer overhead are not measured in characters.",
                  "Active-attempt logs may include cached calls after a restart. The durable ledger is authoritative for actual call count and usage; stage subtotals are log observations.",
                  "Local electricity, hardware amortization and monetary cost are not measured; null does not mean zero."]}, paths


def failure_stages(rows):
    failures = []
    for row in rows:
        if row["status"] != "ok":
            stage = ("memory_update" if row["status"] == "memory_error" else
                     "answer" if row["status"] == "action_error" else "orchestration_or_unexecuted")
            failures.append({"result_id": row["result_id"], "observed_stage": stage,
                             "status": row["status"], "error": row.get("error"),
                             "memory_errors": row.get("memory_errors", []),
                             "job_error": row.get("job_error")})
    return {"technical_failures": failures,
            "selection_events": [{"result_id": x["result_id"], "events": x["memory_events"]}
                                 for x in rows if x.get("memory_events")],
            "semantic_causal_attribution": "not_established",
            "note": "Technical stage follows recorded status. Selection events describe budget handling, not proof of semantic error. Judge findings remain machine allegations; locating a semantic error in writing, selection, answer or judging requires source/trace review."}


def assessment_usage(directory):
    if directory is None:
        return {"status": "not_provided"}, []
    directory = Path(directory)
    paths, groups = [], {}
    for stage in ("calibration", "answers"):
        calls = []
        for path in sorted((directory / "steps" / stage).glob("*/output/calls.jsonl")):
            paths.append(path)
            calls += read_rows(path)
            trace = path.with_name("trace.jsonl")
            if trace.exists():
                paths.append(trace)
        performed = [x for x in calls if x.get("performed") is True]
        summary = {"recorded_calls": len(calls), "performed_calls": len(performed),
                   "failed_calls": sum(bool(x.get("error")) for x in performed)}
        for field, nested in (("input_tokens", "usage"), ("output_tokens", "usage"),
                              ("elapsed_seconds", None), ("input_chars", None),
                              ("response_schema_chars", None)):
            values = [(x.get(nested) or {}).get(field) if nested else x.get(field) for x in performed]
            valid = [v for v in values if type(v) in {int, float} and v >= 0]
            summary[field] = {"known_sum": sum(valid), "known_calls": len(valid),
                              "unknown_calls": len(values) - len(valid)}
        groups[stage] = summary
    return {"status": "recorded_logs", "groups": groups,
            "note": "These are current-workflow step logs. Reused calibration, unknown interrupted calls and earlier spending are described by workflow accounting and the budget receipt, not silently added to these subtotals."}, paths


def review_packet(batch, rows, selection_path, gold, output):
    if selection_path is None:
        return {"status": "not_created", "reason": "No preregistered selection supplied."}, []
    path = Path(selection_path)
    selection = read_json(path)
    selected = selection.get("cases", selection.get("selections"))
    if selected is None and isinstance(selection.get("requests"), list):
        selected = []
        for request_id in selection["requests"]:
            request = batch.requests.get(request_id)
            if request is None:
                raise ValueError("Unknown preselected review request")
            selected.append({"history_id": request["history_id"], "request_id": request_id})
    if not isinstance(selected, list) or len(selected) != 5:
        raise ValueError("Review selection must contain five cases")
    if len({x.get("history_id") for x in selected}) != 5:
        raise ValueError("Review selection needs five distinct histories")
    # Timestamp/order claims are not inferred from filesystem modification times.
    selection_hash = sha(path)
    public, private = output / "human_review", output / "private"
    public.mkdir()
    private.mkdir()
    codes, histories, mapping, disputes = [], [], {}, []
    by_id = {x["result_id"]: x for x in rows}
    for number, item in enumerate(selected, 1):
        history_id, request_id = item["history_id"], item["request_id"]
        request = batch.requests.get(request_id)
        if request is None or request["history_id"] != history_id:
            raise ValueError("Invalid preselected review request")
        history_code = f"H{number:02d}"
        history = batch.histories[history_id]
        histories.append({"history_code": history_code, "source_history": history,
                          "question": request["text"], "human_status": "not_reviewed"})
        selected_rows = [by_id[f"{history_id}:{request_id}:{method}"] for method in METHODS]
        selected_rows.sort(key=lambda x: hashlib.sha256((selection_hash + x["result_id"]).encode()).digest())
        for ordinal, row in enumerate(selected_rows, 1):
            code = f"R{number:02d}{ordinal}"
            mapping[code] = {k: row[k] for k in ("result_id", "history_id", "request_id", "method")}
            codes.append({"answer_code": code, "history_code": history_code,
                          "question": request["text"], "answer": row.get("action"),
                          "source_turns": row.get("source_turns", []), "execution_status": row["status"],
                          "human_review": {"factual_verdict": "", "utility": "",
                                           "scores": {axis: "" for axis in AXES}, "comment": ""}})
    for row in rows:
        judgement = row.get("judgement")
        judge_status = row.get("judge_status")
        reason = ("generation_failure" if row["status"] != "ok" else
                  "judge_error" if judge_status and (judge_status.startswith("judge_error")
                                                        or judge_status == "missing_judgement") else
                  "machine_semantic_flag" if judgement and judgement["verdict"] in
                      {"critical_error", "uncertain"} else None)
        if reason:
            disputes.append({"result_id": row["result_id"], "reason": reason, "judgement": judgement,
                             "action": row.get("action"), "memory_context": row.get("memory_context"),
                             "execution_status": row["status"], "execution_error": row.get("error"),
                             "memory_errors": row.get("memory_errors", []), "judge_error": row.get("judge_error"),
                             "source_history": batch.histories[row["history_id"]],
                             "status": "human_unreviewed"})
    write_rows(public / "histories.jsonl", histories)
    write_rows(public / "answers.jsonl", codes)
    write_json(private / "mapping.private.json", mapping)
    write_rows(private / "disputed_machine_cases.jsonl", disputes)
    packet_lines = ["# Пакет для независимого просмотра", "",
        "Источники, ответы и предварительная рубрика содержательно не проверены человеком. "
        "Оцените существенные фактические искажения, полезность и каждую из пяти осей отдельно. "
        "Названия методов раскрываются после заполнения оценок.", ""]
    for history in histories:
        packet_lines += [f"## {history['history_code']}", "", history["question"], "",
                         "### Источники", "", "```json",
                         json.dumps(history["source_history"], ensure_ascii=False, indent=2), "```", ""]
        for answer in (x for x in codes if x["history_code"] == history["history_code"]):
            packet_lines += [f"### {answer['answer_code']}", "", f"Выполнение: `{answer['execution_status']}`.", "",
                             "```json", json.dumps(answer["answer"], ensure_ascii=False, indent=2), "```", "",
                             "Оценка: фактический вердикт ___; полезность ___; factual_accuracy ___; "
                             "help_attribution ___; action_appropriateness ___; source_support ___; "
                             "unknowns_handling ___; комментарий ___.", ""]
    (public / "packet.md").write_text("\n".join(packet_lines), encoding="utf-8")
    if gold:
        write_rows(private / "reference_rubric.human_unreviewed.jsonl", gold)
    (public / "README.md").write_text(
        "# Просмотр преподавателем\n\n"
        "Пять заранее выбранных историй, по одному вопросу и четыре ответа с закрытыми названиями методов. "
        "Сначала прочитайте histories.jsonl, затем заполните human_review в answers.jsonl. "
        "Данные, рубрика и машинные оценки содержательно не проверены человеком. "
        "Отдельно оцените существенные искажения, полезность и пять осей; баллы не суммируются. "
        "review_required означает просмотр преподавателем; пустой unknowns сам по себе допустим. "
        "Один успех не доказывает устойчивое освоение.\n\n"
        "Каталог private содержит раскрытие методов, предварительную рубрику и спорные машинные случаи. "
        "Откройте его после независимого заполнения основного пакета.\n", encoding="utf-8")
    return {"status": "created", "selection_sha256": selection_hash,
            "preregistration_claim": selection.get("status", selection.get("selection_status",
                                                selection.get("selection_origin", "not_recorded"))),
            "histories": len(histories), "answers": len(codes), "disputed_machine_cases": len(disputes),
            "mapping_path": "private/mapping.private.json"}, [path]


def render_markdown(report, dev_rows):
    heldout = report["heldout"]
    lines = ["# Итоговый пакет исследования памяти", "",
        "Данные, предварительная рубрика и машинные оценки содержательно не проверены человеком. "
        "Технически принятый ответ не равен содержательно правильному. Автоматический победитель не назначается.", "",
        f"Dev – {report['dev']['planned']} пар строк; переходы: {json.dumps(report['dev']['transitions'], ensure_ascii=False)}.",
        f"Отложенный набор – {heldout['planned']} плановых ответов, {heldout['distinct_histories']} историй. "
        "Ошибки и отсутствующие ответы включены в знаменатель.", "",
        f"Состояние судьи: `{report['assessment']['status']}`; калибровка: "
        f"`{report['assessment']['calibration_qualified']}`; машинные метки использованы: "
        f"`{report['assessment']['semantic_labels_used']}`.", "",
        "## Покрытие и независимые показатели", "",
        "| Метод | План | Технически принято | Оценено | Не оценено | Существенное искажение | Без существенного искажения | Неоднозначный вердикт | Полезно | Ограниченно полезно | Неоднозначная полезность |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, group in heldout["methods"].items():
        values = [group[k] for k in ("planned", "technical_ok", "assessed", "not_assessed")]
        values += [group["factual_verdict"][k] for k in ("critical_error", "acceptable", "uncertain")]
        values += [group["utility"][k] for k in UTILITIES]
        lines.append(f"| {method} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "Нули в колонках оценок при отсутствии калибровки означают отсутствие оценённых ответов. "
              "Существенные искажения и полезность – отдельные машинные метки. Пять осей не суммируются.", "",
              "| Метод | Ось | 0 | 1 | 2 | null | Не оценено |", "|---|---|---:|---:|---:|---:|---:|"]
    for method, group in heldout["methods"].items():
        for axis, scores in group["score_axes"].items():
            distribution = scores["distribution"]
            lines.append(f"| {method} | {axis} | " + " | ".join(str(distribution.get(k, 0)) for k in
                         ("0", "1", "2", "null")) + f" | {scores['not_assessed']} |")
    lines += ["", "## Все пары dev", "", "| Строка | Раньше | Сейчас | Переход |", "|---|---|---|---|"]
    lines += [f"| {r['result_id']} | {r['old_status']} | {r['new_status']} | {r['transition']} |" for r in dev_rows]
    lines += ["", "## Сопоставление внутри историй", "",
              "Файл report.json содержит статусы для каждой истории и метода, затем попарные разности долей. "
              "Два запроса одной истории зависимы; короткая и длинная версии могут разделять сценарий. "
              "Семантические пары рассчитаны только для полностью оценённых историй обоих методов. "
              "128 ответов не трактуются как 128 независимых наблюдений. Проверка значимости не проводилась.", "",
              "## Ресурсы и происхождение ошибок", "",
              "В report.json отдельно сохранены вызовы обновления памяти и финального ответа: полные входные токены, "
              "выходные токены, время, символы содержимого сообщений и символы схем. "
              "Отдельная токенизация схем отсутствует; стоимость локального оборудования и электричества не измерена. "
              "Обновление оплачивается один раз на историю и метод, затем используется двумя запросами. "
              "Денежные расходы судьи сохранены отдельно из provider_accounting; неизвестное не заменено нулём.", "",
              "| Метод | Стадия | Вызовы в журнале | Входные токены | Выходные токены | Время, с | Символы сообщений | Символы схемы |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for resource in report["generation_resources"]["heldout"]["active_attempts"]:
        values = [resource["performed_calls"], resource["input_tokens"]["known_sum"],
                  resource["output_tokens"]["known_sum"], round(resource["elapsed_seconds"]["known_sum"], 3),
                  resource["message_content_chars"]["known_sum"], resource["response_schema_chars"]["known_sum"]]
        lines.append(f"| {resource['method']} | {resource['stage']} | " + " | ".join(map(str, values)) + " |")
    lines += ["", "Таблица содержит известные суммы из журналов активных попыток; доли неизвестных значений "
              "сохранены в JSON. При перезапуске возможны записи использования кеша; итог числа реальных вызовов "
              "и токенов берётся из устойчивого журнала вызовов. Нулевые суммы при отсутствии журналов "
              "не являются подтверждённым нулевым расходом.", "",
              "Техническая стадия сбоя взята из записанного статуса. События отбора описывают бюджет памяти. "
              "Содержательная причинность по стадиям записи, отбора, ответа и оценки автоматически не установлена; "
              "для неё нужны первичные источники и trace. Доступные trace учтены в контрольных суммах.", "",
              "## Просмотр человеком", "",
              f"Пакет: `{report['human_review']['status']}`. Названия методов скрыты в human_review; "
              "раскрытие и спорные машинные случаи находятся отдельно в private.", "",
              "Повторный отчёт создаётся в новом каталоге. Входы и старые эксперименты не перезаписываются. "
              "PROCESSING.json подтверждает обработку артефактов, а не научную или педагогическую проверку."]
    return "\n".join(lines) + "\n"


def build_report(dev_before, dev_after, heldout, output, *, workflow=None, gold=None, review_selection=None,
                 budget_receipt=None):
    before, after, batch = (Batch(x) for x in (dev_before, dev_after, heldout))
    dev_rows = dev_comparison(before, after)
    labels, gate, workflow_paths = workflow_labels(workflow, batch)
    rows = [{**row, "assessment_status": "assessed" if labels.get(row["result_id"], {}).get("status") == "assessed" else "not_assessed",
             "judgement": labels.get(row["result_id"], {}).get("judgement"),
             "judge_answer_code": labels.get(row["result_id"], {}).get("answer_code"),
             "judge_status": labels.get(row["result_id"], {}).get("status", gate["status"]),
             "judge_error": labels.get(row["result_id"], {}).get("error"),
             "human_review_status": "not_reviewed"} for row in batch.rows]
    gold_rows = read_rows(gold) if gold else []
    if gold_rows:
        unique(gold_rows, "request_id")
        expected = {x["request_id"] for x in rows}
        if {x["request_id"] for x in gold_rows} != expected:
            raise ValueError("Reference rubric must cover every planned heldout request")
    usage = {}
    source_paths = before.input_paths + after.input_paths + batch.input_paths + workflow_paths
    for name, source in (("dev_before", before), ("dev_after", after), ("heldout", batch)):
        usage[name], paths = generation_usage(source)
        source_paths += paths
    judge_usage, judge_paths = assessment_usage(workflow)
    source_paths += judge_paths
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    packet, packet_paths = review_packet(batch, rows, review_selection, gold_rows, output)
    source_paths += packet_paths
    if gold:
        source_paths.append(Path(gold).resolve())
    budget = read_json(budget_receipt) if budget_receipt else None
    if budget_receipt:
        source_paths.append(Path(budget_receipt).resolve())
    report = {"report_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "human_review_status": "not_reviewed", "assessment_origin": "machine_only_or_not_assessed",
        "winner": None, "scientific_validation_claim": False,
        "inputs": {"dev_before": str(before.path), "dev_after": str(after.path), "heldout": str(batch.path)},
        "dev": {"planned": len(dev_rows), "transitions": dict(Counter(x["transition"] for x in dev_rows)),
                "expected_protocol_denominator": 64, "protocol_denominator_matches": len(dev_rows) == 64,
                "new_failures": [x["result_id"] for x in dev_rows if x["transition"] == "new_failure"]},
        "heldout": {"planned": len(rows), "expected_protocol_denominator": 128,
                    "protocol_denominator_matches": len(rows) == 128,
                    "protocol_shape_matches": (len(batch.definition["history_ids"]) == 16
                        and set(batch.definition["methods"]) == set(METHODS)
                        and all(sum(q["history_id"] == history_id for q in batch.requests.values()) == 2
                                for history_id in batch.definition["history_ids"])),
                    "distinct_histories": len({x["history_id"] for x in rows}),
                    "resolved_config": batch.definition.get("resolved_config"),
                    "methods": method_counts(rows), "paired_analysis": paired_history_analysis(rows),
                    "failures_by_stage": failure_stages(rows)},
        "assessment": gate, "assessment_resources": judge_usage,
        "generation_resources": usage, "budget_receipt": budget,
        "human_review": packet,
        "reference_rubric": {"status": "human_unreviewed" if gold_rows else "not_supplied",
                             "request_count": len(gold_rows), "used_for_generation": False,
                             "used_for_automatic_semantic_labels": False},
        "limitations": ["Synthetic data; no effect on real learners was measured.",
                        "A single model and one judge cannot establish general superiority.",
                        "Calibration qualification is a limited gate, not proof that every subsequent judgement is correct.",
                        "No content, rubric or score was reviewed by a human in this report."]}
    write_rows(output / "dev_rows.jsonl", dev_rows)
    write_rows(output / "heldout_rows.jsonl", rows)
    write_json(output / "report.json", report)
    (output / "report.md").write_text(render_markdown(report, dev_rows), encoding="utf-8")
    unique_paths = sorted({Path(path).resolve() for path in source_paths}, key=str)
    write_json(output / "PROCESSING.json", {"status": "completed", "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(__file__), "human_review_status": "not_reviewed", "model_calls": 0,
        "input_sha256": {str(path): sha(path) for path in unique_paths},
        "output_sha256": {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*")) if path.is_file()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dev-before", "dev-after", "heldout", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("workflow", "gold", "review-selection", "budget-receipt"):
        parser.add_argument("--" + option, type=Path)
    args = parser.parse_args()
    report = build_report(args.dev_before, args.dev_after, args.heldout, args.output,
                          workflow=args.workflow, gold=args.gold, review_selection=args.review_selection,
                          budget_receipt=args.budget_receipt)
    print(json.dumps({"output": str(args.output.resolve()), "planned": report["heldout"]["planned"],
                      "assessment_status": report["assessment"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
