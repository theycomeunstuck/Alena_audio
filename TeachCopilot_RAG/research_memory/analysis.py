"""Machine-only comparison summaries for completed, qualified workflows."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

METHODS = ("profile_only", "tail", "summary", "structured")
VERDICTS = ("acceptable", "critical_error", "uncertain")
UTILITIES = ("useful", "limited", "uncertain")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(path):
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{Path(path).name}:{number}: expected object")
        result.append(row)
    return result


def _unique(rows, key, label):
    values = [row.get(key) for row in rows]
    if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"invalid or duplicate {label}")
    return {row[key]: row for row in rows}


def _write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _money(value, label):
    try:
        amount = Decimal(str(value))
    except Exception:
        raise ValueError(f"invalid {label}") from None
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"invalid {label}")
    return amount


def build_comparison(workflow_dir, output, *, budget_receipt_path=None) -> dict:
    """Aggregate a qualified workflow; the private mapping opens only after its gate."""
    root, output = Path(workflow_dir), Path(output)
    report = _json(root / "report.json")
    if (not str(report.get("status", "")).startswith("completed")
            or report.get("contract_only") is True
            or report.get("calibration_qualified") is not True):
        raise ValueError("workflow is not eligible for comparison")

    definition = _json(root / "workflow.json")
    configured_protocol = definition.get("judge_config", {}).get("protocol")
    reported_protocol = report.get("judge_protocol")
    if configured_protocol and reported_protocol and configured_protocol != reported_protocol:
        raise ValueError("workflow and report judge protocol mismatch")
    protocol = configured_protocol or reported_protocol or "legacy"
    context_protocol = protocol in {"context-v2", "context-v3"}
    batch_dir = Path(definition["batch_dir"])
    cases = _rows(root / "prepared" / "cases.jsonl")
    judgements = _rows(root / "answers" / "judgements.jsonl")
    results = _rows(batch_dir / "results.jsonl")
    cases_by_code = _unique(cases, "answer_code", "answer_code")
    judgements_by_code = _unique(judgements, "answer_code", "judgement answer_code")
    results_by_id = _unique(results, "result_id", "result_id")
    if set(cases_by_code) != set(judgements_by_code):
        raise ValueError("missing or extra judgement cases")
    if any(case.get("contract_only") is True for case in cases):
        raise ValueError("prepared cases include contract-only data")

    manifest = _json(root / "prepared" / "manifest.json")
    result_hash = manifest.get("input_sha256", {}).get("results")
    if result_hash:
        import hashlib
        actual = hashlib.sha256((batch_dir / "results.jsonl").read_bytes()).hexdigest()
        if actual != result_hash:
            raise ValueError("batch results hash mismatch")
    case_hash = manifest.get("cases_sha256")
    if case_hash:
        import hashlib
        if hashlib.sha256((root / "prepared" / "cases.jsonl").read_bytes()).hexdigest() != case_hash:
            raise ValueError("prepared cases hash mismatch")

    mapping = _json(root / "prepared" / "mapping.private.json")
    if set(mapping) != set(cases_by_code):
        raise ValueError("missing or extra mapping entries")
    if {entry.get("result_id") for entry in mapping.values()} != set(results_by_id):
        raise ValueError("missing or extra mapped results")

    counts = {method: {key: 0 for key in
              ("planned", "technical_ok", "generation_error", "assessed", *VERDICTS, "not_assessed")}
              for method in METHODS}
    if context_protocol:
        for group in counts.values():
            group["utility_counts"] = {label: 0 for label in UTILITIES}
    history_ids = set()
    critical_examples = []
    for code, case in cases_by_code.items():
        link = mapping[code]
        result_id = link.get("result_id")
        if result_id not in results_by_id:
            raise ValueError(f"missing result for {code}")
        source = results_by_id[result_id]
        if any(link.get(field) != source.get(field) for field in ("method", "history_id", "request_id")):
            raise ValueError(f"mapping/result mismatch for {code}")
        method = source.get("method")
        if method not in counts:
            raise ValueError(f"unknown method for {code}")
        history_ids.add(source.get("history_id"))
        group = counts[method]
        group["planned"] += 1
        technical_ok = source.get("status") == "ok" and case.get("execution_status") == "ok"
        group["technical_ok" if technical_ok else "generation_error"] += 1
        judgement = judgements_by_code[code]
        if context_protocol and judgement.get("status") == "assessed" and not isinstance(judgement.get("judgement"), dict):
            raise ValueError(f"invalid assessed judgement for {code}")
        if judgement.get("status") == "assessed" and isinstance(judgement.get("judgement"), dict):
            if not technical_ok:
                raise ValueError(f"assessed judgement for generation failure {code}")
            verdict = judgement["judgement"].get("verdict")
            if verdict not in VERDICTS:
                raise ValueError(f"invalid verdict for {code}")
            group["assessed"] += 1
            group[verdict] += 1
            if context_protocol:
                utility = judgement["judgement"].get("utility")
                if utility not in UTILITIES:
                    raise ValueError(f"invalid or missing context utility for {code}")
                group["utility_counts"][utility] += 1
            if verdict == "critical_error" and len(critical_examples) < 8:
                findings = judgement["judgement"].get("findings", [])
                finding = findings[0] if findings else {}
                critical_examples.append({"answer_code": code, "method": method,
                    "request_id": source.get("request_id"), "reason": finding.get("explanation", ""),
                    "evidence_ids": finding.get("evidence_turn_ids", [])})
        else:
            group["not_assessed"] += 1

    for method, group in counts.items():
        if sum(group[k] for k in (*VERDICTS, "not_assessed")) != group["planned"]:
            raise ValueError(f"verdict coverage mismatch for {method}")
        if group["assessed"] != sum(group[k] for k in VERDICTS):
            raise ValueError(f"assessed verdict mismatch for {method}")
        if context_protocol and sum(group["utility_counts"].values()) != group["assessed"]:
            raise ValueError(f"assessed utility mismatch for {method}")
        group["coverage_assessed_over_planned"] = f"{group['assessed']}/{group['planned']}"
        group["acceptable_over_planned"] = f"{group['acceptable']}/{group['planned']}"
        group["critical_error_over_assessed"] = f"{group['critical_error']}/{group['assessed']}"

    accounting = report.get("provider_accounting")
    charged = accounting.get("charged_or_reserved_usd") if isinstance(accounting, dict) else None
    current = _money(charged, "charged or reserved amount") if charged is not None else None
    budget = {"accounting_status": "available" if current is not None else "unavailable",
              "current_phase_charged_or_reserved_usd": str(current) if current is not None else None,
              "prior_stages": "not accounted"}
    if budget_receipt_path:
        receipt = _json(budget_receipt_path)
        cumulative = _money(receipt["cumulative_prior_charged_or_reserved_usd"], "prior cumulative amount")
        cap = _money(receipt["new_cap_usd"], "new phase cap")
        budget.update(cumulative_prior_charged_or_reserved_usd=str(cumulative),
                      new_phase_cap_usd=str(cap))
        budget["prior_stages"] = "accounted_from_receipt"
        if current is not None:
            budget["cumulative_charged_or_reserved_usd"] = str(cumulative + current)

    batch_definition = _json(batch_dir / "batch.json")
    split_note = (f"Split dev: {len(history_ids)} distinct histories; dev is not a final test set."
                  if batch_definition.get("split") == "dev" else
                  f"Split {batch_definition.get('split') or 'unknown'}: {len(history_ids)} distinct histories.")
    summary = {"scientific_result": False, "assessment_origin": "machine_only",
        "mapping_opened_after_assessment": True, "methods": counts,
        "distinct_history_count": len(history_ids), "split": batch_definition.get("split"),
        "dependent_requests_note": "Two requests from one history are dependent.",
        "critical_error_examples": critical_examples, "budget": budget,
        "limitations": [split_note, "One judge was used.", "No pedagogical expert review was performed."],
        "interpretation": "Fractions describe coverage and machine labels, not judge accuracy or learning effect.",
        "score_axes_combined": False}
    if context_protocol:
        summary["judge_protocol"] = protocol
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    split = batch_definition.get("split") or "unknown"
    intro = ("Оценка одного машинного судьи, без педагогической экспертизы. " if context_protocol else
             "Научный результат: нет. Один машинный судья, без педагогической экспертизы. ")
    lines = ["# Сравнение методов", "", intro
             + f"Данные split={split}, {len(history_ids)} разных histories. Запросы одной истории зависимы. "
             + ("Split dev не является финальным test. " if split == "dev" else ""), "",
             "Доли assessed/planned и acceptable/planned показывают покрытие и машинные метки, "
             "а critical_error/assessed – отдельную долю среди оценённых; это не точность судьи "
             "или учебный эффект. Оси score не суммировались.", "",
             ("| Метод | planned | technical_ok | generation_error | assessed | acceptable | critical_error | verdict_uncertain | not_assessed | useful | limited | utility_uncertain |"
              if context_protocol else
              "| Метод | planned | technical_ok | generation_error | assessed | acceptable | critical_error | uncertain | not_assessed |"),
             ("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
              if context_protocol else "|---|---:|---:|---:|---:|---:|---:|---:|---:|")]
    if context_protocol:
        lines.insert(4, "Вердикт и utility показаны отдельно; между собой не суммируются и не ранжируются.")
    fraction_lines = []
    for method, group in counts.items():
        values = [group[k] for k in ("planned", "technical_ok", "generation_error", "assessed", *VERDICTS, "not_assessed")]
        if context_protocol:
            utility_counts = group["utility_counts"]
            values.extend(utility_counts[label] for label in UTILITIES)
            values[-1] = utility_counts["uncertain"]
        lines.append("| " + method + " | " + " | ".join(str(value) for value in values) + " |")
        fraction_lines.append(f"Доли {method}: assessed/planned {group['coverage_assessed_over_planned']}; "
                              f"acceptable/planned {group['acceptable_over_planned']}; "
                              f"critical_error/assessed {group['critical_error_over_assessed']}.")
    lines += ["", *fraction_lines]
    lines += ["", "## Учёт расходов", ""]
    if budget["accounting_status"] == "unavailable":
        lines.append("Стоимость текущего этапа недоступна в provider_accounting.")
    else:
        lines.append(f"Текущий этап (charged or reserved): ${budget['current_phase_charged_or_reserved_usd']}.")
    if "cumulative_prior_charged_or_reserved_usd" in budget:
        lines.append(f"Предыдущие этапы: ${budget['cumulative_prior_charged_or_reserved_usd']}; "
                     f"новый лимит: ${budget['new_phase_cap_usd']}.")
        if "cumulative_charged_or_reserved_usd" in budget:
            lines.append(f"Накопительно charged or reserved: ${budget['cumulative_charged_or_reserved_usd']}.")
    else:
        lines.append("Расходы предыдущих этапов не учтены.")
    lines += ["", "## Примеры critical_error", ""]
    lines += [f"- `{x['answer_code']}` – {x['method']}, request `{x['request_id']}`: "
              f"{x['reason']} (evidence_ids: {', '.join(x['evidence_ids'])})" for x in critical_examples]
    _write_new(output / "report.md", "\n".join(lines) + "\n")
    return summary
