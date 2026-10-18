"""Read-only, deliberately narrow observations over two completed local batches.

No app imports, model calls, database access, eval, or exec. Numeric matches and
code blocks are observations, never a selection of the model's intended example.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / ".local/conversation_boundaries_v2"
DIVISION = re.compile(r"(?<![\w.,+\-−])(?P<a>[0-9]{1,12})\s*[:÷/]\s*(?P<b>[0-9]{1,12})(?!\d|[.,]\d)")
EQUALITY = re.compile(r"\s*=\s*(?P<q>[0-9]{1,12})(?!\d|[.,]\d)")
FENCE = re.compile(r"^\s*```[^\n]*\n(?P<code>.*?)^\s*```\s*$", re.MULTILINE | re.DOTALL)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("input symlinks are not supported")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def literal_print_stdout(code: str) -> str:
    """Interpret only direct print calls whose arguments are AST constants."""
    if len(code) > 20000:
        raise ValueError("block_too_large")
    tree = ast.parse(code)
    if not tree.body or len(tree.body) > 32:
        raise ValueError("expected_1_to_32_direct_print_calls")
    out = io.StringIO()
    for statement in tree.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print"):
            raise ValueError("not_literal_only_direct_print")
        if any(not isinstance(arg, ast.Constant) for arg in call.args):
            raise ValueError("nonconstant_argument")
        values = [arg.value for arg in call.args]
        if any(type(value) not in (str, int, float, bool, type(None)) for value in values):
            raise ValueError("unsupported_constant")
        options = {}
        for keyword in call.keywords:
            if keyword.arg not in {"sep", "end"} or keyword.arg in options or not isinstance(keyword.value, ast.Constant):
                raise ValueError("unsupported_print_keyword")
            value = keyword.value.value
            if value is not None and not isinstance(value, str):
                raise ValueError("separator_must_be_string_or_none")
            options[keyword.arg] = value
        if any(isinstance(value, str) and len(value) > 4096 for value in [*values, *options.values()]):
            raise ValueError("literal_too_large")
        print(*values, **options, file=out)
    return out.getvalue()


def divisions(text: str, constraint: dict) -> list[dict]:
    observations = []
    for match in DIVISION.finditer(text):
        a, b = int(match["a"]), int(match["b"])
        quotient = a // b if b and a % b == 0 else None
        digits = str(quotient) if quotient is not None else ""
        properties = {"integer_quotient": quotient is not None}
        if "quotient_digits" in constraint:
            properties["quotient_digits"] = len(digits) == constraint["quotient_digits"] if digits else None
        if "middle_digit" in constraint:
            properties["middle_digit"] = int(digits[1]) == constraint["middle_digit"] if len(digits) == 3 else None
        if "last_digit_nonzero" in constraint:
            properties["last_digit_nonzero"] = digits[-1] != "0" if digits else None
        if "divisor_digits" in constraint:
            properties["divisor_digits"] = len(str(b)) == constraint["divisor_digits"]
        properties["new_pair"] = {"dividend": a, "divisor": b} not in constraint.get("exclude_pairs", [])
        row = {"span": list(match.span()), "literal": match.group(), "dividend": a, "divisor": b,
               "computed_integer_quotient": quotient, "properties": properties,
               "numeric_property_status": "satisfied" if all(value is True for value in properties.values()) else "wrong"}
        equality = EQUALITY.match(text, match.end())
        if equality:
            stated = int(equality["q"])
            row["stated_quotient"] = stated
            row["stated_equality_status"] = "satisfied" if b and a == b * stated else "wrong"
        observations.append(row)
    return observations


def trusted_gold(cases: list[dict]) -> dict:
    records = []
    for case in cases:
        for check in case["private_gold"]["mechanical_checks"]:
            kind = check["kind"]
            if kind == "division_fact":
                assert check["divisor"] * check["quotient"] == check["dividend"], case["id"]
                records.append({"case_id": case["id"], "kind": kind, "verified": True})
            elif kind == "python_stdout":
                actual = literal_print_stdout(check["code"])
                assert actual == check["stdout"], case["id"]
                records.append({"case_id": case["id"], "kind": kind, "actual_stdout": actual, "verified": True})
            elif kind == "scratch_steps":
                assert check["command_steps"] * check["executions"] == check["total_steps"], case["id"]
                records.append({"case_id": case["id"], "kind": kind, "verified": True})
            elif kind == "proposed_division_property":
                examples = divisions("\n".join(case["private_gold"]["acceptable_behaviors"]), check)
                assert examples and all(item["numeric_property_status"] == "satisfied" for item in examples), case["id"]
                records.append({"case_id": case["id"], "kind": "allowed_numeric_example", "examples": examples, "verified": True})
    # Also verify explicit a:b=q equations in the trusted fixed dialogue/rubric.
    equations = []
    for case in cases:
        for row in divisions(json.dumps(case, ensure_ascii=False), {}):
            if "stated_equality_status" in row:
                assert row["stated_equality_status"] == "satisfied", case["id"]
                equations.append({"case_id": case["id"], "dividend": row["dividend"],
                                  "divisor": row["divisor"], "quotient": row["stated_quotient"]})
    return {"scope": "trusted_fixture_internal_consistency_only", "checks": records, "literal_equations": equations}


def answer_text(row: dict) -> str | None:
    if row.get("status") != "ok":
        return None
    try:
        value = row["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("successful case has no answer text") from None
    if not isinstance(value, str):
        raise ValueError("answer is not text")
    return value


def observe(case: dict, row: dict) -> dict:
    result = {"case_id": case["id"], "technical_status": row.get("status"), "semantic_verdict": "not_assessed"}
    text = answer_text(row)
    if text is None:
        return {**result, "status": "not_observed", "reason": "no_successful_answer"}
    check = case["private_gold"]["mechanical_checks"][0]
    if case["id"] in {"T03", "T04"}:
        matches = divisions(text, check)
        return {**result, "status": "observed" if matches else "not_observed", "reason": "all_symbolic_divisions_listed_without_selecting_proposed_example",
                "division_observations": matches}
    if case["id"] in {"T05", "T06"}:
        blocks = []
        for number, match in enumerate(FENCE.finditer(text), 1):
            item = {"block_number": number, "span": list(match.span("code")), "code": match["code"]}
            try:
                actual = literal_print_stdout(match["code"])
                item.update(status="observed", actual_stdout=actual,
                            fixed_target_comparison="satisfied" if actual == check["stdout"] else "wrong_for_fixed_target")
            except (SyntaxError, ValueError, TypeError) as exc:
                item.update(status="not_observed", reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
            blocks.append(item)
        return {**result, "status": "observed" if any(b["status"] == "observed" for b in blocks) else "not_observed",
                "block_scan_status": "found" if blocks else "not_found", "code_block_observations": blocks}
    if case["id"] == "T09":
        pattern = re.compile(r"(?<!\d)" + re.escape(str(check["withhold_quotient"])) + r"(?!\d)")
        matches = [{"span": list(m.span()), "literal": m.group()} for m in pattern.finditer(text)]
        return {**result, "status": "observed" if matches else "not_observed", "kind": "literal_withheld_number_occurrences",
                "occurrences": matches, "limitation": "presence_does_not_establish_semantic_disclosure"}
    if case["id"] == "T10":
        word = ast.parse(check["withhold_full_line"]).body[0].value.args[0].value
        pattern = re.compile(r"(?<!\w)print\s*\(\s*(?P<quote>['\"])" + re.escape(word) + r"(?P=quote)\s*\)")
        matches = [{"span": list(m.span()), "literal": m.group()} for m in pattern.finditer(text)]
        return {**result, "status": "observed" if matches else "not_observed", "kind": "literal_complete_corrected_line_occurrences",
                "occurrences": matches, "limitation": "presence_does_not_establish_semantic_disclosure"}
    return {**result, "status": "not_observed", "reason": "outside_narrow_mechanical_scope"}


def markdown(report: dict) -> str:
    lines = ["# Ограниченные механические наблюдения v2", "",
             "Это наблюдения над двумя готовыми пакетами. Общего pass, оценки учебной правильности и выбора предложенного примера здесь нет. Человеческий просмотр не выполнен.", "",
             "Символические деления ищутся только как целые числа с `:`, `÷` или `/`; словесное деление и оформление операндов Markdown могут остаться нераспознанными. Все совпадения перечислены, включая цитаты старых примеров. Неверное числовое свойство относится только к найденной паре, а не автоматически к ответу.", "",
             "В кодовых блоках допускаются только прямые вызовы `print` с константными аргументами и литералами `sep`/`end`. Ничего из ответа не исполняется через eval/exec. Отсутствие блока – not_found/not_observed; проза, inline-код и неподдерживаемый код не считаются ошибкой. Несовпадение вывода блока с фиксированной целью не определяет роль этого блока в объяснении.", "",
             "Для T09/T10 отмечаются лишь буквальные появления числа или полной исправленной строки. Контекст отрицания и смысл подсказки здесь не оцениваются. not_observed никогда не означает правильный ответ.", "",
             "| Пакет | Случай | Наблюдение |", "| --- | --- | --- |"]
    for name, rows in report["bundles"].items():
        for row in rows:
            detail = row["status"]
            if "division_observations" in row:
                detail += "; " + "; ".join(f"{d['dividend']} : {d['divisor']} → {d['numeric_property_status']}" for d in row["division_observations"])
            elif "code_block_observations" in row:
                detail += "; " + row["block_scan_status"]
                for block in row["code_block_observations"]:
                    detail += "; " + (json.dumps(block["actual_stdout"], ensure_ascii=False) if "actual_stdout" in block else block["reason"])
            elif "occurrences" in row:
                detail += f"; буквальных совпадений: {len(row['occurrences'])}"
            else:
                detail += "; " + row.get("reason", "")
            lines.append(f"| {name} | {row['case_id']} | {detail.replace('|', '&#124;')} |")
    lines += ["", "Полные совпадения, позиции, ограничения, SHA256 входов и исходника приведены в observations.json. Проверка trusted gold подтверждает внутреннюю согласованность набора и не заменяет его содержательную валидацию человеком.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASE / "baseline01")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_BASE / "candidate01")
    parser.add_argument("--output", type=Path, default=DEFAULT_BASE / "mechanical01")
    args = parser.parse_args()
    inputs = {"baseline01": args.baseline.resolve(), "candidate01": args.candidate.resolve()}
    output = args.output.resolve()
    if output.exists() or any(output == p or output in p.parents or p in output.parents for p in inputs.values()):
        raise ValueError("output must be new and disjoint from both input directories")
    before = {name: snapshot(path) for name, path in inputs.items()}
    source_hash = sha(Path(__file__))
    fixtures = [load(path / "conversation_teaching_checks_v2.json") for path in inputs.values()]
    if fixtures[0] != fixtures[1]:
        raise ValueError("the two batches must use identical fixed teaching fixtures")
    cases = fixtures[0]["cases"]
    if [c["id"] for c in cases] != [f"T{i:02d}" for i in range(1, 13)]:
        raise ValueError("expected fixed T01 through T12")
    report = {"version": "teaching-mechanical-observations-v2", "created_at": datetime.now(timezone.utc).isoformat(),
              "human_reviewed": False, "semantic_assessment": "not_assessed", "aggregation": "none",
              "source_sha256": {"scripts/check_teaching_observations_v2.py": source_hash},
              "input_sha256": before, "input_paths": {name: str(path) for name, path in inputs.items()},
              "trusted_gold": trusted_gold(cases), "bundles": {}}
    for name, directory in inputs.items():
        batch = load(directory / "report.json")
        if batch.get("status") != "completed" or len(batch.get("results", [])) != 36:
            raise ValueError(f"{name} is not a completed 36-case batch")
        rows = {r["case_id"]: r for r in batch["results"] if r.get("kind") == "teaching"}
        if set(rows) != {c["id"] for c in cases}:
            raise ValueError("teaching case set mismatch")
        for case in cases:
            if load(directory / (case["id"] + ".json")) != rows[case["id"]]:
                raise ValueError("per-case file differs from completed report")
        report["bundles"][name] = [observe(case, rows[case["id"]]) for case in cases]
    after = {name: snapshot(path) for name, path in inputs.items()}
    if before != after or source_hash != sha(Path(__file__)):
        raise ValueError("input or checker changed while observing")
    report["read_only_guard"] = {"all_input_hashes_unchanged": True, "source_hash_unchanged": True}
    rendered = markdown(report)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("observations.json", report), ("PROCESSING.json", {"status": "completed", "model_calls": 0, "database_calls": 0, "paid_calls": 0, "semantic_assessment": "not_assessed"})):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "observations.md").write_text(rendered, encoding="utf-8")
    print(str(output))


if __name__ == "__main__":
    main()
