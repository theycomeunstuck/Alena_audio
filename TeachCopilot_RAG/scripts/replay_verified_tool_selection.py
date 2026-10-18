"""Replay only the current literal-tool selector against saved scope decisions.

This is an offline diagnostic, not a new live RAG batch. It imports no runtime,
Store or provider client and performs no retrieval, embeddings or generation.
Historical responses and errors remain unchanged. Public input is the only
input to the selector; private expectations are used afterwards for diagnosis.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_verified_learning_tools_v1 import (
    EXPECTED_CASES, OPERATIONS, error_record, fixture_cases, git_commit,
    mechanical_diagnostic, source_hashes, write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def current_source_hashes() -> dict[str, str]:
    result = source_hashes()
    result[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def validate_saved_resolution(decision: object, public: dict) -> dict:
    """Validate a stored production decision without invoking its classifier.

    A prior anchor must identify a real learner turn within the same last-eight,
    4800-character history window used by the classifier. Assistant turns cannot
    become anchors. An operand still comes exclusively from the latest message.
    """
    required = {"in_scope", "uses_prior_context", "search_query"}
    allowed = required | {"social_turn", "prior_learner_question_index", "query_source"}
    if (not isinstance(decision, dict) or not required <= set(decision)
            or set(decision) - allowed
            or type(decision.get("in_scope")) is not bool
            or type(decision.get("uses_prior_context")) is not bool
            or not isinstance(decision.get("search_query"), str)
            or ("social_turn" in decision and type(decision["social_turn"]) is not bool)):
        raise ValueError("saved_scope_decision_is_absent_or_malformed")
    messages = public["messages"]
    prior = messages[:-1]
    current = messages[-1]["content"].strip()
    prior_questions = [(index, row["content"].strip()) for index, row in enumerate(prior)
                       if row["role"] == "user"][-8:]
    remaining = 4800
    visible_anchors = set()
    for index, content in reversed(prior_questions):
        if content and remaining > 0:
            visible_anchors.add(index)
            remaining -= len(content[:min(1200, remaining)])
    anchor = decision.get("prior_learner_question_index")
    if "prior_learner_question_index" in decision and (
            type(anchor) is not int or not decision["uses_prior_context"] or anchor not in visible_anchors):
        raise ValueError("saved_scope_anchor_is_not_a_visible_prior_learner_turn")
    if decision["uses_prior_context"] and anchor is None:
        raise ValueError("saved_scope_prior_context_requires_an_explicit_learner_anchor")
    query = decision["search_query"].strip()
    if not decision["in_scope"]:
        if query or decision["uses_prior_context"] or "query_source" in decision:
            raise ValueError("saved_out_of_scope_decision_has_retrieval_context")
        if decision.get("social_turn") and len(current) > 800:
            raise ValueError("saved_social_decision_exceeds_current_message_limit")
    elif not query:
        raise ValueError("saved_in_scope_decision_has_no_validated_query")
    elif "query_source" in decision:
        if decision["query_source"] != "original_learner_message" or query != current or len(current) > 1200:
            raise ValueError("saved_original_query_does_not_match_the_actual_learner_message")
    elif len(query) > 500:
        raise ValueError("saved_model_rewrite_exceeds_its_limit")
    return deepcopy(decision)


def markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Повторная проверка только выбора учебного инструмента",
        "",
        "Это локальная проверка selector на сохранённых решениях классификатора. "
        "Она не является новым прогоном чата, проверкой HTTP, поиска, модели или её текущей точности.",
        "",
        f"Исходный запуск: `{report['historical_run_directory']}`.",
        f"Исходный commit: `{report.get('historical_git_commit')}`. "
        f"Текущий commit: `{report['current_git_commit']}`.",
        f"Версия selector: `{report.get('selector_version')}`.",
        "",
        f"Знаменатель – {EXPECTED_CASES} фиксированных случаев. "
        f"Допущены по сохранённому scope decision: {summary['eligible_cases']}. "
        f"Выбраны инструменты: {summary['selected_tools']}. "
        f"Маршрут совпал с заранее заданным: {summary['route_matches_fixed_expectation']}/{EXPECTED_CASES}.",
        "",
        "Модельных, внешних, платных и embedding-вызовов – 0. "
        "Модельная генерация ответа не выполнялась. Детерминированные ответы созданы заново. Идентификатор случая, категория и private_gold "
        "не передавались selector. Содержательная оценка человеком не проведена.",
        "",
        "| Случай | Исходный live status | Допуск | Ожидался маршрут | Выбран офлайн | Совпадение |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["results"]:
        route = row.get("observed_route", "not_eligible")
        match = row.get("route_matches_fixed_expectation")
        match_text = "да" if match is True else "нет" if match is False else "не оценено"
        lines.append(f"| {row['case_id']} | {row['historical_status']} | {row['status']} | "
                     f"{row['expected_route']} | {route} | {match_text} |")
    lines.extend([
        "",
        "Числовые поля и stdout сравниваются механически. Свойство новой практики "
        "вычисляется независимо по видимым операндам; частное для практики не добавляется "
        "в ответ или отчёт. Эти проверки не устанавливают качество объяснения, "
        "самостоятельность ученика или устойчивое освоение темы.",
        "",
        "Если в исходном прогоне был технический отказ, он остаётся отказом. "
        "Выбор инструмента на том же сохранённом решении показывает только локальную ветку, "
        "которая могла бы избежать прежней генерации; он не доказывает новый HTTP-успех. "
        "Обычные модельные ответы для случаев без инструмента здесь не пересоздаются и не оцениваются заново.",
        "",
        f"Обработка: `{report['status']}`. Целостность входов и кода: "
        f"`{report.get('integrity', {}).get('unchanged')}`. "
        "Подробные механические проверки, исходные ошибки и SHA256 находятся в `report.json`.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict:
    # The only application import is a pure selector with bounded arithmetic
    # and literal AST evaluation. No RagService/runtime/Store is instantiated.
    from local_app.learning_tools import VERSION, verified_learning_response

    source = args.input_dir.resolve()
    out = args.output_dir.resolve()
    if not source.is_dir():
        raise ValueError("historical_run_directory_does_not_exist")
    out.mkdir(parents=True, exist_ok=False)
    inputs = [source / "preregistered.json", source / "report.json", source / "ollama_http.jsonl"]
    initial_input_hashes = None
    initial_source_hashes = None
    initial_commit = git_commit()
    report = {
        "kind": "selector_only_not_new_live_run", "status": "initializing", "created_at": utc_now(),
        "historical_run_directory": str(source), "current_git_commit": initial_commit,
        "selector_version": VERSION, "expected_cases": EXPECTED_CASES,
        "local_model_calls": 0, "external_provider_calls": 0, "paid_provider_calls": 0,
        "embedding_calls": 0, "retrieval_calls": 0, "database_calls": 0,
        "saved_scope_decisions_reused_not_regenerated": True,
        "human_reviewed": False, "semantic_assessment": "not_human_reviewed",
        "counts_are_offline_zeros_not_measured_live_usage_or_latency": True,
        "historical_responses_are_not_modified_or_retried": True, "results": [],
    }
    try:
        initial_source_hashes = current_source_hashes()
        initial_input_hashes = input_hashes(inputs)
        preregistered = json.loads(inputs[0].read_text(encoding="utf-8"))
        historical = json.loads(inputs[1].read_text(encoding="utf-8"))
        if historical.get("status") != "completed" or not historical.get("source_integrity", {}).get("unchanged"):
            raise ValueError("historical_live_batch_must_be_completed_with_unchanged_inputs")
        prepared = fixture_cases(preregistered.get("fixture"))
        originals = historical.get("results")
        if not isinstance(originals, list) or len(originals) != EXPECTED_CASES:
            raise ValueError("historical_live_batch_must_preserve_all_24_case_outcomes")
        original_by_id = {row["case_id"]: row for row in originals}
        if len(original_by_id) != EXPECTED_CASES or set(original_by_id) != {case["case_id"] for case, _ in prepared}:
            raise ValueError("historical_results_and_frozen_fixture_do_not_identify_the_same_cases")
        case_inputs = [source / (case["case_id"] + ".json") for case, _ in prepared]
        inputs += case_inputs
        initial_input_hashes.update(input_hashes(case_inputs))
        report["historical_git_commit"] = preregistered.get("git_commit")
        report["historical_technical_summary_unchanged"] = deepcopy(historical.get("technical_summary"))
        report["status"] = "running"
        write_json(out / "PROCESSING.json", {"status": "running", "kind": report["kind"], "local_model_calls": 0})
        for case, public in prepared:
            identifier = case["case_id"]
            original = original_by_id[identifier]
            case_path = source / (identifier + ".json")
            if json.loads(case_path.read_text(encoding="utf-8")) != original:
                raise ValueError("historical_case_file_differs_from_its_original_report_row")
            if original.get("public_input") != public:
                raise ValueError("historical_case_public_input_differs_from_its_frozen_fixture")
            row = {
                "case_id": identifier, "status": "not_eligible", "public_input": deepcopy(public),
                "expected_route": case["private_gold"]["expected_route"],
                "historical_status": original["status"], "historical_error": deepcopy(original.get("error")),
                "historical_case_file": str(case_path), "historical_case_sha256": initial_input_hashes[str(case_path)],
                "historical_answer_generation_calls": original.get("answer_generation_calls"),
                "historical_response_unchanged": True, "human_reviewed": False,
                "semantic_assessment": "not_human_reviewed", "selector_only_not_new_live_run": True,
            }
            try:
                decision = validate_saved_resolution(original.get("scope_resolution"), public)
                row["validated_saved_scope_decision"] = decision
                if not decision["in_scope"]:
                    row["not_eligible_reason"] = "saved_scope_decision_out_of_scope"
                    row["observed_route"] = "out_of_scope"
                    row["route_matches_fixed_expectation"] = row["expected_route"] == "out_of_scope"
                else:
                    row["status"] = "eligible_selector_called"
                    history = [{"role": message["role"], "content": message["content"]}
                               for message in public["messages"][:-1]
                               if message["role"] in {"user", "assistant"}]
                    # This is the complete selector call. No gold or resolved
                    # rewrite is supplied; old turns only exclude repeated practice.
                    selected = verified_learning_response(public["messages"][-1]["content"], public["subject_id"],
                                                          previous_messages=history)
                    row["selector_response"] = selected
                    operation = (selected or {}).get("learning_tool", {}).get("operation")
                    row["observed_route"] = operation if selected is not None else "normal_model"
                    row["route_matches_fixed_expectation"] = row["observed_route"] == row["expected_route"]
                    if selected is not None:
                        if operation not in OPERATIONS or (selected.get("answer_routing") or {}).get("backend") != "deterministic_local":
                            raise ValueError("selector_returned_an_invalid_operation_or_backend")
                        row["mechanical_diagnostic"] = mechanical_diagnostic(
                            selected, decision, case["private_gold"],
                            {"generation_calls": 0, "scope_resolution_calls": 0,
                             "answer_generation_calls": 0, "embedding_calls": 0})
                        row["counterfactual_previous_answer_generation_would_be_bypassed"] = bool(original.get("answer_generation_calls"))
                        row["counterfactual_limit"] = "Only if the same already-saved valid scope decision is reused; no new HTTP success was measured."
            except Exception as exc:
                row["error"] = error_record(exc)
                if "validated_saved_scope_decision" in row and row["validated_saved_scope_decision"]["in_scope"]:
                    row["status"] = "selector_error"
                else:
                    row["not_eligible_reason"] = "saved_scope_decision_absent_or_invalid"
            report["results"].append(row)
            write_json(out / (identifier + ".json"), row)
            write_json(out / "report.json", report)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = error_record(exc)
    finally:
        try:
            final_inputs = input_hashes(inputs) if initial_input_hashes is not None else None
            final_sources = current_source_hashes() if initial_source_hashes is not None else None
            final_commit = git_commit()
            unchanged = (initial_input_hashes is not None and initial_input_hashes == final_inputs
                         and initial_source_hashes == final_sources and initial_commit == final_commit)
            report["integrity"] = {
                "unchanged": unchanged, "start_git_commit": initial_commit, "end_git_commit": final_commit,
                "start_input_sha256": initial_input_hashes, "end_input_sha256": final_inputs,
                "start_source_sha256": initial_source_hashes, "end_source_sha256": final_sources,
            }
            if initial_input_hashes is not None and not unchanged:
                report["status"] = "invalid_inputs_or_source_changed_during_replay"
        except Exception as exc:
            report["status"] = "invalid_integrity_check_failed"
            report["integrity_error"] = error_record(exc)
        rows = report["results"]
        selected = [row for row in rows if row.get("selector_response") is not None]
        supported = [row for row in rows if row["expected_route"] in OPERATIONS]
        boundaries = [row for row in rows if row["expected_route"] not in OPERATIONS]
        report["summary"] = {
            "denominator": EXPECTED_CASES, "recorded_case_outcomes": len(rows),
            "eligible_cases": sum(row["status"] in {"eligible_selector_called", "selector_error"} for row in rows),
            "not_eligible_cases": sum(row["status"] == "not_eligible" for row in rows),
            "selector_errors": sum(row["status"] == "selector_error" for row in rows),
            "selected_tools": len(selected),
            "route_matches_fixed_expectation": sum(row.get("route_matches_fixed_expectation") is True for row in rows),
            "supported_route_matches": sum(row.get("route_matches_fixed_expectation") is True for row in supported),
            "supported_denominator": len(supported),
            "boundary_route_matches": sum(row.get("route_matches_fixed_expectation") is True for row in boundaries),
            "boundary_denominator": len(boundaries),
            "historical_live_errors_still_errors": sum(row["historical_status"] == "error" for row in rows),
            "selector_route_match_is_not_a_new_live_success": True,
        }
        report["completed_at"] = utc_now()
        write_json(out / "report.json", report)
        write_json(out / "summary.json", {key: report[key] for key in (
            "kind", "status", "current_git_commit", "selector_version", "summary",
            "local_model_calls", "paid_provider_calls", "semantic_assessment")})
        (out / "README.md").write_text(markdown(report), encoding="utf-8")
        write_json(out / "PROCESSING.json", {
            "status": report["status"], "kind": report["kind"], "recorded_cases": len(rows),
            "expected_cases": EXPECTED_CASES, "local_model_calls": 0, "paid_provider_calls": 0,
            "semantic_assessment": "not_human_reviewed", "completed_at": report["completed_at"],
        })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Completed historical finite batch, read only")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing paths are refused")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({key: report.get(key) for key in ("kind", "status", "selector_version", "summary", "local_model_calls", "paid_provider_calls")}, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
