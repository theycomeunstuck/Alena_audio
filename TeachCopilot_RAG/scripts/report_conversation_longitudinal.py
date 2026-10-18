"""Read a finished longitudinal run without model calls or source mutations.

Usage: python scripts/report_conversation_longitudinal.py --input-dir RUN --output-dir NEW_DIR
The destination must not exist. Completion and mechanical agreement are not a
semantic verdict. Original run files remain the authoritative unmodified record.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / ".local/conversation_longitudinal_v1/run01"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def normal(value: str, aliases: dict) -> str:
    """Use the preregistered harness rule, without adding new aliases."""
    value = " ".join(value.casefold().split())
    for canonical, variants in aliases.items():
        if value in {canonical.casefold(), *(item.casefold() for item in variants)}:
            return canonical.casefold()
    return value


def check_counts(values) -> dict:
    values = list(values)
    return {"observed": len(values), "true": sum(v is True for v in values),
            "false": sum(v is False for v in values),
            "unknown": sum(v is not True and v is not False for v in values)}


def numeric(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def distribution(values) -> dict:
    values = sorted(v for v in values if numeric(v) is not None)
    if not values:
        return {"count": 0, "sum": 0, "p50": None, "p95": None, "max": None}
    return {"count": len(values), "sum": sum(values), "p50": median(values),
            "p95": values[max(0, math.ceil(len(values) * .95) - 1)], "max": max(values)}


def usage_summary(calls: list[dict]) -> dict:
    responses = [call.get("response") if isinstance(call.get("response"), dict) else {} for call in calls]
    def token_field(key):
        values = [response.get(key) for response in responses]
        known = [v for v in values if numeric(v) is not None]
        return {"total": sum(known), "known_calls": len(known), "unknown_calls": len(values) - len(known)}
    return {"calls": len(calls), "responses": sum(bool(response) for response in responses),
            "errors": sum("error" in call for call in calls),
            "prompt_tokens": token_field("prompt_eval_count"),
            "completion_tokens": token_field("eval_count"),
            "http_seconds": distribution([call.get("elapsed_seconds") for call in calls]),
            "ollama_total_seconds": distribution([response["total_duration"] / 1e9 for response in responses
                                                    if numeric(response.get("total_duration")) is not None])}


def mark(value) -> str:
    return "да" if value is True else "нет" if value is False else "не проверено"


def inline(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False)
    return value.replace("\n", " ").replace("|", "\\|").replace("\u2014", "–")


def as_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2).replace("\u2014", "–")


def write_text(path: Path, text: str):
    # Display-only typography; original trace and report files are never changed.
    path.write_text(text.replace("\u2014", "–").rstrip() + "\n", encoding="utf-8")


def build(args) -> Path:
    source = args.input_dir.resolve()
    destination = args.output_dir.resolve()
    source_paths = [source / name for name in ("report.json", "preregistered.json", "ollama_http.jsonl")]
    report, prereg = (read_json(path) for path in source_paths[:2])
    fixture = prereg.get("fixture")
    if not isinstance(fixture, dict) or not isinstance(fixture.get("learners"), list):
        raise ValueError("preregistered.json must contain the original fixture")
    aliases = fixture.get("expected_aliases", {})
    raw_lines = source_paths[2].read_text(encoding="utf-8").splitlines()
    http_calls, invalid_lines = [], []
    for number, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        try:
            call = json.loads(line)
            if not isinstance(call, dict):
                raise ValueError("HTTP trace line is not an object")
            http_calls.append(call)
        except (json.JSONDecodeError, ValueError) as exc:
            invalid_lines.append({"line": number, "error": str(exc)})
    if invalid_lines:
        raise ValueError(f"Invalid HTTP trace lines; refusing an incomplete token report: {invalid_lines}")
    specifications = [session for learner in fixture["learners"] for session in learner["sessions"]]
    expected_sessions = int(report.get("expected_sessions", len(specifications)))
    expected_messages = int(report.get("expected_messages", sum(len(s["messages"]) for s in specifications)))
    messages, sessions = report.get("messages", []), report.get("sessions", [])
    by_session, by_message = {}, {}
    for row in sessions:
        if row["code"] in by_session:
            raise ValueError("Duplicate session code in report; refusing silent deduplication")
        by_session[row["code"]] = row
    for row in messages:
        if row["code"] in by_message:
            raise ValueError("Duplicate message code in report; refusing silent deduplication")
        by_message[row["code"]] = row
    analyses = []
    for spec in specifications:
        row = by_session.get(spec["code"], {})
        expected = {(operation, normal(value, aliases)) for operation, values in spec["expected_preferences"].items() for value in values}
        candidates = row.get("candidates")
        actual = {(item["operation"], normal(item["interest"], aliases)) for item in candidates} if isinstance(candidates, list) else None
        missing = [list(item) for item in sorted(expected - actual)] if actual is not None else None
        extra = [list(item) for item in sorted(actual - expected)] if actual is not None else None
        matching = not missing and not extra if actual is not None else None
        extraction_status = (row.get("interest_extraction") or {}).get("status")
        state_matching = (row.get("final_interests_match") is True and row.get("final_avoided_match") is True
                          if "final_interests_match" in row and "final_avoided_match" in row else None)
        analyses.append({"code": spec["code"], "status": row.get("status", "not_attempted"),
                         "topic_id": spec["topic_id"], "extraction_status": extraction_status,
                         "expected_candidates": sorted(expected),
                         "actual_candidates": sorted(actual) if actual is not None else None,
                         "missing_candidates": missing, "extra_candidates": extra,
                         "candidate_sets_match": matching, "final_preferences_match": state_matching,
                         "final_interests_match": row.get("final_interests_match"),
                         "final_avoided_match": row.get("final_avoided_match"),
                         "competencies_unchanged": row.get("competencies_unchanged"),
                         "completion_idempotent": row.get("completion_idempotent"),
                         "approval_idempotent": row.get("approval_idempotent"),
                         "http_200": sum(by_message.get(spec["code"] + "M" + str(i + 1), {}).get("status_code") == 200 for i in range(len(spec["messages"]))),
                         "expected_messages": len(spec["messages"]),
                         "harness_missing_expected": row.get("missing_expected"),
                         "harness_extra_candidates": row.get("extra_candidates"),
                         "candidate_difference_matches_harness": (missing == row.get("missing_expected") and extra == row.get("extra_candidates")) if actual is not None else None,
                         "error": row.get("error")})
    chat_calls = [call for call in http_calls if call.get("endpoint") == "/api/chat"]
    memory_calls = [call for call in chat_calls if str(call.get("case", "")).endswith("-memory")]
    dialogue_calls = [call for call in chat_calls if call not in memory_calls]
    source_scope = [row.get("memory_source_scope_ok") for row in messages]
    replay_rows = [row.get("replay_cached") for row in messages if "replay_cached" in row]
    status_counts = Counter(str(row.get("status_code", "unknown")) for row in messages)
    summary = {
        "version": "conversation-longitudinal-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_origin": report.get("data_origin", "synthetic_unreviewed"),
        "human_reviewed": False,
        "semantic_review": {"status": "pending_separate_ai_review", "human_reviewed": False,
                            "automatic_semantic_pass": False},
        "run_status": report.get("status"),
        "teacher_review": report.get("teacher_review"),
        "expected_messages": expected_messages, "expected_sessions": expected_sessions,
        "messages": {"attempted": len(messages), "http_200": status_counts.get("200", 0),
                     "not_attempted": max(0, expected_messages - len(messages)), "http_status_counts": dict(status_counts)},
        "sessions": {"attempted": len(sessions), "completed": sum(row.get("status") == "completed" for row in analyses),
                     "ready": sum(row["extraction_status"] == "ready" for row in analyses),
                     "candidate_sets_matched": sum(row["candidate_sets_match"] is True for row in analyses),
                     "final_preferences_matched": sum(row["final_preferences_match"] is True for row in analyses),
                     "completed_ready_candidates_and_state_matched": sum(row["status"] == "completed" and row["extraction_status"] == "ready"
                         and row["candidate_sets_match"] is True and row["final_preferences_match"] is True for row in analyses),
                     "candidate_differences_unavailable": sum(row["actual_candidates"] is None for row in analyses)},
        "candidate_differences": [{key: row[key] for key in ("code", "missing_candidates", "extra_candidates")} for row in analyses],
        "unexpected_report_session_codes": sorted(set(by_session) - {spec["code"] for spec in specifications}),
        "unexpected_report_message_codes": sorted(set(by_message) - {spec["code"] + "M" + str(i + 1) for spec in specifications for i in range(len(spec["messages"]))}),
        "idempotency": {"message_replay": check_counts(replay_rows), "message_replay_expected_checks": len(fixture["learners"]),
                        "session_completion": check_counts(row["completion_idempotent"] for row in analyses),
                        "teacher_confirmation": check_counts(row["approval_idempotent"] for row in analyses)},
        "isolation": {"authorization": check_counts(row.get("passed") for row in report.get("isolation_checks", [])),
                      "authorization_original_checks": report.get("isolation_checks", []),
                      "memory_source_scope": check_counts(source_scope),
                      "meaning": "Checks authorization and source ownership; semantic cross-learner facts still need review."},
        "competencies_unchanged": check_counts(row["competencies_unchanged"] for row in analyses),
        "generation": {"all_http_chat_calls": usage_summary(chat_calls), "dialogue_including_routing": usage_summary(dialogue_calls),
                       "memory_updates": usage_summary(memory_calls), "reported_generation_calls": report.get("generation_calls"),
                       "http_call_count_matches_report": len(chat_calls) == report.get("generation_calls"),
                       "embedding_http_calls": sum("embed" in str(call.get("endpoint", "")) for call in http_calls),
                       "invalid_http_lines": invalid_lines,
                       "measurement_limits": "HTTP call durations and Ollama-reported tokens only. Not full message latency, queue latency or provider throughput. Sequential synthetic sessions are not concurrent load.",
                       "run_elapsed_seconds_including_setup_and_cleanup": report.get("elapsed_seconds")},
        "paid_provider_calls": report.get("paid_provider_calls"),
        "temporary_database_removed": report.get("temporary_database_removed"),
        "aliases": aliases,
        "source_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths],
        "session_details": analyses,
        "limits": ["Completed does not mean passed.", "Candidate and state agreement are mechanical, not pedagogical semantic approval.",
                   "Teacher confirmations were simulated against preregistered expectations.",
                   "No model calls, retries, source rewriting or expectation changes are performed by this formatter."]}
    destination.mkdir(parents=True, exist_ok=False)
    write_text(destination / "summary.json", as_json(summary))
    table = ["# Машинная проверка 20 последовательных занятий", "",
             "Завершение процедуры не означает содержательную правильность. AI-разбор ожидается, человеческий просмотр не выполнен.", "",
             "| Занятие | HTTP 200 | Процедура | Извлечение | Кандидаты совпали | Карточка совпала | Компетенции неизменны |",
             "|---|---:|---|---|---|---|---|"]
    for row in analyses:
        table.append(f"| {row['code']} | {row['http_200']}/{row['expected_messages']} | {inline(row['status'])} | {inline(row['extraction_status'])} | {mark(row['candidate_sets_match'])} | {mark(row['final_preferences_match'])} | {mark(row['competencies_unchanged'])} |")
    table.extend(["", "## Расхождения кандидатов", ""])
    for row in analyses:
        table.append(f"- {row['code']}: отсутствуют `{inline(row['missing_candidates'])}`; лишние `{inline(row['extra_candidates'])}`.")
        if row["error"]:
            table.append(f"  Ошибка процедуры: `{inline(row['error'])}`.")
    write_text(destination / "sessions.md", "\n".join(table))
    transcript_dir = destination / "transcripts"
    transcript_dir.mkdir()
    for learner in fixture["learners"]:
        lines = [f"# {learner['code']} – полный разговор", "",
                 "Синтетические данные. AI-разбор ожидается; человеческий просмотр не выполнен. Ожидания ниже были зафиксированы до генерации и не входили в запросы модели.", "",
                 "Для отображения длинное тире заменено коротким. Точные исходные тексты находятся в неизменённом report.json исходного прогона.", ""]
        for spec in learner["sessions"]:
            row = by_session.get(spec["code"], {})
            analysis = next(item for item in analyses if item["code"] == spec["code"])
            lines.extend([f"## {spec['code']} – {spec['topic_id']}", "",
                          "### Заранее заданные основания для разбора", "", spec["review_notes"], "",
                          "Ожидаемые операции:", "", "```json", as_json(spec["expected_preferences"]), "```", "",
                          f"Ожидаемые интересы: `{inline(spec['expected_final_interests'])}`. Ожидаемые запреты: `{inline(spec['expected_final_avoided'])}`.", "",
                          f"Процедура: `{inline(analysis['status'])}`; извлечение: `{inline(analysis['extraction_status'])}`.", ""])
            for index, expected_message in enumerate(spec["messages"], 1):
                code = spec["code"] + "M" + str(index)
                message = by_message.get(code)
                lines.extend([f"### {code}", "", "**Ученик**", "", message.get("user_message", expected_message) if message else expected_message, ""])
                if not message:
                    lines.extend(["**Ответ**", "", "Запрос не выполнен; сообщение показано из заранее заданного сценария.", ""])
                    continue
                if message.get("user_message") != expected_message:
                    lines.extend(["Несовпадение с исходным сценарием. Ожидаемый текст:", "", expected_message, ""])
                observed = message.get("observed_context") or {}
                memory = observed.get("memory_context") or {}
                lines.extend([f"HTTP: `{message.get('status_code')}`. Генерационных вызовов: `{message.get('generation_calls')}`.", "",
                              f"Переданные предпочтения: `{inline(observed.get('preferences'))}`. Устаревшая память: `{inline(memory.get('stale'))}`.", "", "**Помощник**", ""])
                response = message.get("response") or {}
                choices = response.get("choices") if isinstance(response, dict) else None
                if choices:
                    for choice in choices:
                        content = (choice.get("message") or {}).get("content")
                        lines.extend([content if isinstance(content, str) else as_json(content), ""])
                else:
                    lines.extend(["```json", as_json(response), "```", ""])
            lines.extend(["### Память после занятия", "", "Кандидаты с буквальными цитатами:", "", "```json",
                          as_json(row.get("candidates")), "```", "",
                          f"Отсутствуют: `{inline(analysis['missing_candidates'])}`. Лишние: `{inline(analysis['extra_candidates'])}`.", "",
                          f"Фактические интересы: `{inline(row.get('final_interests'))}`.", "",
                          "Фактическая разрешённая проекция:", "", "```json", as_json(row.get("final_preferences")), "```", "",
                          "Подтверждение преподавателем имитировалось. Непредусмотренные кандидаты автоматически не принимались.", ""])
            if row.get("error"):
                lines.extend(["Ошибка процедуры:", "", "```json", as_json(row["error"]), "```", ""])
        write_text(transcript_dir / (learner["code"] + ".md"), "\n".join(lines))
    lines = ["# Итоги последовательных разговоров", "",
             f"HTTP 200: **{summary['messages']['http_200']}/{expected_messages}**. Процедура завершена: **{summary['sessions']['completed']}/{expected_sessions}**. Извлечение готово: **{summary['sessions']['ready']}/{expected_sessions}**.", "",
             f"Кандидаты совпали с предварительными ожиданиями: **{summary['sessions']['candidate_sets_matched']}/{expected_sessions}**; конечные интересы и запреты: **{summary['sessions']['final_preferences_matched']}/{expected_sessions}**.", "",
             "Это машинные показатели. Содержательный AI-разбор ожидается, человеческий просмотр не выполнен. Статус completed не равен passed.", "",
             "- [Машинная таблица занятий](sessions.md)", "- [Полные числовые результаты и ограничения](summary.json)"]
    lines.extend(f"- [Разговор {learner['code']}](transcripts/{learner['code']}.md)" for learner in fixture["learners"])
    timing = summary["generation"]["all_http_chat_calls"]
    lines.extend(["", "## Учёт генерации", "",
                  f"Всего HTTP-вызовов генерации: {timing['calls']}; входных токенов по ответам Ollama: {timing['prompt_tokens']['total']}; выходных: {timing['completion_tokens']['total']}.", "",
                  f"HTTP-время отдельных вызовов генерации: суммарно {timing['http_seconds']['sum']:.3f} с; p50 {inline(timing['http_seconds']['p50'])} с; p95 {inline(timing['http_seconds']['p95'])} с.", "",
                  "Это не полная задержка ответа, не время очереди и не пропускная способность публичного API. Занятия выполнялись последовательно. Вызовы маршрутизации и обновления памяти включены отдельно в summary.json. Неизвестные токены не восстанавливаются оценкой.", "",
                  "Исходные report.json, preregistered.json и ollama_http.jsonl не изменены. Их SHA256 сохранены в summary.json. Названия интересов сопоставлены только с исходными aliases. Точная проверка цитаты и совпадение операции не заменяют проверку смысла."])
    write_text(destination / "README.md", "\n".join(lines))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = build(args)
    print(json.dumps({"output_dir": str(destination), "status": "formatted_not_semantically_assessed", "human_reviewed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
