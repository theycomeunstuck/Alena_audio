"""Replay saved resolver replies through current validation, with all network disabled.

This checks query resolution only. It neither searches materials nor generates a
new teaching answer, and does not change the original longitudinal run.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.runtime import OllamaRuntime  # noqa: E402


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SavedHttpSession:
    """Return exactly one original response, verifying the entire new request."""

    trust_env = True

    def __init__(self, record: dict):
        self.record = record
        self.calls = 0

    def post(self, url, *, json, timeout, allow_redirects):
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("Unexpected retry or additional generation")
        if url != "http://127.0.0.1:11434/api/chat" or allow_redirects is not False:
            raise AssertionError("Unexpected HTTP endpoint or redirect policy")
        if json != self.record["request"]:
            raise AssertionError("Replay request differs from the original resolver request")
        saved = deepcopy(self.record["response"])

        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return saved

        return Response()

    def get(self, *args, **kwargs):
        raise AssertionError("A query-resolution replay must not load models or embeddings")


def run(input_dir: Path, output_dir: Path) -> dict:
    input_dir = input_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Output must be a new directory outside the original run")

    diagnosis_path = input_dir / "technical_diagnosis.json"
    raw_path = input_dir / "ollama_http.jsonl"
    diagnosis = read_json(diagnosis_path)
    selected = [row["case"] for row in diagnosis["errors"]
                if row["cause"] == "empty_search_query_after_in_scope_true"]
    if len(selected) != 13 or len(set(selected)) != 13:
        raise ValueError("This fixed replay requires exactly the original 13 diagnosed cases")
    records = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    paths = [diagnosis_path, raw_path, *(input_dir / f"{code}.json" for code in selected)]
    input_hashes = {path.name: sha256(path) for path in paths}
    source_paths = [Path(__file__).resolve(), ROOT / "local_app/runtime.py"]
    source_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                     for path in source_paths}
    blocked_attempts = []

    def reject_network(*args, **kwargs):
        blocked_attempts.append(True)
        raise AssertionError("All real networking is disabled during offline replay")

    cases = []
    policy_versions = set()
    with ExitStack() as guard:
        for target in ("socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
                       "requests.sessions.Session.request"):
            guard.enter_context(patch(target, reject_network))
        for code in selected:
            case_path = input_dir / f"{code}.json"
            original = read_json(case_path)
            matching = [record for record in records if record.get("case") == code
                        and record.get("endpoint") == "/api/chat"]
            if len(matching) != 1 or original["status_code"] != 503:
                raise ValueError(f"Expected one original failed resolver call: {code}")
            record = matching[0]
            saved_decision = json.loads(record["response"]["message"]["content"])
            if saved_decision.get("in_scope") is not True or saved_decision.get("search_query") != "":
                raise ValueError(f"Original resolver does not match the fixed replay case: {code}")
            observed = original["observed_context"]["messages"]
            if not observed or observed[-1]["role"] != "user":
                raise ValueError(f"Missing current learner turn: {code}")
            current = observed[-1]["content"]
            if current != original["user_message"]:
                raise ValueError(f"Current learner message mismatch: {code}")
            prior = observed[:-1]
            original_input = json.loads(record["request"]["messages"][-1]["content"])
            if original_input["latest_message_untrusted"] != current:
                raise ValueError(f"Recorded resolver input mismatch: {code}")
            settings = SimpleNamespace(
                ollama_url="http://127.0.0.1:11434", chat_model=record["request"]["model"],
                embedding_model="offline-unused", context_tokens=record["request"]["options"]["num_ctx"],
                max_output_tokens=1536, request_timeout=1, database_url=None,
            )
            session = SavedHttpSession(record)
            runtime = OllamaRuntime(settings, session=session)
            policy_versions.add(runtime.manifest()["scope_policy_version"])
            result = runtime.resolve_chat_query(current, prior, original_input["subject_hint"])
            if result["search_query"] != current.strip() or result.get("query_source") != "original_learner_message":
                raise AssertionError(f"Original learner message was not preserved: {code}")
            if result["in_scope"] is not True or result["uses_prior_context"] != saved_decision["uses_prior_context"]:
                raise AssertionError(f"Original scope or history decision changed: {code}")
            anchor_index = result.get("prior_learner_question_index")
            anchor = None
            if result["uses_prior_context"]:
                if anchor_index != saved_decision.get("prior_learner_question_index"):
                    raise AssertionError(f"Prior learner anchor changed: {code}")
                if type(anchor_index) is not int or not 0 <= anchor_index < len(prior):
                    raise AssertionError(f"Invalid prior learner anchor: {code}")
                anchor = prior[anchor_index]
                bounded_anchor = next((row for row in original_input["prior_learner_questions_untrusted"]
                                       if row["turn_index"] == anchor_index), None)
                if anchor["role"] != "user" or bounded_anchor is None or bounded_anchor["content"] != anchor["content"]:
                    raise AssertionError(f"Actual original learner anchor was not preserved: {code}")
            if session.calls != 1:
                raise AssertionError(f"Expected exactly one fake call: {code}")
            cases.append({
                "case": code, "original_http_status": 503, "resolution_passed": True,
                "fake_http_calls": session.calls, "request_exactly_matches_original": True,
                "literal_current_message_preserved": True, "current_message_chars": len(current),
                "uses_prior_context": result["uses_prior_context"],
                "prior_learner_question_index": anchor_index,
                "prior_learner_anchor_preserved": True if anchor is not None else None,
                "current_message_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
                "anchor_content_sha256": hashlib.sha256(anchor["content"].encode("utf-8")).hexdigest()
                if anchor is not None else None,
                "decision": result,
            })
    if blocked_attempts:
        raise AssertionError("Replay attempted networking outside the injected fake session")
    if {path.name: sha256(path) for path in paths} != input_hashes:
        raise AssertionError("Original input files changed during offline replay")
    if len(policy_versions) != 1:
        raise AssertionError("Resolver policy changed during replay")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "kind": "offline_query_resolution_replay",
        "source_run": str(input_dir), "scope_policy_version": next(iter(policy_versions)),
        "cases_attempted": len(cases), "cases_resolved": sum(row["resolution_passed"] for row in cases),
        "fake_http_calls": sum(row["fake_http_calls"] for row in cases),
        "preserved_explicit_prior_learner_anchors": sum(row["uses_prior_context"] for row in cases),
        "real_network_calls": 0, "new_model_generations": 0, "paid_calls": 0,
        "network_guard": {"active": True, "blocked_attempts": len(blocked_attempts),
                          "targets": ["socket.create_connection", "socket.socket.connect", "socket.socket.connect_ex",
                                      "requests.sessions.Session.request"]},
        "original_inputs_unchanged": True, "input_sha256": input_hashes, "source_sha256": source_hashes,
        "limits": ["Query resolution only; retrieval and answer generation were not run.",
                   "The original 66/80 live response count is unchanged.",
                   "No conclusion about teaching-answer correctness follows from this replay."],
        "cases": cases,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = ["# Автономная проверка сохранённых решений о поиске", "",
            "Это воспроизведение HTTP-ответов из старого прогона через текущую реализацию `resolve_chat_query`.", "",
            f"Политика: `{report['scope_policy_version']}`. Разрешены 13 из 13 ранее пустых поисковых запросов.",
            "Буквальная текущая реплика сохранена во всех 13 случаях; оба явных указателя на предыдущую реплику ученика сохранены.",
            "Сгенерированный HTTP-запрос полностью совпадает с исходным запросом в каждом случае.", "",
            "Выполнено 13 вызовов подставного HTTP-ответчика – строго один на случай, без повторов.",
            "Реальная сеть заблокирована, попыток обхода: 0. Новых модельных и платных вызовов: 0.", "",
            "Проверен только этап разрешения поискового контекста. Поиск материалов и ответы ученику не запускались.",
            "Исходный результат ночного прогона остаётся 66 ответов из 80 запросов. Корректность учебных ответов этой проверкой не установлена.", "",
            "| Случай | Реплика, символов | Предыдущая реплика ученика |", "| --- | ---: | --- |"]
    rows.extend(f"| {row['case']} | {row['current_message_chars']} | "
                f"{row['prior_learner_question_index'] if row['uses_prior_context'] else 'не используется'} |" for row in cases)
    rows.extend(["", "Исходные файлы не изменены. Хеши входных файлов, текущего кода и все решения сохранены в `report.json`.", ""])
    (output_dir / "README.md").write_text("\n".join(rows), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / ".local/conversation_longitudinal_v1/run01")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("cases_attempted", "cases_resolved", "fake_http_calls",
                                                  "preserved_explicit_prior_learner_anchors", "new_model_generations",
                                                  "real_network_calls", "paid_calls")}))


if __name__ == "__main__":
    main()
