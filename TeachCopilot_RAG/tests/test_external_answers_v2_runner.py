"""The finite runner is exercised with synthetic local and HTTP test doubles."""
from copy import deepcopy
import json
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.request import Request
from urllib.error import HTTPError, URLError

import pytest

from scripts.check_memory_external_v2 import AuditedOpener, IDENTITIES, run_check, scenarios
from scripts.transition_memory_budget_v5 import transition
from tests.test_memory_budget_v5 import setup_transition


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, *args):
        return json.dumps({"choices": [{"message": {"content": json.dumps({"answer": "Проверь пример умножением.",
            "citations": ["S1", "S2"]})}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30}}).encode()


class LocalRuntime:
    def __init__(self, settings):
        self.generation_run_budget = SimpleNamespace(snapshot=lambda: {"generations_reserved": 0})
        self.generation_scheduler = None

    def check_models(self):
        return {"ready": True}

    def manifest(self):
        return {"synthetic_test_double": True}

    def chat_json(self, messages, **kwargs):
        candidate = messages[-1]["content"]
        ambiguous = "подъезда" in candidate or "болезнь" in candidate
        return {"safe_to_send": not ambiguous, "uncertain": ambiguous}

    def chat(self, *args, **kwargs):
        return {"message": {"role": "assistant", "content": '{"answer":"Продолжим локально.","citations":[]}'}}


def test_fixed_scenarios_cover_six_safe_and_six_local_with_exact_current_and_quotes():
    cases = scenarios()
    assert len(cases) == 12
    assert [row["expected_backend"] for row in cases] == ["local"] * 6 + ["spiderssense"] * 6
    assert {row["case_id"] for row in cases} == {f"{prefix}{i:02}" for prefix in ("E", "L") for i in range(1, 7)}
    assert {"хз", "4? Объясни проверку вывода print(2 + 2)."} <= {row["projection"]["current_message"] for row in cases}
    assert "[REDACTED]" in cases[6]["expected_current_message"]
    assert all(row["expected_quote"] for row in cases)


def request_for(case):
    return Request("https://spiderssense.com/v1/chat/completions", data=json.dumps({"messages": [
        {"role": "system", "content": "Учебная инструкция."},
        {"role": "user", "content": json.dumps({"current_message": case["expected_current_message"],
            "memory": [{"source_id": "S1", "quote": case["expected_quote"]}],
            "materials": [{"source_id": "S2"}]}, ensure_ascii=False)}]}, ensure_ascii=False).encode())


@pytest.mark.parametrize("change", ["current", "quote", "identifier", "alias", "local"])
def test_actual_outbound_assertions_run_before_delegate_transport(change):
    case = deepcopy(scenarios()[6])
    transport = Mock()
    audit = AuditedOpener(transport)
    audit.case = case
    request = request_for(case)
    raw = json.loads(request.data)
    payload = json.loads(raw["messages"][1]["content"])
    if change == "local":
        audit.case = scenarios()[0]
    elif change == "identifier":
        raw["private"] = IDENTITIES[1]
    elif change == "current":
        payload["current_message"] = "Потерян текущий вопрос."
    elif change == "quote":
        payload["memory"][0]["quote"] = "Изменённая цитата."
    else:
        payload["memory"][0]["source_id"] = "S3"
    raw["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
    request.data = json.dumps(raw, ensure_ascii=False).encode()
    with pytest.raises(AssertionError):
        audit.open(request)
    transport.open.assert_not_called()


def test_dry_run_saves_frozen_cases_without_loading_configuration_or_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr("local_app.runtime.OllamaRuntime", Mock(side_effect=AssertionError("no runtime")))
    output = tmp_path / "dry"
    report = run_check(tmp_path / "absent.json", output)
    assert report["status"] == "prepared_not_executed" and report["passed"] is None
    assert (output / "preregistered.json").exists()
    prereg = json.loads((output / "preregistered.json").read_text(encoding="utf-8"))
    assert {"scripts/check_memory_external_v2.py", "local_app/runtime.py", "local_app/answer_adapter.py",
            "local_app/budget_authorization.py"} <= set(prereg["settings"]["code_sha256"])
    with pytest.raises(FileExistsError):
        run_check(tmp_path / "absent.json", output)


def test_http_error_raw_body_hash_is_saved_and_key_redacted_before_reraise():
    raw = b'{"error":"synthetic-key-731 upstream unavailable"}'
    error = HTTPError("https://spiderssense.com/v1/chat/completions", 503, "unavailable", {}, io.BytesIO(raw))
    transport = Mock()
    transport.open.side_effect = error
    audit = AuditedOpener(transport, secret="synthetic-key-731")
    audit.case = scenarios()[6]
    with pytest.raises(HTTPError) as caught:
        audit.open(request_for(audit.case))
    assert caught.value is error
    saved = audit.requests[0]
    assert saved["response_sha256"] == hashlib.sha256(raw).hexdigest()
    assert saved["http_status"] == 503 and saved["error_body_truncated"] is False
    assert "synthetic-key-731" not in saved["raw_response_text"]
    assert "[REDACTED_CREDENTIAL]" in saved["raw_response_text"]
    assert transport.open.call_count == 1


def test_full_finite_pass_counts_exact_http_calls_and_preserves_responses(tmp_path, monkeypatch):
    catalog, authorization_dir, _ = setup_transition(tmp_path)
    transport = Mock()
    transport.open.side_effect = lambda *args, **kwargs: Response()
    monkeypatch.setattr("research_memory.spiderssense.build_opener", lambda *args: transport)
    monkeypatch.setenv("SPIDERSSENSE_API_KEY", "synthetic-key-731")
    monkeypatch.setenv("LOCAL_GENERATION_RUN_BUDGET_PATH", "")
    monkeypatch.setenv("LOCAL_GENERATION_SCHEDULER_DSN", "synthetic-main-dsn")
    transition(tmp_path, catalog, authorization_dir, execute=True)
    monkeypatch.setattr("local_app.runtime.OllamaRuntime", LocalRuntime)
    monkeypatch.setattr("local_app.config.Settings.from_env", lambda: SimpleNamespace(chat_model="gemma4:12b", database_url="synthetic-dsn"))
    budget = tmp_path / "existing-product-budget.sqlite3"
    budget.touch()
    output = tmp_path / "executed"
    report = run_check(authorization_dir / "external_answers_v2.json", output, execute=True, budget_ledger=budget)
    assert report["passed"]
    assert report["external_answers"] == report["local_fallback_answers"] == report["new_provider_calls"] == 6
    assert transport.open.call_count == 6
    assert len(report["outbound_requests"]) == 6
    assert all(row["new_provider_calls"] == 0 for row in report["cases"][:6])
    assert all(row["raw_response_text"] for row in report["outbound_requests"])
    assert len(list(output.glob("[EL][0-9][0-9].json"))) == 12
    with pytest.raises(ValueError, match="already used"):
        run_check(authorization_dir / "external_answers_v2.json", tmp_path / "attempted-repeat", execute=True, budget_ledger=budget)


def test_privacy_false_positive_is_a_failure_and_never_reserves_or_sends(tmp_path, monkeypatch):
    catalog, authorization_dir, _ = setup_transition(tmp_path)
    transport = Mock()
    transport.open.side_effect = lambda *args, **kwargs: Response()
    monkeypatch.setattr("research_memory.spiderssense.build_opener", lambda *args: transport)
    monkeypatch.setenv("SPIDERSSENSE_API_KEY", "synthetic-key-731")
    monkeypatch.setenv("LOCAL_GENERATION_RUN_BUDGET_PATH", "")
    monkeypatch.setenv("LOCAL_GENERATION_SCHEDULER_DSN", "synthetic-main-dsn")
    transition(tmp_path, catalog, authorization_dir, execute=True)
    monkeypatch.setattr(LocalRuntime, "chat_json", lambda *args, **kwargs: {"safe_to_send": True, "uncertain": False})
    monkeypatch.setattr("local_app.runtime.OllamaRuntime", LocalRuntime)
    monkeypatch.setattr("local_app.config.Settings.from_env", lambda: SimpleNamespace(chat_model="gemma4:12b", database_url="synthetic-dsn"))
    budget = tmp_path / "existing-budget.sqlite3"
    budget.touch()
    report = run_check(authorization_dir / "external_answers_v2.json", tmp_path / "failed", execute=True, budget_ledger=budget)
    assert report["passed"] is False
    escapes = [row for row in report["cases"] if row.get("error_type") == "UnexpectedExternalRoute"]
    assert {row["case_id"] for row in escapes} == {"L05", "L06"}
    assert all(row["budget_before"] == row["budget_after"] for row in escapes)
    assert report["new_provider_calls"] == transport.open.call_count == 6


def test_unknown_paid_outcome_remains_reserved_without_retry_or_local_replacement(tmp_path, monkeypatch):
    catalog, authorization_dir, _ = setup_transition(tmp_path)
    transport = Mock()
    transport.open.side_effect = [URLError("synthetic unknown outcome"), *[Response() for _ in range(5)]]
    monkeypatch.setattr("research_memory.spiderssense.build_opener", lambda *args: transport)
    monkeypatch.setenv("SPIDERSSENSE_API_KEY", "synthetic-key-731")
    monkeypatch.setenv("LOCAL_GENERATION_RUN_BUDGET_PATH", "")
    monkeypatch.setenv("LOCAL_GENERATION_SCHEDULER_DSN", "synthetic-main-dsn")
    receipt = transition(tmp_path, catalog, authorization_dir, execute=True)
    monkeypatch.setattr("local_app.runtime.OllamaRuntime", LocalRuntime)
    monkeypatch.setattr("local_app.config.Settings.from_env", lambda: SimpleNamespace(chat_model="gemma4:12b", database_url="synthetic-dsn"))
    budget = tmp_path / "existing-budget.sqlite3"
    budget.touch()
    report = run_check(authorization_dir / "external_answers_v2.json", tmp_path / "unknown", execute=True, budget_ledger=budget)
    assert report["passed"] is False
    assert report["external_answers"] == 5 and report["local_fallback_answers"] == 6
    assert report["new_provider_calls"] == transport.open.call_count == 6
    first_paid = report["cases"][6]
    assert first_paid["case_id"] == "E01" and first_paid["error_type"] == "ProviderRequestError"
    import sqlite3
    with sqlite3.connect(receipt["allocations"]["integration"]["ledger_path"]) as db:
        row = db.execute("SELECT status,reserved_usd,charged_usd FROM budget_calls WHERE id=1").fetchone()
    assert row[0] == "reserved_unknown" and row[1] == row[2]
