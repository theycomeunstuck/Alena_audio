"""Finite batch boundaries, never model-answer correctness by controlled stubs."""
from copy import deepcopy
import json

import pytest

from scripts.run_short_division_dialogue import (
    BatchBudget, BatchLimitError, execute, generate_case, load_cases, plan, public_arguments,
)


def test_gold_case_ids_and_categories_never_reach_generation():
    _, cases = load_cases()

    class Service:
        def answer(self, **kwargs):
            self.received = kwargs
            return {"answer": "controlled"}

    service = Service()
    case = deepcopy(cases[0])
    case["case_id"] = "PRIVATE_CASE_MARKER"
    case["category"] = "PRIVATE_CATEGORY_MARKER"
    case["private_gold"] = {"facts": ["PRIVATE_GOLD_MARKER"]}
    generate_case(service, case)
    packed = json.dumps(service.received, ensure_ascii=False)
    assert "PRIVATE_" not in packed
    assert service.received == case["public"]


@pytest.mark.parametrize("field", ["private_gold", "expected_answer", "profile", "known_identifiers"])
def test_unapproved_public_fields_fail_instead_of_reaching_rag(field):
    _, cases = load_cases()
    case = deepcopy(cases[0])
    case["public"][field] = "do not forward"
    with pytest.raises(ValueError, match="unsupported field"):
        public_arguments(case)


def test_public_projection_is_independent_of_mutation_and_preserves_memory_context():
    _, cases = load_cases()
    case = deepcopy(cases[-1])
    projected = public_arguments(case)
    projected["memory_context"]["text"] = "changed"
    projected["messages"][0]["content"] = "changed"
    assert case["public"]["memory_context"]["text"] != "changed"
    assert case["public"]["messages"][0]["content"] == "ага"
    assert case["public"]["memory_context"]["session_end_context"][0]["usage"] == "context_only"


def test_chat_budget_refuses_the_next_call_without_hidden_retry():
    budget = BatchBudget(now=lambda: 0.0, max_chat_calls=2)
    budget.admit("/api/tags")
    budget.admit("/api/embed")
    budget.admit("/api/chat")
    budget.admit("/api/chat")
    with pytest.raises(BatchLimitError, match="chat_budget_exhausted"):
        budget.admit("/api/chat")
    assert budget.chat_calls == 2
    assert budget.embedding_calls == 1
    assert budget.http_calls == 4


def test_elapsed_budget_stops_before_admitting_any_additional_http_request():
    times = iter([0.0, 1200.0])
    budget = BatchBudget(now=lambda: next(times))
    with pytest.raises(BatchLimitError, match="elapsed_budget_exhausted"):
        budget.admit("/api/chat")
    assert budget.chat_calls == budget.http_calls == 0


def test_dry_plan_does_not_initialize_runtime_store_or_environment(monkeypatch):
    import local_app.config
    import local_app.runtime
    import local_app.store

    def forbidden(*args, **kwargs):
        raise AssertionError("a dry run must not initialize services")

    monkeypatch.setattr(local_app.config.Settings, "from_env", forbidden)
    monkeypatch.setattr(local_app.runtime, "OllamaRuntime", forbidden)
    monkeypatch.setattr(local_app.store, "Store", forbidden)
    result = plan()
    assert result["dry_run"] and result["expected_cases"] == 12
    assert result["max_chat_calls"] == 32 and result["max_elapsed_seconds"] == 1200
    assert not result["learner_writes"] and not result["private_gold_enters_generation"]


def test_execute_refuses_existing_output_before_initializing_services(tmp_path, monkeypatch):
    import local_app.config

    marker = tmp_path / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("an existing output path must fail before service setup")

    monkeypatch.setattr(local_app.config.Settings, "from_env", forbidden)
    with pytest.raises(FileExistsError):
        execute(tmp_path)
    assert marker.read_text(encoding="utf-8") == "original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]
