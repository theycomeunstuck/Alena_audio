"""Offline contracts for the finite operation batch; no model or database calls."""
from copy import deepcopy
from contextlib import contextmanager
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_verified_learning_operations_v2 import (
    EXPECTED_CASES, FIXTURE, GENERATION_PHASES, MAX_GENERATION_CALLS, ROOT,
    FiniteBatchBudgetError, bounded_public_store_class, counter_delta,
    fixture_cases, fixture_path, new_counters, public_material_snapshot_hash,
    source_hashes, traced_runtime_class,
)


def fixed_fixture():
    return json.loads((ROOT / FIXTURE).read_text(encoding="utf-8"))


def test_new_fixed_fixture_has_detached_public_inputs_and_never_projects_gold():
    prepared = fixture_cases(fixed_fixture())
    assert len(prepared) == EXPECTED_CASES == 24
    for case, public in prepared:
        assert set(public) == {"messages", "subject_id", "topic_id"}
        assert public == case["public"]
        public["messages"][-1]["content"] += " Detached mutation."
        assert public != case["public"]
        assert all(key not in public for key in ("case_id", "category", "private_gold"))


def test_private_expectation_changes_cannot_change_public_application_input():
    fixture = fixed_fixture()
    changed = deepcopy(fixture)
    for case in changed["cases"]:
        case["private_gold"] = {
            "expected_route": "normal_model", "mechanical_checks": [],
            "allowed_answers": ["PRIVATE_CANARY_NEVER_GENERATE"],
            "forbidden_claims": ["ANOTHER_PRIVATE_CANARY"],
        }
        case["category"] = "private_category_canary"
    assert [public for _, public in fixture_cases(changed)] == [
        public for _, public in fixture_cases(fixture)]
    assert "PRIVATE_CANARY" not in json.dumps([public for _, public in fixture_cases(changed)])


@pytest.mark.parametrize("count", [0, 8, 23, 25])
def test_fixed_denominator_cannot_be_shortened_padded_or_selected_as_smoke(count):
    fixture = fixed_fixture()
    fixture["cases"] = fixture["cases"][:count]
    if count == 25:
        fixture["cases"].append(deepcopy(fixture["cases"][0]))
    with pytest.raises(ValueError, match="exactly 24"):
        fixture_cases(fixture)
    if count != 24:
        with pytest.raises(ValueError, match="exactly 24"):
            fixture_cases(fixed_fixture(), count)


def test_public_injection_duplicate_ids_and_inconsistent_numeric_gold_are_rejected():
    extra = fixed_fixture()
    extra["cases"][0]["public"]["private_gold"] = {"expected_route": "division_check"}
    with pytest.raises(ValueError, match="public fields"):
        fixture_cases(extra)
    duplicate = fixed_fixture()
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    with pytest.raises(ValueError, match="unique safe"):
        fixture_cases(duplicate)
    inconsistent = fixed_fixture()
    numeric = next(check for case in inconsistent["cases"]
                   for check in case["private_gold"]["mechanical_checks"]
                   if check["kind"] in {"division_fact", "division_places"})
    numeric["quotient"] += 1
    with pytest.raises(ValueError, match="arithmetically inconsistent"):
        fixture_cases(inconsistent)
    stdout_fixture = fixed_fixture()
    stdout = next(check for case in stdout_fixture["cases"]
                  for check in case["private_gold"]["mechanical_checks"]
                  if check["kind"] == "python_stdout")
    stdout["stdout"] = ["not literal text"]
    with pytest.raises(ValueError, match="literal string"):
        fixture_cases(stdout_fixture)


def test_project_fixture_path_and_both_runner_sources_are_frozen():
    assert fixture_path(Path(FIXTURE)) == (ROOT / FIXTURE).resolve()
    with pytest.raises(ValueError, match="project-relative"):
        fixture_path((ROOT / FIXTURE).resolve())
    with pytest.raises(ValueError, match="inside the project"):
        fixture_path(Path("../private-fixture.json"))
    hashes = source_hashes()
    assert FIXTURE in hashes
    assert "scripts/run_verified_learning_operations_v2.py" in hashes
    assert "scripts/run_verified_learning_tools_v1.py" in hashes
    assert "local_app/runtime.py" in hashes and "local_app/rag.py" in hashes
    assert all(len(digest) == 64 for digest in hashes.values())


class FakeSelectionError(ValueError):
    raw_response = {"message": {"content": "a malformed selection"}}


class OfflineRuntime:
    """A transport-shaped stub which cannot open network or database connections."""
    def __init__(self):
        self.session = SimpleNamespace(last_post=None)
        self.generation_scheduler = None
        self.transported = []
        self.fail_selection = False

    def _post(self, endpoint, payload):
        self.transported.append((endpoint, deepcopy(payload)))
        raw = b'{"message":{"content":"escaped\\nvalue"}, "spacing":  true}'
        self.session.last_post = {
            "status_code": 200, "raw_body": raw.decode("utf-8"),
            "raw_body_base64": base64.b64encode(raw).decode("ascii"),
        }
        return {"message": {"content": "escaped\nvalue"}, "prompt_eval_count": 3, "eval_count": 2}

    def resolve_chat_query(self, *args, **kwargs):
        self.scope_arguments = (args, kwargs)
        self._post("/api/chat", {"stage": "scope"})
        return {"in_scope": True, "uses_prior_context": False, "search_query": "literal learner question"}

    def select_learning_operation(self, *args, **kwargs):
        self.selection_arguments = (args, kwargs)
        self._post("/api/chat", {"stage": "selection"})
        if self.fail_selection:
            raise FakeSelectionError("invalid operation contract")
        return {"operation": "none", "reason": "unsupported request"}

    def chat(self, *args, **kwargs):
        return self._post("/api/chat", {"stage": "answer"})


def offline_trace(tmp_path, *, enabled=True):
    current = {"phase": "answer_generation", "case_id": "offline_case"}
    counters = new_counters()
    calls = []
    runtime_class = traced_runtime_class(
        OfflineRuntime, current=current, counters=counters, calls=calls,
        output_dir=tmp_path, remaining=lambda: 900, generation_enabled=enabled)
    return runtime_class(), current, counters, calls


def test_selection_return_and_normal_answer_have_separate_calls_and_raw_bodies(tmp_path):
    runtime, current, counters, calls = offline_trace(tmp_path)
    before = dict(counters)
    runtime.resolve_chat_query("current", subject_id="math", topic_id="division")
    decision = runtime.select_learning_operation("literal current", "math", [{"literal": "candidate"}], sentinel="forwarded")
    runtime._post("/api/embed", {"input": ["query"]})
    runtime.chat([{"role": "user", "content": "literal current"}])
    assert current["phase"] == "answer_generation"
    assert runtime.selection_arguments == (("literal current", "math", [{"literal": "candidate"}]), {"sentinel": "forwarded"})
    assert current["operation_selection"]["decision"] == decision
    assert current["operation_selection"]["intent_verified"] is False
    decision["operation"] = "mutated after tracing"
    assert current["operation_selection"]["decision"]["operation"] == "none"
    delta = counter_delta(counters, before)
    assert delta == {"generation_calls": 3, "scope_resolution_calls": 1,
                     "operation_selection_calls": 1, "answer_generation_calls": 1, "embedding_calls": 1}
    assert [row["purpose"] for row in calls] == list(GENERATION_PHASES)
    saved = [json.loads(line) for line in (tmp_path / "ollama_http.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved == calls
    for row in saved:
        assert base64.b64decode(row["http"]["raw_body_base64"]).decode("utf-8") == row["http"]["raw_body"]
        assert '"spacing":  true' in row["http"]["raw_body"]
        assert row["response"]["message"]["content"] == "escaped\nvalue"


def test_caught_selection_error_survives_fallback_and_restores_previous_phase(tmp_path):
    runtime, current, counters, calls = offline_trace(tmp_path)
    runtime.fail_selection = True
    current["phase"] = "scope_resolution"
    try:
        runtime.select_learning_operation("current", "math", [])
    except FakeSelectionError:
        pass  # RAG is allowed to handle a selection error and answer normally.
    assert current["phase"] == "scope_resolution"
    current["phase"] = "answer_generation"
    runtime.chat([])
    selection = current["operation_selection"]
    assert selection["status"] == "error" and selection["decision"] is None
    assert selection["error"]["type"] == "FakeSelectionError"
    assert selection["raw_response"] == FakeSelectionError.raw_response
    assert selection["intent_verified"] is False
    assert counters["generation_calls"] == 2
    assert counters["operation_selection_calls"] == 1
    assert counters["answer_generation_calls"] == 1
    assert [call["purpose"] for call in calls] == ["operation_selection", "answer_generation"]
    assert all(call["http"]["raw_body_base64"] for call in calls)


def test_preflight_cannot_transport_a_generation_even_if_an_application_method_tries(tmp_path):
    runtime, current, counters, calls = offline_trace(tmp_path, enabled=False)
    with pytest.raises(RuntimeError, match="preflight_generation_forbidden"):
        runtime.select_learning_operation("current", "math", [])
    assert runtime.transported == [] and calls == []
    assert counters == new_counters()
    assert current["phase"] == "answer_generation"
    assert current["operation_selection"]["error"]["message"] == "preflight_generation_forbidden"
    assert not (tmp_path / "ollama_http.jsonl").exists()


def test_generation_cap_counts_all_three_phases_and_does_not_charge_embeddings(tmp_path):
    runtime, current, counters, calls = offline_trace(tmp_path)
    for number in range(MAX_GENERATION_CALLS):
        current["phase"] = GENERATION_PHASES[number % 3]
        runtime._post("/api/chat", {"number": number})
    runtime._post("/api/embed", {"input": ["query"]})
    with pytest.raises(FiniteBatchBudgetError, match="generation_budget_exhausted"):
        runtime._post("/api/chat", {"number": "forbidden extra attempt"})
    assert counters["generation_calls"] == MAX_GENERATION_CALLS == 84
    assert sum(counters[phase + "_calls"] for phase in GENERATION_PHASES) == 84
    assert counters["embedding_calls"] == 1
    assert len(calls) == 84 and len(runtime.transported) == 85
    assert counter_delta(counters, new_counters()) == counters
    corrupted = dict(counters, generation_calls=85)
    with pytest.raises(RuntimeError, match="inconsistent_generation_phase_counters"):
        counter_delta(corrupted, new_counters())


class OfflinePublicStore:
    def __init__(self, dsn=None):
        self.dsn = dsn
        self.connection_dsns = []
        self.rows = [{"id": "public-1", "content": "A lesson", "source_version": "1",
                      "embedding_model": "embedding@fixed", "embedding": "[0.1,0.2]",
                      "embedding_dimension": 2, "search_document": "'lesson':2"}]

    @contextmanager
    def _connect(self):
        self.connection_dsns.append(self.dsn)
        yield self

    @contextmanager
    def cursor(self):
        yield self

    def execute(self, query):
        self.query = query

    def fetchall(self):
        return deepcopy(self.rows)


@pytest.mark.parametrize("field,value", [
    ("content", "A changed lesson"), ("source_version", "2"),
    ("embedding", "[0.1,0.3]"), ("embedding_dimension", 3),
    ("embedding_model", "embedding@changed"), ("search_document", "'changed':2"),
])
def test_retrieval_snapshot_detects_content_vector_dimension_and_search_changes(field, value):
    store = OfflinePublicStore()
    original = public_material_snapshot_hash(store, lambda: 900)
    assert public_material_snapshot_hash(store, lambda: 900) == original
    store.rows[0][field] = value
    assert public_material_snapshot_hash(store, lambda: 900) != original


def test_readonly_connections_recompute_statement_timeout_from_remaining_budget():
    remaining_values = iter((1000, 899, 3, 2))
    original_dsn = "postgresql://localhost/offline_public"
    bounded = bounded_public_store_class(
        OfflinePublicStore, make_dsn=lambda dsn, **kwargs: {"original": dsn, **kwargs},
        database_url=original_dsn, base_options="-c search_path=teachcopilot",
        remaining=lambda: next(remaining_values))
    store = bounded(original_dsn)
    with store._connect():
        pass
    with store._connect():
        pass
    first, second = store.connection_dsns
    assert first["original"] == second["original"] == original_dsn
    assert "statement_timeout=900000" in first["options"]
    assert "statement_timeout=3000" in second["options"]
    for connection in store.connection_dsns:
        assert "default_transaction_read_only=on" in connection["options"]
        assert "lock_timeout=5000" in connection["options"]
        assert "search_path=teachcopilot" in connection["options"]
        assert connection["connect_timeout"] == 5


def test_elapsed_budget_is_checked_again_after_public_query():
    checks = []

    def remaining():
        checks.append(True)
        if len(checks) == 2:
            raise FiniteBatchBudgetError("finite_learning_operations_elapsed_budget_exhausted")
        return 900

    with pytest.raises(FiniteBatchBudgetError, match="elapsed_budget_exhausted"):
        public_material_snapshot_hash(OfflinePublicStore(), remaining)
