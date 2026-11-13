from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import uuid

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_personal_graph_selection.py"
SPEC = importlib.util.spec_from_file_location("personal_graph_selection_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(diagnostic)


def test_scenario_has_old_question_help_answer_and_unrelated_recent_sessions():
    scenario = diagnostic.build_scenario()
    query = scenario["query"].casefold()
    outcome = scenario["target_outcome"].casefold()
    old = scenario["old_lesson"]

    assert len(scenario["recent_lessons"]) == 5
    assert all(item["kind"] == "recent_unrelated" for item in scenario["recent_lessons"])
    assert old["turns"][0]["assistant"] == scenario["target_question"]
    assert scenario["target_outcome"] in old["turns"][1]["student"]
    assert outcome not in query
    assert all(part not in query for part in ("25/40", "30/40", "равными знаменателями"))
    assert scenario["source_count_estimate"] == 12

    synthetic_payload = json.dumps(scenario, ensure_ascii=False)
    assert len(synthetic_payload.encode("utf-8")) > 12_000


def test_preflight_uses_real_chunks_and_reports_whether_selection_budget_is_exceeded():
    pytest.importorskip("psycopg2")
    summary = diagnostic.scenario_preflight(diagnostic.build_scenario(), selection_budget=10000)
    assert summary["lesson_count"] == 6
    assert summary["student_source_count"] == 12
    assert summary["source_bytes_exceed_selection_budget"] is True
    assert 0 < summary["fragment_count"] <= 24
    assert summary["estimated_prefilter_calls"] == summary["fragment_count"]


def test_fresh_database_check_uses_named_rows_and_rejects_existing_data():
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def execute(self, sql): self.sql = sql
        def fetchone(self):
            if "current_database" in self.sql:
                return {"db_name": "nirs_personal_graph_it_test"}
            return {"row_count": int("memory_sources" in self.sql and self.populated)}
        populated = False

    class Connection:
        def __init__(self, populated): self.cursor_obj = Cursor(); self.cursor_obj.populated = populated
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def cursor(self): return self.cursor_obj

    class Store:
        def __init__(self, populated=False): self.populated = populated
        def _connect(self): return Connection(self.populated)

    assert diagnostic.assert_fresh_synthetic_database(Store(), "nirs_personal_graph_it_test")["fresh"]
    with pytest.raises(ValueError, match="must_be_fresh"):
        diagnostic.assert_fresh_synthetic_database(Store(True), "nirs_personal_graph_it_test")
    with pytest.raises(ValueError, match="does_not_match"):
        diagnostic.assert_fresh_synthetic_database(Store(), "nirs_personal_graph_it_other")


def test_search_report_keeps_archive_channels_hits_and_skip_reasons_separate():
    fake_workflow = {
        "hits": [
            {"kind": "semantic", "lesson_ids": ["old"], "chunk_id": "chunk-s", "score": 0.91},
            {"kind": "lexical", "lesson_ids": ["old"], "chunk_id": "chunk-l", "score": 2.0},
            {"kind": "graphiti", "lesson_ids": ["old"], "fact": "derived claim"},
        ],
        "unresolved": [{"kind": "graphiti", "fact": "unanchored claim"}],
        "lessons": [{"lesson_id": "old", "fragment_id": "chunk-s", "observed_on": "2026-10-01",
                     "turns": [{"turn_id": "source-old", "role": "student", "text": "synthetic text"}]}],
        "source_ids": ["source-old"],
        "skipped": [{"lesson_id": "new", "fragment_id": "chunk-big",
                      "reason": "memory_budget_or_context_exceeded"}],
    }

    report = diagnostic.search_report(fake_workflow, phase="teacher_prepare", question="neutral query",
                                      selection_budget_bytes=3000)

    assert report["channels"] == {"graphiti": 1, "lexical": 1, "semantic": 1}
    assert [hit["kind"] for hit in report["hits"]] == ["semantic", "lexical", "graphiti"]
    assert report["workflow_budget_selected_source_ids"] == ["source-old"]
    assert report["workflow_skipped"][0]["reason"] == "memory_budget_or_context_exceeded"
    assert report["selection_budget"] == {"kind": "application_utf8_bytes", "limit": 3000}
    assert report["hydrated_candidate_turns"][0]["turns"][0]["turn_id"] == "source-old"


def test_observer_captures_fake_search_and_application_context_independently():
    class FakeWorkflow:
        async def search(self, run_id, learner_id, question, *, mode, budget_tokens, allowed_source_ids=None):
            return {"hits": [{"kind": "lexical", "lesson_ids": ["lesson"], "chunk_id": "c1"}],
                    "lessons": [], "source_ids": [], "skipped": []}

    observer = diagnostic.SearchObserver()
    fake_workflow = FakeWorkflow()
    found = asyncio.run(fake_workflow.search("run", "child", "neutral", mode="graphiti",
                                             budget_tokens=3000, allowed_source_ids={"approved-id"}))
    observer.record_search("chat", found, "neutral", {"approved-id"})
    observer.record_context("chat", {"text": "{}", "evidence_ids": ["approved-id"],
                                      "sources": [{"source_id": "approved-id", "role": "student",
                                                   "text": "synthetic"}], "selection_budget": {
                                          "kind": "utf8_bytes_upper_bound", "limit": 4000, "used": 100}})

    event = observer.events[0]
    assert event["channels"] == {"lexical": 1}
    assert event["allowed_source_ids"] == ["approved-id"]
    assert event["application_context"]["evidence_ids"] == ["approved-id"]
    assert event["application_context"]["selection_budget"]["limit"] == 4000


def test_approval_boundary_reports_missing_target_without_expanding_allowed_ids():
    boundary = diagnostic.approval_boundary_report(
        "old-target", approved_ids={"approved-a", "approved-b"}, chat_ids={"approved-b"})
    assert boundary["chat_ids_within_approved_set"] is True
    assert boundary["target_source_in_approved_set"] is False
    assert boundary["target_source_in_chat_context"] is False
    assert boundary["target_missing_from_approval_limit"] is True

    foreign = diagnostic.approval_boundary_report(
        "old-target", approved_ids={"approved-a"}, chat_ids={"approved-a", "foreign-source"})
    assert foreign["chat_ids_within_approved_set"] is False
    assert foreign["chat_evidence_ids"] == ["approved-a", "foreign-source"]


def test_selected_context_and_literal_checks_do_not_claim_semantic_quality():
    context = diagnostic.selected_context_report({
        "text": "synthetic", "evidence_ids": ["target"],
        "sources": [{"source_id": "target", "role": "student", "text": diagnostic.TARGET_OUTCOME}],
        "dialogue_context": [{"source_id": "target", "usage": "context_only",
                              "text": diagnostic.TARGET_QUESTION}],
        "selection_version": "app_balanced_sources_v1",
        "selection_budget": {"kind": "utf8_bytes_upper_bound", "limit": 4000, "used": 900},
    })
    search = {"hydrated_candidate_turns": [], "workflow_budget_selected_source_ids": []}
    checks = diagnostic.mechanical_presence_checks("target", diagnostic.TARGET_OUTCOME,
        diagnostic.TARGET_QUESTION, search, context)

    assert checks["target_turn_in_application_evidence"] is True
    assert checks["target_literal_text_in_selected_student_source"] is True
    assert checks["assistant_question_literal_in_selected_context_only"] is True
    assert checks["interpretation"].endswith("not_semantic_quality")


def test_database_name_and_loopback_guards_are_fail_closed():
    assert diagnostic.validate_database_name("nirs_personal_graph_it_ab12") == "nirs_personal_graph_it_ab12"
    with pytest.raises(ValueError, match="synthetic_test_database"):
        diagnostic.validate_database_name("teachcopilot_local")

    psycopg2 = pytest.importorskip("psycopg2.extensions")
    local = psycopg2.make_dsn("dbname=nirs_personal_graph_it_ab12 host=127.0.0.1 user=synthetic")
    assert diagnostic.validate_local_postgres_dsn(local)["loopback_only"] is True
    remote = psycopg2.make_dsn("dbname=nirs_personal_graph_it_ab12 host=192.0.2.9 user=synthetic")
    with pytest.raises(ValueError, match="loopback_only"):
        diagnostic.validate_local_postgres_dsn(remote)
    with pytest.raises(ValueError, match="loopback_only"):
        diagnostic.validate_local_postgres_dsn(local, pg_hostaddr="192.0.2.10")

    assert diagnostic.validate_local_ollama_url("http://127.0.0.1:11435") is True
    with pytest.raises(ValueError, match="loopback"):
        diagnostic.validate_local_ollama_url("http://example.invalid:11435")


def test_output_path_is_project_local_and_exclusive():
    output = diagnostic.PROJECT_ROOT / ".local" / ("diagnostic-test-" + uuid.uuid4().hex + ".json")
    try:
        assert diagnostic.validate_output_path(output) == output.resolve()
        diagnostic.write_json_exclusive(output, {"synthetic": True})
        with pytest.raises(FileExistsError):
            diagnostic.validate_output_path(output)
        with pytest.raises(FileExistsError):
            diagnostic.write_json_exclusive(output, {"overwrite": True})
        assert json.loads(output.read_text(encoding="utf-8")) == {"synthetic": True}
    finally:
        output.unlink(missing_ok=True)

    outside = Path("C:/tmp/diagnostic-outside-project.json")
    with pytest.raises(ValueError, match="inside_project"):
        diagnostic.validate_output_path(outside)


def test_possible_refusal_is_only_a_heuristic():
    assert diagnostic.possible_refusal_heuristic("Данных недостаточно.") is True
    assert diagnostic.possible_refusal_heuristic("Возможно, я не уверен.") is False
