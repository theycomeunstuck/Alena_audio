import asyncio
import json
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from scripts import benchmark_personal_graph_retrieval as bench


def test_database_name_file_only_accepts_named_synthetic_local_databases(tmp_path):
    good = tmp_path / "db.txt"
    good.write_text("nirs_personal_graph_demo_ab12\n", encoding="utf-8")
    assert bench.validate_db_name_file(good) == "nirs_personal_graph_demo_ab12"
    good.write_text("nirs_personal_graph_it_9b24803a", encoding="utf-8")
    assert bench.validate_db_name_file(good) == "nirs_personal_graph_it_9b24803a"
    for value in ("teachcopilot_local", "production", "demo_synthetic_01",
                  "nirs_personal_graph_it_a\nother", "nirs_personal_graph_demo_a-b"):
        good.write_text(value, encoding="utf-8")
        with pytest.raises(ValueError):
            bench.validate_db_name_file(good)


def test_common_source_top_k_balances_chunks_and_keeps_two_recent_lessons_separate():
    retrieved = {"hits": [
        {"kind": "semantic", "chunk_id": "c1", "lesson_ids": ["source-1"]},
        {"kind": "semantic", "chunk_id": "c2", "lesson_ids": ["source-2"]},
        {"kind": "semantic", "chunk_id": "c3", "lesson_ids": ["source-3"]},
        {"kind": "lexical", "chunk_id": "c2", "lesson_ids": ["source-2"]},
        {"kind": "lexical", "chunk_id": "c4", "lesson_ids": ["source-4"]},
        {"kind": "graphiti", "lesson_ids": ["graph-only"], "text": "graph fact"},
    ], "lesson_ids": ["recent-1", "recent-2", "source-1", "source-2", "graph-only"],
        "unresolved": [{"kind": "graphiti", "lesson_ids": []}]}
    bounded, diagnostics = bench._limit_source_candidate_state(
        retrieved, 2, ["recent-1", "recent-2"])
    assert [hit["chunk_id"] for hit in bounded["hits"]] == ["c1", "c2"]
    assert bounded["lesson_ids"] == ["recent-1", "recent-2", "source-1", "source-2"]
    assert len(bounded["lesson_ids"][:2]) == 2
    assert diagnostics["raw_archive_channel_counts"] == {"semantic": 3, "lexical": 2}
    assert diagnostics["raw_unique_source_chunk_candidate_count"] == 4
    assert diagnostics["final_source_chunk_candidate_count"] == 2
    assert diagnostics["graph_fact_hit_count"] == 1
    assert diagnostics["graph_hits_passed_to_hydration"] is False
    assert diagnostics["graph_hits_as_evidence"] is False
    assert diagnostics["graph_nodes_as_evidence"] is False
    assert len(diagnostics["retrieved_hits_before_sha256"]) == 64
    assert len(diagnostics["archive_source_hits_after_sha256"]) == 64
    assert "graph-only" not in bounded["lesson_ids"]
    assert not bounded["unresolved"]


def test_archive_proxy_forces_k_on_positional_and_keyword_search_limits():
    class SearchArchive:
        def __init__(self):
            self.limits = []

        def search(self, *args, **kwargs):
            limit = args[4] if len(args) >= 5 else kwargs["limit"]
            self.limits.append(limit)
            return [{"channel": "semantic", "chunk_id": f"c{i}"} for i in range(limit)]

    archive = SearchArchive()
    proxy = bench._ArchiveSearchTopK(archive, 2)
    assert len(proxy.search("run", "learner", "q", [0], 24, ["lesson"])) == 2
    assert len(proxy.search("run", "learner", "q", [0], limit=8)) == 2
    assert archive.limits == [2, 2]
    assert proxy.diagnostics()["archive_search_call_count"] == 2
    assert proxy.diagnostics()["archive_search_rows_by_channel"] == {"semantic": 4}


def test_source_top_k_wrapper_places_gate_between_retrieve_and_hydrate(monkeypatch):
    class FakeGraph:
        def __init__(self, _state_type):
            self.nodes, self.edges = {}, []

        def add_node(self, name, function):
            self.nodes[name] = function

        def add_edge(self, left, right):
            self.edges.append((left, right))

        def compile(self):
            return self

    graph_api = types.ModuleType("langgraph.graph")
    graph_api.END, graph_api.START = "END", "START"
    graph_api.StateGraph = FakeGraph
    langgraph = types.ModuleType("langgraph")
    langgraph.__path__ = []
    workflow_module = types.ModuleType("research_memory.personal_memory_lab.workflow")
    workflow_module.State = dict
    monkeypatch.setitem(sys.modules, "langgraph", langgraph)
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_api)
    monkeypatch.setitem(sys.modules, "research_memory.personal_memory_lab.workflow", workflow_module)

    class Workflow:
        async def retrieve(self, state):
            return state

        async def hydrate(self, state):
            return state

        async def select(self, state):
            return state

    workflow = Workflow()
    proxy = bench._ArchiveSearchTopK(FakeArchive(), 1)
    bench._install_source_top_k_graph(workflow, 1, ["recent-1", "recent-2"], proxy, "graph_hybrid")
    assert workflow.graph.edges == [
        ("START", "retrieve"), ("retrieve", "source_top_k"),
        ("source_top_k", "hydrate"), ("hydrate", "select"), ("select", "END")]
    state = {"hits": [
        {"kind": "semantic", "chunk_id": "c1", "lesson_ids": ["source"]},
        {"kind": "lexical", "chunk_id": "c2", "lesson_ids": ["source-2"]},
        {"kind": "graphiti", "lesson_ids": ["graph-only"]}],
        "lesson_ids": ["graph-only", "source", "source-2", "recent-1", "recent-2"],
        "unresolved": []}
    limited = asyncio.run(workflow.graph.nodes["source_top_k"](state))
    assert [hit["chunk_id"] for hit in limited["hits"]] == ["c1"]
    assert limited["lesson_ids"] == ["recent-1", "recent-2", "source"]
    assert workflow.source_top_k_diagnostics["graph_search_calls"] == 1
    assert workflow.source_top_k_diagnostics["graph_fact_hit_count"] == 1
    assert workflow.source_top_k_diagnostics["final_source_chunk_candidate_count"] == 1
    assert workflow.source_top_k_stage_timings["source_top_k_gate_seconds"] >= 0
    workflow.source_top_k_stage_timings = {}
    result = asyncio.run(workflow.graph.nodes["retrieve"](state))
    result = asyncio.run(workflow.graph.nodes["source_top_k"](result))
    result = asyncio.run(workflow.graph.nodes["hydrate"](result))
    result = asyncio.run(workflow.graph.nodes["select"](result))
    assert set(workflow.source_top_k_stage_timings) == {
        "retrieve_seconds", "source_top_k_gate_seconds", "hydrate_seconds", "select_seconds"}
    assert all(value >= 0 for value in workflow.source_top_k_stage_timings.values())


def test_library_provenance_records_runtime_versions_and_lock_hash():
    provenance = bench._library_provenance()
    assert "langgraph" in provenance["versions"]
    assert len(provenance["requirements_lock_sha256"]) == 64


def test_finish_status_stays_running_until_full_record_and_attempt_coverage():
    report = {"results": [{"status": "ok", "trials": [{"status": "ok"}]}],
              "scientific_result": False}
    bench._update_coverage(report, [{"query_id": "q1"}], repeats=2)
    bench._finish_status(report)
    assert report["status"] == "running"
    report["results"] = [{"status": "ok", "trials": [{"status": "ok"}] * 3}
                      for _ in range(len(bench.MODES))]
    bench._update_coverage(report, [{"query_id": "q1"}], repeats=2)
    bench._finish_status(report)
    assert report["status"] == "complete"


def test_terminal_setup_failures_are_not_misreported_as_in_progress():
    report = {"results": [{"status": "setup_error", "trials": []}
                           for _ in range(len(bench.MODES))]}
    bench._update_coverage(report, [{"query_id": "q1"}], repeats=1)
    bench._finish_status(report)
    assert report["coverage"]["query_attempts_recorded"] == 0
    assert report["coverage"]["query_attempts_terminal_setup_failures"] == len(bench.MODES) * 2
    assert report["status"] == "partial_error"


def test_final_provenance_compares_code_and_model_identities(monkeypatch):
    identities = {"chat": {"name": "chat", "digest": "a"},
                  "embedding": {"name": "embed", "digest": "b"}}
    report = {"code_provenance": bench._code_provenance(), "model": {
        "current_model_identities_before": identities}}
    monkeypatch.setattr(bench, "_code_provenance", lambda: report["code_provenance"])
    bench._finalize_provenance(report, object(), lambda _: identities)
    assert report["model"]["current_model_identities_after"] == identities
    assert report["run_stability"] == {"code_hashes_stable": True,
        "model_identities_stable": True, "stable": True, "model_identity_error_type": None}

    report["model"]["current_model_identities_before"] = identities
    bench._finalize_provenance(report, object(), lambda _: {**identities, "chat": {"digest": "changed"}})
    assert report["run_stability"]["stable"] is False


def test_request_validation_percentiles_and_empty_gold_status(tmp_path):
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps({"query_id": "q1", "question": "triangle?",
                                    "expected_source_ids": ["s1", "s1"]}) + "\n", encoding="utf-8")
    loaded = bench.load_requests(requests)
    assert loaded[0]["expected_source_ids"] == ["s1"]
    assert bench.percentile([1, 2, 3], 50) == 2
    assert bench.percentile([1, 2, 3], 95) == pytest.approx(2.9)
    assert bench.source_recall(["s2"], [])["source_recall_status"] == "no_gold_sources"
    assert bench.source_recall(["s2"], ["s1"])["missing_expected_source_ids"] == ["s1"]


def test_count_policy_keeps_application_bytes_separate_from_local_token_count():
    model = FakeRuntime(16384)
    assert bench._CountPolicy(model, "bytes").count("я") == len("я".encode("utf-8"))
    assert bench._CountPolicy(model, "tokens").count("one two") == 2


def test_model_identity_resolution_uses_exact_aliases_and_rejects_ambiguous_digests(monkeypatch):
    import urllib.request
    from local_app.personal_graph_index import resolve_model_identities

    class TagsResponse:
        def __init__(self, models):
            self.models = models

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"models": self.models}).encode()

    runtime = SimpleNamespace(base_url="http://127.0.0.1:11435", model="gemma4:12b",
                              runtime=SimpleNamespace(settings=SimpleNamespace(embedding_model="bge-m3")))
    tags = [
        {"name": "gemma4:12b", "model": "gemma4:12b", "digest": "a" * 64},
        {"name": "bge-m3:latest", "model": "bge-m3:latest", "digest": "b" * 64},
    ]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: TagsResponse(tags))
    identities = bench._model_identities(runtime)
    assert identities["embedding"] == {"name": "bge-m3", "digest": "b" * 64}

    bad_tags = [tags[0], {"name": "bge-m3:other", "model": "bge-m3:other", "digest": "c" * 64}]
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: TagsResponse(bad_tags))
    with pytest.raises(ValueError, match="digest unavailable or ambiguous"):
        bench._model_identities(runtime)


class FakeArchive:
    def __init__(self):
        self._lessons = [{"lesson_id": "l1", "observed_on": "2025-01-01", "turns": [
            {"turn_id": "s1", "role": "student", "text": "triangle"},
        ]}]

    def lessons(self, run_id, learner_id):
        return self._lessons

    def chunks(self, run_id, learner_id):
        return [{"lesson_id": "l1", "chunk_id": "l1:c0001", "content": {"turns": self._lessons[0]["turns"]}}]

    def index_references(self, run_id, learner_id, backend):
        return [{"lesson_id": "l1", "metadata": {}}] if backend == "graphiti" else []


def test_vector_compatibility_report_checks_all_scope_ids_hashes_and_current_models(tmp_path):
    import hashlib
    from research_memory.personal_memory_lab.archive import canonical
    from research_memory.personal_memory_lab.backends import lesson_text

    content = {"lesson_id": "l1", "observed_on": "2025-01-01", "topic_ids": ["math"],
               "turns": [{"turn_id": "s1", "role": "student", "text": "triangle"}]}
    chunks = [{"chunk_id": "l1:c0001", "lesson_id": "l1", "position": 0, "content": content}]
    archive = FakeArchive()
    archive.chunks = lambda run, learner: chunks if (run, learner) == ("run", "learner") else []
    identities = {"chat": {"name": "gemma4:12b", "digest": "a" * 64},
                  "embedding": {"name": "bge-m3", "digest": "b" * 64}}
    row = {"run_id": "run", "learner_id": "learner", "chunk_id": "l1:c0001",
           "source_hash": hashlib.sha256(lesson_text(content).encode()).hexdigest(),
           "content_hash": hashlib.sha256(canonical(content).encode()).hexdigest(),
           "stored_vector_hash": "c" * 64, "fresh_vector_hash": "c" * 64,
           "max_abs_diff": 0.0, "cosine": 1.0, "status": "pass"}
    report = {"status": "pass", "coverage": {"expected": 1, "found": 1, "processed": 1},
              "scope_expected": {"learner": 1}, "chunks": [row],
              "model_identities_before": identities, "model_identities_after": identities}
    path = tmp_path / "compat.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = bench._validate_vector_compatibility_report(path, identities, archive, "run", "learner", chunks)
    assert result["status"] == "pass"
    assert result["validated_chunk_count"] == 1
    assert result["index_embedding_identity"] == "unknown"

    bad = dict(report)
    bad["chunks"] = [{**row, "source_hash": "0" * 64}]
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes do not match"):
        bench._validate_vector_compatibility_report(path, identities, archive, "run", "learner", chunks)


class FakeRuntime:
    def __init__(self, context):
        self.context, self.model = context, "gemma4:12b"
        self.base_url = "http://127.0.0.1:11435"

        class TagsResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"models": [
                    {"name": "gemma4:12b", "model": "gemma4:12b", "digest": "chat-digest"},
                    {"name": "bge-m3:latest", "model": "bge-m3:latest", "digest": "embed-digest"},
                ]}).encode()

        self.opener = SimpleNamespace(open=lambda *args, **kwargs: TagsResponse())

    def embed(self, _text):
        raise AssertionError("benchmark fixture must not embed")

    def count(self, text):
        return len(text.split())


class FakeGraph:
    def __init__(self):
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


def test_benchmark_runs_five_modes_with_same_budget_and_preserves_exact_memory(tmp_path):
    output = tmp_path / "benchmark.json"
    calls = []
    graph = FakeGraph()

    class FakeWorkflow:
        async def search(self, run_id, learner_id, question, *, mode, budget_tokens):
            calls.append((mode, budget_tokens))
            return {"lessons": [{"lesson_id": "l1", "turns": [
                        {"turn_id": "s1", "role": "student", "text": "triangle"}]}],
                    "source_ids": ["s1"], "memory": '{"exact": "triangle"}',
                    "hits": [{"kind": "semantic"}, {"kind": "lexical"}], "skipped": []}

    def workflow_factory(base, archive, model, graphiti, *, source_order, retrieval_mode):
        return FakeWorkflow()

    def graph_factory(runtime):
        return graph

    requests = [{"query_id": "q1", "question": "triangle?", "expected_source_ids": ["s1"]}]
    settings = SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                               personal_graph_password_file=str(tmp_path / "missing-passfile"))
    result = asyncio.run(bench.run_benchmark(
        database_url="unused", run_id="run", learner_id="learner", requests=requests,
        output_path=output, context=16384, repeats=1, budget_kind="bytes", budget=4000,
        runtime=FakeRuntime(16384), archive=FakeArchive(), graph_factory=graph_factory,
        workflow_factory=workflow_factory, database_name="nirs_personal_graph_demo_synthetic_01",
        settings=settings, base_workflow=object,
        model_identity_resolver=lambda runtime: {
            "chat": {"name": "gemma4:12b", "digest": "chat-digest"},
            "embedding": {"name": "bge-m3", "digest": "embed-digest"}}))
    assert [row["mode"] for row in result["results"]] == list(bench.MODES)
    assert calls == [("graphiti", 4000)] * (len(bench.MODES) * 2)
    assert graph.close_calls == 1
    assert all(row["source_recall"] == 1.0 for row in result["results"])
    assert all(row["trials"][0]["memory"] == '{"exact": "triangle"}' for row in result["results"])
    assert all(row["budget_kind"] == "bytes" for row in result["results"])
    assert output.exists()
    assert result["scientific_result"] is False
    assert result["application_context"] == {"shared_recent_lesson_ids": ["l1"],
                                              "shared_across_modes": True}
    assert result["archive_metadata"]["lessons_content_sha256"] == bench._json_sha256(FakeArchive().lessons("run", "learner"))
    assert result["model"]["chat"]["digest"] == "chat-digest"
    assert result["model"]["embedding"]["digest"] == "embed-digest"
    assert result["model"]["identity_role"] == "current_runtime"
    assert result["model"]["index_embedding_identity"] == "unknown"
    assert result["vector_compatibility"]["status"] == "not_supplied"
    assert result["timing_semantics"]["first"] == "first sequential query in this process; not a cold service restart"
    with pytest.raises(FileExistsError):
        asyncio.run(bench.run_benchmark(
            database_url="unused", run_id="run", learner_id="learner", requests=requests,
            output_path=output, context=16384, repeats=1, budget_kind="bytes", budget=4000,
            runtime=FakeRuntime(16384), archive=FakeArchive(), graph_factory=graph_factory,
            workflow_factory=workflow_factory, base_workflow=object))


def test_benchmark_refuses_tokens_budget_over_context_window(tmp_path):
    with pytest.raises(ValueError, match="working window"):
        asyncio.run(bench.run_benchmark(
            database_url="unused", run_id="r", learner_id="l", requests=[],
            output_path=tmp_path / "out.json", context=16384, budget_kind="tokens", budget=14337,
            runtime=FakeRuntime(16384), archive=FakeArchive()))


def test_query_failures_are_recorded_and_other_modes_continue(tmp_path):
    output = tmp_path / "partial.json"
    visited = []
    graph = FakeGraph()

    class ContextWindowOverflow(Exception):
        pass

    class SometimesFailWorkflow:
        def __init__(self, mode):
            self.mode = mode

        async def search(self, run_id, learner_id, question, *, mode, budget_tokens):
            visited.append(self.mode)
            if self.mode == "lexical":
                raise ContextWindowOverflow("token counter rejected the memory")
            return {"lessons": FakeArchive().lessons(run_id, learner_id), "source_ids": ["s1"],
                    "memory": "text", "hits": [{"kind": self.mode}], "skipped": []}

    def workflow_factory(base, archive, model, graphiti, *, source_order, retrieval_mode):
        return SometimesFailWorkflow(retrieval_mode)

    report = asyncio.run(bench.run_benchmark(
        database_url="unused", run_id="run", learner_id="learner",
        requests=[{"query_id": "q", "question": "triangle", "expected_source_ids": ["s1"]}],
        output_path=output, context=16384, repeats=1, budget_kind="tokens", budget=4000,
        runtime=FakeRuntime(16384), archive=FakeArchive(), graph_factory=lambda runtime: graph,
        workflow_factory=workflow_factory, base_workflow=object))
    assert report["status"] == "partial_error"
    assert len(visited) == len(bench.MODES) * 2
    failed = next(row for row in report["results"] if row["mode"] == "lexical")
    assert failed["status"] == "error"
    assert failed["attempts_planned"] == failed["attempts_completed"] == 2
    assert failed["failed_attempt_count"] == 2
    assert all(item["error_type"] == "ContextWindowOverflow" for item in failed["trials"])
    assert report["coverage"]["query_attempts_planned"] == 10
    assert report["coverage"]["query_attempts_recorded"] == 10
    assert report["coverage"]["query_attempts_failed"] == 2
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status"] == "partial_error"
    assert persisted["results"][-1]["mode"] == "graph_hybrid"
    assert report["timing_semantics"]["includes_token_count_probe"] is True
    assert "latency is included" in report["timing_semantics"]["token_probe_note"]


def test_cli_imports_project_modules_when_run_from_outside_project_without_gpu(tmp_path):
    script = (bench.PROJECT_ROOT / "scripts" / "benchmark_personal_graph_retrieval.py").resolve()
    dbfile = tmp_path / "database.txt"
    dbfile.write_text("invalid_database_name", encoding="utf-8")
    request_file = tmp_path / "requests.jsonl"
    request_file.write_text("{}\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), "--db-name-file", str(dbfile), "--run-id", "r",
                             "--learner-id", "l", "--requests", str(request_file), "--output",
                             str(tmp_path / "out.json"), "--context", "16384"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "database name file" in result.stderr
    assert "No module named 'local_app'" not in result.stderr
