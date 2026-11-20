"""Round-3 live-quality run: per-history isolation, real delivery trace.

Prepares a `teachcopilot_b_smoke_<12 hex>` PostgreSQL database and creates
**one independent learner/student scope per frozen history** (N1 repair):
each history is loaded into its own learner account with a declared initial
profile (the unrelated demo card and its interests are never imported), so
no cross-history source is eligible and the source-key mapping stays
per-history. Questions are asked through the real local API with the
requesting student account scoped to the question's own history.

The unconditional standalone recall probe of the round-2 harness is removed
from the end-to-end answer measurement (N2 repair): retrieval, selection,
hydration and actual final-prompt delivery are read from the trusted
`memory_trace_sink` request trace (protocol zcode-r3-night-v1, correlation
via the synthetic `metadata.quality_row_id`). The probe remains available as
an explicitly separate diagnostic via `--probe`; its ids are recorded as
`probe`, never as delivered evidence. When the merged application does not
yet accept `memory_trace_sink`, the run records `trace_available=false` and
delivered-metrics stay `None` instead of being faked.

Recording happens before scoring (`scripts/r2_live_quality_check.run`);
gold never enters a prompt and is used only evaluation-side. The
evaluation-side canonical truth (every source text of every scope, with
author roles, instants and correction links) is dumped from the complete
synthetic store after loading, so judging never treats a missed source as
absent evidence.

    uv run --locked python scripts/r2_live_quality_run.py --output .local/r3-quality/live-1
    # add --probe for the optional standalone recall diagnostic
    # add --recording for the deterministic recording-runtime plumbing run
      (no live model execution, no answer-quality claims)

No working-database contact, no model downloads, no duplicate servers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HISTORY_IDS = ("r2-mem-a", "r2-mem-b", "r2-mem-final")
LEARNER_PREFIX = "q3"


def runtime_chat_once(system: str, question: str, body: str) -> dict:
    from local_app.config import Settings as _S
    from local_app.runtime import OllamaRuntime as _R
    import re as _re
    runtime = _R(_S.from_env())
    response = runtime.chat([{"role": "system", "content": system},
                             {"role": "user", "content": f"Вопрос: {question}\n\n{body}"}],
                            max_output_tokens=512)
    text = str((response.get("message") or {}).get("content") or "")
    match = _re.search(r"\{.*\}", text, _re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return text
    return text


class CountingRuntime:
    """Counts model/embedding calls and wall time around the real runtime."""

    def __init__(self, inner):
        self._inner = inner
        self.model_calls = 0
        self.embed_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def chat(self, *args, **kwargs):
        self.model_calls += 1
        return self._inner.chat(*args, **kwargs)

    def embed(self, *args, **kwargs):
        self.embed_calls += 1
        return self._inner.embed(*args, **kwargs)


class RecordingRuntime:
    """Deterministic no-model runtime for plumbing runs (client A only).

    Answers are fixed synthetic text and embeddings are hashed bag-of-words
    vectors: enough to exercise the real API/store/retrieval path without
    executing the local Gemma/BGE-M3 models (which only client B owns).
    Rows produced with this runtime are plumbing evidence, never answer
    quality.
    """

    def __init__(self, dims: int = 256):
        self.dims = dims
        self.chat_calls: list[dict] = []

    def check_models(self) -> dict:
        return {"ready": True, "recording": True,
                "note": "recording runtime: no model executed"}

    def chat(self, messages, *, max_output_tokens=512, **kwargs):
        self.chat_calls.append({"messages": messages, "max_output_tokens": max_output_tokens})
        return {"message": {"content": "Запись: ответ сгенерирован записывающей "
                            "заглушкой без живой модели (recording run)."}}

    def embed(self, texts, **kwargs):
        vectors = []
        for text in texts:
            vector = [0.0] * self.dims
            for token in str(text).lower().split():
                slot = int(uuid.uuid5(uuid.NAMESPACE_URL, token).hex[:8], 16) % self.dims
                vector[slot] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def declared_initial_profile(history_id: str, learner_id: str) -> dict:
    """A declared minimal initial profile per isolated history scope.

    Built from the shipped demo card structure with every unrelated personal
    preference removed: no demo interests, no demo current goal/topics/
    difficulties, no demo error patterns. Curriculum fields (grade, age
    group) stay because the frozen histories are grade-4 math conversations.
    The exact fields kept and cleared are recorded in the run metadata.
    """
    demo = json.loads((ROOT.parent / "learner-data" / "demo" / "learners"
                       / "demo-nika-04.json").read_text(encoding="utf-8-sig"))
    if not isinstance(demo, dict):
        raise ValueError("demo template card is not an object")
    card = dict(demo)
    card["learner_id"] = learner_id
    card["pseudonym"] = f"Quality {history_id}"
    cleared = {
        "interests": [],
        "error_patterns": [],
        # current_goal stays a schema-required field: it becomes a neutral
        # declared goal on a real catalog topic instead of the demo goal.
        "rag_context": {**demo.get("rag_context", {}),
                        "current_goal": {"goal": "нейтральный шаблон проверки качества",
                                         "topic_id": "math.g4.fractions.numerator_denominator",
                                         "updated_at": "2026-11-17"},
                        "current_topics": [], "priority_difficulties": []},
    }
    card.update(cleared)
    card["profile"] = {"explanation_style": "нейтральный шаблон проверки качества",
                       "pace": "нейтральный шаблон проверки качества",
                       "autonomy_level": "нейтральный шаблон проверки качества",
                       "motivation": "нейтральный шаблон проверки качества",
                       "prefers_visual": False}
    return card, sorted(cleared)


# declared_initial_profile returns (card, cleared_field_names).


def prepare_environment(keep_database: bool, history_ids=HISTORY_IDS):
    """Disposable database; one isolated learner/student scope per history."""
    from local_app.config import Settings
    from scripts.unified_app_smoke import (create_disposable_database,
                                           drop_disposable_database)
    settings = Settings.from_env()
    dsn, admin = create_disposable_database(settings.database_url)
    from local_app.store import Store
    store = Store(dsn)
    store.migrate()
    # Catalog and knowledge tasks are imported; NO demo learner card is:
    # every history scope starts from its own declared initial profile.
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        data_root = ROOT.parent / "learner-data"
        if not (data_root / "catalog").is_dir():
            data_root = ROOT / "learner-data"
        (Path(tmp) / "learner-data").mkdir(parents=True)
        shutil.copytree(data_root / "catalog", Path(tmp) / "learner-data" / "catalog")
        store.import_repository(tmp)
    from local_app.auth import hash_password
    password = "Quality run password 1"
    teacher = store.create_account("quality-teacher@example.test", "Quality teacher",
                                   hash_password(password), "teacher")
    from tests.r2_memory_quality.loader import load_history, load_history_file
    scopes: dict[str, dict] = {}
    for index, history_id in enumerate(history_ids):
        learner_id = f"{LEARNER_PREFIX}-{history_id}-{uuid.uuid4().hex[:8]}"
        card, cleared = declared_initial_profile(history_id, learner_id)
        # Validated card + teacher binding in one transaction; the student
        # account is then bound to the same isolated learner.
        store.create_assigned_learner(card, teacher["id"])
        student = store.create_student_account(
            f"quality-student-{index + 1}@example.test", "Quality student",
            hash_password(password), learner_id, teacher["id"])
        mapping = load_history(store, load_history_file(history_id), learner_id=learner_id,
                               student_actor_id=student["id"],
                               teacher_actor_id=teacher["id"])
        scopes[history_id] = {"learner_id": learner_id, "student_id": student["id"],
                              "student_email": student["email"], "key_to_id": mapping,
                              "initial_profile_cleared_fields": cleared,
                              "initial_profile_note": "demo template with unrelated "
                              "personal fields cleared; declared per N1 repair"}
    def cleanup():
        if not keep_database:
            from psycopg2.extensions import parse_dsn
            drop_disposable_database(admin, parse_dsn(dsn)["dbname"])
    return store, dsn, scopes, teacher, cleanup


def dump_truth_sources(store, scopes: dict) -> dict:
    """Evaluation-side canonical truth: every source text of every scope.

    Complete-dump truth means a source the answer path never retrieved still
    judges against its real text, and correction links/author roles/instants
    travel with each entry. The dump reads the whole `memory_sources` table
    per learner (corrections included) – not the message-reader projection,
    which intentionally hides non-paired rows.
    """
    dump: dict[str, dict] = {}
    for history_id, scope in scopes.items():
        with store._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT s.id, s.author_role, s.created_at, s.content,
                    s.corrects_source_id,
                    (SELECT c.id FROM teachcopilot.memory_sources c
                     WHERE c.corrects_source_id=s.id AND c.learner_id=s.learner_id
                     ORDER BY c.created_at, c.id LIMIT 1) AS superseded_by
                FROM teachcopilot.memory_sources s WHERE s.learner_id=%s""",
                (scope["learner_id"],))
            for row in cur.fetchall():
                dump[str(row["id"])] = {
                    "text": row["content"] or "",
                    "author_role": row["author_role"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                    "superseded_by": str(row["superseded_by"]) if row["superseded_by"] else None,
                    "corrects_source_id": str(row["corrects_source_id"]) if row["corrects_source_id"] else None,
                    "history_id": history_id,
                }
    return dump


def frozen_source_ids(store, learner_id):
    """The evaluation archive: every source id the loader froze for this scope."""
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM teachcopilot.memory_sources WHERE learner_id=%s",
                    (learner_id,))
        return [str(row["id"]) for row in cur.fetchall()]


def purge_run_created_rows(store, learner_id, frozen_ids):
    """Re-freeze the scope after a harness ask (A-review integrity fix).

    The chat flow persists its own user message and assistant reply into
    learning_messages/memory_sources; without a purge the run's earlier Q&A
    pairs become eligible evidence for later questions (delivered units dated
    the run day, absent from the canonical truth dump). Deleting the run's own
    rows between asks keeps every question's archive exactly the frozen
    history. Chunk rows referencing them are excluded from search by the
    candidate-key restriction (their sources are no longer eligible).
    """
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("""DELETE FROM teachcopilot.learning_messages
                       WHERE learner_id=%s
                         AND (source_id <> ALL(%s::text[]) OR source_id IS NULL)""",
                    (learner_id, frozen_ids))
        cur.execute("""DELETE FROM teachcopilot.memory_sources
                       WHERE learner_id=%s AND id <> ALL(%s::text[])""",
                    (learner_id, frozen_ids))


class TraceCollector:
    """Harness-side trusted sink consumer (protocol zcode-r3-night-v1).

    Events are correlated per request: the collector records the in-flight
    correlation id around each ask and also honours an explicit
    `correlation_id` on the event. It is test-harness state only – the
    application never sees it as a learner selector.
    """

    def __init__(self):
        self.events: dict[str, list[dict]] = {}
        self._in_flight: list[str | None] = [None]
        self.unsafe_events = 0

    def sink(self, event) -> None:
        if not isinstance(event, dict):
            self.unsafe_events += 1
            return
        correlation = event.get("correlation_id")
        if correlation is None:
            correlation = self._in_flight[0]
        if correlation is None:
            self.unsafe_events += 1
            return
        safe = json.loads(json.dumps(event, default=str))
        self.events.setdefault(str(correlation), []).append(safe)

    def begin(self, correlation_id: str) -> None:
        self._in_flight[0] = correlation_id

    def end(self) -> None:
        self._in_flight[0] = None

    def phases(self, correlation_id: str) -> dict:
        return {str(event.get("phase")): event
                for event in self.events.get(correlation_id, [])
                if event.get("phase")}


def summarize_trace(collector: TraceCollector, correlation_id: str) -> dict:
    """Build the harness-facing trace summary for one request.

    Delivered evidence comes only from the `answer_prompt` phase (the exact
    final answer-model input); retrieval/selection from the `retrieval`
    phase; route/generation/technical state from `fallback`/`complete`. A
    resolver-only or search-probe id never masquerades as delivered. The
    full hydrated quotes and the recorded final prompt travel with the row
    so the evaluator can verify provenance and prompt consistency; a phase
    or field the app did not instrument stays absent/None (unavailable),
    never an empty set and never a successful retrieval.
    """
    phases = collector.phases(correlation_id)
    retrieval_view = _phase_view(phases.get("retrieval"),
                                 ("eligible_source_ids", "retrieved_source_ids",
                                  "selected_source_ids", "budget", "diagnostics"))
    summary: dict = {"available": bool(phases),
                     "phases_seen": sorted(phases),
                     "retrieval": retrieval_view,
                     "retrieval_omitted": _omitted_of(phases.get("retrieval")),
                     "retrieval_pool_status": _pool_status(retrieval_view,
                                                           phases.get("answer_prompt")),
                     "answer_prompt": _phase_view(phases.get("answer_prompt"),
                                                  ("delivered_source_ids",
                                                   "final_prompt_sha256", "final_prompt",
                                                   "hydrated_quotes", "usage", "model",
                                                   "backend")),
                     "answer_prompt_omitted": _omitted_of(phases.get("answer_prompt")),
                     "hydrated_quote_count": _quote_count(phases.get("retrieval")) +
                                             _quote_count(phases.get("answer_prompt")),
                     "route": _route_of(phases),
                     }
    terminal = phases.get("complete") or phases.get("fallback") or {}
    summary["generation_called"] = terminal.get("generation_called")
    summary["technical_error"] = terminal.get("technical_error")
    summary["output_status"] = terminal.get("output_status")
    if summary["generation_called"] is None and phases:
        summary["generation_called"] = None  # explicit absence, never inferred true
    return summary


def _pool_status(retrieval_view, answer_prompt_event) -> str:
    """Honesty status of the retrieval-pool instrumentation.

    `ok` when the retrieved pool is a list that is either nonempty or
    accompanied by an empty selection (a genuine zero). `unavailable` when
    the field is missing or an empty pool sits beside nonempty delivered
    evidence – the contradictory empty set the contract forbids.
    """
    retrieved = (retrieval_view or {}).get("retrieved_source_ids") \
        if isinstance(retrieval_view, dict) else None
    delivered = (answer_prompt_event or {}).get("delivered_source_ids") \
        if isinstance(answer_prompt_event, dict) else None
    if not isinstance(retrieved, list):
        return "unavailable"
    if not retrieved and isinstance(delivered, list) and delivered:
        return "unavailable"
    return "ok"


def _phase_view(event, fields):
    if not isinstance(event, dict):
        return None
    return {field: event.get(field) for field in fields}


def _omitted_of(event):
    """Omission list under either accepted field name (contract or B's)."""
    if not isinstance(event, dict):
        return None
    for name in ("omitted_sources", "omitted_source_ids"):
        value = event.get(name)
        if isinstance(value, list):
            return value
    return None


def _quote_count(event) -> int:
    if not isinstance(event, dict):
        return 0
    quotes = event.get("hydrated_quotes")
    return len(quotes) if isinstance(quotes, list) else 0


def _route_of(phases) -> dict | None:
    """The actual (terminal) route of the request, not a provisional one.

    The complete/fallback event carries the final route per the contract;
    earlier phases only offer provisional routes if the run never finished.
    """
    for phase in ("complete", "fallback"):
        event = phases.get(phase)
        if isinstance(event, dict) and event.get("route") is not None:
            return {"phase": phase, "route": event["route"]}
    for phase in ("resolver", "retrieval", "answer_prompt"):
        event = phases.get(phase)
        if isinstance(event, dict) and event.get("route") is not None:
            return {"phase": phase, "route": event["route"]}
    return None


def build_app(settings, store, runtime, rag_service, trace_collector: TraceCollector | None):
    """Build the API app; attach the trusted trace sink when supported."""
    from local_app.api import create_app
    if trace_collector is None:
        return create_app(settings=settings, store=store, runtime=runtime,
                          rag_service=rag_service), False
    try:
        app = create_app(settings=settings, store=store, runtime=runtime,
                         rag_service=rag_service,
                         memory_trace_sink=trace_collector.sink)
        return app, True
    except TypeError:
        # The frozen trace interface is not merged in this tree yet: the run
        # stays honest with trace_available=false instead of faking delivery.
        app = create_app(settings=settings, store=store, runtime=runtime,
                         rag_service=rag_service)
        return app, False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-tokens", type=int, default=32768,
                        help="model context window for the run (16384 or 32768)")
    parser.add_argument("--recall-budget", type=int, default=20000)
    parser.add_argument("--probe", action="store_true",
                        help="optional standalone recall diagnostic, recorded "
                             "separately from the answer request")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--keep-database", action="store_true")
    parser.add_argument("--recording", action="store_true",
                        help="deterministic recording runtime (no live model, "
                             "plumbing evidence only)")
    parser.add_argument("--history", action="append", choices=HISTORY_IDS,
                        help="restrict the run to these histories (repeatable); "
                             "small isolated pilots stay separate output dirs")
    parser.add_argument("--score-only", action="store_true",
                        help="score already-recorded rows; no database, no model ask")
    parser.add_argument("--replay-output", type=Path, default=None,
                        help="write the replayed results.json here instead of the run "
                             "directory (use when replaying an immutable run dir)")
    args = parser.parse_args()

    from local_app.config import Settings
    from scripts.r2_live_quality_check import run as runner_run, score_from_records, QualityCheckError

    if args.score_only:
        output = args.output
        key_to_id = json.loads((output / "source_key_map.json").read_text(encoding="utf-8"))
        sources_by_id = {}
        texts = output / "truth_sources.json"
        if not texts.exists():
            # Legacy round-2 dirs carry only the retrieved-quote subset; the
            # aggregate meaning of such a rescoring stays probe-era legacy.
            texts = output / "source_texts.json"
        if texts.exists():
            sources_by_id = json.loads(texts.read_text(encoding="utf-8"))
        judge = None
        if not args.no_judge:
            def judge(system: str, question: str, body: str) -> dict:
                return runtime_chat_once(system, question, body)
        if args.replay_output is None and (output / "results.json").exists():
            raise QualityCheckError(
                f"{output / 'results.json'} exists: refusing to overwrite a recorded "
                "result in place; pass --replay-output for the replayed results")
        result = score_from_records(output,
                                    ROOT / "tests" / "r2_memory_quality" / "manifest.json",
                                    judge, key_to_id=key_to_id, sources_by_id=sources_by_id,
                                    replay_output_dir=args.replay_output)
        print(json.dumps(result["aggregate"], ensure_ascii=False))
        return 0

    live_settings = Settings.from_env()
    try:
        # A run-specific chat journal keeps the disposable environment fully
        # isolated (the shared default SQLite journal contends between runs
        # and surfaced as 503s); this is harness isolation, not a product
        # code change.
        live_settings = Settings(database_url=live_settings.database_url,
                                 ollama_url=live_settings.ollama_url,
                                 context_tokens=args.context_tokens,
                                 chat_response_journal_path=str(
                                     (args.output / "chat-responses.sqlite3").resolve()))
    except Exception as exc:  # settings validation (e.g. graph flags)
        print(f"settings invalid: {exc}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    if args.recording:
        base_runtime = RecordingRuntime()
        counting = CountingRuntime(base_runtime)
        # Recording runs never require live models: check_models stays local.
        ready = {"ready": True, "recording": True}
    else:
        from local_app.runtime import OllamaRuntime
        base_runtime = OllamaRuntime(live_settings)
        counting = CountingRuntime(base_runtime)
        ready = base_runtime.check_models()
        if not ready.get("ready"):
            print(f"live models unavailable, refusing to fake a run: {ready}", file=sys.stderr)
            return 2

    store, dsn, scopes, teacher, cleanup = prepare_environment(
        args.keep_database, history_ids=tuple(args.history) if args.history else HISTORY_IDS)
    try:
        key_to_id: dict[str, str] = {}
        scope_meta = {}
        for history_id, scope in scopes.items():
            for key, source_id in scope["key_to_id"].items():
                if key_to_id.setdefault(key, source_id) != source_id:
                    raise QualityCheckError(
                        f"source key {key!r} maps to two source ids across histories; "
                        "per-history isolation requires disjoint keys")
            scope_meta[history_id] = {k: str(v) if k in ("learner_id", "student_id") else v
                                      for k, v in scope.items() if k != "key_to_id"}
        (args.output / "source_key_map.json").write_text(
            json.dumps(key_to_id, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        truth_sources = dump_truth_sources(store, scopes)
        (args.output / "truth_sources.json").write_text(
            json.dumps(truth_sources, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8")
        from local_app.rag import RagService
        collector = TraceCollector()
        app, trace_sink_accepted = build_app(live_settings, store, counting,
                                             RagService(store=store, settings=live_settings,
                                                        runtime=counting),
                                             collector)
        if trace_sink_accepted:
            from local_app.unified_app import resolve_unified_memory
            facade = resolve_unified_memory(store, counting)
        else:
            facade = None
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)
        password = "Quality run password 1"
        tokens: dict[str, str] = {}
        for scope in scopes.values():
            login = client.post("/auth/login", json={"email": scope["student_email"],
                                                     "password": password})
            if login.status_code != 200:
                raise QualityCheckError(f"student login failed for "
                                        f"{scope['student_email']}: {login.text}")
            tokens[scope["learner_id"]] = "Bearer " + login.json()["access_token"]
        learner_by_history = {history_id: scope["learner_id"]
                              for history_id, scope in scopes.items()}

        frozen_by_scope = {history_id: frozen_source_ids(store, scope["learner_id"])
                           for history_id, scope in scopes.items()}

        def ask(row: dict) -> dict:
            history_id = row["history_id"]
            if history_id not in scopes:
                raise QualityCheckError(f"question {row['question_id']!r} references "
                                        f"unknown history {history_id!r}")
            learner_id = learner_by_history[history_id]
            purge_run_created_rows(store, learner_id, frozen_by_scope[history_id])
            counting.model_calls = 0
            counting.embed_calls = 0
            correlation_id = f"{row['question_id']}-{uuid.uuid4().hex[:8]}"
            collector.begin(correlation_id)
            started = time.monotonic()
            try:
                reply = client.post("/v1/chat/completions",
                                    headers={"Authorization": tokens[learner_id]},
                                    json={"messages": [{"role": "user", "content": row["prompt"]}],
                                          "metadata": {"quality_row_id": correlation_id}})
            finally:
                collector.end()
            reply.raise_for_status()
            payload = reply.json()
            usage = payload.get("usage") or {}
            result = {"request": {"messages": [{"role": "user", "content": row["prompt"]}],
                                  "learner_scope": {"history_id": history_id,
                                                    "learner_id": learner_id},
                                  "correlation_id": correlation_id},
                      "answer": (payload.get("choices") or [{}])[0].get("message", {}).get("content"),
                      "trace": summarize_trace(collector, correlation_id),
                      "answer_routing": payload.get("answer_routing") or {},
                      "usage": {"prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "total_tokens": usage.get("total_tokens"),
                                "recall_diagnostics": json.loads(json.dumps(
                                    payload.get("recall_diagnostics") or {}, default=str))},
                      "model_call_count": counting.model_calls + counting.embed_calls}
            if args.probe:
                # Explicitly separate diagnostic: standalone recall ids are
                # never used as selected or delivered evidence.
                if facade is None:
                    raise QualityCheckError("--probe requires the unified memory facade")
                probe_started = time.monotonic()
                recall = facade.recall(learner_id, scopes[history_id]["student_id"],
                                       row["prompt"], budget_bytes=args.recall_budget)
                result["probe"] = {"source_ids": list(recall.get("source_ids") or []),
                                   "duration_seconds": round(time.monotonic() - probe_started, 3),
                                   "diagnostics": json.loads(json.dumps(
                                       recall.get("diagnostics", {}), default=str))}
            return result

        judge = None
        if not args.no_judge:
            def judge(system: str, question: str, body: str):
                # Raw text out: judge_answer extracts the JSON object leniently
                # and strict validation marks unusable replies as judge_failure
                # instead of crashing the whole scoring phase.
                response = base_runtime.chat([{"role": "system", "content": system},
                                              {"role": "user", "content": f"Вопрос: {question}\n\n{body}"}],
                                             max_output_tokens=512)
                return str((response.get("message") or {}).get("content") or "")

        run_metadata = {
            "harness_protocol": "zcode-r3-night-v1",
            "history_isolation": "per_history_learner",
            "histories_run": list(scopes),
            "model_mode": "recording_runtime" if args.recording else "live_local_models",
            "answer_quality_evidence": bool(not args.recording),
            "trace_sink_accepted": trace_sink_accepted,
            "probe_mode": "separate_diagnostic" if args.probe else "disabled",
            "context_tokens": args.context_tokens,
            "recall_budget": args.recall_budget,
            "time_anchor": "loader day offsets are relative to the run date (UTC)",
            "scopes": scope_meta,
            "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        result = runner_run(ROOT / "tests" / "r2_memory_quality" / "manifest.json",
                            ask, judge, args.output,
                            key_to_id=key_to_id, sources_by_id=truth_sources,
                            run_metadata=run_metadata,
                            history_ids=tuple(scopes) if len(scopes) < len(HISTORY_IDS) else None)
        print(json.dumps(result["aggregate"], ensure_ascii=False))
        return 0
    except QualityCheckError as exc:
        print(f"quality check failed honestly: {exc}", file=sys.stderr)
        return 1
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())
