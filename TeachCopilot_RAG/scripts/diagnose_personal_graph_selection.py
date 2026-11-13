"""Diagnostic for app-only personal_graph source selection.

This is an operational diagnostic, not a blind evaluation. When explicitly run,
it creates only synthetic learner data in a fresh, prefixed test database and
records raw retrieval channels separately from application-selected evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_ROOT.parent.parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

DB_PREFIX = "nirs_personal_graph_it_"
SYNTHETIC_LEARNER_PREFIX = "pg-diagnostic-"
TOPIC_ID = "math.g4.fractions.numerator_denominator"
QUERY = "Какой вопрос помог мне исправить ошибку при сравнении дробей на одном из прошлых занятий?"
TARGET_QUESTION = (
    "Можно ли сравнить только числители? Что получится, если записать обе дроби "
    "с равными знаменателями? Сначала проверь, что целое разделено на равные части."
)
TARGET_OUTCOME = (
    "Я записал 5/8 как 25/40, а 3/4 как 30/40 и сделал вывод, "
    "что 3/4 больше 5/8."
)
EMPTY_SYNTHETIC_TABLES = (
    "accounts", "account_learners", "learners", "learning_sessions",
    "memory_sources", "learning_messages", "memory_jobs", "memory_versions",
    "next_steps",
)
RECENT_TOPICS = (
    ("number-line", "расположения дробей на числовом луче"),
    ("equivalent-fractions", "пары равных дробей и их сокращения"),
    ("numerator-denominator", "того, что показывают числитель и знаменатель"),
    ("unit-fractions", "долей целого в задаче про бумажную ленту"),
    ("fraction-picture", "раскрашенных равных частей на рисунке"),
    ("fraction-word-problem", "дробной части набора предметов"),
    ("fraction-order", "расположения нескольких дробей по возрастанию"),
    ("fraction-model", "представления дроби полосками одинаковой длины"),
    ("fraction-table", "заполнения таблицы с несколькими дробями"),
    ("fraction-unit", "сравнения дроби с одной целой"),
    ("fraction-practice", "чтения дроби по записи и рисунку"),
    ("fraction-review", "проверки нескольких равных долей"),
    ("fraction-steps", "последовательности шагов в задаче о долях"),
    ("fraction-shapes", "разделения фигур на равные части"),
    ("fraction-recap", "обозначения долей в короткой таблице"),
)


def _padding(label: str, minimum_chars: int = 1150) -> str:
    """Add ordinary lesson notes to create realistic long source turns."""
    sentence = (
        " В тетради я переписал условие рядом с рисунком, оставил место для "
        "вычислений и отдельно отметил порядок действий. Я записываю данные "
        "из задания в одном столбце, а промежуточные шаги в другом, чтобы позже "
        "можно было спокойно проверить решение и не смешать записи."
    )
    repeated = sentence
    while len(repeated) < minimum_chars:
        repeated += sentence
    return repeated


def build_scenario(minimum_source_bytes: int = 0, max_lessons: int = 16) -> dict:
    """Size ordinary synthetic source turns from the live app byte limits."""
    old = {
        "session_key": "old-fraction-comparison", "kind": "target",
        "turns": [
            {"student": (
                "В задаче я сравнивал пять восьмых и три четверти. Сначала решил, "
                "что первая дробь больше, потому что пять больше трёх, но ещё не "
                "проверил, что означают знаменатели." + _padding("old-first-turn", 900)),
             "assistant": TARGET_QUESTION},
            {"student": TARGET_OUTCOME + _padding("old-corrected-turn", 900),
             "assistant": "Хорошо, ты привёл дроби к равным знаменателям и проверил сравнение."},
        ],
    }
    required = max(0, int(minimum_source_bytes))
    lesson_count = max(6, (required + 8_399) // 8_400)
    lesson_count = min(max(1, int(max_lessons)), lesson_count)
    recent_count = min(len(RECENT_TOPICS), lesson_count - 1)
    recent = []
    for session_key, subject in RECENT_TOPICS[:recent_count]:
        recent.append({
            "session_key": session_key, "kind": "recent_unrelated",
            "turns": [
                {"student": (
                    f"В отдельном упражнении я разбирал {subject}. Записал условие, "
                    "проверил отдельные шаги и пока не стал переносить результат "
                    "на другие примеры." + _padding(session_key + "-turn-1", 900)),
                 "assistant": "Какие данные из условия нужны для этого шага?"},
                {"student": (
                    f"Я добавил ещё одну запись про {subject} и указал, какие "
                    "вычисления успел проверить." + _padding(session_key + "-turn-2", 900)),
                 "assistant": "Запиши промежуточный шаг рядом с условием."},
            ],
        })
    lessons = [old, *recent]
    all_student_turns = [turn for lesson in lessons for turn in lesson["turns"]]
    baseline_bytes = sum(len(turn["student"].encode("utf-8")) for turn in all_student_turns)
    per_turn_target = max(1_050, (required + len(all_student_turns) - 1) // len(all_student_turns))
    if baseline_bytes < required:
        for turn in all_student_turns:
            while len(turn["student"].encode("utf-8")) < per_turn_target:
                turn["student"] += _padding("lesson-notes", 160)
    return {
        "topic_id": TOPIC_ID, "query": QUERY, "old_lesson": old,
        "recent_lessons": recent, "target_question": TARGET_QUESTION,
        "target_outcome": TARGET_OUTCOME, "target_turn_index": 1,
        "source_count_estimate": len(all_student_turns),
        "minimum_source_bytes": required,
    }


def scenario_preflight(scenario: dict, selection_budget: int) -> dict:
    """Measure the app's real chunking before creating any synthetic records."""
    from local_app.personal_graph_memory import source_lessons
    from research_memory.personal_memory_lab.chunks import lesson_chunks

    sources = []
    lessons = [scenario["old_lesson"], *scenario["recent_lessons"]]
    for lesson_index, lesson in enumerate(lessons):
        preceding = None
        for turn_index, turn in enumerate(lesson["turns"]):
            item = {"source_id": f"preflight-{lesson_index}-{turn_index}",
                    "author_role": "student", "text": turn["student"],
                    "topic_id": TOPIC_ID, "observed_on": "2026-01-01",
                    "session_id": f"preflight-lesson-{lesson_index}"}
            if preceding:
                item["dialogue_context"] = {"role": "assistant", "text": preceding}
            if lesson_index == len(lessons) - 1 and turn_index == len(lesson["turns"]) - 1:
                item["session_end_context"] = {"role": "assistant", "usage": "context_only",
                                               "text": turn["assistant"]}
            sources.append(item)
            preceding = turn["assistant"]
    snapshot = {"learner_id": "synthetic-preflight", "sources": sources}
    archived, _ = source_lessons(snapshot, "synthetic-preflight")
    parts = [lesson_chunks(lesson, max_bytes=1800, hard_max_bytes=6000) for lesson in archived]
    count = sum(map(len, parts))
    if count > 24:
        raise ValueError("diagnostic_would_create_too_many_graph_fragments")
    raw_bytes = sum(len(json.dumps(lesson, ensure_ascii=False).encode("utf-8")) for lesson in archived)
    return {"lesson_count": len(archived), "student_source_count": len(sources),
            "source_bytes_before_selection": raw_bytes, "selection_budget_bytes": selection_budget,
            "source_bytes_exceed_selection_budget": raw_bytes > selection_budget,
            "fragment_count": count, "fragments_per_lesson": list(map(len, parts)),
            "estimated_prefilter_calls": count,
            "additional_graphiti_calls": "variable; depends on extracted entities and relations"}


def validate_database_name(value: str) -> str:
    value = value.strip()
    if not value.startswith(DB_PREFIX) or not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError("database_name_must_be_a_prefixed_synthetic_test_database")
    return value


def validate_local_postgres_dsn(dsn: str, *, pg_hostaddr: str | None = None) -> dict:
    from psycopg2.extensions import parse_dsn

    params = parse_dsn(dsn)
    hosts = [params.get("host"), params.get("hostaddr"), pg_hostaddr]
    values = [str(value).strip("[]").lower() for value in hosts if value]
    loopback = {"localhost", "127.0.0.1", "::1"}
    if not values or any(value not in loopback for value in values):
        raise ValueError("postgresql_must_use_loopback_only")
    return {"loopback_only": True, "database": params.get("dbname")}


def validate_local_ollama_url(value: str) -> bool:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("ollama_must_use_credential_free_http_loopback")
    return True


def validate_output_path(output: Path) -> Path:
    resolved = output.resolve()
    root = PROJECT_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("diagnostic_output_must_stay_inside_project")
    if resolved.suffix.lower() != ".json":
        raise ValueError("diagnostic_output_must_be_a_new_json_file")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def approval_boundary_report(target_source_id: str, approved_ids, chat_ids) -> dict:
    approved, used = set(map(str, approved_ids)), set(map(str, chat_ids))
    return {
        "approved_evidence_count": len(approved),
        "chat_evidence_ids": sorted(used),
        "chat_ids_within_approved_set": used <= approved,
        "target_source_in_approved_set": target_source_id in approved,
        "target_source_in_chat_context": target_source_id in used,
        "target_missing_from_approval_limit": target_source_id not in approved,
    }


def search_report(found: dict, *, phase: str, question: str,
                  allowed_source_ids=None, selection_budget_bytes: int | None = None) -> dict:
    if selection_budget_bytes is None:
        from local_app.personal_graph_memory import SELECTION_BYTE_BUDGET
        selection_budget_bytes = SELECTION_BYTE_BUDGET
    hits = [dict(hit) for hit in found.get("hits", []) if isinstance(hit, dict)]
    counts = Counter(str(hit.get("kind") or "unknown") for hit in hits)
    candidates = []
    for lesson in found.get("lessons", []) or []:
        if not isinstance(lesson, dict):
            continue
        candidates.append({
            "lesson_id": lesson.get("lesson_id"),
            "fragment_id": lesson.get("fragment_id"),
            "observed_on": lesson.get("observed_on"),
            "turns": [
                {key: turn.get(key) for key in ("turn_id", "role", "text", "anchor_source_id", "usage")
                 if key in turn}
                for turn in lesson.get("turns", []) if isinstance(turn, dict)
            ],
        })
    return {
        "phase": phase,
        "question": question,
        "allowed_source_ids": sorted(map(str, allowed_source_ids)) if allowed_source_ids is not None else None,
        "selection_budget": {"kind": "application_utf8_bytes", "limit": selection_budget_bytes},
        "channels": dict(sorted(counts.items())),
        "hits": hits,
        "unresolved": [dict(item) for item in found.get("unresolved", []) if isinstance(item, dict)],
        "hydrated_candidate_turns": candidates,
        "workflow_budget_selected_source_ids": sorted(map(str, found.get("source_ids", []) or [])),
        "workflow_skipped": [dict(item) for item in found.get("skipped", []) if isinstance(item, dict)],
    }


def selected_context_report(context: dict) -> dict:
    if not isinstance(context, dict):
        return {"available": False}
    return {
        "available": bool(context.get("text")),
        "evidence_ids": sorted(map(str, context.get("evidence_ids", []) or [])),
        "sources": [
            {key: source.get(key) for key in ("source_id", "role", "topic_id", "observed_on", "text",
                                                   "corrects_source_id") if key in source}
            for source in context.get("sources", []) if isinstance(source, dict)
        ],
        "dialogue_context": [dict(item) for item in context.get("dialogue_context", [])
                             if isinstance(item, dict)],
        "session_end_context": [
            {key: item.get(key) for key in ("source_id", "role", "usage", "text", "excerpt_mode", "truncated")
             if key in item}
            for item in context.get("session_end_context", []) if isinstance(item, dict)
        ],
        "selection_version": context.get("selection_version"),
        "selection_budget": context.get("selection_budget"),
        "selection_warnings": context.get("selection_warnings", []),
        "selection_skips": context.get("selection_skips", []),
        "version_id": context.get("version_id"),
        "stale": context.get("stale"),
        "withheld_reason": context.get("withheld_reason"),
    }


def mechanical_presence_checks(target_source_id: str, target_text: str, target_question: str,
                               search: dict, selected: dict) -> dict:
    """Literal ID/text checks only; these do not score semantic answer quality."""
    candidate_turns = [turn for lesson in search.get("hydrated_candidate_turns", [])
                       for turn in lesson.get("turns", [])]
    selected_sources = selected.get("sources", [])
    selected_ids = {str(source.get("source_id")) for source in selected_sources if source.get("source_id")}
    dialogue = selected.get("dialogue_context", [])
    dialogue_text = "\n".join(str(item.get("text") or "") for item in dialogue)
    return {
        "interpretation": "mechanical_id_and_literal_presence_checks_only_not_semantic_quality",
        "target_turn_in_hydrated_candidates": any(turn.get("turn_id") == target_source_id for turn in candidate_turns),
        "target_turn_in_workflow_budget_selection": target_source_id in set(search.get("workflow_budget_selected_source_ids", [])),
        "target_turn_in_application_evidence": target_source_id in selected_ids,
        "target_literal_text_in_selected_student_source": any(
            source.get("source_id") == target_source_id and target_text in str(source.get("text") or "")
            for source in selected_sources
        ),
        "assistant_question_literal_in_selected_context_only": target_question in dialogue_text,
    }


def possible_refusal_heuristic(answer: str) -> bool:
    lowered = answer.casefold()
    phrases = ("данных недостаточно", "не могу сказать", "не помню", "не могу определить",
               "недостаточно информации", "не нашёл в памяти")
    return any(phrase in lowered for phrase in phrases)


def assert_fresh_synthetic_database(store, expected_dbname: str) -> dict:
    """Require the dedicated test DB to have no pre-existing account or learner data."""
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT current_database() AS db_name")
        actual = cursor.fetchone()["db_name"]
        if actual != expected_dbname:
            raise ValueError("connected_database_does_not_match_requested_test_database")
        counts = {}
        for table in EMPTY_SYNTHETIC_TABLES:
            cursor.execute(f"SELECT count(*) AS row_count FROM teachcopilot.{table}")
            counts[table] = int(cursor.fetchone()["row_count"])
    populated = [table for table, count in counts.items() if count]
    if populated:
        raise ValueError("synthetic_test_database_must_be_fresh:" + ",".join(populated))
    return {"database_name_matches": True, "empty_synthetic_tables": sorted(counts), "fresh": True}


def write_json_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")


class SearchObserver:
    def __init__(self):
        self.events: list[dict] = []

    def record_search(self, phase: str, found: dict, question: str, allowed_source_ids=None,
                      selection_budget_bytes: int | None = None) -> None:
        self.events.append(search_report(found, phase=phase, question=question,
            allowed_source_ids=allowed_source_ids, selection_budget_bytes=selection_budget_bytes))

    def record_context(self, phase: str, context: dict) -> None:
        for event in reversed(self.events):
            if event["phase"] == phase and "application_context" not in event:
                event["application_context"] = selected_context_report(context)
                return
        self.events.append({"phase": phase, "application_context": selected_context_report(context),
                            "search_capture_missing": True})


class ObservedProvider:
    """Runtime subclass created lazily so importing this script starts no services."""
    observer: SearchObserver | None = None

    def _base_provider(self):
        from local_app.personal_graph_memory import PersonalGraphProvider
        return PersonalGraphProvider

    def __new__(cls, *args, **kwargs):
        # Keep import of the real app provider behind explicit diagnostic setup.
        base = cls._base_provider(cls)

        class _ObservedProvider(base):
            def _objects(self, *, source_order=()):
                model, archive, graph, adapter, workflow = super()._objects(source_order=source_order)
                original_search = workflow.search

                async def capture_search(run_id, learner_id, question, *, mode,
                                         budget_tokens=4096, allowed_source_ids=None):
                    found = await original_search(run_id, learner_id, question, mode=mode,
                        budget_tokens=budget_tokens, allowed_source_ids=allowed_source_ids)
                    phase = "chat" if allowed_source_ids is not None else "teacher_prepare"
                    if cls.observer is not None:
                        cls.observer.record_search(phase, found, question, allowed_source_ids,
                                                   selection_budget_bytes=budget_tokens)
                    return found

                workflow.search = capture_search
                return model, archive, graph, adapter, workflow

            def prepare(self, job, snapshot, question):
                state, context = super().prepare(job, snapshot, question)
                if cls.observer is not None:
                    cls.observer.record_context("teacher_prepare", context)
                return state, context

            def chat_context(self, snapshot, learner_id, topic_id, question):
                context = super().chat_context(snapshot, learner_id, topic_id, question)
                if cls.observer is not None:
                    cls.observer.record_context("chat", context)
                return context

        return _ObservedProvider(*args, **kwargs)


def _import_synthetic_card(store, learner_id: str, output_dir: Path, nonce: str) -> None:
    fixture = output_dir / ("diagnostic-fixture-" + nonce)
    (fixture / "learner-data" / "learners").mkdir(parents=True)
    data_dir = SCRIPT_ROOT.parent / "learner-data"
    shutil.copytree(data_dir / "catalog", fixture / "learner-data" / "catalog")
    card = json.loads((data_dir / "demo" / "learners" / "demo-nika-04.json").read_text(encoding="utf-8-sig"))
    card.update(learner_id=learner_id, pseudonym="Синтетический ученик")
    (fixture / "learner-data" / "learners" / (learner_id + ".json")).write_text(
        json.dumps(card, ensure_ascii=False), encoding="utf-8")
    store.import_repository(fixture)


def _create_synthetic_accounts(store, learner_id: str, nonce: str):
    from local_app.auth import hash_password

    password = "Synthetic-only-" + nonce
    teacher = store.create_account("teacher-" + nonce + "@synthetic.invalid", "Synthetic teacher",
                                   hash_password(password), "teacher")
    store.assign_learner(teacher["id"], learner_id)
    student = store.create_student_account("student-" + nonce + "@synthetic.invalid", "Synthetic student",
                                           hash_password(password), learner_id, teacher["id"])
    return teacher, student, password


def _seed_scenario(store, learner_id: str, student_id: str, scenario: dict, nonce: str) -> dict:
    target_source_id = None
    source_ids = []
    lessons = [scenario["old_lesson"], *scenario["recent_lessons"]]
    for lesson_index, lesson in enumerate(lessons):
        session = store.create_learning_session(learner_id, student_id, TOPIC_ID,
            client_session_id=f"{nonce}-{lesson['session_key']}")
        for turn_index, turn in enumerate(lesson["turns"]):
            message_id = f"m-{lesson_index}-{turn_index}"
            reserved = store.reserve_chat_message(session["id"], learner_id, message_id,
                                                   turn["student"], student_id)
            source_id = str(reserved["source_id"])
            source_ids.append(source_id)
            store.complete_chat_message(session["id"], message_id, {"choices": [{"message": {
                "role": "assistant", "content": turn["assistant"]}}]})
            if lesson["kind"] == "target" and turn_index == scenario["target_turn_index"]:
                target_source_id = source_id
        store.complete_learning_session(session["id"], learner_id, student_id)
    if not target_source_id:
        raise RuntimeError("synthetic_target_source_was_not_created")
    return {"target_source_id": target_source_id, "student_source_ids": source_ids,
            "session_count": len(lessons), "target_session_id": lessons[0]["session_key"]}


def _safe_call_summary(calls: list[dict]) -> list[dict]:
    return [{key: call.get(key) for key in ("purpose", "input_chars", "elapsed_seconds",
                                             "prompt_tokens", "completion_tokens") if key in call}
            for call in calls]


async def diagnostic_query_preview(store, runtime, snapshot: dict, state: dict,
                                   question: str, target_source_id: str) -> dict:
    """Inspect this synthetic archive without sending preview sources to an answer."""
    from local_app.personal_graph_memory import (PersonalGraphProvider, application_lesson_order,
        select_application_candidates, source_lessons)
    from local_app.personal_graph_contract import personal_graph_budgets

    learner_id = snapshot["learner_id"]
    graph_sources = snapshot.get("personal_graph_sources")
    if not isinstance(graph_sources, list):
        raise ValueError("personal_graph_source_snapshot_unavailable")
    graph_snapshot = {**snapshot, "sources": graph_sources}
    lessons, _ = source_lessons(graph_snapshot, learner_id)
    model, _, graph, _, workflow = PersonalGraphProvider(store, runtime)._objects(
        source_order=application_lesson_order(lessons))
    try:
        selection_budget, _ = personal_graph_budgets(runtime.context_tokens)
        found = await workflow.search(state["archive_run_id"], learner_id, question,
                                      mode="graphiti", budget_tokens=selection_budget)
        smaller = select_application_candidates(found["lessons"], found["hits"], 3000,
            counter=model.count, recent_lesson_ids=found["lesson_ids"][:2],
            max_budget=model.context - 2048)
        return {"purpose": "diagnostic_only_not_sent_to_generation",
                "retrieval": search_report(found, phase="query_preview", question=question,
                                           selection_budget_bytes=selection_budget),
                "same_candidates_budget_comparison": {
                    "smaller_budget_bytes": 3000,
                    "smaller_selected_ids": smaller["source_ids"],
                    "smaller_target_selected": target_source_id in smaller["source_ids"],
                    "expanded_budget_bytes": selection_budget,
                    "expanded_selected_ids": found["source_ids"],
                    "expanded_target_selected": target_source_id in found["source_ids"],
                    "interpretation": "same_hydrated_candidates_and_ranking_no_answer_quality_score",
                }}
    finally:
        await graph.close()


def _run(args) -> dict:
    from psycopg2.extensions import make_dsn
    from fastapi.testclient import TestClient
    from local_app.answer_adapter import AnswerAdapter
    from local_app.api import create_app
    from local_app.config import Settings
    from local_app.memory_response_journal import MemoryResponseJournal, database_fingerprint
    from local_app.memory_service import MemoryWorker
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store
    from local_app.personal_graph_contract import personal_graph_budgets

    args._phase = "preflight"
    dbname = validate_database_name(args.db_name_file.read_text(encoding="ascii"))
    base_settings = Settings.from_env()
    validate_local_postgres_dsn(base_settings.database_url, pg_hostaddr=os.getenv("PGHOSTADDR"))
    validate_local_ollama_url(base_settings.ollama_url)
    if base_settings.chat_model != "gemma4:12b" or not base_settings.embedding_model.startswith("bge-m3"):
        raise ValueError("configured_models_must_be_local_gemma_and_bge_m3")
    dsn = make_dsn(base_settings.database_url, dbname=dbname)
    settings = replace(base_settings, database_url=dsn, generation_scheduler_dsn=dsn,
        context_tokens=args.context_tokens, experimental_personal_graph_enabled=True, request_timeout=300,
        memory_response_journal_path=str(args.output.parent / ("diagnostic-memory-" + uuid4().hex + ".sqlite3")),
        chat_response_journal_path=str(args.output.parent / ("diagnostic-chat-" + uuid4().hex + ".sqlite3")))
    selection_budget, memory_budget = personal_graph_budgets(args.context_tokens)
    # Keep the same scenario across 16k and 32k checks; do not enlarge the
    # history merely because the selection budget changed.
    scenario = build_scenario()
    preflight = scenario_preflight(scenario, selection_budget)
    print(json.dumps({"preflight": preflight}, ensure_ascii=True), flush=True)
    nonce = uuid4().hex[:12]
    learner_id = SYNTHETIC_LEARNER_PREFIX + nonce
    store = Store(dsn)
    db_check = assert_fresh_synthetic_database(store, dbname)
    args._phase = "seed_synthetic_data"
    _import_synthetic_card(store, learner_id, args.output.parent, nonce)
    teacher, student, password = _create_synthetic_accounts(store, learner_id, nonce)
    seeded = _seed_scenario(store, learner_id, student["id"], scenario, nonce)
    snapshot = store.get_learner_memory(learner_id, None, include_personal_graph_snapshot=True)
    observer = SearchObserver()
    ObservedProvider.observer = observer

    memory_journal = MemoryResponseJournal(settings.memory_response_journal_path,
        database_identity=database_fingerprint(dsn))
    runtime = OllamaRuntime(settings)
    worker = MemoryWorker(store, runtime, response_journal=memory_journal,
                          personal_graph_factory=ObservedProvider)

    class ObservedRag(RagService):
        def answer(self, messages, *positional, **kwargs):
            self.last_memory_context = kwargs.get("memory_context")
            self.last_answer_started = time.perf_counter()
            response = super().answer(messages, *positional, **kwargs)
            self.last_answer_elapsed = time.perf_counter() - self.last_answer_started
            return response

    rag = ObservedRag(store, runtime, settings, answer_adapter=AnswerAdapter(runtime))
    app = create_app(settings=settings, store=store, runtime=runtime, rag_service=rag,
                     personal_graph_factory=ObservedProvider)
    client = TestClient(app)
    login = client.post("/auth/login", json={"email": teacher["email"], "password": password})
    if login.status_code != 200:
        raise RuntimeError("synthetic_teacher_login_failed")
    teacher_headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    queued = client.post(f"/learners/{learner_id}/next-step", headers=teacher_headers,
                         json={"topic_id": TOPIC_ID, "method": "personal_graph"})
    if queued.status_code != 202:
        raise RuntimeError("synthetic_teacher_queue_failed")
    job_id = str(queued.json()["job_id"])
    prep_started = time.perf_counter()
    args._phase = "teacher_prepare_graph_and_select"
    completed_job = worker.run_once()
    prep_seconds = time.perf_counter() - prep_started
    teacher_report = {
        "job_id": job_id,
        "status": completed_job.get("status") if completed_job else None,
        "elapsed_seconds": prep_seconds,
        "model_calls": _safe_call_summary(worker.calls),
    }
    if (not completed_job or str(completed_job.get("job_id")) != job_id
            or completed_job.get("status") != "completed"):
        return {
            "database": dbname, "synthetic_database_check": db_check,
            "preflight": preflight,
            "scenario": {"topic_id": TOPIC_ID, "query": QUERY, "session_count": seeded["session_count"],
                         "student_source_count": len(seeded["student_source_ids"])},
            "teacher_job": teacher_report,
            "searches": observer.events,
            "error_phase": "teacher_prepare",
            "ok": False,
        }
    version_id = str(completed_job["result"]["memory_version_id"])
    args._phase = "teacher_approval"
    before_approval = client.get(f"/learners/{learner_id}/memory", headers=teacher_headers)
    if before_approval.status_code != 200:
        raise RuntimeError("synthetic_teacher_memory_view_failed")
    approved = client.post(f"/learners/{learner_id}/memory/versions/{version_id}/confirm",
                           headers=teacher_headers,
                           json={"expected_revision": before_approval.json()["revision"]})
    if approved.status_code != 200:
        raise RuntimeError("synthetic_teacher_approval_failed")
    approved_version = approved.json()["version"]
    approved_ids = set(map(str, (approved_version.get("context") or {}).get("evidence_ids", [])))
    args._phase = "diagnostic_query_preview"
    preview = asyncio.run(diagnostic_query_preview(store, runtime, snapshot,
        approved_version["state"], scenario["query"], seeded["target_source_id"]))
    report = {
        "database": dbname,
        "synthetic_database_check": db_check,
        "scientific_result": False,
        "diagnostic_only": True,
        "preflight": preflight,
        "runtime": {"chat_model": settings.chat_model, "embedding_model": settings.embedding_model,
                    "context_tokens": settings.context_tokens,
                    "postgres_loopback_only": True, "ollama_loopback_only": True,
                    "neo4j_loopback_only": True},
        "application_limits": {"selection_budget": {"kind": "utf8_bytes", "limit": selection_budget},
                               "memory_context_budget": {"kind": "utf8_bytes", "limit": memory_budget}},
        "planned_activity": {
            "lesson_ingest_operations": seeded["session_count"],
            "teacher_action_calls": 1,
            "chat_answer_calls": "1 or 2 depending on chat query resolution",
            "graphiti_llm_calls": "variable per lesson and graph extraction; observed calls are listed below",
            "embedding_calls": "variable by chunk count and Graphiti entity search",
        },
        "scenario": {"topic_id": TOPIC_ID, "query": QUERY,
                     "session_count": seeded["session_count"],
                     "student_source_count": len(seeded["student_source_ids"]),
                     "synthetic_target_source_id": seeded["target_source_id"],
                     "synthetic_target_question": scenario["target_question"],
                     "synthetic_target_outcome": scenario["target_outcome"],
                     "query_contains_target_outcome": scenario["target_outcome"].casefold() in QUERY.casefold()},
        "teacher_job": {**teacher_report, "version_id": version_id,
                        "teacher_approved": bool(approved_version.get("approved")),
                        "approved_evidence_ids": sorted(approved_ids),
                        "teacher_action": completed_job.get("result", {}).get("action"),
                        "teacher_reason": completed_job.get("result", {}).get("reason")},
        "searches": observer.events,
        "diagnostic_query_preview": preview,
        "mechanical_presence_checks": {},
        "chat": {"attempted": False},
    }
    teacher_event = next((item for item in reversed(observer.events) if item.get("phase") == "teacher_prepare"), {})
    teacher_context = teacher_event.get("application_context", {})
    report["mechanical_presence_checks"] = mechanical_presence_checks(
        seeded["target_source_id"], scenario["target_outcome"], scenario["target_question"],
        teacher_event, teacher_context)

    chat_journal_path = settings.chat_response_journal_path
    session = store.create_learning_session(learner_id, student["id"], TOPIC_ID,
                                             client_session_id="diagnostic-chat-" + nonce)
    chat_started = time.perf_counter()
    args._phase = "approved_chat"
    auth = client.post("/auth/login", json={"email": student["email"], "password": password})
    if auth.status_code != 200:
        report["chat"] = {"attempted": False, "login_status": auth.status_code}
        report["ok"] = False
        report["error_phase"] = "synthetic_login"
        return report
    response = client.post("/v1/chat/completions", headers={
        "Authorization": "Bearer " + auth.json()["access_token"]}, json={
            "learning_session_id": str(session["id"]),
            "client_message_id": "diagnostic-question-" + nonce,
            "messages": [{"role": "user", "content": scenario["query"]}],
            "metadata": {"memory_method": "personal_graph"},
        })
    client.close()
    chat_body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    answer = ""
    if response.status_code == 200:
        answer = str((chat_body.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    chat_context = rag.last_memory_context if hasattr(rag, "last_memory_context") else None
    chat_selected = selected_context_report(chat_context or {})
    chat_ids = set(chat_selected.get("evidence_ids") or [])
    boundary = approval_boundary_report(seeded["target_source_id"], approved_ids, chat_ids)
    report["chat"] = {
        "attempted": True,
        "http_status": response.status_code,
        "elapsed_seconds": time.perf_counter() - chat_started,
        "answer": answer or chat_body.get("detail"),
        "possible_refusal_heuristic": possible_refusal_heuristic(answer),
        "usage": chat_body.get("usage"),
        "memory_version_id": chat_body.get("memory_version_id"),
        "memory_context": chat_selected,
        "approval_boundary": boundary,
        "answer_literal_target_match": scenario["target_outcome"].casefold() in answer.casefold(),
        "answer_match_interpretation": "literal_string_check_only_not_semantic_quality",
        "chat_journal_created": Path(chat_journal_path).exists(),
    }
    chat_event = next((item for item in reversed(observer.events) if item.get("phase") == "chat"), {})
    if chat_event:
        chat_event["application_context"] = chat_selected
    report["chat"]["ok_to_claim_memory_boundaries"] = boundary["chat_ids_within_approved_set"]
    report["ok"] = (bool(approved_version.get("approved")) and response.status_code == 200
                    and bool(answer) and boundary["chat_ids_within_approved_set"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name-file", type=Path, required=True,
                        help="Name file created for one fresh synthetic nirs_personal_graph_it_* database")
    parser.add_argument("--output", type=Path, required=True,
                        help="New JSON result path inside the project; existing files are never overwritten")
    parser.add_argument("--context-tokens", type=int, choices=(16384, 32768), default=16384)
    args = parser.parse_args(argv)
    output = validate_output_path(args.output)
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
              "scientific_result": False, "context_tokens_requested": args.context_tokens}
    args._phase = "preflight"
    try:
        args.output = output
        report.update(_run(args))
    except Exception as exc:
        report.update(ok=False, error_phase=getattr(args, "_phase", "preflight"),
                      error_type=type(exc).__name__, error_code="diagnostic_failed_before_completion")
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json_exclusive(output, report)
        except FileExistsError:
            raise
        except Exception as exc:
            # Do not expose connection strings, credentials, or raw model outputs.
            raise RuntimeError("diagnostic_result_could_not_be_written") from exc
    print(json.dumps({key: report.get(key) for key in
                      ("database", "ok", "error_phase", "error_type", "context_tokens_requested")},
                     ensure_ascii=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
