"""Round-2 regressions for the known unified-memory defects (client A).

Every test here reproduces a defect that fails on the round-1 baseline
(95bec61) and passes after the round-2 fix. Known defects covered:
- the fast-memory projection concatenated card values with change events, so a
  durable removal never suppressed the original card interest;
- identical-snapshot concurrent processing duplicated runs and applied changes
  (the pre-insert lookup alone is not concurrency control);
- a summary write failure lost the whole run including the personal changes;
- superseded (corrected) source text surfaced in recall without any
  supersession marking;
- durable change events replayed by (created_at, id) instead of source order,
  which mis-orders events sharing a transaction timestamp;
- task and last-question references of any lesson/session were accepted into
  the lesson state.
"""
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from local_app.auth import hash_password
from local_app.store import Store
from local_app.unified_memory import (
    UnifiedMemoryConflict,
    UnifiedMemoryService,
    UnifiedMemoryValidationError,
)
from local_app.unified_memory_processing import (
    EXTRACT_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_REDUCE_PROMPT,
)

LEARNER = "demo-nika-04"
OTHER_LEARNER = "demo-luna-03"
TOPIC = "math.g4.fractions.numerator_denominator"
VECTOR_DIMENSIONS = 1024


@pytest.fixture
def unified_store(tmp_path):
    """A Store on a fresh disposable loopback database with two demo learners."""
    import os
    import shutil
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn
    from pathlib import Path
    dsn = os.getenv("TEACHCOPILOT_TEST_DSN")
    if not dsn:
        pytest.skip("set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")
    require_local_dsn(dsn)
    name = DATABASE_PREFIX + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(dsn, dbname="postgres"))
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        store = Store(make_dsn(dsn, dbname=name))
        store.migrate()
        repo = Path(__file__).resolve().parents[2]
        shutil.copytree(repo / "learner-data/catalog", tmp_path / "learner-data/catalog",
                        dirs_exist_ok=True)
        learners = tmp_path / "learner-data/learners"
        learners.mkdir(parents=True, exist_ok=True)
        for learner in (LEARNER, OTHER_LEARNER):
            shutil.copy(repo / "learner-data/demo/learners" / f"{learner}.json", learners)
        store.import_repository(tmp_path)
        yield store
    finally:
        with admin.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        admin.close()


class Accounts:
    def __init__(self, store):
        password = "Unified memory test password 123"
        self.teacher = store.create_account("r2-regress-teacher@synthetic.test", "Teacher",
                                            hash_password(password), "teacher")
        store.assign_learner(self.teacher["id"], LEARNER)
        store.assign_learner(self.teacher["id"], OTHER_LEARNER)
        self.student = store.create_student_account("r2-regress-student@synthetic.test", "Student",
                                                    hash_password(password), LEARNER,
                                                    self.teacher["id"])
        self.outsider = store.create_account("r2-regress-outsider@synthetic.test", "Outsider",
                                             hash_password(password), "teacher")
        self.password = password


@pytest.fixture
def accounts(unified_store):
    return Accounts(unified_store)


class FakeRuntime:
    def __init__(self, chat_digest="d" * 64):
        self.settings = SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                        ollama_url="http://127.0.0.1:11435")
        self.chat_digest = chat_digest
        self.embedded = []

    def model_identity(self, model_name=None):
        return "gemma4:12b@sha256:" + self.chat_digest

    def embedding_model_identity(self):
        return "bge-m3@sha256:" + "c" * 64

    def embed(self, texts):
        self.embedded.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * VECTOR_DIMENSIONS
            casefold = text.casefold()
            if "хомяк" in casefold:
                vector[0] = 1.0
            elif "дроб" in casefold or "числитель" in casefold:
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            vectors.append(vector)
        return vectors


class RecordingModelCall:
    """Deterministic recording hook for purposes unified_*_*."""

    model_identity_static = "recording-gemma@sha256:" + "e" * 64

    def __init__(self, *, fail_extract=False, on_chunk_summary=None):
        self.calls = []
        self.fail_extract = fail_extract
        self.on_chunk_summary = on_chunk_summary

    def model_identity(self):
        return self.model_identity_static

    def __call__(self, messages, purpose, schema):
        self.calls.append({"purpose": purpose, "messages": messages})
        if purpose == SUMMARY_CHUNK_PROMPT:
            if self.on_chunk_summary is not None:
                self.on_chunk_summary(json.loads(messages[-1]["content"]))
            return json.dumps({"studied": ["дроби"], "achieved": ["ученик решил пример"],
                               "repeat": ["кратное сравнение"], "next_step": ["повторить деление"]},
                              ensure_ascii=False)
        if purpose == SUMMARY_REDUCE_PROMPT:
            return json.dumps({"studied": ["дроби и числитель"], "achieved": ["ученик решил пример"],
                               "repeat": ["кратное сравнение"], "next_step": ["решить задачу 3"]},
                              ensure_ascii=False)
        if purpose == EXTRACT_PROMPT:
            if self.fail_extract:
                raise RuntimeError("recording extraction failure")
            source_ids = json.loads(messages[-1]["content"])["source_ids"]
            return json.dumps({"changes": [{
                "category": "interest", "operation": "add", "value": "хомяки",
                "scope": "durable", "explicit_durable": True,
                "source_id": source_ids[0],
                "evidence_text": "хомяк"}],
                "candidates": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected purpose {purpose}")


def build_service(store, accounts, *, runtime=None, model_call=None):
    return UnifiedMemoryService(store, runtime or FakeRuntime(), model_call=model_call)


def new_session(store, accounts, learner=LEARNER, actor=None, topic=TOPIC):
    session = store.create_learning_session(learner, actor or accounts.student["id"], topic,
                                            client_session_id=uuid4().hex)
    return session["id"]


def exchange(store, accounts, session_id, client_id, student_text, assistant_text, actor=None):
    store.reserve_chat_message(session_id, LEARNER, client_id, student_text,
                               actor or accounts.student["id"])
    store.complete_chat_message(session_id, client_id, assistant_text)


def seeded_completed_session(store, accounts):
    session_id = new_session(store, accounts)
    exchange(store, accounts, session_id, "msg-1", "У меня есть хомяк, я люблю про него задачи.",
             "Отлично! Сравним 1/2 и 1/3 – какой числитель больше?")
    store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    return session_id


def interest_values(service, learner_id, actor_id):
    result = service.fast_memory(learner_id, actor_id)
    return [item["value"] for item in result["personal"]["interests"]]


# ----------------------------------------------------------------------- R1
def test_durable_removal_suppresses_original_card_interest(unified_store, accounts):
    """P0.1: the removal replays over the manual profile values."""
    service = build_service(unified_store, accounts)
    card_interests = unified_store.get_card(LEARNER).get("interests") or []
    target = card_interests[0]
    assert target in interest_values(service, LEARNER, accounts.student["id"])

    unified_store.append_personal_change(
        LEARNER, category="interest", operation="remove",
        value=f"  {target.upper()}  ", scope="durable", status="applied",
        explicit_durable=True, request_id="regress-remove-1")
    values = interest_values(service, LEARNER, accounts.teacher["id"])
    assert target not in values, "normalized removal must suppress the card interest"
    assert all(item.strip().casefold() != target.strip().casefold() for item in values)


def test_avoidance_does_not_remove_interest(unified_store, accounts):
    """P0.1: interest and avoided story theme stay separate categories."""
    service = build_service(unified_store, accounts)
    before = interest_values(service, LEARNER, accounts.student["id"])
    unified_store.append_personal_change(
        LEARNER, category="avoided_interest", operation="add", value="пираты",
        scope="durable", status="applied", explicit_durable=True,
        request_id="regress-avoid-1")
    after = service.fast_memory(LEARNER, accounts.teacher["id"])
    assert [item["value"] for item in after["personal"]["interests"]] == before
    avoided = [item["value"] for item in after["personal"]["avoided_interests"]]
    assert "пираты" in avoided


def test_remove_add_order_and_revert_replay(unified_store, accounts):
    """P0.1: source-order replay; a revert restores the previous state."""
    service = build_service(unified_store, accounts)
    unified_store.append_personal_change(
        LEARNER, category="interest", operation="add", value="динозавры", scope="durable",
        status="applied", explicit_durable=True, request_id="regress-order-add")
    unified_store.append_personal_change(
        LEARNER, category="interest", operation="remove", value="динозавры", scope="durable",
        status="applied", explicit_durable=True, request_id="regress-order-remove")
    assert "динозавры" not in interest_values(service, LEARNER, accounts.student["id"])
    second = unified_store.append_personal_change(
        LEARNER, category="interest", operation="add", value="Динозавры", scope="durable",
        status="applied", explicit_durable=True, request_id="regress-order-add-2")
    values = interest_values(service, LEARNER, accounts.student["id"])
    assert "динозавры" in values or "Динозавры" in values
    unified_store.revert_personal_change(LEARNER, second["id"], accounts.teacher["id"])
    assert "динозавры" not in interest_values(service, LEARNER, accounts.student["id"])


def test_conversation_interest_carries_actual_provenance(unified_store, accounts):
    """P0.1: effective items keep literal evidence, real source ids, status,
    event/date and scope from the applied change."""
    model = RecordingModelCall()
    service = build_service(unified_store, accounts, model_call=model)
    session_id = seeded_completed_session(unified_store, accounts)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="regress-prov-1")
    assert result["status"] == "completed"
    snapshot_sources = {source["source_id"] for source in
                        unified_store.unified_session_snapshot(LEARNER, session_id)["sources"]}
    fast = service.fast_memory(LEARNER, accounts.teacher["id"])
    item = next(item for item in fast["personal"]["interests"] if item["value"] == "хомяки")
    assert item["origin"] == "conversation"
    assert item["status"] == "applied" and item["scope"] == "durable"
    assert item["source_refs"], "actual source ids must travel with the item"
    assert item["source_refs"][0]["source_id"] in snapshot_sources
    assert item["evidence"] and "хомяк" in item["evidence"][0]["text"]
    assert item["change_id"] and item["event_at"]


# ----------------------------------------------------------------------- R2
def test_concurrent_identical_snapshot_processing_is_claimed_once(unified_store, accounts):
    """P1.2: the pre-insert lookup alone is not concurrency control."""
    nested = {}

    class InterleavingModelCall(RecordingModelCall):
        def __call__(self, messages, purpose, schema):
            if purpose == SUMMARY_CHUNK_PROMPT and not nested.get("ran"):
                nested["ran"] = True
                competing = build_service(unified_store, accounts,
                                          model_call=RecordingModelCall())
                with pytest.raises(UnifiedMemoryConflict):
                    competing.process_session(LEARNER, accounts.teacher["id"], session_id,
                                              request_id="regress-race-2")
            return super().__call__(messages, purpose, schema)

    session_id = seeded_completed_session(unified_store, accounts)
    service = build_service(unified_store, accounts, model_call=InterleavingModelCall())
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="regress-race-1")
    assert result["status"] == "completed"
    runs = unified_store.list_unified_summary_versions(LEARNER, session_id)
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT count(*) AS total FROM teachcopilot.unified_processing_runs
            WHERE learner_id=%s AND session_id=%s""", (LEARNER, session_id))
        assert cur.fetchone()["total"] == 1, "identical snapshots must produce one run"
        cur.execute("""SELECT count(*) AS total FROM teachcopilot.unified_personal_changes
            WHERE learner_id=%s AND category='interest' AND value='хомяки'
                AND status='applied'""", (LEARNER,))
        assert cur.fetchone()["total"] == 1, "applied changes must not duplicate"


# ----------------------------------------------------------------------- R3
def test_summary_write_failure_keeps_the_run_and_changes(unified_store, accounts, monkeypatch):
    """P1.2: a failed summary write degrades that component only."""
    def boom(*args, **kwargs):
        raise RuntimeError("simulated summary write failure")

    monkeypatch.setattr(Store, "_save_unified_summary_version_cur", boom)
    monkeypatch.setattr(Store, "save_unified_summary_version", boom)
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = seeded_completed_session(unified_store, accounts)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="regress-summary-fail")
    assert result["summary"] is None, "no summary masquerades as fresh"
    assert any(warning["code"] == "summary_write_failed" for warning in result["warnings"])
    assert result["personal_changes"], "personal changes survive the summary failure"
    assert unified_store.project_personal_changes(LEARNER)["interest"] == ["хомяки"]
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT status FROM teachcopilot.unified_processing_runs
            WHERE learner_id=%s AND request_id=%s""", (LEARNER, "regress-summary-fail"))
        assert cur.fetchone()["status"] == "partial"


# ----------------------------------------------------------------------- R4
def test_superseded_source_is_marked_in_recall(unified_store, accounts):
    """P1.2: superseded chunk text must not pass as a current fact."""
    session_id = new_session(unified_store, accounts, actor=accounts.teacher["id"])
    exchange(unified_store, accounts, session_id, "t-1",
             "У меня есть хомяк по имени Буся.", "Понятно, продолжим.", actor=accounts.teacher["id"])
    unified_store.complete_learning_session(session_id, LEARNER, accounts.teacher["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    target = next(source["source_id"] for source in
                  unified_store.unified_session_snapshot(LEARNER, session_id)["sources"]
                  if source["author_role"] == "student")
    correction = unified_store.correct_memory_source(
        LEARNER, target, "Правка: хомяка зовут Пушок.", accounts.teacher["id"],
        expected_revision=unified_store.get_revision(LEARNER),
        expected_source_revision=unified_store.get_learner_source_revision(LEARNER),
        client_correction_id="regress-correct-1")
    correction_source_id = correction["source"]["id"]
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    recall = service.recall(LEARNER, accounts.teacher["id"], "как зовут хомяка?",
                            budget_bytes=8000)
    texts = json.dumps(recall["sources"], ensure_ascii=False)
    assert "Буся" in texts or "Пушок" in texts, "the archive must still be searchable"
    superseded = [quote for quote in recall["sources"] if quote["source_id"] == target]
    assert superseded and superseded[0].get("superseded_by") == correction_source_id, \
        "a quote from a corrected original must carry the correcting source id"


# ----------------------------------------------------------------------- R5
def test_same_timestamp_events_replay_in_source_order(unified_store, accounts):
    """P0.1: events sharing a transaction timestamp replay in insertion order."""
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO teachcopilot.unified_personal_changes
            (id,learner_id,category,operation,value,scope,status,explicit_durable,request_id,created_at)
            VALUES
            ('ffffffff-1111-4111-8111-111111111111',%s,'interest','add','динозавры',
             'durable','applied',TRUE,'regress-seq-add','2026-01-01T10:00:00Z'),
            ('00000000-2222-4222-8222-222222222222',%s,'interest','remove','динозавры',
             'durable','applied',TRUE,'regress-seq-remove','2026-01-01T10:00:00Z')""",
            (LEARNER, LEARNER))
    service = build_service(unified_store, accounts)
    assert "динозавры" not in interest_values(service, LEARNER, accounts.student["id"]), \
        "the removal was inserted after the add and must replay after it"


# ----------------------------------------------------------------------- R6
def test_state_rejects_foreign_task_and_question_references(unified_store, accounts):
    """P1.1: no foreign lesson task or foreign session question can enter."""
    from local_app.unified_memory_store import parse_answer_sections
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson_text = ("# План занятия\nТема.\n## Практика\n1. Сравни 1/2 и 1/3.\n"
                   "## Ответы\n1) 1/2 больше")
    lesson = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                                 lesson_text=lesson_text, answers_text="1) 1/2 больше",
                                 validation={"status": "passed"}, request_id="regress-task-lesson",
                                 memory_revision="mem-rev-1")
    service.confirm_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                           expected_revision=1, request_id="regress-task-confirm")
    session_id = new_session(unified_store, accounts)
    service.attach_lesson(LEARNER, teacher["id"], session_id, lesson["lesson_id"],
                          request_id="regress-task-attach")

    with pytest.raises(UnifiedMemoryValidationError):
        service.update_lesson_state(
            LEARNER, student["id"], session_id, expected_revision=1,
            state={"stage": "practice", "pending_task_id": "task:other-lesson:practice:1"},
            request_id="regress-task-bad-1")

    other_session = new_session(unified_store, accounts)
    exchange(unified_store, accounts, other_session, "q-1", "Мой вопрос про дроби.",
             "Ассистент: объяснение про числитель и знаменатель.")
    foreign_assistant_source = next(
        source["source_id"] for source in
        unified_store.unified_session_snapshot(LEARNER, other_session)["sources"]
        if source["author_role"] == "assistant")
    with pytest.raises(UnifiedMemoryValidationError):
        service.update_lesson_state(
            LEARNER, student["id"], session_id, expected_revision=1,
            state={"stage": "practice", "last_question_source_id": foreign_assistant_source},
            request_id="regress-task-bad-2")

    tasks = service.lesson_tasks(LEARNER, teacher["id"], lesson["lesson_id"], audience="teacher")
    own_assistant_source = None
    exchange(unified_store, accounts, session_id, "q-2", "Я готов.", "Вопрос ассистента про дроби?")
    own_assistant_source = next(
        source["source_id"] for source in
        unified_store.unified_session_snapshot(LEARNER, session_id)["sources"]
        if source["author_role"] == "assistant")
    updated = service.update_lesson_state(
        LEARNER, student["id"], session_id, expected_revision=1,
        state={"stage": "practice", "pending_task_id": tasks["items"][0]["task_id"],
               "last_question_source_id": own_assistant_source},
        request_id="regress-task-ok")
    assert updated["applied"] is True and updated["stage"] == "practice"
