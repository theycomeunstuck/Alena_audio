"""Real disposable-PostgreSQL tests for the unified memory backend (client A).

Every test runs against a fresh teachcopilot_memory_verify_* database. Models
are recording fakes: no Ollama, Gemma, Neo4j or network service is contacted.
"""
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from local_app.auth import hash_password
from local_app.memory_store import MemoryIdempotencyConflict
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
    dsn = __import__("os").getenv("TEACHCOPILOT_TEST_DSN")
    if not dsn:
        pytest.skip("set TEACHCOPILOT_TEST_DSN for isolated PostgreSQL tests")
    import os
    import shutil
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn
    from pathlib import Path
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
        self.teacher = store.create_account("unified-teacher@synthetic.test", "Teacher",
                                            hash_password(password), "teacher")
        store.assign_learner(self.teacher["id"], LEARNER)
        store.assign_learner(self.teacher["id"], OTHER_LEARNER)
        self.student = store.create_student_account("unified-student@synthetic.test", "Student",
                                                    hash_password(password), LEARNER,
                                                    self.teacher["id"])
        self.outsider = store.create_account("unified-outsider@synthetic.test", "Outsider",
                                             hash_password(password), "teacher")
        self.password = password


@pytest.fixture
def accounts(unified_store):
    return Accounts(unified_store)


class FakeRuntime:
    """Deterministic 1024-dim embedder with immutable identity metadata."""

    def __init__(self, fail_embedding=False, chat_digest="d" * 64):
        self.settings = SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                        ollama_url="http://127.0.0.1:11435")
        self.fail_embedding = fail_embedding
        self.chat_digest = chat_digest
        self.embedded = []

    def model_identity(self, model_name=None):
        return "gemma4:12b@sha256:" + self.chat_digest

    def embedding_model_identity(self):
        return "bge-m3@sha256:" + "c" * 64

    def embed(self, texts):
        if self.fail_embedding:
            raise RuntimeError("embedding service is down")
        self.embedded.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        casefold = text.casefold()
        vector = [0.0] * VECTOR_DIMENSIONS
        if "хомяк" in casefold:
            vector[0] = 1.0
        elif "дроб" in casefold or "числитель" in casefold:
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        return vector


class RecordingModelCall:
    """Deterministic recording hook for purposes unified_*_*."""

    model_identity_static = "recording-gemma@sha256:" + "e" * 64

    def __init__(self, *, fail_chunk_ids=(), fail_extract=False):
        self.calls = []
        self.fail_chunk_ids = set(fail_chunk_ids)
        self.fail_extract = fail_extract

    def model_identity(self):
        return self.model_identity_static

    def __call__(self, messages, purpose, schema):
        self.calls.append({"purpose": purpose, "messages": messages})
        if purpose == SUMMARY_CHUNK_PROMPT:
            chunk = json.loads(messages[-1]["content"])
            if chunk.get("chunk_id") in self.fail_chunk_ids:
                raise RuntimeError("recording chunk failure")
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
                "candidates": [{
                    "category": "error_pattern",
                    "proposed_value": {"error_tag": "путает числитель и знаменатель",
                                       "topic_id": TOPIC, "note": "при сравнении дробей"},
                    "source_id": source_ids[-1],
                    "evidence_text": "числитель"}]}, ensure_ascii=False)
        raise AssertionError(f"unexpected purpose {purpose}")


def build_service(store, accounts, *, runtime=None, model_call=None):
    return UnifiedMemoryService(store, runtime or FakeRuntime(), model_call=model_call)


def new_session(store, accounts, learner=LEARNER, actor=None, topic=TOPIC):
    session = store.create_learning_session(learner, actor or accounts.student["id"], topic,
                                            client_session_id=uuid4().hex)
    return session["id"]


def exchange(store, accounts, session_id, client_id, student_text, assistant_text):
    store.reserve_chat_message(session_id, LEARNER, client_id, student_text,
                               accounts.student["id"])
    store.complete_chat_message(session_id, client_id, assistant_text)


LESSON_TEXT = ("# План занятия\nРазбор дробей.\n## Задание 1\nСравни 1/2 и 1/3.\n"
               "## Практика\n1. Сравни 1/2 и 1/3.\n"
               "## Ответы\n1) 1/2 больше\n2) подсказка учителю\n## Ход занятия\nОбсуждение.")
VALIDATION = {"status": "passed", "issues": []}


# ----------------------------------------------------------------- lessons
def test_lesson_save_get_confirm_and_redaction_roundtrip(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    saved = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                                lesson_text=LESSON_TEXT, answers_text="Полные ответы учителю.",
                                validation=VALIDATION, request_id="lesson-req-1",
                                memory_revision="mem-rev-1")
    assert saved["status"] == "draft" and saved["revision"] == 1
    assert saved["lesson_id"] and saved["topic_id"] == TOPIC

    replayed = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                                   lesson_text=LESSON_TEXT, answers_text="Полные ответы учителю.",
                                   validation=VALIDATION, request_id="lesson-req-1",
                                   memory_revision="mem-rev-1")
    assert replayed["lesson_id"] == saved["lesson_id"]

    with pytest.raises((UnifiedMemoryConflict, UnifiedMemoryValidationError)):
        service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                            lesson_text="Другой текст", answers_text="", validation=VALIDATION,
                            request_id="lesson-req-1", memory_revision="mem-rev-1")
    with pytest.raises(UnifiedMemoryValidationError):
        service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                            lesson_text="Без валидации", answers_text="",
                            validation={"status": "failed"}, request_id="lesson-req-2",
                            memory_revision="mem-rev-1")
    with pytest.raises(PermissionError):
        service.save_lesson(LEARNER, student["id"], topic_id=TOPIC, minutes=45,
                            lesson_text=LESSON_TEXT, answers_text="", validation=VALIDATION,
                            request_id="lesson-req-3", memory_revision="mem-rev-1")

    confirmed = service.confirm_lesson(LEARNER, teacher["id"], saved["lesson_id"],
                                       expected_revision=1, request_id="confirm-req-1")
    assert confirmed["status"] == "approved" and confirmed["confirmed"] is True
    idempotent = service.confirm_lesson(LEARNER, teacher["id"], saved["lesson_id"],
                                        expected_revision=1, request_id="confirm-req-1")
    assert idempotent["confirmed"] is False and idempotent["status"] == "approved"
    with pytest.raises(UnifiedMemoryConflict):
        service.confirm_lesson(LEARNER, teacher["id"], saved["lesson_id"],
                               expected_revision=2, request_id="confirm-req-2")
    with pytest.raises(PermissionError):
        service.confirm_lesson(LEARNER, student["id"], saved["lesson_id"],
                               expected_revision=1, request_id="confirm-req-3")

    teacher_view = service.get_lesson(LEARNER, teacher["id"], saved["lesson_id"], audience="teacher")
    assert "1) 1/2 больше" in teacher_view["lesson_text"]
    assert teacher_view["answers_text"] == "Полные ответы учителю."
    student_view = service.get_lesson(LEARNER, student["id"], saved["lesson_id"], audience="teacher")
    assert student_view["audience"] == "student" and student_view["answer_redacted"] is True
    assert "1) 1/2 больше" not in student_view["lesson_text"]
    assert student_view["answers_text"] == ""
    assert student_view["status"] == "approved"
    teacher_student_view = service.get_lesson(LEARNER, teacher["id"], saved["lesson_id"],
                                              audience="student")
    assert teacher_student_view["answer_redacted"] is True
    with pytest.raises(LookupError):
        service.get_lesson(LEARNER, teacher["id"], "00000000-0000-0000-0000-000000000000",
                           audience="teacher")
    with pytest.raises(UnifiedMemoryValidationError):
        service.get_lesson(LEARNER, teacher["id"], saved["lesson_id"], audience="admin")


def test_attach_lesson_state_update_and_cas_conflicts(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                                 lesson_text=LESSON_TEXT, answers_text="", validation=VALIDATION,
                                 request_id="attach-lesson-1", memory_revision="mem-rev-1")
    service.confirm_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                           expected_revision=1, request_id="attach-confirm-1")
    draft_attach = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=30,
                                       lesson_text="# План\nЧерновик.", answers_text="",
                                       validation=VALIDATION, request_id="attach-lesson-2",
                                       memory_revision="mem-rev-1")
    session_id = new_session(unified_store, accounts)
    with pytest.raises(UnifiedMemoryValidationError):
        service.attach_lesson(LEARNER, teacher["id"], session_id, draft_attach["lesson_id"],
                              request_id="attach-req-0")
    state = service.attach_lesson(LEARNER, teacher["id"], session_id, lesson["lesson_id"],
                                  request_id="attach-req-1")
    assert state["stage"] == "not_started" and state["revision"] == 1
    replay = service.attach_lesson(LEARNER, teacher["id"], session_id, lesson["lesson_id"],
                                   request_id="attach-req-1")
    assert replay["revision"] == 1
    missing = service.lesson_state(LEARNER, teacher["id"], new_session(unified_store, accounts))
    assert missing["lesson_id"] is None and missing["stage"] is None
    with pytest.raises(PermissionError):
        service.attach_lesson(LEARNER, student["id"], session_id, lesson["lesson_id"],
                              request_id="attach-req-2")
    with pytest.raises(LookupError):
        service.attach_lesson(LEARNER, teacher["id"],
                              "00000000-0000-0000-0000-000000000000", lesson["lesson_id"],
                              request_id="attach-req-3")

    tasks = service.lesson_tasks(LEARNER, teacher["id"], lesson["lesson_id"], audience="teacher")
    assert tasks["catalog"]["status"] == "available" and tasks["items"]
    real_task_id = tasks["items"][0]["task_id"]
    updated = service.update_lesson_state(
        LEARNER, student["id"], session_id, expected_revision=1,
        state={"stage": "warmup", "pending_task_id": real_task_id, "completed_task_ids": []},
        request_id="state-req-1")
    assert updated["stage"] == "warmup" and updated["revision"] == 2 and updated["applied"]
    replayed = service.update_lesson_state(
        LEARNER, student["id"], session_id, expected_revision=99,
        state={"stage": "warmup", "pending_task_id": real_task_id, "completed_task_ids": []},
        request_id="state-req-1")
    assert replayed["applied"] is False and replayed["revision"] == 2
    with pytest.raises(UnifiedMemoryConflict):
        service.update_lesson_state(
            LEARNER, student["id"], session_id, expected_revision=1,
            state={"stage": "practice"}, request_id="state-req-2")
    with pytest.raises(UnifiedMemoryValidationError):
        service.update_lesson_state(
            LEARNER, student["id"], session_id, expected_revision=2,
            state={"stage": "finished"}, request_id="state-req-3")
    other_session = new_session(unified_store, accounts, actor=accounts.teacher["id"])
    with pytest.raises(PermissionError):
        service.update_lesson_state(
            LEARNER, accounts.outsider["id"], other_session, expected_revision=1,
            state={"stage": "practice"}, request_id="state-req-4")
    state_view = service.lesson_state(LEARNER, student["id"], session_id)
    assert state_view["stage"] == "warmup" and state_view["pending_task_id"] == real_task_id


def test_list_lessons_visibility_by_role(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    draft = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                                lesson_text=LESSON_TEXT, answers_text="", validation=VALIDATION,
                                request_id="list-draft-1", memory_revision="mem-rev-1")
    approved = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=30,
                                   lesson_text="# План\nОдобрен.", answers_text="",
                                   validation=VALIDATION, request_id="list-approved-1",
                                   memory_revision="mem-rev-1")
    service.confirm_lesson(LEARNER, teacher["id"], approved["lesson_id"],
                           expected_revision=1, request_id="list-confirm-1")
    teacher_view = service.list_lessons(LEARNER, teacher["id"])
    assert {item["status"] for item in teacher_view["lessons"]} == {"draft", "approved"}
    student_view = service.list_lessons(LEARNER, student["id"])
    assert {item["lesson_id"] for item in student_view["lessons"]} == {approved["lesson_id"]}
    for item in teacher_view["lessons"] + student_view["lessons"]:
        assert "lesson_text" not in item and "answers_text" not in item
    with pytest.raises(PermissionError):
        service.list_lessons(LEARNER, accounts.outsider["id"])


# --------------------------------------------------------------- fast memory
def test_fast_memory_projection_is_cheap_opaque_and_merge_aware(unified_store, accounts):
    service = build_service(unified_store, accounts)
    before = unified_store.get_revision(LEARNER)
    result = service.fast_memory(LEARNER, accounts.student["id"])
    assert result["method"] == "unified_fast_memory"
    assert result["learner_id"] == LEARNER
    assert isinstance(result["revision"], str) and result["revision"].startswith("um-")
    assert set(result["personal"]) == {"interests", "avoided_interests", "preferences"}
    assert set(result["learning"]) == {"goals", "confirmed_observations", "errors", "help",
                                       "unfinished"}
    profile_interests = [item["value"] for item in result["personal"]["interests"]
                         if item["origin"] == "profile"]
    assert profile_interests, "demo card must carry profile interests"
    assert result["learning"]["goals"] and result["learning"]["goals"][0]["goal"]
    assert result["session"] is None
    again = service.fast_memory(LEARNER, accounts.student["id"])
    assert again["revision"] == result["revision"]
    assert unified_store.get_revision(LEARNER) == before

    applied = unified_store.append_personal_change(
        LEARNER, category="interest", operation="add", value="хомяки", scope="durable",
        status="applied", explicit_durable=True, request_id="fast-change-1")
    assert applied["created"]
    after = service.fast_memory(LEARNER, accounts.teacher["id"])
    values = [item["value"] for item in after["personal"]["interests"]]
    assert "хомяки" in values
    assert any(item["origin"] == "conversation" and item["value"] == "хомяки"
               for item in after["personal"]["interests"])
    assert after["revision"] != result["revision"]
    # A personal change must not invalidate the academic card revision.
    assert unified_store.get_revision(LEARNER) == before

    reverted = unified_store.revert_personal_change(LEARNER, applied["id"],
                                                    accounts.teacher["id"])
    assert reverted["reverted"]
    restored = service.fast_memory(LEARNER, accounts.teacher["id"])
    assert "хомяки" not in [item["value"] for item in restored["personal"]["interests"]]

    with pytest.raises(PermissionError):
        service.fast_memory(LEARNER, accounts.outsider["id"])
    with pytest.raises(LookupError):
        service.fast_memory(LEARNER, accounts.teacher["id"], session_id="00000000-0000-0000-0000-000000000000")


# ---------------------------------------------------------- session processing
def seeded_completed_session(store, accounts, *, extra_turns=0):
    session_id = new_session(store, accounts)
    exchange(store, accounts, session_id, "msg-1", "У меня есть хомяк, я люблю про него задачи.",
             "Отлично! Сравним 1/2 и 1/3 – какой числитель больше?")
    exchange(store, accounts, session_id, "msg-2", "хз", "Давай разберём: числитель – сверху.")
    for index in range(extra_turns):
        exchange(store, accounts, session_id, f"extra-{index}",
                 f"Сообщение номер {index} про дроби и хомяка: длинное сообщение для проверки "
                 f"разбиения транскрипта на ограниченные куски при обработке сессии номер {index}.",
                 "Ответ ассистента на сообщение.")
    store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    return session_id


def test_process_session_summary_coverage_and_idempotency(unified_store, accounts):
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = seeded_completed_session(unified_store, accounts)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-1")
    assert result["status"] == "completed" and result["provisional"] is False
    assert result["source_fingerprint"] and result["model_identity"].startswith("gemma4:12b@")
    assert result["prompt_recipe"].startswith("unified_memory_v1:")
    summary = result["summary"]
    assert summary["summary_id"] and not summary["provisional"]
    for heading in ("Что изучали", "Что получилось", "Что стоит повторить", "Следующий шаг"):
        assert heading in summary["text"]
    assert result["coverage"]["chunk_ids"] and result["coverage"]["failed_chunk_ids"] == []

    replay = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-2")
    assert replay["run_id"] == result["run_id"]
    assert replay["summary"]["summary_id"] == summary["summary_id"]

    # Replaying the same request id for the exact same snapshot stays idempotent
    # even from a freshly constructed service with the same recipe.
    conflict_service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    assert conflict_service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                            request_id="proc-1")["run_id"] == result["run_id"]

    # The same request id for a different session snapshot is rejected.
    other_session = seeded_completed_session(unified_store, accounts)
    with pytest.raises(UnifiedMemoryConflict):
        conflict_service.process_session(LEARNER, accounts.teacher["id"], other_session,
                                         request_id="proc-1")

    # A teacher correction of a session source changes the exact snapshot: the
    # corrected archive produces a new run, and both summary versions stay
    # retrievable.
    expected_revision = unified_store.get_revision(LEARNER)
    expected_source_revision = unified_store.get_learner_source_revision(LEARNER)
    target = next(source["source_id"] for source in
                  unified_store.unified_session_snapshot(LEARNER, session_id)["sources"]
                  if source["author_role"] == "student" and "хомяк" in source["text"])
    correction = unified_store.correct_memory_source(
        LEARNER, target, "Правка: хомяка зовут Пух.", accounts.teacher["id"],
        expected_revision=expected_revision,
        expected_source_revision=expected_source_revision,
        client_correction_id="unified-correct-1")
    assert correction["source"]["corrects_source_id"] == target
    second = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-3")
    assert second["status"] == "completed"
    assert second["run_id"] != result["run_id"]
    assert second["summary"]["summary_id"] != summary["summary_id"]
    versions = unified_store.list_unified_summary_versions(LEARNER, session_id)
    assert len(versions) >= 2
    corrected_texts = [source["text"] for source in
                       unified_store.unified_session_snapshot(LEARNER, session_id)["sources"]]
    assert any("Пух" in text for text in corrected_texts), "correction joins the processed snapshot"

    student_view = service.processing_result(LEARNER, accounts.student["id"], session_id)
    assert student_view["available"] is True and student_view["status"] == "completed"


def test_process_session_partial_coverage_on_failed_chunk(unified_store, accounts):
    from local_app.unified_memory import session_model_chunks
    from local_app.unified_memory_processing import session_model_messages
    session_id = seeded_completed_session(unified_store, accounts, extra_turns=6)
    snapshot = unified_store.unified_session_snapshot(LEARNER, session_id)
    chunks = session_model_chunks(LEARNER, session_id,
                                  session_model_messages(snapshot["sources"]))
    assert len(chunks) >= 2, "a longer session must split into several chunks"
    failing = RecordingModelCall(fail_chunk_ids={chunk["chunk_id"] for chunk in chunks[1:]})
    service = build_service(unified_store, accounts, model_call=failing)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-partial")
    assert result["status"] == "partial" and result["provisional"] is True
    assert result["coverage"]["failed_chunk_ids"] == [chunk["chunk_id"] for chunk in chunks[1:]]
    assert result["coverage"]["included_chunk_ids"] == [chunks[0]["chunk_id"]]
    assert result["summary"]["provisional"] is True
    assert result["summary"]["text"]  # built from the successful chunks only
    assert any(warning["code"] == "summary_chunk_failed" for warning in result["warnings"])
    # The exact same snapshot and recipe do not replay into a new run.
    replay = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-partial")
    assert replay["run_id"] == result["run_id"]


def test_process_session_extraction_rules_end_to_end(unified_store, accounts):
    model = RecordingModelCall()
    service = build_service(unified_store, accounts, model_call=model)
    session_id = seeded_completed_session(unified_store, accounts)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-extract")
    # "хз" question + answer stays in the original transcript, verbatim.
    snapshot = unified_store.unified_session_snapshot(LEARNER, session_id)
    texts = [source["text"] for source in snapshot["sources"]]
    assert "хз" in texts and any("числитель больше" in text for text in texts)
    assert result["personal_changes"], "explicit durable interest must auto-apply"
    applied = result["personal_changes"][0]
    assert applied["status"] == "applied" and applied["value"] == "хомяки"
    assert applied["scope"] == "durable" and applied["evidence"]
    assert applied["source_refs"][0]["source_id"] in {source["source_id"] for source in snapshot["sources"]}
    candidates = result["academic_candidates"]
    assert candidates and candidates[0]["status"] == "pending"
    assert candidates[0]["proposed_value"]["error_tag"] == "путает числитель и знаменатель"
    assert "mastery" not in json.dumps(candidates)

    projection = unified_store.project_personal_changes(LEARNER)
    assert projection["interest"] == ["хомяки"]

    # A failed extraction run for a different model identity degrades to
    # partial without deleting anything from the earlier completed run.
    failing = build_service(unified_store, accounts,
                            runtime=FakeRuntime(chat_digest="9" * 64),
                            model_call=RecordingModelCall(fail_extract=True))
    failed = failing.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-extract-fail")
    assert failed["status"] == "partial"
    assert failed["coverage"]["extraction_failed_chunk_ids"]
    assert failed["summary"]["text"]
    assert unified_store.project_personal_changes(LEARNER) == {"interest": ["хомяки"],
                                                               "avoided_interest": [],
                                                               "preference": []}


def test_active_session_processing_is_teacher_only_and_marks_provisional(unified_store, accounts):
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-1", "Я люблю дроби.", "Хорошо!")
    with pytest.raises(PermissionError):
        service.process_session(LEARNER, accounts.student["id"], session_id,
                                request_id="active-student")
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="active-teacher")
    assert result["provisional"] is True
    status = unified_store.get_learning_session(session_id, LEARNER)["status"]
    assert status == "active", "processing must not finish an active session"


def test_second_learner_isolation_in_processing_and_recall(unified_store, accounts):
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = seeded_completed_session(unified_store, accounts)
    other_session = new_session(unified_store, accounts, learner=OTHER_LEARNER,
                                actor=accounts.teacher["id"])
    unified_store.reserve_chat_message(other_session, OTHER_LEARNER, "other-1",
                                       "У меня нет питомца, но я люблю роботов.",
                                       accounts.teacher["id"])
    unified_store.complete_chat_message(other_session, "other-1", "Понятно, про роботов.")
    unified_store.complete_learning_session(other_session, OTHER_LEARNER, accounts.teacher["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (OTHER_LEARNER,))
    other_result = service.process_session(OTHER_LEARNER, accounts.teacher["id"], other_session,
                                           request_id="proc-other")
    # The recording model proposes хомяки with evidence that is not literal in
    # this archive: the validator must reject it, so no change is stored.
    assert other_result["personal_changes"] == [], "other learner must not inherit чужие facts"
    assert unified_store.project_personal_changes(OTHER_LEARNER)["interest"] == []
    assert unified_store.project_personal_changes(LEARNER)["interest"] == []

    recall = service.recall(LEARNER, accounts.student["id"], "где мой хомяк?")
    assert all("нет питомца" not in json.dumps(quote, ensure_ascii=False)
               for quote in recall["sources"]), "recall stays learner-scoped"
    other_recall = service.recall(OTHER_LEARNER, accounts.teacher["id"], "хомяк")
    assert other_recall["diagnostics"]["archive_message_count"] >= 1
    assert all("У меня есть хомяк" not in quote["text"]
               for quote in other_recall["sources"]), "cross-learner text never leaks"


# ------------------------------------------------------------- change review
def test_list_changes_and_teacher_review_actions(unified_store, accounts):
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = seeded_completed_session(unified_store, accounts)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="proc-review")
    listing = service.list_changes(LEARNER, accounts.teacher["id"])
    with pytest.raises(PermissionError):
        service.list_changes(LEARNER, accounts.student["id"])
    assert listing["revision"] >= 1
    kinds = {item["kind"] for item in listing["items"]}
    assert "academic_candidate" in kinds and "personal_change" in kinds
    candidate_item = next(item for item in listing["items"] if item["kind"] == "academic_candidate")
    assert candidate_item["eligible_actions"] == ["accept", "reject"]
    change_item = next(item for item in listing["items"] if item["kind"] == "personal_change"
                       and item["status"] == "applied")
    assert change_item["eligible_actions"] == ["revert"]  # applied durable personal change

    with pytest.raises(UnifiedMemoryConflict):
        service.review_changes(LEARNER, accounts.teacher["id"],
                               actions=[{"id": candidate_item["candidate_id"], "action": "accept"}],
                               expected_revision=listing["revision"] + 5, request_id="review-1")
    with pytest.raises(PermissionError):
        service.review_changes(LEARNER, accounts.student["id"],
                               actions=[{"id": candidate_item["candidate_id"], "action": "accept"}],
                               expected_revision=listing["revision"], request_id="review-0")

    review = service.review_changes(
        LEARNER, accounts.teacher["id"],
        actions=[{"id": candidate_item["candidate_id"], "action": "accept"}],
        expected_revision=listing["revision"], request_id="review-1")
    assert review["applied"] == 1 and review["revision"] == listing["revision"] + 1
    # Canonical application recorded an approved teacher memory source.
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT content,approved FROM teachcopilot.memory_sources
            WHERE learner_id=%s AND source_key LIKE %s""",
            (LEARNER, "unified-candidate:%"))
        rows = cur.fetchall()
    assert rows and rows[0]["approved"] is True
    assert "путает числитель и знаменатель" in rows[0]["content"]

    idempotent = service.review_changes(
        LEARNER, accounts.teacher["id"],
        actions=[{"id": candidate_item["candidate_id"], "action": "accept"}],
        expected_revision=review["revision"], request_id="review-2")
    assert idempotent["applied"] == 0

    # Rejecting a proposed personal change records the teacher decision.
    proposed = unified_store.append_personal_change(
        LEARNER, category="preference", operation="add", value="решает в уме", scope="durable",
        status="proposed", request_id="propose-1")
    listing = service.list_changes(LEARNER, accounts.teacher["id"])
    review = service.review_changes(
        LEARNER, accounts.teacher["id"],
        actions=[{"id": proposed["id"], "action": "reject"}],
        expected_revision=listing["revision"], request_id="review-3")
    assert review["applied"] == 1
    assert unified_store.get_personal_change(LEARNER, proposed["id"])["status"] == "rejected"
    assert "решает в уме" not in unified_store.project_personal_changes(LEARNER)["preference"]


def test_review_revert_of_applied_personal_change(unified_store, accounts):
    service = build_service(unified_store, accounts)
    applied = unified_store.append_personal_change(
        LEARNER, category="interest", operation="add", value="динозавры", scope="durable",
        status="applied", explicit_durable=True, request_id="revert-case-1")
    assert unified_store.project_personal_changes(LEARNER)["interest"] == ["динозавры"]
    listing = service.list_changes(LEARNER, accounts.teacher["id"])
    review = service.review_changes(
        LEARNER, accounts.teacher["id"],
        actions=[{"id": applied["id"], "action": "revert"}],
        expected_revision=listing["revision"], request_id="revert-req-1")
    assert review["applied"] == 1
    assert unified_store.project_personal_changes(LEARNER)["interest"] == []
    events = unified_store.list_personal_changes(LEARNER)
    assert any(event["status"] == "reverted" and event["reverts_id"] == applied["id"]
               for event in events), "append-only history keeps the original event"
    # A second revert of the same change is a no-op, not an error.
    second = service.review_changes(
        LEARNER, accounts.teacher["id"],
        actions=[{"id": applied["id"], "action": "revert"}],
        expected_revision=review["revision"], request_id="revert-req-2")
    assert second["applied"] == 0
    assert sum(1 for event in unified_store.list_personal_changes(LEARNER)
               if event["status"] == "reverted" and event["reverts_id"] == applied["id"]) == 1


# --------------------------------------------------------------------- recall
def test_recall_persisted_rrf_and_scope(unified_store, accounts):
    runtime = FakeRuntime()
    service = build_service(unified_store, accounts, runtime=runtime)
    session_id = seeded_completed_session(unified_store, accounts)
    recall = service.recall(LEARNER, accounts.student["id"], "про какого хомяка я говорил?",
                            budget_bytes=6000)
    assert recall["method"] == "conversation_retrieval"
    assert recall["text"] == recall["system_text"]
    assert recall["diagnostics"]["rrf_k"] == 60
    assert recall["diagnostics"]["search"] == "persisted_postgres_rrf"
    assert recall["evidence_ids"] == recall["source_ids"]
    assert recall["source_ids"], "the хомяк message must be found"
    assert any("хомяк" in quote["text"] for quote in recall["sources"])
    assert all(quote["role"] in ("user", "assistant") for quote in recall["sources"])
    assert any(quote.get("learning_session_id") for quote in recall["sources"])
    assert all(quote.get("activity_day") for quote in recall["sources"] if quote["role"] == "user")
    # Persisted representations: no re-embedding of already stored chunks.
    embedded_after_first = sum(len(batch) for batch in runtime.embedded)
    second = service.recall(LEARNER, accounts.student["id"], "хомяк и числитель")
    embedded_after_second = sum(len(batch) for batch in runtime.embedded)
    assert second["source_ids"]
    assert embedded_after_second - embedded_after_first <= 2, "only the query may embed again"

    empty = service.recall(LEARNER, accounts.student["id"], "квантовая физика панциря")
    assert empty["text"] == "" and empty["sources"] == [] and empty["evidence_ids"] == []

    with pytest.raises(PermissionError):
        service.recall(LEARNER, accounts.outsider["id"], "хомяк")
    with pytest.raises(LookupError):
        service.recall(LEARNER, accounts.student["id"], "хомяк",
                       session_id="00000000-0000-0000-0000-000000000000")


def test_recall_degrades_to_lexical_with_visible_diagnostic(unified_store, accounts):
    service = build_service(unified_store, accounts, runtime=FakeRuntime(fail_embedding=True))
    session_id = seeded_completed_session(unified_store, accounts)
    recall = service.recall(LEARNER, accounts.student["id"], "хомяк")
    assert recall["diagnostics"]["degraded"] is True
    assert recall["diagnostics"]["degraded_reason"] == "embedding_unavailable"
    assert recall["source_ids"], "bounded lexical fallback must still find the хомяк turn"
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT count(*) AS total FROM teachcopilot.unified_conversation_chunks
            WHERE learner_id=%s AND actor_id=%s""", (LEARNER, str(accounts.student["id"])))
        assert cur.fetchone()["total"] >= 1


def test_recall_honors_explicit_time_filter(unified_store, accounts):
    service = build_service(unified_store, accounts)
    session_id = seeded_completed_session(unified_store, accounts)
    recall = service.recall(LEARNER, accounts.student["id"], "хомяк",
                            time_filter={"from": "2020-01-01", "through": "2020-01-02"})
    assert recall["text"] == "" and recall["sources"] == []
    assert recall["diagnostics"]["matched_chunk_count"] == 0


def test_queue_constraints_accept_new_job_kinds(unified_store, accounts):
    job = unified_store.enqueue_memory_job(
        LEARNER, TOPIC, "lesson_generate",
        {"topic_id": TOPIC, "minutes": 45, "client_request_id": "gen-1"},
        actor_id=accounts.teacher["id"], idempotency_key="lesson-gen:gen-1")
    assert job["kind"] == "lesson_generate"
    session_id = new_session(unified_store, accounts)
    job = unified_store.enqueue_memory_job(
        LEARNER, TOPIC, "session_process",
        {"learning_session_id": str(session_id), "client_request_id": "sp-1"},
        actor_id=accounts.teacher["id"], method="personal_memory",
        idempotency_key="session-process:sp-1")
    assert job["kind"] == "session_process" and job["method"] == "personal_memory"
    with pytest.raises(MemoryIdempotencyConflict):
        unified_store.enqueue_memory_job(
            LEARNER, TOPIC, "lesson_generate",
            {"topic_id": TOPIC, "minutes": 30, "client_request_id": "gen-1"},
            actor_id=accounts.teacher["id"], idempotency_key="lesson-gen:gen-1")
    with pytest.raises(ValueError):
        unified_store.enqueue_memory_job(
            LEARNER, TOPIC, "lesson_generate", {"topic_id": TOPIC},
            actor_id=accounts.teacher["id"], idempotency_key="lesson-gen:bad-payload")
    with pytest.raises(ValueError):
        unified_store.enqueue_memory_job(
            LEARNER, TOPIC, "memory_update", {}, actor_id=accounts.teacher["id"],
            method="personal_memory", idempotency_key="old-kind-personal-memory")
