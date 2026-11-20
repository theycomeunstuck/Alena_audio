"""Round-2 feature tests: assignments, task catalog, claims and honesty.

Client A (zcode/r2-memory-20261117). Real disposable PostgreSQL, recording
models only. These cover the P0.2 assignment lifecycle, the P1.1 catalog and
reference validation, the P1.2 claim lease and processing honesty.
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
from local_app.unified_memory_tasks import build_task_catalog

LEARNER = "demo-nika-04"
OTHER_LEARNER = "demo-luna-03"
TOPIC = "math.g4.fractions.numerator_denominator"
VECTOR_DIMENSIONS = 1024

LESSON_TEXT = ("# Урок: Дроби\n\n## Цель\nПонимать числитель и знаменатель.\n"
               "## Разминка\n1. Сколько половинок в целом яблоке?\n"
               "## Объяснение\nЧислитель – сколько частей взяли.\n"
               "## Практика\n1. Сравни 1/2 и 1/3.\n2. Сколько третьих частей в 2/3?\n"
               "## Проверка\n1. Что показывает знаменатель?\n"
               "## Ответы\n1) 1/2 больше\n2) две\n3) на сколько частей разделили")
VALIDATION = {"status": "passed", "issues": []}


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
        self.teacher = store.create_account("r2-feature-teacher@synthetic.test", "Teacher",
                                            hash_password(password), "teacher")
        store.assign_learner(self.teacher["id"], LEARNER)
        store.assign_learner(self.teacher["id"], OTHER_LEARNER)
        self.student = store.create_student_account("r2-feature-student@synthetic.test", "Student",
                                                    hash_password(password), LEARNER,
                                                    self.teacher["id"])
        self.outsider = store.create_account("r2-feature-outsider@synthetic.test", "Outsider",
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

    def model_identity(self, model_name=None):
        return "gemma4:12b@sha256:" + self.chat_digest

    def embedding_model_identity(self):
        return "bge-m3@sha256:" + "c" * 64

    def embed(self, texts):
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
                               "repeat": ["кратное сравнение"], "next_step": ["решить задачу 2"]},
                              ensure_ascii=False)
        if purpose == SUMMARY_REDUCE_PROMPT:
            return json.dumps({"studied": ["дроби"], "achieved": ["ученик решил пример"],
                               "repeat": ["кратное сравнение"], "next_step": ["решить задачу 2"]},
                              ensure_ascii=False)
        if purpose == EXTRACT_PROMPT:
            if self.fail_extract:
                raise RuntimeError("recording extraction failure")
            source_ids = json.loads(messages[-1]["content"])["source_ids"]
            return json.dumps({"changes": [{
                "category": "interest", "operation": "add", "value": "хомяки",
                "scope": "durable", "explicit_durable": True,
                "source_id": source_ids[0], "evidence_text": "хомяк"}],
                "candidates": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected purpose {purpose}")


def build_service(store, accounts, *, runtime=None, model_call=None):
    return UnifiedMemoryService(store, runtime or FakeRuntime(), model_call=model_call)


def approved_lesson(service, accounts, *, request_id, lesson_text=LESSON_TEXT, answers=""):
    lesson = service.save_lesson(LEARNER, accounts.teacher["id"], topic_id=TOPIC, minutes=45,
                                 lesson_text=lesson_text, answers_text=answers,
                                 validation=VALIDATION, request_id=request_id,
                                 memory_revision="mem-rev-1")
    service.confirm_lesson(LEARNER, accounts.teacher["id"], lesson["lesson_id"],
                           expected_revision=1, request_id=f"confirm-{request_id}")
    return lesson


def new_session(store, accounts, learner=LEARNER, actor=None, topic=TOPIC):
    session = store.create_learning_session(learner, actor or accounts.student["id"], topic,
                                            client_session_id=uuid4().hex)
    return session["id"]


def exchange(store, accounts, session_id, client_id, student_text, assistant_text, actor=None):
    store.reserve_chat_message(session_id, LEARNER, client_id, student_text,
                               actor or accounts.student["id"])
    store.complete_chat_message(session_id, client_id, assistant_text)


# ------------------------------------------------------------- P0.2 assignments
def test_assign_lesson_persists_before_any_student_session(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = approved_lesson(service, accounts, request_id="assign-lesson-1")

    # No session exists yet: the assignment still persists.
    assignment = service.assign_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                                       expected_lesson_revision=1, request_id="assign-req-1")
    assert assignment["status"] == "pending"
    assert assignment["learning_session_id"] is None
    assert assignment["lesson_revision"] == 1 and assignment["topic_id"] == TOPIC

    replay = service.assign_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                                   expected_lesson_revision=1, request_id="assign-req-2")
    assert replay["assignment_id"] == assignment["assignment_id"]
    assert replay["created"] is False, "same payload is idempotent"

    with pytest.raises(UnifiedMemoryConflict):
        service.assign_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                              expected_lesson_revision=2, request_id="assign-req-3")
    with pytest.raises(UnifiedMemoryValidationError):
        service.assign_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                              expected_lesson_revision=0, request_id="assign-req-4")
    with pytest.raises(PermissionError):
        service.assign_lesson(LEARNER, student["id"], lesson["lesson_id"],
                              expected_lesson_revision=1, request_id="assign-req-5")
    with pytest.raises(PermissionError):
        service.assign_lesson(LEARNER, accounts.outsider["id"], lesson["lesson_id"],
                              expected_lesson_revision=1, request_id="assign-req-6")
    draft = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=30,
                                lesson_text="# Урок: Черновик\n## Практика\n1. Пример.",
                                answers_text="", validation=VALIDATION,
                                request_id="assign-draft-1", memory_revision="mem-rev-1")
    with pytest.raises(UnifiedMemoryValidationError):
        service.assign_lesson(LEARNER, teacher["id"], draft["lesson_id"],
                              expected_lesson_revision=1, request_id="assign-req-7")
    with pytest.raises(LookupError):
        service.assign_lesson(LEARNER, teacher["id"],
                              "00000000-0000-0000-0000-000000000000",
                              expected_lesson_revision=1, request_id="assign-req-8")

    listing = service.list_lesson_assignments(LEARNER, teacher["id"])
    assert [item["assignment_id"] for item in listing["items"]] == [assignment["assignment_id"]]
    assert listing["items"][0]["assigned_by"] == str(teacher["id"])
    student_listing = service.list_lesson_assignments(LEARNER, student["id"])
    assert [item["assignment_id"] for item in student_listing["items"]] == \
        [assignment["assignment_id"]]
    assert "assigned_by" not in student_listing["items"][0], \
        "student view carries no private teacher provenance"
    for item in listing["items"] + student_listing["items"]:
        assert "lesson_text" not in item and "answers_text" not in item


def test_start_assignment_creates_student_owned_session_idempotently(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = approved_lesson(service, accounts, request_id="start-lesson-1")
    assignment = service.assign_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                                       expected_lesson_revision=1, request_id="start-req-1")

    started = service.start_lesson_assignment(LEARNER, student["id"], assignment["assignment_id"],
                                              request_id="start-exec-1")
    assert started["status"] == "started"
    assert started["learning_session_id"] and started["created"] is True
    state = started["lesson_state"]
    assert state["session_id"] == started["learning_session_id"]
    assert state["lesson_id"] == lesson["lesson_id"] and state["stage"] == "not_started"

    session = unified_store.get_learning_session(started["learning_session_id"], LEARNER)
    assert str(session["actor_id"]) == str(student["id"]), \
        "the created session is owned by the student"
    assert session["status"] == "active"

    repeat = service.start_lesson_assignment(LEARNER, student["id"], assignment["assignment_id"],
                                             request_id="start-exec-2")
    assert repeat["learning_session_id"] == started["learning_session_id"]
    assert repeat["created"] is False, "repeats return the same session"

    # A different child cannot claim the assignment: the learner-scoped lookup
    # hides it (LookupError) before any actor-role check could pass.
    other_account = unified_store.create_student_account(
        "r2-feature-other@synthetic.test", "Other", hash_password(accounts.password),
        OTHER_LEARNER, teacher["id"])
    with pytest.raises(LookupError):
        service.start_lesson_assignment(OTHER_LEARNER, other_account["id"],
                                        assignment["assignment_id"], request_id="start-exec-3")
    with pytest.raises(PermissionError):
        service.start_lesson_assignment(LEARNER, teacher["id"], assignment["assignment_id"],
                                        request_id="start-exec-4")
    with pytest.raises(LookupError):
        service.start_lesson_assignment(LEARNER, student["id"],
                                        "00000000-0000-0000-0000-000000000000",
                                        request_id="start-exec-5")

    # Completing the guided lesson state completes the assignment; no fresh
    # session is created under the same assignment id.
    service.update_lesson_state(LEARNER, student["id"], started["learning_session_id"],
                                expected_revision=1, state={"stage": "completed"},
                                request_id="start-complete-1")
    completed = service.start_lesson_assignment(LEARNER, student["id"],
                                                assignment["assignment_id"],
                                                request_id="start-exec-6")
    assert completed["status"] == "completed"
    assert completed["learning_session_id"] == started["learning_session_id"]
    listing = service.list_lesson_assignments(LEARNER, teacher["id"])
    assert listing["items"][0]["status"] == "completed"
    assert listing["items"][0]["completed_at"] is not None


def test_start_assignment_conflicts_when_approved_revision_changed(unified_store, accounts):
    """A recorded assignment never silently substitutes a different lesson."""
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = approved_lesson(service, accounts, request_id="rev-lesson-1")
    assignment = service.assign_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                                       expected_lesson_revision=1, request_id="rev-req-1")
    with unified_store._connect() as conn, conn.cursor() as cur:
        # Simulate an approval revision change after assignment time.
        cur.execute("""UPDATE teachcopilot.unified_lessons SET revision=revision+1
            WHERE id=%s AND learner_id=%s""", (lesson["lesson_id"], LEARNER))
    with pytest.raises(UnifiedMemoryConflict):
        service.start_lesson_assignment(LEARNER, student["id"], assignment["assignment_id"],
                                        request_id="rev-exec-1")
    assert unified_store.get_lesson_assignment(LEARNER, assignment["assignment_id"])["status"] == \
        "pending"


# --------------------------------------------------------------- P1.1 catalog
def test_task_catalog_is_stable_and_validated(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=45,
                                 lesson_text=LESSON_TEXT, answers_text="1) 1/2 больше",
                                 validation=VALIDATION, request_id="catalog-lesson-1",
                                 memory_revision="mem-rev-1")
    tasks = service.lesson_tasks(LEARNER, teacher["id"], lesson["lesson_id"], audience="teacher")
    assert tasks["lesson_id"] == lesson["lesson_id"]
    assert tasks["lesson_revision"] == 1
    assert tasks["catalog"]["status"] == "available"
    stages = [(item["stage"], item["position"], item["prompt_text"]) for item in tasks["items"]]
    assert [stage for stage, _, _ in stages] == ["warmup", "practice", "practice", "check"]
    assert [position for _, position, _ in stages] == [1, 2, 3, 4]
    assert "Сколько половинок в целом яблоке?" in stages[0][2]
    first = tasks["items"][0]
    assert first["task_id"].startswith(f"task:{lesson['lesson_id']}:warmup:")
    assert first["source_refs"][0]["kind"] == "lesson_document"
    assert first["source_refs"][0]["offsets"]["end"] > first["source_refs"][0]["offsets"]["start"]

    reread = service.lesson_tasks(LEARNER, teacher["id"], lesson["lesson_id"], audience="teacher")
    assert [item["task_id"] for item in reread["items"]] == \
        [item["task_id"] for item in tasks["items"]], "ids stable across reads"

    # No answer keys anywhere in the public catalog projection.
    assert "1/2 больше" not in json.dumps(tasks, ensure_ascii=False)

    # A draft lesson is invisible to the student audience, even when a teacher
    # account requests the student projection.
    draft_tasks = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=20,
                                      lesson_text=LESSON_TEXT, answers_text="",
                                      validation=VALIDATION, request_id="catalog-draft-1",
                                      memory_revision="mem-rev-1")
    with pytest.raises(PermissionError):
        service.lesson_tasks(LEARNER, student["id"], draft_tasks["lesson_id"],
                             audience="student")
    service.confirm_lesson(LEARNER, teacher["id"], lesson["lesson_id"],
                           expected_revision=1, request_id="catalog-confirm-1")
    student_tasks = service.lesson_tasks(LEARNER, student["id"], lesson["lesson_id"],
                                         audience="student")
    assert [item["task_id"] for item in student_tasks["items"]] == \
        [item["task_id"] for item in tasks["items"]]

    # A teacher-requested student audience still follows actual permissions.
    assert student_tasks["audience"] == "student"
    with pytest.raises(UnifiedMemoryValidationError):
        service.lesson_tasks(LEARNER, teacher["id"], lesson["lesson_id"], audience="admin")


def test_lesson_without_parseable_tasks_or_legacy_lesson_reports_honestly(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher = accounts.teacher
    plain = service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=30,
                                lesson_text="# Урок: Тема\n## Объяснение\nПросто текст без заданий.",
                                answers_text="", validation=VALIDATION,
                                request_id="catalog-empty-1", memory_revision="mem-rev-1")
    tasks = service.lesson_tasks(LEARNER, teacher["id"], plain["lesson_id"], audience="teacher")
    assert tasks["catalog"]["status"] == "available" and tasks["items"] == []

    # A legacy lesson row (catalog NULL) reports the explicit limitation and no
    # read path ever writes a catalog.
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE teachcopilot.unified_lessons SET task_catalog=NULL WHERE id=%s",
                    (plain["lesson_id"],))
    legacy = service.lesson_tasks(LEARNER, teacher["id"], plain["lesson_id"], audience="teacher")
    assert legacy["catalog"]["status"] == "unavailable" and legacy["catalog"]["reason"]
    assert legacy["items"] == []
    rebuilt = service.rebuild_lesson_task_catalog(LEARNER, teacher["id"], plain["lesson_id"],
                                                  request_id="catalog-rebuild-1")
    assert rebuilt["rebuilt"] is True and rebuilt["catalog_status"] == "available"
    again = service.rebuild_lesson_task_catalog(LEARNER, teacher["id"], plain["lesson_id"],
                                                request_id="catalog-rebuild-2")
    assert again["rebuilt"] is False, "deterministic content is a no-op"
    with pytest.raises(PermissionError):
        service.rebuild_lesson_task_catalog(LEARNER, accounts.student["id"],
                                            plain["lesson_id"], request_id="catalog-rebuild-3")


def test_update_state_validates_pending_and_completed_task_ids(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = approved_lesson(service, accounts, request_id="validate-lesson-1")
    tasks = service.lesson_tasks(LEARNER, teacher["id"], lesson["lesson_id"], audience="teacher")
    task_id = tasks["items"][0]["task_id"]
    session_id = new_session(unified_store, accounts)
    service.attach_lesson(LEARNER, teacher["id"], session_id, lesson["lesson_id"],
                          request_id="validate-attach-1")
    state = service.update_lesson_state(
        LEARNER, student["id"], session_id, expected_revision=1,
        state={"stage": "warmup", "pending_task_id": task_id, "completed_task_ids": []},
        request_id="validate-state-1")
    assert state["applied"] and state["pending_task_id"] == task_id
    service.update_lesson_state(
        LEARNER, student["id"], session_id, expected_revision=2,
        state={"completed_task_ids": [task_id], "pending_task_id": None},
        request_id="validate-state-2")
    with pytest.raises(UnifiedMemoryValidationError):
        service.update_lesson_state(
            LEARNER, student["id"], session_id, expected_revision=3,
            state={"completed_task_ids": ["task:some-other-lesson:practice:1"]},
            request_id="validate-state-3")
    with pytest.raises(UnifiedMemoryValidationError):
        service.update_lesson_state(
            LEARNER, student["id"], session_id, expected_revision=3,
            state={"pending_task_id": "made-up-task"},
            request_id="validate-state-4")


# ------------------------------------------------- P1.2 claims and honesty
def test_claim_lease_blocks_and_expires(unified_store, accounts):
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-1", "Я люблю дроби и хомяков.", "Хорошо!")
    unified_store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    snapshot = unified_store.unified_session_snapshot(LEARNER, session_id)
    from local_app.unified_memory_processing import source_fingerprint
    fingerprint = source_fingerprint(snapshot["sources"])
    identity = "gemma4:12b@sha256:" + "d" * 64
    from local_app.unified_memory import _recipe_hash
    recipe = _recipe_hash()
    claim = unified_store.acquire_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                   identity, claimed_by="worker-1")
    assert claim["claimed"] is True and claim["claim_token"]
    blocked = unified_store.acquire_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                     identity, claimed_by="worker-2")
    assert blocked["claimed"] is False and blocked["holder"] == "worker-1"
    assert unified_store.extend_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                 identity, claim["claim_token"]) is True
    assert unified_store.extend_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                 identity, "00000000-0000-0000-0000-000000000000") is False
    assert unified_store.release_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                  identity, claim["claim_token"]) is True
    again = unified_store.acquire_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                   identity, claimed_by="worker-2")
    assert again["claimed"] is True
    # A short lease expires and is taken over atomically.
    short = unified_store.acquire_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                   identity, claimed_by="worker-3", lease_seconds=60)
    assert short["claimed"] is False
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""UPDATE teachcopilot.unified_processing_claims
            SET lease_expires_at=now()-interval '1 second'
            WHERE learner_id=%s AND session_id=%s""", (LEARNER, session_id))
    takeover = unified_store.acquire_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                                      identity, claimed_by="worker-4")
    assert takeover["claimed"] is True
    with pytest.raises(ValueError):
        unified_store.acquire_processing_claim(LEARNER, session_id, fingerprint, recipe,
                                               identity, claimed_by="worker-5", lease_seconds=10)


def test_processing_result_never_masquerades_failed_summary(unified_store, accounts):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-1", "Я люблю дроби.", "Хорошо!")
    unified_store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    failing = RecordingModelCall(fail_extract=True)
    from local_app.unified_memory import session_model_chunks
    from local_app.unified_memory_processing import session_model_messages
    snapshot = unified_store.unified_session_snapshot(LEARNER, session_id)
    chunks = session_model_chunks(LEARNER, session_id, session_model_messages(snapshot["sources"]))
    failing.fail_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    service = build_service(unified_store, accounts, model_call=failing)
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="honesty-1")
    assert result["status"] == "failed" and result["summary"] is None
    view = service.processing_result(LEARNER, accounts.student["id"], session_id)
    assert view["available"] is False and view["status"] == "failed"
    assert "summary" not in view, "a failed run never presents a summary"
    # A model identity change means a new run identity, not a replay.
    other = build_service(unified_store, accounts, runtime=FakeRuntime(chat_digest="9" * 64),
                          model_call=RecordingModelCall())
    second = other.process_session(LEARNER, accounts.teacher["id"], session_id,
                                   request_id="honesty-2")
    assert second["run_id"] != result["run_id"]
    assert second["status"] == "completed"


def test_corrections_reach_the_model_input_and_recall(unified_store, accounts):
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-1",
             "У меня есть хомяк, я люблю про него задачи.",
             "Отлично! Сравним 1/2 и 1/3 – какой числитель больше?")
    unified_store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    target = next(source["source_id"] for source in
                  unified_store.unified_session_snapshot(LEARNER, session_id)["sources"]
                  if source["author_role"] == "student")
    correction = unified_store.correct_memory_source(
        LEARNER, target, "Правка: хомяка зовут Пух.", accounts.teacher["id"],
        expected_revision=unified_store.get_revision(LEARNER),
        expected_source_revision=unified_store.get_learner_source_revision(LEARNER),
        client_correction_id="r2-correct-1")
    correction_id = correction["source"]["id"]
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="corrections-1")
    assert result["status"] == "completed"
    chunk_payloads = [json.loads(call["messages"][-1]["content"]) for call in result_model_calls(service)
                      if call["purpose"] == SUMMARY_CHUNK_PROMPT]
    assert chunk_payloads, "summary chunk calls must have been made"
    with_corrections = [payload for payload in chunk_payloads
                        if any(c["superseded_source_id"] == target for c in
                               payload.get("corrections_untrusted") or [])]
    assert with_corrections, "the corrections map must reach the actual model input"
    assert all(c["corrected_by_source_id"] == correction_id
               for payload in with_corrections for c in payload["corrections_untrusted"])
    # The student's own recall keeps their original words but marks the
    # supersession; the correction itself stays a separate teacher note
    # (teacher-reviewed evidence, never an own-conversation message).
    recall = service.recall(LEARNER, accounts.student["id"], "хомяк", budget_bytes=8000)
    original = [quote for quote in recall["sources"] if quote["source_id"] == target]
    assert original and original[0].get("superseded_by") == correction_id
    assert all(quote["source_id"] != correction_id for quote in recall["sources"]), \
        "a teacher correction is not part of the student's own conversation"
    refs = unified_store.memory_source_correction_refs(LEARNER, [correction_id])
    assert refs[correction_id] == target


def result_model_calls(service):
    return service.model_call.calls


def test_failed_run_retry_keeps_failed_artifact_inspectable(unified_store, accounts):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-1", "Я люблю дроби.", "Хорошо!")
    unified_store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))
    snapshot = unified_store.unified_session_snapshot(LEARNER, session_id)
    from local_app.unified_memory import session_model_chunks
    from local_app.unified_memory_processing import session_model_messages
    chunks = session_model_chunks(LEARNER, session_id, session_model_messages(snapshot["sources"]))
    failing = RecordingModelCall(fail_chunk_ids={chunk["chunk_id"] for chunk in chunks})
    service = build_service(unified_store, accounts, model_call=failing)
    failed = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="retry-1")
    assert failed["status"] == "failed"
    # The failed artifact stays retrievable and a retry with a healthy model
    # runs under a fresh request id.
    retried_service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    retried = retried_service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                              request_id="retry-1")
    assert retried["run_id"] != failed["run_id"]
    assert retried["status"] == "completed"
    view = service.processing_result(LEARNER, accounts.teacher["id"], session_id)
    assert view["status"] == "completed"
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT count(*) AS total FROM teachcopilot.unified_processing_runs
            WHERE learner_id=%s AND session_id=%s AND status='failed'""",
            (LEARNER, session_id))
        assert cur.fetchone()["total"] == 1, "the failed artifact stays inspectable"


# --------------------------------------------------- pure catalog derivation
def test_build_task_catalog_is_conservative_and_deterministic():
    lesson_id = "12345678-1111-4111-8111-111111111111"
    text = ("# Урок: Тема\n\n## Цель\nЦель без заданий.\n"
            "## Разминка\nРазогрев без заданий.\n"
            "## Практика\n1. Первое задание.\n- не задача\n2) Второе задание?\n"
            "Случайный абзац без явного задания.\n"
            "## Ответы\n1) ответ")
    first = build_task_catalog(lesson_id, text)
    second = build_task_catalog(lesson_id, text)
    assert first == second
    assert [item["task_id"] for item in first["items"]] == [
        "task:%s:practice:1" % lesson_id, "task:%s:practice:2" % lesson_id]
    assert first["items"][0]["number"] == 1
    assert first["items"][1]["number"] == 2
    assert all("ответ" not in item["prompt_text"].lower() or "задание" in item["prompt_text"].lower()
               for item in first["items"])
    # A line that is not an explicit exercise or question never becomes a task.
    assert all("Случайный абзац" not in item["prompt_text"] for item in first["items"])
    assert all("не задача" not in item["prompt_text"] for item in first["items"])
    assert build_task_catalog(lesson_id, "")["items"] == []
    assert build_task_catalog(lesson_id, None)["items"] == []


# --------------------------------------------------------------- P2 plans
def test_lesson_plan_context_is_bounded_redacted_and_learner_scoped(unified_store, accounts):
    service = build_service(unified_store, accounts)
    teacher, student = accounts.teacher, accounts.student
    lesson = approved_lesson(service, accounts, request_id="p2-lesson-1", answers="1) 1/2 больше")
    seeded_completed_session = new_session(unified_store, accounts)
    exchange(unified_store, accounts, seeded_completed_session, "msg-1",
             "Мы разбирали дроби сегодня.", "Хорошо, продолжим с дробями.")
    unified_store.complete_learning_session(seeded_completed_session, LEARNER,
                                            accounts.student["id"])
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM teachcopilot.memory_jobs WHERE kind='memory_update' AND learner_id=%s",
                    (LEARNER,))

    teacher_recall = service.recall(LEARNER, teacher["id"], "дроби", budget_bytes=6000)
    documents = teacher_recall["context_documents"]
    assert documents and documents[0]["kind"] == "lesson_plan"
    assert documents[0]["lesson_id"] == lesson["lesson_id"]
    assert documents[0]["source_refs"][0]["kind"] == "lesson_document"
    assert documents[0]["truncated"] in (True, False) and documents[0]["text"]
    assert documents[0]["context_note"]
    assert teacher_recall["diagnostics"]["context_documents_count"] == len(documents)

    student_recall = service.recall(LEARNER, student["id"], "дроби", budget_bytes=6000)
    student_documents = student_recall["context_documents"]
    assert student_documents, "the student sees their own approved plan as context"
    assert "1/2 больше" not in json.dumps(student_documents, ensure_ascii=False), \
        "student plan context is redacted like the lesson projection"
    # Draft lessons never become context; other learners never leak.
    service.save_lesson(LEARNER, teacher["id"], topic_id=TOPIC, minutes=20,
                        lesson_text="# Урок: Черновик дробей\n## Практика\n1. Черновик.",
                        answers_text="", validation=VALIDATION,
                        request_id="p2-draft-1", memory_revision="mem-rev-1")
    draft_recall = service.recall(LEARNER, student["id"], "черновик дроби", budget_bytes=6000)
    assert all("Черновик" not in document["text"] for document in draft_recall["context_documents"]), \
        "draft lessons are not eligible plan context"
    other_lesson = service.save_lesson(OTHER_LEARNER, teacher["id"], topic_id="math.g4.other",
                                       minutes=30, lesson_text="# Урок: Дроби другого ученика",
                                       answers_text="", validation=VALIDATION,
                                       request_id="p2-other-1", memory_revision="mem-rev-1")
    service.confirm_lesson(OTHER_LEARNER, teacher["id"], other_lesson["lesson_id"],
                           expected_revision=1, request_id="p2-other-confirm")
    student_recall2 = service.recall(LEARNER, student["id"], "дроби", budget_bytes=6000)
    assert all(document["lesson_id"] != other_lesson["lesson_id"]
               for document in student_recall2["context_documents"]), \
        "another learner's plan never enters this learner's context"


def test_queue_constraints_still_accept_r2_job_kinds(unified_store, accounts):
    job = unified_store.enqueue_memory_job(
        LEARNER, TOPIC, "session_process",
        {"learning_session_id": str(new_session(unified_store, accounts)),
         "client_request_id": "r2-sp-1"},
        actor_id=accounts.teacher["id"], method="personal_memory",
        idempotency_key="r2-session-process:sp-1")
    assert job["kind"] == "session_process"
    with pytest.raises(MemoryIdempotencyConflict):
        unified_store.enqueue_memory_job(
            LEARNER, TOPIC, "session_process",
            {"learning_session_id": str(new_session(unified_store, accounts)),
             "client_request_id": "r2-sp-2"},
            actor_id=accounts.teacher["id"], method="personal_memory",
            idempotency_key="r2-session-process:sp-1")
