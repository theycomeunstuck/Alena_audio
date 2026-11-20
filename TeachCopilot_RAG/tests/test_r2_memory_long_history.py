"""Round-2 synthetic archive verification (client A, P1.3).

Three frozen six-session histories are loaded into a disposable PostgreSQL
through the immutable source-key mapping. This file verifies A's side: the
archives load honestly, the oversized history exceeds the 32768-token window,
every gold key resolves, retrieval reaches the gold sources for the
development histories with a deterministic recording embedder, long-session
processing coverage stays honest with a middle-chunk failure, and gold answers
never enter model inputs. The live model quality evaluation itself belongs to
client B and is only protocolized here.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from local_app.auth import hash_password
from local_app.store import Store
from local_app.unified_memory import UnifiedMemoryService
from local_app.unified_memory_processing import (
    EXTRACT_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_REDUCE_PROMPT,
)
from tests.r2_memory_quality.loader import (HISTORIES_DIR, load_corrections, load_history,
                                            load_history_file)

VECTOR_DIMENSIONS = 1024
TOPIC = "math.g4.synthetic"


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
        for learner in ("demo-nika-04", "demo-luna-03"):
            shutil.copy(repo / "learner-data/demo/learners" / f"{learner}.json", learners)
        store.import_repository(tmp_path)
        yield store
    finally:
        with admin.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
            cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))
        admin.close()


class HashRuntime:
    """Deterministic token-hash embedder with an immutable identity.

    No model service: vectors are bag-of-token hashes. Morphology is left to
    the PostgreSQL full-text channel; this embedder only exercises the vector
    channel end to end.
    """

    def __init__(self):
        self.settings = SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                        ollama_url="http://127.0.0.1:11434")
        self.embedded = 0

    def model_identity(self, model_name=None):
        return "recording-gemma@sha256:" + "a" * 64

    def embedding_model_identity(self):
        return "hash-embedder@sha256:" + "b" * 64

    def embed(self, texts):
        self.embedded += len(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        import hashlib
        vector = [0.0] * VECTOR_DIMENSIONS
        token = []
        for char in text.casefold() + " ":
            if char.isalpha() or char.isdigit():
                token.append(char)
            elif token:
                digest = hashlib.sha256("".join(token).encode("utf-8")).digest()
                vector[int.from_bytes(digest[:4], "big") % VECTOR_DIMENSIONS] = 1.0
                token = []
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


class RecordingModelCall:
    model_identity_static = "recording-gemma@sha256:" + "e" * 64

    def __init__(self, *, fail_chunk_ids=()):
        self.calls = []
        self.fail_chunk_ids = set(fail_chunk_ids)

    def model_identity(self):
        return self.model_identity_static

    def __call__(self, messages, purpose, schema):
        self.calls.append({"purpose": purpose, "messages": messages})
        import json as json_module
        if purpose == SUMMARY_CHUNK_PROMPT:
            chunk = json_module.loads(messages[-1]["content"])
            if chunk.get("chunk_id") in self.fail_chunk_ids:
                raise RuntimeError("recording chunk failure")
            return json_module.dumps({"studied": ["арифметика"],
                                      "achieved": ["ученик решил примеры"],
                                      "repeat": ["неверный ответ в одном примере"],
                                      "next_step": ["продолжить упражнения"]},
                                     ensure_ascii=False)
        if purpose == SUMMARY_REDUCE_PROMPT:
            return json_module.dumps({"studied": ["арифметика"],
                                      "achieved": ["ученик решил примеры"],
                                      "repeat": ["неверный ответ в одном примере"],
                                      "next_step": ["продолжить упражнения"]},
                                     ensure_ascii=False)
        if purpose == EXTRACT_PROMPT:
            source_ids = json_module.loads(messages[-1]["content"])["source_ids"]
            return json_module.dumps({"changes": [], "candidates": []}, ensure_ascii=False)
        raise AssertionError(f"unexpected purpose {purpose}")


def make_service(store, accounts, *, model_call=None, runtime=None):
    return UnifiedMemoryService(store, runtime or HashRuntime(), model_call=model_call)


def load_archive(store, history_id):
    """Create accounts for a fresh synthetic learner and load the archive."""
    import copy
    teacher = store.create_account(f"r2q-{history_id}-teacher@synthetic.test", "Teacher",
                                   hash_password("R2 quality password 123"), "teacher")
    learner_id = f"r2q-{history_id}"
    repo = Path(__file__).resolve().parents[2]
    template = json.loads((repo / "learner-data" / "demo" / "learners" / "demo-nika-04.json")
                          .read_text(encoding="utf-8"))
    card = copy.deepcopy(template)
    card["learner_id"] = learner_id
    card["pseudonym"] = f"Синтетика-{history_id}"
    store.create_assigned_learner(card, teacher["id"])
    student = store.create_student_account(f"r2q-{history_id}-student@synthetic.test",
                                           "Student", hash_password("R2 quality password 123"),
                                           learner_id, teacher["id"])
    history = load_history_file(history_id)
    mapping = load_history(store, history, learner_id=learner_id,
                           student_actor_id=student["id"], teacher_actor_id=teacher["id"])
    return {"learner_id": learner_id, "teacher": teacher, "student": student,
            "mapping": mapping, "history": history}


MANIFEST = json.loads((Path(__file__).resolve().parent / "r2_memory_quality" / "manifest.json")
                      .read_text(encoding="utf-8"))


def test_frozen_files_are_consistent_and_complete():
    """The frozen package itself must be internally consistent before any run."""
    histories = {history["history_id"] for history in MANIFEST["histories"]}
    assert histories == {"r2-mem-a", "r2-mem-b", "r2-mem-final"}
    assert len(MANIFEST["questions"]) == 18
    per_history = {}
    for question in MANIFEST["questions"]:
        per_history.setdefault(question["history_id"], []).append(question)
    assert set(per_history) == histories
    assert all(len(questions) == 6 for questions in per_history.values())
    covered = {question["case"] for question in MANIFEST["questions"]}
    assert covered == set(MANIFEST["case_types"]), "every declared case type is covered"
    final_questions = [question["question_id"] for question in MANIFEST["questions"]
                       if question["history_id"] == "r2-mem-final"]
    assert len(final_questions) == 6
    # Match the exact source keys the loader creates for each history.
    source_keys, semantic_keys = {}, {}
    for history_id in histories:
        data = load_history_file(history_id)
        turns = {turn["key"]: turn for session in data["sessions"] for turn in session["turns"]}
        keys = set(turns)
        keys.update(f"{key}:u" for key in turns)
        keys.update(f"{key}:a" for key, turn in turns.items() if turn.get("assistant"))
        aliases = {}
        for session in data["sessions"]:
            aliases.update(session.get("key_block_keys") or {})
        for alias, concrete in aliases.items():
            assert concrete in turns, f"semantic key {alias} refers to an unknown turn"
            keys.add(alias)
            if turns[concrete].get("assistant"):
                keys.add(f"{alias}:a")
        source_keys[history_id] = keys
        semantic_keys[history_id] = set(aliases)
    for correction in load_corrections():
        history_id = correction["history_id"]
        assert history_id in histories
        assert correction["corrects_key"] in semantic_keys[history_id]
        source_keys[history_id].add(correction["key"])
    for question in MANIFEST["questions"]:
        for key in question["expected_source_keys"]:
            assert key in source_keys[question["history_id"]], \
                f"manifest key {key} is not frozen in {question['history_id']}"


def test_oversized_history_exceeds_the_model_window(unified_store):
    """History A exceeds 32768 tokens even under a conservative char/token ratio."""
    archive = load_archive(unified_store, "r2-mem-a")
    total_chars = sum(len(turn["text"]) + len(turn.get("assistant") or "")
                      for session in archive["history"]["sessions"]
                      for turn in session["turns"])
    assert total_chars >= 110000, "the oversized archive must exceed the 32768-token window"
    # Conservative bound: Russian text at 3.2 chars/token is still > 34000 tokens.
    assert total_chars / 3.2 >= 32768
    sessions = unified_store.list_completed_learning_sessions(
        archive["learner_id"], archive["student"]["id"])
    assert len(sessions["sessions"]) == 6


def test_gold_keys_resolve_and_seven_day_view_holds(unified_store):
    """The immutable mapping resolves and the old facts sit outside 7 active days."""
    archive = load_archive(unified_store, "r2-mem-a")
    for question in MANIFEST["questions"]:
        if question["history_id"] != "r2-mem-a":
            continue
        for key in question["expected_source_keys"]:
            assert key in archive["mapping"], f"gold key {key} did not resolve"
    messages = unified_store.get_own_conversation_messages(
        archive["learner_id"], archive["student"]["id"], limit=10000, active_day_limit=7)
    recent_ids = {message["source_id"] for message in messages}
    all_messages = unified_store.get_own_conversation_messages(
        archive["learner_id"], archive["student"]["id"], limit=10000, active_day_limit=None)
    assert len(all_messages) > len(messages), "the archive spans more than seven active days"
    aquarium = archive["mapping"]["a.aquarium"]
    dog = archive["mapping"]["a.dog_baitik"]
    assert aquarium not in recent_ids and dog not in recent_ids, \
        "the old gold facts must lie outside the seven-active-day view"
    days = {str(message["activity_day"]) for message in all_messages}
    assert len(days) >= 6, "six sessions on distinct days"


@pytest.mark.parametrize("history_id", ["r2-mem-a", "r2-mem-b"])
def test_retrieval_reaches_gold_sources(unified_store, history_id):
    """Retrieval checks on the development histories (recording embedder)."""
    archive = load_archive(unified_store, history_id)
    service = make_service(unified_store, archive)
    found, missed = [], []
    for question in MANIFEST["questions"]:
        if question["history_id"] != history_id or question.get("absence"):
            continue
        recall = service.recall(archive["learner_id"], archive["student"]["id"],
                                question["probe_query"], budget_bytes=20000)
        quoted_ids = {quote["source_id"] for quote in recall["sources"]}
        user_ids = set(recall["source_ids"])
        for key in question["expected_source_keys"]:
            source_id = archive["mapping"][key]
            if source_id in quoted_ids or source_id in user_ids:
                found.append(f"{question['question_id']}:{key}")
            else:
                missed.append(f"{question['question_id']}:{key}")
        if question.get("absence"):
            assert recall["sources"] == [] or not recall["text"]
    assert missed == [], \
        f"gold sources not reachable through recall: {missed}; found {len(found)}"
    assert found


def test_absence_questions_have_no_borrowed_evidence(unified_store):
    """Missing-fact questions must not return plausible-but-wrong archives."""
    archive = load_archive(unified_store, "r2-mem-b")
    service = make_service(unified_store, archive)
    for question in MANIFEST["questions"]:
        if question["history_id"] != "r2-mem-b" or not question.get("absence"):
            continue
        recall = service.recall(archive["learner_id"], archive["student"]["id"],
                                question["question"], budget_bytes=6000)
        claimed_color = [quote for quote in recall["sources"]
                         if "цвет" in quote["text"].casefold()]
        assert claimed_color == [], "the archive never states a favorite color"


def test_long_session_processing_coverage_and_middle_failure(unified_store):
    """Every processed chunk is covered; a middle failure degrades to partial."""
    archive = load_archive(unified_store, "r2-mem-a")
    service = make_service(unified_store, archive)
    sessions = unified_store.list_completed_learning_sessions(
        archive["learner_id"], archive["student"]["id"])
    session_ids = [row["learning_session_id"] for row in sessions["sessions"]]
    assert len(session_ids) == 6
    from local_app.unified_memory import session_model_chunks
    from local_app.unified_memory_processing import session_model_messages
    snapshot = unified_store.unified_session_snapshot(archive["learner_id"], session_ids[0])
    chunks = session_model_chunks(archive["learner_id"], session_ids[0],
                                  session_model_messages(snapshot["sources"]))
    assert len(chunks) >= 15, "an oversized session must split into many bounded chunks"
    # A failure in a middle chunk of the second session.
    mid_snapshot = unified_store.unified_session_snapshot(archive["learner_id"], session_ids[1])
    mid_chunks = session_model_chunks(archive["learner_id"], session_ids[1],
                                      session_model_messages(mid_snapshot["sources"]))
    failing = RecordingModelCall(fail_chunk_ids={mid_chunks[len(mid_chunks) // 2]["chunk_id"]})
    failing_service = make_service(unified_store, archive, model_call=failing)
    result = failing_service.process_session(archive["learner_id"], archive["teacher"]["id"],
                                             session_ids[1], request_id="r2q-mid-fail")
    assert result["status"] == "partial" and result["provisional"] is True
    assert mid_chunks[len(mid_chunks) // 2]["chunk_id"] in result["coverage"]["failed_chunk_ids"]
    assert sorted(result["coverage"]["chunk_ids"]) == \
        sorted(result["coverage"]["included_chunk_ids"] + result["coverage"]["failed_chunk_ids"]), \
        "coverage must refer to all processed chunks"
    summary = result["summary"]
    assert summary["provisional"] is True

    # A healthy processing run of the first session covers every chunk.
    healthy = make_service(unified_store, archive, model_call=RecordingModelCall())
    complete = healthy.process_session(archive["learner_id"], archive["teacher"]["id"],
                                       session_ids[0], request_id="r2q-full")
    assert complete["status"] == "completed"
    assert sorted(complete["coverage"]["chunk_ids"]) == sorted(
        complete["coverage"]["included_chunk_ids"])
    assert len(complete["coverage"]["chunk_ids"]) == len(chunks)


def test_gold_answers_never_enter_model_inputs(unified_store):
    """Model inputs carry archive text only, never manifest gold wording."""
    archive = load_archive(unified_store, "r2-mem-b")
    model = RecordingModelCall()
    service = make_service(unified_store, archive, model_call=model)
    sessions = unified_store.list_completed_learning_sessions(
        archive["learner_id"], archive["student"]["id"])
    session_id = sessions["sessions"][0]["learning_session_id"]
    service.process_session(archive["learner_id"], archive["teacher"]["id"], session_id,
                            request_id="r2q-prompt-safety")
    model_input = json.dumps([call["messages"] for call in model.calls], ensure_ascii=False)
    for question in MANIFEST["questions"]:
        gold = question["expected_answer"]
        assert gold[:60] not in model_input, \
            f"gold wording for {question['question_id']} leaked into a model prompt"
    # The manifest is evaluation-side: no product module imports the package.
    import local_app.unified_memory as facade_module
    assert "r2_memory_quality" not in facade_module.__file__
