"""Regression tests for the post-ready review fixes F1–F7 (client B).

A's post-ready code review (A-report.md section 7) documented crash paths in
the frozen memory-backend commit; the fixes are applied in the merged B tree
as integration corrections. Unit tests run without a database; F1/F4/F5/F6
run against a fresh disposable PostgreSQL database. No live model is used.
"""
import json

import pytest

from local_app.memory_store import MemoryQueueFullError
from local_app.unified_memory import UnifiedMemoryService, UnifiedMemoryUnavailable
from local_app.unified_memory_processing import (
    EXTRACT_PROMPT,
    SUMMARY_CHUNK_PROMPT,
    normalize_chunk_summary,
    validate_extraction,
)
from local_app.unified_memory_store import UnifiedMemoryStoreMixin

from tests.test_unified_memory_postgres import (  # noqa: F401 - pytest fixtures
    accounts,
    unified_store,
)
from tests.test_unified_memory_postgres import (
    LEARNER,
    RecordingModelCall,
    build_service,
    exchange,
    new_session,
)

NON_OBJECT_CHUNK = json.dumps(["не объект"])


# --------------------------------------------------------------------- F2
def test_validate_extraction_keeps_nonempty_value_for_remove():
    chunk = {"chunk_id": "c1", "source_ids": ["s1"]}
    source = {"source_id": "s1", "author_role": "student", "text": "раньше любил роботов, теперь нет",
              "session_id": "sess"}
    payload = {"changes": [{"category": "interest", "operation": "remove",
                            "value": "роботы", "scope": "durable", "explicit_durable": True,
                            "source_id": "s1", "evidence_text": "любил роботов"}],
               "candidates": []}
    changes, candidates, warnings = validate_extraction(payload, chunk, {"s1": source})
    assert not warnings
    assert changes[0]["operation"] == "remove"
    assert changes[0]["value"] == "роботы"  # F2: the withdrawn fact stays literal


def test_validate_extraction_rejects_empty_remove_value():
    chunk = {"chunk_id": "c1", "source_ids": ["s1"]}
    source = {"source_id": "s1", "author_role": "student", "text": "не знаю", "session_id": "sess"}
    payload = {"changes": [{"category": "interest", "operation": "remove",
                            "value": "", "scope": "durable", "explicit_durable": True,
                            "source_id": "s1", "evidence_text": "не знаю"}],
               "candidates": []}
    changes, candidates, warnings = validate_extraction(payload, chunk, {"s1": source})
    assert not changes and not candidates
    assert any(warning["code"] == "invalid_value" for warning in warnings)


def test_store_change_validation_requires_value_for_remove():
    with pytest.raises(ValueError):
        UnifiedMemoryStoreMixin._validate_change_fields("interest", "remove", "", "durable")
    # A nonempty remove value is accepted.
    assert UnifiedMemoryStoreMixin._validate_change_fields(
        "interest", "remove", "роботы", "durable") is None


# --------------------------------------------------------------------- F3
def test_normalize_chunk_summary_rejects_non_object_payload():
    with pytest.raises(ValueError):
        normalize_chunk_summary(NON_OBJECT_CHUNK)
    with pytest.raises(ValueError):
        normalize_chunk_summary('"строка"')


def test_chunk_summary_non_object_degrades_one_chunk(unified_store, accounts):
    """A schema-ignoring model response fails that chunk, not the whole run."""
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f3-1", "хомяк – мой питомец",
             "Расскажи про дроби: числитель и знаменатель.")

    class NonObjectChunkCall(RecordingModelCall):
        def __call__(self, messages, purpose, schema):
            if purpose == SUMMARY_CHUNK_PROMPT:
                return NON_OBJECT_CHUNK  # valid JSON, not an object (A-report F3)
            if purpose == EXTRACT_PROMPT:
                source_ids = json.loads(messages[-1]["content"])["source_ids"]
                return json.dumps({"changes": [], "candidates": []}, ensure_ascii=False)
            return super().__call__(messages, purpose, schema)

    service = build_service(unified_store, accounts, model_call=NonObjectChunkCall())
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="f3-run-1")
    # The run completes with honest coverage instead of crashing (F3).
    assert result["status"] in ("completed", "failed")
    assert result["coverage"]["chunk_ids"]
    assert result["coverage"]["chunk_ids"] == result["coverage"]["failed_chunk_ids"]
    assert any(warning["code"] == "summary_chunk_failed" for warning in result["warnings"])


# --------------------------------------------------------------------- F1
def test_failed_run_retry_saves_new_run_without_unique_violation(unified_store, accounts):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f1-1", "хомяк – мой питомец",
             "Расскажи про дроби: числитель и знаменатель.")
    service = build_service(unified_store, accounts,
                            model_call=RecordingModelCall(fail_chunk_ids={"*"}))

    class AlwaysFail(RecordingModelCall):
        def __call__(self, messages, purpose, schema):
            raise RuntimeError("recording total failure")

    failed = build_service(unified_store, accounts, model_call=AlwaysFail()).process_session(
        LEARNER, accounts.teacher["id"], session_id, request_id="f1-run")
    assert failed["status"] == "failed"

    retried = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                      request_id="f1-run")
    # F1: the retry (<request_id>:r2) inserts its own run row; no UniqueViolation.
    assert retried["status"] == "completed"
    assert retried["run_id"] != failed["run_id"]
    runs = unified_store.latest_unified_run(LEARNER, session_id)
    assert str(runs["id"]) == retried["run_id"]


def test_failed_run_artifact_stays_retrievable(unified_store, accounts):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f1-2", "хомяк – мой питомец",
             "Расскажи про дроби: числитель и знаменатель.")

    class AlwaysFail(RecordingModelCall):
        def __call__(self, messages, purpose, schema):
            raise RuntimeError("recording total failure")

    service = build_service(unified_store, accounts, model_call=AlwaysFail())
    failed = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="f1-artifact")
    assert failed["status"] == "failed"
    by_request = unified_store.get_unified_run_by_request(LEARNER, "f1-artifact")
    assert by_request is not None and str(by_request["id"]) == failed["run_id"]


# --------------------------------------------------------------------- F4
def test_recall_without_runtime_degrades_to_lexical(unified_store, accounts):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f4-1", "хомяк – мой питомец",
             "Расскажи про дроби: числитель и знаменатель.")
    unified_store.complete_learning_session(session_id, LEARNER, accounts.student["id"])
    service = UnifiedMemoryService(unified_store, None)  # A-report F4: runtime=None
    result = service.recall(LEARNER, accounts.student["id"], "хомяк", session_id=session_id)
    assert result["method"] == "conversation_retrieval"
    assert result["diagnostics"]["degraded"] is True
    assert result["diagnostics"]["degraded_reason"] == "embedding_unavailable"


# --------------------------------------------------------------------- F5
def test_recall_database_outcome_maps_to_unavailable(unified_store, accounts, monkeypatch):
    import psycopg2

    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f5-1", "хомяк – мой питомец",
             "Расскажи про дроби: числитель и знаменатель.")
    service = build_service(unified_store, accounts)

    def broken_search(*args, **kwargs):
        raise psycopg2.OperationalError("search backend is down")

    monkeypatch.setattr(unified_store, "search_conversation_chunks", broken_search)
    with pytest.raises(UnifiedMemoryUnavailable):
        service.recall(LEARNER, accounts.student["id"], "хомяк", session_id=session_id)


# --------------------------------------------------------------------- F6
def test_candidate_acceptance_applies_canonically_in_one_transaction(unified_store, accounts):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f6-1", "хомяк – мой питомец",
             "Я путаю числитель и знаменатель при сравнении дробей.")
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="f6-run")
    candidates = result["academic_candidates"]
    assert candidates, "the recording model proposes one error-pattern candidate"
    candidate_id = candidates[0]["candidate_id"]
    changes = service.list_changes(LEARNER, accounts.teacher["id"])

    review = service.review_changes(LEARNER, accounts.teacher["id"],
                                    actions=[{"id": candidate_id, "action": "accept"}],
                                    expected_revision=changes["revision"],
                                    request_id="f6-accept")

    assert review["applied"] == 1
    # The canonical note source and its durable job committed with the decision.
    import psycopg2
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, decided_by FROM teachcopilot.unified_academic_candidates WHERE id=%s",
                    (candidate_id,))
        assert cur.fetchone()["status"] == "accepted"
        cur.execute("""SELECT id FROM teachcopilot.memory_sources
            WHERE source_key=%s AND approved IS TRUE""", (f"unified-candidate:{candidate_id}",))
        source_row = cur.fetchone()
        assert source_row is not None
        cur.execute("""SELECT kind, status FROM teachcopilot.memory_jobs
            WHERE idempotency_key=%s""", (f"source:{source_row['id']}",))
        job = cur.fetchone()
        assert job is not None and job["kind"] == "memory_update"


def test_queue_full_rolls_back_acceptance(unified_store, accounts, monkeypatch):
    session_id = new_session(unified_store, accounts)
    exchange(unified_store, accounts, session_id, "msg-f6-2", "хомяк – мой питомец",
             "Я путаю числитель и знаменатель при сравнении дробей.")
    service = build_service(unified_store, accounts, model_call=RecordingModelCall())
    result = service.process_session(LEARNER, accounts.teacher["id"], session_id,
                                     request_id="f6-run-2")
    candidate_id = result["academic_candidates"][0]["candidate_id"]
    changes = service.list_changes(LEARNER, accounts.teacher["id"])

    def full_queue(*args, **kwargs):
        raise MemoryQueueFullError("memory queue contains 40 waiting requests")

    monkeypatch.setattr(unified_store, "_enqueue_memory_job", full_queue)
    with pytest.raises(UnifiedMemoryUnavailable):
        service.review_changes(LEARNER, accounts.teacher["id"],
                               actions=[{"id": candidate_id, "action": "accept"}],
                               expected_revision=changes["revision"],
                               request_id="f6-accept-2")
    # F6: the acceptance rolled back – the candidate stays pending, no note.
    import psycopg2
    with unified_store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT status FROM teachcopilot.unified_academic_candidates WHERE id=%s",
                    (candidate_id,))
        assert cur.fetchone()["status"] == "pending"
        cur.execute("SELECT count(*) AS n FROM teachcopilot.memory_sources WHERE source_key=%s",
                    (f"unified-candidate:{candidate_id}",))
        assert cur.fetchone()["n"] == 0
