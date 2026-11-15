"""Regression tests for audit findings F-031, F-043 and F-044.

F-031 – complete_learning_session must take the memory-queue advisory lock
before it row-locks the session and learning-message rows, matching the
claim/enqueue order; otherwise a concurrent claim_memory_job can deadlock
(PostgreSQL 40P01 aborts one of the two transactions).

F-043 – one long earlier learner message (the API allows up to 12000
characters) must never exceed the 4000-character retrieval anchor budget and
turn every follow-up in the conversation into a permanent failure; the
prior-question anchor is clamped to the remaining budget instead.

F-044 – the resolver's mandatory-size reservation must include the late
system-prompt appends, so priors admitted by the admission formula always fit
the complete-memory byte check performed by chat().

Pure Python fakes; no database, no network, no model calls.
"""
from __future__ import annotations

import ast
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from local_app.memory_store import MemoryStoreMixin
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime

TEACHCOPILOT_RAG = Path(__file__).resolve().parents[1]

ANCHOR_BUDGET = 4000
CONTEXT_TOKENS = 8192


# ---------------------------------------------------------------------------
# F-031: lock-order inversion in complete_learning_session
# ---------------------------------------------------------------------------

class _RecordingCursor:
    def __init__(self, statements, responses):
        self._statements = statements
        self._responses = responses
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, sql, params=None):
        self._statements.append(" ".join(str(sql).split()))

    def fetchone(self):
        return self._responses.pop(0) if self._responses else None

    def fetchall(self):
        return []


class _RecordingConnection:
    def __init__(self, statements, responses):
        self._statements = statements
        self._responses = responses

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        return _RecordingCursor(self._statements, self._responses)


class _RecordingStore(MemoryStoreMixin):
    """Drive MemoryStoreMixin against a fake connection that records SQL order."""

    def __init__(self, responses):
        self.statements: list[str] = []
        self._responses = list(responses)

    def _connect(self):
        return _RecordingConnection(self.statements, self._responses)


def _first_index(statements, marker):
    for index, sql in enumerate(statements):
        if marker in sql:
            return index
    return None


def test_complete_learning_session_takes_queue_advisory_lock_before_row_locks():
    # Scripted fetchone responses for the whole completion flow: session row,
    # pending-message check, learner row, idempotency check, queue count, job.
    responses = [
        {"id": "session-1", "learner_id": "learner-a", "actor_id": "parent-1",
         "status": "completed", "topic_id": "math.geometry"},
        None,
        {"revision": 1, "memory_source_revision": 0, "memory_chat_source_revision": 0},
        None,
        {"count": 0},
        {"id": "job-1", "learner_id": "learner-a", "kind": "memory_update",
         "provider": "local", "status": "queued"},
    ]
    store = _RecordingStore(responses)
    result = store.complete_learning_session("session-1", "learner-a", "parent-1")
    assert result["job_id"] == "job-1"

    statements = store.statements
    advisory = _first_index(statements, "pg_advisory_xact_lock")
    session_row_lock = _first_index(statements, "FROM teachcopilot.learning_sessions")
    message_row_lock = _first_index(statements, "UPDATE teachcopilot.learning_messages")
    assert advisory is not None, "complete_learning_session never acquired the queue advisory lock"
    assert session_row_lock is not None, "session row lock (SELECT ... FOR UPDATE) not reached"
    assert message_row_lock is not None, "message row lock (_recover_chat_messages) not reached"
    assert "teachcopilot_memory_queue" in statements[advisory], (
        "the advisory lock taken is not the memory-queue lock used by claim/enqueue")
    assert advisory < session_row_lock, (
        "lock-order inversion: the queue advisory lock (position %d) must be acquired "
        "before the learning_sessions row lock (position %d), matching claim/enqueue"
        % (advisory, session_row_lock))
    assert advisory < message_row_lock, (
        "lock-order inversion: the queue advisory lock (position %d) must be acquired "
        "before the learning_messages row locks (position %d), matching claim/enqueue"
        % (advisory, message_row_lock))


# ---------------------------------------------------------------------------
# F-043: unclamped prior-question retrieval anchor
# ---------------------------------------------------------------------------

class _StubRuntime:
    context_tokens = CONTEXT_TOKENS

    def approved_memory_request(self):
        @contextmanager
        def _scope():
            yield
        return _scope()

    def resolve_chat_query(self, query, prior_messages, subject_id=None, **kwargs):
        # Mirrors a resolver decision that anchors to the long earlier question
        # and the approved memory at the same time (combined antecedents).
        return {"in_scope": True, "uses_prior_context": True,
                "search_query": "отрезки и лучи",
                "personal_memory_recall": False,
                "uses_approved_memory": True,
                "prior_learner_question_index": 0}

    def manifest(self):
        return {}


class _StubSettings:
    chat_model = "stub"


class _StubAnswerAdapter:
    def answer(self, messages, **kwargs):
        return {"id": "stub-answer", "created_at": None, "model": "stub",
                "prompt_eval_count": 0, "eval_count": 0,
                "message": {"role": "assistant",
                            "content": '{"answer":"Разберём отрезки по шагам.","citations":[]}'}}


def test_long_prior_question_is_clamped_to_the_anchor_budget_not_fatal():
    # One earlier learner message of 4512 characters (the API allows up to
    # 12000) plus a short follow-up must still complete.
    long_question = "Реши задачу про отрезки и лучи: " + ("длинное условие про отрезки " * 160)
    assert len(long_question) > ANCHOR_BUDGET
    messages = [
        {"role": "user", "content": long_question},
        {"role": "assistant", "content": "Давай разберём эту задачу вместе."},
        {"role": "user", "content": "а теперь объясни это ещё раз"},
    ]
    memory_context = {"method": "personal_memory", "version_id": "v1",
                      "topic_id": "math.geometry",
                      "text": "Ученик решал задачу про отрезки.", "sources": []}

    service = RagService(store=None, runtime=_StubRuntime(), settings=_StubSettings(),
                         answer_adapter=_StubAnswerAdapter())
    queries: list[str] = []

    def _fake_search(query, *args, **kwargs):
        queries.append(query)
        return []

    service.search = _fake_search

    result = service.answer(messages, subject_id="math", topic_id="math.geometry",
                            memory_context=memory_context)
    assert result["choices"][0]["message"]["content"] == "Разберём отрезки по шагам."
    assert queries, "the clamped anchor must still reach retrieval"

    marker = "Earlier learner question: "
    assert any(marker in query for query in queries), (
        "the prior-question anchor must remain part of the retrieval query")
    query = next(query for query in queries if marker in query)
    clamped_anchor = query[query.index(marker):]
    assert len(clamped_anchor) <= ANCHOR_BUDGET, (
        "the prior-question anchor was not clamped to the %d-character budget" % ANCHOR_BUDGET)
    assert len(clamped_anchor) < len(marker) + len(long_question), (
        "the long earlier message was not truncated")


# ---------------------------------------------------------------------------
# F-044: resolver slack undercounts the late system-prompt appends
# ---------------------------------------------------------------------------

class _DeadSession:
    def post(self, *args, **kwargs):
        raise AssertionError("HTTP post attempted")


class _ResolverSettings:
    ollama_url = "http://127.0.0.1:9"
    request_timeout = 2
    context_tokens = CONTEXT_TOKENS
    max_output_tokens = 1024
    chat_model = "stub-chat"
    embedding_model = "stub-embed"


def _base_resolver_system_bytes():
    """Measure the base system literal of resolve_chat_query from the source."""
    source = (TEACHCOPILOT_RAG / "local_app" / "runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_chat_query":
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == "system"
                        and isinstance(stmt.value, ast.Constant)):
                    return len(stmt.value.value.encode("utf-8"))
    raise AssertionError("base system literal not found in resolve_chat_query")


def _fake_complete_memory_chat(runtime, captured, decision):
    """Replicate chat()'s complete-memory byte check without any HTTP call."""

    def fake_chat(messages, response_format=None, max_output_tokens=None, truncate_history=True):
        bounded = [{"role": str(row.get("role", "user")), "content": str(row.get("content", ""))}
                   for row in messages if isinstance(row, dict)]
        output_tokens = runtime.max_output_tokens if max_output_tokens is None else max(
            128, min(int(max_output_tokens), 3072, runtime.context_tokens - 1024))
        input_size = sum(len(row["content"].encode("utf-8")) for row in bounded)
        ceiling = max(0, runtime.context_tokens - output_tokens) * 2
        captured["messages"] = bounded
        captured["input_size"] = input_size
        captured["ceiling"] = ceiling
        captured["output_tokens"] = output_tokens
        if input_size > ceiling:
            raise ValueError("Complete memory input exceeds the local context budget")
        return {"message": {"content": json.dumps(decision)}, "done": True, "done_reason": "stop"}

    return fake_chat


_DECISION = {"in_scope": True, "uses_prior_context": False,
             "search_query": "отрезки и лучи", "personal_memory_recall": False}

_OWN_SUMMARY = ("Сводка предыдущего разговора ученика: "
                + "краткое содержание диалога с уроком. " * 17)
_CURRENT = "а теперь объясни это ещё раз"
_APPROVED = {"method": "personal_memory",
             "text": "Одобренная память: ученик решал задачу про отрезки.",
             "topic_id": "math.geometry", "version_id": "v1", "stale": False}
_CONVERSATION_CONTEXT = {"method": "conversation_history", "system_text": _OWN_SUMMARY}
_FILLER = ("Объясни пожалуйста вот такую задачу про отрезки и лучи на числовой прямой ")


def test_resolver_reservation_covers_late_system_appends():
    # Replicate the admission formula to size priors that fill the reservation
    # the resolver grants (the audit's view: the mandatory size was computed
    # before the late system-prompt appends). The assembled prompt must then
    # still pass the complete-memory byte check instead of hard-failing.
    base_system_bytes = _base_resolver_system_bytes()
    mandatory = (base_system_bytes + len(_OWN_SUMMARY.encode("utf-8"))
                 + len(_CURRENT.encode("utf-8")) + len(_APPROVED["text"].encode("utf-8")))
    remaining = max(0, (CONTEXT_TOKENS - 256) * 2 - mandatory - 2400)
    assert remaining > 0, "test sizing expects a nonzero admission budget"

    per_prior = remaining // 2
    prior_messages = [{"role": "user", "content": (_FILLER * 40)[:per_prior]}
                      for _ in range(2)]

    runtime = OllamaRuntime(_ResolverSettings(), session=_DeadSession())
    captured: dict = {}
    runtime.chat = _fake_complete_memory_chat(runtime, captured, _DECISION)

    runtime.resolve_chat_query(_CURRENT, prior_messages, subject_id="math",
                               approved_memory=_APPROVED, topic_id="math.geometry",
                               conversation_context=_CONVERSATION_CONTEXT)
    assert captured["input_size"] <= captured["ceiling"], (
        "the resolver admitted priors up to its reservation but the assembled "
        "prompt exceeded the complete-memory ceiling: %d > %d bytes"
        % (captured["input_size"], captured["ceiling"]))


def test_resolver_still_admits_priors_that_fit_the_measured_budget():
    # Guard against over-conservation: reasonably sized priors must still be
    # admitted whole and the assembled prompt must fit the ceiling. Two priors
    # of 200 characters (400 UTF-8 bytes each) sit well inside the corrected
    # reservation, so both must survive the admission loop.
    prior_messages = [{"role": "user", "content": (_FILLER * 40)[:200]}
                      for _ in range(2)]

    runtime = OllamaRuntime(_ResolverSettings(), session=_DeadSession())
    captured: dict = {}
    runtime.chat = _fake_complete_memory_chat(runtime, captured, _DECISION)

    runtime.resolve_chat_query(_CURRENT, prior_messages, subject_id="math",
                               approved_memory=_APPROVED, topic_id="math.geometry",
                               conversation_context=_CONVERSATION_CONTEXT)
    assert captured["input_size"] <= captured["ceiling"]
    user_payload = json.loads(captured["messages"][1]["content"])
    admitted = user_payload["prior_learner_questions_untrusted"]
    assert len(admitted) == 2, (
        "the reservation must not become so conservative that fitting priors "
        "are all dropped (admitted %d of 2)" % len(admitted))
