"""Public unified memory backend facade (client A, contract v1).

``UnifiedMemoryService(store, runtime, *, model_call=None)`` accepts the
existing Store/runtime objects, does no model or database work at import time,
and validates actor access at every service boundary. String identifiers and
JSON-serializable dictionaries only. Client B depends on this facade, not on
its table layout.

Errors: :class:`UnifiedMemoryConflict` (409), :class:`UnifiedMemoryUnavailable`
(503), :class:`UnifiedMemoryValidationError` (422), ``PermissionError`` (403)
and ``LookupError`` (404). Public errors never contain credentials or raw
private transcripts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .memory_store import MemoryIdempotencyConflict
from .unified_memory_profile import replay_personal_changes
from .unified_memory_processing import (
    EXTRACT_PROMPT,
    EXTRACT_SCHEMA,
    MAX_ITEM_CHARS,
    SUMMARY_CHUNK_PROMPT,
    SUMMARY_CHUNK_SCHEMA,
    SUMMARY_HEADING_NEXT,
    SUMMARY_HEADING_REPEAT,
    SUMMARY_REDUCE_PROMPT,
    normalize_chunk_summary,
    prompt_recipe,
    reduce_chunk_summaries,
    session_model_messages,
    source_fingerprint,
    summarize_text,
    validate_extraction,
)
from .unified_memory_recall import recall_own_conversation
from .unified_memory_store import redact_lesson_text

# A-report F2/F5: store-level integrity failures are component degradations
# inside _persist_run, never contract exceptions; a database outage maps to
# UnifiedMemoryUnavailable at the service boundary.
try:
    from psycopg2 import Error as _Psycopg2Error
    _STORE_INTEGRITY_ERRORS = (_Psycopg2Error,)
    _DATABASE_ERRORS = (_Psycopg2Error,)
except ImportError:  # pragma: no cover - deployments without the DB driver
    _STORE_INTEGRITY_ERRORS = ()
    _DATABASE_ERRORS = ()

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts" / "unified_memory"
_PROMPT_FILES = {
    EXTRACT_PROMPT: "personal_extract_v1.txt",
    SUMMARY_CHUNK_PROMPT: "session_summary_chunk_v1.txt",
    SUMMARY_REDUCE_PROMPT: "session_summary_reduce_v1.txt",
}

TEACHER_ROLES = ("teacher", "admin")
LESSON_STAGES = ("not_started", "warmup", "explanation", "practice", "check", "completed")


class UnifiedMemoryConflict(RuntimeError):
    """A revision or idempotency conflict; maps to HTTP 409."""


class UnifiedMemoryUnavailable(RuntimeError):
    """A required backend capability is temporarily unavailable; maps to 503."""


class UnifiedMemoryValidationError(ValueError):
    """The request violates the backend contract; maps to HTTP 422."""


def _recipe_hash():
    hashes = {}
    for name, filename in _PROMPT_FILES.items():
        hashes[name] = hashlib.sha256((_PROMPT_DIR / filename).read_bytes()).hexdigest()
    return prompt_recipe(hashes)


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _json_dict(value):
    return json.loads(value) if isinstance(value, str) else value


class UnifiedMemoryService:
    """The only supported entry point for client B."""

    def __init__(self, store, runtime, *, model_call=None):
        self.store = store
        self.runtime = runtime
        self.model_call = model_call
        self.recipe = _recipe_hash()

    # ------------------------------------------------------------- plumbing
    def _account(self, actor_id):
        reader = getattr(self.store, "get_account", None)
        if not callable(reader) or actor_id is None or str(actor_id) == "":
            raise PermissionError("actor account is unavailable")
        account = reader(account_id=str(actor_id))
        if not account or not account.get("active", True):
            raise PermissionError("actor account is unavailable")
        return account

    def _require_learner_access(self, actor_id, learner_id):
        """Every actor is checked against the store's own authorization data."""
        account = self._account(actor_id)
        if not callable(getattr(self.store, "authorize_learner_access", None)):
            raise PermissionError("authorization is unavailable")
        if not self.store.authorize_learner_access(account["id"], learner_id):
            raise PermissionError("actor has no access to this learner")
        return account

    def _require_teacher(self, actor_id, learner_id):
        account = self._require_learner_access(actor_id, learner_id)
        if account.get("role") not in TEACHER_ROLES:
            raise PermissionError("this operation is teacher-only")
        return account

    def _model_json(self, messages, purpose, schema, *, max_output_tokens=2048):
        # One attempt per call: an unknown-outcome model failure is never
        # retried here; callers degrade per component instead.
        try:
            if self.model_call is not None:
                text = self.model_call(messages, purpose, schema)
            else:
                chat_json = getattr(self.runtime, "chat_json", None)
                if not callable(chat_json):
                    raise UnifiedMemoryUnavailable(
                        "no local model path is available for unified memory")
                # Direct runtime use respects the local admission path.
                text_response = chat_json(messages, response_schema=schema,
                                          max_output_tokens=max_output_tokens)
                return text_response
        except UnifiedMemoryUnavailable:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise UnifiedMemoryUnavailable(f"model call failed for {purpose}") from exc
        if not isinstance(text, str):
            raise UnifiedMemoryUnavailable("model_call must return response text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise UnifiedMemoryUnavailable(f"model returned invalid JSON for {purpose}") from exc

    def _model_identity(self):
        runtime = self.runtime
        if runtime is not None:
            reader = getattr(runtime, "model_identity", None)
            settings = getattr(runtime, "settings", None)
            if callable(reader) and settings is not None:
                try:
                    identity = reader(getattr(settings, "chat_model", None))
                except Exception as exc:
                    raise UnifiedMemoryUnavailable(
                        "the local model identity is unavailable for processing") from exc
                if isinstance(identity, str) and identity:
                    return identity
        reader = getattr(self.model_call, "model_identity", None)
        if callable(reader):
            identity = reader()
            if isinstance(identity, str) and identity:
                return identity
        raise UnifiedMemoryUnavailable(
            "unified processing requires an immutable model identity: provide a runtime "
            "with model_identity() or a model_call with a model_identity attribute")

    @staticmethod
    def _validated_request_id(request_id):
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 180:
            raise UnifiedMemoryValidationError("request_id must contain 1 to 180 characters")
        return request_id

    def _opaque_revision(self, learner_id):
        counters = self.store.get_learner_card_revision(learner_id)
        session_state = self.store.latest_lesson_state(learner_id)
        payload = {"learner_revision": counters["revision"],
                   "source_revision": counters["memory_source_revision"],
                   "changes_revision": counters["unified_changes_revision"],
                   "lesson_state_revision": session_state["revision"] if session_state else None,
                   "lesson_state_updated_at": _iso(session_state["updated_at"]) if session_state else None}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "um-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]

    # ---------------------------------------------------------- fast memory
    def fast_memory(self, learner_id, actor_id, *, session_id=None):
        """Cheap current profile plus confirmed academic and session state.

        No LLM or embedding calls, no full-history scan. The effective personal
        projection starts from the manual profile values and replays the
        applied durable changes (and reverts) in source order, so a removal
        also suppresses the matching original card value and normalized keys
        are compared consistently. Interests and avoided story themes stay
        separate categories. Unavailable fields are empty/unknown, never
        fabricated.
        """
        self._require_learner_access(actor_id, learner_id)
        card = self.store.get_card(learner_id)
        if card is None:
            raise LookupError(learner_id)
        events = self.store.durable_personal_change_events(learner_id)
        rag = card.get("rag_context") or {}
        profile = card.get("profile") or {}
        manual = {
            "interests": [{"value": str(value),
                           "source_refs": [{"kind": "learner_card", "field": "interests"}]}
                          for value in (card.get("interests") or []) if str(value).strip()],
            "avoided_interests": [{"value": str(value),
                                   "source_refs": [{"kind": "learner_card",
                                                    "field": "rag_context.avoid"}]}
                                  for value in ((rag.get("avoid") or []) if isinstance(rag, dict) else [])
                                  if str(value).strip()],
            "preferences": [{"name": name, "value": value,
                             "source_refs": [{"kind": "learner_card", "field": f"profile.{name}"}]}
                            for name, value in sorted(profile.items())
                            if isinstance(value, (str, int, bool))],
        }
        effective = replay_personal_changes(manual, events)
        goals = []
        current_goal = rag.get("current_goal") if isinstance(rag, dict) else None
        if isinstance(current_goal, dict) and current_goal.get("goal"):
            goals.append({"topic_id": current_goal.get("topic_id"), "goal": str(current_goal["goal"]),
                          "updated_at": current_goal.get("updated_at"), "origin": "profile",
                          "source_refs": [{"kind": "learner_card", "field": "rag_context.current_goal"}]})
        confirmed = [{"topic_id": item.get("topic_id"), "status": item.get("status"),
                      "notes": item.get("notes"),
                      "source_refs": [{"kind": "learner_card", "field": "knowledge",
                                       "topic_id": item.get("topic_id")}]}
                     for item in (card.get("knowledge") or []) if isinstance(item, dict)]
        errors = [{"error_tag": item.get("error_tag"), "topic_id": item.get("topic_id"),
                   "subject": item.get("subject"), "count": item.get("count"),
                   "last_seen": item.get("last_seen"),
                   "source_refs": [{"kind": "learner_card", "field": "error_patterns",
                                    "error_tag": item.get("error_tag")}]}
                  for item in (card.get("error_patterns") or []) if isinstance(item, dict)]
        help_items = [{"topic_id": item.get("topic_id"), "strategies": item.get("strategies"),
                       "source_refs": [{"kind": "learner_card", "field": "scaffolding_by_topic",
                                        "topic_id": item.get("topic_id")}]}
                      for item in (card.get("scaffolding_by_topic") or []) if isinstance(item, dict)]
        state = None
        if session_id is not None:
            state = self.store.get_unified_lesson_state(session_id, learner_id)
            if state is None:
                session = self.store.get_learning_session(session_id, learner_id)
                if session is None:
                    raise LookupError(session_id)
        else:
            state = self.store.latest_open_lesson_state(learner_id)
        unfinished = None
        session_view = None
        if state is not None:
            unfinished = {"lesson_id": str(state["lesson_id"]), "stage": state["stage"],
                          "pending_task_id": state.get("pending_task_id"),
                          "completed_task_ids": list(state.get("completed_task_ids") or []),
                          "source_refs": [{"kind": "unified_lesson_state",
                                           "session_id": str(state["learning_session_id"])}]}
            session_view = {"session_id": str(state["learning_session_id"]),
                            "lesson_id": str(state["lesson_id"]), "stage": state["stage"],
                            "revision": state["revision"],
                            "topic_id": state.get("topic_id"), "minutes": state.get("minutes")}
        return {"method": "unified_fast_memory", "learner_id": learner_id,
                "revision": self._opaque_revision(learner_id),
                "personal": {"interests": effective["interests"],
                             "avoided_interests": effective["avoided_interests"],
                             "preferences": effective["preferences"]},
                "learning": {"goals": goals, "confirmed_observations": confirmed,
                             "errors": errors, "help": help_items, "unfinished": unfinished},
                "session": session_view,
                "source_refs": {"profile_authority": "learner_card",
                                "conversation_authority": "unified_personal_changes"}}

    def previous_lesson_context(self, learner_id, actor_id):
        """Read the latest finished lesson's recommendation for teacher generation."""
        self._require_teacher(actor_id, learner_id)
        summary = self.store.latest_completed_lesson_summary(learner_id)
        if summary is None:
            return None
        # Existing summary versions store the four sections as rendered text.
        def first_item(heading):
            _, _, section = summary["text"].rpartition(f"\n\n{heading}\n")
            first_line, _, _ = section.partition("\n")
            item = first_line.removeprefix("- ").strip()
            if first_line.startswith("- ") and item != "(нет данных)":
                return item[:MAX_ITEM_CHARS]
            return ""

        next_step = first_item(SUMMARY_HEADING_NEXT)
        if not next_step:
            return None
        context = {"completed_on": summary["completed_on"].isoformat(),
                   "topic_id": summary["topic_id"], "next_step": next_step,
                   "source_ref": {"kind": "unified_summary_version",
                                  "summary_id": str(summary["summary_id"]),
                                  "session_id": str(summary["session_id"]),
                                  "lesson_id": str(summary["lesson_id"])}}
        if review_focus := first_item(SUMMARY_HEADING_REPEAT):
            context["review_focus"] = review_focus
        return context

    # --------------------------------------------------------------- recall
    def recall(self, learner_id, actor_id, query, *, session_id=None, budget_bytes=4000,
               time_filter=None, exclude_source_id=None):
        account = self._require_learner_access(actor_id, learner_id)
        try:
            return recall_own_conversation(self.store, self.runtime, learner_id, actor_id, query,
                                           session_id=session_id, budget_bytes=budget_bytes,
                                           time_filter=time_filter,
                                           exclude_source_id=exclude_source_id,
                                           student_audience=account.get("role")
                                           not in TEACHER_ROLES)
        except PermissionError:
            raise
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        except (RuntimeError, OSError) + _DATABASE_ERRORS as exc:
            # A-report F5: raw psycopg2 errors map to the contract, not 500s.
            raise UnifiedMemoryUnavailable("recall backend is temporarily unavailable") from exc

    # -------------------------------------------------------------- lessons
    def save_lesson(self, learner_id, actor_id, *, topic_id, minutes, lesson_text, answers_text,
                    validation, request_id, memory_revision):
        self._require_teacher(actor_id, learner_id)
        self._validated_request_id(request_id)
        try:
            row = self.store.save_unified_lesson(
                learner_id, topic_id, minutes, lesson_text, answers_text or "", validation,
                [{"kind": "lesson_generation", "topic_id": topic_id}],
                memory_revision, request_id, actor_id)
        except MemoryIdempotencyConflict as exc:
            raise UnifiedMemoryConflict(str(exc)) from exc
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        return self._lesson_public(row, audience="teacher")

    def get_lesson(self, learner_id, actor_id, lesson_id, *, audience):
        account = self._require_learner_access(actor_id, learner_id)
        if audience not in ("teacher", "student"):
            raise UnifiedMemoryValidationError("audience must be 'teacher' or 'student'")
        row = self.store.get_unified_lesson(lesson_id, learner_id)
        effective = audience
        if account.get("role") not in TEACHER_ROLES:
            # Actual actor permissions override the requested audience.
            effective = "student"
        return self._lesson_public(row, audience=effective)

    def list_lessons(self, learner_id, actor_id):
        """Optional listing requested by client B: lesson metadata without text.

        Teachers and admins see every version; students see approved lessons
        only. No lesson or answer text is included, so drafts stay safe.
        """
        account = self._require_learner_access(actor_id, learner_id)
        approved_only = account.get("role") not in TEACHER_ROLES
        return {"learner_id": learner_id,
                "lessons": [{"lesson_id": str(row["id"]), "topic_id": row["topic_id"],
                             "minutes": row["minutes"], "revision": row["revision"],
                             "status": row["status"],
                             "memory_revision": row.get("memory_revision"),
                             "created_at": _iso(row.get("created_at")),
                             "confirmed_at": _iso(row.get("confirmed_at"))}
                            for row in self.store.list_unified_lessons(
                                learner_id, approved_only=approved_only)]}

    def confirm_lesson(self, learner_id, actor_id, lesson_id, *, expected_revision, request_id):
        self._require_teacher(actor_id, learner_id)
        self._validated_request_id(request_id)
        if type(expected_revision) is not int or expected_revision < 1:
            raise UnifiedMemoryValidationError("expected_revision must be a positive integer")
        try:
            row = self.store.confirm_unified_lesson(lesson_id, learner_id, expected_revision,
                                                    actor_id, request_id)
        except _DATABASE_ERRORS as exc:
            # A-report F5: a database outage is 503, not a revision conflict.
            raise UnifiedMemoryUnavailable("unified memory backend is temporarily unavailable") from exc
        except RuntimeError as exc:
            raise UnifiedMemoryConflict(str(exc)) from exc
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        result = self._lesson_public(row, audience="teacher")
        result["confirmed"] = bool(row.get("confirmed"))
        return result

    def attach_lesson(self, learner_id, actor_id, session_id, lesson_id, *, request_id):
        self._require_teacher(actor_id, learner_id)
        self._validated_request_id(request_id)
        try:
            row = self.store.attach_unified_lesson(session_id, learner_id, lesson_id,
                                                   actor_id, request_id)
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        return self._state_public(row)

    def lesson_state(self, learner_id, actor_id, session_id):
        self._require_learner_access(actor_id, learner_id)
        state = self.store.get_unified_lesson_state(session_id, learner_id)
        if state is None:
            session = self.store.get_learning_session(session_id, learner_id)
            if session is None:
                raise LookupError(session_id)
            return {"session_id": str(session_id), "lesson_id": None, "revision": None,
                    "stage": None, "pending_task_id": None, "last_question_source_id": None,
                    "completed_task_ids": []}
        return self._state_public(state)

    def update_lesson_state(self, learner_id, actor_id, session_id, *, expected_revision, state,
                            request_id):
        account = self._require_learner_access(actor_id, learner_id)
        self._validated_request_id(request_id)
        session = self.store.get_learning_session(session_id, learner_id)
        if session is None:
            raise LookupError(session_id)
        if account.get("role") not in TEACHER_ROLES and str(session.get("actor_id")) != str(actor_id):
            raise PermissionError("learning session belongs to another actor")
        if not isinstance(state, dict):
            raise UnifiedMemoryValidationError("state must be an object")
        stage = state.get("stage")
        if stage is not None and stage not in LESSON_STAGES:
            raise UnifiedMemoryValidationError("unknown lesson stage")
        try:
            row = self.store.update_unified_lesson_state(
                session_id, learner_id, expected_revision,
                stage=stage,
                pending_task_id=state.get("pending_task_id", ...),
                last_question_source_id=state.get("last_question_source_id", ...),
                completed_task_ids=state.get("completed_task_ids"),
                request_id=request_id, actor_id=actor_id)
        except _DATABASE_ERRORS as exc:
            raise UnifiedMemoryUnavailable("unified memory backend is temporarily unavailable") from exc
        except RuntimeError as exc:
            raise UnifiedMemoryConflict(str(exc)) from exc
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        result = self._state_public(row)
        result["applied"] = bool(row.get("applied"))
        return result

    # ------------------------------------------------------------- assignments
    def assign_lesson(self, learner_id, actor_id, lesson_id, *, expected_lesson_revision,
                      request_id):
        """Teacher-only pre-login assignment (P0.2).

        The approved lesson revision is validated exactly; the assignment can
        exist before any student session exists. Idempotent for the same
        payload; a revision mismatch is the existing conflict type, never a
        silent substitute.
        """
        self._require_teacher(actor_id, learner_id)
        self._validated_request_id(request_id)
        if type(expected_lesson_revision) is not int or expected_lesson_revision < 1:
            raise UnifiedMemoryValidationError(
                "expected_lesson_revision must be a positive integer")
        try:
            lesson = self.store.get_unified_lesson(lesson_id, learner_id)
        except KeyError:
            raise LookupError(lesson_id) from None
        if lesson["status"] != "approved":
            raise UnifiedMemoryValidationError("only an approved lesson can be assigned")
        if lesson["revision"] != expected_lesson_revision:
            raise UnifiedMemoryConflict(
                f"lesson revision is {lesson['revision']}, expected {expected_lesson_revision}")
        try:
            row = self.store.create_lesson_assignment(learner_id, lesson_id,
                                                      expected_lesson_revision, actor_id,
                                                      request_id)
        except MemoryIdempotencyConflict as exc:
            raise UnifiedMemoryConflict(str(exc)) from exc
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        result = self._assignment_public(row, teacher_view=True)
        result["topic_id"] = lesson["topic_id"]
        result["minutes"] = lesson["minutes"]
        result["created"] = bool(row.get("created", True))
        return result

    def list_lesson_assignments(self, learner_id, actor_id):
        """Authorized assignment listing; teachers see the learner's
        assignments, linked students see only what they may use. No lesson or
        answer text and no private teacher material is included."""
        account = self._require_learner_access(actor_id, learner_id)
        teacher_view = account.get("role") in TEACHER_ROLES
        items = [self._assignment_public(row, teacher_view=teacher_view)
                 for row in self.store.list_lesson_assignments(learner_id)]
        return {"learner_id": learner_id, "items": items}

    def start_lesson_assignment(self, learner_id, actor_id, assignment_id, *, request_id):
        """Called by the linked student (P0.2).

        One transaction validates the assignment, creates a NEW standard
        session owned by that student, attaches the approved version and marks
        the assignment started. Repeats return the same persisted session; a
        different child cannot claim it; completed assignments never create a
        fresh session under the same assignment id.
        """
        account = self._require_learner_access(actor_id, learner_id)
        self._validated_request_id(request_id)
        if account.get("role") != "student":
            raise PermissionError("start_lesson_assignment is called by the linked student")
        try:
            row = self.store.start_lesson_assignment(learner_id, assignment_id, actor_id,
                                                     request_id)
        except MemoryIdempotencyConflict as exc:
            raise UnifiedMemoryConflict(str(exc)) from exc
        except RuntimeError as exc:
            raise UnifiedMemoryConflict(str(exc)) from exc
        except ValueError as exc:
            raise UnifiedMemoryValidationError(str(exc)) from exc
        result = self._assignment_public(row, teacher_view=False)
        result["created"] = bool(row.get("created", False))
        state = row.get("state")
        if state is None and row.get("learning_session_id"):
            state = self.store.get_unified_lesson_state(row["learning_session_id"], learner_id)
        result["lesson_state"] = self._state_public(state) if state is not None else None
        return result

    def lesson_tasks(self, learner_id, actor_id, lesson_id, *, audience):
        """Task catalog of one saved lesson version (P1.1).

        ``items`` carry ``task_id``, ``stage``, ``position``, ``prompt_text``
        and lesson-document source offsets/refs. Answers are never part of the
        catalog. A lesson saved before the catalog existed reports an explicit
        ``unavailable`` limitation; a read never writes a catalog.
        """
        account = self._require_learner_access(actor_id, learner_id)
        if audience not in ("teacher", "student"):
            raise UnifiedMemoryValidationError("audience must be 'teacher' or 'student'")
        row = self.store.get_lesson_task_catalog(lesson_id, learner_id)
        effective = audience
        if account.get("role") not in TEACHER_ROLES:
            effective = "student"
            if row["status"] != "approved":
                raise PermissionError("tasks are available for approved lessons only")
        catalog = row.get("task_catalog")
        result = {"lesson_id": str(row["id"]), "lesson_revision": row["revision"],
                  "lesson_status": row["status"], "audience": effective}
        if catalog is None:
            result["catalog"] = {"status": "unavailable",
                                 "reason": "lesson version was saved before the task catalog "
                                           "existed; an explicit worker/admin operation may rebuild it"}
            result["items"] = []
            return result
        result["catalog"] = {"status": "available", "recipe": catalog.get("recipe")}
        result["items"] = [{"task_id": item["task_id"], "stage": item["stage"],
                            "position": item["position"], "prompt_text": item["prompt_text"],
                            "section": item.get("section"),
                            "source_refs": item.get("source_refs") or []}
                           for item in (catalog.get("items") or [])
                           if isinstance(item, dict) and item.get("task_id")]
        return result

    def rebuild_lesson_task_catalog(self, learner_id, actor_id, lesson_id, *, request_id):
        """Explicit teacher/admin backfill of a missing task catalog (P1.1).

        Never triggered by a read path; deterministic content is a no-op.
        """
        self._require_teacher(actor_id, learner_id)
        self._validated_request_id(request_id)
        try:
            row = self.store.rebuild_lesson_task_catalog(learner_id, lesson_id, actor_id,
                                                         request_id)
        except KeyError:
            raise LookupError(lesson_id) from None
        catalog = row.get("task_catalog")
        return {"lesson_id": str(row["id"]), "lesson_revision": row["revision"],
                "rebuilt": bool(row.get("rebuilt")),
                "catalog_status": "available" if catalog is not None else "unavailable",
                "task_count": len((catalog or {}).get("items") or [])}

    def _assignment_public(self, row, *, teacher_view):
        public = {"assignment_id": str(row["id"]), "learner_id": row["learner_id"],
                  "lesson_id": str(row["lesson_id"]), "lesson_revision": row["lesson_revision"],
                  "status": row["status"],
                  "learning_session_id": str(row["learning_session_id"])
                  if row.get("learning_session_id") else None,
                  "topic_id": row.get("topic_id"), "minutes": row.get("minutes"),
                  "assigned_at": _iso(row.get("assigned_at")),
                  "started_at": _iso(row.get("started_at")),
                  "completed_at": _iso(row.get("completed_at"))}
        if teacher_view:
            # Actor identities are administrative provenance; the student view
            # keeps only the schedule facts it needs.
            public.update({"assigned_by": row.get("assigned_by"),
                           "started_by": row.get("started_by")})
        return public

    def _lesson_public(self, row, *, audience):
        lesson_id = str(row["id"])
        lesson_text = row["lesson_text"]
        answers_text = row["answers_text"]
        if audience == "student":
            sections = _json_dict(row.get("answer_sections")) or []
            lesson_text, _ = redact_lesson_text(lesson_text, sections)
            answers_text = ""
        return {"lesson_id": lesson_id, "learner_id": row["learner_id"],
                "topic_id": row["topic_id"], "minutes": row["minutes"],
                "revision": row["revision"], "status": row["status"],
                "lesson_text": lesson_text,
                "answers_text": answers_text if audience == "teacher" else "",
                "validation": _json_dict(row.get("validation")) if audience == "teacher" else
                              {"status": (_json_dict(row.get("validation")) or {}).get("status")},
                "source_refs": _json_dict(row.get("source_refs")) or [],
                "memory_revision": row.get("memory_revision"),
                "audience": audience,
                "answer_redacted": audience == "student",
                "created_at": _iso(row.get("created_at")),
                "confirmed_at": _iso(row.get("confirmed_at"))}

    def _state_public(self, row):
        return {"session_id": str(row["learning_session_id"]), "lesson_id": str(row["lesson_id"]),
                "revision": row["revision"], "stage": row["stage"],
                "pending_task_id": row.get("pending_task_id"),
                "last_question_source_id": row.get("last_question_source_id"),
                "completed_task_ids": list(row.get("completed_task_ids") or []),
                "topic_id": row.get("topic_id"), "minutes": row.get("minutes")}

    # ------------------------------------------------------ session processing
    def process_session(self, learner_id, actor_id, session_id, *, request_id):
        """Worker-facing processing; never call inside an HTTP transaction."""
        account = self._require_learner_access(actor_id, learner_id)
        self._validated_request_id(request_id)
        snapshot = self.store.unified_session_snapshot(learner_id, session_id)
        session = snapshot["session"]
        provisional = session["status"] != "completed"
        if provisional and account.get("role") not in TEACHER_ROLES:
            # On an active session only an explicit teacher request processes a
            # provisional snapshot; it never finishes the session.
            raise PermissionError("processing an active session is teacher-only")
        if not snapshot["sources"]:
            raise UnifiedMemoryValidationError("session has no stored sources to process")

        sources = snapshot["sources"]
        fingerprint = source_fingerprint(sources)
        identity = self._model_identity()
        existing = self.store.find_unified_run(learner_id, session_id, fingerprint,
                                               self.recipe, identity)
        effective_request_id = request_id
        prior_request = self.store.get_unified_run_by_request(learner_id, request_id)
        if prior_request is not None:
            same_input = (prior_request["source_fingerprint"] == fingerprint
                          and prior_request["prompt_recipe"] == self.recipe
                          and prior_request["model_identity"] == identity
                          and str(prior_request["session_id"]) == str(session_id))
            if not same_input:
                raise UnifiedMemoryConflict(
                    "request_id was reused for a different session snapshot")
            if prior_request["status"] in ("completed", "partial"):
                return self._run_result(prior_request, learner_id, session_id)
            # A failed run for the exact same input is retried under a fresh
            # request identifier; the failed artifact stays retrievable.
            suffix = 2
            while self.store.get_unified_run_by_request(
                    learner_id, f"{request_id}:r{suffix}") is not None:
                suffix += 1
            effective_request_id = f"{request_id}:r{suffix}"
        elif existing is not None and existing["status"] in ("completed", "partial"):
            return self._run_result(existing, learner_id, session_id)

        # P1.2: the pre-insert run lookup alone is not concurrency control. The
        # durable identity claim serializes identical-snapshot processing with
        # a bounded lease and no long row transaction across model calls.
        claim = self.store.acquire_processing_claim(learner_id, session_id, fingerprint,
                                                    self.recipe, identity, claimed_by=str(actor_id))
        if not claim["claimed"]:
            raise UnifiedMemoryConflict(
                "an identical processing run is already in progress for this snapshot; "
                "retry after it completes")
        claim_token = claim["claim_token"]
        try:
            return self._process_snapshot(learner_id, session_id, actor_id, sources, fingerprint,
                                          identity, effective_request_id, provisional,
                                          claim_token)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            # Nothing was persisted on this path, so releasing the claim cannot
            # duplicate applied changes; unknown-outcome model calls stay
            # distinguishable through the injected model_call journal.
            try:
                self.store.release_processing_claim(learner_id, session_id, fingerprint,
                                                    self.recipe, identity, claim_token)
            except Exception:  # pragma: no cover - best-effort release
                pass
            raise

    def _process_snapshot(self, learner_id, session_id, actor_id, sources, fingerprint, identity,
                          effective_request_id, provisional, claim_token):
        messages = session_model_messages(sources)
        chunks = session_model_chunks(learner_id, session_id, messages)
        sources_by_id = {source["source_id"]: source for source in sources}
        superseded_by = {source["source_id"]: source["superseded_by"]
                         for source in sources if source.get("superseded_by")}

        chunk_summaries, coverage, warnings = [], {"chunk_ids": [], "included_chunk_ids": [],
                                                   "failed_chunk_ids": [],
                                                   "extraction_failed_chunk_ids": []}, []
        changes, candidates = [], []
        extraction_attempts = 0
        for chunk in chunks:
            coverage["chunk_ids"].append(chunk["chunk_id"])
            corrections = [{"superseded_source_id": source_id,
                            "corrected_by_source_id": superseded_by[source_id]}
                           for source_id in chunk["source_ids"] if source_id in superseded_by]
            payload = None
            try:
                payload = self._model_json(
                    [{"role": "system", "content": self._prompt(SUMMARY_CHUNK_PROMPT)},
                     {"role": "user", "content": json.dumps(
                         {"chunk_id": chunk["chunk_id"], "transcript_chunk_untrusted": chunk["text"],
                          "corrections_untrusted": corrections},
                         ensure_ascii=False)}],
                    SUMMARY_CHUNK_PROMPT, SUMMARY_CHUNK_SCHEMA, max_output_tokens=1024)
                chunk_summaries.append(normalize_chunk_summary(payload))
                coverage["included_chunk_ids"].append(chunk["chunk_id"])
            except (UnifiedMemoryUnavailable, ValueError):
                # A-report F3: a schema-ignoring chunk response degrades that
                # chunk; the run itself continues with honest coverage.
                coverage["failed_chunk_ids"].append(chunk["chunk_id"])
                warnings.append({"code": "summary_chunk_failed", "chunk_id": chunk["chunk_id"]})
            try:
                extraction_attempts += 1
                extract_payload = self._model_json(
                    [{"role": "system", "content": self._prompt(EXTRACT_PROMPT)},
                     {"role": "user", "content": json.dumps(
                         {"chunk_id": chunk["chunk_id"],
                          "source_ids": chunk["source_ids"],
                          "transcript_chunk_untrusted": chunk["text"],
                          "corrections_untrusted": corrections},
                         ensure_ascii=False)}],
                    EXTRACT_PROMPT, EXTRACT_SCHEMA, max_output_tokens=1024)
                new_changes, new_candidates, extract_warnings = validate_extraction(
                    extract_payload, chunk, sources_by_id)
                warnings.extend(extract_warnings)
                changes.extend(new_changes)
                candidates.extend(new_candidates)
            except (UnifiedMemoryUnavailable, ValueError):
                coverage["extraction_failed_chunk_ids"].append(chunk["chunk_id"])
                warnings.append({"code": "extraction_chunk_failed", "chunk_id": chunk["chunk_id"]})

        summary_payload = None
        if chunk_summaries:
            reduced = None
            try:
                reduced = normalize_chunk_summary(self._model_json(
                    [{"role": "system", "content": self._prompt(SUMMARY_REDUCE_PROMPT)},
                     {"role": "user", "content": json.dumps(
                         {"chunk_summaries_untrusted": chunk_summaries}, ensure_ascii=False)}],
                    SUMMARY_REDUCE_PROMPT, SUMMARY_CHUNK_SCHEMA, max_output_tokens=1024))
            except (UnifiedMemoryUnavailable, ValueError):
                # A-report F3: a non-object reduce response degrades to the
                # per-chunk summaries instead of aborting the run.
                warnings.append({"code": "summary_reduce_failed"})
            # Deterministic de-duplication applies to the model result too.
            reduced = reduce_chunk_summaries([reduced] if reduced else chunk_summaries)
            summary_payload = {"text": summarize_text(reduced),
                               "parts": {key: reduced[key] for key in
                                         ("studied", "achieved", "repeat", "next_step")}}
        elif coverage["failed_chunk_ids"] and extraction_attempts and not changes and not candidates:
            warnings.append({"code": "summary_unavailable_all_chunks_failed"})

        status = "completed"
        if coverage["failed_chunk_ids"] or coverage["extraction_failed_chunk_ids"]:
            status = "partial"
        if not chunk_summaries:
            status = "failed"
        provisional_flag = provisional or status != "completed"

        run = self._persist_run(learner_id, session_id, effective_request_id, status,
                                provisional_flag, fingerprint, identity, sources,
                                summary_payload, coverage, warnings, changes, candidates,
                                claim_token)
        return self._run_result(run, learner_id, session_id)

    def _prompt(self, name):
        filename = _PROMPT_FILES[name]
        return (_PROMPT_DIR / filename).read_text(encoding="utf-8")

    def _persist_run(self, learner_id, session_id, request_id, status, provisional, fingerprint,
                     identity, sources, summary_payload, coverage, warnings, changes, candidates,
                     claim_token=None):
        """One-transaction persistence of the run, its summary version and all
        side effects, then the claim release (P1.2)."""
        summary_version = None
        if summary_payload is not None:
            summary_fingerprint = hashlib.sha256(
                json.dumps({"fingerprint": fingerprint, "recipe": self.recipe,
                            "identity": identity, "summary": summary_payload},
                           ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            summary_version = {"input_fingerprint": summary_fingerprint,
                               "text": summary_payload["text"],
                               "source_refs": [{"kind": "conversation_source",
                                                "source_id": source["source_id"],
                                                "author_role": source["author_role"]}
                                               for source in sources],
                               "coverage": coverage, "provisional": provisional}
        change_rows = []
        for change in changes:
            change_rows.append({**change,
                                "request_id": f"{request_id}:{change['category']}:"
                                              f"{change['operation']}:{change['value'][:80]}"})
        candidate_rows = []
        for candidate in candidates:
            candidate_rows.append({**candidate,
                                   "request_id": f"{request_id}:candidate:{candidate['category']}:"
                                                 f"{json.dumps(candidate['proposed_value'], ensure_ascii=False, sort_keys=True)[:120]}"})
        release_claim = None
        if claim_token is not None:
            release_claim = (fingerprint, self.recipe, identity, claim_token)
        try:
            run = self.store.persist_unified_run(
                learner_id, session_id, status=status, provisional=provisional,
                source_fingerprint=fingerprint, prompt_recipe=self.recipe, model_identity=identity,
                summary=None, coverage=coverage, warnings=warnings,
                request_id=request_id, summary_version=summary_version,
                changes=change_rows, candidates=candidate_rows, release_claim=release_claim)
        except _DATABASE_ERRORS as exc:
            text = str(getattr(exc, "diag", None) and getattr(exc.diag, "constraint_name", "") or "")
            if "request_id" in text and "unified_processing_runs" in text:
                # A concurrent writer claimed this request id for a different
                # snapshot: an explicit conflict, not a service outage.
                raise UnifiedMemoryConflict(
                    "request_id was reused concurrently for a different session snapshot") from exc
            raise UnifiedMemoryUnavailable(
                "unified memory backend is temporarily unavailable") from exc
        return run

    def _run_result(self, run, learner_id, session_id):
        summary = _json_dict(run.get("summary")) if run.get("summary") else None
        return {"session_id": str(session_id), "status": run["status"],
                "run_id": str(run["id"]), "provisional": bool(run.get("provisional")),
                "source_fingerprint": run["source_fingerprint"],
                "prompt_recipe": run["prompt_recipe"], "model_identity": run["model_identity"],
                "summary": summary,
                "personal_changes": [_change_public(item) for item in
                                     self.store.list_personal_changes(learner_id)
                                     if item.get("run_id") and str(item["run_id"]) == str(run["id"])],
                "academic_candidates": [_candidate_public(item) for item in
                                        self.store.list_academic_candidates(learner_id)
                                        if item.get("run_id") and str(item["run_id"]) == str(run["id"])],
                "coverage": _json_dict(run.get("coverage")) or {},
                "warnings": _json_dict(run.get("warnings")) or []}

    def processing_result(self, learner_id, actor_id, session_id):
        self._require_learner_access(actor_id, learner_id)
        session = self.store.get_learning_session(session_id, learner_id)
        if session is None:
            raise LookupError(session_id)
        run = self.store.latest_unified_run(learner_id, session_id)
        if run is not None and run["status"] in ("completed", "partial"):
            return {"available": True, **self._run_result(run, learner_id, session_id)}
        pending = self.store.pending_session_process_job(learner_id, session_id)
        if pending is not None:
            return {"available": False, "status": "pending", "session_id": str(session_id),
                    "job_id": str(pending["id"])}
        if run is not None:
            return {"available": False, "status": run["status"], "session_id": str(session_id),
                    "run_id": str(run["id"]),
                    "warnings": _json_dict(run.get("warnings")) or []}
        return {"available": False, "status": "unavailable", "session_id": str(session_id)}

    # ------------------------------------------------------------ change review
    def list_changes(self, learner_id, actor_id):
        self._require_teacher(actor_id, learner_id)
        items = []
        for candidate in self.store.list_academic_candidates(learner_id):
            public = _candidate_public(candidate)
            public["kind"] = "academic_candidate"
            public["eligible_actions"] = (["accept", "reject"] if candidate["status"] == "pending"
                                          else [])
            items.append(public)
        for change in self.store.list_personal_changes(learner_id):
            public = _change_public(change)
            public["kind"] = "personal_change"
            if change["status"] == "proposed":
                public["eligible_actions"] = ["accept", "reject"]
            elif change["status"] == "applied" and change["scope"] == "durable":
                public["eligible_actions"] = ["revert"]
            else:
                public["eligible_actions"] = []
            items.append(public)
        items.sort(key=lambda item: item["created_at"] or "", reverse=True)
        return {"revision": self.store.get_unified_changes_revision(learner_id), "items": items}

    def review_changes(self, learner_id, actor_id, *, actions, expected_revision, request_id):
        account = self._require_teacher(actor_id, learner_id)
        self._validated_request_id(request_id)
        if not isinstance(actions, list) or not actions:
            raise UnifiedMemoryValidationError("actions must be a nonempty list")
        if type(expected_revision) is not int or expected_revision < 0:
            raise UnifiedMemoryValidationError("expected_revision must be a nonnegative integer")
        parsed = []
        for action in actions:
            if not isinstance(action, dict) or set(action) != {"id", "action"}:
                raise UnifiedMemoryValidationError("each action must be {id, action}")
            if action["action"] not in ("accept", "reject", "revert"):
                raise UnifiedMemoryValidationError("action must be accept, reject or revert")
            parsed.append(action)
        current = self.store.get_unified_changes_revision(learner_id)
        if current != expected_revision:
            raise UnifiedMemoryConflict(
                f"changes revision is {current}, expected {expected_revision}")
        results = []
        for action in parsed:
            results.append(self._apply_change_action(learner_id, account, action))
        return {"applied": sum(1 for item in results if item["applied"]),
                "items": results,
                "revision": self.store.get_unified_changes_revision(learner_id)}

    def _apply_change_action(self, learner_id, account, action):
        action_id, verb = str(action["id"]), action["action"]
        change = self.store.get_personal_change(learner_id, action_id)
        candidate = None if change is not None else self.store.get_academic_candidate(
            learner_id, action_id)
        if change is None and candidate is None:
            raise LookupError(action_id)
        if change is not None:
            if verb == "revert":
                if change["status"] != "applied" or change["scope"] != "durable":
                    raise UnifiedMemoryValidationError(
                        "revert is only valid for an applied durable personal change")
                outcome = self.store.revert_personal_change(learner_id, action_id, account["id"],
                                                            request_id=None)
                return {"id": action_id, "action": verb, "applied": bool(outcome["reverted"])}
            if change["status"] != "proposed":
                return {"id": action_id, "action": verb, "applied": False}
            decision = "applied" if verb == "accept" else "rejected"
            row = self.store.decide_personal_change(learner_id, action_id, decision, account["id"])
            return {"id": action_id, "action": verb, "applied": bool(row.get("applied"))}
        if verb == "revert":
            raise UnifiedMemoryValidationError("revert is not valid for an academic candidate")
        if candidate["status"] != "pending":
            return {"id": action_id, "action": verb, "applied": False}
        decision = "accepted" if verb == "accept" else "rejected"
        if decision == "accepted":
            # A-report F6: decision + canonical note + durable job commit in
            # one transaction; a queue-full failure rolls the whole acceptance
            # back and surfaces as an explicit unavailability.
            return self._accept_candidate_canonically(learner_id, account, candidate)
        row = self.store.decide_academic_candidate(learner_id, action_id, decision, account["id"])
        return {"id": action_id, "action": verb, "applied": bool(row.get("applied"))}

    def _accept_candidate_canonically(self, learner_id, account, candidate):
        """Record an accepted observation through the existing teacher-note path.

        The canonical educational update rules for conversation evidence are an
        approved teacher memory source plus its durable update job. Numbers are
        never written: the validator already stripped them, and the note is
        text only.
        """
        from .memory_store import MemoryQueueFullError
        value = _json_dict(candidate.get("proposed_value")) or {}
        topic_id = str(value.get("topic_id") or "general")
        parts = []
        if value.get("error_tag"):
            parts.append(f"Типичная ошибка: {value['error_tag']}")
        if value.get("note"):
            parts.append(str(value["note"]))
        if value.get("goal"):
            parts.append(f"Цель: {value['goal']}")
        if not parts:
            raise UnifiedMemoryValidationError("accepted candidate has no textual content")
        text = "Наблюдение от обработки занятия (подтверждено преподавателем): " + "; ".join(parts)
        try:
            row = self.store.accept_academic_candidate_canonically(
                learner_id, candidate["id"], account["id"], topic_id=topic_id, note_text=text)
        except MemoryQueueFullError as exc:
            # A-report F6: the acceptance rolled back with the transaction.
            raise UnifiedMemoryUnavailable(
                "memory queue is full; the candidate stays pending; retry later") from exc
        return {"id": str(candidate["id"]), "action": "accept",
                "applied": bool(row.get("applied"))}


def session_model_chunks(learner_id, session_id, messages):
    from .conversation_retrieval import CHUNK_RECIPE, chunk_conversation_messages
    chunks = []
    for chunk in chunk_conversation_messages(messages, learner_id, str(session_id)):
        chunk["recipe"] = CHUNK_RECIPE
        chunks.append(chunk)
    return chunks


def _change_public(change):
    return {"change_id": str(change["id"]), "category": change["category"],
            "operation": change["operation"], "value": change.get("value") or "",
            "scope": change["scope"], "status": change["status"],
            "explicit_durable": bool(change.get("explicit_durable")),
            "source_refs": _json_dict(change.get("source_refs")) or [],
            "evidence": _json_dict(change.get("evidence")) or [],
            "session_id": str(change["session_id"]) if change.get("session_id") else None,
            "run_id": str(change["run_id"]) if change.get("run_id") else None,
            "created_at": _iso(change.get("created_at")),
            "decided_by": change.get("decided_by")}


def _candidate_public(candidate):
    return {"candidate_id": str(candidate["id"]), "category": candidate["category"],
            "proposed_value": _json_dict(candidate.get("proposed_value")) or {},
            "source_refs": _json_dict(candidate.get("source_refs")) or [],
            "status": candidate["status"],
            "session_id": str(candidate["session_id"]) if candidate.get("session_id") else None,
            "run_id": str(candidate["run_id"]) if candidate.get("run_id") else None,
            "created_at": _iso(candidate.get("created_at")),
            "decided_by": candidate.get("decided_by"),
            "decision_note": candidate.get("decision_note")}
