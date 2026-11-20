"""Private per-request actual-delivery trace for trusted local harnesses.

The chat endpoint can forward JSON-safe events about one answer request to a
callable handed to ``create_app`` as ``memory_trace_sink``. HTTP clients cannot
enable a sink: only the trusted constructor installs one, and the events never
enter either role's chat response or any public endpoint. With no sink the
application behaves exactly as before.

Event phases and their evidence fields follow the frozen round-3 contract:

- ``resolver`` – the single follow-up resolver's actual decision and the
  provisional route it implies.
- ``retrieval`` – one product retrieval: the authorized candidate pool
  (``eligible_source_ids``), what the search returned (``retrieved_source_ids``),
  what selection kept (``selected_source_ids``) and what it dropped
  (``omitted_source_ids``).
- ``answer_prompt`` – the exact final answer-model input after all budget
  fitting: ``final_prompt`` plus ``final_prompt_sha256`` for verification, the
  evidence units actually admitted into that input (``delivered_source_ids``
  with ``hydrated_quotes`` carrying source id, author role, date, correction
  links and the admitted quote), and ``generation_called``.
- ``fallback`` – a reply produced without an answer-model call. It always
  states ``generation_called=false``; a clarification is never recorded as a
  generated answer.
- ``complete`` – the request's final route, output status, authoritative
  ``generation_called``, delivered ids and ``technical_error``.

Every event carries ``request_id``, ``correlation_id`` (the harness's own
bounded ``metadata.quality_row_id``; it grants no authorization and never
selects a learner), ``learner_id``, ``phase`` and ``route``. Events are
JSON-safe: PostgreSQL numeric and time types keep their exact provenance
semantics (Decimal becomes its string value, timestamps their ISO form), and a
value the serializer cannot represent stays visibly unrepresentable instead of
being silently stringified into fake evidence. Sink failures never break the
chat and never fabricate evidence; they are counted and reported on
``complete``.

Each request owns its trace instance, which the endpoint passes explicitly
into the answer service; there is deliberately no process-global last-query
state, so concurrent requests cannot cross-contaminate events.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import logging
from typing import Any, Callable
from uuid import UUID, uuid4


logger = logging.getLogger(__name__)

PHASES = ("resolver", "retrieval", "answer_prompt", "fallback", "complete")

# Frozen provisional/final route names. ``unknown`` is only used before the
# resolver has decided anything (for example a request that failed validation).
ROUTES = ("curriculum_retrieval", "personal_recall", "social", "creative",
          "clarification", "out_of_scope", "deterministic_followup",
          "learning_operation", "guided_lesson", "unknown")


def _json_safe(value: Any) -> Any:
    """Keep database numeric/time provenance exact; refuse opaque objects visibly."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"unserializable": "bytes", "byte_count": len(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    # Never invent a value: an opaque object stays a visible non-evidence marker.
    return {"unserializable": type(value).__name__}


def bounded_correlation_id(value: Any) -> str | None:
    """Accept only a bounded plain string as the harness correlation id."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 200:
        return None
    try:
        text.encode("utf-8")
    except UnicodeError:
        return None
    return text


class MemoryTrace:
    """One request's trace; the endpoint passes it explicitly to the answer service."""

    def __init__(self, sink: Callable[[dict], None], *, learner_id: str,
                 correlation_id: str | None = None, request_id: str | None = None):
        if not callable(sink):
            raise ValueError("A memory trace sink must be a callable")
        self.sink = sink
        self.request_id = request_id or uuid4().hex
        self.correlation_id = correlation_id
        self.learner_id = str(learner_id)
        self.route = "unknown"
        self.generation_called = False
        self.delivered_source_ids: list[str] = []
        self.sink_errors = 0
        self._completed = False

    # -- state used by the final complete event ------------------------------
    def mark_generation(self) -> None:
        self.generation_called = True

    def add_delivered(self, source_ids) -> None:
        for source_id in source_ids or []:
            if source_id and source_id not in self.delivered_source_ids:
                self.delivered_source_ids.append(str(source_id))

    # -- phase emitters ------------------------------------------------------
    def emit(self, phase: str, *, route: str | None = None, **fields: Any) -> None:
        if phase not in PHASES:
            raise ValueError(f"Unknown memory trace phase {phase!r}")
        if route is not None:
            if route not in ROUTES:
                raise ValueError(f"Unknown memory trace route {route!r}")
            if phase != "complete" or not self._completed:
                self.route = route
        event = {
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "learner_id": self.learner_id,
            "phase": phase,
            "route": route or self.route,
            "event_time_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        try:
            self.sink(_json_safe(event))
        except Exception as exc:  # tracing must never break the chat
            self.sink_errors += 1
            logger.warning("Memory trace sink failed (%s); event dropped", type(exc).__name__)

    def resolver(self, route: str, *, decision: Any = None, resolver_model: str | None = None,
                 invalid_decision: str | None = None, **extra: Any) -> None:
        self.emit("resolver", route=route, decision=_json_safe(decision),
                  resolver_model=resolver_model, invalid_decision=invalid_decision, **extra)

    def retrieval(self, route: str, *, eligible_source_ids: Any = None,
                  retrieved_source_ids: Any = None, selected_source_ids: Any = None,
                  omitted_source_ids: Any = None, budget: Any = None,
                  diagnostics: Any = None, **extra: Any) -> None:
        self.emit("retrieval", route=route,
                  eligible_source_ids=_json_safe(eligible_source_ids),
                  retrieved_source_ids=_json_safe(retrieved_source_ids),
                  selected_source_ids=_json_safe(selected_source_ids),
                  omitted_source_ids=_json_safe(omitted_source_ids),
                  budget=_json_safe(budget), diagnostics=_json_safe(diagnostics), **extra)

    def answer_prompt(self, route: str, *, delivered_source_ids: Any = None,
                      hydrated_quotes: Any = None, omitted_source_ids: Any = None,
                      budget: Any = None, final_prompt: Any = None,
                      final_prompt_sha256: str | None = None,
                      generation_called: bool = True, model: str | None = None,
                      backend: str | None = None, usage: Any = None, **extra: Any) -> None:
        self.emit("answer_prompt", route=route,
                  delivered_source_ids=_json_safe(delivered_source_ids),
                  hydrated_quotes=_json_safe(hydrated_quotes),
                  omitted_source_ids=_json_safe(omitted_source_ids),
                  budget=_json_safe(budget), final_prompt=_json_safe(final_prompt),
                  final_prompt_sha256=final_prompt_sha256,
                  generation_called=bool(generation_called), model=model, backend=backend,
                  usage=_json_safe(usage), **extra)

    def fallback(self, route: str, *, reason: str | None = None,
                 generation_called: bool = False, **extra: Any) -> None:
        # A fallback never made an answer-model call; state that explicitly.
        self.emit("fallback", route=route, reason=reason,
                  generation_called=bool(generation_called), **extra)

    def complete(self, *, route: str | None = None, output_status: str,
                 technical_error: str | None = None, answer_routing: Any = None,
                 duration_seconds: float | None = None) -> None:
        """Emit the final event exactly once; it is authoritative for the request."""
        if self._completed:
            return
        self._completed = True
        self.emit("complete", route=route or self.route, output_status=output_status,
                  generation_called=self.generation_called,
                  delivered_source_ids=list(self.delivered_source_ids),
                  technical_error=technical_error, answer_routing=_json_safe(answer_routing),
                  duration_seconds=duration_seconds, trace_sink_errors=self.sink_errors)


def new_request_trace(sink, *, learner_id: str, correlation_id: Any = None) -> MemoryTrace | None:
    """Build a request trace when a trusted sink is installed, else None."""
    if sink is None:
        return None
    return MemoryTrace(sink, learner_id=learner_id,
                       correlation_id=bounded_correlation_id(correlation_id))
