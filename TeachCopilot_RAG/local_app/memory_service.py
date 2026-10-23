"""Shared research memory rules behind durable application jobs.

Only literal student and teacher sources enter evidence memory. The application
may show a bounded preceding assistant turn as labelled conversational context.
Generated conclusions stay drafts until a teacher approves a version;
competencies are never written here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import time
import uuid

from psycopg2 import InterfaceError, OperationalError

from research_memory.contracts import action_schema, validate_action
from research_memory.clients import input_budget_diagnostic
from research_memory.methods import METHODS, Memory, messages
from .conversation_memory import (interest_messages, interest_schema, preference_projection,
                                  validate_interest_candidates)
from .memory_store import DIALOGUE_CONTEXT_CHARS, SESSION_END_CONTEXT_CHARS
from .memory_response_journal import (MemoryResponseJournal, ResponseJournalError,
                                      database_fingerprint, default_journal_path)


MEMORY_BUDGET = 4000
STATE_FIELDS = ("sources", "summary", "records", "blocks", "events")
TAIL_PRESENTATION_VERSION = "dialogue-pairs-v4-session-end-context"
DIALOGUE_CONTEXT_NOTICE = (
    "Слова ассистента ниже – только контекст ответа ученика. Они не доказывают действия, "
    "самостоятельность или освоение. Короткое согласие ученика тоже не доказывает освоение."
)
SOURCE_CORRECTION_NOTICE = (
    "Исправление источника уточняет прежнюю запись, а не означает новое действие "
    "или прогресс ученика. Исходное утверждение читай вместе со всеми исправлениями; "
    "отозванное утверждение не используй как доказанный факт."
)
SESSION_END_CONTEXT_NOTICE = (
    "Последняя реплика помощника сохранена только как контекст занятия. Она не доказывает, "
    "что ученик решил задание или что задание до сих пор не завершено. Слова помощника могут "
    "содержать ошибку; при возвращении уточни текущее состояние у ученика."
)


def memory_state(memory: Memory) -> dict:
    return {"method": memory.method, "topic_id": memory.topic_id,
            **{key: deepcopy(getattr(memory, key)) for key in STATE_FIELDS}}


def restore_memory(state: dict, call, *, method: str, topic_id: str) -> Memory:
    memory = Memory(method, call, MEMORY_BUDGET, topic_id)
    if state and state.get("method") == method and state.get("topic_id") == topic_id:
        for key in STATE_FIELDS:
            if key in state:
                setattr(memory, key, deepcopy(state[key]))
    return memory


def source_projection(source: dict) -> dict:
    return {"source_id": str(source.get("source_id") or source.get("turn_id")),
            "role": source.get("author_role") or source.get("role"),
            "text": str(source.get("text") or source.get("content") or ""),
            "observed_on": str(source.get("observed_on") or source.get("created_at") or "")[:10],
            **({"corrects_source_id": source["corrects_source_id"]} if source.get("corrects_source_id") else {})}


def _dialogue_contexts(sources: list[dict]) -> dict[str, dict]:
    contexts = {}
    for source in sources:
        if (source.get("author_role") or source.get("role")) != "student":
            continue
        preceding = source.get("dialogue_context") or {}
        text = preceding.get("text") if isinstance(preceding, dict) else None
        if not isinstance(text, str) or not text or preceding.get("role") != "assistant":
            continue
        source_id = str(source.get("source_id") or source.get("turn_id"))
        contexts[source_id] = {"source_id": source_id, "role": "assistant", "usage": "context_only",
                               "text": text[-DIALOGUE_CONTEXT_CHARS:], "excerpt_mode": "tail",
                               "truncated": bool(preceding.get("truncated")) or len(text) > DIALOGUE_CONTEXT_CHARS}
    return contexts


def _session_end_contexts(sources: list[dict]) -> dict[str, dict]:
    contexts = {}
    fields = {"assistant_source_id", "session_id", "role", "usage", "text", "excerpt_mode", "truncated"}
    for source in sources:
        closing = source.get("session_end_context")
        if closing is None:
            continue
        source_id = str(source.get("source_id") or source.get("turn_id"))
        if ((source.get("author_role") or source.get("role")) != "student"
                or not isinstance(closing, dict) or set(closing) != fields
                or closing.get("role") != "assistant" or closing.get("usage") != "context_only"
                or closing.get("excerpt_mode") != "whole" or closing.get("truncated") is not False
                or not isinstance(closing.get("text"), str) or not closing["text"].strip()
                or len(closing["text"]) > SESSION_END_CONTEXT_CHARS
                or not all(isinstance(closing.get(key), str) and closing[key]
                           for key in ("assistant_source_id", "session_id"))):
            raise ValueError("invalid_session_end_context")
        contexts[source_id] = {"source_id": source_id, **deepcopy(closing)}
    return contexts


def _tail_with_dialogue_context(memory: Memory, sources: list[dict], *,
                               selection_events: list[dict] | None = None,
                               session_end_context: list[dict] | None = None) -> tuple[str, set[str], list[dict]]:
    """App-only tail presentation; frozen research state and rules stay literal.

    A selected student's turn and its known preceding assistant context are one
    budget unit. When the pair does not fit, neither side is emitted alone.
    """
    contexts = _dialogue_contexts(sources)
    closing_contexts = _session_end_contexts(sources)
    positions = {source_id: index for index, source_id in enumerate(
        dict.fromkeys(source_id for source_id, _ in memory.blocks))}
    corrections, corrected_by = {}, {}
    links = {source_id: set() for source_id in positions}
    for source in sources:
        target = source.get("corrects_source_id")
        if not target:
            continue
        source_id = str(source.get("source_id") or source.get("turn_id"))
        if (not isinstance(target, str) or source_id not in positions or target not in positions
                or positions[target] >= positions[source_id]
                or (source.get("author_role") or source.get("role")) != "teacher"):
            raise ValueError("source_correction_requires_an_earlier_available_source")
        corrections[source_id] = target
        corrected_by.setdefault(target, []).append(source_id)
        links[source_id].add(target)
        links[target].add(source_id)

    def render(blocks):
        parts, notes, closing_notes, seen = [], [], [], set()
        for source_id, block in blocks:
            if source_id in corrected_by:
                parts.append(f"[Источник {source_id}; исправлен источниками: {', '.join(corrected_by[source_id])}]")
            if source_id in corrections:
                parts.append(f"[Исправление источника {corrections[source_id]}; это не новая попытка ученика]")
            if source_id in contexts and source_id not in seen:
                note = contexts[source_id]
                label = f"[Контекст перед ответом ученика {source_id}; ассистент, не доказательство]"
                excerpt = label + "\n" + note["text"]
                if note["truncated"]:
                    excerpt += f"\n[Показан конец предыдущей реплики, до {DIALOGUE_CONTEXT_CHARS} символов.]"
                parts.append(excerpt)
                notes.append(deepcopy(note))
                seen.add(source_id)
            parts.append(block)
            if source_id in closing_contexts:
                note = closing_contexts[source_id]
                parts.append(f"[Последняя реплика помощника после {source_id}; только контекст, не доказательство]\n"
                             + note["text"])
                closing_notes.append(deepcopy(note))
        has_corrections = any(source_id in corrections for source_id, _ in blocks)
        text = "\n".join(([SOURCE_CORRECTION_NOTICE] if has_corrections else [])
                         + ([DIALOGUE_CONTEXT_NOTICE] if notes else [])
                         + ([SESSION_END_CONTEXT_NOTICE] if closing_notes else []) + parts)
        return text, notes, closing_notes

    # The frozen tail splits long turns into chunks. The application presents
    # whole literal turns so a budget boundary cannot leave half an answer or
    # detach a correction. The stored research state is left unchanged.
    whole_turns = []
    for source_id in dict.fromkeys(source_id for source_id, _ in memory.blocks):
        turn = memory.sources[source_id]
        header = f"[{source_id} | {turn['observed_on']} | {turn['role']}] "
        whole_turns.append((source_id, header + turn['text']))
    by_id = dict(whole_turns)
    selected, text, notes, closing_notes, visited = [], "", [], [], set()
    for source_id, block in reversed(whole_turns):
        if source_id in visited:
            continue
        pending, group = [source_id], set()
        while pending:
            current = pending.pop()
            if current not in group:
                group.add(current)
                pending.extend(links[current] - group)
        visited.update(group)
        candidate_ids = {item[0] for item in selected} | group
        candidate = [(item, by_id[item]) for item in sorted(candidate_ids, key=positions.__getitem__)]
        candidate_text, candidate_notes, candidate_closing = render(candidate)
        if len(candidate_text) > memory.budget:
            if selection_events is not None and group & closing_contexts.keys():
                selection_events.append({"event": "session_end_context_group_skipped",
                    "source_ids": sorted(group, key=positions.__getitem__), "budget_chars": memory.budget,
                    "reason": "remaining_budget_exceeded"})
            if len(group) == 1:
                break  # Ordinary turns keep the existing contiguous-tail rule.
            if selection_events is not None:
                group_text, _, _ = render([(item, by_id[item]) for item in sorted(group, key=positions.__getitem__)])
                selection_events.append({"event": "source_correction_group_skipped",
                    "source_ids": sorted(group, key=positions.__getitem__), "budget_chars": memory.budget,
                    "group_chars": len(group_text),
                    "reason": "group_exceeds_budget" if len(group_text) > memory.budget else "remaining_budget_exceeded"})
            continue
        selected, text, notes, closing_notes = candidate, candidate_text, candidate_notes, candidate_closing
    if session_end_context is not None:
        session_end_context.extend(closing_notes)
    return text, {source_id for source_id, _ in selected}, notes


def profile_projection(card: dict, topic_id: str) -> dict:
    """No private journal, identifiers or raw learner card in action prompts."""
    return {"topic_id": topic_id, "grade": card.get("grade"),
            "language": card.get("language"),
            **preference_projection(card),
            "current_goal": (card.get("rag_context") or {}).get("current_goal", {})}


class MemoryWorker:
    def __init__(self, store, runtime, worker_id: str | None = None, *, response_journal=None):
        self.store, self.runtime = store, runtime
        self.worker_id = worker_id or "memory-" + uuid.uuid4().hex
        self.response_journal = response_journal if response_journal is not None else MemoryResponseJournal(
            default_journal_path(), database_identity=database_fingerprint(getattr(store, "dsn", None)))
        self.calls: list[dict] = []
        self._active_job: dict | None = None

    def _call(self, prompt_messages: list[dict], purpose: str, schema: dict | None) -> str:
        # Unlike conversation bounding, writer inputs may never be silently cut.
        encoded = json.dumps(prompt_messages, ensure_ascii=False)
        context_tokens = int(getattr(self.runtime, "context_tokens", 8192))
        diagnostic = input_budget_diagnostic(prompt_messages, schema, context_tokens=context_tokens,
                                              max_output_tokens=1536, purpose=purpose)
        if purpose == "conversation_interests":
            # The frozen research helper knows only its academic writer names.
            diagnostic["enforced"] = True
        started = time.perf_counter()
        event = {"purpose": purpose, "input_chars": len(encoded), "input_budget": diagnostic}
        self.calls.append(event)
        try:
            if diagnostic.get("enforced") and not diagnostic.get("within_guard"):
                raise ValueError("memory_writer_input_exceeds_context")
            if self._active_job and hasattr(self.store, "start_memory_job_call"):
                job_id = self._active_job["job_id"]
                if not self.store.heartbeat_memory_job(job_id, self.worker_id, lease_seconds=600):
                    raise ValueError("memory_job_lease_expired")
            call_id = self.response_journal.prepare(
                job_id=str((self._active_job or {}).get("job_id") or "unbound"),
                worker_id=self.worker_id, purpose=purpose, call_index=len(self.calls))
            event["response_journal_id"] = call_id
            if self._active_job and hasattr(self.store, "start_memory_job_call"):
                self.store.start_memory_job_call(job_id, self.worker_id)
            try:
                response = self.runtime.chat(prompt_messages, response_format=schema,
                                             max_output_tokens=1536, truncate_history=False)
            except Exception as exc:
                self.response_journal.record(call_id, outcome="raised",
                    raw_response=getattr(exc, "raw_response", None), error_type=type(exc).__name__,
                    elapsed_seconds=time.perf_counter() - started)
                raise
            else:
                # Commit locally before parsing or any PostgreSQL completion.
                self.response_journal.record(call_id, outcome="returned", raw_response=response,
                                             elapsed_seconds=time.perf_counter() - started)
            text = response.get("message", {}).get("content")
            event.update(raw_response=response, prompt_tokens=response.get("prompt_eval_count"),
                         completion_tokens=response.get("eval_count"))
            if purpose in {"summary", "extract", "conversation_interests"} and int(response.get("prompt_eval_count") or 0) >= context_tokens - 1536:
                raise ValueError("context_saturation_requires_review")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty_memory_completion")
            return text
        except Exception as exc:
            if isinstance(getattr(exc, "raw_response", None), dict):
                event["raw_response"] = deepcopy(exc.raw_response)
                event["prompt_tokens"] = exc.raw_response.get("prompt_eval_count")
                event["completion_tokens"] = exc.raw_response.get("eval_count")
            raise
        finally:
            event["elapsed_seconds"] = time.perf_counter() - started

    def _prepare(self, job: dict) -> tuple[dict | None, dict]:
        learner_id, topic_id = job["learner_id"], job.get("topic_id") or "general"
        method = job.get("method", "tail")
        if method not in METHODS:
            raise ValueError("unsupported_memory_method")
        snapshot = self.store.get_learner_memory(learner_id, topic_id)
        versions = [row for row in snapshot.get("versions", [])
                    if (row.get("method") or (row.get("state") or {}).get("method")) == method
                    and (row.get("state") or {}).get("topic_id") == topic_id]
        # The store returns newest versions first. Failed drafts never replace a
        # ready version because they are not committed by finish_memory_job.
        prior = versions[0].get("state", {}) if versions else {}
        memory = restore_memory(prior, self._call, method=method, topic_id=topic_id)
        sources = [source for source in snapshot.get("sources", [])
                   if (source.get("author_role") or source.get("role")) in {"student", "teacher"}]
        if method in {"summary", "structured"} and any(source.get("corrects_source_id") for source in sources):
            # Source relations are an application feature. Until other writers
            # have a matching contract, never silently discard the relation.
            raise ValueError("Для истории с исправлениями источников используйте метод tail.")
        groups: dict[str, list[dict]] = {}
        for source in sources:
            projected = source_projection(source)
            if projected["source_id"] in memory.sources:
                continue
            # A dialogue is one update; individual teacher notes form one lesson.
            key = str(source.get("session_id") or source.get("learning_session_id") or projected["source_id"])
            groups.setdefault(key, []).append(projected)
        update_failure = None
        try:
            for key, group in groups.items():
                memory.update({"lesson_id": key, "observed_on": group[-1]["observed_on"] or date.today().isoformat(),
                               "turns": [{"turn_id": row["source_id"], "role": row["role"], "text": row["text"]}
                                         for row in group]})
        except ValueError as exc:
            if not prior or not any(code in str(exc) for code in (
                    "memory_writer_input_exceeds_context", "context_saturation_requires_review",
                    "Complete memory input exceeds")):
                raise
            # Application availability fallback. The failed update and raw call
            # remain visible; an old state is never resaved as a fresh version.
            update_failure = {"type": type(exc).__name__, "message": str(exc),
                              "fallback_version_id": versions[0].get("version_id") or versions[0].get("id")}
            memory = restore_memory(prior, self._call, method=method, topic_id=topic_id)
        selected = memory.context(topic_id)
        dialogue_context = []
        session_end_context = []
        selection_events = []
        if method == "tail":
            for source in sources:
                if source.get("session_end_context_omitted"):
                    selection_events.append({"event": "session_end_context_omitted",
                        "source_id": str(source.get("source_id")), **source["session_end_context_omitted"]})
            selected.text, selected.evidence_ids, dialogue_context = _tail_with_dialogue_context(
                memory, sources, selection_events=selection_events, session_end_context=session_end_context)
        visible = [source_projection(row) for row in sources
                   if str(row.get("source_id")) in selected.evidence_ids]
        context = {"text": selected.text, "evidence_ids": sorted(selected.evidence_ids),
                   "sources": visible, "method": method, "topic_id": topic_id}
        if method == "tail":
            context.update(presentation_version=TAIL_PRESENTATION_VERSION, dialogue_context=dialogue_context)
            if session_end_context:
                context["session_end_context"] = session_end_context
            if selection_events:
                context["selection_events"] = selection_events
        if update_failure:
            context.update(stale=True, update_failure=update_failure,
                           memory_version_id=update_failure["fallback_version_id"])
        return None if update_failure else memory_state(memory), context

    def _conversation_interests(self, job: dict, context: dict) -> None:
        session_id = (job.get("payload") or {}).get("learning_session_id")
        if job["kind"] != "memory_update" or not session_id:
            return
        # Old workers/test stores without session support preserve their contract.
        if not all(hasattr(self.store, name) for name in ("get_learning_session", "list_learning_messages")):
            return
        context["interest_candidates"] = []
        try:
            session = self.store.get_learning_session(session_id, job["learner_id"])
            if not session or session.get("status") != "completed":
                raise ValueError("interest extraction requires a completed learner session")
            sources = [{"source_id": str(row["source_id"]), "author_role": "student",
                        "content": row.get("content", row.get("text", ""))}
                       for row in self.store.list_learning_messages(session_id)
                       if row.get("author_role") == "student"]
            if not sources:
                context["interest_extraction"] = {"status": "skipped", "reason": "no_student_sources"}
                return
            raw = self._call(interest_messages(sources), "conversation_interests", interest_schema(sources))
            context["interest_candidates"] = validate_interest_candidates(raw, sources)
            context["interest_extraction"] = {"status": "ready", "learning_session_id": session_id}
        except (OperationalError, InterfaceError, ResponseJournalError):
            # DB acknowledgement and local durability failures must stop this
            # job even when interest extraction is otherwise an optional draft.
            raise
        except Exception as exc:
            # A new conversational draft must never discard a ready academic tail.
            # Raw model outputs are kept in the job call diagnostics, without retry.
            context["interest_extraction"] = {"status": "failed", "type": type(exc).__name__,
                                               "message": str(exc), "learning_session_id": session_id}

    def run_once(self) -> dict | None:
        job = self.store.claim_memory_job(self.worker_id, lease_seconds=600)
        if not job:
            return None
        self.calls = []
        self._active_job = job
        try:
            from local_app.runtime import generation_priority
            with generation_priority("background" if job["kind"] == "memory_update" else "interactive"):
                state, context = self._prepare(job)
                result = {"method": job.get("method", "tail"), "review_required": True}
                if context.get("update_failure"):
                    result.update(stale=True, memory_update_error=context["update_failure"],
                                  memory_version_id=context.get("memory_version_id"))
                    if job["kind"] == "memory_update":
                        self.store.finish_memory_job(job["job_id"], result=result,
                            error={**context["update_failure"], "calls": self.calls}, worker_id=self.worker_id)
                        return self.store.get_memory_job(job["job_id"])
                self._conversation_interests(job, context)
                action = None
                if job["kind"] == "next_step":
                    topic_id = job.get("topic_id") or "general"
                    card = self.store.get_card(job["learner_id"])
                    if not card:
                        raise ValueError("learner_not_found")
                    payload = {"profile": profile_projection(card, topic_id),
                               "request": (job.get("payload") or {}).get("request") or
                               "Предложи одно следующее учебное действие по выбранной теме.",
                               "memory": context["text"],
                               "memory_status": {"stale": bool(context.get("stale")),
                                                 "update_failure": context.get("update_failure")}}
                    raw = self._call(messages("action", payload), "action",
                                     action_schema(set(context["evidence_ids"])))
                    action = validate_action(raw, set(context["evidence_ids"]))
                    from .recommendation_review import recommendation_source_review
                    action["evidence_review"] = recommendation_source_review(action, context)
                    action.update(sources=context["sources"], method=job.get("method", "tail"),
                                  learner_id=job["learner_id"], topic_id=topic_id,
                                  revision=job.get("base_revision"), stale=bool(context.get("stale")))
                    result.update(action)
                result["calls"] = self.calls
                self.store.finish_memory_job(job["job_id"], result=result, memory_state=state,
                    next_step=action, context=context, worker_id=self.worker_id, approved=False)
        except (OperationalError, InterfaceError, ResponseJournalError):
            # Neither a lost DB acknowledgement nor a journal failure permits
            # a second finish or generation. Persisted lease/call markers keep
            # their conservative recovery semantics.
            raise
        except Exception as exc:
            # Preserve failed/truncated responses and diagnostics. No repair,
            # citation deduplication or hidden repeated model request occurs.
            self.store.finish_memory_job(job["job_id"], error={"type": type(exc).__name__,
                "message": str(exc), "calls": self.calls}, worker_id=self.worker_id,
                unknown_outcome=job.get("provider") == "external")
        return self.store.get_memory_job(job["job_id"])


def approved_chat_memory(snapshot: dict, topic_id: str | None = None) -> dict:
    versions = [row for row in snapshot.get("versions", []) if row.get("approved")
                and (row.get("method") or (row.get("state") or {}).get("method")) == "tail"
                and (not topic_id or row.get("topic_id") == topic_id)]
    if not versions:
        return {"text": "", "sources": [], "stale": False, "version_id": None}
    version = versions[0]
    context = deepcopy(version.get("context") or {})
    # A normal new attempt may make memory stale while still useful. An explicit
    # correction must not leave a withdrawn assertion in the answer meanwhile.
    visible = set(context.get("evidence_ids") or []) | {
        str(source.get("source_id")) for source in context.get("sources", []) if source.get("source_id")}
    links = {}
    for source in snapshot.get("sources", []):
        target = source.get("corrects_source_id")
        if not target or source.get("topic_id") != version.get("topic_id"):
            continue
        source_id = str(source.get("source_id"))
        links.setdefault(source_id, set()).add(target)
        links.setdefault(target, set()).add(source_id)
    pending, related = list(visible), set(visible)
    while pending:
        for linked in links.get(pending.pop(), ()):
            if linked not in related:
                related.add(linked)
                pending.append(linked)
    if (related - visible) or (links and not visible and context.get("text")):
        return {"text": "", "sources": [], "evidence_ids": [], "stale": True,
                "version_id": None, "topic_id": version.get("topic_id"),
                "withheld_reason": "source_correction_requires_review"}
    context.update(version_id=version.get("version_id"), topic_id=version.get("topic_id"),
                   stale=bool(version.get("stale")) or version.get("base_revision") != snapshot.get("revision"))
    return context
