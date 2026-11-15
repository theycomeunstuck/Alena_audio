"""Opt-in bridge from approved application sources to experimental personal graph memory.

The default application does not import Graphiti or LangGraph. This module does
not accept client-supplied source text; its caller supplies a store snapshot
after the existing authorization checks. Graph facts guide retrieval, but only
original teacher/student turns enter the generated context as evidence.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import date
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from urllib.parse import urlsplit

from .personal_graph_contract import (PERSONAL_GRAPH_CONTEXT_PROTOCOL,
                                      PERSONAL_GRAPH_MEMORY_BYTE_BUDGET,
                                      PERSONAL_GRAPH_SELECTION_BYTE_BUDGET,
                                      personal_graph_budgets,
                                      personal_graph_snapshot_fingerprint)

METHOD = "personal_graph"
MEMORY_BYTE_BUDGET = PERSONAL_GRAPH_MEMORY_BYTE_BUDGET
SELECTION_BYTE_BUDGET = PERSONAL_GRAPH_SELECTION_BYTE_BUDGET
APP_SELECTOR_VERSION = "app_budgeted_sources_v2"
_SOURCE_HIT_CHANNELS = ("semantic", "lexical", "source_semantic", "source_lexical", "chronological")
RETRIEVAL_MODES = frozenset({"chronological", "lexical", "vector", "hybrid", "graph_hybrid"})
SOURCE_CONTEXT_NOTICE = (
    "Реплики ассистента – только контекст. Исправления относятся к конкретным источникам; "
    "не используй исправленное утверждение как действующий факт. "
    "Поисковые совпадения – кандидаты: проверь, действительно ли оригинальная реплика отвечает на вопрос, "
    "и сообщи о недостатке данных, если подтверждения нет.")
SOURCE_CONTEXT_SKIPPED_NOTICE = " Некоторые кандидаты не вошли в память; предупреждения показывают причину пропуска."


class PersonalGraphSelectionBudgetExceeded(ValueError):
    """The approved source context cannot fit without losing required evidence."""


def round_robin_source_hits(hits):
    """Interleave archive channel ranks before long-lesson hydration truncates them."""
    queues = {channel: [] for channel in _SOURCE_HIT_CHANNELS}
    other = []
    for hit in hits:
        channel = hit.get("kind") if isinstance(hit, dict) else None
        if channel in queues and isinstance(hit.get("chunk_id"), str):
            queues[channel].append(hit)
        else:
            other.append(hit)
    ordered, positions = [], {channel: 0 for channel in _SOURCE_HIT_CHANNELS}
    while True:
        advanced = False
        for channel in _SOURCE_HIT_CHANNELS:
            position = positions[channel]
            if position >= len(queues[channel]):
                continue
            advanced = True
            ordered.append(queues[channel][position])
            positions[channel] += 1
        if not advanced:
            break
    return ordered + other


def select_application_candidates(lessons, hits, budget, *, counter=None, recent_lesson_ids=(), max_budget=None):
    """Select exact source hits and recent context atomically within the app byte budget.

    Semantic and lexical retrieval ranks are kept in separate queues because their
    scores are not comparable. A graph hit without an exact chunk ID never raises
    a fallback fragment's priority. Recent context gets one of the first two
    selection slots alongside the best directly retrieved source when both fit.
    """
    if budget <= 0 or (max_budget is not None and budget > max_budget):
        raise ValueError("Invalid selected-memory budget for the working window")
    if not isinstance(lessons, list) or not isinstance(hits, list):
        raise ValueError("Invalid application memory candidates")
    measure = counter or (lambda value: len(value.encode("utf-8")))

    blocks, by_fragment, by_lesson, context_only = [], {}, {}, set()
    source_turns_by_candidate = {}
    recent_rank = {str(lesson_id): index for index, lesson_id in enumerate(recent_lesson_ids)}
    for index, lesson in enumerate(lessons):
        if not isinstance(lesson, dict) or not isinstance(lesson.get("turns"), list):
            raise ValueError("Invalid hydrated application lesson")
        app_recent = lesson.get("_application_recent_context") is True
        canonical_lesson = {key: value for key, value in lesson.items()
                            if key != "_application_recent_context"}
        block = {"derived_candidates": [], "original_lesson": canonical_lesson}
        blocks.append(block)
        lesson_id = str(canonical_lesson.get("lesson_id") or "")
        fragment_id = canonical_lesson.get("fragment_id")
        if not any(turn.get("role") in {"teacher", "student"} and turn.get("turn_id")
                   for turn in canonical_lesson.get("turns", [])):
            context_only.add(index)
            continue
        source_turn_ids = [turn["turn_id"] for turn in canonical_lesson["turns"]
                           if turn.get("role") in {"teacher", "student"}]
        if len(source_turn_ids) != len(set(source_turn_ids)):
            raise ValueError("hydrated application candidate contains duplicate source turns")
        source_turns_by_candidate[index] = set(source_turn_ids)
        if fragment_id:
            by_fragment.setdefault(str(fragment_id), []).append(index)
        by_lesson.setdefault(lesson_id, []).append(index)
        if app_recent or (not fragment_id and lesson_id in recent_rank):
            # Larger indices in recent_lesson_ids refer to newer lessons.
            canonical_lesson["_selection_recent_rank"] = recent_rank.get(lesson_id, -1)

    channel_queues = {channel: [] for channel in _SOURCE_HIT_CHANNELS}
    channel_seen = {channel: set() for channel in _SOURCE_HIT_CHANNELS}
    unhydrated_hits = {}
    for hit in hits:
        if not isinstance(hit, dict):
            continue
        channel, chunk_id = hit.get("kind"), hit.get("chunk_id")
        lesson_ids = hit.get("lesson_ids")
        if channel not in channel_queues or not isinstance(chunk_id, str) or not isinstance(lesson_ids, list):
            continue
        # An exact fragment ID is required for a long lesson. For a hydrated
        # whole lesson, the archive hit still identifies that lesson's source.
        matching = []
        for lesson_id in lesson_ids:
            for candidate_index in by_fragment.get(chunk_id, []):
                candidate = blocks[candidate_index]["original_lesson"]
                if candidate.get("lesson_id") == lesson_id:
                    matching.append(candidate_index)
            for candidate_index in by_lesson.get(str(lesson_id), []):
                candidate = blocks[candidate_index]["original_lesson"]
                if not candidate.get("fragment_id"):
                    matching.append(candidate_index)
        for candidate_index in matching:
            if candidate_index not in channel_seen[channel]:
                channel_seen[channel].add(candidate_index)
                channel_queues[channel].append(candidate_index)
        if not matching:
            for lesson_id in lesson_ids:
                lesson_key = str(lesson_id)
                if lesson_key in by_lesson:
                    unhydrated_hits[(lesson_key, chunk_id)] = {
                        "lesson_id": lesson_key, "fragment_id": chunk_id,
                        "reason": "source_fragment_not_hydrated",
                    }

    # Round-robin channel ranks so semantic results cannot crowd out lexical
    # results merely because archive.search returns the channels consecutively.
    direct_order, direct_seen = [], set()
    queue_position = {channel: 0 for channel in _SOURCE_HIT_CHANNELS}
    while True:
        advanced = False
        for channel in _SOURCE_HIT_CHANNELS:
            position = queue_position[channel]
            queue = channel_queues[channel]
            if position >= len(queue):
                continue
            advanced = True
            candidate_index = queue[position]
            queue_position[channel] += 1
            if candidate_index not in direct_seen:
                direct_seen.add(candidate_index)
                direct_order.append(candidate_index)
        if not advanced:
            break

    recent_order = []
    for candidate_index, block in enumerate(blocks):
        lesson = block["original_lesson"]
        if "_selection_recent_rank" in lesson:
            recent_order.append(candidate_index)
    recent_order.sort(key=lambda candidate_index: (
        -blocks[candidate_index]["original_lesson"].get("_selection_recent_rank", -1),
        str(blocks[candidate_index]["original_lesson"].get("observed_on") or ""),
        candidate_index,
    ))
    recent_set = set(recent_order)

    # Internal scheduling metadata must never be serialized as memory.
    for block in blocks:
        block["original_lesson"].pop("_selection_recent_rank", None)

    def memory_text(indices):
        if not indices:
            return ""
        return json.dumps([blocks[index] for index in indices], ensure_ascii=False, separators=(",", ":"))

    def memory_size(indices):
        return measure(memory_text(indices)) if indices else 0

    selected, selected_set, skipped = [], set(), []
    overlap_skipped = set()

    def overlaps(indices):
        seen_turns = set()
        for candidate_index in indices:
            candidate_turns = source_turns_by_candidate.get(candidate_index, set())
            if seen_turns & candidate_turns:
                return True
            seen_turns.update(candidate_turns)
        return False

    def add_if_fits(candidate_indices):
        trial = list(dict.fromkeys(selected + candidate_indices))
        if overlaps(trial):
            overlap_skipped.update(candidate_index for candidate_index in candidate_indices
                                   if candidate_index not in selected_set)
            return False
        if memory_size(trial) <= budget:
            selected[:] = trial
            selected_set.update(candidate_indices)
            return True
        return False

    # Find a pair before spending the budget: one exact retrieval candidate and
    # one recent-context candidate. This avoids a large hit taking all available
    # space before the recency lane gets a chance.
    seed_pair = None
    seed_direct = None
    for direct_index in direct_order:
        if memory_size([direct_index]) > budget:
            continue
        seed_direct = direct_index
        for recent_index in recent_order:
            pair = [direct_index] if direct_index == recent_index else [direct_index, recent_index]
            if overlaps(pair):
                if direct_index != recent_index:
                    overlap_skipped.add(recent_index)
                continue
            if memory_size(pair) <= budget:
                seed_pair = pair
                break
        if seed_pair is not None:
            break
        # Keep the best fitting exact hit even when no recent block can fit
        # beside it; a weaker hit must not replace it just to create a pair.
        break
    if seed_pair is not None:
        selected.extend(seed_pair)
        selected_set.update(seed_pair)
    elif seed_direct is not None:
        selected.append(seed_direct)
        selected_set.add(seed_direct)
    else:
        # No exact source hit fits on its own, so leave room for a recent block.
        for candidate_index in recent_order:
            if add_if_fits([candidate_index]):
                break

    # Work-conserving alternation keeps both lanes represented when possible.
    lanes = [("direct", direct_order), ("recent", recent_order)]
    lane_positions = {name: 0 for name, _ in lanes}
    lane_name = "direct" if seed_pair or (selected and selected[0] in recent_set) else "recent"
    while True:
        advanced = False
        for offset in range(len(lanes)):
            name = lanes[(0 if lane_name == "direct" else 1) ^ offset][0]
            order = direct_order if name == "direct" else recent_order
            position = lane_positions[name]
            while position < len(order) and order[position] in selected_set:
                position += 1
            lane_positions[name] = position
            if position >= len(order):
                continue
            advanced = True
            candidate_index = order[position]
            lane_positions[name] += 1
            add_if_fits([candidate_index])
            lane_name = "recent" if name == "direct" else "direct"
            break
        if not advanced:
            break

    candidate_indexes = direct_seen | recent_set
    for candidate_index in sorted(candidate_indexes):
        if candidate_index in selected_set:
            continue
        lesson = blocks[candidate_index]["original_lesson"]
        skipped.append({"lesson_id": lesson.get("lesson_id"),
                        "fragment_id": lesson.get("fragment_id"),
                        "reason": ("duplicate_source_turns" if candidate_index in overlap_skipped
                                   else "memory_budget_or_context_exceeded")})
    # A non-recent lesson introduced only by a chunkless graph edge is context,
    # not a directly retrieved source fragment, so it is withheld and reported.
    for candidate_index, block in enumerate(blocks):
        if candidate_index in context_only:
            continue
        if candidate_index in candidate_indexes:
            continue
        lesson = block["original_lesson"]
        skipped.append({"lesson_id": lesson.get("lesson_id"),
                        "fragment_id": lesson.get("fragment_id"),
                        "reason": "no_exact_source_chunk_match"})
    skipped.extend(unhydrated_hits.values())

    source_ids = [turn["turn_id"] for index in selected
                  for turn in blocks[index]["original_lesson"].get("turns", [])
                  if turn.get("role") in {"teacher", "student"}]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("hydrated application candidates contain duplicate source turns")
    text = memory_text(selected)
    used = memory_size(selected)
    return {"memory": text, "memory_tokens": used, "source_ids": source_ids, "skipped": skipped}


def add_recent_context_candidates(lessons, chunk_rows, recent_lesson_ids, *, allowed_source_ids=None):
    """Add each recent lesson's last original-source chunk without widening chat scope."""
    candidates = [dict(lesson) for lesson in lessons]
    by_lesson = {}
    for row in chunk_rows:
        by_lesson.setdefault(row["lesson_id"], []).append(row)
    allowed = None if allowed_source_ids is None else set(allowed_source_ids)
    for lesson_id in recent_lesson_ids:
        existing_indexes = [index for index, lesson in enumerate(candidates)
                            if lesson.get("lesson_id") == lesson_id]
        if any(not candidates[index].get("fragment_id") for index in existing_indexes):
            for index in existing_indexes:
                candidates[index]["_application_recent_context"] = True
            continue
        rows = sorted(by_lesson.get(lesson_id, []), key=lambda row: row.get("position", 0))
        tail = next((row for row in reversed(rows)
                     if any(turn.get("role") in {"teacher", "student"} and
                            (allowed is None or turn.get("turn_id") in allowed)
                            for turn in row.get("content", {}).get("turns", []))), None)
        if tail is None:
            continue
        chunk_id = tail["chunk_id"]
        existing_index = next((index for index in existing_indexes
                               if candidates[index].get("fragment_id") == chunk_id), None)
        content = tail["content"]
        if allowed is not None:
            evidence = {turn.get("turn_id") for turn in content.get("turns", [])
                        if turn.get("role") in {"teacher", "student"} and turn.get("turn_id") in allowed}
            if not evidence:
                continue
            turns = [turn for turn in content.get("turns", []) if turn.get("turn_id") in evidence or
                     (turn.get("role") == "assistant" and turn.get("anchor_source_id") in evidence)]
            content = {**content, "turns": turns}
        if existing_index is None:
            candidates.append({**content, "fragment_id": chunk_id, "_application_recent_context": True})
        else:
            candidates[existing_index]["_application_recent_context"] = True
    return candidates


def application_lesson_order(lessons):
    """Order dates chronologically, preserving source_order for ties within a day."""
    return [lesson["lesson_id"] for lesson in sorted(lessons, key=lambda lesson: lesson["observed_on"])]


def application_memory_workflow(base_workflow, archive, model, graphiti, *,
                                full_lesson_max_bytes=1800, source_order=(), retrieval_mode="graph_hybrid"):
    """Build the app-only workflow variant while leaving research selection intact."""
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError("Unknown application retrieval mode")

    class ApplicationPersonalMemoryWorkflow(base_workflow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.application_source_order = list(dict.fromkeys(str(item) for item in source_order))

        async def retrieve(self, state):
            if retrieval_mode == "graph_hybrid":
                retrieved = await super().retrieve(state)
            else:
                run, child = state["run_id"], state["learner_id"]
                allowed = state.get("allowed_source_ids")
                if retrieval_mode == "chronological":
                    lessons = await asyncio.to_thread(archive.lessons, run, child)
                    order = self.application_source_order or application_lesson_order(lessons)
                    order_rank = {lesson_id: index for index, lesson_id in enumerate(order)}
                    rows = await asyncio.to_thread(archive.chunks, run, child)
                    rows.sort(key=lambda row: (order_rank[row["lesson_id"]], row["position"]), reverse=True)
                    if allowed is not None:
                        rows = [row for row in rows if any(turn.get("turn_id") in allowed
                                for turn in row["content"].get("turns", []))]
                    hits = [{"kind": "chronological", "lesson_ids": [row["lesson_id"]],
                             "chunk_id": row["chunk_id"]} for row in rows]
                else:
                    channels = {"lexical": ("lexical",), "vector": ("semantic",),
                                "hybrid": ("semantic", "lexical")}[retrieval_mode]
                    vector = (await asyncio.to_thread(model.embed, state["question"])
                              if "semantic" in channels else None)
                    rows = await asyncio.to_thread(archive.search, run, child, state["question"], vector,
                                                   source_ids=allowed, channels=channels)
                    hits = [{"kind": row["channel"], "lesson_ids": [row["lesson_id"]],
                             "chunk_id": row["chunk_id"], "score": row["score"]} for row in rows]
                retrieved = {"hits": hits, "unresolved": [],
                             "lesson_ids": list(dict.fromkeys(item for hit in hits for item in hit["lesson_ids"]))}
            if self.application_source_order:
                recent_ids = self.application_source_order[-2:]
                retrieved["lesson_ids"] = recent_ids + [item for item in retrieved["lesson_ids"]
                                                        if item not in recent_ids]
            return retrieved

        async def hydrate(self, state):
            # Hydrate every directly retrieved fragment. The selection budget,
            # not a fixed four-fragment cap, determines which evidence survives.
            run, child = state["run_id"], state["learner_id"]
            ids = state.get("lesson_ids", [])
            lessons = await asyncio.to_thread(archive.lessons, run, child, ids)
            by_lesson = {lesson["lesson_id"]: lesson for lesson in lessons}
            if set(by_lesson) != set(ids):
                raise ValueError("A retrieved source does not belong to the requested child and run")
            rows = await asyncio.to_thread(archive.chunks, run, child, ids)
            by_chunk = {row["chunk_id"]: row for row in rows}
            allowed = set(state["allowed_source_ids"]) if "allowed_source_ids" in state else None
            candidates = []

            def append_permitted(lesson):
                if allowed is None:
                    candidates.append(lesson)
                    return
                evidence = {turn.get("turn_id") for turn in lesson["turns"]
                            if turn.get("role") in {"teacher", "student"} and turn.get("turn_id") in allowed}
                if evidence:
                    candidates.append({**lesson, "turns": [turn for turn in lesson["turns"]
                        if turn.get("turn_id") in evidence or (turn.get("role") == "assistant"
                                                               and turn.get("anchor_source_id") in evidence)]})

            balanced_hits = round_robin_source_hits(state.get("hits", []))
            for lesson_id in ids:
                lesson = by_lesson[lesson_id]
                if len(json.dumps(lesson, ensure_ascii=False).encode("utf-8")) <= self.full_lesson_max_bytes:
                    append_permitted(lesson)
                    continue
                ordered = list(dict.fromkeys(hit["chunk_id"] for hit in balanced_hits
                    if hit.get("chunk_id") in by_chunk and lesson_id in hit.get("lesson_ids", [])))
                for chunk_id in ordered:
                    row = by_chunk[chunk_id]
                    if row["lesson_id"] != lesson_id or row["content"].get("lesson_id") != lesson_id:
                        raise ValueError("Fragment belongs to a different lesson")
                    append_permitted({**row["content"], "fragment_id": chunk_id})
            recent_ids = list(state.get("lesson_ids", []))[:2]
            if not recent_ids:
                return {"lessons": candidates}
            candidates = add_recent_context_candidates(
                candidates, rows, recent_ids, allowed_source_ids=allowed)
            return {"lessons": candidates}

        async def select(self, state):
            return select_application_candidates(
                state["lessons"], state["hits"], state["budget_tokens"],
                counter=model.count, recent_lesson_ids=state.get("lesson_ids", [])[:2],
                max_budget=model.context - 2048,
            )

    return ApplicationPersonalMemoryWorkflow(archive, model, graphiti=graphiti,
                                              include_candidates=False,
                                              full_lesson_max_bytes=full_lesson_max_bytes)


def source_lessons(snapshot, learner_id):
    if snapshot.get("learner_id") != learner_id:
        raise ValueError("memory snapshot belongs to another learner")
    groups, by_id = OrderedDict(), {}
    for source in snapshot.get("sources", []):
        source_id = str(source.get("source_id") or "")
        role = source.get("author_role") or source.get("role")
        text = source.get("text") or source.get("content")
        if not source_id or source_id in by_id or role not in {"teacher", "student"}:
            raise ValueError("invalid or duplicate application source")
        if role == "teacher" and source.get("approved") is not True:
            raise ValueError("teacher source is not approved")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty application source")
        observed = str(source.get("observed_on") or source.get("created_at") or "")[:10]
        try:
            date.fromisoformat(observed)
        except ValueError:
            raise ValueError("source date is missing or invalid") from None
        topic = str(source.get("topic_id") or "general")
        session = str(source.get("session_id") or source.get("learning_session_id") or "source:" + source_id)
        key = (session, observed)
        if key not in groups:
            digest = hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:24]
            groups[key] = {"lesson_id": "app-lesson-" + digest, "observed_on": observed,
                           "topic_ids": [], "turns": []}
        group = groups[key]
        if topic not in group["topic_ids"]:
            group["topic_ids"].append(topic)
        preceding = source.get("dialogue_context")
        if role == "student" and isinstance(preceding, dict) and preceding.get("role") == "assistant":
            question = preceding.get("text")
            if isinstance(question, str) and question.strip():
                from .memory_store import DIALOGUE_CONTEXT_CHARS
                group["turns"].append({"turn_id": "context:before:" + source_id, "role": "assistant",
                    "usage": "context_only", "anchor_source_id": source_id,
                    "excerpt_mode": "tail", "truncated": bool(preceding.get("truncated")) or len(question) > DIALOGUE_CONTEXT_CHARS,
                    "text": question[-DIALOGUE_CONTEXT_CHARS:]})
        turn = {"turn_id": source_id, "role": role, "text": text}
        if source.get("corrects_source_id"):
            turn["corrects_source_id"] = str(source["corrects_source_id"])
        group["turns"].append(turn)
        ending = source.get("session_end_context")
        if role == "student" and isinstance(ending, dict) and ending.get("role") == "assistant" \
                and ending.get("usage") == "context_only" and isinstance(ending.get("text"), str):
            group["turns"].append({"turn_id": "context:after:" + source_id, "role": "assistant",
                                   "usage": "context_only", "anchor_source_id": source_id,
                                   "excerpt_mode": "whole", "truncated": False,
                                   "text": ending["text"]})
        by_id[source_id] = {"source_id": source_id, "role": role, "text": text,
                            "observed_on": observed, "topic_id": topic,
                            **({"corrects_source_id": str(source["corrects_source_id"])}
                               if source.get("corrects_source_id") else {})}
    for item in by_id.values():
        if item.get("corrects_source_id") and item["corrects_source_id"] not in by_id:
            raise ValueError("a correction target is absent from the source snapshot")
    return list(groups.values()), by_id


def correction_closure(sources, source_ids):
    links = {source_id: set() for source_id in sources}
    for source_id, source in sources.items():
        target = source.get("corrects_source_id")
        if target:
            links[source_id].add(target)
            links[target].add(source_id)
    pending, closure = list(source_ids), set(source_ids)
    while pending:
        current = pending.pop()
        for related in links.get(current, ()):
            if related not in closure:
                closure.add(related)
                pending.append(related)
    return closure


def source_scoped_found(lessons, source_ids, *, skipped_source_ids=()):
    """Build retrieval blocks from selected turns and their anchored contexts only."""
    selected_ids = set(source_ids)
    skipped_ids = set(skipped_source_ids) - selected_ids
    blocks = []
    lesson_by_source = {}
    for lesson in lessons:
        for turn in lesson.get("turns", []):
            if turn.get("role") in {"teacher", "student"}:
                lesson_by_source[turn.get("turn_id")] = lesson["lesson_id"]
        turns = [turn for turn in lesson.get("turns", [])
                 if (turn.get("role") in {"teacher", "student"} and turn.get("turn_id") in selected_ids)
                 or (turn.get("role") == "assistant" and turn.get("anchor_source_id") in selected_ids)]
        if turns:
            blocks.append({"original_lesson": {"lesson_id": lesson["lesson_id"],
                                                "observed_on": lesson["observed_on"],
                                                "topic_ids": lesson["topic_ids"], "turns": turns}})
    skipped = [{"lesson_id": lesson_by_source.get(source_id),
                "reason": "memory_budget_or_context_exceeded"}
               for source_id in sorted(skipped_ids) if source_id in lesson_by_source]
    return {"memory": json.dumps(blocks, ensure_ascii=False, separators=(",", ":")),
            "unresolved": [], "skipped": skipped}


def source_scoped_search(question, lessons, allowed_ids, embed_many, *, limit=8, include_memory=True,
                         source_embed_many=None, mode="hybrid"):
    """Rank allowed source turns with only their directly anchored assistant context.

    Every mode, hybrid included, goes through personal_graph_retrieval.rank_source_candidates. The private hybrid
    implementation that used to live here was merged away (FIX_LIST item 6). Accepted differences from it:
    candidates are ordered newest lesson first, equal scores are broken by recency, and embeddings must be
    1024-dimensional and non-zero.
    """
    if mode not in {"chronological", "lexical", "vector", "hybrid"}:
        raise ValueError("Selected context does not permit graph retrieval")
    from .personal_graph_retrieval import rank_source_candidates
    ranked = rank_source_candidates(question, lessons, allowed_ids, mode=mode,
        embed_many=embed_many, source_embed_many=source_embed_many, limit=limit)
    found = (source_scoped_found(lessons, ranked["ranked_source_ids"]) if include_memory
             else {"memory": "", "unresolved": [], "skipped": []})
    return {**found, **ranked}


def select_context(found, snapshot, learner_id, topic_id, *, allowed_ids=None, memory_byte_budget=None):
    """Keep canonical turns only and require every linked correction together."""
    memory_byte_budget = MEMORY_BYTE_BUDGET if memory_byte_budget is None else memory_byte_budget
    lessons, sources = source_lessons(snapshot, learner_id)
    allowed = set(sources) if allowed_ids is None else set(allowed_ids)
    if not allowed <= set(sources):
        raise ValueError("approved sources no longer belong to this learner")
    expected_lessons = {lesson["lesson_id"]: lesson for lesson in lessons}
    expected = {turn["turn_id"]: (lesson["lesson_id"], index, turn)
                for lesson in lessons for index, turn in enumerate(lesson["turns"])}
    blocks = json.loads(found.get("memory") or "[]")
    lesson_order = {lesson["lesson_id"]: index for index, lesson in
                    enumerate(sorted(lessons, key=lambda lesson: lesson["observed_on"]))}
    selected, visible, seen = [], set(), set()

    def with_topic(turn):
        anchor = turn.get("anchor_source_id") if turn.get("role") == "assistant" else turn.get("turn_id")
        if anchor not in sources:
            raise ValueError("context anchor is absent from the source snapshot")
        return {**turn, "topic_id": sources[anchor]["topic_id"]}

    for block in blocks:
        lesson = block.get("original_lesson")
        if not isinstance(lesson, dict):
            raise ValueError("invalid original lesson in retrieved memory")
        canonical = expected_lessons.get(lesson.get("lesson_id"))
        if canonical is None or any(lesson.get(key) != canonical.get(key) for key in ("observed_on", "topic_ids")):
            raise ValueError("retrieved lesson does not match the canonical snapshot")
        turns = []
        last_position = -1
        visible_in_block = {turn.get("turn_id") for turn in lesson.get("turns", [])
                            if turn.get("role") in {"teacher", "student"} and turn.get("turn_id") in allowed}
        for turn in lesson.get("turns", []):
            turn_id = turn.get("turn_id")
            source = expected.get(turn_id)
            if (source is None or source[0] != lesson["lesson_id"] or source[1] <= last_position
                    or turn_id in seen or turn != source[2]):
                raise ValueError("retrieved turn does not match the canonical snapshot")
            seen.add(turn_id)
            last_position = source[1]
            if turn["role"] == "assistant":
                if turn.get("anchor_source_id") in visible_in_block:
                    turns.append(with_topic(turn))
            elif turn_id in allowed:
                turns.append(with_topic(turn))
                visible.add(turn_id)
        if turns:
            selected.append({"lesson_id": lesson["lesson_id"], "observed_on": lesson["observed_on"],
                             "topic_ids": list(dict.fromkeys(turn["topic_id"] for turn in turns)), "turns": turns})
    # Search rank determines which blocks fit; presentation follows the source
    # archive so several hits from one lesson remain in event order.
    for item in selected:
        item["turns"].sort(key=lambda turn: expected[turn["turn_id"]][1])
        item["topic_ids"] = list(dict.fromkeys(turn["topic_id"] for turn in item["turns"]))
    selected.sort(key=lambda item: (
        lesson_order[item["lesson_id"]],
        min(expected[turn["turn_id"]][1] for turn in item["turns"]),
    ))
    closure = correction_closure(sources, visible)
    if not closure <= allowed:
        raise ValueError("source correction requires a new approved memory version")
    linked = [sources[source_id] for source_id in sources if source_id in closure - visible]
    included = {turn["turn_id"] for block in selected for turn in block["turns"]}
    related_context = []
    for lesson in lessons:
        extra = [with_topic(turn) for turn in lesson["turns"] if turn.get("role") == "assistant"
                 and turn.get("anchor_source_id") in closure and turn["turn_id"] not in included]
        if extra:
            related_context.append({"lesson_id": lesson["lesson_id"], "observed_on": lesson["observed_on"],
                                    "topic_ids": list(dict.fromkeys(turn["topic_id"] for turn in extra)),
                                    "turns": extra})
    skipped = found.get("skipped", [])
    if not isinstance(skipped, list):
        raise ValueError("invalid skipped source fragments")
    if skipped and not closure:
        if any(item.get("reason") == "memory_budget_or_context_exceeded" for item in skipped):
            raise PersonalGraphSelectionBudgetExceeded(
                "Найденные исходные фрагменты не поместились в бюджет памяти. Уточните запрос и проверьте источники.")
        raise PersonalGraphSelectionBudgetExceeded(
            "Не найдено доступных оригинальных реплик, подтверждающих контекст. Проверьте источники.")
    budget_skipped = sum(item.get("reason") == "memory_budget_or_context_exceeded" for item in skipped)
    graph_only_skipped = sum(item.get("reason") == "no_exact_source_chunk_match" for item in skipped)
    hydration_skipped = sum(item.get("reason") == "source_fragment_not_hydrated" for item in skipped)
    selection_warnings = []
    if budget_skipped:
        selection_warnings.append({"code": "memory_budget_exceeded", "count": budget_skipped})
    if graph_only_skipped:
        selection_warnings.append({"code": "withheld_without_exact_source_chunk", "count": graph_only_skipped})
    if hydration_skipped:
        selection_warnings.append({"code": "retrieved_source_fragment_not_hydrated", "count": hydration_skipped})
    notice = SOURCE_CONTEXT_NOTICE
    if skipped:
        notice += SOURCE_CONTEXT_SKIPPED_NOTICE
    payload = {"notice": notice, "lessons": selected, "related_corrections": linked}
    if related_context:
        payload["related_context"] = related_context
    if selection_warnings:
        payload["selection_warnings"] = selection_warnings
    text = json.dumps(payload,
                      ensure_ascii=False, separators=(",", ":")) if selected or linked else ""
    if len(text.encode("utf-8")) > memory_byte_budget:
        raise PersonalGraphSelectionBudgetExceeded(
            "Полный контекст источников и исправлений не поместился в бюджет памяти. Проверьте исходные реплики.")
    projected = [sources[source_id] for source_id in sources if source_id in closure]
    from .memory_service import _dialogue_contexts, _session_end_contexts
    dialogue = _dialogue_contexts(snapshot.get("sources", []))
    endings = _session_end_contexts(snapshot.get("sources", []))
    dialogue_context = [dialogue[source_id] for source_id in sources if source_id in closure and source_id in dialogue]
    session_end_context = [endings[source_id] for source_id in sources if source_id in closure
                           and source_id in endings]
    return {"text": text, "sources": projected, "evidence_ids": sorted(closure),
            "method": METHOD, "topic_id": topic_id, "stale": False,
            "dialogue_context": dialogue_context, "session_end_context": session_end_context,
            "selection_warnings": selection_warnings,
            "selection_skips": [{key: item[key] for key in ("lesson_id", "fragment_id", "reason") if key in item}
                                for item in skipped],
            "selection_version": APP_SELECTOR_VERSION,
            "selection_budget": {"kind": "utf8_bytes_upper_bound", "limit": memory_byte_budget,
                                 "used": len(text.encode("utf-8"))},
            "unresolved_count": len(found.get("unresolved", []))}


def approved_graph_version(snapshot, learner_id, topic_id):
    if snapshot.get("learner_id") != learner_id:
        raise ValueError("memory snapshot belongs to another learner")
    try:
        _, current_sources = source_lessons(snapshot, learner_id)
    except (TypeError, ValueError):
        return None
    for version in snapshot.get("versions", []):
        state = version.get("state") or {}
        if ((version.get("method") or state.get("method")) != METHOD or not version.get("approved")
                or version.get("topic_id") != topic_id):
            continue
        from .memory_store import reviewed_snapshot_context
        if reviewed_snapshot_context(snapshot,version,learner_id,topic_id) is not None:
            return version
        if (version.get("stale") or version.get("base_revision") != snapshot.get("revision")
                or state.get("source_revision") != version.get("source_revision")
                or state.get("learner_id") != learner_id or not state.get("archive_run_id")
                or state.get("source_context_protocol") != PERSONAL_GRAPH_CONTEXT_PROTOCOL):
            return None
        source_ids = state.get("source_ids")
        current_ids = [str(row.get("source_id") or "") for row in snapshot.get("sources", [])]
        if (not isinstance(source_ids, list) or not all(isinstance(x, str) and x for x in source_ids)
                or len(source_ids) != len(set(source_ids)) or set(source_ids) != set(current_ids)):
            return None
        ids = (version.get("context") or {}).get("evidence_ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            return None
        evidence_ids = set(ids)
        if (not evidence_ids <= set(current_sources)
                or not correction_closure(current_sources, evidence_ids) <= evidence_ids):
            return None
        source_access = (version.get("context") or {}).get("source_access")
        if source_access is not None:
            if not isinstance(source_access, dict):
                return None
            access_mode = source_access.get("mode")
            access_ids = source_access.get("source_ids")
            if (not isinstance(access_ids, list) or not all(isinstance(x, str) and x for x in access_ids)
                    or len(access_ids) != len(set(access_ids))):
                return None
            if access_mode == "full_archive":
                fingerprint = state.get("archive_snapshot_fingerprint")
                if (not isinstance(fingerprint, str) or source_access.get("snapshot_fingerprint") != fingerprint
                        or personal_graph_snapshot_fingerprint(snapshot.get("sources", [])) != fingerprint
                        or set(access_ids) != set(source_ids)):
                    return None
            elif access_mode == "selected_context":
                if set(access_ids) != set(ids) or not set(access_ids) <= set(source_ids):
                    return None
            else:
                return None
        return version
    return None


class ApplicationModel:
    """Use the application's scheduler and durable response journal for LLM writes."""
    def __init__(self, runtime, model_call=None):
        settings = runtime.settings
        if runtime.context_tokens not in (16384, 32768):
            raise ValueError("personal graph requires a 16k or 32k model context")
        if settings.chat_model != "gemma4:12b" or settings.embedding_model.split(":")[0] != "bge-m3":
            raise ValueError("personal graph requires local Gemma and BGE-M3")
        endpoint = urlsplit(settings.ollama_url)
        if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("personal graph only permits a local Ollama endpoint")
        self.runtime, self.model_call = runtime, model_call
        self.context, self.model, self.base_url = runtime.context_tokens, settings.chat_model, settings.ollama_url
        self.lock = threading.Lock()
        self.model_identities = None

    def check_model_identities(self):
        if self.model_identities is not None:
            from .personal_graph_index import resolve_model_identities
            if resolve_model_identities(self) != self.model_identities:
                raise ValueError("Local model changed during personal graph operation")

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        self.check_model_identities()
        vectors = self.runtime.embed(texts)
        self.check_model_identities()
        if len(vectors) != len(texts) or any(len(v) != 1024 for v in vectors):
            raise ValueError("BGE-M3 returned an invalid vector")
        if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
            raise ValueError("BGE-M3 returned a non-finite vector")
        if any(not 0 < math.hypot(*vector) < math.inf for vector in vectors):
            raise ValueError("BGE-M3 returned a zero or invalid vector norm")
        return vectors

    def count(self, text):
        # Conservative application-only upper bound, not an exact Gemma count.
        return len(text.encode("utf-8"))

    def _post(self, route, payload, purpose):
        if route != "/api/chat" or purpose not in {"graphiti_llm", "personal_filter"} or self.model_call is None:
            raise RuntimeError("this application operation cannot call the graph writer")
        with self.lock:
            self.check_model_identities()
            content = self.model_call(payload["messages"], "personal_graph_extract", payload.get("format"))
            self.check_model_identities()
        return {"done": True, "done_reason": "stop", "message": {"content": content}}


class _ChannelModel:
    """Search-only model shim: exposes what make_graphiti and the adapter read."""

    def __init__(self, source_model):
        self._source = source_model
        self.model, self.context = source_model.model, source_model.context
        self.base_url = source_model.base_url
        self.model_identities = source_model.model_identities

    def check_model_identities(self):
        self._source.check_model_identities()

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        return self._source.embed_many(texts)

    def count(self, text):
        return len(text.encode("utf-8"))

    def _post(self, route, payload, purpose):
        raise RuntimeError("the unified memory channel never calls the graph writer")


async def graph_channel_source_ids(store, runtime, model, learner_id, question, allowed,
                                   lessons, snapshot_fingerprint):
    """Best-effort graph channel for the unified personal_memory hybrid retrieval.

    Consumes an already-built, ready graph index for this exact learner scope;
    never builds, claims or invalidates one. Returns
    ``{"source_ids": [...]|None, "run_id": ..., "reason": str|None}`` — any
    failure degrades to plain hybrid retrieval with a recorded reason. Graph
    facts stay navigation only: the returned source IDs point at canonical
    teacher/student replies.
    """
    settings = getattr(runtime, "settings", None)
    if not getattr(settings, "memory_graph_channel", False):
        return {"source_ids": None, "run_id": None, "reason": "channel_disabled"}
    dsn = getattr(store, "dsn", None)
    if not dsn:
        return {"source_ids": None, "run_id": None, "reason": "no_database_dsn"}
    from psycopg2.extensions import parse_dsn
    params = parse_dsn(dsn)
    addresses = [params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR")]
    if not any(addresses) or any(value and value not in {"127.0.0.1", "localhost", "::1"} for value in addresses):
        return {"source_ids": None, "run_id": None, "reason": "non_loopback_database"}
    path = Path(settings.personal_graph_password_file)
    if not path.is_file():
        return {"source_ids": None, "run_id": None, "reason": "password_file_unavailable"}
    if not isinstance(allowed, set) or not allowed:
        return {"source_ids": None, "run_id": None, "reason": "no_approved_sources"}
    if not isinstance(lessons, list) or not lessons:
        return {"source_ids": None, "run_id": None, "reason": "no_lessons"}
    from .personal_graph_index import (find_ready_personal_graph_index,
                                       personal_graph_config_fingerprint)
    try:
        config_fingerprint = personal_graph_config_fingerprint(model, model.model_identities)
    except Exception as exc:
        return {"source_ids": None, "run_id": None, "reason": f"config_fingerprint:{type(exc).__name__}"}
    try:
        index = await asyncio.to_thread(find_ready_personal_graph_index, dsn, learner_id,
                                        snapshot_fingerprint, lessons, config_fingerprint)
    except Exception as exc:
        return {"source_ids": None, "run_id": None, "reason": f"registry_lookup:{type(exc).__name__}"}
    if index is None:
        return {"source_ids": None, "run_id": None, "reason": "no_ready_index"}
    run_id = index["run_id"]
    from local_app.memory_ported.personal_memory_lab.archive import Archive
    from local_app.memory_ported.personal_memory_lab.backends import GraphitiAdapter, make_graphiti
    archive = None
    graph = None
    try:
        password = await asyncio.to_thread(path.read_text, encoding="utf-8")
        archive = Archive(dsn)
        graph = make_graphiti(_ChannelModel(model), password.strip())
        adapter = GraphitiAdapter(graph, archive, None)
        await asyncio.to_thread(archive.initialize)
        await adapter.initialize()
        hits = await adapter.search(run_id, learner_id, question)
        lessons_by_id = {lesson["lesson_id"]: lesson for lesson in lessons}
        ordered, seen = [], set()
        for hit in hits:
            for lesson_id in hit.get("lesson_ids") or []:
                lesson = lessons_by_id.get(lesson_id)
                if not lesson:
                    continue
                for turn in lesson.get("turns", []):
                    source_id = turn.get("turn_id")
                    if (turn.get("role") in {"teacher", "student"} and source_id in allowed
                            and source_id not in seen):
                        seen.add(source_id)
                        ordered.append(source_id)
        if not ordered:
            return {"source_ids": None, "run_id": run_id, "reason": "no_allowed_hits"}
        return {"source_ids": ordered, "run_id": run_id, "reason": None}
    except Exception as exc:
        return {"source_ids": None, "run_id": run_id, "reason": f"graph_channel_error:{type(exc).__name__}"}
    finally:
        if graph is not None:
            try:
                await graph.close()
            except Exception:
                pass


class PersonalGraphProvider:
    def __init__(self, store, runtime, *, model_call=None):
        self.store, self.runtime, self.model_call = store, runtime, model_call

    def _application_model(self):
        from local_app.memory_ported.personal_memory_lab.runtime import LocalModel
        model = ApplicationModel(self.runtime, self.model_call)
        # Refuse to use a server that silently ignores no-truncation flags.
        # The probe contains generated synthetic text only.
        LocalModel(base_url=model.base_url, model=model.model, context=model.context).verify_no_truncation()
        from .personal_graph_index import resolve_model_identities
        model.model_identities = resolve_model_identities(model)
        return model

    def _source_embedder(self, model, learner_id):
        """Cache only source texts; authorization is applied before this callback."""
        if not getattr(self.store, "dsn", None):
            return model.embed_many
        from .personal_graph_index import resolve_model_identities
        from .personal_graph_retrieval import SourceEmbeddingCache
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        recipe = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in (
            "local_app/runtime.py", "local_app/personal_graph_memory.py", "local_app/personal_graph_retrieval.py")}
        model.check_model_identities()
        identity = {"model": model.model_identities["embedding"], "recipe": recipe}
        cache = SourceEmbeddingCache(self.store.dsn, learner_id,
                                     json.dumps(identity, sort_keys=True, separators=(",", ":")))
        def cached_sources(texts):
            model.check_model_identities()
            result = cache.embed_many(texts, model.embed_many)
            model.check_model_identities()
            return result
        return cached_sources

    def _objects(self, *, source_order=()):
        from pathlib import Path
        from psycopg2.extensions import parse_dsn
        params = parse_dsn(self.store.dsn)
        addresses = [params.get("host"), params.get("hostaddr"), os.getenv("PGHOSTADDR")]
        if not any(addresses) or any(value and value not in {"127.0.0.1", "localhost", "::1"} for value in addresses):
            raise ValueError("personal graph requires a local PostgreSQL database")
        model = self._application_model()
        from local_app.memory_ported.personal_memory_lab.archive import Archive
        from local_app.memory_ported.personal_memory_lab.backends import GraphitiAdapter, make_graphiti
        from local_app.memory_ported.personal_memory_lab.workflow import PersonalMemoryWorkflow
        path = Path(self.runtime.settings.personal_graph_password_file)
        password = path.read_text(encoding="utf-8").strip()
        if not password:
            raise ValueError("personal graph password is unavailable")
        archive = Archive(self.store.dsn)
        graph = make_graphiti(model, password)
        adapter = GraphitiAdapter(graph, archive, model)
        workflow = application_memory_workflow(PersonalMemoryWorkflow, archive, model, adapter,
            full_lesson_max_bytes=1800, source_order=source_order,
            retrieval_mode=getattr(self.runtime.settings, "personal_graph_retrieval_mode", "graph_hybrid"))
        return model, archive, graph, adapter, workflow

    async def _prepare(self, job, snapshot, question):
        from .personal_graph_index import ensure_personal_graph_index, personal_graph_config_fingerprint
        from .memory_store import get_learner_memory_snapshot
        learner_id = job["learner_id"]
        if snapshot.get("source_revision") != job.get("source_revision"):
            raise ValueError("source snapshot changed before personal graph indexing")
        graph_sources = snapshot.get("personal_graph_sources")
        if not isinstance(graph_sources, list):
            raise ValueError("personal_graph_source_snapshot_unavailable")
        graph_snapshot = {**snapshot, "sources": graph_sources}
        lessons, source_map = source_lessons(graph_snapshot, learner_id)
        snapshot_fingerprint = personal_graph_snapshot_fingerprint(graph_sources)
        model, archive, graph, adapter, workflow = self._objects(
            source_order=application_lesson_order(lessons))
        try:
            await asyncio.to_thread(archive.initialize)
            await adapter.initialize()
            def current_snapshot():
                current = get_learner_memory_snapshot(self.store, learner_id, None,
                                                      include_personal_graph_snapshot=True)
                return personal_graph_snapshot_fingerprint(current["personal_graph_sources"])

            config_fingerprint = await asyncio.to_thread(personal_graph_config_fingerprint,
                                                         model, model.model_identities)
            index = await ensure_personal_graph_index(dsn=self.store.dsn, learner_id=learner_id,
                job_id=job["job_id"], snapshot_fingerprint=snapshot_fingerprint, lessons=lessons,
                runtime=model, archive=archive, graph=graph, adapter=adapter,
                config_fingerprint=config_fingerprint, current_snapshot=current_snapshot)
            run_id = index["run_id"]
            selection_budget, memory_budget = personal_graph_budgets(self.runtime.context_tokens)
            found = await workflow.search(run_id, learner_id, question, mode="graphiti",
                                          budget_tokens=selection_budget)
            context = select_context(found, graph_snapshot, learner_id, job.get("topic_id") or "general",
                                     memory_byte_budget=memory_budget)
            state = {"method": METHOD, "topic_id": job.get("topic_id") or "general", "learner_id": learner_id,
                     "archive_run_id": run_id, "source_revision": snapshot["source_revision"],
                     "source_ids": sorted(source_map), "source_context_protocol": PERSONAL_GRAPH_CONTEXT_PROTOCOL,
                     "archive_snapshot_fingerprint": snapshot_fingerprint,
                     "index_config_fingerprint": config_fingerprint, "index_reused": index["reused"],
                     "index_manifest": index["manifest"],
                     "retrieval_mode": getattr(self.runtime.settings, "personal_graph_retrieval_mode", "graph_hybrid"),
                     "selection_version": APP_SELECTOR_VERSION}
            return state, context
        finally:
            await graph.close()

    def prepare(self, job, snapshot, question):
        return asyncio.run(self._prepare(job, snapshot, question))

    async def _chat(self, snapshot, learner_id, topic_id, question, version):
        state = version["state"]
        version_context = version.get("context") or {}
        source_access = version_context.get("source_access") or {}
        full_archive = source_access.get("mode") == "full_archive"
        allowed = set(source_access.get("source_ids") if full_archive
                       else version_context.get("evidence_ids") or [])
        selection_budget, memory_budget = personal_graph_budgets(self.runtime.context_tokens)
        if full_archive:
            from .personal_graph_index import PersonalGraphIndexStale, validate_ready_personal_graph_index
            lessons, _ = source_lessons(snapshot, learner_id)
            snapshot_fingerprint = personal_graph_snapshot_fingerprint(snapshot.get("sources", []))
            if (state.get("archive_snapshot_fingerprint") != snapshot_fingerprint
                    or source_access.get("snapshot_fingerprint") != snapshot_fingerprint
                    or not state.get("index_config_fingerprint") or not state.get("index_manifest")):
                raise PersonalGraphIndexStale()
            model, archive, graph, _, workflow = self._objects(
                source_order=application_lesson_order(lessons))
            try:
                await validate_ready_personal_graph_index(dsn=self.store.dsn, run_id=state.get("archive_run_id"),
                    learner_id=learner_id, snapshot_fingerprint=snapshot_fingerprint, lessons=lessons,
                    runtime=model, archive=archive, graph=graph,
                    approved_config_fingerprint=state.get("index_config_fingerprint"),
                    approved_manifest=state.get("index_manifest"))
                try:
                    found = await workflow.search(state["archive_run_id"], learner_id, question, mode="graphiti",
                                                  budget_tokens=selection_budget, allowed_source_ids=allowed)
                except ValueError:
                    try:
                        await asyncio.to_thread(model.check_model_identities)
                    except ValueError as exc:
                        raise PersonalGraphIndexStale() from exc
                    raise
                try:
                    await asyncio.to_thread(model.check_model_identities)
                except ValueError as exc:
                    raise PersonalGraphIndexStale() from exc
            finally:
                await graph.close()
            context = select_context(found, snapshot, learner_id, topic_id, allowed_ids=allowed,
                                     memory_byte_budget=memory_budget)
        else:
            _, current_sources = source_lessons(snapshot, learner_id)
            if not allowed <= set(current_sources):
                raise ValueError("approved sources no longer belong to this learner")
            if not correction_closure(current_sources, allowed) <= allowed:
                raise ValueError("source correction requires a new approved memory version")
            scoped_sources = [row for row in snapshot.get("sources", [])
                              if str(row.get("source_id") or "") in allowed]
            if {str(row.get("source_id") or "") for row in scoped_sources} != allowed:
                raise ValueError("approved sources no longer belong to this learner")
            scoped_snapshot = {**snapshot, "sources": scoped_sources}
            lessons, _ = source_lessons(scoped_snapshot, learner_id)
            model = self._application_model()
            source_embed_many = await asyncio.to_thread(self._source_embedder, model, learner_id)
            selected_mode = getattr(getattr(self.runtime, "settings", None),
                                    "personal_graph_retrieval_mode", "graph_hybrid")
            # The default graph method retains source-only hybrid retrieval for
            # selected_context; a broad grant is required to query the graph.
            if selected_mode == "graph_hybrid":
                selected_mode = "hybrid"
            ranked = await asyncio.to_thread(source_scoped_search, question, lessons, allowed, model.embed_many,
                limit=max(1, len(allowed)), include_memory=False, source_embed_many=source_embed_many,
                mode=selected_mode)
            ranked_ids = ranked.get("ranked_source_ids", [])
            ranked_groups, groups_by_ids = [], {}
            for source_id in ranked_ids:
                source_group = frozenset(correction_closure(current_sources, {source_id}))
                group = groups_by_ids.get(source_group)
                if group is None:
                    group = {"source_ids": set(source_group), "representatives": []}
                    groups_by_ids[source_group] = group
                    ranked_groups.append(group)
                group["representatives"].append(source_id)

            accepted_groups, budget_skipped_ids = [], set()

            def context_for_groups(groups, skipped_ids):
                found = source_scoped_found(lessons, [group["selected"] for group in groups],
                                            skipped_source_ids=skipped_ids)
                return select_context(found, scoped_snapshot, learner_id, topic_id,
                                      allowed_ids=allowed, memory_byte_budget=memory_budget)

            def fit_accepted_groups():
                while accepted_groups:
                    try:
                        return context_for_groups(accepted_groups, budget_skipped_ids)
                    except PersonalGraphSelectionBudgetExceeded:
                        replacement_found = False
                        for group in reversed(accepted_groups):
                            previous = group["selected"]
                            for source_id in group["representatives"]:
                                if source_id == previous:
                                    continue
                                group["selected"] = source_id
                                try:
                                    replacement_context = context_for_groups(accepted_groups,
                                                                             budget_skipped_ids)
                                except PersonalGraphSelectionBudgetExceeded:
                                    continue
                                replacement_found = True
                                break
                            if replacement_found:
                                return replacement_context
                            group["selected"] = previous
                        dropped_group = accepted_groups.pop()
                        budget_skipped_ids.update(dropped_group["source_ids"])
                return None

            for group in ranked_groups:
                if group["source_ids"] & budget_skipped_ids:
                    continue
                accepted = False
                for source_id in group["representatives"]:
                    trial_group = {**group, "selected": source_id}
                    try:
                        trial_context = context_for_groups(accepted_groups + [trial_group],
                                                           budget_skipped_ids)
                    except PersonalGraphSelectionBudgetExceeded:
                        continue
                    if trial_context["evidence_ids"]:
                        accepted_groups.append(trial_group)
                        accepted = True
                        break
                if not accepted:
                    budget_skipped_ids.update(group["source_ids"])
                    fit_accepted_groups()
            context = context_for_groups(accepted_groups, budget_skipped_ids)
        context["version_id"] = version.get("version_id") or version.get("id")
        return context

    def chat_context(self, snapshot, learner_id, topic_id, question):
        graph_sources = snapshot.get("personal_graph_sources")
        if not isinstance(graph_sources, list):
            raise ValueError("personal_graph_source_snapshot_unavailable")
        graph_snapshot = {**snapshot, "sources": graph_sources}
        version = approved_graph_version(graph_snapshot, learner_id, topic_id)
        if version is None:
            return {"text": "", "sources": [], "evidence_ids": [], "stale": True,
                    "version_id": None, "topic_id": topic_id, "withheld_reason": "personal_graph_not_approved_or_stale"}
        from .memory_store import reviewed_snapshot_context
        reviewed = reviewed_snapshot_context(graph_snapshot,version,learner_id,topic_id)
        if reviewed is not None:
            return reviewed
        return asyncio.run(self._chat(graph_snapshot, learner_id, topic_id, question, version))
