"""LangGraph retrieval, source hydration and a measured shared memory budget."""
from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .runtime import ContextWindowOverflow, UnsafeMemoryProbe


class State(TypedDict, total=False):
    run_id: str
    learner_id: str
    question: str
    mode: str
    budget_tokens: int
    hits: list[dict]
    unresolved: list[dict]
    lesson_ids: list[str]
    lessons: list[dict]
    memory: str
    memory_tokens: int
    source_ids: list[str]
    skipped: list[dict]
    allowed_source_ids: list[str]


class PersonalMemoryWorkflow:
    def __init__(self, archive, runtime, *, graphiti=None, mem0=None, include_candidates=True,
                 full_lesson_max_bytes=6000):
        self.archive, self.runtime, self.graphiti, self.mem0 = archive, runtime, graphiti, mem0
        self.include_candidates = include_candidates
        self.full_lesson_max_bytes = full_lesson_max_bytes
        graph = StateGraph(State)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("hydrate", self.hydrate)
        graph.add_node("select", self.select)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "hydrate")
        graph.add_edge("hydrate", "select")
        graph.add_edge("select", END)
        self.graph = graph.compile()

    async def retrieve(self, state: State):
        mode = state["mode"]
        if mode not in {"mem0", "graphiti", "graphiti_mem0"}:
            raise ValueError("Unknown personal memory condition")
        if mode in {"graphiti", "graphiti_mem0"} and self.graphiti is None:
            raise RuntimeError("Graphiti adapter is unavailable")
        if mode in {"mem0", "graphiti_mem0"} and self.mem0 is None:
            raise RuntimeError("Mem0 adapter is unavailable")
        run, child, question = state["run_id"], state["learner_id"], state["question"]
        hits = []
        vector = None
        allowed = state.get("allowed_source_ids")
        if mode in {"graphiti", "graphiti_mem0"}:
            vector = await asyncio.to_thread(self.runtime.embed, question)
            rows = await asyncio.to_thread(self.archive.search, run, child, question, vector,
                                           source_ids=allowed) if allowed is not None else \
                   await asyncio.to_thread(self.archive.search, run, child, question, vector)
            hits.extend({"kind": row["channel"], "lesson_ids": [row["lesson_id"]],
                         "chunk_id": row["chunk_id"], "score": row["score"]} for row in rows)
            hits.extend(await self.graphiti.search(run, child, question))
        if mode in {"mem0", "graphiti_mem0"}:
            hits.extend(await asyncio.to_thread(self.mem0.search, run, child, question))
        # A derived claim without a source never reaches the answer generator.
        unresolved = [hit for hit in hits if not hit.get("lesson_ids")]
        usable = [hit for hit in hits if hit.get("lesson_ids")]
        ids = list(dict.fromkeys(lesson_id for hit in usable for lesson_id in hit["lesson_ids"]))
        # Source lookup is restricted to lessons found by the chosen memory. This
        # locates supporting fragments for long lessons without giving Mem0 an
        # unrestricted second retrieval system.
        derived_ids = list(dict.fromkeys(lesson_id for hit in usable
            if hit["kind"] in {"graphiti", "mem0"} for lesson_id in hit["lesson_ids"]))
        if derived_ids:
            if vector is None:
                vector = await asyncio.to_thread(self.runtime.embed, question)
            source_rows = await asyncio.to_thread(self.archive.search, run, child, question, vector,
                                                   24, derived_ids, source_ids=allowed) if allowed is not None else \
                          await asyncio.to_thread(self.archive.search, run, child, question, vector, 24, derived_ids)
            usable.extend({"kind": "source_" + row["channel"], "lesson_ids": [row["lesson_id"]],
                           "chunk_id": row["chunk_id"], "score": row["score"]} for row in source_rows)
        if mode in {"graphiti", "graphiti_mem0"}:
            recent = await asyncio.to_thread(self.archive.lessons, run, child)
            recent_ids = [item["lesson_id"] for item in recent[-2:]]
            ids = recent_ids + [item for item in ids if item not in recent_ids]
        return {"hits": usable, "unresolved": unresolved, "lesson_ids": ids}

    async def hydrate(self, state: State):
        lessons = await asyncio.to_thread(self.archive.lessons, state["run_id"], state["learner_id"],
                                          state["lesson_ids"])
        found = {lesson["lesson_id"] for lesson in lessons}
        if found != set(state["lesson_ids"]):
            raise ValueError("A retrieved source does not belong to the requested child and run")
        by_id = {lesson["lesson_id"]: lesson for lesson in lessons}
        fragments = await asyncio.to_thread(self.archive.chunks, state["run_id"], state["learner_id"],
                                            state["lesson_ids"])
        allowed = set(state["allowed_source_ids"]) if "allowed_source_ids" in state else None
        if allowed is not None:
            fragments = [row for row in fragments if any(turn.get("turn_id") in allowed
                         for turn in row["content"].get("turns", []))]
        by_chunk = {row["chunk_id"]: row for row in fragments}
        hydrated = []
        def append_permitted(lesson):
            if allowed is None:
                hydrated.append(lesson)
                return
            evidence = {turn.get("turn_id") for turn in lesson["turns"]
                        if turn.get("role") in {"teacher", "student"} and turn.get("turn_id") in allowed}
            turns = [turn for turn in lesson["turns"] if turn.get("turn_id") in evidence or
                     (turn.get("role") == "assistant" and turn.get("anchor_source_id") in evidence)]
            if evidence:
                hydrated.append({**lesson, "turns": turns})
        for lesson_id in state["lesson_ids"]:
            lesson = by_id[lesson_id]
            if len(json.dumps(lesson, ensure_ascii=False).encode("utf-8")) <= self.full_lesson_max_bytes:
                append_permitted(lesson)
                continue
            ordered = list(dict.fromkeys(hit["chunk_id"] for hit in state["hits"]
                if hit.get("chunk_id") in by_chunk and lesson_id in hit["lesson_ids"]))
            own = [row for row in fragments if row["lesson_id"] == lesson_id]
            if own and not ordered:
                ordered = [own[-1]["chunk_id"]]
            for chunk_id in ordered[:4]:
                row = by_chunk[chunk_id]
                if row["lesson_id"] != lesson_id:
                    raise ValueError("Fragment belongs to a different lesson")
                append_permitted({**row["content"], "fragment_id": chunk_id})
        return {"lessons": hydrated}

    async def select(self, state: State):
        budget = state["budget_tokens"]
        if budget <= 0 or budget > self.runtime.context - 2048:
            raise ValueError("Invalid selected-memory budget for the working window")
        selected, skipped = [], []
        for lesson in state["lessons"]:
            lesson_id = lesson["lesson_id"]
            related = [hit for hit in state["hits"] if self.include_candidates and not lesson.get("fragment_id")
                       and set(hit["lesson_ids"]) == {lesson_id}
                       and hit["kind"] in {"graphiti", "mem0"}]
            selected.append({"derived_candidates": related, "original_lesson": lesson})
        async def measured(prefix):
            if not prefix:
                return 0
            candidate = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
            try:
                return await asyncio.to_thread(self.runtime.count, candidate)
            except (ContextWindowOverflow, UnsafeMemoryProbe):
                return None
        tokens = await measured(selected)
        if tokens is None or tokens > budget:
            if not self.include_candidates:
                # Application budgets are conservative byte counts. Skip an
                # oversized block and keep looking for smaller approved ones.
                kept, skipped = [], []
                tokens = 0
                for block in selected:
                    count = await measured(kept + [block])
                    if count is not None and count <= budget:
                        kept.append(block)
                        tokens = count
                    else:
                        skipped.append({"lesson_id": block["original_lesson"]["lesson_id"],
                                        "fragment_id": block["original_lesson"].get("fragment_id"),
                                        "reason": "memory_budget_or_context_exceeded"})
                selected = kept
            else:
                low, high, best_tokens = 0, len(selected) - 1, 0
                while low < high:
                    middle = (low + high + 1) // 2
                    count = await measured(selected[:middle])
                    if count is not None and count <= budget:
                        low, best_tokens = middle, count
                    else:
                        high = middle - 1
                if low and not best_tokens:
                    best_tokens = await measured(selected[:low]) or 0
                skipped = [{"lesson_id": block["original_lesson"]["lesson_id"],
                            "fragment_id": block["original_lesson"].get("fragment_id"),
                            "reason": "memory_budget_or_context_exceeded"} for block in selected[low:]]
                selected, tokens = selected[:low], best_tokens
        selected_ids = [turn["turn_id"] for block in selected
                        for turn in block["original_lesson"]["turns"]
                        if turn["role"] in {"teacher", "student"}]
        text = json.dumps(selected, ensure_ascii=False, separators=(",", ":")) if selected else ""
        return {"memory": text, "memory_tokens": tokens, "source_ids": selected_ids, "skipped": skipped}

    async def search(self, run_id, learner_id, question, *, mode, budget_tokens=4096, allowed_source_ids=None):
        state = {"run_id": run_id, "learner_id": learner_id,
                 "question": question, "mode": mode, "budget_tokens": budget_tokens}
        if allowed_source_ids is not None:
            state["allowed_source_ids"] = list(allowed_source_ids)
        return await self.graph.ainvoke(state)

    async def answer(self, run_id, learner_id, question, *, mode, budget_tokens=4096, profile=None):
        result = await self.search(run_id, learner_id, question, mode=mode, budget_tokens=budget_tokens)
        result["answer"] = await asyncio.to_thread(self.runtime.answer, question, result["memory"],
                                                    set(result["source_ids"]), profile)
        return result
