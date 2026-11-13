"""Exact compatibility with the original selector, including byte boundaries."""
import random

import pytest

from local_app import personal_memory
from local_app.personal_graph_memory import (PersonalGraphSelectionBudgetExceeded, correction_closure,
                                             select_context, source_lessons, source_scoped_found)
from local_app.personal_memory import select_ranked_source_context
from local_app.personal_memory_contract import PERSONAL_MEMORY_SELECTION_VERSION


def original_selector(ranked_ids, snapshot, learner_id, topic_id, allowed_ids, memory_byte_budget):
    """Frozen pre-optimization behavior used only as a compatibility oracle."""
    lessons, sources = source_lessons(snapshot, learner_id)
    allowed = set(allowed_ids)
    groups, by_group = [], {}
    for source_id in ranked_ids:
        key = frozenset(correction_closure(sources, {source_id}))
        if key not in by_group:
            by_group[key] = []
            groups.append((key, by_group[key]))
        if source_id not in by_group[key]:
            by_group[key].append(source_id)
    candidates = set().union(*(key for key, _ in groups)) if groups else set()
    selected, selected_ids = [], set()

    def build(representatives, visible_ids):
        found = source_scoped_found(lessons, representatives, skipped_source_ids=candidates - visible_ids)
        context = select_context(found, snapshot, learner_id, topic_id,
                                 allowed_ids=allowed, memory_byte_budget=memory_byte_budget)
        context.update(method="personal_memory", selection_version=PERSONAL_MEMORY_SELECTION_VERSION)
        return context

    for group_ids, representatives in groups:
        for source_id in representatives:
            try:
                build(selected + [source_id], selected_ids | group_ids)
            except PersonalGraphSelectionBudgetExceeded:
                continue
            selected.append(source_id)
            selected_ids.update(group_ids)
            break
    return build(selected, selected_ids)


def synthetic_snapshot(seed):
    rng = random.Random(seed)
    sources = []
    for index in range(18):
        role = "student" if index % 3 else "teacher"
        source = {"source_id": f"s{index}", "author_role": role, "approved": role == "teacher",
                  "text": f'Наблюдение {index}: "a\\b"\n🙂 ' * rng.randint(1, 9),
                  "topic_id": "math" if index % 2 else "reading", "session_id": f"lesson{index // 4}",
                  "observed_on": f"2026-11-{1 + (index // 4) % 3:02d}"}
        if index >= 3 and rng.random() < 0.4:
            source.update(author_role="teacher", approved=True, corrects_source_id=f"s{rng.randrange(index)}")
        if source["author_role"] == "student":
            source["dialogue_context"] = {"role": "assistant", "text": "Почему? " * rng.randint(1, 100)}
            if index % 2:
                source["session_end_context"] = {"role": "assistant", "usage": "context_only",
                    "text": "Продолжим завтра?", "session_id": source["session_id"],
                    "assistant_source_id": f"end{index}", "excerpt_mode": "whole", "truncated": False}
        sources.append(source)
    return {"learner_id": "synthetic", "sources": sources}


def outcome(selector, ranked, snapshot, budget, allowed=None):
    try:
        return selector(ranked, snapshot, "synthetic", "math",
                        allowed if allowed is not None else {source["source_id"] for source in snapshot["sources"]}, budget)
    except PersonalGraphSelectionBudgetExceeded as exc:
        return (type(exc), str(exc))


@pytest.mark.parametrize("seed", range(8))
def test_exact_original_selection_with_corrections_contexts_and_utf8_boundaries(seed):
    value = synthetic_snapshot(seed)
    ranked = [source["source_id"] for source in value["sources"]]
    random.Random(seed).shuffle(ranked)
    # Include duplicate ranks and a correction group without all of its members
    # directly retrieved. The closure, assistant context and warnings still fit.
    ranked = ranked[:12] + ranked[:3]
    roomy = outcome(original_selector, ranked, value, 30000)
    threshold = roomy["selection_budget"]["used"]
    for budget in (0, 900, 1600, 2500, 4000, 6000, 12000, threshold - 1, threshold, threshold + 1):
        assert outcome(select_ranked_source_context, ranked, value, budget) == outcome(
            original_selector, ranked, value, budget), (seed, budget)


def test_exact_selection_for_a_restricted_correction_closed_grant():
    value = synthetic_snapshot(19)
    _, sources = source_lessons(value, "synthetic")
    allowed = correction_closure(sources, {"s1", "s9"})
    ranked = list(reversed(sorted(allowed)))
    for budget in (1000, 3000, 6000, 12000):
        assert outcome(select_ranked_source_context, ranked, value, budget, allowed) == outcome(
            original_selector, ranked, value, budget, allowed)


def test_selector_renders_and_validates_canonical_context_once(monkeypatch):
    value = synthetic_snapshot(6)
    ranked = [source["source_id"] for source in value["sources"]]
    render = personal_memory.select_context
    calls = []

    def checked_render(*args, **kwargs):
        calls.append(True)
        return render(*args, **kwargs)

    monkeypatch.setattr(personal_memory, "select_context", checked_render)
    result = outcome(select_ranked_source_context, ranked, value, 4000)
    assert isinstance(result, dict) and len(calls) == 1


def test_all_oversized_groups_keep_the_original_explicit_budget_error():
    value = synthetic_snapshot(3)
    for source in value["sources"]:
        source["text"] *= 100
    ranked = [source["source_id"] for source in value["sources"]]
    expected = outcome(original_selector, ranked, value, 2000)
    assert isinstance(expected, tuple)
    assert outcome(select_ranked_source_context, ranked, value, 2000) == expected
    empty = {"learner_id": "synthetic", "sources": []}
    assert outcome(select_ranked_source_context, [], empty, 3000) == outcome(original_selector, [], empty, 3000)
