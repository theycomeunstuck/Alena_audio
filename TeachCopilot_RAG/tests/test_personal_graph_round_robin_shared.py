"""FIX_LIST item 6, the behaviour-preserving part: both approved-memory hybrids use one round-robin helper.

source_scoped_search (personal_graph_memory) used to carry its own copy of the interleave loop. The loop is
identical in effect to personal_graph_retrieval._round_robin, so the copy is replaced by a call. The rest of
the two hybrids (candidate order, tokenising, cosine) differs on purpose and is pinned by other tests.
"""
import inspect
import random

from local_app import personal_graph_memory
from local_app.personal_graph_retrieval import _round_robin


def reference_old_loop(semantic_order, lexical_order, limit):
    """The loop that used to live in source_scoped_search, copied verbatim in behaviour."""
    selected_indexes = []
    for rank in range(max(len(semantic_order), len(lexical_order))):
        for order in (semantic_order, lexical_order):
            if rank < len(order) and order[rank] not in selected_indexes:
                selected_indexes.append(order[rank])
                if len(selected_indexes) == limit:
                    break
        if len(selected_indexes) == limit:
            break
    return selected_indexes


def test_shared_round_robin_matches_the_old_private_loop_on_random_inputs():
    rng = random.Random(20261112)
    for _ in range(2000):
        size = rng.randint(0, 12)
        semantic = rng.sample(range(size), rng.randint(0, size)) if size else []
        lexical = rng.sample(range(size), rng.randint(0, size)) if size else []
        limit = rng.randint(1, 14)
        assert _round_robin([semantic, lexical], limit) == reference_old_loop(semantic, lexical, limit), (semantic, lexical, limit)


def test_edge_cases_agree():
    for semantic, lexical, limit in [([], [], 3), ([0, 1, 2], [], 2), ([], [3, 1], 5), ([0, 1], [1, 0], 10), ([2, 0], [0, 2], 1)]:
        assert _round_robin([semantic, lexical], limit) == reference_old_loop(semantic, lexical, limit)


def test_source_scoped_search_has_no_interleave_loop_of_its_own_any_more():
    # Superseded by the full merge (test_source_scoped_search_unified.py): it delegates every mode to
    # rank_source_candidates, which owns the only round-robin.
    source = inspect.getsource(personal_graph_memory.source_scoped_search)
    assert "selected_indexes.append" not in source
    assert "rank_source_candidates" in source
