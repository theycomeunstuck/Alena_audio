"""FIX_LIST item 6, full merge: source_scoped_search no longer has its own hybrid ranking.

All modes now go through personal_graph_retrieval.rank_source_candidates. Differences from the old private
hybrid, accepted on purpose: candidates are ordered newest lesson first (not by turn id), equal scores are
broken by recency, tokenising and cosine follow rank_source_candidates (strict 1024-dimension, non-zero vectors).
"""
import inspect

import pytest

from local_app import personal_graph_memory
from local_app.personal_graph_memory import source_scoped_search
from local_app.personal_graph_retrieval import rank_source_candidates


def vector(index):
    values = [0.0] * 1024
    values[index] = 1.0
    return values


class Embedder:
    """Texts that mention a keyword get that keyword's axis; everything else gets axis 9."""

    def __init__(self, keywords):
        self.keywords = keywords

    def embed_many(self, texts):
        out = []
        for text in texts:
            axis = next((i for i, word in enumerate(self.keywords) if word in text.casefold()), 9)
            out.append(vector(axis))
        return out


def lessons():
    return [
        {"lesson_id": "later", "observed_on": "2025-02-01", "topic_ids": ["math"], "turns": [
            {"turn_id": "s2", "role": "student", "text": "algebra fraction"},
            {"turn_id": "a2", "role": "assistant", "anchor_source_id": "s2", "text": "fraction explanation"},
        ]},
        {"lesson_id": "earlier", "observed_on": "2025-01-01", "topic_ids": ["math"], "turns": [
            {"turn_id": "s1", "role": "teacher", "text": "geometry triangle"},
            {"turn_id": "a1", "role": "assistant", "anchor_source_id": "s1", "text": "triangle explanation"},
            {"turn_id": "s3", "role": "student", "text": "geometry circle"},
        ]},
    ]


ALLOWED = {"s1", "s2", "s3"}


def test_hybrid_scoped_search_ranks_exactly_like_rank_source_candidates():
    model = Embedder(["triangle", "fraction"])
    for question in ("geometry triangle", "fraction", "something unrelated"):
        scoped = source_scoped_search(question, lessons(), ALLOWED, model.embed_many, limit=3)
        direct = rank_source_candidates(question, lessons(), ALLOWED, mode="hybrid", embed_many=model.embed_many, limit=3)
        assert scoped["ranked_source_ids"] == direct["ranked_source_ids"], question


def test_result_keeps_its_memory_payload_keys():
    model = Embedder(["triangle"])
    result = source_scoped_search("geometry triangle", lessons(), ALLOWED, model.embed_many, limit=2)
    assert {"memory", "unresolved", "skipped", "ranked_source_ids"} <= set(result)
    assert result["ranked_source_ids"][0] == "s1"


def test_no_private_hybrid_implementation_is_left_in_source_scoped_search():
    source = inspect.getsource(source_scoped_search)
    for leftover in ("def cosine", "semantic_order", "lexical_order", "candidates.sort"):
        assert leftover not in source, leftover
    assert "rank_source_candidates" in source


def test_equal_scores_are_now_broken_by_recency_not_by_turn_id():
    both_same = [
        {"lesson_id": "old", "observed_on": "2025-01-01", "topic_ids": ["m"], "turns": [
            {"turn_id": "a-old", "role": "student", "text": "plain words"}]},
        {"lesson_id": "new", "observed_on": "2025-03-01", "topic_ids": ["m"], "turns": [
            {"turn_id": "z-new", "role": "student", "text": "plain words"}]},
    ]
    model = Embedder(["nothing-matches"])
    result = source_scoped_search("plain words", both_same, {"a-old", "z-new"}, model.embed_many, limit=2)
    assert result["ranked_source_ids"] == ["z-new", "a-old"]   # the newer lesson first (old rule: a-old first by id)


def test_graph_modes_are_still_refused_for_selected_context():
    with pytest.raises(ValueError, match="graph"):
        source_scoped_search("q", lessons(), ALLOWED, Embedder([]).embed_many, mode="graph_hybrid")


def test_vectors_that_are_not_1024_dimensional_are_rejected_now():
    with pytest.raises(ValueError, match="1024"):
        source_scoped_search("geometry", lessons(), ALLOWED, lambda texts: [[1.0, 0.0] for _ in texts])
