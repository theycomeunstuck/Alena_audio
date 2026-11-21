"""Audit F-070: check_arithmetic missed a wrong equality after a sentence
period, a long expression escaped as RecursionError, and a failed RAG search
logged the child's message verbatim."""
import logging

import pytest

from pipeline import rag
from pipeline.task_checks import check_arithmetic


def test_equalities_joined_by_a_sentence_period_are_each_checked():
    problems = check_arithmetic("2 + 2 = 4. 3 + 3 = 7")
    assert len(problems) == 1 and "3 + 3 = 7" in problems[0]


def test_a_correct_pair_after_a_period_raises_nothing():
    assert check_arithmetic("2 + 2 = 4. 3 + 3 = 6.") == []


def test_decimal_point_is_not_a_sentence_break():
    assert check_arithmetic("2.5 + 2.5 = 5") == []


def test_a_very_long_expression_is_skipped_not_crashed():
    expression = "+".join(["1"] * 4000) + " = 4000"
    assert check_arithmetic(expression) == []


def test_failed_rag_search_does_not_log_the_message_text(monkeypatch, caplog):
    secret = "меня зовут Маша, я живу на улице Ленина 5"

    class BrokenModel:
        def encode(self, query):
            raise RuntimeError("embedding service down")

    monkeypatch.setattr(rag, "_get_model", lambda: BrokenModel())
    with caplog.at_level(logging.DEBUG):
        assert rag.search_knowledge(secret) == []
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "Маша" not in logged and "Ленина" not in logged
    assert "RAG search failed" in logged and f"{len(secret)} chars" in logged
