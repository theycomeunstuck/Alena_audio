import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.learner_context import LearnerContextError, load_learner_context  # noqa: E402


def test_compact_context_loads_without_full_card_data():
    context = load_learner_context("volk-08")

    assert context.learner_id == "volk-08"
    assert "math.g4.numbers.division_by_1_2_digit" in context.topic_ids
    assert "<learner_rag_context>" in context.prompt_text
    assert "learner_model" not in context.prompt_text
    assert "points_ledger" not in context.prompt_text
    assert "interests" not in context.prompt_text


def test_invalid_learner_id_is_rejected_before_file_lookup():
    try:
        load_learner_context("../volk-08")
    except LearnerContextError as exc:
        assert "learner_id" in str(exc)
    else:
        raise AssertionError("invalid learner id was accepted")
