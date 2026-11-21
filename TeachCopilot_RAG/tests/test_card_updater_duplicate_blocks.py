"""Audit F-069: when the model returned two blocks for the same learner,
propose_updates kept only the last one – the first block's observations were
dropped silently. Blocks for one learner are now merged."""
import logging

from tests.test_card_updater import LEARNER, TOPIC, _fake_llm, propose_updates


def _block(note, help_level, **extra):
    return {"learner_id": LEARNER, "journal_note": note,
            "observations": [{"topic_id": TOPIC, "help_level": help_level, "solved": True, "note": note}], **extra}


def test_two_blocks_for_one_learner_keep_both_observations(caplog):
    payload = {"learners": [_block("первая попытка", "hint_question"), _block("вторая попытка", "independent")]}
    with caplog.at_level(logging.WARNING):
        proposal = propose_updates("итоги урока", [LEARNER], complete=_fake_llm(payload))[LEARNER]
    assert [item.note for item in proposal.observations] == ["первая попытка", "вторая попытка"]
    assert "первая попытка" in proposal.journal_note and "вторая попытка" in proposal.journal_note
    assert any("duplicate" in record.getMessage().lower() for record in caplog.records)


def test_identical_duplicate_blocks_do_not_double_the_facts():
    block = _block("одна и та же попытка", "hint_question")
    payload = {"learners": [block, dict(block)]}
    proposal = propose_updates("итоги урока", [LEARNER], complete=_fake_llm(payload))[LEARNER]
    assert len(proposal.observations) == 1
    assert proposal.journal_note == "одна и та же попытка"


def test_conflicting_goal_keeps_the_first_and_warns():
    payload = {"learners": [_block("a", "hint_question", new_goal="Освоить луч"),
                            _block("b", "independent", new_goal="Освоить отрезок")]}
    proposal = propose_updates("итоги урока", [LEARNER], complete=_fake_llm(payload))[LEARNER]
    assert proposal.new_goal == "Освоить луч"
    assert any("new_goal" in warning for warning in proposal.warnings)
