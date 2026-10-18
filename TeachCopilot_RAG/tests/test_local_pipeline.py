"""Domain integration checks for database-backed programming contexts."""
import copy
import json
from pathlib import Path

from pipeline.learner_context import load_task_design_context, load_learner_data_tools
from pipeline.card_updater import update_from_notes
from pipeline.task_designer import design_task_card, TaskDesignError
import pytest

ROOT = Path(__file__).resolve().parents[2]


class CardStore:
    def __init__(self):
        self.card = json.loads((ROOT / 'learner-data/learners/ivanov-ivan.json').read_text(encoding='utf-8'))
        self.topics = load_learner_data_tools().load_catalog(ROOT / 'learner-data/catalog')

    def get_card(self, learner_id):
        return copy.deepcopy(self.card)

    def catalog(self):
        return self.topics


def test_programming_context_uses_database_and_has_no_private_identity():
    store = CardStore()
    context = load_task_design_context('ivanov-ivan', 'programming.python.loops', store=store)
    assert context.topic_id == 'programming.python.loops'
    assert context.topic_title == store.topics[context.topic_id]['title_ru']
    for key in ('legal_name', 'points_ledger', 'first_name', 'last_name'):
        assert key not in context.prompt_text


def test_programming_observation_is_not_rejected_as_wrong_school_grade():
    store = CardStore()
    topic = 'programming.python.loops'
    store.card['rag_context']['current_topics'] = [topic]
    def complete(prompt, system):
        return json.dumps({'learners': [{'learner_id': 'ivanov-ivan',
            'observations': [{'topic_id': topic, 'help_level': 'visual', 'solved': True}],
            'new_error_patterns': [{'topic_id': topic, 'error_tag': 'loop boundary'}]}]})
    results = update_from_notes('Practiced Python loops', ['ivanov-ivan'], store=store, complete=complete)
    assert results
    assert any(c['topic_id'] == topic for c in results[0].card['learner_model']['competencies'])
    assert next(e for e in results[0].card['error_patterns'] if e['topic_id'] == topic)['subject'] == 'programming'
    assert topic not in [c['topic_id'] for c in store.card['learner_model']['competencies']]


def test_database_task_generation_refuses_empty_indexed_material():
    class EmptyRag:
        def search(self, *args, **kwargs):
            return []
    with pytest.raises(TaskDesignError, match='No indexed curriculum'):
        design_task_card('ivanov-ivan', store=CardStore(), rag_service=EmptyRag(),
                         complete=lambda *args: (_ for _ in ()).throw(AssertionError('LLM must not run')))
