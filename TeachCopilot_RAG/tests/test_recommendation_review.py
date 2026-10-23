"""Literal review notes cannot turn generated interpretation into checked fact."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from local_app.memory_service import MemoryWorker
from local_app.recommendation_review import recommendation_source_review
from tests.test_memory_service import ACTION, ApiStore, Rag, Runtime, WorkerStore, client


def source(source_id, text, role="student", **extra):
    return {"source_id": source_id, "text": text, "role": role, "observed_on": "2026-10-23", **extra}


def review(sources, cited=(), **extra):
    context = {"sources": sources, "evidence_ids": [row["source_id"] for row in sources], **extra}
    action = {**ACTION, "evidence_turn_ids": list(cited), "unknowns": []}
    before = deepcopy((action, context))
    result = recommendation_source_review(action, context)
    assert (action, context) == before
    assert result["meaning_verified"] is False
    return result


def test_worker_exposes_explicit_unknowns_that_are_missing_from_the_model_output():
    store = WorkerStore()
    store.snapshot["sources"] = [
        source("unknown-conditions", "Условия помощи неизвестны; самостоятельное решение и освоение дробей не подтверждены.", "teacher"),
        source("pause", "Приостановить сохранённый пример."),
    ]
    action = {**ACTION, "action": "continue_work", "reason": "Вернуться к сохранённому примеру.",
              "unknowns": [], "evidence_turn_ids": ["pause"]}
    runtime = Runtime(action)
    result = MemoryWorker(store, runtime).run_once()
    assert not result.get("error")
    saved = result["next_step"]
    assert saved["unknowns"] == [] and saved["reason"] == action["reason"]
    notes = saved["evidence_review"]["source_unknowns"]
    assert {row["condition"] for row in notes} == {"support", "independence"}
    assert all(row["source_id"] == "unknown-conditions" and row["cited"] is False for row in notes)
    assert all(row["quote"] in store.snapshot["sources"][0]["text"] for row in notes)
    assert saved["evidence_review"]["unknowns_empty"] is True
    assert len(runtime.calls) == 1
    assert json.loads(result["result"]["calls"][0]["raw_response"]["message"]["content"]) == action


def test_short_answer_and_its_question_keep_their_distinct_roles():
    text = "4? Это число всех равных частей?"
    question = "Сколько всего равных частей в круге?"
    result = review([source("reply", text)], ["reply"], dialogue_context=[
        {"source_id": "reply", "role": "assistant", "usage": "context_only", "text": question, "truncated": False}])
    assert result["sources"][0]["text"] == text
    assert result["sources"][0]["role"] == "student"
    assert result["sources"][0]["dialogue_context"]["text"] == question
    assert result["sources"][0]["dialogue_context"]["usage"] == "context_only"
    assert result["source_unknowns"] == []


def test_corrections_are_shown_with_the_cited_original_without_changing_model_citations():
    sources = [source("old", "Помощь неизвестна.", "teacher"),
               source("fix", "Исправление: ученик работал с наводящим вопросом.", "teacher", corrects_source_id="old"),
               source("unrelated", "Другая запись.")]
    result = review(sources, ["old"])
    assert [row["source_id"] for row in result["sources"]] == ["old", "fix"]
    assert result["sources"][0]["corrected_by"] == ["fix"]
    assert result["sources"][1]["cited"] is False
    assert result["source_unknowns"] == [], "A directly corrected unknown is not presented as active"


def test_a_correction_and_its_chain_remain_one_review_group():
    result = review([source("old", "Самостоятельность неизвестна.", "teacher"),
        source("fix", "Самостоятельность не подтверждена.", "teacher", corrects_source_id="old"),
        source("last", "Нужно уточнить условия наблюдения.", "teacher", corrects_source_id="fix")], ["last"])
    assert [row["source_id"] for row in result["sources"]] == ["old", "fix", "last"]
    assert result["source_unknowns"] == []


@pytest.mark.parametrize("text,conditions", [
    ("Помощь: unknown. independent_work=unknown.", {"support", "independence"}),
    ("Условия помощи пока неизвестны.", {"support"}),
    ("Самостоятельная перепроверка не подтверждена.", {"independence"}),
    ("Условия помощи известны, самостоятельность неизвестна.", {"independence"}),
    ("Самостоятельность подтверждена, помощь неизвестна.", {"support"}),
    ("Работал с помощью учителя. Неизвестно, какой сюжет нравится.", set()),
    ("Помощь: joint; самостоятельность: unknown.", {"independence"}),
    ("Ученик ответил ага.", set()),
])
def test_explicit_condition_notes_are_literal_and_do_not_cross_between_conditions(text, conditions):
    result = review([source("note", text, "teacher")])
    assert {row["condition"] for row in result["source_unknowns"]} == conditions
    assert all(row["quote"] in text for row in result["source_unknowns"])


def test_assistant_unknowns_stay_context_and_do_not_become_observations():
    result = review([source("student", "ага")], ["student"], dialogue_context=[
        {"source_id": "student", "role": "assistant", "usage": "context_only", "text": "Помощь неизвестна.", "truncated": True},
        {"source_id": "absent", "role": "assistant", "usage": "context_only", "text": "Foreign context"}])
    assert result["source_unknowns"] == []
    assert result["sources"][0]["dialogue_context"]["truncated"] is True


def test_unavailable_source_or_correction_cannot_leak_through_review_notes():
    with pytest.raises(ValueError, match="recommendation_source_unavailable"):
        recommendation_source_review({**ACTION, "evidence_turn_ids": ["foreign"]},
            {"sources": [source("own", "own")], "evidence_ids": ["own", "foreign"]})
    with pytest.raises(ValueError, match="recommendation_source_unavailable"):
        review([source("not-visible", "Помощь неизвестна.")], evidence_ids=[])
    with pytest.raises(ValueError, match="invalid_recommendation_correction"):
        review([source("fix", "Исправление", "teacher", corrects_source_id="foreign")], ["fix"])


def test_review_display_metadata_is_not_duplicated_into_the_task_generation_prompt(monkeypatch):
    store, captured = ApiStore(), []
    recommendation = {**ACTION, "sources": [source("own", "Помощь неизвестна.")],
                      "evidence_review": {"version": "recommendation-source-review-v1", "view_only": "DISPLAY-ONLY"}}
    before = deepcopy(recommendation)
    store.get_next_step = lambda _: {"learner_id": "learner-a", "base_revision": 1,
        "topic_id": "math.test", "result": recommendation}
    browser = client(store, Rag())
    browser.app.state.runtime = SimpleNamespace(complete=lambda prompt, **kwargs: captured.append(prompt) or "controlled")

    def design(*args, complete, **kwargs):
        assert complete("Synthetic base prompt", system_prompt="Synthetic system") == "controlled"
        return SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr("pipeline.task_designer.design_task_card", design)
    monkeypatch.setattr("pipeline.task_designer.render_task_card", lambda card, **kwargs: "Synthetic task")
    response = browser.post("/task/design", json={"learner_id": "learner-a", "next_step_id": "step"})
    assert response.status_code == 200, response.text
    projected = json.loads(captured[0].split("Recommendation draft for teacher review: ", 1)[1])
    assert "evidence_review" not in projected and "DISPLAY-ONLY" not in captured[0]
    assert projected["unknowns"] == ACTION["unknowns"] and projected["sources"] == recommendation["sources"]
    assert recommendation == before
