"""Source corrections stay paired and cannot revive a withdrawn assertion."""
from copy import deepcopy

import pytest

from local_app.memory_service import (MemoryWorker, SOURCE_CORRECTION_NOTICE,
    _tail_with_dialogue_context, approved_chat_memory, memory_state, restore_memory)
from research_memory.methods import Memory
from tests.test_dialogue_memory_context import SnapshotStore, TOPIC, prepare, source


def note(source_id, text, target=None):
    row = source(source_id, text, role="teacher", question=None, session=source_id)
    if target:
        row["corrects_source_id"] = target
    return row


def corrected_rows():
    return [note("original", "В журнале записано: ученик пропустил ноль и написал 33."),
            note("middle", "Обычная промежуточная заметка. " * 220),
            note("correction", "Исправляю журнал: в исходной работе изначально 303, помощь неизвестна.", "original")]


def test_tail_restores_original_with_correction_when_an_ordinary_tail_would_evict_it():
    rows = corrected_rows()
    state, context = prepare(rows)
    frozen_tail = restore_memory(state, None, method="tail", topic_id=TOPIC).context(TOPIC)
    assert "original" not in frozen_tail.evidence_ids and "correction" in frozen_tail.evidence_ids
    assert set(context["evidence_ids"]) == {"original", "correction"}
    assert rows[0]["text"] in context["text"] and rows[2]["text"] in context["text"]
    assert rows[1]["text"] not in context["text"]
    assert SOURCE_CORRECTION_NOTICE in context["text"]
    assert "Исправление источника original" in context["text"]
    assert context["sources"][1]["corrects_source_id"] == "original"
    assert "corrects_source_id" not in state["sources"]["correction"]


def test_oversized_correction_group_is_skipped_whole_with_diagnostics_and_without_state_changes():
    rows = [note("original", "Очень длинное старое наблюдение. " * 120),
            note("correction", "Исправляю: помощь не фиксировалась.", "original")]
    memory = Memory("tail", None, 500, TOPIC)
    for row in rows:
        memory.update({"lesson_id": row["source_id"], "observed_on": row["observed_on"],
                       "turns": [{"turn_id": row["source_id"], "role": "teacher", "text": row["text"]}]})
    before, events = memory_state(memory), []
    text, ids, contexts = _tail_with_dialogue_context(memory, rows, selection_events=events)
    assert (text, ids, contexts) == ("", set(), [])
    assert events[0]["event"] == "source_correction_group_skipped"
    assert events[0]["source_ids"] == ["original", "correction"]
    assert events[0]["reason"] == "group_exceeds_budget" and events[0]["group_chars"] > 500
    assert memory_state(memory) == before


def test_correction_chain_and_branches_are_one_group_while_new_success_is_an_ordinary_source():
    rows = [note("original", "Ошибочная заметка о 33."),
            note("first-fix", "В исходнике 303, помощь неизвестна.", "original"),
            note("second-fix", "Уточняю первую поправку: пример был 3636 : 12.", "first-fix"),
            note("parallel-fix", "Уточнение к исходнику: работа уже была завершена.", "original"),
            source("new-attempt", "Новый пример 1248 : 12 решил: 104.", question=None)]
    _, context = prepare(rows)
    assert set(context["evidence_ids"]) == {row["source_id"] for row in rows}
    assert [(row["source_id"], row.get("corrects_source_id")) for row in context["sources"]] == [
        ("original", None), ("first-fix", "original"), ("second-fix", "first-fix"),
        ("parallel-fix", "original"), ("new-attempt", None)]
    assert context["text"].index(rows[0]["text"]) < context["text"].index(rows[1]["text"])


@pytest.mark.parametrize("target,role", [("missing-source", "teacher"), ("original", "student"), ("correction", "teacher")])
def test_invalid_source_relation_fails_instead_of_presenting_an_unpaired_correction(target, role):
    rows = [note("original", "Ранее сохранённое наблюдение."), note("correction", "Уточнение.", target)]
    rows[1]["author_role"] = role
    with pytest.raises(ValueError, match="source_correction_requires"):
        prepare(rows)


@pytest.mark.parametrize("method", ["summary", "structured"])
def test_other_demo_writers_fail_explicitly_until_they_support_application_source_links(method):
    with pytest.raises(ValueError, match="исправлениями источников"):
        prepare(corrected_rows(), method=method)


def test_profile_only_keeps_empty_observations_even_when_archive_contains_source_corrections():
    _, context = prepare(corrected_rows(), method="profile_only")
    assert context["text"] == "" and context["sources"] == [] and context["evidence_ids"] == []


def version(rows, version_id="approved", approved=True):
    state, context = prepare(rows)
    return {"version_id": version_id, "method": "tail", "topic_id": TOPIC, "state": state,
            "context": context, "approved": approved, "base_revision": 1, "stale": True}


def test_new_teacher_correction_withholds_old_approved_claim_until_the_updated_version_is_confirmed():
    rows = corrected_rows()
    old = version(rows[:1])
    draft = version(rows, "draft", False)
    snapshot = {"revision": 1, "sources": rows, "versions": [draft, old]}
    before = deepcopy(snapshot)
    selected = approved_chat_memory(snapshot, TOPIC)
    assert selected["text"] == "" and selected["sources"] == [] and selected["version_id"] is None
    assert selected["withheld_reason"] == "source_correction_requires_review" and selected["stale"]
    assert snapshot == before
    snapshot["versions"][0]["approved"] = True
    selected = approved_chat_memory(snapshot, TOPIC)
    assert selected["version_id"] == "draft" and rows[0]["text"] in selected["text"]
    assert rows[2]["text"] in selected["text"] and "помощь неизвестна" in selected["text"]


def test_new_ordinary_attempt_or_correction_of_another_topic_does_not_withhold_old_approved_memory():
    row = note("original", "Ученик завершил одну задачу; помощь неизвестна.")
    old = version([row])
    ordinary = source("new-student", "хз", question=None)
    other_correction = note("other-correction", "Уточнение другой темы.", "original")
    other_correction["topic_id"] = "programming.python.conditions"
    snapshot = {"revision": 1, "sources": [row, ordinary, other_correction], "versions": [old]}
    selected = approved_chat_memory(snapshot, TOPIC)
    assert selected["version_id"] == "approved" and row["text"] in selected["text"]
    assert selected["stale"]


def test_later_fix_of_a_fix_or_missing_original_also_withholds_the_entire_old_context():
    rows = [note("original", "Исходная заметка."), note("first", "Первое исправление.", "original")]
    approved = version(rows)
    rows.append(note("last", "Поправка к исправлению.", "first"))
    selected = approved_chat_memory({"revision": 1, "sources": rows, "versions": [approved]}, TOPIC)
    assert selected["version_id"] is None and not selected["text"]
    approved["context"]["sources"] = approved["context"]["sources"][1:]
    approved["context"]["evidence_ids"] = ["first"]
    selected = approved_chat_memory({"revision": 1, "sources": rows[:2], "versions": [approved]}, TOPIC)
    assert selected["version_id"] is None


def test_legacy_unattributed_context_is_not_used_when_its_topic_has_an_explicit_correction():
    rows = corrected_rows()
    old = version(rows[:1])
    old["context"] = {"text": "Старая заметка без сохранённых ссылок."}
    selected = approved_chat_memory({"revision": 1, "sources": rows, "versions": [old]}, TOPIC)
    assert not selected["text"] and selected["withheld_reason"] == "source_correction_requires_review"
