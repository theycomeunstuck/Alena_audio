"""Round-2 unit tests for the new pure backend modules (client A).

No PostgreSQL and no model: the effective-profile replay and the task-catalog
derivation are pure functions and are specified here.
"""
import pytest

from local_app.unified_memory_profile import normalize_fact_key, replay_personal_changes
from local_app.unified_memory_tasks import build_task_catalog


def event(eid, category, operation, value, *, status="applied", scope="durable",
          explicit=True, created_at="2026-01-01T10:00:00Z", source_refs=None,
          evidence=None, reverts_id=None):
    return {"id": eid, "category": category, "operation": operation, "value": value,
            "scope": scope, "status": status, "explicit_durable": explicit,
            "source_refs": source_refs if source_refs is not None else
            [{"kind": "conversation_source", "source_id": f"src-{eid}"}],
            "evidence": evidence or [{"source_id": f"src-{eid}", "text": value}],
            "reverts_id": reverts_id, "created_at": created_at}


PROFILE = {
    "interests": [{"value": "головоломки", "source_refs": [{"kind": "learner_card",
                                                            "field": "interests"}]}],
    "avoided_interests": [],
    "preferences": [{"name": "формулировки", "value": "короткие",
                     "source_refs": [{"kind": "learner_card", "field": "profile.формулировки"}]}],
}


def test_normalize_fact_key_is_whitespace_and_case_consistent():
    assert normalize_fact_key("  Динозавры  ") == "динозавры"
    assert normalize_fact_key("Динамики   и  роботы") == "динамики и роботы"
    assert normalize_fact_key("") == "" and normalize_fact_key(None) == ""


def test_removal_suppresses_profile_value_with_normalized_key():
    result = replay_personal_changes(PROFILE, [
        event("e1", "interest", "remove", "ГОЛОЛОЛО").__class__ and
        event("e1", "interest", "remove", "  Головоломки  ")])
    assert result["interests"] == []
    # Unrelated values survive.
    result = replay_personal_changes(PROFILE, [
        event("e1", "interest", "remove", "динозавры")])
    assert [item["value"] for item in result["interests"]] == ["головоломки"]


def test_avoidance_never_touches_interests():
    result = replay_personal_changes(PROFILE, [
        event("e1", "avoided_interest", "add", "головоломки")])
    assert [item["value"] for item in result["interests"]] == ["головоломки"]
    assert [item["value"] for item in result["avoided_interests"]] == ["головоломки"]


def test_source_order_replay_and_revert():
    events = [
        event("e1", "interest", "add", "динозавры"),
        event("e2", "interest", "remove", "динозавры"),
        event("e3", "interest", "add", "Динозавры  ", created_at="2026-02-01T09:00:00Z"),
    ]
    result = replay_personal_changes(PROFILE, events)
    assert [item["value"] for item in result["interests"]] == ["головоломки", "Динозавры"]
    item = result["interests"][1]
    assert item["origin"] == "conversation" and item["status"] == "applied"
    assert item["scope"] == "durable" and item["change_id"] == "e3"
    assert item["source_refs"][0]["source_id"] == "src-e3"
    assert item["evidence"][0]["text"] == "Динозавры  "
    # Reverting the latest add removes the effective value again.
    reverted = replay_personal_changes(PROFILE, events + [
        event("e4", "interest", "add", "Динозавры", reverts_id="e3", status="reverted")])
    assert [item["value"] for item in reverted["interests"]] == ["головоломки"]
    # History is untouched: the reverted event keeps its original status.
    assert events[2]["status"] == "applied"


def test_proposed_and_rejected_events_never_replay():
    result = replay_personal_changes(PROFILE, [
        event("e1", "interest", "add", "космос", status="proposed"),
        event("e2", "interest", "add", "роботы", status="rejected"),
        event("e3", "interest", "add", "хомяки", scope="session"),
    ])
    assert [item["value"] for item in result["interests"]] == ["головоломки"]


def test_duplicate_add_confirms_the_manual_value():
    result = replay_personal_changes(PROFILE, [
        event("e1", "interest", "add", "головоломки")])
    assert len(result["interests"]) == 1
    item = result["interests"][0]
    assert item["origin"] == "profile", "the manual value stays authoritative"
    assert item["confirmations"] and item["confirmations"][0]["change_id"] == "e1"


def test_preference_remove_matches_by_normalized_value():
    result = replay_personal_changes(PROFILE, [
        event("e1", "preference", "remove", "Короткие")])
    assert result["preferences"] == []
    result = replay_personal_changes(PROFILE, [
        event("e1", "preference", "add", "работает с примерами"),
        event("e2", "preference", "update", "работает с примерами на доске")])
    # `update` is add-if-absent by the documented round-1 semantics: it brings
    # in the new wording; withdrawing the old wording is a `remove` event.
    values = [item["value"] for item in result["preferences"]]
    assert values == ["короткие", "работает с примерами", "работает с примерами на доске"]
    assert result["preferences"][-1]["name"] == "stated_preference"


def test_event_without_real_source_refs_gets_change_reference():
    result = replay_personal_changes(PROFILE, [
        event("e1", "interest", "add", "роботы", source_refs=[])])
    item = next(item for item in result["interests"] if item["value"] == "роботы")
    assert item["source_refs"] == [{"kind": "unified_personal_change", "change_id": "e1"}]


# ------------------------------------------------------------------ catalog
def test_catalog_numbers_questions_and_skips_non_tasks():
    lesson_id = "12345678-1111-4111-8111-111111111111"
    text = ("# Урок: Дроби\n\n## Цель\nПонимать дроби.\n"
            "## Разминка\nРазогрев.\n1. Первый вопрос?\n"
            "## Объяснение\nТекст объяснения без заданий.\n"
            "## Практика\n1. Первая задача.\n2. Вторая задача?\n"
            "## Проверка\nЧто показывает знаменатель?\n"
            "## Ответы\n1) ответ")
    catalog = build_task_catalog(lesson_id, text)
    items = catalog["items"]
    assert catalog["recipe"] == "unified_task_catalog_v1"
    assert [(item["stage"], item["position"]) for item in items] == [
        ("warmup", 1), ("practice", 2), ("practice", 3), ("check", 4)]
    assert items[0]["number"] == 1
    assert items[2]["number"] == 2
    assert items[3]["number"] is None, "a bare question line has no number"
    assert all(item["task_id"] == f"task:{lesson_id}:{item['stage']}:{item['position']}"
               for item in items)
    # The goal, the explanation prose and the answer section never yield tasks.
    assert all(item["section"] in ("Разминка", "Практика", "Проверка") for item in items)
    # Offsets point into the lesson text and round-trip.
    for item in items:
        span = text[item["offsets"]["start"]:item["offsets"]["end"]]
        assert span.strip() == item["prompt_text"]


def test_catalog_is_stable_and_never_invents_tasks():
    lesson_id = "12345678-1111-4111-8111-111111111111"
    text = "## Практика\n1. Одна явная задача.\n"
    assert build_task_catalog(lesson_id, text) == build_task_catalog(lesson_id, text)
    assert build_task_catalog(lesson_id, "")["items"] == []
    assert build_task_catalog(lesson_id, None)["items"] == []
    assert build_task_catalog(lesson_id, "## Цель\nНет секций с заданиями.")["items"] == []
    # Marked-up/bullet lines and answers are not tasks.
    messy = "## Практика\n- пункт списка\n> цитата\n# не заголовок задания\n1. явная задача.\n"
    items = build_task_catalog(lesson_id, messy)["items"]
    assert [item["prompt_text"] for item in items] == ["явная задача."]
    long_line = "1. " + "очень длинное задание " * 60
    items = build_task_catalog(lesson_id, f"## Практика\n{long_line}\n")["items"]
    assert all(len(item["prompt_text"]) <= 500 for item in items)


def test_catalog_task_ids_are_qualified_by_lesson_version():
    first = build_task_catalog("11111111-1111-4111-8111-111111111111", "## Практика\n1. А.")
    second = build_task_catalog("22222222-2222-4222-8222-222222222222", "## Практика\n1. А.")
    assert first["items"][0]["task_id"] != second["items"][0]["task_id"]
    assert first["items"][0]["task_id"].startswith("task:11111111")
