"""Round-2 P0.1: lesson generation consumes the real facade fast-memory shape.

The facade returns effective facts as records with ``value``/``source_refs``.
These tests verify the actual model input (a recording model), not just the
saved revision: changed interests, separate avoidance, confirmed errors and
help, goals, the unfinished step, no card fallback for empty effective fields,
and whole-unit budgeting with an explicit omitted-data diagnostic.
"""
from __future__ import annotations

import json

import pytest

from local_app.lesson_agent.fast_memory import bounded_fast_memory_text, fast_memory_block
from local_app.lesson_agent.generation import generate_lesson
from tests.test_unified_app_api import LESSON_TEXT
from tests.test_unified_app_worker import WorkerRuntime, WorkerStore, make_worker
from tests.test_unified_app_worker import FakeUnifiedMemory  # noqa: F401  (contract fake)

pytest.importorskip("pydantic")


def _record(value, category="interest", kind="unified_personal_change", origin="conversation"):
    return {"value": value, "origin": origin,
            "source_refs": [{"kind": kind, "category": category, "value": value}]}


def _fast_memory(*, interests=None, avoided=None, preferences=None, goals=None,
                 confirmed=None, errors=None, help=None, unfinished=None):
    return {
        "method": "unified_fast_memory", "learner_id": "learner-a", "revision": "um-abc123",
        "personal": {"interests": interests or [], "avoided_interests": avoided or [],
                     "preferences": preferences or []},
        "learning": {"goals": goals or [], "confirmed_observations": confirmed or [],
                     "errors": errors or [], "help": help or [], "unfinished": unfinished},
        "session": {}, "source_refs": {"profile_authority": "learner_card",
                                       "conversation_authority": "unified_personal_changes"},
    }


VALID_LESSON = LESSON_TEXT


class RecordingModel:
    """Records every prompt it is asked to generate against."""

    def __init__(self, answer=VALID_LESSON):
        self.prompts = []
        self.answer = answer

    def __call__(self, messages, purpose, schema):
        self.prompts.append({"purpose": purpose, "messages": messages})
        return self.answer


class TestFastMemoryBlockProjection:
    def test_records_with_value_and_source_refs_reach_block(self):
        fast = _fast_memory(
            interests=[_record("космос"), _record("роботы")],
            avoided=[_record("страшные истории", category="avoided_interest")],
            preferences=[{"name": "stated_preference", "value": "короткие примеры",
                          "origin": "conversation", "source_refs": [{"kind": "x"}]}],
            goals=[{"topic_id": "math", "goal": "выучить умножение", "updated_at": "2026-11-10",
                    "origin": "profile", "source_refs": [{"kind": "learner_card"}]}],
            confirmed=[{"topic_id": "math", "status": "needs_practice", "notes": "n",
                        "source_refs": [{"kind": "learner_card"}]}],
            errors=[{"error_tag": "digit_swap", "topic_id": "math", "subject": "math",
                     "count": 2, "last_seen": "2026-11-10", "source_refs": [{"kind": "c"}]}],
            help=[{"topic_id": "math", "strategies": ["разбор по разрядам"],
                   "source_refs": [{"kind": "c"}]}],
            unfinished={"lesson_id": "lesson-9", "stage": "practice",
                        "pending_task_id": "task-2", "completed_task_ids": ["task-1"],
                        "source_refs": [{"kind": "unified_lesson_state"}]},
        )
        block, diagnostics = fast_memory_block(fast)
        assert [item["value"] for item in block["interests"]] == ["космос", "роботы"]
        assert [item["value"] for item in block["avoided_themes"]] == ["страшные истории"]
        assert block["preferences"] == [{"name": "stated_preference", "value": "короткие примеры"}]
        assert block["learning_goals"] == [{"topic_id": "math", "goal": "выучить умножение"}]
        assert block["confirmed_observations"] == [{"topic_id": "math", "status": "needs_practice"}]
        assert block["known_errors"][0]["error_tag"] == "digit_swap"
        assert block["helpful_strategies"] == [{"topic_id": "math",
                                                "strategies": "разбор по разрядам"}]
        assert block["unfinished_step"] == {"lesson_id": "lesson-9", "stage": "practice",
                                            "pending_task_id": "task-2",
                                            "completed_task_ids": ["task-1"]}
        assert diagnostics["included"]["interests"] == 2
        assert diagnostics["empty_sections"] == []
        # The serialized block stays complete valid JSON.
        json.dumps(block, ensure_ascii=False)

    def test_avoidance_never_merges_into_interests(self):
        fast = _fast_memory(interests=[_record("космос")],
                            avoided=[_record("страшные истории", category="avoided_interest")])
        block, _diagnostics = fast_memory_block(fast)
        assert "страшные истории" not in [item["value"] for item in block["interests"]]
        assert block["avoided_themes"][0]["value"] == "страшные истории"

    def test_empty_effective_field_reported_not_filled(self):
        fast = _fast_memory()  # every effective section empty
        block, diagnostics = fast_memory_block(fast)
        assert "interests" not in block
        assert "interests" in diagnostics["empty_sections"]
        assert diagnostics["shape_problems"] == []

    def test_unusable_records_reported_not_silently_dropped(self):
        fast = _fast_memory(interests=[{"origin": "conversation", "source_refs": []},
                                       {"value": "   "}])
        block, diagnostics = fast_memory_block(fast)
        assert "interests" not in block
        assert len(diagnostics["shape_problems"]) == 2

    def test_budget_omits_whole_units_and_stays_valid_json(self):
        long_value = "очень-очень-длинный-интерес-" * 40
        fast = _fast_memory(interests=[_record(f"интерес-{i}-{long_value}") for i in range(12)])
        block, diagnostics = fast_memory_block(fast, budget_chars=1200)
        assert block["interests"], "at least one unit must fit a 1200-char budget"
        assert len(json.dumps(block, ensure_ascii=False)) <= 1200
        budget_omits = [o for o in diagnostics["omitted"] if o["reason"] == "budget"]
        assert budget_omits and budget_omits[0]["section"] == "interests"
        # 12 candidates hit the interests section limit (8) first, then the
        # character budget stops the whole-unit inclusion at 5.
        assert budget_omits[0]["count"] == 8 - len(block["interests"])
        assert any(o["reason"] == "section_limit" and o["section"] == "interests"
                   for o in diagnostics["omitted"])

    def test_unfinished_with_malformed_completed_ids_stays_bounded(self):
        # Regression B3: a non-list completed_task_ids must not explode into
        # per-character units in the prompt block.
        fast = _fast_memory(unfinished={"lesson_id": "lesson-1", "stage": "practice",
                                        "completed_task_ids": "task-1,task-2",
                                        "source_refs": []})
        block, _diagnostics = fast_memory_block(fast)
        assert "completed_task_ids" not in block["unfinished_step"]

    def test_plain_string_items_still_supported(self):
        # Legacy callers pass plain strings; the projection keeps working.
        fast = _fast_memory(interests=["роботы"])
        block, _diagnostics = fast_memory_block(fast)
        assert block["interests"] == [{"value": "роботы"}]

    def test_bounded_fast_memory_text_is_complete_json(self):
        fast = _fast_memory(goals=[{"topic_id": "math", "goal": "выучить умножение",
                                    "source_refs": [], "origin": "profile"}],
                            unfinished={"lesson_id": "lesson-1", "stage": "warmup",
                                        "source_refs": []})
        text = bounded_fast_memory_text(fast, budget_chars=200)
        parsed = json.loads(text)  # must not raise: no mid-JSON truncation
        assert "learning_goals" in parsed and "unfinished_step" in parsed

    def test_bounded_fast_memory_text_none_when_empty(self):
        assert bounded_fast_memory_text(_fast_memory()) is None


class TestGenerationConsumesEffectiveMemory:
    def test_effective_records_reach_model_and_card_fallback_suppressed(self):
        model = RecordingModel()
        fast = _fast_memory(
            interests=[_record("космос")],
            avoided=[_record("страшные истории", category="avoided_interest")],
            errors=[{"error_tag": "digit_swap", "topic_id": "math", "source_refs": []}],
            help=[{"topic_id": "math", "strategies": ["разбор по разрядам"], "source_refs": []}],
            goals=[{"topic_id": "math", "goal": "выучить умножение", "source_refs": [],
                    "origin": "profile"}],
            unfinished={"lesson_id": "lesson-1", "stage": "practice",
                        "pending_task_id": "task-2", "source_refs": []},
        )
        # The card still says "роботы" and an old goal: with the facade present
        # these stale fields must not be concatenated into the prompt.
        card_profile = {"grade": 4, "language": "ru", "interests": ["роботы"],
                        "learning_goal": "старая цель"}
        result = generate_lesson(model, profile=card_profile, topic_id="math",
                                 topic="Умножение", minutes=20, fast_memory=fast)
        user_content = model.prompts[0]["messages"][1]["content"]
        assert "космос" in user_content
        assert "страшные истории" in user_content
        assert "digit_swap" in user_content and "разбор по разрядам" in user_content
        assert "выучить умножение" in user_content
        assert "lesson-1" in user_content and "task-2" in user_content
        # The opaque revision never enters the prompt; stale card data does not.
        assert "um-abc123" not in user_content
        assert "роботы" not in user_content
        assert "старая цель" not in user_content
        payload = result["profile_payload"]
        assert [item["value"] for item in payload["interests"]] == ["космос"]
        assert payload["avoided_themes"][0]["value"] == "страшные истории"
        assert result["fast_memory_diagnostics"]["included"]["interests"] == 1

    def test_empty_effective_interests_do_not_fall_back_to_card(self):
        model = RecordingModel()
        card_profile = {"grade": 4, "interests": ["роботы"], "learning_goal": "старая цель"}
        result = generate_lesson(model, profile=card_profile, topic_id="math",
                                 topic="Умножение", minutes=20, fast_memory=_fast_memory())
        user_content = model.prompts[0]["messages"][1]["content"]
        assert "роботы" not in user_content
        assert "старая цель" not in user_content
        assert result["fast_memory_diagnostics"]["empty_sections"]
        # An empty effective field is not a service failure.
        assert result["validation"]["ok"]

    def test_legacy_card_projection_without_facade(self):
        model = RecordingModel()
        card_profile = {"grade": 4, "interests": ["роботы"], "learning_goal": "старая цель"}
        result = generate_lesson(model, profile=card_profile, topic_id="math",
                                 topic="Умножение", minutes=20, fast_memory=None)
        user_content = model.prompts[0]["messages"][1]["content"]
        assert "роботы" in user_content and "старая цель" in user_content
        assert result["fast_memory_diagnostics"] is None


class TestWorkerGenerationUsesFacadeShape:
    def test_worker_prompt_uses_effective_records_not_card(self):
        store, fake = WorkerStore(), FakeUnifiedMemory()
        runtime = WorkerRuntime([VALID_LESSON])
        worker, _journal = make_worker(store, runtime, fake)
        store.enqueue("lesson_generate", payload={"topic_id": "math", "minutes": 20,
                                                  "client_request_id": "req-r2"})
        store.next_claim = ["job-1"]
        finished = worker.run_once()
        assert finished["status"] == "completed"
        user_content = runtime.chat_calls[0]["messages"][1]["content"]
        # Effective record values reach the actual model input...
        assert "космос" in user_content and "страшные истории" in user_content
        assert "выучить умножение" in user_content
        # ...while the card-only interest stays suppressed.
        assert "роботы" not in user_content
        diagnostics = finished["result"]["fast_memory_diagnostics"]
        assert diagnostics["included"]["interests"] == 1
        assert diagnostics["budget_chars"] > 0
