"""Confirmed conversational interests retain literal student provenance."""
import json

import pytest


def test_interest_candidates_keep_student_quotes_and_exclude_other_authors():
    from local_app.conversation_memory import validate_interest_candidates
    sources = [{"source_id": "s1", "author_role": "student", "content": "Я люблю Minecraft."},
               {"source_id": "s2", "author_role": "assistant", "content": "Ты любишь шахматы."}]
    result = validate_interest_candidates(json.dumps({"candidates": [
        {"operation": "add", "interest": "Minecraft", "source_id": "s1", "quote": "Я люблю Minecraft."}]}), sources)
    assert result == [{"candidate_id": result[0]["candidate_id"], "operation": "add", "interest": "Minecraft",
                       "source_id": "s1", "quote": "Я люблю Minecraft."}]
    with pytest.raises(ValueError, match="student"):
        validate_interest_candidates(json.dumps({"candidates": [
            {"operation": "add", "interest": "шахматы", "source_id": "s2", "quote": "Ты любишь шахматы."}]}), sources)


def test_completed_session_worker_keeps_interest_failure_separate_from_ready_tail():
    from local_app.memory_service import MemoryWorker
    from tests.test_memory_service import WorkerStore, Runtime

    store = WorkerStore()
    store.job.update(kind="memory_update", payload={"learning_session_id": "session-one"})
    store.get_learning_session = lambda session_id, learner_id: {"status": "completed"}
    store.list_learning_messages = lambda session_id: [
        {"source_id": "student-1", "author_role": "student", "content": "Я люблю Minecraft."}]
    result = MemoryWorker(store, Runtime()).run_once()
    assert result.get("error") is None
    assert result["context"]["interest_extraction"]["status"] == "failed"
    assert result["context"]["interest_candidates"] == []
    assert result["memory_state"]["method"] == "tail"


def test_worker_interest_drafts_use_only_completed_student_sources_without_auto_approval():
    from local_app.memory_service import MemoryWorker
    from tests.test_memory_service import WorkerStore, Runtime

    sources = [{"source_id":"student-1","author_role":"student","content":"Я люблю Minecraft."},
               {"source_id":"assistant-1","author_role":"assistant","content":"Любишь шахматы?"},
               {"source_id":"teacher-1","author_role":"teacher","content":"Он любит футбол."}]
    store = WorkerStore()
    store.job.update(kind="memory_update", payload={"learning_session_id":"session-one"})
    store.get_learning_session = lambda session_id, learner_id: {"status":"completed"}
    store.list_learning_messages = lambda session_id: sources
    runtime = Runtime({"candidates":[{"operation":"add","interest":"Minecraft", "source_id":"student-1",
                                      "quote":"Я люблю Minecraft."}]})
    result = MemoryWorker(store,runtime).run_once()
    assert result["context"]["interest_candidates"][0]["interest"] == "Minecraft"
    assert result["context"]["interest_extraction"]["status"] == "ready"
    assert not result["approved"]
    sent = json.loads(runtime.calls[0][0][-1]["content"])
    assert sent["sources"] == [sources[0]]


def test_oversized_conversation_extraction_keeps_ready_tail_without_truncation_or_retry():
    from local_app.memory_service import MemoryWorker
    from tests.test_memory_service import WorkerStore, Runtime

    store = WorkerStore()
    store.job.update(kind="memory_update", payload={"learning_session_id":"session-one"})
    store.get_learning_session = lambda session_id, learner_id: {"status":"completed"}
    store.list_learning_messages = lambda session_id: [{"source_id":"student-1","author_role":"student",
                                                       "content":"Много разных слов о разговоре. " * 10000}]
    runtime = Runtime({"candidates":[]})
    result = MemoryWorker(store,runtime).run_once()
    assert result.get("error") is None
    assert result["context"]["interest_extraction"]["message"] == "memory_writer_input_exceeds_context"
    assert result["memory_state"]["method"] == "tail"
    assert runtime.calls == []
    assert len(result["result"]["calls"]) == 1


def test_conversation_context_saturation_preserves_raw_completion_and_discards_only_interest_draft():
    from local_app.memory_service import MemoryWorker
    from tests.test_memory_service import WorkerStore

    raw_response = {"message":{"content":json.dumps({"candidates":[]})},
                    "prompt_eval_count":6656,"eval_count":15}
    class SaturatedRuntime:
        context_tokens = 8192
        def chat(self, messages, **kwargs):
            return raw_response
    store = WorkerStore()
    store.job.update(kind="memory_update", payload={"learning_session_id":"session-one"})
    store.get_learning_session = lambda session_id, learner_id: {"status":"completed"}
    store.list_learning_messages = lambda session_id: [{"source_id":"student-1","author_role":"student",
                                                       "content":"Я люблю Minecraft."}]
    result = MemoryWorker(store,SaturatedRuntime()).run_once()
    assert result.get("error") is None
    assert result["context"]["interest_extraction"]["message"] == "context_saturation_requires_review"
    assert result["context"]["interest_candidates"] == []
    assert result["memory_state"]["method"] == "tail"
    assert result["result"]["calls"][0]["raw_response"] == raw_response


def test_preference_projection_applies_ordered_changes_and_preserves_history():
    from local_app.conversation_memory import apply_interest_choices, preference_projection, validate_interest_candidates
    sources = [{"source_id":"s1","author_role":"student","content":"Больше не люблю Minecraft."},
               {"source_id":"s2","author_role":"student","content":"А теперь снова люблю Minecraft."}]
    # Deliberately reverse model output; source order determines the current state.
    raw = json.dumps({"candidates":[
        {"source_id":"s2","operation":"add","interest":"Minecraft","quote":sources[1]["content"]},
        {"source_id":"s1","operation":"remove","interest":"Minecraft","quote":sources[0]["content"]}]})
    candidates = validate_interest_candidates(raw,sources)
    original = {"interests":["Minecraft","космос"],"rag_context":{"avoid":["без длинных серий"]}}
    rejected = apply_interest_choices(original,candidates[:1])
    assert preference_projection(rejected) == {"interests":["космос"],"avoided_interests":["Minecraft"]}
    restored = apply_interest_choices(rejected,candidates[1:])
    assert restored["interests"] == ["космос","Minecraft"]
    assert preference_projection(restored) == {"interests":["космос"],"avoided_interests":["Minecraft"]}
    assert restored["rag_context"]["avoid"] == ["без длинных серий","Не использовать сюжет: Minecraft"]
    assert [item["quote"] for item in candidates] == [source["content"] for source in sources]
    assert original["interests"] == ["Minecraft","космос"]


def test_avoiding_a_story_does_not_revoke_the_childs_interest():
    from local_app.conversation_memory import apply_interest_choices, preference_projection, validate_interest_candidates
    source = {"source_id":"s1","author_role":"student","content":"Люблю Minecraft, но задачи про него не хочу."}
    raw = json.dumps({"candidates":[
        {"source_id":"s1","operation":"add","interest":"Minecraft","quote":source["content"]},
        {"source_id":"s1","operation":"avoid","interest":"Minecraft","quote":source["content"]}]})
    candidates = validate_interest_candidates(raw,[source])
    card = apply_interest_choices({"interests":[],"rag_context":{"avoid":[]}},candidates)
    assert card["interests"] == ["Minecraft"]
    assert preference_projection(card) == {"interests":[],"avoided_interests":["Minecraft"]}


def test_explicit_story_permission_clears_only_generated_avoid_and_adds_no_interest():
    from local_app.conversation_memory import apply_interest_choices, preference_projection, validate_interest_candidates
    source = {"source_id":"s1","author_role":"student","content":"Теперь можно задачи про Minecraft."}
    candidates = validate_interest_candidates(json.dumps({"candidates":[
        {"source_id":"s1","operation":"allow","interest":"Minecraft","quote":source["content"]}]}),[source])
    card = apply_interest_choices({"interests":[],"rag_context":{"avoid":[
        "Не использовать сюжет: Minecraft","Учебные сюжеты выбирает ребёнок"]}},candidates)
    assert card["rag_context"]["avoid"] == ["Учебные сюжеты выбирает ребёнок"]
    assert preference_projection(card) == {"interests":[],"avoided_interests":[]}


@pytest.mark.parametrize("candidate,diagnostic", [
    ({"source_id":"unknown","operation":"add","interest":"Minecraft","quote":"Я люблю Minecraft."},"unavailable"),
    ({"source_id":"s1","operation":"add","interest":"Minecraft","quote":"Я люблю шахматы."},"literal"),
])
def test_interest_candidates_reject_invalid_sources_and_quotes(candidate,diagnostic):
    from local_app.conversation_memory import validate_interest_candidates
    with pytest.raises(ValueError,match=diagnostic):
        validate_interest_candidates(json.dumps({"candidates":[candidate]}),[
            {"source_id":"s1","author_role":"student","content":"Я люблю Minecraft."}])
