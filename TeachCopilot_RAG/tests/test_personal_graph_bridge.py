from copy import deepcopy
import asyncio
import json
from types import SimpleNamespace

import pytest

from local_app.personal_graph_memory import (ApplicationModel, MEMORY_BYTE_BUDGET, approved_graph_version,
                                             PersonalGraphSelectionBudgetExceeded,
                                             PersonalGraphProvider,
                                             application_lesson_order,
                                             application_memory_workflow,
                                             add_recent_context_candidates,
                                             round_robin_source_hits, select_application_candidates,
                                             select_context, source_lessons, source_scoped_found,
                                             source_scoped_search)
from local_app.personal_graph_contract import (PERSONAL_GRAPH_CONTEXT_PROTOCOL,
                                               personal_graph_snapshot_fingerprint)
from local_app.memory_store import MemoryStoreMixin, get_learner_memory_snapshot


def snapshot():
    return {"learner_id": "child-a", "revision": 2, "source_revision": 7, "versions": [], "sources": [
        {"source_id": "s1", "author_role": "student", "text": "Хз", "topic_id": "math",
         "observed_on": "2026-01-01", "session_id": "lesson-1",
         "dialogue_context": {"role": "assistant", "text": "Почему так получилось?"}},
        {"source_id": "t1", "author_role": "teacher", "text": "Ребёнок ошибся в счёте.",
         "approved": True, "topic_id": "math", "observed_on": "2026-01-02"},
        {"source_id": "t2", "author_role": "teacher", "text": "Уточнение: ошибка была в записи.",
         "approved": True, "topic_id": "math", "observed_on": "2026-01-03", "corrects_source_id": "t1"},
    ]}


def found_for(snapshot_value, *indices):
    lessons, _ = source_lessons(snapshot_value, "child-a")
    return {"memory": json.dumps([{"original_lesson": lessons[i], "derived_candidates": []}
                                   for i in indices], ensure_ascii=False), "unresolved": []}


def source_fragment(lesson_id, fragment_id, observed_on, prefix, text="x"):
    return {"lesson_id": lesson_id, "fragment_id": fragment_id, "observed_on": observed_on,
            "topic_ids": ["math"], "turns": [
                {"turn_id": prefix + "-teacher", "role": "teacher", "text": "Вопрос?"},
                {"turn_id": prefix + "-student", "role": "student", "text": text},
            ]}


def memory_bytes(lessons):
    canonical = [{key: value for key, value in lesson.items() if key != "_application_recent_context"}
                 for lesson in lessons]
    return len(json.dumps([{"derived_candidates": [], "original_lesson": lesson} for lesson in canonical],
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def test_dialogue_question_is_context_not_evidence():
    value = snapshot()
    lessons, _ = source_lessons(value, "child-a")
    assert lessons[0]["turns"][0]["role"] == "assistant"
    assert lessons[0]["turns"][0]["usage"] == "context_only"
    context = select_context(found_for(value, 0), value, "child-a", "math")
    assert context["evidence_ids"] == ["s1"]
    assert "Почему так получилось?" in context["text"]
    assert all(row["role"] != "assistant" for row in context["sources"])


def test_truncated_assistant_context_is_marked_and_never_detached_from_its_anchor():
    value = snapshot()
    value["sources"][0]["dialogue_context"] = {"role": "assistant", "text": "А" * 700, "truncated": True}
    lessons, _ = source_lessons(value, "child-a")
    assistant = lessons[0]["turns"][0]
    assert assistant["truncated"] is True and assistant["excerpt_mode"] == "tail"
    assert len(assistant["text"]) == 600
    found = found_for(value, 0)
    context = select_context(found, value, "child-a", "math")
    assert json.loads(context["text"])["lessons"][0]["turns"][0]["truncated"] is True
    orphan = deepcopy(lessons[0])
    orphan["turns"] = [assistant]
    found["memory"] = json.dumps([{"original_lesson": orphan}])
    context = select_context(found, value, "child-a", "math", allowed_ids={"s1"})
    assert context["text"] == "" and context["evidence_ids"] == []


def test_corrections_are_included_with_their_original_and_cannot_cross_approval():
    value = snapshot()
    context = select_context(found_for(value, 1), value, "child-a", "math")
    assert context["evidence_ids"] == ["t1", "t2"]
    assert "Уточнение" in context["text"]
    with pytest.raises(ValueError, match="new approved"):
        select_context(found_for(value, 1), value, "child-a", "math", allowed_ids={"t1"})
    permitted = select_context(found_for(value, 1), value, "child-a", "math", allowed_ids={"t1", "t2"})
    assert permitted["evidence_ids"] == ["t1", "t2"]


def test_correction_reloads_the_original_students_question():
    value = snapshot()
    value["sources"][2]["corrects_source_id"] = "s1"
    context = select_context(found_for(value, 2), value, "child-a", "math", allowed_ids={"s1", "t2"})
    assert context["evidence_ids"] == ["s1", "t2"]
    assert "Почему так получилось?" in context["text"]
    assert all(row["role"] != "assistant" for row in context["sources"])
    with pytest.raises(ValueError, match="new approved"):
        select_context(found_for(value, 2), value, "child-a", "math", allowed_ids={"t2"})


def test_split_student_and_assistant_context_are_reunited_without_duplicates():
    value = snapshot()
    value["sources"][0]["session_end_context"] = {
        "role": "assistant", "usage": "context_only", "text": "Сначала нарисуем схему.",
        "assistant_source_id": "assistant-end", "session_id": "lesson-1",
        "excerpt_mode": "whole", "truncated": False,
    }
    lesson = source_lessons(value, "child-a")[0][0]
    before, student, after = lesson["turns"]
    first, second = deepcopy(lesson), deepcopy(lesson)
    first["turns"], second["turns"] = [student], [after]
    found = {"memory": json.dumps([{"original_lesson": first}, {"original_lesson": second}], ensure_ascii=False)}
    context = select_context(found, value, "child-a", "math", allowed_ids={"s1"})
    assert context["evidence_ids"] == ["s1"]
    assert context["text"].count(before["text"]) == 1
    assert context["text"].count(after["text"]) == 1


def test_selected_source_keeps_its_topic_for_the_model():
    value = snapshot()
    value["sources"][0]["topic_id"] = "reading.other_topic"
    context = select_context(found_for(value, 0), value, "child-a", "math")
    selected = json.loads(context["text"])["lessons"][0]
    assert "reading.other_topic" in selected["topic_ids"]
    assert next(turn for turn in selected["turns"] if turn["turn_id"] == "s1")["topic_id"] == "reading.other_topic"


def test_application_selection_balances_exact_semantic_lexical_hits_and_recent_context():
    semantic_first = source_fragment("old", "old:c0001", "2026-01-01", "old-1", "Старый факт.")
    lexical_hit = source_fragment("old", "old:c0002", "2026-01-01", "old-2", "Редкий термин.")
    graph_fallback = source_fragment("old", "old:c0003", "2026-01-01", "old-3", "Не найденный фрагмент.")
    recent = source_fragment("recent", "recent:c0004", "2026-01-04", "recent-4", "Недавний ответ.")
    recent["_application_recent_context"] = True
    expected = memory_bytes([semantic_first, recent, lexical_hit])
    found = select_application_candidates(
        [semantic_first, lexical_hit, graph_fallback, recent],
        [
            {"kind": "semantic", "lesson_ids": ["old"], "chunk_id": "old:c0001", "score": 0.9},
            {"kind": "semantic", "lesson_ids": ["old"], "chunk_id": "old:c0002", "score": 0.8},
            {"kind": "lexical", "lesson_ids": ["old"], "chunk_id": "old:c0002", "score": 99.0},
            {"kind": "graphiti", "lesson_ids": ["old"], "text": "a graph claim without a chunk id"},
        ],
        expected, recent_lesson_ids=["recent"],
    )
    blocks = json.loads(found["memory"])
    assert [block["original_lesson"]["fragment_id"] for block in blocks] == [
        "old:c0001", "recent:c0004", "old:c0002"]
    assert "old-1-student" in found["source_ids"]
    assert "old-2-student" in found["source_ids"]
    assert "recent-4-student" in found["source_ids"]
    assert "old-3-student" not in found["source_ids"]
    assert found["memory_tokens"] == expected
    assert any(row["fragment_id"] == "old:c0003" and row["reason"] == "no_exact_source_chunk_match"
               for row in found["skipped"])
    assert all("_selection_recent_rank" not in block["original_lesson"] for block in blocks)


def test_narrow_recent_budget_prefers_newer_lesson_in_source_order():
    older = source_fragment("recent-older", "recent-older:c0001", "2026-01-03", "recent-old", "Одинаковый размер.")
    newer = source_fragment("recent-newer", "recent-newer:c0001", "2026-01-03", "recent-new", "Одинаковый размер.")
    older["_application_recent_context"] = True
    newer["_application_recent_context"] = True
    found = select_application_candidates(
        [older, newer], [], memory_bytes([newer]), recent_lesson_ids=["recent-older", "recent-newer"])
    assert "recent-new-student" in found["source_ids"]
    assert "recent-old-student" not in found["source_ids"]


def test_archive_channel_hits_are_round_robined_before_hydration_cap():
    hits = [
        *[{"kind": "semantic", "chunk_id": f"s{i}"} for i in range(1, 6)],
        {"kind": "lexical", "chunk_id": "l1"},
        {"kind": "source_semantic", "chunk_id": "ss1"},
        {"kind": "source_lexical", "chunk_id": "sl1"},
        {"kind": "graphiti", "lesson_ids": ["old"]},
    ]
    assert [hit.get("chunk_id", "graphiti") for hit in round_robin_source_hits(hits)] == [
        "s1", "l1", "ss1", "sl1", "s2", "s3", "s4", "s5", "graphiti"]


def test_application_lesson_recency_uses_event_date_then_source_order_for_date_ties():
    lessons = [
        {"lesson_id": "older-but-recorded-later", "observed_on": "2026-01-01"},
        {"lesson_id": "same-day-first", "observed_on": "2026-01-03"},
        {"lesson_id": "same-day-second", "observed_on": "2026-01-03"},
        {"lesson_id": "newer", "observed_on": "2026-01-04"},
    ]
    assert application_lesson_order(lessons) == [
        "older-but-recorded-later", "same-day-first", "same-day-second", "newer"]


def test_selected_lesson_blocks_are_displayed_by_event_date_then_source_order():
    value = {"learner_id": "child-a", "sources": [
        {"source_id": "recorded-first", "author_role": "student", "text": "Событие позднее.",
         "topic_id": "math", "observed_on": "2026-01-03", "session_id": "session-late"},
        {"source_id": "recorded-second", "author_role": "student", "text": "Событие раньше.",
         "topic_id": "math", "observed_on": "2026-01-01", "session_id": "session-early"},
    ]}
    found = found_for(value, 0, 1)
    context = select_context(found, value, "child-a", "math")
    shown = json.loads(context["text"])["lessons"]
    assert [lesson["observed_on"] for lesson in shown] == ["2026-01-01", "2026-01-03"]
    assert [lesson["turns"][0]["turn_id"] for lesson in shown] == ["recorded-second", "recorded-first"]


def test_best_exact_hit_is_kept_when_only_a_weaker_hit_can_pair_with_recent_context():
    best = source_fragment("old", "old:c0001", "2026-01-01", "best", "b" * 800)
    weaker = source_fragment("old", "old:c0002", "2026-01-01", "weak", "w")
    recent = source_fragment("recent", "recent:c0001", "2026-01-04", "recent", "r" * 80)
    recent["_application_recent_context"] = True
    budget = max(memory_bytes([best]), memory_bytes([weaker, recent]))
    assert memory_bytes([best, recent]) > budget
    assert memory_bytes([weaker, recent]) <= budget
    found = select_application_candidates(
        [best, weaker, recent],
        [{"kind": "semantic", "lesson_ids": ["old"], "chunk_id": "old:c0001"},
         {"kind": "semantic", "lesson_ids": ["old"], "chunk_id": "old:c0002"}],
        budget, recent_lesson_ids=["recent"],
    )
    blocks = json.loads(found["memory"])
    assert [block["original_lesson"]["fragment_id"] for block in blocks] == ["old:c0001"]
    assert "best-student" in found["source_ids"]
    assert "weak-student" not in found["source_ids"]


def test_direct_and_recent_candidates_with_overlapping_turn_ids_are_not_duplicated():
    value = snapshot()
    lesson = source_lessons(value, "child-a")[0][0]
    direct_whole = deepcopy(lesson)
    recent_overlap = {**lesson, "fragment_id": "app:c0001", "_application_recent_context": True}
    found = select_application_candidates(
        [direct_whole, recent_overlap],
        [{"kind": "semantic", "lesson_ids": [lesson["lesson_id"]],
          "chunk_id": lesson["lesson_id"] + ":c0000"}],
        3000, recent_lesson_ids=[lesson["lesson_id"]],
    )
    context = select_context(found, value, "child-a", "math")
    assert context["evidence_ids"] == ["s1"]
    assert found["source_ids"].count("s1") == 1
    assert len(found["source_ids"]) == len(set(found["source_ids"]))
    assert any(item["reason"] == "duplicate_source_turns" for item in found["skipped"])


def test_oversized_hit_stays_atomic_and_does_not_block_a_smaller_hit_or_recent_pair():
    oversized = source_fragment("old", "old:c0001", "2026-01-01", "large", "я" * 900)
    smaller = source_fragment("old", "old:c0002", "2026-01-01", "small", "Короткий ответ.")
    recent = source_fragment("recent", "recent:c0001", "2026-01-04", "recent", "Недавний ответ.")
    recent["_application_recent_context"] = True
    budget = memory_bytes([smaller, recent])
    found = select_application_candidates(
        [oversized, smaller, recent],
        [{"kind": "semantic", "lesson_ids": ["old"], "chunk_id": "old:c0001"},
         {"kind": "lexical", "lesson_ids": ["old"], "chunk_id": "old:c0002"}],
        budget, recent_lesson_ids=["recent"],
    )
    blocks = json.loads(found["memory"])
    assert {block["original_lesson"]["fragment_id"] for block in blocks} == {"old:c0002", "recent:c0001"}
    assert "large-student" not in found["source_ids"]
    assert any(row["fragment_id"] == "old:c0001" and row["reason"] == "memory_budget_or_context_exceeded"
               for row in found["skipped"])


def test_recent_tail_uses_original_turn_anchor_and_stays_inside_approved_ids():
    value = snapshot()
    value["sources"][0]["session_end_context"] = {
        "role": "assistant", "usage": "context_only", "text": "Продолжим в следующий раз.",
        "assistant_source_id": "assistant-end", "session_id": "lesson-1",
        "excerpt_mode": "whole", "truncated": False,
    }
    lesson = source_lessons(value, "child-a")[0][0]
    before, student, after = lesson["turns"]
    rows = [
        {"lesson_id": lesson["lesson_id"], "chunk_id": "app:c0001", "position": 1,
         "content": {**lesson, "turns": [before, student]}},
        {"lesson_id": lesson["lesson_id"], "chunk_id": "app:c0002", "position": 2,
         "content": {**lesson, "turns": [{"turn_id": "s2", "role": "student", "text": "Не утверждено."}]}},
        {"lesson_id": lesson["lesson_id"], "chunk_id": "app:c0003", "position": 3,
         "content": {**lesson, "turns": [after]}},
    ]
    candidates = add_recent_context_candidates([], rows, [lesson["lesson_id"]], allowed_source_ids={"s1"})
    assert len(candidates) == 1
    assert candidates[0]["fragment_id"] == "app:c0001"
    assert candidates[0]["_application_recent_context"] is True
    assert "s2" not in {turn.get("turn_id") for turn in candidates[0]["turns"]}
    found = select_application_candidates(candidates, [], 3000, recent_lesson_ids=[lesson["lesson_id"]])
    context = select_context(found, value, "child-a", "math", allowed_ids={"s1"})
    assert context["evidence_ids"] == ["s1"]
    assert "Почему так получилось?" in context["text"]
    assert "Продолжим в следующий раз." in context["text"]
    assert all(row["role"] != "assistant" for row in context["sources"])


def test_application_workflow_uses_source_order_and_hydrates_lexical_hits_before_the_four_chunk_cap():
    pytest.importorskip("langgraph")
    from research_memory.personal_memory_lab.workflow import PersonalMemoryWorkflow

    def lesson(lesson_id):
        return {"lesson_id": lesson_id, "observed_on": "2026-02-01", "topic_ids": ["math"],
                "turns": [{"turn_id": lesson_id + "-teacher", "role": "teacher", "text": "Вопрос?"},
                          {"turn_id": lesson_id + "-student", "role": "student", "text": "Ответ."}]}

    old = {"lesson_id": "old-hash", "observed_on": "2026-02-01", "topic_ids": ["math"], "turns": []}
    chunks = []
    for index in range(1, 7):
        chunk_lesson = lesson("old-hash")
        teacher = {"turn_id": f"old-c{index}-teacher", "role": "teacher", "text": f"Вопрос {index}?"}
        student = {"turn_id": f"old-c{index}-student", "role": "student", "text": f"Ответ {index}."}
        chunk_lesson["turns"] = [teacher, student]
        chunks.append({"lesson_id": "old-hash", "chunk_id": f"old-hash:c{index:04d}",
                       "position": index, "content": chunk_lesson})
        old["turns"].extend([teacher, student])
    recent_b, recent_c = lesson("recent-b-hash"), lesson("recent-c-hash")
    chunks.extend([
        {"lesson_id": recent_b["lesson_id"], "chunk_id": "recent-b-hash:c0001", "position": 1,
         "content": recent_b},
        {"lesson_id": recent_c["lesson_id"], "chunk_id": "recent-c-hash:c0001", "position": 1,
         "content": recent_c},
    ])
    archive_order = [recent_b, old, recent_c]  # Simulates date/hash ordering, not source order.
    old_hits = [
        *[{"lesson_id": "old-hash", "chunk_id": f"old-hash:c{i:04d}", "score": 1 - i / 100,
           "channel": "semantic"} for i in range(1, 6)],
        {"lesson_id": "old-hash", "chunk_id": "old-hash:c0006", "score": 0.1, "channel": "lexical"},
    ]

    class Archive:
        def lessons(self, run_id, learner_id, ids=None):
            if ids is None:
                return list(archive_order)
            by_id = {item["lesson_id"]: item for item in [old, recent_b, recent_c]}
            return [by_id[item] for item in ids if item in by_id]

        def chunks(self, run_id, learner_id, ids=None):
            return [row for row in chunks if ids is None or row["lesson_id"] in ids]

        def search(self, run_id, learner_id, query, embedding, limit=8, lesson_ids=None, source_ids=None):
            rows = old_hits if lesson_ids is None else []
            if source_ids is not None:
                rows = [row for row in rows if any(
                    turn["turn_id"] in source_ids for turn in
                    next(item for item in chunks if item["chunk_id"] == row["chunk_id"])["content"]["turns"])]
            return [dict(row) for row in rows]

    class Runtime:
        context = 16384
        def embed(self, _text):
            return [0.0] * 1024
        def count(self, text):
            return len(text.encode("utf-8"))

    class Graph:
        async def search(self, run_id, learner_id, question):
            return []

    workflow = application_memory_workflow(
        PersonalMemoryWorkflow, Archive(), Runtime(), Graph(),
        full_lesson_max_bytes=1, source_order=["old-hash", "recent-b-hash", "recent-c-hash"],
    )
    result = asyncio.run(workflow.search("run", "child", "вопрос", mode="graphiti", budget_tokens=8000))
    assert "old-c1-student" in result["source_ids"]
    assert "old-c6-student" in result["source_ids"]
    assert "old-c5-student" in result["source_ids"]  # Budget permits evidence past the old four-fragment cap.
    assert "recent-b-hash-student" in result["source_ids"]
    assert "recent-c-hash-student" in result["source_ids"]
    assert "old-hash-student" not in result["source_ids"]


def test_application_selection_of_a_correction_keeps_its_approved_source_closure():
    value = snapshot()
    lesson = source_lessons(value, "child-a")[0][1]
    found = select_application_candidates(
        [lesson], [{"kind": "semantic", "lesson_ids": [lesson["lesson_id"]],
                    "chunk_id": lesson["lesson_id"] + ":c0001"}], 3000,
    )
    context = select_context(found, value, "child-a", "math", allowed_ids={"t1", "t2"})
    assert context["evidence_ids"] == ["t1", "t2"]
    assert "Уточнение" in context["text"]
    assert context["selection_version"] == "app_budgeted_sources_v2"


def test_application_selection_still_rejects_a_new_or_foreign_source_id():
    value = snapshot()
    lesson = source_lessons(value, "child-a")[0][0]
    altered = deepcopy(lesson)
    altered["turns"][-1] = {**altered["turns"][-1], "turn_id": "foreign-source"}
    found = select_application_candidates(
        [altered], [{"kind": "semantic", "lesson_ids": [lesson["lesson_id"]],
                     "chunk_id": lesson["lesson_id"] + ":c0001"}], 3000,
    )
    with pytest.raises(ValueError, match="canonical snapshot"):
        select_context(found, value, "child-a", "math")


def test_chunkless_graph_hit_is_not_reported_as_found_original_evidence():
    value = snapshot()
    lesson = source_lessons(value, "child-a")[0][0]
    fallback = {**lesson, "fragment_id": lesson["lesson_id"] + ":fallback"}
    found = select_application_candidates(
        [fallback], [{"kind": "graphiti", "lesson_ids": [lesson["lesson_id"]], "text": "derived claim"}], 3000,
    )
    assert found["memory"] == ""
    assert found["source_ids"] == []
    assert found["skipped"][0]["reason"] == "no_exact_source_chunk_match"
    with pytest.raises(PersonalGraphSelectionBudgetExceeded, match="оригинальных реплик"):
        select_context(found, value, "child-a", "math")


def test_cross_topic_final_context_is_tracked_for_freshness():
    value = snapshot()
    value["sources"][0]["topic_id"] = "reading.other_topic"
    value["sources"][0]["session_end_context"] = {
        "role": "assistant", "usage": "context_only", "text": "Вернёмся к этому вопросу.",
        "assistant_source_id": "assistant-end", "session_id": "lesson-1",
        "excerpt_mode": "whole", "truncated": False,
    }
    context = select_context(found_for(value, 0), value, "child-a", "math")
    assert "Вернёмся к этому вопросу." in context["text"]
    assert len(context["session_end_context"]) == 1
    assert context["session_end_context"][0]["source_id"] == "s1"


def test_rehydrated_dialogue_does_not_include_unapproved_sources():
    value = snapshot()
    value["sources"].append({"source_id": "s2", "author_role": "student", "text": "Другой ответ",
        "topic_id": "math", "observed_on": "2026-01-04", "session_id": "lesson-2",
        "dialogue_context": {"role": "assistant", "text": "Контекст второго источника"}})
    context = select_context(found_for(value, 0, 3), value, "child-a", "math", allowed_ids={"s1"})
    assert context["evidence_ids"] == ["s1"]
    assert "Почему так получилось?" in context["text"]
    assert "Контекст второго источника" not in context["text"]


def test_rehydrated_context_must_fit_the_complete_memory_budget():
    value = snapshot()
    value["sources"][0]["text"] = "a" * MEMORY_BYTE_BUDGET
    value["sources"][0]["dialogue_context"] = {"role": "assistant", "text": "b" * 600}
    lesson = source_lessons(value, "child-a")[0][0]
    lesson["turns"] = [turn for turn in lesson["turns"] if turn["role"] == "student"]
    found = {"memory": json.dumps([{"original_lesson": lesson}])}
    with pytest.raises(ValueError, match="не поместился"):
        select_context(found, value, "child-a", "math", allowed_ids={"s1"})


def test_personal_graph_can_keep_multiple_whole_sources_beyond_the_old_byte_limit():
    from local_app.rag import RagService

    value = snapshot()
    value["sources"] = [
        {"source_id": f"s{index}", "author_role": "student",
         "text": f"Занятие {index}: " + "заметка о решении и проверке. " * 25,
         "topic_id": "math", "observed_on": f"2026-01-0{index}", "session_id": f"lesson-{index}"}
        for index in range(1, 5)
    ]
    found = found_for(value, 0, 1, 2, 3)
    context = select_context(found, value, "child-a", "math")
    assert 4000 < len(context["text"].encode("utf-8")) <= MEMORY_BYTE_BUDGET
    assert set(context["evidence_ids"]) == {f"s{index}" for index in range(1, 5)}
    assert RagService._approved_memory_text(context) == context["text"]
    assert all(source["text"] in context["text"] for source in value["sources"])


def test_chat_accepts_expanded_graph_memory_without_expanding_legacy_memory():
    from local_app.rag import RagService

    expanded = "a" * 6000
    assert RagService._approved_memory_text({"method": "personal_graph", "text": expanded}, context_tokens=16384) == expanded
    with pytest.raises(ValueError, match="budget"):
        RagService._approved_memory_text({"method": "tail", "text": expanded})
    with pytest.raises(ValueError, match="budget"):
        RagService._approved_memory_text({"method": "personal_graph", "text": "я" * MEMORY_BYTE_BUDGET}, context_tokens=16384)


def test_query_resolver_receives_the_whole_expanded_graph_memory(monkeypatch):
    from local_app.config import Settings
    from local_app.runtime import OllamaRuntime

    runtime = OllamaRuntime(Settings(context_tokens=16384))
    captured = []
    def fake_chat_json(messages, **_kwargs):
        captured.append(json.loads(messages[1]["content"]))
        return {"in_scope": True, "uses_prior_context": False,
                "search_query": "дроби", "uses_approved_memory": True}
    monkeypatch.setattr(runtime, "chat_json", fake_chat_json)
    memory = {"method": "personal_graph", "text": "a" * 6000,
              "topic_id": "math", "version_id": "v1"}
    result = runtime.resolve_chat_query("Что было на прошлом занятии?", [], "math",
                                        approved_memory=memory, topic_id="math")
    assert result["uses_approved_memory"] is True
    assert captured[0]["approved_memory_untrusted"]["text"] == memory["text"]
    with pytest.raises(ValueError, match="memory"):
        runtime.resolve_chat_query("Вопрос", [], "math", approved_memory={**memory, "method": "tail"}, topic_id="math")
    with pytest.raises(ValueError, match="memory"):
        runtime.resolve_chat_query("Вопрос", [], "math",
            approved_memory={**memory, "text": "я" * MEMORY_BYTE_BUDGET}, topic_id="math")


def test_32k_graph_keeps_more_approved_sources_with_their_full_text():
    from local_app.personal_graph_contract import personal_graph_budgets
    from local_app.rag import RagService

    value = snapshot()
    value["sources"] = [
        {"source_id": f"s{index}", "author_role": "student", "text": f"Занятие {index}: " + "a" * 1400,
         "topic_id": "math", "observed_on": "2026-01-01", "session_id": f"lesson-{index}"}
        for index in range(12)
    ]
    found = found_for(value, *range(12))
    with pytest.raises(PersonalGraphSelectionBudgetExceeded):
        select_context(found, value, "child-a", "math")
    selection_limit, memory_limit = personal_graph_budgets(32768)
    assert selection_limit == 20000 and memory_limit == 24000
    context = select_context(found, value, "child-a", "math", memory_byte_budget=memory_limit)
    assert MEMORY_BYTE_BUDGET < len(context["text"].encode("utf-8")) <= memory_limit
    assert len(context["evidence_ids"]) == 12
    assert RagService._approved_memory_text(context, context_tokens=32768) == context["text"]
    with pytest.raises(ValueError, match="budget"):
        RagService._approved_memory_text(context, context_tokens=16384)


def test_graph_query_resolver_respects_the_32k_memory_limit(monkeypatch):
    from local_app.config import Settings
    from local_app.runtime import OllamaRuntime

    runtime = OllamaRuntime(Settings(context_tokens=32768))
    received = []
    def fake_chat_json(messages, **_kwargs):
        received.append(json.loads(messages[1]["content"]))
        return {"in_scope": True, "uses_prior_context": False,
                "search_query": "дроби", "uses_approved_memory": True}
    monkeypatch.setattr(runtime, "chat_json", fake_chat_json)
    memory = {"method": "personal_graph", "text": "a" * 16000,
              "topic_id": "math", "version_id": "v1"}
    assert runtime.resolve_chat_query("Что было раньше?", [], "math", approved_memory=memory,
                                      topic_id="math")["uses_approved_memory"]
    assert received[0]["approved_memory_untrusted"]["text"] == memory["text"]
    with pytest.raises(ValueError, match="budget"):
        runtime.resolve_chat_query("Вопрос", [], "math",
            approved_memory={**memory, "text": "я" * 12001}, topic_id="math")


def test_budget_skips_are_reported_and_cannot_create_empty_approved_memory():
    value = snapshot()
    found = found_for(value, 0)
    found["skipped"] = [{"lesson_id": "unfitted", "reason": "memory_budget_or_context_exceeded"}]
    context = select_context(found, value, "child-a", "math")
    assert context["selection_warnings"][0]["count"] == 1
    assert "не вошли" in context["text"]
    with pytest.raises(ValueError, match="не поместились"):
        select_context({"memory": "[]", "skipped": found["skipped"]}, value, "child-a", "math")


def test_unapproved_and_altered_sources_are_rejected():
    value = snapshot()
    wrong = deepcopy(value)
    wrong["sources"][1]["approved"] = False
    with pytest.raises(ValueError, match="not approved"):
        source_lessons(wrong, "child-a")
    with pytest.raises(ValueError, match="another learner"):
        source_lessons(value, "child-b")
    found = found_for(value, 0)
    parsed = json.loads(found["memory"])
    parsed[0]["original_lesson"]["turns"][-1]["text"] = "changed"
    found["memory"] = json.dumps(parsed)
    with pytest.raises(ValueError, match="canonical snapshot"):
        select_context(found, value, "child-a", "math")


def test_retrieved_turns_cannot_be_moved_between_lessons_or_given_another_date():
    value = snapshot()
    found = found_for(value, 0)
    blocks = json.loads(found["memory"])
    blocks[0]["original_lesson"]["observed_on"] = "2026-01-02"
    found["memory"] = json.dumps(blocks)
    with pytest.raises(ValueError, match="canonical snapshot"):
        select_context(found, value, "child-a", "math")
    lessons, _ = source_lessons(value, "child-a")
    altered = deepcopy(lessons[0])
    altered["turns"].append(lessons[1]["turns"][0])
    found["memory"] = json.dumps([{"original_lesson": altered}])
    with pytest.raises(ValueError, match="canonical snapshot"):
        select_context(found, value, "child-a", "math")
    altered = deepcopy(lessons[0])
    altered["turns"].reverse()
    found["memory"] = json.dumps([{"original_lesson": altered}])
    with pytest.raises(ValueError, match="canonical snapshot"):
        select_context(found, value, "child-a", "math")


def test_chat_cannot_use_sources_outside_the_approved_context():
    value = snapshot()
    context = select_context(found_for(value, 0, 1), value, "child-a", "math", allowed_ids={"s1"})
    assert context["evidence_ids"] == ["s1"]
    assert "Ребёнок ошибся" not in context["text"]


def test_source_scoped_search_ranks_only_allowed_turn_text():
    value = snapshot()
    value["sources"][0]["text"] = "permitted needle answer"
    for source in value["sources"][1:]:
        source.update(session_id="lesson-1", observed_on="2026-01-01", topic_id="secret-topic",
                      text="forbidden private marker")

    class Embedder:
        def __init__(self):
            self.inputs = []

        def embed_many(self, texts):
            self.inputs.extend(texts)
            return [([1.0, 0.0] if "needle" in text.casefold() else [0.0, 1.0]) + [0.0] * 1022 for text in texts]

    first_model = Embedder()
    first_lessons, _ = source_lessons(value, "child-a")
    first_found = source_scoped_search("needle question", first_lessons, {"s1"}, first_model.embed_many)
    first_context = select_context(first_found, value, "child-a", "math", allowed_ids={"s1"})

    changed = deepcopy(value)
    changed["sources"][1]["text"] = "changed hidden wording"
    changed["sources"][1]["topic_id"] = "different-secret-topic"
    changed_model = Embedder()
    changed_lessons, _ = source_lessons(changed, "child-a")
    changed_found = source_scoped_search("needle question", changed_lessons, {"s1"}, changed_model.embed_many)
    changed_context = select_context(changed_found, changed, "child-a", "math", allowed_ids={"s1"})

    assert first_model.inputs == changed_model.inputs
    assert all("forbidden" not in text and "secret-topic" not in text for text in first_model.inputs)
    assert first_context["text"] == changed_context["text"]
    assert first_context["evidence_ids"] == changed_context["evidence_ids"] == ["s1"]


def test_source_scoped_search_uses_only_context_anchored_to_allowed_turns():
    value = {"learner_id": "child-a", "sources": []}
    for index in range(11):
        value["sources"].append({"source_id": f"a-source-{index:02d}", "author_role": "student",
            "text": "ага", "topic_id": "math", "observed_on": "2026-01-01",
            "session_id": f"session-{index:02d}",
            "dialogue_context": {"role": "assistant", "text": f"Другой вопрос {index}?"}})
    value["sources"].append({"source_id": "z-target", "author_role": "student", "text": "хз",
        "topic_id": "math", "observed_on": "2026-01-01", "session_id": "session-target",
        "dialogue_context": {"role": "assistant", "text": "Почему так получилось?"}})
    lessons, _ = source_lessons(value, "child-a")

    class Embedder:
        def embed_many(self, texts):
            return [([1.0, 0.0] if "Почему так получилось?" in text else [0.0, 1.0]) + [0.0] * 1022 for text in texts]

    allowed = {source["source_id"] for source in value["sources"]}
    found = source_scoped_search("Почему так получилось?", lessons, allowed, Embedder().embed_many)
    context = select_context(found, value, "child-a", "math", allowed_ids=allowed)

    assert "z-target" in context["evidence_ids"]
    assert "Почему так получилось?" in context["text"]
    payload = json.loads(context["text"])
    for lesson in payload["lessons"] + payload.get("related_context", []):
        assert all(turn.get("anchor_source_id") in context["evidence_ids"]
                   for turn in lesson["turns"] if turn.get("role") == "assistant")


def test_selected_context_chat_skips_graphiti_and_uses_source_scoped_search(monkeypatch):
    value = snapshot()
    value["sources"][0]["text"] = "permitted needle answer"
    value["sources"][1]["text"] = "forbidden private marker"

    class Embedder:
        def __init__(self):
            self.inputs = []

        def embed_many(self, texts):
            self.inputs.extend(texts)
            return [([1.0, 0.0] if "needle" in text.casefold() else [0.0, 1.0]) + [0.0] * 1022 for text in texts]

    embedder = Embedder()
    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: embedder)
    monkeypatch.setattr(provider, "_objects", lambda **kwargs: pytest.fail("selected context must not open Graphiti"))
    version = {"id": "v-selected", "state": {}, "context": {"evidence_ids": ["s1"]}}

    context = asyncio.run(provider._chat(value, "child-a", "math", "needle question", version))

    assert context["evidence_ids"] == ["s1"]
    assert "forbidden" not in context["text"]
    assert all("forbidden" not in text for text in embedder.inputs)


def test_selected_context_chat_rejects_a_correction_chain_outside_the_approved_ids(monkeypatch):
    value = snapshot()
    version = {"id": "v-old-selected", "method": "personal_graph", "approved": True,
        "topic_id": "math", "base_revision": 2, "source_revision": 7,
        "context": {"evidence_ids": ["t1"]},
        "state": {"method": "personal_graph", "learner_id": "child-a", "archive_run_id": "app-old",
            "source_revision": 7, "source_ids": ["s1", "t1", "t2"],
            "source_context_protocol": PERSONAL_GRAPH_CONTEXT_PROTOCOL}}
    value["versions"] = [version]
    assert approved_graph_version(value, "child-a", "math") is None

    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: pytest.fail("must reject before retrieval"))
    with pytest.raises(ValueError, match="new approved memory version"):
        asyncio.run(provider._chat(value, "child-a", "math", "question", version))


def test_selected_context_chat_reselects_sources_to_fit_the_current_context_budget(monkeypatch):
    value = {"learner_id": "child-a", "sources": [
        {"source_id": source_id, "author_role": "student", "text": "x" * 4500,
         "topic_id": "math", "observed_on": "2026-01-01", "session_id": source_id}
        for source_id in ("a-long", "b-long", "c-long")
    ]}
    lessons, _ = source_lessons(value, "child-a")
    allowed = {source["source_id"] for source in value["sources"]}
    at_32k = select_context(source_scoped_found(lessons, allowed), value, "child-a", "math",
                            allowed_ids=allowed, memory_byte_budget=24000)
    assert set(at_32k["evidence_ids"]) == allowed

    class Embedder:
        def embed_many(self, texts):
            return [[1.0, 0.0] + [0.0] * 1022 for _ in texts]

    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: Embedder())
    version = {"id": "v-32k", "state": {}, "context": {"evidence_ids": sorted(allowed)}}
    context = asyncio.run(provider._chat(value, "child-a", "math", "needle", version))

    # All three sources tie on score and date; the merged ranking breaks ties newest-first (later lesson first),
    # so the two that fit the budget are b and c. The budget behaviour itself is what this test is about.
    assert context["evidence_ids"] == ["b-long", "c-long"]
    assert context["selection_budget"]["limit"] == 12000
    assert context["selection_warnings"] == [{"code": "memory_budget_exceeded", "count": 1}]
    assert "Некоторые кандидаты не вошли в память" in context["text"]


def test_selected_context_chat_skips_an_oversized_correction_chain_atomically(monkeypatch):
    value = {"learner_id": "child-a", "sources": [
        {"source_id": "a-correction", "author_role": "teacher", "approved": True,
         "text": "c" * 5600, "topic_id": "math", "observed_on": "2026-01-01",
         "corrects_source_id": "b-original"},
        {"source_id": "b-original", "author_role": "student", "text": "o" * 5600,
         "topic_id": "math", "observed_on": "2026-01-02"},
        {"source_id": "c-short", "author_role": "student", "text": "short answer",
         "topic_id": "math", "observed_on": "2026-01-03"},
    ]}
    lessons, _ = source_lessons(value, "child-a")
    allowed = {source["source_id"] for source in value["sources"]}
    at_32k = select_context(source_scoped_found(lessons, allowed), value, "child-a", "math",
                            allowed_ids=allowed, memory_byte_budget=24000)
    assert set(at_32k["evidence_ids"]) == allowed

    class Embedder:
        def embed_many(self, texts):
            return [[1.0, 0.0] + [0.0] * 1022 for _ in texts]

    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: Embedder())
    version = {"id": "v-corrections", "state": {}, "context": {"evidence_ids": sorted(allowed)}}
    context = asyncio.run(provider._chat(value, "child-a", "math", "needle", version))

    assert context["evidence_ids"] == ["c-short"]
    assert context["selection_warnings"] == [{"code": "memory_budget_exceeded", "count": 2}]
    assert "a-correction" not in context["evidence_ids"]
    assert "b-original" not in context["evidence_ids"]


def test_budget_warning_overhead_drops_a_source_instead_of_overflowing(monkeypatch):
    value = {"learner_id": "child-a", "sources": [
        {"source_id": source_id, "author_role": "student", "text": "x" * 4000,
         "topic_id": "math", "observed_on": "2026-01-01", "session_id": source_id}
        for source_id in ("a", "b", "c")
    ]}
    lessons, _ = source_lessons(value, "child-a")
    allowed = {"a", "b", "c"}
    without_warning = select_context(source_scoped_found(lessons, {"a", "b"}), value, "child-a", "math",
                                    allowed_ids=allowed, memory_byte_budget=100000)
    with_warning = select_context(source_scoped_found(lessons, {"a", "b"}, skipped_source_ids={"c"}),
                                  value, "child-a", "math", allowed_ids=allowed, memory_byte_budget=100000)
    raw_size = without_warning["selection_budget"]["used"]
    warning_size = with_warning["selection_budget"]["used"]
    assert raw_size < warning_size
    budget = (raw_size + warning_size) // 2
    assert raw_size < budget < warning_size
    monkeypatch.setattr("local_app.personal_graph_memory.personal_graph_budgets",
                        lambda _context: (10000, budget))

    class Embedder:
        def embed_many(self, texts):
            return [[1.0, 0.0] + [0.0] * 1022 for _ in texts]

    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: Embedder())
    version = {"id": "v-warning-budget", "state": {}, "context": {"evidence_ids": ["a", "b", "c"]}}
    context = asyncio.run(provider._chat(value, "child-a", "math", "needle", version))

    assert context["evidence_ids"] == ["c"]  # ties are broken newest-first by the merged ranking (was "a" by turn id)
    assert context["selection_budget"]["used"] <= budget
    assert context["selection_warnings"] == [{"code": "memory_budget_exceeded", "count": 2}]


def test_source_already_in_accepted_correction_closure_is_not_added_twice(monkeypatch):
    value = {"learner_id": "child-a", "sources": [
        {"source_id": "a-original", "author_role": "student", "text": "o" * 5000,
         "topic_id": "math", "observed_on": "2026-01-01"},
        {"source_id": "b-correction", "author_role": "teacher", "approved": True,
         "text": "c" * 5800, "topic_id": "math", "observed_on": "2026-01-02",
         "corrects_source_id": "a-original"},
    ]}
    lessons, _ = source_lessons(value, "child-a")
    allowed = {"a-original", "b-correction"}
    original_first = select_context(source_scoped_found(lessons, {"a-original"}), value, "child-a", "math",
                                   allowed_ids=allowed, memory_byte_budget=100000)
    both_direct = select_context(source_scoped_found(lessons, allowed), value, "child-a", "math",
                                 allowed_ids=allowed, memory_byte_budget=100000)
    closure_size = original_first["selection_budget"]["used"]
    direct_size = both_direct["selection_budget"]["used"]
    assert closure_size < direct_size
    budget = (closure_size + direct_size) // 2
    assert closure_size < budget < direct_size
    monkeypatch.setattr("local_app.personal_graph_memory.personal_graph_budgets",
                        lambda _context: (10000, budget))

    class Embedder:
        def embed_many(self, texts):
            return [[1.0, 0.0] + [0.0] * 1022 for _ in texts]

    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: Embedder())
    version = {"id": "v-correction-closure", "state": {},
               "context": {"evidence_ids": sorted(allowed)}}
    context = asyncio.run(provider._chat(value, "child-a", "math", "needle", version))

    assert set(context["evidence_ids"]) == allowed
    assert context["selection_budget"]["used"] <= budget
    assert context["selection_warnings"] == []


def test_budget_search_tries_another_representative_of_a_correction_group(monkeypatch):
    value = {"learner_id": "child-a", "sources": [
        {"source_id": "a-correction", "author_role": "teacher", "approved": True,
         "text": "а" * 1900, "topic_id": "math", "observed_on": "2026-01-02",
         "corrects_source_id": "b-original"},
        {"source_id": "b-original", "author_role": "student", "text": "я" * 1700,
         "topic_id": "math", "observed_on": "2026-01-01",
         "dialogue_context": {"role": "assistant", "text": "Почему так получилось?"}},
    ]}
    lessons, _ = source_lessons(value, "child-a")
    allowed = {"a-correction", "b-original"}
    correction_direct = select_context(source_scoped_found(lessons, {"a-correction"}), value,
                                       "child-a", "math", allowed_ids=allowed, memory_byte_budget=100000)
    original_direct = select_context(source_scoped_found(lessons, {"b-original"}), value,
                                     "child-a", "math", allowed_ids=allowed, memory_byte_budget=100000)
    correction_size = correction_direct["selection_budget"]["used"]
    original_size = original_direct["selection_budget"]["used"]
    assert original_size < correction_size
    budget = (original_size + correction_size) // 2
    assert original_size < budget < correction_size
    monkeypatch.setattr("local_app.personal_graph_memory.personal_graph_budgets",
                        lambda _context: (10000, budget))

    class Embedder:
        def embed_many(self, texts):
            return [[1.0, 0.0] + [0.0] * 1022 for _ in texts]

    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_application_model", lambda: Embedder())
    version = {"id": "v-alt-correction", "state": {}, "context": {"evidence_ids": sorted(allowed)}}
    context = asyncio.run(provider._chat(value, "child-a", "math", "needle", version))

    assert set(context["evidence_ids"]) == allowed
    assert context["selection_budget"]["used"] == original_size
    assert "Почему так получилось?" in context["text"]
    assert context["selection_warnings"] == []


def test_full_archive_chat_keeps_graphiti_search_for_the_reviewed_snapshot(monkeypatch):
    from local_app import personal_graph_index as index
    value = snapshot()
    source_ids = [source["source_id"] for source in value["sources"]]
    fingerprint = personal_graph_snapshot_fingerprint(value["sources"])
    manifest = {"graph_node_count": 1}

    class Graph:
        closed = False

        async def close(self):
            self.closed = True

    class Workflow:
        calls = []

        async def search(self, run_id, learner_id, question, **kwargs):
            self.calls.append((run_id, learner_id, question, kwargs))
            return found_for(value, 0, 1, 2)

    graph, workflow = Graph(), Workflow()
    model = SimpleNamespace(model_identities={"bound": "model"}, check_model_identities=lambda: None)
    provider = PersonalGraphProvider(SimpleNamespace(dsn="synthetic"), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, "_objects", lambda **kwargs: (model, object(), graph, None, workflow))
    monkeypatch.setattr(index, 'database_fingerprint', lambda dsn: 'db')
    monkeypatch.setattr(index, 'personal_graph_config_fingerprint', lambda runtime, identities: 'config')
    checked = []
    monkeypatch.setattr(index, '_ready_index_manifest',
                        lambda dsn, run, key: checked.append((run, key)) or manifest)
    async def healthy(*args, **kwargs): return manifest
    monkeypatch.setattr(index, '_health', healthy)
    version = {"id": "v-full", "state": {"archive_run_id": "app-run",
               "archive_snapshot_fingerprint": fingerprint, "index_config_fingerprint": "config",
               "index_manifest": manifest},
               "context": {"evidence_ids": ["s1"], "source_access": {
                   "mode": "full_archive", "source_ids": source_ids, "snapshot_fingerprint": fingerprint}}}

    context = asyncio.run(provider._chat(value, "child-a", "math", "question", version))

    assert context["evidence_ids"] == ["s1", "t1", "t2"]
    assert len(checked) == 2 and checked[0][0] == "app-run"
    assert workflow.calls[0][3]["allowed_source_ids"] == set(source_ids)
    assert graph.closed


def test_old_full_archive_without_index_metadata_requires_refresh_before_opening_graph(monkeypatch):
    from local_app.personal_graph_index import PersonalGraphIndexStale
    value = snapshot()
    fingerprint = personal_graph_snapshot_fingerprint(value['sources'])
    provider = PersonalGraphProvider(object(), SimpleNamespace(context_tokens=16384))
    monkeypatch.setattr(provider, '_objects', lambda **kwargs: pytest.fail('old index must not be searched'))
    version = {'state': {'archive_run_id': 'pre-registry', 'archive_snapshot_fingerprint': fingerprint},
               'context': {'source_access': {'mode': 'full_archive', 'snapshot_fingerprint': fingerprint,
                                             'source_ids': [source['source_id'] for source in value['sources']]}}}
    with pytest.raises(PersonalGraphIndexStale, match='обновить'):
        asyncio.run(provider._chat(value, 'child-a', 'math', 'question', version))


def test_only_fresh_approved_version_can_be_used_for_chat():
    value = snapshot()
    version = {"id": "v1", "method": "personal_graph", "approved": True, "topic_id": "math",
               "base_revision": 2, "source_revision": 7, "context": {"evidence_ids": ["s1"]},
               "state": {"method": "personal_graph", "learner_id": "child-a", "archive_run_id": "app-1",
                         "source_revision": 7, "source_ids": ["s1", "t1", "t2"],
                         "source_context_protocol": PERSONAL_GRAPH_CONTEXT_PROTOCOL}}
    value["versions"] = [version]
    assert approved_graph_version(value, "child-a", "math") == version
    for key, replacement in (("approved", False), ("source_revision", 6), ("base_revision", 1), ("stale", True)):
        changed = deepcopy(value)
        changed["versions"][0][key] = replacement
        assert approved_graph_version(changed, "child-a", "math") is None
    assert approved_graph_version(value, "child-a", "other") is None
    legacy = deepcopy(value)
    legacy["versions"][0]["state"].pop("source_context_protocol")
    assert approved_graph_version(legacy, "child-a", "math") is None
    # An active, not-yet-eligible conversation may advance the revision without
    # changing the teacher-approved completed source snapshot.
    active = deepcopy(value)
    active["source_revision"] = 8
    assert approved_graph_version(active, "child-a", "math") == version
    active["sources"].append({"source_id": "new-completed"})
    assert approved_graph_version(active, "child-a", "math") is None


def test_full_archive_access_is_bound_to_the_approved_source_snapshot():
    value = snapshot()
    source_ids = sorted(source["source_id"] for source in value["sources"])
    fingerprint = personal_graph_snapshot_fingerprint(value["sources"])
    version = {"id": "v-full", "method": "personal_graph", "approved": True, "topic_id": "math",
        "base_revision": 2, "source_revision": 7,
        "context": {"evidence_ids": ["s1"], "source_access": {"mode": "full_archive",
            "source_ids": source_ids, "snapshot_fingerprint": fingerprint}},
        "state": {"method": "personal_graph", "learner_id": "child-a", "archive_run_id": "app-1",
            "source_revision": 7, "source_ids": source_ids, "archive_snapshot_fingerprint": fingerprint,
            "source_context_protocol": PERSONAL_GRAPH_CONTEXT_PROTOCOL}}
    value["versions"] = [version]
    assert approved_graph_version(value, "child-a", "math") == version

    changed_text = deepcopy(value)
    changed_text["sources"][1]["text"] += " changed"
    assert approved_graph_version(changed_text, "child-a", "math") is None

    changed_question = deepcopy(value)
    changed_question["sources"][0]["dialogue_context"]["text"] += " changed"
    assert approved_graph_version(changed_question, "child-a", "math") is None

    incomplete_access = deepcopy(value)
    incomplete_access["versions"][0]["context"]["source_access"]["source_ids"].pop()
    assert approved_graph_version(incomplete_access, "child-a", "math") is None


def test_snapshot_fingerprint_tracks_order_corrections_and_session_end_context():
    value = snapshot()
    baseline = personal_graph_snapshot_fingerprint(value["sources"])
    reversed_sources = list(reversed(value["sources"]))
    assert personal_graph_snapshot_fingerprint(reversed_sources) != baseline

    ending = deepcopy(value["sources"])
    ending[0]["session_end_context"] = {"role": "assistant", "usage": "context_only", "text": "Продолжим?"}
    assert personal_graph_snapshot_fingerprint(ending) != baseline

    correction = deepcopy(value["sources"])
    correction[1]["corrects_source_id"] = "s1"
    assert personal_graph_snapshot_fingerprint(correction) != baseline


def test_store_snapshot_fingerprint_uses_eligible_sources_and_attached_context():
    from datetime import date, datetime

    class Cursor:
        def __init__(self):
            self.rows = []
            self.queries = []
        def execute(self, query, params):
            self.queries.append(query)
            if "WITH latest AS" in query:
                self.rows = [{"source_id": "s1", "assistant_source_id": "assistant-end",
                    "session_id": "session-1", "text": "Завершим на этом."}]
            else:
                self.rows = [
                    {"source_id": "s1", "session_id": "session-1", "author_role": "student",
                     "text": "Ответ ученика.", "observed_on": None, "created_at": datetime(2026, 1, 1, 12), "topic_id": "math",
                     "approved": False, "corrects_source_id": None,
                     "preceding_assistant_text": "Как ты рассуждал?", "preceding_assistant_truncated": False},
                    {"source_id": "t1", "session_id": None, "author_role": "teacher",
                     "text": "Уточнение преподавателя.", "observed_on": date(2026, 1, 2), "created_at": datetime(2026, 1, 2, 12), "topic_id": "math",
                     "approved": True, "corrects_source_id": "s1",
                     "preceding_assistant_text": None, "preceding_assistant_truncated": False},
                ]
        def fetchall(self): return self.rows

    cursor = Cursor()
    source_ids, fingerprint = MemoryStoreMixin._current_personal_graph_snapshot(cursor, "child-a")
    expected = [
        {"source_id": "s1", "session_id": "session-1", "author_role": "student", "text": "Ответ ученика.",
         "observed_on": None, "created_at": datetime(2026, 1, 1, 12), "topic_id": "math", "approved": False,
         "dialogue_context": {"role": "assistant", "usage": "context_only", "text": "Как ты рассуждал?", "excerpt_mode": "tail", "truncated": False},
         "session_end_context": {"assistant_source_id": "assistant-end", "session_id": "session-1",
             "role": "assistant", "usage": "context_only", "text": "Завершим на этом.", "excerpt_mode": "whole", "truncated": False}},
        {"source_id": "t1", "session_id": None, "author_role": "teacher", "text": "Уточнение преподавателя.",
         "observed_on": date(2026, 1, 2), "created_at": datetime(2026, 1, 2, 12), "topic_id": "math", "approved": True, "corrects_source_id": "s1"},
    ]
    assert source_ids == ["s1", "t1"]
    assert fingerprint == personal_graph_snapshot_fingerprint(expected)
    assert "ls.status='completed'" in cursor.queries[0]
    assert "learning_messages m" in cursor.queries[0]
    assert "m.status <> 'completed'" in cursor.queries[0]
    assert "AND NOT EXISTS (SELECT 1 FROM teachcopilot.learning_messages m" in cursor.queries[0]
    assert "ORDER BY s.source_order" in cursor.queries[0]


def test_memory_snapshot_drops_entire_correction_chain_when_target_is_ineligible():
    sources = [
        {"source_id": "eligible-student", "corrects_source_id": None},
        {"source_id": "orphan-correction", "corrects_source_id": "failed-student"},
        {"source_id": "correction-of-orphan", "corrects_source_id": "orphan-correction"},
        {"source_id": "independent-teacher-note", "corrects_source_id": None},
    ]
    result = MemoryStoreMixin._remove_orphaned_memory_corrections(sources)
    assert [item["source_id"] for item in result] == ["eligible-student", "independent-teacher-note"]


def test_graph_snapshot_read_fails_closed_for_legacy_store_contracts():
    class LegacyStore:
        def get_learner_memory(self, learner_id, topic_id=None):
            return {"learner_id": learner_id, "sources": [{"source_id": "unfinished"}]}

    legacy = LegacyStore()
    assert get_learner_memory_snapshot(legacy, "child-a")["sources"][0]["source_id"] == "unfinished"
    with pytest.raises(RuntimeError, match="personal_graph_source_snapshot_unavailable"):
        get_learner_memory_snapshot(legacy, "child-a", include_personal_graph_snapshot=True)


def test_application_model_uses_injected_journalled_writer_and_context_limit():
    calls = []
    runtime = SimpleNamespace(context_tokens=16384,
        settings=SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                 ollama_url="http://127.0.0.1:11435"),
        embed=lambda texts: [[0.0] * 1024 for _ in texts])
    model = ApplicationModel(runtime, lambda messages, purpose, schema: calls.append((purpose, schema)) or "{}")
    result = model._post("/api/chat", {"messages": [], "format": {"type": "object"}}, "graphiti_llm")
    assert result["message"]["content"] == "{}" and calls[0][0] == "personal_graph_extract"
    with pytest.raises(RuntimeError):
        ApplicationModel(runtime)._post("/api/chat", {"messages": []}, "graphiti_llm")
    runtime.context_tokens = 65536
    with pytest.raises(ValueError, match="16k or 32k"):
        ApplicationModel(runtime)
