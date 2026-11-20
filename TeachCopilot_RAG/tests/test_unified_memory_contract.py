"""Contract-level unit tests for the unified memory backend (client A).

These run without PostgreSQL: pure validation, projection and rendering
logic. Real-database behavior lives in test_unified_memory_postgres.py.
"""
import json

import pytest

from local_app.unified_memory import (
    UnifiedMemoryConflict,
    UnifiedMemoryValidationError,
    _candidate_public,
    _change_public,
)
from local_app.unified_memory_processing import (
    EXTRACT_SCHEMA,
    SUMMARY_CHUNK_SCHEMA,
    normalize_chunk_summary,
    reduce_chunk_summaries,
    source_fingerprint,
    summarize_text,
    validate_extraction,
)
from local_app.unified_memory_store import parse_answer_sections, redact_lesson_text

SOURCE = {"source_id": "src-1", "author_role": "student", "text": "Я люблю Minecraft и хочу задачу про роботов.",
          "session_id": "sess-1", "delivery_status": "completed"}
CHUNK = {"chunk_id": "chunk-1", "source_ids": ["src-1"]}
SOURCES_BY_ID = {"src-1": SOURCE}


def extract(payload):
    return validate_extraction(payload, CHUNK, SOURCES_BY_ID)


def test_summary_chunk_schema_is_strict():
    assert SUMMARY_CHUNK_SCHEMA["additionalProperties"] is False
    assert set(SUMMARY_CHUNK_SCHEMA["required"]) == {"studied", "achieved", "repeat", "next_step"}
    assert EXTRACT_SCHEMA["additionalProperties"] is False


def test_summarize_text_uses_the_four_lesson_agent_headings():
    text = summarize_text({"studied": ["Дроби"], "achieved": ["Решил две задачи"],
                           "repeat": ["Кратное сравнение"], "next_step": ["Потренировать деление"]})
    assert text.splitlines()[0] == "Что изучали"
    assert "Что получилось" in text
    assert "Что стоит повторить" in text
    assert "Следующий шаг" in text


def test_reduce_merge_is_deterministic_and_bounded():
    merged = reduce_chunk_summaries([
        normalize_chunk_summary({"studied": ["A", "A"], "achieved": ["B"], "repeat": [], "next_step": ["N"]}),
        normalize_chunk_summary({"studied": ["A", "C"], "achieved": [], "repeat": ["R"], "next_step": ["N"]}),
    ])
    assert merged["studied"] == ["A", "C"]
    assert merged["achieved"] == ["B"]
    assert merged["repeat"] == ["R"]
    assert merged["next_step"] == ["N"]
    long = normalize_chunk_summary({key: [f"item-{index}" for index in range(50)] for key in
                                    ("studied", "achieved", "repeat", "next_step")})
    assert all(len(long[key]) <= 30 for key in long)


def test_source_fingerprint_is_order_and_content_bound():
    first = [dict(SOURCE), {**SOURCE, "source_id": "src-2"}]
    second = [{**SOURCE, "source_id": "src-2"}, dict(SOURCE)]
    assert source_fingerprint(first) != source_fingerprint(second)
    assert source_fingerprint(first) == source_fingerprint([dict(SOURCE), {**SOURCE, "source_id": "src-2"}])
    assert source_fingerprint([SOURCE]) != source_fingerprint([{**SOURCE, "text": "changed"}])


def test_explicit_durable_preference_applies_with_literal_evidence():
    changes, candidates, warnings = extract({"changes": [{
        "category": "interest", "operation": "add", "value": "Minecraft",
        "scope": "durable", "explicit_durable": True, "source_id": "src-1",
        "evidence_text": "Я люблю Minecraft"}], "candidates": []})
    assert warnings == [] and candidates == []
    assert changes == [{"category": "interest", "operation": "add", "value": "Minecraft",
                        "scope": "durable", "status": "applied", "explicit_durable": True,
                        "source_id": "src-1",
                        "evidence": [{"source_id": "src-1", "text": "Я люблю Minecraft"}],
                        "source_refs": [{"kind": "conversation_source", "source_id": "src-1",
                                         "author_role": "student", "session_id": "sess-1"}]}]


def test_temporary_scope_and_ambiguous_non_explicit_claims_stay_out_of_durable_state():
    # Session-scoped claims are recorded nowhere durable: temporary requests
    # belong to session state that the application layer owns.
    changes, _, warnings = extract({"changes": [{
        "category": "interest", "operation": "add", "value": "задача про роботов",
        "scope": "session", "explicit_durable": True, "source_id": "src-1",
        "evidence_text": "хочу задачу про роботов"}], "candidates": []})
    assert changes == [] and any(w["code"] == "session_scope_not_durable" for w in warnings)
    # Non-explicit durable claim stays a proposal for teacher review.
    changes, _, _ = extract({"changes": [{
        "category": "preference", "operation": "add", "value": "решает в уме",
        "scope": "durable", "explicit_durable": False, "source_id": "src-1",
        "evidence_text": "Я люблю Minecraft"}], "candidates": []})
    assert changes[0]["status"] == "proposed" and changes[0]["explicit_durable"] is False


def test_extraction_prompt_carries_the_required_negative_examples():
    """Semantic filtering is the prompt's job; it must name every required negative."""
    from local_app.unified_memory import _PROMPT_DIR, _PROMPT_FILES
    from local_app.unified_memory_processing import EXTRACT_PROMPT
    prompt = (_PROMPT_DIR / _PROMPT_FILES[EXTRACT_PROMPT]).read_text(encoding="utf-8")
    for needle in ("request for today", "hypothetical", "assistant suggestion",
                   "change of interest", "story prohibition", "not sure", "Never include"):
        assert needle in prompt, needle


def test_fabricated_evidence_is_discarded():
    changes, candidates, warnings = extract({"changes": [{
        "category": "interest", "operation": "add", "value": "космос",
        "scope": "durable", "explicit_durable": True, "source_id": "src-1",
        "evidence_text": "Я обожаю космос"}], "candidates": [{
            "category": "knowledge_observation", "proposed_value": {"topic_id": "math.g3", "note": "считает быстро"},
            "source_id": "src-1", "evidence_text": "Этого текста нет в источнике"}]})
    assert changes == [] and candidates == []
    assert {warning["code"] for warning in warnings} == {"evidence_not_literal"}


def test_unavailable_or_cross_chunk_sources_are_discarded():
    changes, _, warnings = extract({"changes": [{
        "category": "interest", "operation": "add", "value": "хомяки",
        "scope": "durable", "explicit_durable": True, "source_id": "src-other",
        "evidence_text": "хомяки"}], "candidates": []})
    assert changes == [] and any(w["code"] == "unavailable_source" for w in warnings)


def test_numeric_academic_proposals_are_never_stored():
    _, candidates, warnings = extract({"changes": [], "candidates": [{
        "category": "knowledge_observation",
        "proposed_value": {"topic_id": "math.g3", "mastery": 0.8, "note": "считает быстро"},
        "source_id": "src-1", "evidence_text": "Minecraft"}]})
    assert candidates == [] and any(w["code"] == "numeric_proposal_discarded" for w in warnings)


def test_text_academic_candidate_is_kept_with_provenance():
    _, candidates, warnings = extract({"changes": [], "candidates": [{
        "category": "error_pattern",
        "proposed_value": {"error_tag": "путает делитель и вычитаемое", "topic_id": "math.g3",
                           "note": "при кратном сравнении"},
        "source_id": "src-1", "evidence_text": "Minecraft"}]})
    assert warnings == []
    assert candidates[0]["category"] == "error_pattern"
    assert candidates[0]["source_refs"][0]["source_id"] == "src-1"
    assert "mastery" not in json.dumps(candidates)


def test_change_and_candidate_public_projections_keep_provenance():
    change = {"id": "11111111-1111-1111-1111-111111111111", "category": "interest",
              "operation": "add", "value": "Minecraft", "scope": "durable", "status": "applied",
              "explicit_durable": True, "source_refs": [{"kind": "conversation_source"}],
              "evidence": [], "session_id": None, "run_id": None, "created_at": None,
              "decided_by": None}
    public = _change_public(change)
    assert public["change_id"] == change["id"] and public["status"] == "applied"
    candidate = {"id": "22222222-2222-2222-2222-222222222222", "category": "learning_goal",
                 "proposed_value": '{"topic_id": "math.g3"}', "source_refs": "[]",
                 "status": "pending", "session_id": None, "run_id": None, "created_at": None,
                 "decided_by": None, "decision_note": None}
    public = _candidate_public(candidate)
    assert public["proposed_value"] == {"topic_id": "math.g3"} and public["status"] == "pending"


def test_answer_section_parsing_is_defensive_and_redaction_is_exact():
    text = "# Занятие\nПлан урока.\n## Ответы\n1) 5\n2) 7\n## Ход занятия\nПродолжаем."
    sections = parse_answer_sections(text)
    assert [section["heading"] for section in sections] == ["Ответы"]
    redacted, removed = redact_lesson_text(text, sections)
    assert removed and "1) 5" not in redacted and "Ход занятия" in redacted
    # No answer headings: text stays byte-identical.
    plain = "# Занятие\nНет ответов здесь."
    assert parse_answer_sections(plain) == []
    assert redact_lesson_text(plain, [])[0] == plain
    # Malformed stored spans never corrupt the text.
    redacted, removed = redact_lesson_text(text, [{"heading": "x", "start": 5, "end": 3},
                                                  {"heading": "y", "start": 10 ** 9, "end": 10 ** 9 + 1}])
    assert removed is False


def test_service_error_types_exist_for_b_mapping():
    assert issubclass(UnifiedMemoryValidationError, ValueError)
    assert issubclass(UnifiedMemoryConflict, RuntimeError)


def test_service_rejects_instantiation_time_model_work():
    """Import-time side effects would violate the frozen contract."""
    import local_app.unified_memory as module
    import inspect
    source = inspect.getsource(module)
    assert "Store(" not in source.split("class UnifiedMemoryService")[0]
