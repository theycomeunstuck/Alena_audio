"""Authored contract regressions; fixture output is not a measured model result."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.clients import Completion, FixtureClient
from research_memory.contracts import ContractValidationError, extract_schema, validate_action, validate_records
from research_memory.methods import Memory
from research_memory.runner import run
from research_memory.tests.test_stand import TOPIC, lesson, write_inputs


def record(turn_id, quote, **overrides):
    return {"kind": "learning_episode", "topic_id": TOPIC, "event": quote,
            "support_level": "unknown", "independent_recheck": "unknown",
            "evidence_turn_ids": [turn_id], "source_quote": quote,
            "support_evidence": None, "recheck_evidence": None,
            "corrects_record_id": None, **overrides}


def evidence(turn_id, quote):
    return {"source_turn_id": turn_id, "quote": quote}


def encoded(*records):
    return json.dumps({"records": records}, ensure_ascii=False)


class EducationalMemoryRegressions(unittest.TestCase):
    def make_memory(self, outputs, budget=4000):
        outputs = iter(outputs)
        return Memory("structured", lambda *args: next(outputs), budget, TOPIC)

    def assert_code(self, code, callback):
        with self.assertRaises(ContractValidationError) as failed:
            callback()
        self.assertEqual(failed.exception.code, code)

    def test_unknown_help_stays_unknown_without_condition_evidence(self):
        text = "Ответ 18 верный. Условия решения не записаны."
        raw = record("t1", text)
        memory = self.make_memory([encoded(raw)])
        memory.update(lesson(1, text))
        self.assertEqual(memory.records[0]["support_level"], "unknown")
        self.assertIsNone(memory.records[0]["support_evidence"])
        unsupported = {**raw, "support_level": "none"}
        self.assert_code("missing_condition_evidence", lambda: validate_records(
            encoded(unsupported), memory.sources, {"t1"}, TOPIC))

    def test_teacher_note_correction_targets_one_record_not_every_shared_source(self):
        first = "Записано: решил без помощи. Работа остановилась после схемы."
        corrected = "Исправляю заметку: я задавал наводящие вопросы, самостоятельной попытки не было."
        old = record("t1", first, support_level="none", support_evidence=evidence("t1", "без помощи"))
        unfinished = record("t1", "Работа остановилась после схемы.", kind="unfinished_work")
        correction = record("t2", corrected, kind="source_correction", corrects_record_id="r0001",
                            support_level="hint_question", support_evidence=evidence("t2", "наводящие вопросы"),
                            independent_recheck="not_observed",
                            recheck_evidence=evidence("t2", "самостоятельной попытки не было"))
        memory = self.make_memory([encoded(old, unfinished), encoded(correction)])
        memory.update(lesson(1, first))
        memory.update(lesson(2, corrected))
        self.assertEqual(memory.records[0]["source_quote"], first)
        self.assertEqual(memory.records[0]["corrected_by_record_ids"], ["r0003"])
        self.assertEqual(memory.records[1]["corrected_by_record_ids"], [])
        selected = json.loads(memory.context(TOPIC).text)
        self.assertEqual({r["record_id"] for r in selected}, {"r0001", "r0002", "r0003"})

    def test_real_student_error_then_fix_is_an_episode_and_cannot_retract_history(self):
        error = "Сначала написал 14. После вопроса учителя исправил на 12."
        raw = record("t1", error, support_level="hint_question",
                     support_evidence=evidence("t1", "После вопроса учителя"))
        memory = self.make_memory([encoded(raw)])
        memory.update(lesson(1, error))
        self.assertIsNone(memory.records[0]["corrects_record_id"])
        self.assertIn("Сначала написал 14", memory.records[0]["event"])
        self.assert_code("invalid_correction_kind", lambda: validate_records(
            encoded({**raw, "corrects_record_id": "r0001"}), memory.sources, {"t1"}, TOPIC, memory.records))

    def test_new_independent_success_preserves_earlier_help(self):
        first = "Решил с наводящим вопросом учителя."
        second = "Другой пример решил верно без помощи."
        old = record("t1", first, support_level="hint_question",
                     support_evidence=evidence("t1", "с наводящим вопросом учителя"))
        success = record("t2", second, support_level="none", independent_recheck="success",
                         support_evidence=evidence("t2", "без помощи"), recheck_evidence=evidence("t2", second))
        memory = self.make_memory([encoded(old), encoded(success)])
        memory.update(lesson(1, first))
        memory.update(lesson(2, second))
        self.assertEqual(memory.records[0]["support_level"], "hint_question")
        self.assertEqual(memory.records[0]["corrected_by_record_ids"], [])
        self.assertEqual(memory.records[1]["independent_recheck"], "success")
        self.assertEqual(memory.records[1]["corrected_by_record_ids"], [])

    def test_unfinished_point_survives_later_completion(self):
        stop = "Составил схему, остановился перед вычислением площади."
        finish = "На следующем занятии вычислил площадь с подсказкой."
        memory = self.make_memory([encoded(record("t1", stop, kind="unfinished_work")),
                                   encoded(record("t2", finish))])
        memory.update(lesson(1, stop))
        memory.update(lesson(2, finish))
        self.assertEqual(memory.records[0]["kind"], "unfinished_work")
        self.assertEqual(memory.records[0]["corrected_by_record_ids"], [])
        self.assertIn("перед вычислением площади", memory.context(TOPIC).text)

    def test_bad_references_duplicates_and_truncation_remain_distinct(self):
        sources = {"t1": {"text": "Ответ 12."}}
        raw = record("t1", "Ответ 12.")
        cases = [("missing_source", {**raw, "evidence_turn_ids": ["future"]}),
                 ("duplicate_citation", {**raw, "evidence_turn_ids": ["t1", "t1"]}),
                 ("invalid_correction_target", {**raw, "kind": "source_correction", "corrects_record_id": "r9999"}),
                 ("invalid_quote", {**raw, "source_quote": "Ответ  12."})]
        for code, broken in cases:
            with self.subTest(code=code):
                self.assert_code(code, lambda: validate_records(encoded(broken), sources, {"t1"}, TOPIC))
        self.assert_code("duplicate_record", lambda: validate_records(encoded(raw, raw), sources, {"t1"}, TOPIC))
        self.assert_code("invalid_json", lambda: validate_records('{"records":[', sources, {"t1"}, TOPIC))
        self.assert_code("duplicate_json_key", lambda: validate_records(
            '{"records":[],"records":[]}', sources, {"t1"}, TOPIC))

    def test_literal_presence_does_not_claim_semantic_validity(self):
        # The quote is literal but says nothing about help. A judge/human must flag the meaning.
        sources = {"t1": {"text": "Ответ 12.", "role": "teacher"}}
        raw = record("t1", "Ответ 12.", support_level="none", support_evidence=evidence("t1", "Ответ 12."))
        accepted = validate_records(encoded(raw), sources, {"t1"}, TOPIC)
        self.assertEqual(accepted, [raw])

    def test_known_recheck_requires_separate_literal_evidence(self):
        sources = {"t1": {"text": "Самостоятельной проверки не было."}}
        raw = record("t1", sources["t1"]["text"], independent_recheck="not_observed")
        self.assert_code("missing_condition_evidence", lambda: validate_records(encoded(raw), sources, {"t1"}, TOPIC))
        raw["recheck_evidence"] = evidence("t1", "проверки не было")
        self.assertEqual(len(validate_records(encoded(raw), sources, {"t1"}, TOPIC)), 1)

    def test_assistant_result_cannot_be_independent_learner_evidence(self):
        sources = {"t1": {"text": "Решение верное, проверено самостоятельно.", "role": "assistant"}}
        raw = record("t1", sources["t1"]["text"], independent_recheck="success",
                     recheck_evidence=evidence("t1", sources["t1"]["text"]))
        self.assert_code("assistant_as_learner_evidence", lambda: validate_records(encoded(raw), sources, {"t1"}, TOPIC))

    def test_correction_chain_and_all_originals_are_indivisible(self):
        texts = ["Первая заметка.", "Исправлена первая заметка.", "Исправлено первое исправление."]
        outputs = [encoded(record("t1", texts[0])), encoded(record(
            "t2", texts[1], kind="source_correction", corrects_record_id="r0001")), encoded(record(
            "t3", texts[2], kind="source_correction", corrects_record_id="r0002"))]
        memory = self.make_memory(outputs)
        for ordinal, text in enumerate(texts, 1):
            memory.update(lesson(ordinal, text))
        complete = memory.context(TOPIC)
        self.assertEqual(len(json.loads(complete.text)), 3)
        memory.budget = len(complete.text) - 1
        self.assertEqual(memory.context(TOPIC).text, "[]")
        self.assertEqual(memory.events[-1]["group_record_ids"], ["r0001", "r0002", "r0003"])

    def test_schema_only_allows_existing_record_correction_ids(self):
        fields = extract_schema({"t1", "t2"}, TOPIC, {"r0001"})["properties"]["records"]["items"]["properties"]
        self.assertEqual(fields["corrects_record_id"]["enum"], [None, "r0001"])
        self.assertEqual(fields["evidence_turn_ids"]["maxItems"], 2)
        self.assertEqual(extract_schema({"t1"}, TOPIC)["properties"]["records"]["items"]["properties"][
            "corrects_record_id"]["enum"], [None])

    def test_runner_retains_duplicate_raw_output_and_machine_diagnostic(self):
        duplicate = encoded(record("t1", "Решил с вопросом учителя.", evidence_turn_ids=["t1", "t1"]))

        class DuplicateClient(FixtureClient):
            def complete(self, messages, purpose, response_schema=None):
                if purpose == "extract":
                    return Completion(duplicate, {"input_tokens": 20, "output_tokens": 30}, 0, {})
                return super().complete(messages, purpose, response_schema)

        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory, histories=[{"history_id": "h1", "learner_id": "synthetic1", "split": "dev",
                "data_origin": "synthetic_unreviewed", "profile_as_of_cutoff": {"topic_id": TOPIC}, "lessons": [lesson()]}])
            output = Path(directory) / "run"
            manifest = run(*paths, output, methods=["structured"], client=DuplicateClient())
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            trace = [json.loads(line) for line in (output / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(manifest["performed_calls"], 1)
        self.assertEqual(rows[0]["memory_errors"][0]["error_code"], "duplicate_citation")
        self.assertEqual(trace[0]["completion_text"], duplicate)
        self.assertEqual(rows[0]["status"], "memory_error")


if __name__ == "__main__":
    unittest.main()
