from __future__ import annotations

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_memory.clients import Completion, FixtureClient
from research_memory.contracts import ContractValidationError, validate_action, validate_records
from research_memory.dataset import load_dataset
from research_memory.methods import Memory, _whole_sentences
from research_memory.runner import ROOT, CallJournal, export_review, run


TOPIC = "math.test"


def lesson(n=1, text="Решил с вопросом учителя."):
    return {"lesson_id": f"l{n}", "observed_on": f"2026-09-{n:02d}",
            "turns": [{"turn_id": f"t{n}", "role": "student", "text": text}]}


def record(n=1, **overrides):
    return {"kind": "learning_episode", "topic_id": TOPIC, "event": "Наблюдение, не вывод об освоении.",
            "support_level": "hint_question", "independent_recheck": "unknown",
            "evidence_turn_ids": [f"t{n}"], "source_quote": "Решил с вопросом учителя.",
            "support_evidence": {"source_turn_id": f"t{n}", "quote": "с вопросом учителя"},
            "recheck_evidence": None, "corrects_record_id": None, **overrides}


def record_text(*records):
    return json.dumps({"records": list(records)}, ensure_ascii=False)


def write_inputs(directory, histories=None, requests=None):
    histories = histories or [{"history_id": "h1", "learner_id": "synthetic1", "split": "dev",
                               "data_origin": "synthetic_unreviewed", "profile_as_of_cutoff": {
                                   "topic_id": TOPIC, "grade": 4}, "lessons": [lesson(1), lesson(2)]}]
    requests = requests or [{"request_id": "q1", "history_id": "h1", "topic_id": TOPIC,
                             "text": "FUTURE_REQUEST_SENTINEL"}]
    paths = [Path(directory) / "histories.jsonl", Path(directory) / "requests.jsonl"]
    for path, values in zip(paths, (histories, requests)):
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in values), encoding="utf-8")
    return paths


class SpyClient(FixtureClient):
    def __init__(self):
        self.seen = []

    def complete(self, messages, purpose, response_schema=None):
        self.seen.append((purpose, json.loads(messages[-1]["content"])))
        if purpose == "summary":
            return Completion("SUMMARY_SENTINEL [t1].", {}, 0, {})
        return super().complete(messages, purpose, response_schema)


class DatasetTests(unittest.TestCase):
    def test_current_data_and_no_reference_answer_access(self):
        opened = []
        original = Path.open

        def guarded(path, *args, **kwargs):
            self.assertNotIn("gold", path.name.lower())
            opened.append(path.name)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded), tempfile.TemporaryDirectory() as directory:
            manifest = run(ROOT / "data/histories.dev.jsonl", ROOT / "data/requests.dev.jsonl",
                           Path(directory) / "run")
        self.assertEqual(manifest["result_count"], 40)
        self.assertIn("histories.dev.jsonl", opened)
        self.assertNotIn("gold.dev.jsonl", opened)

    def test_reference_named_input_rejected_before_open(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("Must not open")):
            with self.assertRaisesRegex(ValueError, "Reference-answer"):
                load_dataset("gold.dev.jsonl", "requests.dev.jsonl")

    def test_broken_links_ids_dates_and_splits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory)
            history = json.loads(paths[0].read_text())
            request = json.loads(paths[1].read_text())
            corruptions = [lambda h, q: h["lessons"][1].update(observed_on="2026-08-01"),
                           lambda h, q: h["lessons"][1]["turns"][0].update(turn_id="t1"),
                           lambda h, q: q.update(history_id="missing"),
                           lambda h, q: q.update(split="test"),
                           lambda h, q: h["lessons"][0].update(observed_on="2026-02-30")]
            for corrupt in corruptions:
                h, q = copy.deepcopy(history), copy.deepcopy(request)
                corrupt(h, q)
                write_inputs(directory, [h], [q])
                with self.assertRaises(ValueError):
                    load_dataset(*paths)


class MemoryTests(unittest.TestCase):
    def test_updates_do_not_see_requests_future_lessons_or_other_methods(self):
        client = SpyClient()
        with tempfile.TemporaryDirectory() as directory:
            histories, requests = write_inputs(directory)
            output = Path(directory) / "run"
            run(histories, requests, output, client=client)
            rows = [json.loads(x) for x in (output / "results.jsonl").read_text().splitlines()]
        update_calls = [(p, payload) for p, payload in client.seen if p != "action"]
        self.assertEqual(len(update_calls), 4)
        for purpose, payload in update_calls:
            self.assertNotIn("FUTURE_REQUEST_SENTINEL", json.dumps(payload))
            if payload["new_lesson"]["lesson_id"] == "l1":
                self.assertNotIn('"t2"', json.dumps(payload))
            if purpose == "extract":
                self.assertEqual(payload["previous_records"], [])
                self.assertEqual(payload["allowed_topic_id"], TOPIC)
                self.assertNotIn("SUMMARY_SENTINEL", json.dumps(payload))
            else:
                self.assertEqual(set(payload), {"previous_summary", "new_lesson", "budget_chars"})
        by_method = {row["method"]: row for row in rows}
        self.assertEqual(by_method["profile_only"]["memory_context"], "")
        self.assertIn("2026-09-02", by_method["tail"]["memory_context"])
        self.assertIn("SUMMARY_SENTINEL", by_method["summary"]["memory_context"])
        self.assertEqual(by_method["structured"]["memory_context"], "[]")

    def test_structured_sources_quotes_topic_and_new_turn_required(self):
        sources = {"t1": {"text": "Решил с вопросом учителя."}, "t2": {"text": "Новое наблюдение."}}
        self.assertEqual(len(validate_records(record_text(record()), sources, {"t1"}, TOPIC)), 1)
        invalid = [record(evidence_turn_ids=["future"]), record(source_quote="Я решил сам."),
                   record(topic_id="invented"), record(corrects_record_id="future")]
        for item in invalid:
            with self.assertRaises(ValueError):
                validate_records(record_text(item), sources, {"t1"}, TOPIC)
        with self.assertRaisesRegex(ValueError, "new sources"):
            validate_records(record_text(record()), sources, {"t2"}, TOPIC)
        with self.assertRaisesRegex(ValueError, "new turn"):
            validate_records(record_text(record(evidence_turn_ids=["t1", "t2"])), sources, {"t2"}, TOPIC)

    def test_source_correction_preserves_history_new_success_does_not_correct(self):
        outputs = [record_text(record()), record_text(record(2, kind="source_correction", corrects_record_id="r0001")),
                   record_text(record(3, support_level="none", independent_recheck="success",
                                      support_evidence={"source_turn_id": "t3", "quote": "без помощи"},
                                      recheck_evidence={"source_turn_id": "t3", "quote": "Новый пример решил без помощи"},
                                      source_quote="Новый пример решил без помощи."))]
        memory = Memory("structured", lambda *args: outputs.pop(0), 4000, TOPIC)
        for n in range(1, 3):
            memory.update(lesson(n))
        memory.update(lesson(3, "Новый пример решил без помощи."))
        self.assertEqual(len(memory.records), 3)
        self.assertEqual(memory.records[0]["corrected_by_record_ids"], ["r0002"])
        self.assertEqual(memory.records[1]["corrected_by_record_ids"], [])

    def test_tail_large_message_has_nonempty_contiguous_suffix(self):
        memory = Memory("tail", None, 180, TOPIC)
        memory.update(lesson(1, "OLD_ONLY"))
        memory.update(lesson(2, " ".join(f"word{i}" for i in range(300))))
        context = memory.context(TOPIC)
        self.assertGreater(len(context.text), 0)
        self.assertLessEqual(len(context.text), 180)
        self.assertIn("word299", context.text)
        self.assertNotIn("OLD_ONLY", context.text)
        self.assertEqual(context.evidence_ids, {"t2"})

    def test_summary_over_budget_rejected_without_shortening_or_trimming(self):
        seen = []
        outputs = ["Первая фраза [t1]. " + "Большая " * 100 + ". Последняя фраза [t1]."] * 2

        def call(messages, purpose, schema):
            seen.append(purpose)
            return outputs.pop(0)

        memory = Memory("summary", call, 128, TOPIC)
        memory.summary = "Предыдущая память."
        with self.assertRaises(ContractValidationError) as rejected:
            memory.update(lesson())
        self.assertEqual(rejected.exception.code, "summary_exceeds_budget")
        self.assertEqual(seen, ["summary"])
        self.assertEqual(memory.summary, "Предыдущая память.")
        self.assertEqual(memory.events[0]["event"], "summary_exceeds_budget")
        self.assertEqual(_whole_sentences("А. " + "Б" * 200 + ". В.", 128), "А.")

    def test_structured_selection_keeps_valid_json_records_and_prioritizes_topic(self):
        memory = Memory("structured", None, 900, TOPIC)
        memory.records = [dict(record(), record_id="r1", observed_on="2026-01-01", corrected_by_record_ids=[]),
                          dict(record(2, topic_id="other"), record_id="r2", observed_on="2026-10-01",
                               corrected_by_record_ids=[])]
        memory.budget = len(json.dumps([memory.records[0]], ensure_ascii=False, separators=(",", ":")))
        context = memory.context(TOPIC)
        self.assertLessEqual(len(context.text), memory.budget)
        self.assertEqual([r["record_id"] for r in json.loads(context.text)], ["r1"])
        memory.budget = 128
        self.assertEqual(memory.context(TOPIC).text, "[]")

    def test_action_citations_must_be_visible_and_review_required(self):
        action = json.loads(FixtureClient().complete([], "action").text)
        self.assertEqual(validate_action(json.dumps(action), set())["evidence_turn_ids"], [])
        action["evidence_turn_ids"] = ["t1"]
        with self.assertRaisesRegex(ValueError, "absent"):
            validate_action(json.dumps(action), set())
        action["review_required"] = False
        with self.assertRaisesRegex(ValueError, "review"):
            validate_action(json.dumps(action), {"t1"})

    def test_structured_selection_skips_oversized_new_record_and_keeps_older_one(self):
        memory = Memory("structured", None, 700, TOPIC)
        old = dict(record(), record_id="r1", observed_on="2026-01-01", corrected_by_record_ids=[])
        oversized = dict(record(2, event="Длинное наблюдение. " * 100), record_id="r2",
                         observed_on="2026-10-01", corrected_by_record_ids=[])
        memory.records = [old, oversized]
        context = memory.context(TOPIC)
        self.assertLessEqual(len(context.text), memory.budget)
        self.assertEqual([item["record_id"] for item in json.loads(context.text)], ["r1"])
        self.assertEqual(context.evidence_ids, {"t1"})
        self.assertTrue(any(event.get("record_id") == "r2" and event.get("reason") == "record_exceeds_budget"
                            for event in memory.events))
        events_before = list(memory.events)
        memory.context(TOPIC)
        self.assertEqual(memory.events, events_before)

    def test_correction_and_original_are_selected_or_skipped_together(self):
        memory = Memory("structured", None, 4000, TOPIC)
        old = dict(record(), record_id="r1", observed_on="2026-01-01", corrected_by_record_ids=["r2"])
        correction = dict(record(2, kind="source_correction", corrects_record_id="r1"), record_id="r2",
                          observed_on="2026-10-01", corrected_by_record_ids=[])
        memory.records = [old, correction]
        memory.budget = len(json.dumps([correction], ensure_ascii=False, separators=(",", ":")))
        context = memory.context(TOPIC)
        self.assertEqual(json.loads(context.text), [])
        self.assertEqual(context.evidence_ids, set())
        memory.budget = len(json.dumps([old, correction], ensure_ascii=False, separators=(",", ":")))
        context = memory.context(TOPIC)
        self.assertEqual([item["record_id"] for item in json.loads(context.text)], ["r1", "r2"])
        self.assertEqual(context.evidence_ids, {"t1", "t2"})


class RunnerTests(unittest.TestCase):
    def test_actual_call_schemas_use_only_processed_or_visible_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory)
            output = Path(directory) / "run"
            run(*paths, output, client=SpyClient())
            trace = [json.loads(line) for line in (output / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
            calls = [json.loads(line) for line in (output / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
        for entry, call in zip(trace, calls):
            schema = entry["response_schema"]
            self.assertEqual(call["response_schema_chars"],
                             len(json.dumps(schema, ensure_ascii=False)) if schema is not None else 0)
            if entry["purpose"] == "extract":
                expected = ["t1"] if entry["lesson_id"] == "l1" else ["t1", "t2"]
                fields = schema["properties"]["records"]["items"]["properties"]
                self.assertEqual(fields["evidence_turn_ids"]["items"]["enum"], expected)
                self.assertEqual(fields["evidence_turn_ids"]["maxItems"], len(expected))
                self.assertEqual(fields["support_evidence"]["anyOf"][1]["properties"]["source_turn_id"]["enum"], expected)
            elif entry["purpose"] == "action":
                field = schema["properties"]["evidence_turn_ids"]
                expected = {"tail": ["t1", "t2"], "summary": ["t1"]}.get(entry["method"], [])
                if expected:
                    self.assertEqual(field["items"]["enum"], expected)
                else:
                    self.assertEqual(field["maxItems"], 0)

    def test_fixture_is_contract_only_hashed_and_exported_blind(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory)
            output, review = Path(directory) / "run", Path(directory) / "review"
            manifest = run(*paths, output)
            self.assertFalse(manifest["scientific_result"])
            self.assertFalse(manifest["real_model_calls"])
            self.assertEqual(manifest["result_type"], "contract_only")
            self.assertEqual(manifest["performed_calls"], 8)
            self.assertEqual(set(manifest["input_sha256"]), {"histories", "requests"})
            self.assertTrue(all(len(x) == 64 for x in manifest["prompt_sha256"].values()))
            self.assertEqual(export_review(output, review), 4)
            rows = [json.loads(x) for x in (review / "review.jsonl").read_text().splitlines()]
            mapping = json.loads((review / "mapping.private.json").read_text())
            self.assertEqual(len(mapping), 4)
            for row in rows:
                self.assertNotIn("method", row)
                self.assertFalse(row["scientific_result"])
                self.assertEqual(set(row["human_review"].values()), {""})
            with self.assertRaises(FileExistsError):
                run(*paths, output)

    def test_call_limit_is_persisted_as_errors_without_losing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory)
            output = Path(directory) / "run"
            manifest = run(*paths, output, max_calls=1)
            rows = [json.loads(x) for x in (output / "results.jsonl").read_text().splitlines()]
            calls = [json.loads(x) for x in (output / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(manifest["performed_calls"], 1)
            self.assertEqual(manifest["error_result_count"], 3)
            self.assertTrue(any(x["error"] == "max_calls_exceeded" for x in calls))
            self.assertTrue(any(x["status"] == "memory_error" for x in rows))

    def test_timeout_and_invalid_outputs_leave_explicit_result_rows(self):
        class BrokenClient(FixtureClient):
            def complete(self, messages, purpose, response_schema=None):
                if purpose == "summary":
                    raise TimeoutError("SECRET_PROVIDER_PAYLOAD")
                return Completion("invalid-json", {}, 0, {})

        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory)
            output = Path(directory) / "run"
            manifest = run(*paths, output, client=BrokenClient())
            text = (output / "results.jsonl").read_text()
            rows = [json.loads(x) for x in text.splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(manifest["error_result_count"], 4)
            self.assertNotIn("SECRET_PROVIDER_PAYLOAD", text + (output / "calls.jsonl").read_text())
            self.assertEqual({x["status"] for x in rows}, {"memory_error", "action_error"})

    def test_call_journal_logs_tokens_latency_purpose_but_not_prompts(self):
        stream = io.StringIO()
        journal = CallJournal(FixtureClient(), stream)
        journal([{"role": "user", "content": "PRIVATE_INPUT"}], "action")
        row = json.loads(stream.getvalue())
        self.assertEqual(row["purpose"], "action")
        self.assertIn("input_tokens", row["usage"])
        self.assertIsNone(row["error"])
        self.assertEqual(row["elapsed_seconds"], 0)
        self.assertNotIn("PRIVATE_INPUT", stream.getvalue())

    def test_synthetic_trace_preserves_actual_messages_schema_and_response(self):
        calls, trace = io.StringIO(), io.StringIO()
        journal = CallJournal(FixtureClient(), calls, trace_stream=trace)
        messages = [{"role": "user", "content": "SYNTHETIC_TRACE_SENTINEL"}]
        schema = {"type": "object"}
        response = journal(messages, "action", schema)
        row = json.loads(trace.getvalue())
        self.assertEqual(row["messages"], messages)
        self.assertEqual(row["response_schema"], schema)
        self.assertEqual(row["completion_text"], response)
        self.assertEqual(row["call_id"], json.loads(calls.getvalue())["call_id"])

    def test_non_synthetic_data_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_inputs(directory)
            history = json.loads(paths[0].read_text())
            history["data_origin"] = "real_unreviewed"
            paths[0].write_text(json.dumps(history), encoding="utf-8")
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "synthetic data only"):
                run(*paths, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
