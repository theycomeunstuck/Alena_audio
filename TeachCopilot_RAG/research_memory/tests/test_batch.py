"""Durable queue checks with synthetic input and fake clients only."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research_memory import batch
from research_memory.clients import Completion, CompletionValidationError, FixtureClient, InputBudgetError
from research_memory.runner import run as original_run


class Crash(BaseException):
    pass


class Spy(FixtureClient):
    def __init__(self, crash_at=None):
        self.seen = []
        self.crash_at = crash_at
        self.timeout = 180

    def complete(self, messages, purpose, response_schema=None):
        self.seen.append((purpose, self.timeout))
        if len(self.seen) == self.crash_at:
            raise Crash()
        return super().complete(messages, purpose, response_schema)


def inputs(directory, count=1, requests_each=2):
    histories, requests = [], []
    for i in range(1, count + 1):
        hid = f"h{i}"
        histories.append({"history_id": hid, "learner_id": f"synthetic-{i}", "split": "dev",
                          "data_origin": "synthetic_unreviewed", "profile_as_of_cutoff": {"topic_id": "math.test"},
                          "lessons": [{"lesson_id": f"{hid}-l{n}", "observed_on": f"2026-09-{n:02d}",
                                       "turns": [{"turn_id": f"{hid}-t{n}", "role": "student",
                                                  "text": "Вымышленное наблюдение."}]} for n in range(1, 4)]})
        requests.extend({"request_id": f"{hid}-q{n}", "history_id": hid, "topic_id": "math.test",
                         "text": "Предложи следующее действие."} for n in range(1, requests_each + 1))
    paths = Path(directory) / "histories.jsonl", Path(directory) / "requests.jsonl"
    for path, rows in zip(paths, (histories, requests)):
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return paths


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = inputs(self.root)
        self.output = self.root / "batch"

    def create(self, **kwargs):
        return batch.create_batch(*self.paths, self.output, **kwargs)

    def rows(self):
        return [json.loads(x) for x in (self.output / "results.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_nonfinite_limits_are_rejected_before_creating_queue(self):
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.create(max_seconds=value)
            self.assertFalse(self.output.exists())

    def test_completed_jobs_are_not_repeated_and_no_fake_usage(self):
        self.create()
        client = Spy()
        first = batch.work_batch(self.output, client=client, max_jobs=1)
        self.assertEqual(len(client.seen), 2)
        self.assertEqual(first["job_counts"]["succeeded"], 1)
        self.assertEqual(first["usage"]["fixture_calls"], 2)
        self.assertEqual(first["usage"]["actual_model_calls"], 0)
        second = batch.work_batch(self.output, client=client)
        self.assertEqual(second["job_counts"]["succeeded"], 4)
        self.assertEqual(len(client.seen), 14)
        batch.work_batch(self.output, client=client)
        self.assertEqual(len(client.seen), 14)
        self.assertEqual(second["usage"]["known_input_tokens"], 0)
        self.assertEqual(second["not_assessed_results"], 8)

    def test_cached_partial_job_is_replayed_without_repeating_model_calls(self):
        self.create(methods=["summary"])

        def finish_then_crash(*args, **kwargs):
            original_run(*args, **kwargs)
            raise Crash()

        first = Spy()
        with patch.object(batch, "run", side_effect=finish_then_crash), self.assertRaises(Crash):
            batch.work_batch(self.output, client=first)
        self.assertEqual(len(first.seen), 5)
        resumed = Spy()
        status = batch.work_batch(self.output, client=resumed)
        self.assertEqual(len(resumed.seen), 0)
        self.assertEqual(status["usage"]["new_calls"], 5)
        self.assertEqual(status["usage"]["cache_hits"], 5)
        self.assertEqual(status["job_counts"]["succeeded"], 1)
        self.assertEqual(status["jobs"][0]["attempt"], 2)

    def test_unknown_started_call_is_failed_not_repeated_and_other_jobs_continue(self):
        self.create(methods=["summary", "tail"])
        with self.assertRaises(Crash):
            batch.work_batch(self.output, client=Spy(crash_at=2))
        resumed = Spy()
        status = batch.work_batch(self.output, client=resumed)
        self.assertEqual(len(resumed.seen), 2)
        self.assertEqual(status["usage"]["unknown_outcome_calls"], 1)
        self.assertEqual(status["job_counts"]["failed"], 1)
        self.assertEqual(status["job_counts"]["succeeded"], 1)
        self.assertEqual(len(self.rows()), 4)
        self.assertEqual(self.rows()[0]["job_error"], "unknown_model_call_outcome")

    def test_partial_replay_retains_summary_budget_failures_without_regeneration(self):
        self.create(methods=["summary"])

        class LongSummary(Spy):
            def complete(self, messages, purpose, response_schema=None):
                self.seen.append(purpose)
                if purpose == "summary":
                    payload = json.loads(messages[-1]["content"])
                    text = ("Длинное резюме. " * 400) if "new_lesson" in payload else "Короткое резюме."
                    return Completion(text, {}, 0, {})
                return FixtureClient.complete(self, messages, purpose, response_schema)

        original_complete = batch._CachedClient.complete

        def cached_then_crash(cached, *args, **kwargs):
            result = original_complete(cached, *args, **kwargs)
            if cached.ordinal == 2:
                raise Crash()
            return result

        first = LongSummary()
        with patch.object(batch._CachedClient, "complete", cached_then_crash), self.assertRaises(Crash):
            batch.work_batch(self.output, client=first)
        resumed = LongSummary()
        status = batch.work_batch(self.output, client=resumed)
        self.assertEqual(len(first.seen), 2)
        self.assertEqual(len(resumed.seen), 1)
        self.assertEqual(status["usage"]["cache_hits"], 2)
        self.assertEqual(status["usage"]["new_calls"], 3)
        self.assertEqual(status["job_counts"]["failed"], 1)
        self.assertTrue(all(row["status"] == "memory_error" for row in self.rows()))
        self.assertTrue(all(error["error_code"] == "summary_exceeds_budget"
                            for row in self.rows() for error in row["memory_errors"]))

    def test_frozen_input_tampering_is_detected_before_calls(self):
        self.create()
        snapshot = self.output / "snapshots/histories.jsonl"
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        client = Spy()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            batch.work_batch(self.output, client=client)
        self.assertFalse(client.seen)

    def test_current_code_or_prompt_change_is_detected(self):
        self.create()
        with patch.object(batch, "_versions", return_value={"code": {}, "prompts": {}}):
            with self.assertRaisesRegex(ValueError, "code or prompt hash mismatch"):
                batch.work_batch(self.output)

    def test_global_call_limit_survives_resume_and_keeps_all_rows(self):
        self.create(methods=["summary"], max_calls=2)
        first = Spy()
        status = batch.work_batch(self.output, client=first)
        self.assertEqual(len(first.seen), 2)
        self.assertEqual(status["job_counts"]["pending"], 1)
        resumed = Spy()
        status = batch.work_batch(self.output, client=resumed)
        self.assertEqual(len(resumed.seen), 0)
        self.assertEqual(status["usage"]["new_calls"], 2)
        self.assertEqual(status["usage"]["cache_hits"], 2)
        self.assertEqual(status["jobs"][0]["error"], "max_calls_exceeded")
        self.assertEqual(len(self.rows()), 2)
        self.assertTrue(all(x["status"] == "not_executed" for x in self.rows()))

    def test_deadline_survives_resume_and_clips_client_timeout(self):
        self.create(methods=["profile_only"], max_seconds=10)
        client = Spy()
        with patch.object(batch.time, "time", return_value=100):
            status = batch.work_batch(self.output, client=client, max_jobs=0)
        self.assertEqual(status["deadline_unix"], 110)
        with patch.object(batch.time, "time", return_value=105):
            batch.work_batch(self.output, client=client)
        self.assertTrue(all(timeout == 5 for _, timeout in client.seen))
        self.assertEqual(client.timeout, 180)
        with patch.object(batch.time, "time", return_value=111):
            status = batch.work_batch(self.output, client=Spy())
        self.assertTrue(status["deadline_exceeded"])
        self.assertEqual(status["deadline_unix"], 110)

    def test_expired_deadline_does_not_start_a_job(self):
        self.create(max_seconds=1)
        with patch.object(batch.time, "time", return_value=100):
            batch.work_batch(self.output, max_jobs=0)
        client = Spy()
        with patch.object(batch.time, "time", return_value=102):
            status = batch.work_batch(self.output, client=client)
        self.assertFalse(client.seen)
        self.assertEqual(status["job_counts"]["pending"], 4)

    def test_second_worker_fails_fast_but_status_is_readable(self):
        self.create()
        code = ("from research_memory.batch import work_batch; import sys; "
                "work_batch(sys.argv[1])")
        with batch._worker_lock(self.output):
            status = batch.batch_status(self.output)
            result = subprocess.run([sys.executable, "-c", code, str(self.output)],
                                    capture_output=True, text=True, timeout=10,
                                    cwd=batch.ROOT.parent)
        self.assertEqual(status["job_counts"]["pending"], 4)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Another worker owns this batch", result.stderr)

    def test_all_forty_expected_rows_retained(self):
        self.paths = inputs(self.root, count=5)
        initial = self.create()
        self.assertEqual(initial["result_count"], 40)
        self.assertEqual(initial["unexecuted_results"], 40)
        final = batch.work_batch(self.output, client=Spy())
        self.assertEqual(final["result_count"], 40)
        self.assertEqual(final["executed_results"], 40)
        self.assertEqual(final["usage"]["fixture_calls"], 70)
        self.assertFalse(final["scientific_result"])
        self.assertEqual(len({x["result_id"] for x in self.rows()}), 40)
        self.assertTrue(all(x["assessment_status"] == "not_assessed" for x in self.rows()))

    def test_known_call_failure_is_not_retried_and_usage_is_preserved(self):
        self.create(backend="ollama", methods=["summary", "tail"])

        class Fail(Spy):
            def complete(self, messages, purpose, response_schema=None):
                self.seen.append(purpose)
                if purpose == "summary":
                    raise CompletionValidationError("truncated", Completion("partial", {
                        "input_tokens": 10, "output_tokens": 3}, 0, {}), "truncated_completion")
                result = FixtureClient.complete(self, messages, purpose, response_schema)
                result.usage = {"input_tokens": 7, "output_tokens": 2}
                return result

        client = Fail()
        status = batch.work_batch(self.output, client=client)
        self.assertEqual(len(client.seen), 3)
        self.assertEqual(status["usage"]["known_input_tokens"], 24)
        self.assertEqual(status["usage"]["known_output_tokens"], 7)
        self.assertEqual(status["job_counts"]["failed"], 1)
        self.assertEqual(status["job_counts"]["succeeded"], 1)
        batch.work_batch(self.output, client=client)
        self.assertEqual(len(client.seen), 3)

    def test_token_reservation_stops_before_call(self):
        self.create(backend="ollama", methods=["profile_only"], max_output_tokens_total=1535)
        client = Spy()
        status = batch.work_batch(self.output, client=client)
        self.assertFalse(client.seen)
        self.assertEqual(status["jobs"][0]["error"], "max_output_tokens_total_reservation_exceeded")

    def test_preflight_refusal_counts_attempt_without_inventing_a_model_call(self):
        self.create(backend="ollama", methods=["structured"])

        class Refuse(Spy):
            def complete(self, messages, purpose, response_schema=None):
                self.seen.append(purpose)
                raise InputBudgetError("Writer budget exceeded", Completion("", {
                    "input_tokens": 0, "output_tokens": 0}, 0,
                    {"network_request_performed": False, "input_budget": {"within_guard": False}}),
                    "writer_input_budget_exceeded")

        client = Refuse()
        status = batch.work_batch(self.output, client=client)
        self.assertEqual(client.seen, ["extract"])
        self.assertEqual(status["usage"]["new_calls"], 1)
        self.assertEqual(status["usage"]["actual_model_calls"], 0)
        self.assertEqual(status["usage"]["preflight_refusals"], 1)
        self.assertEqual(status["usage"]["known_input_tokens"], 0)
        self.assertFalse(status["real_model_calls"])
        self.assertEqual(status["job_counts"]["failed"], 1)
        self.assertEqual(len(self.rows()), 2)
        self.assertTrue(all(row["status"] == "memory_error" for row in self.rows()))
        self.assertEqual(self.rows()[0]["memory_errors"][0]["error_code"], "writer_input_budget_exceeded")
        batch.work_batch(self.output, client=client)
        self.assertEqual(client.seen, ["extract"])

    def test_missing_real_usage_stops_token_limited_batch_without_fake_estimate(self):
        self.create(backend="ollama", methods=["profile_only"], max_input_tokens=100000)
        first = Spy()
        status = batch.work_batch(self.output, client=first)
        self.assertEqual(len(first.seen), 1)
        self.assertEqual(status["usage"]["known_input_tokens"], 0)
        self.assertEqual(status["usage"]["unknown_usage_calls"], 1)
        self.assertEqual(status["jobs"][0]["error"], "token_usage_unknown")
        resumed = Spy()
        status = batch.work_batch(self.output, client=resumed)
        self.assertFalse(resumed.seen)
        self.assertEqual(status["usage"]["cache_hits"], 1)
        self.assertFalse(status["usage"]["token_usage_complete"])

    def test_fixture_has_no_invented_token_cost_and_no_token_budget_charge(self):
        self.create(methods=["profile_only"], max_input_tokens=1, max_output_tokens_total=1)
        status = batch.work_batch(self.output, client=Spy())
        self.assertEqual(status["job_counts"]["succeeded"], 1)
        self.assertEqual(status["usage"]["known_output_tokens"], 0)
        self.assertEqual(status["usage"]["unknown_usage_calls"], 0)

    def test_non_synthetic_input_is_rejected(self):
        history = json.loads(self.paths[0].read_text(encoding="utf-8"))
        history["data_origin"] = "real_child"
        self.paths[0].write_text(json.dumps(history), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "synthetic data only"):
            self.create()


if __name__ == "__main__":
    unittest.main()
