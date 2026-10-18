from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research_memory.dataset import load_dataset
from research_memory.judge import (CONTEXT_PROTOCOL, HELDOUT_PROTOCOL, _validate_case,
                                   judge_cases, prompt_path, summarize_calibration)
from research_memory.methods import METHODS, Memory
from research_memory.tests import test_judge_context as context_helpers


DATA = Path(__file__).parents[1] / "data"
V4 = DATA / "v4"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class HeldoutDataTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset(V4 / "histories.test.jsonl", V4 / "requests.test.jsonl")
        self.design = json.loads((V4 / "design.private.json").read_text(encoding="utf-8"))["design"]

    def test_comparison_has_sixteen_new_histories_eight_pairs_and_128_answers(self):
        self.assertEqual(self.dataset.split, "test")
        self.assertEqual(len(self.dataset.histories), 16)
        self.assertEqual(len(self.dataset.requests) * len(METHODS), 128)
        self.assertEqual(set(Counter(r["history_id"] for r in self.dataset.requests).values()), {2})
        pairs = Counter((item["situation"], item["length"]) for item in self.design)
        self.assertEqual(len(pairs), 16)
        self.assertEqual(set(pairs.values()), {1})
        self.assertEqual(len({item["situation"] for item in self.design}), 8)
        self.assertEqual(Counter(item["length"] for item in self.design), {"short": 8, "long": 8})
        old = load_dataset(DATA / "v2/histories.dev.jsonl", DATA / "v2/requests.dev.jsonl")
        self.assertTrue(set(self.dataset.histories).isdisjoint(old.histories))
        old_text = {t["text"] for h in old.histories.values() for l in h["lessons"] for t in l["turns"]}
        for history in self.dataset.histories.values():
            self.assertEqual(history["data_origin"], "synthetic_unreviewed")
            self.assertEqual(set(history), {"history_id", "learner_id", "split", "data_origin",
                                            "profile_as_of_cutoff", "lessons"})
            for lesson in history["lessons"]:
                for turn in lesson["turns"]:
                    self.assertNotIn(turn["text"], old_text)

    def test_long_histories_exceed_budget_with_old_and_recent_key_observations(self):
        for item in self.design:
            with self.subTest(history=item["history_id"]):
                history = self.dataset.histories[item["history_id"]]
                topic = history["profile_as_of_cutoff"]["topic_id"]
                memory = Memory("tail", None, 4000, topic)
                texts = []
                for lesson in history["lessons"]:
                    memory.update(lesson)
                    texts.extend(t["text"] for t in lesson["turns"])
                self.assertEqual(len(texts), len(set(texts)))
                context = memory.context(topic)
                self.assertLessEqual(len(context.text), 4000)
                if item["length"] == "long":
                    self.assertGreater(sum(map(len, texts)), 4000)
                    visible = set(item["key_turn_ids"]) & context.evidence_ids
                    if item["situation"] in {"assistance", "unknown_conditions", "old_event_eviction", "ambiguous_authorship"}:
                        self.assertFalse(visible)
                    else:
                        self.assertTrue(visible)
                else:
                    self.assertLess(sum(map(len, texts)), 4000)

    def test_frozen_rubric_sources_are_real_and_excluded_from_generator_inputs(self):
        gold = rows(V4 / "gold.test.jsonl")
        by_request = {r["request_id"]: r for r in self.dataset.requests}
        self.assertEqual(len(gold), 32)
        self.assertEqual({r["request_id"] for r in gold}, set(by_request))
        for reference in gold:
            request = by_request[reference["request_id"]]
            history = self.dataset.histories[request["history_id"]]
            sources = {t["turn_id"] for l in history["lessons"] for t in l["turns"]}
            self.assertTrue(set(reference["evidence_turn_ids"]) <= sources)
            self.assertGreaterEqual(len(reference["allowed_actions"]), 2)
            self.assertGreaterEqual(len(reference["acceptable_action_examples"]), 2)
            self.assertTrue(reference["forbidden_claims"])
        with self.assertRaisesRegex(ValueError, "Reference-answer"):
            load_dataset(V4 / "gold.test.jsonl", V4 / "requests.test.jsonl")
        manifest = json.loads((V4 / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["human_semantic_review"])
        self.assertEqual(manifest["generator_inputs"], ["histories.test.jsonl", "requests.test.jsonl"])
        for name, digest in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((V4 / name).read_bytes()).hexdigest(), digest)

    def test_preregistered_review_has_five_histories_and_twenty_answers(self):
        selection = json.loads((V4 / "review.selection.json").read_text(encoding="utf-8"))
        requests = {r["request_id"]: r for r in self.dataset.requests}
        self.assertEqual(len(selection["requests"]), 5)
        self.assertEqual(len({requests[r]["history_id"] for r in selection["requests"]}), 5)
        self.assertEqual(selection["methods"], list(METHODS))
        self.assertEqual(selection["expected_answers"], 20)

    def test_declared_key_correct_results_are_arithmetically_valid(self):
        triples = [(618, 3, 206), (1428, 7, 204), (1015, 5, 203), (1836, 6, 306),
                   (2412, 12, 201), (2842, 14, 203), (1632, 8, 204), (2652, 13, 204),
                   (2727, 9, 303), (3232, 16, 202), (3648, 12, 304), (3672, 12, 306),
                   (4060, 20, 203), (4080, 20, 204), (1331, 11, 121), (2442, 11, 222),
                   (1717, 17, 101), (3468, 17, 204), (4872, 24, 203), (7105, 35, 203),
                   (5226, 26, 201), (5304, 26, 204), (6496, 32, 203), (6528, 32, 204)]
        for dividend, divisor, quotient in triples:
            self.assertEqual(dividend, divisor * quotient)


class HeldoutJudgeTests(unittest.TestCase):
    def test_calibration_preserves_eighteen_cases_and_adds_six_disjoint_checks(self):
        cases, expected = rows(V4 / "calibration.cases.jsonl"), rows(V4 / "calibration.expected.jsonl")
        self.assertEqual(cases[:18], rows(DATA / "v3/calibration.cases.jsonl"))
        self.assertEqual(expected[:18], rows(DATA / "v3/calibration.expected.jsonl"))
        self.assertEqual(len(cases), 24)
        self.assertEqual(len(expected), 24)
        self.assertEqual([r["expected_verdict"] for r in expected[18:]],
                         ["acceptable", "critical_error"] * 3)
        heldout_ids = {t["turn_id"] for h in rows(V4 / "histories.test.jsonl")
                       for l in h["lessons"] for t in l["turns"]}
        for case, reference in zip(cases, expected):
            sources = _validate_case(case, protocol=HELDOUT_PROTOCOL)
            self.assertTrue(set(reference["evidence_turn_ids"]) <= sources.keys())
            if case["answer_code"] >= "C019":
                self.assertTrue(heldout_ids.isdisjoint(sources))

    def test_v3_fixture_processes_24_cases_and_cannot_qualify_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = judge_cases(V4 / "calibration.cases.jsonl", root / "judged", protocol=HELDOUT_PROTOCOL)
            report = summarize_calibration(root / "judged/judgements.jsonl",
                                           V4 / "calibration.expected.jsonl", root / "gate.json",
                                           protocol=HELDOUT_PROTOCOL)
            self.assertEqual(manifest["case_count"], 24)
            self.assertEqual(manifest["performed_calls"], 24)
            self.assertEqual(manifest["protocol_version"], "memory-judge-context-v3")
            self.assertEqual(report["protocol_version"], "memory-judge-calibration-context-v3")
            self.assertFalse(report["qualified"])
            self.assertFalse(report["gate"]["not_contract_only"])
            self.assertEqual(report["uncertain_count"], 24)
            self.assertFalse(manifest["real_model_calls"])

    def test_new_protocol_keeps_old_prompt_and_sends_actual_generator_context(self):
        self.assertNotEqual(prompt_path(HELDOUT_PROTOCOL), prompt_path(CONTEXT_PROTOCOL))
        setup = context_helpers.ContextPreparationTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = setup.source(root, setup.row())
            from research_memory.judge import prepare_judging
            prepare_judging(run_dir, root / "histories.jsonl", root / "blind", protocol=HELDOUT_PROTOCOL)
            client = context_helpers.StubClient()
            manifest = judge_cases(root / "blind/cases.jsonl", root / "judged", backend="ollama",
                                   model="stub", client=client, protocol=HELDOUT_PROTOCOL)
            payload = json.loads(client.seen[0][0][-1]["content"])
            self.assertEqual(payload["generator_context"]["selected_memory"], "ACTUAL_MEMORY_SENTINEL")
            self.assertNotIn("method", payload)
            self.assertEqual(manifest["utility_counts"], {"limited": 1})


if __name__ == "__main__":
    unittest.main()
