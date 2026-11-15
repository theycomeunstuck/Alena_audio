#!/usr/bin/env python3
"""Regression tests for learner-data tooling hygiene (audit findings F-054 – F-057).

Four confirmed defects, one test class each:

* F-054 – NaN/Infinity pass the 0..1 score checks in zpd/validate, get
  classified as zpd instead of unknown, and poison the mastery EMA forever;
  json.loads/json.dumps admit NaN literals by default.
* F-055 – ``zpd --write`` truncates the card before writing (a crash leaves a
  torn JSON holding the only copy of the history) and silently drops malformed
  competencies from the derived zone.
* F-056 – progress_report treats history as date-ordered (backfilled
  observations become the report end point), crashes on a missing topic_id
  (ZpdPolicyError) and does raw arithmetic on null/string scores.
* F-057 – validate_card raises TypeError (unhashable type) on a list-valued
  difficulty, aborting repo-wide validation.

Style follows learner-data/tools/selftest.py: stdlib unittest only, in-memory
fixtures plus TemporaryDirectory, no real learner files touched.

Run:  python learner-data/tools/test_learner_tools_hygiene.py
  or from TeachCopilot_RAG:
      python -m pytest ../learner-data/tools/test_learner_tools_hygiene.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import progress_report
    import validate
    import zpd
else:
    from . import progress_report
    from . import validate
    from . import zpd

TOPIC = "math.g3.numbers.multiplication_table.times_table"
# validate_card only does catalog.get(topic_id) lookups, so a mini catalog keeps
# these tests hermetic (same shape the audit repros used).
MINI_CATALOG = {TOPIC: {"title_ru": "Демо-тема", "deprecated": False}}
NAN = float("nan")


def base_card(learner_id: str = "demo-learner") -> dict:
    """Minimal in-memory card in the shape the audit repros used."""
    return {
        "learner_id": learner_id,
        "pseudonym": "Демо",
        "grade": 3,
        "learner_model": {
            "competencies": [
                {"topic_id": TOPIC, "mastery": 0.5, "independence": 0.5, "history": []}
            ]
        },
        "knowledge": [{"topic_id": TOPIC, "status": "learning"}],
    }


def base_card_with_nan() -> dict:
    """A card whose mastery is NaN; json.dumps writes the NaN literal back out."""
    card = base_card()
    card["learner_model"]["competencies"][0]["mastery"] = NAN
    return card


def write_card_file(tmp: Path, card: dict, learner_id: str = "demo-learner") -> Path:
    """Write the card under tmp/learners/ and return the learners directory."""
    learners = tmp / "learners"
    learners.mkdir(exist_ok=True)
    (learners / f"{learner_id}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return learners


def write_catalog_file(tmp: Path) -> Path:
    catalog_file = tmp / "_catalog.json"
    catalog_file.write_text(
        json.dumps({"topics": [{"id": TOPIC, "title_ru": "Демо-тема"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    return catalog_file


def run_main(module, argv: list[str]) -> tuple[int, str]:
    """Call a tool's main() in-process and capture stderr.

    ``configure_utf8_streams`` is neutralised: sys.stderr is a test double here
    (StringIO under redirect_stderr), and console re-encoding is not what these
    tests exercise.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        with mock.patch.object(module, "configure_utf8_streams", lambda: None):
            rc = module.main(argv)
    return rc, buffer.getvalue()


# --- F-054: NaN passes 0..1 checks, classified as zpd, poisons the EMA -------


class TestF054NonFiniteScores(unittest.TestCase):
    def test_unit_number_rejects_non_finite(self):
        self.assertIsNone(zpd._unit_number(NAN))
        self.assertIsNone(zpd._unit_number(float("inf")))
        self.assertIsNone(zpd._unit_number(float("-inf")))
        # Unusable types and finite values keep their existing behaviour.
        self.assertIsNone(zpd._unit_number("0.5"))
        self.assertIsNone(zpd._unit_number(True))
        self.assertEqual(zpd._unit_number(0.25), 0.25)

    def test_validator_unit_check_rejects_non_finite(self):
        findings = validate._check_unit_number(NAN, "f.json", "/mastery", 'поле "mastery"')
        self.assertTrue(findings and findings[0].is_error(), findings)
        self.assertIn("конечным", findings[0].message)  # explicit NaN message, not just the range one
        self.assertEqual(validate._check_unit_number(0.25, "f.json", "/mastery", "mastery"), [])

    def test_nan_mastery_is_classified_unknown_not_zpd(self):
        verdict = zpd.classify({"topic_id": TOPIC, "mastery": NAN})
        self.assertEqual(verdict.zone, zpd.ZONE_UNKNOWN)

    def test_nan_mastery_does_not_poison_the_ema(self):
        updated = zpd.update_competency(
            {"topic_id": TOPIC, "mastery": NAN, "independence": 0.2},
            zpd.Observation(TOPIC, "independent", True, "2026-11-01"),
        )
        self.assertFalse(math.isnan(updated["mastery"]))
        # The EMA restarts from the prior (0.5): a clean solo success must recover.
        self.assertAlmostEqual(updated["mastery"], 0.7)
        self.assertFalse(math.isnan(updated["history"][-1]["mastery"]))

    def test_validate_card_flags_nan_mastery(self):
        card = base_card()
        card["learner_model"]["competencies"][0]["mastery"] = NAN
        findings = validate.validate_card(card, MINI_CATALOG, "demo-learner.json")
        mastery_errors = [
            f for f in findings
            if f.is_error() and f.json_path.endswith("/learner_model/competencies/0/mastery")
        ]
        self.assertTrue(mastery_errors, [f.format() for f in findings])

    def test_validate_card_flags_non_finite_number_anywhere(self):
        card = base_card()
        card["learner_model"] = dict(card["learner_model"], learning_preferences={"visual": NAN})
        findings = validate.validate_card(card, MINI_CATALOG, "demo-learner.json")
        nonfinite = [
            f for f in findings
            if f.is_error() and "NaN" in f.message and "/learning_preferences" in f.json_path
        ]
        self.assertTrue(nonfinite, [f.format() for f in findings])

    def test_json_admits_nan_but_the_scan_finds_it(self):
        # json really does admit NaN literals by default – the explicit scan is
        # what closes that door for the tools' load/dump paths.
        parsed = json.loads('{"visual": NaN}')
        self.assertTrue(math.isnan(parsed["visual"]))
        self.assertIn("NaN", json.dumps(parsed))
        self.assertEqual(zpd.nonfinite_json_paths(parsed), ["/visual"])
        self.assertEqual(zpd.nonfinite_json_paths(base_card()), [])

    def test_validate_all_flags_nan_literal_on_disk(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            learners = write_card_file(tmp, base_card_with_nan())
            self.assertIn("NaN", (learners / "demo-learner.json").read_text(encoding="utf-8"))
            findings, _count = validate.validate_all(learners, write_catalog_file(tmp))
            self.assertTrue(
                any(f.is_error() and "NaN" in f.message for f in findings),
                [f.format() for f in findings],
            )

    def test_zpd_main_rejects_nan_card_without_rewriting(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            learners = write_card_file(tmp, base_card_with_nan())
            original = (learners / "demo-learner.json").read_text(encoding="utf-8")
            rc, stderr = run_main(zpd, [
                "demo-learner", "--write",
                "--learners-dir", str(tmp / "learners"), "--catalog", str(write_catalog_file(tmp)),
            ])
            self.assertEqual(rc, 1)
            self.assertIn("NaN", stderr)
            # Nothing is persisted back: no rewritten zone, no temp leftovers.
            self.assertEqual((learners / "demo-learner.json").read_text(encoding="utf-8"), original)
            self.assertEqual(list(learners.glob("*.tmp")), [])

    def test_progress_report_main_rejects_nan_card(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            write_card_file(tmp, base_card_with_nan())
            rc, stderr = run_main(progress_report, [
                "demo-learner",
                "--learners-dir", str(tmp / "learners"), "--catalog", str(write_catalog_file(tmp)),
            ])
            self.assertEqual(rc, 1)
            self.assertIn("NaN", stderr)


# --- F-055: --write must be atomic and refuse incomplete derived zones -------


class TestF055CardWrite(unittest.TestCase):
    def _write(self, tmp: Path) -> int:
        return run_main(zpd, [
            "demo-learner", "--write",
            "--learners-dir", str(tmp / "learners"), "--catalog", str(write_catalog_file(tmp)),
        ])[0]

    def test_write_success_updates_card_without_temp_leftovers(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            learners = write_card_file(tmp, base_card())  # no stored zpd block – write happens
            self.assertEqual(self._write(tmp), 0)
            card = json.loads((learners / "demo-learner.json").read_text(encoding="utf-8"))
            self.assertEqual(card["learner_model"]["zpd"], {"current": [TOPIC], "outside": []})
            self.assertEqual(list(learners.glob("*.tmp")), [])

    def test_crash_mid_write_keeps_the_old_card(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            learners = write_card_file(tmp, base_card())
            target = learners / "demo-learner.json"
            original = target.read_text(encoding="utf-8")

            real_fdopen = os.fdopen

            class DyingStream:
                """Writes half the payload, flushes, then dies – the on-disk
                state of a process kill inside the write window."""

                def __init__(self, stream):
                    self._stream = stream

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return self._stream.__exit__(*exc)

                def write(self, data):
                    self._stream.write(data[: len(data) // 2])
                    self._stream.flush()
                    raise RuntimeError("simulated crash mid-write")

            def fake_fdopen(fd, *args, **kwargs):
                return DyingStream(real_fdopen(fd, *args, **kwargs))

            with mock.patch("os.fdopen", side_effect=fake_fdopen):
                with self.assertRaises(RuntimeError):
                    self._write(tmp)

            self.assertEqual(target.read_text(encoding="utf-8"), original)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["pseudonym"], "Демо")
            self.assertEqual(list(learners.glob("*.tmp")), [])

    def test_replace_failure_cleans_up_temp_file(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            learners = write_card_file(tmp, base_card())
            original = (learners / "demo-learner.json").read_text(encoding="utf-8")
            payload = json.dumps(base_card(), ensure_ascii=False, indent=2) + "\n"
            with mock.patch("os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    zpd._write_card_atomic(learners / "demo-learner.json", payload)
            self.assertEqual((learners / "demo-learner.json").read_text(encoding="utf-8"), original)
            self.assertEqual(list(learners.glob("*.tmp")), [])

    def test_write_refuses_and_keeps_zone_when_competency_is_malformed(self):
        card = base_card()
        card["learner_model"]["competencies"].append({"mastery": 0.5})  # no topic_id
        card["learner_model"]["zpd"] = {"current": [TOPIC], "outside": []}
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            learners = write_card_file(tmp, card)
            original = (learners / "demo-learner.json").read_text(encoding="utf-8")
            rc, stderr = run_main(zpd, [
                "demo-learner", "--write",
                "--learners-dir", str(tmp / "learners"), "--catalog", str(write_catalog_file(tmp)),
            ])
            self.assertNotEqual(rc, 0)  # fail-closed: nonzero exit
            self.assertIn("компетенц", stderr)
            # The stored zone is NOT silently emptied to {"current": [], "outside": []}.
            self.assertEqual((learners / "demo-learner.json").read_text(encoding="utf-8"), original)

    def test_write_refuses_on_non_object_competency(self):
        card = base_card()
        card["learner_model"]["competencies"].append("not-an-object")
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            write_card_file(tmp, card)
            self.assertNotEqual(self._write(tmp), 0)

    def test_read_paths_still_skip_malformed_entries(self):
        card = base_card()
        card["learner_model"]["competencies"].append({"mastery": 0.5})
        verdicts, dropped = zpd.classify_card_dropped(card)
        self.assertEqual([verdict.topic_id for verdict in verdicts], [TOPIC])
        self.assertEqual(dropped, [1])
        self.assertEqual(zpd.derive_zpd(card), {"current": [TOPIC], "outside": []})


# --- F-056: progress_report crashes or is silently wrong ---------------------


class TestF056ProgressReport(unittest.TestCase):
    def _backfilled_card(self) -> dict:
        card = base_card()
        card = zpd.record_observation(card, topic_id=TOPIC, support_level="joint", solved=True,
                                      observed_on="2026-09-10")
        card = zpd.record_observation(card, topic_id=TOPIC, support_level="independent", solved=True,
                                      observed_on="2026-10-20")
        card = zpd.record_observation(card, topic_id=TOPIC, support_level="hint_question", solved=True,
                                      observed_on="2026-09-25")
        return card

    def test_backfilled_history_is_reported_by_date_not_append_order(self):
        card = self._backfilled_card()
        dates = [entry["date"] for entry in card["learner_model"]["competencies"][0]["history"]]
        self.assertEqual(dates, ["2026-09-10", "2026-10-20", "2026-09-25"])  # append order on disk

        report = progress_report.build_report(card, MINI_CATALOG)
        topic = report["topics"][0]
        self.assertEqual(topic["from_date"], "2026-09-10")
        self.assertEqual(topic["to_date"], "2026-10-20")  # the newest observation, not the last appended
        self.assertEqual(topic["help_after"], "independent")
        self.assertAlmostEqual(topic["mastery_before"], 0.4)
        self.assertAlmostEqual(topic["mastery_after"], 0.64)

    def test_missing_topic_id_is_skipped_with_loud_warning(self):
        card = base_card()
        card["learner_model"]["competencies"][0]["history"] = [
            {"date": "2026-10-02", "mastery": 0.6, "help_level": "joint",
             "support_level": "joint", "solved": True}
        ]
        card["learner_model"]["competencies"].append(
            {"mastery": 0.5,
             "history": [{"date": "2026-10-01", "mastery": 0.5, "help_level": "joint",
                          "support_level": "joint", "solved": True}]}
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            report = progress_report.build_report(card, MINI_CATALOG)
        self.assertEqual([topic["topic_id"] for topic in report["topics"]], [TOPIC])  # report still built
        self.assertIn("topic_id", stderr.getvalue())  # the skip is announced, not silent

    def test_null_error_count_is_explicit_validation_error(self):
        card = base_card()
        card["error_patterns"] = [{"error_tag": "a", "count": None}, {"error_tag": "b", "count": 2}]
        with self.assertRaises(ValueError) as ctx:
            progress_report.build_report(card, MINI_CATALOG)
        self.assertIn("count", str(ctx.exception))

    def test_null_points_is_explicit_validation_error(self):
        card = base_card()
        card["mvp"] = {"points_ledger": [{"date": "2026-07-01", "points": None, "reason": "x", "role": "tutor"}]}
        with self.assertRaises(ValueError) as ctx:
            progress_report.build_report(card, MINI_CATALOG)
        self.assertIn("points", str(ctx.exception))

    def test_non_integer_minutes_is_explicit_validation_error(self):
        card = base_card()
        card["mvp"] = {"lessons": [{"date": "2026-07-01", "topic_ids": [], "minutes": "forty"}]}
        with self.assertRaises(ValueError) as ctx:
            progress_report.build_report(card, MINI_CATALOG)
        self.assertIn("minutes", str(ctx.exception))

    def test_null_minutes_stays_tolerated(self):
        card = base_card()
        card["mvp"] = {"lessons": [{"date": "2026-07-01", "topic_ids": [], "minutes": None}]}
        report = progress_report.build_report(card, MINI_CATALOG)
        self.assertEqual(report["minutes"], 0)

    def test_string_mastery_is_explicit_validation_error(self):
        card = base_card()
        card["learner_model"]["competencies"][0]["history"] = [
            {"date": "2026-10-01", "mastery": "0.5", "help_level": "joint",
             "support_level": "joint", "solved": True}
        ]
        with self.assertRaises(ValueError) as ctx:
            progress_report.build_report(card, MINI_CATALOG)
        self.assertIn("mastery", str(ctx.exception))

    def test_render_text_rejects_non_numeric_mastery_explicitly(self):
        report = {
            "learner_id": "demo-learner", "pseudonym": "Демо", "grade": 3,
            "since": None, "until": None,
            "topics": [{
                "topic_id": TOPIC, "mastery_before": "0.5", "mastery_after": 0.6,
                "mastery_delta": 0.1, "help_before": "joint", "help_after": "independent",
                "observations": 2, "from_date": "2026-09-01", "to_date": "2026-10-01",
                "zone": "zpd", "zone_ru": "в ЗБР (делает с опорой)",
                "stale": False, "stale_days": None,
            }],
            "journal": [], "lessons": 0, "minutes": 0, "points_earned": 0,
            "error_patterns": [], "zpd_now": {"current": [], "outside": []},
        }
        with self.assertRaises(ValueError) as ctx:
            progress_report.render_text(report, MINI_CATALOG)
        # An explicit validation message, not "Unknown format code 'f'".
        self.assertIn("освоени", str(ctx.exception))

    def test_main_reports_validation_error_with_nonzero_exit(self):
        card = base_card()
        card["error_patterns"] = [{"error_tag": "a", "count": None}]
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            write_card_file(tmp, card)
            rc, stderr = run_main(progress_report, [
                "demo-learner",
                "--learners-dir", str(tmp / "learners"), "--catalog", str(write_catalog_file(tmp)),
            ])
            self.assertEqual(rc, 1)
            self.assertIn("count", stderr)


# --- F-057: unhashable difficulty aborts repo-wide validation ----------------


class TestF057UnhashableDifficulty(unittest.TestCase):
    def _card_with_history(self, difficulty: object) -> dict:
        card = base_card()
        card["learner_model"]["competencies"][0]["history"] = [
            {"date": "2026-07-01", "mastery": 0.5, "help_level": "joint",
             "support_level": "joint", "solved": True, "difficulty": difficulty}
        ]
        return card

    def test_list_difficulty_is_a_finding_not_a_crash(self):
        findings = validate.validate_card(self._card_with_history(["easy"]), MINI_CATALOG, "demo-learner.json")
        self.assertTrue(
            any(f.is_error() and f.json_path.endswith("/history/0/difficulty") for f in findings),
            [f.format() for f in findings],
        )

    def test_dict_difficulty_is_a_finding_not_a_crash(self):
        findings = validate.validate_card(self._card_with_history({"level": "easy"}), MINI_CATALOG, "demo-learner.json")
        self.assertTrue(
            any(f.is_error() and f.json_path.endswith("/history/0/difficulty") for f in findings),
            [f.format() for f in findings],
        )

    def test_valid_difficulty_still_passes(self):
        findings = validate.validate_card(self._card_with_history("easy"), MINI_CATALOG, "demo-learner.json")
        self.assertFalse([f for f in findings if f.json_path.endswith("/difficulty")])

    def test_null_difficulty_is_still_a_finding(self):
        findings = validate.validate_card(self._card_with_history(None), MINI_CATALOG, "demo-learner.json")
        self.assertTrue(
            any(f.is_error() and f.json_path.endswith("/history/0/difficulty") for f in findings),
            [f.format() for f in findings],
        )

    def test_finite_card_gets_no_non_finite_findings(self):
        findings = validate.validate_card(base_card(), MINI_CATALOG, "demo-learner.json")
        self.assertFalse([f for f in findings if "NaN" in f.message], [f.format() for f in findings])


if __name__ == "__main__":
    unittest.main(verbosity=2)
