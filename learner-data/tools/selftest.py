#!/usr/bin/env python3
"""Self-test suite for learner-data tools (stdlib unittest only).

Run: python learner-data/tools/selftest.py   (works from any cwd)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learner_common import find_package_root, iter_learner_files, load_catalog, load_json
    import build_context
    import progress_report
    import validate
    import zpd
else:
    from .learner_common import find_package_root, iter_learner_files, load_catalog, load_json
    from . import build_context
    from . import progress_report
    from . import validate
    from . import zpd

PACKAGE_ROOT = find_package_root()
LEARNERS_DIR = PACKAGE_ROOT / "learners"
CATALOG_FILE = PACKAGE_ROOT / "catalog" / "math_g3_g4.json"
BENCHMARK_FILE = PACKAGE_ROOT / "benchmarks" / "rag_behavior_cases.json"
TOOLS_DIR = Path(__file__).resolve().parent


def _minimal_valid_card(learner_id: str = "test-01", pseudonym: str = "Тест") -> dict:
    """A minimal structurally-valid card, used as the base for negative fixtures."""
    return {
        "schema_version": 2,
        "learner_id": learner_id,
        "pseudonym": pseudonym,
        "age_group": "9-10",
        "grade": "3",
        "language": "russian",
        "rag_context": {
            "current_goal": {
                "topic_id": "math.g3.numbers.multiplication_table.times_table",
                "goal": "закрепить таблицу умножения",
                "updated_at": "2026-06-01",
            },
            "current_topics": ["math.g3.numbers.multiplication_table.times_table"],
            "priority_difficulties": [],
            "effective_strategies": ["разбить на короткие шаги"],
            "avoid": ["не давать длинную инструкцию"],
            "recent_progress": {"date": "2026-06-01", "note": "Первое занятие."},
        },
        "profile": {
            "explanation_style": "просто",
            "pace": "средний",
            "autonomy_level": "средний",
            "motivation": "похвала",
            "prefers_visual": False,
        },
        "interests": [],
        "knowledge": [],
        "error_patterns": [],
        "mvp": {
            "story_preferences": [],
            "learning_preferences_observed": [],
            "help_strategies": [],
            "journal": [],
            "points_ledger": [],
        },
    }


class TestValidatorOnRealData(unittest.TestCase):
    """Check 1: validator over real learners/ + catalog: 0 errors, 0 warnings."""

    def test_real_data_is_clean(self):
        findings, file_count = validate.validate_all(LEARNERS_DIR, CATALOG_FILE)
        errors = [f for f in findings if f.is_error()]
        warnings = [f for f in findings if not f.is_error()]
        details = "\n".join(f.format() for f in findings)
        self.assertEqual(len(errors), 0, f"unexpected errors:\n{details}")
        self.assertEqual(len(warnings), 0, f"unexpected warnings:\n{details}")
        self.assertEqual(file_count, 17)


class TestCatalogShape(unittest.TestCase):
    """Check 2: catalog has 40-60 non-deprecated leaf entries; both grades have
    all 6 sections; all parents resolve."""

    def setUp(self):
        self.catalog = load_catalog(CATALOG_FILE)
        data, _ = load_json(CATALOG_FILE)
        self.topics = data["topics"]

    def test_leaf_count_in_range(self):
        parent_ids = {t["parent"] for t in self.topics if t["parent"]}
        leaves = [t for t in self.topics if t["id"] not in parent_ids]
        non_deprecated_leaves = [t for t in leaves if not t["deprecated"]]
        self.assertGreaterEqual(len(non_deprecated_leaves), 40)
        self.assertLessEqual(len(non_deprecated_leaves), 60)

    def test_six_sections_per_grade(self):
        for grade in (3, 4):
            sections = {
                t["id"] for t in self.topics
                if t["grade"] == grade and t["parent"] is None
            }
            self.assertEqual(len(sections), 6, f"grade {grade} should have 6 top-level sections, got {sections}")

    def test_all_parents_resolve(self):
        ids = {t["id"] for t in self.topics}
        for t in self.topics:
            if t["parent"] is not None:
                self.assertIn(t["parent"], ids, f'parent "{t["parent"]}" of "{t["id"]}" not found in catalog')


class TestContextPerLearner(unittest.TestCase):
    """Check 3: the compact RAG projection contains only curated data.

    The test verifies isolation between learners, balanced tags and that direct
    identifiers do not leak while anonymised personalisation does. Plus one
    subprocess smoke test.
    """

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(CATALOG_FILE)
        cls.files = iter_learner_files(LEARNERS_DIR)
        cls.cards = []
        for path in cls.files:
            card, _ = load_json(path)
            cls.cards.append((path.stem, card))

    def test_all_demo_cards_present(self):
        self.assertEqual(len(self.cards), 17)

    def test_each_card_context_isolated_and_balanced(self):
        all_pseudonyms = {card["pseudonym"] for _, card in self.cards}
        for learner_id, card in self.cards:
            with self.subTest(learner_id=learner_id):
                text = build_context.build_context_text(card, self.catalog)

                own_pseudonym = card["pseudonym"]
                self.assertIn(own_pseudonym, text)

                other_pseudonyms = all_pseudonyms - {own_pseudonym}
                for other in other_pseudonyms:
                    self.assertNotIn(other, text, f'leaked pseudonym "{other}" into context for "{learner_id}"')

                self._assert_tags_balanced(text)
                self.assertIn("<learner_rag_context>", text)
                self.assertIn(card["rag_context"]["current_goal"]["goal"], text)
                self.assertIn("<learner_personalization>", text)
                self.assertLess(len(text), 5000, "RAG projection must stay bounded")
                legal_name = card.get("legal_name") or {}
                for value in (legal_name.get("last_name"), legal_name.get("patronymic")):
                    if not value or value == "-":
                        continue
                    self.assertNotIn(value, text, "legal name must not reach RAG")
                if card["interests"]:
                    self.assertIn(card["interests"][0], text)
                if card["mvp"]["points_ledger"]:
                    self.assertIn(card["mvp"]["points_ledger"][-1]["reason"], text)

    def _assert_tags_balanced(self, text: str):
        stack = []
        for tag_match in re.finditer(r"</?([a-z_]+)>", text):
            full = tag_match.group(0)
            name = tag_match.group(1)
            if full.startswith("</"):
                self.assertTrue(stack, f"closing tag {full} with empty stack")
                self.assertEqual(stack.pop(), name, f"mismatched closing tag {full}")
            else:
                stack.append(name)
        self.assertEqual(stack, [], f"unclosed tags: {stack}")

    def test_subprocess_smoke_one_learner(self):
        learner_id = self.cards[0][0]
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "build_context.py"), learner_id],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())


class TestJsonFormat(unittest.TestCase):
    """Check 4: --format json output parses and equals on-disk content."""

    def test_json_format_matches_disk(self):
        for path in iter_learner_files(LEARNERS_DIR):
            learner_id = path.stem
            with self.subTest(learner_id=learner_id):
                on_disk, _ = load_json(path)
                text_output = build_context.build_context_json(on_disk)
                parsed = json.loads(text_output)
                self.assertEqual(parsed, on_disk)

    def test_json_format_subprocess(self):
        learner_id = iter_learner_files(LEARNERS_DIR)[0].stem
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "build_context.py"), learner_id, "--format", "json"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        on_disk, _ = load_json(LEARNERS_DIR / f"{learner_id}.json")
        self.assertEqual(parsed, on_disk)


class TestRagBehaviorBenchmark(unittest.TestCase):
    """The future RAG team receives one behaviour case for every demo learner."""

    def test_cases_cover_every_demo_learner_once(self):
        data, _ = load_json(BENCHMARK_FILE)
        self.assertEqual(data["schema_version"], 1)
        cases = data["cases"]
        learner_ids = [case["learner_id"] for case in cases]
        self.assertEqual(len(cases), 17)
        self.assertEqual(len(learner_ids), len(set(learner_ids)))
        self.assertEqual(
            set(learner_ids),
            {path.stem for path in iter_learner_files(LEARNERS_DIR)},
        )
        for case in cases:
            self.assertTrue(case["question"].strip())
            self.assertTrue(case["must_do"])


class TestMissingLearnerSubprocess(unittest.TestCase):
    """Check 5: build_context.py no-such-child -> exit 2, empty stdout, non-empty stderr."""

    def test_missing_learner_exit_2(self):
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "build_context.py"), "no-such-child"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.strip())

    def test_invalid_learner_id_is_rejected_before_lookup(self):
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "build_context.py"), "../catalog/math_g3_g4"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("Некорректный learner_id", result.stderr)


class TestNegativeFixtures(unittest.TestCase):
    """Check 6: in-memory dict fixtures (not files) — each produces the expected
    validator finding via validate_card(card, catalog, filename)."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(CATALOG_FILE)

    def _findings_for(self, card: dict):
        return validate.validate_card(card, self.catalog, "fixture.json")

    def test_bad_status(self):
        card = _minimal_valid_card()
        card["knowledge"] = [{"topic_id": "math.g3.numbers.multiplication_table.times_table", "status": "amazing"}]
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and "/knowledge/0/status" in f.json_path for f in findings),
            [f.format() for f in findings],
        )

    def test_unknown_topic_id(self):
        card = _minimal_valid_card()
        card["knowledge"] = [{"topic_id": "math.g3.does_not_exist", "status": "learning"}]
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and "/knowledge/0/topic_id" in f.json_path and "неизвестный topic_id" in f.message for f in findings),
            [f.format() for f in findings],
        )

    def test_malformed_date(self):
        card = _minimal_valid_card()
        card["error_patterns"] = [{
            "subject": "math",
            "topic_id": "math.g3.numbers.multiplication_table.times_table",
            "error_tag": "test",
            "count": 1,
            "last_seen": "2026-7-1",
        }]
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and "/error_patterns/0/last_seen" in f.json_path for f in findings),
            [f.format() for f in findings],
        )

    def test_balance_key_forbidden(self):
        # Place the balance-like key inside mvp.story_preferences items, which have
        # no dedicated key allowlist (unlike e.g. mvp itself or points_ledger items).
        # This isolates the recursive _contains_balance_key rule: if it were deleted,
        # no other check in validate.py would flag this key, and the test would fail.
        card = _minimal_valid_card()
        card["mvp"]["story_preferences"] = [{"title": "space", "points_balance": 100}]
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and "баланс всегда вычисляется" in f.message for f in findings),
            [f.format() for f in findings],
        )

    def test_unknown_top_level_key(self):
        card = _minimal_valid_card()
        card["favorite_color"] = "blue"
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and f.json_path == "/favorite_color" for f in findings),
            [f.format() for f in findings],
        )

    def test_points_bool_rejected(self):
        card = _minimal_valid_card()
        card["mvp"]["points_ledger"] = [{
            "date": "2026-06-01", "points": True, "reason": "test", "role": "tutor",
        }]
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and "/mvp/points_ledger/0/points" in f.json_path for f in findings),
            [f.format() for f in findings],
        )

    def test_non_string_element_in_string_array(self):
        card = _minimal_valid_card()
        card["mvp"]["help_strategies"] = ["дать пример", "разбить на шаги", 42]
        findings = self._findings_for(card)
        self.assertTrue(
            any(
                f.is_error() and f.json_path == "/mvp/help_strategies/2" and "должен быть строкой" in f.message
                for f in findings
            ),
            [f.format() for f in findings],
        )

    def test_non_string_topic_id_in_knowledge(self):
        card = _minimal_valid_card()
        card["knowledge"] = [{"topic_id": 123, "status": "learning"}]
        findings = self._findings_for(card)
        self.assertTrue(
            any(
                f.is_error() and f.json_path == "/knowledge/0/topic_id" and "должно быть строкой" in f.message
                for f in findings
            ),
            [f.format() for f in findings],
        )

    def test_rag_context_limit_is_enforced(self):
        card = _minimal_valid_card()
        card["rag_context"]["effective_strategies"] = ["один", "два", "три", "четыре"]
        findings = self._findings_for(card)
        self.assertTrue(
            any(f.is_error() and f.json_path == "/rag_context/effective_strategies" for f in findings),
            [f.format() for f in findings],
        )


class TestZpdPolicy(unittest.TestCase):
    """Check 7: the ЗБР policy derived from the competence map (concept section 2)."""

    def test_high_mastery_worked_alone_is_mastered(self):
        verdict = zpd.classify({"topic_id": "math.g3.quantities.length", "mastery": 0.92, "independence": 0.88})
        self.assertEqual(verdict.zone, zpd.ZONE_MASTERED)
        self.assertFalse(verdict.needs_scaffolding)

    def test_high_mastery_but_low_independence_stays_in_zone(self):
        """«Умеет с помощью, но не сам» — это и есть зона, а не освоенная тема."""
        verdict = zpd.classify({"topic_id": "math.g3.quantities.length", "mastery": 0.9, "independence": 0.4})
        self.assertEqual(verdict.zone, zpd.ZONE_ZPD)

    def test_mid_mastery_is_zone_and_low_mastery_is_outside(self):
        self.assertEqual(zpd.classify({"topic_id": "math.g3.geometry.circle", "mastery": 0.55}).zone, zpd.ZONE_ZPD)
        self.assertEqual(zpd.classify({"topic_id": "math.g3.geometry.circle", "mastery": 0.2}).zone, zpd.ZONE_OUTSIDE)

    def test_missing_mastery_is_unknown_not_guessed(self):
        self.assertEqual(zpd.classify({"topic_id": "math.g3.geometry.circle"}).zone, zpd.ZONE_UNKNOWN)

    def test_hint_depth_shrinks_as_mastery_grows(self):
        deep = zpd.classify({"topic_id": "math.g3.geometry.circle", "mastery": 0.4}).hint_depth
        shallow = zpd.classify({"topic_id": "math.g3.geometry.circle", "mastery": 0.75}).hint_depth
        self.assertGreater(deep, shallow)

    def test_unknown_help_level_is_rejected(self):
        with self.assertRaises(zpd.ZpdPolicyError):
            zpd.help_level("magic_hint")
        with self.assertRaises(zpd.ZpdPolicyError):
            zpd.help_level(True)  # bool is an int in Python and must not pass as level 1

    def test_unsolved_attempt_gives_no_credit(self):
        self.assertEqual(zpd.credit_for("independent", solved=False), 0.0)
        self.assertGreater(zpd.credit_for("independent", solved=True), zpd.credit_for("joint", solved=True))

    def test_one_solo_success_is_not_mastery_but_three_are(self):
        competency = None
        zones = []
        for _ in range(3):
            competency = zpd.update_competency(
                competency,
                zpd.Observation("math.g3.quantities.length", "independent", True, "2026-07-26"),
            )
            zones.append(zpd.classify(competency).zone)
        self.assertEqual(zones[0], zpd.ZONE_ZPD, "одна удачная попытка не означает освоения")
        self.assertEqual(zones[-1], zpd.ZONE_MASTERED)
        self.assertFalse(competency["needs_scaffolding"], "опоры должны сниматься автоматически")

    def test_needs_scaffolding_does_not_latch(self):
        """Флаг производный: он не должен запирать тему в зоне навсегда."""
        competency = {"topic_id": "math.g3.quantities.length", "mastery": 0.84, "independence": 0.84,
                      "needs_scaffolding": True}
        updated = zpd.update_competency(
            competency, zpd.Observation("math.g3.quantities.length", "independent", True, "2026-07-26")
        )
        self.assertEqual(zpd.classify(updated).zone, zpd.ZONE_MASTERED)

    def test_one_bad_attempt_cannot_erase_the_history(self):
        competency = {"topic_id": "math.g3.quantities.length", "mastery": 0.93, "independence": 0.9}
        updated = zpd.update_competency(
            competency, zpd.Observation("math.g3.quantities.length", "joint", False, "2026-07-26")
        )
        self.assertAlmostEqual(updated["mastery"], 0.93 - zpd.MAX_MASTERY_DROP, places=3)

    def test_history_is_capped(self):
        competency = None
        for day in range(1, zpd.HISTORY_LIMIT + 4):
            competency = zpd.update_competency(
                competency,
                zpd.Observation("math.g3.quantities.length", "visual", True, f"2026-07-{day:02d}"),
            )
        self.assertEqual(len(competency["history"]), zpd.HISTORY_LIMIT)

    def test_confidence_grows_with_observations(self):
        first = zpd.update_competency(None, zpd.Observation("math.g3.quantities.length", "visual", True, "2026-07-26"))
        second = zpd.update_competency(first, zpd.Observation("math.g3.quantities.length", "visual", True, "2026-07-27"))
        self.assertGreater(second["confidence"], first["confidence"])

    def test_old_assessment_is_marked_stale(self):
        """У освоения нет забывания, поэтому давность оценки — единственный сигнал."""
        fresh = zpd.classify({"topic_id": "math.g3.quantities.length", "mastery": 0.9,
                              "independence": 0.9, "last_assessed": "2026-07-22"})
        old = zpd.classify({"topic_id": "math.g3.quantities.length", "mastery": 0.9,
                            "independence": 0.9, "last_assessed": "2026-05-01"})

        self.assertFalse(fresh.is_stale("2026-07-26"))
        self.assertTrue(old.is_stale("2026-07-26"))
        self.assertEqual(old.stale_days("2026-07-26"), 86)
        # Зона от давности не меняется: занижать оценку без наблюдения нельзя.
        self.assertEqual(old.zone, fresh.zone)

    def test_missing_or_broken_date_is_not_stale(self):
        without = zpd.classify({"topic_id": "math.g3.quantities.length", "mastery": 0.5})
        broken = zpd.classify({"topic_id": "math.g3.quantities.length", "mastery": 0.5,
                               "last_assessed": "01.05.2026"})
        self.assertIsNone(without.stale_days("2026-07-26"))
        self.assertFalse(broken.is_stale("2026-07-26"))

    def test_preferred_support_prefers_topic_specific_and_deduplicates(self):
        card, _ = load_json(LEARNERS_DIR / "zvezdin-artem.json")
        support = zpd.preferred_support(card, "math.g3.geometry.point_line_ray_segment")
        self.assertTrue(support[0].startswith("сравнить луч"), support)
        self.assertEqual(len(support), len(set(support)))
        self.assertEqual(
            sum(1 for item in support if "фонарик" in item), 1,
            f"одна и та же опора попала в промпт дважды: {support}",
        )

    def test_select_target_uses_requested_topic_then_goal(self):
        card, _ = load_json(LEARNERS_DIR / "zvezdin-artem.json")
        requested = zpd.select_target(card, "math.g3.quantities.length")
        self.assertEqual(requested.topic_id, "math.g3.quantities.length")
        default = zpd.select_target(card)
        self.assertEqual(default.topic_id, card["rag_context"]["current_goal"]["topic_id"])

    def test_stored_zpd_matches_derived_for_every_real_card(self):
        """Пункт 3 концепта — производная проекция пункта 2, а не второй источник."""
        for path in iter_learner_files(LEARNERS_DIR):
            card, _ = load_json(path)
            learner_model = card.get("learner_model")
            if not isinstance(learner_model, dict) or not learner_model.get("competencies"):
                continue
            with self.subTest(learner_id=path.stem):
                self.assertEqual(learner_model.get("zpd"), zpd.derive_zpd(card))


class TestCompetencyValidation(unittest.TestCase):
    """Check 8: the competence map is validated, because an LLM may write into it."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(CATALOG_FILE)

    def _findings_for(self, competencies: list) -> list:
        card = _minimal_valid_card()
        card["learner_model"] = {
            "competencies": competencies,
            "zpd": validate.zpd.derive_zpd({"learner_model": {"competencies": competencies}}),
            "learning_preferences": {}, "engagement": {}, "ai_usage": {}, "gamification": {},
            "projects": [], "strengths": [], "support_needs": [], "scaffolding_by_topic": [],
        }
        return validate.validate_card(card, self.catalog, "test-01.json")

    def _assert_error_at(self, findings: list, json_path: str) -> None:
        self.assertTrue(
            any(f.is_error() and f.json_path == json_path for f in findings),
            [f.format() for f in findings],
        )

    def test_mastery_out_of_range_is_an_error(self):
        findings = self._findings_for([{"topic_id": "math.g3.geometry.circle", "mastery": 1.4}])
        self._assert_error_at(findings, "/learner_model/competencies/0/mastery")

    def test_mastery_must_be_a_number_not_a_string(self):
        findings = self._findings_for([{"topic_id": "math.g3.geometry.circle", "mastery": "0.5"}])
        self._assert_error_at(findings, "/learner_model/competencies/0/mastery")

    def test_missing_mastery_is_an_error(self):
        findings = self._findings_for([{"topic_id": "math.g3.geometry.circle"}])
        self._assert_error_at(findings, "/learner_model/competencies/0/mastery")

    def test_unknown_topic_id_is_an_error(self):
        findings = self._findings_for([{"topic_id": "math.g3.geometry.no_such_topic", "mastery": 0.5}])
        self._assert_error_at(findings, "/learner_model/competencies/0/topic_id")

    def test_duplicate_topic_id_is_an_error(self):
        findings = self._findings_for([
            {"topic_id": "math.g3.geometry.circle", "mastery": 0.5},
            {"topic_id": "math.g3.geometry.circle", "mastery": 0.6},
        ])
        self._assert_error_at(findings, "/learner_model/competencies/1/topic_id")

    def test_unknown_help_level_in_history_is_an_error(self):
        findings = self._findings_for([{
            "topic_id": "math.g3.geometry.circle",
            "mastery": 0.5,
            "history": [{"date": "2026-07-01", "help_level": "telepathy"}],
        }])
        self._assert_error_at(findings, "/learner_model/competencies/0/history/0/help_level")

    def test_unknown_key_in_competency_is_an_error(self):
        findings = self._findings_for([{"topic_id": "math.g3.geometry.circle", "mastery": 0.5, "score": 80}])
        self._assert_error_at(findings, "/learner_model/competencies/0/score")

    def test_stale_zpd_block_is_a_warning_not_an_error(self):
        card = _minimal_valid_card()
        card["learner_model"] = {
            "competencies": [{"topic_id": "math.g3.geometry.circle", "mastery": 0.5}],
            "zpd": {"current": [], "outside": []},
            "learning_preferences": {}, "engagement": {}, "ai_usage": {}, "gamification": {},
            "projects": [], "strengths": [], "support_needs": [], "scaffolding_by_topic": [],
        }
        findings = validate.validate_card(card, self.catalog, "test-01.json")
        self.assertFalse([f for f in findings if f.is_error()], [f.format() for f in findings])
        self.assertTrue(
            any(not f.is_error() and f.json_path == "/learner_model/zpd/current" for f in findings),
            [f.format() for f in findings],
        )


class TestProgressReport(unittest.TestCase):
    """Check 9: отчёт о динамике собирается из истории оценок и ничего не выдумывает."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(CATALOG_FILE)
        cls.card, _ = load_json(LEARNERS_DIR / "zvezdin-artem.json")

    def _topic(self, report: dict, topic_id: str) -> dict:
        return next(item for item in report["topics"] if item["topic_id"] == topic_id)

    def test_report_shows_growth_and_less_help_needed(self):
        report = progress_report.build_report(self.card, self.catalog)
        topic = self._topic(report, "math.g3.geometry.point_line_ray_segment")

        self.assertEqual(topic["observations"], 3)
        self.assertAlmostEqual(topic["mastery_delta"], 0.27, places=3)
        self.assertEqual(topic["help_before"], "joint")
        self.assertEqual(topic["help_after"], "hint_question")
        self.assertEqual(topic["zone"], zpd.ZONE_ZPD)

    def test_since_filters_earlier_observations(self):
        report = progress_report.build_report(self.card, self.catalog, since="2026-07-15")
        topic = self._topic(report, "math.g3.geometry.point_line_ray_segment")

        self.assertEqual(topic["observations"], 2)
        self.assertEqual(topic["from_date"], "2026-07-15")
        self.assertEqual(report["journal"][0]["date"], "2026-07-15")

    def test_report_contains_no_direct_identifiers(self):
        text = json.dumps(progress_report.build_report(self.card, self.catalog), ensure_ascii=False)
        legal_name = self.card.get("legal_name") or {}
        self.assertNotIn(legal_name.get("last_name", "нет-такой-фамилии"), text)
        self.assertNotIn("legal_name", text)

    def test_card_without_competence_map_gives_an_empty_but_valid_report(self):
        card, _ = load_json(LEARNERS_DIR / "belka-06.json")
        report = progress_report.build_report(card, self.catalog)

        self.assertEqual(report["topics"], [])
        self.assertIn("оценок по темам не записывали", progress_report.render_text(report, self.catalog))

    def test_subprocess_smoke_and_missing_learner(self):
        ok = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "progress_report.py"), "zvezdin-artem"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("Динамика по темам", ok.stdout)

        missing = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "progress_report.py"), "no-such-child"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(missing.returncode, 2)

    def test_bad_date_is_rejected(self):
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "progress_report.py"), "zvezdin-artem", "--since", "01.07.2026"],
            capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("YYYY-MM-DD", result.stderr)


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from learner_common import configure_utf8_streams
    else:
        from .learner_common import configure_utf8_streams
    configure_utf8_streams()
    runner = unittest.main(exit=False, argv=[sys.argv[0], "-v"])
    sys.exit(0 if runner.result.wasSuccessful() else 1)
