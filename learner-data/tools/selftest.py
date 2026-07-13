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
    import validate
else:
    from .learner_common import find_package_root, iter_learner_files, load_catalog, load_json
    from . import build_context
    from . import validate

PACKAGE_ROOT = find_package_root()
LEARNERS_DIR = PACKAGE_ROOT / "learners"
CATALOG_FILE = PACKAGE_ROOT / "catalog" / "math_g3_g4.json"
TOOLS_DIR = Path(__file__).resolve().parent


def _minimal_valid_card(learner_id: str = "test-01", pseudonym: str = "Тест") -> dict:
    """A minimal structurally-valid card, used as the base for negative fixtures."""
    return {
        "schema_version": 1,
        "learner_id": learner_id,
        "pseudonym": pseudonym,
        "age_group": "9-10",
        "grade": "3",
        "language": "russian",
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
        self.assertEqual(file_count, 12)


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
    """Check 3: for each of the 12 cards, text context builds in-process, contains
    own pseudonym and none of the 11 others', all XML tags balanced, Баллы: line
    matches independently-computed ledger sum. Plus one subprocess smoke test."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(CATALOG_FILE)
        cls.files = iter_learner_files(LEARNERS_DIR)
        cls.cards = []
        for path in cls.files:
            card, _ = load_json(path)
            cls.cards.append((path.stem, card))

    def test_twelve_cards_present(self):
        self.assertEqual(len(self.cards), 12)

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

                expected_balance = sum(p.get("points", 0) for p in card["mvp"]["points_ledger"])
                self.assertIn(
                    f"Баллы: {expected_balance}",
                    text.splitlines(),
                    f'expected an exact "Баллы: {expected_balance}" line in text',
                )

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
            capture_output=True, text=True,
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
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        parsed = json.loads(result.stdout)
        on_disk, _ = load_json(LEARNERS_DIR / f"{learner_id}.json")
        self.assertEqual(parsed, on_disk)


class TestMissingLearnerSubprocess(unittest.TestCase):
    """Check 5: build_context.py no-such-child -> exit 2, empty stdout, non-empty stderr."""

    def test_missing_learner_exit_2(self):
        result = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "build_context.py"), "no-such-child"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.strip())


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


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from learner_common import configure_utf8_streams
    else:
        from .learner_common import configure_utf8_streams
    configure_utf8_streams()
    runner = unittest.main(exit=False, argv=[sys.argv[0], "-v"])
    sys.exit(0 if runner.result.wasSuccessful() else 1)
