from __future__ import annotations

from copy import deepcopy
import unittest

from research_memory.contracts import (
    ACTION_SCHEMA,
    EXTRACT_SCHEMA,
    action_schema,
    extract_schema,
)


class CitationSchemaTests(unittest.TestCase):
    def test_extract_uses_processed_canonical_ids_and_exact_topic(self):
        source_ids = {"h07-l03-t04", "h07-l01-t01"}
        fields = extract_schema(source_ids, "math.division")["properties"]["records"]["items"]["properties"]

        self.assertEqual(fields["topic_id"], {"type": "string", "enum": ["math.division"]})
        for name in ("evidence_turn_ids",):
            self.assertEqual(fields[name], {
                "type": "array", "items": {"type": "string", "enum": ["h07-l01-t01", "h07-l03-t04"]},
                "uniqueItems": True, "maxItems": 2,
            })
            for malformed_id in ("r0001", "2026-09-03", "h07-l03-t4"):
                self.assertNotIn(malformed_id, fields[name]["items"]["enum"])

    def test_action_only_exposes_visible_ids(self):
        fields = action_schema({"h06-l07-t06", "h06-l07-t04"})["properties"]

        self.assertEqual(fields["evidence_turn_ids"]["items"]["enum"], ["h06-l07-t04", "h06-l07-t06"])
        self.assertEqual(fields["evidence_turn_ids"]["maxItems"], 2)
        for absent_id in ("h06-l01-t01", "r0001", "2026-09-07", "h06-l07-t4"):
            self.assertNotIn(absent_id, fields["evidence_turn_ids"]["items"]["enum"])
        self.assertEqual(fields["review_required"], ACTION_SCHEMA["properties"]["review_required"])

    def test_empty_context_allows_only_empty_citation_arrays(self):
        extract_fields = extract_schema(set(), "math.division")["properties"]["records"]["items"]["properties"]
        citation_schemas = [
            extract_fields["evidence_turn_ids"],
            action_schema(set())["properties"]["evidence_turn_ids"],
        ]

        for schema in citation_schemas:
            self.assertEqual(schema, {
                "type": "array", "items": {"type": "string"}, "uniqueItems": True, "maxItems": 0,
            })

    def test_factories_are_deeply_independent_of_globals_and_previous_results(self):
        original_extract = deepcopy(EXTRACT_SCHEMA)
        original_action = deepcopy(ACTION_SCHEMA)
        first_extract = extract_schema({"t01"}, "topic.first")
        first_action = action_schema({"t01"})
        second_extract = extract_schema({"t02"}, "topic.second")
        second_action = action_schema({"t02"})
        saved_second_extract = deepcopy(second_extract)
        saved_second_action = deepcopy(second_action)

        first_fields = first_extract["properties"]["records"]["items"]["properties"]
        first_fields["evidence_turn_ids"]["items"]["enum"].append("changed")
        first_fields["kind"]["enum"].append("changed")
        first_extract["properties"]["records"]["items"]["required"].append("changed")
        first_action["properties"]["evidence_turn_ids"]["items"]["enum"].append("changed")
        first_action["properties"]["action"]["enum"].append("changed")
        first_action["required"].append("changed")

        self.assertEqual(first_fields["support_evidence"]["anyOf"][1]["properties"]["source_turn_id"]["enum"], ["t01"])
        self.assertEqual(EXTRACT_SCHEMA, original_extract)
        self.assertEqual(ACTION_SCHEMA, original_action)
        self.assertEqual(second_extract, saved_second_extract)
        self.assertEqual(second_action, saved_second_action)

    def test_input_set_mutation_does_not_change_existing_schema(self):
        source_ids = {"t01"}
        extraction = extract_schema(source_ids, "topic")
        action = action_schema(source_ids)
        source_ids.add("t02")

        extraction_fields = extraction["properties"]["records"]["items"]["properties"]
        self.assertEqual(extraction_fields["evidence_turn_ids"]["items"]["enum"], ["t01"])
        self.assertEqual(extraction_fields["support_evidence"]["anyOf"][1]["properties"]["source_turn_id"]["enum"], ["t01"])
        self.assertEqual(action["properties"]["evidence_turn_ids"]["items"]["enum"], ["t01"])


if __name__ == "__main__":
    unittest.main()
