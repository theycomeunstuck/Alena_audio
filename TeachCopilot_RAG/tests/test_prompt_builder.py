"""Regression tests for prompt_builder fixes:
- child interests rendered once as a clean join (not a raw Python list repr)
- adult prompt grounds answers in retrieved RAG material
These need no DB/model: build_system_prompt takes a profile dict directly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline.prompt_builder as pb


def test_child_interests_rendered_once_not_raw_list():
    profile = {
        "name": "Миша",
        "age_group": "9-11",
        "grade": "3-4",
        "interests": ["конструкторы", "машинки"],
        "knowledge": [],
        "error_patterns": [],
    }
    out = pb.build_system_prompt("BASE", profile, rag=[], speaker_type="child")
    # No raw Python list repr leaking into the prompt.
    assert "['конструкторы'" not in out
    assert "[\"конструкторы\"" not in out
    # Rendered exactly once as a comma-joined string.
    assert out.count("конструкторы, машинки") == 1


def test_adult_prompt_grounds_rag_material():
    profile = {"name": "Миша", "knowledge": []}
    rag = [{"topic": "Отрезки", "content": "Отрезок — часть прямой между двумя точками."}]
    grounded = pb._build_adult_prompt(profile, rag)
    assert "relevant_material" in grounded
    assert "Отрезок — часть прямой" in grounded
    # Empty RAG must not add the block.
    assert "relevant_material" not in pb._build_adult_prompt(profile, [])
