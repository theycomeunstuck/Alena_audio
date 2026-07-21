import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import json_rag  # noqa: E402


def test_build_and_search_json_index_without_postgres(monkeypatch, tmp_path):
    def fake_embed(text):
        return [1.0, 0.0] if "дел" in text.lower() else [0.0, 1.0]

    monkeypatch.setattr(json_rag, "embed_text", fake_embed)
    source = tmp_path / "tasks.json"
    source.write_text(
        '{"tasks":[{"task_id":"division","topic_id":"math.g4.numbers.division_by_1_2_digit","grade":"4","content":"Деление 408 на 4","answer":"102"}]}',
        encoding="utf-8",
    )
    index = tmp_path / "index.json"

    assert json_rag.build_index(source, index) == 1
    results = json_rag.search_index(index, "Как сделать деление?", topic_id="math.g4.numbers.division_by_1_2_digit", grade="4")

    assert results[0]["metadata"]["task_id"] == "division"
    assert results[0]["score"] == 1.0
