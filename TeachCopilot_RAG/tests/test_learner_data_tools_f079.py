"""Audit F-079: five small defects in the learner-data CLI tools."""
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "learner-data" / "tools"
LEARNER_DATA = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_context  # noqa: E402
import learner_common  # noqa: E402
import progress_report  # noqa: E402
import validate  # noqa: E402
import zpd  # noqa: E402

CATALOG = LEARNER_DATA / "catalog" / "math_g3_g4.json"
LEARNER = "artemov-artem"


def _card():
    return json.loads((LEARNER_DATA / "learners" / f"{LEARNER}.json").read_text(encoding="utf-8-sig"))


def _learners_dir(tmp_path, card):
    directory = tmp_path / "learners"
    directory.mkdir()
    (directory / f"{LEARNER}.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
    return directory


def test_null_knowledge_status_is_an_error():
    card = _card()
    assert card["knowledge"], "the sample card must have knowledge entries"
    catalog = learner_common.load_catalog(LEARNER_DATA / "catalog")
    baseline = [f for f in validate.validate_card(card, catalog, "card.json") if f.level == "ERROR"]
    broken = copy.deepcopy(card)
    broken["knowledge"][0]["status"] = None
    errors = [f for f in validate.validate_card(broken, catalog, "card.json") if f.level == "ERROR"]
    assert len(errors) == len(baseline) + 1
    assert any("/status" in f.json_path for f in errors)


def test_card_json_syntax_error_gets_the_detailed_message(tmp_path, capsys):
    directory = tmp_path / "learners"
    directory.mkdir()
    (directory / f"{LEARNER}.json").write_text('{"learner_id": "artemov-artem",\n  "broken": }', encoding="utf-8")
    code = build_context.main([LEARNER, "--learners-dir", str(directory), "--catalog", str(CATALOG)])
    assert code == 1
    assert "Ошибка разбора JSON в карточке ученика: строка 2" in capsys.readouterr().err


@pytest.mark.parametrize("tool", [progress_report, zpd])
def test_missing_catalog_is_not_reported_as_a_missing_learner(tool, tmp_path, capsys):
    directory = _learners_dir(tmp_path, _card())
    code = tool.main([LEARNER, "--learners-dir", str(directory), "--catalog", str(tmp_path / "missing.json")])
    err = capsys.readouterr().err
    assert code != 0
    assert "не найден" not in err or "каталог" in err.lower()
    assert "каталог" in err.lower()


def test_progress_report_warns_about_malformed_dates(tmp_path, capsys):
    card = _card()
    mvp = card.setdefault("mvp", {})
    mvp["journal"] = [{"date": "01.07.2026", "note": "урок был"}, {"date": "2026-07-02", "note": "второй урок"}]
    directory = _learners_dir(tmp_path, card)
    progress_report.main([LEARNER, "--learners-dir", str(directory), "--catalog", str(LEARNER_DATA / "catalog")])
    err = capsys.readouterr().err
    assert "01.07.2026" in err


def test_single_file_catalog_rejects_duplicate_topic_ids(tmp_path):
    data = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
    data["topics"].append(dict(data["topics"][0]))
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate curriculum identifiers"):
        learner_common.load_catalog(path)
