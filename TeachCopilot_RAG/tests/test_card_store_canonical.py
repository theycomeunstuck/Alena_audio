"""F-071, user decision 2026-11-22 (option A): PostgreSQL (teachcopilot.learners)
is the source of truth for learner cards; learner-data JSON files are the seed
and an export target.

- scripts/update_learner_card.py --apply writes through the store (revision
  checked, audited) instead of overwriting JSON; --json-seed keeps the old
  file mode for maintaining the seed explicitly.
- scripts/export_learner_cards.py writes validated DB cards as learner-data JSON
  so the CLI reports (zpd, progress_report, build_context) can read them.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for path in (str(ROOT), str(ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import export_learner_cards  # noqa: E402
import update_learner_card  # noqa: E402

LEARNER = "artemov-artem"


def _card():
    return json.loads((REPO / "learner-data" / "learners" / f"{LEARNER}.json").read_text(encoding="utf-8-sig"))


def _catalog():
    sys.path.insert(0, str(REPO / "learner-data" / "tools"))
    import learner_common
    return learner_common.load_catalog(REPO / "learner-data" / "catalog")


class FakeStore:
    def __init__(self, cards):
        self.cards = {card["learner_id"]: card for card in cards}

    def list_cards(self):
        return [{"learner_id": learner_id} for learner_id in sorted(self.cards)]

    def get_card(self, learner_id):
        return copy.deepcopy(self.cards.get(learner_id))

    def catalog(self):
        return _catalog()


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    def fake_update(notes, learner_ids, **kwargs):
        calls.update(kwargs, learner_ids=learner_ids)
        return []

    monkeypatch.setattr(update_learner_card, "update_from_notes", fake_update)
    monkeypatch.setattr(update_learner_card, "open_card_store", lambda: "DB-STORE")
    return calls


def test_cli_writes_through_the_database_store_by_default(captured):
    assert update_learner_card.main(["--learners", LEARNER, "--notes", "итоги", "--apply"]) == 0
    assert captured["store"] == "DB-STORE"
    assert captured["actor_id"] == update_learner_card.CLI_ACTOR_ID
    assert captured["apply"] is True


def test_cli_json_seed_mode_is_explicit(captured):
    assert update_learner_card.main(["--learners", LEARNER, "--notes", "итоги", "--apply", "--json-seed"]) == 0
    assert captured["store"] is None


def test_export_writes_validated_database_cards(tmp_path):
    out = tmp_path / "learners"
    code = export_learner_cards.export_cards(FakeStore([_card()]), out)
    assert code == 0
    written = json.loads((out / f"{LEARNER}.json").read_text(encoding="utf-8"))
    assert written == _card()


def test_export_refuses_invalid_cards_and_reports_them(tmp_path, capsys):
    broken = _card()
    broken["knowledge"][0]["status"] = None
    out = tmp_path / "learners"
    code = export_learner_cards.export_cards(FakeStore([broken]), out)
    assert code == 1
    assert not (out / f"{LEARNER}.json").exists()
    assert LEARNER in capsys.readouterr().err
