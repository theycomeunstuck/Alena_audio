"""Hygiene regressions for card_updater, from the 2026-11-15 audit (F-037,
F-038, F-039): backup pruning must stay inside one learner's own backups, the
write paths must respect the revision guards, and a partially overlapping
re-run must re-apply only the new facts.

The patterns follow the verified V4 repros: temp directories and the real
backup_card / apply_proposal / update_from_notes driven with fakes for the
store and the LLM. Nothing here touches the real learner-data files: every
write is redirected into tmp_path.
"""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.store import RevisionConflict  # noqa: E402
from pipeline import card_updater  # noqa: E402
from pipeline.card_updater import (  # noqa: E402
    BACKUPS_KEPT,
    CardUpdateError,
    CardUpdateProposal,
    ObservedAttempt,
    apply_proposal,
    backup_card,
    update_from_notes,
)
from pipeline.config import LEARNER_DATA_DIR  # noqa: E402
from pipeline.learner_context import load_learner_data_tools, load_validated_card  # noqa: E402

LEARNER = "zvezdin-artem"
TOPIC = "math.g3.geometry.point_line_ray_segment"
TODAY = "2026-07-26"


def _reply(**overrides):
    block = {
        "learner_id": LEARNER,
        "observations": [
            {"topic_id": TOPIC, "help_level": "hint_question", "solved": True, "note": "начертил после вопроса"}
        ],
        "new_error_patterns": [{"topic_id": TOPIC, "error_tag": "рисует у луча два конца, как у отрезка"}],
        "journal_note": "Луч начертил после наводящего вопроса.",
    }
    block.update(overrides)
    return {"learners": [block]}


def _fake_llm(payload):
    def complete(_prompt, _system):
        return json.dumps(payload, ensure_ascii=False)

    return complete


def _real_catalog():
    return load_learner_data_tools().load_catalog(LEARNER_DATA_DIR / "catalog")


def _redirect_writes_to(tmp_path, monkeypatch):
    """Point backup_card's read and backup locations at a throwaway directory."""
    learners = tmp_path / "learners"
    learners.mkdir()
    (learners / "anna.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(card_updater, "card_path", lambda learner_id: learners / f"{learner_id}.json")
    backups = tmp_path / ".backups"
    monkeypatch.setattr(card_updater, "backup_dir", lambda: backups)
    return backups


# --- F-037: backup pruning must never touch a sibling learner's backups --------

def test_backup_pruning_never_touches_sibling_learner_backups(tmp_path, monkeypatch):
    """Sibling ids (anna vs anna-2, the shape _unique_learner_id generates) keep
    backups in the shared .backups dir; the old anna-*.json glob also matches
    anna-2's files and lexicographic order puts them first, so anna's updates
    destroyed anna-2's oldest backups before anna ever exceeded its own quota."""
    backups = _redirect_writes_to(tmp_path, monkeypatch)
    backups.mkdir()
    sibling_old = [
        "anna-2-20261001-090000.json",
        "anna-2-20261005-090000.json",
        "anna-2-20261010-090000.json",
    ]
    for name in sibling_old:
        (backups / name).write_text("{}", encoding="utf-8")

    for i in range(BACKUPS_KEPT + 3):
        backup_card("anna", stamp=f"20261115-{100000 + i * 1000:06d}")

    remaining = sorted(path.name for path in backups.glob("*.json"))
    assert [name for name in sibling_old if name in remaining] == sibling_old, (
        "backup_card('anna') must not delete another learner's backups"
    )
    kept_own = [name for name in remaining if name.startswith("anna-2026")]
    assert len(kept_own) == BACKUPS_KEPT, "anna's own quota is still enforced"
    assert min(kept_own) == "anna-20261115-103000.json", "anna's own oldest backups are the ones pruned"


def test_backup_pruning_keeps_only_timestamp_shaped_files(tmp_path, monkeypatch):
    """A same-prefix file that is not a backup_card product (no timestamp shape)
    is never pruned, even when it sorts before the real backups."""
    backups = _redirect_writes_to(tmp_path, monkeypatch)
    backups.mkdir()
    (backups / "anna-.json").write_text("{}", encoding="utf-8")

    for i in range(BACKUPS_KEPT + 2):
        backup_card("anna", stamp=f"20261115-{100000 + i * 1000:06d}")

    assert (backups / "anna-.json").is_file(), "only timestamp-shaped backups are pruned"
    kept_own = [path.name for path in backups.glob("anna-2026*.json")]
    assert len(kept_own) == BACKUPS_KEPT


# --- F-038: writes must respect the revision guards ------------------------------

class FakeStore:
    """Duck-typed local_app.store.Store with the real optimistic-concurrency guard."""

    def __init__(self, card, revision=7):
        self._card = copy.deepcopy(card)
        self.revision = revision
        self.save_calls = []

    def get_card(self, learner_id):
        return copy.deepcopy(self._card)

    def catalog(self):
        return _real_catalog()

    def get_revision(self, learner_id):
        return self.revision

    def save_card(self, card, actor_id, expected_revision=None):
        self.save_calls.append({"expected_revision": expected_revision, "actor_id": actor_id})
        if expected_revision is not None and self.revision != expected_revision:
            raise RevisionConflict(f"revision is {self.revision}, expected {expected_revision}")
        self.revision += 1
        self._card = copy.deepcopy(card)


def test_store_save_passes_the_revision_seen_at_load():
    store = FakeStore(load_validated_card(LEARNER)[0])
    results = update_from_notes(
        "итоги урока", [LEARNER], apply=True, store=store,
        complete=_fake_llm(_reply()), today=TODAY,
    )

    assert results and results[0].changed()
    assert len(store.save_calls) == 1
    assert store.save_calls[0]["expected_revision"] == 7, (
        "save_card must carry the revision the card was read at, so the store "
        "can reject a write that would clobber a concurrent edit"
    )


def test_store_save_fails_closed_when_the_revision_moved():
    class ConcurrentWriter(FakeStore):
        """Bumps the revision on the update's own card read.

        Call sequence inside update_from_notes: the roster build reads the card
        first (call 1), then the revision is fetched (must still see 7), then
        the update's own load reads the card again (call 2) — the concurrent
        save lands right after that read, before the write.
        """

        def __init__(self, card):
            super().__init__(card)
            self._get_card_calls = 0

        def get_card(self, learner_id):
            self._get_card_calls += 1
            if self._get_card_calls == 2:
                self.revision = 8  # another teacher saved while these notes were processed
            return super().get_card(learner_id)

    store = ConcurrentWriter(load_validated_card(LEARNER)[0])
    with pytest.raises(RevisionConflict):
        update_from_notes(
            "итоги урока", [LEARNER], apply=True, store=store,
            complete=_fake_llm(_reply()), today=TODAY,
        )
    assert store.save_calls[0]["expected_revision"] == 7


def test_store_preview_path_still_does_not_write():
    store = FakeStore(load_validated_card(LEARNER)[0])
    results = update_from_notes(
        "итоги урока", [LEARNER], apply=False, store=store,
        complete=_fake_llm(_reply()), today=TODAY,
    )

    assert results and results[0].changed()
    assert store.save_calls == []


def _redirect_file_writes(tmp_path, monkeypatch):
    learners = tmp_path / "learners"
    learners.mkdir()
    target = learners / f"{LEARNER}.json"
    original, _catalog = load_validated_card(LEARNER)
    target.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(card_updater, "card_path", lambda learner_id: target)
    monkeypatch.setattr(card_updater, "backup_dir", lambda: tmp_path / ".backups")
    return target, original


def test_file_write_fails_closed_when_the_card_changed_on_disk(tmp_path, monkeypatch):
    """The file path overwrites card_path(learner_id) directly; a human edit
    landing between the load and the write must abort the update, not be
    clobbered by a card computed from a stale snapshot."""
    target, _original = _redirect_file_writes(tmp_path, monkeypatch)

    real_load = card_updater.load_validated_card

    def load_then_concurrent_edit(learner_id, *, store=None):
        card, catalog = real_load(learner_id, store=store)
        edited = json.loads(target.read_text(encoding="utf-8"))
        edited["pseudonym"] = "Тёма (правка репетитора)"
        target.write_text(json.dumps(edited, ensure_ascii=False, indent=2), encoding="utf-8")
        return card, catalog

    monkeypatch.setattr(card_updater, "load_validated_card", load_then_concurrent_edit)

    with pytest.raises(CardUpdateError):
        update_from_notes("итоги урока", [LEARNER], apply=True, complete=_fake_llm(_reply()), today=TODAY)

    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk["pseudonym"] == "Тёма (правка репетитора)", (
        "the human edit must survive; the stale update is not written"
    )


def test_file_write_proceeds_when_disk_is_unchanged(tmp_path, monkeypatch):
    target, _original = _redirect_file_writes(tmp_path, monkeypatch)

    results = update_from_notes("итоги урока", [LEARNER], apply=True, complete=_fake_llm(_reply()), today=TODAY)

    assert results and results[0].changed()
    assert json.loads(target.read_text(encoding="utf-8")) == results[0].card


# --- F-039: per-observation repeat detection --------------------------------------

def _minimal_card():
    return {
        "learner_id": "repro-child",
        "pseudonym": "Ребёно",
        "grade": "3",
        "learner_model": {
            "competencies": [{"topic_id": TOPIC}],
            "zpd": {"current": [], "outside": []},
        },
        "knowledge": [],
        "error_patterns": [],
    }


def _proposal(observations, error_patterns=()):
    return CardUpdateProposal(
        learner_id="repro-child",
        observations=tuple(observations),
        new_error_patterns=tuple(error_patterns),
    )


def _competency(card):
    return next(c for c in card["learner_model"]["competencies"] if c["topic_id"] == TOPIC)


def test_partial_rerun_applies_only_the_new_observation():
    """A re-run that repeats observation A (same date + help level) and adds a
    new observation C must apply only C: the EMA moves once per recorded
    attempt and the error counter is not fed by an already-recorded mistake."""
    zpd = load_learner_data_tools().zpd
    first = _proposal(
        [ObservedAttempt(TOPIC, "hint_question", True, note="A")],
        [(TOPIC, "ray_vs_segment")],
    )
    r1 = apply_proposal(_minimal_card(), first, mode="auto_ema", today=TODAY)
    after_first = _competency(r1.card)["mastery"]
    assert after_first != 0.5, "the first run moved the EMA off the prior"
    assert next(e for e in r1.card["error_patterns"])["count"] == 1

    rerun = _proposal(
        [ObservedAttempt(TOPIC, "hint_question", True, note="A re-worded"),
         ObservedAttempt(TOPIC, "visual", True, note="C")],
        [(TOPIC, "ray_vs_segment")],
    )
    r2 = apply_proposal(r1.card, rerun, mode="auto_ema", today=TODAY)

    assert r2.changed(), "the genuinely new observation is still applied"
    c2 = _competency(r2.card)
    expected = zpd.update_competency(
        _competency(r1.card),
        zpd.Observation(topic_id=TOPIC, help_key="visual", solved=True, date=TODAY),
    )
    assert c2["mastery"] == pytest.approx(expected["mastery"]), (
        "only C moves the EMA on top of the first run's state"
    )
    history = [(entry["date"], entry["help_level"]) for entry in c2["history"]]
    assert history.count((TODAY, "hint_question")) == 1, "A must not be recorded twice"
    assert history.count((TODAY, "visual")) == 1
    assert next(e for e in r2.card["error_patterns"])["count"] == 1, (
        "the re-reported mistake is already recorded today; the counter must not grow"
    )
    assert any("уже записан" in warning for warning in r2.warnings), (
        "the tutor sees why a reported observation did not move the numbers"
    )


def test_identical_rerun_is_still_skipped_entirely():
    """The existing all-or-nothing fast path keeps its semantics."""
    first = _proposal(
        [ObservedAttempt(TOPIC, "hint_question", True, note="A")],
        [(TOPIC, "ray_vs_segment")],
    )
    r1 = apply_proposal(_minimal_card(), first, mode="auto_ema", today=TODAY)
    r2 = apply_proposal(r1.card, first, mode="auto_ema", today=TODAY)

    assert not r2.changes
    assert any("уже применены" in warning for warning in r2.warnings)
    assert _competency(r2.card)["mastery"] == _competency(r1.card)["mastery"]


def test_force_still_reapplies_a_full_repeat():
    first = _proposal(
        [ObservedAttempt(TOPIC, "hint_question", True, note="A")],
        [(TOPIC, "ray_vs_segment")],
    )
    r1 = apply_proposal(_minimal_card(), first, mode="auto_ema", today=TODAY)
    r2 = apply_proposal(r1.card, first, mode="auto_ema", today=TODAY, force=True)

    assert r2.changes, "--force means the tutor wants the entry written again"


def test_error_counter_still_increments_on_a_new_day():
    first = _proposal(
        [ObservedAttempt(TOPIC, "hint_question", True, note="A")],
        [(TOPIC, "ray_vs_segment")],
    )
    r1 = apply_proposal(_minimal_card(), first, mode="auto_ema", today="2026-07-25")
    r2 = apply_proposal(r1.card, first, mode="auto_ema", today="2026-07-26")

    assert r2.changes
    assert next(e for e in r2.card["error_patterns"])["count"] == 2, (
        "a new day is a new fact; the same-day suppression must not swallow it"
    )
    history = [(entry["date"], entry["help_level"]) for entry in _competency(r2.card)["history"]]
    assert history.count(("2026-07-26", "hint_question")) == 1
