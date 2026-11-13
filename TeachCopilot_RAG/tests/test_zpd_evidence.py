from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT.parent / "learner-data" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import zpd  # noqa: E402

TODAY = "2026-10-13"


@pytest.fixture(autouse=True)
def fixed_evidence_date(monkeypatch):
    """Keep fixed-date evidence recent regardless of the machine's calendar."""
    class FixedDate(zpd.datetime.date):
        @classmethod
        def today(cls):
            return cls.fromisoformat(TODAY)

    monkeypatch.setattr(zpd.datetime, "date", FixedDate)


def profile(topic_id: str, *, mastery: float = 0.95, independence: float = 0.95) -> dict:
    return {
        "schema_version": 2,
        "learner_id": "fictional-demo",
        "pseudonym": "Лена",
        "age_group": "9-10",
        "grade": 4,
        "language": "ru",
        "learner_model": {
            "competencies": [{
                "topic_id": topic_id,
                "mastery": mastery,
                "independence": independence,
                "confidence": 0.9,
                "last_assessed": TODAY,
                "history": [],
            }],
            "zpd": {"current": [], "outside": []},
        },
        "knowledge": [],
    }


def add_attempt(card: dict, topic: str, pair: str, support: str, solved: bool,
                *, date: str = TODAY, difficulty: str = "medium") -> dict:
    return zpd.record_observation(
        card,
        topic_id=topic,
        support_level=support,
        solved=solved,
        observed_on=date,
        task_id=f"task-{pair}-{support}",
        assessment_pair_id=pair,
        difficulty=difficulty,
    )


def report(card: dict, topic: str) -> dict:
    return next(row for row in zpd.zone_report(card, today=TODAY) if row["topic_id"] == topic)


@pytest.mark.parametrize("topic", [
    "math.g3.geometry.circle",
    "programming.scratch.loops",
    "programming.python.output",
])
def test_full_fictional_profiles_show_confirmed_zone_from_paired_evidence(topic):
    card = profile(topic)
    for pair in ("assessment-1", "assessment-2"):
        card = add_attempt(card, topic, pair, "independent", False)
        card = add_attempt(card, topic, pair, "visual", True)

    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_ZPD
    assert row["evidence_status"] == "confirmed"
    assert row["evidence"]["paired"]["tasks"] == 2
    assert row["evidence"]["paired"]["independent"]["successes"] == 0
    assert row["evidence"]["paired"]["assisted"]["successes"] == 2


def test_high_mastery_without_independence_is_unknown_not_mastered():
    competency = {"topic_id": "math.g3.geometry.circle", "mastery": 0.99}
    verdict = zpd.classify(competency)
    assert verdict.zone == zpd.ZONE_UNKNOWN
    assert "самостоятельность не записана" in verdict.reason
    evidence = zpd.confirmed_zone(competency, TODAY)
    assert evidence["zone"] == zpd.ZONE_UNKNOWN
    assert evidence["evidence_status"] == "insufficient"


def test_high_mastery_with_low_independence_stays_provisional_zpd():
    verdict = zpd.classify({
        "topic_id": "math.g3.quantities.length", "mastery": 0.96, "independence": 0.2,
    })
    assert verdict.zone == zpd.ZONE_ZPD


def test_one_assisted_success_cannot_confirm_mastery_or_zpd():
    topic = "math.g3.geometry.circle"
    card = profile(topic)
    card = add_attempt(card, topic, "assessment-1", "independent", False)
    card = add_attempt(card, topic, "assessment-1", "hint_question", True)
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_UNKNOWN
    assert row["evidence_status"] == "insufficient"
    assert row["evidence"]["paired_observations"] == 1


def test_observations_on_different_tasks_do_not_count_as_pairs():
    topic = "programming.python.output"
    card = profile(topic)
    card = add_attempt(card, topic, "independent-a", "independent", False)
    card = add_attempt(card, topic, "assisted-a", "visual", True)
    card = add_attempt(card, topic, "independent-b", "independent", False)
    card = add_attempt(card, topic, "assisted-b", "visual", True)
    row = report(card, topic)
    assert row["evidence"]["independent"]["attempts"] == 2
    assert row["evidence"]["assisted"]["attempts"] == 2
    assert row["evidence"]["paired_observations"] == 0
    assert row["evidence_status"] == "insufficient"


def test_matching_pair_id_with_different_difficulty_is_not_comparable():
    topic = "programming.scratch.loops"
    card = profile(topic)
    card = add_attempt(card, topic, "pair-1", "independent", False, difficulty="easy")
    card = add_attempt(card, topic, "pair-1", "visual", True, difficulty="hard")
    card = add_attempt(card, topic, "pair-2", "independent", False, difficulty="medium")
    card = add_attempt(card, topic, "pair-2", "visual", True, difficulty="easy")
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_UNKNOWN
    assert row["evidence_status"] == "insufficient"


def test_latest_paired_independent_successes_show_support_fading():
    topic = "math.g3.geometry.circle"
    card = profile(topic)
    for pair in ("early-1", "early-2"):
        card = add_attempt(card, topic, pair, "independent", False, date="2026-10-09")
        card = add_attempt(card, topic, pair, "visual", True, date="2026-10-09")
    assert report(card, topic)["confirmed_zone"] == zpd.ZONE_ZPD

    for pair in ("recent-1", "recent-2"):
        card = add_attempt(card, topic, pair, "independent", True, date="2026-10-12")
        card = add_attempt(card, topic, pair, "visual", True, date="2026-10-12")
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_MASTERED
    assert row["evidence"]["paired"]["decision_window"]["tasks"] == 2


def test_later_same_day_attempt_replaces_prior_rung_outcome_for_zone_and_task_selection():
    topic = "math.g3.geometry.circle"
    card = profile(topic, mastery=0.62, independence=0.48)
    for pair in ("same-day-1", "same-day-2"):
        card = add_attempt(card, topic, pair, "independent", False)
        card = add_attempt(card, topic, pair, "visual", True)
    for pair in ("same-day-1", "same-day-2"):
        # The repeated attempt is appended later but has the same calendar day.
        card = add_attempt(card, topic, pair, "independent", True)

    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_MASTERED
    target = zpd.select_target(card, topic)
    assert target.zone == zpd.ZONE_MASTERED
    assert target.difficulty == "hard"
    assert target.hint_depth == 1


def test_latest_two_pairs_use_history_order_when_all_pair_dates_match():
    topic = "math.g3.geometry.circle"
    card = profile(topic, mastery=0.62, independence=0.48)
    for pair in ("pair-1", "pair-2", "pair-3"):
        card = add_attempt(card, topic, pair, "independent", False)
        card = add_attempt(card, topic, pair, "visual", True)
    # pair-1 was opened first, but its later same-day success belongs in the
    # decision window alongside pair-3, which is the latest assessment.
    card = add_attempt(card, topic, "pair-1", "independent", True)
    card = add_attempt(card, topic, "pair-3", "independent", True)

    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_MASTERED
    assert row["evidence"]["paired"]["decision_window"]["independent"]["successes"] == 2


def test_outside_requires_failures_independently_and_with_joint_adult_support():
    topic = "math.g3.geometry.circle"
    card = profile(topic, mastery=0.1, independence=0.1)
    for pair in ("assessment-1", "assessment-2"):
        card = add_attempt(card, topic, pair, "independent", False)
        card = add_attempt(card, topic, pair, "joint", False)
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_OUTSIDE
    assert row["evidence_status"] == "confirmed"
    assert row["evidence"]["paired"]["decision_window"]["joint_failures"] == 2


def test_visual_support_failures_do_not_confirm_outside_before_joint_support():
    topic = "math.g3.geometry.circle"
    card = profile(topic, mastery=0.1, independence=0.1)
    for pair in ("assessment-1", "assessment-2"):
        card = add_attempt(card, topic, pair, "independent", False)
        card = add_attempt(card, topic, pair, "visual", False)
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_UNKNOWN
    assert row["evidence_status"] == "inconclusive"
    assert row["evidence"]["paired"]["decision_window"]["joint_failures"] == 0
    assert "совместное решение со взрослым" in row["confirmed_reason"]


def test_joint_failure_does_not_override_a_success_with_lower_support():
    topic = "programming.python.output"
    card = profile(topic, mastery=0.1, independence=0.1)
    for pair in ("assessment-1", "assessment-2"):
        card = add_attempt(card, topic, pair, "independent", False)
        card = add_attempt(card, topic, pair, "visual", True)
        card = add_attempt(card, topic, pair, "joint", False)
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_UNKNOWN
    assert row["evidence_status"] == "inconclusive"


def test_low_score_alone_does_not_confirm_outside_and_stale_pairs_are_rejected():
    topic = "math.g3.geometry.circle"
    card = profile(topic, mastery=0.1, independence=0.1)
    assert report(card, topic)["confirmed_zone"] == zpd.ZONE_UNKNOWN
    for pair in ("old-1", "old-2"):
        card = add_attempt(card, topic, pair, "independent", False, date="2026-08-01")
        card = add_attempt(card, topic, pair, "joint", False, date="2026-08-01")
    row = report(card, topic)
    assert row["confirmed_zone"] == zpd.ZONE_UNKNOWN
    assert row["evidence_status"] == "stale"
    assert row["evidence"]["stale"] is True


def test_record_observation_copies_card_and_validates_evidence_fields():
    topic = "programming.python.output"
    original = profile(topic)
    updated = zpd.record_observation(
        original,
        topic_id=topic,
        support_level="independent",
        solved=True,
        observed_on=TODAY,
        task_id="task-1",
        assessment_pair_id="pair-1",
        difficulty="easy",
        notes="Printed the expected value",
    )
    entry = updated["learner_model"]["competencies"][0]["history"][-1]
    assert original["learner_model"]["competencies"][0]["history"] == []
    assert entry["solved"] is True
    assert entry["support_level"] == "independent"
    assert entry["task_id"] == "task-1"
    assert entry["assessment_pair_id"] == "pair-1"
    assert entry["difficulty"] == "easy"
    assert entry["note"] == "Printed the expected value"
    with pytest.raises(zpd.ZpdPolicyError):
        zpd.record_observation(original, topic_id=topic, support_level="not_solved", solved=False)
    with pytest.raises(zpd.ZpdPolicyError, match="future"):
        zpd.record_observation(
            original, topic_id=topic, support_level="independent", solved=True,
            observed_on="2999-01-01",
        )


def test_recorded_observation_remains_valid_under_the_learner_card_contract():
    from pipeline.card_updater import validate_or_raise
    from pipeline.learner_context import load_validated_card

    card, _catalog = load_validated_card("zvezdin-artem")
    topic = "math.g3.geometry.point_line_ray_segment"
    updated = zpd.record_observation(
        card,
        topic_id=topic,
        support_level="visual",
        solved=True,
        observed_on=TODAY,
        task_id="teacher-observed-task-1",
        assessment_pair_id="teacher-assessment-1",
        difficulty="medium",
        notes="Learner completed the construction after a visual cue.",
    )
    validate_or_raise(updated, "zvezdin-artem")


def _task_context_with_observations(
    topic: str,
    *,
    mastery: float,
    independence: float,
    pairs: list[tuple[str, bool, str, bool]],
):
    import copy
    from pipeline.learner_context import load_task_design_context, load_validated_card

    card, catalog = load_validated_card("zvezdin-artem")
    card = copy.deepcopy(card)
    competency = next(
        item for item in card["learner_model"]["competencies"] if item["topic_id"] == topic
    )
    competency.update({
        "mastery": mastery,
        "independence": independence,
        "last_assessed": TODAY,
        "history": [],
    })
    for pair_id, independent_solved, support, assisted_solved in pairs:
        for level, solved in (
            ("independent", independent_solved),
            (support, assisted_solved),
        ):
            competency["history"].append({
                "date": TODAY,
                "mastery": mastery,
                "independence": independence,
                "help_level": level,
                "support_level": level,
                "solved": solved,
                "task_id": f"{pair_id}-{level}",
                "assessment_pair_id": pair_id,
                "difficulty": "medium",
            })
    card["learner_model"]["zpd"] = zpd.derive_zpd(card)

    class MemoryStore:
        def catalog(self):
            return catalog

        def get_card(self, learner_id):
            assert learner_id == "zvezdin-artem"
            return copy.deepcopy(card)

    return load_task_design_context("zvezdin-artem", topic, store=MemoryStore())


def test_task_context_follows_recent_independent_wins_despite_low_ema_scores():
    topic = "math.g3.geometry.point_line_ray_segment"
    context = _task_context_with_observations(
        topic,
        mastery=0.62,
        independence=0.48,
        pairs=[
            ("selection-mastered-1", True, "visual", False),
            ("selection-mastered-2", True, "visual", False),
        ],
    )
    assert context.zone == zpd.ZONE_MASTERED
    assert context.difficulty == "hard"
    assert context.hint_depth == 1
    assert "подтверждённым парным наблюдениям" in context.zone_reason


def test_task_context_follows_recent_visual_support_need_despite_high_ema_scores():
    topic = "math.g3.geometry.point_line_ray_segment"
    context = _task_context_with_observations(
        topic,
        mastery=0.99,
        independence=0.99,
        pairs=[
            ("selection-zpd-1", False, "visual", True),
            ("selection-zpd-2", False, "visual", True),
        ],
    )
    assert context.zone == zpd.ZONE_ZPD
    assert context.difficulty == "medium"
    assert context.hint_depth == zpd.HELP_BY_KEY["visual"].level
    assert "подтверждённым парным наблюдениям" in context.zone_reason


def test_task_context_uses_easy_diagnostic_for_incomplete_explicit_pairs():
    topic = "math.g3.geometry.point_line_ray_segment"
    context = _task_context_with_observations(
        topic,
        mastery=0.99,
        independence=0.99,
        pairs=[("selection-incomplete", False, "visual", True)],
    )
    assert context.zone == zpd.ZONE_UNKNOWN
    assert context.difficulty == "easy"
    assert context.hint_depth == len(zpd.OFFERABLE_HELP)
    assert "короткий диагностический шаг" in context.zone_reason

