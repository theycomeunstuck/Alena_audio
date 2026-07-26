"""Lesson notes → validated JSON card updates.

The model here is a fake: what is under test is that the *code* turns reported
help levels into numbers, refuses whatever the model made up, and never writes a
card that would fail learner-data validation.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import card_updater  # noqa: E402
from pipeline.card_updater import (  # noqa: E402
    CONFIRM_ALL,
    CardUpdateError,
    apply_proposal,
    build_roster,
    propose_updates,
    update_from_notes,
    write_card,
)
from pipeline.learner_context import load_validated_card  # noqa: E402

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
        "effective_scaffolding": [{"topic_id": TOPIC, "strategy": "сравнение с лучом фонарика"}],
        "journal_note": "Луч начертил после наводящего вопроса.",
        "progress_note": "Чертит луч после одного вопроса.",
    }
    block.update(overrides)
    return {"learners": [block]}


def _fake_llm(payload):
    def complete(_prompt, _system):
        return json.dumps(payload, ensure_ascii=False)

    return complete


def _proposal(payload):
    return propose_updates("итоги урока", [LEARNER], complete=_fake_llm(payload))[LEARNER]


# --- roster -------------------------------------------------------------------

def test_roster_lists_topics_zones_and_known_errors():
    text, cards, catalog = build_roster([LEARNER])

    assert LEARNER in cards and catalog
    assert TOPIC in text
    assert "в ЗБР" in text
    assert "рисует у луча два конца, как у отрезка" in text


def test_roster_covers_the_whole_group_in_one_prompt():
    text, cards, _catalog = build_roster([LEARNER, "belka-06"])

    assert set(cards) == {LEARNER, "belka-06"}
    assert "Тёма" in text and "Белка" in text


# --- parsing the model reply ---------------------------------------------------

def test_invented_topic_is_dropped_with_a_warning():
    proposal = _proposal(_reply(observations=[
        {"topic_id": "math.g3.geometry.wormholes", "help_level": "visual", "solved": True}
    ]))

    assert proposal.observations == ()
    assert any("не из каталога" in warning for warning in proposal.warnings)


def test_unknown_help_level_is_dropped_with_a_warning():
    proposal = _proposal(_reply(observations=[
        {"topic_id": TOPIC, "help_level": "telepathy", "solved": True}
    ]))

    assert proposal.observations == ()
    assert any("уровень помощи" in warning for warning in proposal.warnings)


def test_unknown_learner_in_the_reply_is_ignored():
    payload = _reply()
    payload["learners"].append({"learner_id": "no-such-child", "journal_note": "выдумка"})
    proposals = propose_updates("итоги", [LEARNER], complete=_fake_llm(payload))

    assert set(proposals) == {LEARNER}


def test_topic_from_another_grade_is_dropped():
    """Тема не своего класса — почти всегда выдумка модели."""
    proposal = _proposal(_reply(observations=[
        {"topic_id": "math.g4.geometry.angles", "help_level": "visual", "solved": True}
    ]))

    assert proposal.observations == ()
    assert any("не для 3 класса" in warning for warning in proposal.warnings)


def test_large_group_is_split_into_several_requests():
    """Один ростер на 17 человек не влезает в контекст и путает детей местами."""
    calls: list[str] = []

    def counting_llm(prompt, _system):
        calls.append(prompt)
        return json.dumps(_reply(), ensure_ascii=False)

    learners = ["zvezdin-artem", "belka-06", "ezh-02", "filin-07", "kit-04"]
    propose_updates("итоги", learners, complete=counting_llm, batch_size=2)

    assert len(calls) == 3, "5 учеников при batch_size=2 — это три запроса"
    assert "zvezdin-artem" in calls[0] and "ezh-02" not in calls[0]


def test_batching_keeps_every_learner():
    seen = []

    def per_batch_llm(prompt, _system):
        # Модель отвечает про того ученика, который есть в её ростере.
        learner = "zvezdin-artem" if "zvezdin-artem" in prompt else "belka-06"
        seen.append(learner)
        return json.dumps({"learners": [{"learner_id": learner, "journal_note": "Занимались."}]},
                          ensure_ascii=False)

    proposals = propose_updates("итоги", [LEARNER, "belka-06"], complete=per_batch_llm, batch_size=1)

    assert set(proposals) == {LEARNER, "belka-06"}
    assert seen == ["zvezdin-artem", "belka-06"]


def test_empty_notes_are_refused():
    with pytest.raises(CardUpdateError):
        propose_updates("   ", [LEARNER], complete=_fake_llm(_reply()))


# --- applying ------------------------------------------------------------------

def test_help_level_moves_mastery_and_leaves_the_source_card_untouched():
    card, _catalog = load_validated_card(LEARNER)
    before = json.dumps(card, ensure_ascii=False)

    result = apply_proposal(card, _proposal(_reply()), mode="auto_ema", today=TODAY)

    assert json.dumps(card, ensure_ascii=False) == before, "исходная карточка не должна меняться"
    competency = next(
        item for item in result.card["learner_model"]["competencies"] if item["topic_id"] == TOPIC
    )
    assert competency["mastery"] > 0.62, "подсказка-вопрос — это успех, mastery должна вырасти"
    assert competency["last_assessed"] == TODAY
    assert competency["history"][-1]["help_level"] == "hint_question"


def test_model_cannot_write_mastery_itself():
    """Числа считает код: присланное моделью значение просто не читается."""
    payload = _reply(observations=[
        {"topic_id": TOPIC, "help_level": "joint", "solved": True, "mastery": 0.99}
    ])
    result = apply_proposal(load_validated_card(LEARNER)[0], _proposal(payload), today=TODAY)

    competency = next(
        item for item in result.card["learner_model"]["competencies"] if item["topic_id"] == TOPIC
    )
    assert competency["mastery"] < 0.7


def test_repeated_error_increments_the_counter_instead_of_duplicating():
    card, _catalog = load_validated_card(LEARNER)
    before = [item for item in card["error_patterns"] if item["error_tag"].startswith("рисует у луча")][0]

    result = apply_proposal(card, _proposal(_reply()), today=TODAY)
    after = [item for item in result.card["error_patterns"] if item["error_tag"].startswith("рисует у луча")]

    assert len(after) == 1
    assert after[0]["count"] == before["count"] + 1
    assert after[0]["last_seen"] == TODAY


def test_new_error_pattern_is_added():
    result = apply_proposal(
        load_validated_card(LEARNER)[0],
        _proposal(_reply(new_error_patterns=[{"topic_id": TOPIC, "error_tag": "путает вершину и звено"}])),
        today=TODAY,
    )
    tags = [item["error_tag"] for item in result.card["error_patterns"]]
    assert "путает вершину и звено" in tags


def test_journal_and_progress_are_written():
    result = apply_proposal(load_validated_card(LEARNER)[0], _proposal(_reply()), today=TODAY)

    assert result.card["mvp"]["journal"][-1] == {
        "date": TODAY,
        "note": "Луч начертил после наводящего вопроса.",
    }
    assert result.card["rag_context"]["recent_progress"]["note"] == "Чертит луч после одного вопроса."


def test_zpd_block_and_knowledge_status_are_recomputed():
    """Три самостоятельных решения подряд переводят тему из ЗБР в «делает сам»."""
    card, _catalog = load_validated_card(LEARNER)
    solo = _reply(observations=[{"topic_id": TOPIC, "help_level": "independent", "solved": True}])

    # Разные даты: три занятия, а не одно, применённое трижды.
    for day in ("2026-07-24", "2026-07-25", "2026-07-26"):
        card = apply_proposal(card, _proposal(solo), today=day).card

    assert TOPIC not in card["learner_model"]["zpd"]["current"]
    status = next(item for item in card["knowledge"] if item["topic_id"] == TOPIC)["status"]
    assert status == "confident"


def test_competence_map_is_created_for_a_card_without_learner_model():
    card, _catalog = load_validated_card("belka-06")
    assert "learner_model" not in card

    payload = {"learners": [{
        "learner_id": "belka-06",
        "observations": [{
            "topic_id": "math.g3.word_problems.comparison.multiple",
            "help_level": "independent",
            "solved": True,
        }],
    }]}
    proposal = propose_updates("итоги", ["belka-06"], complete=_fake_llm(payload))["belka-06"]
    result = apply_proposal(card, proposal, today=TODAY)

    assert result.card["learner_model"]["competencies"][0]["mastery"] == pytest.approx(0.7)
    card_updater.validate_or_raise(result.card, "belka-06")


# --- повторное применение тех же итогов -----------------------------------------

def test_repeating_the_same_notes_is_detected_and_skipped():
    card, _catalog = load_validated_card(LEARNER)
    proposal = _proposal(_reply())

    once = apply_proposal(card, proposal, today=TODAY)
    twice = apply_proposal(once.card, proposal, today=TODAY)

    assert once.changes and not twice.changes
    assert any("уже применены" in warning for warning in twice.warnings)

    competency_once = next(c for c in once.card["learner_model"]["competencies"] if c["topic_id"] == TOPIC)
    competency_twice = next(c for c in twice.card["learner_model"]["competencies"] if c["topic_id"] == TOPIC)
    assert competency_once["mastery"] == competency_twice["mastery"]

    errors_once = [e for e in once.card["error_patterns"] if e["error_tag"].startswith("рисует у луча")][0]
    errors_twice = [e for e in twice.card["error_patterns"] if e["error_tag"].startswith("рисует у луча")][0]
    assert errors_once["count"] == errors_twice["count"], "счётчик ошибок не должен накручиваться"


def test_force_applies_a_repeat_anyway():
    card, _catalog = load_validated_card(LEARNER)
    proposal = _proposal(_reply())

    once = apply_proposal(card, proposal, today=TODAY)
    twice = apply_proposal(once.card, proposal, today=TODAY, force=True)

    assert twice.changes


def test_repeat_on_another_day_is_a_normal_update():
    card, _catalog = load_validated_card(LEARNER)
    proposal = _proposal(_reply())

    once = apply_proposal(card, proposal, today="2026-07-25")
    next_day = apply_proposal(once.card, proposal, today="2026-07-26")

    assert next_day.changes


def test_second_lesson_with_different_help_level_is_not_blocked():
    """Guard не должен глотать настоящее второе наблюдение в тот же день."""
    card, _catalog = load_validated_card(LEARNER)
    first = apply_proposal(card, _proposal(_reply()), today=TODAY)

    solo = _reply(
        observations=[{"topic_id": TOPIC, "help_level": "independent", "solved": True}],
        journal_note="Вторую задачу сделал сам.",
    )
    second = apply_proposal(first.card, _proposal(solo), today=TODAY)

    assert second.changes


# --- tutor_confirmed mode -------------------------------------------------------

def test_tutor_confirmed_holds_numbers_back_but_still_records_facts():
    card, _catalog = load_validated_card(LEARNER)
    result = apply_proposal(card, _proposal(_reply()), mode="tutor_confirmed", today=TODAY)

    competency = next(
        item for item in result.card["learner_model"]["competencies"] if item["topic_id"] == TOPIC
    )
    assert competency["mastery"] == 0.62, "без подтверждения оценка не меняется"
    assert result.pending, "предложенное изменение должно быть показано репетитору"
    assert any("журнал" in change for change in result.changes), "факты пишутся в обоих режимах"


def test_tutor_confirmed_applies_the_confirmed_topic():
    card, _catalog = load_validated_card(LEARNER)
    result = apply_proposal(
        card, _proposal(_reply()), mode="tutor_confirmed", today=TODAY, confirmed_topics=[TOPIC]
    )

    competency = next(
        item for item in result.card["learner_model"]["competencies"] if item["topic_id"] == TOPIC
    )
    assert competency["mastery"] > 0.62
    assert not result.pending


def test_confirm_all_sentinel_applies_everything():
    card, _catalog = load_validated_card(LEARNER)
    result = apply_proposal(
        card, _proposal(_reply()), mode="tutor_confirmed", today=TODAY, confirmed_topics=[CONFIRM_ALL]
    )
    assert not result.pending


def test_unknown_mode_is_refused():
    with pytest.raises(CardUpdateError):
        apply_proposal(load_validated_card(LEARNER)[0], _proposal(_reply()), mode="vibes", today=TODAY)


# --- persisting ------------------------------------------------------------------

def test_write_card_refuses_a_card_that_breaks_the_contract(tmp_path, monkeypatch):
    target = tmp_path / f"{LEARNER}.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(card_updater, "card_path", lambda learner_id: target)
    monkeypatch.setattr(card_updater, "backup_dir", lambda: tmp_path / ".backups")

    card, _catalog = load_validated_card(LEARNER)
    card["learner_model"]["competencies"][0]["mastery"] = 5

    with pytest.raises(CardUpdateError):
        write_card(LEARNER, card)
    assert target.read_text(encoding="utf-8") == "{}", "битая карточка не должна записываться"


def test_write_card_persists_a_valid_update(tmp_path, monkeypatch):
    target = tmp_path / f"{LEARNER}.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(card_updater, "card_path", lambda learner_id: target)
    # Без подмены папки копий тест писал бы бэкапы в настоящие данные проекта.
    monkeypatch.setattr(card_updater, "backup_dir", lambda: tmp_path / ".backups")

    card, _catalog = load_validated_card(LEARNER)
    result = apply_proposal(card, _proposal(_reply()), today=TODAY)
    write_card(LEARNER, result.card)

    assert json.loads(target.read_text(encoding="utf-8")) == result.card


def test_update_from_notes_does_not_write_without_apply(monkeypatch):
    def explode(*_args, **_kwargs):
        raise AssertionError("write_card must not be called without apply=True")

    monkeypatch.setattr(card_updater, "write_card", explode)
    results = update_from_notes("итоги", [LEARNER], complete=_fake_llm(_reply()), today=TODAY)

    assert results and results[0].changed()


# --- бэкап и пошаговое согласование ---------------------------------------------

def test_backup_is_written_before_overwriting(tmp_path, monkeypatch):
    """Откатить неудачное обновление должно быть можно и без git."""
    learners = tmp_path / "learners"
    learners.mkdir()
    target = learners / f"{LEARNER}.json"
    original, _catalog = load_validated_card(LEARNER)
    target.write_text(json.dumps(original, ensure_ascii=False, indent=2), encoding="utf-8")

    # Патчим только пути записи: каталог тем должен остаться настоящим,
    # иначе валидация карточки перед записью просто не найдёт данные.
    monkeypatch.setattr(card_updater, "backup_dir", lambda: tmp_path / ".backups")
    monkeypatch.setattr(card_updater, "card_path", lambda learner_id: target)

    result = apply_proposal(original, _proposal(_reply()), today=TODAY)
    write_card(LEARNER, result.card)

    backups = list((tmp_path / ".backups").glob(f"{LEARNER}-*.json"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original
    assert json.loads(target.read_text(encoding="utf-8")) == result.card


def test_backups_are_pruned(tmp_path, monkeypatch):
    learners = tmp_path / "learners"
    learners.mkdir()
    target = learners / f"{LEARNER}.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(card_updater, "backup_dir", lambda: tmp_path / ".backups")
    monkeypatch.setattr(card_updater, "card_path", lambda learner_id: target)

    for minute in range(card_updater.BACKUPS_KEPT + 5):
        card_updater.backup_card(LEARNER, stamp=f"20260726-0000{minute:02d}")

    kept = list((tmp_path / ".backups").glob(f"{LEARNER}-*.json"))
    assert len(kept) == card_updater.BACKUPS_KEPT


def test_tutor_can_reject_individual_items():
    proposal = _proposal(_reply())
    asked = []

    def decide(kind, description):
        asked.append(kind)
        return kind != "observation"  # репетитор не согласен с оценкой попытки

    filtered = card_updater.filter_proposal(proposal, decide)

    assert filtered.observations == ()
    assert filtered.journal_note == proposal.journal_note
    assert filtered.new_error_patterns == proposal.new_error_patterns
    assert "observation" in asked and "journal" in asked


def test_rejecting_everything_produces_an_empty_proposal():
    filtered = card_updater.filter_proposal(_proposal(_reply()), lambda kind, text: False)
    assert filtered.is_empty()


def test_interactive_filtering_reaches_the_applied_card():
    results = update_from_notes(
        "итоги", [LEARNER], complete=_fake_llm(_reply()), today=TODAY,
        decide=lambda kind, text: kind != "observation",
    )
    changes = " ".join(results[0].changes)

    assert "mastery" not in changes, "отклонённое наблюдение не должно менять оценку"
    assert "журнал" in changes


# --- группы и журнал занятий ------------------------------------------------

def test_group_resolves_to_its_members():
    from pipeline.learner_context import list_group_ids, load_group

    assert "detektivy-3" in list_group_ids()
    members = load_group("detektivy-3")
    assert LEARNER in members and len(members) >= 3


def test_unknown_group_names_the_known_ones():
    from pipeline.learner_context import LearnerContextError, load_group

    with pytest.raises(LearnerContextError) as exc:
        load_group("no-such-group")
    assert "detektivy-3" in str(exc.value)


def test_lesson_is_recorded_with_topics_and_duration():
    proposal = _proposal(_reply(lesson_minutes=45))
    result = apply_proposal(load_validated_card(LEARNER)[0], proposal, today=TODAY)

    lesson = result.card["mvp"]["lessons"][-1]
    assert lesson["date"] == TODAY
    assert lesson["topic_ids"] == [TOPIC]
    assert lesson["minutes"] == 45


def test_absurd_duration_is_dropped_not_written():
    proposal = _proposal(_reply(lesson_minutes=5000))
    assert proposal.lesson_minutes is None
    assert any("длительность" in warning for warning in proposal.warnings)


def test_lesson_is_not_duplicated_on_the_same_day():
    proposal = _proposal(_reply())
    once = apply_proposal(load_validated_card(LEARNER)[0], proposal, today=TODAY)
    twice = apply_proposal(once.card, proposal, today=TODAY, force=True)

    assert len(twice.card["mvp"]["lessons"]) == len(once.card["mvp"]["lessons"])
