"""Round-2 P1.1: task-level continuation consumes the facade's task catalog.

Only a verified arithmetic attempt completes the current task; a model stage
marker, silence or «хз» keeps it pending. The guided reply resumes the
unfinished step with its original catalog wording after a restart.
"""
from __future__ import annotations

import pytest

from local_app.lesson_agent.task_progress import (normalize_non_answer, parse_attempt,
                                                  resolve_pending_task, task_attempt_outcome,
                                                  task_prompt_line)
from tests.test_unified_app_api import LEARNER, LESSON_TEXT, make_api

pytest.importorskip("pydantic")


class TestAttemptVerification:
    def test_correct_attempt_completes(self):
        assert task_attempt_outcome("24 × 3 = ?", "смотри: 24 × 3 = 72") == "completed"

    def test_wrong_result_stays_pending(self):
        assert task_attempt_outcome("24 × 3 = ?", "24 × 3 = 70") == "pending"

    def test_swapped_factors_still_complete(self):
        assert task_attempt_outcome("24 × 3 = ?", "3 × 24 = 72") == "completed"

    def test_different_exercise_is_uncertain(self):
        assert task_attempt_outcome("24 × 3 = ?", "13 × 2 = 26") == "uncertain"

    def test_non_answer_keeps_task_pending(self):
        for message in ("хз", "?", "...", "", "  ", "не знаю"):
            assert task_attempt_outcome("24 × 3 = ?", message) == "pending"

    def test_non_arithmetic_wording_is_uncertain(self):
        assert task_attempt_outcome("24 × 3 = ?", "я не понял задание") == "uncertain"
        assert task_attempt_outcome("Объясни деление с остатком", "двадцать четыре") == "uncertain"

    def test_parse_attempt_variants(self):
        assert parse_attempt("24*3=72") == (24, 3, 72)
        assert parse_attempt("24 x 3 = 72") == (24, 3, 72)
        assert parse_attempt("нет равенства") is None

    def test_bare_number_answer_completes_the_task(self):
        # A bare correct number is a verifiable attempt at the task's
        # expression; a wrong one stays pending.
        assert task_attempt_outcome("6 × 2 = ?", "12") == "completed"
        assert task_attempt_outcome("6 × 2 = ?", "= 12") == "completed"
        assert task_attempt_outcome("6 × 2 = ?", "11") == "pending"

    def test_cyrillic_h_multiplication_is_recognized(self):
        assert task_attempt_outcome("24 × 3 = ?", "24 х 3 = 72") == "completed"

    def test_bare_number_with_non_arithmetic_task_stays_unverifiable(self):
        assert task_attempt_outcome("Объясни деление с остатком", "12") == "pending"


class TestPendingTaskResolution:
    ITEMS = [
        {"task_id": "t1", "stage": "warmup", "position": 1, "prompt_text": "Сосчитай 2 × 3."},
        {"task_id": "t2", "stage": "practice", "position": 1, "prompt_text": "24 × 3 = ?"},
        {"task_id": "t3", "stage": "practice", "position": 2, "prompt_text": "13 × 2 = ?"},
        {"task_id": "t4", "stage": "review", "position": 1, "prompt_text": "Чему равно 12 × 4?"},
    ]

    def test_persisted_pending_task_wins_when_valid(self):
        assert resolve_pending_task(self.ITEMS, [], "t3")["task_id"] == "t3"

    def test_next_incomplete_by_stage_then_position(self):
        assert resolve_pending_task(self.ITEMS, ["t1", "t2"])["task_id"] == "t3"
        assert resolve_pending_task(self.ITEMS, [])["task_id"] == "t1"

    def test_unknown_state_ids_are_filtered(self):
        assert resolve_pending_task(self.ITEMS, ["ghost"], "ghost")["task_id"] == "t1"

    def test_completed_pending_task_falls_through_to_next(self):
        # The realistic in-order flow: completing t3 implies t1/t2 done.
        assert resolve_pending_task(self.ITEMS, ["t1", "t2", "t3"], "t3")["task_id"] == "t4"

    def test_all_done_returns_none(self):
        assert resolve_pending_task(self.ITEMS, ["t1", "t2", "t3", "t4"]) is None

    def test_prompt_line_keeps_original_wording(self):
        assert task_prompt_line({"position": 2, "prompt_text": "13 × 2 = ?"}) == \
            "Задание 2: 13 × 2 = ?"
        assert task_prompt_line(None) == ""


class TestGuidedChatTaskProgress:
    def _setup(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "t", topic_id="math", minutes=20, lesson_text=LESSON_TEXT,
                                 answers_text="24 × 3 = 72", validation={"ok": True, "problems": []},
                                 request_id="r1", memory_revision=None)
        fake.confirm_lesson(LEARNER, "t", saved["lesson_id"], expected_revision=1, request_id="c")
        session = client.post(f"/learners/{LEARNER}/learning-sessions", headers=student,
                              json={"topic_id": "math"}).json()["learning_session_id"]
        fake.attach_lesson(LEARNER, "t", session, saved["lesson_id"], request_id="a1")
        return client, fake, saved, session, student

    def _chat(self, client, student, session, message, client_message_id="m1"):
        return client.post("/v1/chat/completions",
                           json={"learning_session_id": session, "client_message_id": client_message_id,
                                 "messages": [{"role": "user", "content": message}],
                                 "metadata": {"lesson_mode": "guided"}}, headers=student)

    def test_verified_attempt_completes_the_task_and_moves_to_next(self):
        client, fake, saved, session, student = self._setup()
        response = self._chat(client, student, session, "24 × 3 = 72")
        assert response.status_code == 200, response.text
        updates = [entry for entry in fake.calls if entry[0] == "update_lesson_state"]
        assert updates, "task progress must be persisted"
        state_update = updates[-1][2]
        assert state_update["completed_task_ids"] == [f"{saved['lesson_id']}:p1"]
        assert state_update["pending_task_id"] == f"{saved['lesson_id']}:p2"

    def test_short_non_answer_keeps_the_task_pending(self):
        client, fake, saved, session, student = self._setup()
        response = self._chat(client, student, session, "хз")
        assert response.status_code == 200
        updates = [entry for entry in fake.calls if entry[0] == "update_lesson_state"]
        assert updates and updates[-1][2]["pending_task_id"] == f"{saved['lesson_id']}:p1"
        assert "completed_task_ids" not in updates[-1][2]

    def test_guided_prompt_carries_original_task_wording(self):
        client, fake, saved, session, student = self._setup()
        response = self._chat(client, student, session, "что делать?")
        assert response.status_code == 200
        call = client.app.state.rag_service.guided_calls[-1]
        assert "Задание 1: 24 × 3 = ?" in call["fast_memory_text"]
        assert "недоверенные данные" in call["fast_memory_text"]

    def test_stage_marker_alone_never_completes_a_task(self):
        client, fake, saved, session, student = self._setup()
        response = self._chat(client, student, session, "давай дальше")
        assert response.status_code == 200
        updates = [entry for entry in fake.calls if entry[0] == "update_lesson_state"]
        assert updates and "completed_task_ids" not in updates[-1][2]

    def test_task_progress_without_catalog_stays_stage_level(self):
        client, fake, saved, session, student = self._setup()
        fake.lesson_tasks = None  # the merged facade lacks the catalog method
        response = self._chat(client, student, session, "24 × 3 = 72")
        assert response.status_code == 200
        updates = [entry for entry in fake.calls if entry[0] == "update_lesson_state"]
        assert updates and updates[-1][2] == {"stage": "warmup"}

    def test_all_tasks_completed_guided_turn_does_not_crash(self):
        # Regression B1: with every catalog task completed the pending-task
        # projection is None; the guided turn must answer 200 without a
        # TypeError and without pinning a fabricated task.
        client, fake, saved, session, student = self._setup()
        fake.states[session]["completed_task_ids"] = [
            f"{saved['lesson_id']}:p1", f"{saved['lesson_id']}:p2", f"{saved['lesson_id']}:p3"]
        fake.states[session]["pending_task_id"] = None
        response = self._chat(client, student, session, "что дальше?")
        assert response.status_code == 200, response.text
        call = client.app.state.rag_service.guided_calls[-1]
        assert call["fast_memory_text"] is None or "Текущее задание" not in call["fast_memory_text"]
