"""Round-2 P0.2: the assignment lifecycle at the application boundary.

A teacher assigns an approved lesson before the student has any session; the
linked student later starts it into their own standard session. The facade
owns persistence and authorization; these tests exercise B's API routes
against the contract fake, including idempotency, revision conflicts,
role boundaries and second-child isolation.
"""
from __future__ import annotations

import pytest

from tests.test_unified_app_api import (FOREIGN_LEARNER, LEARNER, LESSON_TEXT, make_api)

pytest.importorskip("pydantic")


def _make_approved_lesson(client, teacher, *, request_id="gen-r2"):
    """Save and confirm one lesson draft through the fake-backed endpoints."""
    saved = client.post(f"/learners/{LEARNER}/lessons", headers=teacher,
                        json={"topic_id": "math", "minutes": 20, "client_request_id": request_id})
    assert saved.status_code == 202, saved.text
    job_id = saved.json()["job_id"]
    # The API test store has no worker; the fake lesson is produced directly.
    return job_id


class TestAssignmentLifecycle:
    def _prepare(self, *, request_id="gen-r2"):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "teacher-unified", topic_id="math", minutes=20,
                                 lesson_text=LESSON_TEXT, answers_text="24 × 3 = 72",
                                 validation={"ok": True, "problems": []},
                                 request_id=request_id, memory_revision=None)
        fake.confirm_lesson(LEARNER, "teacher-unified", saved["lesson_id"],
                            expected_revision=1, request_id=f"confirm-{request_id}")
        return client, store, fake, teacher, student, saved["lesson_id"]

    @staticmethod
    def _learning_sessions(store):
        # The fake store keeps auth sessions and learning sessions in one
        # dict; learning session ids carry the unified-session prefix.
        return [key for key in store.sessions if key.startswith("unified-session-")]

    def test_teacher_assigns_approved_lesson_before_any_session(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        # The student has no session at assignment time.
        assert not self._learning_sessions(store)
        response = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 1, "request_id": "asg-1"})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["assignment_id"] and body["lesson_id"] == lesson_id
        assert body["lesson_revision"] == 1 and body["status"] == "pending"
        assert body["learning_session_id"] is None
        # No session was created by the teacher as a delivery mechanism.
        assert not self._learning_sessions(store)

    def test_assign_is_idempotent_for_the_same_payload(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        first = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                            headers=teacher,
                            json={"expected_lesson_revision": 1, "request_id": "asg-a"})
        second = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                             headers=teacher,
                             json={"expected_lesson_revision": 1, "request_id": "asg-b"})
        assert second.status_code == 201
        assert second.json()["assignment_id"] == first.json()["assignment_id"]

    def test_assign_with_changed_payload_stays_idempotent_per_lesson(self):
        # A repeated assign of the same lesson never creates a second
        # assignment, even with a new request id.
        client, store, fake, teacher, student, lesson_id = self._prepare()
        client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments", headers=teacher,
                    json={"expected_lesson_revision": 1, "request_id": "asg-a"})
        third = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                            headers=teacher,
                            json={"expected_lesson_revision": 1, "request_id": "asg-c"})
        assert third.json()["assignment_id"] == "assign-1"

    def test_assign_with_wrong_expected_revision_conflicts(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        response = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 99, "request_id": "asg-x"})
        assert response.status_code == 409, response.text

    def test_assign_unapproved_draft_is_rejected(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "teacher-unified", topic_id="math", minutes=20,
                                 lesson_text=LESSON_TEXT, answers_text="",
                                 validation={"ok": True, "problems": []},
                                 request_id="gen-draft", memory_revision=None)
        response = client.post(f"/learners/{LEARNER}/lessons/{saved['lesson_id']}/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 1, "request_id": "asg-d"})
        assert response.status_code == 422, response.text

    def test_student_cannot_assign(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        response = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                               headers=student,
                               json={"expected_lesson_revision": 1, "request_id": "asg-s"})
        assert response.status_code == 403, response.text

    def test_student_lists_and_starts_pending_assignment(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        assigned = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 1, "request_id": "asg-1"}).json()
        listing = client.get(f"/learners/{LEARNER}/lesson-assignments", headers=student)
        assert listing.status_code == 200
        items = listing.json()["items"]
        assert [item["assignment_id"] for item in items] == [assigned["assignment_id"]]
        # The listing must not expose teacher answer material.
        assert "answers" not in json_text(items)

        started = client.post(
            f"/learners/{LEARNER}/lesson-assignments/{assigned['assignment_id']}/start",
            headers=student, json={"request_id": "start-1"})
        assert started.status_code == 200, started.text
        body = started.json()
        session_id = body["learning_session_id"]
        assert session_id and body["status"] == "started" and body["lesson_id"] == lesson_id
        assert body["state"]["lesson_id"] == lesson_id
        # The created session belongs to the student, not the teacher.
        session = store.sessions[session_id]
        assert str(session["actor_id"]) == "student-unified"
        assert not any(str(s.get("actor_id")) == "teacher-unified"
                       for s in store.sessions.values() if "account_id" not in s)

        # A repeat start returns the same session, no new one.
        again = client.post(
            f"/learners/{LEARNER}/lesson-assignments/{assigned['assignment_id']}/start",
            headers=student, json={"request_id": "start-2"})
        assert again.status_code == 200
        assert again.json()["learning_session_id"] == session_id
        assert len(self._learning_sessions(store)) == 1

    def test_teacher_cannot_start_and_second_learner_cannot_claim(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        assigned = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 1, "request_id": "asg-1"}).json()
        teacher_start = client.post(
            f"/learners/{LEARNER}/lesson-assignments/{assigned['assignment_id']}/start",
            headers=teacher, json={"request_id": "start-t"})
        assert teacher_start.status_code == 403, teacher_start.text
        # A different child (another learner's student) cannot even see or
        # claim the assignment: learner access is checked before the facade.
        store.add_account("student-foreign", "student.foreign@example.test", "student",
                          FOREIGN_LEARNER)
        from tests.test_local_api import auth, token
        foreign = auth(token(client, "student.foreign@example.test"))
        foreign_view = client.get("/learners/learner-b/lesson-assignments", headers=foreign)
        assert foreign_view.status_code == 200
        assert foreign_view.json()["items"] == []
        foreign_start = client.post(
            f"/learners/{LEARNER}/lesson-assignments/{assigned['assignment_id']}/start",
            headers=foreign, json={"request_id": "start-f"})
        assert foreign_start.status_code == 403, foreign_start.text

    def test_completed_assignment_does_not_start_a_fresh_session(self):
        client, store, fake, teacher, student, lesson_id = self._prepare()
        assigned = client.post(f"/learners/{LEARNER}/lessons/{lesson_id}/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 1, "request_id": "asg-1"}).json()
        fake.assignments[assigned["assignment_id"]]["status"] = "completed"
        response = client.post(
            f"/learners/{LEARNER}/lesson-assignments/{assigned['assignment_id']}/start",
            headers=student, json={"request_id": "start-late"})
        assert response.status_code == 422, response.text

    def test_unknown_assignment_is_404(self):
        client, store, fake, teacher, student = make_api()
        response = client.post("/learners/learner-a/lesson-assignments/assign-404/start",
                               headers=student, json={"request_id": "start-404"})
        assert response.status_code == 404, response.text


def json_text(items):
    import json
    return json.dumps(items, ensure_ascii=False)


class TestLessonTasksRoute:
    def test_student_tasks_hide_answers(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "teacher-unified", topic_id="math", minutes=20,
                                 lesson_text=LESSON_TEXT, answers_text="24 × 3 = 72",
                                 validation={"ok": True, "problems": []},
                                 request_id="gen-tasks", memory_revision=None)
        response = client.get(f"/learners/{LEARNER}/lessons/{saved['lesson_id']}/tasks",
                              headers=student)
        assert response.status_code == 200
        body = response.json()
        assert body["lesson_id"] == saved["lesson_id"]
        assert [item["task_id"] for item in body["items"]] == [
            f"{saved['lesson_id']}:p1", f"{saved['lesson_id']}:p2", f"{saved['lesson_id']}:p3"]
        assert all("72" not in item["prompt_text"] for item in body["items"])
        assert "answers" not in body

    def test_teacher_tasks_include_answers(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "teacher-unified", topic_id="math", minutes=20,
                                 lesson_text=LESSON_TEXT, answers_text="24 × 3 = 72",
                                 validation={"ok": True, "problems": []},
                                 request_id="gen-tasks-t", memory_revision=None)
        response = client.get(
            f"/learners/{LEARNER}/lessons/{saved['lesson_id']}/tasks?audience=teacher",
            headers=teacher)
        assert response.status_code == 200
        assert response.json()["answers"][f"{saved['lesson_id']}:p1"] == "72"

    def test_student_cannot_request_teacher_audience(self):
        client, store, fake, teacher, student = make_api()
        saved = fake.save_lesson(LEARNER, "teacher-unified", topic_id="math", minutes=20,
                                 lesson_text=LESSON_TEXT, answers_text="24 × 3 = 72",
                                 validation={"ok": True, "problems": []},
                                 request_id="gen-tasks-s", memory_revision=None)
        response = client.get(
            f"/learners/{LEARNER}/lessons/{saved['lesson_id']}/tasks?audience=teacher",
            headers=student)
        assert response.status_code == 403, response.text

    def test_tasks_503_when_facade_lacks_catalog(self):
        client, store, fake, teacher, student = make_api()
        fake.lesson_tasks = None  # instance shadow: the facade lacks the method
        saved = fake.save_lesson(LEARNER, "teacher-unified", topic_id="math", minutes=20,
                                 lesson_text=LESSON_TEXT, answers_text="",
                                 validation={"ok": True, "problems": []},
                                 request_id="gen-tasks-x", memory_revision=None)
        response = client.get(f"/learners/{LEARNER}/lessons/{saved['lesson_id']}/tasks",
                              headers=teacher)
        assert response.status_code == 503
        assert "lesson tasks" in response.json()["detail"]


class TestAssignmentsWithoutFacadeSupport:
    def test_assign_503_when_facade_missing_methods(self):
        client, store, fake, teacher, student = make_api()
        fake.assign_lesson = None  # instance shadow: the facade lacks the method
        response = client.post("/learners/learner-a/lessons/lesson-1/assignments",
                               headers=teacher,
                               json={"expected_lesson_revision": 1, "request_id": "asg-503"})
        assert response.status_code == 503
        assert "lesson assignments" in response.json()["detail"]

    def test_assignment_routes_503_without_facade(self):
        client, store, fake, teacher, student = make_api(facade=False)
        assert client.get("/learners/learner-a/lesson-assignments",
                          headers=student).status_code == 503
        assert client.post("/learners/learner-a/lessons/lesson-1/assignments", headers=teacher,
                           json={"expected_lesson_revision": 1,
                                 "request_id": "asg-503-b"}).status_code == 503
        assert client.post("/learners/learner-a/lesson-assignments/assign-1/start",
                           headers=student, json={"request_id": "s-503"}).status_code == 503
