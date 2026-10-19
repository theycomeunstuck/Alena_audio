"""Readable labels are display metadata, never replacements for topic IDs."""
from tests.test_local_api import auth, make_client, token


def test_named_learning_session_preserves_its_exact_topic_id(monkeypatch):
    client, store, _ = make_client()
    topic_id = "math.g4.numbers.division_by_1_2_digit"
    monkeypatch.setattr(store, "get_learner_memory", lambda *args: {}, raising=False)
    monkeypatch.setattr(store, "create_learning_session", lambda learner, actor, topic, **kwargs: {
        "learning_session_id": "title-session", "learner_id": learner, "topic_id": topic, "status": "active",
    }, raising=False)
    student = token(client, "student@example.org")
    response = client.post("/learners/learner-a/learning-sessions", headers=auth(student), json={"topic_id": topic_id})
    assert response.status_code == 201
    assert response.json()["topic_id"] == topic_id
    assert response.json()["topic_title"] == "Деление на однозначное и двузначное число"


def test_teacher_memory_labels_do_not_merge_general_and_specific_topics(monkeypatch):
    client, store, _ = make_client()
    topic_id = "math.g4.numbers.division_by_1_2_digit"
    monkeypatch.setattr(store, "get_learner_memory", lambda *args: {
        "learner_id": "learner-a", "sources": [{"topic_id": topic_id, "text": "хз"}],
        "versions": [{"topic_id": "math", "version_id": "general"}, {"topic_id": topic_id, "version_id": "specific"}],
    }, raising=False)
    teacher = token(client, "teacher@example.org")
    response = client.get("/learners/learner-a/memory", headers=auth(teacher))
    assert response.status_code == 200
    payload = response.json()
    assert [(row["topic_id"], row["topic_title"]) for row in payload["versions"]] == [
        ("math", "Математика"), (topic_id, "Деление на однозначное и двузначное число"),
    ]
    assert payload["sources"][0]["text"] == "хз"
    student = token(client, "student@example.org")
    assert client.get("/learners/learner-a/memory", headers=auth(student)).status_code == 403
