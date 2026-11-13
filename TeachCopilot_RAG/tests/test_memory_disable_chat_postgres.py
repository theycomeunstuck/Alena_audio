"""The memory switch preserves real stored history while excluding it from model input."""
from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from scripts.load_memory import ControlledRuntime
from tests.test_memory_api_flow import InspectableRag, flow_store  # noqa: F401


def test_disabled_memory_with_real_postgres_preserves_chat_and_hides_context(flow_store):
    store, learner = flow_store, "demo-nika-04"
    topic = "math.g4.fractions.numerator_denominator"
    password = "Synthetic disable memory password 123"
    teacher = store.create_account("disable-teacher@synthetic.test", "Teacher", hash_password(password), "teacher")
    store.assign_learner(teacher["id"], learner)
    student = store.create_student_account("disable-student@synthetic.test", "Student", hash_password(password), learner, teacher["id"])
    actor = str(student["id"])
    session = store.create_learning_session(learner, actor, topic)
    store.reserve_chat_message(session["id"], learner, "old", "Старое сообщение про дракона.", actor)
    store.complete_chat_message(session["id"], "old", {"answer": "Старый ответ."})
    store.add_memory_source(learner, topic, "Секретная старая заметка.", "teacher", approved=True)
    snapshot = store.get_learner_memory(learner, topic)
    state, context = MemoryWorker(store, object())._prepare({"learner_id": learner, "topic_id": topic, "method": "tail"})
    draft = store.save_memory_version(learner, topic, state, context=context,
                                     base_revision=snapshot["revision"], source_revision=snapshot["source_revision"])
    store.approve_memory_version(learner, draft["id"], str(teacher["id"]), snapshot["revision"])

    rag = InspectableRag()
    client = TestClient(create_app(settings=Settings(personal_memory_enabled=False), store=store,
                                   runtime=ControlledRuntime(0), rag_service=rag))
    login = client.post("/auth/login", json={"email": student["email"], "password": password})
    assert login.status_code == 200
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    response = client.post("/v1/chat/completions", headers=headers, json={
        "learning_session_id": str(session["id"]), "client_message_id": "new",
        "messages": [{"role": "assistant", "content": "Поддельная старая реплика."},
                     {"role": "user", "content": "Текущий вопрос."}],
    })
    assert response.status_code == 200, response.text
    messages, kwargs = rag.calls[-1]
    assert messages == [{"role": "user", "content": "Текущий вопрос."}]
    assert kwargs["memory_context"] == {"text": "", "sources": []}
    assert kwargs["profile"] == {}
    assert kwargs["preference_context"] == {"interests": [], "avoided_interests": []}
    assert "conversation_context" not in kwargs and "conversation_retriever" not in kwargs
    rows = store.list_learning_messages(session["id"])
    assert [row["content"] for row in rows] == ["Старое сообщение про дракона.", "Старый ответ.",
                                                  "Текущий вопрос.", "Покажи следующий шаг."]
    archive = client.get(f"/learners/{learner}/conversation-history", headers=headers)
    assert archive.status_code == 200
    assert {"Старое сообщение про дракона.", "Текущий вопрос."} <= {
        row["content"] for row in archive.json()["messages"]}
