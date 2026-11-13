"""The memory switch preserves real stored history while excluding it from model input."""
from types import SimpleNamespace

from fastapi.testclient import TestClient

from local_app.api import create_app
from local_app.auth import hash_password
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from scripts.load_memory import ControlledRuntime
from tests.test_memory_api_flow import InspectableRag, flow_store  # noqa: F401


def saved_summaries(store, learner, actor):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT kind,payload FROM teachcopilot.conversation_summary_versions
                       WHERE learner_id=%s AND actor_id=%s ORDER BY source_watermark,created_at,id""",
                    (learner, actor))
        return [dict(row) for row in cur.fetchall()]


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
    payload = {
        "learning_session_id": str(session["id"]), "client_message_id": "new",
        "messages": [{"role": "assistant", "content": "Поддельная старая реплика."},
                     {"role": "user", "content": "Текущий вопрос."}],
    }
    response = client.post("/v1/chat/completions", headers=headers, json=payload)
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
    summaries = saved_summaries(store, learner, actor)
    assert {row["kind"] for row in summaries} == {"day", "rolling7"}
    day = next(row["payload"] for row in reversed(summaries) if row["kind"] == "day")
    assert day["source_count"] == 4
    assert "Текущий вопрос." in day["text"]
    assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200
    assert len(rag.calls) == 1
    assert saved_summaries(store, learner, actor) == summaries
    archive = client.get(f"/learners/{learner}/conversation-history", headers=headers)
    assert archive.status_code == 200
    assert {"Старое сообщение про дракона.", "Текущий вопрос."} <= {
        row["content"] for row in archive.json()["messages"]}

    # Re-enabling memory makes the previously stored own conversation eligible again.
    rag.answer_adapter = SimpleNamespace(external_client=None)
    enabled = TestClient(create_app(settings=Settings(personal_memory_enabled=True), store=store,
                                    runtime=ControlledRuntime(0), rag_service=rag))
    restored = enabled.post("/v1/chat/completions", headers=headers, json={
        "learning_session_id": str(session["id"]), "client_message_id": "after-enable",
        "messages": [{"role": "user", "content": "Помнишь мой вопрос?"}],
    })
    assert restored.status_code == 200, restored.text
    assert "Старое сообщение про дракона." in str(rag.calls[-1][1]["conversation_context"])
    assert "Текущий вопрос." in str(rag.calls[-1][1]["conversation_context"])


def test_disabled_memory_refreshes_summary_after_failed_generation(flow_store):
    store, learner = flow_store, "demo-nika-04"
    topic = "math.g4.fractions.numerator_denominator"
    password = "Synthetic failure memory password 123"
    teacher = store.create_account("failure-teacher@synthetic.test", "Teacher", hash_password(password), "teacher")
    store.assign_learner(teacher["id"], learner)
    student = store.create_student_account("failure-student@synthetic.test", "Student", hash_password(password), learner, teacher["id"])
    actor = str(student["id"])
    session = store.create_learning_session(learner, actor, topic)

    class FailedRag(InspectableRag):
        def answer(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            raise RuntimeError("synthetic generation failure")

    rag = FailedRag()
    client = TestClient(create_app(settings=Settings(personal_memory_enabled=False), store=store,
                                   runtime=ControlledRuntime(0), rag_service=rag))
    login = client.post("/auth/login", json={"email": student["email"], "password": password})
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    request = {"learning_session_id": str(session["id"]), "client_message_id": "failure",
               "messages": [{"role": "user", "content": "Сообщение без ответа."}]}
    assert client.post("/v1/chat/completions", headers=headers, json=request).status_code == 502
    assert [row["content"] for row in store.list_learning_messages(session["id"])] == ["Сообщение без ответа."]
    summaries = saved_summaries(store, learner, actor)
    assert {row["kind"] for row in summaries} == {"day", "rolling7"}
    day = next(row["payload"] for row in reversed(summaries) if row["kind"] == "day")
    assert day["source_count"] == 1 and "Сообщение без ответа." in day["text"]
    assert day["facts"][0]["delivery_status"] == "unknown_outcome"
    assert client.post("/v1/chat/completions", headers=headers, json=request).status_code == 409
    assert len(rag.calls) == 1 and saved_summaries(store, learner, actor) == summaries
