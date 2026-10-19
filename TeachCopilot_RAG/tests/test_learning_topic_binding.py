"""A selected curriculum topic survives ambiguous text and untrusted metadata."""
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest

from local_app.api import create_app
from local_app.learning_topics import bind_session_subject, canonical_topic_subject
from tests.test_local_api import auth, token
from tests.test_social_chat import Settings
from tests.test_verified_learning_api import LEARNER, LearningApiStore


FRACTIONS = "math.g4.fractions.numerator_denominator"
PYTHON = "programming.python.output"
SCRATCH = "programming.scratch.conditions"


class RecordingRag:
    def __init__(self):
        self.calls = []

    def answer(self, messages, **kwargs):
        self.calls.append((deepcopy(messages), deepcopy(kwargs)))
        return {"choices": [{"message": {"role": "assistant", "content": "Покажи один шаг."}}]}


@pytest.fixture
def topic_api():
    store, rag = LearningApiStore(), RecordingRag()
    store.add_account("topic-student", "topic.student@example.test", "student", LEARNER)
    with TestClient(create_app(settings=Settings(), store=store, rag_service=rag)) as client:
        yield client, store, rag, auth(token(client, "topic.student@example.test"))


def _session(client, headers, topic):
    result = client.post(f"/learners/{LEARNER}/learning-sessions", headers=headers, json={"topic_id": topic})
    assert result.status_code == 201, result.text
    return result.json()["learning_session_id"]


def _payload(session_id, subject=None):
    return {"learning_session_id": session_id, "client_message_id": "short-reply",
            "messages": [{"role": "user", "content": "хз, можно про кометы"}],
            "metadata": {} if subject is None else {"subject_id": subject}}


@pytest.mark.parametrize("topic,requested,expected", [
    (FRACTIONS, None, "math"), (PYTHON, None, "python"),
    (PYTHON, "programming", "python"), (SCRATCH, "programming", "scratch"),
])
def test_canonical_session_supplies_subject_without_inferring_it_from_child_text(topic_api, topic, requested, expected):
    client, store, rag, headers = topic_api
    response = client.post("/v1/chat/completions", headers=headers,
                           json=_payload(_session(client, headers, topic), requested))
    assert response.status_code == 200, response.text
    assert rag.calls[-1][1]["subject_id"] == expected
    assert rag.calls[-1][1]["topic_id"] == topic
    assert rag.calls[-1][0] == [{"role": "user", "content": "хз, можно про кометы"}]
    assert next(iter(store.learning_sessions.values()))["topic_id"] == topic


@pytest.mark.parametrize("topic,requested", [(FRACTIONS, "python"), (PYTHON, "math"),
    (SCRATCH, "python"), (FRACTIONS, "made-up"), (FRACTIONS, ["math"])])
def test_conflicting_metadata_fails_before_source_or_job_reservation(topic_api, topic, requested):
    client, store, rag, headers = topic_api
    session_id = _session(client, headers, topic)
    response = client.post("/v1/chat/completions", headers=headers, json=_payload(session_id, requested))
    assert response.status_code == 422, response.text
    assert not store.sources and not store.memory_jobs and not rag.calls
    assert not store.learning_messages
    # A rejected request does not consume the idempotency key.
    fixed = client.post("/v1/chat/completions", headers=headers, json=_payload(session_id))
    assert fixed.status_code == 200, fixed.text


@pytest.mark.parametrize("topic", ["math", "python", "math.legacy-missing", "custom-old-topic"])
def test_generic_and_unknown_legacy_sessions_keep_their_original_ids(topic_api, topic):
    client, store, rag, headers = topic_api
    response = client.post("/v1/chat/completions", headers=headers,
                           json=_payload(_session(client, headers, topic), "math"))
    assert response.status_code == 200, response.text
    assert rag.calls[-1][1]["topic_id"] == topic
    assert rag.calls[-1][1]["subject_id"] == "math"


def test_goal_selector_projection_exposes_only_public_binding_fields(topic_api):
    client, store, rag, headers = topic_api
    store.cards[LEARNER]["rag_context"]["current_goal"]["teacher_note"] = "private source note"
    dashboard = client.get(f"/learners/{LEARNER}/dashboard", headers=headers)
    assert dashboard.status_code == 200
    goal = dashboard.json()["profile"]["goal"]
    assert goal["topic_id"] == FRACTIONS and goal["subject_id"] == "math"
    assert "private source note" not in dashboard.text
    assert not {"mastery", "confidence", "teacher_note"}.intersection(goal)
    # Merely sharing a topic prefix is insufficient to bind an unknown legacy ID.
    assert canonical_topic_subject("math.forged", store.catalog()) is None
    assert bind_session_subject("math.forged", "python", store.catalog()) is None
