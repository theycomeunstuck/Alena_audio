"""Check approved memory reaches final chat independently of material retrieval.

The runtime is a stub: these checks do not measure model answer quality.
"""
from __future__ import annotations

import json

import pytest

from local_app.rag import RagService


TOPIC = "math.g4.geometry.perimeter"
QUESTION = "Как найти длину всех сторон прямоугольника вместе?"
REPLY = "Сложить длины четырёх сторон."
SOURCE_ID = "student-answer-73"
MATERIAL = {"id": "lesson-2", "content": "Периметр – сумма длин сторон.",
            "source_file": "geometry.md", "topic_id": TOPIC, "subject_id": "math"}


class Settings:
    chat_model = "stub"
    context_tokens = 32768


class Runtime:
    context_tokens = 32768

    def __init__(self):
        self.messages = None
        self.resolver_memory = None

    def resolve_chat_query(self, latest, history, subject, *, approved_memory=None, topic_id=None):
        self.resolver_memory = approved_memory
        return {"in_scope": True, "uses_prior_context": False,
                "uses_approved_memory": bool(approved_memory), "search_query": "периметр"}

    def chat(self, messages, response_format=None):
        self.messages = messages
        return {"message": {"content": json.dumps({"answer": "Ответ модели", "citations": []})}}


class Service(RagService):
    def __init__(self, runtime, materials):
        super().__init__(object(), runtime, Settings())
        self.materials = materials

    def search(self, *args, **kwargs):
        return self.materials


def memory():
    turns = [
        {"turn_id": "context:before:" + SOURCE_ID, "role": "assistant", "usage": "context_only",
         "anchor_source_id": SOURCE_ID, "truncated": False, "text": QUESTION},
        {"turn_id": SOURCE_ID, "role": "student", "text": REPLY},
    ]
    return {"topic_id": TOPIC, "version_id": "approved-version", "method": "personal_graph",
            "stale": False, "text": json.dumps({"lessons": [{"turns": turns}]}, ensure_ascii=False),
            "sources": [{"source_id": SOURCE_ID, "role": "student", "text": REPLY}],
            "dialogue_context": [{"source_id": SOURCE_ID, "role": "assistant", "usage": "context_only",
                                  "truncated": False, "excerpt_mode": "tail", "text": QUESTION}]}


@pytest.mark.parametrize("has_memory,has_material", [(False, False), (False, True), (True, False), (True, True)])
def test_memory_and_materials_reach_generation_independently(has_memory, has_material):
    runtime = Runtime()
    approved = memory() if has_memory else None
    result = Service(runtime, [MATERIAL] if has_material else []).answer(
        [{"role": "user", "content": "Какой вопрос был перед моим ответом?"}],
        subject_id="math", topic_id=TOPIC, memory_context=approved)

    if not has_memory and not has_material:
        assert runtime.messages is None
        assert result["sources"] == [] and result["citations"] == []
        assert "материалов" in result["choices"][0]["message"]["content"]
        return

    system = runtime.messages[0]["content"]
    assert ("<retrieved_sources_untrusted>" in system) is has_material
    assert ("<approved_memory_untrusted>" in system) is has_memory
    if has_memory:
        body = system.split("<approved_memory_untrusted>\n", 1)[1].split("\n</approved_memory_untrusted>", 1)[0]
        assert body == approved["text"]
        before, after = json.loads(body)["lessons"][0]["turns"]
        assert (before["role"], before["usage"], before["anchor_source_id"], before["text"]) == (
            "assistant", "context_only", after["turn_id"], QUESTION)
        assert (after["role"], after["text"]) == ("student", REPLY)
        assert "можешь процитировать" in system
        assert "не доказывает причинность" in system
        if not has_material:
            assert "Учебные материалы не найдены" in system
            assert "подтверждённую память" in system
            assert "Не утверждай, что опираешься на источник" not in system
            assert result["citations"] == []
    else:
        assert QUESTION not in system and REPLY not in system


def test_different_topic_memory_never_reaches_resolver_or_answer():
    runtime = Runtime()
    approved = memory()
    Service(runtime, [MATERIAL]).answer(
        [{"role": "user", "content": "Объясни новую тему."}], subject_id="math",
        topic_id="math.g4.geometry.area", memory_context=approved)
    assert runtime.resolver_memory is None
    assert "<approved_memory_untrusted>" not in runtime.messages[0]["content"]
    assert QUESTION not in json.dumps(runtime.messages, ensure_ascii=False)
    assert REPLY not in json.dumps(runtime.messages, ensure_ascii=False)
