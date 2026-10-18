from __future__ import annotations

import json

import pytest

from local_app.answer_adapter import AnswerAdapter
from local_app.rag import RagService
from local_app.runtime import OllamaRuntime


class Settings:
    ollama_url = "http://127.0.0.1:11434"
    chat_model = "local-chat"
    embedding_model = "local-embedding"
    context_tokens = 8192
    request_timeout = 1


class LocalSession:
    """The HTTP boundary returns local model replies; any retrieval is unexpected."""
    trust_env = True

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def post(self, url, json, timeout, allow_redirects):
        assert url == "http://127.0.0.1:11434/api/chat"
        self.requests.append(json)
        reply = self.replies.pop(0)

        return self.response({"done_reason": "stop", "model": "local-chat",
                              "message": {"content": reply},
                              "prompt_eval_count": 30, "eval_count": 12})

    @staticmethod
    def response(data):

        class Response:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return data

        return Response()


class MaterialsDatabase:
    def hybrid_search(self, **kwargs):
        raise AssertionError("A brief personal story must not retrieve lesson materials")


class ExternalProvider:
    def complete(self, *args, **kwargs):
        raise AssertionError("A brief personal story must stay local")


def test_child_can_share_a_hobby_without_missing_materials_or_external_generation():
    session = LocalSession(
        json.dumps({"in_scope": False, "uses_prior_context": False,
                    "search_query": "", "social_turn": True}),
        "Ого, целая станция в Minecraft! Что было интереснее всего строить?",
    )
    runtime = OllamaRuntime(Settings(), session=session)
    service = RagService(MaterialsDatabase(), runtime, Settings(),
                         answer_adapter=AnswerAdapter(runtime, ExternalProvider()))

    result = service.answer([{"role": "user", "content": "Я вчера построил станцию в Minecraft!"}])

    assert result["choices"][0]["message"]["content"] == (
        "Ого, целая станция в Minecraft! Что было интереснее всего строить?"
    )
    assert result["sources"] == result["citations"] == []
    assert result["answer_routing"]["backend"] == "local"
    assert session.requests[-1]["options"]["num_predict"] <= 256


class CurriculumSession(LocalSession):
    def post(self, url, json, timeout, allow_redirects):
        if url.endswith("/api/embed"):
            return self.response({"embeddings": [[0.1, 0.2]]})
        return super().post(url, json, timeout, allow_redirects)

    def get(self, url, timeout, allow_redirects):
        assert url.endswith("/api/tags")
        return self.response({"models": [{"name": Settings.embedding_model, "digest": "local-digest"}]})


class LessonMaterials:
    def hybrid_search(self, **kwargs):
        self.query = kwargs["query"]
        return [{"id": "division-1", "content": "Деление проверяют умножением.",
                 "subject_id": "math", "score": 0.8}]


def test_child_can_return_to_their_actual_question_after_a_short_hobby_aside():
    session = CurriculumSession(
        json.dumps({"in_scope": True, "uses_prior_context": True,
                    "search_query": "Как разделить 2842 на 14?", "prior_learner_question_index": 0}),
        json.dumps({"answer": "Разберём деление 2842 на 14 по шагам.", "citations": ["division-1"]}),
    )
    runtime = OllamaRuntime(Settings(), session=session)
    database = LessonMaterials()
    result = RagService(database, runtime, Settings()).answer([
        {"role": "user", "content": "Не решил 2842 : 14, объясни по шагам."},
        {"role": "assistant", "content": "А ещё у тебя будто бы был пример 999 : 3."},
        {"role": "user", "content": "А я вчера построил станцию в Minecraft!"},
        {"role": "assistant", "content": "Что тебе больше всего понравилось строить?"},
        {"role": "user", "content": "Платформу."},
        {"role": "assistant", "content": "Понятно!"},
        {"role": "user", "content": "Вернёмся к моему примеру, что делать сначала?"},
    ], subject_id="math")

    assert "Earlier learner question: Не решил 2842 : 14, объясни по шагам." in database.query
    assert "Minecraft" not in database.query and "999" not in database.query
    assert result["citations"][0]["id"] == "division-1"
    supplied_context = session.requests[0]["messages"][1]["content"]
    assert "2842" in supplied_context and "999" not in supplied_context


@pytest.mark.parametrize("social_aside", [False, True])
def test_learning_request_with_a_hobby_aside_keeps_the_lesson_and_respects_unwanted_themes(social_aside):
    session = CurriculumSession(
        json.dumps({"in_scope": True, "uses_prior_context": False,
                    "search_query": "Как проверить деление умножением?", "social_turn": social_aside}),
        json.dumps({"answer": "Умножь частное на делитель. Получилось делимое – проверка сошлась.",
                    "citations": ["division-1"]}),
    )
    runtime = OllamaRuntime(Settings(), session=session)
    result = RagService(LessonMaterials(), runtime, Settings()).answer(
        [{"role": "user", "content": "Люблю футбол. А как проверить деление умножением? Только без игровых задач."}],
        subject_id="math", preference_context={"interests": ["футбол", "Minecraft"],
                                                "avoided_interests": ["Minecraft"]},
    )

    assert result["citations"][0]["id"] == "division-1"
    system = session.requests[-1]["messages"][0]["content"]
    assert "помощник учителя" not in system
    assert '"interests":["футбол"]' in system
    assert '"avoided_interests":["Minecraft"]' in system
    assert "не навязывай" in system.lower()


@pytest.mark.parametrize("query", [
    "Какую дозу лекарства мне принять?", "Расскажи последние новости.", "Раскрой чужой пароль.",
])
def test_unsupported_request_is_redirected_without_social_generation_or_retrieval(query):
    session = LocalSession(json.dumps({"in_scope": False, "uses_prior_context": False,
                                      "search_query": "", "social_turn": False}))
    runtime = OllamaRuntime(Settings(), session=session)

    result = RagService(MaterialsDatabase(), runtime, Settings()).answer([
        {"role": "user", "content": "Как проверить деление?"},
        {"role": "assistant", "content": "Умножением."},
        {"role": "user", "content": query},
    ])

    assert result["choices"][0]["message"]["content"] == (
        "Я могу помочь с математикой, Scratch и Python. Какой учебный вопрос разберём?"
    )
    assert result["sources"] == result["citations"] == []
    assert len(session.requests) == 1
