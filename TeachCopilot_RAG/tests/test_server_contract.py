import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402


def test_rag_search_passes_limit_and_returns_metadata(monkeypatch):
    calls = {}

    def fake_search_knowledge(query, limit=None, **kwargs):
        calls["query"] = query
        calls["limit"] = limit
        return [
            {
                "topic": "Отрезок",
                "content": "Отрезок — часть прямой.",
                "source_file": "book.pdf",
                "page_number": 12,
                "image_path": "data/images/book_p012.png",
                "score": 0.91,
            }
        ]

    monkeypatch.setattr(server, "search_knowledge", fake_search_knowledge)
    client = TestClient(server.app)

    response = client.post("/rag/search", json={"query": "что такое отрезок", "limit": 7})

    assert response.status_code == 200
    assert calls == {"query": "что такое отрезок", "limit": 7}
    data = response.json()
    assert data["count"] == 1
    assert data["results"][0]["source_file"] == "book.pdf"
    assert data["results"][0]["page_number"] == 12


def test_openai_proxy_forces_non_stream_and_strips_reasoning(monkeypatch):
    captured = {}

    def fake_search_knowledge(query):
        captured["rag_query"] = query
        return [
            {
                "topic": "Отрезок",
                "content": "Отрезок — часть прямой между двумя точками.",
                "source_file": "math.pdf",
                "page_number": 5,
                "score": 0.88,
            }
        ]

    class FakeResponse:
        status_code = 200
        text = '{"ok": true}'

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Ответ",
                            "reasoning_content": "hidden",
                        }
                    }
                ]
            }

    def fake_post(url, json, timeout, stream):
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        captured["stream"] = stream
        return FakeResponse()

    monkeypatch.setattr(server, "search_knowledge", fake_search_knowledge)
    monkeypatch.setattr(server.requests, "post", fake_post)
    monkeypatch.setattr(server, "CHAT_MODEL", "real-upstream-model")

    client = TestClient(server.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "teachcopilot-rag",
            "stream": True,
            "messages": [{"role": "user", "content": "Объясни отрезок"}],
        },
    )

    assert response.status_code == 200
    assert captured["rag_query"] == "Объясни отрезок"
    assert captured["stream"] is False
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["model"] == "real-upstream-model"
    assert captured["payload"]["messages"][0]["role"] == "system"
    assert "Источник: math.pdf, стр. 5" in captured["payload"]["messages"][0]["content"]
    assert "reasoning_content" not in response.json()["choices"][0]["message"]


def test_rag_search_uses_validated_learner_topics_as_soft_hint(monkeypatch):
    captured = {}

    def fake_search_knowledge(query, limit=None, **kwargs):
        captured["query"] = query
        captured["limit"] = limit
        return []

    monkeypatch.setattr(server, "search_knowledge", fake_search_knowledge)
    client = TestClient(server.app)

    response = client.post(
        "/rag/search",
        json={"query": "Как делить 408 на 4?", "learner_id": "volk-08", "limit": 2},
    )

    assert response.status_code == 200
    assert response.json()["learner_id"] == "volk-08"
    assert captured["limit"] == 2
    assert "math.g4.numbers.division_by_1_2_digit" in captured["query"]


def test_rag_search_rejects_unknown_learner_before_retrieval(monkeypatch):
    def retrieval_must_not_run(*args, **kwargs):
        raise AssertionError("retrieval ran for an unknown learner")

    monkeypatch.setattr(server, "search_knowledge", retrieval_must_not_run)
    client = TestClient(server.app)

    response = client.post(
        "/rag/search",
        json={"query": "Привет", "learner_id": "not-a-learner"},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_chat_injects_only_compact_learner_context(monkeypatch):
    captured = {}

    def fake_search_knowledge(query):
        captured["query"] = query
        return []

    class FakeResponse:
        status_code = 200
        text = '{"ok": true}'

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "Ответ"}}]}

    def fake_post(url, json, timeout, stream):
        captured["payload"] = json
        return FakeResponse()

    monkeypatch.setattr(server, "search_knowledge", fake_search_knowledge)
    monkeypatch.setattr(server.requests, "post", fake_post)
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Как делить 408 на 4?"}],
            "learner_id": "volk-08",
        },
    )

    assert response.status_code == 200
    system_prompt = captured["payload"]["messages"][0]["content"]
    assert "<learner_rag_context>" in system_prompt
    assert "math.g4.numbers.division_by_1_2_digit" in system_prompt
    assert "learner_model" not in system_prompt
    assert "points_ledger" not in system_prompt
