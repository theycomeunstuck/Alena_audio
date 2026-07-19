import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import server


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
