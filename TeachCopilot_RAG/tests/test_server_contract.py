"""Contract checks for the authenticated storage-backed local API."""

from test_local_api import FakeRag, make_client, token, auth


def test_rag_search_passes_limit_and_returns_metadata():
    item = {"topic_id": "math.division", "content": "A segment is part of a line.", "source_file": "book.pdf", "page_number": 12, "score": 0.91}
    client, _, rag = make_client(rag=FakeRag(search_result=[item]))
    teacher = token(client, "teacher@example.org")
    response = client.post("/rag/search", headers=auth(teacher), json={"query": "what is a segment", "learner_id": "learner-a", "limit": 7})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["results"][0]["source_file"] == "book.pdf"
    assert rag.search_calls[0]["limit"] == 7
    assert rag.search_calls[0]["topic_id"] == "math.division"


def test_openai_contract_keeps_safe_messages_and_omits_client_model_controls(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "pipeline.learner_context.load_learner_context",
        lambda learner_id, store=None: SimpleNamespace(prompt_text="safe learner context"),
    )
    client, _, rag = make_client()
    teacher = token(client, "teacher@example.org")
    response = client.post("/v1/chat/completions", headers=auth(teacher), json={
        "model": "client-supplied-model", "temperature": 0.99, "stream": True,
        "learner_id": "learner-a", "messages": [
            {"role": "system", "content": "untrusted client system"},
            {"role": "user", "content": "help me"},
        ],
    })
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert rag.answer_calls[0]["messages"] == [{"role": "user", "content": "help me"}]
    assert rag.answer_calls[0]["profile"] == "safe learner context"
    assert "client-supplied-model" not in str(rag.answer_calls)
