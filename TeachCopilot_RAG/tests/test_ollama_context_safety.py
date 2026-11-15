"""Do not trust token counts from silently truncated Ollama requests."""

import io
import json
from urllib.error import HTTPError

import pytest

from local_app.config import Settings
from local_app.personal_graph_memory import PersonalGraphProvider
from local_app.runtime import IncompleteCompletionError, OllamaRuntime
from local_app.memory_ported.personal_memory_lab.runtime import ContextWindowOverflow, LocalModel


def test_context_overflow_is_not_silently_truncated():
    class Opener:
        def open(self, request, timeout):
            payload = json.loads(request.data)
            assert payload["truncate"] is False
            assert payload["shift"] is False
            nested = json.dumps({"error": {"type": "exceed_context_size_error"}})
            raise HTTPError(request.full_url, 400, "bad request", {},
                            io.BytesIO(json.dumps({"error": nested}).encode()))

    model = LocalModel(context=32768)
    model.opener = Opener()
    with pytest.raises(ContextWindowOverflow):
        model._post("/api/generate", {"model": model.model, "prompt": "synthetic"}, "memory_token_count")
    assert model.calls[-1]["status"] == "error"
    assert model.calls[-1]["error_type"] == "ContextWindowOverflow"


def test_other_http_errors_are_not_misreported_as_context_overflow():
    class Opener:
        def open(self, request, timeout):
            raise HTTPError(request.full_url, 400, "bad request", {},
                            io.BytesIO(b'{"error":"different failure"}'))

    model = LocalModel(context=32768)
    model.opener = Opener()
    with pytest.raises(HTTPError):
        model._post("/api/generate", {"model": model.model, "prompt": "synthetic"}, "memory_token_count")


def test_embedding_overflow_is_reported_and_not_truncated():
    class Opener:
        def open(self, request, timeout):
            assert json.loads(request.data)["truncate"] is False
            raise HTTPError(request.full_url, 400, "bad request", {},
                            io.BytesIO(b'{"error":"the input length exceeds the context length"}'))

    model = LocalModel(context=32768)
    model.opener = Opener()
    with pytest.raises(ContextWindowOverflow):
        model._post("/api/embed", {"model": "bge-m3", "input": "synthetic"}, "embedding")


def test_capability_check_fails_closed_if_server_accepts_oversized_prompt():
    class IgnoresControl(LocalModel):
        def _post(self, *args, **kwargs):
            return {"done": True, "prompt_eval_count": 16387}

    with pytest.raises(RuntimeError, match="no-truncation"):
        IgnoresControl(context=32768).verify_no_truncation()


def test_capability_check_fails_closed_if_embedding_ignores_guard():
    class IgnoresEmbeddingControl(LocalModel):
        def _post(self, route, *args, **kwargs):
            if route == "/api/embed":
                return {"embeddings": [[0.0]]}
            raise ContextWindowOverflow("synthetic rejection")

    with pytest.raises(RuntimeError, match="no-truncation"):
        IgnoresEmbeddingControl(context=32768).verify_no_truncation()


@pytest.mark.parametrize(("experimental", "truncate_history", "strict"), [
    (True, True, True), (False, False, True), (False, True, False),
])
def test_application_sends_no_truncation_controls_for_memory_work(experimental, truncate_history, strict):
    settings = Settings(database_url="", context_tokens=16384,
                        experimental_personal_graph_enabled=experimental)
    runtime = OllamaRuntime(settings)
    captured = {}

    def post(path, payload):
        captured.update(payload)
        return {"done": True, "message": {"content": "ok"}, "done_reason": "stop"}

    runtime._run_budget_post = post
    runtime.chat([{"role": "user", "content": "synthetic"}], truncate_history=truncate_history)
    if strict:
        assert captured["truncate"] is False and captured["shift"] is False
    else:
        assert "truncate" not in captured and "shift" not in captured


def test_personal_graph_fails_closed_when_server_ignores_context_guard(monkeypatch):
    settings = Settings(database_url="", context_tokens=16384, experimental_personal_graph_enabled=True)
    runtime = OllamaRuntime(settings)
    store = type("Store", (), {"dsn": "host=127.0.0.1 dbname=synthetic"})()

    def reject(*args, **kwargs):
        raise RuntimeError("no-truncation guard unavailable")

    monkeypatch.setattr(LocalModel, "verify_no_truncation", reject)
    with pytest.raises(RuntimeError, match="no-truncation"):
        PersonalGraphProvider(store, runtime)._objects()


@pytest.mark.parametrize("experimental", [True, False])
def test_experimental_graph_embedding_rejects_server_truncation(experimental):
    settings = Settings(database_url="", context_tokens=16384,
                        experimental_personal_graph_enabled=experimental)
    runtime = OllamaRuntime(settings)
    captured = {}

    def post(path, payload):
        captured.update(payload)
        return {"embeddings": [[0.0, 1.0]]}

    runtime._run_budget_post = post
    assert runtime.embed("synthetic") == [[0.0, 1.0]]
    if experimental:
        assert captured["truncate"] is False
    else:
        assert "truncate" not in captured


@pytest.mark.parametrize("done_field", [{"done": False}, {"done": None}, {}])
def test_experimental_graph_rejects_an_incomplete_or_unconfirmed_response(done_field):
    settings = Settings(database_url="", context_tokens=16384, experimental_personal_graph_enabled=True)
    runtime = OllamaRuntime(settings)
    runtime._run_budget_post = lambda path, payload: {
        **done_field, "done_reason": "stop", "message": {"content": '{"facts": []}'}
    }
    with pytest.raises(IncompleteCompletionError):
        runtime.chat([{"role": "user", "content": "synthetic"}], truncate_history=False)
