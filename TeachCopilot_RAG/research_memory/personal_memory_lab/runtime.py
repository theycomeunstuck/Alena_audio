"""Loopback-only Ollama transport with measured counts and explicit context limits."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class AnswerValidationError(ValueError):
    """Keep a model's invalid or truncated answer available for the audit."""

    def __init__(self, message, *, raw_text=None, answer=None):
        super().__init__(message)
        self.raw_text = raw_text
        self.answer = answer


class ContextWindowOverflow(ValueError):
    """The local server rejected an input that does not fit without truncation."""


class UnsafeMemoryProbe(ValueError):
    """A token-count probe is too large or too close to the context boundary."""


def _context_overflow(error):
    if error.code != 400:
        return False
    try:
        value = json.loads(error.read(65536).decode("utf-8"))
        # Ollama may wrap the runner's JSON error inside an error string.
        for _ in range(4):
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    text = value.strip().casefold()
                    # Ollama <=0.3x and >=0.40 word the overflow rejection differently.
                    return (text == "the input length exceeds the context length"
                            or "exceeds the available context size" in text)
            elif isinstance(value, dict) and "error" in value:
                value = value["error"]
            else:
                break
        return isinstance(value, dict) and value.get("type") == "exceed_context_size_error"
    except (ValueError, UnicodeDecodeError):
        return False


class LocalModel:
    def __init__(self, base_url="http://127.0.0.1:11435", model="gemma4:12b", *, context=16384,
                 call_stream=None):
        url = urlsplit(base_url)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama must use a loopback HTTP endpoint")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid local endpoint")
        if context not in (16384, 32768):
            raise ValueError("Only the measured 16k and 32k contexts are allowed")
        self.base_url, self.model, self.context = base_url.rstrip("/"), model, context
        self.calls: list[dict] = []
        self.call_stream = call_stream
        self.call_scope = {}
        self._lock = threading.Lock()
        self.opener = build_opener(ProxyHandler({}))

    def _post(self, route, payload, purpose, *, _skip_preflight=False):
        started = time.perf_counter()
        event = {**self.call_scope, "purpose": purpose, "route": route}
        try:
            payload = dict(payload)
            if route in {"/api/chat", "/api/generate"}:
                # The Ollama default can silently truncate an oversized prompt
                # and report the truncated token count as if it were complete.
                payload.update(truncate=False, shift=False)
            elif route == "/api/embed":
                payload["truncate"] = False
            if route == "/api/chat" and not _skip_preflight:
                messages = payload.get("messages", [])
                message_bytes = sum(len(str(m.get("content", "")).encode("utf-8")) for m in messages)
                output_reserve = int(payload.get("options", {}).get("num_predict", 1024))
                if message_bytes + output_reserve + 512 >= self.context:
                    # UTF-8 is a deliberately conservative first check for
                    # Cyrillic. When it rejects a potentially valid prompt,
                    # measure the text with the same model and reserve 512
                    # tokens for the chat template and boundary differences.
                    raw = "\n".join(str(m.get("role", "")) + "\n" + str(m.get("content", ""))
                                    for m in messages)
                    measured = self.count(raw)
                    event["preflight_text_tokens"] = measured
                    if measured + output_reserve + 512 >= self.context:
                        raise ValueError("Input may exceed the working context; refusing possible truncation")
            req = Request(self.base_url + route, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          {"Content-Type": "application/json"}, method="POST")
            with self.opener.open(req, timeout=300) as res:
                result = json.load(res)
            event.update(status="ok", input_tokens=result.get("prompt_eval_count"),
                         output_tokens=result.get("eval_count"), done_reason=result.get("done_reason"))
            return result
        except HTTPError as exc:
            if _context_overflow(exc):
                event.update(status="expected_rejection" if _skip_preflight else "error",
                             error_type="ContextWindowOverflow")
                raise ContextWindowOverflow("Input exceeds the local model context without truncation") from None
            event.update(status="error", error_type=type(exc).__name__)
            raise
        except Exception as exc:
            event.update(status="error", error_type=type(exc).__name__)
            raise
        finally:
            event["seconds"] = round(time.perf_counter() - started, 4)
            with self._lock:
                self.calls.append(event)
                if self.call_stream is not None:
                    self.call_stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                    self.call_stream.flush()

    def verify_no_truncation(self):
        """Fail closed if a model endpoint ignores the explicit overflow guard."""
        probe = " ".join(hashlib.sha256(str(i).encode()).hexdigest() for i in range(2500))
        options = {"num_ctx": self.context, "num_predict": 1, "temperature": 0, "seed": 42}
        requests = (
            ("/api/generate", {"model": self.model, "prompt": probe, "raw": True,
                               "stream": False, "think": False, "options": options}),
            ("/api/chat", {"model": self.model, "messages": [{"role": "user", "content": probe}],
                           "stream": False, "think": False, "options": options}),
            ("/api/embed", {"model": "bge-m3", "input": probe}),
        )
        for route, payload in requests:
            try:
                self._post(route, payload, "strict_overflow_probe", _skip_preflight=True)
            except ContextWindowOverflow:
                continue
            raise RuntimeError("Ollama did not enforce the no-truncation context guard")
        return {"generate": "rejected", "chat": "rejected", "embed": "rejected"}

    def embed(self, text):
        return self.embed_many([text])[0]

    def embed_many(self, texts):
        if not texts:
            return []
        result = self._post("/api/embed", {"model": "bge-m3", "input": texts, "truncate": False}, "embedding")
        vectors = result.get("embeddings") or []
        if len(vectors) != len(texts) or any(len(vector) != 1024 for vector in vectors):
            raise ValueError("Invalid BGE-M3 embedding")
        return vectors

    def count(self, text):
        """Measure model tokens for the exact memory text, without a chat template.

        num_predict=1 forces Ollama to evaluate the prompt; its output is ignored.
        Requests near the model context limit are refused because Ollama may truncate.
        """
        if not text:
            return 0
        # Refuse very large requests before they reach Ollama. Near the actual
        # window boundary an observed saturated count is rejected below.
        if len(text.encode("utf-8")) > self.context * 20:
            raise UnsafeMemoryProbe("Memory too large for an unambiguous token-count probe")
        result = self._post("/api/generate", {"model": self.model, "prompt": text,
            "raw": True, "stream": False, "think": False,
            "options": {"num_ctx": self.context, "num_predict": 1, "temperature": 0, "seed": 42}},
            "memory_token_count")
        count = result.get("prompt_eval_count")
        if type(count) is not int or count <= 0:
            raise RuntimeError("Token count was unavailable or invalid")
        if count >= self.context - 512:
            raise UnsafeMemoryProbe("Token count is too close to the context boundary")
        return count

    def answer(self, question, memory, source_ids, profile=None):
        allowed_sources = sorted(source_ids)
        citation_schema = {"type": "array", "items": {"type": "string"}, "maxItems": 25}
        if allowed_sources:
            citation_schema["items"]["enum"] = allowed_sources
        else:
            citation_schema["maxItems"] = 0
        schema = {"type": "object", "properties": {
            "answer": {"type": "string"},
            "source_ids": citation_schema,
            "uncertain": {"type": "boolean"}},
            "required": ["answer", "source_ids", "uncertain"]}
        instruction = ("Отвечай только по предоставленным сведениям. Не выдумывай. "
                       "Если данных недостаточно, прямо скажи об этом. Разделяй старые и новые "
                       "наблюдения, помощь и самостоятельную работу. Одно пожелание не доказывает "
                       "постоянное предпочтение. Неясное «не знаю» не интерпретируй без вопроса. "
                       "Извлечённые кандидаты могут быть ошибочными: проверяй каждое утверждение по "
                       "исходным репликам, а не по кандидату. Исправление после подсказки не означает "
                       "самостоятельное решение. Не утверждай самостоятельность без основания. Реплики "
                       "ассистента служат только контекстом и не являются свидетельствами действий ребёнка. "
                       "Верни JSON и укажи только ID приведённых исходных реплик.")
        result = self._post("/api/chat", {"model": self.model, "stream": False, "think": False,
            "format": schema, "options": {"num_ctx": self.context, "num_predict": 1024,
            "temperature": 0, "seed": 42}, "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"question": question, "memory": memory,
                 "profile": profile or {}}, ensure_ascii=False)}]},
            "answer")
        raw = result.get("message", {}).get("content")
        if result.get("done") is not True or result.get("done_reason") == "length":
            raise AnswerValidationError("Answer did not finish", raw_text=raw)
        try:
            answer = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raise AnswerValidationError("Answer is not valid JSON", raw_text=raw) from None
        if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
            raise AnswerValidationError("Invalid answer", raw_text=raw, answer=answer)
        ids = answer.get("source_ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) and x in source_ids for x in ids):
            raise AnswerValidationError("Answer cites an unavailable source", raw_text=raw, answer=answer)
        if not isinstance(answer.get("uncertain"), bool):
            raise AnswerValidationError("Invalid uncertainty field", raw_text=raw, answer=answer)
        return answer


class LegacyClient:
    """Expose existing memory writers through the same guarded local transport."""
    def __init__(self, runtime, max_output_tokens=2048):
        if type(max_output_tokens) is not int or max_output_tokens not in {2048, 4096, 8192}:
            raise ValueError("Unsupported legacy output limit")
        self.runtime = runtime
        self.timeout = 300
        self.max_output_tokens = max_output_tokens

    def complete(self, messages, purpose, response_schema=None):
        from research_memory.clients import Completion
        started = time.perf_counter()
        payload = {"model": self.runtime.model, "stream": False, "think": False,
                   "messages": messages, "options": {"num_ctx": self.runtime.context,
                   "num_predict": self.max_output_tokens, "temperature": 0, "seed": 42}}
        if response_schema is not None:
            payload["format"] = response_schema
        result = self.runtime._post("/api/chat", payload, "legacy_" + purpose)
        if result.get("done") is not True or result.get("done_reason") == "length":
            raise ValueError("Legacy memory write did not finish")
        value = result.get("message", {}).get("content")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Legacy memory write returned no text")
        return Completion(text=value, usage={"input_tokens": result.get("prompt_eval_count"),
                    "output_tokens": result.get("eval_count")},
                    elapsed_seconds=time.perf_counter()-started,
                    metadata={"backend": "ollama", "model": self.runtime.model, "purpose": purpose,
                              "network_request_performed": True})
