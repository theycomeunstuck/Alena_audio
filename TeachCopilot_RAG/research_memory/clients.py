"""Local model and explicit demo clients for research memory workflows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import ipaddress
import json
import math
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


@dataclass
class Completion:
    text: str
    usage: dict
    elapsed_seconds: float
    metadata: dict


class CompletionValidationError(ValueError):
    """A parsed model response failed its contract but still consumed tokens."""

    def __init__(self, message: str, completion: Completion, code: str):
        super().__init__(message)
        self.completion = completion
        self.code = code


class ModelRequestError(RuntimeError):
    """A transport or HTTP failure with safe, structured diagnostics only."""

    _CODES = {"model_http_error", "model_timeout", "model_connection_error"}

    def __init__(self, code: str, *, http_status: int | None = None):
        if code not in self._CODES:
            raise ValueError("Unsupported model request error code")
        if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
            raise ValueError("http_status must be a valid HTTP status")
        self.code = code
        self.http_status = http_status
        super().__init__("Ollama request failed")


class InputBudgetError(CompletionValidationError):
    """A writer was refused before a generator request; zero tokens were consumed."""


@lru_cache(maxsize=1)
def _estimate_encoding():
    # This estimates input size only. Gemma's tokenizer is different, so the
    # reserve below and the returned diagnostics must never imply an exact count.
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def input_budget_diagnostic(messages: list[dict[str, str]], response_schema: dict | None,
                            *, context_tokens: int, max_output_tokens: int,
                            purpose: str) -> dict:
    """Count the complete request projection, with an explicit heuristic reserve.

    No message or schema is shortened. Ollama does not expose a tokenizer for
    this local model, and its prompt_eval_count is available only after a call.
    cl100k_base plus the reserve is a preflight heuristic, not a proof that the
    model will not truncate input. The conservative byte fallback is deliberately
    stricter when the optional tokenizer cannot be loaded.
    """
    message_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    schema_json = (json.dumps(response_schema, ensure_ascii=False, separators=(",", ":"))
                   if response_schema is not None else "")
    try:
        encoding = _estimate_encoding()
    except (ImportError, OSError, ValueError):
        # Keep the research package usable offline without a cached encoding.
        estimated = len(message_json.encode("utf-8")) + len(schema_json.encode("utf-8"))
        estimator = "utf8_bytes_fallback"
    else:
        estimated = (len(encoding.encode(message_json, disallowed_special=()))
                     + len(encoding.encode(schema_json, disallowed_special=())))
        estimator = "cl100k_base"
    # JSON includes roles; this separate allowance accounts for model chat
    # framing that the public Ollama API does not expose in the request.
    framing = 128 + 64 * len(messages)
    estimate_with_framing = estimated + framing
    reserve = max(512, math.ceil(estimate_with_framing * 0.25))
    available = context_tokens - max_output_tokens
    guarded = estimate_with_framing + reserve
    return {
        "method": "cl100k_base_with_safety_reserve_v1",
        "estimator": estimator,
        "token_budget_kind": "heuristic_not_model_tokenizer",
        "exact_token_count": False,
        "estimated_payload_tokens": estimated,
        "framing_token_allowance": framing,
        "estimated_input_tokens": estimate_with_framing,
        "safety_reserve_tokens": reserve,
        "guarded_input_tokens": guarded,
        "available_input_tokens": available,
        "context_tokens": context_tokens,
        "reserved_output_tokens": max_output_tokens,
        "message_utf8_bytes": len(message_json.encode("utf-8")),
        "schema_utf8_bytes": len(schema_json.encode("utf-8")),
        "within_guard": guarded <= available,
        "enforced": purpose in {"summary", "extract"},
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def _validate_base_url(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("base_url must be a valid loopback HTTP URL") from exc
    if (parsed.scheme != "http" or not hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError("base_url must be a loopback HTTP URL without credentials, query, or fragment")
    host = hostname.rstrip(".").lower()
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError("base_url must target loopback")
        except ValueError as exc:
            raise ValueError("base_url must target loopback") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("base_url port is invalid")
    return base_url.rstrip("/")


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11435",
                 model: str = "gemma4:12b", timeout: float = 180,
                 context_tokens: int = 8192, max_output_tokens: int = 1024):
        self.base_url = _validate_base_url(base_url)
        self.model = model
        self.timeout = timeout
        self.context_tokens = context_tokens
        self.max_output_tokens = max_output_tokens
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if context_tokens < 2048:
            raise ValueError("context_tokens must be at least 2048")
        if max_output_tokens <= 0 or max_output_tokens >= context_tokens:
            raise ValueError("max_output_tokens must be positive and less than context_tokens")
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def complete(self, messages: list[dict[str, str]], purpose: str,
                 response_schema: dict | None = None) -> Completion:
        budget = input_budget_diagnostic(messages, response_schema,
            context_tokens=self.context_tokens, max_output_tokens=self.max_output_tokens,
            purpose=purpose)
        if budget["enforced"] and not budget["within_guard"]:
            refusal = Completion("", {"input_tokens": 0, "output_tokens": 0}, 0.0,
                {"backend": "ollama", "model": self.model, "purpose": purpose,
                 "request_timeout_seconds": self.timeout, "input_budget": budget,
                 "network_request_performed": False})
            raise InputBudgetError("Writer input exceeds the preflight budget heuristic",
                                   refusal, "writer_input_budget_exceeded")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self.context_tokens,
                "num_predict": self.max_output_tokens,
                "seed": 42,
                "temperature": 0,
            },
        }
        if response_schema is not None:
            payload["format"] = response_schema
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url + "/api/chat", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            # Do not include the server body, which may echo private prompts.
            status = exc.code if type(exc.code) is int and 100 <= exc.code <= 599 else None
            raise ModelRequestError("model_http_error", http_status=status) from None
        except URLError as exc:
            code = "model_timeout" if isinstance(exc.reason, TimeoutError) else "model_connection_error"
            raise ModelRequestError(code) from None
        except TimeoutError:
            raise ModelRequestError("model_timeout") from None
        except ConnectionError:
            raise ModelRequestError("model_connection_error") from None
        elapsed = time.perf_counter() - started
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Ollama response exceeds 8 MiB")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise ValueError("Ollama returned malformed JSON") from None
        if not isinstance(result, dict):
            raise ValueError("Ollama response must be a JSON object")
        message = result.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        completion = Completion(
            text=text if isinstance(text, str) else "",
            usage={"input_tokens": result.get("prompt_eval_count"),
                   "output_tokens": result.get("eval_count")},
            elapsed_seconds=elapsed,
            metadata={"backend": "ollama", "model": self.model, "purpose": purpose,
                      "max_output_tokens": self.max_output_tokens,
                      "prompt_eval_count": result.get("prompt_eval_count"),
                      "eval_count": result.get("eval_count"),
                      "total_duration": result.get("total_duration"),
                      "load_duration": result.get("load_duration"),
                      "request_timeout_seconds": self.timeout,
                      "input_budget": {**budget, "actual_input_tokens": result.get("prompt_eval_count"),
                                       "actual_output_tokens": result.get("eval_count")},
                      "network_request_performed": True},
        )
        if not isinstance(text, str) or not text.strip():
            raise CompletionValidationError("Ollama returned an empty or malformed completion",
                                            completion, "empty_completion")
        done_reason = result.get("done_reason")
        if isinstance(done_reason, str) and done_reason.lower() in {
            "length", "max_tokens", "max_output_tokens", "limit",
        }:
            raise CompletionValidationError("Ollama completion was truncated by the output token limit",
                                            completion, "truncated_completion")
        if result.get("done") is not True:
            raise CompletionValidationError("Ollama response did not confirm completion",
                                            completion, "unfinished_completion")
        actual_input = result.get("prompt_eval_count")
        if (budget["enforced"] and type(actual_input) is int
                and actual_input >= budget["available_input_tokens"]):
            # Saturation is observable, hidden input truncation is not. Keep the
            # response and actual usage for review without claiming a diagnosis
            # that the public API cannot establish.
            raise CompletionValidationError("Writer input saturated the reserved context budget",
                                            completion, "context_saturation_requires_review")
        return completion


class FixtureClient:
    def complete(self, messages: list[dict[str, str]], purpose: str,
                 response_schema: dict | None = None) -> Completion:
        if purpose == "summary":
            text = "Демонстрационное резюме: наблюдения не анализировались моделью."
        elif purpose == "extract":
            text = json.dumps({"records": []}, ensure_ascii=False)
        elif purpose == "action":
            text = json.dumps({
                "action": "clarify_missing_observation",
                "reason": "Проверка контракта без модели.",
                "evidence_turn_ids": [],
                "unknowns": ["Модель не запускалась."],
                "review_required": True,
            }, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported fixture purpose: {purpose}")
        return Completion(text=text, usage={"input_tokens": None, "output_tokens": None},
                          elapsed_seconds=0.0,
                          metadata={"backend": "fixture", "scientific_result": False,
                                    "purpose": purpose})
