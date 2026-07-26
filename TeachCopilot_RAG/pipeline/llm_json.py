"""One place that asks an OpenAI-compatible runtime for a JSON answer.

Both stage-4 task design and lesson-note ingestion need the same thing: send a
prompt, get back a JSON object, fail loudly when the model returns prose. The
callable is injectable (``complete=...``) so every caller can be tested without
a running LM Studio.

``requests`` is used when installed and falls back to ``urllib.request`` so the
module imports on a bare interpreter — the ZPD tooling must stay runnable
without the full service environment.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pipeline.config import (
    CHAT_API_BASE_URL,
    CHAT_API_KEY,
    CHAT_MODEL,
    JSON_MODE,
    PROMPT_WARN_CHARS,
)

logger = logging.getLogger(__name__)

# A JSON-returning chat call: (prompt, system_prompt) -> raw model text.
CompleteFn = Callable[[str, str], str]


class LlmError(RuntimeError):
    """Raised when the model could not be reached or did not return JSON."""


class LlmContractError(LlmError):
    """Рантайм не принял запрос — например, не умеет запрошенный response_format."""


def _post_json(url: str, payload: dict, timeout: float, api_key: str = "", attempts: int = 2) -> dict:
    """POST с одной повторной попыткой на транспортную ошибку.

    Разрыв TLS-соединения на первом запросе к облачному API — обычное дело, а
    прогон карточек на всю группу из-за него терял бы работу целого ребёнка.
    Повторяются только сетевые сбои: отказ с кодом 4xx — это ответ по существу.
    """
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return _post_json_once(url, payload, timeout, api_key)
        except LlmContractError:
            raise
        except LlmError as exc:
            last_error = exc
            if attempt < attempts:
                logger.warning("Сетевая ошибка при запросе к LLM, повтор %d: %s", attempt + 1, exc)
    raise last_error  # noqa: RSE102 - последняя транспортная ошибка уже описана


def _post_json_once(url: str, payload: dict, timeout: float, api_key: str = "") -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        # Значение заголовка нигде не логируется: ключ не должен попасть ни в
        # вывод команды, ни в отчёт об ошибке.
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import requests  # noqa: PLC0415 - optional dependency, resolved per call
    except ImportError:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # У 4xx тело обычно содержит внятное объяснение провайдера —
            # оно полезнее, чем голый код ответа.
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise LlmContractError(f"провайдер отклонил запрос ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise LlmError(f"LLM недоступна ({url}): {exc}") from exc
        except ValueError as exc:
            raise LlmError(f"LLM вернула не JSON-ответ ({url}): {exc}") from exc

    try:
        response = requests.post(url, data=body, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise LlmError(f"LLM недоступна ({url}): {exc}") from exc
    if 400 <= response.status_code < 500:
        raise LlmContractError(
            f"провайдер отклонил запрос ({response.status_code}): {response.text[:400]}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise LlmError(
            f"LLM вернула не JSON-ответ ({url}), status={response.status_code}: {response.text[:300]}"
        ) from exc


def chat_text(
    prompt: str,
    system_prompt: str = "",
    *,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.4,
    timeout: float = 180.0,
    response_format: dict | None = None,
    api_key: str | None = None,
) -> str:
    """Send one prompt to the configured runtime and return the raw reply text."""
    url = f"{(base_url or CHAT_API_BASE_URL).rstrip('/')}/chat/completions"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    resolved_model = (model or CHAT_MODEL).strip()
    if resolved_model:
        payload["model"] = resolved_model
    if response_format:
        payload["response_format"] = response_format

    data = _post_json(url, payload, timeout, CHAT_API_KEY if api_key is None else api_key)
    if isinstance(data.get("error"), dict):
        message = data["error"].get("message", "без описания")
        raise LlmContractError(f"провайдер вернул ошибку: {message}")
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"неожиданная структура ответа LLM: {json.dumps(data, ensure_ascii=False)[:300]}") from exc

    text = message.get("content") or ""
    if not text.strip():
        # Reasoning models keep their scratchpad in a separate field and
        # sometimes leave `content` empty, putting the answer at the end of the
        # reasoning. Failing outright would make such models unusable, so we
        # try the scratchpad — the JSON extractor takes the object out of it.
        reasoning = message.get("reasoning_content") or ""
        if reasoning.strip():
            logger.warning(
                "Модель вернула пустой content, ответ ищем в reasoning_content. "
                "Для таких моделей лучше отключить размышление (/no_think)."
            )
            return reasoning
        raise LlmError("LLM вернула пустой ответ")
    return text


def extract_json_object(text: str) -> dict:
    """Parse the first JSON object in a model reply.

    Local models routinely wrap JSON in ```json fences or add a closing remark,
    so a strict ``json.loads`` of the whole reply is too brittle to rely on.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1] if stripped.count("```") >= 2 else stripped[3:]
        if stripped.lstrip().lower().startswith("json"):
            stripped = stripped.lstrip()[4:]
        stripped = stripped.strip()

    try:
        parsed = json.loads(stripped)
    except ValueError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise LlmError(f"в ответе LLM нет JSON-объекта: {text[:300]}") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except ValueError as exc:
            raise LlmError(f"не удалось разобрать JSON из ответа LLM: {exc}; ответ: {text[:300]}") from exc

    if not isinstance(parsed, dict):
        raise LlmError(f"ожидался JSON-объект, получен {type(parsed).__name__}")
    return parsed


def json_response_format(schema: dict | None = None) -> dict | None:
    """Как просить строгий JSON у рантайма, с учётом настройки JSON_MODE.

    Строгая схема надёжнее уговоров в промпте, но поддерживается не везде:
    LM Studio умеет json_schema, часть облачных API — только json_object,
    некоторые не умеют ничего. Поэтому режим auto пробует мягкий вариант и
    откатывается на текстовый контракт при отказе.
    """
    if JSON_MODE == "off":
        return None
    if JSON_MODE == "json_schema" and schema:
        return {"type": "json_schema", "json_schema": {"name": "task_card", "schema": schema, "strict": True}}
    return {"type": "json_object"}


def complete_json(
    prompt: str,
    system_prompt: str = "",
    *,
    complete: CompleteFn | None = None,
    schema: dict | None = None,
    **kwargs: Any,
) -> dict:
    """Ask for JSON and return the parsed object.

    ``complete`` replaces the HTTP call entirely; tests pass a fake that returns
    a canned reply, which keeps the whole task-design pipeline testable offline.
    """
    size = len(prompt) + len(system_prompt)
    if size > PROMPT_WARN_CHARS:
        # Локальные рантаймы по умолчанию дают окно 4096 токенов и молча
        # обрезают вход — модель возвращает неполный JSON без всякой ошибки.
        logger.warning(
            "Промпт длиной %d символов (~%d токенов). Проверьте, что у модели окно "
            "контекста хотя бы 8192, иначе запрос обрежется без сообщения об ошибке.",
            size, size // 3,
        )

    if complete is not None:
        raw = complete(prompt, system_prompt)
        logger.debug("LLM reply: %d chars", len(raw))
        return extract_json_object(raw)

    response_format = json_response_format(schema)
    if response_format:
        try:
            raw = chat_text(prompt, system_prompt, response_format=response_format, **kwargs)
            logger.debug("LLM reply: %d chars (json mode)", len(raw))
            return extract_json_object(raw)
        except LlmContractError as exc:
            if JSON_MODE != "auto":
                raise
            # Не каждый рантайм умеет response_format. В режиме auto это не
            # повод падать: повторяем обычным запросом, формат держит промпт.
            logger.warning("Рантайм не принял response_format (%s); повторяю без него", exc)

    raw = chat_text(prompt, system_prompt, **kwargs)
    logger.debug("LLM reply: %d chars", len(raw))
    return extract_json_object(raw)
