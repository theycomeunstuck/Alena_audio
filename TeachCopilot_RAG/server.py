
# $env:TEACHCOPILOT_DEBUG_RAG="1"
# uv run uvicorn server:app --host 0.0.0.0 --port 8099

import logging
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline.db import get_child_full_profile
from pipeline.rag import search_knowledge
from pipeline.config import (
    CHAT_API_BASE_URL,
    CHAT_MODEL,
    CHAT_REQUEST_TIMEOUT_SEC,
    DEBUG_RAG,
    DEFAULT_CHILD_ID,
    STREAM_MODE,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("teachcopilot")

app = FastAPI(title="TeachCopilot RAG API")

class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    child_id: str = DEFAULT_CHILD_ID
    limit: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/profile")
def profile(child_id: str = DEFAULT_CHILD_ID):
    profile_data = get_child_full_profile(child_id)
    return {
        "child_id": child_id,
        "profile": profile_data,
    }


@app.post("/rag/search")
def rag_search(request: RagRequest):
    results = search_knowledge(query=request.query, limit=request.limit)

    return {
        "query": request.query,
        "child_id": request.child_id,
        "count": len(results),
        "results": results,
    }


@app.get("/v1/models")
def openai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "teachcopilot-rag",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts)

    return ""


def _build_rag_system_message(user_text: str) -> str:
    results = search_knowledge(query=user_text)

    if DEBUG_RAG:
        logger.info("=" * 80)
        logger.info("RAG QUERY: %s", user_text)
        logger.info("RAG RESULTS COUNT: %s", len(results))

        for i, item in enumerate(results[:5], start=1):
            content_preview = (
                item.get("content")
                or item.get("image_descriptions")
                or ""
            )[:300]

            logger.info(
                "RAG RESULT %s: topic=%r score=%s content_start=%r",
                i,
                item.get("topic"),
                item.get("score"),
                content_preview,
            )

        logger.info("=" * 80)


    chunks = []
    for index, item in enumerate(results[:5], start=1):
        topic = item.get("topic") or "Без темы"
        content = item.get("content") or ""
        score = item.get("score")
        source = item.get("source_file") or "unknown"
        page = item.get("page_number")
        image_path = item.get("image_path")

        chunks.append(
            f"[Фрагмент {index}]\n"
            f"Тема: {topic}\n"
            f"Источник: {source}" + (f", стр. {page}" if page else "") + "\n"
            f"Score: {score}\n"
            f"{content}"
            + (f"\nИзображение_страницы: {image_path}" if image_path else "")
        )

    if chunks:
        rag_context = "\n\n".join(chunks)
    else:
        rag_context = "По базе знаний ничего релевантного не найдено."

    return f"""
Ты — TeachCopilot, спокойный детский учебный помощник.

Сейчас всегда считается, что с тобой говорит ребёнок.
Объясняй мягко, короткими шагами, без давления.
Не ругай ребёнка за ошибки.
Если вопрос учебный, сначала объясни простыми словами, потом дай пример.

Используй материалы RAG ниже, если они подходят к вопросу.
Не выдавай RAG-фрагменты за полный учебник или единственный источник истины.
Если материалы не подходят, не выдумывай ссылку на них, а отвечай обычным способом.

<RAG_CONTEXT>
{rag_context}
</RAG_CONTEXT>
""".strip()


@app.post("/v1/chat/completions")
def openai_chat_completions(payload: dict[str, Any]):
    messages = payload.get("messages", [])
    user_text = _extract_last_user_text(messages)

    rag_system_message = _build_rag_system_message(user_text)

    new_messages = [
        {
            "role": "system",
            "content": rag_system_message,
        }
    ]

    for message in messages:
        if message.get("role") == "system":
            continue
        new_messages.append(message)

    lmstudio_payload = dict(payload)
    # If CHAT_MODEL is configured it must override the proxy-facing model id.
    # Open WebUI always forwards our advertised id ("teachcopilot-rag"), which the
    # upstream runtime does not have, so the old `not payload.get("model")` guard
    # never fired on the real path and CHAT_MODEL was dead config. Override when the
    # incoming model is empty or is our own advertised id; otherwise honor an
    # explicit upstream model the caller chose.
    incoming_model = payload.get("model")
    if CHAT_MODEL and (not incoming_model or incoming_model == "teachcopilot-rag"):
        lmstudio_payload["model"] = CHAT_MODEL
    lmstudio_payload["messages"] = new_messages

    requested_stream = bool(payload.get("stream", False))
    if requested_stream and STREAM_MODE in {"diagnostic", "debug"}:
        logger.warning(
            "stream=True requested by client, but TeachCopilot proxy keeps stream disabled. "
            "See docs/STREAMING_DIAGNOSIS.md before enabling it."
        )
    lmstudio_payload["stream"] = False

    try:
        response = requests.post(
            f"{CHAT_API_BASE_URL}/chat/completions",
            json=lmstudio_payload,
            timeout=CHAT_REQUEST_TIMEOUT_SEC,
            stream=False,
        )
    except requests.RequestException as exc:
        logger.exception("LM Studio request failed")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"LM Studio request failed: {exc}",
                    "type": "lmstudio_request_error",
                }
            },
        )

    try:
        data = response.json()
    except ValueError:
        logger.error(
            "LM Studio returned non-JSON response: status=%s body_start=%r",
            response.status_code,
            response.text[:500],
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "LM Studio returned non-JSON response. Check whether the model is loaded and LM Studio is healthy.",
                    "type": "lmstudio_non_json_response",
                    "status_code": response.status_code,
                    "body_start": response.text[:500],
                }
            },
        )
    # Чистим reasoning_content в обычном JSON.
    for choice in data.get("choices", []):
        message = choice.get("message") or {}
        message.pop("reasoning_content", None)



    return JSONResponse(
        status_code=response.status_code,
        content=data,
    )
