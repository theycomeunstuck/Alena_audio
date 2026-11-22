# 🔌 API & Open WebUI Integration

Сейчас `server.py` запускает API `local_app`. Разделы Endpoints, The Open WebUI contract и Quick smoke test описывают прежний прокси, проверенный с имитацией сервиса модели.

Base URL in examples: `http://localhost:8099`.

---

## Endpoints

### `GET /health`
Liveness probe. → `{"status": "ok"}`

### `GET /profile?child_id=<uuid>`
Returns the full child profile from Postgres.
```json
{ "child_id": "…", "profile": { "name": "Миша", "grade": "3-4", "knowledge": [ … ], "interests": [ … ] } }
```

### `POST /rag/search`
Direct retrieval (handy for debugging relevance).
```bash
curl -s localhost:8099/rag/search -H 'content-type: application/json' \
     -d '{"query":"что такое луч","learner_id":"ivanov-ivan","limit":3}'
```
```json
{ "query": "что такое луч", "count": 1,
  "results": [ {
    "topic": "луч",
    "content": "…",
    "source_file": "3 класс 1 часть_Математика.pdf",
    "page_number": 12,
    "image_path": "data/images/3 класс 1 часть_Математика_p012.png",
    "score": 0.62,
    "tags": null,
    "difficulty": null
  } ] }
```

`limit` is validated as `1..50` and passed to retrieval.  `learner_id` is
optional; when present it must come from the authenticated application session
(not from the child's message).  The server validates exactly that JSON card,
uses only its compact `rag_context` in the prompt, and adds its `topic_id`s as
soft retrieval hints.  Full learner cards are never sent to the model.

### `GET /v1/models`
OpenAI-style model list. Advertises a single logical model:
```json
{ "object": "list", "data": [ { "id": "teachcopilot-rag", "object": "model", "owned_by": "local" } ] }
```

### `POST /v1/chat/completions`
The main entry point. It:

1. extracts the last user message,
2. runs RAG and builds a **system message** ("use RAG only if relevant; don't
   invent"),
3. includes source/page metadata in the RAG block,
4. prepends it (dropping any client system message),
5. forwards to the upstream LLM with **`stream=false` forced**,
6. **strips `reasoning_content`** from the returned message,
7. returns the upstream JSON verbatim (minus reasoning).

```bash
curl -s localhost:8099/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "teachcopilot-rag",
  "learner_id": "ivanov-ivan",
  "messages": [{"role":"user","content":"что такое отрезок?"}]
}'
```

---

## The Open WebUI contract (verified behavior)

> Таблица описывает контракт прежнего прокси. Коды `lmstudio_request_error` и `lmstudio_non_json_response` относятся к прежней интеграции с LM Studio.

| Behavior | Guarantee |
|----------|-----------|
| **Model id** | Open WebUI forwards the selected id (`teachcopilot-rag`). If `CHAT_MODEL` is set, the proxy **replaces** it with the real upstream model before calling the LLM. |
| **Streaming** | Even if the client sends `stream: true`, the proxy responds with **plain JSON** (`application/json`, not `text/event-stream`) and forces `stream=false` upstream. Streaming stays off by design — see [STREAMING_DIAGNOSIS.md](STREAMING_DIAGNOSIS.md). |
| **`reasoning_content`** | Removed from every `choices[].message` before returning, so chain-of-thought never leaks to Open WebUI. |
| **RAG system message** | Injected as the first `system` message; any client-supplied system message is dropped. |
| **Errors** | Upstream connection failure → `502 lmstudio_request_error`. Upstream non-JSON → `502 lmstudio_non_json_response` with a body snippet. |

## Connecting Open WebUI

> **Устаревший путь (заморожен 2026-11-22).** Интерфейс продукта работает в [`local_app`](../README.md#personal-memory-alёna), вне Open WebUI. Режим A с Filter Function заморожен (F-012).

### Connecting Open WebUI (mode B – OpenAI-compatible API)

Для ученической учётной записи чат определяет ученика по аккаунту `local_app` и его правам доступа. Для учётной записи преподавателя запрос указывает `learner_id` в теле или в `metadata`, а API проверяет доступ. Связки «пользователь Open WebUI → ученик» в `local_app` нет.

Open WebUI → **Settings → Connections → add an OpenAI connection**:

- **API Base URL:** `http://<host>:8099/v1`
- **API Key** – токен сессии учётной записи `local_app` (`access_token` из `POST /auth/login`). `GET /v1/models` и `POST /v1/chat/completions` проверяют `Authorization: Bearer <токен>` через `require_user`. Срок действия сессии ограничен (`expires_in`).
- **Model:** `teachcopilot-rag`

### Connecting Open WebUI (mode A — Filter Function)

Архивные шаги установки фильтра приведены в [FILTER_INSTALL.md](../FILTER_INSTALL.md).

---

## Legacy PostgreSQL `child_id` resolution

The Filter Function resolves its legacy PostgreSQL profile this way:

1. Explicit message prefix `[child_id:<uuid>] question` (testing).
2. `user_mappings` table: Open WebUI `user_id` → `child_id` (production).
3. `DEFAULT_CHILD_ID` (fallback, `child_only` mode).

The former proxy's JSON learner integration accepted top-level `learner_id`
or `metadata.learner_id`. The calling application or API gateway was responsible
for authentication and mapping users to learners.

---

## Quick smoke test

```bash
curl -s localhost:8099/health
curl -s "localhost:8099/profile?child_id=00000000-0000-0000-0000-000000000001" | head -c 200
curl -s localhost:8099/rag/search -H 'content-type: application/json' -d '{"query":"луч"}'
```
