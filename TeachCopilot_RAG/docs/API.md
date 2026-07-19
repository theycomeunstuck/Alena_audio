# 🔌 API & Open WebUI Integration

`server.py` is a small FastAPI app that exposes an **OpenAI-compatible** surface
plus a couple of helper endpoints. Everything below was verified against a running
instance with a fake upstream.

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
     -d '{"query":"что такое луч","limit":3}'
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

`limit` is validated as `1..50` and passed to retrieval.

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
  "messages": [{"role":"user","content":"что такое отрезок?"}]
}'
```

---

## The Open WebUI contract (verified behavior)

| Behavior | Guarantee |
|----------|-----------|
| **Model id** | Open WebUI forwards the selected id (`teachcopilot-rag`). If `CHAT_MODEL` is set, the proxy **replaces** it with the real upstream model before calling the LLM. |
| **Streaming** | Even if the client sends `stream: true`, the proxy responds with **plain JSON** (`application/json`, not `text/event-stream`) and forces `stream=false` upstream. Streaming stays off by design — see [STREAMING_DIAGNOSIS.md](STREAMING_DIAGNOSIS.md). |
| **`reasoning_content`** | Removed from every `choices[].message` before returning, so chain-of-thought never leaks to Open WebUI. |
| **RAG system message** | Injected as the first `system` message; any client-supplied system message is dropped. |
| **Errors** | Upstream connection failure → `502 lmstudio_request_error`. Upstream non-JSON → `502 lmstudio_non_json_response` with a body snippet. |

### Connecting Open WebUI (mode B — proxy)

Open WebUI → **Settings → Connections → add an OpenAI connection**:

- **API Base URL:** `http://<host>:8099/v1`
- **API Key:** any non-empty string (not validated locally)
- **Model:** `teachcopilot-rag`

### Connecting Open WebUI (mode A — Filter Function)

Alternatively, skip the proxy and paste `pipeline/filter_function.py`'s `Filter`
class into Open WebUI → Workspace → Functions. `inlet()` personalizes and grounds
each request; `outlet()` runs the adaptive profile update. Full steps:
[`../FILTER_INSTALL.md`](../FILTER_INSTALL.md).

> Use **one** mode, not both — otherwise personalization runs twice.

---

## `child_id` resolution order

Both the proxy and the Filter Function resolve the child the same way:

1. Explicit message prefix `[child_id:<uuid>] question` (testing).
2. `user_mappings` table: Open WebUI `user_id` → `child_id` (production).
3. `DEFAULT_CHILD_ID` (fallback, `child_only` mode).

---

## Quick smoke test

```bash
curl -s localhost:8099/health
curl -s "localhost:8099/profile?child_id=00000000-0000-0000-0000-000000000001" | head -c 200
curl -s localhost:8099/rag/search -H 'content-type: application/json' -d '{"query":"луч"}'
```
