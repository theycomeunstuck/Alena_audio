# Runtime Contracts

Last verified: 2026-07-09.

This document describes behavior that tests now lock down. If code changes break
these contracts, update the tests and docs intentionally.

## `/rag/search`

Request:

```json
{
  "query": "что такое отрезок",
  "child_id": "00000000-0000-0000-0000-000000000001",
  "limit": 3
}
```

To personalise from the checked-in JSON learner cards, pass the verified
application identity as `learner_id` (for example `volk-08`).  `child_id` is a
legacy PostgreSQL UUID and does not select a JSON card.

```json
{
  "query": "Как делить 408 на 4 столбиком?",
  "learner_id": "volk-08",
  "limit": 3
}
```

Contract:

- `query` is required and non-empty.
- `limit` is 1..50.
- `limit` is passed into retrieval, not only sliced after retrieval.
- Response includes source metadata when present.
- With `learner_id`, the service validates exactly that card, adds its compact
  `rag_context` to the model prompt and uses its curriculum `topic_id`s as soft
  retrieval hints. Full cards, journals, points and `learner_model` never enter
  the prompt.

Response shape:

```json
{
  "query": "что такое отрезок",
  "child_id": "00000000-0000-0000-0000-000000000001",
  "count": 1,
  "results": [
    {
      "topic": "Отрезок",
      "content": "Отрезок — часть прямой...",
      "image_descriptions": null,
      "source_file": "small_geometry_test.md",
      "page_number": null,
      "image_path": null,
      "difficulty": null,
      "tags": null,
      "score": 0.91
    }
  ]
}
```

## `search_knowledge()`

Contract:

- Empty/blank query returns `[]`.
- Default top-k comes from `RAG_TOP_K`.
- Optional runtime `limit` overrides top-k and is clamped to `1..50`.
- Result rows include `topic`, `content`, `image_descriptions`, `source_file`,
  `page_number`, `image_path`, `difficulty`, `tags`, `score`.
- DB/model failures are logged and return `[]`.

Important limitation: `[]` can mean either "no relevant material" or "retrieval
failed". Check logs with `TEACHCOPILOT_DEBUG_RAG=1` before concluding content is
missing.

## `/v1/models`

Response advertises one logical model:

```json
{
  "object": "list",
  "data": [
    {
      "id": "teachcopilot-rag",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

Open WebUI should select `teachcopilot-rag`. The proxy maps that to the real
upstream model via `TEACHCOPILOT_CHAT_MODEL`.

## `/v1/chat/completions`

Contract:

- Last user message is extracted from OpenAI-compatible `messages`.
- RAG runs on that last user text.
- An optional top-level `learner_id` (or `metadata.learner_id`) selects a
  validated JSON learner context. The calling application must derive it from
  its authenticated session, never from the child's message text.
- The proxy prepends one system message with RAG instructions and fragments.
- Client-supplied system messages are dropped.
- If `TEACHCOPILOT_CHAT_MODEL` is set and incoming model is `teachcopilot-rag`,
  the upstream payload uses `TEACHCOPILOT_CHAT_MODEL`.
- `stream` is forced to `false` upstream.
- Response is regular JSON, not SSE.
- `choices[].message.reasoning_content` is removed before returning.
- Request failures return `502 lmstudio_request_error`.
- Non-JSON upstream responses return `502 lmstudio_non_json_response`.

Injected RAG context includes:

- fragment number;
- topic;
- source file;
- page number when available;
- similarity score;
- content;
- image path when available.

## Open WebUI Integration

Use one integration mode:

- Proxy mode: Open WebUI connection points at `http://<host>:8099/v1`.
- Filter Function mode: paste `pipeline/filter_function.py` into Open WebUI.

Do not use both. If both are enabled, personalization and RAG can be injected
twice, which makes answers longer and harder to debug.

## Streaming

Current contract: streaming is disabled.

Even if Open WebUI sends `stream: true`, the proxy uses `stream=false` upstream
and returns normal JSON. This is intentional because prior streaming produced
garbled output and loops.

Enable streaming only behind a new explicit mode after validating:

- exact SSE framing;
- `data: [DONE]`;
- UTF-8/Cyrillic behavior;
- removal of `reasoning_content` from streamed deltas;
- reverse-proxy buffering behavior.
