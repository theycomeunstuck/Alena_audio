# Streaming Diagnosis

Current default: `TEACHCOPILOT_STREAM_MODE=off`.

`server.py` accepts OpenAI-compatible requests from Open WebUI but always sends
`stream=false` to the upstream model runtime. This preserves the last known good
behavior: full JSON responses work, while the previous real-time path produced
garbled Russian output and could loop.

Likely failure points to verify before re-enabling streaming:

- The proxy must return `text/event-stream` whenever the client requests
  `stream=true`; returning plain JSON to a streaming client can desynchronize
  Open WebUI's parser.
- Every SSE frame must be exactly `data: <json>\n\n`, with `data: [DONE]\n\n`
  at the end.
- Upstream `reasoning_content` must be removed from streamed `delta` and
  non-stream `message` payloads before forwarding to Open WebUI.
- UTF-8 chunks must not split or be decoded line-by-line with lossy settings.
  If Open WebUI still corrupts Cyrillic text, serialize SSE JSON with
  `ensure_ascii=True`.
- The proxy should keep upstream `stream=true` and downstream SSE framing in
  the same code path; mixed "upstream non-stream, downstream fake-stream" should
  be tested separately.
- Disable HTTP buffering in reverse proxies and keep `Cache-Control: no-cache`,
  `Connection: keep-alive`, and `X-Accel-Buffering: no`.

Recommended diagnostic sequence:

1. Capture raw upstream LM Studio SSE with `curl -N`.
2. Replay the same request through `server.py` with `TEACHCOPILOT_STREAM_MODE=diagnostic`.
3. Compare frame boundaries, JSON keys, `reasoning_content`, and `[DONE]`.
4. Only after parity is proven, add a guarded streaming implementation behind
   `TEACHCOPILOT_STREAM_MODE=enabled`.
