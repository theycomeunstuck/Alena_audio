# Code Review Report

## Findings

1. `pyproject.toml` and the generated `requirements.txt` pinned CUDA PyTorch
   wheels (`+cu128`) even though the local debug machine has no GPU. This made
   local installs fragile and contradicted the CPU-only runtime constraint.
2. `server.py` had a hardcoded fallback model and accepted `stream=true` from
   Open WebUI while silently forcing non-streaming upstream behavior. That
   preserved working responses, but made streaming failures hard to diagnose.
   Open WebUI's read-only code confirms that streaming callers expect
   `text/event-stream`, `data: [DONE]`, and may surface `reasoning_content`.
3. `pipeline/filter_function.py` and `pipeline/prompt_builder.py` contained
   commented-out adult/child routing instead of an explicit runtime setting.
4. RAG prompt assembly did not clearly tell the model that retrieved fragments
   are optional context, not a complete source of truth.
5. `AudioAPI` defaulted Whisper to `small` on CPU and `core/TTS/tts_engine.py`
   forced CUDA-only Qwen TTS settings (`device_map="cuda:0"`,
   `flash_attention_2`, `bfloat16`).
6. `AudioAPI` ASR WebSocket cancelled its background partial emitter on the
   first `stop`, although the route documentation says the socket remains open
   for later utterances.

## Changes Made

- Added CPU-debug runtime settings in `pipeline/config.py`.
- Switched RAG dependency resolution to CPU PyTorch wheels through `uv` source
  config and regenerated `requirements.txt`.
- Added `.env.cpu.example` for local CPU debugging.
- Kept RAG streaming disabled, but added `TEACHCOPILOT_STREAM_MODE=diagnostic`
  behavior and `docs/STREAMING_DIAGNOSIS.md`.
- Made adult/child prompt behavior controlled by `TEACHCOPILOT_SPEAKER_MODE`.
- Added RAG context boundary instructions in `server.py` and
  `pipeline/prompt_builder.py`.
- Made `profile_agent` skip assessment when no profile LLM model is configured.
- Made AudioAPI device/model selection env-driven for Whisper and Qwen TTS.
- Fixed `/ws/asr` so `stop` finalizes and resets under a lock without killing
  the long-lived background emitter.
- Added focused RAG runtime config tests.

## Tests Run

- `python -m py_compile server.py pipeline/config.py pipeline/rag.py pipeline/prompt_builder.py pipeline/filter_function.py pipeline/profile_agent.py`
- `python -m py_compile core/config.py core/TTS/tts_engine.py app/api/routes_ws.py app/services/streaming_asr.py app/services/streaming_verify.py`
- `uv run pytest tests/test_runtime_config.py` — 3 passed.
- Verified Open WebUI context files are read-only.

## Not Fully Verified

- Full RAG integration tests need PostgreSQL + pgvector and populated data.
- AudioAPI FastAPI tests still import real model startup paths through
  `app.main`; they need installed ASR/SpeechBrain/TTS dependencies and model
  assets or a separate lightweight test mode.
- Streaming was intentionally not re-enabled. See `docs/STREAMING_DIAGNOSIS.md`.
