# Verification Report — 2026-07-09

Environment:

- Machine: CPU-only local machine.
- RAG DB: fresh `pgvector/pgvector:pg16` container on port `5434`.
- RAG model cache: `pretrained_models/hf`.
- AudioAPI mode: `AUDIOAPI_LIGHTWEIGHT_MODE=1`, `DEVICE=cpu`, `WHISPER_MODEL=tiny`.

## RAG

Commands:

```bash
cd /home/dev/projects/alena/_work/TeachCopilot_RAG
export DATABASE_URL="postgresql://postgres:postgres@localhost:5434/teachcopilot"
export HF_HOME="$PWD/pretrained_models/hf"

uv run pytest -q
uv run python scripts/apply_schema.py
uv run python -c "from pipeline.db import insert_test_child; insert_test_child()"
uv run python scripts/ingest.py --dir books --mode text
uv run python scripts/test_pipeline.py
```

Results:

- Unit/contract tests: `7 passed`.
- Schema apply: OK.
- Test child seed: OK.
- Ingest from `books/`: 3 files, idempotent.
- Integration pipeline: `27/27 passed`.
- PDF multimodal checks: skipped because no PDF pages were ingested in this run.

Verified behavior:

- `/rag/search` passes `limit` into retrieval.
- RAG results include `source_file`, `page_number`, `image_path`.
- OpenAI-compatible proxy forces `stream=false`.
- Proxy strips `reasoning_content`.
- Prompt builder includes child profile and RAG material correctly.
- Filter Function imports and runs `inlet`/`outlet`.
- User mapping fallback remains child mode in default `child_only` mode.

## AudioAPI / TTS Contract

Commands:

```bash
cd /home/dev/projects/alena/_work/AudioAPI

AUDIOAPI_LIGHTWEIGHT_MODE=1 DEVICE=cpu WHISPER_MODEL=tiny \
  .venv/bin/python -m pytest -q
```

Result:

- `42 passed, 5 deselected`.

Verified behavior:

- REST health/upload/download/enhance/transcribe/speaker verify contracts.
- ASR websocket contract.
- Streaming verification contract.
- TTS validation errors.
- TTS happy path in lightweight mode returns valid WAV bytes.
- Voice metadata and transcription cache behavior.
- Audio read/write avoids `torchcodec`/CUDA dependency in CPU mode.
- Pytest collection ignores deep model tests and client examples.

## Known Blockers

Production TTS is not fully verified:

- Local Qwen weights exist under `pretrained_models/Qwen3-TTS-12Hz-1.7B-Base`.
- Current venv does not have `qwen_tts`.
- `qwentts_update.zip` contains project files, not an installable `qwen_tts`
  backend package.

Streaming remains intentionally disabled:

- Full JSON responses are verified.
- SSE streaming needs the checklist in `STREAMING_DIAGNOSIS.md` before enabling.

PDF multimodal RAG is not verified in this run:

- Page PNGs exist in `data/images`.
- A vision-capable upstream LLM is required to produce `image_descriptions`.
