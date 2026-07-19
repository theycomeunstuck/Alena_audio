# TeachCopilot RAG Runbook

Last verified: 2026-07-09, CPU-only machine.

This is the practical operator guide for bringing the RAG backend from a fresh
checkout to a working local service.

## What Must Work

The service is considered healthy when all of these are true:

- PostgreSQL accepts connections and has `pgvector`.
- `db/schema.sql` has been applied.
- At least one child profile exists.
- `knowledge_base` has embedded rows.
- `/rag/search` returns relevant fragments with source metadata.
- `/v1/chat/completions` injects the RAG system message and forwards to the real
  upstream LLM with `stream=false`.
- Open WebUI is connected either through the proxy or the Filter Function, not both.

## Known-Good Local Setup

Use a clean pgvector container. If port `5433` is occupied, use `5434`.

```bash
cd /home/dev/projects/alena/_work/TeachCopilot_RAG

docker run -d --name teachcopilot_pg_local -p 5434:5432 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=teachcopilot \
  pgvector/pgvector:pg16

export DATABASE_URL="postgresql://postgres:postgres@localhost:5434/teachcopilot"
export HF_HOME="$PWD/pretrained_models/hf"

uv sync --group dev
uv run python scripts/apply_schema.py
uv run python -c "from pipeline.db import insert_test_child; insert_test_child()"
uv run python scripts/ingest.py --dir books --mode text
uv run python scripts/test_pipeline.py
```

Expected final line:

```text
Results: 27/27 passed, 0/27 failed
ALL STEPS PASSED
```

## Content Ingestion

Current checkout has real text sources in `books/`:

- `rag_secret_test.md`
- `small_geometry_test.md`
- `широкая база.rtf`

`data/` currently contains rendered PDF page images, not text documents. Running
text ingest on `data/` is therefore a no-op.

Text/RTF/Markdown ingestion:

```bash
uv run python scripts/ingest.py --dir books --mode text
```

PDF ingestion requires a working vision LLM endpoint because pages are rendered
and described before embedding:

```bash
uv run python scripts/ingest.py \
  --mode pdf \
  --file "books/3 класс 1 часть_Математика.pdf" \
  --subject math \
  --topic "математика 3 класс" \
  --pages 1-20
```

Do not run full PDF ingest until `PROFILE_LLM_API_URL` and `PROFILE_LLM_MODEL`
point to a real vision-capable OpenAI-compatible model.

## Smoke Checks

```bash
curl -s http://localhost:8099/health

curl -s http://localhost:8099/rag/search \
  -H 'content-type: application/json' \
  -d '{"query":"что такое отрезок","limit":3}' | jq

curl -s http://localhost:8099/v1/models | jq
```

For `/v1/chat/completions`, an upstream LLM must be running:

```bash
export TEACHCOPILOT_CHAT_API_BASE_URL="http://127.0.0.1:1234/v1"
export TEACHCOPILOT_CHAT_MODEL="<real-upstream-model-id>"

uv run uvicorn server:app --host 0.0.0.0 --port 8099
```

## What Was Verified

On 2026-07-09:

- Unit/contract tests: `7 passed`.
- Fresh pgvector container on `5434`.
- Schema apply: OK.
- Test child seed: OK.
- Text ingest from `books/`: 3 rows inserted.
- Integration pipeline: `27/27 passed`.

The existing `teachcopilot_pg` container on `5433` had a broken role state
(`postgres` not permitted to log in). For clean local work, use a fresh container
or reset that database manually.

## Stop Conditions

Stop and fix before product work if any of these happen:

- `/rag/search` returns `[]` for known `books/` questions after ingest.
- `scripts/test_pipeline.py` fails any DB/RAG/prompt/filter step.
- Open WebUI is configured with both Filter Function and proxy at the same time.
- Streaming is enabled without passing the checks in `STREAMING_DIAGNOSIS.md`.
- `EMBEDDING_MODEL` changes without re-ingesting all knowledge rows.
