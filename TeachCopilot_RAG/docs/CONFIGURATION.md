# ⚙️ Configuration

All configuration is environment-driven (via `.env`, loaded by
`pipeline/config.py`). Real environment variables **override** `.env` values.

## Core

| Variable | Default | Notes |
|----------|---------|-------|
| `DATABASE_URL` | `postgresql://localhost/teachcopilot` | Postgres + pgvector connection string. |
| `DEFAULT_CHILD_ID` | `""` | Child served when no mapping/prefix resolves (used in `child_only` mode). |
| `TEACHCOPILOT_SALT` | — | Salt for hashing identifiers. Keep secret. |
| `LOG_LEVEL` | `INFO` | Standard logging level. |

## Retrieval (RAG)

| Variable | Default | Notes |
|----------|---------|-------|
| `EMBEDDING_MODEL` | `auto` | `auto` → try `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, best for Russian), fall back to `all-MiniLM-L6-v2`. Or pin any SentenceTransformer id. **Must be 384-dim** to match `knowledge_base.embedding vector(384)`. |
| `EMBEDDING_DEVICE` | `cpu` (in `cpu_debug`) | `cpu` / `cuda` / `auto`. |
| `RAG_TOP_K` | `3` | Max fragments returned. |
| `RAG_MIN_SCORE` | `0.5` (config) / `0.3` (dev `.env`) | Cosine-similarity cutoff, `score = 1 - (embedding <=> query)`. Raise after switching to a stronger model. |

> **Changing the embedding model requires re-ingesting** — old vectors won't match
> a new model's space (and dims must stay 384).

## Runtime profile & device

| Variable | Default | Notes |
|----------|---------|-------|
| `TEACHCOPILOT_RUNTIME_PROFILE` | `cpu_debug` | `cpu_debug` favors CPU + small models. Set to anything else to let `EMBEDDING_DEVICE` default to `auto`. |

## Chat proxy (`server.py`)

| Variable | Default | Notes |
|----------|---------|-------|
| `TEACHCOPILOT_CHAT_API_BASE_URL` / `LMSTUDIO_BASE_URL` | `http://127.0.0.1:1234/v1` | Upstream OpenAI-compatible endpoint. |
| `TEACHCOPILOT_CHAT_MODEL` / `LMSTUDIO_MODEL` | `""` | Model sent upstream. **When set, it overrides the proxy-facing id** (Open WebUI always sends `teachcopilot-rag`). When empty, the caller's model is forwarded unchanged. |
| `TEACHCOPILOT_CHAT_TIMEOUT_SEC` | `180` | Upstream request timeout. |
| `TEACHCOPILOT_STREAM_MODE` | `off` | `off` \| `diagnostic` \| `debug`. **Streaming stays disabled**; `diagnostic` only logs when a client requested `stream=true`. See [STREAMING_DIAGNOSIS.md](STREAMING_DIAGNOSIS.md). |
| `TEACHCOPILOT_DEBUG_RAG` | `false` | Logs each RAG query + top results. |
| `TEACHCOPILOT_LEARNER_DATA_DIR` | `../learner-data` relative to this project | Directory containing the JSON `learners/`, `catalog/` and `tools/`. Required only when callers send `learner_id`; set an explicit path if code and data deploy separately. |

## Speaker mode

| Variable | Default | Notes |
|----------|---------|-------|
| `TEACHCOPILOT_SPEAKER_MODE` | `child_only` | `child_only`: every unknown user gets the default child prompt. `mapped_or_adult`: mapped users are children; unknown users get the **adult/teacher** analytical prompt. |

## Profile Agent (adaptive scoring)

Runs on a **separate** LLM endpoint — never loops back through the Open WebUI proxy.

| Variable | Default | Notes |
|----------|---------|-------|
| `PROFILE_LLM_API_URL` | = `CHAT_API_BASE_URL` | OpenAI-compatible endpoint for assessments. |
| `PROFILE_LLM_MODEL` | = `CHAT_MODEL` | If empty/unconfigured, assessment is **skipped gracefully** (no crash). |
| `PROFILE_UPDATE_INTERVAL` | `5` | Assess every _N_ user messages. |

---

## Example `.env` (CPU debug)

```dotenv
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/teachcopilot
EMBEDDING_MODEL=auto
RAG_TOP_K=3
RAG_MIN_SCORE=0.3
LOG_LEVEL=DEBUG
DEFAULT_CHILD_ID=00000000-0000-0000-0000-000000000001

LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_MODEL=qwen/qwen3-vl-4b

TEACHCOPILOT_RUNTIME_PROFILE=cpu_debug
TEACHCOPILOT_STREAM_MODE=off
TEACHCOPILOT_SPEAKER_MODE=child_only
TEACHCOPILOT_DEBUG_RAG=1
# Optional when learner-data is not alongside TeachCopilot_RAG:
# TEACHCOPILOT_LEARNER_DATA_DIR=/srv/teachcopilot/learner-data
```

## Production notes

- Set `TEACHCOPILOT_RUNTIME_PROFILE` to something other than `cpu_debug` (or set
  `EMBEDDING_DEVICE=cuda`) to use a GPU for embeddings.
- Keep `EMBEDDING_MODEL` consistent between ingest and serving; re-ingest on change.
- Raise `RAG_MIN_SCORE` (e.g. `0.5`) once you've validated retrieval quality.
- Leave `TEACHCOPILOT_STREAM_MODE=off` until streaming parity is proven.
- Resolve `learner_id` from an authenticated session before invoking the proxy;
  never accept it directly from a child's message. The current Open WebUI Filter
  does not yet map Open WebUI users to JSON learner IDs, so use proxy mode for
  JSON-card personalisation.
