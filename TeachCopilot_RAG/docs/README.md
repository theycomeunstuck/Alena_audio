# 📚 TeachCopilot — Documentation

> Current direction: JSON-first learner data. The learner profile source of
> truth is `../learner-data/`; PostgreSQL/pgvector documents describe a legacy
> retrieval implementation, not the active storage decision.

This folder is the documentation hub. Everything here was **verified against a
running instance** (Postgres + pgvector, embedding model, proxy) — the commands
below actually work.

| Doc | What's inside |
|-----|---------------|
| **[LEARNER_DATA_USAGE.md](LEARNER_DATA_USAGE.md)** | Current JSON-first flow, commands, examples and sample output. |
| **[../knowledge-data/README.md](../knowledge-data/README.md)** | Add JSON learning material, build the local vector index and search it. |
| **[SETUP.md](SETUP.md)** | Legacy pgvector setup; do not use for the current JSON-only phase. |
| **[RAG_RUNBOOK.md](RAG_RUNBOOK.md)** | Known-good local runbook and verified RAG startup path. |
| **[CONFIGURATION.md](CONFIGURATION.md)** | Every environment variable, its default, and CPU vs production notes. |
| **[API.md](API.md)** | HTTP endpoints, the OpenAI-compatible contract, and Open WebUI integration. |
| **[CONTRACTS.md](CONTRACTS.md)** | Runtime contracts locked by tests: search, proxy, model id, streaming behavior. |
| **[EVALUATION.md](EVALUATION.md)** | How to evaluate retrieval quality and build a golden query set. |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | The gotchas that actually bite (schema drift, empty `data/`, streaming). |
| **[STREAMING_DIAGNOSIS.md](STREAMING_DIAGNOSIS.md)** | Why streaming is off and how to safely diagnose it. |
| **[VERIFICATION_2026_07_09.md](VERIFICATION_2026_07_09.md)** | Exact commands/results from the latest CPU-only verification pass. |
| **[REVIEW_REPORT.md](REVIEW_REPORT.md)** | History of the code-review/hardening pass. |

---

## Current JSON-first scope

1. `learner-data/learners/*.json` is the source of truth for learner cards.
2. `build_context.py <learner_id>` renders the safe, compact projection for an
   LLM; full cards remain outside the prompt.
3. Educational material remains JSON files linked by `topic_id` and `grade`.
4. A file-backed vector index for those educational JSON files is the next
   implementation task. Do not start PostgreSQL merely to read learner cards.

## Legacy pgvector implementation

A child (or a teacher/parent) chats in Open WebUI. In the legacy path, before the message reaches the
LLM, TeachCopilot:

1. **Identifies the speaker** — maps the Open WebUI user to a child profile (or an
   adult/teacher), see [`SPEAKER_MODE`](CONFIGURATION.md).
2. **Personalizes** — pulls the child's name, grade, learning level, difficulties,
   interests and error patterns from Postgres and injects them into the system
   prompt.
3. **Grounds with RAG** — embeds the question and retrieves the most relevant
   lesson fragments from a pgvector knowledge base (cosine similarity).
4. **Adapts over time** — every _N_ messages a separate Profile-Agent LLM assesses
   understanding and nudges the child's mastery scores.

Retrieved fragments are always presented as **optional context** ("use only if
relevant; do not invent facts") — never as the sole source of truth.

---

## Architecture

```
            ┌──────────────────────────────────────────────┐
            │                 Open WebUI                    │
            │        (child / teacher types a message)      │
            └───────────────┬───────────────┬──────────────┘
       Integration mode A   │               │   Integration mode B
      (Filter Function)     │               │   (Connection / proxy)
                            ▼               ▼
                ┌────────────────┐   ┌────────────────────────┐
                │ filter_function │   │   server.py (FastAPI)  │
                │  inlet / outlet │   │  /v1/chat/completions  │
                └───────┬────────┘   └───────────┬────────────┘
                        │  build personalized + RAG system prompt
                        ▼
        ┌───────────────────────────────┐        ┌──────────────────────┐
        │   pipeline/                    │        │  Upstream LLM runtime │
        │   ├─ db.py       (Postgres)    │        │  (LM Studio / vLLM /  │
        │   ├─ rag.py      (pgvector)    │───────▶│   any OpenAI API)     │
        │   ├─ prompt_builder.py         │        └──────────────────────┘
        │   ├─ profile_agent.py  ────────┼──▶ separate Profile-Agent LLM
        │   └─ event_logger.py           │
        └───────────────┬───────────────┘
                        ▼
             ┌────────────────────────┐
             │  PostgreSQL + pgvector │
             │  children / profiles / │
             │  knowledge_base / ...  │
             └────────────────────────┘
```

**Two ways to connect Open WebUI** (pick one):

- **A — Filter Function** (`pipeline/filter_function.py`): pasted into Open WebUI
  → Workspace → Functions. `inlet()` personalizes the request; `outlet()` updates
  the profile. See [`FILTER_INSTALL.md`](../FILTER_INSTALL.md).
- **B — Connection / proxy** (`server.py`): Open WebUI points at this service as an
  OpenAI-compatible endpoint. It injects the RAG system message and forwards to the
  upstream LLM. See [API.md](API.md).

---

## Legacy quickstart (CPU, verified)

```bash
# 0) deps
uv sync

# 1) Postgres + pgvector (Docker; use 5434 if 5433 is taken)
docker run -d --name teachcopilot_pg_local -p 5434:5432 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=teachcopilot \
  pgvector/pgvector:pg16
export DATABASE_URL="postgresql://postgres:postgres@localhost:5434/teachcopilot"

# 2) schema + test child
uv run python scripts/apply_schema.py
uv run python -c "from pipeline.db import insert_test_child; insert_test_child()"

# 3) ingest lesson texts available in this checkout
uv run python scripts/ingest.py --dir books --mode text

# 4) sanity check (23+ integration checks)
uv run python scripts/test_pipeline.py

# 5) run the proxy
uv run uvicorn server:app --host 0.0.0.0 --port 8099
```

Full walkthrough → **[SETUP.md](SETUP.md)** and **[RAG_RUNBOOK.md](RAG_RUNBOOK.md)**.

---

## Runtime facts (as verified)

| Aspect | Value |
|--------|-------|
| Default embedding model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, good for Russian) — downloaded on first use (~470 MB) |
| Default device | **CPU** (`TEACHCOPILOT_RUNTIME_PROFILE=cpu_debug`) |
| Vector search | pgvector cosine, `score = 1 - (embedding <=> query)`, `RAG_MIN_SCORE` cutoff, top-`RAG_TOP_K` |
| Streaming to Open WebUI | **OFF by default** — full JSON responses only (see [STREAMING_DIAGNOSIS.md](STREAMING_DIAGNOSIS.md)) |
| `reasoning_content` | Stripped from responses before returning to Open WebUI |
| Proxy model id | Advertised as `teachcopilot-rag`; the real upstream model is set by `CHAT_MODEL` |
