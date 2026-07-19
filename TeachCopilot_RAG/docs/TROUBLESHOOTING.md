# 🧯 Troubleshooting

Real failure modes, with the fix. Ordered roughly by how often they bite.

---

### RAG search always returns `[]`

`search_knowledge` **swallows exceptions** and returns an empty list, so a broken
query looks like "no results". Check, in order:

1. **DB reachable?** `psql "$DATABASE_URL" -c 'select 1'`.
2. **Extension present?** `CREATE EXTENSION IF NOT EXISTS vector;`
3. **Missing columns.** `search_knowledge` selects `difficulty` and `tags`. If you
   applied a **bare/old `schema.sql`** those columns may be absent → every query
   errors → `[]`. Fix: run `python scripts/apply_schema.py` (adds them idempotently).
   The current `schema.sql` already declares them.
4. **Nothing ingested / score too high.** Lower `RAG_MIN_SCORE`, confirm rows:
   `psql "$DATABASE_URL" -c 'select count(*) from knowledge_base'`.
5. Set `TEACHCOPILOT_DEBUG_RAG=1` to log the query + top hits.

---

### "No supported files found in data"

`data/` in this checkout has only `images/` — **no text files**. Ingest needs
`.txt` / `.md` / `.docx` / `.rtf` files with a `TOPIC:` / `SUBJECT:` header (see
[SETUP.md](SETUP.md) §5). Add yours, or point `--dir` at another folder.

---

### `psycopg2` / connection errors on a fresh machine

- Wrong port: this repo's `.env` defaults to `5432`; the docs run pgvector on
  `5433`. Export `DATABASE_URL` to match, or run pgvector on 5432.
- Real env vars override `.env` (`load_dotenv` doesn't override) — check
  `echo $DATABASE_URL` if the app ignores your `.env` edit.

---

### `pip install -r requirements.txt` fails on `torch==…+cpu`

The file carries `--extra-index-url https://download.pytorch.org/whl/cpu`. If it
was regenerated with `uv export` (which drops that line), re-add it, or just use
`uv sync`.

---

### First query is slow / downloads a model

`EMBEDDING_MODEL=auto` downloads `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB)
on first `embed_text`. It's cached afterwards (under your HF cache).

---

### Vector dimension mismatch on ingest

`knowledge_base.embedding` is `vector(384)`. Both default models are 384-dim. If
you pin a different `EMBEDDING_MODEL`, keep it 384-dim **or** change the column and
re-ingest.

---

### Open WebUI shows garbled / looping text, or nothing streams

Streaming is intentionally **off**. The proxy returns full JSON even when the
client asks for `stream=true`. That's expected — the previous streaming path
produced garbled Russian and could loop. See [STREAMING_DIAGNOSIS.md](STREAMING_DIAGNOSIS.md)
before attempting to enable it.

---

### Profile scores never change

The Profile Agent needs a **separate** LLM (`PROFILE_LLM_API_URL` /
`PROFILE_LLM_MODEL`). If unconfigured it **skips silently** (by design — no crash).
Configure it and confirm updates land in the `events` table:
`psql "$DATABASE_URL" -c "select event_type, created_at from events order by created_at desc limit 5"`.

---

### `pytest` tries to hit a live DB / downloads models

`pyproject.toml` sets `testpaths = ["tests"]` so `pytest` only collects the fast
unit tests. Don't run `scripts/test_pipeline.py` via pytest — it's a full
integration script (`python scripts/test_pipeline.py`) and needs a live DB.
