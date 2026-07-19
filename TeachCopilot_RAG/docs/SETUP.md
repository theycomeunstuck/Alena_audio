# 🛠️ Setup (CPU / local debug)

Every step here was run end-to-end on a **GPU-less** machine. Times/notes reflect
what actually happens.

## Prerequisites

- **Python 3.12** (the project pins `>=3.12,<3.13`).
- **[uv](https://docs.astral.sh/uv/)** for dependency management.
- **PostgreSQL 16 with the `pgvector` extension.** The easiest source is the
  official `pgvector/pgvector:pg16` Docker image (used below).
- Optional: an OpenAI-compatible LLM runtime (LM Studio, vLLM, llama.cpp server…)
  for the chat proxy and the Profile Agent. Retrieval and personalization work
  **without** it.

---

## 1. Install dependencies

```bash
uv sync
```

`uv` resolves **CPU** PyTorch wheels automatically (see `[tool.uv.sources]` in
`pyproject.toml` → the `pytorch-cpu` index). No CUDA is pulled.

> Installing via plain `pip`? Use `pip install -r requirements.txt` — the file now
> carries `--extra-index-url https://download.pytorch.org/whl/cpu` so the
> `torch==…+cpu` pin resolves. (If you regenerate it with `uv export`, re-add that line.)

---

## 2. PostgreSQL + pgvector

```bash
docker run -d --name teachcopilot_pg -p 5433:5432 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=teachcopilot \
  pgvector/pgvector:pg16
```

Point the app at it (overrides `.env` because `load_dotenv` doesn't override real
env vars):

```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5433/teachcopilot"
```

> Using port **5433** avoids clashing with any Postgres already on 5432. If you run
> pgvector on 5432, keep the `.env` default and skip the export.

Already have Postgres? Just enable the extension once:
`CREATE EXTENSION IF NOT EXISTS vector;`

---

## 3. Apply the schema

```bash
uv run python scripts/apply_schema.py
```

This runs `db/schema.sql` **and** a few idempotent `ALTER TABLE … ADD COLUMN IF
NOT EXISTS` migrations (`difficulty`, `tags`, `avg_response_time_sec`).

> `db/schema.sql` is now self-consistent — those columns are declared in it too, so
> applying the bare file (`psql -f db/schema.sql`) also works. Prefer
> `apply_schema.py`: it's cross-platform and safe to re-run.

---

## 4. Seed a test child

```bash
uv run python -c "from pipeline.db import insert_test_child; insert_test_child()"
```

Creates **Миша** (`00000000-0000-0000-0000-000000000001`) with sample knowledge,
interests and error patterns. Idempotent.

---

## 5. Ingest lesson material

Text files use a tiny header format — first lines are `TOPIC:` / `SUBJECT:`, the
rest is content:

```
TOPIC: отрезок
SUBJECT: math
Отрезок — это часть прямой, ограниченная двумя точками…
```

```bash
uv run python scripts/ingest.py --dir books --mode text
```

> ⚠️ In this checkout `data/` contains only `images/` — **no `.txt` files**, so a
> plain run reports "No supported files". The current text sources live in
> `books/`. Drop new `.txt`/`.md`/`.docx`/`.rtf` lesson files into `data/` or
> pass `--dir <your_folder>`. The first embedding call downloads the multilingual
> model (~470 MB).

Other modes:

- **PDF (multimodal, needs a vision LLM):**
  `uv run python scripts/ingest.py --mode pdf --file book.pdf --subject math --topic "геометрия 3 класс" --pages 1-50`
- **JSON task bank:** `uv run python scripts/ingest.py --mode json --file tasks.json`

---

## 6. Verify

```bash
uv run python scripts/test_pipeline.py
```

Expected: **`27/27 passed`** (the multimodal step SKIPs unless you've ingested a
PDF). This exercises DB, RAG search + filters, prompt building, the filter
inlet/outlet, user mappings and the profile-agent gate.

---

## 7. Run

**As a standalone OpenAI-compatible proxy:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8099
# then in Open WebUI → Settings → Connections, add:
#   Base URL: http://<host>:8099/v1     (Model: teachcopilot-rag)
```

**As a Filter Function** (no proxy needed): follow
[`../FILTER_INSTALL.md`](../FILTER_INSTALL.md) to paste the `Filter` class into
Open WebUI → Workspace → Functions.

See **[API.md](API.md)** for endpoints and the request/response contract, and
**[CONFIGURATION.md](CONFIGURATION.md)** for every environment variable.
