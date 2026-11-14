# Alёna – memory devops manual (November 14, 2026)

Working tree: repository `Alena_audio`, application `TeachCopilot_RAG`. This
manual describes the local memory loop: how it is built, what configures it, how
to start and verify it. Before moving to another machine, compare with the actual
code and `docs/ALENA_MEMORY_STATUS_2026_11_13.md` – part of the work may remain
uncommitted. Architecture reference: `docs/ALENA_MEMORY_ARCHITECTURE.md`;
design decisions: `docs/adr/` and `docs/glossary.md`.

## 1. Memory map

Four retrieval loops, each with its own role and implementation.

**(a) Learning materials (product RAG).** `Store.hybrid_search` in
`local_app/store.py:630` – pgvector + PostgreSQL full-text hybrid. Vector leg:
cosine similarity (`embedding <=> %s::vector`, similarity = 1 − distance); text
leg: `ts_rank` over `plainto_tsquery('russian')` and `plainto_tsquery('simple')`.
Fusion: reciprocal rank fusion with k = 60, final tie-break by `id` (deterministic
order). If not all active materials are indexed for the requested embedding
model, the search raises `IndexNotReadyError` instead of silently returning a
partial result.

**(b) Personal memory.** Hybrid mode – round-robin merge of a vector leg
(BGE-M3, 1024 dimensions) and a lexical leg over teacher-approved source
lessons. Ranking: `rank_source_candidates` from `local_app/personal_graph_retrieval.py`;
`local_app/personal_memory.py` wires it with the source embedding cache
(`SourceEmbeddingCache`). Only teacher-approved sources are searchable.

**(c) Conversation retrieval.** `local_app/conversation_retrieval.py`, two
formulas: `legacy` – shipped default (cosine + 0.04 × exact-term fraction), `rrf`
– opt-in (reciprocal rank fusion of cosine and stemmed BM25). Switched by the
`fusion` argument of the calling code; there is no env switch.

**(d) Personal graph (experimental).** Graphiti over Neo4j on
`bolt://127.0.0.1:17687` (`research_memory/personal_memory_lab/backends.py`).
Index registry – `teachcopilot.personal_graph_indexes` (migration
`016_personal_graph_index_registry.sql`); graph embedding cache – migration
`017_personal_graph_embedding_cache.sql`. The application does not import
Graphiti or LangGraph by default. The experiment is only valid with a context of
16384 or 32768 tokens and the models `gemma4:12b` + `bge-m3` (validated in
`local_app/config.py`). Graph facts NEVER enter the answer text: graph findings
only point at canonical source replies, which are the quoted evidence.

**(e) Channel unification [landed November 14, commit a2825bd].** The graph is an
automatic channel inside personal_memory hybrid: `LOCAL_MEMORY_GRAPH_CHANNEL`
(on by default) adds graph findings as a third round-robin leg when a ready index
matches the exact snapshot fingerprint; without running Neo4j or without a ready
index the channel degrades softly with a reason in diagnostics
(`channel_disabled`, `no_ready_index`, `password_file_unavailable`, …). The
channel only consumes a ready index and never builds one; graph facts never enter
the answer text. The index is built by an explicit teacher `personal_graph` job
(requires Neo4j and a 16k/32k context).

## 2. Env flags

All values are read by `Settings.from_env()` from `local_app/config.py`; the
`.local.env` file at the `TeachCopilot_RAG` root is loaded with `override=False`
(real environment variables win over the file). Memory-related variables:

| Variable | Default | Effect |
|---|---|---|
| `DATABASE_URL` | `postgresql://teachcopilot@127.0.0.1:5434/teachcopilot_local` | Working local DB (Docker container, port 5434 → 5432). Never used for tests – see section 6. |
| `OLLAMA_URL` | `http://127.0.0.1:11435` | HTTP loopback only, no credentials/query/fragment (`require_local_url` raises otherwise). |
| `LOCAL_CHAT_MODEL` | `gemma4:12b` | Ollama chat model. Strictly `gemma4:12b` for the experimental graph. |
| `LOCAL_EMBEDDING_MODEL` | `bge-m3` | Embedding model (BGE-M3, 1024 dims). Family `bge-m3` for the graph. |
| `LOCAL_CONTEXT_TOKENS` | `32768` | Model context; minimum 2048. **Unified memory chat requires ≥16384** – see section 4b. The experimental graph requires exactly 16384 or 32768. |
| `LOCAL_MIN_SIMILARITY` | `0.35` | Minimum semantic similarity in conversation retrieval. |
| `LOCAL_PERSONAL_MEMORY_ENABLED` | `1` | `0` excludes previous dialogue, own archive, summaries, personal preferences and approved memory from the new answer's context. The current message and reply are still stored; the owner can view the history. |
| `LOCAL_PERSONAL_MEMORY_RETRIEVAL_MODE` | `hybrid` | One of `chronological`, `lexical`, `vector`, `hybrid`; anything else fails at startup. |
| `LOCAL_MEMORY_GRAPH_CHANNEL` | `1` | Automatic graph channel inside personal_memory hybrid; without Neo4j or without a ready index it degrades softly with a reason in diagnostics. |
| `LOCAL_EXPERIMENTAL_PERSONAL_GRAPH` | `0` | `1` enables the explicit experimental graph method. Requires a 16k/32k context and `gemma4:12b` + `bge-m3`, otherwise the config is invalid. |
| `LOCAL_PERSONAL_GRAPH_RETRIEVAL_MODE` | `graph_hybrid` | One of `chronological`, `lexical`, `vector`, `hybrid`, `graph_hybrid`. |
| `LOCAL_PERSONAL_GRAPH_PASSWORD_FILE` | `C:/Users/name/PycharmProjects/misc/teachCopilot/.local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt` | Neo4j password file (user `neo4j`). Never commit or print. |

Related (not memory, same config): `LOCAL_REQUEST_TIMEOUT` (180),
`LOCAL_MAX_OUTPUT_TOKENS` (1024), `LOCAL_SESSION_HOURS` (12),
`LOCAL_MEMORY_RESPONSE_JOURNAL`, `LOCAL_CHAT_RESPONSE_JOURNAL`,
`LOCAL_GENERATION_SCHEDULER_DSN`, `LOCAL_DEMO_LOGIN`.

## 3. Setup and start

### 3.1 Environment (uv)

Python 3.12 and `uv`; dependency sources are the application's `pyproject.toml`
and `uv.lock`. All commands run from the `TeachCopilot_RAG` directory.

```powershell
cd C:\Users\name\PycharmProjects\misc\teachCopilot\Alena_audio\TeachCopilot_RAG
uv sync --locked
```

New dependencies only via `uv add <package>` (dev: `uv add --dev <package>`) with
`pyproject.toml` and `uv.lock` saved; never replace this with a system `pip`
install. Run project commands through `uv run --locked`. `torch` is already a
dependency (CPU PyTorch index) – its absence in a separate environment is not a
reason to re-add it. `graphiti-core` (0.31.2) and the `neo4j` driver (6.4.1) are
dev dependencies as of November 14 – required for the graph path and `pytest -m live`.

### 3.2 PostgreSQL in Docker

`compose.local.yml` – image `pgvector/pgvector:0.8.7-pg18`, container
`teachcopilot_local_db`, port `127.0.0.1:5434 → 5432`, password from `.local.env`
(`POSTGRES_PASSWORD`), data in the named volume `local_database`.

```powershell
docker compose --env-file .local.env -f compose.local.yml up -d --wait
```

If `.local.env` does not exist yet, `scripts/setup_local.ps1` creates it
(generates the password, writes `DATABASE_URL`, `OLLAMA_URL`, `LOCAL_CHAT_MODEL`,
`LOCAL_EMBEDDING_MODEL`, `LOCAL_CONTEXT_TOKENS`), creates `.venv-local`, installs
dependencies from `requirements-local.lock.txt`, applies migrations and imports data:

```powershell
powershell -File scripts\setup_local.ps1
```

### 3.3 One-script start

```powershell
powershell -File scripts\start_local.ps1
```

Optional `param([int]$ApiPort = 8099)`. The script: brings up the Docker DB
(`up -d --wait`), applies migrations (`scripts/manage_local.py migrate`), prepares
Ollama (`OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_NO_CLOUD=1`,
`OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=2`;
binary – `.local\ollama\ollama.exe`, else from PATH; starts `ollama serve` if the
port is silent) and waits up to 30 seconds for readiness. Then it starts the
memory worker (`scripts/memory_worker.py`, hidden process, logs
`.local\memory-worker-<guid>.stdout.log` and `.stderr.log`) and the API
`uvicorn server:app` on `127.0.0.1:8099`. When the script exits, the worker stops.

Migrations separately (`manage_local.py` subcommands: `migrate`, `import`,
`bootstrap`, `export`, `backup`, `restore`):

```powershell
uv run --locked python scripts\manage_local.py migrate
```

### 3.4 Ollama

Local Ollama on `http://127.0.0.1:11435`, models `gemma4:12b` (chat) and
`bge-m3` (embeddings). On November 13 Ollama was updated from 0.24.0 to 0.40.16 –
before the update `gemma4:12b` failed to load. Liveness check:

```powershell
Invoke-RestMethod http://127.0.0.1:11435/api/version
ollama list
```

### 3.5 Neo4j

The experimental graph uses Neo4j Community on `bolt://127.0.0.1:17687`
(loopback only), user `neo4j`, password from the file at
`LOCAL_PERSONAL_GRAPH_PASSWORD_FILE` (default
`.local/neo4j-runtime/neo4j-community-5.26.32/personal-memory-password.txt`;
a bundled JDK 21 sits next to the distribution). Without running Neo4j the
product still works: the graph method is simply unavailable, and since the
November 14 unification the channel degrades softly.

## 4. Verification

### 4.1 Regular tests

```powershell
uv run --locked python -m pytest
```

The `pyproject.toml` config collects `tests/` only. The `deep` marker (repo-root
`pytest.ini`) marks long real-model integration tests; they are excluded by
default (`addopts = -m "not deep"`). Full serial runs were executed on fresh
disposable databases: 2392 tests. PostgreSQL tests run only on a disposable DB
(section 6).

### 4.2 Disposable databases (disposable-DB launcher)

The launcher
`C:/Users/name/Documents/ChatGPT/Claude/teachcopilot-work/codex_disposable_db_suite_20261113.py`
reads `DATABASE_URL` from `.local.env` only to reach the local PostgreSQL server,
creates a random `teachcopilot_memory_verify_<12 hex>` database, points both DSNs
(`TEACHCOPILOT_TEST_DSN` and `DATABASE_URL`) at it, runs `python -m pytest
<targets>` from `TeachCopilot_RAG` and drops only its own database (prefix and
name-length asserted). Arguments are pytest targets (default `tests`):

```powershell
python C:\Users\name\Documents\ChatGPT\Claude\teachcopilot-work\codex_disposable_db_suite_20261113.py tests
```

The same prefix `teachcopilot_memory_verify_` is used by `scripts/verify_memory.py`
– it creates the disposable DB itself and passes it via `TEACHCOPILOT_TEST_DSN`.
The load runner `scripts/load_memory.py` is guarded by `require_disposable_dsn` –
it only accepts DSNs of the form `teachcopilot_memory_verify_` + 12 hex chars;
the provider is replaced with `ControlledRuntime` (delay injection, all prompts
recorded), any random HTTP call kills the process before sending. For graph
integration checks `scripts/create_personal_graph_test_db.py` creates a
`nirs_personal_graph_it_<12 hex>` database with migrations:

```powershell
uv run --locked python scripts\create_personal_graph_test_db.py --name-file <path-to-name-file>
```

The smoke `scripts/smoke_personal_graph_api.py` («Real API, PostgreSQL, Neo4j and
local Gemma with synthetic accounts and data») runs a real API on that disposable
database with synthetic accounts and a learner.

### 4.2b Starting Neo4j (for the graph channel)

Neo4j sits next to the application and starts with its bundled JDK:

```powershell
$env:JAVA_HOME = 'C:\Users\name\PycharmProjects\misc\teachCopilot\.local\neo4j-runtime\jdk-21.0.12+8'
cd C:\Users\name\PycharmProjects\misc\teachCopilot\.local\neo4j-runtime\neo4j-community-5.26.32
.\bin\neo4j.bat console
```

Bolt comes up on `127.0.0.1:17687` (password – in `personal-memory-password.txt`
next to it). The graph index is built by a teacher `personal_graph` job in the
app or by `memory_doctor --e2e --graph`; a ready index is automatically consumed
by the personal_memory hybrid channel. Status – the «Диагностика памяти» tab in
the app.

### 4.3 Live tests and memory_doctor

```powershell
uv run --locked python -m pytest -m live
```

The `live` marker requires a running Ollama; the model is taken from the
`LOCAL_CHAT_MODEL` env variable, not hardcoded. Graph-cycle tests additionally
skip when graphiti-core, the Neo4j driver or the Neo4j port is unavailable.

The diagnostics script `scripts/memory_doctor.py`: `--check` (read-only
PostgreSQL, memory tables, Ollama models, embed probe, env flags, `/health/ready`
of a running app), `--live` (plus a real model call), `--e2e` (the full
personal_memory cycle on a disposable `nirs_memory_doctor_*` database: job →
worker → approval → chat with the real model; JSON report), `--graph` (extends
e2e with the personal_graph cycle when Neo4j is reachable; otherwise skip).
Example: `uv run --locked python scripts/memory_doctor.py --check --live`.
Reports go to `teachcopilot-work/` and are never published to git.

### 4.4 Memory diagnostics in the UI (teacher only)

The teacher workspace has the memory panels: «Основания следующего шага –
Память занятий» and «Мои беседы» with a «Обновить» button. Each memory version
shows its method, date, the `stale` flag and the approval status
(«подтверждена» / «черновик»); a stale approved snapshot is not used by the
assistant. Plus the «Диагностика памяти» tab: graph-index status card and a
retrieval probe form (query, mode, byte budget) with the channel table. This is
functional diagnostics, not a numeric dashboard.

## 4b. Context tokens: the 16k floor (English note)

**Set `LOCAL_CONTEXT_TOKENS=32768` for the unified memory path.** The
application accepts smaller values, but the personal-memory chat path
breaks below 16384:

- the complete-memory resolver prompt alone is larger than the byte
  budget the app derives as `(context_tokens - max_output_tokens) * 2`;
  at `LOCAL_CONTEXT_TOKENS=8192` a real chat returns HTTP 413 (or 502
  through some routes) — verified independently twice on 2026-11-14;
- the graph channel requires `context_tokens` of 16384 or 32768
  (`ApplicationModel` refuses anything else), and the index config
  fingerprint binds an index to the context it was built with. After
  changing `LOCAL_CONTEXT_TOKENS`, previously approved memory versions
  go stale once (fail-closed by design) and the teacher must
  re-approve them in the app.

Current deployment: `.local.env` sets `LOCAL_CONTEXT_TOKENS=32768`
(since 2026-11-14). If you change it, restart the app
(`scripts/start_local.ps1`) and verify `/health/ready`, then re-approve
memory versions as prompted.

## 4c. Moving to a new server (English note)

**Code is portable.** `local_app/` contains zero absolute paths - all defaults
are `Path(__file__)`-relative (repo tree) or env-driven. The only exception was
the memory_doctor default report folder, now `MEMORY_DOCTOR_OUTPUT_DIR`-driven
(falls back to `<repo>/TeachCopilot_RAG/.local/memory-doctor-reports`).

What is MACHINE-FIXED and must be migrated or recreated explicitly:

| Item | Where it lives now | Migration |
|---|---|---|
| Working DB data (child data) | Docker named volume `local_database` on the old host | `pg_dump`/`pg_restore` the `teachcopilot_local` DB (or copy the volume); never start tests against it on the new host either |
| `.local.env` | `TeachCopilot_RAG/.local.env` (gitignored) | Copy by hand - contains DB password, `LOCAL_CONTEXT_TOKENS=32768`, graph password-file path |
| Neo4j data + config + password | `.local/neo4j-runtime/neo4j-community-5.26.32/` (data/ inside) + bundled `jdk-21.0.12+8` | Copy the whole `neo4j-runtime` folder, or reinstall + restore a dump; graph facts live here |
| Ollama models (~7.4 GB blobs) | `C:/Users/<user>/.ollama/models` | `ollama pull gemma4:12b` + `ollama pull bge-m3` on the new host, or copy `~/.ollama/models` |
| `.venv-local` | `TeachCopilot_RAG/.venv-local` | Recreate: run `scripts/setup_local.ps1` (never copy the folder) |
| Playwright Chromium | `.local/ms-playwright` or shared cache | `uv run --locked python -m playwright install chromium` |

What moves WITH the repo automatically: all code, `web/`, migrations, `docs/`,
config defaults (all `Path(__file__)`-relative), `web/` static files.

Repo-relative defaults that assume the same layout next to the repo:
`LOCAL_PERSONAL_GRAPH_PASSWORD_FILE` (`.local/neo4j-runtime/...` - override via
env if the layout differs), demo credentials file, response journals
(`.local/*.sqlite3` - fresh ones are created on first run; old journals are
historical records, copy them if you need the archive).

Ports are loopback conventions, not paths: app 8099, Ollama 11435, PostgreSQL
5434, Neo4j bolt 17687 - keep or override via env/compose.

## 5. Troubleshooting

**HTTP 409 during memory work – normal re-verification.** After new messages
from an assigned learner, the previous memory snapshot must be re-approved by
the teacher (the UI shows a warning and an explicit confirmation). New sources
are not added to the previously approved scope; if the card, corrections or
other sources changed – a new review is required. The API also returns 409 on
re-confirmation with a different selection (`memory was already confirmed with a
different selection`) and on revision change (`learner revision changed; reload
before continuing`).

**`PersonalGraphIndexStale`.** Retrieval refuses to use a stale approved graph
index without touching the registry `teachcopilot.personal_graph_indexes` or the
graph itself (`local_app/personal_graph_index.py`). Fix: build a new index
version and approve it; old versions without a stored base revision are
conservatively treated as stale.

**«unknown model architecture» in Ollama.** Means Ollama is older than the model
blobs: 0.24.0 did not know all architectures of the newer blobs. Update Ollama
(on November 13 updated to 0.40.16 via winget), after which `gemma4:12b` loads
and generates.

**gemma4 reasoning mode.** The model may return a «thinking» field; the app
disables reasoning with `think: false` (`local_app/runtime.py`). If you see
service reasoning inside an answer, check the request goes through
`OllamaRuntime`, not directly to Ollama.

**Ollama 0.40 silent truncation.** Without `truncate: false` an oversized prompt
is cut to roughly half the requested context (verified: a ~55k-token probe at
`num_ctx: 16384` evaluated 8195 tokens). The strict graph path therefore sends
`truncate: false, shift: false` and fail-closes on the overflow rejection
(`_context_overflow` recognizes both the legacy and the 0.40 wording). Product
prompts stay safe because byte budgets cap them far below the truncation
threshold.

**Playwright cannot find the browser.** Install Chromium through uv and use the
same `PLAYWRIGHT_BROWSERS_PATH` for install and run; some tests look in
`TeachCopilot_RAG/.local/ms-playwright` by default – installing into the shared
cache does not change that path by itself.

```powershell
uv run --locked python -m playwright install chromium
```

**`neo4j.time` – optional dependency.** One full-run skip because of the missing
`neo4j.time` module is expected, not a regression.

**`WSAEADDRINUSE` (10048) on `127.0.0.1:5434` in live browser tests.** The app
opens a fresh connection per operation; a burst of thousands of short
connections exhausts Windows ephemeral ports (range 49152–65535, partly taken by
Hyper-V exclusions). This is a test-environment effect, not app logic and not a
PostgreSQL failure (10 of 100 connections used at failure time; isolated runs
green). Possible fix – widen the dynamic range `netsh int ipv4 set dynamicport
tcp start=1025 num=64510` (admin rights, reversible) – apply only by user
decision. Diagnostics – `teachcopilot-work/503-diag-20261113-run1` and `-run2`.

## 6. Safety rules

- PostgreSQL tests – only on disposable `nirs_*` / `teachcopilot_memory_verify_*`
  databases. The working DB `teachcopilot_local` is never used for tests; before
  any cleanup verify the exact object and preserve user data.
- No real child data in tests, reports or the public site – synthetic accounts
  and learners only (as in `smoke_personal_graph_api.py`: «Синтетический
  ученик», `@synthetic.test`).
- Secrets – the Neo4j password, `.local.env` contents, tokens, response journals
  – never committed, never printed in reports, never published. Local backups,
  test journals and credentials are not published.
- Synthetic test results do not represent quality verification on real children
  and do not prove that retrieval always finds what is needed or that the model
  uses it correctly. Keep НИРС (`research_memory/`) and product changes separate.
- Applying schema 024 to the working DB is not a new app release; verify the
  running process separately and never assume it auto-updated.

## Verification checklist

Run top to bottom; every step is either green or points to section 5.

```powershell
cd C:\Users\name\PycharmProjects\misc\teachCopilot\Alena_audio\TeachCopilot_RAG
```

1. Environment reproduces: `uv sync --locked` finishes without errors.

```powershell
uv sync --locked
```

2. Docker Desktop is running; the DB comes up and is healthy.

```powershell
docker compose --env-file .local.env -f compose.local.yml up -d --wait
```

3. Ollama is alive on 11435, models `gemma4:12b` and `bge-m3` are present.

```powershell
Invoke-RestMethod http://127.0.0.1:11435/api/version
ollama list
```

4. DB schema is current (migrations 001–024 applied).

```powershell
uv run --locked python scripts\manage_local.py migrate
```

5. Regular tests are green (without `-m live`, without `deep`).

```powershell
uv run --locked python -m pytest
```

6. For PostgreSQL tests a disposable `teachcopilot_memory_verify_*` database (or
`nirs_personal_graph_it_*` for the graph) is created – and only that one.

```powershell
python C:\Users\name\Documents\ChatGPT\Claude\teachcopilot-work\codex_disposable_db_suite_20261113.py tests
```

7. Live tests – with Ollama running: `uv run --locked python -m pytest -m live`;
the model comes from `LOCAL_CHAT_MODEL`/`LOCAL_EMBEDDING_MODEL`; without a
running Ollama the tests skip.

```powershell
uv run --locked python -m pytest -m live
```

8. Memory diagnostics – `memory_doctor.py --check`, then `--e2e`; for the graph
experiment – `--graph` and `--live`.

```powershell
uv run --locked python scripts\memory_doctor.py --check
```

9. The whole stack starts with one script; the API answers on 8099.

```powershell
powershell -File scripts\start_local.ps1
```

10. In the teacher UI the «Память занятий» and «Мои беседы» panels open; memory
versions show method and approval status; a 409 on staleness is normal –
re-approve the snapshot as the teacher.

11. Neo4j (if the graph experiment is on): reachable at
`bolt://127.0.0.1:17687`, the password is read from
`LOCAL_PERSONAL_GRAPH_PASSWORD_FILE`, the password is never printed or committed.

12. Checklist result: no step touched the working DB `teachcopilot_local`, no
secrets published, synthetic results never attributed to real children.
