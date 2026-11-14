# Alёna memory state (November 13–14, 2026)

This file describes the local working tree. Before transferring, compare with
the commits: part of the earlier work and memory changes may remain uncommitted.
The `alena-retrieval-fixes-20261112` branch does not contain all required
migrations and modules and is not a ready release. Current architecture
reference: `docs/ALENA_MEMORY_ARCHITECTURE.md` (consolidated, English);
operations: `docs/ALENA_MEMORY_DEVOPS_MANUAL_2026_11_14.md`; decisions:
`docs/adr/`.

## Memory disabled

`LOCAL_PERSONAL_MEMORY_ENABLED=0` excludes previous dialogue, the own archive,
summaries, personal preferences and approved memory from the new answer's
context. The current message and reply are still stored; the owner can open the
saved history. Learning-material access and the current topic stay separate from
personal memory.

## Previous-version approval

If only new messages from the assigned learner appeared after a version was
prepared, the teacher sees a warning and explicitly re-approves the previous
snapshot. New sources are not included in the previously approved scope. Updates
run as a separate action. If the card, corrections or other sources changed, a
new review is required; approving the whole changed archive is not granted by
that path. Old drafts without a stored base revision are conservatively stale.

For an approved previous selection of `personal_memory` and `personal_graph`,
the saved reviewed context is used instead of re-searching new data. This does
not prove that retrieval always finds the right information or that the model
uses it correctly.

## November 14: unified method, graph channel, diagnostics

`personal_memory` is the single product memory method. In `hybrid` retrieval the
system automatically adds a graph channel (third round-robin leg) when
`LOCAL_MEMORY_GRAPH_CHANNEL=1` (default) and a ready graph index matches the
exact snapshot fingerprint; otherwise it degrades softly with a reason in
`retrieval_diagnostics.graph_channel`. The channel is consume-only – it never
builds, claims or invalidates an index; indexes are built by the explicit
teacher `personal_graph` job (experimental flag, Neo4j on bolt 17687,
graphiti-core installed as a dev dependency, 16k/32k context). Graph facts never
enter the answer text. Commits: `a2825bd` (unified method + PG tests),
`341982e` (Ollama 0.40 overflow wording; graphiti-core declared).

Teacher-only diagnostics: `GET /learners/{id}/memory/diagnostics/index`
(settings, budgets, read-only graph-index status) and
`GET .../memory/diagnostics/retrieval` (the same retrieval the approved chat
runs, with channels, scores, selection warnings and the selected context) plus
the «Диагностика памяти» tab in the web UI (`f65cbe6`). `scripts/memory_doctor.py`
(`--check --live --e2e --graph`) verifies the whole loop on disposable databases
(`376571b`). Live marker tests (`pytest -m live`) cover real Ollama embed/chat
and the personal_memory cycle; graph-cycle tests skip without graphiti-core or
Neo4j (`3aee737`).

## Deployment state

- Working DB on schema 024; `.local.env` sets `LOCAL_CONTEXT_TOKENS=32768`
  (since November 14 – the unified memory chat path requires ≥16384; below that
  it fails with 413/502, see the devops manual §4b). Changing the context stales
  previously approved memory versions once (fail-closed) – the teacher
  re-approves them in the app.
- Ollama 0.40.16: `gemma4:12b` loads and generates; reasoning disabled with
  `think: false`. Note: Ollama ≥0.40 silently truncates oversized prompts when
  `truncate` is not disabled – the strict graph path fail-closes on this via
  `verify_no_truncation` (overflow matcher recognizes the new 0.40 wording).
- Neo4j Community on bolt 17687 with a bundled JDK 21; start command in the
  devops manual §4.2b.
- The running app instance must be restarted to pick up code changes; check
  `/health/ready` (status, schema 24, models) after restarting.

## Verification history (November 13)

The source state was committed as `961bf6e` on `dev` – whole verified
memory-work files together with the previously uncommitted product stack
(migrations 014–024, memory modules, web UI, tests). Left out of the commit on
purpose: НИРС (`research_memory/`), `.local_*` scratch scripts, `outputs/` and a
concept document.

Targeted checks on disposable databases: 160 memory tests, 18 Chromium
correction checks, 39 hierarchical-memory checks – all green. Two full serial
runs on fresh disposable databases: 2392 tests with one failure, then 2392 tests
all green (one skip – optional `neo4j.time`). The single failure was diagnosed
to root cause: `psycopg2.OperationalError ... Address already in use
(WSAEADDRINUSE 10048)` – Windows ephemeral-port exhaustion during the live
browser test's connection burst (no connection pool in the app; Hyper-V
exclusions shrink the port range). It is a test-environment effect, not app
logic and not a PostgreSQL failure; a post-fix confirmation run passed cleanly.
Diagnostics: `teachcopilot-work/503-diag-20261113-run1` and `-run2`.

The synthetic answer eval ran as planned: 124 rows – 112 November first-person
questions in six languages and 12 hand-written scenarios («хз», topic switches,
exact wording) on ru archives; retrieval by the shipped formula with a 4000-byte
budget, generation `gemma4:12b` at temperature 0, deterministic verdicts.
Answer correctness was scored only where key terms were pre-registered before
the run (12 new scenarios, ru): 11 of 12 correct; two honest failures – an
incomplete answer and a refusal although the child's reply was in the passed
context. November answers were published unscored for manual review. On the ru
subset, retrieval found at least one gold source for 76% of November and 92% of
new questions, all gold sources for 55% and 92%. Files:
`teachcopilot-work/benchmark/rows_answer_20261113.json` and the «Ответы модели
на вопрос по своей истории» section on the site.

Run dependencies and tests through `uv` per the root `CLAUDE.md`. PostgreSQL
tests run only on a disposable database, never the working one. Local backups,
test journals and credentials are not published. Synthetic results are not
quality claims about real children.
