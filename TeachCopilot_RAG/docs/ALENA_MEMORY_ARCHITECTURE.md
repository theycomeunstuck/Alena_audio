# Alёna memory architecture (consolidated)

Status: current reference, consolidated on 2026-11-14. This document merges the
still-relevant parts of the November 7 handoff and architecture plan
(`ALENA_HANDOFF_2026_11_07.md`, `ALENA_ARCHITECTURE_PLAN_2026_11_07.md` — both
now superseded and kept for history) with the November 12–14 changes:
the unified personal_memory method with the graph channel, the fail-closed
approval design, the 16k context floor and the teacher diagnostics endpoints.

Operational setup and commands live in `ALENA_MEMORY_DEVOPS_MANUAL_2026_11_14.md`.
Design decisions and their rationale live in `docs/adr/` and `docs/glossary.md`.
Terminology below matches `docs/glossary.md` exactly.

## 1. Binding user requirements

| Requirement | Accepted definition |
| --- | --- |
| One session | All of a child's conversations on one calendar day of activity |
| Recent messages | Up to 20 individual messages, user and assistant roles included |
| Brief memory | One-day summary plus a summary of the last 7 active days |
| Seven days | Last dates with conversations, gaps of weeks allowed |
| Full history | Original messages of all chats in an append-only archive with author, exact time and links |
| Old facts | Search over the whole allowed own archive, including events outside the 7 active days |
| Time | UTC `TIMESTAMPTZ` at the source, IANA zones in metadata, `Asia/Tomsk` as learning and display zone |
| User timezone | Separate display preference |
| Time in summaries | Date and at most an hour; the exact timestamp stays available in the archive |
| Summary placement | SYSTEM prompt with explicit untrusted-data boundaries and provenance |
| Passive collection | Real dialogues are stored; derived views are built from them |
| Active collection | One voluntary age-appropriate question per active day, shown/skipped/answered tracked |
| Archive splitting | Fixed windows with overlap and preserved source-turn boundaries |
| Catalog | Common conversation topics and time facets; one current classifier recipe; atomically replaced labels |
| Reclassification | Atomic replacement of derived state while the source archive stays untouched |

`learning_sessions` keep serving the API, dialogue restore and learning actions.
Daily memory groups them by `activity_day` regardless of topic.

## 2. Data flow (chat turn)

1. The API authorizes the account and learner access, resolves the technical
   session, reserves the new turn with an idempotency key.
2. The store adds source text, role, UTC timestamp, timezones and the activity day.
3. The server builds the allowed own transcript, the last 20 messages and the daily summaries.
4. The resolver decides the query intent; `personal_memory_recall` explicitly gates
   full-archive retrieval.
5. On a recall question, a server-side callback retrieves fixed chunks from the own
   archive (conversation_retrieval, shipped legacy fusion = cosine + 0.04 exact-term
   fraction; RRF fusion exists but is opt-in and not wired to any runtime call).
6. Approved teacher observations are validated through the reviewed grant and the
   approved memory version (fail-closed, see ADR-002).
7. The runtime assembles SYSTEM context (approved memory + summaries), recent
   messages, learning materials and selected sources within the model budget
   (`num_ctx` passed per request; deployment currently at 32768).
8. The assistant reply is stored as its own source message. Derived summaries and
   the catalog refresh when their source snapshot changes.

A generation failure keeps the accepted user turn and its delivery status; a retry
with the same request id returns the existing result or a clear in-progress status.

## 3. Archive, time and migrations

Canonical archive: `teachcopilot.memory_sources` plus the verified
`learning_messages` links. Corrections are separate sources linked via
`corrects_source_id`; correction groups are atomic in every context selection.

| Field | Meaning |
| --- | --- |
| `created_at` | Exact server UTC moment of the source turn |
| `event_timezone` | Recorded IANA zone; legacy rows may be `NULL` |
| `activity_timezone` | Zone used to assign the learning day |
| `activity_day` | Stored day derived from `created_at` + `activity_timezone` |
| `learning_timezone` | Calendar-boundary setting for new child turns |
| `display_timezone` | User display preference |

Migrations 020–024: timezones (020), hierarchical summaries (021), check-ins (022),
dialogue catalog (023), chat-revision counter + memory-version base revision (024).
Migration verification rules (row counts, text hashes, idempotent re-run, IANA data
on Windows) are in the superseded plan and remain the standard for new migrations.

## 4. Conversation memory layers

- `recent_messages` — up to 20 individual messages; the current query is included once.
- `last_day` — one explicitly named active day with a visible date.
- `last_seven_active_days` — up to seven last active dates, current day included.
- Summaries live in `conversation_summary_versions` (migration 021), scope =
  child + participant, extractive algorithm, source-fingerprint-bound.
- Placement order in SYSTEM: stable instructions/profile → seven-day summary →
  one-day summary → full-archive retrieval sources + reviewed evidence → recent
  messages. Summary text is an explicitly bounded untrusted block.

## 5. Full-archive retrieval

Fixed chunks: 900 Unicode characters window, overlap 120 (240 for long
paragraphs), recipe `own-conversation-fixed900-overlap120-long240-v1`. Chunk keys
and the embedding cache include source ids, text hashes, recipe, scope and the
embedding-model identity. Retrieval uses BGE-M3 vectors; lexical overlap helps
exact names and rare words. Catalog topics are navigation hints only — they do not
rank and do not restrict full-archive search.

Owner signature: `retrieve_conversation_context(store, runtime, learner_id,
actor_id, query, exclude_source_id=None, budget_bytes=4000, time_filter=None,
fusion="legacy")` (conversation_retrieval.py). The API passes a server-side
callback into the RAG service after the resolver's `personal_memory_recall`
decision.

## 6. Unified personal memory + graph channel (November 14)

`personal_memory` is the single product method. In `hybrid` retrieval mode
(`rank_source_candidates`, round-robin of vector and lexical orders) the system
automatically adds a **graph channel** as a third round-robin input when ALL of
the following hold: `LOCAL_MEMORY_GRAPH_CHANNEL=1` (default), a ready graph index
matches the exact snapshot fingerprint (`find_ready_personal_graph_index`), Neo4j
is reachable on bolt://127.0.0.1:17687 and the password file exists. Otherwise the
channel degrades with a reason recorded in `retrieval_diagnostics.graph_channel`
and retrieval is plain hybrid.

The channel is **consume-only**: it never builds, claims or invalidates an index.
Indexes are built by the explicit teacher job `method="personal_graph"`
(experimental flag, 16k/32k context, Neo4j + graphiti-core). Graph facts are
navigation only — they re-rank canonical source replies and never enter the answer
text. See ADR-001.

## 7. Approval cycle (fail-closed)

Teacher flow: `POST /learners/{id}/next-step {method:"personal_memory"}` → job →
`MemoryWorker.run_once` builds the draft version → teacher reviews and
`POST .../memory/versions/{vid}/confirm` → student chat validates the approved
version (`approved_personal_memory_version`: protocol, archive fingerprint,
revisions, recipe fingerprint, correction closure) → the approved context is
inserted as `<approved_memory_untrusted>` in the SYSTEM prompt.

The recipe fingerprint binds the SHA-256 of the implementation files, model
digests, context size and protocol — any code or model change stales existing
approvals (one re-confirmation). See ADR-002.

## 8. Context floor and budgets

`LOCAL_CONTEXT_TOKENS` must be ≥16384 for the unified memory chat path (the
complete-memory resolver prompt exceeds the 8k byte budget) and exactly
16384/32768 for the graph method. Byte budgets: 10000/12000 at 16k, 20000/24000
at 32k. Ollama ≥0.40 silently truncates oversized prompts when `truncate` is not
disabled, and cuts them to roughly half the requested context even when disabled —
product prompts stay safe because byte budgets cap them far below that threshold;
the strict no-truncation probe (`verify_no_truncation`) keeps the experimental
graph path fail-closed. See ADR-003.

## 9. Teacher diagnostics

`GET /learners/{id}/memory/diagnostics/index` — settings flags, budgets and the
read-only graph-index status (registry lookup only, Neo4j never contacted).
`GET /learners/{id}/memory/diagnostics/retrieval?q=&mode=&budget=` — the same
retrieval the approved chat would run: channels, scores, selection warnings,
graph-channel reason and the selected context payload. Both are teacher-only,
read-only, and perform no generation and no graph writes. See ADR-004.

## 10. Verification matrix (current)

- New PG integration tests: `tests/test_hybrid_retrieval_postgres.py` (hybrid on
  real pgvector, cache, learner isolation, graceful degrade),
  `tests/test_memory_worker_chain_postgres.py` (job → draft → approval → chat →
  correction invalidation → 409 → re-approval).
- Diagnostics: `tests/test_memory_diagnostics_api.py`,
  `tests/test_memory_diagnostics_ui.py` (Chromium render, student 403).
- Live marker: `tests/test_memory_live_ollama.py` (real embed/chat/personal_memory
  cycle) and `tests/test_personal_graph_live_postgres.py` (real Neo4j graph cycle;
  skips without graphiti-core/Neo4j). Run with `pytest -m live` on a disposable DB.
- Doctor: `scripts/memory_doctor.py --check --live --e2e --graph`.

Synthetic-data results are implementation checks, never quality claims about real
children.
