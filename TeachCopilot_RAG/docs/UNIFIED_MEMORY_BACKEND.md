# Unified memory backend (client A, 2026-11-17)

Status: implemented and verified on a disposable PostgreSQL database with
recording/mock models. Live Gemma checks are owned by client B after
integration. This document describes what is actually implemented, the exact
interface B consumes, the migrations, the tested behavior and the limitations.
Round 2 (branch `zcode/r2-memory-20261117`, base `95bec61`) adds the sections
"Round 2 additions" below; sections 1–8 describe the round-1 facade that
remains compatible.

## 1. Public facade

`local_app.unified_memory.UnifiedMemoryService(store, runtime, *, model_call=None)`
accepts the existing `Store` and runtime objects. No model or database work
happens at import time. All identifiers are strings, all payloads are
JSON-serializable dictionaries, and optional keyword fields never become
positional.

Errors exported from `local_app.unified_memory`:

| Error | HTTP mapping | Raised for |
| --- | --- | --- |
| `UnifiedMemoryConflict` | 409 | revision CAS mismatch, request id reused with changed payload, duplicate idempotency keys |
| `UnifiedMemoryUnavailable` | 503 | model/embedding backend unavailable, missing immutable model identity |
| `UnifiedMemoryValidationError` | 422 | contract violations (bad stage, unvalidated draft, malformed actions, bad request ids) |
| `PermissionError` | 403 | actor has no access, teacher-only operation, active-session processing by a student |
| `LookupError` (incl. `KeyError`) | 404 | unknown lesson/session/learner/change id |

The service validates actor access at every boundary using the store's own
authorization data (`accounts.role`, `account_learners`); a learner id supplied
by a client is never trusted as authorization.

## 2. Core methods

- `fast_memory(learner_id, actor_id, *, session_id=None)` – returns
  `method="unified_fast_memory"`, `learner_id`, an opaque `revision` string
  (`um-<sha256 prefix>` over learner/source/changes/lesson-state counters),
  `personal` (`interests`, `avoided_interests`, `preferences`), `learning`
  (`goals`, `confirmed_observations`, `errors`, `help`, `unfinished`),
  `session` and `source_refs`. No LLM/embedding calls and no full-history
  scan: it reads the learner card, the personal-change projection and at most
  one lesson-state row. Collection items carry origin (`profile` or
  `conversation`) and source refs; unavailable fields are empty, never
  fabricated. Applying or reverting a personal change moves the opaque
  revision without touching the learner card revision, so unchanged academic
  approvals are not invalidated.
- `recall(learner_id, actor_id, query, *, session_id=None, budget_bytes=4000,
  time_filter=None)` – returns `method="conversation_retrieval"` with `text`,
  `system_text`, `sources`, `source_ids`, `evidence_ids` and `diagnostics`,
  compatible with the shipped conversation retrieval shape. Sources keep role
  (`author_role`), `activity_day`, `learning_session_id`, `corrects_source_id`
  when present, and delivery status on user quotes. No match returns empty
  evidence. Summaries never expand access; the search covers only the actor's
  own authorized archive.

### Lesson methods

- `save_lesson(...)` (teacher-only) – stores a validated draft only
  (`validation.status in {"passed","valid"}` or `passed: true`). Answer
  sections are located by defensive Markdown heading parsing at save time
  (`unified_lessons.answer_sections`), never from a client-provided split.
  Idempotent per `(learner_id, request_id)`; a different payload under the
  same request id is a 409.
- `get_lesson(..., audience)` – `audience` is `teacher` or `student`; the
  actor's actual permissions override the requested audience (a student always
  receives the student projection). The student projection redacts
  `answers_text` and the parsed answer sections and reduces `validation` to
  its status.
- `confirm_lesson(...)` (teacher-only) – compare-and-set on
  `expected_revision`; replay of an already approved lesson with the matching
  revision returns `confirmed=False` (idempotent).
- `list_lessons(learner_id, actor_id)` – optional listing requested by B:
  lesson metadata only (no lesson or answer text). Teachers see every version;
  students see approved lessons only.
- `attach_lesson(...)` (teacher-only) – requires an approved lesson and an
  existing session of the same learner; re-attach with a different lesson
  restarts the state at `not_started` with a revision bump.
- `lesson_state(...)` / `update_lesson_state(...)` – state holds
  `session_id`, `lesson_id`, `revision`, `stage`
  (`not_started|warmup|explanation|practice|check|completed`),
  `pending_task_id`, `last_question_source_id`, `completed_task_ids`; unknown
  identifiers are null. Updates use compare-and-set revisions; accepted
  requests replay idempotently via the stored request id. No hidden chain of
  thought is stored.

### Processing and review

- `process_session(learner_id, actor_id, session_id, *, request_id)` –
  worker-facing; B must not call it inside an HTTP transaction. On an active
  session it is teacher-only and processes a provisional snapshot without
  finishing the session; a completed session may be processed by the session's
  actor or a teacher. Run identity is
  `(learner_id, session_id, source_fingerprint, prompt_recipe, model_identity)`;
  the exact same identity replays the persisted run, a changed snapshot under
  the same request id is rejected. The snapshot is the ordered session
  transcript plus the full correction chain attached to its sources
  (correction-safe closure). Status is `completed`, `partial` or `failed`;
  chunk failures and extraction failures degrade independently, each becomes
  a warning plus a coverage entry, and a failed run is retried under a fresh
  request id (`<request_id>:r2`, …) while the failed artifact stays
  retrievable. A model failure inside one chunk never deletes stored sources
  or an earlier valid state, and no unknown-outcome model call is retried
  automatically.
- Summary text uses the four lesson-agent headings: `Что изучали`,
  `Что получилось`, `Что стоит повторить`, `Следующий шаг`. Coverage records
  every input chunk id plus `included_chunk_ids`, `failed_chunk_ids` and
  `extraction_failed_chunk_ids`; a summary built with failed chunks is marked
  `provisional`. Summary versions are immutable and retrievable per session
  (`store.list_unified_summary_versions`).
- Academic candidates are stored `pending` with `candidate_id`, `category`
  (`knowledge_observation|error_pattern|learning_goal`), `proposed_value`,
  `source_refs`. The validator discards any proposal containing numeric
  fields, so the model can never author a mastery delta. Accepting a candidate
  records the teacher decision and applies it through the existing canonical
  teacher-note path: an approved teacher memory source
  (`source_key="unified-candidate:<id>"`) plus its durable `memory_update`
  job. Numbers are never written anywhere.
- Personal changes are append-only events (`unified_personal_changes`) with a
  current projection (`store.project_personal_changes`). Only
  `explicit_durable=true` durable facts apply automatically and only with
  literal evidence: the validator requires the quoted `evidence_text` to be a
  verbatim substring of the referenced learner-scoped source. Session-scoped
  claims are never applied; non-explicit durable claims stay `proposed` for
  teacher review. Teacher `revert` appends a reverting event and never
  rewrites history.
- `processing_result(...)` returns the latest authorized artifact or an
  explicit `pending` (queued/running job exists), `failed` or `unavailable`
  state; a failed run never masquerades as a valid summary.
- `list_changes(...)` (teacher-only) returns the current `revision` and items
  with provenance and `eligible_actions` (`accept/reject` for pending items,
  `revert` for applied durable personal changes). `review_changes(...)` is
  teacher-only, validates every `{id, action}`, enforces the category/action
  eligibility rules, uses compare-and-set on the revision from `list_changes`
  and replays idempotently for already-decided items.

## 3. Model interface

The optional `model_call(messages, purpose, schema)` hook returns model
response text. Purposes: `unified_personal_extract`,
`session_summary_chunk_v1`-bound `unified_session_summary_chunk` and
`unified_session_summary_reduce`. `model_call.model_identity()` (or the
runtime's `model_identity(settings.chat_model)`) supplies the immutable
identity required for run identity; without either, processing raises
`UnifiedMemoryUnavailable`. Every model call is a single attempt; failures
degrade per component. Without a hook, the runtime's `chat_json` (local
admission path) is used. B owns worker dispatch for `lesson_generate` and
`session_process` and updates `_call` budgets without bypassing context
checks.

## 4. Migrations (025_unified_memory_backend.sql)

Additive only; the working learner database is never used for testing.

- `memory_jobs`: `kind` now also accepts `lesson_generate` and
  `session_process`; `method='personal_memory'` is allowed for
  `next_step` and `session_process` (provider `local`). The structured_v2 and
  personal_graph restrictions keep their original scope.
- `unified_lessons` – durable lesson versions with `revision`, draft/approved
  status, parsed answer sections and request-id idempotency.
- `unified_lesson_states` – one attached lesson per session with CAS revision.
- `unified_personal_changes` – append-only personal change events with
  `scope`, `explicit_durable`, evidence, source refs and `reverts_id`.
- `unified_academic_candidates` – pending teacher-review proposals.
- `unified_processing_runs` – one artifact per exact snapshot/recipe/identity
  with summary, coverage and warnings.
- `unified_summary_versions` – immutable, versioned summaries per session.
- `unified_conversation_chunks` – persisted own-conversation search
  representations: chunk text, source ids/spans, activity days,
  `search_document` tsvector (russian||simple), optional 1024-dim pgvector
  embedding with its model identity, GIN + HNSW indexes.
- `learners.unified_changes_revision` – coordinator revision for review CAS.

## 5. Recall implementation

- Chunking reuses `chunk_conversation_messages` (fixed 900-char windows,
  paragraph-based overlap) over `read_own_messages`, which establishes
  ownership and preserves failed user deliveries. Chunks are persisted once
  per `(learner, actor, chunk_key, recipe)`; embeddings are stored on the same
  rows keyed by the embedding model identity, so no query re-embeds the
  archive.
- Search is one SQL statement over both channels with identical scope
  predicates (`learner_id`, `actor_id`, recipe, optional session and activity
  day range) before ranking: full-text via
  `plainto_tsquery('russian')||plainto_tsquery('simple')` with `ts_rank`, and
  pgvector cosine with the shipped engineering floor (similarity ≥ 0.20).
  Fusion is deterministic RRF with k=60; ties break by chunk key. The
  curriculum RAG algorithm is untouched.
- A query-derived date filter that yields no indexed hits retries bounded
  without the date; an explicit caller range is never widened. Embedding
  outage degrades to lexical-only with `diagnostics.degraded=true` and a
  reason. Cross-learner or cross-actor retrieval is never a fallback.
- `time_filter` reuses the shipped parser (explicit ISO from/through or
  relative Russian date phrases against the learning timezone).

## 6. Tested behavior

Unit (no DB), `tests/test_unified_memory_contract.py` – 15 passed: schema
strictness, four-heading rendering, deterministic reduce merge, fingerprint
order/content binding, explicit-durable vs proposed vs session-scope rules,
literal-evidence rejection, numeric-proposal discard, provenance projections,
defensive answer parsing/redaction, error types, prompt negative examples.

Disposable PostgreSQL, `tests/test_unified_memory_postgres.py` – 14 passed:
lesson save/get/confirm redaction roundtrip, attach/state CAS and conflicts,
fast memory projection/opaque revision/rollback, processing idempotency and
summary versioning, partial coverage on failed chunks, extraction rules end
to end (question + «хз» preserved verbatim, explicit durable interest,
pending candidate, failed extraction degrades to partial), teacher-only
processing of active sessions with provisional flag, two-learner isolation
for processing and recall, list/review/revert with canonical teacher-note
application, persisted RRF recall with no re-embedding per query, lexical
degradation with visible diagnostics, explicit time filter, and the widened
queue constraints including payload and duplicate-key rejection.

Regression: 113 unit-level and 30 PostgreSQL tests of the directly affected
existing suites (`test_local_store`, `test_conversation_retrieval*`,
`test_hierarchical_memory*`, `test_memory_api_flow`,
`test_hybrid_retrieval_postgres`, `test_memory_worker_chain_postgres`,
`test_session_summaries`) pass unchanged. Migration 025 applies once
(`schema_version` 25) and re-runs apply nothing.

Commands (from `TeachCopilot_RAG`, with `TEACHCOPILOT_TEST_DSN` pointing at a
disposable loopback PostgreSQL server):

```
uv run --locked python -m pytest tests/test_unified_memory_contract.py -q
uv run --locked python -m pytest tests/test_unified_memory_postgres.py -q
```

## 7. Integration notes for B

- Construct one `UnifiedMemoryService(store, runtime, model_call=...)` per
  worker/request; the recipe fingerprint is computed at construction from the
  prompt files.
- `process_session` is worker-facing: call it from the
  `session_process` job handler, never inside an HTTP transaction. Job
  payloads are documented in the shared contract; `enqueue_memory_job` now
  validates them.
- `model_call` should supply `model_identity()`; the worker should pass the
  runtime-based identity so run identity matches the live model.
- Error mapping: 409 `UnifiedMemoryConflict`, 503 `UnifiedMemoryUnavailable`,
  422 `UnifiedMemoryValidationError`, 403 `PermissionError`, 404
  `LookupError`.
- Fast memory is intentionally age-free; B must not require an exact age when
  only an age group is stored on the card.
- Live vector/recall quality and the live model end-to-end scenario remain
  pending for B after merge; A executed none of them.

## 8. Known limitations

- The vector floor (0.20) and RRF k=60 reuse existing engineering defaults;
  no held-out calibration was performed, and no quality percentage is claimed.
- Extraction/summary quality with the real Gemma model is unverified (mock
  and recording models only, per the parallel-work contract).
- `update` operation for personal changes is validated but the projection
  treats it as add-if-absent; deduplication relies on value equality.
- Summary chunk count grows linearly with transcript length; there is no
  cross-chunk map budget beyond the reduce call's `max_output_tokens`.
- Answer-section redaction covers Markdown headings matching
  ответ/answer/ключ/key/решение prefixes; prose-embedded answers are not
  detected.
- The service does not enqueue jobs itself; B owns completion-triggered and
  manual enqueueing (`lesson_generate`, `session_process`).

## 9. Round 2 additions (zcode/r2-memory-20261117, base 95bec61)

### 9.1 Effective personal projection (P0.1)

`fast_memory` no longer concatenates card values with change events. The
effective projection starts from the manual profile values and replays applied
durable changes in source order (`unified_memory_profile.replay_personal_changes`):

- keys are compared consistently (whitespace-collapsed, case-folded);
- a durable `remove` suppresses the matching original card value as well as
  conversation-added values; a later `add` restores it with new provenance;
- interests and avoided story themes remain separate categories: an avoidance
  never removes an interest;
- reverted events and their revert markers never replay; a revert restores the
  previous effective value without touching history;
- every conversation-origin item carries the literal evidence, the actual
  source ids of the applied change, its `change_id`, `event_at`, `scope`,
  `status` and `explicit_durable`; a repeated statement adds a `confirmations`
  entry to the authoritative manual item.

Store-side, `unified_personal_changes.event_seq` (migration 027) gives every
event a stable source-order key, so events sharing a transaction timestamp
replay in insertion order. Personal changes still touch only
`unified_changes_revision`; academic approvals stay untouched.

### 9.2 Pre-login lesson assignments (P0.2)

New facade operations (keyword-only options preserved):

- `assign_lesson(learner_id, actor_id, lesson_id, *, expected_lesson_revision, request_id)`
  – teacher-only; validates the exact approved lesson revision and is
  idempotent for the same payload; the assignment exists before any student
  session. Lesson content is never mutated to represent an assignment.
- `list_lesson_assignments(learner_id, actor_id)` – authorized `items`;
  teachers see the learner's assignments (with `assigned_by`/`started_by`
  provenance), linked students see the schedule facts without actor
  identities; no lesson or answer text is included.
- `start_lesson_assignment(learner_id, actor_id, assignment_id, *, request_id)`
  – called by the linked student (role `student`, relationship `student`). In
  ONE transaction it validates the assignment, re-validates the approved
  revision (conflict type on mismatch, never a silent substitute), creates a
  NEW standard session owned by that student, attaches the approved lesson and
  marks the assignment `started`. Repeats return the same persisted session; a
  different child gets a learner-scoped LookupError; completed assignments
  never create a fresh session. Returns the assignment fields plus the
  existing lesson-state projection under `lesson_state`.
- Completing the guided lesson state (`update_lesson_state` with
  `stage="completed"`) completes the corresponding assignment in the same
  transaction; the response carries `completed_assignment_id` when that
  happened.

Table `unified_lesson_assignments` (migration 027) freezes `lesson_revision`
at assignment time and keeps learner-consistent foreign keys
(`UNIQUE(learner_id, lesson_id, ...)`, FKs to `(id, learner_id)` of lessons and
sessions).

### 9.3 Deterministic task catalog (P1.1)

`unified_memory_tasks.build_task_catalog` derives the catalog once at lesson
save time from explicit numbered exercises and standalone questions in the
Разминка/Объяснение/Практика/Проверка sections. Task ids are
`task:<lesson_id>:<stage>:<position>` – qualified by the saved lesson version
and stable across reads, retries and re-attachments. The catalog is stored on
the lesson row (`unified_lessons.task_catalog`, migration 027); `NULL` marks a
lesson saved before the catalog existed.

- `lesson_tasks(learner_id, actor_id, lesson_id, *, audience)` returns
  `lesson_id`, `lesson_revision`, `catalog` (`available`/`unavailable` with an
  explicit reason) and `items` with `task_id`, `stage`, `position`,
  `prompt_text`, `section` and lesson-document source offsets/refs. Answers
  are never part of the catalog; a student actor receives the student
  projection of approved lessons only (draft lessons are a 403 for students).
- `update_lesson_state` validates `pending_task_id`/`completed_task_ids`
  against the attached lesson's catalog and `last_question_source_id` against
  an actual assistant source of this session; no foreign lesson's task or
  foreign session's question can enter the state. Lessons without a catalog
  stay permissive (explicit limitation).
- `rebuild_lesson_task_catalog(learner_id, actor_id, lesson_id, *, request_id)`
  is the explicit teacher/admin backfill; deterministic content is a no-op and
  no read path ever writes a catalog.

### 9.4 Processing recovery and concurrency (P1.2)

- Durable identity claim: `unified_processing_claims` (migration 027) serializes
  identical-snapshot processing. `acquire_processing_claim` is atomic
  (expired leases are taken over), `extend_processing_claim` supports worker
  heartbeats, and the claim is released inside the persist transaction. A busy
  claim surfaces as `UnifiedMemoryConflict` for the competing worker. There
  are no model calls inside any transaction.
- One-transaction persistence: `persist_unified_run` commits the run row, the
  summary version, every personal change and every academic candidate with
  per-artifact savepoints, then releases the claim. One rejected item or a
  failed summary write becomes a warning (`summary_write_failed`,
  `personal_change_rejected`, `academic_candidate_rejected`); a run whose
  summary write failed is downgraded to `partial`/provisional instead of
  claiming completion. Failed artifacts stay inspectable and retries keep
  using fresh request ids.
- Correction metadata: the session snapshot and the own-conversation reader
  carry `superseded_by`; the extraction/summary model payloads include a
  `corrections_untrusted` map and the prompts instruct that the correcting
  wording is current. Recall quotes from superseded sources carry
  `superseded_by`. Teacher corrections stay separate teacher-reviewed evidence
  and never enter an own-conversation archive.
- The lexical recall channel no longer ANDs all query words
  (`plainto_tsquery` over the whole query); it ranks by IDF-weighted matched
  words, so paraphrased questions with one absent word still hit. A
  query-derived date that yields no indexed hits now really retries without
  the date (the round-1 fallback kept filtering by it – fixed; the requested
  range stays in diagnostics).

### 9.5 Lesson-plan context (P2, optional)

`recall` results carry an optional `context_documents` list: approved lesson
plans of the same learner matched by full text (unindexed scan over the
learner's own lesson versions), bounded to 3 documents × 1200 characters with
`kind="lesson_plan"`, `lesson_id`, `lesson_revision`, topic/minutes and
`source_refs` of kind `lesson_document`. The student projection redacts the
stored answer spans exactly like `get_lesson`; a plan is context only, never
student evidence, and every document carries
`context_note="План занятия – только контекст; он не доказывает выполнение."`.
B renders the warning; the field never widens access. An unavailable plan
channel degrades to an empty list with `context_documents_degraded`.

### 9.6 Round-2 migrations and tests

Migration `027_unified_r2_assignments_tasks_claims.sql` is additive:
`unified_lesson_assignments`, `unified_lessons.task_catalog`,
`unified_processing_claims`, `unified_personal_changes.event_seq`
(backfilled in `(created_at, id)` order; new rows use a sequence starting at
1000000). It applies once and re-runs apply nothing.

New tests (all recording models, disposable PostgreSQL only):
`tests/test_r2_memory_contract.py` (pure replay/catalog specification),
`tests/test_r2_memory_regressions.py` (nine regressions; seven fail
meaningfully on the baseline when product files are stashed),
`tests/test_r2_memory_postgres.py` (assignments, catalog, claims, honesty,
P2), `tests/test_r2_memory_long_history.py` (frozen archives).

### 9.7 Frozen synthetic archives and gold (P1.3)

`tests/r2_memory_quality/` is the frozen package: three six-session histories
(`histories/r2-mem-a.json` – the oversized archive, at least 110000 characters,
above the 32768-token window at a conservative 3.2 characters/token;
`r2-mem-b.json`; `r2-mem-final.json` – the held-out final check), the
deterministic generator (`generate_histories.py`, seed 20261117), the loader
(`loader.py`) that inserts an archive through the real persistence invariants
and returns the immutable source-key mapping, and the frozen gold manifest
(`manifest.json`) with 18 questions (6 per history) covering all twelve
required case types. Gold answers and source keys are evaluation-side data
only and never enter generation, resolver or extraction prompts; the
evaluation protocol for B is written into the manifest.

### 9.8 Round-2 limitations

- Extraction/summary quality with the real model is still B-s live evaluation
  (A used recording models only).
- The task catalog treats numbered/question lines in Объяснение as
  explanation-stage prompts; a worked example numbered like an exercise
  becomes a task (conservative parse of explicit lines only, never invented).
- Lessons saved before migration 027 have no catalog; task-id validation is
  permissive for them until an explicit rebuild.
- The claim lease defaults to 1800 seconds; workers processing very long
  archives should heartbeat via `extend_processing_claim` or pass a longer
  lease.
- `update` personal-change semantics remain add-if-absent (round-1
  limitation, kept and documented).
- A synthetic archive is inserted by direct source rows because
  `memory_sources.created_at` is immutable by trigger; the loader mirrors the
  real persistence invariants rather than rewriting history.
