# Unified application – lesson workflows and final integration (client B)

The Alena dev integration uses `theycomeunstuck/Alena_audio:dev` at
`843ca30` as its base and the unified-lesson and memory code from `f5792e3`.
Use the previous-lesson continuity check below to verify this branch.
The historical merge commits and run counts below refer to the original
candidate.

Status: integrated. The peer's ready commit `7fbace3` (unified memory
backend) is merged as `c687d2f`; integration corrections, disposable
PostgreSQL verification, the live model smoke and the browser check are
complete. This document describes the exact launch/setup, the changed
API/behavior, and the smoke reproduction. English documentation, Russian
product labels, short dash `–`.

## 1. What was delivered in the independent phase

The lesson capabilities of `D3strovich/child-lesson-agent` (commit
`43e6e36`) are ported into the existing TeachCopilot web application. The
upstream behavior is preserved; the SQLite/Streamlit product and the LM
Studio/OpenAI client are not transplanted. All model access goes through the
existing local Ollama/Gemma runtime and its admission, journal and lease
machinery.

| Upstream behavior | Where it lives now |
| --- | --- |
| Lesson prompt, exact section contract (`# Урок:` + six `##` sections) | `local_app/lesson_agent/prompts.py` |
| generate → validate → one corrective attempt (attempts < 2) | `local_app/lesson_agent/generation.py` |
| Validation: sections, praise-before-answer, arithmetic equality, multiplication practice contract (3 tasks, examples, no division story tasks, answers present) | `local_app/lesson_agent/validation.py` |
| Guided chat: one question at a time, no answers before the attempt, lesson/profile as untrusted data, last-12 bounded merged history | `local_app/lesson_agent/guided.py`, `prompts.py` |
| Summary presentation with the four fixed headings (`Что изучали`, `Что получилось`, `Что стоит повторить`, `Следующий шаг`) | `local_app/lesson_agent/summary.py`, `prompts.py` |
| Teacher-only answer section | `student_lesson_content`, facade student projection |

New integration surface owned by B:

- `local_app/unified_app.py` – lazy facade resolution, honest 503 before the
  facade is merged, contract error mapping (`UnifiedMemoryConflict` → 409,
  `UnifiedMemoryUnavailable` → 503, `UnifiedMemoryValidationError` → 422,
  `PermissionError` → 403, `LookupError` → 404).
- `local_app/lesson_agent/` – the ported workflows (above).
- API endpoints (section 3), guided-lesson chat selection, worker dispatch
  for `kind="lesson_generate"` and `kind="session_process"`, web UI panels.

## 2. Launch and setup

`server.py` only exports the ASGI application and does not start a web
server. Start Uvicorn explicitly, then the worker in a second process:

```powershell
cd TeachCopilot_RAG
uv sync --locked
# .local.env carries DATABASE_URL (disposable smoke DBs are created from it)
uv run --locked uvicorn local_app.api:app --host 127.0.0.1 --port 8000
# second process – the durable memory queue (lesson_generate / session_process)
uv run --locked python scripts/memory_worker.py --idle-seconds 1
```

- A true restart is a new OS process against the same disposable database:
  stop the Uvicorn process (Ctrl+C), start it again with the same
  `DATABASE_URL`, and verify persisted session/task state resumes –
  `scripts/unified_app_smoke.py` does this on a disposable database.
- PostgreSQL is canonical; migrations come from the peer's ready commit
  (migration `025_unified_memory_backend.sql` at round-1 baseline, plus the
  peer's round-2 additive migrations from 027 up when merged). B ships no
  migrations of its own and never migrates the working database.
- No new runtime dependencies were added; `pyproject.toml` and `uv.lock` are
  unchanged in the independent phase.
- Audio UI, images and the nine-method research run stay out of scope.

## 3. Changed API

All new endpoints answer `503` with an explicit detail while
`local_app.unified_memory` is absent; none of them fabricate a success.

| Method and path | Who | Behavior |
| --- | --- | --- |
| `POST /learners/{id}/lessons` | teacher | Enqueues `lesson_generate` (payload `topic_id`, `minutes`, `client_request_id`); duplicate idempotency keys with a changed payload → 409; minutes validated 5–90 |
| `GET /learners/{id}/lessons` | teacher | Optional facade `list_lessons(learner_id, actor_id)`; 503 when the facade does not provide listing |
| `GET /learners/{id}/lessons/{lesson_id}?audience=teacher|student` | both | Default audience from the role; the facade overrides by actual permissions. Student projection excludes answer keys |
| `POST /learners/{id}/lessons/{lesson_id}/confirm` | teacher | `{expected_revision, request_id}`; CAS revision check |
| `POST /learners/{id}/lessons/{lesson_id}/attach` | teacher | `{learning_session_id, request_id}`; the session must exist for this learner and the lesson must be approved. Kept for explicitly selected authorized sessions; not the delivery mechanism |
| `POST /learners/{id}/lessons/{lesson_id}/assignments` | teacher | Round-2 P0.2: `{expected_lesson_revision, request_id}`; facade `assign_lesson`, teacher-only, idempotent per payload, revision conflict → 409. The student needs no session yet |
| `GET /learners/{id}/lesson-assignments` | both | Round-2 P0.2: facade `list_lesson_assignments`; teachers see the learner's assignments, linked students only what they may use; no answer keys |
| `POST /learners/{id}/lesson-assignments/{aid}/start` | student | Round-2 P0.2: facade `start_lesson_assignment`; creates the student-owned standard session, attaches the approved revision, marks started; repeats return the same session |
| `GET /learners/{id}/lessons/{lesson_id}/tasks?audience=teacher|student` | both | Round-2 P1.1: facade `lesson_tasks` catalog; answers stay teacher-only; 503 while the backend provides no catalog |
| `GET/POST /learners/{id}/learning-sessions/{sid}/lesson-state` | both | Facade `lesson_state` / `update_lesson_state` (CAS `expected_revision`) |
| `POST /learners/{id}/learning-sessions/{sid}/process` | teacher or owning student | Enqueues `session_process` (`learning_session_id`, `client_request_id`); active sessions are teacher-only per the contract |
| `GET /learners/{id}/learning-sessions/{sid}/processing-result` | both | Latest authorized artifact or explicit pending/unavailable (503) |
| `GET /learners/{id}/memory-changes` | teacher | Facade `list_changes` (revision + items with provenance) |
| `POST /learners/{id}/memory-changes/review` | teacher | `{expected_revision, actions:[{id, action}], request_id}`; actions limited to `accept`/`reject`/`revert` |

Chat behavior changes (`POST /v1/chat/completions`):

- `metadata.lesson_mode == "guided"` opts into the guided-lesson flow. The
  lesson, profile and history always come from server state (facade lesson
  state + stored session messages); a browser cannot inject them.
- Without an attached lesson the guided request answers `409`; without the
  facade `503`; without a learning session `409`.
- The guided prompt receives only the student-audience lesson text (answer
  keys never enter a student prompt) plus a bounded fast-memory block
  (goals/unfinished step, ≤1500 chars).
- The model ends each guided reply with a `[Этап: <название>]` marker; the
  server parses it defensively, strips it from the child-visible text, and
  records the stage through `update_lesson_state` with a compare-and-set
  revision. A missing/invalid marker never advances the stage and never
  fails the durable reply.
- The ordinary chat contract, `/task/design`, autonomous/guided division and
  daily check-ins are unchanged.

Worker behavior changes (`MemoryWorker`):

- Claims `lesson_generate`/`session_process` jobs and routes them to the
  facade with `model_call=self._call`, so the response journal, lease
  heartbeat before every model call and generation admission stay effective.
- `lesson_generate`: `fast_memory` → bounded profile plus the previous lesson's
  recommendation → ported workflow → `save_lesson` (answers split defensively
  from the draft). With personal memory enabled, the worker adds the completion
  date, topic, first next step, optional review focus and exact summary-version
  reference for the learner's latest completed session with an attached lesson.
  The date uses the session's event time zone, with the learner's learning time
  zone as the fallback for older sessions. The block is omitted while that session lacks a
  final summary or if its summary has no recommendation. The recommendation is
  untrusted planning context and leaves the profile and confirmed observations
  untouched. The job result records `previous_lesson_summary_ref` separately
  from the fast-memory revision. Validation
  failures after the corrective attempt fail the job with the problems in the
  error; nothing is stored.
- New model purposes (`lesson_generate`, `unified_personal_extract`,
  `unified_session_summary_chunk`, `unified_session_summary_reduce`) use
  2048 output tokens and keep the whole-input context guard; no silent
  truncation, no hidden retries.
- Without the facade, unified jobs fail with
  `unified memory backend is unavailable`; no fake completion.
- On learning-session completion the API enqueues `session_process` once
  (idempotency key `session-process:<sid>:on-complete`) when a lesson is
  attached; a failed enqueue never undoes the durable completion and is
  reported as `session_processing_status: enqueue_failed`.

## 4. Web UI

- Teacher: «Полный урок» panel – topic + minutes → job status → draft with
  validation results → «Подтвердить урок» → «Прикрепить к занятию ученика»
  (creates the session and attaches). «Изменения памяти» panel lists
  personal changes (accept/reject/rollback) and academic candidates
  (accept/reject). Processing summary/provenance panel.
- Student: «Вести подготовленный урок» toggle (visible only with an attached
  lesson; the stage is shown from server state), «Итоги занятия» panel
  reads `processing-result` for the current session.
- Existing teacher approval, correction handling, memory-disable
  (`LOCAL_PERSONAL_MEMORY_ENABLED=0`), authorization and lesson-proposal
  flows are untouched.

## 5. Smoke reproduction

```powershell
cd TeachCopilot_RAG
uv run --locked python scripts/unified_app_smoke.py            # scripted model, no network
uv run --locked python scripts/unified_app_smoke.py --live     # real Ollama/Gemma + BGE-M3
uv run --locked python scripts/unified_app_browser_check.py    # real uvicorn + Chromium
```

- The smoke creates `teachcopilot_b_smoke_<12 hex>` on the configured
  loopback PostgreSQL server, migrates it, imports the synthetic demo
  learner, runs the scenario and drops the database. `--live` uses the
  configured local Ollama models (no duplicate servers, no downloads);
  `--keep-database` keeps the DB for inspection. `--output` writes the JSON
  report.
- Round-2 smoke checks cover the mandatory scenario: generation →
  validation → confirm → assignment while the student has no session
  (replay idempotent, wrong revision → 409) → student lists and starts it
  into their own session (owner check, replay returns the same session) →
  guided chat (marker hidden, stage recorded with CAS) → a verified
  arithmetic attempt completes the current catalog task and moves the
  pending pointer → «хз» stays unfinished and preserved verbatim → genuine
  restart with new application/store instances on the same database →
  active session, task progress and messages restored without duplicates →
  ordinary no-lesson chat completes and processes once → lesson-session
  completion processing with a four-heading summary grounded in the
  student's attempt and source refs → memory-changes list → own-archive
  recall finds the processed session chunk → the next lesson is generated
  from effective fast memory (diagnostics prove the records reached the
  prompt) → the old attach endpoint stays compatible → second-child
  isolation (403).
- Pre-merge the smoke honestly reported `facade_missing` (exit 1); a green
  smoke was impossible before the facade existed.
- The browser check runs the real application in two separate Chromium
  contexts: the teacher creates the lesson through the real queue, confirms
  it and assigns it («Задание выдано») without any teacher-owned session;
  the student context sees «Задания от учителя», starts it into its own
  session and receives the guided-lesson state. The uvicorn process is then
  actually stopped and a new process starts on the same database; the
  student context reloads, sees the persisted lesson state and a
  «Продолжить занятие» action. Screenshots are saved under `.local/`.

### Previous-lesson continuity check

Run from `TeachCopilot_RAG` in PowerShell. The test uses the configured loopback
PostgreSQL credentials to create and drop a fresh database with synthetic learners.

```powershell
$env:TEACHCOPILOT_TEST_DSN = uv run --locked python -c "from local_app.config import Settings; print(Settings.from_env().database_url)"
uv run --locked python -m pytest -q tests/test_lesson_summary_continuity.py -m 'not live'

$env:OLLAMA_URL = 'http://127.0.0.1:11434'  # Set to the local Ollama instance with both models.
$env:LOCAL_CHAT_MODEL = 'gemma4:12b'
$env:LOCAL_EMBEDDING_MODEL = 'bge-m3'
$env:LOCAL_CONTEXT_TOKENS = '32768'
uv run --locked python -m pytest -q tests/test_lesson_summary_continuity.py -m live
```

The scripted case uses a 16k window. Both modes confirm, assign, start and
complete two lessons about division. The first student reply loses a zero and
requests more practice. The assertions check the actual second generation
request, teacher access, two separate immutable summary versions, repeat
completion, and delivery of the first lesson's verbatim reply to ordinary
chat in a fresh OS process. A second learner exercises both generation and
archive isolation. To run the same live checks on Qwen, set `LOCAL_CHAT_MODEL`
to the exact Qwen 3.5 tag served by the loopback Ollama instance.

## 6. Tests

- `tests/test_lesson_agent_workflows.py` (30): ported validation, the
  corrective-attempt workflow, marker parsing/history handling, summary
  contract.
- `tests/test_unified_app_api.py`: endpoints against a contract fake
  facade (idempotency/conflict 409, audience override, teacher-only
  confirm/review, attach rules, CAS lesson state, processing gating, guided
  selection via server state, completion auto-enqueue once – attached lesson
  or not – teacher-completion of the owner session, fast-memory wiring with
  the real record shape and its degradation).
- `tests/test_unified_app_worker.py`: worker dispatch through the real
  `model_call` hook (journal/heartbeat/admission), failed-validation job,
  honest failure when the backend is absent, effective-records prompt proof.
- Round-2 additions (`tests/test_r2_app_*`):
  `test_r2_app_generation_memory` – P0.1 projection of the real facade
  records into the generation prompt, no card fallback, whole-unit budget
  diagnostics; `test_r2_app_assignments` – P0.2 assignment lifecycle,
  roles, isolation, 503-before-facade-support; `test_r2_app_task_progress`
  – P1.1 verified attempts, «хз»/marker/silence never complete, original
  wording in the guided prompt; `test_r2_app_creative_routing` – P0.4
  bounded creative route (2048 tokens, archive retrieval, fiction
  separation, brief greetings, legacy compatibility);
  `test_r2_app_failure_handling` – P1.2 queue-full honesty, lease takeover,
  malformed output with journals, repeated submissions;
  `test_r2_app_quality_runner` – P1.3 record-before-score, gold-leak scan,
  final-check preservation.
- PostgreSQL-backed suites on disposable databases (merged tree):
  `test_unified_memory_postgres` + `test_memory_worker_chain_postgres` +
  `test_memory_api_flow` + `test_hybrid_retrieval_postgres` 23 passed;
  with the hierarchical/history suites 30 passed.
- Full sweep on the merged tree: 2207 passed / 413 skipped; the only
  failures are the 6 pre-existing
  `tests/test_personal_graph_retrieval_benchmark.py` cases (identical on
  the baseline) and the pre-existing `tests/test_mem0_strict_v1.py` import
  error (both present before any of this work).

## 7. Integration corrections and remaining limitations

Post-ready review fixes (A-report.md section 7, findings F1–F7) applied in
the merged tree as integration corrections; the peer's frozen branch is
untouched. Regression tests: `tests/test_unified_app_review_fixes.py`
(11 tests; F1/F4/F5/F6 on disposable PostgreSQL).

- F1: migration `026_unified_processing_run_retry_fix.sql` drops the
  UNIQUE identity tuple on `unified_processing_runs`, so the documented
  failed-run retry (`<request_id>:r2`) inserts its own run and the failed
  artifact stays retrievable; request-id uniqueness plus the pre-insert
  lookup remain the replay guards.
- F2: the `remove` personal-change operation keeps its nonempty literal
  value (extraction validation, store validation and the INSERT), and the
  processing persist guards treat store-level integrity failures as
  per-change rejections instead of crashing the run.
- F3: a non-object JSON response for a summary chunk (or the reduce call)
  degrades that chunk with honest coverage instead of aborting the run.
- F4: `recall` without a configured runtime degrades to the lexical channel
  with `diagnostics.degraded` instead of raising AttributeError.
- F5: raw psycopg2 errors map to the contract (recall → 503; the lesson
  CAS wrappers distinguish a database outage (503) from a real revision
  conflict (409)).
- F6: accepting an academic candidate commits decision + canonical teacher
  note + durable job in ONE transaction; a queue-full failure rolls the
  acceptance back (the candidate stays pending) and surfaces as 503.
- F7: unused imports removed.

Resolved during the merge phase:

- The queue `kind` widening and payload validation arrived with the peer's
  commit (migration 025); the endpoints now enqueue `lesson_generate` and
  `session_process` for real.
- `list_lessons` was delivered by the peer as requested and is wired into
  `GET /learners/{id}/lessons`.
- The lesson draft validation record now carries `status`/`passed` next to
  the workflow diagnostics, matching the backend's save contract.
- Ordinary chat degrades gracefully when the facade is unavailable: fast
  memory enrichment logs a warning and the approved-memory paths stay
  authoritative; the recall callback falls back to the shipped same-scope
  own-conversation retrieval (never a scope widening).

Remaining limitations (honest):

- Lesson `pending_task_id`/`completed_task_ids` stay null in this port: the
  model marker carries the stage only. Task-level identifiers need a
  lesson-task ID convention shared with the backend contract.
- The student-side history restore check accepts the completed-session
  endpoint contract (`409` for an active session); own-history readability
  is verified instead. A dedicated student "continue lesson" restore view
  reuses the existing learning-session restore flow.
- The live smoke exercises one bounded scenario with synthetic input; it is
  an integration check, not a quality evaluation. Recall vector quality at
  the 0.20 floor and extraction/summary quality with real Gemma need
  separate evaluation before any claim about real children.
- Audio UI, images and the nine-method research run stay out of scope.
