# Alena own-archive retrieval fixes – change guide for a reviewer (2026-11-12)

Audience: a human or a diff-review agent. Paths are relative to `Alena_audio/TeachCopilot_RAG` unless stated.
Everything is synthetic-data tested; nothing was run against real children's data or the working database.

## 1. How to see exactly what changed (read this first)

`git diff dev..alena-retrieval-fixes-20261112` is NOT a clean view of this work: the repository already had about 165
uncommitted entries from earlier (7 November) sessions, and several of my files (`api.py`, `rag.py`, `store.py`,
`memory_store.py`, `memory_service.py`, the new modules) contained that earlier work, so the branch also carries it.

To see ONLY the changes made on 2026-11-12:
- Use the patch `docs/changes/ALENA_RETRIEVAL_FIXES_2026_11_12.patch` (unified diff of every pre-existing file I touched,
  taken against a copy of the code made before the first edit). It covers 19 files, about 965 lines.
- Eight test files are entirely new and are not in the patch (they have no "before"): see section 3.
- Pre-fix copy, if available on this machine: `C:\Users\name\Documents\ChatGPT\Claude\teachcopilot-work\snapshots\alena-20261112-pre-fix\`
  (with `MANIFEST.json` of SHA-256 hashes). Per file: `diff -u snapshots/alena-20261112-pre-fix/<path> <path>`.

Git: branch `alena-retrieval-fixes-20261112` (commits `1c1b671`, then a handoff-text follow-up, then this guide) was cut from
`dev` (`9eb9ad3`) with a temporary index, so `dev`, the index and the working tree were not modified. Not pushed.
It lists whole files and does not include migrations 015, 021, 023, so it is not self-contained. Drop it with
`git branch -D alena-retrieval-fixes-20261112`.

## 2. Change map (existing files)

| File | What changed | Why | Review focus |
|---|---|---|---|
| `local_app/conversation_retrieval.py` (+230/-24, a module that was new and untracked) | (a) whole-word month table, impossible ISO dates ignored, date taken from the question falls back to the full archive when it matches nothing (explicit ranges never widened); (b) embedding outage -> word-overlap fallback with `diagnostics.degraded`; (c) `ConversationEmbedder.source_vectors` isolates a text the model cannot embed (returns `None` for it) after a batch failure, aborts as an outage after 3 failed singles; (d) cache key no longer hashes the module source (`CACHE_FORMAT_VERSION`); (e) `pack()` escapes `<`/`>` as `\u003c`/`\u003e`; (f) opt-in `fusion="rrf"` (RRF k=60 over cosine and stemmed BM25) and opt-in `absence_rule="provisional-v1"` (abstain if best cosine < 0.47), both OFF; (g) always-on observational `diagnostics.absence_signal`. Default scoring is unchanged (`legacy`: cosine + 0.04 x exact-term fraction, 0.20 gate, top 12). | FIX_LIST items 2,3,4,5,10,11,1 | exception types caught in (b)/(c) (`RuntimeError`, `OSError`, DB errors; `ValueError` and `PermissionError` must still propagate); `None` vectors handled in the scoring loop; first request after deploy re-embeds each learner once (cache key change) |
| `local_app/dialogue_catalog.py` (+9/-1) | a chunk with `None` vector is classified "unknown, embedding_unavailable"; an unembeddable prototype raises `ConversationRetrievalUnavailable` | follows from (c) | catalog persistence of the new `embedding_unavailable` key and empty `topic_scores` |
| `local_app/hierarchical_memory.py` (+14/-3) | `_wire()` escapes `<`/`>` when assembling the summary block `<own_conversation_untrusted>` | same prompt-wrapper hole as (e) | applied at assembly so stored immutable summary versions are covered; byte budget counted after escaping |
| `local_app/memory_store.py` (+7/-4) | `get_own_conversation_messages`: each UNION branch returns the stored owner (`s.learner_id`/`s.actor_id`; assistant rows use the session owner `m.actor_id`; check-ins `a.learner_id`/`c.actor_id`); final select is `SELECT * FROM eligible` | the Python owner check in `read_own_messages` compared request values with themselves (the SQL echoed the parameters) | UNION column order/types (all `text`), assistant rows must carry the session owner, not `'assistant'` |
| `local_app/personal_graph_memory.py` (+13/-78) | `source_scoped_search` no longer has a private hybrid: every mode delegates to `rank_source_candidates` | FIX_LIST item 6 (merge approved) | BEHAVIOUR CHANGE: ties are broken newest-first (was turn id); tokenising and cosine follow `rank_source_candidates`; embeddings must be 1024-dimensional and non-zero |
| `local_app/rag.py` (+14/-3) | `TOPIC_RANK_BOOST` 1.15 -> 1.03, `GRADE_RANK_BOOST` 1.05 -> 1.01; rows with `text_score > 0` survive the cosine floor | a x1.15 boost lifted a tenth-ranked material over the first | BEHAVIOUR CHANGE in curriculum-material ranking (accepted by the owner) |
| `local_app/store.py` (+5/-4) | `id` as the last ORDER BY key in `hybrid_search` (both branches) | deterministic ties | none beyond SQL syntax; exercised by `test_local_store` hybrid tests |
| `local_app/api.py` (+10/-2) | `/learners/{id}/learning-sessions/current` and session history no longer return `source_id` per message (`_without_internal_source_ids`) | privacy: internal source ids must not reach the learner client; `test_guided_restart` pins it | the web UI does not read `source_id` from these endpoints (it uses it in the own-conversation view, unchanged) |
| `local_app/memory_service.py` (+31/0) | `MemoryWorker.run_once` retries a background local job that failed only as `stale_revision` as explicit new jobs, max `STALE_RETRY_LIMIT = 2`; chat jobs and external providers never retried; the old body is `_run_once_single` | direct chats are persisted and move the source revision, which failed running jobs | each retry is its own job row (idempotency key `stale-retry:<job id>`); no hidden repeated call inside one job; only the retry itself is followed |
| tests (9 existing files, small) | test doubles updated, no production behaviour pinned weaker: fake model replies gained `"done": true` (`test_social_chat`, `test_chat_response_journal_api`, `test_memory_generation`, `test_generation_run_budget`); fake resolver returns `personal_memory_recall` (`test_guided_division_api`); fake embeddings padded to 1024 dims and two budget tests expect newest-first survivors (`test_personal_graph_bridge`); `test_local_store` uses unique source names per run; message-dict helpers ignore the four time fields (`test_current_learning_session_api`, `test_learning_session_archive_api`) and archive list rows accept `created_at`/`event_timezone`; legacy-chat journal row now has a server-side session | stale expectations after the 7 November work | confirm none of these edits hides a real defect (the `source_id` shortcut was reverted: `source_id` is NOT ignored) |
| `docs/ALENA_HANDOFF_2026_11_07.md` (+20) | dated addendum (Russian) with results | handoff | wording only |

## 3. New test files (no "before")

`test_conversation_retrieval_hardening.py` (51), `test_summary_wrapper_hardening.py` (3), `test_rag_search_ranking_hardening.py` (8),
`test_own_messages_owner_columns.py` (3), `test_conversation_embedder_cache_postgres.py` (2, needs `TEACHCOPILOT_TEST_DSN`),
`test_personal_graph_round_robin_shared.py` (3), `test_source_scoped_search_unified.py` (6), `test_memory_stale_retry.py` (11).
Each was written before its fix and failed first (the Postgres cache test fails on the old code too).

## 4. Things a reviewer should challenge

1. `conversation_retrieval.py`: can the fallback in (b) hide a real configuration error? (`ValueError` is not caught by design.)
2. The 0.47 absence threshold is read off the same 14 synthetic archives it was measured on and loses roughly 30-56% of answerable questions. It is off by default; confirm nothing passes it.
3. The unified approved-memory search: confirm production embeddings are always 1024-dimensional (BGE-M3) and that no caller passes shorter vectors.
4. `stale_revision` retries: confirm a retry cannot loop (limit 2 via payload counter) and that it never applies to `chat` jobs.
5. `source_id` removal: grep the web client for any other reader of `source_id` on `/current` or session history.
6. `_without_internal_source_ids` removes only `source_id` from each message; check whether other internal message fields (for example ordering or provenance keys) should not reach the learner either.
7. `LOCAL_PERSONAL_MEMORY_ENABLED=0` still does NOT switch off own-history context in `/v1/chat/completions` (api.py around 1703-1715). Not changed here; see `ALENA_MEMORY_GUIDE.md` in the `teachcopilot-work` scratch folder.

## 5. How to run the tests

- Python with pytest and psycopg2: the project venv `..\.local\personal-memory-venv\Scripts\python.exe` (system Python has no pytest). No-database tests: `python -m pytest tests/test_conversation_retrieval_hardening.py tests/test_memory_stale_retry.py tests/test_source_scoped_search_unified.py -q`.
- Database tests must only target the DEDICATED test database, never the working one. The launcher used for the full runs refuses any other target (it is `teachcopilot-work/run_db_suite.py`; it needs the multiprocessing guard it contains).
- Last full run on the dedicated test database (everything in place): 2360 passed, 2 failed, 1 collection error (`tests/test_json_rag.py` needs `torch`). The 2 failures: `test_memory_source_corrections` (the memory job now completes after a retry, but the teacher's confirm returns 409 because the test sends another direct chat, which moves the source revision; needs a decision on whether own-conversation messages should count toward memory staleness) and `test_rich_demo_seed` (demo accounts exist in the test database but `.local/rich-demo-credentials.json` is missing).
- Quality numbers behind the decisions (synthetic only): `teachcopilot-work/benchmark/README.md`.

## 6. Prompt you can give a diff agent

> Review the change set described in `docs/ALENA_CHANGES_2026_11_12.md`. Use the patch `docs/changes/ALENA_RETRIEVAL_FIXES_2026_11_12.patch` for existing files and read the eight new test files in full. For each row of the change map decide: correct / risky / wrong, with a file:line citation. Then answer the seven questions in section 4. Do not modify files, do not run anything against a database. Report defects ranked by severity and list anything you could not verify.
