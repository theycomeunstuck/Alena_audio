# ADR-004: Teacher-only read-only memory diagnostics endpoints

- Status: Accepted (2026-11-14)
- Implementation: commit `f65cbe6` in `dev` (2026-11-14); approval-source fix from the 2026-11-14 review folded in
- Key files: `local_app/api.py`, `web/index.html`, `web/app.js`, `local_app/personal_graph_index.py`

## Context

After the unified graph channel landed (ADR-001), a teacher needs answers to operational questions the chat cannot provide: is the graph channel on, is there a ready index for the current snapshot, why did retrieval degrade, and what do the channels actually return for a sample question. Exposing that wrongly would leak draft memory, other learners' data or graph-write capabilities, so the surface had to be designed around what it refuses to do.

## Decision

1. **Two endpoints, both teacher-only.**
   - `GET /learners/{learner_id}/memory/diagnostics/index` (`local_app/api.py:559-648`) returns settings flags, byte budgets, snapshot statistics with the snapshot fingerprint, the graph-index status `{ready, reason, run_id, manifest}` and the approved topic. Its docstring states the contract: "Teacher-only read-only memory status… Never connects to Neo4j… every failure degrades into {ready: false, reason} without exposing DSNs, passwords or file paths" (`local_app/api.py:561-566`).
   - `GET /learners/{learner_id}/memory/diagnostics/retrieval?q&mode&budget` (`local_app/api.py:650-717`) is a dry run of the approved personal-memory retrieval path: the same snapshot, approval scope and `PersonalMemoryProvider._retrieve` as the approved chat (`local_app/api.py:656-663`, call at `693-694`), returning channel orders, ranked source IDs, selection warnings, the graph-channel reason and the selected context.

2. **Authorization is the same as the rest of the memory API.** Both endpoints require `Depends(teacher)` plus the learner `access()` check (`local_app/api.py:560`, `567-568`, `655`, `664-665`; `access` at `local_app/api.py:257`). The web counterpart is a teacher-only tab, «Диагностика памяти», with an index card and a retrieval test form (`web/index.html:87-96`; API calls at `web/app.js:1092` and `web/app.js:1959`).

3. **Conservative without approval.** Without an approved version the retrieval endpoint answers `{"status": "not_approved"}` with a Russian hint and nothing else – no draft scope, no candidate ranking, no diagnostics of an unapproved selection (`local_app/api.py:677-679`). With approval, the query runs strictly inside the approved `source_access` scope (`local_app/api.py:685`).

4. **Bounded inputs.** The query is limited to 1–2000 characters (`local_app/api.py:652`), the mode to `hybrid|graph_hybrid` (`local_app/api.py:653`), and the budget override to 500–30000 bytes (`local_app/api.py:654`); the applied budget re-runs selection only through the same `select_ranked_source_context` used by the product path (`local_app/api.py:696-701`).

5. **Approval status is computed from the canonical source set (2026-11-14 review fix).** The index endpoint validates the approval against `{**snapshot, "sources": graph_sources}` – the same canonical form the chat path passes – rather than the raw snapshot whose `sources` may contain in-flight or orphaned rows (`local_app/api.py:636-640`); the retrieval endpoint builds the same canonical view before consulting approvals (`local_app/api.py:675-676`). Without this, the endpoint could report an approved topic that the chat path itself would refuse, or vice versa, because approval validation compares fingerprints against the canonical ordered source set (`local_app/personal_graph_contract.py:9-53`).

6. **Graph status is a read-only registry lookup.** `find_ready_personal_graph_index` "never claims, invalidates or rebuilds anything" (`local_app/personal_graph_index.py:288-294`); the endpoint reads the manifest of a ready row without touching the registry or Neo4j (`_ready_index_manifest`, `local_app/personal_graph_index.py:277-285`). Every failure becomes a reason code (`channel_disabled`, `no_ready_index`, `model_identity:*`, `config_fingerprint:*`, `registry_lookup:*`) instead of an error page (`local_app/api.py:602-635`), mirroring the channel's degrade-with-reason semantics (ADR-001).

## What the endpoints deliberately do NOT do

- **No generation.** The diagnostics provider is constructed without a `model_call` (`local_app/api.py:555-557`), so no LLM answer can be produced from the dry run – only embeddings against the local model are permitted (`local_app/api.py:658-659`). This prevents the endpoint from becoming a free-form local-LLM proxy.
- **No graph writes.** No build, claim, invalidate or lease transition is reachable; the only index interaction is the read-only ready lookup (`local_app/personal_graph_index.py:288-307`). Writes stay exclusively with the experimental `personal_graph` job (`local_app/api.py:444-445`) and the doctor script.
- **No memory writes or approvals.** Version confirmation and corrections remain on their own endpoints with their own review acknowledgments (`local_app/api.py:513-542`, `498-511`); diagnostics cannot approve, refresh or modify anything.
- **No cross-learner data.** Beyond `access()`, every downstream validator binds the snapshot to the learner (`source_lessons` refuses another learner's snapshot, `local_app/personal_graph_memory.py:456-457`; approval validators require `state.learner_id` match, `local_app/personal_memory.py:113-114`, `local_app/personal_graph_memory.py:697-698`). The student-side counterpart is restricted by design: students may inspect only update progress of their own jobs and never raw sources or diagnostics (`local_app/api.py:462-468`), and chat responses are filtered for students (`local_app/api.py:2009-2011`).
- **No secret or infrastructure disclosure.** Failure reasons carry exception type names only; DSNs, the Neo4j password file contents and file paths are never returned (`local_app/api.py:561-566`, `706-708`).

## Abuse prevented

- Draft or unapproved selection leaking through an API side door (the `not_approved` answer, `local_app/api.py:677-679`).
- Students or other learners probing personal memory (`Depends(teacher)` + `access()` + learner-bound validators).
- Turning diagnostics into a generation or graph-write surface (no `model_call`; read-only index lookup).
- Resource abuse through oversized inputs (query/budget/mode bounds, `local_app/api.py:652-654`).

## Alternatives considered

- **Student-visible diagnostics.** Rejected: the same numbers let a student infer what the teacher approved and what was withheld about them; the existing student surface (job status, filtered chat) already covers their legitimate needs.
- **Generative "try the answer" mode in diagnostics.** Rejected for this iteration: it would need a `model_call`, creating a second path that can produce learner-facing text without the chat path's guards (resolver, continuation guard, identifier projection). The dry run gives the teacher the ranking and selection, which is what the channel decision needs.
- **Caching the index status.** Rejected: the lookup is a single indexed registry query (`local_app/personal_graph_index.py:301-307`); caching would risk showing a stale ready state after a code/model change staled the fingerprint.

## Consequences

- Teachers can self-serve the questions "is the graph on", "why did it degrade", "what does retrieval return" without reading logs; the same reason codes as the product path make the tab trustworthy.
- The dry run executes the full retrieval path (embeddings, and graph search when a ready index exists) on every call; for a single-teacher local deployment this is acceptable, and it is deliberately un-cached so the answer always reflects the current code and registry (open question on rate limiting).
- The 2026-11-14 canonical-snapshot fix means the endpoint's `approved_topic` agrees with the chat path by construction; any future change to approval validation must update both callers together.

## Open questions for the product owner

– Нужен ли лимит частоты на `diagnostics/retrieval`? Каждый вызов выполняет полный путь поиска с эмбеддингами (и графовым поиском при готовом индексе); для одного локального учителя это приемлемо, но при нескольких учителях или автономных аккаунтах вопрос становится реальным.

– Должна ли вкладка «Диагностика памяти» показывать причину деградации граф-канала постоянно, а не только по кнопке «Обновить», и нужен ли экспорт диагностики в файл для поддержки?

– Достаточно ли роли teacher для этих эндпоинтов в долгосрочной модели доступа (автономные аккаунты разделения, будущие роли), или нужна отдельная capability-проверка «memory diagnostics»?

– Полезно ли учителю сравнение «hybrid vs graph_hybrid» одним запросом (сейчас режим выбирается вручную в форме), и должен ли сервис показывать разницу ранжирования между режимами?
