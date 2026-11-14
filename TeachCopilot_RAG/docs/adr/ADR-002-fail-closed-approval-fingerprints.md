# ADR-002: Fail-closed approval fingerprints for personal memory versions

- Status: Accepted (2026-11-14)
- Implementation: commit `a2825bd` in `dev` (2026-11-14) – fingerprints existed earlier, the fail-closed staling was re-verified and tightened on this date
- Key files: `local_app/personal_memory.py`, `local_app/personal_graph_index.py`, `local_app/personal_memory_contract.py`

## Context

A teacher approves a `personal_memory` version after reviewing the selection the recipe produced. Later chats re-run retrieval per question under that approval. Between approval and use, several things can change silently: the Ollama model blobs can be swapped or re-pulled (same tag, different digest), `LOCAL_CONTEXT_TOKENS` or `LOCAL_PERSONAL_MEMORY_RETRIEVAL_MODE` can be edited, and the selection implementation itself can be edited in place (the deployment runs from a working tree, not releases). A teacher's consent covers what they saw; anything that changes what will be served must therefore make the approval unusable rather than quietly serve different behavior.

## Decision

1. **Recipe fingerprint.** `personal_memory_recipe_fingerprint` binds an approval to the context protocol, immutable model identities, the model context size, the retrieval mode, the selection version and the SHA-256 hashes of the eight implementation files that produce the selection (`local_app/personal_memory.py:53-64`, file list at `local_app/personal_memory.py:57-60`). Model identities are Ollama blob digests resolved from `/api/tags` with ambiguity refused (`local_app/personal_graph_index.py:38-58`); `SourceModel` rejects anything without valid immutable digests (`local_app/personal_memory.py:55-56`, `78-80`).

2. **Continuous re-verification, not approval-time-only.** The draft state records the fingerprint (`local_app/personal_memory.py:413-421`); `validate_state_identity` rechecks it after generation, before commit (`local_app/personal_memory.py:424-435`; called by the worker at `local_app/memory_service.py:592-593` and by the chat handler after generation at `local_app/api.py:1934-1937`); `chat_context` rechecks before and after every retrieval (`local_app/personal_memory.py:448-457`); model digests are re-checked around every embedding batch (`local_app/personal_memory.py:82-94`). The persistent source-embedding cache is keyed by an identity that also includes code hashes, so cached vectors cannot be reused across recipes (`local_app/personal_memory.py:320-326`).

3. **Approval validation is fail-closed.** `approved_personal_memory_version` accepts an approval only when every recorded field still matches the live snapshot exactly, and a newer invalid approval does not revive an older grant (`local_app/personal_memory.py:111-174`, guard comment at `132-146`). Any mismatch surfaces as `PersonalMemoryStale` with a teacher-facing Russian message (`local_app/personal_memory_contract.py:21-25`) and HTTP 409 in chat (`local_app/api.py:1875-1876`).

4. **Same discipline for the graph index.** `personal_graph_config_fingerprint` hashes nine implementation files, `graphiti-core`/`neo4j`/`pydantic` versions, model digests and the context size (`local_app/personal_graph_index.py:61-85`). An approved full-archive graph context re-validates the ready index against the approved manifest on every chat (`local_app/personal_graph_memory.py:1029-1040`; `validate_ready_personal_graph_index` at `local_app/personal_graph_index.py:310-339`), raising `PersonalGraphIndexStale` (`local_app/personal_graph_index.py:20-27`). This is what makes ADR-001's consume-only channel safe: it can never silently serve an index the teacher did not review.

5. **Graph availability is deliberately NOT part of the approval identity.** The recipe fingerprint binds configuration and selection semantics, not runtime infrastructure. Whether the graph channel contributed is recorded as diagnostics in the draft state (`local_app/personal_memory.py:419`) but does not enter the fingerprint. Rationale: Neo4j being up or down is incidental; the channel can only elevate source IDs already inside the approved `allowed` set (`local_app/personal_graph_memory.py:893-895`, chat scope at `local_app/api.py:685`), so its availability changes ranking, never scope. Including it would stale approvals on infrastructure flaps while changing nothing the teacher consented to.

## Staling semantics

Any of the following stales **all** approved `personal_memory` versions for **all** learners, requiring one re-approval per version:

- a changed chat or embedding model digest (`local_app/personal_graph_index.py:50-56`);
- a changed `LOCAL_CONTEXT_TOKENS` or retrieval mode (`local_app/personal_memory.py:61-63`; mode whitelist at `local_app/config.py:66-67`);
- a changed protocol or selection version constant (`local_app/personal_memory_contract.py:5-7`);
- an edit to any of the eight fingerprinted files (`local_app/personal_memory.py:57-60`).

This is intentional even for apparently cosmetic changes: the code hash proves that "the behavior the teacher reviewed is the behavior that serves". The re-approval burden was accepted and recorded in the decision history of 2026-11-13/14 (one re-confirmation after the fail-closed change landed).

## Alternatives considered

- **Semantic version pinning only** (trust `PERSONAL_MEMORY_SELECTION_VERSION` without code hashes). Rejected: version strings describe intent, not behavior; they can be bumped without a behavior change and left unchanged across a behavior change. Byte-level hashes make the claim "the teacher reviewed what will serve" literally true, including for a tree that is not released through immutable artifacts.
- **Fingerprint at approval time only.** Rejected: a model swap mid-session would serve different embeddings than the reviewed selection. Hence the per-call re-checks around every embed (`local_app/personal_memory.py:82-94`) and the post-generation identity re-validation before commit (`local_app/memory_service.py:592-593`).
- **Exempt refactors by maintaining a manual allowlist of "harmless" files.** Rejected: judgment about harm would itself be unaudited; the cost of an occasional re-approval is bounded and the failure mode of a wrong exemption is silent divergence from reviewed behavior.

## Consequences

- No silent drift is possible between what the teacher approved and what later chats serve; the embedding cache cannot smuggle vectors from another recipe (`local_app/personal_memory.py:320-326`).
- Every code touch to the eight files stales every approval – the operational cost is real and was consciously accepted for a single-teacher local deployment; at larger scale it becomes a batch re-approval problem (open question below).
- Refactoring inside those files is not free: moving selection logic between files requires updating the fingerprint file list or accepting the stale-then-re-approve cycle.
- Changing `LOCAL_CONTEXT_TOKENS` (for example 16384 → 32768) intentionally stales approvals because the byte budgets and prompt shape change (see ADR-003); the devops manual documents the re-approval step after a context change (`docs/ALENA_MEMORY_DEVOPS_MANUAL_2026_11_14.md`, section 4b).
- The fail-closed direction errs against availability: a stale approval blocks chat memory (409) instead of serving stale context. This matches the product rule that a withdrawn or unreviewed selection must not reach the learner.

## Open questions for the product owner

– Приемлема ли цена пере-подтверждения при каждом изменении восьми кодовых файлов рецепта – одно изменение состаривает все одобрения всех учеников? Нужна ли группировка изменений (рефакторинг без изменения поведения) так, чтобы пере-подтверждение требовалось только при изменении логики выбора?

– Должно ли пере-подтверждение показывать учителю, что именно изменилось в рецепте (модели, контекст, версия выбора), чтобы согласие было информированным, а не повторным нажатием «подтвердить»?

– Каков ожидаемый масштаб эксплуатации (ученики × темы × версии)? Если счёт идёт на десятки одобрений, нужен ли пакетный интерфейс пере-подтверждения после обновления кода?

– Нужен ли аудит-журнал отпечатков (кто, когда, каким recipe fingerprint подтверждал версию) для разбора инцидентов – сейчас отпечаток хранится в состоянии версии, но отдельного журнала переходов между отпечатками нет?
