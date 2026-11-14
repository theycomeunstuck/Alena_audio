# Glossary – personal memory and retrieval terms

Scope: the local memory contour of TeachCopilot (`local_app/`, `research_memory/personal_memory_lab/`) as of 2026-11-14. Related decisions: `docs/adr/ADR-001-unified-personal-memory-graph-channel.md`, `ADR-002`, `ADR-003`, `ADR-004`. References are `file:line` in the repository root.

## The three hybrids

**Materials hybrid (product RAG, RRF).** Retrieval over the curriculum corpus: pgvector cosine leg fused with a PostgreSQL full-text leg by reciprocal rank fusion with k = 60 and a deterministic id tie-break (`local_app/store.py:630-657`, fusion at `656`); called from `RagService.search` (`local_app/rag.py:137-143`) with small soft boosts for topic and grade (`local_app/rag.py:36-37`). It ranks published materials, never learner memory, and refuses to serve from a partially indexed corpus (`local_app/store.py:675-676`).

**Personal memory hybrid (round-robin).** Retrieval over approved teacher/student source turns: a BGE-M3 vector order and a lexical order are merged by round-robin so neither channel crowds out the other (`rank_source_candidates`, `local_app/personal_graph_retrieval.py:91-166`, merge at `157-158`; `_round_robin` at `77-88`). It is the `hybrid` mode of the `personal_memory` method (`local_app/personal_memory.py:361-377`) and admits only sources inside the approved scope (`local_app/personal_graph_retrieval.py:56-74`).

**Conversation retrieval fusion (legacy/rrf).** Retrieval over the user's own chat history: `legacy` is the shipped formula (cosine + 0.04 × exact-term fraction) and `rrf` is the opt-in reciprocal rank fusion of cosine and stemmed BM25; the recipe is a caller parameter, not an env flag (`local_app/conversation_retrieval.py:42`, `474-484`, fusion dispatch at `483-484`). It serves conversational continuity with `self_report`/`context_only` statuses – never teacher-approved evidence.

These three are unrelated algorithms that happen to share the word "hybrid"; the devops manual maps them in section 1 (`docs/ALENA_MEMORY_DEVOPS_MANUAL_2026_11_14.md`).

## Graph integration

**Graph channel.** The automatic third retrieval leg inside `personal_memory` hybrid: graph hits for the exact snapshot join vector and lexical orders in the round-robin (`local_app/personal_memory.py:369-376`). It consumes a ready index, degrades with a recorded reason when unavailable, and returns source IDs only (`local_app/personal_memory.py:335-359`).

**Consume-only channel.** The invariant that the channel never builds, claims, invalidates or rebuilds an index: lookup is a read-only registry query (`find_ready_personal_graph_index`, `local_app/personal_graph_index.py:288-294`), the channel shim refuses writer calls (`local_app/personal_graph_memory.py:825-826`). Building happens only in the experimental `personal_graph` teacher job (`local_app/personal_graph_memory.py:992-995`) or `memory_doctor --graph`.

**Graph method (personal_graph).** The experimental memory method that builds and searches a Graphiti-over-Neo4j index of the approved snapshot (`METHOD` at `local_app/personal_graph_memory.py:28`; provider at `910-1015`). It requires the `LOCAL_EXPERIMENTAL_PERSONAL_GRAPH` flag, a 16k/32k context and local Gemma/BGE-M3 (`local_app/config.py:74-78`), and produces its own approval versions (`approved_graph_version`, `local_app/personal_graph_memory.py:696-749`).

**graph_hybrid mode.** The retrieval mode in which graph hits participate: `rank_source_candidates` merges vector, lexical and graph orders by round-robin and refuses the mode without explicit graph hit IDs (`local_app/personal_graph_retrieval.py:12`, `105-106`, `159-164`). In the unified path it is entered dynamically only when the channel returned source IDs (`local_app/personal_memory.py:371`).

**Navigation-only graph facts.** The rule that LLM-extracted graph entities never enter the answer text: the channel returns only canonical source IDs filtered to the approved scope (`local_app/personal_graph_memory.py:884-899`), context is rendered exclusively from snapshot turns with a "search matches are candidates" notice (`SOURCE_CONTEXT_NOTICE`, `local_app/personal_graph_memory.py:34-39`; `select_context` at `574-693`).

## Identity and approval

**Snapshot fingerprint.** SHA-256 over the ordered eligible source set and its attached contexts – texts, roles, correction links, assistant dialogue context, session-end context – excluding volatile timestamps (`personal_graph_snapshot_fingerprint`, `local_app/personal_graph_contract.py:9-53`). Index reuse and approval binding both compare this fingerprint, so any source change forces re-review.

**Recipe fingerprint (approval fingerprint).** Hash of the context protocol, immutable model digests, context size, retrieval mode, selection version and the SHA-256 of the eight selection implementation files (`local_app/personal_memory.py:53-64`). It is recorded in the draft state, rechecked after generation and around every chat retrieval, and stales approvals on any change (ADR-002).

**Config fingerprint (index fingerprint).** Hash of nine index-producing files, `graphiti-core`/`neo4j`/`pydantic` versions, model digests and the context size (`local_app/personal_graph_index.py:61-85`). A ready index matches only for this exact fingerprint, which is why a code, library, model or context change silently retires ready indexes.

**Approval cycle.** Teacher requests a draft (`POST /learners/{id}/next-step`, `local_app/api.py:440-453`) → worker prepares state plus context with recorded diagnostics (`local_app/memory_service.py:340-375`, retry semantics at `509-530`) → teacher confirms the version, with a separate acknowledgment for full-archive access (`local_app/api.py:513-542`) → approved chats replay or re-run retrieval inside the approved scope (`local_app/personal_memory.py:437-459`). New learner messages or recipe changes make the version stale and require explicit re-confirmation.

**evidence_ids vs source_ids.** `source_ids` in an approval state enumerates every source the grant covers (always the full current snapshot for `personal_memory`: `local_app/personal_memory.py:147-156, 414`); `evidence_ids` enumerates the sources actually selected into the context, always extended by their correction closure (`local_app/personal_graph_memory.py:684`, closure rule at `514-528`). Scope checks use `source_ids`; citations and projections use `evidence_ids`.

**Correction closure.** The transitive set of a source plus everything correcting or corrected by it, via `corrects_source_id` links (`correction_closure`, `local_app/personal_graph_memory.py:514-528`). Selection either includes the whole closure or none of it, so a correction can never be severed from the claim it corrects (`local_app/personal_memory.py:270-272`).

**PersonalMemoryStale / PersonalGraphIndexStale.** The fail-closed errors shown to the teacher when the recipe, sources or graph index no longer match what was approved (`local_app/personal_memory_contract.py:21-25`; `local_app/personal_graph_index.py:20-27`). In chat they surface as HTTP 409 (`local_app/api.py:1875-1876`).

## Budgets and context

**Byte budgets.** UTF-8 byte upper bounds for selected and whole memory, derived from the model context: (3000, 4000) at 8k, (10000, 12000) at 16k, (20000, 24000) at 32k (`local_app/personal_memory_contract.py:10-18`; graph constants at `local_app/personal_graph_contract.py:57-68`). The same limits are enforced for generation and query resolution (`approved_memory_fits_budget`, `local_app/personal_graph_contract.py:71-91`).

**Context floor.** The operational requirement that the full personal-memory chat cycle needs `LOCAL_CONTEXT_TOKENS >= 16384`: the graph method refuses other contexts outright (`local_app/personal_graph_memory.py:756-757`), and at 8192 the complete-memory chat fails with 413/502 (ADR-003; `local_app/runtime.py:426-429`, `local_app/api.py:1996-1999`).

**Complete-memory scope.** The runtime state entered when serving approved memory: history is not bounded, Ollama receives `truncate=False, shift=False`, and the whole input must fit `(context_tokens – max_output_tokens) × 2` bytes or the request fails (`local_app/runtime.py:354-370`, `415-445`). It exists so approved sources can never be silently cut by the transport.

**verify_no_truncation.** The strict overflow probe that sends oversized requests to generate, chat and embed and fails closed if the server does not reject them (`research_memory/personal_memory_lab/runtime.py:122-139`), recognizing both the Ollama ≤0.3x and ≥0.40 rejection wordings (`30-51`). It is wired into the experimental graph provider's model construction (`local_app/personal_graph_memory.py:916-919`).

**SourceEmbeddingCache.** Persistent exact-text vector cache scoped by learner, model digest and code recipe – rows are always filtered by `learner_id` so no other learner's vectors are read (`local_app/personal_graph_retrieval.py:169-223`, query at `197-199`). Cache identity includes code hashes, so vectors cannot be reused across selection recipes (`local_app/personal_memory.py:320-326`).

## Selection bookkeeping

**selection_warnings.** Machine-readable reasons for candidates that did not make the context: `memory_budget_exceeded`, `withheld_without_exact_source_chunk`, `retrieved_source_fragment_not_hydrated` (`local_app/personal_graph_memory.py:654-671`). They travel with the context into drafts, actions and the diagnostics endpoint (`local_app/memory_service.py:546-547, 569-570, 585-586`; `local_app/api.py:716`).

**selection_events.** Structured events for the tail presentation's whole-group decisions, such as a correction group skipped for budget (`local_app/memory_service.py:196-207`). Unlike warnings they belong to the `tail` method's presentation layer, not to source-scoped selection.

**Diagnostics endpoints.** The two teacher-only read-only endpoints `GET /learners/{id}/memory/diagnostics/index` and `GET /learners/{id}/memory/diagnostics/retrieval` (`local_app/api.py:559-648`, `650-717`); they never generate, never write graph state and never expose draft scope (ADR-004).

**Approved memory context in the prompt.** The wrapper that keeps approved memory whole and escaped: text is angle-bracket-escaped, size-checked against the byte budget and wrapped in an untrusted block (`local_app/rag.py:969-988`, escape at `40-42`).
