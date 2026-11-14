# ADR-001: Unified personal memory retrieval with a consume-only graph channel

- Status: Accepted (2026-11-14)
- Implementation: commit `a2825bd` in `dev` (2026-11-14)
- Key files: `local_app/personal_memory.py`, `local_app/personal_graph_memory.py`, `local_app/personal_graph_retrieval.py`, `local_app/personal_graph_index.py`

## Context

`personal_memory` is the ordinary teacher-draft and approved student-recall memory method (`local_app/personal_memory.py:307-308`), enabled by default (`local_app/config.py:56`). Its `hybrid` retrieval mode merges two legs – a BGE-M3 vector leg and a lexical leg – by round-robin over the ranked orders (`local_app/personal_graph_retrieval.py:157-158`, `_round_robin` at `local_app/personal_graph_retrieval.py:77-88`).

Separately, an experimental graph method (`personal_graph`, `local_app/personal_graph_memory.py:28`) builds a Graphiti-over-Neo4j index of the approved source snapshot and searches it. Before this decision the two were entirely separate products: using graph value required the experimental method, its own teacher approval cycle, Neo4j, and a 16k/32k context (`local_app/config.py:74-78`).

The 2026-11-14 plan asked: can the default path get graph value without a second approval workflow and without making student chat depend on Neo4j?

## Decision

1. **Unified hybrid.** In `personal_memory` `hybrid` mode, `PersonalMemoryProvider._retrieve` consults a graph channel (`local_app/personal_memory.py:369`). When the channel returns source IDs, retrieval mode becomes `graph_hybrid` and graph hits join the round-robin as a **third leg** beside vector and lexical (`local_app/personal_memory.py:371-376`; `rank_source_candidates` merges three orders at `local_app/personal_graph_retrieval.py:159-164`). The selection, correction-group fitting and byte budgets downstream are unchanged (`local_app/personal_memory.py:387-389`).

2. **Consume-only channel.** The channel never builds, claims, invalidates or rebuilds an index. `_graph_channel` is documented as consuming a ready index for the exact snapshot (`local_app/personal_memory.py:335-342`), `graph_channel_source_ids` "never builds, claims or invalidates one" (`local_app/personal_graph_memory.py:829-839`), and the lookup `find_ready_personal_graph_index` is a read-only registry query that "never claims, invalidates or rebuilds anything" (`local_app/personal_graph_index.py:288-294`). A channel model shim refuses any graph-writer call outright (`local_app/personal_graph_memory.py:825-826`). Building remains exclusively with the experimental `personal_graph` teacher job (`local_app/memory_service.py:357-375`, gated at `local_app/api.py:444-445`; index build via `ensure_personal_graph_index` at `local_app/personal_graph_index.py:342-418`) or the `memory_doctor --graph` script (`scripts/memory_doctor.py`).

3. **Default on, scoped by flags.** `LOCAL_MEMORY_GRAPH_CHANNEL` defaults to on (`local_app/config.py:58` field default `True`; env default `"1"` at `local_app/config.py:117`). The channel activates only in `hybrid` mode (`local_app/personal_memory.py:343-344`, reason `mode_not_hybrid` otherwise) and only when a ready index matches the exact learner scope: database identity, learner ID, snapshot fingerprint, lessons hash and config fingerprint (`local_app/personal_graph_index.py:300-307`).

4. **Graceful degrade with a recorded reason.** Every failure mode returns `{"source_ids": None, "run_id": ..., "reason": ...}` and retrieval deterministically falls back to plain hybrid: `mode_not_hybrid`, `channel_disabled`, `lessons_error:*`, `no_snapshot_fingerprint` (`local_app/personal_memory.py:343-359`); `no_database_dsn`, `non_loopback_database`, `password_file_unavailable`, `no_approved_sources`, `no_lessons`, `config_fingerprint:*`, `registry_lookup:*`, `no_ready_index`, `no_allowed_hits`, `graph_channel_error:*` (`local_app/personal_graph_memory.py:841-901`). The reason is recorded in `retrieval_diagnostics.graph_channel` (`local_app/personal_memory.py:377`) and surfaced by the teacher diagnostics endpoint (`local_app/api.py:710-717`).

5. **Graph facts stay navigation-only.** The channel returns source IDs only, filtered to canonical teacher/student turns inside this learner's snapshot and approved scope (`local_app/personal_graph_memory.py:884-899`); graph entities are never serialized into the prompt. The context payload is built exclusively from canonical snapshot turns (`select_context`, `local_app/personal_graph_memory.py:574-693`) and opens with a notice that search matches are candidates to verify, not facts (`SOURCE_CONTEXT_NOTICE`, `local_app/personal_graph_memory.py:34-39`). Because graph facts derive only from the same approved snapshot – `source_lessons` refuses unapproved teacher sources (`local_app/personal_graph_memory.py:465-466`) – the channel adds no new approval burden: the unified path still requires an approved `personal_memory` version (`approved_personal_memory_version` at `local_app/personal_memory.py:111-174`; chat gate at `local_app/api.py:1859-1862`).

## Alternatives considered

- **Keep `personal_graph` as the only graph integration (a separate third method).** Rejected as the sole path: it forces Neo4j, a 16k/32k context and a second approval workflow on anyone who wants graph value, and it duplicates the evidence base (the experimental method is retained unchanged at `local_app/memory_service.py:357-375` for research and heavy use).
- **Auto-build the index from the chat path when no ready index exists.** Rejected. A build runs Graphiti LLM extraction under a leased claim with registry writes and Neo4j writes (`local_app/personal_graph_index.py:342-418`); doing this from a chat request would couple student-chat latency and availability to Neo4j, perform side effects from a read path, and require the consumer to hold writer credentials and lease logic. The consume-only contract keeps the chat path side-effect-free.
- **Hard-fail when the graph is unavailable.** Rejected. The `personal_memory` contract is "retrieval independent of Neo4j" (`local_app/personal_memory_contract.py:1`); student chat must keep working when an optional experimental service is down. Degrading with a recorded reason preserves observability instead of hiding the failure.

## Consequences

- The default approved-chat path gains graph navigation whenever a matching ready index exists, with zero new approvals: the index is derived from the same snapshot fingerprint the approval already binds.
- Degradation is deterministic and testable – every outcome carries a reason code (`tests/test_hybrid_retrieval_postgres.py`, `tests/test_memory_diagnostics_api.py`), and the same retrieval path runs in diagnostics (`local_app/api.py:693-694`).
- Ranking becomes availability-dependent: the same question may rank sources differently with and without Neo4j. This is accepted because scope is unchanged (the channel can only elevate IDs already inside the approved `allowed` set, `local_app/personal_graph_memory.py:893-895`), and the actual channel outcome is recorded in diagnostics.
- Any change to graph code, library versions, model digests or context tokens stales every ready index via the config fingerprint (`local_app/personal_graph_index.py:61-85`), silently switching the channel off (`no_ready_index`) until a rebuild; recovery currently requires the experimental job or `memory_doctor --graph` (see open questions).
- Consumer and builder must stay fingerprint-compatible; both sides share `personal_graph_config_fingerprint` (`local_app/personal_graph_index.py:61-85`), so a divergence fails safe (no match) rather than serving a wrong index.

## Open questions for the product owner

– Кто и когда строит графовые индексы в продуктовом развёртывании? Канал только потребляет, явная сборка существует лишь в экспериментальном методе `personal_graph` (по умолчанию выключен – `LOCAL_EXPERIMENTAL_PERSONAL_GRAPH=0`) и в `memory_doctor --graph`; в интерфейсе диагностики нет действия «построить индекс». Нужна ли учителю кнопка сборки и кто владеет эксплуатацией Neo4j?

– После любого изменения кода, библиотек, моделей или контекста готовые индексы перестают совпадать по config fingerprint, и канал молча отключается (`no_ready_index`) до пересборки. Нужна ли учителю проактивная индикация «графовый канал отключился» – сейчас причина видна только на вкладке диагностики и в `retrieval_diagnostics`?

– Есть ли целевая метрика пользы канала? Сквозные проверки подтверждают корректность (evidence ⊆ approved, recall в doctor --e2e), но не выигрыш от третьей ноги. Нужен ли замер «hybrid vs graph_hybrid» на реальных занятиях, прежде чем считать канал обязательным?

– Достаточно ли правила «новейший готовый индекс по `updated_at`» при нескольких совпадающих строках реестра (`local_app/personal_graph_index.py:305`), или нужна явная политика выбора/очистки старых индексов?
