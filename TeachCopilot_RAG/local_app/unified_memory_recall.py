"""Own-archive recall over persisted PostgreSQL search representations.

The source reader (``read_own_messages``) establishes ownership and provides
correction-safe hydration behavior identical to the shipped conversation
retrieval. This module adds durable chunk rows with a PostgreSQL full-text
representation and optional pgvector embeddings, fused with deterministic RRF
(k=60) over two learner- and actor-scoped channels. A full-text/vector outage
degrades with a visible diagnostic and a bounded lexical fallback; the scope
never widens.
"""
from __future__ import annotations

import time
from datetime import date, datetime

from psycopg2 import Error as DatabaseError

from .conversation_retrieval import (
    CHUNK_RECIPE,
    ConversationEmbedder,
    ConversationRetrievalUnavailable,
    _select_quotes,
    _time_filter,
    chunk_conversation_messages,
    read_own_messages,
    unit_vector,
)

RRF_K = 60
SEARCH_LIMIT = 12
MIN_SEMANTIC_SIMILARITY = 0.20  # same engineering gate as the shipped retrieval
# Round-3 follow-up: archives full of same-template verification turns
# («Верно: промежуточный 48, финальный 35…» with different numbers) flood the
# fused candidate list with near-duplicates and crowd out distinct evidence.
# Digit-masked trigram similarity detects the shared template; the top-ranked
# instance of a template is kept and the freed slots backfill from deeper in
# the channels. Distinct evidence (different words) is never suppressed.
NEAR_DUPLICATE_JACCARD = 0.75
# Short texts inflate trigram Jaccard; a template match also needs enough
# absolute shared trigrams before two chunks count as the same template.
NEAR_DUPLICATE_MIN_SHARED = 60
# Drill archives carry many verification turns whose 900-char windows differ
# only in the side exercises around a shared verification line: digit-masked
# trigrams miss them, but their chunk embeddings are near-identical. A high
# cosine between stored chunk vectors is the second duplicate detector.
NEAR_DUPLICATE_COSINE = 0.93
NEAR_DUPLICATE_BACKFILL = SEARCH_LIMIT
# Next round: a delete-style collapse lost different facts. Two chunks that
# differ only in their numbers (34 vs 96 candies) or in one corrected word are
# NOT duplicates — they are different answers — yet both the digit-masked
# template match and the embedding cosine happily equate them. The collapse
# therefore became an admission cap: the detectors only decide the near-
# duplicate GROUP, and only the top-ranked member of a group occupies the
# delivered depth before distinct material. Deferred members are never
# removed: they stay in the retrieved pool, keep their sources visible in the
# trace, and backfill any delivered slots that distinct evidence does not
# fill (so a small pool still delivers the whole 34/96 pair). Verbatim
# repeats (same template, same digits) stop crowding; different values,
# corrections and assistant context survive; the live rank probe recorded the
# drill crowd pushing the answer-bearing chunk to fused position 125/177.
NEAR_DUPLICATE_DELIVERY_CAP = 1
# Bounded candidate scan (A2/A6 rank evidence: the gold chunk sat at fused
# position 125/177 and 48/177 — far below the fused top-24 — and a reranker
# cannot recover a chunk outside its pool). The recall now scans the whole
# authorized snapshot in ONE store call bounded by SNAPSHOT_SCAN_ROWS: the
# previous batch walk excluded already-seen keys on every expansion call, and
# because the store computes its IDF statistics and both rank channels after
# that exclusion, each sample scored its rows against a different corpus —
# the fused order could drift between samples and identical keys were cut in
# Python afterwards anyway. One call over the fixed snapshot keeps IDF, both
# rank channels and the fused order computed on exactly the same set, costs
# one round trip instead of one per batch, and stays bounded: the store caps
# a scan at 1000 rows and the caller only ever passes the current snapshot's
# chunk keys, so the scan cannot widen the scope. No query rewriting, no
# golden cues.
SNAPSHOT_SCAN_ROWS = 800
MAX_CANDIDATE_ROWS = 800
# Kept for diagnostic compatibility: the recorded name of the search-call
# cost. The one-pass scan performs exactly one search per attempt.
EXPANSION_MAX_QUERIES = 1
# Message-level rerank inside the retrieved pool (the bounded promotion
# scheme for the recorded A2/A6/F4/B2/F3 losses: the needed episode was IN
# the pool at fused positions 13-119, crowded out by 25-41 same-template
# drill groups whose whole-chunk vectors dominate both fused channels). The
# rerank scores SINGLE-MESSAGE embeddings against the query: a 900-char drill
# window dilutes the chunk vector, while the one student sentence that
# answers the question does not dilute its own message vector. Promotion is
# a bounded band, not a full reorder: chunks whose best message cosine
# reaches SPAN_RERANK_MIN_COSINE lead the pool in cosine order (then scan
# order); every chunk below the floor keeps its exact scan-order position.
# A weak-match archive therefore keeps its fused order (no noise-driven
# scrambling), a strong match cannot be buried by template crowd, and no
# query rewriting or golden cues are involved. The cost is bounded and
# measured: only messages of pool chunks are embedded, each once per model
# identity (persistent text cache, so repeat questions and later runs pay
# nothing); an outage or failure degrades to the un-reranked order with a
# visible diagnostic. The floor is an engineering constant like
# MIN_SEMANTIC_SIMILARITY: provisional, untuned on held-out data, and
# recorded with every rerank so the calibration stays visible. The recorded
# archives measured (selection-diag-1, real BGE-M3): the needed episode's
# best message cosine was 0.5092-0.6437 for the rows the span channel can
# separate, while the crowd's 90th percentile stayed at 0.43-0.48 — the
# floor sits between them. A2's gold (0.4732, rank 24) stays below the floor
# on purpose: its chunk is recovered by the lexical channel instead
# (text_rank 3/177) through the channel guarantee below.
SPAN_RERANK_MIN_COSINE = 0.50
SPAN_RERANK_MAX_MESSAGES = 2000
# Channel-delivery guarantee (the A2/F4 recorded shape: the gold chunk's
# lexical rank was 3/177 and 2/48 while its fused position was 129 and 13 —
# a dense band of same-template chunks makes the vector ranks inside the
# template family effectively arbitrary, and the fusion average buries a
# chunk one channel knows is relevant). Whatever the fused order says, a
# chunk that ANY ranking channel puts in its top CHANNEL_GUARANTEE_TOP still
# reaches the packed evidence: the packed input window extends past the
# front to include such chunks from the diversified pool, bounded by
# CHANNEL_GUARANTEE_TOP x 2 chunks in total. No golden cues: the guarantee
# reads the store's own per-channel ranks, whichever query is asked.
CHANNEL_GUARANTEE_TOP = 3


def _iso(value):
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def _template_trigrams(text):
    """Digit-masked, whitespace-normalized trigram set of a chunk text."""
    import re
    normalized = re.sub(r"\d+", "#", str(text).casefold())
    normalized = re.sub(r"\s+", " ", normalized)
    return {normalized[index:index + 3]
            for index in range(max(0, len(normalized) - 2))}


def _cosine(left, right):
    import numpy as np
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _diversify_near_duplicates(hits, store, learner_id, actor_id,
                               delivered_limit=SEARCH_LIMIT):
    """Admit the top-ranked member of each near-duplicate template group
    into the delivered depth; defer the rest behind distinct material and
    backfill unfilled slots.

    The two round-3 detectors (digit-masked trigram template, chunk-embedding
    cosine) decide only the GROUP. Nothing is deleted: a deferred chunk keeps
    its rank, stays in the retrieved pool and returns when distinct evidence
    does not fill the depth, so near-identical exercises with different
    numbers (34 vs 96 candies) or a corrected fact are never lost, while truly
    repeated material cannot crowd the depth with its 12th copy.
    """
    texts = store.conversation_chunk_texts(learner_id, actor_id, CHUNK_RECIPE,
                                           [hit["chunk_key"] for hit in hits])
    vectors_raw = store.conversation_chunk_embeddings(learner_id, actor_id, CHUNK_RECIPE,
                                                      [hit["chunk_key"] for hit in hits])
    vectors = {}
    for key, literal in vectors_raw.items():
        if literal:
            try:
                vectors[key] = [float(part) for part in str(literal).strip().strip('[]').split(',')]
            except (ValueError, TypeError):
                continue
    groups = []  # {"trigrams": set, "vector": list|None}
    assignment = {}
    for hit in hits:
        key = hit["chunk_key"]
        text = texts.get(key, "")
        trigrams = _template_trigrams(text)
        vector = vectors.get(key)
        assigned = None
        for index, group in enumerate(groups):
            shared = len(trigrams & group["trigrams"])
            if shared >= NEAR_DUPLICATE_MIN_SHARED and \
                    shared / max(1, len(trigrams | group["trigrams"])) >= NEAR_DUPLICATE_JACCARD:
                assigned = index
                break
            group_vector = group["vector"]
            if vector and group_vector and len(vector) == len(group_vector):
                if _cosine(vector, group_vector) >= NEAR_DUPLICATE_COSINE:
                    assigned = index
                    break
        if assigned is None:
            groups.append({"trigrams": trigrams, "vector": vector})
            assigned = len(groups) - 1
        assignment[key] = assigned
    counts: dict[int, int] = {}
    kept, deferred = [], []
    for hit in hits:
        group = assignment[hit["chunk_key"]]
        if counts.get(group, 0) < NEAR_DUPLICATE_DELIVERY_CAP:
            counts[group] = counts.get(group, 0) + 1
            kept.append(hit)
        else:
            deferred.append(hit)
    admitted_count = len(kept)
    if len(kept) < delivered_limit and deferred:
        kept.extend(deferred[:delivered_limit - len(kept)])
    # Candidates beyond the delivered depth are ordinary truncation, not
    # duplicates: the diagnostic counts only group members deferred by the cap.
    # admitted_count (before backfill) drives the candidate-expansion loop:
    # a crowded pool must keep fetching deeper batches even though backfill
    # already filled the delivered depth with template members.
    return kept, len(deferred), len(groups), admitted_count


def _embedder(store, runtime, learner_id, actor_id):
    """Reuse the shipped BGE-M3 embedder with its persistent cache contract."""
    return ConversationEmbedder(store, runtime, learner_id, actor_id)


def _rerank_pool_by_message_similarity(pool, query_vector, store, learner_id, actor_id,
                                       embedder, message_by_id):
    """Bounded message-level promotion inside the retrieved pool.

    Every pool chunk is scored by its best single-message cosine to the
    query. Chunks whose best cosine reaches SPAN_RERANK_MIN_COSINE lead the
    pool in cosine order (ties keep the scan order); every chunk below the
    floor keeps its exact scan-order position, so a weak-match archive keeps
    its fused order while a strong answer-bearing message cannot stay buried
    under the template crowd. Returns the pool reordered plus a diagnostics
    dict; the input list is not mutated. Degrades to the un-reranked order
    with a visible diagnostic on any embedding-service failure and records
    the measured embedding cost.
    """
    diagnostics = {"applied": False, "messages_embedded": 0,
                   "min_cosine_floor": SPAN_RERANK_MIN_COSINE}
    if not pool or query_vector is None or embedder is None:
        diagnostics["reason"] = "unavailable_no_embeddings"
        return pool, diagnostics
    try:
        started = time.monotonic()
        # Whole-message texts of the pool's spans: the semantic unit is the
        # recorded exchange member, not the 900-char search window around it.
        candidate_texts, text_by_key = [], {}
        for hit in pool:
            for span in hit.get("spans") or []:
                source_id = span.get("source_id")
                message = message_by_id.get(source_id)
                if not message:
                    continue
                text = str(message.get("text") or "")
                if not text.strip() or source_id in text_by_key:
                    continue
                text_by_key[source_id] = text
                candidate_texts.append(text)
        if not candidate_texts:
            diagnostics["reason"] = "no_span_messages"
            return pool, diagnostics
        diagnostics["messages_eligible"] = len(candidate_texts)
        if len(candidate_texts) > SPAN_RERANK_MAX_MESSAGES:
            diagnostics["messages_truncated"] = len(candidate_texts) - SPAN_RERANK_MAX_MESSAGES
            candidate_texts = candidate_texts[:SPAN_RERANK_MAX_MESSAGES]
        # A distinct cache identity family (message-level, not chunk-transcript
        # level) so the two vector kinds never share cache entries.
        from .personal_graph_retrieval import SourceEmbeddingCache
        span_embedder = ConversationEmbedder(store, embedder.runtime, learner_id, actor_id,
                                             document_recipe="own-message-span-v1")
        cache = (SourceEmbeddingCache(store.dsn, learner_id, span_embedder.cache_identity)
                 if getattr(store, "dsn", None) else None)
        if cache is not None:
            vectors = cache.embed_many(candidate_texts, span_embedder.embed_many)
            diagnostics["cache"] = "persistent"
        else:
            vectors = span_embedder.source_vectors(candidate_texts)
            diagnostics["cache"] = "memo"
        diagnostics["messages_embedded"] = len(candidate_texts)
        vector_by_text = {text: vector for text, vector in zip(candidate_texts, vectors)
                          if vector is not None}
        if not vector_by_text:
            diagnostics["reason"] = "no_message_vectors"
            return pool, diagnostics
        # Best message cosine per chunk (same normalization as the shipped
        # retrieval: unit vectors, plain dot product).
        best_by_chunk = []
        for hit in pool:
            best = None
            for span in hit.get("spans") or []:
                text = text_by_key.get(span.get("source_id"))
                vector = vector_by_text.get(text)
                if vector is None:
                    continue
                similarity = float(sum(left * right for left, right
                                       in zip(query_vector, unit_vector(vector))))
                if best is None or similarity > best:
                    best = similarity
            best_by_chunk.append(best)
        top_cosine = max((best for best in best_by_chunk if best is not None), default=None)
        diagnostics["top_cosine"] = round(top_cosine, 4) if top_cosine is not None else None
        if top_cosine is None or top_cosine < SPAN_RERANK_MIN_COSINE:
            # No message reaches the floor: the scan order stays, honestly.
            diagnostics.update(applied=False, reason="below_floor",
                               promoted_count=0,
                               latency_ms=round((time.monotonic() - started) * 1000, 1))
            return pool, diagnostics
        order = sorted(
            range(len(pool)),
            key=lambda index: (
                # Promoted band first (cosine desc), the exact scan order
                # inside both bands: a sub-floor chunk never moves relative
                # to its scan-order neighbours.
                0.0 if (best_by_chunk[index] is not None
                        and best_by_chunk[index] >= SPAN_RERANK_MIN_COSINE) else 1.0,
                -(best_by_chunk[index] if best_by_chunk[index] is not None
                  and best_by_chunk[index] >= SPAN_RERANK_MIN_COSINE else 0.0),
                index))
        promoted = sum(1 for best in best_by_chunk
                       if best is not None and best >= SPAN_RERANK_MIN_COSINE)
        diagnostics.update(applied=True, promoted_count=promoted,
                           reranked=len(order),
                           latency_ms=round((time.monotonic() - started) * 1000, 1))
        return [pool[index] for index in order], diagnostics
    except PermissionError:
        raise
    except (RuntimeError, OSError, DatabaseError, ConversationRetrievalUnavailable,
            ValueError, AttributeError) as exc:
        diagnostics.update(applied=False, reason="rerank_degraded",
                           rerank_error=type(exc).__name__)
        return pool, diagnostics


def recall_own_conversation(store, runtime, learner_id, actor_id, query, *, session_id=None,
                            budget_bytes=4000, time_filter=None, student_audience=False,
                            exclude_source_id=None):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("recall query is required")
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget_bytes must be a nonnegative integer")
    if session_id is not None:
        session = store.get_learning_session(session_id, learner_id)
        if session is None:
            raise LookupError(session_id)

    messages = read_own_messages(store, learner_id, actor_id,
                                 exclude_source_id=exclude_source_id)
    temporal = _time_filter(query, store, actor_id, time_filter, learner_id=learner_id)
    filtered = [message for message in messages if (
        not (temporal.get("from") or temporal.get("through")) or message.get("activity_day") is not None
        and temporal.get("from", "0001-01-01") <= str(_iso(message["activity_day"])) <= temporal.get("through", "9999-12-31"))]
    fallback_widened = False
    if (temporal.get("source") == "query" and (temporal.get("from") or temporal.get("through"))
            and not filtered and messages):
        # A date guessed from the wording of the question must not hide the
        # whole archive; an explicit caller range is never widened.
        filtered, fallback_widened = messages, True
        temporal = {**temporal, "applied": False, "fallback": "no_messages_in_range"}

    result = {"method": "conversation_retrieval", "text": "", "system_text": "", "sources": [],
              "source_ids": [], "evidence_ids": [], "context_documents": [], "diagnostics": {
                  "archive_message_count": len(messages), "eligible_message_count": len(filtered),
                  "chunk_recipe": CHUNK_RECIPE, "rrf_k": RRF_K,
                  "min_semantic_similarity": MIN_SEMANTIC_SIMILARITY,
                  "scope": "own_account_conversation",
                  "budget_bytes": budget_bytes, "search": "persisted_postgres_rrf"}}
    # Provenance stages (round-3 follow-up): the authorized candidate pool and
    # what the fused search actually returned, as actual pipeline data.
    result["eligible_source_ids"] = [message["source_id"] for message in filtered]
    if fallback_widened:
        result["diagnostics"]["time_filter_fallback"] = True
    if not filtered or not budget_bytes:
        result["diagnostics"].update(chunk_count=0, matched_chunk_count=0, selected_hit_count=0)
        # The plan channel is independent of the conversation archive: an empty
        # own archive still carries eligible approved lesson plans as context.
        _lesson_plan_context(result, store, learner_id, query,
                             student_audience=student_audience)
        return result

    def _index_sources(eligible):
        current_chunks = chunk_conversation_messages(eligible, learner_id, actor_id)
        order_by_id = {message["source_id"]: message.get("source_order", 0) for message in eligible}
        for chunk in current_chunks:
            orders = [order_by_id[source_id] for source_id in chunk["source_ids"]]
            chunk["first_source_order"] = min(orders)
            chunk["last_source_order"] = max(orders)
        store.upsert_conversation_chunks(learner_id, actor_id, current_chunks)
        return {chunk["chunk_id"]: chunk for chunk in current_chunks}

    chunks = _index_sources(filtered)
    query_vector = None
    identity = None
    degraded = None

    def _fill_embeddings(embedder, current_chunks):
        missing = store.chunk_embedding_status(learner_id, actor_id, CHUNK_RECIPE, embedder.identity)
        keys = [key for key in missing if key in current_chunks]
        if not keys:
            return
        vectors = embedder.source_vectors([current_chunks[key]["text"] for key in keys])
        for key, vector in zip(keys, vectors):
            if vector is not None:
                store.store_chunk_embedding(learner_id, actor_id, CHUNK_RECIPE,
                                            embedder.identity, key, unit_vector(vector))

    def _embedding_failed(exc):
        result["diagnostics"].update(degraded=True, degraded_reason="embedding_unavailable",
                                     degraded_error=type(exc).__name__)

    embedding_errors = (RuntimeError, OSError, DatabaseError, ConversationRetrievalUnavailable,
                        ValueError, AttributeError)
    if runtime is None:
        # A-report F4: no runtime means no embedding channel; degrade to the
        # lexical channel with a visible diagnostic instead of leaking an
        # AttributeError from the embedder constructor.
        degraded = "runtime_unavailable"
        result["diagnostics"].update(degraded=True, degraded_reason="embedding_unavailable",
                                     degraded_error="runtime is not configured")
    else:
        try:
            embedder = _embedder(store, runtime, learner_id, actor_id)
            identity = embedder.identity
            _fill_embeddings(embedder, chunks)
            query_vector = unit_vector(embedder.embed_many([query])[0])
        except PermissionError:
            raise
        except embedding_errors as exc:
            # AttributeError keeps an unexpected embedder incompatibility a
            # degradation rather than a crash (A-report F4 probe path).
            degraded = type(exc).__name__
            _embedding_failed(exc)

    def _search(use_vector, limit=SEARCH_LIMIT):
        return store.search_conversation_chunks(
            learner_id, actor_id, query, chunk_recipe=CHUNK_RECIPE,
            embedding_identity=identity if use_vector else None,
            query_vector=query_vector if use_vector else None,
            limit=limit, session_id=str(session_id) if session_id else None,
            time_filter=temporal if temporal.get("applied", True) else None, rrf_k=RRF_K,
            min_similarity=MIN_SEMANTIC_SIMILARITY, candidate_chunk_keys=chunks)

    def _eligible(batch):
        # Stale-proofing: a hit may only contribute sources that are still in
        # the authorized snapshot this request read (hydration builds its
        # by_id from exactly that snapshot).
        eligible = set(result["eligible_source_ids"])
        return [hit for hit in batch
                if eligible.issuperset(hit["source_ids"])
                and all(span["source_id"] in eligible for span in hit["spans"])]

    def _channel_guaranteed(hit):
        text_rank, vector_rank = hit.get("text_rank"), hit.get("vector_rank")
        return ((text_rank is not None and text_rank <= CHANNEL_GUARANTEE_TOP)
                or (vector_rank is not None and vector_rank <= CHANNEL_GUARANTEE_TOP))

    def _packed_input(hits):
        """The bounded packed-evidence window with the channel guarantee.

        The front is the relevance-first diversified order. A chunk the
        lexical or vector channel ranks in its own top-CHANNEL_GUARANTEE_TOP
        but whose fused position sits behind the front enters the window AT
        ITS CHANNEL RANK, not appended at the end: appending would put the
        recovered evidence behind the budget frontier again (the recorded
        crowding shape — the tail never reaches the model when the front
        consumes the bytes). The merged order is by effective position
        (channel rank for guaranteed deep chunks, fused position otherwise),
        ties keep the fused order ahead of the recovered deep chunks. The
        recovered tail is bounded (2 x CHANNEL_GUARANTEE_TOP chunks) and the
        byte budget still binds every acceptance.
        """
        front = hits[:SEARCH_LIMIT]
        deep = [hit for hit in hits[SEARCH_LIMIT:] if _channel_guaranteed(hit)]
        deep = deep[:2 * CHANNEL_GUARANTEE_TOP]
        if deep:
            result["diagnostics"]["channel_guaranteed_keys"] = \
                [hit["chunk_key"] for hit in deep]
        if not deep:
            return front
        entries = []
        for fused_position, hit in enumerate(front, start=1):
            channel = min((rank for rank in (hit.get("text_rank"), hit.get("vector_rank"))
                           if rank is not None and rank <= CHANNEL_GUARANTEE_TOP),
                          default=None)
            effective = min(fused_position, channel) if channel else fused_position
            entries.append((effective, fused_position, hit))
        for hit in deep:
            channel = min(rank for rank in (hit.get("text_rank"), hit.get("vector_rank"))
                          if rank is not None)
            entries.append((channel, 1000 + channel, hit))
        entries.sort(key=lambda row: (row[0], row[1]))
        return [row[2] for row in entries]

    def _collect_pool(use_vector):
        """One ranked scan over the fixed authorized snapshot.

        The whole candidate set is fetched in a single store call (bounded by
        SNAPSHOT_SCAN_ROWS, restricted to the current snapshot's chunk keys):
        IDF statistics, both rank channels and the fused order are computed
        on exactly the same set, so no sample-to-sample rank drift can move a
        candidate between calls, and the search costs one round trip. The
        caller-side eligibility filter stays: a hit may only contribute
        sources that are still in the authorized snapshot this request read.
        """
        started = time.monotonic()
        batch = _search(use_vector=use_vector, limit=SNAPSHOT_SCAN_ROWS)
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        pool = _eligible(batch)
        return pool, 1, latency_ms

    pool, expansion_queries, search_latency_ms = _collect_pool(use_vector=not degraded)
    if not pool and not degraded and temporal.get("applied", True) \
            and (temporal.get("from") or temporal.get("through")) \
            and temporal.get("source") == "query":
        # Indexed full-text/date narrowing found nothing for a query-derived
        # date; retry bounded WITHOUT the date instead of returning empty. The
        # requested range stays recorded in diagnostics, an explicit caller
        # range never reaches this fallback.
        temporal = {**temporal, "from": None, "through": None, "applied": False,
                    "fallback": "no_index_hits_in_range",
                    "requested_from": temporal.get("from"),
                    "requested_through": temporal.get("through")}
        filtered = messages
        chunks = _index_sources(filtered)
        result["eligible_source_ids"] = [message["source_id"] for message in filtered]
        result["diagnostics"]["eligible_message_count"] = len(filtered)
        try:
            _fill_embeddings(embedder, chunks)
        except PermissionError:
            raise
        except embedding_errors as exc:
            degraded = type(exc).__name__
            _embedding_failed(exc)
        result["diagnostics"]["time_filter_fallback"] = True
        pool, more_queries, more_latency = _collect_pool(use_vector=not degraded)
        expansion_queries += more_queries
        search_latency_ms = round(search_latency_ms + more_latency, 1)
    if not pool and not degraded:
        # Vector identity missing rows entirely (fresh index): lexical channel
        # alone still runs through the same RRF SQL with vector absent.
        pool, more_queries, more_latency = _collect_pool(use_vector=False)
        expansion_queries += more_queries
        search_latency_ms = round(search_latency_ms + more_latency, 1)

    # Bounded message-level rerank inside the retrieved pool: the recorded
    # A2/A6/F4/B2/F3 losses sat at fused positions 13-119 under 25-41
    # same-template drill groups. The rerank is a third RRF channel over
    # single-message embeddings; a failure degrades to the scan order.
    span_rerank_diag = {"applied": False}
    if pool and not degraded and query_vector is not None:
        message_by_id = {message["source_id"]: message for message in filtered}
        pool, span_rerank_diag = _rerank_pool_by_message_similarity(
            pool, query_vector, store, learner_id, actor_id, embedder, message_by_id)

    hits, near_duplicate_deferred, near_duplicate_groups, _admitted = \
        _diversify_near_duplicates(pool, store, learner_id, actor_id)

    retrieved_source_ids = list(dict.fromkeys(source_id for hit in pool
                                              for source_id in hit.get("source_ids") or []))

    result["diagnostics"].update(
        chunk_count=len(chunks), matched_chunk_count=len(hits),
        near_duplicate_deferred=near_duplicate_deferred,
        near_duplicate_groups=near_duplicate_groups,
        near_duplicate_delivery_cap=NEAR_DUPLICATE_DELIVERY_CAP,
        expansion_queries=expansion_queries,
        search_queries=expansion_queries,
        # Measured search cost, kept separate from the model work: wall time
        # of the ranked snapshot scans (SQL only), plus the rerank cost.
        search_latency_ms=search_latency_ms,
        span_rerank=span_rerank_diag,
        candidate_rows_examined=len(pool))
    result["retrieved_source_ids"] = retrieved_source_ids
    if not hits:
        result["diagnostics"].update(selected_hit_count=0)
        _lesson_plan_context(result, store, learner_id, query,
                             student_audience=student_audience)
        return result

    # Correction links are known before selection so a delivered quote can
    # carry its counterpart (corrected original or correction) through the
    # same budget trials as its chunk.
    correction_refs = store.memory_source_correction_refs(
        learner_id, list({source_id for hit in hits[:SEARCH_LIMIT]
                          for source_id in hit.get("source_ids") or []}))
    text, quotes, selected_hits, selection_diagnostics = _select_quotes(
        [{"source_ids": hit["source_ids"], "spans": hit["spans"], "score": hit["score"],
          "chunk_id": hit["chunk_key"]} for hit in _packed_input(hits)], filtered, budget_bytes,
        correction_refs=correction_refs)
    superseded_by = {message["source_id"]: message["superseded_by"]
                     for message in filtered if message.get("superseded_by")}
    for quote in quotes:
        if quote["source_id"] in correction_refs:
            quote["corrects_source_id"] = correction_refs[quote["source_id"]]
        if quote["source_id"] in superseded_by:
            # A quote from a corrected original never masquerades as a current
            # fact: the correcting source id travels with it.
            quote["superseded_by"] = superseded_by[quote["source_id"]]
    result["diagnostics"].update(
        selected_hit_count=len(selected_hits),
        pair_incomplete_source_ids=selection_diagnostics["pair_incomplete_source_ids"],
        correction_incomplete_source_ids=selection_diagnostics["correction_incomplete_source_ids"],
        channels=[{"chunk_key": hit["chunk_key"], "score": hit["score"],
                   "vector_rank": hit.get("vector_rank"), "text_rank": hit.get("text_rank")}
                  for hit in hits[:SEARCH_LIMIT]],
        embedding_model_identity=identity)
    result.update(text=text, system_text=text, sources=quotes,
                  source_ids=list(dict.fromkeys(quote["source_id"] for quote in quotes
                                                if quote["role"] == "user")))
    result["evidence_ids"] = list(result["source_ids"])
    _lesson_plan_context(result, store, learner_id, query, student_audience=student_audience)
    return result


def _lesson_plan_context(result, store, learner_id, query, *, student_audience, limit=3,
                         max_chars=1200):
    """P2: approved lesson plans as bounded context, never student evidence.

    Only this learner's own approved plans are eligible; the student projection
    redacts the stored answer spans exactly like ``get_lesson`` does. A plan
    does not prove completion - the field is context for the reply, never
    evidence of a student achievement. An unavailable plan channel degrades to
    an empty list with a visible diagnostic instead of failing the evidence
    path.
    """
    from .unified_memory_store import redact_lesson_text
    from psycopg2 import Error as DatabaseError
    try:
        plans = store.search_lesson_plan_context(learner_id, query, limit=limit)
    except (RuntimeError, OSError, DatabaseError, ValueError) as exc:
        result["context_documents"] = []
        result["diagnostics"]["context_documents_count"] = 0
        result["diagnostics"]["context_documents_degraded"] = type(exc).__name__
        return
    documents = []
    for row in plans:
        lesson_text = row["lesson_text"]
        if student_audience:
            lesson_text, _ = redact_lesson_text(lesson_text, row.get("answer_sections") or [])
        documents.append({
            "kind": "lesson_plan",
            "lesson_id": str(row["id"]),
            "lesson_revision": row["revision"],
            "topic_id": row.get("topic_id"),
            "minutes": row.get("minutes"),
            "text": lesson_text[:max_chars],
            "truncated": len(lesson_text) > max_chars,
            "source_refs": [{"kind": "lesson_document", "lesson_id": str(row["id"]),
                             "revision": row["revision"]}],
            "context_note": "План занятия – только контекст; он не доказывает выполнение.",
        })
    result["context_documents"] = documents
    result["diagnostics"]["context_documents_count"] = len(documents)
