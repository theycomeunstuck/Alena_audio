"""Isolated source-scoped retrieval modes and learner-scoped embedding cache."""
from __future__ import annotations

import hashlib
import math
import re
from datetime import date

import psycopg2


RETRIEVAL_MODES = frozenset({"chronological", "lexical", "vector", "hybrid", "graph_hybrid"})
VECTOR_DIMENSIONS = 1024


def _validated_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != VECTOR_DIMENSIONS:
        raise ValueError("Embedding must contain exactly 1024 dimensions")
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Embedding contains a non-real value") from exc
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains a non-finite value")
    if not any(value != 0.0 for value in vector):
        raise ValueError("Embedding has zero norm")
    return vector


def _vector_literal(values):
    return "[" + ",".join(format(value, ".17g") for value in _validated_vector(values)) + "]"


def _parse_cached_vector(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1].split(",")
    return _validated_vector(value)


def _tokens(text):
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def _cosine(left, right):
    left, right = _validated_vector(left), _validated_vector(right)
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not math.isfinite(dot + left_norm + right_norm):
        raise ValueError("Non-finite source similarity")
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _candidate_sources(lessons, allowed_source_ids):
    allowed = set(allowed_source_ids)
    result = {}
    for lesson_index, lesson in enumerate(lessons):
        lesson_id = lesson.get("lesson_id")
        for turn_index, turn in enumerate(lesson.get("turns", [])):
            source_id = turn.get("turn_id")
            if turn.get("role") not in {"teacher", "student"} or source_id not in allowed:
                continue
            anchored = [item for item in lesson.get("turns", [])
                        if item.get("role") == "assistant" and item.get("anchor_source_id") == source_id]
            parts = [f"{turn['role']}: {turn.get('text', '')}"]
            parts.extend(f"assistant context: {item.get('text', '')}" for item in anchored)
            candidate = {"source_id": source_id, "lesson_id": lesson_id,
                         "text": "\n".join(parts), "lesson_index": lesson_index,
                         "turn_index": turn_index, "observed_on": lesson.get("observed_on")}
            # Duplicate source IDs are represented once, at their first canonical occurrence.
            result.setdefault(source_id, candidate)
    return result


def _round_robin(orders, limit):
    selected, seen = [], set()
    rank = 0
    while len(selected) < limit and any(rank < len(order) for order in orders):
        for order in orders:
            if rank < len(order) and order[rank] not in seen:
                selected.append(order[rank])
                seen.add(order[rank])
                if len(selected) == limit:
                    break
        rank += 1
    return selected


def rank_source_candidates(question, lessons, allowed_source_ids, *, mode="hybrid",
                           embed_many=None, source_embed_many=None,
                           graph_hit_source_ids=None, limit=8):
    """Rank only approved source turns plus assistant text anchored to those turns.

    `graph_hit_source_ids` is an explicit, already scoped graph result. This helper
    never reads graph state or treats selected assistant context as graph evidence.
    """
    if mode not in RETRIEVAL_MODES:
        raise ValueError("Unknown personal graph retrieval mode")
    if not isinstance(question, str) or not question.strip() or not isinstance(limit, int) or limit <= 0:
        raise ValueError("Invalid source retrieval request")
    if mode in {"vector", "hybrid", "graph_hybrid"} and not callable(embed_many):
        raise ValueError("Vector retrieval requires an embedder")
    if mode == "graph_hybrid" and graph_hit_source_ids is None:
        raise ValueError("graph_hybrid requires explicit graph hit source IDs")

    candidates = _candidate_sources(lessons, allowed_source_ids)
    # Recent lessons come first; later input lessons break same-date ties and
    # later source turns come first within the same lesson.
    ordered = sorted(candidates.values(), key=lambda item: (
        -date.fromisoformat(str(item["observed_on"])).toordinal(), -item["lesson_index"],
        -item["turn_index"], item["source_id"]))
    ids = [item["source_id"] for item in ordered]
    diagnostics = {"mode": mode, "candidate_count": len(ordered), "channels": {}}
    if not ordered:
        return {"ranked_source_ids": [], "channels": diagnostics["channels"], "diagnostics": diagnostics}

    chronological = ids
    diagnostics["channels"]["chronological"] = chronological
    query_terms = set(_tokens(question))
    lexical_scored = []
    for item in ordered:
        text_terms = set(_tokens(item["text"]))
        overlap = len(query_terms & text_terms)
        if overlap:
            lexical_scored.append((item["source_id"], overlap / max(1, len(query_terms))))
    lexical = [source_id for source_id, _ in sorted(lexical_scored, key=lambda pair: (-pair[1], ids.index(pair[0])))]
    diagnostics["channels"]["lexical"] = lexical

    vector_order = []
    if mode in {"vector", "hybrid", "graph_hybrid"}:
        query_result = embed_many([question])
        if not isinstance(query_result, (list, tuple)) or len(query_result) != 1:
            raise ValueError("Invalid query embeddings")
        query_vector = _validated_vector(query_result[0])
        vectors = []
        for start in range(0, len(ordered), 16):
            source_batch = [item["text"] for item in ordered[start:start + 16]]
            batch = (source_embed_many(source_batch) if source_embed_many is not None
                     else embed_many(source_batch))
            expected = min(16, len(ordered) - start)
            if not isinstance(batch, (list, tuple)) or len(batch) != expected:
                raise ValueError("Invalid source embeddings")
            vectors.extend(_validated_vector(vector) for vector in batch)
        scores = [(item["source_id"], _cosine(query_vector, vectors[index]))
                  for index, item in enumerate(ordered)]
        vector_order = [source_id for source_id, _ in sorted(scores, key=lambda pair: (-pair[1], ids.index(pair[0])))]
        diagnostics["channels"]["vector"] = vector_order

    if mode == "chronological":
        ranked = chronological[:limit]
    elif mode == "lexical":
        ranked = lexical[:limit]
    elif mode == "vector":
        ranked = vector_order[:limit]
    elif mode == "hybrid":
        ranked = _round_robin([vector_order, lexical], limit)
    else:
        graph_allowed = set(graph_hit_source_ids) & set(candidates)
        graph_order = list(dict.fromkeys(source_id for source_id in graph_hit_source_ids
                                         if source_id in graph_allowed))
        diagnostics["channels"]["graph"] = graph_order
        ranked = _round_robin([vector_order, lexical, graph_order], limit)
    diagnostics["ranked_count"] = len(ranked)
    return {"ranked_source_ids": ranked, "channels": diagnostics["channels"], "diagnostics": diagnostics}


class SourceEmbeddingCache:
    """Persistent exact-text source vector cache, scoped to learner and model."""

    def __init__(self, dsn, learner_id, model_identity):
        if not all(isinstance(value, str) and value.strip()
                   for value in (dsn, learner_id, model_identity)):
            raise ValueError("DSN, learner ID, and model identity are required")
        self.dsn, self.learner_id, self.model_identity = dsn, learner_id, model_identity
        self.model_identity_digest = (model_identity if re.fullmatch(r"[0-9a-f]{64}", model_identity)
                                      else hashlib.sha256(model_identity.encode("utf-8")).hexdigest())

    def _connect(self):
        return psycopg2.connect(self.dsn, connect_timeout=10)

    @staticmethod
    def _text_hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_many(self, texts, embedder):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(text, str) for text in texts):
            raise ValueError("Source texts must be strings")
        if not callable(embedder):
            raise ValueError("Embedder must be callable")
        if not texts:
            return []
        hashes = [self._text_hash(text) for text in texts]
        vectors = {}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("""SELECT text_sha256, embedding::text FROM teachcopilot.personal_graph_source_embedding_cache
                WHERE learner_id=%s AND model_identity_digest=%s AND text_sha256=ANY(%s)""",
                        (self.learner_id, self.model_identity_digest, list(dict.fromkeys(hashes))))
            for text_hash, raw_vector in cur.fetchall():
                try:
                    vectors[text_hash] = _parse_cached_vector(raw_vector)
                except ValueError:
                    cur.execute("""DELETE FROM teachcopilot.personal_graph_source_embedding_cache
                        WHERE learner_id=%s AND model_identity_digest=%s AND text_sha256=%s""",
                                (self.learner_id, self.model_identity_digest, text_hash))
        missing_hashes = list(dict.fromkeys(text_hash for text_hash in hashes if text_hash not in vectors))
        text_for_hash = {text_hash: text for text_hash, text in zip(hashes, texts)}
        for start in range(0, len(missing_hashes), 16):
            chunk = missing_hashes[start:start + 16]
            embedded = embedder([text_for_hash[text_hash] for text_hash in chunk])
            if not isinstance(embedded, (list, tuple)) or len(embedded) != len(chunk):
                raise ValueError("Embedder returned an invalid batch")
            checked = [_validated_vector(vector) for vector in embedded]
            vectors.update(zip(chunk, checked))
            with self._connect() as conn, conn.cursor() as cur:
                for text_hash, vector in zip(chunk, checked):
                    cur.execute("""INSERT INTO teachcopilot.personal_graph_source_embedding_cache
                        (learner_id,model_identity_digest,text_sha256,embedding) VALUES (%s,%s,%s,%s::vector)
                        ON CONFLICT (learner_id,model_identity_digest,text_sha256) DO UPDATE
                        SET embedding=EXCLUDED.embedding""",
                                (self.learner_id, self.model_identity_digest, text_hash, _vector_literal(vector)))
        return [vectors[text_hash] for text_hash in hashes]
