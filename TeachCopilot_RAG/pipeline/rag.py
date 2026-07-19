# pipeline/rag.py
import logging
import psycopg2, psycopg2.extras
import torch
from sentence_transformers import SentenceTransformer
from pipeline.config import (
    DATABASE_URL, EMBEDDING_MODEL, RAG_TOP_K, RAG_MIN_SCORE,
    _EMBEDDING_PREFERRED, _EMBEDDING_FALLBACK, EMBEDDING_DEVICE
)

logger = logging.getLogger(__name__)
_model = None
_resolved_model_name = None


def _resolve_device() -> str:
    """Return 'cuda' if available (and requested), otherwise 'cpu'."""
    if EMBEDDING_DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
        logger.warning("EMBEDDING_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU")
        device = "cpu"
    else:
        device = EMBEDDING_DEVICE
    logger.info("Embedding device: %s", device)
    return device


def _get_model() -> SentenceTransformer:
    global _model, _resolved_model_name
    if _model is None:
        device = _resolve_device()
        if EMBEDDING_MODEL == "auto":
            # Try multilingual first, fallback to English-only
            try:
                logger.info("Trying preferred model: %s", _EMBEDDING_PREFERRED)
                _model = SentenceTransformer(_EMBEDDING_PREFERRED, device=device)
                _resolved_model_name = _EMBEDDING_PREFERRED
                logger.info("Loaded preferred model: %s", _EMBEDDING_PREFERRED)
            except Exception:
                logger.warning("Preferred model unavailable, falling back to: %s",
                             _EMBEDDING_FALLBACK)
                _model = SentenceTransformer(_EMBEDDING_FALLBACK, device=device)
                _resolved_model_name = _EMBEDDING_FALLBACK
        else:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
            _resolved_model_name = EMBEDDING_MODEL
    return _model


def get_model_name() -> str:
    """Return name of loaded model (loads model if not yet loaded)."""
    _get_model()
    return _resolved_model_name or EMBEDDING_MODEL


def embed_text(text: str) -> list[float]:
    """Encode a single text string into an embedding vector."""
    return _get_model().encode(text).tolist()


def search_knowledge(query: str, subject: str | None = None,
                     difficulty: str | None = None,
                     tags: list[str] | None = None,
                     limit: int | None = None) -> list[dict]:
    """Return ranked knowledge fragments.

    Empty list means either "no relevant rows" or "retrieval failed"; failures are
    logged with traceback. Callers that need to distinguish those states should
    inspect logs or add a stricter diagnostic path.
    """
    if not query or not query.strip():
        logger.debug("RAG: empty query")
        return []

    try:
        embedding = _get_model().encode(query).tolist()
        top_k = max(1, min(int(limit or RAG_TOP_K), 50))
        filters = []
        if subject:
            filters.append("AND subject = %(subject)s")
        if difficulty:
            filters.append("AND difficulty = %(difficulty)s")
        if tags:
            filters.append("AND tags && %(tags)s::text[]")
        filter_clause = " ".join(filters)
        sql = f"""
            SELECT topic, content, image_descriptions, source_file,
                   page_number, image_path, difficulty, tags,
                   1 - (embedding <=> %(emb)s::vector) AS score
            FROM knowledge_base
            WHERE 1 - (embedding <=> %(emb)s::vector) >= %(min_score)s
            {filter_clause}
            ORDER BY score DESC
            LIMIT %(top_k)s
        """
        params = {"emb": embedding, "min_score": RAG_MIN_SCORE,
                  "top_k": top_k, "subject": subject,
                  "difficulty": difficulty, "tags": tags}
        with psycopg2.connect(DATABASE_URL,
                              cursor_factory=psycopg2.extras.RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results = [dict(r) for r in cur.fetchall()]
                logger.debug("RAG: %d results for query=%r", len(results), query)
                return results
    except Exception:
        logger.exception("RAG search failed for query=%r", query)
        return []
