# scripts/eval_rag.py
"""Golden-set RAG evaluation harness (docs/EVALUATION.md).

Measures the LIVE retrieval system faithfully: embeds each query exactly as the
server does (pipeline.rag.embed_text) and runs raw cosine over knowledge_base with
NO score threshold, so it reports the true ranking + score distribution. Use it to:
  * see top-1 / top-3 hit-rate on known queries,
  * calibrate RAG_MIN_SCORE PER MODEL (in-domain vs off-topic score bands),
  * compare embedding models (re-ingest with the new model, then re-run),
  * catch "empty because DB is broken" vs "empty because nothing matches".

Usage:
  DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5445/teachcopilot \
    .venv/bin/python scripts/eval_rag.py [--golden tests/fixtures/rag_golden.json]

Exit code 0 always (diagnostic tool). Reads DATABASE_URL/EMBEDDING_MODEL from env/.env.
"""
import sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import psycopg2, psycopg2.extras
from pipeline.config import DATABASE_URL, RAG_MIN_SCORE
from pipeline.rag import embed_text, get_model_name

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85]


def _rank_all(cur, emb):
    """Return [(source_file, topic, score)] over the whole KB, best first, no cutoff."""
    cur.execute(
        """SELECT source_file, topic, 1 - (embedding <=> %s::vector) AS score
           FROM knowledge_base ORDER BY score DESC""",
        (emb,),
    )
    return [(r["source_file"], r["topic"], float(r["score"])) for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(Path(__file__).parent.parent / "tests/fixtures/rag_golden.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    items = data["items"]
    in_dom = [q for q in items if q["kind"] == "in_domain"]
    off = [q for q in items if q["kind"] == "off_topic"]

    print(f"DB: {DATABASE_URL.rsplit('@', 1)[-1]}")
    print(f"Embedding model: {get_model_name()}")
    print(f"Configured RAG_MIN_SCORE={RAG_MIN_SCORE}\n")

    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    except Exception as e:
        print(f"!! DB CONNECT FAILED: {e}\n   (This is the difference between 'no matches' and 'broken'.)")
        return
    cur = conn.cursor()
    cur.execute("SELECT count(*) c FROM knowledge_base")
    kb_n = cur.fetchone()["c"]
    print(f"knowledge_base rows: {kb_n}\n" + "=" * 72)

    top1 = top3 = 0
    in_scores, lat = [], []
    for q in in_dom:
        t0 = time.time(); emb = embed_text(q["query"]); lat.append((time.time() - t0) * 1000)
        ranked = _rank_all(cur, emb)
        # rank of first row whose source matches expectation
        want = q["expect_source"]
        rank = next((i for i, r in enumerate(ranked) if r[0] == want), None)
        best_correct = max((r[2] for r in ranked if r[0] == want), default=0.0)
        in_scores.append(best_correct)
        hit1 = rank == 0; hit3 = rank is not None and rank < 3
        top1 += hit1; top3 += hit3
        tag = "OK  " if hit1 else ("top3" if hit3 else "MISS")
        top = ranked[0] if ranked else ("-", "-", 0.0)
        print(f"[{tag}] {q['query']!r}\n       want={want}  best_correct={best_correct:.3f} rank={rank}"
              f"  |  actual#1={top[0]} ({top[1]}) {top[2]:.3f}")
    print("=" * 72)

    off_max = []
    for q in off:
        emb = embed_text(q["query"]); ranked = _rank_all(cur, emb)
        m = ranked[0][2] if ranked else 0.0; off_max.append(m)
        print(f"[off ] {q['query']!r}  max_score={m:.3f} (want LOW)")
    print("=" * 72)

    n = max(1, len(in_dom))
    print(f"top1 hit-rate: {top1}/{len(in_dom)} = {top1/n:.0%}")
    print(f"top3 hit-rate: {top3}/{len(in_dom)} = {top3/n:.0%}")
    if in_scores:
        print(f"in-domain best-correct score: min={min(in_scores):.3f} "
              f"avg={sum(in_scores)/len(in_scores):.3f} max={max(in_scores):.3f}")
    if off_max:
        print(f"off-topic top score:          min={min(off_max):.3f} "
              f"avg={sum(off_max)/len(off_max):.3f} max={max(off_max):.3f}")
    if lat:
        print(f"embed latency (ms): avg={sum(lat)/len(lat):.0f} max={max(lat):.0f}")

    # Threshold sweep: at each cutoff, how many in-domain still pass (recall) and how
    # many off-topic leak through (false positives). The sweet spot separates the two.
    print("\nThreshold sweep (in-domain kept @rank1 / off-topic leaked):")
    print("  thr    kept_in_domain   leaked_off_topic")
    for t in THRESHOLDS:
        kept = sum(1 for s in in_scores if s >= t)
        leak = sum(1 for s in off_max if s >= t)
        star = "  <-- clean split" if kept == len(in_scores) and leak == 0 else ""
        print(f"  {t:.2f}   {kept}/{len(in_scores)}              {leak}/{len(off_max)}{star}")
    conn.close()


if __name__ == "__main__":
    main()
