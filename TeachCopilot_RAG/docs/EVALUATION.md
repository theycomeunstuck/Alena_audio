# RAG Evaluation

Last verified: 2026-07-09.

The current RAG works, but quality should be treated as measurable behavior, not
as a feeling. This file is the suggested evaluation loop.

## Minimum Local Gate

Run this before changing retrieval, chunking, prompts, or embedding models:

```bash
cd /home/dev/projects/alena/_work/TeachCopilot_RAG
export DATABASE_URL="postgresql://postgres:postgres@localhost:5434/teachcopilot"
export HF_HOME="$PWD/pretrained_models/hf"

uv run pytest -q
uv run python scripts/test_pipeline.py
```

Expected:

- unit/contract tests pass;
- integration reports `27/27 passed`.

## Golden Query Set

Create `tests/fixtures/rag_golden.json` with real examples:

```json
[
  {
    "query": "что такое отрезок",
    "expected_topic_contains": "Отрезок",
    "expected_source_file": "small_geometry_test.md",
    "min_score": 0.45
  },
  {
    "query": "чем луч отличается от отрезка",
    "expected_topic_contains": "геометр",
    "min_score": 0.35
  }
]
```

Track:

- top-1 hit rate;
- top-3 hit rate;
- score distribution;
- empty result count;
- wrong-source count;
- latency on CPU.

## What To Log

With `TEACHCOPILOT_DEBUG_RAG=1`, logs should be enough to answer:

- What query was embedded?
- Which model/device embedded it?
- Which chunk ids/sources were returned?
- What were the scores?
- Was the failure "0 relevant rows" or "DB/model exception"?

Recommended future fields:

- `knowledge_base.id`
- `source_file`
- `page_number`
- `topic`
- `score`
- `content_len`
- `embedding_model`
- `ingestion_version`

## Retrieval Risks

Current risks:

- Broad exception handling returns `[]`, so retrieval failures are easy to miss.
- Text ingest is coarse; long RTF content may produce broad chunks.
- PDF pages are stored page-by-page, which can be noisy for exercises.
- No reranker.
- No BM25/hybrid retrieval for exact math terms.
- No formal eval set yet.

## Improvement Order

1. Add `knowledge_base.id` and metadata to RAG result logs.
2. Add a golden query test file.
3. Split long documents into semantic chunks before embedding.
4. Add source/page citations to final user-visible answers.
5. Add hybrid search: vector + keyword.
6. Consider a small reranker only after measuring latency on CPU.

## Prompt Quality Checks

For each golden query, inspect the final system prompt:

```bash
uv run python scripts/show_prompt.py
```

The prompt should:

- include the child profile only once;
- include only relevant RAG material;
- mark RAG snippets as excerpts, not the full textbook;
- include source/page metadata;
- avoid adult analytics in child mode;
- avoid child-style wording in adult mode when `TEACHCOPILOT_SPEAKER_MODE=mapped_or_adult`.

## Pass/Fail Guidance

Pass:

- top result is from the expected topic/source;
- answer can be grounded in retrieved material;
- unrelated query does not force irrelevant source citation;
- no hidden errors in logs.

Fail:

- known textbook query returns `[]`;
- source metadata is missing after ingest;
- answer cites a page/source that was not in RAG context;
- streaming path is used for production without the streaming checklist.
