# Round-2 synthetic archives and frozen gold (client A)

This package is A-owned and frozen before B's live quality run. It contains
three six-session synthetic archives, the immutable source-key mapping, a
loader for disposable PostgreSQL databases and the gold manifest with 18
questions.

## Contents

- `histories/r2-mem-a.json` – development history A, the oversized archive
  (≥110000 characters; above the 32768-token model window at a conservative
  3.2 characters/token; individual model requests stay bounded by the 900-char
  chunk recipe).
- `histories/r2-mem-b.json` – development history B.
- `histories/r2-mem-final.json` – the held-out FINAL CHECK history. The first
  live result on it is the final-check evidence; after adapting to its
  failures, later attempts are regression runs, not fresh held-out evidence.
- `histories/corrections.json` – the teacher correction of history F's
  brother-age fact.
- `manifest.json` – the frozen gold: per-question case type, natural question
  text, gold expected answer, expected source keys (semantic, with `:a` for
  the assistant source of the same exchange), absence flags and the
  evaluation protocol. A's plumbing check additionally uses `probe_query`
  (lexical-channel reachability with a recording embedder); B's live run uses
  the natural `question` with the real BGE-M3.
- `loader.py` – `load_history(store, history, learner_id=..., student_actor_id=...,
  teacher_actor_id=...)` inserts one archive through the real persistence
  invariants (learner/session scoping, author roles, content hash, completed
  pairs, source revisions) with explicit historical instants, applies the
  correction through the standard store correction path, and returns
  `{source_key: actual source id}` for both concrete (`r2-mem-a-s2-t49`, `:a`,
  `:u`) and semantic (`a.dog_baitik`) keys.
- `generate_histories.py` – deterministic generator (seed 20261117). The
  committed JSON files are frozen; regeneration reproduces them byte-identically
  and must never replace a frozen file during a round.

## Source-key mapping

Every dialogue block in a frozen history file has the stable concrete key
`<history>-s<session>-t<index>`; the same exchange's assistant source adds the
`:a` suffix. Authored key blocks additionally carry semantic keys resolved in
`key_block_keys` per session (e.g. `final.brother_age`). The gold manifest
references semantic keys; the loader resolves them to actual persisted source
ids at insert time, so no UUID is ever hardcoded.

## Freeze and usage rules

1. Gold answers and expected source keys never enter generation, resolver or
   extraction prompts; they are evaluation-side data only.
2. Load into a disposable database only (never the working learner database).
3. Record immutable prompts, selected source ids, answers and errors before
   scoring; compare answers semantically against sources, not by heading
   presence or substring-only scoring.
4. Label the assessment automated; the coding agent's review is not an
   independent human study.
5. Preserve the first final-check result on `r2-mem-final` per the rules above.
