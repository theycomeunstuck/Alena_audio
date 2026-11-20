"""Next round: one-pass snapshot scan, bounded message-level rerank,
fact-preserving selection, pair/correction completeness and the cumulative
chunk bound.

The recorded defects this file pins:
  - A2/A6/F4/B2/F3: the gold episode was IN the retrieved pool (fused
    positions 13-119 of 177) but never reached the model: the old batch walk
    excluded already-seen keys on every expansion call, so the store computed
    its IDF statistics and ranks on a different set each sample, and only the
    top-12 diversified hits were packed while 25-41 same-template drill
    groups outranked the gold. The recall now scans the fixed snapshot in ONE
    search call (no rescore drift, one round trip, measured latency) and adds
    a bounded message-level rerank channel inside the pool, so the one
    student sentence that answers the question is promoted above template
    crowd without any golden cues.
  - F4: the aquarium exchange was retrieved but never selected into the
    bounded prompt; the grouped depth now admits distinct material before
    deferred template members.
  - F5: the exact 96-candy passage was delivered near the end of an
    archive-ordered block and the answer quoted a different exchange; the
    packed evidence is now relevance-first and the evidence notice tells the
    answer model to separate near-identical episodes by their exact numbers
    and details.
  - codex review P2: the chunker's message and pair completions each
    measured EXCHANGE_EXTENSION from the new end and doubled the bound
    (900 -> 3257 chars); the budget is now cumulative from the window end.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime

from local_app import unified_memory_recall as recall_module
from local_app.unified_memory_recall import recall_own_conversation
from local_app.conversation_retrieval import (CHUNK_RECIPE, EXCHANGE_EXTENSION, CHUNK_SIZE,
                                              chunk_conversation_messages, _select_quotes)


class ScriptedRuntime:
    context_tokens = 32768

    def __init__(self):
        from types import SimpleNamespace
        self.settings = SimpleNamespace(chat_model="gemma4:12b", embedding_model="bge-m3",
                                        ollama_url="http://127.0.0.1:11434")

    def model_identity(self, model_name=None):
        return "recording-gemma@sha256:" + "c" * 64

    def embedding_model_identity(self):
        return "hash-embedder@sha256:" + "d" * 64

    def embed(self, texts):
        import hashlib
        vectors = []
        for text in (texts if isinstance(texts, list) else [texts]):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([byte / 255.0 for byte in digest[:8]])
        return vectors

    def check_models(self):
        return {"ready": True, "chat": "scripted", "embedding": "scripted"}


class BagOfWordsRuntime(ScriptedRuntime):
    """Deterministic embeddings with real overlap geometry: shared words share
    slots, so a message that answers the question scores a high cosine."""

    DIMS = 1024

    def embed(self, texts):
        vectors = []
        for text in (texts if isinstance(texts, list) else [texts]):
            vector = [0.0] * self.DIMS
            import hashlib
            for word in re.findall(r"\w+", str(text).casefold()):
                if len(word) < 3:
                    continue
                slot = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % self.DIMS
                vector[slot] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def _row(source_id, text, *, role="user", order=1, session="session-1", day="2026-10-01",
         paired=None, pair=None, author_role=None):
    row = {"source_id": source_id, "source_order": order, "learner_id": "child",
           "actor_id": "account", "role": role,
           "author_role": author_role or ("student" if role == "user" else "assistant"),
           "text": text, "activity_day": date.fromisoformat(day),
           "activity_timezone": "Asia/Tomsk", "event_timezone": "Asia/Tomsk",
           "created_at": datetime.fromisoformat(day + "T10:15:00+00:00"),
           "learning_session_id": session, "topic_id": "general",
           "delivery_status": "completed",
           "usage": "self_report" if role == "user" else "context_only"}
    if paired:
        row["paired_source_id"] = paired
    if pair:
        row["pair_source_id"] = pair
    return row


class DeepCrowdStore:
    """In-memory store whose search ranks a one-template crowd above the
    answer-bearing distinct chunk (the recorded A2 shape)."""

    def __init__(self, rows, ranked_keys):
        self.rows = rows
        self.ranked_keys = ranked_keys  # chunk_key order == fused rank order
        self.queries = 0
        self.exclude_args = []
        self.limit_args = []
        self.texts = {row["source_id"]: row["text"] for row in rows}
        # Optional per-channel ranks by chunk key (the recorded diag shape).
        self.vector_ranks: dict = {}
        self.text_ranks: dict = {}

    def get_own_conversation_messages(self, learner_id, actor_id, *, limit=None,
                                      active_day_limit=None, exclude_source_id=None):
        rows = [row for row in self.rows
                if exclude_source_id is None or row["source_id"] != exclude_source_id]
        return rows[:limit] if limit else rows

    def get_learner_learning_timezone(self, learner_id):
        return "Asia/Tomsk"

    def upsert_conversation_chunks(self, *args, **kwargs):
        return None

    def chunk_embedding_status(self, *args, **kwargs):
        return []

    def store_chunk_embedding(self, *args, **kwargs):
        return None

    def memory_source_correction_refs(self, learner_id, source_ids):
        return {}

    def search_lesson_plan_context(self, learner_id, query, limit=3):
        return []

    def conversation_chunk_texts(self, learner_id, actor_id, chunk_recipe, chunk_keys):
        return {key: self.texts.get(key, "") for key in chunk_keys}

    def conversation_chunk_embeddings(self, learner_id, actor_id, chunk_recipe, chunk_keys):
        return {}

    def search_conversation_chunks(self, learner_id, actor_id, query, *, chunk_recipe,
                                   embedding_identity=None, query_vector=None, limit=12,
                                   session_id=None, time_filter=None, rrf_k=60,
                                   min_similarity=0.20, candidate_chunk_keys=None,
                                   exclude_chunk_keys=None):
        self.queries += 1
        self.exclude_args.append(exclude_chunk_keys)
        self.limit_args.append(limit)
        keys = list(self.ranked_keys)[:limit]
        # Every chunk maps to exactly one message, with the span the recall
        # needs for the message-level rerank. Channel ranks mirror the
        # recorded diag shape: the crowd's vector ranks lead, the gold chunk's
        # lexical rank can be excellent while its fused position is deep.
        return [{"chunk_key": key, "source_ids": [key],
                 "spans": [{"source_id": key, "role": "user", "start": 0,
                            "end": len(self.texts.get(key, "")),
                            "chunk_start": 0, "chunk_end": len(self.texts.get(key, ""))}],
                 "score": 1.0,
                 "vector_rank": self.vector_ranks.get(key),
                 "text_rank": self.text_ranks.get(key)}
                for key in keys]


CROWD_TEMPLATE = ("Задача про деньги: у ученика {a} рублей. Он купил 3 булочки по 7 рублей за каждую. "
                  "Сколько рублей осталось на проезд домой? Объясни решение.\n\n"
                  "3 × 7 = 21 рублей за булочки, {a} – 21 = {b}.\n\n"
                  "Правильно: на булочки ушло 21 рублей, осталось {b}. Можно проверить: {b} + 21 = {a}.")
GOLD_TEXT = ("Я часто путаюсь: «во сколько раз больше» и «на сколько больше». "
             "Помоги разобраться на числах 6 и 2.")


def _crowd_archive(crowd_count=170):
    rows = []
    for index in range(crowd_count):
        rows.append(_row(f"crowd-{index}",
                         CROWD_TEMPLATE.format(a=50 + index, b=29 + index),
                         order=index + 1))
    gold_order = crowd_count + 1
    rows.append(_row("gold-times-error", GOLD_TEXT, order=gold_order))
    return rows


def test_one_pass_snapshot_scan_reaches_a_deep_distinct_chunk():
    # The gold chunk's fused rank is 150 of 171: one scan over the fixed
    # snapshot returns the whole ranked pool (no rescoring between samples,
    # no exclude keys), and the bounded message-level rerank promotes the
    # answer-bearing message above the drill crowd.
    rows = _crowd_archive(170)
    ranked = [f"crowd-{index}" for index in range(150)] + ["gold-times-error"] + \
             [f"crowd-{index}" for index in range(150, 170)]
    store = DeepCrowdStore(rows, ranked)
    result = recall_own_conversation(store, BagOfWordsRuntime(), "child", "account",
                                     "во сколько раз больше и на сколько больше",
                                     budget_bytes=24000)
    delivered = {str(quote.get("source_id")) for quote in result.get("sources", [])}
    assert "gold-times-error" in delivered, \
        "the answer-bearing chunk must reach the model through the one-pass scan and rerank"
    assert "gold-times-error" in set(result["retrieved_source_ids"])
    diagnostics = result["diagnostics"]
    assert diagnostics["expansion_queries"] == 1, "the snapshot scan is a single search call"
    assert diagnostics["search_queries"] == 1
    assert diagnostics["candidate_rows_examined"] == len(ranked), \
        "the whole fixed snapshot was examined in that one call"
    assert store.queries == diagnostics["expansion_queries"], \
        "the recorded query count is the actual cost"
    assert all(excluded is None for excluded in store.exclude_args), \
        "the scan never excludes seen keys: ranking always uses the fixed snapshot"
    assert diagnostics["search_latency_ms"] >= 0.0, "the measured search latency is recorded"
    assert diagnostics["span_rerank"]["applied"] is True, \
        "the message-level rerank ran with embeddings available"
    assert diagnostics["span_rerank"]["messages_embedded"] >= 1


def test_scan_without_usable_overlap_still_delivers_by_fused_order():
    # Same archive, but the scripted embeddings carry no overlap geometry
    # (near-orthogonal hash vectors): the rerank cannot promote anything and
    # degrades honestly. The pool itself is unchanged by the one-pass scan.
    rows = _crowd_archive(170)
    ranked = [f"crowd-{index}" for index in range(150)] + ["gold-times-error"] + \
             [f"crowd-{index}" for index in range(150, 170)]
    store = DeepCrowdStore(rows, ranked)
    result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                     "во сколько раз больше и на сколько больше",
                                     budget_bytes=24000)
    diagnostics = result["diagnostics"]
    assert diagnostics["expansion_queries"] == 1
    assert diagnostics["candidate_rows_examined"] == len(ranked)
    assert "gold-times-error" in set(result["retrieved_source_ids"]), \
        "the deep gold stays inside the one-pass pool even when the rerank is neutral"


class ConstantEmbedRuntime(ScriptedRuntime):
    """Every text maps to the same vector: the rerank channel is neutral
    (all cosines tie, ranks fall back to the deterministic key order)."""

    def embed(self, texts):
        count = len(texts) if isinstance(texts, list) else 1
        return [[1.0] + [0.0] * 1023 for _ in range(count)]


def test_deep_distinct_chunk_without_a_promotion_signal_keeps_the_scan_order():
    # The recorded A2 shape, 30 pairwise-distinct episodes outranking the
    # gold, with a neutral embedding (all cosines tie): no message can be
    # separated, so no promotion happens and the scan order stands.
    rows = []
    for index in range(30):
        letter = chr(ord("а") + index)
        rows.append(_row(f"distinct-{index}",
                         f"Разбор номер {index}: " + (letter * 40) +
                         " обсудили и записали. " + (letter * 40),
                         order=index + 1))
    rows.append(_row("gold-times-error", GOLD_TEXT, order=31))
    ranked = [f"distinct-{index}" for index in range(30)] + ["gold-times-error"]
    store = DeepCrowdStore(rows, ranked)
    result = recall_own_conversation(store, ConstantEmbedRuntime(), "child", "account",
                                     "совершенно иной вопрос про погоду",
                                     budget_bytes=24000)
    delivered = {str(quote.get("source_id")) for quote in result.get("sources", [])}
    assert "gold-times-error" not in delivered, \
        "with no separable signal the 31st-ranked chunk stays outside the packed depth"
    assert "gold-times-error" in set(result["retrieved_source_ids"]), \
        "the gold chunk is still inside the examined pool"


def test_message_rerank_promotes_the_needed_episode_under_a_distinct_crowd():
    # The recorded A2 repair: 30 distinct episodes outrank the gold chunk in
    # the fused scan order, but the gold MESSAGE shares the question's exact
    # topic words while the crowd does not. The bounded band promotion moves
    # the answer-bearing chunk into the delivered depth without any golden
    # cues: the only signal is the message-level cosine to the query.
    rows = []
    for index in range(30):
        letter = chr(ord("а") + index)
        rows.append(_row(f"distinct-{index}",
                         f"Разбор номер {index}: " + (letter * 40) +
                         " обсудили и записали. " + (letter * 40),
                         order=index + 1))
    rows.append(_row("gold-times-error", GOLD_TEXT, order=31))
    ranked = [f"distinct-{index}" for index in range(30)] + ["gold-times-error"]
    store = DeepCrowdStore(rows, ranked)
    result = recall_own_conversation(store, BagOfWordsRuntime(), "child", "account",
                                     "во сколько раз больше и на сколько больше",
                                     budget_bytes=24000)
    delivered = {str(quote.get("source_id")) for quote in result.get("sources", [])}
    assert "gold-times-error" in delivered, \
        "the promoted answer-bearing chunk reaches the packed depth"
    diagnostics = result["diagnostics"]
    rerank = diagnostics["span_rerank"]
    assert rerank["applied"] is True
    assert rerank["promoted_count"] >= 1
    assert rerank["top_cosine"] >= rerank["min_cosine_floor"]
    assert diagnostics["channels"][0]["chunk_key"] == "gold-times-error", \
        "the promoted episode leads the packed evidence"


def test_channel_guarantee_recovers_a_lexically_confident_deep_chunk():
    # The recorded A2 shape: 30 distinct episodes outrank the gold in the
    # fused order and the rerank signal is neutral, but the gold chunk is the
    # lexical channel's rank-3 result (the drill crowd's vector ranks are
    # dense noise inside the template family). The channel guarantee extends
    # the packed window so one channel's confident evidence reaches the
    # model.
    rows = []
    for index in range(30):
        letter = chr(ord("а") + index)
        rows.append(_row(f"distinct-{index}",
                         f"Разбор номер {index}: " + (letter * 40) +
                         " обсудили и записали. " + (letter * 40),
                         order=index + 1))
    rows.append(_row("gold-times-error", GOLD_TEXT, order=31))
    ranked = [f"distinct-{index}" for index in range(30)] + ["gold-times-error"]
    store = DeepCrowdStore(rows, ranked)
    store.text_ranks = {f"distinct-{index}": 10 + index for index in range(30)}
    store.text_ranks["gold-times-error"] = 3
    store.vector_ranks = {f"distinct-{index}": index + 1 for index in range(30)}
    store.vector_ranks["gold-times-error"] = 49
    result = recall_own_conversation(store, ConstantEmbedRuntime(), "child", "account",
                                     "совершенно иной вопрос про погоду",
                                     budget_bytes=24000)
    delivered = {str(quote.get("source_id")) for quote in result.get("sources", [])}
    assert "gold-times-error" in delivered,         "the lexical channel's rank-3 chunk reaches the packed evidence"
    assert "gold-times-error" in result["diagnostics"].get("channel_guaranteed_keys", []),         "the guarantee records which deep chunk it recovered"
    assert len(result["diagnostics"].get("channel_guaranteed_keys", [])) <= 6,         "the appended tail stays bounded"


def test_expansion_cost_is_bounded_for_an_all_repetitive_archive():
    # No distinct material anywhere: one bounded scan examines the whole
    # snapshot (capped), the delivered depth is still filled (deferred
    # members backfill), and the cost is recorded.
    rows = _crowd_archive(300)[:300]
    ranked = [f"crowd-{index}" for index in range(300)]
    store = DeepCrowdStore(rows, ranked)
    result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                     "задача про деньги и булочки",
                                     budget_bytes=24000)
    diagnostics = result["diagnostics"]
    assert diagnostics["candidate_rows_examined"] <= recall_module.MAX_CANDIDATE_ROWS
    assert diagnostics["expansion_queries"] == 1
    assert store.limit_args[-1] <= 1000, "the bounded scan cap holds"
    assert result["sources"], "a fully repetitive archive still fills the delivered depth"


def test_distinct_material_never_triggers_expansion_queries():
    # A small fully distinct archive: one search, no expansion cost.
    rows = [_row(f"note-{index}", f"Совершенно другой разбор номер {index} про корабль Сокол.",
                 order=index + 1) for index in range(6)]
    store = DeepCrowdStore(rows, [f"note-{index}" for index in range(6)])
    result = recall_own_conversation(store, ScriptedRuntime(), "child", "account",
                                     "корабль Сокол разбор",
                                     budget_bytes=24000)
    assert result["diagnostics"]["expansion_queries"] == 1


class TestPairCompletenessUnderBudget:
    def _exchange_rows(self):
        # question -> short ambiguous answer -> follow-up explanation
        question = _row("q", "Вопрос по дробям: какая дробь больше – 2/5 или 3/5?",
                        role="assistant", order=1)
        reply = _row("a", "хз", order=2, paired="expl")
        explanation = _row("expl", "Давай разберём: знаменатель одинаковый, значит сравниваем "
                           "числители: 3 больше 2, поэтому 3/5 больше. "
                           "Правило: при одинаковых знаменателях больше та дробь, "
                           "у которой числитель больше.",
                           role="assistant", order=3, pair="a")
        return [question, reply, explanation]

    def test_short_ambiguous_answer_stays_with_its_pair(self):
        rows = self._exchange_rows()
        chunk_hit = {"chunk_key": "chunk-exchange", "source_ids": ["q", "a", "expl"],
                     "spans": [{"source_id": "q", "role": "assistant", "start": 0,
                                "end": len(rows[0]["text"]), "chunk_start": 0, "chunk_end": 44},
                               {"source_id": "a", "role": "user", "start": 0, "end": 2,
                                "chunk_start": 44, "chunk_end": 46},
                               {"source_id": "expl", "role": "assistant", "start": 0,
                                "end": len(rows[2]["text"]), "chunk_start": 46, "chunk_end": 200}],
                     "score": 1.0, "chunk_id": "chunk-exchange"}
        # The budget is exactly the size of the complete exchange: the packed
        # evidence keeps the question, the ambiguous reply and its explanation.
        whole_size = len(_select_quotes([chunk_hit], rows, 999999)[0].encode("utf-8"))
        text, quotes, selected, diag = _select_quotes([chunk_hit], rows, whole_size)
        delivered = {quote["source_id"] for quote in quotes}
        assert {"q", "a", "expl"} <= delivered, \
            "the ambiguous reply, its question and the explanation travel together"
        assert diag["pair_incomplete_source_ids"] == []
        assert "хз" in text and "3 больше 2" in text

    def test_squeezed_mode_prefers_the_pair_and_records_incompleteness(self):
        rows = self._exchange_rows()
        full_hit = {"chunk_key": "chunk-exchange", "source_ids": ["q", "a", "expl"],
                    "spans": [{"source_id": "q", "role": "assistant", "start": 0,
                               "end": len(rows[0]["text"]), "chunk_start": 0, "chunk_end": 44},
                              {"source_id": "a", "role": "user", "start": 0, "end": 2,
                               "chunk_start": 44, "chunk_end": 46},
                              {"source_id": "expl", "role": "assistant", "start": 0,
                               "end": len(rows[2]["text"]), "chunk_start": 46, "chunk_end": 200}],
                    "score": 1.0, "chunk_id": "chunk-exchange"}
        complete_size = len(_select_quotes([full_hit], rows, 999999)[0].encode("utf-8"))
        # The window of this hit covers only the reply; its pair must be
        # completed by the paired assistant message when the budget allows.
        chunk_hit = {"chunk_key": "chunk-exchange", "source_ids": ["q", "a", "expl"],
                     "spans": [{"source_id": "a", "role": "user", "start": 0, "end": 2,
                                "chunk_start": 0, "chunk_end": 2}],
                     "score": 1.0, "chunk_id": "chunk-exchange"}
        # Measure the bare reply without its pair link: pair completion would
        # otherwise always include the paired explanation.
        bare_rows = [rows[0], dict(rows[1]), dict(rows[2])]
        bare_rows[1] = {key: value for key, value in bare_rows[1].items() if key != "paired_source_id"}
        bare_hit = {"chunk_key": "chunk-exchange", "source_ids": ["a"],
                    "spans": [{"source_id": "a", "role": "user", "start": 0, "end": 2,
                               "chunk_start": 0, "chunk_end": 2}],
                    "score": 1.0, "chunk_id": "chunk-exchange"}
        bare_size = len(_select_quotes([bare_hit], bare_rows, 999999)[0].encode("utf-8"))
        # A budget where only the bare reply fits: the incomplete pair must be
        # visible in the selection diagnostics, never lost silently.
        budget = bare_size + 25
        assert budget < complete_size
        text, quotes, selected, diag = _select_quotes([chunk_hit], rows, budget)
        delivered = {quote["source_id"] for quote in quotes}
        assert "a" in delivered, "the learner's recorded words are not silently dropped"
        assert diag["pair_incomplete_source_ids"] == ["a"], \
            "a delivered reply without its paired explanation is reported"


class TestRelevanceFirstPacking:
    def test_the_best_ranked_chunk_leads_the_packed_evidence(self):
        rows = [_row("late-exact", "Вопрос ровно в такой формулировке: раздели 96 конфет между "
                     "четырьмя друзьями поровну. Сколько получит каждый?", order=10),
                _row("late-exact-a", "24 конфеты, потому что 96 : 4 = 24.", role="assistant",
                     author_role="assistant", order=11),
                _row("early-other", "Совершенно другой разбор: рисовали корабль Сокол.", order=1)]
        # Rank 0: the exact passage (found by search); rank 1: the early
        # unrelated note. The archive order would put the early note first.
        hits = [{"chunk_key": "k0", "source_ids": ["late-exact", "late-exact-a"], "spans": [],
                 "score": 2.0, "chunk_id": "k0"},
                {"chunk_key": "k1", "source_ids": ["early-other"], "spans": [],
                 "score": 1.0, "chunk_id": "k1"}]
        text, quotes, selected, diag = _select_quotes(hits, rows, 8000)
        assert quotes[0]["source_id"] == "late-exact", \
            "the exact passage leads the evidence block"
        assert text.index("раздели 96 конфет") < text.index("корабль Сокол"), \
            "the packed prompt leads with the best-ranked material"
        assert quotes[0]["quote_alias"] == "Q1"


class TestCorrectionCompleteness:
    def _rows(self):
        original = _row("orig", "Сверим запись: брат старше ученика на пять лет.", order=1)
        corrected = dict(_row("corr", "Зафиксировали исправление: брат старше ученика на семь лет.",
                              role="assistant", author_role="assistant", order=2))
        # The correcting source carries the link to the original it corrects
        # (as the correction refs provide evaluation-side and store-side).
        corrected["corrects_source_id"] = "orig"
        original["superseded_by"] = "corr"
        return [original, corrected]

    def test_a_delivered_correction_pulls_its_original_into_the_evidence(self):
        rows = self._rows()
        hit = {"chunk_key": "k1", "source_ids": ["corr"],
               "spans": [{"source_id": "corr", "role": "assistant", "start": 0,
                          "end": len(rows[1]["text"]), "chunk_start": 0,
                          "chunk_end": len(rows[1]["text"])}],
               "score": 1.0, "chunk_id": "k1"}
        text, quotes, selected, diag = _select_quotes([hit], rows, 8000,
                                                      correction_refs={"corr": "orig"})
        delivered = {quote["source_id"] for quote in quotes}
        assert {"corr", "orig"} <= delivered, \
            "the correction travels with the original it corrects"
        assert diag["correction_incomplete_source_ids"] == []

    def test_the_original_without_its_correction_is_reported_when_budget_blocks(self):
        rows = self._rows()
        hit = {"chunk_key": "k1", "source_ids": ["orig"],
               "spans": [{"source_id": "orig", "role": "user", "start": 0,
                          "end": len(rows[0]["text"]), "chunk_start": 0,
                          "chunk_end": len(rows[0]["text"])}],
               "score": 1.0, "chunk_id": "k1"}
        # A budget that fits only the bare original: the correction must be
        # visible as an incomplete link, never silently absent.
        bare_size = len(_select_quotes([hit], [rows[0]], 999999)[0].encode("utf-8"))
        text, quotes, selected, diag = _select_quotes([hit], rows, bare_size,
                                                      correction_refs={"corr": "orig"})
        delivered = {quote["source_id"] for quote in quotes}
        assert "orig" in delivered, "the recorded words are not silently dropped"
        assert diag["correction_incomplete_source_ids"] == ["orig"], \
            "a delivered original without its correction is reported"


class TestNearEpisodeDisambiguation:
    def test_the_notice_tells_the_answer_model_to_separate_near_episodes(self):
        # The recorded F5 shape: two candy exercises share the topic; only
        # their exact numbers distinguish the requested episode from the
        # distractor. The evidence notice must carry that instruction.
        from local_app.conversation_retrieval import _NOTICE
        assert "различай" in _NOTICE and "числам" in _NOTICE, \
            "the notice instructs separating near-identical episodes by exact details"


class TestCumulativeExtensionBound:
    def test_message_and_pair_extensions_share_one_budget(self):
        # The codex-review case: a 2080-char reply followed by a 1150-char
        # explanation must not grow a 900-char window into a 3257-char chunk.
        rows = []
        rows.append(_row("long-user", "Р" * 2080, order=1))
        rows.append(_row("long-assistant", "А" * 1150, role="assistant",
                         author_role="assistant", order=2))
        chunks = chunk_conversation_messages(rows, "child", "account")
        assert chunks, "the long exchange is chunked"
        assert max(len(chunk["text"]) for chunk in chunks) <= CHUNK_SIZE + EXCHANGE_EXTENSION, \
            "the extension budget is cumulative, not doubled"
        # The cut completes the user message itself (1194 <= 1200 from the
        # window end) but the pair completion no longer doubles the chunk.
        first = chunks[0]
        assert len(first["text"]) > CHUNK_SIZE, "the cut still completes the message"

    def test_short_ambiguous_pair_still_completes_in_one_chunk(self):
        rows = []
        for index in range(3):
            rows.append(_row(f"filler-user-{index}", "Продолжаем: " + "п" * 300, order=index * 2 + 1))
            rows.append(_row(f"filler-assistant-{index}", "Верно, продолжаем: " + "в" * 300,
                             role="assistant", order=index * 2 + 2))
        rows.append(_row("q", "Вопрос по дробям: какая дробь больше – 2/5 или 3/5?",
                         role="assistant", author_role="assistant",
                         order=len(rows) + 1))
        rows.append(_row("a", "хз", order=len(rows) + 1, paired="expl"))
        rows.append(_row("expl", "Давай разберём: знаменатель одинаковый, значит сравниваем "
                         "числители: 3 больше 2, поэтому 3/5 больше.",
                         role="assistant", order=len(rows) + 1,
                         pair="a"))
        chunks = chunk_conversation_messages(rows, "child", "account")
        pair_chunks = [chunk for chunk in chunks
                       if "a" in chunk["source_ids"] or "expl" in chunk["source_ids"]]
        assert pair_chunks, "the short exchange is chunked"
        for chunk in pair_chunks:
            both = ("a" in chunk["source_ids"]) + ("expl" in chunk["source_ids"])
            assert both == 2, "a short completed exchange is never split"
        assert any("3 больше 2" in chunk["text"] for chunk in pair_chunks)
