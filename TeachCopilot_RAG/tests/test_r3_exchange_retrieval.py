"""Round-3 follow-up: complete exchange units and near-duplicate collapse.

A fixed-window chunker could cut a completed user+assistant exchange in
half — the child's short reply («хз») delivered while the paired assistant
explanation stayed in the next (unranked) chunk, losing the answer-bearing
half (B3/F4 reproductions). The chunker now completes messages and pairs at
the cut, the chunk rows carry their recipe, and the fused recall collapses
same-template near-duplicates so distinct evidence is not crowded out.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from local_app import unified_memory_recall as recall_module
from local_app.conversation_retrieval import (CHUNK_RECIPE,
                                              chunk_conversation_messages)


def pair(index, user_text, assistant_text, *, session="opaque-session", day="2026-10-01"):
    """A completed user+assistant exchange with the shipped pair links."""
    user_id = f"opaque-user-{index}"
    assistant_id = f"opaque-assistant-{index}"
    base = {"learner_id": "child", "actor_id": "account", "topic_id": "general",
            "activity_day": date.fromisoformat(day), "activity_timezone": "Asia/Tomsk",
            "event_timezone": "Asia/Tomsk", "delivery_status": "completed",
            "learning_session_id": session}
    return [
        {**base, "source_id": user_id, "source_order": index * 2 + 1, "role": "user",
         "author_role": "student", "text": user_text, "usage": "self_report",
         "created_at": datetime.fromisoformat(day + "T10:15:00+00:00"),
         "paired_source_id": assistant_id},
        {**base, "source_id": assistant_id, "source_order": index * 2 + 2, "role": "assistant",
         "author_role": "assistant", "text": assistant_text, "usage": "context_only",
         "created_at": datetime.fromisoformat(day + "T10:16:00+00:00"),
         "pair_source_id": user_id},
    ]


class TestExchangeCompleteChunking:
    def test_pair_is_never_split_across_a_chunk_boundary(self):
        # A filler exchange positions the window so the fixed 900-char cut
        # would land between the short reply and its explanation.
        rows = []
        rows += pair(0, "ф" * 600, "Разбор: " + "а" * 600)
        rows += pair(1, "хз", "Давай разберём: знаменатель одинаковый, значит сравниваем "
                               "числители: 3 больше 2, поэтому 3/5 больше.")
        for _ in range(6):
            rows += pair(len(rows), "Продолжаем: " + "п" * 300, "Верно, продолжаем: " + "в" * 300)
        chunks = chunk_conversation_messages(rows, "child", "account")
        pair_chunks = [c for c in chunks
                       if "opaque-user-1" in c["source_ids"] or "opaque-assistant-1" in c["source_ids"]]
        assert pair_chunks, "the хз exchange must be chunked"
        for chunk in pair_chunks:
            both = ("opaque-user-1" in chunk["source_ids"]) + ("opaque-assistant-1" in chunk["source_ids"])
            assert both == 2, "a completed exchange must never be split across chunks"
        # The full explanation text reached the chunk (not a truncated excerpt).
        delivered = [c for c in pair_chunks if "opaque-assistant-1" in c["source_ids"]]
        assert any("3 больше 2" in c["text"] for c in delivered)

    def test_chunk_rows_carry_their_recipe(self):
        rows = pair(0, "хз", "Разбор дробей: сравниваем числители.")
        chunks = chunk_conversation_messages(rows, "child", "account")
        assert chunks and all(chunk.get("recipe") == CHUNK_RECIPE for chunk in chunks)
        assert CHUNK_RECIPE.endswith("v3"), "the recipe version reflects the packing change"

    def test_oversized_single_message_still_windows_with_overlap(self):
        rows = pair(0, "хз", "Длинный разбор: " + "о" * 4000)
        chunks = chunk_conversation_messages(rows, "child", "account")
        assert len(chunks) >= 2, "one oversized turn must still be windowed"
        assert all(len(chunk["text"]) <= 900 + 1200 for chunk in chunks), \
            "the exchange extension is bounded"


class TestNearDuplicateDiversification:
    """Next round: the delete-style collapse equated near-identical chunks
    with different numbers (34 vs 96 candies) and lost corrected facts.
    The detectors now decide only the near-duplicate GROUP; at most
    NEAR_DUPLICATE_DELIVERY_CAP members of a group occupy the delivered depth
    before distinct material, and deferred members are never removed."""

    def _fake_store(self, texts):
        class FakeStore:
            def __init__(self):
                self.texts = texts
                self.asked = 0

            def conversation_chunk_texts(self, learner_id, actor_id, chunk_recipe, chunk_keys):
                self.asked += 1
                return {key: self.texts.get(key, "") for key in chunk_keys}

            def conversation_chunk_embeddings(self, learner_id, actor_id, chunk_recipe, chunk_keys):
                return {}
        return FakeStore()

    def _hits(self, keys):
        return [{"chunk_key": key, "source_ids": [key], "spans": [], "score": 1.0}
                for key in keys]

    def test_verbatim_repeats_stop_crowding_and_distinct_evidence_survives(self):
        template = "Верно: промежуточный 48, финальный 35. Если промежуточный совпал, значит порядок действий соблюдён."
        texts = {f"chunk-repeat-{index}": template for index in range(4)}
        texts["chunk-distinct"] = "Я часто путаюсь: «во сколько раз больше» и «на сколько больше». Объяснение на числах 6 и 2."
        store = self._fake_store(texts)
        hits = self._hits(["chunk-repeat-0", "chunk-repeat-1", "chunk-distinct",
                           "chunk-repeat-2", "chunk-repeat-3"])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        kept_keys = [hit["chunk_key"] for hit in kept]
        assert kept_keys[:3] == ["chunk-repeat-0", "chunk-distinct", "chunk-repeat-1"], \
            "the distinct chunk is admitted before the deferred repeats"
        assert deferred == 3 and groups == 2 and admitted == 2
        assert store.asked == 1

    def test_same_exercise_with_different_numbers_is_not_a_duplicate(self):
        # The codex-review synthetic: two chunks differing only in 34 vs 96
        # must both survive — they are different answers, not repeats.
        texts = {
            "chunk-candies-34": "Вопрос ровно в такой формулировке: раздели 34 конфеты между четырьмя "
                                "друзьями поровну. Сколько получит каждый? Ответ: 8 конфет, потому что 34 : 4 = 8.",
            "chunk-candies-96": "Вопрос ровно в такой формулировке: раздели 96 конфет между четырьмя "
                                "друзьями поровну. Сколько получит каждый? Ответ: 24 конфеты, потому что 96 : 4 = 24.",
        }
        store = self._fake_store(texts)
        hits = self._hits(["chunk-candies-34", "chunk-candies-96"])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        kept_keys = [hit["chunk_key"] for hit in kept]
        assert kept_keys == ["chunk-candies-34", "chunk-candies-96"], \
            "both different-value exercises survive and backfill the depth"
        assert deferred == 1 and admitted == 1, \
            "only the top-ranked member occupies a slot first; the other backfills"

    def test_corrected_fact_is_not_collapsed_into_the_original(self):
        # Word-level correction («пять лет» → «семь лет»): digit-masked
        # trigrams see one template, but the correction must survive it.
        texts = {
            "chunk-original": "Сверим одну запись из твоей карточки: у меня записано «брат старше "
                              "ученика на пять лет». Проверим это? Спасибо за проверку, зафиксируем: брат старше на пять лет.",
            "chunk-corrected": "Сверим одну запись из твоей карточки: у меня записано «брат старше "
                               "ученика на семь лет». Проверим это? Спасибо за проверку, зафиксируем: брат старше на семь лет.",
        }
        store = self._fake_store(texts)
        hits = self._hits(["chunk-original", "chunk-corrected"])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        assert [hit["chunk_key"] for hit in kept] == ["chunk-original", "chunk-corrected"], \
            "a corrected fact survives next to the original claim"
        assert deferred == 1 and admitted == 1, \
            "the correction is deferred only behind distinct material and backfills here"

    def test_all_distinct_chunks_are_kept_unchanged(self):
        texts = {"chunk-a": "Разбирали деление с остатком на числах 17 и 5.",
                 "chunk-b": "Рисовали корабль Сокол и считали его скорость.",
                 "chunk-c": "Готовили задачу про аквариум и дроби с полным решением."}
        store = self._fake_store(texts)
        hits = self._hits(["chunk-a", "chunk-b", "chunk-c"])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        assert [hit["chunk_key"] for hit in kept] == ["chunk-a", "chunk-b", "chunk-c"]
        assert (deferred, groups, admitted) == (0, 3, 3)

    def test_delivered_depth_stays_at_the_search_limit(self):
        # Digit-variant template members are capped, not deleted; distinct
        # chunks keep their slots and the delivered depth stays SEARCH_LIMIT.
        template_texts = {f"chunk-dup-{i}": f"Верно: промежуточный {10 + i}, финальный {20 + i}. "
                          "Если промежуточный совпал, значит порядок действий соблюдён."
                          for i in range(14)}
        distinct_topics = [
            "деление с остатком на числах семнадцать и пять, проверка умножением",
            "рисовали корабль Сокол и считали его скорость в пути",
            "готовили задачу про аквариум и дроби с полным решением",
            "периметр квадрата по контуру и площадь внутри, разные величины",
            "часы и минуты: тренировка длилась два часа пятнадцать минут",
            "сложение десятков и единиц по разрядам без перехода",
            "любимая книга про динозавров и пиратский корабль",
            "плавание по вторникам и четвергам после обеда",
            "брат старше на семь лет, решаем примеры вместе",
            "мечта работать программистом и писать игры",
        ]
        import string
        distinct_texts = {f"chunk-distinct-{i}": (f"Совершенно другой разбор: {topic}. "
                          + letter * 400)
                          for i, (letter, topic) in enumerate(zip(string.ascii_lowercase[:10],
                                                                  distinct_topics))}
        texts = {**template_texts, **distinct_texts}
        store = self._fake_store(texts)
        hits = self._hits([f"chunk-dup-{i}" for i in range(14)]
                          + [f"chunk-distinct-{i}" for i in range(10)])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        kept_keys = [hit["chunk_key"] for hit in kept]
        assert len(kept_keys) == len(set(kept_keys))
        assert all(key in kept_keys for key in distinct_texts), \
            "every distinct chunk keeps a delivered slot"
        assert sum(1 for key in kept_keys if key in template_texts) == \
            len(kept_keys) - len(distinct_texts), \
            "template members only fill slots the distinct material leaves free"
        assert kept_keys[0] in template_texts, \
            "the top-ranked template member leads; the rest are deferred, not deleted"
        assert deferred == len(template_texts) - recall_module.NEAR_DUPLICATE_DELIVERY_CAP
        assert groups == 11
        assert admitted == len(distinct_texts) + recall_module.NEAR_DUPLICATE_DELIVERY_CAP

    def test_deferred_members_backfill_unfilled_delivered_slots(self):
        # A one-template archive still fills the delivered depth: deferral is
        # an ordering, never a removal.
        template = "Верно: промежуточный 48, финальный 35. Если промежуточный совпал, значит порядок действий соблюдён."
        texts = {f"chunk-repeat-{index}": template for index in range(6)}
        store = self._fake_store(texts)
        hits = self._hits([f"chunk-repeat-{index}" for index in range(6)])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        assert len(kept) == 6, "all members return when distinct material is exhausted"
        assert deferred == 5 and groups == 1 and admitted == 1

    def test_short_texts_are_never_grouped(self):
        texts = {"chunk-a": "хз", "chunk-b": "не знаю", "chunk-c": "давай продолжать"}
        store = self._fake_store(texts)
        hits = self._hits(["chunk-a", "chunk-b", "chunk-c"])
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        assert len(kept) == 3 and deferred == 0 and groups == 3


class TestEmbeddingNearDuplicateGrouping:
    """Drill-verification chunks whose windows differ only in side exercises
    are near-identical by chunk-embedding cosine even when digit-masked
    trigrams differ; the detectors put them in one group and the delivery cap
    (not deletion) bounds how much of the group reaches the depth."""

    def _store(self, texts, vectors):
        class FakeStore:
            def conversation_chunk_texts(self, learner_id, actor_id, chunk_recipe, chunk_keys):
                return {key: texts.get(key, "") for key in chunk_keys}

            def conversation_chunk_embeddings(self, learner_id, actor_id, chunk_recipe, chunk_keys):
                return {key: vectors.get(key) for key in chunk_keys}
        return FakeStore()

    def test_embedding_identical_templates_are_capped_not_deleted(self):
        import numpy as np
        base = np.zeros(64, dtype=float)
        base[0:10] = 1.0
        variants = {}
        for index in range(6):
            vector = base.copy()
            vector[10 + index] = 0.15  # small per-variant drift: different neighbors
            variants[f"chunk-perimeter-{index}"] = [float(x) for x in vector]
        distinct_vector = np.zeros(64, dtype=float)
        distinct_vector[30:40] = 1.0
        variants["chunk-aquarium"] = [float(x) for x in distinct_vector]
        texts = {key: f"Всё верно: периметр {index} см по контуру, площадь {index * index} кв. см внутри."
                 for index, key in enumerate(variants)}
        store = self._store(texts, variants)
        hits = [{"chunk_key": key, "source_ids": [key], "spans": [], "score": 1.0}
                for key in variants]
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        kept_keys = [hit["chunk_key"] for hit in kept]
        assert kept_keys[:2] == ["chunk-perimeter-0", "chunk-aquarium"], \
            "the distinct chunk outranks the deferred template members"
        assert groups == 2
        assert deferred == 5, "extra template members are deferred, not removed"
        assert len(kept) == 7, "a small pool backfills its depth after the cap"

    def test_genuinely_different_embeddings_are_kept(self):
        import numpy as np
        variants = {}
        for index in range(6):
            vector = np.zeros(64, dtype=float)
            vector[index * 6:index * 6 + 6] = 1.0
            variants[f"chunk-{index}"] = [float(x) for x in vector]
        texts = {key: f"Совершенно разные тексты номер {index}." for index, key in enumerate(variants)}
        store = self._store(texts, variants)
        hits = [{"chunk_key": key, "source_ids": [key], "spans": [], "score": 1.0}
                for key in variants]
        kept, deferred, groups, admitted = recall_module._diversify_near_duplicates(
            hits, store, "child", "account")
        assert len(kept) == 6 and deferred == 0 and groups == 6
