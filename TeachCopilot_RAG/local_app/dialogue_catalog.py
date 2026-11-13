"""Current topic facets over own conversation chunks; labels are retrieval hints.

Original messages are never updated or deleted by classification. BGE-M3's
existing vectors provide a lightweight prototype classifier; scores are cosine
similarities, not calibrated confidence probabilities.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

from psycopg2.extras import Json

from .conversation_retrieval import (CHUNK_RECIPE, ConversationEmbedder, _hash, chunk_conversation_messages,
                                     conversation_snapshot_fingerprint, read_own_messages, unit_vector,
                                     ConversationRetrievalUnavailable)


CLASSIFICATION_METHOD = "bge-m3-prototype-v1"
MIN_TOPIC_SIMILARITY = 0.35
MIN_TOPIC_MARGIN = 0.02
SECOND_TOPIC_DISTANCE = 0.08
TOPICS = (
    {"topic_id": "learning", "label": "Учебные задания и объяснения", "examples": (
        "Я решал задачу по математике, считал дроби и проверял ответ.",
        "Объясни школьное задание, правило чтения или написания слова.",
        "На уроке было трудно понять способ решения, я попросил подсказку.")},
    {"topic_id": "hobbies", "label": "Игры и увлечения", "examples": (
        "Мне нравится рисовать корабли, играть в шахматы и собирать конструктор.",
        "Мы обсуждали игру, любимую книгу, музыку или спортивную тренировку.",
        "Я не люблю эту игру, но хочу рассказать про своё увлечение.")},
    {"topic_id": "events", "label": "События и рассказы", "examples": (
        "Вчера я ездил с семьёй в гости и рассказывал о прогулке.",
        "На выходных случилось интересное событие, мы встретились с друзьями.",
        "Я придумал историю про героя, который построил синий маяк.")},
    {"topic_id": "wellbeing", "label": "Самочувствие со слов ребёнка", "examples": (
        "Сегодня я устал, мне грустно и не хочется заниматься.",
        "Я радуюсь и волнуюсь перед занятием, расскажу как себя чувствую.",
        "У меня болела голова, я попросил перерыв и попить воды.")},
    {"topic_id": "plans", "label": "Планы и просьбы", "examples": (
        "Завтра я хочу продолжить рисунок и потом закончить задачу.",
        "Можно сменить упражнение, сделать перерыв или выбрать другой пример?",
        "Я планирую поездку и прошу напомнить о моём замысле.")},
    {"topic_id": "other", "label": "Общение и прочее", "examples": (
        "Привет, спасибо, давай познакомимся и поговорим.",
        "Не знаю, что рассказать, или уточняю значение твоего вопроса.",
        "Короткий ответ без достаточного контекста для общей темы.")},
)


class DialogueCatalogConflict(ValueError):
    """The source snapshot changed while a replacement catalog was being built."""


def classify_chunks(chunks, vectors, embedder):
    """Classify existing vectors against fixed examples, with an unknown state."""
    if len(chunks) != len(vectors):
        raise ValueError("Every catalog chunk needs its own embedding")
    prototypes = [example for topic in TOPICS for example in topic["examples"]]
    raw_prototypes = embedder.source_vectors(prototypes)
    if any(vector is None for vector in raw_prototypes):
        raise ConversationRetrievalUnavailable("A catalog prototype text could not be embedded")
    prototype_vectors = [unit_vector(vector) for vector in raw_prototypes]
    metadata = {"method": CLASSIFICATION_METHOD, "model_identity": embedder.identity,
                "chunk_recipe": CHUNK_RECIPE, "score_kind": "cosine_similarity",
                "threshold_status": "engineering_defaults_not_heldout_calibrated",
                "min_similarity": MIN_TOPIC_SIMILARITY, "min_margin": MIN_TOPIC_MARGIN,
                "second_topic_distance": SECOND_TOPIC_DISTANCE,
                "taxonomy": [{"topic_id": topic["topic_id"], "label": topic["label"]} for topic in TOPICS]}
    metadata["recipe_id"] = _hash({**metadata, "prototype_texts": prototypes,
        "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    classified = []
    for chunk, raw_vector in zip(chunks, vectors):
        if raw_vector is None:  # the model cannot embed this text: unknown topic, still part of the archive
            classified.append({key: value for key, value in chunk.items() if key != "text"} | {
                "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
                "labels": [], "topic_scores": {}, "unknown": True, "embedding_unavailable": True})
            continue
        vector, scores, offset = unit_vector(raw_vector), {}, 0
        for topic in TOPICS:
            group = prototype_vectors[offset:offset + len(topic["examples"])]
            # A thematic mention is classified independently of its polarity.
            scores[topic["topic_id"]] = max(sum(left * right for left, right in zip(vector, example)) for example in group)
            offset += len(group)
        ranked = sorted(scores, key=lambda topic: (-scores[topic], topic))
        top, second = scores[ranked[0]], scores[ranked[1]]
        labels = []
        if top >= MIN_TOPIC_SIMILARITY and top - second >= MIN_TOPIC_MARGIN:
            labels.append(ranked[0])
            if second >= MIN_TOPIC_SIMILARITY and top - second <= SECOND_TOPIC_DISTANCE:
                labels.append(ranked[1])
        classified.append({key: value for key, value in chunk.items() if key != "text"} | {
            "text_sha256": hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest(),
            "labels": labels, "topic_scores": {topic: round(score, 6) for topic, score in scores.items()},
            "unknown": not labels})
    embedder.check()
    return classified, metadata


def summarize_catalog(chunks, classifier, snapshot_fingerprint):
    topics = {topic["topic_id"]: {"topic_id": topic["topic_id"], "label": topic["label"],
                                 "chunk_count": 0, "source_ids": set(), "activity_days": set()} for topic in TOPICS}
    days = defaultdict(lambda: {"chunk_count": 0, "topic_ids": set()})
    unknown_count = 0
    for chunk in chunks:
        labels = chunk.get("labels") or []
        unknown_count += not labels
        for topic_id in labels:
            item = topics[topic_id]
            item["chunk_count"] += 1
            item["source_ids"].update(chunk["source_ids"])
            item["activity_days"].update(chunk["activity_days"])
        for day in chunk["activity_days"]:
            days[day]["chunk_count"] += 1
            days[day]["topic_ids"].update(labels)
    return {"classifier": classifier, "snapshot_fingerprint": snapshot_fingerprint, "chunk_count": len(chunks),
            "unknown_chunk_count": unknown_count,
            "topics": [{"topic_id": item["topic_id"], "label": item["label"], "chunk_count": item["chunk_count"],
                        "source_count": len(item["source_ids"]), "activity_days": sorted(item["activity_days"])}
                       for item in topics.values()],
            "days": [{"activity_day": day, "chunk_count": value["chunk_count"], "topic_ids": sorted(value["topic_ids"])}
                     for day, value in sorted(days.items(), reverse=True)],
            "classification_is_fact_inference": False, "topic_filter_is_mandatory": False}


def _current(store):
    with store._connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT generation,recipe_id,classifier FROM teachcopilot.dialogue_catalog_classifier WHERE singleton")
        row = cur.fetchone()
        return dict(row) if row is not None else None


def get_dialogue_catalog(store, learner_id, actor_id):
    """Read only current labels; mark an appended or changed archive as stale."""
    messages = read_own_messages(store, learner_id, actor_id)
    if not callable(getattr(store, "_connect", None)):
        return {"state": "unavailable", "topics": [], "days": [], "reason": "catalog_storage_unavailable"}
    fingerprint = conversation_snapshot_fingerprint(messages)
    with store._connect() as connection, connection.cursor() as cur:
        connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
        cur.execute("SELECT generation,recipe_id,classifier FROM teachcopilot.dialogue_catalog_classifier WHERE singleton")
        current = cur.fetchone()
        if current is None:
            return {"state": "pending", "topics": [], "days": [], "classifier": None}
        cur.execute("""SELECT generation,state,source_snapshot_fingerprint,source_watermark,summary,updated_at
            FROM teachcopilot.dialogue_catalog_scopes WHERE learner_id=%s AND actor_id=%s""", (learner_id, str(actor_id)))
        scope = cur.fetchone()
        meta = {"generation": current["generation"], "classifier": current["classifier"]}
        if scope is None or scope["generation"] != current["generation"] or scope["state"] != "ready":
            return {**meta, "state": "pending", "topics": [], "days": []}
        if scope["source_snapshot_fingerprint"] != fingerprint:
            return {**meta, "state": "stale", "topics": [], "days": [],
                    "snapshot_fingerprint": scope["source_snapshot_fingerprint"], "reason": "archive_changed"}
        return {**scope["summary"], **meta, "state": "ready", "updated_at": scope["updated_at"].isoformat()}


def rebuild_dialogue_catalog(store, runtime, learner_id, actor_id, *, classifier=None):
    """Build outside the transaction; atomically replace the only active labels.

    A recipe switch clears only derived assignments of other scopes, leaving
    them pending. An older worker cannot switch the global generation back.
    """
    if not callable(getattr(store, "_connect", None)):
        raise ConversationRetrievalUnavailable("Catalog storage is unavailable")
    initial = _current(store)
    expected_generation = initial["generation"] if initial else 0
    messages = read_own_messages(store, learner_id, actor_id)
    fingerprint = conversation_snapshot_fingerprint(messages)
    watermark = max((message.get("source_order", 0) for message in messages), default=0)
    chunks = chunk_conversation_messages(messages, learner_id, actor_id)
    embedder = ConversationEmbedder(store, runtime, learner_id, actor_id)
    vectors = embedder.source_vectors([chunk["text"] for chunk in chunks])
    classified, metadata = (classifier(chunks, vectors, embedder) if classifier is not None
                            else classify_chunks(chunks, vectors, embedder))
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("recipe_id"), str)
            or len(metadata["recipe_id"]) != 64 or len(classified) != len(chunks)
            or {chunk["chunk_id"] for chunk in classified} != {chunk["chunk_id"] for chunk in chunks}):
        raise ValueError("Classifier replacement is incomplete or has no immutable recipe")
    summary = summarize_catalog(classified, metadata, fingerprint)
    # Appends and delivery completion can both change the eligible archive.
    if conversation_snapshot_fingerprint(read_own_messages(store, learner_id, actor_id)) != fingerprint:
        raise DialogueCatalogConflict("Conversation archive changed during classification; retry the current snapshot")
    embedder.check()
    with store._connect() as connection, connection.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended('dialogue_catalog_global_current',0))")
        cur.execute("SELECT generation,recipe_id FROM teachcopilot.dialogue_catalog_classifier WHERE singleton FOR UPDATE")
        current = cur.fetchone()
        actual_generation = current["generation"] if current else 0
        if actual_generation != expected_generation:
            raise DialogueCatalogConflict("Classifier generation changed while this replacement was being built")
        changed = current is None or current["recipe_id"] != metadata["recipe_id"]
        generation = actual_generation + 1 if changed else actual_generation
        if changed:
            cur.execute("DELETE FROM teachcopilot.dialogue_catalog_chunks")
            cur.execute("""UPDATE teachcopilot.dialogue_catalog_scopes SET generation=%s,state='pending',
                source_snapshot_fingerprint=NULL,source_watermark=NULL,summary=NULL,updated_at=now()""", (generation,))
        cur.execute("""INSERT INTO teachcopilot.dialogue_catalog_classifier(singleton,generation,recipe_id,classifier)
            VALUES(TRUE,%s,%s,%s) ON CONFLICT(singleton) DO UPDATE SET generation=EXCLUDED.generation,
            recipe_id=EXCLUDED.recipe_id,classifier=EXCLUDED.classifier,updated_at=now()""",
            (generation, metadata["recipe_id"], Json(metadata)))
        cur.execute("""INSERT INTO teachcopilot.dialogue_catalog_scopes
            (learner_id,actor_id,generation,state,source_snapshot_fingerprint,source_watermark,summary)
            VALUES(%s,%s,%s,'ready',%s,%s,%s) ON CONFLICT(learner_id,actor_id) DO UPDATE SET
            generation=EXCLUDED.generation,state='ready',source_snapshot_fingerprint=EXCLUDED.source_snapshot_fingerprint,
            source_watermark=EXCLUDED.source_watermark,summary=EXCLUDED.summary,updated_at=now()""",
            (learner_id, str(actor_id), generation, fingerprint, watermark, Json(summary)))
        for chunk in classified:
            cur.execute("""INSERT INTO teachcopilot.dialogue_catalog_chunks(learner_id,actor_id,generation,chunk_id,
                text_sha256,source_ids,source_spans,activity_days,labels,topic_scores)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s::date[],%s,%s)
                ON CONFLICT(learner_id,actor_id,chunk_id) DO UPDATE SET generation=EXCLUDED.generation,
                text_sha256=EXCLUDED.text_sha256,source_ids=EXCLUDED.source_ids,source_spans=EXCLUDED.source_spans,
                activity_days=EXCLUDED.activity_days,labels=EXCLUDED.labels,topic_scores=EXCLUDED.topic_scores""",
                (learner_id, str(actor_id), generation, chunk["chunk_id"], chunk["text_sha256"], chunk["source_ids"],
                 Json(chunk["spans"]), chunk["activity_days"], chunk["labels"], Json(chunk["topic_scores"])))
        cur.execute("""DELETE FROM teachcopilot.dialogue_catalog_chunks WHERE learner_id=%s AND actor_id=%s
            AND NOT(chunk_id=ANY(%s::bpchar[]))""", (learner_id, str(actor_id), [chunk["chunk_id"] for chunk in classified]))
    return {**summary, "state": "ready", "generation": generation, "other_scopes_pending": changed}
