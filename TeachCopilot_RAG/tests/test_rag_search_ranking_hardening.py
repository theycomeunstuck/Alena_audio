"""Materials search: soft preferences must stay soft, and lexical matches must survive the cosine floor."""
import pytest

from local_app.rag import RagService


class Settings:
    ollama_url = "http://127.0.0.1:11434"
    chat_model = "qwen3.5:9b"
    embedding_model = "qwen3-embedding:0.6b"
    context_tokens = 8192
    request_timeout = 180
    minimum_similarity = 0.35


class Runtime:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class Store:
    def __init__(self, rows):
        self.rows = rows

    def hybrid_search(self, query, query_embedding, filters=None, limit=5, embedding_model=None):
        return [dict(row) for row in self.rows]


def rrf_rows(topics, grades=None):
    """Rows scored like the database's reciprocal rank fusion: rank r gets 1/(60+r)."""
    return [{"id": f"m{rank}", "content": f"text {rank}", "subject_id": "math", "topic_id": topic,
             "grade": (grades or {}).get(rank, "3"), "score": 1.0 / (60 + rank), "similarity": 0.9}
            for rank, topic in enumerate(topics, 1)]


def search(rows, **kwargs):
    return RagService(Store(rows), Runtime(), Settings()).search("вопрос", subject_id="math", limit=3, **kwargs)


def test_topic_match_far_down_the_list_does_not_overtake_the_top_hit():
    topics = ["other"] * 9 + ["wanted"]                 # the topic match is only tenth
    results = search(rrf_rows(topics), topic_id="wanted")
    assert results[0]["id"] == "m1"
    assert "m10" not in [row["id"] for row in results]


def test_topic_match_still_breaks_a_near_tie():
    results = search(rrf_rows(["other", "wanted", "other"]), topic_id="wanted")
    assert [row["id"] for row in results][:2] == ["m2", "m1"]


def test_grade_preference_is_weaker_than_topic_preference():
    rows = rrf_rows(["other", "wanted", "other"], grades={3: "5"})
    by_topic = search(rows, topic_id="wanted", grade=5)
    assert by_topic[0]["id"] == "m2"                    # topic match wins over a grade-only match at rank 3


def test_grade_match_far_down_the_list_does_not_overtake_the_top_hit():
    rows = rrf_rows(["other"] * 10, grades={10: "5"})
    assert search(rows, grade=5)[0]["id"] == "m1"


@pytest.mark.parametrize("text_score,kept", [(0.30, True), (0.0, False), (None, False)])
def test_lexical_match_survives_the_cosine_floor_but_a_pure_miss_does_not(text_score, kept):
    row = {"id": "m1", "content": "Пончик", "subject_id": "math", "topic_id": "t", "grade": "3",
           "score": 0.016, "similarity": 0.10}
    if text_score is not None:
        row["text_score"] = text_score
    assert bool(search([row])) is kept


def test_confident_vector_hit_is_unaffected_by_the_lexical_exemption():
    row = {"id": "m1", "content": "x", "subject_id": "math", "topic_id": "t", "grade": "3", "score": 0.016,
           "similarity": 0.80, "text_score": 0.0}
    assert [r["id"] for r in search([row])] == ["m1"]
