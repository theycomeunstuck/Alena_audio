"""Банк заданий должен покрывать темы, на которых реально стоят ученики.

Тест намеренно связывает два каталога — learner-data и knowledge-data. Если
кто-то заведёт ученика на теме без учебного материала, генератор 4 этапа начнёт
собирать задание без опоры на программу, и об этом лучше узнать здесь, а не на
занятии.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import check_material_coverage as coverage_tool  # noqa: E402
from pipeline.retrieval import KNOWLEDGE_DIR, _load_task_bank, retrieve_material  # noqa: E402
from pipeline.task_checks import check_arithmetic  # noqa: E402

COVERAGE = coverage_tool.build_coverage()


def test_every_topic_learners_work_on_has_material():
    gaps = COVERAGE.gaps
    assert not gaps, "нет учебного материала для тем: " + ", ".join(
        f"{topic} ({', '.join(who)})" for topic, who in sorted(gaps.items())
    )


def test_task_bank_uses_only_catalog_topic_ids():
    assert not COVERAGE.unknown_topics, COVERAGE.unknown_topics


def test_task_bank_grade_matches_topic_id():
    assert not COVERAGE.grade_mismatch, COVERAGE.grade_mismatch


def test_every_covered_topic_has_at_least_two_tasks():
    """Одна задача на тему — это не банк: генератору нужен выбор."""
    thin = {topic: count for topic, count in COVERAGE.material.items() if count < 2}
    assert not thin, f"слишком мало задач: {thin}"


def test_arithmetic_in_the_task_bank_adds_up():
    """Материал — эталон для генератора: ошибка здесь размножится по карточкам."""
    problems = [
        f"{item.source}/{item.topic}: {problem}"
        for item in _load_task_bank(KNOWLEDGE_DIR)
        for text in (item.content, item.answer)
        for problem in check_arithmetic(text)
    ]
    assert not problems, "\n".join(problems)


def test_retrieval_actually_returns_material_for_each_needed_topic(monkeypatch):
    """Покрытие по метаданным ничего не стоит, если поиск это не находит.

    Audit F-080: with DATABASE_URL set this used to query the real database
    and load the embedding model, and any failure there was swallowed into
    the lexical fallback. The test checks the shipped knowledge bank, so the
    pgvector source is switched off explicitly.
    """
    import pipeline.retrieval as retrieval

    def no_pgvector(*args, **kwargs):
        return []

    monkeypatch.setattr(retrieval, "_search_pgvector", no_pgvector)
    empty = []
    for topic_id in sorted(COVERAGE.needed):
        grade = "3" if ".g3." in topic_id else "4"
        found = retrieve_material(topic_id, topic_id=topic_id, grade=grade, limit=3)
        if not found:
            empty.append(topic_id)
    assert not empty, f"поиск не вернул материал по темам: {empty}"


def test_coverage_retrieval_never_touches_the_database(monkeypatch):
    """Audit F-080 guard: the coverage check must not reach pipeline.rag."""
    import pipeline.retrieval as retrieval

    def forbidden(*args, **kwargs):
        raise AssertionError("coverage test reached the pgvector source")

    monkeypatch.setattr(retrieval, "_search_pgvector", lambda *a, **k: [])
    monkeypatch.setattr("pipeline.rag.search_knowledge", forbidden, raising=False)
    topic_id = sorted(COVERAGE.needed)[0]
    assert retrieve_material(topic_id, topic_id=topic_id, grade="3" if ".g3." in topic_id else "4", limit=1)
