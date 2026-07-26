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


def test_retrieval_actually_returns_material_for_each_needed_topic():
    """Покрытие по метаданным ничего не стоит, если поиск это не находит."""
    empty = []
    for topic_id in sorted(COVERAGE.needed):
        grade = "3" if ".g3." in topic_id else "4"
        found = retrieve_material(topic_id, topic_id=topic_id, grade=grade, limit=3)
        if not found:
            empty.append(topic_id)
    assert not empty, f"поиск не вернул материал по темам: {empty}"
