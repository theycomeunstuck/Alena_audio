"""Prepare one explicitly synthetic memory example inside the existing demo."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def seed() -> dict:
    from local_app.config import Settings
    from local_app.store import Store

    store = Store(Settings.from_env().database_url)
    store.migrate()
    learner, topic = "demo-nika-04", "math.g4.fractions.numerator_denominator"
    teacher = store.get_account(email="demo.rich.teacher@local.test")
    if not teacher or not store.get_card(learner):
        raise ValueError("Run the existing rich synthetic demo seed first")
    if not store.authorize_learner_access(teacher["id"], learner):
        raise ValueError("Synthetic demo teacher must already have learner access")
    examples = (
        "[СИНТЕТИЧЕСКОЕ ДЕМО] На первом рисунке записано 3/8. Условия помощи и авторство записи не указаны.",
        "[СИНТЕТИЧЕСКОЕ ДЕМО] На втором рисунке ученик подписал целое после вопроса преподавателя. "
        "Работа остановлена перед выбором знаменателя; новая самостоятельная проверка не проводилась.",
    )
    sources = [store.add_memory_source(learner, topic, text, str(teacher["id"]),
        approved=True, observed_on="2026-10-17", source_key=f"synthetic-memory-demo-v4-{number}")
        for number, text in enumerate(examples, 1)]
    job = store.enqueue_memory_job(learner, topic, "next_step", {}, str(teacher["id"]),
        method="tail", idempotency_key="synthetic-memory-demo-next-step-v4")
    result = {"learner_id": learner, "topic_id": topic, "job_id": job["job_id"],
              "status": job["status"], "source_ids": [source["source_id"] for source in sources],
              "human_content_review": "not_performed", "synthetic": True,
              "url": "http://127.0.0.1:8099"}
    output = ROOT / ".local/memory-verification/demo.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(seed(), ensure_ascii=False, indent=2))
