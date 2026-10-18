"""Three synthetic conversations through the real local chat and material search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from local_app.answer_adapter import AnswerAdapter
    from local_app.config import Settings
    from local_app.rag import RagService
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings = Settings.from_env()
    cases = [
        {"case_id": "D01", "messages": [{"role": "user", "content": "Я вчера построил железнодорожную станцию в Minecraft!"}],
         "preferences": {"interests": [], "avoided_interests": []},
         "expected": "Короткий уместный отклик на рассказ, без отказа об отсутствии материалов, выдуманной памяти, компетенций и навязанной задачи."},
        {"case_id": "D02", "messages": [{"role": "user", "content": "Люблю Minecraft, но сейчас давай без игровых историй. Как проверить 1248 : 12 = 104?"}],
         "preferences": {"interests": ["Minecraft"], "avoided_interests": ["Minecraft"]},
         "expected": "Учебная ветка, обращение к ребёнку; 104 × 12 = 1248, сюжет Minecraft не навязывается."},
        {"case_id": "D03", "messages": [
            {"role": "user", "content": "Помоги проверить 1248 : 12, я ещё не закончил."},
            {"role": "assistant", "content": "Покажи свой первый шаг."},
            {"role": "user", "content": "Кстати, вчера построил станцию в Minecraft."},
            {"role": "assistant", "content": "Здорово, какая получилась станция?"},
            {"role": "user", "content": "Вернёмся к тому делению. Получилось 104. Как проверить?"}],
         "preferences": {"interests": [], "avoided_interests": []},
         "expected": "Возврат к исходным 1248 : 12, проверка 104 × 12 = 1248; без выдуманного успеха или нового задания вместо продолжения."},
    ]

    class TracedRuntime(OllamaRuntime):
        def _post(self, endpoint, payload):
            started = time.perf_counter()
            response = super()._post(endpoint, payload)
            if endpoint == "/api/chat":
                calls.append({"endpoint": endpoint, "request": payload, "response": response,
                              "elapsed_seconds": time.perf_counter() - started})
                (out / "http_calls.json").write_text(json.dumps(calls, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return response

    calls = []
    runtime = TracedRuntime(settings)
    # Force local operation even if the process has an external adapter config.
    service = RagService(Store(settings.database_url), runtime, settings, answer_adapter=AnswerAdapter(runtime))
    prereg = {"created_at": datetime.now(timezone.utc).isoformat(), "data_origin": "synthetic_unreviewed",
              "model_identity": runtime.model_identity(settings.chat_model), "manifest": runtime.manifest(),
              "seed": 42, "temperature": 0, "think": False, "cases": cases,
              "attempts_per_case": 1, "external_provider_disabled": True,
              "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ("local_app/runtime.py", "local_app/rag.py", "local_app/answer_adapter.py")}}
    (out / "preregistered.json").write_text(json.dumps(prereg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for case in cases:
        started = time.perf_counter()
        row = {"case_id": case["case_id"], "status": "failed"}
        try:
            row["response"] = service.answer(case["messages"], profile={"grade": 4}, subject_id="math",
                topic_id="math.g4.numbers.division_by_1_2_digit", grade=4, preference_context=case["preferences"])
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if getattr(exc, "raw_response", None) is not None:
                row["raw_response"] = exc.raw_response
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
        (out / (case["case_id"] + ".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(case["case_id"] + ": " + row["status"], flush=True)
    report = {"status": "completed", "technical_ok": sum(r["status"] == "ok" for r in rows),
              "cases": len(rows), "generative_calls": len(calls), "paid_provider_calls": 0,
              "semantic_assessment": "pending_separate_review", "results": rows}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["technical_ok"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
