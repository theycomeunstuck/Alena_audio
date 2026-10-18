"""Finite paid adapter check and synthetic demo after local research generation."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    from research_memory.__main__ import _provider_client
    from scripts.check_memory_external import run_check
    from local_app.config import Settings
    from local_app.memory_service import MemoryWorker
    from local_app.runtime import OllamaRuntime
    from local_app.store import Store

    root = ROOT / ".local/research_memory/final_v4_20261018"
    state_path = root / "integration_state.json"
    if state_path.exists():
        raise RuntimeError("Integration package already started; inspect durable state before any further call")
    state = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_local_generation"}
    write(state_path, state)
    deadline = time.monotonic() + 21600
    try:
        while True:
            research = json.loads((root / "state.json").read_text(encoding="utf-8"))
            if research["status"] not in {"starting", "dev_generation", "heldout_generation"}:
                if research["status"] == "failed" and "heldout" not in research:
                    raise RuntimeError("Research generation failed; integration is not started automatically")
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Finite integration wait expired")
            time.sleep(2)
        state["status"] = "external_check_started"
        write(state_path, state)
        config = ROOT / ".local/research_memory/budget_v4_20261017/integration.json"
        client = _provider_client(config)
        if client.preflight()["calls_reserved"]:
            raise RuntimeError("Integration ledger already contains calls; automatic repetition is forbidden")
        external = run_check(config, root / "external_integration.json", execute=True, requests=5)
        state["external"] = {key: external.get(key) for key in ("passed", "external_answers", "local_fallback_answers", "new_ledger_calls", "new_charged_or_reserved_usd", "errors")}
        state["status"] = "synthetic_demo"
        write(state_path, state)
        settings = Settings.from_env()
        store = Store(settings.database_url)
        demo = json.loads((ROOT / ".local/memory-verification/demo.json").read_text(encoding="utf-8"))
        job = store.get_memory_job(demo["job_id"])
        if job["status"] == "queued":
            with store._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT count(*) AS count FROM teachcopilot.memory_jobs WHERE status IN ('queued','running') AND id<>%s", (demo["job_id"],))
                if cur.fetchone()["count"]:
                    raise RuntimeError("Other application jobs exist; demo worker requires explicit operator selection")
            MemoryWorker(store, OllamaRuntime(settings)).run_once()
            job = store.get_memory_job(demo["job_id"])
        state["demo"] = job
        state["status"] = "completed" if external.get("passed") and job["status"] == "completed" else "completed_with_limitations"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)
        write(root / "INTEGRATION_PROCESSING.json", {"status": state["status"], "finished_at_utc": state["finished_at_utc"],
            "external_report": "external_integration.json", "demo_job_id": demo["job_id"],
            "human_content_review": "not_performed", "automatic_retries": 0})
        print(state["status"], flush=True)
        return 0
    except Exception as exc:
        state.update(status="failed", safe_error=type(exc).__name__, error=str(exc),
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
