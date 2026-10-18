"""Finite 5/10/20-session memory load with PostgreSQL and an injected provider.

No model or public API is contacted. Provider latency and token counts are
synthetic; PostgreSQL memory reads, queue transitions and vector search are real.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values, quantile: float):
    """Linear interpolation; an empty measurement set is explicitly absent."""
    values = sorted(float(value) for value in values)
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be 0..1")
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def timing_summary(values) -> dict:
    values = list(values)
    return {"count": len(values), "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95), "max_ms": max(values) if values else None}


class ResourceSampler:
    """Runner RSS and host RAM samples; GPU samples are device-wide snapshots."""
    def __init__(self, interval=0.02):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    @staticmethod
    def gpu_snapshot() -> dict:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                     "--format=csv,noheader,nounits"], capture_output=True,
                                    text=True, timeout=3, check=True)
            rows = []
            for line in result.stdout.splitlines():
                index, used, total = [float(value.strip()) for value in line.split(",")]
                rows.append({"index": int(index), "used_mib": used, "total_mib": total})
            return {"available": True, "scope": "device-wide, may include unrelated processes", "devices": rows}
        except (OSError, subprocess.SubprocessError, ValueError):
            return {"available": False, "scope": "device-wide", "reason": "nvidia-smi unavailable"}

    def _sample(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            self.samples.append({"runner_rss_bytes": process.memory_info().rss,
                                 "host_used_ram_bytes": psutil.virtual_memory().used})
            self.stop.wait(self.interval)

    def __enter__(self):
        self.gpu_before = self.gpu_snapshot()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
        self.gpu_after = self.gpu_snapshot()

    def report(self) -> dict:
        return {
            "sample_count": len(self.samples), "sampling_interval_seconds": self.interval,
            "runner_peak_rss_bytes": max((row["runner_rss_bytes"] for row in self.samples), default=None),
            "host_peak_used_ram_bytes": max((row["host_used_ram_bytes"] for row in self.samples), default=None),
            "ram_scope": "runner process and host RAM; PostgreSQL container RSS not isolated",
            "gpu_before": self.gpu_before, "gpu_after": self.gpu_after,
            "gpu_peak_measured": False,
        }


class ControlledRuntime:
    """An injectable latency fixture, preserving every actual submitted prompt."""
    context_tokens = 8192

    def __init__(self, delay_ms=40.0):
        self.delay_seconds = delay_ms / 1000
        self.lock = threading.Lock()
        self.calls = []
        self.active = 0
        self.peak_active = 0

    def chat(self, messages, response_format=None, max_output_tokens=None, **kwargs):
        started = time.perf_counter()
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            time.sleep(self.delay_seconds)
            fields = (response_format or {}).get("properties", {})
            allowed = fields.get("evidence_turn_ids", {}).get("items", {}).get("enum", [])
            answer = {"records": []} if "records" in fields else {
                "action": "independent_check", "reason": "Предложить один новый пример и уточнить условия помощи.",
                "evidence_turn_ids": allowed[:1], "unknowns": ["Устойчивость навыка не проверена."],
                "review_required": True}
            return {"message": {"content": json.dumps(answer, ensure_ascii=False)},
                    "prompt_eval_count": 123, "eval_count": 45}
        finally:
            with self.lock:
                self.active -= 1
                self.calls.append({"messages": messages, "elapsed_ms": (time.perf_counter() - started) * 1000})


def require_disposable_dsn(dsn: str) -> None:
    from scripts.verify_memory import DATABASE_PREFIX, require_local_dsn
    params = require_local_dsn(dsn)
    name = params.get("dbname", "")
    if not name.startswith(DATABASE_PREFIX) or len(name) != len(DATABASE_PREFIX) + 12:
        raise ValueError("Use verify_memory.py to create an isolated disposable database")
    if any(character not in "0123456789abcdef" for character in name[len(DATABASE_PREFIX):]):
        raise ValueError("Disposable database suffix is invalid")


def _elapsed_ms(start, end):
    if not start or not end:
        return None
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return (end - start).total_seconds() * 1000


def _seed_sessions(store, count, topic_id):
    """SQL creates synthetic minimal fixtures only in this runner's private DB."""
    from psycopg2.extras import Json
    learners = []
    run_key = uuid4().hex[:10]
    for index in range(count):
        learner = f"load-{run_key}-{index}"
        card = {"learner_id": learner, "pseudonym": f"Synthetic {index}", "grade": 4,
                "language": "russian", "interests": [], "rag_context": {"current_goal": {"topic_id": topic_id}}}
        with store._connect() as conn, conn.cursor() as cursor:
            cursor.execute("INSERT INTO teachcopilot.learners(learner_id,pseudonym,grade,card) VALUES (%s,%s,4,%s)",
                           (learner, card["pseudonym"], Json(card)))
        session = store.create_learning_session(learner, "synthetic-load", topic_id, "session-" + learner)
        session_id = session.get("session_id") or session.get("id")
        reservation = store.reserve_chat_message(session_id, learner, "message-1",
            f"Synthetic marker {learner}. Вычислил 408 : 4 = 102. Условия помощи не указаны.", "synthetic-load")
        if not reservation.get("created"):
            raise AssertionError("new synthetic message was unexpectedly duplicated")
        store.complete_chat_message(session_id, "message-1", {
            "choices": [{"message": {"role": "assistant", "content": "Объясни запись нуля в частном."}}]})
        store.complete_learning_session(session_id, learner, "synthetic-load")
        learners.append(learner)
    return learners


def _drain(store, runtime, timeout_seconds):
    from local_app.memory_service import MemoryWorker
    completed = []
    deadline = time.monotonic() + timeout_seconds

    def work():
        worker = MemoryWorker(store, runtime)
        results = []
        while time.monotonic() < deadline:
            result = worker.run_once()
            if result is None:
                break
            results.append(result)
        return results

    with ThreadPoolExecutor(max_workers=4) as executor:
        for results in executor.map(lambda _: work(), range(4)):
            completed.extend(results)
    if time.monotonic() >= deadline:
        raise TimeoutError("finite load batch exceeded worker deadline")
    return completed


def run_wave(store, count, lane, delay_ms, timeout_seconds=60.0) -> dict:
    topic_id = "memory.synthetic.division"
    learners = _seed_sessions(store, count, topic_id)
    # Application-default updates must finish before next-step timing. Extracted
    # model records are fixture output, so this is not a semantic memory test.
    update_runtime = ControlledRuntime(delay_ms)
    updates = _drain(store, update_runtime, timeout_seconds)
    if len(updates) != count or any(row.get("status") != "completed" for row in updates):
        raise AssertionError("application session memory update did not complete")
    material = "synthetic-material-" + uuid4().hex
    model = "synthetic-two-dimensional-vector-" + uuid4().hex
    source_file = material + ".json"
    store.upsert_chunks([{"id": material, "content": "division 408 4 place value zero",
                          "source_file": source_file, "embedding": [1.0, 0.0]}], model, 2)
    measurements = {"memory_get": [], "search": [], "enqueue": []}
    lock = threading.Lock()

    def request(learner):
        started = time.perf_counter()
        snapshot = store.get_learner_memory(learner, topic_id)
        memory_ms = (time.perf_counter() - started) * 1000
        sources = snapshot.get("sources", [])
        if not sources or any(row.get("learner_id", learner) != learner for row in sources):
            raise AssertionError("missing memory or foreign learner source")
        started = time.perf_counter()
        rows = store.hybrid_search("division place value", [1.0, 0.0], {"source_file": source_file}, 1, model)
        search_ms = (time.perf_counter() - started) * 1000
        if len(rows) != 1 or rows[0]["id"] != material:
            raise AssertionError("local vector search did not return its synthetic material")
        started = time.perf_counter()
        job = store.enqueue_memory_job(learner, topic_id, "next_step", actor_id="synthetic-load",
            expected_revision=snapshot["revision"], method="tail", provider=lane,
            idempotency_key="wave-next-step")
        with lock:
            measurements["memory_get"].append(memory_ms)
            measurements["search"].append(search_ms)
            measurements["enqueue"].append((time.perf_counter() - started) * 1000)
        return job

    runtime = ControlledRuntime(delay_ms)
    errors = []
    started = time.perf_counter()
    with ResourceSampler() as resources:
        with ThreadPoolExecutor(max_workers=count) as executor:
            queued = list(executor.map(request, learners))
        _drain(store, runtime, timeout_seconds)
        results = [store.get_memory_job(row.get("job_id") or row.get("id")) for row in queued]
    elapsed = time.perf_counter() - started
    queue_ms, response_ms = [], []
    for learner, result in zip(learners, results):
        if result.get("status") != "completed":
            errors.append({"learner_id": learner, "status": result.get("status"), "error": result.get("error")})
            continue
        value = _elapsed_ms(result.get("created_at"), result.get("started_at"))
        if value is not None:
            queue_ms.append(value)
        value = _elapsed_ms(result.get("created_at"), result.get("completed_at"))
        if value is not None:
            response_ms.append(value)
        response = result.get("result") or {}
        if response.get("learner_id") != learner or not response.get("review_required"):
            errors.append({"learner_id": learner, "error": "result scope or review flag is wrong"})
        allowed = {row.get("source_id") or row.get("id") for row in store.get_learner_memory(learner, topic_id)["sources"]}
        if any(citation not in allowed for citation in response.get("evidence_turn_ids", [])):
            errors.append({"learner_id": learner, "error": "foreign or missing citation"})
    limit = 1 if lane == "local" else 4
    if runtime.peak_active > limit or len(runtime.calls) != count:
        errors.append({"error": "provider concurrency or call count violated", "peak": runtime.peak_active,
                       "expected_calls": count, "actual_calls": len(runtime.calls)})
    return {
        "logical_sessions": count, "provider_lane": lane, "provider": "controlled simulation",
        "request_pattern": "concurrent reads and enqueue, followed by a synchronized queued wave",
        "provider_delay_ms": delay_ms, "provider_calls": len(runtime.calls), "provider_peak_active": runtime.peak_active,
        "memory_update_provider_calls": len(update_runtime.calls),
        "provider_concurrency_limit": limit, "elapsed_seconds": elapsed,
        "simulated_completed_requests_per_second": (count - len(errors)) / elapsed,
        "completed": sum(row.get("status") == "completed" for row in results),
        "errors": errors, "error_count": len(errors), "passed": not errors,
        "timings": {**{key: timing_summary(values) for key, values in measurements.items()},
                    "queue_wait": timing_summary(queue_ms), "generation": timing_summary(row["elapsed_ms"] for row in runtime.calls),
                    "memory_update_generation": timing_summary(row["elapsed_ms"] for row in update_runtime.calls),
                    "queued_response": timing_summary(response_ms)},
        "resources": resources.report(),
    }


def run_load(dsn, sessions=(5, 10, 20), delay_ms=40.0) -> dict:
    from local_app.store import Store
    require_disposable_dsn(dsn)
    store = Store(dsn)
    store.migrate()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "provider": "controlled simulation", "paid_calls": 0,
        "public_api_throughput_measured": False, "model_inference_performed": False,
        "search": "real PostgreSQL hybrid/vector search with deterministic synthetic vectors; no embedding inference",
        "token_counts": "synthetic fixture values, excluded from measured token or cost claims",
        "human_content_review": "not performed", "waves": [],
    }
    # Any accidental HTTP call fails before it can leave the process.
    with patch("requests.sessions.Session.request", side_effect=AssertionError("HTTP forbidden during simulated memory load")):
        for count in sessions:
            for lane in ("local", "external"):
                wave = run_wave(store, count, lane, delay_ms)
                report["waves"].append(wave)
                print(f"sessions={count} lane={lane} completed={wave['completed']} errors={wave['error_count']}", flush=True)
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = all(row["passed"] for row in report["waves"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--provider-delay-ms", type=float, default=40.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 1 or value > 20 for value in args.sessions) or not 0 <= args.provider_delay_ms <= 1000:
        parser.error("sessions must be 1..20 and provider delay must be 0..1000 ms")
    dsn = os.getenv("TEACHCOPILOT_TEST_DSN")
    if not dsn:
        parser.error("run scripts/verify_memory.py to create the disposable database")
    try:
        report = run_load(dsn, args.sessions, args.provider_delay_ms)
    except Exception as exc:
        report = {"passed": False, "error_type": type(exc).__name__, "error": str(exc),
                  "provider": "controlled simulation", "paid_calls": 0, "public_api_throughput_measured": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
