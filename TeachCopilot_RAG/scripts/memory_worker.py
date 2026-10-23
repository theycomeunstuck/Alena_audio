"""Run the durable memory queue, or process a finite package with --max-jobs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

from psycopg2 import InterfaceError, OperationalError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from local_app.config import Settings
from local_app.memory_service import MemoryWorker
from local_app.runtime import OllamaRuntime
from local_app.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 runs continuously")
    parser.add_argument("--idle-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_jobs < 0 or not 0 < args.idle_seconds <= 30:
        parser.error("invalid job or polling limit")
    settings = Settings.from_env()
    worker = MemoryWorker(Store(settings.database_url), OllamaRuntime(settings))
    processed = 0
    retry_seconds = args.idle_seconds
    database_unavailable = False
    while not args.max_jobs or processed < args.max_jobs:
        try:
            result = worker.run_once()
        except (OperationalError, InterfaceError):
            # Resume queue polling, never an in-memory job or provider request.
            # Persisted leases and call markers decide whether a job can return.
            # Do not log exception text: a connection error may include a DSN.
            if args.max_jobs:
                print("Database connection unavailable; finite worker stopped without replay.",
                      file=sys.stderr, flush=True)
                return 1
            print(f"Database connection unavailable; queue polling resumes in {retry_seconds:g}s; no job replay.",
                  file=sys.stderr, flush=True)
            time.sleep(retry_seconds)
            retry_seconds = min(30.0, retry_seconds * 2)
            database_unavailable = True
            continue
        if database_unavailable:
            print("Database connection restored; queue polling resumed.", file=sys.stderr, flush=True)
        retry_seconds = args.idle_seconds
        database_unavailable = False
        if result:
            processed += 1
            print(result["job_id"], result["status"], flush=True)
        elif args.max_jobs:
            break
        else:
            time.sleep(args.idle_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
