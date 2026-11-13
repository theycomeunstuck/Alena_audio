"""Sample resource usage while one named local research run is active."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.output.exists() or not args.run_name or not 1 <= args.interval <= 30:
        parser.error("choose a new output, run name and interval between 1 and 30 seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        seen = False
        started = time.monotonic()
        while True:
            targets = []
            for process in psutil.process_iter(["pid", "cmdline"]):
                try:
                    command = process.info.get("cmdline") or []
                    if any("research_memory.personal_memory_lab.experiment" in str(arg) for arg in command) \
                            and any(args.run_name in str(arg) for arg in command):
                        targets.append(process)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            if not targets and seen:
                break
            if not targets and time.monotonic() - started >= 60:
                raise RuntimeError("The named experiment process was not observed within 60 seconds")
            seen |= bool(targets)
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
            samples = []
            for process in targets:
                try:
                    samples.append({"pid": process.pid, "rss_bytes": process.memory_info().rss,
                                    "cpu_seconds": sum(process.cpu_times()[:2])})
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                "run_processes": samples, "gpu_csv_mib_mib_percent": gpu.stdout.strip(),
                "gpu_status": gpu.returncode}) + "\n")
            stream.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
