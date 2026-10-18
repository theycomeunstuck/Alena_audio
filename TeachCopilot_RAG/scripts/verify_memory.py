"""Run memory integration checks and simulated load in a disposable local DB."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATABASE_PREFIX = "teachcopilot_memory_verify_"


def require_local_dsn(dsn: str) -> dict:
    from psycopg2.extensions import parse_dsn
    params = parse_dsn(dsn)
    if params.get("host") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Memory verification requires a loopback PostgreSQL connection")
    return params


def run_checks(*, sessions=(5, 10, 20), delay_ms=40.0, output_dir=None) -> dict:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn
    from local_app.config import Settings

    settings = Settings.from_env()
    require_local_dsn(settings.database_url)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    report_dir = Path(output_dir) if output_dir else ROOT / ".local" / "memory-verification" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    database = DATABASE_PREFIX + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(settings.database_url, dbname="postgres", connect_timeout=5))
    admin.autocommit = True
    env = dict(os.environ, TEACHCOPILOT_TEST_DSN=make_dsn(settings.database_url, dbname=database),
               PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT))
    report = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "database": "disposable loopback PostgreSQL", "checks": [],
        "provider": "controlled simulation", "paid_calls": 0,
        "public_api_throughput_measured": False,
    }

    def execute(name: str, command: list[str], timeout: int) -> bool:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=timeout)
            output = result.stdout + result.stderr
            code = result.returncode
        except subprocess.TimeoutExpired as exc:
            output = "Verification process exceeded its finite timeout.\n"
            for value in (exc.stdout, exc.stderr):
                if value:
                    output += value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            code = 124
        log = report_dir / (name + ".log")
        log.write_text(output, encoding="utf-8")
        report["checks"].append({"name": name, "passed": code == 0, "exit_code": code, "log": str(log)})
        print(f"{name}: {'passed' if code == 0 else 'failed'} ({log})", flush=True)
        return code == 0

    with admin.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        tests = sorted({*ROOT.glob("tests/test_*memory*.py"), *ROOT.glob("tests/test_answer_adapter.py")})
        if not tests:
            raise RuntimeError("No memory integration tests found")
        passed = execute("memory-tests", [sys.executable, "-m", "pytest", "-q", *map(str, tests)], 600)
        if passed:
            # Tests intentionally leave unique fixtures. Recreate this disposable
            # database so the load worker cannot consume a test's pending job.
            with admin.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
                cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
            execute("simulated-load", [sys.executable, str(ROOT / "scripts" / "load_memory.py"),
                    "--sessions", *map(str, sessions), "--provider-delay-ms", str(delay_ms),
                    "--output", str(report_dir / "load.json")], 600)
        else:
            report["load_status"] = "not_run_because_integration_tests_failed"
    finally:
        # This exact generated database is the only database removed by this tool.
        assert database.startswith(DATABASE_PREFIX) and len(database) == len(DATABASE_PREFIX) + 12
        with admin.cursor() as cursor:
            cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
            cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        admin.close()
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report["passed"] = bool(report["checks"]) and all(item["passed"] for item in report["checks"])
        (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--provider-delay-ms", type=float, default=40.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if any(value < 1 or value > 20 for value in args.sessions) or not 0 <= args.provider_delay_ms <= 1000:
        parser.error("sessions must be 1..20 and provider delay must be 0..1000 ms")
    try:
        report = run_checks(sessions=args.sessions, delay_ms=args.provider_delay_ms, output_dir=args.output_dir)
    except Exception as exc:
        print(f"Memory verification could not complete ({type(exc).__name__}). Check local PostgreSQL and dependencies.", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
