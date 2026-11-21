"""Run isolated PostgreSQL, policy, and optional browser checks."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Audit F-077: a hung test (for example a browser acceptance test whose server
# never comes up) used to block verification forever. The full suite takes
# about 15 minutes locally; one hour is a generous finite bound.
CHECK_TIMEOUT_SECONDS = 3600


def run_check(command: list[str], *, env, timeout: float) -> tuple[str, int]:
    """Run one verification subprocess; a timeout is reported as exit code 124
    with the partial output, as scripts/verify_memory.py already does."""
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        output = f"Verification process exceeded its finite timeout of {timeout} s.\n"
        for value in (exc.stdout, exc.stderr):
            if value:
                output += value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return output, 124


def run_checks(include_ui: bool = False) -> dict:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import make_dsn, parse_dsn
    from local_app.config import Settings

    settings = Settings.from_env()
    try:
        params = parse_dsn(settings.database_url)
    except Exception as exc:
        raise ValueError("Invalid local database configuration") from exc
    if params.get("host") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Verification requires a loopback PostgreSQL connection")
    if include_ui and importlib.util.find_spec("playwright") is None:
        raise ValueError("Install requirements-dev-local.txt before using --ui")

    database = "teachcopilot_verify_" + uuid4().hex[:12]
    admin = psycopg2.connect(make_dsn(settings.database_url, dbname="postgres"))
    admin.autocommit = True
    report_dir = ROOT / ".local" / "verification"
    report_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        TEACHCOPILOT_TEST_DSN=make_dsn(settings.database_url, dbname=database),
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=str(ROOT),
    )
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".local" / "ms-playwright"))
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "ui_requested": include_ui, "checks": []}

    def execute(name: str, command: list[str]) -> None:
        output, returncode = run_check(command, env=env, timeout=CHECK_TIMEOUT_SECONDS)
        log = report_dir / (name + ".log")
        log.write_text(output, encoding="utf-8")
        item = {"name": name, "passed": returncode == 0, "exit_code": returncode, "log": str(log.relative_to(ROOT))}
        if name == "pytest":
            match = re.search(r"(\d+) passed", output)
            item["passed_tests"] = int(match.group(1)) if match else 0
            match = re.search(r"(\d+) skipped", output)
            item["skipped_tests"] = int(match.group(1)) if match else 0
        elif name == "learner-policy":
            match = re.search(r"Ran (\d+) tests", output)
            item["tests"] = int(match.group(1)) if match else 0
        report["checks"].append(item)
        print(name + ": " + ("passed" if item["passed"] else "failed") + " (" + item["log"] + ")", flush=True)

    with admin.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        command = [sys.executable, "-m", "pytest", "-q", "tests"]
        if not include_ui:
            command += ["--ignore=tests/test_ui_acceptance.py", "--ignore-glob=tests/*_ui.py"]
        execute("pytest", command)
        execute("learner-policy", [sys.executable, str(ROOT.parent / "learner-data" / "tools" / "selftest.py")])
    finally:
        # Only the exact randomly named database created above is removed.
        assert database.startswith("teachcopilot_verify_") and len(database) == 32
        with admin.cursor() as cursor:
            cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()", (database,))
            cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
        admin.close()
    report["passed"] = all(check["passed"] for check in report["checks"])
    (report_dir / "latest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local implementation in a disposable database.")
    parser.add_argument("--ui", action="store_true", help="also run Playwright browser acceptance tests")
    args = parser.parse_args()
    try:
        report = run_checks(args.ui)
    except Exception:
        print("Verification could not run. Check the local database and development dependencies.", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
