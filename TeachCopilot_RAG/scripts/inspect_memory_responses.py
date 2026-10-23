"""Read one job's local worker diagnostics without connecting to PostgreSQL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.config import Settings
from local_app.memory_response_journal import MemoryResponseJournal, ResponseJournalError, database_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--job-id")
    selector.add_argument("--call-id")
    parser.add_argument("--journal", help="defaults to LOCAL_MEMORY_RESPONSE_JOURNAL")
    parser.add_argument("--include-response", action="store_true", help="include private raw runtime response")
    args = parser.parse_args()
    settings = Settings.from_env()
    try:
        journal = MemoryResponseJournal(args.journal or settings.memory_response_journal_path,
                                         database_identity=database_fingerprint(settings.database_url))
        records = [journal.get(args.call_id)] if args.call_id else journal.list_calls(job_id=args.job_id)
    except (ResponseJournalError, KeyError):
        print("Worker response diagnostics unavailable for this database and selector.", file=sys.stderr)
        return 1
    if not args.include_response:
        records = [{key: value for key, value in record.items() if key != "raw_response"} for record in records]
    print(json.dumps({"calls": records, "automatic_replay": False}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
