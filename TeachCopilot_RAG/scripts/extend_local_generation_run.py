"""Apply one separately authorized 60 minute local package recovery window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_app.generation_run_budget import GenerationRunBudget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Consume the authorization and start the window now")
    args = parser.parse_args(argv)
    budget = GenerationRunBudget(args.ledger)
    result = (budget.extend_authorized(args.authorization) if args.execute else
              budget.preview_authorized_extension(args.authorization))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
