"""Export learner cards from PostgreSQL into learner-data JSON files.

F-071, user decision 2026-11-22: teachcopilot.learners is the source of truth
for learner cards. learner-data/learners/*.json are the seed for a new
database (``manage_local.py import`` only inserts missing learners) and an
export target. This script writes the current database cards in the same
format, validated against the catalog, so the learner-data CLI tools can read
them:

    uv run --locked python scripts/export_learner_cards.py
    python ../learner-data/tools/progress_report.py <learner_id> --learners-dir .local/learner-export/learners

Invalid cards are reported and not written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / ".local" / "learner-export" / "learners"


def export_cards(store, out_dir: Path) -> int:
    """Write every valid card to <out_dir>/<learner_id>.json; 1 if any was invalid."""
    from pipeline.learner_context import load_learner_data_tools

    tools = load_learner_data_tools()
    catalog = store.catalog()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for row in store.list_cards():
        learner_id = row["learner_id"]
        card = store.get_card(learner_id)
        if not isinstance(card, dict) or not tools.learner_id_re.fullmatch(str(learner_id)):
            failed.append(learner_id)
            print(f"{learner_id}: not a valid card object, skipped", file=sys.stderr)
            continue
        errors = [finding.format() for finding in tools.validate_card(card, catalog, f"{learner_id}.json")
                  if finding.is_error()]
        if errors:
            failed.append(learner_id)
            print(f"{learner_id}: {len(errors)} validation error(s), not exported:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            continue
        target = out_dir / f"{learner_id}.json"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
    print(f"exported to {out_dir}; failed: {len(failed)}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export learner cards from PostgreSQL to learner-data JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    from local_app.config import Settings
    from local_app.store import Store
    return export_cards(Store(Settings.from_env().database_url), args.out)


if __name__ == "__main__":
    raise SystemExit(main())
