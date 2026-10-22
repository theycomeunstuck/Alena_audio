"""Finite additional assessment of existing answers; paid phases require --execute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_assessment_v5.workflow import freeze, prepare, report, run_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "dev", "complete-dev", "freeze", "calibrate", "assess", "report"))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(".local/research_memory/budget_v5_20261022/evaluation.json"))
    parser.add_argument("--authorization", type=Path, default=Path(".local/research_memory/budget_v5_20261022/receipt.json"))
    parser.add_argument("--execute", action="store_true", help="Authorize the selected finite paid phase once")
    parser.add_argument("--correction-note", help="Document the one correction based solely on the eight dev results")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare(args.workspace)
    elif args.phase == "freeze":
        result = freeze(args.workspace, args.config, args.authorization, correction_note=args.correction_note)
    elif args.phase == "report":
        result = report(args.workspace)
    else:
        phase = {"complete-dev": "dev_completion", "calibrate": "calibration", "assess": "main"}.get(args.phase, args.phase)
        result = run_phase(args.workspace, phase, args.config, args.authorization, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
