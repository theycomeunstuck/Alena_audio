"""Retire previous spending and allocate two finite budgets without refunds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def transition(root: Path, catalog_path: Path, output: Path) -> dict:
    """Local operation only. Existing receipts and ledgers are never reset."""
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Budget transition already exists; inspect its receipt")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    matches = [model for model in catalog["data"] if model["id"] == "openai/gpt-6.1-sol"]
    if len(matches) != 1:
        raise ValueError("Verified provider model is missing or duplicated")
    pricing = matches[0]["pricing"]
    input_price = Decimal(pricing["prompt"]) * 1_000_000
    output_price = Decimal(pricing["completion"]) * 1_000_000
    if input_price <= 0 or output_price <= 0:
        raise ValueError("Provider pricing must be positive")
    prior_names = (
        "spiderssense_pilot_budget.sqlite3", "spiderssense_structured_pilot_budget.sqlite3",
        "spiderssense_json_mode_pilot_budget.sqlite3", "spiderssense_context_v3_budget.sqlite3",
    )
    local = root / ".local/research_memory"
    # Review all balances before retiring the active ledger. Unknown reservations count in full.
    balances = []
    for name in prior_names:
        ledger_path = local / name
        if not ledger_path.is_file():
            # Audit F-050: sqlite3.connect used to run before any existence
            # check and silently materialized an empty canonical-named ledger
            # file, later misleading the v5 is_file guard.
            raise FileNotFoundError(f"A canonical prior ledger is missing: {ledger_path}")
        with sqlite3.connect(ledger_path) as conn:
            rows = conn.execute("SELECT charged_usd,status FROM budget_calls").fetchall()
            blocked = conn.execute("SELECT blocked FROM budget_config WHERE id=1").fetchone()
            if blocked is None:
                raise ValueError("Previous ledger has no configuration")
            balances.append({"ledger": name, "calls": len(rows),
                "charged_or_reserved_usd": str(sum((Decimal(row[0]) for row in rows), Decimal(0))),
                "unknown_reservations": sum(row[1] == "reserved_unknown" for row in rows)})
    prior = sum((Decimal(row["charged_or_reserved_usd"]) for row in balances), Decimal(0))
    if prior + Decimal("2.40") + Decimal("0.40") > Decimal("4"):
        raise ValueError("Requested phase allocations exceed the overall USD 4 cap")
    output.mkdir(parents=True, exist_ok=False)
    for name in prior_names:
        with sqlite3.connect(local / name) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE budget_config SET blocked=1 WHERE id=1")
    configs = {}
    from local_app.memory_ported.spiderssense import SpiderssenseClient
    for phase, cap, calls in (("evaluation", "2.40", 152), ("integration", "0.40", 20)):
        ledger = local / f"spiderssense_{phase}_v4_budget.sqlite3"
        if ledger.exists():
            raise FileExistsError("New phase ledger already exists; spending cannot be reset")
        config = {"model": "openai/gpt-6.1-sol", "input_usd_per_million": str(input_price),
            "output_usd_per_million": str(output_price), "pricing_verified": True,
            "pricing_source": "https://spiderssense.com/v1/models", "ledger_path": str(ledger),
            "max_cost_usd": cap, "max_calls": calls,
            "max_output_tokens": 8192 if phase == "evaluation" else 1536,
            "token_field": "max_completion_tokens", "reasoning_effort": "medium",
            "timeout": 180, "key_file": str(local / "spiderssense.key"),
            "use_json_schema": False, "use_json_object": phase == "evaluation"}
        client = SpiderssenseClient(**config)
        path = output / f"{phase}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        configs[phase] = {"configuration": str(path), "cap_usd": cap, "call_cap": calls,
                         "configuration_hash": client.preflight()["configuration_hash"]}
    receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_authorized_cap_usd": "4", "previous_stages": balances,
        "cumulative_prior_charged_or_reserved_usd": str(prior), "prior_ledgers_blocked": True,
        "unknown_cost_reservations_retained": True, "allocations": configs,
        "unallocated_usd": str(Decimal("4") - prior - Decimal("2.80")),
        "pricing_catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "pricing_source": "https://spiderssense.com/v1/models",
        "catalog_observation": "Fresh public catalog retrieved before any new paid call"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(transition(ROOT, args.catalog, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
