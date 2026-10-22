"""Retire the canonical six ledgers and authorize only the new USD 20 package."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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

from local_app.budget_authorization import ALLOCATIONS, AUTHORIZATION_KIND
from research_memory.spiderssense import SpiderssenseClient

PRIOR_NAMES = tuple(f"spiderssense_{name}_budget.sqlite3" for name in (
    "pilot", "structured_pilot", "json_mode_pilot", "context_v3", "evaluation_v4", "integration_v4"))
PRIOR_RECEIPTS = (
    "budget_transition_structured_20261016.json", "budget_transition_json_mode_20261016.json",
    "budget_transition_context_v3_20261016.json", "budget_v4_20261017/receipt.json",
    "final_v4_20261018/budget_final.json",
)
EXPECTED_PRIOR = Decimal("1.2405025")
CATALOG_URL = "https://spiderssense.com/v1/models"


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def transition(root: Path, catalog_path: Path, output: Path, *, execute=False) -> dict:
    """Default is a read-only plan. Existing rows and unknown reserves never change."""
    root, catalog_path, output = root.resolve(), catalog_path.resolve(), output.resolve()
    local = root / ".local/research_memory"
    if output.exists():
        raise FileExistsError("Transition output already exists; spending cannot be reset")
    raw = catalog_path.read_bytes()
    catalog = json.loads(raw)
    matches = [item for item in catalog["data"] if item.get("id") == "openai/gpt-6.1-sol"]
    if len(matches) != 1:
        raise ValueError("Verified provider model is missing or duplicated")
    model = matches[0]
    supported = model.get("supported_parameters", [])
    if not {"max_completion_tokens", "reasoning_effort", "response_format"} <= set(supported):
        raise ValueError("Provider catalog does not support the fixed judge settings")
    prices = [Decimal(model["pricing"][name]) * 1_000_000 for name in ("prompt", "completion")]
    if any(not price.is_finite() or price <= 0 for price in prices):
        raise ValueError("Model-specific provider prices must be positive")
    receipt_hashes = {name: hashlib.sha256((local / name).read_bytes()).hexdigest()
                      for name in PRIOR_RECEIPTS}
    balances = []
    paths = [local / name for name in PRIOR_NAMES]
    # Fixed canonical names are read once. Report/export copies are never counted.
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("A canonical prior ledger is missing")
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
            rows = db.execute("SELECT charged_usd,status FROM budget_calls ORDER BY id").fetchall()
            state = db.execute("SELECT blocked FROM budget_config WHERE id=1").fetchone()
        if state is None:
            raise ValueError("A prior ledger has no configuration")
        balances.append({"ledger": path.name, "calls": len(rows),
                         "charged_or_reserved_usd": str(sum((Decimal(row[0]) for row in rows), Decimal(0))),
                         "unknown_reservations": sum(row[1] == "reserved_unknown" for row in rows),
                         "previously_blocked": bool(state[0]),
                         "budget_calls_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()})
    prior = sum((Decimal(row["charged_or_reserved_usd"]) for row in balances), Decimal(0))
    historical = json.loads((local / PRIOR_RECEIPTS[-1]).read_text(encoding="utf-8"))
    if prior != EXPECTED_PRIOR or Decimal(historical["charged_or_reserved_usd"]) != prior:
        raise ValueError("Prior spending differs from the reviewed canonical chain; inspect before transition")
    new_paths = {phase: local / f"spiderssense_{phase}_v5_budget.sqlite3" for phase in ALLOCATIONS}
    if any(path.exists() for path in new_paths.values()):
        raise FileExistsError("A v5 ledger already exists; spending cannot be reset")
    receipt = {"kind": AUTHORIZATION_KIND, "status": "prepared", "executed": execute,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "previous_authorized_cap_usd": "4", "additional_authorized_cap_usd": "20",
               "previous_charged_or_reserved_usd": str(prior),
               "previous_unused_budget_usd": str(Decimal("4") - prior),
               "old_unused_budget_rolled_over_usd": "0", "new_unallocated_usd": "0",
               "previous_stages": balances, "prior_receipts_sha256": receipt_hashes,
               "prior_ledgers_blocked": False, "unknown_cost_reservations_retained": True,
               "automatic_paid_retries_allowed": False,
               "pricing_source": CATALOG_URL, "pricing_catalog_path": str(catalog_path),
               "pricing_catalog_sha256": hashlib.sha256(raw).hexdigest(),
               "pricing_checked_at_utc": datetime.fromtimestamp(catalog_path.stat().st_mtime, timezone.utc).isoformat(),
               "pricing_usd_per_million": {"input": str(prices[0]), "output": str(prices[1])},
               "allocations": {phase: {"cap_usd": str(cap), "call_cap": calls,
                                        "ledger_path": str(new_paths[phase])}
                               for phase, (cap, calls) in ALLOCATIONS.items()}}
    if not execute:
        return receipt
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "PREPARED.json", receipt)
    # Hold all prior SQLite write locks until balances are checked again. Crashes
    # leave blocked ledgers or a prepared receipt, never a new active authority.
    with ExitStack() as stack:
        dbs = []
        for path, balance in zip(paths, balances):
            db = stack.enter_context(sqlite3.connect(path))
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT charged_usd,status FROM budget_calls ORDER BY id").fetchall()
            if hashlib.sha256(json.dumps(rows).encode()).hexdigest() != balance["budget_calls_sha256"]:
                raise ValueError("Prior ledger changed during transition; no authorization created")
            db.execute("UPDATE budget_config SET blocked=1 WHERE id=1")
            dbs.append(db)
        for db in dbs:
            db.commit()
    receipt["prior_ledgers_blocked"] = True
    for phase, (cap, calls) in ALLOCATIONS.items():
        config = {"model": "openai/gpt-6.1-sol", "input_usd_per_million": str(prices[0]),
                  "output_usd_per_million": str(prices[1]), "pricing_verified": True,
                  "pricing_source": CATALOG_URL, "ledger_path": str(new_paths[phase]),
                  "max_cost_usd": str(cap), "max_calls": calls,
                  "max_output_tokens": 8192 if phase == "evaluation" else 1536,
                  "token_field": "max_completion_tokens", "reasoning_effort": "medium",
                  "timeout": 180, "key_file": str(local / "spiderssense.key"),
                  "use_json_schema": False, "use_json_object": True}
        client = SpiderssenseClient(**config)
        config_path = output / f"{phase}.json"
        _json(config_path, config)
        receipt["allocations"][phase].update(configuration=str(config_path),
            configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            configuration_hash=client.preflight()["configuration_hash"])
    receipt.update(status="active", completed_at_utc=datetime.now(timezone.utc).isoformat())
    _json(output / "receipt.json", receipt)
    _json(output / "external_answers_v2.json", {"kind": "local_app_external_answers_v2", "enabled": True,
          "budget_authorization_path": str(output / "receipt.json"),
          "spiderssense": json.loads((output / "integration.json").read_text(encoding="utf-8"))})
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/research_memory/budget_v5_20261022")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(transition(ROOT, args.catalog, args.output, execute=args.execute), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
