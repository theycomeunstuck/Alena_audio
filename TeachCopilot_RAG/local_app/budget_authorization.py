"""Explicit finite authorization for the additional USD 20 research package.

This module performs local validation only. It never discovers credentials or
prices, sends requests, refunds reservations, or resets an existing ledger.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path

from research_memory.spiderssense import SpiderssenseClient


AUTHORIZATION_KIND = "teachcopilot_additional_budget_v5"
ALLOCATIONS = {"evaluation": (Decimal("16"), 126), "integration": (Decimal("4"), 6)}


def _resolve(value, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An explicit authorization path is required")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _cost(value) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Authorization amount must be finite")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError("Authorization amount must be finite") from None
    if not result.is_finite():
        raise ValueError("Authorization amount must be finite")
    return result


def validate_budget_authorization(client_config: dict, authorization_path: str | Path,
                                  *, phase: str = "evaluation") -> dict:
    """Bind the complete configuration, durable ledger and phase to one receipt.

    Configurations passed in memory must equal the frozen configuration file.
    The receipt deliberately grants no carryover from the old USD 4 package.
    """
    if phase not in ALLOCATIONS or not isinstance(client_config, dict):
        raise ValueError("Unknown budget authorization phase")
    path = Path(authorization_path).resolve()
    auth = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(auth, dict) or auth.get("kind") != AUTHORIZATION_KIND
            or auth.get("status") != "active" or auth.get("prior_ledgers_blocked") is not True
            or auth.get("unknown_cost_reservations_retained") is not True
            or _cost(auth.get("additional_authorized_cap_usd")) != Decimal("20")
            or _cost(auth.get("old_unused_budget_rolled_over_usd")) != 0):
        raise ValueError("Additional budget authorization is invalid or inactive")
    allocations = auth.get("allocations")
    if not isinstance(allocations, dict) or set(allocations) != set(ALLOCATIONS):
        raise ValueError("Both finite allocations are required")
    for name, (cap, calls) in ALLOCATIONS.items():
        item = allocations[name]
        if (not isinstance(item, dict) or _cost(item.get("cap_usd")) != cap
                or type(item.get("call_cap")) is not int or item["call_cap"] != calls):
            raise ValueError("Authorized finite allocations differ from the approved package")
    allocation = allocations[phase]
    config_path = _resolve(allocation.get("configuration"), path.parent)
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != allocation.get("configuration_sha256"):
        raise ValueError("Authorized configuration file changed")
    if json.loads(config_bytes) != client_config:
        raise ValueError("Client settings differ from the authorized configuration")
    ledger = _resolve(client_config.get("ledger_path"), config_path.parent)
    if ledger != _resolve(allocation.get("ledger_path"), path.parent):
        raise ValueError("Client ledger is not authorized for this phase")
    cap, calls = ALLOCATIONS[phase]
    if (_cost(client_config.get("max_cost_usd")) != cap
            or type(client_config.get("max_calls")) is not int or client_config["max_calls"] != calls):
        raise ValueError("Client limits differ from the authorized finite allocation")
    if phase == "evaluation" and (client_config.get("model") != "openai/gpt-6.1-sol"
            or client_config.get("reasoning_effort") != "medium"
            or client_config.get("max_output_tokens") != 8192):
        raise ValueError("Research judge configuration differs from the fixed protocol")
    if not ledger.is_file():
        raise ValueError("Authorized durable ledger does not exist")
    return auth


def build_authorized_client(config_path: str | Path, authorization_path: str | Path,
                            *, purpose: str = "evaluation") -> SpiderssenseClient:
    """Construct an existing budgeted client only after the receipt is verified."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    auth = validate_budget_authorization(config, authorization_path, phase=purpose)
    client = SpiderssenseClient(**config)
    state = client.preflight()
    if state["configuration_hash"] != auth["allocations"][purpose].get("configuration_hash"):
        raise ValueError("Authorized ledger fingerprint differs")
    if state["blocked"]:
        raise ValueError("Authorized ledger is blocked")
    return client
