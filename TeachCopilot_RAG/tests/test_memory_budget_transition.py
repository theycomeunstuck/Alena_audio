from decimal import Decimal
import json
import sqlite3

import pytest

from scripts.transition_memory_budget import transition


def setup_prior(tmp_path, total="1.1371987625"):
    local = tmp_path / ".local/research_memory"
    local.mkdir(parents=True)
    names = ("pilot", "structured_pilot", "json_mode_pilot", "context_v3")
    for number, name in enumerate(names):
        with sqlite3.connect(local / f"spiderssense_{name}_budget.sqlite3") as conn:
            conn.executescript("CREATE TABLE budget_config(id INTEGER PRIMARY KEY, blocked INTEGER);"
                               "INSERT INTO budget_config VALUES(1,0);"
                               "CREATE TABLE budget_calls(charged_usd TEXT,status TEXT);")
            conn.execute("INSERT INTO budget_calls VALUES(?,?)", (total if number == 3 else "0", "reserved_unknown"))
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"data": [{"id": "openai/gpt-6.1-sol",
        "pricing": {"prompt": "0.0000005", "completion": "0.000003"}}]}), encoding="utf-8")
    return catalog, local


def test_transition_preserves_unknown_costs_and_blocks_every_prior_ledger(tmp_path):
    catalog, local = setup_prior(tmp_path)
    receipt = transition(tmp_path, catalog, tmp_path / "transition")
    assert receipt["cumulative_prior_charged_or_reserved_usd"] == "1.1371987625"
    assert Decimal(receipt["unallocated_usd"]) == Decimal("0.0628012375")
    assert sum(row["unknown_reservations"] for row in receipt["previous_stages"]) == 4
    for prior in receipt["previous_stages"]:
        with sqlite3.connect(local / prior["ledger"]) as conn:
            assert conn.execute("SELECT blocked FROM budget_config").fetchone()[0] == 1
            assert conn.execute("SELECT status FROM budget_calls").fetchone()[0] == "reserved_unknown"
    with pytest.raises(FileExistsError):
        transition(tmp_path, catalog, tmp_path / "transition")


def test_overspend_is_refused_before_any_ledger_is_retired(tmp_path):
    catalog, local = setup_prior(tmp_path, total="1.21")
    with pytest.raises(ValueError, match="overall"):
        transition(tmp_path, catalog, tmp_path / "transition")
    assert not (tmp_path / "transition").exists()
    for path in local.glob("*budget.sqlite3"):
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT blocked FROM budget_config").fetchone()[0] == 0
