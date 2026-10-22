"""Finite authorization checks use only temporary SQLite files and fake prices."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import sqlite3

import pytest

from local_app.answer_adapter import AnswerAdapter
from local_app.budget_authorization import build_authorized_client, validate_budget_authorization
from scripts.transition_memory_budget_v5 import EXPECTED_PRIOR, PRIOR_NAMES, PRIOR_RECEIPTS, transition


def setup_transition(tmp_path):
    local = tmp_path / ".local/research_memory"
    local.mkdir(parents=True)
    for index, name in enumerate(PRIOR_NAMES):
        with sqlite3.connect(local / name) as db:
            db.executescript("CREATE TABLE budget_config(id INTEGER PRIMARY KEY,blocked INTEGER);"
                             "INSERT INTO budget_config VALUES(1,0);"
                             "CREATE TABLE budget_calls(id INTEGER PRIMARY KEY,charged_usd TEXT,status TEXT);")
            db.execute("INSERT INTO budget_calls VALUES(1,?,?)",
                       (str(EXPECTED_PRIOR) if index == 0 else "0", "reserved_unknown"))
    for name in PRIOR_RECEIPTS:
        path = local / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"charged_or_reserved_usd": str(EXPECTED_PRIOR)}), encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"data": [{"id": "openai/gpt-6.1-sol",
        "pricing": {"prompt": "0.0000005", "completion": "0.000003"},
        "supported_parameters": ["max_completion_tokens", "reasoning_effort", "response_format"]}]}), encoding="utf-8")
    return catalog, tmp_path / "budget_v5", local


def test_dry_run_never_creates_or_retires_ledgers(tmp_path):
    catalog, output, local = setup_transition(tmp_path)
    report = transition(tmp_path, catalog, output)
    assert report["status"] == "prepared" and not output.exists()
    assert report["previous_charged_or_reserved_usd"] == str(EXPECTED_PRIOR)
    for name in PRIOR_NAMES:
        with sqlite3.connect(local / name) as db:
            assert db.execute("SELECT blocked FROM budget_config").fetchone()[0] == 0
    assert not list(local.glob("*v5*"))


def test_transition_preserves_unknown_reserves_and_discards_old_remainder(tmp_path):
    catalog, output, local = setup_transition(tmp_path)
    report = transition(tmp_path, catalog, output, execute=True)
    assert report["status"] == "active"
    assert report["old_unused_budget_rolled_over_usd"] == "0"
    assert Decimal(report["previous_unused_budget_usd"]) == Decimal("2.7594975")
    assert sum(x["unknown_reservations"] for x in report["previous_stages"]) == 6
    for row in report["previous_stages"]:
        with sqlite3.connect(local / row["ledger"]) as db:
            assert db.execute("SELECT blocked FROM budget_config").fetchone()[0] == 1
            assert db.execute("SELECT status FROM budget_calls").fetchone()[0] == "reserved_unknown"
    for phase, cap, calls in (("evaluation", Decimal("16"), 126), ("integration", Decimal("4"), 6)):
        client = build_authorized_client(output / f"{phase}.json", output / "receipt.json", purpose=phase)
        assert client.max_cost == cap and client.max_calls == calls
        assert client.preflight()["calls_reserved"] == 0
    with pytest.raises(FileExistsError):
        transition(tmp_path, catalog, output, execute=True)


@pytest.mark.parametrize("change", ["missing_model", "missing_price", "prior_drift", "duplicate_model"])
def test_transition_refuses_unreviewed_inputs_before_mutation(tmp_path, change):
    catalog, output, local = setup_transition(tmp_path)
    data = json.loads(catalog.read_text())
    if change == "missing_model":
        data["data"] = []
    elif change == "missing_price":
        data["data"][0]["pricing"]["prompt"] = "0"
    elif change == "duplicate_model":
        data["data"].append(deepcopy(data["data"][0]))
    else:
        with sqlite3.connect(local / PRIOR_NAMES[0]) as db:
            db.execute("UPDATE budget_calls SET charged_usd='9'")
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        transition(tmp_path, catalog, output, execute=True)
    assert not output.exists()
    with sqlite3.connect(local / PRIOR_NAMES[0]) as db:
        assert db.execute("SELECT blocked FROM budget_config").fetchone()[0] == 0


@pytest.mark.parametrize("override", [{"ledger_path": "unapproved.sqlite3"}, {"max_cost_usd": "4.01"},
    {"max_calls": 7}, {"max_calls": True}, {"max_output_tokens": 1537}, {"model": "other-model"}])
def test_v2_rejects_config_drift_without_creating_a_new_ledger(tmp_path, override):
    catalog, output, _ = setup_transition(tmp_path)
    transition(tmp_path, catalog, output, execute=True)
    wrapper = json.loads((output / "external_answers_v2.json").read_text())
    wrapper["spiderssense"].update(override)
    path = output / "changed-wrapper.json"
    path.write_text(json.dumps(wrapper))
    with pytest.raises((ValueError, TypeError)):
        AnswerAdapter.from_config(object(), path)
    assert not (output / "unapproved.sqlite3").exists()


def test_v2_requires_receipt_and_matches_existing_ledger_fingerprint(tmp_path):
    catalog, output, _ = setup_transition(tmp_path)
    transition(tmp_path, catalog, output, execute=True)
    assert AnswerAdapter.from_config(object(), output / "external_answers_v2.json").external_client.max_calls == 6
    wrapper = json.loads((output / "external_answers_v2.json").read_text())
    del wrapper["budget_authorization_path"]
    path = output / "no-auth.json"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError):
        AnswerAdapter.from_config(object(), path)
    config = json.loads((output / "integration.json").read_text())
    with sqlite3.connect(config["ledger_path"]) as db:
        db.execute("UPDATE budget_config SET blocked=1")
    with pytest.raises(ValueError, match="blocked"):
        build_authorized_client(output / "integration.json", output / "receipt.json", purpose="integration")


def test_authorization_detects_modified_frozen_configuration(tmp_path):
    catalog, output, _ = setup_transition(tmp_path)
    transition(tmp_path, catalog, output, execute=True)
    path = output / "evaluation.json"
    config = json.loads(path.read_text())
    path.write_text(json.dumps(config) + " ")
    with pytest.raises(ValueError, match="changed"):
        validate_budget_authorization(config, output / "receipt.json")


def test_disabled_v2_does_not_require_existing_receipt_or_provider(tmp_path):
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps({"kind": "local_app_external_answers_v2", "enabled": False,
        "budget_authorization_path": "absent.json", "spiderssense": {}}))
    assert AnswerAdapter.from_config(object(), path).external_client is None
