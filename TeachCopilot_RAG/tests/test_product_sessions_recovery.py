"""Recovery gates preserve unknown outcomes and the original pilot evidence."""
from argparse import Namespace
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.run_product_sessions_v1 import FIXTURE, load_fixture
from scripts.recover_product_sessions_v1 import (COMPLETED_CODES, INTERRUPTED_CODE, UNKNOWN_MESSAGE_ID,
    immutable_files, load_checkpoint, ordinary_message_coverage, recovery_plan, restored_interrupted_events,
    run, validate_database_boundary)


def state_fixture(*, recovered=False):
    fixture, _ = load_fixture()
    sessions = [{"client_session_id": code, "id": "sid-" + code, "status": "completed"}
                for code in COMPLETED_CODES]
    sessions.append({"client_session_id": INTERRUPTED_CODE, "id": "sid-L02S2", "status": "active"})
    events = fixture["learners"][1]["sessions"][1]["events"]
    messages = [{"client_message_id": "L02S2-E01", "status": "completed", "literal_user_text": events[0]["text"],
                 "source_id": "source-1", "response": {"choices": [{"message": {"content": "Уточним."}}]}},
                {"client_message_id": UNKNOWN_MESSAGE_ID, "status": "unknown_outcome" if recovered else "pending",
                 "literal_user_text": events[1]["text"], "source_id": "source-2", "response": None}]
    jobs = [{"id": "job-2", "idempotency_key": f"chat:sid-L02S2:{UNKNOWN_MESSAGE_ID}",
             "call_started_at": "2026-10-22T15:40:00+00:00", "status": "unknown_outcome" if recovered else "running"}]
    return {"sessions": sessions, "interrupted_messages": messages, "jobs": jobs}, fixture


def test_expired_started_request_is_unknown_and_never_requeued():
    before, fixture = state_fixture()
    assert validate_database_boundary(before, fixture) == "sid-L02S2"
    after, _ = state_fixture(recovered=True)
    assert validate_database_boundary(after, fixture, recovered=True) == "sid-L02S2"
    after["jobs"][0]["status"] = "queued"
    with pytest.raises(ValueError, match="never queued"):
        validate_database_boundary(after, fixture, recovered=True)


@pytest.mark.parametrize("mutation", ["later_lesson", "third_turn", "completed_unknown", "changed_source", "new_queued_job"])
def test_recovery_fails_closed_before_new_generations_if_boundary_changed(mutation):
    state, fixture = state_fixture(recovered=True)
    if mutation == "later_lesson":
        state["sessions"].append({"client_session_id": "L01S3", "id": "sid-5", "status": "active"})
    elif mutation == "third_turn":
        state["interrupted_messages"].append({"client_message_id": "L02S2-E03", "status": "completed"})
    elif mutation == "completed_unknown":
        state["interrupted_messages"][1]["status"] = "completed"
    elif mutation == "changed_source":
        state["interrupted_messages"][1]["literal_user_text"] = "Иной запрос"
    else:
        state["jobs"].append({"id": "unexpected", "status": "queued"})
    with pytest.raises(ValueError):
        validate_database_boundary(state, fixture, recovered=True)


def test_reconstruction_keeps_one_unknown_and_two_unexecuted_original_turns(tmp_path):
    state, fixture = state_fixture(recovered=True)
    session = fixture["learners"][1]["sessions"][1]
    result = restored_interrupted_events(state["interrupted_messages"], session, tmp_path)
    assert [event["status"] for event in result] == ["completed", "unknown_outcome", "not_executed", "not_executed"]
    assert result[1]["response"] is None and result[1]["response_completed"] is False
    assert all(event["generation_repeated"] is False for event in result)
    assert result[2]["literal_preregistered_text"] == session["events"][2]["text"]
    assert state["interrupted_messages"][1]["status"] == "unknown_outcome"


def test_immutable_manifest_excludes_only_shared_journal_and_its_sidecars(tmp_path):
    journal = tmp_path / "budget.sqlite3"
    journal.write_bytes(b"initial")
    (tmp_path / "budget.sqlite3-wal").write_bytes(b"runtime")
    artifact = tmp_path / "report.partial.json"
    artifact.write_text('{"status":"running"}', encoding="utf-8")
    before = immutable_files(tmp_path, journal)
    journal.write_bytes(b"authorized_extension_and_additional_calls")
    assert immutable_files(tmp_path, journal) == before
    artifact.write_text('{"status":"falsely_completed"}', encoding="utf-8")
    assert immutable_files(tmp_path, journal) != before


def checkpoint(tmp_path):
    fixture, _ = load_fixture()
    ledger = tmp_path / "local-generation-budget.sqlite3"
    ledger.write_bytes(b"existing durable journal")
    report = {"status": "running", "preflight_only": False,
              "database_name": "teachcopilot_product_sessions_bd0772934848", "budget_ledger": str(ledger),
              "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
              "sessions": [{"code": code, "status": "attempted"} for code in COMPLETED_CODES] +
                          [{"code": INTERRUPTED_CODE, "status": "running"}],
              "processes": [{"api": {"pid": 1}, "worker": {"pid": 2}}, {"api": {"pid": 3}, "worker": {"pid": 4}}]}
    (tmp_path / "report.partial.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "preregistered.json").write_text(json.dumps({"fixture": fixture}), encoding="utf-8")
    return report


def test_dry_run_reads_only_checkpoint_and_does_not_start_or_modify_anything(tmp_path, monkeypatch):
    report = checkpoint(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run tried to start a process")

    monkeypatch.setattr("subprocess.Popen", forbidden)
    result = run(Namespace(continue_from=tmp_path, output_dir=None, execute=False))
    assert result["status"] == "dry_run"
    assert result["remaining_new_ordinary_messages"] == 8
    assert result["remaining_new_explicit_actions"] == 0
    assert result["engineering_iteration"] is False
    assert result["unknown_request_never_repeated"] == UNKNOWN_MESSAGE_ID
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_a_finished_or_changed_checkpoint_cannot_be_silently_recovered(tmp_path):
    report = checkpoint(tmp_path)
    report["status"] = "completed_with_findings"
    (tmp_path / "report.partial.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="interrupted"):
        load_checkpoint(tmp_path)


def test_coverage_comes_from_canonical_delivery_status_not_expected_counts():
    fixture, _ = load_fixture()
    statuses = [{"client_message_id": f"{s['code']}-E{index:02d}", "status": "completed"}
                for learner in fixture["learners"] for s in learner["sessions"]
                for index, event in enumerate(s["events"], 1) if event["kind"] == "message"]
    statuses = [row for row in statuses if row["client_message_id"] not in {"L02S2-E03", "L02S2-E04"}]
    next(row for row in statuses if row["client_message_id"] == UNKNOWN_MESSAGE_ID)["status"] = "unknown_outcome"
    result = ordinary_message_coverage(statuses, fixture)
    assert result["planned"] == 24 and result["completed"] == 21
    assert result["unknown_outcome"] == 1 and result["not_executed"] == 2
    assert result["coverage_complete"] is False
    statuses[-1]["status"] = "failed"
    result = ordinary_message_coverage(statuses, fixture)
    assert result["completed"] == 20 and result["failed"] == 1
