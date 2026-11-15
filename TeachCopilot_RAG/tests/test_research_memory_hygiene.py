"""Hygiene regressions for the research memory module.

Covers three verified audit findings, offline only (fake clients, temporary
directories, no network, no database):

- F-051: resuming a finished spiderssense workflow must replay its completed
  steps instead of overwriting assessed results with unassessed rows when the
  provider credential is absent or the budget ledger is blocked.
- F-052: relative provider ledger_path/key_file values must resolve to one
  stable absolute location, independent of the working directory.
- F-053: the batch call cache must be keyed to the client implementation, and
  fixture-produced completions must never be accounted as real model calls.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from research_memory import batch, spiderssense, workflow
from research_memory.clients import Completion, FixtureClient
from research_memory.judge import CRITERIA


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def history():
    return {"profile_as_of_cutoff": {"topic_id": "math.test", "grade": 4},
            "lessons": [{"lesson_id": "l1", "observed_on": "2026-09-01", "turns": [
                {"turn_id": "t1", "role": "student", "text": "Исправил после помощи."}]}]}


def make_calibration(root):
    cases = []
    for code, marker in (("C001", "CALIBRATION_ACCEPT"), ("C002", "CALIBRATION_CRITICAL")):
        cases.append({"answer_code": code, "data_origin": "synthetic_unreviewed", "execution_status": "ok",
                      "contract_only": False, "payload": {"history": history(), "question": "Что дальше?",
                                                          "answer": {"reason": marker}}})
    paths = root / "calibration.cases.jsonl", root / "calibration.expected.jsonl"
    write_rows(paths[0], cases)
    write_rows(paths[1], [{"answer_code": "C001", "expected_verdict": "acceptable"},
                          {"answer_code": "C002", "expected_verdict": "critical_error", "critical": True}])
    return paths


class JudgeProviderStub:
    """Provider-shaped judge client without a network.

    Mirrors the real client's preflight contract: the configuration hash does
    not cover credential presence, exactly like the durable budget client.
    """

    model = "fake"
    max_output_tokens = 100

    def __init__(self, *, ready=True, blocked=False):
        self.seen, self.ready, self.blocked = [], ready, blocked

    def preflight(self):
        return {"configuration_hash": "stub-config-v1", "key_present": self.ready,
                "blocked": self.blocked, "calls_reserved": len(self.seen),
                "charged_or_reserved_usd": "0", "available_budget_usd": "100"}

    def complete(self, messages, purpose, response_schema=None):
        payload = json.loads(messages[-1]["content"])
        answer = payload.get("answer")
        reason = answer.get("reason", "") if isinstance(answer, dict) else ""
        self.seen.append(reason)
        verdict = "critical_error" if reason == "CALIBRATION_CRITICAL" else "acceptable"
        value = {"verdict": verdict, "scores": {key: 2 for key in CRITERIA},
                 "findings": [{"criterion": "help_attribution", "claim": "Ошибка в ответе.",
                               "evidence_turn_ids": ["t1"], "quote": "после помощи",
                               "explanation": "В истории есть помощь."}] if verdict == "critical_error" else [],
                 "rationale": "Тестовый судья без модели."}
        return Completion(json.dumps(value, ensure_ascii=False), {"input_tokens": 10, "output_tokens": 5}, 0,
                          {"backend": "spiderssense", "purpose": purpose})


def build_spiderssense_workflow(tmp_path):
    calibration = make_calibration(tmp_path)
    hist = {**history(), "history_id": "h1", "learner_id": "synthetic1", "split": "dev",
            "data_origin": "synthetic_unreviewed"}
    histories, requests = tmp_path / "histories.jsonl", tmp_path / "requests.jsonl"
    write_rows(histories, [hist])
    write_rows(requests, [{"request_id": f"q{i}", "history_id": "h1", "topic_id": "math.test",
                           "text": f"Следующее действие {i}?"} for i in range(1, 4)])
    batch_dir, output = tmp_path / "batch", tmp_path / "workflow"
    batch.create_batch(histories, requests, batch_dir, methods=["profile_only"])
    return calibration, batch_dir, output


class TestF051FinishedWorkflowResume:
    @staticmethod
    def run(batch_dir, output, calibration, client):
        return workflow.run_workflow(batch_dir, output, judge_backend="spiderssense", judge_model="fake",
                                     judge_client=client, calibration_cases_path=calibration[0],
                                     calibration_expected_path=calibration[1])

    @pytest.mark.parametrize("blocked", [False, True], ids=["missing_key", "blocked_ledger"])
    def test_resume_without_credentials_keeps_assessed_results(self, tmp_path, blocked):
        calibration, batch_dir, output = build_spiderssense_workflow(tmp_path)
        stub = JudgeProviderStub(ready=True)
        first = self.run(batch_dir, output, calibration, stub)
        assert first["status"] == "completed"
        assert json.loads((output / "state.json").read_text(encoding="utf-8"))["status"] == "completed"
        rows = read_rows(output / "answers/judgements.jsonl")
        assert all(row["status"] == "assessed" and row["judgement"] is not None for row in rows)
        assert first["answer_counts"] == {"assessed": 3}
        assert len(stub.seen) == 5

        stub.ready = False
        stub.blocked = blocked
        second = self.run(batch_dir, output, calibration, stub)
        state = json.loads((output / "state.json").read_text(encoding="utf-8"))
        resumed_rows = read_rows(output / "answers/judgements.jsonl")
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert second["status"] == "completed"
        assert state["status"] == "completed"
        assert all(row["status"] == "assessed" and row["judgement"] is not None for row in resumed_rows)
        assert report["answer_counts"] == {"assessed": 3}
        assert len(stub.seen) == 5


class TestF052LedgerAnchor:
    @staticmethod
    def config():
        return {"model": "openai/gpt-6.1-sol", "input_usd_per_million": "0.5", "output_usd_per_million": "3",
                "pricing_verified": True, "pricing_source": "https://spiderssense.com/v1/models",
                "ledger_path": ".local/research_memory/hygiene_budget.sqlite3", "max_cost_usd": "4",
                "max_calls": 10, "max_output_tokens": 64,
                "key_file": ".local/research_memory/hygiene.key"}

    def test_relative_paths_do_not_depend_on_the_working_directory(self, tmp_path, monkeypatch):
        # The anchor is redirected into the temporary tree so the durable
        # ledger file is created there and nowhere else.
        monkeypatch.setattr(spiderssense, "_ANCHOR", tmp_path, raising=False)
        dir_a, dir_b = tmp_path / "dirA", tmp_path / "dirB"
        for directory in (dir_a, dir_b):
            directory.mkdir()
        monkeypatch.chdir(dir_a)
        client_a = spiderssense.SpiderssenseClient(**self.config())
        monkeypatch.chdir(dir_b)
        client_b = spiderssense.SpiderssenseClient(**self.config())
        assert client_a.ledger_path.is_absolute()
        assert client_a.ledger_path == client_b.ledger_path
        assert client_a._key_file.is_absolute()
        assert client_a._key_file == client_b._key_file
        # One configuration means one durable ledger: spending recorded while
        # working in one directory is visible from the other directory.
        connection = sqlite3.connect(client_a.ledger_path)
        with connection:
            connection.execute("INSERT INTO budget_calls(reserved_usd,charged_usd,status,input_bound) "
                               "VALUES('0.1','0.1','settled',100)")
        connection.close()
        assert client_b.preflight()["calls_reserved"] == 1

    def test_absolute_paths_are_preserved_exactly(self, tmp_path):
        config = self.config()
        ledger, key = tmp_path / "ledger.sqlite3", tmp_path / "key.file"
        config.update(ledger_path=ledger, key_file=key)
        client = spiderssense.SpiderssenseClient(**config)
        assert client.ledger_path == ledger
        assert client._key_file == key


class TestF053BatchClientCache:
    @staticmethod
    def build_ollama_batch(tmp_path):
        hist = {**history(), "history_id": "h1", "learner_id": "synthetic1", "split": "dev",
                "data_origin": "synthetic_unreviewed"}
        histories, requests = tmp_path / "histories.jsonl", tmp_path / "requests.jsonl"
        write_rows(histories, [hist])
        write_rows(requests, [{"request_id": f"q{i}", "history_id": "h1", "topic_id": "math.test",
                               "text": f"Следующее действие {i}?"} for i in range(1, 3)])
        batch_dir = tmp_path / "batch"
        # max_calls=1 leaves the job pending with one recorded call, so a
        # resume actually reaches the cached-completion decision.
        batch.create_batch(histories, requests, batch_dir, backend="ollama", methods=["profile_only"],
                           max_calls=1)
        return batch_dir

    def test_fixture_execution_of_ollama_batch_is_never_reported_as_real(self, tmp_path):
        batch_dir = self.build_ollama_batch(tmp_path)
        status = batch.work_batch(batch_dir, client=FixtureClient())
        usage = status["usage"]
        assert status["real_model_calls"] is False
        assert usage["actual_model_calls"] == 0
        assert usage["fixture_calls"] == 1
        connection = sqlite3.connect(batch_dir / "queue.sqlite3")
        try:
            stored = connection.execute(
                "SELECT completion FROM calls WHERE completion IS NOT NULL ORDER BY ordinal").fetchall()
        finally:
            connection.close()
        assert stored
        assert all(json.loads(row[0])["metadata"]["backend"] == "fixture" for row in stored)

    def test_cached_fixture_completions_are_not_replayed_by_a_different_client(self, tmp_path):
        batch_dir = self.build_ollama_batch(tmp_path)
        batch.work_batch(batch_dir, client=FixtureClient())

        class RealOllamaStandIn:
            """A differently implemented client standing in for the real one."""

            timeout = 180

            def __init__(self):
                self.seen = []

            def complete(self, messages, purpose, response_schema=None):
                self.seen.append(purpose)
                return Completion("Автономная проверка", {"input_tokens": 3, "output_tokens": 2}, 0,
                                  {"backend": "ollama", "network_request_performed": True})

        stand_in = RealOllamaStandIn()
        status = batch.work_batch(batch_dir, client=stand_in)
        assert status["usage"]["cache_hits"] == 0
        assert stand_in.seen == []
        assert status["jobs"][0]["status"] != "succeeded"
