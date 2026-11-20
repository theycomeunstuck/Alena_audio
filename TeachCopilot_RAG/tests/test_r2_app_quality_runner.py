"""Round-2 P1.3: the live-quality runner's gold-safety and evidence contract.

The runner consumes the peer's frozen manifest (gold inline, per the frozen
`evaluation_protocol`), must record every row before scoring, never leak gold
into generation prompts, score semantically against recorded sources, resolve
frozen source keys to actual ids, refuse final-check overwrites, and stay
honestly unscored without a judge.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import importlib.util


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "r2_live_quality_check",
        Path(__file__).resolve().parents[1] / "scripts" / "r2_live_quality_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _manifest_and_gold(tmp_path: Path):
    questions = [{"question_id": f"q{i}", "history_id": "dev-1",
                  "case": "changed_interest", "question": f"Вопрос {i} про прошлое?",
                  "expected_answer": f"Ожидаемый ответ {i} уникальной длины",
                  "expected_source_keys": [f"a.key-{i}"], "must_not_sources": [],
                  "notes": "", "probe_query": f"probe {i}"} for i in range(1, 7)]
    manifest = {"protocol": "zcode-r2-v1", "histories": [
        {"history_id": "dev-1", "role": "development", "sessions": 6}],
        "questions": questions}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest, manifest_path


def _fake_ask(prompt_log):
    def ask(row):
        prompt_log.append(row["prompt"])
        return {"request": {"messages": [{"role": "user", "content": row["prompt"]}]},
                "answer": f"Автоматический ответ на вопрос {row['question_id']}",
                "probe": {"source_ids": [f"src-{row['question_id'][1:]}"]},
                "trace": {"available": False, "phases_seen": [],
                          "retrieval": None, "answer_prompt": None,
                          "route": None, "generation_called": None,
                          "technical_error": None, "hydrated_quote_count": 0},
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "model_call_count": 2}
    return ask


def _judge_calls(judge_log, verdicts=None):
    def judge(system, question, body):
        judge_log.append({"question": question, "body": body})
        return verdicts or {"relevant": True, "supported": True, "complete": True,
                            "honest_missing": False, "false_claims": [],
                            "reason": "по источникам"}
    return judge


class TestManifestContract:
    def test_split_gold_strips_every_gold_field_from_prompt_rows(self, tmp_path):
        manifest, _path = _manifest_and_gold(tmp_path)
        prompts, gold = runner.split_gold(manifest)
        assert len(prompts) == 6 and len(gold) == 6
        blob = json.dumps(prompts, ensure_ascii=False)
        for field in runner.GOLD_FIELDS:
            assert field not in blob
        for i in range(1, 7):
            assert f"Ожидаемый ответ {i}" not in blob
            assert gold[f"q{i}"]["expected_source_keys"] == [f"a.key-{i}"]

    def test_manifest_and_gold_validation(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        bad_protocol = dict(manifest, protocol="other")
        (tmp_path / "bad.json").write_text(json.dumps(bad_protocol), encoding="utf-8")
        with pytest.raises(runner.QualityCheckError, match="protocol"):
            runner.load_manifest(tmp_path / "bad.json")
        short = json.loads(json.dumps(manifest))
        short["questions"] = short["questions"][:3]
        (tmp_path / "short.json").write_text(json.dumps(short), encoding="utf-8")
        with pytest.raises(runner.QualityCheckError, match="6 questions"):
            runner.load_manifest(tmp_path / "short.json")
        # A final-check history must freeze its sessions.
        final = json.loads(json.dumps(manifest))
        final["histories"][0]["role"] = "final_check"
        final["histories"][0]["sessions"] = None
        (tmp_path / "final.json").write_text(json.dumps(final), encoding="utf-8")
        with pytest.raises(runner.QualityCheckError, match="frozen sessions"):
            runner.load_manifest(tmp_path / "final.json")
        # The real frozen manifest from the peer must load as-is.
        frozen = Path(__file__).resolve().parents[1] / "tests" / "r2_memory_quality" / "manifest.json"
        loaded, digest = runner.load_manifest(frozen)
        assert len(loaded["questions"]) == 18
        assert len(digest) == 64


class TestRunnerContract:
    def test_record_before_score_and_gold_never_in_prompts(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        prompt_log, judge_log = [], []
        key_to_id = {f"a.key-{i}": f"src-{i}" for i in range(1, 7)}
        result = runner.run(manifest_path, _fake_ask(prompt_log),
                            _judge_calls(judge_log), tmp_path / "out",
                            key_to_id=key_to_id)
        assert len(prompt_log) == 6 and len(judge_log) == 6
        for i in range(1, 7):
            assert f"Ожидаемый ответ {i}" not in "".join(prompt_log)
        assert all(entry["question"] == row["prompt"] for entry, row in
                   zip(judge_log, result["records"]))
        assert result["aggregate"]["supported"] == 6
        assert result["aggregate"]["positive_score_total"] == 6
        assert "not an independent human study" in result["assessment"]
        assert result["harness_protocol"] == "zcode-r3-night-v1"
        assert (tmp_path / "out" / "records" / "q1.json").exists()
        assert len(result["manifest_sha256"]) == 64

    def test_gold_leak_aborts_before_scoring(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        leaked = "Ожидаемый ответ 3 уникальной длины"

        def leaking_ask(row):
            return {"request": {"messages": [{"role": "user",
                                              "content": row["prompt"] + " " + leaked}]},
                    "answer": "x", "selected_source_ids": []}
        with pytest.raises(runner.QualityCheckError, match="gold leak"):
            runner.run(manifest_path, leaking_ask, None, tmp_path / "out2")

    def test_unmapped_frozen_keys_stay_honest(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        result = runner.run(manifest_path, _fake_ask([]), None, tmp_path / "out5",
                            key_to_id={"a.key-1": "src-1"})
        rows = {row["question_id"]: row for row in result["scored"]}
        # q1's frozen key maps to src-1 and the diagnostic probe found it: hit.
        assert rows["q1"]["probe_hit"] is True
        # q2's key has no mapping in this reduced key map: recorded honestly,
        # with no invented hit or miss.
        assert rows["q2"]["unmapped_keys"] == ["a.key-2"]
        assert rows["q2"]["probe_hit"] is None
        assert result["aggregate"]["probe_hit"] == 1

    def test_final_check_first_result_is_preserved(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        manifest["histories"][0]["role"] = "final_check"
        manifest["histories"][0]["sessions"] = 6
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        runner.run(manifest_path, _fake_ask([]), None, tmp_path / "final1")
        assert (tmp_path / "final1" / "final_check_attempt_1_dev-1.json").exists()
        with pytest.raises(runner.QualityCheckError, match="preserved"):
            runner.run(manifest_path, _fake_ask([]), None, tmp_path / "final1")

    def test_unavailable_judge_leaves_rows_unscored_not_faked(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        result = runner.run(manifest_path, _fake_ask([]), None, tmp_path / "out3")
        assert result["aggregate"]["unscored"] == 6
        assert all(row["verdict"] == "unscored" for row in result["scored"])

    def test_errors_and_misses_stay_in_the_result(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        key_to_id = {f"a.key-{i}": f"src-{i}" for i in range(1, 7)}

        def failing_ask(row):
            if row["question_id"] == "q3":
                raise RuntimeError("model timeout")
            return {"request": {}, "answer": "не нашёл этого в наших разговорах",
                    "probe": {"source_ids": ["wrong-id"]}, "usage": {}}

        judge = _judge_calls([], verdicts={"relevant": True, "supported": False,
                                           "complete": False, "honest_missing": True,
                                           "false_claims": [], "reason": "источников нет"})
        result = runner.run(manifest_path, failing_ask, judge, tmp_path / "out4",
                            key_to_id=key_to_id)
        rows = {row["question_id"]: row for row in result["scored"]}
        assert rows["q3"]["verdict"] == "error"
        # The gold here does not declare canonical absence: an honest-looking
        # absence report on a question whose truth exists is unsupported.
        assert rows["q1"]["verdict"] == "unsupported"
        assert rows["q1"]["honest_missing"] is True
        assert rows["q1"]["probe_hit"] is False
        assert result["aggregate"]["errors"] == 1
        assert result["aggregate"]["unsupported"] == 5
        assert result["aggregate"]["positive_score_total"] == 0

    def test_declared_absence_questions_keep_the_honest_missing_verdict(self, tmp_path):
        manifest, manifest_path = _manifest_and_gold(tmp_path)
        manifest["questions"][0]["absence"] = True
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        def asking(row):
            return {"request": {}, "answer": "в архиве этих данных нет",
                    "probe": {"source_ids": []}, "usage": {}}

        judge = _judge_calls([], verdicts={"relevant": True, "supported": False,
                                           "complete": True, "honest_missing": True,
                                           "false_claims": [], "reason": "явное сообщение об отсутствии"})
        result = runner.run(manifest_path, asking, judge, tmp_path / "out6",
                            key_to_id={})
        assert result["scored"][0]["verdict"] == "honest_missing_evidence"
        assert result["aggregate"]["honest_missing_evidence"] == 1
        assert result["aggregate"]["positive_score_total"] == 1
