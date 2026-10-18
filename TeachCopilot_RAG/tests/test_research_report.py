"""Reporting must preserve missing answers and respect the calibration gate."""

import json

import pytest

from scripts.report_memory_research import (
    AXES, METHODS, Batch, build_report, dev_comparison, paired_history_analysis,
    sha, workflow_labels, write_json, write_rows,
)


def make_batch(path, *, omitted=(), statuses=None, histories=2):
    path.mkdir()
    snapshots = path / "snapshots"
    snapshots.mkdir()
    history_rows, requests, results = [], [], []
    statuses = statuses or {}
    for number in range(histories):
        history_id = f"h{number:02d}"
        history_rows.append({"history_id": history_id, "data_origin": "synthetic",
            "profile_as_of_cutoff": {"topic_id": "math", "grade": 4},
            "lessons": [{"lesson_id": f"{history_id}-lesson", "observed_on": "2026-10-10",
                         "turns": [{"turn_id": f"{history_id}-t", "role": "student", "text": "Ответ 102"}]}]})
        for question in range(2):
            request_id = f"{history_id}-q{question}"
            requests.append({"history_id": history_id, "request_id": request_id, "text": "Следующий шаг?"})
            for method in METHODS:
                identity = f"{history_id}:{request_id}:{method}"
                if identity in omitted:
                    continue
                status = statuses.get(identity, "ok")
                results.append({"result_id": identity, "history_id": history_id, "request_id": request_id,
                    "method": method, "request_text": "Следующий шаг?", "status": status,
                    "action": {"action": "independent_check", "reason": "Проверить перенос", "review_required": True,
                               "evidence_turn_ids": [], "unknowns": []} if status == "ok" else None,
                    "profile_as_of_cutoff": history_rows[-1]["profile_as_of_cutoff"], "memory_context": "",
                    "source_turns": []})
    write_rows(snapshots / "histories.jsonl", history_rows)
    write_rows(snapshots / "requests.jsonl", requests)
    write_rows(path / "results.jsonl", results)
    write_json(path / "batch.json", {"history_ids": [h["history_id"] for h in history_rows],
                                    "methods": METHODS, "backend": "ollama", "split": "heldout"})
    return path


def make_workflow(path, batch, *, qualified=True):
    path.mkdir()
    (path / "calibration").mkdir()
    write_json(path / "workflow.json", {"batch_dir": str(batch.path),
               "sha256": {"batch_definition": sha(batch.path / "batch.json")},
               "judge_config": {"backend": "spiderssense", "protocol": "context-v3"}})
    write_json(path / "report.json", {"status": "completed" if qualified else "calibration_not_qualified",
               "calibration_qualified": qualified, "contract_only": False})
    write_json(path / "calibration/summary.json", {"qualified": qualified})
    if not qualified:
        return
    prepared = path / "prepared"
    prepared.mkdir()
    (path / "answers").mkdir()
    cases, judgements, mapping = [], [], {}
    for number, row in enumerate(batch.rows):
        code = f"J{number:04d}"
        cases.append({"answer_code": code, "execution_status": row["status"], "contract_only": False,
                      "payload": {"answer": row["action"]}})
        mapping[code] = {k: row[k] for k in ("result_id", "history_id", "request_id", "method")}
        judgements.append({"answer_code": code, "status": "assessed", "judgement": {
            "verdict": "acceptable", "utility": "useful", "scores": {k: 2 for k in AXES},
            "findings": [], "rationale": "Поддерживается историей"}})
    write_rows(prepared / "cases.jsonl", cases)
    write_json(prepared / "mapping.private.json", mapping)
    write_rows(path / "answers/judgements.jsonl", judgements)
    write_json(prepared / "manifest.json", {"input_sha256": {"results": sha(batch.path / "results.jsonl"),
                                                           "histories": sha(batch.path / "snapshots/histories.jsonl")},
                                           "cases_sha256": sha(prepared / "cases.jsonl")})


def test_missing_and_failed_answers_remain_in_denominator(tmp_path):
    missing = "h00:h00-q0:structured"
    failure = "h01:h01-q1:tail"
    before = make_batch(tmp_path / "before")
    after = make_batch(tmp_path / "after", omitted=[missing], statuses={failure: "action_error"})
    output = tmp_path / "report"
    report = build_report(before, after, after, output)
    assert report["dev"]["planned"] == 16
    assert report["dev"]["transitions"]["new_failure"] == 2
    assert report["heldout"]["methods"]["structured"]["planned"] == 4
    assert report["heldout"]["methods"]["structured"]["technical_ok"] == 3
    assert report["heldout"]["methods"]["structured"]["not_assessed"] == 4
    assert report["heldout"]["planned"] == 16
    assert report["heldout"]["protocol_denominator_matches"] is False
    assert (output / "PROCESSING.json").exists()
    assert report["winner"] is None


def test_failed_calibration_never_opens_mapping_or_uses_labels(tmp_path):
    batch = Batch(make_batch(tmp_path / "batch"))
    workflow = tmp_path / "workflow"
    make_workflow(workflow, batch, qualified=False)
    # Deliberately invalid mapping must never be read behind the failed gate.
    (workflow / "prepared").mkdir()
    (workflow / "prepared/mapping.private.json").write_text("not json", encoding="utf-8")
    labels, gate, _ = workflow_labels(workflow, batch)
    assert labels == {}
    assert gate["semantic_labels_used"] is False


def test_qualified_workflow_requires_matching_results(tmp_path):
    batch = Batch(make_batch(tmp_path / "batch"))
    workflow = tmp_path / "workflow"
    make_workflow(workflow, batch)
    labels, gate, _ = workflow_labels(workflow, batch)
    assert len(labels) == 16
    assert gate["semantic_labels_used"] is True
    mapping_path = workflow / "prepared/mapping.private.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["J0000"]["method"] = "summary"
    write_json(mapping_path, mapping)
    with pytest.raises(ValueError, match="identity mismatch"):
        workflow_labels(workflow, batch)


def test_pairs_use_histories_and_complete_semantic_coverage(tmp_path):
    batch = Batch(make_batch(tmp_path / "batch"))
    rows = []
    for row in batch.rows:
        unassessed = row["method"] == "structured" and row["request_id"] == "h00-q0"
        rows.append({**row, "assessment_status": "not_assessed" if unassessed else "assessed",
                     "judgement": {"verdict": "acceptable", "utility": "useful"}})
    result = paired_history_analysis(rows)
    pair = next(p for p in result["pairs"] if p["first"] == "tail" and p["second"] == "structured"
                and p["metric"] == "useful_fraction")
    assert pair["paired_histories"] == 1
    assert pair["planned_shared_histories"] == 2
    assert result["unit"] == "history"


def test_preselected_packet_hides_methods_and_keeps_private_mapping(tmp_path):
    batch = make_batch(tmp_path / "batch", histories=5)
    selection = tmp_path / "selection.json"
    write_json(selection, {"status": "preregistered_before_generation",
                          "cases": [{"history_id": f"h{i:02d}", "request_id": f"h{i:02d}-q0"} for i in range(5)]})
    output = tmp_path / "report"
    report = build_report(batch, batch, batch, output, review_selection=selection)
    assert report["human_review"]["answers"] == 20
    public_text = (output / "human_review/answers.jsonl").read_text(encoding="utf-8")
    assert all(method not in public_text for method in METHODS)
    mapping = json.loads((output / "private/mapping.private.json").read_text(encoding="utf-8"))
    assert len(mapping) == 20
    assert {x["method"] for x in mapping.values()} == set(METHODS)
    with pytest.raises(FileExistsError):
        build_report(batch, batch, batch, output)


def test_dev_identity_changes_are_rejected(tmp_path):
    before = Batch(make_batch(tmp_path / "before"))
    after = Batch(make_batch(tmp_path / "after", histories=3))
    with pytest.raises(ValueError, match="different planned identities"):
        dev_comparison(before, after)


def test_v4_review_selection_schema_is_supported(tmp_path):
    batch = make_batch(tmp_path / "batch", histories=5)
    selection = tmp_path / "selection.json"
    write_json(selection, {"selection_origin": "preselected before generation", "seed": 20261017,
                          "expected_answers": 20, "methods": METHODS,
                          "requests": [f"h{i:02d}-q0" for i in range(5)]})
    report = build_report(batch, batch, batch, tmp_path / "report", review_selection=selection)
    assert report["human_review"]["preregistration_claim"] == "preselected before generation"
    assert report["human_review"]["answers"] == 20
