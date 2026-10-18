"""One finite dev, held-out, calibration and assessment package, without retries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_identity(base_url: str, model: str) -> dict:
    opener = build_opener(ProxyHandler({}))
    with opener.open(base_url + "/api/tags", timeout=15) as response:
        data = json.load(response)
    matches = [item for item in data["models"] if item["name"] == model]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise ValueError("Pinned local model is unavailable")
    return {key: matches[0][key] for key in ("name", "digest", "size", "details")}


def execute(output: Path, provider_config: Path) -> dict:
    from research_memory.batch import _versions, batch_status, create_batch, work_batch
    from research_memory.__main__ import _provider_client
    from research_memory.workflow import run_workflow

    output = output.resolve()
    data = ROOT / "research_memory/data"
    config = ROOT / "research_memory/config.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    files = [config, *sorted((data / "v4").glob("*.json*"))]
    identity = model_identity(settings["base_url"], settings["model"])
    manifest = {"protocol": "memory-final-v4", "code_and_prompts": _versions(),
        "inputs": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "model": identity, "generation": settings, "seed": 42, "temperature": 0,
        "human_content_review": "not_performed", "scientific_result": False,
        "evaluation_config_sha256": sha(provider_config),
        "policy": "One full dev batch, one held-out batch, one calibration; no recovery or regeneration"}
    if output.exists():
        if json.loads((output / "freeze.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Frozen code, data, model or provider configuration changed")
    else:
        output.mkdir(parents=True)
        write(output / "freeze.json", manifest)
    state_path = output / "state.json"
    state = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "starting"}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    try:
        for phase, histories, requests, count in (
            ("dev", data / "v2/histories.dev.jsonl", data / "v2/requests.dev.jsonl", 64),
            ("heldout", data / "v4/histories.test.jsonl", data / "v4/requests.test.jsonl", 128),
        ):
            state["status"] = phase + "_generation"
            write(state_path, state)
            print(state["status"], flush=True)
            batch = output / phase
            if not batch.exists():
                create_batch(histories, requests, batch, backend="ollama", budget_chars=4000,
                    max_calls=600 if phase == "heldout" else 220, max_seconds=21600,
                    config_path=config)
            before = batch_status(batch)
            if before["job_counts"]["pending"] or before["job_counts"]["running"]:
                report = work_batch(batch)
            else:
                report = before
            if model_identity(settings["base_url"], settings["model"]) != identity:
                raise ValueError("Local model identity changed during generation")
            if _versions() != manifest["code_and_prompts"]:
                raise ValueError("Research source changed during generation")
            state[phase] = report
            write(state_path, state)
            if any(report["job_counts"][status] for status in ("pending", "running", "blocked")):
                raise RuntimeError("Finite generation did not complete; results retained")
            rows = [json.loads(line) for line in (batch / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(rows) != count:
                raise RuntimeError("Planned answer denominator mismatch")
        state["status"] = "calibration_and_assessment"
        write(state_path, state)
        print(state["status"], flush=True)
        client = _provider_client(provider_config)
        report = run_workflow(output / "heldout", output / "assessment",
            judge_backend="spiderssense", judge_model=client.model, judge_client=client,
            judge_max_calls=152, judge_timeout_seconds=client.timeout,
            judge_max_output_tokens=client.max_output_tokens, judge_protocol="context-v3",
            calibration_cases_path=data / "v4/calibration.cases.jsonl",
            calibration_expected_path=data / "v4/calibration.expected.jsonl")
        state["assessment"] = report
        state["status"] = report["status"]
        from scripts.report_memory_research import build_report
        report_dir = output / "final_report"
        if not report_dir.exists():
            build_report(ROOT / ".local/research_memory/background_canonical_sources_v4_20261016",
                output / "dev", output / "heldout", report_dir,
                workflow=output / "assessment", gold=data / "v4/gold.test.jsonl",
                review_selection=data / "v4/review.selection.json",
                budget_receipt=provider_config.parent / "receipt.json")
        state["final_report"] = str(report_dir)
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)
        print(state["status"], flush=True)
        return state
    except Exception as exc:
        state.update(status="failed", safe_error=type(exc).__name__, error=str(exc),
                     finished_at_utc=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    args = parser.parse_args()
    execute(args.output, args.provider_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
