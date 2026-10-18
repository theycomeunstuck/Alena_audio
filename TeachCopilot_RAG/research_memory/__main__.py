from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import load_dataset
from .methods import METHODS
from .runner import ROOT, export_review, run


def _provider_client(path, model=None):
    from .spiderssense import SpiderssenseClient

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"model", "input_usd_per_million", "output_usd_per_million", "pricing_verified",
               "pricing_source", "ledger_path", "max_cost_usd", "max_calls", "max_output_tokens",
               "token_field", "key_file", "use_json_schema", "use_json_object", "timeout", "reasoning_effort"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unexpected provider configuration fields; credentials belong in the local key file")
    if model is not None and model != config.get("model"):
        raise ValueError("A different provider model requires its own verified pricing configuration")
    return SpiderssenseClient(**config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated educational-memory pilot; no automatic quality claims")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--histories", default=str(ROOT / "data" / "histories.dev.jsonl"))
        sub.add_argument("--requests", default=str(ROOT / "data" / "requests.dev.jsonl"))
        if name == "run":
            sub.add_argument("--backend", choices=("fixture", "ollama"), default="fixture")
            sub.add_argument("--output", required=True)
            sub.add_argument("--history-id", action="append", dest="history_ids")
            sub.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
            sub.add_argument("--budget-chars", type=int, default=4000)
            sub.add_argument("--max-calls", type=int)
            sub.add_argument("--config")
            sub.add_argument("--model")
            sub.add_argument("--base-url")
    sub = commands.add_parser("export-review")
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--output", required=True)
    sub = commands.add_parser("batch-create", help="Freeze a synthetic dataset and create a resumable local queue")
    sub.add_argument("--histories", default=str(ROOT / "data/histories.dev.jsonl"))
    sub.add_argument("--requests", default=str(ROOT / "data/requests.dev.jsonl"))
    sub.add_argument("--output", required=True)
    sub.add_argument("--backend", choices=("fixture", "ollama"), default="fixture")
    sub.add_argument("--history-id", action="append", dest="history_ids")
    sub.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    sub.add_argument("--budget-chars", type=int, default=4000)
    sub.add_argument("--max-calls", type=int, default=100)
    sub.add_argument("--max-seconds", type=float, default=1800)
    sub.add_argument("--max-input-tokens", type=int)
    sub.add_argument("--max-output-tokens-total", type=int)
    sub.add_argument("--config")
    sub.add_argument("--model")
    sub.add_argument("--base-url")
    for name in ("batch-work", "batch-status", "batch-report"):
        sub = commands.add_parser(name)
        sub.add_argument("--batch-dir", required=True)
        if name == "batch-work":
            sub.add_argument("--max-jobs", type=int)
    sub = commands.add_parser("prepare-judging", help="Export full-source cases without method labels")
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--histories", default=str(ROOT / "data/histories.dev.jsonl"))
    sub.add_argument("--output", required=True)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--judge-protocol", choices=("legacy", "context-v2", "context-v3"), default="legacy")
    sub = commands.add_parser("judge", help="Run a separate automatic judge on synthetic cases")
    sub.add_argument("--cases", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--backend", choices=("fixture", "ollama", "spiderssense"), default="fixture")
    sub.add_argument("--model")
    sub.add_argument("--base-url", default="http://127.0.0.1:11435")
    sub.add_argument("--max-calls", type=int, default=20)
    sub.add_argument("--timeout-seconds", type=float, default=180)
    sub.add_argument("--max-output-tokens", type=int, default=2048)
    sub.add_argument("--provider-config")
    sub.add_argument("--judge-protocol", choices=("legacy", "context-v2", "context-v3"), default="legacy")
    sub = commands.add_parser("calibration-summary")
    sub.add_argument("--judgements", required=True)
    sub.add_argument("--expected")
    sub.add_argument("--output", required=True)
    sub.add_argument("--judge-protocol", choices=("legacy", "context-v2", "context-v3"), default="legacy")
    sub = commands.add_parser("provider-preflight", help="Inspect local provider configuration without a model call")
    sub.add_argument("--provider-config", required=True)
    sub = commands.add_parser("compare", help="Aggregate completed, qualified machine assessments by memory method")
    sub.add_argument("--workflow-dir", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--budget-receipt")
    for name in ("workflow", "background"):
        sub = commands.add_parser(name, help="Run a finite workflow" if name == "workflow"
                                  else "Start the same finite workflow in a hidden background process")
        sub.add_argument("--batch-dir", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--judge-backend", choices=("fixture", "ollama", "spiderssense"), default="fixture")
        sub.add_argument("--judge-model")
        sub.add_argument("--judge-base-url", default="http://127.0.0.1:11435")
        sub.add_argument("--judge-max-calls", type=int, default=100)
        sub.add_argument("--judge-timeout-seconds", type=float, default=180)
        sub.add_argument("--judge-max-output-tokens", type=int, default=2048)
        sub.add_argument("--provider-config")
        sub.add_argument("--judge-protocol", choices=("legacy", "context-v2", "context-v3"), default="legacy")
        sub.add_argument("--calibration-cases")
        sub.add_argument("--calibration-expected")
        sub.add_argument("--reuse-calibration-from")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            dataset = load_dataset(args.histories, args.requests)
            result = {"valid": True, "histories": len(dataset.histories),
                      "requests": len(dataset.requests), "split": dataset.split,
                      "note": "Only input contracts checked; no reference answers opened or educational truth verified"}
        elif args.command == "run":
            manifest = run(args.histories, args.requests, args.output, backend=args.backend,
                           history_ids=args.history_ids, methods=args.methods, budget_chars=args.budget_chars,
                           max_calls=args.max_calls, config_path=args.config, model=args.model, base_url=args.base_url)
            result = {key: manifest[key] for key in ("status", "run_kind", "result_type", "scientific_result",
                                                     "real_model_calls", "result_count", "error_result_count", "performed_calls")}
        elif args.command == "export-review":
            result = {"exported_answers": export_review(args.run_dir, args.output),
                      "note": "Keep mapping.private.json separate from the blind reviewer"}
        elif args.command == "batch-create":
            from .batch import create_batch

            result = create_batch(args.histories, args.requests, args.output, backend=args.backend,
                                  history_ids=args.history_ids, methods=args.methods,
                                  budget_chars=args.budget_chars, max_calls=args.max_calls,
                                  max_seconds=args.max_seconds, max_input_tokens=args.max_input_tokens,
                                  max_output_tokens_total=args.max_output_tokens_total,
                                  config_path=args.config, model=args.model, base_url=args.base_url)
        elif args.command in {"batch-work", "batch-status", "batch-report"}:
            from .batch import batch_report, batch_status, work_batch

            if args.command == "batch-work":
                result = work_batch(args.batch_dir, max_jobs=args.max_jobs)
            else:
                result = (batch_report if args.command == "batch-report" else batch_status)(args.batch_dir)
        elif args.command == "prepare-judging":
            from .judge import prepare_judging

            result = prepare_judging(args.run_dir, args.histories, args.output, seed=args.seed,
                                     protocol=args.judge_protocol)
        elif args.command == "judge":
            from .judge import judge_cases

            client = None
            model = args.model
            output_tokens, timeout_seconds = args.max_output_tokens, args.timeout_seconds
            if args.backend == "spiderssense":
                if not args.provider_config:
                    raise ValueError("Spiderssense requires --provider-config with verified pricing and a persistent budget")
                client = _provider_client(args.provider_config, model)
                model = client.model
                output_tokens, timeout_seconds = client.max_output_tokens, client.timeout
            result = judge_cases(args.cases, args.output, backend=args.backend, model=model,
                                 base_url=args.base_url, max_calls=args.max_calls,
                                 timeout_seconds=timeout_seconds,
                                 max_output_tokens=output_tokens, client=client, protocol=args.judge_protocol)
        elif args.command == "calibration-summary":
            from .judge import summarize_calibration

            calibration_dir = ROOT / "data" / {"context-v2": "v3", "context-v3": "v4"}.get(args.judge_protocol, "")
            result = summarize_calibration(args.judgements,
                args.expected or calibration_dir / "calibration.expected.jsonl", args.output,
                protocol=args.judge_protocol)
        elif args.command == "provider-preflight":
            result = _provider_client(args.provider_config).preflight()
        elif args.command == "compare":
            from .analysis import build_comparison

            result = build_comparison(args.workflow_dir, args.output, budget_receipt_path=args.budget_receipt)
        elif args.command == "background":
            import os
            import subprocess
            import sys
            import time

            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.time_ns()
            stdout = output.with_name(output.name + f".{stamp}.stdout.log")
            stderr = output.with_name(output.name + f".{stamp}.stderr.log")
            child_args = sys.argv[1:]
            child_args[0] = "workflow"
            options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                       else {"start_new_session": True})
            with stdout.open("xb") as out_stream, stderr.open("xb") as err_stream:
                process = subprocess.Popen([sys.executable, "-m", "research_memory", *child_args],
                                           stdin=subprocess.DEVNULL, stdout=out_stream, stderr=err_stream,
                                           close_fds=True, **options)
            result = {"status": "started", "pid": process.pid, "report": str(output / "report.md"),
                      "stdout_log": str(stdout), "stderr_log": str(stderr),
                      "note": "Finite local worker; inspect its report for completion and calibration status"}
        else:
            from .workflow import run_workflow

            calibration_dir = ROOT / "data" / {"context-v2": "v3", "context-v3": "v4"}.get(args.judge_protocol, "")
            client, model = None, args.judge_model
            timeout_seconds, output_tokens = args.judge_timeout_seconds, args.judge_max_output_tokens
            if args.judge_backend == "spiderssense":
                if not args.provider_config:
                    raise ValueError("Spiderssense requires --provider-config")
                client = _provider_client(args.provider_config, model)
                model, timeout_seconds, output_tokens = client.model, client.timeout, client.max_output_tokens
            result = run_workflow(args.batch_dir, args.output, judge_backend=args.judge_backend,
                                  judge_model=model, judge_base_url=args.judge_base_url,
                                  judge_client=client, judge_max_calls=args.judge_max_calls,
                                  judge_timeout_seconds=timeout_seconds, judge_max_output_tokens=output_tokens,
                                  calibration_cases_path=args.calibration_cases or calibration_dir / "calibration.cases.jsonl",
                                  calibration_expected_path=args.calibration_expected or calibration_dir / "calibration.expected.jsonl",
                                  judge_protocol=args.judge_protocol,
                                  calibration_reuse_source=args.reuse_calibration_from)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("error_result_count", 0) else 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
