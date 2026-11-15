"""Finite, resumable generation–calibration–judging workflow for synthetic data."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from .batch import _sha, _versions, _write, batch_report, batch_status, work_batch
from .clients import OllamaClient
from .dataset import require
from .judge import ROOT, judge_cases, prepare_judging, prompt_path, summarize_calibration


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _lines(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@contextmanager
def _lock(output):
    with (output / "workflow.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another worker owns this workflow") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _empty(case, code):
    return {"answer_code": case["answer_code"], "status": "not_assessed", "judgement": None,
            "contract_only": case["contract_only"], "assessment_origin": "machine_only",
            "error": {"code": code}}


class _AttemptClient:
    def __init__(self, client, state, state_path):
        self.client, self.state, self.state_path = client, state, state_path

    def complete(self, *args, **kwargs):
        self.state["real_client_attempts"] += 1
        _write(self.state_path, self.state)
        return self.client.complete(*args, **kwargs)


def _read_completed(step, case, protocol="legacy"):
    output = step / "output"
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] not in {"completed", "completed_with_errors"}:
        return None
    require(manifest["input_sha256"]["cases"] == _sha(step / "input.jsonl")
            and manifest["prompt_sha256"] == _sha(prompt_path(protocol)), "Completed judge step hash mismatch")
    require(manifest.get("judge_protocol", "legacy") == protocol, "Completed judge protocol mismatch")
    rows = _rows(output / "judgements.jsonl")
    require(len(rows) == 1 and rows[0]["answer_code"] == case["answer_code"], "Invalid completed judge step")
    return rows[0]


def _group_completed(output, group, count):
    """True when every journaled step of a judge group already completed."""
    for number in range(1, count + 1):
        stamp_path = output / "steps" / group / f"{number:05d}" / "step.json"
        if not stamp_path.is_file():
            return False
        if json.loads(stamp_path.read_text(encoding="utf-8")).get("status") != "completed":
            return False
    return True


def _judge_group(cases, group, output, config, state, client):
    result = []
    protocol = config.get("protocol", "legacy")
    for number, case in enumerate(cases, 1):
        step = output / "steps" / group / f"{number:05d}"
        stamp_path = step / "step.json"
        step_input = json.dumps(case, ensure_ascii=False) + "\n"
        if stamp_path.exists():
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            require((step / "input.jsonl").read_text(encoding="utf-8") == step_input,
                    "Judge step input hash mismatch")
            if stamp["status"] == "completed":
                require(_sha(step / "output/judgements.jsonl") == stamp["judgements_sha256"],
                        "Completed judgement hash mismatch")
                row = _read_completed(step, case, protocol)
                require(row is not None, "Completed step lost its final manifest")
                result.append(row)
                continue
            if stamp["status"] == "unknown":
                result.append(_empty(case, "judge_error_unknown"))
                continue
            # A response may have been fully persisted before the workflow died.
            row = _read_completed(step, case, protocol)
            if row is not None:
                stamp.update(status="completed", judgements_sha256=_sha(step / "output/judgements.jsonl"))
                _write(stamp_path, stamp)
                result.append(row)
                continue
            if stamp["status"] == "running":
                stamp["status"] = "unknown"
                _write(stamp_path, stamp)
                result.append(_empty(case, "judge_error_unknown"))
                continue
        else:
            step.mkdir(parents=True, exist_ok=True)
            require(not (step / "output").exists(), "Unjournaled judge output requires inspection")
            (step / "input.jsonl").write_text(step_input, encoding="utf-8")
            stamp = {"status": "pending", "input_sha256": _sha(step / "input.jsonl")}
            _write(stamp_path, stamp)
        eligible = case["execution_status"] == "ok" and case["payload"]["answer"] is not None
        if eligible and state["judge_calls_reserved"] >= config["max_calls"]:
            result.append(_empty(case, "judge_max_calls_exceeded"))
            continue
        if eligible:
            state["judge_calls_reserved"] += 1
            if config["backend"] == "fixture":
                state["fixture_call_reservations"] += 1
            _write(output / "state.json", state)
        stamp["status"] = "running"
        _write(stamp_path, stamp)
        try:
            judge_cases(step / "input.jsonl", step / "output", backend=config["backend"],
                        model=config["model"], base_url=config["base_url"], max_calls=1,
                        timeout_seconds=config["timeout_seconds"], max_output_tokens=config["max_output_tokens"],
                        client=None if config["backend"] == "fixture" else client, protocol=protocol)
            row = _read_completed(step, case, protocol)
            require(row is not None, "Judge did not persist a completed step")
            stamp.update(status="completed", judgements_sha256=_sha(step / "output/judgements.jsonl"))
        except Exception as exc:
            stamp.update(status="unknown", error_type=type(exc).__name__)
            row = _empty(case, "judge_error_unknown")
        _write(stamp_path, stamp)
        provider = getattr(client, "client", client)
        if config["backend"] == "spiderssense" and callable(getattr(provider, "preflight", None)):
            state["provider_accounting"] = provider.preflight()
            _write(output / "state.json", state)
        result.append(row)
    _lines(output / group / "judgements.jsonl", result)
    return result


def _report(output, state, generation, calibration=None, answers=None):
    answers = answers or []
    counts = Counter(row["status"] for row in answers)
    verdicts = Counter(row["judgement"]["verdict"] for row in answers if row.get("judgement"))
    utilities = Counter(row["judgement"]["utility"] for row in answers
                        if row.get("judgement") and "utility" in row["judgement"])
    errors = Counter(row["error"]["code"] for row in answers if row.get("error"))
    report = {"status": state["status"], "scientific_result": False,
              "contract_only": state["judge_backend"] == "fixture" or generation["run_kind"] == "contract_only",
              "generation_job_counts": generation["job_counts"], "generation_expected_results": generation["expected_results"],
              "calibration_qualified": calibration.get("qualified") if calibration else None,
              "judge_calls_reserved": state["judge_calls_reserved"],
              "real_client_attempts": state["real_client_attempts"],
              "fixture_call_reservations": state["fixture_call_reservations"],
              "answer_counts": dict(counts), "answer_verdict_counts": dict(verdicts),
              "judge_protocol": state.get("judge_protocol", "legacy"),
              "answer_utility_counts": dict(utilities),
              "reused_calibration_cases": state.get("reused_calibration_cases", 0),
              "transport_retry_cases": state.get("transport_retry_cases", 0),
              "answer_error_counts": dict(errors), "answer_rows": len(answers),
              "provider_accounting": state.get("provider_accounting"),
              "assessment_origin": "machine_only; no expert validation", "mapping_opened": False}
    _write(output / "report.json", report)
    text = ("# Отчёт конечного фонового процесса\n\n"
            f"Состояние: `{state['status']}`. Научный результат: нет. "
            + ("Проверка контракта; смысловая пригодность не измерялась.\n\n" if report["contract_only"] else
               "Оценка одного машинного судьи; экспертной проверки нет.\n\n")
            + f"Генерация: {json.dumps(generation['job_counts'], ensure_ascii=False)}; "
            f"ожидаемых ответов {generation['expected_results']}.\n\n"
            + f"Калибровка qualified: {report['calibration_qualified']}. "
            "Непрошедшая смысловую калибровку модель не оценивает основные ответы.\n\n"
            + f"Оценки ответов: {json.dumps(dict(counts), ensure_ascii=False)}. "
            f"Вердикты: {json.dumps(dict(verdicts), ensure_ascii=False)}. "
            f"Полезность отдельно: {json.dumps(dict(utilities), ensure_ascii=False)}. "
            f"Ошибки/причины пропуска: {json.dumps(dict(errors), ensure_ascii=False)}.\n\n"
            + f"Зарезервировано вызовов судьи: {state['judge_calls_reserved']}; "
            f"реальных попыток клиента: {state['real_client_attempts']}; "
            f"fixture-резервов: {state['fixture_call_reservations']}. "
            f"Сохранённых оценок калибровки перенесено: {report['reused_calibration_cases']}; "
            f"транспортных случаев для однократного повтора: {report['transport_retry_cases']}. "
            "Fixture не создаёт модельных токенов или денежной стоимости.\n\n"
            "Завершённые оценки используются повторно. Незавершённый вызов с неизвестным исходом "
            "получает judge_error_unknown и автоматически не повторяется. "
            "Лимит вызовов сохраняется между запусками. Результат не подбирается повторными попытками. "
            "Неудачные генерации остаются not_assessed. mapping.private.json не открывался.\n")
    (output / "report.md").write_text(text, encoding="utf-8")
    return report


def run_workflow(batch_dir, output, *, judge_backend="fixture", judge_model=None,
                 judge_base_url="http://127.0.0.1:11435", judge_client=None, judge_max_calls=100,
                 judge_timeout_seconds=180, judge_max_output_tokens=2048,
                 calibration_cases_path=ROOT / "data/calibration.cases.jsonl",
                 calibration_expected_path=ROOT / "data/calibration.expected.jsonl", generation_client=None,
                 judge_protocol="legacy", calibration_reuse_source=None):
    """Resume or complete a fixed finite workflow; this function starts no daemon."""
    batch_dir, output = Path(batch_dir).resolve(), Path(output)
    judge_prompt = prompt_path(judge_protocol)
    require(judge_backend in {"fixture", "ollama", "spiderssense"}, "Unsupported judge backend")
    require(type(judge_max_calls) is int and judge_max_calls >= 0, "Invalid judge_max_calls")
    require(judge_backend != "fixture" or judge_client is None, "Fixture judging uses the built-in stub")
    require(judge_backend == "fixture" or isinstance(judge_model, str) and bool(judge_model),
            "An explicit judge model is required")
    require(judge_backend != "spiderssense" or judge_client is not None, "Provider client is required")
    require(judge_protocol != "context-v3" or calibration_reuse_source is None,
            "The held-out protocol permits one calibration only; recovery is disabled")
    if judge_client is not None and hasattr(judge_client, "model"):
        require(judge_client.model == judge_model, "Injected judge model differs from the workflow model")
    if judge_client is not None and hasattr(judge_client, "max_output_tokens"):
        require(judge_client.max_output_tokens <= judge_max_output_tokens,
                "Injected judge output limit exceeds the workflow limit")
    config = {"backend": judge_backend, "model": judge_model, "base_url": judge_base_url,
              "max_calls": judge_max_calls, "timeout_seconds": judge_timeout_seconds,
              "max_output_tokens": judge_max_output_tokens}
    if judge_protocol != "legacy":
        config["protocol"] = judge_protocol
    if judge_client is not None and callable(getattr(judge_client, "preflight", None)):
        preflight = judge_client.preflight()
        require(isinstance(preflight.get("configuration_hash"), str), "Provider preflight needs a configuration hash")
        config["provider_configuration_hash"] = preflight["configuration_hash"]
        if hasattr(judge_client, "ledger_path"):
            config["provider_ledger_path"] = str(Path(judge_client.ledger_path).resolve())
    definition = {"workflow_version": 1, "batch_dir": str(batch_dir), "judge_config": config,
                  "sha256": {"batch_definition": _sha(batch_dir / "batch.json"),
                             "calibration_cases": _sha(calibration_cases_path),
                             "calibration_expected": _sha(calibration_expected_path),
                             "judge_prompt": _sha(judge_prompt)}, "versions": _versions()}
    if calibration_reuse_source is not None:
        from .reuse import describe_calibration_reuse

        definition["calibration_reuse"] = describe_calibration_reuse(calibration_reuse_source, definition)
    output.mkdir(parents=True, exist_ok=True)
    with _lock(output):
        workflow_path, state_path = output / "workflow.json", output / "state.json"
        if workflow_path.exists():
            require(json.loads(workflow_path.read_text(encoding="utf-8")) == definition,
                    "Workflow hash or configuration mismatch; create a new output")
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            _write(workflow_path, definition)
            state = {"status": "created", "judge_backend": judge_backend, "judge_calls_reserved": 0,
                     "real_client_attempts": 0, "fixture_call_reservations": 0,
                     "judge_protocol": judge_protocol}
            _write(state_path, state)
            (output / "inputs").mkdir()
            for name, source in (("calibration.cases.jsonl", calibration_cases_path),
                                 ("calibration.expected.jsonl", calibration_expected_path)):
                (output / "inputs" / name).write_bytes(Path(source).read_bytes())
        for name, key in (("calibration.cases.jsonl", "calibration_cases"),
                          ("calibration.expected.jsonl", "calibration_expected")):
            require(_sha(output / "inputs" / name) == definition["sha256"][key], "Frozen calibration input hash mismatch")
        generation = batch_status(batch_dir)
        if any(generation["job_counts"][key] for key in ("pending", "running")):
            generation = work_batch(batch_dir, client=generation_client)
        else:
            generation = batch_report(batch_dir)
        if any(generation["job_counts"][key] for key in ("pending", "running", "blocked")):
            state["status"] = "generation_incomplete"
            _write(state_path, state)
            return _report(output, state, generation)
        result_hash = _sha(batch_dir / "results.jsonl")
        if "generation_results_sha256" in state:
            require(state["generation_results_sha256"] == result_hash, "Generation results hash mismatch")
        else:
            state["generation_results_sha256"] = result_hash
            _write(state_path, state)
        prepared = output / "prepared"
        if prepared.exists() and not (prepared / "manifest.json").exists():
            os.replace(prepared, output / f"prepared.partial.{time.time_ns()}")
        if not prepared.exists():
            temporary = output / f"prepared.build.{time.time_ns()}"
            prepare_judging(batch_dir, batch_dir / "snapshots/histories.jsonl", temporary,
                            protocol=judge_protocol)
            os.replace(temporary, prepared)
        prepared_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        require(prepared_manifest["input_sha256"]["results"] == result_hash
                and prepared_manifest["input_sha256"]["histories"] == _sha(batch_dir / "snapshots/histories.jsonl")
                and prepared_manifest["cases_sha256"] == _sha(prepared / "cases.jsonl"),
                "Prepared cases provenance mismatch")
        require("prepared_cases_sha256" not in state
                or _sha(prepared / "cases.jsonl") == state["prepared_cases_sha256"],
                "Prepared cases hash mismatch")
        state["prepared_cases_sha256"] = _sha(prepared / "cases.jsonl")
        _write(state_path, state)
        cases = _rows(prepared / "cases.jsonl")
        if "calibration_reuse" in definition:
            from .reuse import import_calibration_reuse

            state.update(import_calibration_reuse(definition["calibration_reuse"], output))
            _write(state_path, state)
        if judge_backend == "spiderssense" and callable(getattr(judge_client, "preflight", None)):
            provider = judge_client.preflight()
            state["provider_accounting"] = provider
            # A workflow whose judge steps all completed has no provider work
            # left: resuming it must replay those steps and report their rows,
            # never overwrite the assessed results with unassessed placeholder
            # rows just because the credential is absent or the ledger is
            # blocked right now.
            finished = (_group_completed(output, "calibration",
                                        len(_rows(output / "inputs/calibration.cases.jsonl")))
                        and _group_completed(output, "answers", len(cases)))
            if (not provider.get("key_present") or provider.get("blocked")) and not finished:
                state["status"] = "provider_budget_blocked" if provider.get("blocked") else "awaiting_credentials"
                _write(state_path, state)
                answers = [_empty(case, state["status"]) for case in cases]
                _lines(output / "answers/judgements.jsonl", answers)
                return _report(output, state, generation, answers=answers)
        if judge_client is None and judge_backend == "ollama":
            judge_client = OllamaClient(base_url=judge_base_url, model=judge_model, timeout=judge_timeout_seconds,
                                        context_tokens=16384, max_output_tokens=judge_max_output_tokens)
        client = _AttemptClient(judge_client, state, state_path) if judge_backend != "fixture" else None
        state["status"] = "calibrating"
        _write(state_path, state)
        _judge_group(_rows(output / "inputs/calibration.cases.jsonl"), "calibration", output, config, state, client)
        calibration_path = output / "calibration/summary.json"
        summary_build = calibration_path.with_name(f"summary.build.{time.time_ns()}.json")
        calibration = summarize_calibration(output / "calibration/judgements.jsonl",
                                           output / "inputs/calibration.expected.jsonl", summary_build,
                                           protocol=judge_protocol)
        os.replace(summary_build, calibration_path)
        if not calibration["qualified"] and judge_backend != "fixture":
            answers = [_empty(case, "calibration_not_qualified") for case in cases]
            _lines(output / "answers/judgements.jsonl", answers)
            state["status"] = "calibration_not_qualified"
        else:
            state["status"] = "judging"
            _write(state_path, state)
            answers = _judge_group(cases, "answers", output, config, state, client)
            state["status"] = "completed_contract_only" if judge_backend == "fixture" else "completed"
            if any(row["status"] != "assessed" for row in answers):
                state["status"] += "_with_unassessed"
        if judge_backend == "spiderssense" and callable(getattr(judge_client, "preflight", None)):
            state["provider_accounting"] = judge_client.preflight()
        _write(state_path, state)
        return _report(output, state, generation, calibration, answers)
