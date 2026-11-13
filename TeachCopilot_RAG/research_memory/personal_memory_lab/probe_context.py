"""Probe the local model and record observed resource use, without inference claims."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


def gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return {"returncode": result.returncode, "output": result.stdout.strip()}


def request(path, payload=None, timeout=600):
    opener = build_opener(ProxyHandler({}))
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request("http://127.0.0.1:11435" + path, data=data, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def probe(context_tokens):
    # The number of repetitions is calibrated from actual prompt_eval_count, not
    # claimed to be the exact model-token count. It intentionally stays below the
    # requested window to reserve room for generation and the chat template.
    repetitions = 450 if context_tokens == 16384 else 900
    passage = ("На занятии учитель попросил объяснить решение. Ребёнок сначала сомневался, "
               "а после вопроса проверил ответ и описал один следующий шаг.\n")
    prompt = passage * repetitions + "В конце ответь одним словом: готово."
    before = gpu()
    started = time.perf_counter()
    result = request("/api/generate", {
        "model": "gemma4:12b", "prompt": prompt, "stream": False, "think": False,
        "keep_alive": "5m", "options": {"num_ctx": context_tokens, "num_predict": 8,
                                            "seed": 42, "temperature": 0},
    })
    elapsed = time.perf_counter() - started
    running = request("/api/ps")
    return {"context_tokens": context_tokens, "gpu_before": before, "gpu_after": gpu(),
            "elapsed_seconds": round(elapsed, 3), "result": result, "running_models": running}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, choices=(16384, 32768), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = probe(args.context)
    except Exception as exc:
        payload = {"context_tokens": args.context, "error_type": type(exc).__name__, "error": str(exc),
                   "gpu_after": gpu()}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"result", "running_models"}},
                     ensure_ascii=False, indent=2))
    if "result" in payload:
        print(json.dumps({k: payload["result"].get(k) for k in (
            "response", "done", "done_reason", "prompt_eval_count", "eval_count", "total_duration")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
