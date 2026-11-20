"""Round-3 quality check over the frozen synthetic manifests (strict scorer).

Consumes `tests/r2_memory_quality/manifest.json` (frozen at commit, protocol
`zcode-r2-v1`): three six-session histories, 18 questions with inline gold
(`expected_answer`, `expected_source_keys`). Two strictly separated phases:

1. **Record** – the manifest is split first: gold fields are stripped from
   every prompt row, each question is asked through the real local API path,
   and the immutable artifacts (prompt row, request trace, answer text,
   timings, model-call counts, token usage, errors) are written atomically
   before any scoring. A leak scan aborts the run if any gold string ever
   appears in a recorded prompt.

2. **Score** – only after all rows are recorded. The evaluator loads its
   canonical truth from the **complete** synthetic store dump (every source
   text, author role, instant and correction link), never from the subset the
   answer path happened to retrieve: a missed source is not `(text
   unavailable)` truth and not proof the archive lacks the fact. The judge
   answers a strict typed schema (`relevant`, `supported`, `complete`,
   `honest_missing`, `false_claims`, `reason`); string booleans, missing
   fields, empty reasons and internally inconsistent judgements are visible
   `judge_failure` rows – never coerced into a pass. A generic lesson
   clarification is scored as a non-answer, and an honest absence statement
   is scored separately from a supported answer. Raw judge replies are
   preserved verbatim.

Retrieval metrics are separated per the request trace contract:
`probe_hit` (optional standalone diagnostic), `product_retrieval_hit`,
`selected_hit` and `delivered_hit` (evidence actually admitted into the final
answer prompt) are computed from the recorded trace, and a probe's source
ids are never reused as delivered evidence. Denominators stay visible:
questions with unmapped frozen keys keep `None` instead of an invented hit
or miss.

The assessment is automated (LLM judge + deterministic checks). It is not an
independent human study. The first final-check result per final history
(`role == "final_check"`) is preserved: later runs write to new output
directories, never over it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL = "zcode-r2-v1"  # frozen manifest protocol (peer artifact, immutable)
HARNESS_PROTOCOL = "zcode-r3-night-v1"
QUESTIONS_PER_HISTORY = 6
GOLD_FIELDS = ("expected_answer", "expected_source_keys", "must_not_sources", "notes")
ANSWER_TIMEOUT_SECONDS = 300

JUDGE_BOOL_FIELDS = ("relevant", "supported", "complete", "honest_missing")


class QualityCheckError(RuntimeError):
    """A configuration or evidence problem that must abort the run."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: Path) -> tuple[dict, str]:
    """Load and validate the frozen manifest; returns (manifest, sha256).

    The sha is computed over LF-normalized bytes so a checkout with CRLF
    conversion hashes identically to the committed blob (found during the
    follow-up review: B's working tree checked the manifest out with CRLF,
    and the raw-byte sha falsely reported a manifest mismatch).
    """
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise QualityCheckError(f"manifest protocol is {manifest.get('protocol')!r}, "
                                f"expected {PROTOCOL!r}")
    histories = manifest.get("histories")
    questions = manifest.get("questions")
    if not isinstance(histories, list) or not histories:
        raise QualityCheckError("manifest has no histories")
    if not isinstance(questions, list) or not questions:
        raise QualityCheckError("manifest has no questions")
    by_history: dict[str, int] = {}
    seen_ids: set[str] = set()
    for question in questions:
        for field in ("question_id", "history_id", "question"):
            if not question.get(field):
                raise QualityCheckError(f"manifest question is missing {field!r}")
        if question["question_id"] in seen_ids:
            raise QualityCheckError(
                f"manifest question {question['question_id']!r} is duplicated: a repeated "
                "row would silently overwrite its gold entry and its record file")
        seen_ids.add(question["question_id"])
        by_history[question["history_id"]] = by_history.get(question["history_id"], 0) + 1
    for history in histories:
        count = by_history.get(history["history_id"], 0)
        if count != QUESTIONS_PER_HISTORY:
            raise QualityCheckError(
                f"history {history['history_id']!r} must freeze exactly "
                f"{QUESTIONS_PER_HISTORY} questions, found {count}")
        if history.get("role") == "final_check" and not history.get("sessions"):
            raise QualityCheckError(
                f"final-check history {history['history_id']!r} has no frozen sessions; "
                "refusing to score questions against fabricated context")
    extra = set(by_history) - {h["history_id"] for h in histories}
    if extra:
        raise QualityCheckError(f"questions reference unknown histories: {sorted(extra)}")
    return manifest, _sha256(raw)


def split_gold(manifest: dict) -> tuple[list[dict], dict]:
    """Split the frozen manifest into prompt rows and evaluation-side gold.

    The prompt rows never carry gold fields; the gold stays evaluation-side
    only, per the manifest's own evaluation protocol.
    """
    prompts, gold = [], {}
    for question in manifest["questions"]:
        prompts.append({"question_id": question["question_id"],
                        "history_id": question["history_id"],
                        "prompt": question["question"],
                        "case": question.get("case"),
                        "absence": bool(question.get("absence"))})
        gold[question["question_id"]] = {
            "expected_answer": question.get("expected_answer"),
            "expected_source_keys": list(question.get("expected_source_keys") or []),
            "must_not_sources": list(question.get("must_not_sources") or []),
            "notes": question.get("notes"),
            "absence": bool(question.get("absence")),
        }
    return prompts, gold


def gold_leak_scan(records: list[dict], gold: dict) -> None:
    """Gold strings must never appear in any recorded prompt (defence in depth)."""
    needles = [str(item.get("expected_answer") or "").strip() for item in gold.values()]
    needles = [needle for needle in needles if len(needle) >= 8]
    for record in records:
        prompt_blob = json.dumps(record.get("request", {}), ensure_ascii=False)
        for needle in needles:
            if needle in prompt_blob:
                raise QualityCheckError(
                    f"gold leak: the expected answer for {record.get('question_id')!r} "
                    "appears in the recorded prompt; aborting before scoring")


def _json_default(value):
    """PostgreSQL numerics (Decimal) and similar DB types stay evidence-safe."""
    from datetime import date, datetime
    from decimal import Decimal
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    return str(value)


def record_phase(prompts: list[dict], ask: Callable[[dict], dict],
                 output_dir: Path) -> list[dict]:
    """Ask every question once and persist immutable records before any scoring."""
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for row in prompts:
        started = time.monotonic()
        try:
            answer_payload = ask(row)
            error = None
        except Exception as exc:  # failures are evidence, not a reason to hide the row
            answer_payload = None
            error = {"type": type(exc).__name__, "message": str(exc)}
        record = {
            "question_id": row["question_id"],
            "history_id": row["history_id"],
            "case": row.get("case"),
            "absence": row.get("absence", False),
            "prompt": row["prompt"],
            "request": answer_payload.get("request") if answer_payload else None,
            "answer": (answer_payload or {}).get("answer"),
            "probe": (answer_payload or {}).get("probe"),
            "trace": (answer_payload or {}).get("trace"),
            "answer_routing": (answer_payload or {}).get("answer_routing") or {},
            "usage": (answer_payload or {}).get("usage") or {},
            "model_call_count": (answer_payload or {}).get("model_call_count"),
            "error": error,
            "duration_seconds": round(time.monotonic() - started, 3),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        target = records_dir / f"{record['question_id']}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=_json_default),
                       encoding="utf-8")
        tmp.replace(target)  # atomic: the record exists before scoring reads it
        records.append(record)
    return records


JUDGE_SYSTEM = (
    "Ты оцениваешь автоматический ответ учебного помощника. Тебе дают вопрос, ответ "
    "помощника, эталонный ответ (оценочная истина) и материалы источников из архива. "
    "Реши строго по источникам и верни только JSON ровно вида:\n"
    '{"relevant": true|false, "supported": true|false, "complete": true|false, '
    '"honest_missing": true|false, "false_claims": ["..."], "reason": "..."}\n'
    "Значения полей: relevant – ответ вообще относится к вопросу; supported – фактические "
    "утверждения ответа подтверждаются источниками (пересказ источников своими словами и "
    "дословные цитаты – это подтверждение); complete – ответ покрывает существенные пункты "
    "эталонного ответа; honest_missing – true только если ответ явно сообщает, что нужных "
    "данных в архиве нет, и при этом ничего не выдумывает; false_claims – список конкретных "
    "утверждений ответа, которых нет в источниках или которые им противоречат (пустой список, "
    "если таких нет); reason – краткое обоснование на русском.\n"
    "Строгие правила соответствия флагов: если ответ сообщает об отсутствии нужных данных, "
    "ставь supported=false и honest_missing=true. Никогда не ставь supported=true и "
    "honest_missing=true одновременно – это недопустимая комбинация; оценивая сообщение об "
    "отсутствии, всегда supported=false. Если эталонный ответ описывает факты, а ответ "
    "помощника утверждает, что этих данных нет, – это supported=false, honest_missing=false "
    "(ответ противоречит эталону и источникам).\n"
    "Перед оценкой перечитай источники: утверждать, что в источниках нет нужных сведений, "
    "нельзя, если такие сведения в них приведены ниже.\n"
    "Дополнительные правила: общая учебная фраза-уточнение (например «Хочешь другой пример "
    "по этой теме или разберём непонятный шаг?») – это НЕ ответ на вопрос и НЕ сообщение об "
    "отсутствии данных: relevant=false, honest_missing=false. Если в false_claims есть хотя "
    "бы одно утверждение, supported=true недопустим. Не используй внешние знания. Верни "
    "только JSON без пояснений вокруг него."
)


def judge_answer(judge: Callable[[str, str, str], Any], prompt: str, answer: str,
                 sources_text: str) -> tuple[dict | None, str]:
    """A judged decision and the raw reply, or ``None`` when unusable.

    A non-JSON judge reply is a judge failure, not answer evidence: the row
    is scored as judge_failure/unscored instead of inventing a verdict. The
    raw reply travels with the row either way.
    """
    raw_text = ""
    decision = judge(JUDGE_SYSTEM, prompt, answer + "\n\n" + sources_text)
    if isinstance(decision, dict):
        return decision, json.dumps(decision, ensure_ascii=False)
    if isinstance(decision, str):
        raw_text = decision
    else:
        raw_text = json.dumps(decision, ensure_ascii=False, default=str)
    import re
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed, raw_text
        except json.JSONDecodeError:
            pass
    # A reply that quotes a schema example before the decision defeats the
    # greedy span (first "{" to last "}" is not valid JSON). Decode every
    # balanced candidate and prefer the last one that carries the decision
    # fields (a quoted example usually precedes the real verdict); an
    # unusable reply still stays a judge failure below.
    decoder = json.JSONDecoder()
    candidates = []
    for start in [index for index, char in enumerate(raw_text) if char == "{"]:
        try:
            parsed, _end = decoder.raw_decode(raw_text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    for parsed in reversed(candidates):
        if all(field in parsed for field in JUDGE_BOOL_FIELDS):
            return parsed, raw_text
    if candidates:
        return candidates[-1], raw_text
    return None, raw_text


def validate_decision(decision: Any) -> tuple[dict | None, str | None]:
    """Strict schema and consistency validation of one judge decision.

    Returns (normalized_decision, None) or (None, failure_reason). Booleans
    must be JSON booleans: the string "false" is not boolean false. A
    nonempty false-claims list cannot accompany a supported verdict, and
    honest_missing cannot combine with supported or with irrelevance.
    """
    if not isinstance(decision, dict):
        return None, "judge reply is not a JSON object"
    for field in JUDGE_BOOL_FIELDS:
        if type(decision.get(field)) is not bool:
            return None, f"field {field!r} must be a JSON boolean, got {decision.get(field)!r}"
    claims = decision.get("false_claims")
    if not isinstance(claims, list) or any(
            not isinstance(claim, str) or not claim.strip() for claim in claims):
        return None, f"false_claims must be a list of nonempty strings, got {claims!r}"
    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None, f"reason must be a nonempty string, got {reason!r}"
    normalized = {field: decision[field] for field in JUDGE_BOOL_FIELDS}
    normalized["false_claims"] = [claim.strip() for claim in claims]
    normalized["reason"] = reason
    if normalized["supported"] and normalized["false_claims"]:
        return None, "inconsistent judgement: supported=true with a nonempty false_claims list"
    if normalized["supported"] and normalized["honest_missing"]:
        return None, "inconsistent judgement: supported=true and honest_missing=true"
    if normalized["supported"] and not normalized["relevant"]:
        return None, "inconsistent judgement: supported=true but relevant=false"
    if normalized["honest_missing"] and not normalized["relevant"]:
        return None, "inconsistent judgement: honest_missing=true but relevant=false"
    return normalized, None


def derive_verdict(decision: dict, *, generation_called: bool | None = True,
                   absence_declared: bool | None = None) -> str:
    """Deterministic verdict from a validated decision.

    A generic clarification never becomes a supported answer; an explicit
    honest absence statement is its own verdict, separate from support.
    A fallback row without an answer-model call is never an answer success.

    ``honest_missing_evidence`` is reserved for questions whose frozen gold
    declares canonical absence (``absence: true``): when the archive
    contains the fact, an answer reporting absence is a wrong answer, not a
    positive honest-missing result (found in review: A6 was scored as an
    honest absence although the canonical history contains the aquarium
    fact).
    """
    if generation_called is False:
        return "fallback_no_generation"
    if decision["relevant"] and decision["supported"] and decision["complete"] \
            and not decision["honest_missing"]:
        return "supported"
    if decision["relevant"] and decision["supported"] and not decision["complete"] \
            and not decision["honest_missing"]:
        return "partial_answer"
    if decision["relevant"] and decision["honest_missing"] and not decision["supported"] \
            and absence_declared is True:
        return "honest_missing_evidence"
    return "unsupported"


def _truth_entry(sources_map: dict, source_id: str):
    """Accept both the new dict-shaped truth dump and legacy plain-text maps."""
    entry = sources_map.get(source_id)
    if entry is None:
        return None
    if isinstance(entry, str):
        return {"text": entry, "author_role": None, "created_at": None,
                "superseded_by": None, "corrects_source_id": None}
    if isinstance(entry, dict):
        return {"text": str(entry.get("text") or ""),
                "author_role": entry.get("author_role"),
                "created_at": entry.get("created_at"),
                "superseded_by": entry.get("superseded_by"),
                "corrects_source_id": entry.get("corrects_source_id")}
    return None


def _expected_sources_text(expected_ids: list[str], sources_map: dict,
                           absence: bool) -> tuple[str, list[str]]:
    """Evaluation-side canonical truth for the judge.

    Truth comes from the complete store dump, so a source the answer path
    never retrieved still shows its real text. A source missing even from
    the complete dump is an evaluation-side failure and is labelled as such
    – never presented as absent evidence.
    """
    if absence and not expected_ids:
        return ("Эталон (оценочная истина): в полном архиве этих данных нет – "
                "это absence-вопрос, корректный ответ обязан явно сообщить об "
                "отсутствии данных и ничего не выдумать."), []
    lines, missing = [], []
    for source_id in sorted(expected_ids):
        entry = _truth_entry(sources_map, source_id)
        if entry is None:
            missing.append(source_id)
            lines.append(f"[{source_id}] (ОШИБКА ЭТАЛОНА: текст источника отсутствует "
                         "в полной выгрузке архива – это нельзя считать отсутствием "
                         "доказательства)")
            continue
        prefix_parts = []
        if entry["author_role"]:
            prefix_parts.append(f"роль: {entry['author_role']}")
        if entry["created_at"]:
            prefix_parts.append(f"время: {entry['created_at']}")
        if entry["corrects_source_id"]:
            prefix_parts.append(f"исправление источника {entry['corrects_source_id']}")
        if entry["superseded_by"]:
            prefix_parts.append(f"заменён источником {entry['superseded_by']} (устаревшее "
                                "доказательство, актуальна правка)")
        prefix = "; ".join(prefix_parts)
        lines.append(f"[{source_id}]{' (' + prefix + ')' if prefix else ''} {entry['text']}")
    return "\n\n".join(lines), missing


def _phase_ids(event: dict, names: tuple[str, ...]) -> list | None:
    """Read one evidence id list under any accepted field name.

    The frozen contract names the omission list `omitted_sources`; the
    reviewed B implementation emits `omitted_source_ids`. The harness
    accepts both rather than guessing one and losing the field.
    """
    for name in names:
        value = event.get(name)
        if isinstance(value, list):
            return value
    return None


def _trace_hit(record: dict, phase: str, field: str, wanted: set[str]) -> bool | None:
    """One trace-based hit metric; None when instrumentation is unavailable.

    The recorded trace is the harness summary (see
    r2_live_quality_run.summarize_trace): `retrieval` and `answer_prompt`
    phase views sit directly under the trace mapping. A missing field means
    unavailable (None), never an empty set and never a successful retrieval.
    """
    trace = record.get("trace")
    if not isinstance(trace, dict):
        return None
    event = trace.get(phase)
    if not isinstance(event, dict):
        return None
    ids = _phase_ids(event, (field,))
    if ids is None:
        return None
    return bool(wanted & {str(source_id) for source_id in ids}) if wanted else None


def _product_retrieval_hit(record: dict, wanted: set[str]) -> bool | None:
    """Hit on the retrieval phase's retrieved pool, or None when unusable.

    An empty retrieved pool beside a nonempty delivered set is contradictory
    instrumentation, not a genuine zero: the personal-recall path recorded
    retrieved=[] while delivering ~40 units, so the pool is marked
    unavailable instead of scoring a false miss.
    """
    hit = _trace_hit(record, "retrieval", "retrieved_source_ids", wanted)
    if hit is False and wanted:
        trace = record.get("trace") or {}
        retrieval = trace.get("retrieval") or {}
        prompt = trace.get("answer_prompt") or {}
        retrieved = retrieval.get("retrieved_source_ids")
        delivered = prompt.get("delivered_source_ids") or []
        if isinstance(retrieved, list) and not retrieved and delivered:
            return None  # contradictory instrumentation: unavailable, not a miss
    return hit


PROMPT_IN_EVIDENCE_MIN_CHARS = 24


def _quote_text(quote) -> str:
    """The quote unit's carried text (B's label form or a plain text form)."""
    if isinstance(quote, dict):
        return str(quote.get("quote") or quote.get("text") or "")
    return str(quote or "")


def _delivered_quotes(record: dict) -> list | None:
    trace = record.get("trace") or {}
    prompt = trace.get("answer_prompt") or {}
    quotes = prompt.get("hydrated_quotes") if isinstance(prompt, dict) else None
    return quotes if isinstance(quotes, list) else None


def _current_request_in_evidence(record: dict) -> bool | None:
    """True when the just-asked question itself surfaces as delivered evidence.

    The current request must stay separated from historical evidence. Two
    instruments are checked: a quote unit literally carrying the prompt
    text, and the final-prompt fingerprint of a pending own delivery inside
    the retrieval block (only the just-reserved current message is pending;
    historical sources are completed). None when neither instrument is
    available in the trace.
    """
    trace = record.get("trace") or {}
    prompt_view = trace.get("answer_prompt") or {}
    final_prompt = prompt_view.get("final_prompt") if isinstance(prompt_view, dict) else None
    import re
    pending_in_retrieval = False
    if isinstance(final_prompt, list):
        contents = [str(message.get("content") or "") for message in final_prompt
                    if isinstance(message, dict)]
    elif isinstance(final_prompt, str):
        contents = [final_prompt]
    else:
        contents = []
    for content in contents:
        # The retrieval block embeds delivery statuses as literal JSON text;
        # only the just-reserved current message is pending.
        if re.search(r"delivery_status[\"']?\s*:\s*[\"']?pending", content):
            pending_in_retrieval = True
            break
    quotes = _delivered_quotes(record)
    direct = None
    if quotes is not None:
        prompt = str(record.get("prompt") or "").strip()
        if len(prompt) >= PROMPT_IN_EVIDENCE_MIN_CHARS:
            direct = any(prompt in _quote_text(quote) for quote in quotes)
    if direct is True or pending_in_retrieval:
        return True
    if direct is None and not contents:
        return None
    return False


def _fragment_in_blob(fragment: str, blob: str, collapsed_blob: str | None = None) -> bool:
    """Substring match with JSON-escape aware fallbacks.

    The recorded final prompt is compared as its json.dumps rendering, so
    every quote, backslash and newline inside evidence text is escaped
    there; a raw substring check falsely reports such a quote as absent
    from the prompt it was delivered in (found in the adversarial
    self-review of the evaluator). Evidence text can also sit inside an
    embedded JSON payload (the retrieval block renders nested JSON), which
    escapes it twice, so up to two escape levels are tried before the
    whitespace-collapsed comparison.
    """
    if not fragment:
        return False
    candidate = fragment
    for _level in range(3):
        if candidate and candidate in blob:
            return True
        escaped = json.dumps(candidate, ensure_ascii=False)
        candidate = escaped[1:-1] if len(escaped) >= 2 else ""
    if collapsed_blob is None:
        collapsed_blob = " ".join(blob.split())
    compact = " ".join(fragment.split())
    return bool(compact) and compact in collapsed_blob


def _final_prompt_consistent(record: dict) -> dict | None:
    """Delivered IDs must correspond to evidence units inside the final prompt.

    For every delivered source id the trace must carry a hydrated quote, and
    that quote must be recognisable in the recorded final prompt (by its
    citation label or its leading text). None when either side is not
    instrumented – a missing prompt or quote list is unavailable, never an
    automatic pass.
    """
    trace = record.get("trace") or {}
    prompt_view = trace.get("answer_prompt") or {}
    if not isinstance(prompt_view, dict):
        return None
    quotes = prompt_view.get("hydrated_quotes")
    delivered = prompt_view.get("delivered_source_ids")
    final_prompt = prompt_view.get("final_prompt")
    if not isinstance(delivered, list) or not isinstance(quotes, list) \
            or final_prompt in (None, ""):
        return None
    blob = json.dumps(final_prompt, ensure_ascii=False, default=str)
    collapsed_blob = " ".join(blob.split())
    quote_by_id: dict[str, dict] = {}
    for quote in quotes:
        if isinstance(quote, dict) and quote.get("source_id") is not None:
            quote_by_id.setdefault(str(quote["source_id"]), quote)
    problems = []
    for source_id in sorted({str(item) for item in delivered}):
        quote = quote_by_id.get(source_id)
        if quote is None:
            problems.append({"source_id": source_id,
                             "issue": "no hydrated quote for a delivered id"})
            continue
        label = quote.get("quote")
        label_ok = isinstance(label, str) and bool(label.strip()) \
            and _fragment_in_blob(label, blob, collapsed_blob)
        # The delivered text can live in "text" (label form carries "quote")
        # or be the "quote" field itself (B's recorded shape).
        text_ok = any(isinstance(text, str) and text
                      and _fragment_in_blob(text[:40], blob, collapsed_blob)
                      for text in (quote.get("text"), quote.get("quote")))
        if not (label_ok or text_ok):
            problems.append({"source_id": source_id,
                             "issue": "quote neither by label nor by text in final prompt"})
    return {"consistent": not problems, "problems": problems}


def score_phase(records: list[dict], gold: dict, key_to_id: dict[str, str],
                judge: Callable[[str, str, str], Any] | None,
                sources_by_id: dict | None) -> list[dict]:
    """Score recorded rows only; gold enters this phase, never generation."""
    source_map = sources_by_id or {}
    scored = []
    for record in records:
        item = gold.get(record["question_id"], {})
        expected_keys = list(item.get("expected_source_keys") or [])
        expected_ids, unmapped_keys = [], []
        for key in expected_keys:
            source_id = key_to_id.get(key)
            if source_id:
                expected_ids.append(source_id)
            else:
                unmapped_keys.append(key)
        must_not_keys = list(item.get("must_not_sources") or [])
        must_not_ids, unmapped_must_not = [], []
        for key in must_not_keys:
            source_id = key_to_id.get(key)
            if source_id:
                must_not_ids.append(str(source_id))
            else:
                unmapped_must_not.append(key)
        wanted = set(expected_ids)
        trace = record.get("trace") if isinstance(record.get("trace"), dict) else {}
        trace_available = bool(trace.get("available"))
        generation_called = trace.get("generation_called")
        technical_error = trace.get("technical_error")
        probe = record.get("probe") if isinstance(record.get("probe"), dict) else None
        row = {
            "question_id": record["question_id"],
            "history_id": record["history_id"],
            "absence": bool(item.get("absence")),
            "retrieval_expected_keys": sorted(expected_keys),
            "unmapped_keys": sorted(unmapped_keys),
            "retrieval_expected_ids": sorted(wanted),
            "must_not_source_keys": sorted(must_not_keys),
            "unmapped_must_not_keys": sorted(unmapped_must_not),
            "probe_hit": bool(wanted & {str(source_id) for source_id in (probe or {}).get("source_ids") or []})
                         if wanted and probe is not None else None,
            "product_retrieval_hit": _product_retrieval_hit(record, wanted),
            "selected_hit": _trace_hit(record, "retrieval", "selected_source_ids", wanted),
            "delivered_hit": _trace_hit(record, "answer_prompt", "delivered_source_ids", wanted),
            "generation_called": generation_called,
            "trace_technical_error": technical_error,
            "trace_available": trace_available,
            "current_request_in_evidence": _current_request_in_evidence(record),
            "final_prompt_check": _final_prompt_consistent(record),
            "answer": record.get("answer"),
            "error": record.get("error"),
        }
        wanted_must_not = set(must_not_ids)
        if wanted_must_not:
            prompt_view = trace.get("answer_prompt") or {}
            delivered_ids = {str(source_id) for source_id in
                             (prompt_view.get("delivered_source_ids") or [])} \
                if isinstance(prompt_view, dict) else None
            # Forbidden sources delivered into the prompt are a violation;
            # without delivery instrumentation it stays unavailable, never a
            # clean pass.
            row["must_not_violations"] = sorted(wanted_must_not & delivered_ids) \
                if delivered_ids is not None else None
            row["must_not_source_ids"] = sorted(wanted_must_not)
        else:
            row["must_not_violations"] = None
            row["must_not_source_ids"] = []
        if probe is not None:
            row["probe_source_ids"] = list((probe or {}).get("source_ids") or [])
        if record.get("error"):
            row.update({"verdict": "error", "relevant": None, "supported": None,
                        "complete": None, "honest_missing": None, "false_claims": []})
        elif technical_error and not record.get("answer"):
            row.update({"verdict": "error", "relevant": None, "supported": None,
                        "complete": None, "honest_missing": None, "false_claims": [],
                        "reason": f"technical_error in request trace: {technical_error}"})
        elif judge is None:
            row.update({"verdict": "unscored", "reason": "judge unavailable"})
        elif generation_called is False:
            # A fallback/clarification without an answer-model call is never
            # an answer success: the trace makes that explicit, so no judge
            # call can upgrade it.
            row.update({"verdict": "fallback_no_generation", "relevant": None,
                        "supported": None, "complete": None, "honest_missing": None,
                        "false_claims": []})
        else:
            truth_text, missing_truth = _expected_sources_text(
                expected_ids, source_map, bool(item.get("absence")))
            row["truth_source_missing"] = missing_truth
            reference = item.get("expected_answer")
            body = ("Эталонный ответ (оценочная истина, не для копирования в ответ "
                    "помощника):\n" + str(reference or "(эталон не задан)")
                    + "\n\nИсточники (полная эталонная выгрузка архива):\n" + truth_text)
            decision, raw_reply = judge_answer(judge, record["prompt"],
                                               str(record.get("answer") or ""), body)
            normalized, failure = validate_decision(decision)
            if normalized is None:
                # Invalid or inconsistent judge output stays a visible failure
                # with the raw reply preserved; it never becomes a pass.
                row.update({"verdict": "judge_failure", "judge_failure_reason": failure,
                            "judge_raw": raw_reply, "relevant": None, "supported": None,
                            "complete": None, "honest_missing": None, "false_claims": []})
            else:
                verdict = derive_verdict(normalized,
                                         generation_called=generation_called,
                                         absence_declared=bool(item.get("absence")))
                row.update({"verdict": verdict, **{field: normalized[field]
                                                   for field in JUDGE_BOOL_FIELDS},
                            "absence_declared": bool(item.get("absence")),
                            "false_claims": normalized["false_claims"],
                            "judge_reason": normalized["reason"],
                            "judge_raw": raw_reply})
        scored.append(row)
    return scored


def aggregate(scored: list[dict]) -> dict:
    positive = {"supported", "honest_missing_evidence", "partial_answer"}
    rows = [row for row in scored if row["verdict"] != "error"]
    return {
        "questions_total": len(scored),
        "errors": sum(1 for row in scored if row["verdict"] == "error"),
        "unscored": sum(1 for row in scored if row["verdict"] == "unscored"),
        "judge_failures": sum(1 for row in scored if row["verdict"] == "judge_failure"),
        "fallback_no_generation": sum(1 for row in scored if row["verdict"] == "fallback_no_generation"),
        "supported": sum(1 for row in rows if row["verdict"] == "supported"),
        "partial_answer": sum(1 for row in rows if row["verdict"] == "partial_answer"),
        "honest_missing_evidence": sum(1 for row in rows if row["verdict"] == "honest_missing_evidence"),
        "unsupported": sum(1 for row in rows if row["verdict"] == "unsupported"),
        "answer_relevant": sum(1 for row in rows if row.get("relevant") is True),
        "answer_complete": sum(1 for row in rows if row.get("complete") is True),
        "answer_supported": sum(1 for row in rows if row.get("supported") is True),
        "false_claim_rows": sum(1 for row in rows if row.get("false_claims")),
        "probe_hit": sum(1 for row in rows if row.get("probe_hit")),
        "product_retrieval_hit": sum(1 for row in rows if row.get("product_retrieval_hit")),
        "selected_hit": sum(1 for row in rows if row.get("selected_hit")),
        "delivered_hit": sum(1 for row in rows if row.get("delivered_hit")),
        # Denominators: the hit counters above are hits / denominator, where a
        # denominator row is one whose metric is available (frozen gold keys
        # mapped and the trace instrument present). Absence controls and
        # unavailable instrumentation stay out of a stage denominator instead
        # of silently deflating the rate.
        "questions_with_expected_ids": sum(1 for row in rows if row.get("retrieval_expected_ids")),
        "absence_control_rows": sum(1 for row in rows if row.get("absence")),
        "probe_hit_denominator": sum(1 for row in rows if row.get("probe_hit") is not None),
        "product_retrieval_hit_denominator": sum(
            1 for row in rows if row.get("product_retrieval_hit") is not None),
        "selected_hit_denominator": sum(1 for row in rows if row.get("selected_hit") is not None),
        "delivered_hit_denominator": sum(1 for row in rows if row.get("delivered_hit") is not None),
        "must_not_rows": sum(1 for row in rows if row.get("must_not_source_ids")),
        "must_not_violation_rows": sum(1 for row in rows if row.get("must_not_violations")),
        "trace_available_rows": sum(1 for row in rows if row.get("trace_available")),
        "current_request_in_evidence_rows": sum(1 for row in rows if row.get("current_request_in_evidence")),
        "final_prompt_check_rows": sum(1 for row in rows if isinstance(row.get("final_prompt_check"), dict)),
        "final_prompt_inconsistent_rows": sum(1 for row in rows if isinstance(row.get("final_prompt_check"), dict)
                                              and not row["final_prompt_check"]["consistent"]),
        "positive_score_total": sum(1 for row in rows if row["verdict"] in positive),
    }


def _result_envelope(result: dict, manifest_sha256: str, final_histories: list[str],
                     score_only: bool = False) -> dict:
    result.update({
        "assessment": "automated (LLM judge + deterministic checks); not an independent human study",
        "harness_protocol": HARNESS_PROTOCOL,
        "manifest_protocol": PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "final_check_histories": final_histories,
        "score_only": score_only,
    })
    return result


def score_from_records(output_dir: Path, manifest_path: Path,
                       judge: Callable[[str, str, str], Any] | None,
                       key_to_id: dict[str, str] | None = None,
                       sources_by_id: dict[str, str] | None = None,
                       replay_output_dir: Path | None = None) -> dict:
    """Score already-recorded rows (score-only mode); no question is re-asked.

    The recorded rows are the immutable evidence; a failed judging phase can
    be replayed against them without touching the models again. When the
    records live in an immutable run directory, pass ``replay_output_dir``
    so the replayed results.json lands there instead of overwriting the
    directory's own result.
    """
    manifest, manifest_sha256 = load_manifest(manifest_path)
    _prompts, gold = split_gold(manifest)
    records_dir = output_dir / "records"
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(records_dir.glob("*.json"))]
    gold_leak_scan(records, gold)
    scored = score_phase(records, gold, key_to_id or {}, judge, sources_by_id)
    final_histories = [h["history_id"] for h in manifest["histories"]
                       if h.get("role") == "final_check"]
    result = {
        "records": records,
        "scored": scored,
        "aggregate": aggregate(scored),
    }
    result = _result_envelope(result, manifest_sha256, final_histories, score_only=True)
    destination = replay_output_dir or output_dir
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")
    return result


def run(manifest_path: Path, ask: Callable[[dict], dict],
        judge: Callable[[str, str, str], Any] | None, output_dir: Path,
        key_to_id: dict[str, str] | None = None,
        sources_by_id: dict[str, str] | None = None,
        run_metadata: dict | None = None,
        history_ids: tuple | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, manifest_sha256 = load_manifest(manifest_path)
    prompts, gold = split_gold(manifest)
    if history_ids is not None:
        prompts = [row for row in prompts if row["history_id"] in history_ids]
        if not prompts:
            raise QualityCheckError("history filter left no questions to ask")
    key_to_id = key_to_id or {}
    final_histories = [h["history_id"] for h in manifest["histories"]
                       if h.get("role") == "final_check"]
    for history_id in final_histories:
        marker = output_dir / f"final_check_attempt_1_{history_id}.json"
        if marker.exists():
            raise QualityCheckError(
                f"{marker} exists: the first final-check result is preserved. "
                "Write later runs to a fresh output directory and label them as regressions.")
    records = record_phase(prompts, ask, output_dir)
    gold_leak_scan(records, gold)
    (output_dir / "records_manifest.json").write_text(
        json.dumps({"manifest_sha256": manifest_sha256,
                    "recorded": [record["question_id"] for record in records],
                    "row_sha256": [_sha256((output_dir / "records" / f"{record['question_id']}.json").read_bytes())
                                   for record in records]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    scored = score_phase(records, gold, key_to_id, judge, sources_by_id)
    result = {
        "records": records,
        "scored": scored,
        "aggregate": aggregate(scored),
        "run_metadata": run_metadata or {},
    }
    result = _result_envelope(result, manifest_sha256, final_histories)
    for history_id in final_histories:
        marker = output_dir / f"final_check_attempt_1_{history_id}.json"
        marker.write_text(json.dumps(
            {"aggregate": result["aggregate"],
             "note": "first final-check attempt; preserved"},
            ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "tests" / "r2_memory_quality" / "manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-judge", action="store_true",
                        help="record and score deterministically only; rows stay unscored")
    args = parser.parse_args()
    try:
        manifest, _digest = load_manifest(args.manifest)
        raise QualityCheckError(
            "live mode runs through scripts/r2_live_quality_run.py, which prepares the "
            "disposable database, loads the frozen histories and asks through the real API")
    except (QualityCheckError, OSError, json.JSONDecodeError) as exc:
        print(f"quality check failed honestly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
