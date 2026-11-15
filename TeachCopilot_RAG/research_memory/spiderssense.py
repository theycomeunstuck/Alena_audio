"""Explicitly priced, budgeted Spiderssense chat client. No automatic retries."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .clients import Completion, CompletionValidationError, MAX_RESPONSE_BYTES


BASE_URL = "https://spiderssense.com/v1"
KEY_ENV = "SPIDERSSENSE_API_KEY"

# Relative ledger_path/key_file values are anchored to the project root (the
# parent of this package) at construction time, so one configuration always
# addresses one durable ledger no matter which working directory the client
# was constructed from. Resolving them against the process CWD instead would
# give the same configuration a fresh budget from another directory.
_ANCHOR = Path(__file__).resolve().parent.parent


def _anchored(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else _ANCHOR / value


class BudgetError(RuntimeError):
    """A request cannot safely proceed under the persisted client budget."""


class ProviderRequestError(RuntimeError):
    """Transport failure with safe diagnostic fields and no provider body."""

    def __init__(self, message: str, *, code: str, http_status: int | None = None,
                 ledger_call_id: int | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.ledger_call_id = ledger_call_id


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def _decimal(value: Any, label: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{label} must be a finite positive number") from None
    if not result.is_finite() or (result < 0 if allow_zero else result <= 0):
        raise ValueError(f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _redact_credential_text(value: str, key: str) -> str:
    """Inspect bounded JSON escape layers without retaining an encoded key."""
    replacement = next(value for value in ("[REDACTED_CREDENTIAL]", "###", "***", "") if key not in value)
    candidate = value
    modified = False
    for _ in range(16):
        if key in candidate:
            candidate = candidate.replace(key, replacement)
            modified = True

        def decode(match):
            try:
                return json.loads('"' + match.group(0) + '"')
            except (ValueError, TypeError):
                return match.group(0)

        decoded = re.sub(r'\\(?:u[0-9a-fA-F]{4}|[\\/"bfnrt])', decode, candidate)
        if decoded == candidate:
            return candidate if modified else value
        candidate = decoded
    marker = "[REDACTED_UNINSPECTABLE_CREDENTIAL_ENCODING]"
    return marker if key not in marker else replacement


def _journal_body(raw: bytes, key: str, *, http_status: int | None = None) -> dict:
    """Keep provider bytes and JSON before interpretation, never a reflected key."""
    replacement = "[REDACTED_CREDENTIAL]"
    # Inspect complete JSON string tokens even if the surrounding body is
    # malformed, too deeply nested or contains non-finite numbers. This removes
    # reflected credentials encoded with JSON escapes without rewriting the
    # other provider bytes. surrogateescape preserves invalid UTF-8 bytes.
    text = raw.decode("utf-8", errors="surrogateescape")

    def redact_token(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            value = json.loads(token)
        except (ValueError, TypeError):
            return token
        safe_value = _redact_credential_text(value, key)
        if safe_value != value:
            return json.dumps(safe_value, ensure_ascii=True)
        return token

    safe_text = re.sub(r'"(?:[^"\\]|\\.)*"', redact_token, text)
    safe_text = _redact_credential_text(safe_text, key)
    # Retain invalid UTF-8 bytes, while escaping any decoded lone surrogate.
    safe_text = re.sub(r"[\ud800-\udc7f\udd00-\udfff]",
                       lambda match: "\\u%04x" % ord(match.group(0)), safe_text)
    safe_raw = safe_text.encode("utf-8", errors="surrogateescape")
    result = {"body_base64": base64.b64encode(safe_raw).decode("ascii"),
              "body_text": safe_raw.decode("utf-8", errors="replace")}
    try:
        parsed = json.loads(safe_raw.decode("utf-8"))
        # Some JSON decoders admit NaN or numeric overflow. Keep such bytes as
        # raw evidence without allowing non-finite values into journal JSON.
        json.dumps(parsed, ensure_ascii=True, allow_nan=False)
        result["parsed"] = parsed
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError):
        pass
    if type(http_status) is int:
        result["http_status"] = http_status
    return result


class SpiderssenseClient:
    """One model and one immutable price configuration per durable budget ledger.

    Prices are USD per million tokens, supplied from a verified provider catalog.
    They are not discovered or inferred from another provider's prices. A budget
    reservation is retained when billing or network outcome remains unknown.
    """

    def __init__(self, *, model: str, input_usd_per_million: Any,
                 output_usd_per_million: Any, pricing_verified: bool,
                 pricing_source: str,
                 ledger_path: str | Path = ".local/research_memory/spiderssense_budget.sqlite3",
                 max_cost_usd: Any = "4", max_calls: int = 32,
                 max_output_tokens: int = 1536, token_field: str = "max_completion_tokens",
                 key_file: str | Path | None = None, timeout: float = 180,
                 reasoning_effort: str | None = None,
                 use_json_schema: bool = False, use_json_object: bool = False,
                 input_overhead_tokens: int = 4096,
                 input_cost_multiplier: Any = "1.25", reservation_margin: Any = "1.10"):
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise ValueError("model must be explicitly specified")
        if pricing_verified is not True:
            raise ValueError("verified model-specific provider prices are required")
        try:
            source = urlsplit(pricing_source)
        except (TypeError, ValueError):
            raise ValueError("pricing_source must be an official public provider URL") from None
        if (source.scheme != "https" or source.netloc != "spiderssense.com"
                or source.query or source.fragment or source.username or source.password):
            raise ValueError("pricing_source must be an official public provider URL without credentials")
        self.model = model
        self.input_price = _decimal(input_usd_per_million, "input_usd_per_million")
        self.output_price = _decimal(output_usd_per_million, "output_usd_per_million")
        self.max_cost = _decimal(max_cost_usd, "max_cost_usd")
        self.max_calls = _positive_int(max_calls, "max_calls")
        self.max_output_tokens = _positive_int(max_output_tokens, "max_output_tokens")
        self.input_overhead_tokens = _positive_int(input_overhead_tokens, "input_overhead_tokens")
        self.input_cost_multiplier = _decimal(input_cost_multiplier, "input_cost_multiplier")
        self.reservation_margin = _decimal(reservation_margin, "reservation_margin")
        if self.input_cost_multiplier < Decimal("1.25") or self.reservation_margin < 1:
            raise ValueError("input_cost_multiplier must be at least 1.25 and reservation_margin at least 1")
        if token_field not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError("token_field must be max_completion_tokens or max_tokens")
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not isinstance(use_json_schema, bool):
            raise ValueError("use_json_schema must be a boolean")
        if not isinstance(use_json_object, bool):
            raise ValueError("use_json_object must be a boolean")
        if use_json_schema and use_json_object:
            raise ValueError("use_json_schema and use_json_object are mutually exclusive")
        if reasoning_effort is not None and (not isinstance(reasoning_effort, str)
                or not reasoning_effort.isascii() or not reasoning_effort.isalpha()
                or len(reasoning_effort) > 32):
            raise ValueError("reasoning_effort must be an explicit short ASCII effort name")
        self.timeout = timeout
        self.token_field = token_field
        self.use_json_schema = use_json_schema
        self.use_json_object = use_json_object
        self.reasoning_effort = reasoning_effort
        self.ledger_path = _anchored(ledger_path)
        self._key_file = (_anchored(key_file) if key_file is not None
                          else _anchored(".local/research_memory/spiderssense.key"))
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._config = {
            "version": 1, "backend": "spiderssense", "base_url": BASE_URL,
            "model": model, "input_usd_per_million": str(self.input_price),
            "output_usd_per_million": str(self.output_price), "pricing_source": pricing_source,
            "max_cost_usd": str(self.max_cost), "max_calls": self.max_calls,
            "max_output_tokens": self.max_output_tokens, "token_field": token_field,
            "reasoning_effort": reasoning_effort,
            "use_json_schema": use_json_schema, "input_overhead_tokens": self.input_overhead_tokens,
            "input_cost_multiplier": str(self.input_cost_multiplier),
            "reservation_margin": str(self.reservation_margin),
        }
        if use_json_object:
            self._config["use_json_object"] = True
        self._config_json = json.dumps(self._config, sort_keys=True, separators=(",", ":"))
        self._config_hash = hashlib.sha256(self._config_json.encode()).hexdigest()
        with self._connect() as connection:
            self._begin(connection)
            connection.execute("CREATE TABLE IF NOT EXISTS budget_config (id INTEGER PRIMARY KEY CHECK(id=1), fingerprint TEXT NOT NULL, configuration TEXT NOT NULL, blocked INTEGER NOT NULL DEFAULT 0)")
            connection.execute("CREATE TABLE IF NOT EXISTS budget_calls (id INTEGER PRIMARY KEY, reserved_usd TEXT NOT NULL, charged_usd TEXT NOT NULL, status TEXT NOT NULL, input_bound INTEGER NOT NULL, input_tokens INTEGER, output_tokens INTEGER, provider_cost_usd TEXT)")
            stored = connection.execute("SELECT fingerprint FROM budget_config WHERE id=1").fetchone()
            if stored is None:
                connection.execute("INSERT INTO budget_config(id,fingerprint,configuration) VALUES(1,?,?)", (self._config_hash, self._config_json))
            elif stored[0] != self._config_hash:
                raise BudgetError("Budget ledger configuration differs; spending cannot be reset by resuming with changed settings")
            connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.ledger_path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _begin(connection: sqlite3.Connection) -> None:
        # SQLite's OS-backed write lock serializes check-and-reserve across processes.
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _spent(connection: sqlite3.Connection) -> tuple[int, Decimal]:
        costs = connection.execute("SELECT charged_usd FROM budget_calls").fetchall()
        return len(costs), sum((Decimal(row[0]) for row in costs), Decimal("0"))

    def preflight(self) -> dict:
        """Sanitized, non-network state. Credential contents are not read here."""
        with self._connect() as connection:
            count, spent = self._spent(connection)
            blocked = bool(connection.execute("SELECT blocked FROM budget_config WHERE id=1").fetchone()[0])
        return {**self._config, "configuration_hash": self._config_hash,
                "key_present": bool(os.environ.get(KEY_ENV, "").strip()) or (
                    self._key_file.is_file() and self._key_file.stat().st_size > 0),
                "endpoint_allowed": True, "calls_reserved": count,
                "calls_remaining": max(0, self.max_calls - count),
                "charged_or_reserved_usd": str(spent),
                "available_budget_usd": str(max(Decimal("0"), self.max_cost - spent)),
                "blocked": blocked, "network_used": False}

    def _key(self) -> str:
        # Called only immediately before a completion. Never include values or paths in errors.
        try:
            if KEY_ENV in os.environ:
                value = os.environ[KEY_ENV]
            else:
                with self._key_file.open("r", encoding="utf-8") as stream:
                    value = stream.read(4097)
        except (OSError, UnicodeError):
            raise RuntimeError("Spiderssense credential is unavailable; set SPIDERSSENSE_API_KEY or the configured local key file") from None
        value = value.strip()
        if not value or len(value) > 4096 or any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise RuntimeError("Spiderssense credential is empty or malformed")
        return value

    def _reserve(self, body_bytes: int) -> tuple[int, Decimal, int]:
        input_bound = body_bytes + self.input_overhead_tokens
        reservation = ((Decimal(input_bound) * self.input_price * self.input_cost_multiplier
                        + Decimal(self.max_output_tokens) * self.output_price)
                       * self.reservation_margin / 1_000_000)
        with self._connect() as connection:
            self._begin(connection)
            blocked = connection.execute("SELECT blocked FROM budget_config WHERE id=1").fetchone()[0]
            count, spent = self._spent(connection)
            if blocked:
                raise BudgetError("Budget ledger is blocked after an accounting mismatch")
            if count >= self.max_calls:
                raise BudgetError("Spiderssense persisted call limit reached")
            if spent + reservation > self.max_cost:
                raise BudgetError("Spiderssense request reservation would exceed the persisted USD budget")
            cursor = connection.execute("INSERT INTO budget_calls(reserved_usd,charged_usd,status,input_bound) VALUES(?,?,?,?)",
                                        (str(reservation), str(reservation), "reserved_unknown", input_bound))
            call_id = int(cursor.lastrowid)
            connection.commit()
        return call_id, reservation, input_bound

    def _settle(self, call_id: int, reservation: Decimal, input_bound: int,
                usage: dict) -> tuple[dict, bool]:
        input_tokens = _token_count(usage.get("prompt_tokens"))
        output_tokens = _token_count(usage.get("completion_tokens"))
        reported = None
        invalid_reported = False
        if "cost" in usage:
            try:
                reported = _decimal(usage["cost"], "provider usage cost", allow_zero=True)
            except ValueError:
                invalid_reported = True
        known_rate = None
        if input_tokens is not None and output_tokens is not None:
            known_rate = (Decimal(input_tokens) * self.input_price * self.input_cost_multiplier
                          + Decimal(output_tokens) * self.output_price) / 1_000_000
        # No refund while any billing component is unknown or malformed.
        known = known_rate is not None and not invalid_reported
        charge = max(known_rate, reported or Decimal("0")) if known else reservation
        exceeded = ((reported is not None and reported > reservation)
                    or (known_rate is not None and known_rate > reservation)
                    or (input_tokens is not None and input_tokens > input_bound)
                    or (output_tokens is not None and output_tokens > self.max_output_tokens))
        if exceeded:
            charge = max(reservation, reported or Decimal("0"), known_rate or Decimal("0"))
        with self._connect() as connection:
            self._begin(connection)
            connection.execute("UPDATE budget_calls SET charged_usd=?,status=?,input_tokens=?,output_tokens=?,provider_cost_usd=? WHERE id=?",
                               (str(charge), "accounting_mismatch" if exceeded else "settled" if known else "reserved_unknown",
                                input_tokens, output_tokens, str(reported) if reported is not None else None, call_id))
            if exceeded:
                connection.execute("UPDATE budget_config SET blocked=1 WHERE id=1")
            connection.commit()
        return {"input_tokens": input_tokens, "output_tokens": output_tokens,
                "cost_usd": str(charge), "provider_cost_usd": str(reported) if reported is not None else None,
                "rate_upper_cost_usd": str(known_rate) if known_rate is not None else None,
                "cost_status": "accounting_mismatch" if exceeded else "settled" if known else "reserved_unknown"}, exceeded

    def complete(self, messages: list[dict[str, str]], purpose: str,
                 response_schema: dict | None = None) -> Completion:
        if not isinstance(messages, list) or any(not isinstance(item, dict)
                or set(item) != {"role", "content"} or item["role"] not in {"system", "user", "assistant"}
                or not isinstance(item["content"], str) for item in messages):
            raise ValueError("messages must contain only role/content text objects")
        payload: dict[str, Any] = {"model": self.model, "messages": [dict(item) for item in messages],
                                   "stream": False, self.token_field: self.max_output_tokens}
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.use_json_object:
            payload["response_format"] = {"type": "json_object"}
        if response_schema is not None:
            if not isinstance(response_schema, dict):
                raise ValueError("response_schema must be an object")
            if self.use_json_schema:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "research_memory_response", "strict": True, "schema": response_schema}}
            else:
                payload["messages"].append({"role": "user", "content":
                    "Return only one JSON value satisfying this JSON Schema. No Markdown fences.\n"
                    + json.dumps(response_schema, ensure_ascii=False, sort_keys=True)})
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        key = self._key()
        # Lazy import keeps the standalone research client independent until a
        # call is made. Outside an API journal request this context yields None.
        from local_app.chat_response_journal import captured_call
        with captured_call(purpose=purpose, backend="spiderssense") as capture:
            return self._complete_request(body, key, purpose, capture)

    def _complete_request(self, body: bytes, key: str, purpose: str,
                          capture: Any) -> Completion:
        call_id, reservation, input_bound = self._reserve(len(body))
        request = Request(BASE_URL + "/chat/completions", data=body,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}, method="POST")
        started = time.perf_counter()
        raw = None
        http_status = None
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if capture is not None:
                    http_status = getattr(response, "status", None)
        except HTTPError as exc:
            error = ProviderRequestError(f"Spiderssense returned HTTP {exc.code}; reservation retained",
                                         code="provider_http_error", http_status=exc.code,
                                         ledger_call_id=call_id)
            if capture is not None:
                try:
                    error_body = exc.read(MAX_RESPONSE_BYTES + 1)
                except Exception as read_error:
                    error_body = getattr(read_error, "partial", None)
                capture.raised(error, raw_response=_journal_body(error_body, key, http_status=exc.code)
                               if isinstance(error_body, bytes) else None)
            raise error from None
        except (URLError, TimeoutError, OSError) as exc:
            error = ProviderRequestError("Spiderssense connection failed; reservation retained",
                                         code="provider_connection_error", ledger_call_id=call_id)
            if capture is not None:
                partial = raw if isinstance(raw, bytes) else getattr(exc, "partial", None)
                capture.raised(error, raw_response=_journal_body(partial, key, http_status=http_status)
                               if isinstance(partial, bytes) else None)
            raise error from None
        except Exception as exc:
            error = ProviderRequestError("Spiderssense transport failed; reservation retained",
                                         code="provider_transport_error", ledger_call_id=call_id)
            if capture is not None:
                partial = raw if isinstance(raw, bytes) else getattr(exc, "partial", None)
                capture.raised(error, raw_response=_journal_body(partial, key, http_status=http_status)
                               if isinstance(partial, bytes) else None)
            raise error from None
        elapsed = time.perf_counter() - started
        if capture is not None:
            capture.returned(_journal_body(raw, key, http_status=http_status))
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Spiderssense response exceeds 8 MiB; reservation retained")
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise ValueError("Spiderssense returned malformed JSON; reservation retained") from None
        if not isinstance(result, dict):
            raise ValueError("Spiderssense response must be an object; reservation retained")
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        accounted_usage, exceeded = self._settle(call_id, reservation, input_bound, usage)
        choices = result.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        text = message.get("content")
        safe_text = _redact_credential_text(text, key) if isinstance(text, str) else ""
        credential_echo = isinstance(text, str) and safe_text != text
        finish_reason = choice.get("finish_reason")
        completion = Completion(text=safe_text, usage=accounted_usage,
                                elapsed_seconds=elapsed, metadata={
                                    "backend": "spiderssense", "model": self.model, "purpose": purpose,
                                    "base_url": BASE_URL, "request_timeout_seconds": self.timeout,
                                    "ledger_call_id": call_id, "reserved_usd": str(reservation),
                                    "configuration_hash": self._config_hash,
                                    "finish_reason": finish_reason if finish_reason in {"stop", "length", "content_filter", "tool_calls", "function_call"} else None,
                                })
        if exceeded:
            raise CompletionValidationError("Spiderssense usage exceeded the client reservation; ledger blocked",
                                            completion, "budget_accounting_mismatch")
        if credential_echo:
            raise CompletionValidationError("Spiderssense returned a credential echo; response redacted",
                                            completion, "credential_echo")
        if choice.get("finish_reason") != "stop":
            raise CompletionValidationError("Spiderssense completion did not finish with stop",
                                            completion, "truncated_completion" if choice.get("finish_reason") == "length" else "unfinished_completion")
        if not isinstance(text, str) or not text.strip() or message.get("refusal"):
            raise CompletionValidationError("Spiderssense returned an empty, refused, or malformed completion",
                                            completion, "empty_completion")
        return completion
