"""Authentication helpers for the local TeachCopilot API."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

PBKDF2_ITERATIONS = 600_000
# Audit F-075: session lifetime comes only from settings.session_hours
# (LOCAL_SESSION_HOURS); the unused SESSION_TTL duplicate was removed.


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    if not isinstance(password, str) or len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, rounds, salt_hex, digest_hex = encoded.split("$", 3)
        if scheme != "pbkdf2_sha256" or int(rounds) < PBKDF2_ITERATIONS:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds))
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (ValueError, TypeError, AttributeError):
        return False


_DUMMY_PASSWORD_HASH = (f"pbkdf2_sha256${PBKDF2_ITERATIONS}${secrets.token_hex(16)}"
                        f"${secrets.token_hex(32)}")


def dummy_password_hash() -> str:
    """Audit F-074: a well-formed hash with the production iteration count that
    no password matches (random salt and digest, created without PBKDF2 work).
    Verifying against it costs the same as a real account, so failed logins for
    unknown or inactive emails no longer answer measurably faster."""
    return _DUMMY_PASSWORD_HASH


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expiry(now: datetime | None = None, *, hours: int = 12) -> datetime:
    return (now or datetime.now(timezone.utc)) + timedelta(hours=hours)
