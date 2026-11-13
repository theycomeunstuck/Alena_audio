"""Time conventions shared by chat history and day-based memory."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


DEFAULT_LEARNING_TIMEZONE = "Asia/Tomsk"
DEFAULT_DISPLAY_TIMEZONE = "Asia/Tomsk"


def validate_timezone(value):
    """Return an available IANA zone key without accepting display offsets."""
    if (not isinstance(value, str) or not value or len(value) > 128
            or value != value.strip() or any(ord(char) < 32 for char in value)):
        raise ValueError("timezone must be an available IANA identifier")
    try:
        return ZoneInfo(value).key
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("timezone must be an available IANA identifier") from exc


def utc_datetime(value):
    """Normalize an aware timestamp while preserving its exact instant."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("timestamp must include a UTC offset") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value.astimezone(timezone.utc)
