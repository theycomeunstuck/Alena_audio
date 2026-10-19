"""Explicitly enabled loopback-only entry to the existing synthetic demo."""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from local_app.auth import verify_password


DEMO_TEACHER_EMAIL = "demo.rich.teacher@local.test"
DEMO_STUDENT_EMAIL = "demo.nika04@local.test"
DEMO_STUDENT_ID = "demo-nika-04"
DEMO_LEARNERS = frozenset({"demo-luna-03", "demo-mika-03", "demo-nika-04", "demo-ilya-04", "demo-rita-04", "demo-oren-03"})
DEFAULT_CREDENTIALS = Path(__file__).resolve().parents[1] / ".local" / "rich-demo-credentials.json"


def require_local_demo_request(request: Request, *, login: bool = False) -> None:
    """Require loopback ASGI peer/Host and a same-origin browser POST.

    This intentionally is not a reverse-proxy authentication mechanism. Keep
    the opt-in application bound to loopback and do not publish it through a
    proxy while demo login is enabled.
    """
    try:
        peer = ipaddress.ip_address(request.client.host) if request.client else None
        host = urlsplit("http://" + request.headers.get("host", ""))
        if not peer or not peer.is_loopback or host.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("not loopback")
        port = host.port or 80
        if request.url.scheme != "http" or host.username or host.password or host.path or host.query or host.fragment:
            raise ValueError("invalid local host")
        if request.headers.get("sec-fetch-site") not in {None, "same-origin", "none"}:
            raise ValueError("cross origin")
        if login:
            origin = urlsplit(request.headers.get("origin", ""))
            if (request.headers.get("x-teachcopilot-demo") != "1" or origin.scheme != "http"
                    or origin.hostname != host.hostname or (origin.port or 80) != port
                    or origin.username or origin.password or origin.path or origin.query or origin.fragment):
                raise ValueError("same-origin demo request required")
    except ValueError:
        raise HTTPException(status_code=403, detail="local same-origin demo access required") from None


def _synthetic_card(card: dict) -> bool:
    return (str(card.get("learner_id", "")) in DEMO_LEARNERS
            and str(card.get("pseudonym", "")).casefold().startswith(("демо", "demo"))
            and card.get("learner_model", {}).get("engagement", {}).get("record_type") == "synthetic_demo")


def demo_account(store, credentials_path: str | Path, role: str) -> dict:
    """Recheck the seed password, real role and all assigned profiles each time."""
    try:
        path = Path(credentials_path)
        if path.stat().st_size > 65536:
            raise ValueError("invalid credentials")
        credentials = json.loads(path.read_text(encoding="utf-8"))
        if role == "teacher":
            entry = credentials["teacher"]
            email = DEMO_TEACHER_EMAIL
        else:
            entries = [row for row in credentials["students"] if row.get("learner_id") == DEMO_STUDENT_ID]
            if len(entries) != 1:
                raise ValueError("ambiguous demo student")
            entry = entries[0]
            email = DEMO_STUDENT_EMAIL
        if entry.get("email") != email or not isinstance(entry.get("password"), str):
            raise ValueError("invalid demo identity")
        account = store.get_account(email=email)
        if (not account or account.get("role") != role or not account.get("active", True)
                or not verify_password(entry["password"], account.get("password_hash", ""))):
            raise ValueError("demo account unavailable")
        if role == "teacher" and account.get("payload", {}).get("synthetic_demo") is not True:
            raise ValueError("not a synthetic teacher")
        account_id = str(account.get("account_id") or account.get("id"))
        assigned = []
        for summary in store.list_cards():
            learner_id = str(summary.get("learner_id", ""))
            if not store.authorize_learner_access(account_id, learner_id):
                continue
            # The real Store roster returns summaries, without learner_model.
            # Fetch full records only after checking this account's access.
            card = store.get_card(learner_id)
            if not isinstance(card, dict) or str(card.get("learner_id", "")) != learner_id:
                raise ValueError("demo profile unavailable")
            assigned.append(card)
        if (not assigned or not all(_synthetic_card(card) for card in assigned)
                or DEMO_STUDENT_ID not in {card["learner_id"] for card in assigned}
                or (role == "student" and len(assigned) != 1)):
            raise ValueError("demo access changed")
        return account
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        # Neither credential contents nor account details leave this boundary.
        raise HTTPException(status_code=503, detail="local synthetic demo is unavailable") from None
