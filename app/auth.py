"""Participant-code normalization, keyed hashing, and signed access tokens.

Participant and IP identifiers are represented by domain-specific keyed hashes.
Token verification checks the signature and expiration before returning claims.
Secrets are supplied by Settings and must be unique to each deployment.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from typing import Any

from .config import Settings


class TokenError(ValueError):
    pass


PARTICIPANT_CODE_PATTERN = re.compile(r"^TTA-[A-Z0-9]{4}-[A-Z0-9]{4}$")


def normalize_participant_code(code: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9]", "", str(code or "")).upper()
    if raw.startswith("TTA") and len(raw) >= 11:
        return f"TTA-{raw[3:7]}-{raw[7:11]}"
    return str(code or "").strip().upper()


def is_full_participant_code(code: str) -> bool:
    return bool(PARTICIPANT_CODE_PATTERN.fullmatch(normalize_participant_code(code)))


def code_hash(code: str, settings: Settings) -> str:
    normalized = code.strip().upper().encode("utf-8")
    return hmac.new(settings.code_pepper.encode("utf-8"), normalized, hashlib.sha256).hexdigest()


def ip_hash(ip: str, settings: Settings) -> str:
    return hmac.new(
        settings.ip_pepper.encode("utf-8"), ip.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def issue_token(
    participant_id: str,
    client_role: str,
    settings: Settings,
    ttl_seconds: int = 12 * 3600,
) -> str:
    payload = {
        "sub": participant_id,
        "role": client_role,
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_seconds,
    }
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    body = base64.urlsafe_b64encode(raw).rstrip(b"=")
    sig = hmac.new(settings.token_secret.encode("utf-8"), body, hashlib.sha256).digest()
    return f"{body.decode()}.{base64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"


def decode_token(token: str, settings: Settings) -> dict[str, Any]:
    try:
        body_text, sig_text = token.split(".", 1)
        body = body_text.encode("ascii")
        supplied_sig = base64.urlsafe_b64decode(sig_text + "=" * (-len(sig_text) % 4))
    except (ValueError, UnicodeError) as exc:
        raise TokenError("Malformed token") from exc

    expected_sig = hmac.new(
        settings.token_secret.encode("utf-8"), body, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(supplied_sig, expected_sig):
        raise TokenError("Invalid token signature")

    try:
        raw = base64.urlsafe_b64decode(body_text + "=" * (-len(body_text) % 4))
        payload = json.loads(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise TokenError("Invalid token payload") from exc

    if int(payload.get("exp", 0)) < int(time.time()):
        raise TokenError("Expired token")
    if not payload.get("sub") or payload.get("role") not in {"study", "companion"}:
        raise TokenError("Invalid token claims")
    return payload

