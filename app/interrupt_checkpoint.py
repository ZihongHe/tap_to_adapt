"""Read-only interrupt checkpoint for setup-screen continue/restart.

Never writes orchestration, models, or participant rows. Callers persist
only by appending ordinary study events.
"""

from __future__ import annotations

import json
import re
from typing import Any


HINT_RE = re.compile(r"^[A-Za-z0-9]{4}$")
INTERRUPT_TYPES = (
    "session.interrupted",
    "session.interaction_presence_expired",
)
CONTINUE_TYPE = "session.interrupt_continue_chosen"
RESTART_TYPE = "session.interrupt_restart_chosen"


def normalize_code_hint(value: str | None) -> str:
    return str(value or "").strip().upper()


def is_code_hint(value: str | None) -> bool:
    return bool(HINT_RE.fullmatch(str(value or "").strip()))


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("payload")
    if isinstance(raw, dict):
        return raw
    try:
        loaded = json.loads(str(raw or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def infer_interrupt_checkpoint(
    *,
    completed: bool,
    sessions: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build a continue/restart checkpoint from already-loaded rows.

    An aborted study session still counts when it recorded an interrupt,
    so logout does not void the 4-character homepage resume.
    End-to-end receipts do not hide an open or aborted interrupted session;
    only a finished/ended study session without an abort drops the checkpoint.
    """
    _ = completed
    open_session = next(
        (
            row
            for row in sessions
            if str(row.get("client_role") or "") == "study"
            and str(row.get("state") or "") in {"started", "active"}
            and row.get("ended_ms") is None
        ),
        None,
    )
    if open_session is None:
        aborted = [
            row
            for row in sessions
            if str(row.get("client_role") or "") == "study"
            and str(row.get("state") or "") == "aborted"
        ]
        open_session = max(
            aborted,
            key=lambda row: (
                int(row.get("ended_ms") or 0),
                int(row.get("started_ms") or 0),
                str(row.get("id") or ""),
            ),
            default=None,
        )
    if open_session is None:
        return None
    session_id = str(open_session.get("id") or "")
    session_events = [
        row
        for row in events
        if str(row.get("session_id") or "") == session_id
    ]
    interrupted = [
        row
        for row in session_events
        if str(row.get("event_type") or "") in INTERRUPT_TYPES
    ]
    if not interrupted:
        return None
    last = interrupted[-1]
    payload = _payload(last)
    cursor = _int(payload.get("cursor"))
    if cursor is None:
        cursor = 0
        for row in session_events:
            item = _payload(row)
            block = _int(
                item.get("active_block_index")
                or item.get("block")
                or item.get("window")
            )
            if block and block >= 1:
                cursor = max(cursor, (block - 1) * 2)
    continues = [
        row
        for row in session_events
        if str(row.get("event_type") or "") == CONTINUE_TYPE
    ]
    restarts = [
        row
        for row in session_events
        if str(row.get("event_type") or "") == RESTART_TYPE
    ]
    return {
        "available": True,
        "session_id": session_id,
        "cursor": max(0, cursor),
        "stage_key": str(payload.get("stage_key") or "") or None,
        "active_elapsed_seconds": float(payload.get("active_elapsed_seconds") or 0),
        "opportunities": _int(payload.get("opportunities")) or 0,
        "reason": str(payload.get("reason") or last.get("event_type") or ""),
        "interrupted_ms": _int(last.get("server_ms")),
        "continue_count": len(continues),
        "restart_count": len(restarts),
        "continue_timestamps_ms": [
            _int(row.get("server_ms"))
            for row in continues
            if _int(row.get("server_ms")) is not None
        ],
    }


def public_interrupt_checkpoint(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row or not row.get("available"):
        return None
    return {
        "available": True,
        "session_id": row.get("session_id"),
        "cursor": int(row.get("cursor") or 0),
        "stage_key": row.get("stage_key"),
        "active_elapsed_seconds": float(row.get("active_elapsed_seconds") or 0),
        "opportunities": int(row.get("opportunities") or 0),
        "reason": str(row.get("reason") or ""),
        "interrupted_ms": row.get("interrupted_ms"),
        "continue_count": int(row.get("continue_count") or 0),
    }
