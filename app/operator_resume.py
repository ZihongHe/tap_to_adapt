"""Participant-scoped operator resume hints.

A sidecar file is the only way to enable this path. Missing, invalid, or
mismatched files must return None so every other subject is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA = "tap-to-adapt-operator-resume/v1"


def operator_resume_path(data_dir: str | Path, participant_id: str) -> Path:
    return Path(data_dir) / "operator_resume" / f"{participant_id}.json"


def load_operator_resume(
    data_dir: str | Path | None,
    participant_id: str,
) -> dict[str, Any] | None:
    pid = str(participant_id or "").strip()
    if not data_dir or not pid:
        return None
    path = operator_resume_path(data_dir, pid)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    if str(raw.get("schema") or "") != SCHEMA:
        return None
    if str(raw.get("participant_id") or "") != pid:
        return None
    try:
        resume_block = int(raw.get("resume_block") or 0)
        hide_before = int(raw.get("hide_block5_before_ms") or 0)
        keep_before = int(raw.get("keep_milestones_before_ms") or 0)
        drop_until = int(raw.get("resume_after_ms") or hide_before or 0)
    except (TypeError, ValueError):
        return None
    if resume_block != 5 or hide_before <= 0 or keep_before <= 0 or drop_until <= 0:
        return None
    session_id = str(raw.get("session_id") or "").strip()
    return {
        "schema": SCHEMA,
        "participant_id": pid,
        "session_id": session_id,
        "code_hint": str(raw.get("code_hint") or ""),
        "resume_block": resume_block,
        "keep_milestones_before_ms": keep_before,
        "resume_after_ms": drop_until,
        "hide_block5_before_ms": hide_before,
        "reason": str(raw.get("reason") or ""),
    }


def public_operator_resume(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "resume_block": int(row["resume_block"]),
        "session_id": str(row.get("session_id") or ""),
        "reason": str(row.get("reason") or ""),
    }


def hide_block5_event(resume: dict[str, Any] | None, *, block: int, server_ms: int) -> bool:
    if not resume or int(block) != 5:
        return False
    return int(server_ms or 0) < int(resume["hide_block5_before_ms"])


def keep_milestone_event(resume: dict[str, Any] | None, event: dict[str, Any]) -> bool:
    """Keep first-run blocks 1-4 plus post-resume block 5; drop the gap."""
    if not resume:
        return True
    server_ms = int(event.get("server_ms") or 0)
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        block = int(
            payload.get(
                "block",
                payload.get("window", payload.get("active_block_index", 0)),
            )
            or 0
        )
    except (TypeError, ValueError):
        block = 0
    keep_before = int(resume["keep_milestones_before_ms"])
    resume_after = int(resume["resume_after_ms"])
    if server_ms < keep_before:
        return block in {1, 2, 3, 4}
    if server_ms >= resume_after:
        return True
    return False
