"""Per-instance collective-session lock windows.

Stored beside orchestration state as ``session_lock.json`` so setting or
clearing a lock never writes ``orchestration_state.json``, models, or SQLite.
Missing file means unlocked. Multiple day-windows share one password.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path
from typing import Any

PASSWORD_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
PASSWORD_LENGTH = 8
MAX_LOCK_MS = 24 * 60 * 60 * 1000
MIN_LOCK_MS = 60 * 1000
MAX_WINDOWS = 8
SLOT_MS = 30 * 60 * 1000
SCHEMA = "tap-to-adapt-session-lock/v1"


class SessionLockError(RuntimeError):
    """Invalid lock window or lock-file transition."""


def normalize_session_password(value: str | None) -> str:
    return "".join(
        char for char in str(value or "").upper() if char.isalnum()
    )


def display_session_password(value: str | None) -> str:
    normalized = normalize_session_password(value)
    if len(normalized) == PASSWORD_LENGTH:
        return f"{normalized[:4]}-{normalized[4:]}"
    return normalized


def generate_session_password() -> str:
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def _window_overlaps(slot_start_ms: int, start_ms: int, end_ms: int, slot_ms: int) -> bool:
    return int(slot_start_ms) < int(end_ms) and (
        int(slot_start_ms) + int(slot_ms)
    ) > int(start_ms)


class SessionLockStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read_unlocked(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _empty_snapshot(raw: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "configured": False,
            "active": False,
            "upcoming": False,
            "expired": False,
            "start_ms": None,
            "end_ms": None,
            "windows": [],
            "created_ms": int(raw.get("created_ms") or 0) if raw else None,
            "updated_ms": int(raw.get("updated_ms") or 0) if raw else None,
            "cleared_ms": int(raw.get("cleared_ms") or 0) if raw else None,
            "password": None,
            "password_display": None,
        }

    @staticmethod
    def _parse_windows(raw: dict[str, Any]) -> list[tuple[int, int]]:
        parsed: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        rows = raw.get("windows")
        if not isinstance(rows, list) or not rows:
            start_ms = int(raw.get("start_ms") or 0)
            end_ms = int(raw.get("end_ms") or 0)
            rows = [{"start_ms": start_ms, "end_ms": end_ms}] if start_ms and end_ms else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            start_ms = int(row.get("start_ms") or 0)
            end_ms = int(row.get("end_ms") or 0)
            if start_ms <= 0 or end_ms <= start_ms:
                continue
            key = (start_ms, end_ms)
            if key in seen:
                continue
            seen.add(key)
            parsed.append(key)
        return sorted(parsed)

    @staticmethod
    def _window_views(
        windows: list[tuple[int, int]], now_ms: int
    ) -> list[dict[str, Any]]:
        views = []
        for start_ms, end_ms in windows:
            views.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "active": start_ms <= now_ms < end_ms,
                    "upcoming": now_ms < start_ms,
                    "expired": now_ms >= end_ms,
                }
            )
        return views

    @staticmethod
    def _primary_window(
        views: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for row in views:
            if row["active"]:
                return row
        for row in views:
            if row["upcoming"]:
                return row
        return views[-1] if views else None

    def snapshot(
        self,
        *,
        now_ms: int | None = None,
        include_password: bool = False,
    ) -> dict[str, Any]:
        now_ms = int(now_ms or time.time() * 1000)
        with self._lock:
            raw = self._read_unlocked()
        if not raw or not raw.get("active"):
            return self._empty_snapshot(raw)
        windows = self._parse_windows(raw)
        if not windows:
            return self._empty_snapshot(raw)
        views = self._window_views(windows, now_ms)
        primary = self._primary_window(views)
        active = any(row["active"] for row in views)
        upcoming = (not active) and any(row["upcoming"] for row in views)
        expired = all(row["expired"] for row in views)
        password = str(raw.get("password") or "")
        result = {
            "configured": not expired,
            "active": active,
            "upcoming": upcoming,
            "expired": expired,
            "start_ms": None if primary is None else primary["start_ms"],
            "end_ms": None if primary is None else primary["end_ms"],
            "windows": views,
            "created_ms": int(raw.get("created_ms") or 0),
            "updated_ms": int(raw.get("updated_ms") or 0),
            "cleared_ms": None,
            "password": None,
            "password_display": None,
        }
        if include_password and password:
            result["password"] = normalize_session_password(password)
            result["password_display"] = display_session_password(password)
        return result

    def public_status(self, *, now_ms: int | None = None) -> dict[str, Any]:
        snap = self.snapshot(now_ms=now_ms)
        return {
            "configured": snap["configured"],
            "active": snap["active"],
            "upcoming": snap["upcoming"],
            "expired": snap["expired"],
            "start_ms": snap["start_ms"],
            "end_ms": snap["end_ms"],
            "windows": [
                {
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "active": row["active"],
                    "upcoming": row["upcoming"],
                    "expired": row["expired"],
                }
                for row in (snap.get("windows") or [])
            ],
        }

    def is_active(self, now_ms: int | None = None) -> bool:
        return bool(self.snapshot(now_ms=now_ms)["active"])

    def verify(self, password: str | None, *, now_ms: int | None = None) -> bool:
        snap = self.snapshot(now_ms=now_ms, include_password=True)
        if not snap["active"]:
            return False
        submitted = normalize_session_password(password)
        expected = normalize_session_password(snap.get("password"))
        return bool(expected) and secrets.compare_digest(submitted, expected)

    def blocks_slot(
        self,
        slot_start_ms: int,
        *,
        slot_ms: int = SLOT_MS,
        now_ms: int | None = None,
    ) -> bool:
        snap = self.snapshot(now_ms=now_ms)
        if not snap["configured"]:
            return False
        for row in snap.get("windows") or []:
            if row.get("expired"):
                continue
            if _window_overlaps(
                slot_start_ms, int(row["start_ms"]), int(row["end_ms"]), slot_ms
            ):
                return True
        return False

    def _validated_windows(
        self,
        windows: list[tuple[int, int] | dict[str, Any]],
        *,
        now_ms: int,
    ) -> list[tuple[int, int]]:
        parsed: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for raw in windows:
            if isinstance(raw, dict):
                start_ms = int(raw.get("start_ms") or 0)
                end_ms = int(raw.get("end_ms") or 0)
            else:
                start_ms, end_ms = int(raw[0]), int(raw[1])
            if start_ms <= 0 or end_ms <= 0:
                raise SessionLockError("锁定起止时间无效")
            if end_ms <= start_ms:
                raise SessionLockError("结束时间必须晚于开始时间")
            if end_ms <= now_ms:
                raise SessionLockError("结束时间必须晚于当前时间")
            if end_ms - start_ms < MIN_LOCK_MS:
                raise SessionLockError("锁定时长至少 1 分钟")
            if end_ms - start_ms > MAX_LOCK_MS:
                raise SessionLockError("单次锁定最长 24 小时")
            key = (start_ms, end_ms)
            if key in seen:
                continue
            seen.add(key)
            parsed.append(key)
        if not parsed:
            raise SessionLockError("请至少设置一段锁定时间")
        if len(parsed) > MAX_WINDOWS:
            raise SessionLockError(f"最多设置 {MAX_WINDOWS} 段锁定时间")
        return sorted(parsed)

    def set_window(
        self,
        start_ms: int,
        end_ms: int,
        *,
        actor: str = "admin",
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        return self.set_windows([(start_ms, end_ms)], actor=actor, now_ms=now_ms)

    def set_windows(
        self,
        windows: list[tuple[int, int] | dict[str, Any]],
        *,
        actor: str = "admin",
        now_ms: int | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        now_ms = int(now_ms or time.time() * 1000)
        parsed = self._validated_windows(windows, now_ms=now_ms)
        chosen = normalize_session_password(password) or generate_session_password()
        if len(chosen) != PASSWORD_LENGTH:
            raise SessionLockError("召集密码无效")
        with self._lock:
            previous = self._read_unlocked() or {}
            created_ms = (
                int(previous.get("created_ms") or now_ms)
                if previous.get("active")
                else now_ms
            )
            payload = {
                "schema": SCHEMA,
                "active": True,
                "start_ms": parsed[0][0],
                "end_ms": parsed[0][1],
                "windows": [
                    {"start_ms": start_ms, "end_ms": end_ms}
                    for start_ms, end_ms in parsed
                ],
                "password": chosen,
                "actor": str(actor or "admin"),
                "created_ms": created_ms,
                "updated_ms": now_ms,
            }
            self._write_unlocked(payload)
        return self.snapshot(now_ms=now_ms, include_password=True)

    def clear(self, *, actor: str = "admin", now_ms: int | None = None) -> dict[str, Any]:
        now_ms = int(now_ms or time.time() * 1000)
        with self._lock:
            previous = self._read_unlocked() or {}
            self._write_unlocked(
                {
                    "schema": SCHEMA,
                    "active": False,
                    "actor": str(actor or "admin"),
                    "created_ms": int(previous.get("created_ms") or now_ms),
                    "updated_ms": now_ms,
                    "cleared_ms": now_ms,
                }
            )
        return self.snapshot(now_ms=now_ms)
