"""Capacity-bounded leases for participant streaming connections.

The registry tracks active ownership and expiry so reconnecting clients can
recover a slot without exceeding the configured concurrent-session capacity.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


REALTIME_PROFILE = {
    "ws_timing": True,
    "ws_pair": True,
    "timing_fps_hint": 10,
    "event_batch_interval_sec": 2,
    "estimated_kbps": 200,
}


class CapacityExceeded(RuntimeError):
    """Raised when a new realtime lease would exceed the concurrency cap."""


@dataclass
class RealtimeLease:
    session_id: str
    participant_id: str
    day: int
    acquired_at_ms: int
    last_seen_ms: int
    profile: dict[str, Any] = field(default_factory=lambda: dict(REALTIME_PROFILE))
    released_at_ms: int | None = None

    @property
    def active(self) -> bool:
        return self.released_at_ms is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "day": self.day,
            "acquired_at_ms": self.acquired_at_ms,
            "last_seen_ms": self.last_seen_ms,
            "released_at_ms": self.released_at_ms,
            "active": self.active,
            "profile": dict(self.profile),
        }


class RealtimeLeaseRegistry:
    """In-memory leases for active TAP realtime sessions."""

    def __init__(self, *, stale_after_ms: int = 45 * 60 * 1000) -> None:
        self._leases: dict[str, RealtimeLease] = {}
        self._lock = threading.RLock()
        self._stale_after_ms = max(60_000, int(stale_after_ms))

    def _purge_stale_locked(self, now_ms: int) -> int:
        purged = 0
        for lease in self._leases.values():
            if not lease.active:
                continue
            if now_ms - int(lease.last_seen_ms) > self._stale_after_ms:
                lease.released_at_ms = now_ms
                purged += 1
        return purged

    def acquire(
        self,
        *,
        session_id: str,
        participant_id: str,
        day: int,
        max_concurrent: int | None = None,
    ) -> RealtimeLease:
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._purge_stale_locked(now_ms)
            existing = self._leases.get(session_id)
            if existing is not None and existing.active:
                existing.last_seen_ms = now_ms
                return existing
            active = [lease for lease in self._leases.values() if lease.active]
            if max_concurrent is not None and len(active) >= max(1, int(max_concurrent)):
                raise CapacityExceeded(
                    f"active realtime leases {len(active)} >= {max_concurrent}"
                )
            lease = RealtimeLease(
                session_id=session_id,
                participant_id=participant_id,
                day=day,
                acquired_at_ms=now_ms,
                last_seen_ms=now_ms,
            )
            self._leases[session_id] = lease
            return lease

    def touch(self, session_id: str) -> RealtimeLease | None:
        now_ms = int(time.time() * 1000)
        with self._lock:
            lease = self._leases.get(session_id)
            if lease is None or not lease.active:
                return lease
            lease.last_seen_ms = now_ms
            return lease

    def release(self, session_id: str) -> RealtimeLease | None:
        with self._lock:
            lease = self._leases.get(session_id)
            if lease is None or not lease.active:
                return lease
            lease.released_at_ms = int(time.time() * 1000)
            return lease

    def release_participant(self, participant_id: str) -> list[RealtimeLease]:
        released: list[RealtimeLease] = []
        with self._lock:
            now_ms = int(time.time() * 1000)
            for lease in self._leases.values():
                if lease.participant_id == participant_id and lease.active:
                    lease.released_at_ms = now_ms
                    released.append(lease)
        return released

    def peek(self) -> dict[str, Any]:
        """List currently marked-active leases without purging stale ones."""
        now_ms = int(time.time() * 1000)
        with self._lock:
            active = [lease.to_dict() for lease in self._leases.values() if lease.active]
        return {
            "active_count": len(active),
            "active": active,
            "server_ms": now_ms,
            "readonly": True,
        }

    def snapshot(self) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        with self._lock:
            self._purge_stale_locked(now_ms)
            active = [lease.to_dict() for lease in self._leases.values() if lease.active]
            history = [
                lease.to_dict()
                for lease in self._leases.values()
                if not lease.active
            ]
        return {
            "active_count": len(active),
            "active": active,
            "recent_released": history[-20:],
            "profile_template": dict(REALTIME_PROFILE),
            "stale_after_ms": self._stale_after_ms,
        }
