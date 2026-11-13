"""Regression tests for realtime lease.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from app.realtime_lease import RealtimeLeaseRegistry


def test_realtime_lease_acquire_and_release():
    registry = RealtimeLeaseRegistry()
    lease = registry.acquire(session_id="s1", participant_id="p1", day=1)
    assert lease.active is True
    snapshot = registry.snapshot()
    assert snapshot["active_count"] == 1
    released = registry.release("s1")
    assert released is not None
    assert released.active is False
    assert registry.snapshot()["active_count"] == 0


def test_realtime_lease_idempotent_acquire():
    registry = RealtimeLeaseRegistry()
    first = registry.acquire(session_id="s1", participant_id="p1", day=1)
    second = registry.acquire(session_id="s1", participant_id="p1", day=1)
    assert first.acquired_at_ms == second.acquired_at_ms
    assert registry.snapshot()["active_count"] == 1
