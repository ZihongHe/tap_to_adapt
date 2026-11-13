"""Regression tests for block schedule.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.study_state import StudyState


def _event(
    event_type: str,
    server_ms: int,
    *,
    block: int,
    phase: str | None = None,
    completed: int | None = None,
    planned: int | None = None,
) -> dict:
    payload = {
        "active_block_index": block,
        "analysis_eligible": not event_type.startswith("rest."),
    }
    if phase is not None:
        payload["phase"] = phase
        payload["policy_token"] = "A"
    if event_type.startswith("companion."):
        payload["window"] = block
    if event_type.startswith("rest."):
        payload["block"] = block
        payload["excluded_from_analysis"] = True
    if completed is not None:
        payload["completed_seconds"] = completed
    if planned is not None:
        payload["planned_seconds"] = planned
    return {
        "event_type": event_type,
        "server_ms": server_ms,
        "payload": payload,
    }


def test_strict_block_report_requires_complete_active_and_rest_boundaries() -> None:
    rules = {
        "min_phase_server_ratio": 0.9,
        "min_phase_reported_ratio": 0.98,
        "block_schedule": {
            "enabled": True,
            "active_blocks": 2,
            "primary_seconds": 240,
            "companion_seconds": 120,
            "rest_seconds": 60,
            "rest_after_final_block": True,
        },
    }
    events = []
    clock = 0
    for block in (1, 2):
        events.append(_event("phase.start", clock, block=block, phase="learning"))
        clock += 240_000
        events.append(
            _event(
                "phase.end",
                clock,
                block=block,
                phase="learning",
                completed=240,
            )
        )
        events.append(_event("companion.window_start", clock, block=block))
        clock += 120_000
        events.append(
            _event("companion.window_end", clock, block=block, completed=120)
        )
        events.append(
            _event("rest.start", clock, block=block, planned=60)
        )
        clock += 60_000
        events.append(_event("rest.end", clock, block=block, completed=60))

    report = StudyState._block_phase_report(
        object(),
        {"group_name": "personalized", "protocol_mode": "strict"},
        events,
        rules,
    )
    assert report["valid"] is True
    assert report["active_blocks"] == 2
    assert report["observed_milestones"] == 12
    assert report["rest_excluded_from_analysis"] is True

    events[-1]["payload"]["analysis_eligible"] = True
    invalid = StudyState._block_phase_report(
        object(),
        {"group_name": "personalized", "protocol_mode": "strict"},
        events,
        rules,
    )
    assert invalid["valid"] is False
