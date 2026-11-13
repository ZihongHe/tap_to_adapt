"""Regression tests for operator resume.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.operator_resume import (
    hide_block5_event,
    keep_milestone_event,
    load_operator_resume,
    public_operator_resume,
)
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


XSLX = "15482239-3fbe-4aa3-885f-6c76bc20e252"


def _write_resume(tmp_path: Path, **overrides) -> dict:
    payload = {
        "schema": "tap-to-adapt-operator-resume/v1",
        "participant_id": XSLX,
        "session_id": "8bfccc9b-8fce-4307-bc88-f41d7406e44b",
        "code_hint": "XSLX",
        "resume_block": 5,
        "keep_milestones_before_ms": 1_000,
        "resume_after_ms": 5_000,
        "hide_block5_before_ms": 5_000,
        "reason": "resume_first_run_block5",
    }
    payload.update(overrides)
    path = tmp_path / "operator_resume" / f"{XSLX}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_load_operator_resume_requires_exact_participant(tmp_path: Path) -> None:
    _write_resume(tmp_path)
    assert load_operator_resume(tmp_path, XSLX)["resume_block"] == 5
    assert load_operator_resume(tmp_path, "someone-else") is None
    assert load_operator_resume(tmp_path, "") is None
    other = tmp_path / "operator_resume" / "someone-else.json"
    other.write_text(json.dumps({"schema": "nope"}), encoding="utf-8")
    assert load_operator_resume(tmp_path, "someone-else") is None


def test_hide_block5_only_drops_old_fifth_window() -> None:
    resume = {
        "hide_block5_before_ms": 5_000,
    }
    assert hide_block5_event(resume, block=5, server_ms=4_999) is True
    assert hide_block5_event(resume, block=5, server_ms=5_000) is False
    assert hide_block5_event(resume, block=4, server_ms=100) is False
    assert hide_block5_event(None, block=5, server_ms=1) is False


def test_block_report_stitches_first_four_and_resumed_fifth() -> None:
    rules = {
        "min_phase_server_ratio": 0.9,
        "min_phase_reported_ratio": 0.98,
        "block_schedule": {
            "enabled": True,
            "active_blocks": 5,
            "primary_seconds": 240,
            "companion_seconds": 120,
            "rest_seconds": 60,
            "rest_after_final_block": False,
        },
    }
    events = []
    clock = 0
    for block in (1, 2, 3, 4, 5):
        events.append(_event("phase.start", clock, block=block, phase="learning"))
        clock += 240_000
        events.append(_event("phase.end", clock, block=block, phase="learning", completed=240))
        events.append(_event("companion.window_start", clock, block=block))
        clock += 120_000
        events.append(_event("companion.window_end", clock, block=block, completed=120))
        if block < 5:
            events.append(_event("rest.start", clock, block=block, planned=60))
            clock += 60_000
            events.append(_event("rest.end", clock, block=block, completed=60))
    first_four = events[:24]
    keep_before = int(first_four[-1]["server_ms"]) + 1
    incomplete_fifth = [_event("phase.start", keep_before + 1_000, block=5, phase="learning")]
    second_run = [
        _event("phase.start", keep_before + 2_000, block=1, phase="learning"),
        _event("phase.end", keep_before + 2_000 + 240_000, block=1, phase="learning", completed=240),
    ]
    resume_after = keep_before + 1_000_000
    fifth_origin = int(events[24]["server_ms"])
    resumed_fifth = [
        {
            **row,
            "payload": dict(row["payload"]),
            "server_ms": resume_after + int(row["server_ms"]) - fifth_origin,
        }
        for row in events[24:]
    ]
    mixed = [
        *first_four,
        *incomplete_fifth,
        {
            "event_type": "session.interrupted",
            "server_ms": keep_before + 1_100,
            "payload": {"reason": "timing_socket_closed"},
        },
        *second_run,
        *resumed_fifth,
    ]
    resume = {
        "keep_milestones_before_ms": keep_before,
        "resume_after_ms": resume_after,
        "hide_block5_before_ms": resume_after,
        "resume_block": 5,
    }
    report = StudyState._block_phase_report(
        object(),
        {
            "id": XSLX,
            "group_name": "personalized",
            "protocol_mode": "strict",
            "operator_resume": resume,
        },
        mixed,
        rules,
    )
    assert report["valid"] is True
    assert report["active_blocks"] == 5

    other = StudyState._block_phase_report(
        object(),
        {"id": "other", "group_name": "personalized", "protocol_mode": "strict"},
        mixed,
        rules,
    )
    assert other["valid"] is False


def test_public_payload_omits_internal_timestamps(tmp_path: Path) -> None:
    row = load_operator_resume(tmp_path, XSLX)
    assert row is None
    _write_resume(tmp_path)
    public = public_operator_resume(load_operator_resume(tmp_path, XSLX))
    assert public == {
        "resume_block": 5,
        "session_id": "8bfccc9b-8fce-4307-bc88-f41d7406e44b",
        "reason": "resume_first_run_block5",
    }
    assert keep_milestone_event(
        load_operator_resume(tmp_path, XSLX),
        {"server_ms": 500, "payload": {"active_block_index": 4}},
    )
    assert not keep_milestone_event(
        load_operator_resume(tmp_path, XSLX),
        {"server_ms": 500, "payload": {"active_block_index": 5}},
    )


def test_client_only_resumes_when_bootstrap_requests_block_five() -> None:
    script = Path(__file__).resolve().parents[1] / "static" / "app.js"
    text = script.read_text(encoding="utf-8")
    assert "function operatorResumeCursor()" in text
    assert "Number(resume.resume_block) !== 5" in text
    assert "async function postEventBatch(events)" in text
    assert "err.status === 413 && events.length > 1" in text
