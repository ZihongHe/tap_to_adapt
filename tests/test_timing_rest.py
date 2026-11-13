"""Regression tests for timing rest.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.timing import ParticipantRuntime


def test_rest_phase_rejects_pcm_inference_and_taps() -> None:
    runtime = ParticipantRuntime(
        participant_id="participant",
        group_name="personalised",
        cohort="pilot",
        settings=None,  # set_phase/process rest do not access settings
    )
    transition = runtime.set_phase(
        "rest",
        "current",
        drain_active=True,
        active_block_index=2,
        analysis_eligible=False,
    )
    assert transition["committed"] is True
    assert runtime.ring.samples == 0

    prediction = runtime.process_pcm(b"\x00\x00" * 1600, 1000)
    assert prediction["skipped"] is True
    assert prediction["skip_reason"] == "scheduled_rest_excluded"
    assert prediction["analysis_eligible"] is False
    assert runtime.ring.samples == 0

    tap = runtime.tap(1100)
    assert tap["action"] == "scheduled_rest_ignored"
    assert tap["training_applied"] is False
    assert tap["quarantine_reason"] == "scheduled_rest_excluded"
