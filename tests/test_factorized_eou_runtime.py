"""Regression tests for factorized eou runtime.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.timing import ParticipantRuntime


def pcm_frame(amplitude: int) -> bytes:
    return np.full(3_200, amplitude, dtype="<i2").tobytes()


def runtime() -> ParticipantRuntime:
    return ParticipantRuntime(
        participant_id="factorized-test",
        group_name="personalised",
        cohort="simulation",
        settings=Settings.from_env(),
        phase="learning",
        activation_threshold=0.01,
        minimum_end_of_turn_silence_ms=400,
        decision_context_required=True,
        reply_needed_threshold=0.60,
        decision_context_ready=False,
    )


def test_activation_requires_eou_addressee_and_reply_needed() -> None:
    instance = runtime()
    instance.set_decision_context(
        conversation_target="experimental_agent",
        reply_needed_probability=0.9,
        source="unit_test",
        text_sha256="a" * 64,
    )
    assert instance.process_pcm(pcm_frame(4_000), 10_000)["activate"] is False
    first_silence = instance.process_pcm(pcm_frame(0), 10_200)
    assert first_silence["activate"] is False
    assert "end_of_turn_silence" in first_silence["decision_gate_reasons"]
    second_silence = instance.process_pcm(pcm_frame(0), 10_400)
    assert second_silence["activate"] is True
    assert second_silence["silence_ms"] >= 400

    tap = instance.tap(
        10_450,
        pointer_agent_state="thinking",
        page_visible=True,
        intended_action="interrupt",
        expected_state="thinking",
        expected_activation_id=str(second_silence["activation_id"]),
        expected_state_version=int(second_silence["activation_state_version"]),
        feedback_attribution="reply_needed_context",
    )
    assert tap["training_eligible"] is False
    assert tap["training_applied"] is False
    assert tap["quarantine_reason"] == "feedback_attributed_to_context_gate"

    companion = runtime()
    companion.set_decision_context(
        conversation_target="companion_agent",
        reply_needed_probability=1.0,
        source="unit_test",
        text_sha256="b" * 64,
    )
    companion.process_pcm(pcm_frame(4_000), 20_000)
    companion.process_pcm(pcm_frame(0), 20_200)
    blocked = companion.process_pcm(pcm_frame(0), 20_400)
    assert blocked["activate"] is False
    assert "conversation_target" in blocked["decision_gate_reasons"]


def test_missing_context_is_fail_closed() -> None:
    instance = runtime()
    instance.process_pcm(pcm_frame(4_000), 30_000)
    instance.process_pcm(pcm_frame(0), 30_200)
    blocked = instance.process_pcm(pcm_frame(0), 30_400)
    assert blocked["activate"] is False
    assert "decision_context_missing" in blocked["decision_gate_reasons"]


def test_completed_response_consumes_context_and_prevents_reactivation_loop() -> None:
    instance = runtime()
    instance.set_decision_context(
        conversation_target="experimental_agent",
        reply_needed_probability=0.9,
        source="unit_test",
        text_sha256="c" * 64,
    )
    instance.process_pcm(pcm_frame(4_000), 40_000)
    instance.process_pcm(pcm_frame(0), 40_200)
    activation = instance.process_pcm(pcm_frame(0), 40_400)
    assert activation["activate"] is True

    instance.set_state("speaking")
    instance.set_state("waiting")
    assert instance.decision_context_ready is False
    assert instance.reply_needed_probability == 0.0

    instance.process_pcm(pcm_frame(0), 42_000)
    blocked = instance.process_pcm(pcm_frame(0), 42_400)
    assert blocked["activate"] is False
    assert "decision_context_missing" in blocked["decision_gate_reasons"]
