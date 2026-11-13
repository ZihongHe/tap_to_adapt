"""Regression tests for phone quality.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from app.quality import evaluate_quality


STUDY = {
    "quality_gate": {
        "network": {
            "minimum_samples": 10,
            "maximum_failures": 1,
            "median_ms_max": 200,
            "p95_ms_max": 500,
            "jitter_ms_max": 150,
            "hard_p95_ms_max": 1000,
        },
        "microphone": {
            "snr_db_min": 10,
            "quiet_rms_p90_max": 0.09,
            "speech_rms_p90_min": 0.008,
            "clipping_ratio_max": 0.01,
            "require_asr_probe": True,
        },
        "reaction_latency": {
            "minimum_samples": 6,
            "median_ms_min": 100,
            "median_ms_max": 900,
        },
    }
}


def attempt(**metrics):
    return {"attempt_id": "phone-check-1", "attempt_number": 1, **metrics}


def test_phone_quality_aliases_use_server_thresholds() -> None:
    network_passed, _ = evaluate_quality(
        "phone_network",
        attempt(
            samples=[60.0] * 12,
            failures=0,
            median_ms=60.0,
            p95_ms=60.0,
            jitter_ms=0.0,
        ),
        STUDY,
    )
    microphone_passed, _ = evaluate_quality(
        "phone_microphone",
        attempt(
            quiet_rms_median=0.01,
            quiet_rms_p90=0.02,
            speech_rms_p90=0.12,
            snr_db=21.6,
            clipping_ratio=0.0,
        ),
        STUDY,
    )
    reaction_passed, _ = evaluate_quality(
        "phone_reaction_latency",
        attempt(samples_ms=[220, 240, 260, 280, 300, 320], median_ms=260),
        STUDY,
    )

    assert network_passed
    assert microphone_passed
    assert reaction_passed


def test_phone_browser_requires_typed_sentence_recall() -> None:
    metrics = attempt(
        secure_context=True,
        media_devices=True,
        audio_context=True,
        websocket=True,
        speech_recognition=True,
        audio_element=True,
        speech_synthesis=True,
        playback_api="speechSynthesis",
        phrase_id="zh-blue-kite-v1",
        test_language="zh-CN",
        verification_method="typed_keyword_recall",
        playback_verified=False,
        typed_answer_stored=False,
        answer_chars=9,
        matched_keyword_count=3,
        required_keyword_count=4,
    )
    failed, failed_metrics = evaluate_quality("phone_browser", metrics, STUDY)
    metrics["playback_verified"] = True
    metrics["matched_keyword_count"] = 4
    passed, passed_metrics = evaluate_quality("phone_browser", metrics, STUDY)

    assert not failed
    assert "playback_verified" in failed_metrics["server_decision"]["missing"]
    assert "keyword_recall" in failed_metrics["server_decision"]["missing"]
    assert passed
    assert passed_metrics["server_decision"]["missing"] == []


def test_desktop_browser_rejects_the_legacy_capability_only_probe() -> None:
    metrics = attempt(
        secure_context=True,
        media_devices=True,
        audio_worklet=True,
        websocket=True,
        speech_recognition=True,
        media_recorder=True,
    )
    passed, evaluated = evaluate_quality("browser", metrics, STUDY)

    assert not passed
    assert "speech_synthesis" in evaluated["server_decision"]["missing"]
    assert "verification_method" in evaluated["server_decision"]["missing"]
