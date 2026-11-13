"""Device-quality and reaction-latency validation for study entry.

Input measurements are checked for finite values, valid attempts, microphone
levels, playback confirmation, and reaction latency. Results describe device
readiness and do not replace the participant's corrective timing feedback.
"""

from __future__ import annotations

import math
from typing import Any


class QualityPayloadError(ValueError):
    pass


PLAYBACK_PHRASE_IDS = {"zh-blue-kite-v1", "en-blue-kite-v1"}

NETWORK_THRESHOLDS = {
    "minimum_samples": 10,
    "minimum_successful": 11,
    "maximum_failures": 1,
    "median_ms_max": 250,
    "p95_ms_max": 800,
    "jitter_ms_max": 200,
    "hard_p95_ms_max": 1500,
}

# Operational device-compatibility amendment (2026-08-21).  The v29 study-run
# snapshot is immutable, so the narrowly relaxed desktop admission values live
# here instead of rewriting the protocol metadata of already enrolled users.
DESKTOP_MICROPHONE_ADMISSION = {
    "snr_db_min": 8.0,
    "speech_rms_p90_min": 0.006,
}


def _number(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityPayloadError(f"quality metric {key} is missing or invalid")
    number = float(value)
    if not math.isfinite(number):
        raise QualityPayloadError(f"quality metric {key} must be finite")
    return number


def _optional_finite_number(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _calibration_output_gain(metrics: dict[str, Any]) -> float:
    # iOS WeChat often leaves HTMLAudioElement.volume readable as 1 after a 0.5
    # write. Prefer the gain the companion page requested for this probe.
    requested = _optional_finite_number(metrics, "output_gain_requested")
    if requested is not None:
        return requested
    return _number(metrics, "output_gain")


def _attempt(metrics: dict[str, Any]) -> tuple[str, int]:
    attempt_id = str(metrics.get("attempt_id") or "").strip()
    attempt_number = metrics.get("attempt_number")
    if not attempt_id or len(attempt_id) > 100:
        raise QualityPayloadError("quality attempt_id is required")
    if isinstance(attempt_number, bool) or not isinstance(attempt_number, int):
        raise QualityPayloadError("quality attempt_number is required")
    if attempt_number < 1:
        raise QualityPayloadError("quality attempt_number must be at least 1")
    return attempt_id, attempt_number


def _issue(label: str, actual: str, expected: str) -> dict[str, str]:
    return {"label": label, "actual": actual, "expected": expected}


def _playback_api_valid(metrics: dict[str, Any]) -> bool:
    return metrics.get("playback_api") in {
        "speechSynthesis",
        "server_web_audio",
        "server_mp3",
    }


def _playback_sentence_decision(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    uses_server_tts = metrics.get("playback_api") in {"server_web_audio", "server_mp3"}
    checks = {
        "speech_synthesis": metrics.get("speech_synthesis") is True or uses_server_tts,
        "playback_api": _playback_api_valid(metrics),
        "phrase_id": metrics.get("phrase_id") in PLAYBACK_PHRASE_IDS,
        "test_language": metrics.get("test_language") in {"zh-CN", "en-US"},
        "verification_method": metrics.get("verification_method")
        == "typed_keyword_recall",
        "playback_verified": metrics.get("playback_verified") is True,
        "typed_answer_not_stored": metrics.get("typed_answer_stored") is False,
        "answer_length": isinstance(metrics.get("answer_chars"), int)
        and not isinstance(metrics.get("answer_chars"), bool)
        and int(metrics["answer_chars"]) >= 6,
        "keyword_recall": metrics.get("matched_keyword_count") == 4
        and metrics.get("required_keyword_count") == 4,
    }
    return all(checks.values()), [name for name, passed in checks.items() if not passed]


def evaluate_quality(kind: str, metrics: dict[str, Any], study: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    normalized = dict(metrics)
    gate = study.get("quality_gate", {})
    attempt_id, attempt_number = _attempt(normalized)
    normalized["attempt_id"] = attempt_id
    normalized["attempt_number"] = attempt_number

    if kind in {"network", "phone_network"}:
        rules = {**gate.get("network", {}), **NETWORK_THRESHOLDS}
        samples = normalized.get("samples")
        if not isinstance(samples, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in samples
        ):
            raise QualityPayloadError("network samples must be a numeric list")
        failures = int(_number(normalized, "failures"))
        median = _number(normalized, "median_ms")
        p95 = _number(normalized, "p95_ms")
        jitter = _number(normalized, "jitter_ms")
        successful_samples = normalized.get("successful_samples_ms")
        if isinstance(successful_samples, list):
            successful_count = len(successful_samples)
        else:
            successful_count = max(0, len(samples) - failures)
        minimum_successful = int(rules.get("minimum_successful", 11))
        maximum_failures = int(rules.get("maximum_failures", 1))
        median_max = float(rules.get("median_ms_max", 250))
        p95_max = float(rules.get("p95_ms_max", 800))
        jitter_max = float(rules.get("jitter_ms_max", 200))
        enough_samples = len(samples) >= int(rules.get("minimum_samples", 10))
        issues: list[dict[str, str]] = []
        if successful_count < minimum_successful:
            issues.append(
                _issue("成功次数", f"{successful_count}/12", f"≥ {minimum_successful}/12")
            )
        if failures > maximum_failures:
            issues.append(_issue("丢失", f"{failures}/12", f"≤ {maximum_failures}/12"))
        if median > median_max:
            issues.append(_issue("中位延迟", f"{round(median)} ms", f"≤ {round(median_max)} ms"))
        if p95 > p95_max:
            issues.append(_issue("P95 延迟", f"{round(p95)} ms", f"≤ {round(p95_max)} ms"))
        if jitter > jitter_max:
            issues.append(_issue("抖动", f"{round(jitter)} ms", f"≤ {round(jitter_max)} ms"))
        passed = enough_samples and not issues
        normalized["server_decision"] = {
            "passed": passed,
            "issues": issues,
            "hard_failure": failures > maximum_failures
            or p95 > float(rules.get("hard_p95_ms_max", 1500)),
            "sample_count": len(samples),
        }
        return passed, normalized

    if kind in {"microphone", "phone_microphone"}:
        rules = {**gate.get("microphone", {})}
        if kind == "phone_microphone":
            rules = {**rules, **gate.get("phone_microphone", {})}
        else:
            rules = {**rules, **DESKTOP_MICROPHONE_ADMISSION}
        require_asr_probe = bool(rules.get("require_asr_probe", True)) and kind == "microphone"
        issues: list[dict[str, str]] = []
        if normalized.get("error"):
            passed = False
            issues.append(_issue("麦克风", "不可用", "可正常采集音频"))
        elif normalized.get("simulation") is True:
            passed = True
        else:
            snr = _number(normalized, "snr_db")
            quiet = _number(normalized, "quiet_rms_p90")
            speech = _number(normalized, "speech_rms_p90")
            clipping = _number(normalized, "clipping_ratio")
            snr_min = float(rules.get("snr_db_min", 8))
            quiet_max = float(rules.get("quiet_rms_p90_max", 0.09))
            speech_min = float(rules.get("speech_rms_p90_min", 0.006))
            clipping_max = float(rules.get("clipping_ratio_max", 0.01))
            if quiet >= quiet_max:
                issues.append(_issue("环境噪声", f"{quiet:.3f}", f"< {quiet_max}"))
            if speech <= speech_min:
                issues.append(_issue("正常说话音量", f"{speech:.3f}", f"> {speech_min}"))
            if clipping >= clipping_max:
                issues.append(_issue("削波比例", f"{clipping:.3f}", f"< {clipping_max}"))
            if snr < snr_min:
                issues.append(_issue("信噪比 SNR", f"{snr:.1f} dB", f"≥ {snr_min} dB"))
            acoustic_ok = not issues
            semantic_confirmation = (
                normalized.get("asr_verification_version") == "semantic-confirmation-v1"
            )
            if semantic_confirmation:
                # New clients show the actual transcript and require the
                # participant to confirm that its meaning is approximately
                # correct.  Old pages keep the legacy soft-ASR behaviour so an
                # in-progress setup cannot be invalidated by a hot update.
                transcript = str(normalized.get("asr_probe_transcript") or "").strip()
                if not transcript or normalized.get("asr_user_confirmed") is not True:
                    issues.append(_issue("语音识别语义确认", "未通过", "参与者确认大意正确"))
            # Browser Web Speech is flaky (especially embedded browsers). For
            # legacy clients only, keep treating it as soft evidence when the
            # acoustic measurements prove the microphone works.
            elif require_asr_probe and normalized.get("asr_probe_passed") is not True:
                if acoustic_ok:
                    normalized["asr_probe_soft_failed"] = True
                else:
                    issues.append(_issue("语音识别", "未通过", "通过"))
            passed = not issues
        normalized["server_decision"] = {"passed": passed, "issues": issues}
        return passed, normalized

    if kind in {"reaction_latency", "phone_reaction_latency"}:
        rules = gate.get("reaction_latency", {})
        samples = normalized.get("samples_ms")
        if not isinstance(samples, list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in samples
        ):
            raise QualityPayloadError("reaction samples_ms must be a numeric list")
        median = _number(normalized, "median_ms")
        minimum_samples = int(rules.get("minimum_samples", 6))
        median_min = float(rules.get("median_ms_min", 100))
        median_max = float(rules.get("median_ms_max", 900))
        issues: list[dict[str, str]] = []
        if len(samples) < minimum_samples:
            issues.append(_issue("有效点击", f"{len(samples)} 次", f"≥ {minimum_samples} 次"))
        if median < median_min:
            issues.append(_issue("反应中位数", f"{round(median)} ms", f"≥ {round(median_min)} ms"))
        if median > median_max:
            issues.append(_issue("反应中位数", f"{round(median)} ms", f"≤ {round(median_max)} ms"))
        passed = not issues
        normalized["server_decision"] = {
            "passed": passed,
            "issues": issues,
            "sample_count": len(samples),
        }
        return passed, normalized

    if kind == "browser":
        required = (
            "secure_context",
            "media_devices",
            "audio_worklet",
            "websocket",
            "media_recorder",
        )
        capabilities_passed = all(normalized.get(name) is True for name in required)
        playback_passed, playback_missing = _playback_sentence_decision(normalized)
        passed = capabilities_passed and playback_passed
        normalized["server_decision"] = {
            "passed": passed,
            "missing": [
                *[name for name in required if normalized.get(name) is not True],
                *playback_missing,
            ],
        }
        return passed, normalized

    if kind == "phone_browser":
        required = (
            "secure_context",
            "media_devices",
            "audio_context",
            "websocket",
            "audio_element",
        )
        capabilities_passed = all(normalized.get(name) is True for name in required)
        playback_passed, playback_missing = _playback_sentence_decision(normalized)
        output_gain = _calibration_output_gain(normalized)
        volume_gate_checks = {
            "minimum_volume_probe": normalized.get("probe_context")
            == "minimum_volume_gate",
            "hard_volume_gate": normalized.get("volume_gate_hard") is True,
            "calibration_output_gain": 0.49 <= output_gain <= 0.51,
            "media_volume_minimum": normalized.get("media_volume_min_percent") == 50,
            "minimum_confirmed": normalized.get("user_confirmed_media_volume_minimum")
            is True,
        }
        volume_gate_missing = [
            name for name, passed in volume_gate_checks.items() if not passed
        ]
        missing = [
            *[name for name in required if normalized.get(name) is not True],
            *playback_missing,
            *volume_gate_missing,
        ]
        passed = capabilities_passed and playback_passed and not volume_gate_missing
        reason_labels = {
            "keyword_recall": "听辨内容",
            "playback_verified": "听辨内容",
            "answer_length": "听辨内容长度",
            "calibration_output_gain": "校准播放增益",
            "minimum_volume_probe": "最低音量校准播放",
            "hard_volume_gate": "最低音量硬校准标记",
            "media_volume_minimum": "系统媒体音量下限",
            "minimum_confirmed": "音量确认勾选",
            "phrase_id": "校准句编号",
            "test_language": "校准语言",
            "verification_method": "听辨核对方式",
            "playback_api": "声音播放接口",
            "speech_recognition": "语音识别接口",
            "secure_context": "安全上下文",
            "media_devices": "媒体设备接口",
            "audio_context": "音频上下文",
            "websocket": "WebSocket",
            "audio_element": "音频播放元素",
        }
        normalized["server_decision"] = {
            "passed": passed,
            "missing": missing,
            "issues": [
                _issue(reason_labels.get(name, name), "未通过", "已满足")
                for name in missing
            ],
            "hard_volume_gate": True,
            "media_volume_min_percent": 50,
            "calibration_output_gain": 0.5,
        }
        return passed, normalized

    raise QualityPayloadError(f"unsupported quality kind: {kind}")
