"""Validate a public intercept-only positive-feedback sample.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import websocket

try:
    from .simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )
    from .smoke_negative_feedback_public import (
        cleanup_smoke_run,
        pcm_tone,
        receive_json,
        submit_quality,
    )
    from .voice_pcm import stream_timing_pcm
except ImportError:  # Direct execution from the scripts directory.
    from simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )
    from smoke_negative_feedback_public import (
        cleanup_smoke_run,
        pcm_tone,
        receive_json,
        submit_quality,
    )
    from voice_pcm import stream_timing_pcm


def timing_url(base_url: str, token: str, session_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_root = urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, "", ""))
    query = urllib.parse.urlencode({"token": token, "session_id": session_id})
    return f"{ws_root}/ws/timing?{query}"


def run(
    base_url: str,
    code: str,
    *,
    expect_training: bool = True,
    voiced_pcm: bytes | None = None,
    pace_voiced_pcm: bool = True,
) -> dict[str, Any]:
    study_login = login(base_url, code, "study")
    companion_login = login(base_url, code, "companion")
    study_token = str(study_login["token"])
    companion_token = str(companion_login["token"])
    study_session: dict[str, Any] | None = None
    study_pair: websocket.WebSocket | None = None
    companion_pair: websocket.WebSocket | None = None
    timing_socket: websocket.WebSocket | None = None
    pending_error: BaseException | None = None
    try:
        bootstrap = ensure_consent(base_url, study_token)
        study_session = request_json(
            base_url,
            "/api/sessions/start",
            method="POST",
            token=study_token,
            body={"day": 1},
        )
        companion_session = request_json(
            base_url,
            "/api/sessions/start",
            method="POST",
            token=companion_token,
            body={"day": 1},
        )
        submit_baseline(
            base_url,
            study_token,
            str(study_session["id"]),
            str(bootstrap["questionnaire_version"]),
        )
        submit_quality(base_url, study_token, str(study_session["id"]))
        study_pair, companion_pair = pair_websockets(
            base_url,
            study_token,
            str(study_session["id"]),
            companion_token,
            str(companion_session["id"]),
        )
        request_json(
            base_url,
            f"/api/sessions/{study_session['id']}/activate",
            method="POST",
            token=study_token,
        )
        request_json(
            base_url,
            "/api/events",
            method="POST",
            token=study_token,
            body={
                "session_id": study_session["id"],
                "event_type": "topic.selected",
                "payload": {
                    "topic_id": "free_topic",
                    "category_id": "free",
                    "reused_topic_id": False,
                    "simulation": True,
                },
            },
        )
        options: dict[str, Any] = {"timeout": 90, "enable_multithread": True}
        if base_url.startswith("https://"):
            options["sslopt"] = {"cert_reqs": ssl.CERT_REQUIRED}
        timing_socket = websocket.create_connection(
            timing_url(base_url, study_token, str(study_session["id"])), **options
        )
        ready = receive_json(timing_socket)
        ready_personal = dict(ready.get("calibration", {}).get("personal", {}))
        calibration_enabled = ready_personal.get("enabled") is True
        if calibration_enabled is not expect_training:
            raise RuntimeError(
                "Positive-feedback fixture update policy does not match its group"
            )
        calibration_bias_before = float(ready_personal.get("logit_bias", 0.0))

        timing_socket.send(
            json.dumps(
                {
                    "type": "phase",
                    "phase": "micro_probe",
                    "policy_token": "A",
                    "drain_active": True,
                }
            )
        )
        if receive_json(timing_socket).get("type") != "phase_ack":
            raise RuntimeError("Micro-probe phase was not committed")
        # Match the browser AudioWorklet's sub-8 KiB PCM frames.  A 0.6 s
        # mono/int16 frame is 19,200 bytes and is correctly rejected by the
        # production 8,192-byte WebSocket frame limit before it reaches the
        # application.
        if voiced_pcm is None:
            for _ in range(3):
                timing_socket.send_binary(pcm_tone(0.2))
                probe_prediction = receive_json(timing_socket)
                if probe_prediction.get("type") != "prediction":
                    raise RuntimeError(
                        f"Unexpected micro-probe response: {probe_prediction}"
                    )
        else:
            stream_timing_pcm(
                timing_socket,
                voiced_pcm,
                receive_json,
                pace=pace_voiced_pcm,
            )

        timing_socket.send(
            json.dumps(
                {
                    "type": "phase",
                    "phase": "learning",
                    "policy_token": "A",
                    "drain_active": True,
                }
            )
        )
        learning_ack = receive_json(timing_socket)
        if learning_ack.get("type") != "phase_ack":
            raise RuntimeError("Learning phase was not committed")
        tap_id = str(uuid.uuid4())
        timing_socket.send(
            json.dumps(
                {
                    "type": "tap",
                    "tap_id": tap_id,
                    "pointer_agent_state": "waiting",
                    "client_ms": time.time() * 1000,
                    "monotonic_ms": time.perf_counter() * 1000,
                    "visible": True,
                    "intended_action": "manual_activate",
                    "expected_state": "waiting",
                    "expected_activation_id": learning_ack.get("server_activation_id") or "",
                    "expected_state_version": int(learning_ack["server_state_version"]),
                }
            )
        )
        tap_result = receive_json(timing_socket)
        if tap_result.get("type") != "tap_result" or tap_result.get("tap_id") != tap_id:
            raise RuntimeError(f"Unexpected positive tap response: {tap_result}")
        expected: dict[str, Any] = {
            "action": "manual_activate",
            "intended_action": "manual_activate",
            "label": 1,
        }
        if expect_training:
            expected.update(
                {
                    "training_eligible": True,
                    "training_applied": True,
                    "personal_calibration_applied": True,
                    "network_update_scale": 0.0,
                    "current_network_update_applied": False,
                    "requested_training_steps": 0,
                    "quarantined": False,
                }
            )
        else:
            expected.update(
                {
                    "training_eligible": False,
                    "training_applied": False,
                    "quarantined": True,
                    "quarantine_reason": "activation_not_learning_current",
                }
            )
        mismatches = {
            key: {"expected": value, "actual": tap_result.get(key)}
            for key, value in expected.items()
            if tap_result.get(key) != value
        }
        before = float(
            tap_result.get("personal_logit_bias_before", calibration_bias_before)
        )
        after = float(tap_result.get("personal_logit_bias_after", before))
        if expect_training and after <= before:
            mismatches["personal_logit_bias"] = {
                "expected": "after > before",
                "actual": [before, after],
            }
        if not expect_training:
            if tap_result.get("personal_calibration_applied") not in {None, False}:
                mismatches["personal_calibration_applied"] = {
                    "expected": "absent or false",
                    "actual": tap_result.get("personal_calibration_applied"),
                }
            if tap_result.get("current_network_update_applied") not in {None, False}:
                mismatches["current_network_update_applied"] = {
                    "expected": "absent or false",
                    "actual": tap_result.get("current_network_update_applied"),
                }
            if after != before:
                mismatches["personal_logit_bias"] = {
                    "expected": "unchanged",
                    "actual": [before, after],
                }
        if mismatches:
            raise RuntimeError(f"Positive-feedback public smoke failed: {mismatches}")

        # A delayed replay of the same physical intent must never be re-read as
        # another activation.  This mirrors the browser's generation guard and
        # proves the public conditional-tap CAS contract for positive feedback.
        stale_tap_id = str(uuid.uuid4())
        timing_socket.send(
            json.dumps(
                {
                    "type": "tap",
                    "tap_id": stale_tap_id,
                    "pointer_agent_state": "waiting",
                    "client_ms": time.time() * 1000,
                    "monotonic_ms": time.perf_counter() * 1000,
                    "visible": True,
                    "intended_action": "manual_activate",
                    "expected_state": "waiting",
                    "expected_activation_id": learning_ack.get("server_activation_id")
                    or "",
                    "expected_state_version": int(
                        learning_ack["server_state_version"]
                    ),
                }
            )
        )
        stale_result = receive_json(timing_socket)
        if (
            stale_result.get("type") != "tap_result"
            or stale_result.get("tap_id") != stale_tap_id
            or stale_result.get("action") != "conditional_tap_cancelled"
            or stale_result.get("accepted") is not False
            or stale_result.get("training_applied") is not False
        ):
            raise RuntimeError(
                f"Stale manual activation was not safely cancelled: {stale_result}"
            )
        reply = request_json(
            base_url,
            "/api/chat",
            method="POST",
            token=study_token,
            body={
                "session_id": study_session["id"],
                "source": "primary",
                "transcript": "请用一句简短的话回应这次语音链路测试。",
            },
        )
        tts_audio = request_bytes(
            base_url,
            "/api/tts",
            token=study_token,
            body={"session_id": study_session["id"], "text": str(reply["text"])},
        )
        return {
            "ok": True,
            "session_id": study_session["id"],
            "timing_ready_mode": ready.get("mode"),
            "personal_logit_bias_before": before,
            "personal_logit_bias_after": after,
            "personal_calibration_enabled": calibration_enabled,
            "stale_manual_action": stale_result.get("action"),
            "stale_manual_cancel_reason": stale_result.get("cancel_reason"),
            "chat_mode": reply.get("mode"),
            "tts_bytes": len(tts_audio),
            **expected,
        }
    except BaseException as error:
        pending_error = error
        raise
    finally:
        cleanup_errors = cleanup_smoke_run(
            base_url,
            study_token,
            str(study_session["id"]) if study_session is not None else None,
            (
                ("timing_socket", timing_socket),
                ("study_pair", study_pair),
                ("companion_pair", companion_pair),
            ),
        )
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            if pending_error is not None:
                pending_error.add_note(f"Smoke cleanup also failed: {detail}")
            else:
                raise RuntimeError(f"Smoke cleanup failed: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a public intercept-only positive-feedback sample."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.code_file.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(
            item
            for item in csv.DictReader(handle)
            if item["purpose"] in {"positive", "personalized_positive", "frozen_positive"}
        )
    result = run(
        args.base_url.rstrip("/"),
        row["participant_code"],
        expect_training=row.get("internal_group") != "frozen",
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
