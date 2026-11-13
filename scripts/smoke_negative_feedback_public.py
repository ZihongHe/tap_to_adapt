"""Validate a public model-originated negative-feedback sample.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
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
        request_json,
        submit_baseline,
    )
    from .voice_pcm import stream_timing_pcm
except ImportError:  # Direct execution from the scripts directory.
    from simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_json,
        submit_baseline,
    )
    from voice_pcm import stream_timing_pcm


LATENCY_CLOCK = "server_monotonic_activation_commit_v2"


def pcm_tone(seconds: float = 0.2) -> bytes:
    samples = array.array(
        "h",
        (
            int(0.12 * 32767 * math.sin(2 * math.pi * 220 * index / 16000))
            for index in range(int(16000 * seconds))
        ),
    )
    return samples.tobytes()


def receive_json(socket: websocket.WebSocket) -> dict[str, Any]:
    raw = socket.recv()
    if raw in {"", b""}:
        raise RuntimeError(
            "WebSocket closed before a JSON response was received; "
            "check proxy close codes and the configured maximum frame size"
        )
    return json.loads(raw)


def timing_url(base_url: str, token: str, session_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_root = urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, "", ""))
    query = urllib.parse.urlencode({"token": token, "session_id": session_id})
    return f"{ws_root}/ws/timing?{query}"


def submit_quality(base_url: str, token: str, session_id: str) -> None:
    payloads = {
        "network": {
            "median_ms": 40,
            "p95_ms": 100,
            "jitter_ms": 10,
            "failures": 0,
            "samples": [35, 36, 38, 39, 40, 41, 43, 45, 50, 60, 80, 100],
            "simulation": True,
        },
        "microphone": {"snr_db": 24, "clipping_ratio": 0, "simulation": True},
        "reaction_latency": {
            "median_ms": 300,
            "samples_ms": [230, 260, 285, 315, 340, 370],
            "simulation": True,
        },
        "browser": {
            "secure_context": True,
            "media_devices": True,
            "audio_worklet": True,
            "websocket": True,
            "speech_recognition": True,
            "media_recorder": True,
            "device_id": "negative-feedback-public-smoke",
            "simulation": True,
        },
    }
    for kind, metrics in payloads.items():
        request_json(
            base_url,
            "/api/quality",
            method="POST",
            token=token,
            body={
                "session_id": session_id,
                "kind": kind,
                "passed": True,
                "metrics": {
                    **metrics,
                    "attempt_id": f"{session_id}-{kind}-1",
                    "attempt_number": 1,
                },
            },
        )


def cleanup_smoke_run(
    base_url: str,
    study_token: str,
    study_session_id: str | None,
    sockets: tuple[tuple[str, websocket.WebSocket | None], ...],
) -> list[str]:
    """Close local sockets and abort both same-day sessions through the study role."""
    errors: list[str] = []
    for name, socket in sockets:
        if socket is None:
            continue
        try:
            socket.close()
        except Exception as error:  # Cleanup must continue for every resource.
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if study_session_id is not None:
        try:
            request_json(
                base_url,
                f"/api/sessions/{study_session_id}/abort",
                method="POST",
                token=study_token,
            )
        except Exception as error:  # Preserve the primary smoke failure below.
            errors.append(f"session_abort: {type(error).__name__}: {error}")
    return errors


def run(
    base_url: str,
    code: str,
    agent_state: str = "thinking",
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
            timing_url(base_url, study_token, str(study_session["id"])),
            **options,
        )
        ready = receive_json(timing_socket)
        if ready.get("latency_clock") != LATENCY_CLOCK:
            raise RuntimeError(f"Unexpected timing clock: {ready}")
        ready_personal = dict(ready.get("calibration", {}).get("personal", {}))
        calibration_enabled = ready_personal.get("enabled") is True
        if calibration_enabled is not expect_training:
            raise RuntimeError(
                "Negative-feedback fixture update policy does not match its group"
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
        if voiced_pcm is None:
            for _ in range(3):
                timing_socket.send_binary(pcm_tone(0.2))
                if receive_json(timing_socket).get("type") != "prediction":
                    raise RuntimeError("Micro-probe PCM did not produce a prediction")
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
        if receive_json(timing_socket).get("type") != "phase_ack":
            raise RuntimeError("Learning phase was not committed")

        activation: dict[str, Any] | None = None
        silence = b"\x00\x00" * 3200
        for _ in range(40):
            time.sleep(0.25)
            timing_socket.send_binary(silence)
            prediction = receive_json(timing_socket)
            if prediction.get("activate") is True:
                activation = prediction
                break
        if activation is None:
            raise RuntimeError("Model did not activate within the smoke window")

        local_activation_received = time.perf_counter() * 1000
        tap_snapshot = {
            "state": "thinking",
            "activation_id": str(activation["activation_id"]),
            "state_version": int(activation["activation_state_version"]),
        }
        if agent_state == "speaking":
            timing_socket.send(
                json.dumps(
                    {
                        "type": "state",
                        "state": "speaking",
                        "transition_id": str(uuid.uuid4()),
                    }
                )
            )
            state_ack = receive_json(timing_socket)
            if state_ack.get("type") != "state_ack" or state_ack.get("state") != "speaking":
                raise RuntimeError(f"Speaking state was not committed: {state_ack}")
            tap_snapshot = {
                "state": "speaking",
                "activation_id": str(state_ack["server_activation_id"]),
                "state_version": int(state_ack["server_state_version"]),
            }
        # Deliberately provide no post-activation speech and wait beyond the
        # retired two-second gate. A visible user interrupt must still label
        # the activation-time window as negative.
        time.sleep(2.25)
        local_tap_sent = time.perf_counter() * 1000
        tap_id = str(uuid.uuid4())
        timing_socket.send(
            json.dumps(
                {
                    "type": "tap",
                    "tap_id": tap_id,
                    "pointer_agent_state": agent_state,
                    "client_ms": time.time() * 1000,
                    "monotonic_ms": local_tap_sent,
                    "visible": True,
                    "intended_action": "interrupt",
                    "expected_state": tap_snapshot["state"],
                    "expected_activation_id": tap_snapshot["activation_id"],
                    "expected_state_version": tap_snapshot["state_version"],
                }
            )
        )
        tap_result: dict[str, Any] | None = None
        for _ in range(8):
            message = receive_json(timing_socket)
            if message.get("type") == "tap_result" and message.get("tap_id") == tap_id:
                tap_result = message
                break
        if tap_result is None:
            raise RuntimeError("No tap_result was returned")
        expected = {
            "action": "interrupt",
            "intended_action": "interrupt",
            "label": 0,
            "user_speech_resumed": False,
            "label_semantics": "observable_activation_interrupt_v2",
            "label_confidence": "high",
            "interrupt_agent_state": agent_state,
            "latency_clock": LATENCY_CLOCK,
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
        latency_ms = float(tap_result.get("activation_to_tap_ms", 999999))
        if latency_ms <= 2000:
            mismatches["activation_to_tap_ms"] = {
                "expected": ">2000",
                "actual": latency_ms,
            }
        bias_before = float(
            tap_result.get("personal_logit_bias_before", calibration_bias_before)
        )
        bias_after = float(tap_result.get("personal_logit_bias_after", bias_before))
        if expect_training and bias_after >= bias_before:
            mismatches["personal_logit_bias"] = {
                "expected": "after < before",
                "actual": [bias_before, bias_after],
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
            if bias_after != bias_before:
                mismatches["personal_logit_bias"] = {
                    "expected": "unchanged",
                    "actual": [bias_before, bias_after],
                }
        if mismatches:
            raise RuntimeError(f"Negative-feedback smoke failed: {mismatches}")

        # Replaying the stale physical intent after the accepted interrupt must
        # be cancelled, not reinterpreted as a manual activation in Waiting.
        stale_tap_id = str(uuid.uuid4())
        timing_socket.send(
            json.dumps(
                {
                    "type": "tap",
                    "tap_id": stale_tap_id,
                    "pointer_agent_state": agent_state,
                    "client_ms": time.time() * 1000,
                    "monotonic_ms": time.perf_counter() * 1000,
                    "visible": True,
                    "intended_action": "interrupt",
                    "expected_state": tap_snapshot["state"],
                    "expected_activation_id": tap_snapshot["activation_id"],
                    "expected_state_version": tap_snapshot["state_version"],
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
            raise RuntimeError(f"Stale interrupt was not safely cancelled: {stale_result}")
        return {
            "ok": True,
            "session_id": study_session["id"],
            "activation_id": activation.get("activation_id"),
            "activation_compute_ms": activation.get("activation_compute_ms"),
            "local_activation_to_tap_ms": local_tap_sent - local_activation_received,
            "server_activation_to_tap_ms": latency_ms,
            "client_clock_offset_ms": tap_result.get("client_clock_offset_ms"),
            "training_audio_samples": tap_result.get("training_audio_samples"),
            "loss": tap_result.get("loss"),
            "optimizer_step_before": tap_result.get("optimizer_step_before"),
            "optimizer_step_after": tap_result.get("optimizer_step_after"),
            "personal_logit_bias_before": bias_before,
            "personal_logit_bias_after": bias_after,
            "personal_calibration_enabled": calibration_enabled,
            "stale_interrupt_action": stale_result.get("action"),
            "stale_interrupt_cancel_reason": stale_result.get("cancel_reason"),
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
        description="Validate a public model-originated negative-feedback sample."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--agent-state", choices=["thinking", "speaking"], default="thinking"
    )
    args = parser.parse_args()
    with args.code_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        (
            item
            for item in rows
            if item.get("purpose")
            in {"negative", "personalized_negative", "frozen_negative"}
        ),
        rows[0],
    )
    result = run(
        args.base_url.rstrip("/"),
        row["participant_code"],
        args.agent_state,
        expect_training=row.get("internal_group") != "frozen",
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
