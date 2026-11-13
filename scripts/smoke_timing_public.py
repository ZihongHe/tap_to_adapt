"""Exercise the public DTCN timing WebSocket.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

import websocket

try:
    from .smoke_negative_feedback_public import (
        cleanup_smoke_run,
        pcm_tone,
        receive_json,
    )
    from .simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )
except ImportError:  # Direct execution from the scripts directory.
    from smoke_negative_feedback_public import cleanup_smoke_run, pcm_tone, receive_json
    from simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )

def run_account(
    base_url: str,
    code_file: Path,
    label: str,
    assignment_id: str | None = None,
) -> dict[str, Any]:
    with code_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = (
        next(
            (item for item in rows if item.get("assignment_id") == assignment_id),
            None,
        )
        if assignment_id
        else rows[0] if rows else None
    )
    if row is None:
        raise RuntimeError(f"Assignment {assignment_id!r} was not found in {code_file}")
    code = row["participant_code"]
    group = row.get("group") or row.get("internal_group") or row.get("assignment_id")
    study_login = login(base_url, code, "study")
    study_token = str(study_login["token"])
    companion_login = login(base_url, code, "companion")
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
            study_session["id"],
            str(bootstrap["questionnaire_version"]),
        )
        quality = {
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
                "device_id": "public-smoke-browser",
                "simulation": True,
            },
        }
        for kind, metrics in quality.items():
            request_json(
                base_url,
                "/api/quality",
                method="POST",
                token=study_token,
                body={
                    "session_id": study_session["id"],
                    "kind": kind,
                    "passed": True,
                    "metrics": {
                        **metrics,
                        "attempt_id": f"{study_session['id']}-{kind}-1",
                        "attempt_number": 1,
                    },
                },
            )
        study_pair, companion_pair = pair_websockets(
            base_url,
            study_token,
            study_session["id"],
            companion_token,
            companion_session["id"],
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
                },
            },
        )
        parsed = urllib.parse.urlsplit(base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_root = urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, "", ""))
        query = urllib.parse.urlencode(
            {"token": study_token, "session_id": study_session["id"]}
        )
        timing_socket = websocket.create_connection(
            f"{ws_root}/ws/timing?{query}", timeout=90, enable_multithread=True
        )
        ready = receive_json(timing_socket)
        policies: dict[str, str] = {}
        tone = pcm_tone()
        comparison_policy = "base" if row.get("cohort") == "study_a" else "matched_other"
        for policy, token in (("current", "A"), (comparison_policy, "B"), ("vad", "C")):
            timing_socket.send(
                json.dumps(
                    {
                        "type": "phase",
                        "phase": "micro_probe",
                        "policy_token": token,
                        "drain_active": True,
                    }
                )
            )
            acknowledgement = receive_json(timing_socket)
            if acknowledgement.get("type") != "phase_ack":
                raise RuntimeError(f"Unexpected phase response: {acknowledgement}")
            time.sleep(0.25)
            timing_socket.send_binary(tone)
            prediction = receive_json(timing_socket)
            policies[policy] = str(prediction.get("policy_effective"))
            if policies[policy] != policy:
                raise RuntimeError(
                    f"Policy {policy} executed as {policies[policy]} for {label}"
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
            raise RuntimeError(f"Unexpected learning phase response: {learning_ack}")
        timing_socket.send(
            json.dumps(
                {
                    "type": "state",
                    "state": "waiting",
                    "transition_id": str(uuid.uuid4()),
                }
            )
        )
        state_ack = receive_json(timing_socket)
        if state_ack.get("type") != "state_ack":
            raise RuntimeError(f"Unexpected state response: {state_ack}")
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
                    "expected_activation_id": state_ack.get("server_activation_id") or "",
                    "expected_state_version": int(state_ack["server_state_version"]),
                }
            )
        )
        tap_result = receive_json(timing_socket)
        if tap_result.get("type") != "tap_result" or tap_result.get("tap_id") != tap_id:
            raise RuntimeError(f"Unexpected tap response: {tap_result}")
        reply = request_json(
            base_url,
            "/api/chat",
            method="POST",
            token=study_token,
            body={
                "session_id": study_session["id"],
                "source": "primary",
                "transcript": "请用一句简短的话回应这次系统连通性测试。",
            },
        )
        tts_audio = request_bytes(
            base_url,
            "/api/tts",
            token=study_token,
            body={"session_id": study_session["id"], "text": str(reply["text"])},
        )
        return {
            "label": label,
            "group": group,
            "timing_ready_mode": ready.get("mode"),
            "policy_effective": policies,
            "tap_action": tap_result.get("action"),
            "tap_training_eligible": tap_result.get("training_eligible"),
            "tap_has_loss": "loss" in tap_result,
            "chat_mode": reply.get("mode"),
            "tts_bytes": len(tts_audio),
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
    parser = argparse.ArgumentParser(description="Exercise the public DTCN timing WebSocket.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--personalized-code-file", type=Path, required=True)
    parser.add_argument("--frozen-code-file", type=Path, required=True)
    parser.add_argument("--personalized-assignment", default="BP01")
    parser.add_argument("--frozen-assignment", default="BF01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    results = [
        run_account(
            base_url,
            args.personalized_code_file,
            "QA-BP-TIMING",
            args.personalized_assignment,
        ),
        run_account(
            base_url,
            args.frozen_code_file,
            "QA-BF-TIMING",
            args.frozen_assignment,
        ),
    ]
    report = {"simulation_only": True, "results": results}
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
