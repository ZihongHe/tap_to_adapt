"""Command-line utility: smoke reconnect complete public.

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

from simulate_feasibility import (
    ensure_consent,
    login,
    pair_websockets,
    request_json,
    submit_baseline,
)
from smoke_negative_feedback_public import submit_quality


def timing_url(base_url: str, token: str, session_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_root = urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, "", ""))
    query = urllib.parse.urlencode({"token": token, "session_id": session_id})
    return f"{ws_root}/ws/timing?{query}"


def open_timing(base_url: str, token: str, session_id: str) -> websocket.WebSocket:
    options: dict[str, Any] = {"timeout": 20, "enable_multithread": True}
    if base_url.startswith("https://"):
        options["sslopt"] = {"cert_reqs": ssl.CERT_REQUIRED}
    socket = websocket.create_connection(
        timing_url(base_url, token, session_id), **options
    )
    ready = json.loads(socket.recv())
    if ready.get("type") != "timing_ready":
        socket.close()
        raise RuntimeError(f"Timing socket was not ready: {ready}")
    return socket


def setup(base_url: str, code: str) -> dict[str, Any]:
    study_login = login(base_url, code, "study")
    companion_login = login(base_url, code, "companion")
    study_token = str(study_login["token"])
    companion_token = str(companion_login["token"])
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
    return {
        "study_token": study_token,
        "companion_token": companion_token,
        "study_session_id": str(study_session["id"]),
        "companion_session_id": str(companion_session["id"]),
        "study_pair": study_pair,
        "companion_pair": companion_pair,
    }


def abort(base_url: str, context: dict[str, Any]) -> None:
    for key in ("timing_socket", "study_pair", "companion_pair"):
        socket = context.get(key)
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
    request_json(
        base_url,
        f"/api/sessions/{context['study_session_id']}/abort",
        method="POST",
        token=str(context["study_token"]),
    )


def reconnect_smoke(base_url: str, code: str) -> dict[str, Any]:
    context = setup(base_url, code)
    try:
        context["study_pair"].close()
        context["companion_pair"].close()
        time.sleep(0.5)
        study_pair, companion_pair = pair_websockets(
            base_url,
            str(context["study_token"]),
            str(context["study_session_id"]),
            str(context["companion_token"]),
            str(context["companion_session_id"]),
        )
        context["study_pair"] = study_pair
        context["companion_pair"] = companion_pair
        context["timing_socket"] = open_timing(
            base_url,
            str(context["study_token"]),
            str(context["study_session_id"]),
        )
        bootstrap = request_json(
            base_url, "/api/bootstrap", token=str(context["study_token"])
        )
        intervals = bootstrap["progress"]["pair_intervals"]
        qualified = [row for row in intervals if int(row["qualified"]) == 1]
        pairs = {
            (row["study_session_id"], row["companion_session_id"])
            for row in qualified
        }
        if len(qualified) < 2 or len(pairs) != 1:
            raise RuntimeError(
                f"Reconnect was not requalified for the original pair: {intervals}"
            )
        return {
            "ok": True,
            "qualified_interval_count": len(qualified),
            "qualified_pair_count": len(pairs),
            "timing_reopened": True,
        }
    finally:
        abort(base_url, context)


def after_complete_smoke(base_url: str, code: str) -> dict[str, Any]:
    context = setup(base_url, code)
    try:
        timing_socket = open_timing(
            base_url,
            str(context["study_token"]),
            str(context["study_session_id"]),
        )
        context["timing_socket"] = timing_socket
        transition_id = str(uuid.uuid4())
        timing_socket.send(
            json.dumps(
                {
                    "type": "phase",
                    "phase": "complete",
                    "policy_token": "A",
                    "drain_active": True,
                    "transition_id": transition_id,
                }
            )
        )
        phase_ack = json.loads(timing_socket.recv())
        if phase_ack.get("type") != "phase_ack" or phase_ack.get("phase") != "complete":
            raise RuntimeError(f"Complete phase was not committed: {phase_ack}")
        timing_socket.send(
            json.dumps(
                {
                    "type": "tap",
                    "tap_id": str(uuid.uuid4()),
                    "pointer_agent_state": "waiting",
                    "client_ms": time.time() * 1000,
                    "monotonic_ms": time.perf_counter() * 1000,
                    "visible": True,
                }
            )
        )
        tap_result = json.loads(timing_socket.recv())
        expected = {
            "type": "tap_result",
            "action": "after_complete_ignored",
            "training_eligible": False,
            "training_applied": False,
            "label": None,
            "quarantine_reason": "interaction_complete",
        }
        mismatches = {
            key: {"expected": value, "actual": tap_result.get(key)}
            for key, value in expected.items()
            if tap_result.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Late complete tap was not ignored: {mismatches}")
        status = request_json(
            base_url,
            f"/api/sessions/{context['study_session_id']}/protocol-status",
            token=str(context["study_token"]),
        )
        if status["tap_receipts"]["after_complete_ignored"] != 1:
            raise RuntimeError(f"Complete-tap audit was not visible: {status['tap_receipts']}")
        return {
            "ok": True,
            "action": tap_result["action"],
            "training_eligible": tap_result["training_eligible"],
            "agent_state_after_tap": "waiting",
            "after_complete_ignored": status["tap_receipts"][
                "after_complete_ignored"
            ],
        }
    finally:
        abort(base_url, context)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.code_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise SystemExit("The smoke requires exactly two isolated simulation codes")
    result = {
        "reconnect": reconnect_smoke(
            args.base_url.rstrip("/"), rows[0]["participant_code"]
        ),
        "after_complete": after_complete_smoke(
            args.base_url.rstrip("/"), rows[1]["participant_code"]
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
