"""Run the locked smooth 4+2 public-route duration simulation.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import ssl
import threading
import time
import traceback
import urllib.parse
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import websocket

try:
    from .run_duration_simulation_public import (
        PairHeartbeat,
        request_pcm,
        send_conditional_tap,
        send_phase,
        send_state,
        wait_while_servicing_timing_socket,
    )
    from .simulate_feasibility import (
        SimulationError,
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )
    from .smoke_negative_feedback_public import receive_json, submit_quality
    from .smooth_simulation_protocol import RUN_SLUG_PATTERN, SCHEMA, validate_schedule
    from .voice_pcm import (
        PCM_SAMPLE_RATE_HZ,
        PCM_SAMPLE_WIDTH_BYTES,
        TIMING_FRAME_BYTES,
        stream_timing_pcm,
    )
except ImportError:
    from run_duration_simulation_public import (
        PairHeartbeat,
        request_pcm,
        send_conditional_tap,
        send_phase,
        send_state,
        wait_while_servicing_timing_socket,
    )
    from simulate_feasibility import (
        SimulationError,
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )
    from smoke_negative_feedback_public import receive_json, submit_quality
    from smooth_simulation_protocol import RUN_SLUG_PATTERN, SCHEMA, validate_schedule
    from voice_pcm import (
        PCM_SAMPLE_RATE_HZ,
        PCM_SAMPLE_WIDTH_BYTES,
        TIMING_FRAME_BYTES,
        stream_timing_pcm,
    )


def timing_url(base_url: str, token: str, session_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    root = urllib.parse.urlunsplit(
        ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, parsed.path, "", "")
    )
    return (
        f"{root}/ws/timing?"
        + urllib.parse.urlencode({"token": token, "session_id": session_id})
    )


def append_event(
    base_url: str, token: str, session_id: str, record: dict[str, Any]
) -> None:
    request_json(
        base_url,
        "/api/events",
        method="POST",
        token=token,
        body={
            "session_id": session_id,
            "event_type": "simulation.smooth_opportunity",
            "client_ms": int(time.time() * 1000),
            "monotonic_ms": time.perf_counter() * 1000,
            "payload": record,
        },
    )


def activate_after_pair_ready(
    base_url: str,
    token: str,
    session_id: str,
    *,
    attempts: int = 6,
) -> dict[str, Any]:
    """Bridge the brief pair-presence propagation window without hiding 409s."""
    for attempt in range(attempts):
        try:
            return request_json(
                base_url,
                f"/api/sessions/{session_id}/activate",
                method="POST",
                token=token,
            )
        except SimulationError as error:
            message = str(error)
            pair_presence_race = (
                "HTTP 409" in message
                and (
                    "同时在线" in message
                    or "companion" in message.lower()
                    or "same-day" in message.lower()
                )
            )
            if not pair_presence_race or attempt + 1 == attempts:
                raise
            time.sleep(0.25 * (attempt + 1))
    raise RuntimeError("Unreachable pair activation retry state")


def call_while_servicing_timing_socket(
    socket: websocket.WebSocket,
    callback: Callable[[], Any],
) -> Any:
    """Run blocking HTTP work while continuing timing WebSocket ping/pong."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(callback)
        while not future.done():
            wait_while_servicing_timing_socket(socket, 0.25)
        return future.result()


def connect_pair_with_heartbeat(
    base_url: str,
    study_token: str,
    study_session_id: str,
    companion_token: str,
    companion_session_id: str,
    *,
    attempts: int = 2,
) -> tuple[websocket.WebSocket, websocket.WebSocket, PairHeartbeat]:
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            study_pair, companion_pair = pair_websockets(
                base_url,
                study_token,
                study_session_id,
                companion_token,
                companion_session_id,
            )
            heartbeat = PairHeartbeat(study_pair, companion_pair)
            heartbeat.start()
            return study_pair, companion_pair, heartbeat
        except BaseException as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Pair channels could not be restored") from last_error


def fast_forward_silent_gap(
    socket: websocket.WebSocket, *, gap_seconds: float
) -> dict[str, Any]:
    transition_id = str(uuid.uuid4())
    socket.send(
        json.dumps(
            {
                "type": "simulation_gap",
                "transition_id": transition_id,
                "gap_seconds": gap_seconds,
                "client_ms": time.time() * 1000,
                "monotonic_ms": time.perf_counter() * 1000,
            }
        )
    )
    while True:
        response = receive_json(socket)
        if response.get("transition_id") != transition_id:
            continue
        if response.get("type") != "simulation_gap_ack":
            raise RuntimeError(f"Silent-gap fast-forward failed: {response}")
        return response


def send_decision_context(
    socket: websocket.WebSocket,
    *,
    conversation_target: str,
    reply_needed_probability: float,
    source: str,
    text_sha256: str,
) -> dict[str, Any]:
    transition_id = str(uuid.uuid4())
    socket.send(
        json.dumps(
            {
                "type": "decision_context",
                "transition_id": transition_id,
                "conversation_target": conversation_target,
                "reply_needed_probability": reply_needed_probability,
                "source": source,
                "text_sha256": text_sha256,
            }
        )
    )
    while True:
        response = receive_json(socket)
        if response.get("transition_id") != transition_id:
            continue
        if response.get("type") != "decision_context_ack":
            raise RuntimeError(f"Decision-context transition failed: {response}")
        return response


def load_context_predictions(path: Path) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "tap-to-adapt-reply-needed-predictions/v1":
        raise ValueError("Unsupported reply-needed prediction manifest")
    rows = {
        (int(row["participant_index"]), int(row["opportunity_index"])): dict(row)
        for row in document["rows"]
    }
    if len(rows) != len(document["rows"]):
        raise ValueError("Duplicate participant/opportunity in prediction manifest")
    return rows, document


def load_agent_replies(
    path: Path,
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema") != "tap-to-adapt-frozen-agent-replies/v1"
        or document.get("simulation_only") is not True
        or document.get("llm_provider") != "OpenRouter"
        or document.get("llm_model") != "z-ai/glm-5.2"
        or document.get("provider_order") != ["z-ai"]
        or document.get("allow_fallbacks") is not False
    ):
        raise ValueError("Unsupported frozen GLM-5.2 reply manifest")
    rows = {
        (
            int(row["participant_index"]),
            int(row["opportunity_index"]),
            str(row["source"]),
        ): dict(row)
        for row in document["rows"]
    }
    if len(rows) != len(document["rows"]):
        raise ValueError("Duplicate participant/opportunity/source in reply manifest")
    canonical_rows = json.dumps(
        document["rows"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if hashlib.sha256(canonical_rows).hexdigest() != document["rows_sha256"]:
        raise ValueError("Frozen reply manifest checksum failed")
    return rows, document


def _synthesize_schedule(
    *,
    base_url: str,
    study_token: str,
    study_session_id: str,
    companion_token: str,
    companion_session_id: str,
    roster: dict[str, str],
    events: list[dict[str, Any]],
    bundle_root: Path | None = None,
) -> tuple[dict[tuple[str, str], bytes], dict[str, Any] | None]:
    cache: dict[tuple[str, str], bytes] = {}
    for row in events:
        role = str(row["speaker_role"])
        text = str(row["text"])
        key = (role, text)
        if key in cache:
            continue
        if role == "participant":
            cache[key] = request_pcm(
                base_url,
                study_token,
                session_id=study_session_id,
                text=text,
                speaker_id=int(roster["voice_id"]),
                speed=float(roster["speed"]),
                source="participant",
            )
        elif role == "companion":
            cache[key] = request_pcm(
                base_url,
                companion_token,
                session_id=companion_session_id,
                text=text,
                speaker_id=None,
                speed=1.0,
                source="companion",
            )
        else:
            raise RuntimeError(f"Unknown simulation speaker role: {role}")
    if bundle_root is None:
        return cache, None
    participant_index = int(roster["participant_index"])
    participant_dir = bundle_root / f"participant-{participant_index:02d}"
    participant_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    for (role, text), pcm in sorted(cache.items()):
        cache_key = hashlib.sha256(
            f"{role}\0{text}".encode("utf-8")
        ).hexdigest()
        filename = f"{cache_key}.pcm"
        (participant_dir / filename).write_bytes(pcm)
        entries.append(
            {
                "cache_key": cache_key,
                "speaker_role": role,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "filename": filename,
                "pcm_bytes": len(pcm),
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            }
        )
    manifest = {
        "schema": "tap-to-adapt-frozen-pcm-participant/v1",
        "simulation_only": True,
        "participant_index": participant_index,
        "target_voice_id": int(roster["voice_id"]),
        "companion_voice": roster["companion_agent_voice_name"],
        "speed": float(roster["speed"]),
        "entries": entries,
    }
    manifest_path = participant_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cache, {
        "mode": "write",
        "participant_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "entries": len(entries),
    }


def _load_frozen_schedule(
    *,
    bundle_root: Path,
    roster: dict[str, str],
    events: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], bytes], dict[str, Any]]:
    participant_index = int(roster["participant_index"])
    participant_dir = bundle_root / f"participant-{participant_index:02d}"
    manifest_path = participant_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("schema") != "tap-to-adapt-frozen-pcm-participant/v1"
        or int(manifest["participant_index"]) != participant_index
        or int(manifest["target_voice_id"]) != int(roster["voice_id"])
        or manifest.get("companion_voice") != roster["companion_agent_voice_name"]
        or not math.isclose(
            float(manifest["speed"]),
            float(roster["speed"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            f"Frozen PCM manifest does not match participant {participant_index}"
        )
    entries = {str(row["cache_key"]): row for row in manifest["entries"]}
    cache: dict[tuple[str, str], bytes] = {}
    for row in events:
        role = str(row["speaker_role"])
        text = str(row["text"])
        key = (role, text)
        if key in cache:
            continue
        cache_key = hashlib.sha256(
            f"{role}\0{text}".encode("utf-8")
        ).hexdigest()
        entry = entries.get(cache_key)
        if (
            entry is None
            or entry.get("speaker_role") != role
            or entry.get("text_sha256")
            != hashlib.sha256(text.encode("utf-8")).hexdigest()
        ):
            raise RuntimeError(
                f"Frozen PCM entry is absent for participant {participant_index}"
            )
        pcm = (participant_dir / str(entry["filename"])).read_bytes()
        if (
            len(pcm) != int(entry["pcm_bytes"])
            or hashlib.sha256(pcm).hexdigest() != entry["pcm_sha256"]
        ):
            raise RuntimeError(
                f"Frozen PCM checksum failed for participant {participant_index}"
            )
        cache[key] = pcm
    return cache, {
        "mode": "read",
        "participant_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "entries": len(entries),
    }


def run_participant(
    *,
    base_url: str,
    roster: dict[str, str],
    participant_turns: dict[str, Any],
    clock_scale: float,
    start_stagger_seconds: float,
    pair_rotation_seconds: float,
    timing_rotation_seconds: float,
    pair_rotation_jitter_window_seconds: float,
    timing_rotation_jitter_window_seconds: float,
    roster_size: int,
    progress_lock: threading.Lock,
    pcm_bundle_read: Path | None,
    pcm_bundle_write: Path | None,
    context_predictions: dict[tuple[int, int], dict[str, Any]],
    context_manifest: dict[str, Any],
    agent_replies: dict[tuple[int, int, str], dict[str, Any]],
    agent_reply_manifest: dict[str, Any],
) -> dict[str, Any]:
    events = list(participant_turns["events"])
    max_minutes = int((int(events[-1]["elapsed_seconds"]) + 15) // 60)
    validate_schedule(events, max_minutes=max_minutes)
    profile = dict(participant_turns["profile"])
    study_login = login(base_url, roster["participant_code"], "study")
    companion_login = login(base_url, roster["participant_code"], "companion")
    study_token = str(study_login["token"])
    companion_token = str(companion_login["token"])
    bootstrap = ensure_consent(base_url, study_token)
    study_session = request_json(
        base_url, "/api/sessions/start", method="POST", token=study_token, body={"day": 1}
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
    study_pair: websocket.WebSocket | None = None
    companion_pair: websocket.WebSocket | None = None
    timing: websocket.WebSocket | None = None
    heartbeat: PairHeartbeat | None = None
    records: list[dict[str, Any]] = []
    error: BaseException | None = None
    pair_reconnects = 0
    pair_reconnect_errors: list[str] = []
    pair_reconnect_reasons: list[str] = []
    pair_connected_monotonic = 0.0
    timing_connected_monotonic = 0.0
    timing_reconnects = 0
    timing_reconnect_reasons: list[str] = []
    timing_reconnect_audits: list[dict[str, Any]] = []
    runtime_instance_id = ""
    last_model_update_count = 0
    started_wall = time.time()
    participant_offset = max(0, int(roster["participant_index"]) - 1)
    offset_denominator = max(1, roster_size - 1)
    pair_rotation_interval_seconds = (
        pair_rotation_seconds
        + pair_rotation_jitter_window_seconds
        * participant_offset
        / offset_denominator
    )
    timing_rotation_interval_seconds = (
        timing_rotation_seconds
        + timing_rotation_jitter_window_seconds
        * participant_offset
        / offset_denominator
    )
    try:
        # Public TTS fixture construction can take several minutes for 32
        # concurrent profiles.  Build it before opening the paired WebSockets;
        # otherwise reverse proxies may retire an otherwise healthy pair while
        # no interaction traffic has started.
        if pcm_bundle_read is not None:
            pcm_cache, pcm_bundle = _load_frozen_schedule(
                bundle_root=pcm_bundle_read,
                roster=roster,
                events=events,
            )
        else:
            pcm_cache, pcm_bundle = _synthesize_schedule(
                base_url=base_url,
                study_token=study_token,
                study_session_id=str(study_session["id"]),
                companion_token=companion_token,
                companion_session_id=str(companion_session["id"]),
                roster=roster,
                events=events,
                bundle_root=pcm_bundle_write,
            )
        study_pair, companion_pair, heartbeat = connect_pair_with_heartbeat(
            base_url,
            study_token,
            str(study_session["id"]),
            companion_token,
            str(companion_session["id"]),
        )
        pair_connected_monotonic = time.monotonic()

        def reconnect_pair(reason: str) -> None:
            nonlocal study_pair, companion_pair, heartbeat
            nonlocal pair_reconnects, pair_connected_monotonic
            if pair_reconnects >= 4:
                heartbeat_error = heartbeat.error if heartbeat is not None else None
                raise RuntimeError(
                    "Pair presence exceeded the reconnect safety limit"
                ) from heartbeat_error
            if heartbeat is not None and heartbeat.error is not None:
                pair_reconnect_errors.append(
                    f"{type(heartbeat.error).__name__}: {heartbeat.error}"
                )
            pair_reconnect_reasons.append(reason)
            if heartbeat is not None:
                heartbeat.close()
            for pair_socket in (study_pair, companion_pair):
                if pair_socket is not None:
                    try:
                        pair_socket.close()
                    except Exception:
                        pass
            study_pair, companion_pair, heartbeat = (
                connect_pair_with_heartbeat(
                    base_url,
                    study_token,
                    str(study_session["id"]),
                    companion_token,
                    str(companion_session["id"]),
                )
            )
            pair_connected_monotonic = time.monotonic()
            pair_reconnects += 1

        def ensure_pair_alive() -> None:
            if heartbeat is None:
                raise RuntimeError("Pair heartbeat is unavailable")
            if (
                heartbeat.error is None
                and time.monotonic() - pair_connected_monotonic
                < pair_rotation_interval_seconds
            ):
                return
            reconnect_pair(
                "heartbeat_error"
                if heartbeat.error is not None
                else "proactive_pair_rotation"
            )

        activate_after_pair_ready(
            base_url,
            study_token,
            str(study_session["id"]),
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
                    "simulation_purpose": "smooth_duration_engineering",
                },
            },
        )
        options: dict[str, Any] = {"timeout": 120, "enable_multithread": True}
        if base_url.startswith("https://"):
            options["sslopt"] = {"cert_reqs": ssl.CERT_REQUIRED}
        def open_timing_socket() -> tuple[websocket.WebSocket, dict[str, Any]]:
            opened = websocket.create_connection(
                timing_url(base_url, study_token, str(study_session["id"])),
                **options,
            )
            opened_ready = receive_json(opened)
            if (
                opened_ready.get("type") != "timing_ready"
                or opened_ready.get("mode") != "distilhubert"
            ):
                opened.close()
                raise RuntimeError(f"Unexpected timing runtime: {opened_ready}")
            opened_fingerprint = dict(
                opened_ready.get("model_fingerprint") or {}
            )
            opened_factorization = dict(
                opened_ready.get("decision_factorization") or {}
            )
            if opened_fingerprint.get("participant_disjoint_training") is not True:
                opened.close()
                raise RuntimeError(
                    "Runtime is not bound to a participant-disjoint release"
                )
            if (
                opened_factorization.get("decision_context_required") is not True
                or float(
                    opened_factorization.get(
                        "minimum_end_of_turn_silence_ms"
                    )
                    or 0
                )
                < 200
            ):
                opened.close()
                raise RuntimeError(
                    "Runtime decision factorization/EOU gate is not enabled"
                )
            return opened, opened_ready

        timing, ready = open_timing_socket()
        timing_connected_monotonic = time.monotonic()
        fingerprint = dict(ready.get("model_fingerprint") or {})
        factorization = dict(ready.get("decision_factorization") or {})
        runtime_instance_id = str(ready.get("runtime_instance_id") or "")
        if not runtime_instance_id:
            raise RuntimeError("Timing runtime lacks a stable instance ID")
        initial_head_state = dict(ready.get("model_head_state") or {})
        last_model_update_count = int(initial_head_state.get("update_count") or 0)
        phase = send_phase(timing, "learning")
        state_version = int(phase["server_state_version"])
        activation_id = str(phase.get("server_activation_id") or "")

        def reconnect_timing(reason: str) -> None:
            nonlocal timing, timing_connected_monotonic, timing_reconnects
            nonlocal state_version, activation_id
            if timing_reconnects >= 9:
                raise RuntimeError(
                    "Timing channel exceeded the reconnect safety limit"
                )
            if timing is not None:
                try:
                    timing.close()
                except Exception:
                    pass
            replacement, replacement_ready = open_timing_socket()
            replacement_fingerprint = dict(
                replacement_ready.get("model_fingerprint") or {}
            )
            replacement_instance_id = str(
                replacement_ready.get("runtime_instance_id") or ""
            )
            replacement_head_state = dict(
                replacement_ready.get("model_head_state") or {}
            )
            replacement_update_count = int(
                replacement_head_state.get("update_count") or 0
            )
            immutable_fingerprint_fields = (
                "model_family",
                "revision",
                "release_manifest_sha256",
                "shared_head_artifact_sha256",
                "shared_head_fingerprint",
                "replay_per_class",
                "participant_disjoint_training",
            )
            if any(
                replacement_fingerprint.get(field) != fingerprint.get(field)
                for field in immutable_fingerprint_fields
            ):
                replacement.close()
                raise RuntimeError(
                    "Timing reconnect changed the locked model fingerprint"
                )
            if replacement_instance_id != runtime_instance_id:
                replacement.close()
                raise RuntimeError(
                    "Timing reconnect replaced the participant runtime"
                )
            if replacement_update_count < last_model_update_count:
                replacement.close()
                raise RuntimeError(
                    "Timing reconnect rolled back the personal head"
                )
            timing = replacement
            timing_connected_monotonic = time.monotonic()
            timing_reconnects += 1
            timing_reconnect_reasons.append(reason)
            state_version = int(
                replacement_ready.get("server_state_version") or 0
            )
            activation_id = str(
                replacement_ready.get("server_activation_id") or ""
            )
            timing_reconnect_audits.append(
                {
                    "reason": reason,
                    "runtime_instance_id": replacement_instance_id,
                    "model_update_count_before": last_model_update_count,
                    "model_update_count_after": replacement_update_count,
                    "server_state_version": state_version,
                }
            )

        def ensure_timing_alive() -> None:
            if (
                time.monotonic() - timing_connected_monotonic
                >= timing_rotation_interval_seconds
            ):
                reconnect_timing("proactive_timing_rotation")
        start_delay = (int(roster["participant_index"]) - 1) * start_stagger_seconds
        if start_delay:
            wait_while_servicing_timing_socket(timing, start_delay)
        timed_start = time.monotonic()
        previous_elapsed = -15.0
        for row in events:
            scheduled = timed_start + float(row["elapsed_seconds"]) * clock_scale
            while time.monotonic() < scheduled:
                ensure_pair_alive()
                ensure_timing_alive()
                wait_while_servicing_timing_socket(
                    timing, min(0.5, scheduled - time.monotonic())
                )
            ensure_pair_alive()
            ensure_timing_alive()
            gap_seconds = max(15.0, float(row["elapsed_seconds"]) - previous_elapsed)
            fast_forward_silent_gap(timing, gap_seconds=gap_seconds)
            pcm = pcm_cache[(str(row["speaker_role"]), str(row["text"]))]
            text_sha256 = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
            prediction_key = (
                int(roster["participant_index"]),
                int(row["opportunity_index"]),
            )
            if row["phase"] == "primary":
                context_prediction = dict(context_predictions[prediction_key])
                if (
                    context_prediction["text_sha256"] != text_sha256
                    or int(context_prediction["label"]) != int(row["response_expected"])
                ):
                    raise RuntimeError(
                        f"Reply-needed prediction provenance mismatch: {prediction_key}"
                    )
                conversation_target = "experimental_agent"
                reply_needed_probability = float(context_prediction["probability"])
                context_source = str(context_manifest["source"])
            else:
                context_prediction = None
                conversation_target = "companion_agent"
                reply_needed_probability = 0.0
                context_source = "protocol_companion_window"
            responses: list[dict[str, Any]] = []
            tap_results: list[dict[str, Any]] = []
            correction_types: list[str] = []
            valid_activation: dict[str, Any] | None = None
            response_expected = bool(row["response_expected"])

            # Keep the independent context gate closed for the complete TTS
            # playback.  This explicit turn boundary is required in addition
            # to acoustic VAD because synthesized files may contain trailing
            # zero-valued PCM while playback is still in progress.
            voiced_responses = stream_timing_pcm(
                timing,
                pcm,
                receive_json,
                pace=True,
                stop_on_activation=False,
                on_pair_required=lambda: reconnect_pair(
                    "timing_pair_required_during_voiced_pcm"
                ),
            )
            responses.extend(voiced_responses)
            voiced_activation = next(
                (item for item in voiced_responses if item.get("activate") is True),
                None,
            )
            if voiced_activation is not None:
                raise RuntimeError(
                    "EOU invariant violated: activation occurred before TTS ended"
                )

            context_ack = send_decision_context(
                timing,
                conversation_target=conversation_target,
                reply_needed_probability=reply_needed_probability,
                source=context_source,
                text_sha256=text_sha256,
            )
            context_allows_reply = bool(
                context_ack["context_allows_activation"]
            )

            deadline_seconds = float(row["reply_deadline_seconds"])
            silence_frames_planned = max(5, math.ceil(deadline_seconds / 0.2))
            silence_pcm = bytes(silence_frames_planned * TIMING_FRAME_BYTES)
            silence_responses = stream_timing_pcm(
                timing,
                silence_pcm,
                receive_json,
                pace=True,
                stop_on_activation=True,
                on_pair_required=lambda: reconnect_pair(
                    "timing_pair_required_during_eou_silence"
                ),
            )
            responses.extend(silence_responses)
            silence_activation = next(
                (item for item in silence_responses if item.get("activate") is True),
                None,
            )
            silence_activation_frame = next(
                (
                    index + 1
                    for index, item in enumerate(silence_responses)
                    if item.get("activate") is True
                ),
                None,
            )
            if silence_activation is not None:
                state_version = int(
                    silence_activation["activation_state_version"]
                )
                activation_id = str(silence_activation["activation_id"])
                if response_expected:
                    valid_activation = silence_activation
                else:
                    tap_result = send_conditional_tap(
                        timing,
                        intended_action="interrupt",
                        state="thinking",
                        activation_id=activation_id,
                        state_version=state_version,
                        feedback_attribution="reply_needed_context",
                    )
                    if tap_result.get("accepted") is False:
                        raise RuntimeError(
                            f"Conditional interrupt was cancelled: {tap_result}"
                        )
                    if (
                        tap_result.get("training_eligible") is not False
                        or tap_result.get("training_applied") is not False
                        or tap_result.get("quarantine_reason")
                        != "feedback_attributed_to_context_gate"
                    ):
                        raise RuntimeError(
                            f"Context interrupt leaked into acoustic training: {tap_result}"
                        )
                    tap_results.append(tap_result)
                    correction_types.append("interrupt")
                    state_version = int(tap_result["server_state_version"])
                    activation_id = str(
                        tap_result.get("server_activation_id") or ""
                    )
                    # The simulated user retracts the erroneous reply-needed
                    # decision immediately.  This prevents the remaining
                    # end-of-turn silence from causing a second activation,
                    # while preserving the protocol requirement that the full
                    # planned 1--3 s silence tail is physically streamed.
                    send_decision_context(
                        timing,
                        conversation_target=conversation_target,
                        reply_needed_probability=0.0,
                        source="context_correction_applied",
                        text_sha256=text_sha256,
                    )

            remaining_silence_frames = (
                silence_frames_planned - len(silence_responses)
            )
            if remaining_silence_frames < 0:
                raise RuntimeError("Sent more silence frames than planned")
            if remaining_silence_frames:
                remaining_silence_responses = stream_timing_pcm(
                    timing,
                    bytes(remaining_silence_frames * TIMING_FRAME_BYTES),
                    receive_json,
                    pace=True,
                    stop_on_activation=False,
                    on_pair_required=lambda: reconnect_pair(
                        "timing_pair_required_during_remaining_silence"
                    ),
                )
                silence_responses.extend(remaining_silence_responses)
                responses.extend(remaining_silence_responses)
            post_speech_frames_sent = len(silence_responses)
            if post_speech_frames_sent != silence_frames_planned:
                raise RuntimeError(
                    "The full planned end-of-turn silence tail was not streamed"
                )

            eligible = [
                item for item in responses if item.get("decision_eligible") is True
            ]
            final_prediction = eligible[-1] if eligible else responses[-1]
            total_pcm_frames = math.ceil(len(pcm) / TIMING_FRAME_BYTES)
            label = int(row["label"])

            if response_expected and valid_activation is None:
                feedback_attribution = (
                    "acoustic_personalizer"
                    if context_allows_reply
                    else "reply_needed_context"
                )
                tap_result = send_conditional_tap(
                    timing,
                    intended_action="manual_activate",
                    state="waiting",
                    activation_id=activation_id,
                    state_version=state_version,
                    feedback_attribution=feedback_attribution,
                )
                if tap_result.get("accepted") is False:
                    raise RuntimeError(
                        f"Conditional manual activation was cancelled: {tap_result}"
                    )
                if feedback_attribution == "acoustic_personalizer":
                    if (
                        tap_result.get("training_eligible") is not True
                        or tap_result.get("training_applied") is not True
                    ):
                        raise RuntimeError(
                            f"Eligible acoustic activation was not learned: {tap_result}"
                        )
                elif (
                    tap_result.get("training_eligible") is not False
                    or tap_result.get("training_applied") is not False
                    or tap_result.get("quarantine_reason")
                    != "feedback_attributed_to_context_gate"
                ):
                    raise RuntimeError(
                        f"Context activation leaked into acoustic training: {tap_result}"
                    )
                tap_results.append(tap_result)
                correction_types.append("manual_activate")
                state_version = int(tap_result["server_state_version"])
                activation_id = str(tap_result.get("server_activation_id") or "")
            if len(set(correction_types)) > 1:
                raise RuntimeError(
                    "Conflicting positive/negative feedback in one opportunity"
                )

            cycle = int(row["cycle_index"])
            if row["phase"] == "primary" and response_expected:
                frozen_reply = dict(
                    agent_replies[
                        (
                            int(roster["participant_index"]),
                            int(row["opportunity_index"]),
                            "primary",
                        )
                    ]
                )
                if frozen_reply["transcript_sha256"] != text_sha256:
                    raise RuntimeError("Frozen primary reply provenance mismatch")

                def request_primary_reply() -> None:
                    request_bytes(
                        base_url,
                        "/api/tts",
                        token=study_token,
                        body={
                            "session_id": study_session["id"],
                            "text": str(frozen_reply["reply"]),
                        },
                    )

                call_while_servicing_timing_socket(timing, request_primary_reply)
            elif row["category"] == "participant_to_companion":
                frozen_reply = dict(
                    agent_replies[
                        (
                            int(roster["participant_index"]),
                            int(row["opportunity_index"]),
                            "companion",
                        )
                    ]
                )
                if frozen_reply["transcript_sha256"] != text_sha256:
                    raise RuntimeError("Frozen companion reply provenance mismatch")

                def request_companion_reply() -> None:
                    request_bytes(
                        base_url,
                        "/api/tts",
                        token=companion_token,
                        body={
                            "session_id": companion_session["id"],
                            "text": str(frozen_reply["reply"]),
                        },
                    )

                call_while_servicing_timing_socket(
                    timing, request_companion_reply
                )

            if valid_activation is not None or any(
                result.get("action") == "manual_activate"
                for result in tap_results
            ):
                state = send_state(timing, "waiting")
                state_version = int(state["server_state_version"])
                activation_id = str(state.get("server_activation_id") or "")

            comparison = dict(final_prediction.get("model_head_comparison") or {})
            final_tap = tap_results[-1] if tap_results else None
            head_after_opportunity = dict(
                (final_tap or {}).get(
                    "model_head_comparison_after_update", comparison
                )
                or {}
            )
            last_model_update_count = max(
                last_model_update_count,
                int(head_after_opportunity.get("update_count") or 0),
            )
            record = {
                "schema": "tap-to-adapt-factorized-eou-opportunity/v4",
                "simulation": True,
                "participant_id": roster["participant_id"],
                "participant_index": int(roster["participant_index"]),
                "profile_id": profile["profile_id"],
                "voice_id": int(roster["voice_id"]),
                "companion_voice": roster["companion_agent_voice_name"],
                "session_id": study_session["id"],
                "opportunity_index": int(row["opportunity_index"]),
                "cycle_index": cycle,
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "bin_6m": cycle,
                "phase": row["phase"],
                "category": row["category"],
                "speaker_role": row["speaker_role"],
                "interaction_intent": row["interaction_intent"],
                "response_expected": response_expected,
                "reply_deadline_seconds": float(
                    row["reply_deadline_seconds"]
                ),
                "difficulty_band": row["difficulty_band"],
                "text_length": int(row["text_length"]),
                "text_sha256": text_sha256,
                "label": label,
                "conversation_target": conversation_target,
                "reply_needed_probability": reply_needed_probability,
                "reply_needed_prediction": context_allows_reply,
                "reply_needed_model_sha256": context_manifest.get("model_sha256"),
                "reply_needed_threshold": float(context_manifest["threshold"]),
                "decision_context_source": context_source,
                "decision_context_version": int(
                    context_ack["decision_context_version"]
                ),
                "corrected": bool(correction_types),
                "correction_count": len(correction_types),
                "correction_types": correction_types,
                "manual_activate_count": correction_types.count(
                    "manual_activate"
                ),
                "manual_interrupt_count": correction_types.count("interrupt"),
                "correction_type": (
                    correction_types[0]
                    if len(correction_types) == 1
                    else "+".join(correction_types) or None
                ),
                "activated": valid_activation is not None,
                "any_activation": bool(
                    silence_activation is not None
                ),
                "valid_response_activation": valid_activation is not None,
                "premature_activation_count": 0,
                "eou_gate_violation": False,
                "minimum_end_of_turn_silence_ms": float(
                    factorization["minimum_end_of_turn_silence_ms"]
                ),
                "post_speech_silence_frame_ms": 200,
                "post_speech_silence_frames_planned": silence_frames_planned,
                "post_speech_silence_frames_sent": post_speech_frames_sent,
                "post_speech_silence_ms_sent": post_speech_frames_sent * 200,
                "first_valid_response_latency_ms": (
                    float(silence_activation_frame * 200)
                    if valid_activation is not None
                    and silence_activation_frame is not None
                    else None
                ),
                "opposite_label_conflict": False,
                "feedback_attributions": [
                    result.get("feedback_attribution") for result in tap_results
                ],
                "prediction_frames_streamed": len(responses),
                "total_pcm_frames": total_pcm_frames,
                "speech_truncated_on_activation": False,
                "probability": final_prediction.get("probability"),
                "shared_probability": comparison.get("shared_probability"),
                "personal_probability": comparison.get("personal_probability"),
                "full_utterance_shared_error_propensity": row.get(
                    "calibration_shared_error_propensity"
                ),
                "personal_update_count": dict(
                    (final_tap or {}).get(
                        "model_head_comparison_after_update", comparison
                    )
                    or {}
                ).get("update_count"),
                "acoustic_training_applied": (
                    all(
                        result.get("training_applied") is True
                        for result in tap_results
                        if result.get("feedback_attribution")
                        == "acoustic_personalizer"
                    )
                    if any(
                        result.get("feedback_attribution")
                        == "acoustic_personalizer"
                        for result in tap_results
                    )
                    else False
                ),
                "correction_delay_ms": int(profile["correction_delay_ms"]),
                "llm_generated_text": True,
                "agent_reply_mode": "frozen_remote_recorded",
                "agent_reply_manifest_rows_sha256": agent_reply_manifest[
                    "rows_sha256"
                ],
                "agent_reply_sha256": (
                    frozen_reply["reply_sha256"]
                    if (
                        (row["phase"] == "primary" and response_expected)
                        or row["category"] == "participant_to_companion"
                    )
                    else None
                ),
                "participant_tts_provider": "sherpa-onnx",
                "participant_tts_model": "vits-icefall-zh-aishell3",
                "experimental_agent_tts_provider": "sherpa-onnx-local",
                "experimental_agent_voice": roster[
                    "experimental_agent_voice_name"
                ],
                "companion_agent_tts_provider": "sherpa-onnx-local",
                "companion_agent_voice": roster["companion_agent_voice_name"],
                "pair_reconnect_count_so_far": pair_reconnects,
                "timing_reconnect_count_so_far": timing_reconnects,
            }
            append_event(base_url, study_token, str(study_session["id"]), record)
            records.append(record)
            previous_elapsed = float(row["elapsed_seconds"])
            if len(records) == 1 or len(records) % 24 == 0:
                with progress_lock:
                    print(
                        f"P{roster['participant_index']}: "
                        f"{float(row['elapsed_seconds']) / 60:.1f}m, "
                        f"opportunities={len(records)}, "
                        f"corrections={sum(int(item['correction_count']) for item in records)}",
                        flush=True,
                    )
        send_phase(timing, "complete")
    except BaseException as caught:
        error = caught
        raise
    finally:
        if heartbeat is not None:
            heartbeat.close()
        for socket in (timing, study_pair, companion_pair):
            if socket is not None:
                try:
                    socket.close()
                except Exception:
                    pass
        try:
            request_json(
                base_url,
                f"/api/sessions/{study_session['id']}/abort",
                method="POST",
                token=study_token,
            )
        except Exception:
            if error is None:
                raise
    return {
        "participant_id": roster["participant_id"],
        "participant_index": int(roster["participant_index"]),
        "profile": profile,
        "voice_id": int(roster["voice_id"]),
        "companion_voice": roster["companion_agent_voice_name"],
        "model_fingerprint": fingerprint,
        "session_id": study_session["id"],
        "wall_elapsed_seconds": time.time() - started_wall,
        "pair_reconnect_count": pair_reconnects,
        "pair_reconnect_errors": pair_reconnect_errors,
        "pair_reconnect_reasons": pair_reconnect_reasons,
        "pair_rotation_interval_seconds": pair_rotation_interval_seconds,
        "timing_reconnect_count": timing_reconnects,
        "timing_reconnect_reasons": timing_reconnect_reasons,
        "timing_reconnect_audits": timing_reconnect_audits,
        "timing_rotation_interval_seconds": timing_rotation_interval_seconds,
        "runtime_instance_id": runtime_instance_id,
        "pcm_bundle": pcm_bundle,
        "opportunities": len(records),
        "corrected_opportunities": sum(
            int(item["corrected"]) for item in records
        ),
        "corrections": sum(
            int(item["correction_count"]) for item in records
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked smooth 4+2 public-route duration simulation."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--context-predictions", type=Path, required=True)
    parser.add_argument("--agent-replies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--start-stagger-seconds", type=float, default=1.0)
    parser.add_argument(
        "--pair-rotation-seconds",
        type=float,
        default=20 * 60,
        help=(
            "Proactively refresh public pair-presence sockets. Values below "
            "600 seconds are restricted to time-compressed smoke QA."
        ),
    )
    parser.add_argument(
        "--timing-rotation-seconds",
        type=float,
        default=8 * 60,
        help=(
            "Proactively refresh the timing WebSocket at an opportunity "
            "boundary while retaining the server-side participant runtime."
        ),
    )
    parser.add_argument(
        "--pair-rotation-jitter-window-seconds",
        type=float,
        default=90.0,
        help=(
            "Deterministically spread pair rotations across profiles over "
            "this additional window to prevent a reconnect herd."
        ),
    )
    parser.add_argument(
        "--timing-rotation-jitter-window-seconds",
        type=float,
        default=45.0,
        help=(
            "Deterministically spread timing rotations across profiles over "
            "this additional window while retaining each personal runtime."
        ),
    )
    parser.add_argument("--clock-scale", type=float, default=1.0)
    parser.add_argument("--confirm-simulation", action="store_true")
    parser.add_argument("--allow-time-compression-for-smoke-only", action="store_true")
    parser.add_argument(
        "--smoke-cycles",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Run only one or two complete 4+2 cycles for compressed QA. "
            "Zero keeps the full schedule."
        ),
    )
    pcm_group = parser.add_mutually_exclusive_group()
    pcm_group.add_argument("--read-pcm-bundle", type=Path)
    pcm_group.add_argument("--write-pcm-bundle", type=Path)
    args = parser.parse_args()
    if not args.confirm_simulation:
        raise SystemExit("Pass --confirm-simulation")
    if args.clock_scale != 1.0 and not args.allow_time_compression_for_smoke_only:
        raise SystemExit("Duration evidence requires --clock-scale 1")
    if args.smoke_cycles and (
        args.clock_scale == 1.0
        or not args.allow_time_compression_for_smoke_only
    ):
        raise SystemExit("--smoke-cycles is restricted to time-compressed QA")
    if not 0.02 <= args.clock_scale <= 1.0:
        raise SystemExit("--clock-scale must be in [0.02, 1]")
    if not 0 <= args.start_stagger_seconds <= 10:
        raise SystemExit("--start-stagger-seconds must be in [0, 10]")
    if not 2 <= args.pair_rotation_seconds <= 3600:
        raise SystemExit("--pair-rotation-seconds must be in [2, 3600]")
    if (
        args.pair_rotation_seconds < 600
        and not args.allow_time_compression_for_smoke_only
    ):
        raise SystemExit(
            "Short pair rotations are restricted to time-compressed smoke QA"
        )
    if not 2 <= args.timing_rotation_seconds <= 3600:
        raise SystemExit("--timing-rotation-seconds must be in [2, 3600]")
    if (
        args.timing_rotation_seconds < 300
        and not args.allow_time_compression_for_smoke_only
    ):
        raise SystemExit(
            "Short timing rotations are restricted to time-compressed smoke QA"
        )
    if not 0 <= args.pair_rotation_jitter_window_seconds <= 300:
        raise SystemExit(
            "--pair-rotation-jitter-window-seconds must be in [0, 300]"
        )
    if not 0 <= args.timing_rotation_jitter_window_seconds <= 300:
        raise SystemExit(
            "--timing-rotation-jitter-window-seconds must be in [0, 300]"
        )
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    if args.read_pcm_bundle is not None and not args.read_pcm_bundle.is_dir():
        raise SystemExit("--read-pcm-bundle must be an existing directory")
    if args.write_pcm_bundle is not None:
        if args.write_pcm_bundle.exists():
            raise SystemExit("Refusing to overwrite --write-pcm-bundle")
        args.write_pcm_bundle.mkdir(parents=True)
    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    turns = json.loads(args.turns.read_text(encoding="utf-8"))
    context_predictions, context_manifest = load_context_predictions(
        args.context_predictions
    )
    agent_replies, agent_reply_manifest = load_agent_replies(
        args.agent_replies
    )
    if turns.get("schema") != SCHEMA:
        raise SystemExit("Turns do not use the smooth 4+2 schema")
    by_id = {}
    for item in turns["participants"]:
        scoped = item
        if args.smoke_cycles:
            scoped = {
                **item,
                "events": item["events"][: args.smoke_cycles * 12],
            }
        by_id[str(item["participant_id"])] = scoped
    if {row["participant_id"] for row in roster} != set(by_id):
        raise SystemExit("Roster and turns participant IDs do not match")
    expected_context_keys = {
        (int(row["participant_index"]), int(event["opportunity_index"]))
        for row in roster
        for event in by_id[row["participant_id"]]["events"]
        if event["phase"] == "primary"
    }
    prediction_keys = set(context_predictions)
    if not expected_context_keys.issubset(prediction_keys):
        missing = len(expected_context_keys - prediction_keys)
        extra = len(prediction_keys - expected_context_keys)
        raise SystemExit(
            f"Context predictions do not match primary opportunities: missing={missing}, extra={extra}"
        )
    expected_agent_reply_keys = {
        (
            int(row["participant_index"]),
            int(event["opportunity_index"]),
            (
                "primary"
                if event["phase"] == "primary"
                and bool(event["response_expected"])
                else "companion"
            ),
        )
        for row in roster
        for event in by_id[row["participant_id"]]["events"]
        if (
            event["phase"] == "primary"
            and bool(event["response_expected"])
        )
        or event["category"] == "participant_to_companion"
    }
    available_agent_reply_keys = set(agent_replies)
    if not expected_agent_reply_keys.issubset(available_agent_reply_keys):
        raise SystemExit(
            "Frozen agent reply manifest does not cover the schedule"
        )
    if not args.smoke_cycles and available_agent_reply_keys != expected_agent_reply_keys:
        raise SystemExit(
            "Full duration run requires an exact frozen agent reply manifest"
        )
    run_slugs = {row["run_slug"] for row in roster}
    if (
        len(run_slugs) != 1
        or RUN_SLUG_PATTERN.fullmatch(next(iter(run_slugs))) is None
        or run_slugs != {turns["run_slug"]}
    ):
        raise SystemExit("Roster and turns are outside the same smooth run")
    if any(
        row.get("data_origin") != "simulation"
        or row.get("analysis_use") != "smooth_4plus2_engineering_only"
        for row in roster
    ):
        raise SystemExit("Refusing accounts outside the smooth simulation fixture")

    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(roster)))) as pool:
        futures = {
            pool.submit(
                run_participant,
                base_url=args.base_url.rstrip("/"),
                roster=row,
                participant_turns=by_id[row["participant_id"]],
                clock_scale=args.clock_scale,
                start_stagger_seconds=args.start_stagger_seconds,
                pair_rotation_seconds=args.pair_rotation_seconds,
                timing_rotation_seconds=args.timing_rotation_seconds,
                pair_rotation_jitter_window_seconds=(
                    args.pair_rotation_jitter_window_seconds
                ),
                timing_rotation_jitter_window_seconds=(
                    args.timing_rotation_jitter_window_seconds
                ),
                roster_size=len(roster),
                progress_lock=lock,
                pcm_bundle_read=args.read_pcm_bundle,
                pcm_bundle_write=args.write_pcm_bundle,
                context_predictions=context_predictions,
                context_manifest=context_manifest,
                agent_replies=agent_replies,
                agent_reply_manifest=agent_reply_manifest,
            ): row
            for row in roster
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as caught:
                failure = {
                    "participant_index": row["participant_index"],
                    "error": f"{type(caught).__name__}: {caught}",
                    "traceback": "".join(
                        traceback.format_exception(
                            type(caught), caught, caught.__traceback__
                        )
                    ),
                }
                failures.append(failure)
                with lock:
                    print(
                        f"P{row['participant_index']}: FAILED {failure['error']}",
                        flush=True,
                    )
                    print(failure["traceback"], flush=True)
    pcm_manifests = [
        {
            "participant_index": int(item["participant_index"]),
            **dict(item["pcm_bundle"]),
        }
        for item in sorted(results, key=lambda item: item["participant_index"])
        if item.get("pcm_bundle") is not None
    ]
    pcm_manifest_identity = [
        {
            "participant_index": row["participant_index"],
            "participant_manifest_sha256": row["participant_manifest_sha256"],
            "entries": row["entries"],
        }
        for row in pcm_manifests
    ]
    pcm_manifest_digest = (
        hashlib.sha256(
            json.dumps(
                pcm_manifest_identity,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if pcm_manifests
        else None
    )
    report = {
        "schema": "tap-to-adapt-factorized-eou-public-run/v4",
        "simulation_only": True,
        "duration_evidence_valid": (
            args.clock_scale == 1.0
            and args.smoke_cycles == 0
            and turns["cohort"] == "duration32"
            and len(roster) == 32
            and args.read_pcm_bundle is not None
            and all(
                not record["eou_gate_violation"]
                and not record["opposite_label_conflict"]
                and record["post_speech_silence_ms_sent"] >= 1000
                for item in results
                for record in item["records"]
            )
        ),
        "run_slug": next(iter(run_slugs)),
        "cohort": turns["cohort"],
        "base_url": args.base_url.rstrip("/"),
        "llm_provider": turns["llm_provider"],
        "llm_model": turns["llm_model"],
        "llm_execution_mode": "frozen_remote_recorded_replay",
        "agent_reply_manifest": {
            "source_run_slug": agent_reply_manifest["source_run_slug"],
            "rows": len(agent_reply_manifest["rows"]),
            "rows_sha256": agent_reply_manifest["rows_sha256"],
            "provider_order": agent_reply_manifest["provider_order"],
            "allow_fallbacks": agent_reply_manifest["allow_fallbacks"],
        },
        "source_pool_sha256": turns["source_pool_sha256"],
        "tts_provider": "sherpa-onnx",
        "tts_model": "vits-icefall-zh-aishell3",
        "model": "runtime locked by release manifest",
        "protocol": {
            "max_minutes": turns["max_minutes"],
            "cycle_minutes": 6,
            "primary_minutes": 4,
            "companion_minutes": 2,
            "opportunity_seconds": 30,
            "smoke_cycles": args.smoke_cycles,
            "opportunities_per_profile": (
                args.smoke_cycles * 12
                if args.smoke_cycles
                else int(turns["max_minutes"]) * 2
            ),
            "difficulty_control": turns["difficulty_control"],
            "target_voice_count": len({row["voice_id"] for row in roster}),
            "companion_voice_count": 1,
            "companion_voice": roster[0]["companion_agent_voice_name"],
            "frozen_pcm_bundle_required": True,
            "frozen_pcm_bundle_used": args.read_pcm_bundle is not None,
            "stimulus_difficulty_score_mode": (
                "full_utterance_shared_only_no_truncation"
            ),
            "shared_score_map_sha256": dict(
                turns.get("voice_calibration") or {}
            ).get("shared_score_map_sha256"),
            "end_of_turn_gate": {
                "minimum_silence_ms": 400,
                "post_tts_silence_frame_ms": 200,
                "post_tts_silence_deadline_seconds": "per-opportunity 1-3",
            },
            "decision_factorization": {
                "speaker_personalisation_head": "distilhubert_replay_head",
                "addressee_gate": "protocol_target",
                "reply_needed_context_manifest": str(
                    args.context_predictions.resolve()
                ),
                "reply_needed_context_source": context_manifest["source"],
                "reply_needed_threshold": float(context_manifest["threshold"]),
                "reply_needed_model_sha256": context_manifest.get("model_sha256"),
            },
            "one_label_per_opportunity_enforced": True,
            "pair_rotation_seconds": args.pair_rotation_seconds,
            "pair_rotation_jitter_window_seconds": (
                args.pair_rotation_jitter_window_seconds
            ),
            "timing_rotation_seconds": args.timing_rotation_seconds,
            "timing_rotation_jitter_window_seconds": (
                args.timing_rotation_jitter_window_seconds
            ),
        },
        "clock_scale": args.clock_scale,
        "frozen_pcm_bundle": {
            "mode": (
                "read"
                if args.read_pcm_bundle is not None
                else "write"
                if args.write_pcm_bundle is not None
                else "none"
            ),
            "participant_manifests": pcm_manifests,
            "aggregate_manifest_sha256": pcm_manifest_digest,
        },
        "started_ms": int(started * 1000),
        "finished_ms": int(time.time() * 1000),
        "participants_expected": len(roster),
        "participants_succeeded": len(results),
        "pair_reconnects_total": sum(
            int(item.get("pair_reconnect_count", 0)) for item in results
        ),
        "profiles_with_pair_reconnect": sum(
            int(item.get("pair_reconnect_count", 0)) > 0 for item in results
        ),
        "pair_reconnect_reason_counts": dict(
            Counter(
                reason
                for item in results
                for reason in item.get("pair_reconnect_reasons", [])
            )
        ),
        "timing_reconnects_total": sum(
            int(item.get("timing_reconnect_count", 0)) for item in results
        ),
        "profiles_with_timing_reconnect": sum(
            int(item.get("timing_reconnect_count", 0)) > 0
            for item in results
        ),
        "timing_reconnect_reason_counts": dict(
            Counter(
                reason
                for item in results
                for reason in item.get("timing_reconnect_reasons", [])
            )
        ),
        "failures": failures,
        "results": sorted(results, key=lambda item: item["participant_index"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
