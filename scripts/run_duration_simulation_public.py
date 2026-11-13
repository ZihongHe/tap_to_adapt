"""Run participant-isolated, real-public-route duration simulation.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import ssl
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    from .smoke_negative_feedback_public import receive_json, submit_quality
    from .voice_pcm import stream_timing_pcm, validate_voiced_pcm
except ImportError:
    from simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        submit_baseline,
    )
    from smoke_negative_feedback_public import receive_json, submit_quality
    from voice_pcm import stream_timing_pcm, validate_voiced_pcm


def timing_url(base_url: str, token: str, session_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    root = urllib.parse.urlunsplit((scheme, parsed.netloc, parsed.path, "", ""))
    query = urllib.parse.urlencode({"token": token, "session_id": session_id})
    return f"{root}/ws/timing?{query}"


def wait_while_servicing_timing_socket(
    socket: websocket.WebSocket, seconds: float
) -> None:
    """Keep browser-equivalent WebSocket ping/pong handling active while idle."""

    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    previous_timeout = socket.gettimeout()
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            socket.settimeout(min(0.5, remaining))
            try:
                raw = socket.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if raw in {"", b""}:
                raise RuntimeError("Timing WebSocket closed during an idle interval")
            message = json.loads(raw)
            raise RuntimeError(
                "Unexpected timing message during an idle interval: "
                f"{message.get('type')}"
            )
    finally:
        socket.settimeout(previous_timeout)


def request_pcm(
    base_url: str,
    token: str,
    *,
    session_id: str,
    text: str,
    speaker_id: int | None,
    speed: float,
    source: str = "participant",
) -> bytes:
    payload = json.dumps(
        {
            "session_id": session_id,
            "text": text,
            "speaker_id": speaker_id,
            "speed": speed,
            "source": source,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/simulation/tts-pcm",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/octet-stream",
            "User-Agent": "TTA-Duration-Simulator/1.0",
        },
    )
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                audio = response.read()
                content_type = response.headers.get_content_type()
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise RuntimeError(f"TTS PCM HTTP {error.code}: {detail}") from None
            last_error = error
        except urllib.error.URLError as error:
            if attempt == 3:
                raise RuntimeError(f"TTS PCM transport failure: {error}") from None
            last_error = error
        time.sleep(float(attempt))
    else:
        raise RuntimeError(f"TTS PCM failed after retries: {last_error}")
    if content_type != "application/octet-stream":
        raise RuntimeError(f"Unexpected TTS PCM content type: {content_type}")
    validate_voiced_pcm(audio)
    return audio


class PairHeartbeat:
    def __init__(self, *sockets: websocket.WebSocket):
        self.sockets = sockets
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.sender = threading.Thread(target=self._send_loop, daemon=True)
        self.readers = [
            threading.Thread(target=self._read_loop, args=(socket,), daemon=True)
            for socket in sockets
        ]

    def _record_error(self, error: BaseException) -> None:
        if not self.stop.is_set() and self.error is None:
            self.error = error
        self.stop.set()

    def _send_loop(self) -> None:
        sequence = 0
        while not self.stop.is_set():
            sequence += 1
            payload = json.dumps(
                {
                    "type": "heartbeat",
                    "seq": sequence,
                    "client_sent_ms": int(time.time() * 1000),
                }
            )
            try:
                for socket in self.sockets:
                    socket.send(payload)
            except BaseException as error:
                self._record_error(error)
                return
            self.stop.wait(4.0)

    def _read_loop(self, socket: websocket.WebSocket) -> None:
        socket.settimeout(1.0)
        while not self.stop.is_set():
            try:
                raw = socket.recv()
                if raw in {"", b""}:
                    raise RuntimeError("Pair WebSocket closed")
            except websocket.WebSocketTimeoutException:
                continue
            except BaseException as error:
                self._record_error(error)
                return

    def start(self) -> None:
        for reader in self.readers:
            reader.start()
        self.sender.start()

    def close(self) -> None:
        self.stop.set()
        self.sender.join(timeout=5)
        for reader in self.readers:
            reader.join(timeout=5)


def send_phase(socket: websocket.WebSocket, phase: str) -> dict[str, Any]:
    transition_id = str(uuid.uuid4())
    socket.send(
        json.dumps(
            {
                "type": "phase",
                "phase": phase,
                "policy_token": "A",
                "drain_active": True,
                "transition_id": transition_id,
            }
        )
    )
    while True:
        response = receive_json(socket)
        if response.get("transition_id") == transition_id:
            if response.get("type") != "phase_ack":
                raise RuntimeError(f"Phase transition failed: {response}")
            return response


def send_state(socket: websocket.WebSocket, state: str) -> dict[str, Any]:
    transition_id = str(uuid.uuid4())
    socket.send(
        json.dumps(
            {"type": "state", "state": state, "transition_id": transition_id}
        )
    )
    while True:
        response = receive_json(socket)
        if response.get("transition_id") == transition_id:
            if response.get("type") != "state_ack":
                raise RuntimeError(f"State transition failed: {response}")
            return response


def send_conditional_tap(
    socket: websocket.WebSocket,
    *,
    intended_action: str,
    state: str,
    activation_id: str,
    state_version: int,
    feedback_attribution: str = "acoustic_personalizer",
) -> dict[str, Any]:
    tap_id = str(uuid.uuid4())
    socket.send(
        json.dumps(
            {
                "type": "tap",
                "tap_id": tap_id,
                "pointer_agent_state": state,
                "client_ms": time.time() * 1000,
                "monotonic_ms": time.perf_counter() * 1000,
                "visible": True,
                "intended_action": intended_action,
                "expected_state": state,
                "expected_activation_id": activation_id,
                "expected_state_version": state_version,
                "feedback_attribution": feedback_attribution,
            }
        )
    )
    for _ in range(20):
        response = receive_json(socket)
        if response.get("type") == "tap_result" and response.get("tap_id") == tap_id:
            return response
    raise RuntimeError("No matching tap_result")


def append_opportunity_event(
    base_url: str,
    token: str,
    session_id: str,
    record: dict[str, Any],
) -> None:
    request_json(
        base_url,
        "/api/events",
        method="POST",
        token=token,
        body={
            "session_id": session_id,
            "event_type": "simulation.duration_opportunity",
            "client_ms": int(time.time() * 1000),
            "monotonic_ms": time.perf_counter() * 1000,
            "payload": record,
        },
    )


def run_participant(
    *,
    base_url: str,
    roster: dict[str, str],
    turns: dict[str, Any],
    max_minutes: float,
    opportunity_seconds: float,
    clock_scale: float,
    start_stagger_seconds: float,
    seed: int,
    progress_lock: threading.Lock,
) -> dict[str, Any]:
    rng = random.Random(seed)
    study_login = login(base_url, roster["participant_code"], "study")
    companion_login = login(base_url, roster["participant_code"], "companion")
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
    study_pair: websocket.WebSocket | None = None
    companion_pair: websocket.WebSocket | None = None
    timing: websocket.WebSocket | None = None
    heartbeat: PairHeartbeat | None = None
    records: list[dict[str, Any]] = []
    error: BaseException | None = None
    started_wall = time.time()
    try:
        study_pair, companion_pair = pair_websockets(
            base_url,
            study_token,
            str(study_session["id"]),
            companion_token,
            str(companion_session["id"]),
        )
        heartbeat = PairHeartbeat(study_pair, companion_pair)
        heartbeat.start()
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
                    "simulation_purpose": "duration_engineering",
                },
            },
        )
        # Synthesis is intentionally outside the measured duration and before
        # the timing WebSocket is opened.  Keeping an otherwise idle timing
        # socket open during a multi-voice cache fill can trigger the public
        # proxy's idle timeout.
        target_texts = list(turns["target_user_turns"])
        background_texts = list(turns["background_speaker_turns"])
        target_pcm = [
            request_pcm(
                base_url,
                study_token,
                session_id=str(study_session["id"]),
                text=text,
                speaker_id=int(roster["voice_id"]),
                speed=float(roster["speed"]),
            )
            for text in target_texts
        ]
        background_pcm = [
            request_pcm(
                base_url,
                study_token,
                session_id=str(study_session["id"]),
                text=text,
                speaker_id=int(roster["background_voice_id"]),
                speed=float(roster["speed"]),
            )
            for text in background_texts
        ]
        if heartbeat.error is not None:
            raise RuntimeError("Pair heartbeat failed during TTS cache fill") from heartbeat.error
        options: dict[str, Any] = {"timeout": 120, "enable_multithread": True}
        if base_url.startswith("https://"):
            options["sslopt"] = {"cert_reqs": ssl.CERT_REQUIRED}
        timing = websocket.create_connection(
            timing_url(base_url, study_token, str(study_session["id"])), **options
        )
        ready = receive_json(timing)
        if ready.get("type") != "timing_ready" or ready.get("mode") != "distilhubert":
            raise RuntimeError(f"Unexpected timing runtime: {ready}")
        fingerprint = dict(ready.get("model_fingerprint") or {})
        if (
            fingerprint.get("participant_disjoint_training") is not True
            or int(fingerprint.get("replay_per_class") or 0) <= 0
        ):
            raise RuntimeError("Timing runtime is not the locked participant-disjoint replay release")
        phase = send_phase(timing, "learning")
        state_version = int(phase["server_state_version"])
        activation_id = str(phase.get("server_activation_id") or "")
        # Thread workers otherwise reach their first audio chunk almost
        # simultaneously, an artificial synchronization that real participants
        # do not exhibit.  Deterministically spread starts while keeping each
        # participant's measured clock at real-time scale.
        start_delay = (int(roster["participant_index"]) - 1) * start_stagger_seconds
        if start_delay > 0:
            wait_while_servicing_timing_socket(timing, start_delay)
        timed_start = time.monotonic()
        next_opportunity = timed_start
        index = 0
        llm_reply_count = 0
        while (time.monotonic() - timed_start) / 60.0 < max_minutes * clock_scale:
            if heartbeat.error is not None:
                raise RuntimeError("Pair heartbeat failed") from heartbeat.error
            now = time.monotonic()
            if now < next_opportunity:
                wait_while_servicing_timing_socket(
                    timing, min(0.5, next_opportunity - now)
                )
                continue
            elapsed_real_seconds = (now - timed_start) / clock_scale
            label = 1 if rng.random() < 0.58 else 0
            pool = target_pcm if label == 1 else background_pcm
            texts = target_texts if label == 1 else background_texts
            selected = index % len(pool)
            responses = stream_timing_pcm(
                timing,
                pool[selected],
                receive_json,
                pace=True,
            )
            eligible = [
                item for item in responses if item.get("decision_eligible") is True
            ]
            final_prediction = eligible[-1] if eligible else responses[-1]
            activation = next(
                (item for item in responses if item.get("activate") is True), None
            )
            corrected = False
            correction_type: str | None = None
            tap_result: dict[str, Any] | None = None
            if label == 1 and activation is None:
                tap_result = send_conditional_tap(
                    timing,
                    intended_action="manual_activate",
                    state="waiting",
                    activation_id=activation_id,
                    state_version=state_version,
                )
                corrected = True
                correction_type = "manual_activate"
            elif label == 0 and activation is not None:
                tap_result = send_conditional_tap(
                    timing,
                    intended_action="interrupt",
                    state="thinking",
                    activation_id=str(activation["activation_id"]),
                    state_version=int(activation["activation_state_version"]),
                )
                corrected = True
                correction_type = "interrupt"

            if tap_result is not None:
                if tap_result.get("accepted") is False:
                    raise RuntimeError(f"Conditional correction was cancelled: {tap_result}")
                if (
                    tap_result.get("training_eligible") is not True
                    or tap_result.get("training_applied") is not True
                ):
                    raise RuntimeError(f"Eligible correction was not learned: {tap_result}")
                state_version = int(tap_result["server_state_version"])
                activation_id = str(tap_result.get("server_activation_id") or "")
            elif activation is not None:
                state_version = int(activation["activation_state_version"])
                activation_id = str(activation["activation_id"])

            # A real response finishes before the next user turn.  Exercise the
            # public LLM+TTS reply path periodically and at least once,
            # while all simulated-user speech still uses the assigned voice.
            if label == 1 and (llm_reply_count == 0 or index % 20 == 0):
                reply = request_json(
                    base_url,
                    "/api/chat",
                    method="POST",
                    token=study_token,
                    body={
                        "session_id": study_session["id"],
                        "source": "primary",
                        "transcript": texts[selected],
                    },
                )
                request_bytes(
                    base_url,
                    "/api/tts",
                    token=study_token,
                    body={
                        "session_id": study_session["id"],
                        "text": str(reply["text"]),
                    },
                )
                llm_reply_count += 1
            if activation is not None or (
                tap_result is not None
                and tap_result.get("action") == "manual_activate"
            ):
                state = send_state(timing, "waiting")
                state_version = int(state["server_state_version"])
                activation_id = str(state.get("server_activation_id") or "")

            comparison = dict(final_prediction.get("model_head_comparison") or {})
            record = {
                "schema": "tap-to-adapt-duration-opportunity/v1",
                "simulation": True,
                "participant_id": roster["participant_id"],
                "participant_index": int(roster["participant_index"]),
                "voice_id": int(roster["voice_id"]),
                "background_voice_id": int(roster["background_voice_id"]),
                "session_id": study_session["id"],
                "opportunity_index": index + 1,
                "elapsed_seconds": elapsed_real_seconds,
                "bin_5m": int(elapsed_real_seconds // 300) + 1,
                "label": label,
                "corrected": corrected,
                "correction_type": correction_type,
                "activated": activation is not None,
                "probability": final_prediction.get("probability"),
                "shared_probability": comparison.get("shared_probability"),
                "personal_probability": comparison.get("personal_probability"),
                "personal_update_count": (
                    dict(
                        (tap_result or {}).get(
                            "model_head_comparison_after_update", comparison
                        )
                        or {}
                    ).get("update_count")
                ),
                "training_applied": (
                    tap_result.get("training_applied") if tap_result else False
                ),
                "text_pool_index": selected,
                "llm_generated_text": True,
                "tts_provider": "sherpa-onnx",
                "tts_model": "vits-icefall-zh-aishell3",
            }
            append_opportunity_event(
                base_url, study_token, str(study_session["id"]), record
            )
            records.append(record)
            if index % 8 == 0:
                with progress_lock:
                    print(
                        f"P{roster['participant_index']}: "
                        f"{elapsed_real_seconds / 60:.1f}m, "
                        f"opportunities={index + 1}, "
                        f"corrections={sum(int(item['corrected']) for item in records)}",
                        flush=True,
                    )
            index += 1
            jitter = rng.uniform(0.85, 1.15)
            next_opportunity += opportunity_seconds * jitter * clock_scale

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
        # Engineering-only sessions are deliberately aborted after their
        # immutable event log is written; they can never count as completions.
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
        "voice_id": int(roster["voice_id"]),
        "session_id": study_session["id"],
        "wall_elapsed_seconds": time.time() - started_wall,
        "opportunities": len(records),
        "corrections": sum(int(item["corrected"]) for item in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run participant-isolated, real-public-route duration simulation."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-minutes", type=float, default=90.0)
    parser.add_argument("--opportunity-seconds", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-stagger-seconds", type=float, default=8.0)
    parser.add_argument("--clock-scale", type=float, default=1.0)
    parser.add_argument("--confirm-simulation", action="store_true")
    parser.add_argument("--allow-time-compression-for-smoke-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_simulation:
        raise SystemExit("Pass --confirm-simulation")
    if not 5 <= args.max_minutes <= 90:
        raise SystemExit("--max-minutes must be in [5, 90]")
    if not 20 <= args.opportunity_seconds <= 180:
        raise SystemExit("--opportunity-seconds must be in [20, 180]")
    if not 0 <= args.start_stagger_seconds <= 30:
        raise SystemExit("--start-stagger-seconds must be in [0, 30]")
    if args.clock_scale != 1.0 and not args.allow_time_compression_for_smoke_only:
        raise SystemExit(
            "Duration evidence must use --clock-scale 1; compression is smoke-only"
        )
    if not 0.02 <= args.clock_scale <= 1.0:
        raise SystemExit("--clock-scale must be in [0.02, 1.0]")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    turn_data = json.loads(args.turns.read_text(encoding="utf-8"))
    by_id = {
        str(item["participant_id"]): item for item in turn_data["participants"]
    }
    if {row["participant_id"] for row in roster} != set(by_id):
        raise SystemExit("Roster and LLM turn-pool participant IDs do not match")
    run_slugs = {row["run_slug"] for row in roster}
    if run_slugs != {turn_data["run_slug"]}:
        raise SystemExit("Roster and LLM turn pool belong to different runs")
    if any(
        row.get("data_origin") != "simulation"
        or row.get("analysis_use") != "duration_simulation_engineering_only"
        for row in roster
    ):
        raise SystemExit("Refusing accounts outside the simulation-only fixture")

    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.workers, len(roster)))
    ) as executor:
        future_rows = {
            executor.submit(
                run_participant,
                base_url=args.base_url.rstrip("/"),
                roster=row,
                turns=by_id[row["participant_id"]],
                max_minutes=args.max_minutes,
                opportunity_seconds=args.opportunity_seconds,
                clock_scale=args.clock_scale,
                start_stagger_seconds=args.start_stagger_seconds,
                seed=20260723 + int(row["participant_index"]) * 97,
                progress_lock=lock,
            ): row
            for row in roster
        }
        for future in as_completed(future_rows):
            row = future_rows[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "participant_index": row["participant_index"],
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": "".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                    }
                )
                with lock:
                    print(
                        f"P{row['participant_index']}: FAILED "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    print(failures[-1]["traceback"], flush=True)

    report = {
        "schema": "tap-to-adapt-duration-public-run/v1",
        "simulation_only": True,
        "duration_evidence_valid": args.clock_scale == 1.0,
        "run_slug": next(iter(run_slugs)),
        "base_url": args.base_url.rstrip("/"),
        "llm_provider": turn_data["llm_provider"],
        "llm_model": turn_data["llm_model"],
        "tts_provider": "sherpa-onnx",
        "tts_model": "vits-icefall-zh-aishell3",
        "model": (
            "ntu-spml/distilhubert frozen backbone + participant-disjoint "
            "shared logistic head + per-participant replay personal head"
        ),
        "max_minutes": args.max_minutes,
        "opportunity_seconds": args.opportunity_seconds,
        "start_stagger_seconds": args.start_stagger_seconds,
        "clock_scale": args.clock_scale,
        "started_ms": int(started * 1000),
        "finished_ms": int(time.time() * 1000),
        "participants_expected": len(roster),
        "participants_succeeded": len(results),
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
