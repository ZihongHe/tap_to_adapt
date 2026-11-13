"""Time-compressed, analysis-ineligible end-to-end feasibility simulation.

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
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import websocket


class SimulationError(RuntimeError):
    pass


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    token: str = "",
    body: dict[str, Any] | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json", "User-Agent": "TTA-Feasibility-Simulator/1.0"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{base_url}{path}", data=payload, headers=headers, method=method
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read()
            return json.loads(content.decode("utf-8")) if content else {}
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            last_error = SimulationError(
                f"HTTP {exc.code} for {method} {path}: {detail[:1000]}"
            )
            if exc.code < 500:
                break
            if attempt + 1 < retries:
                time.sleep(0.4 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.4 * (attempt + 1))
    raise SimulationError(f"{method} {path} failed: {last_error}")


def request_bytes(
    base_url: str,
    path: str,
    *,
    token: str,
    body: dict[str, Any],
    retries: int = 3,
) -> bytes:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Accept": "audio/mpeg",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "TTA-Feasibility-Simulator/1.0",
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{base_url}{path}", data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
                if response.headers.get_content_type() != "audio/mpeg":
                    raise SimulationError("TTS endpoint did not return audio/mpeg")
            if not content:
                raise SimulationError("TTS endpoint returned an empty response")
            return content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = SimulationError(
                f"HTTP {exc.code} for POST {path}: {detail[:1000]}"
            )
            if exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, SimulationError) as exc:
            last_error = exc
        if attempt + 1 < retries:
            time.sleep(0.5 * (attempt + 1))
    raise SimulationError(f"POST {path} failed: {last_error}")


def login(base_url: str, code: str, role: str) -> dict[str, Any]:
    result = request_json(
        base_url,
        "/api/login",
        method="POST",
        body={"code": code, "role": role},
    )
    if result.get("preview_mode") is not True or result.get("pilot_only") is not True:
        raise SimulationError("Refusing to simulate an account not marked preview_mode")
    return result


def validate_roster_rows(rows: list[dict[str, str]]) -> str:
    if len(rows) != 72:
        raise SimulationError(f"Expected 72 simulation roster rows, found {len(rows)}")
    required = {
        "assignment_id",
        "participant_code",
        "cohort",
        "internal_group",
        "data_origin",
        "protocol_mode",
        "run_slug",
        "day1_probe_order",
        "day2_probe_order",
        "day3_probe_order",
    }
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        raise SimulationError(
            "Roster is missing simulation-safety columns: " + ", ".join(missing_columns)
        )
    origins = {row["data_origin"] for row in rows}
    modes = {row["protocol_mode"] for row in rows}
    run_slugs = {row["run_slug"] for row in rows}
    codes = [row["participant_code"] for row in rows]
    labels = [row["assignment_id"] for row in rows]
    if origins != {"simulation"} or modes != {"compressed"}:
        raise SimulationError(
            "Refusing to run: every row must be immutable simulation/compressed; "
            f"found origins={sorted(origins)}, modes={sorted(modes)}"
        )
    if len(run_slugs) != 1:
        raise SimulationError("All simulation accounts must belong to one isolated run")
    if len(set(codes)) != 72 or len(set(labels)) != 72:
        raise SimulationError("Simulation roster contains duplicate codes or assignment labels")
    counts = {
        (cohort, group): sum(
            row["cohort"] == cohort and row["internal_group"] == group for row in rows
        )
        for cohort, group in (
            ("study_a", "seed_donor"),
            ("study_a", "evaluator_donor"),
            ("study_a", "holdout_evaluator"),
            ("study_b", "personalized"),
            ("study_b", "frozen"),
        )
    }
    expected = {
        ("study_a", "seed_donor"): 12,
        ("study_a", "evaluator_donor"): 16,
        ("study_a", "holdout_evaluator"): 8,
        ("study_b", "personalized"): 18,
        ("study_b", "frozen"): 18,
    }
    if counts != expected:
        raise SimulationError(f"Simulation roster allocation mismatch: {counts}")
    return next(iter(run_slugs))


def preflight_accounts(base_url: str, rows: list[dict[str, str]]) -> None:
    failures: list[str] = []
    for row in rows:
        try:
            result = login(base_url, row["participant_code"], "study")
            if result.get("pilot_only") is not True:
                raise SimulationError("server did not mark account pilot_only")
        except Exception as exc:
            failures.append(f"{row['assignment_id']}: {exc}")
            if len(failures) >= 5:
                break
    if failures:
        raise SimulationError(
            "Simulation preflight failed before any sessions were created. "
            "Do not use a formal roster. " + " | ".join(failures)
        )


def ensure_consent(base_url: str, token: str) -> dict[str, Any]:
    bootstrap = request_json(base_url, "/api/bootstrap", token=token)
    if bootstrap.get("consent_valid"):
        return bootstrap
    consent = bootstrap["consent"]
    request_json(
        base_url,
        "/api/consent",
        method="POST",
        token=token,
        body={
            "version": consent["version"],
            "document_hash": consent["sha256"],
            "participation": True,
            "raw_audio": False,
            "future_sharing": False,
            "affirmation": consent["required_affirmation"],
        },
    )
    return request_json(base_url, "/api/bootstrap", token=token)


def submit_baseline(base_url: str, token: str, session_id: str, version: str) -> None:
    request_json(
        base_url,
        "/api/responses",
        method="POST",
        token=token,
        body={
            "session_id": session_id,
            "instrument": "baseline",
            "version": version,
            "answers": {
                "voice_technology_familiarity": 4,
                "privacy_concern": 4,
                "speech_agent_confidence": 4,
                "age_range": "18_22",
                "gender": "prefer_not",
                "primary_spoken_language": "mandarin",
                "voice_agent_frequency": "weekly",
                "hearing_or_speech_condition": "no",
                "tap_rule_check": "felt_need",
                "failure_rule_check": "pause_contact",
                "device_rule_check": "same_code_day",
            },
        },
    )


def pair_websockets(
    base_url: str,
    study_token: str,
    study_session: str,
    companion_token: str,
    companion_session: str,
) -> tuple[websocket.WebSocket, websocket.WebSocket]:
    parsed = urllib.parse.urlsplit(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_root = urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, "", ""))

    def ws_url(token: str, session_id: str) -> str:
        query = urllib.parse.urlencode({"token": token, "session_id": session_id})
        return f"{ws_root}/ws/pair?{query}"

    # Pair presence should settle promptly on the public route.  Bounding each
    # receive prevents one overloaded reconnect from pinning an entire
    # simulation worker for several minutes.
    options: dict[str, Any] = {"timeout": 8, "enable_multithread": True}
    if ws_scheme == "wss":
        options["sslopt"] = {"cert_reqs": ssl.CERT_REQUIRED}
    study_socket = websocket.create_connection(ws_url(study_token, study_session), **options)
    companion_socket: websocket.WebSocket | None = None
    try:
        study_socket.recv()
        companion_socket = websocket.create_connection(
            ws_url(companion_token, companion_session), **options
        )
        companion_socket.recv()
        companion_socket.send(
            json.dumps(
                {
                    "type": "background_state",
                    "action": "ready",
                    "ready": True,
                    "simulation": True,
                }
            )
        )
        ready = False
        for _ in range(6):
            message = json.loads(study_socket.recv())
            if (
                message.get("type") == "pair_status"
                and message.get("same_day_online") is True
                and message.get("companion_ready") is True
            ):
                ready = True
                break
        if not ready:
            raise SimulationError("Pair did not reach same-day companion-ready state")
        return study_socket, companion_socket
    except Exception:
        study_socket.close()
        if companion_socket is not None:
            companion_socket.close()
        raise


def simulated_events(
    label: str, cohort: str, group: str, day: int, probe_order: str
) -> list[dict[str, Any]]:
    seed = sum(ord(character) for character in label) * 10 + day
    rng = random.Random(seed)
    now = int(time.time() * 1000)
    events: list[dict[str, Any]] = []

    def add(event_type: str, payload: dict[str, Any]) -> None:
        events.append(
            {
                "event_type": event_type,
                "client_ms": now + len(events) * 250,
                "monotonic_ms": len(events) * 250.0,
                "payload": {
                    **payload,
                    "simulation": True,
                    "assignment_label": label,
                    "cohort": cohort,
                    "group": group,
                    "day": day,
                },
            }
        )

    add("simulation.session_start", {"generator": "feasibility-v1"})
    add(
        "topic.selected",
        {
            "topic_id": "free_topic",
            "category_id": "free",
            "reused_topic_id": day > 1,
        },
    )
    add("phase.start", {"phase": "frozen", "policy_token": "A"})
    for index in range(4):
        add(
            "response.opportunity",
            {"phase": "frozen", "opportunity": index + 1, "gap_ms": rng.randint(250, 1400)},
        )
    add("phase.end", {"phase": "frozen", "opportunities": 4, "elapsed_seconds": 0})

    tap_count = max(2, 7 - day) if group != "frozen" else 7
    interrupt_count = max(1, 5 - day) if group != "frozen" else 5
    for cycle in range(4):
        add(
            "phase.start",
            {"phase": "learning", "policy_token": "A", "cycle": cycle + 1},
        )
        for index in range(tap_count):
            add(
                "simulation.manual_activate_plan",
                {"cycle": cycle + 1, "index": index + 1, "label": 1},
            )
        for index in range(interrupt_count):
            add(
                "simulation.interrupt_plan",
                {
                    "cycle": cycle + 1,
                    "index": index + 1,
                    "label": 0,
                    "activation_to_tap_ms": rng.randint(450, 1900),
                    "user_speech_resumed": True,
                },
            )
        add(
            "phase.end",
            {"phase": "learning", "cycle": cycle + 1, "completed_seconds": 0},
        )
        if cycle < 3:
            add("companion.window_start", {"window": cycle + 1})
            add("companion.window_end", {"window": cycle + 1})

    for index, letter in enumerate(probe_order, start=1):
        add(
            "phase.start",
            {"phase": "micro_probe", "policy_token": letter, "block": index},
        )
        add(
            "probe.summary",
            {
                "policy_token": letter,
                "activation_count": rng.randint(3, 8),
                "interrupt_count": rng.randint(1, 5),
            },
        )
        add(
            "phase.end",
            {
                "phase": "micro_probe",
                "policy_token": letter,
                "block": index,
                "completed_seconds": 0,
            },
        )
    add("simulation.session_end", {"generator": "feasibility-v1"})
    return events


def simulate_participant(
    base_url: str, row: dict[str, str], progress_lock: threading.Lock
) -> dict[str, Any]:
    code = row["participant_code"]
    label = row["assignment_id"]
    cohort = row["cohort"]
    group = row["internal_group"]
    study_login = login(base_url, code, "study")
    study_token = str(study_login["token"])
    bootstrap = ensure_consent(base_url, study_token)
    companion_login = login(base_url, code, "companion")
    companion_token = str(companion_login["token"])
    completed_days = {
        int(item["day"])
        for item in bootstrap["progress"]["sessions"]
        if item["client_role"] == "study" and item["state"] in {"completed", "ended"}
    }
    result = {"label": label, "cohort": cohort, "group": group, "days": 0, "skipped": 0}

    for day in range(1, 4):
        if day in completed_days:
            result["skipped"] += 1
            continue
        study_session = request_json(
            base_url,
            "/api/sessions/start",
            method="POST",
            token=study_token,
            body={"day": day},
        )
        companion_session = request_json(
            base_url,
            "/api/sessions/start",
            method="POST",
            token=companion_token,
            body={"day": day},
        )
        if day == 1:
            submit_baseline(
                base_url,
                study_token,
                study_session["id"],
                str(bootstrap["questionnaire_version"]),
            )
        quality = {
            "network": {
                "median_ms": 48,
                "p95_ms": 110,
                "jitter_ms": 18,
                "failures": 0,
                "samples": [42, 44, 45, 46, 47, 48, 49, 52, 60, 72, 88, 110],
                "simulation": True,
            },
            "microphone": {
                "snr_db": 22.5,
                "quiet_rms_p90": 0.018,
                "speech_rms_p90": 0.18,
                "clipping_ratio": 0.0,
                "asr_probe_passed": True,
                "simulation": True,
            },
            "reaction_latency": {
                "median_ms": 315,
                "samples_ms": [250, 275, 300, 330, 350, 380],
                "simulation": True,
            },
            "browser": {
                "secure_context": True,
                "media_devices": True,
                "audio_worklet": True,
                "websocket": True,
                "speech_recognition": True,
                "media_recorder": True,
                "device_id": "simulation-browser",
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
        study_socket, companion_socket = pair_websockets(
            base_url,
            study_token,
            study_session["id"],
            companion_token,
            companion_session["id"],
        )
        try:
            request_json(
                base_url,
                f"/api/sessions/{study_session['id']}/activate",
                method="POST",
                token=study_token,
            )
            events = simulated_events(
                label, cohort, group, day, row[f"day{day}_probe_order"]
            )
            request_json(
                base_url,
                "/api/events/batch",
                method="POST",
                token=study_token,
                body={"session_id": study_session["id"], "events": events},
            )
            primary_reply = request_json(
                base_url,
                "/api/chat",
                method="POST",
                token=study_token,
                body={
                    "session_id": study_session["id"],
                    "source": "primary",
                    "transcript": "我最近在考虑怎样安排自己的空闲时间。",
                },
            )
            request_bytes(
                base_url,
                "/api/tts",
                token=study_token,
                body={
                    "session_id": study_session["id"],
                    "text": str(primary_reply["text"]),
                },
            )
            request_json(
                base_url,
                "/api/chat",
                method="POST",
                token=companion_token,
                body={
                    "session_id": companion_session["id"],
                    "source": "companion",
                    "transcript": "我比较喜欢散步，也会听一些音乐。",
                },
            )
            request_json(
                base_url,
                "/api/responses",
                method="POST",
                token=study_token,
                body={
                    "session_id": study_session["id"],
                    "instrument": f"daily_day_{day}",
                    "version": bootstrap["questionnaire_version"],
                    "answers": {
                        "timing_today": 4,
                        "effort_today": 4,
                        "annoyance_today": 4,
                        "predictable_today": 4,
                        "speed_today": 4,
                        "like_today": 4,
                        "fatigue_today": 4,
                        "control_today": 4,
                        "timing_example": "Synthetic feasibility response; exclude from analysis.",
                        "technical_issue": "simulation run",
                    },
                },
            )
            if day == 3:
                request_json(
                    base_url,
                    "/api/responses",
                    method="POST",
                    token=study_token,
                    body={
                        "session_id": study_session["id"],
                        "instrument": "final",
                        "version": bootstrap["questionnaire_version"],
                        "answers": {
                            "trend_accuracy": 4,
                            "trend_likeability": 4,
                            "trend_demand": 4,
                            "trend_annoyance": 4,
                            "trend_habitability": 4,
                            "trend_speed": 4,
                            "overall_accuracy": 4,
                            "overall_likeability": 4,
                            "overall_demand": 4,
                            "overall_annoyance": 4,
                            "overall_habitability": 4,
                            "overall_speed": 4,
                            "overall_impression": "Synthetic feasibility response.",
                            "best_change": "Synthetic feasibility response.",
                            "tap_reasons": "Synthetic feasibility response.",
                            "user_adaptation": "Synthetic feasibility response.",
                            "companion_effect": "simulation run",
                            "improvement": "simulation run",
                        },
                    },
                )
            completion = request_json(
                base_url,
                f"/api/sessions/{study_session['id']}/end",
                method="POST",
                token=study_token,
            )
            if completion.get("completed") is not True:
                raise SimulationError(f"{label} day {day} did not complete")
        finally:
            study_socket.close()
            companion_socket.close()
        result["days"] += 1

    with progress_lock:
        print(
            f"{label}: completed={result['days']} skipped={result['skipped']}",
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-compressed, analysis-ineligible end-to-end feasibility simulation."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-partial-roster", action="store_true")
    parser.add_argument("--confirm-simulation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_simulation:
        raise SystemExit("Pass --confirm-simulation; this script writes synthetic study records.")

    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["cohort"] in {"study_a", "study_b"}
        ]
    run_slug = validate_roster_rows(rows)
    if args.limit:
        rows = rows[: args.limit]
    elif args.allow_partial_roster:
        raise SystemExit("--allow-partial-roster requires --limit")

    preflight_accounts(args.base_url.rstrip("/"), rows)

    started = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    progress_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = {
            executor.submit(
                simulate_participant,
                args.base_url.rstrip("/"),
                row,
                progress_lock,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(
                    {
                        "label": row["assignment_id"],
                        "cohort": row["cohort"],
                        "error": str(exc),
                    }
                )
                print(f"{row['assignment_id']}: FAILED {exc}", flush=True)

    report = {
        "simulation_only": True,
        "run_slug": run_slug,
        "base_url": args.base_url,
        "started_ms": int(started * 1000),
        "finished_ms": int(time.time() * 1000),
        "elapsed_seconds": round(time.time() - started, 2),
        "participants_expected": len(rows),
        "participants_succeeded": len(results),
        "days_completed": sum(item["days"] for item in results),
        "days_skipped_as_already_complete": sum(item["skipped"] for item in results),
        "failures": failures,
        "results": sorted(results, key=lambda item: item["label"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
