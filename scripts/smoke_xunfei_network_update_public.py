"""Run a simulation-only three-feedback public DTCN update/reload gate. iFLYTEK credentials are read only from process environment variables.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import ssl
import sys
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable

import websocket

try:
    from .disable_simulation_run import (
        disable_runner_smoke_fixture,
        is_canonical_runner_smoke_run_slug,
    )
    from .simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        simulated_events,
        submit_baseline,
    )
    from .smoke_negative_feedback_public import (
        cleanup_smoke_run,
        receive_json,
        submit_quality,
    )
    from .voice_pcm import stream_timing_pcm, synthesize_xunfei_pcm
except ImportError:  # Direct execution from the scripts directory.
    from disable_simulation_run import (
        disable_runner_smoke_fixture,
        is_canonical_runner_smoke_run_slug,
    )
    from simulate_feasibility import (
        ensure_consent,
        login,
        pair_websockets,
        request_bytes,
        request_json,
        simulated_events,
        submit_baseline,
    )
    from smoke_negative_feedback_public import (
        cleanup_smoke_run,
        receive_json,
        submit_quality,
    )
    from voice_pcm import stream_timing_pcm, synthesize_xunfei_pcm


DEFAULT_UTTERANCE = "我想补充一个具体例子，说明刚才提到的生活习惯。"
ANALYSIS_USE = "simulation_network_update_smoke_v11"
EXPECTED_ROWS = {
    "network_update_personalized": (
        "personalized",
        "three_feedback_network_update",
    ),
    "network_update_frozen": ("frozen", "three_feedback_network_control"),
}


def load_private_fixture(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    if not path.is_file():
        raise ValueError("Network-update fixture CSV does not exist")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if len(raw_rows) != 2:
        raise ValueError("Network-update smoke requires exactly two one-shot accounts")

    run_slugs = {str(row.get("run_slug", "")) for row in raw_rows}
    if len(run_slugs) != 1:
        raise ValueError("Network-update fixture rows must share exactly one run slug")
    run_slug = next(iter(run_slugs))
    if not is_canonical_runner_smoke_run_slug(run_slug):
        raise ValueError("Network-update fixture is outside the canonical smoke-v11 namespace")

    rows: dict[str, dict[str, str]] = {}
    for raw_row in raw_rows:
        row = {str(key): str(value) for key, value in raw_row.items()}
        purpose = row.get("purpose", "")
        if purpose in rows:
            raise ValueError("Network-update fixture purposes must be unique")
        rows[purpose] = row
    if set(rows) != set(EXPECTED_ROWS):
        raise ValueError("Network-update fixture has the wrong purpose cells")

    for purpose, (group, scenario) in EXPECTED_ROWS.items():
        row = rows[purpose]
        expected = {
            "internal_group": group,
            "smoke_scenario": scenario,
            "analysis_use": ANALYSIS_USE,
            "fixture_design": "network-update",
            "run_slug": run_slug,
        }
        mismatches = {
            field: {"expected": value, "actual": row.get(field)}
            for field, value in expected.items()
            if row.get(field) != value
        }
        if mismatches:
            raise ValueError(f"Invalid {purpose} fixture metadata: {mismatches}")
        for field in ("participant_code", "participant_id", "assignment_id"):
            if not row.get(field):
                raise ValueError(f"{purpose} fixture has no {field}")

    for field in ("participant_code", "participant_id", "assignment_id"):
        if len({row[field] for row in rows.values()}) != 2:
            raise ValueError(f"Network-update fixture {field} values must be unique")
    return run_slug, rows


def _local_fixture_db(
    run_slug: str, expected_data_dir: Path
) -> tuple[Any, dict[str, Any], Any]:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.config import Settings
    from app.db import StudyDB

    settings = Settings.from_env()
    actual_data_dir = settings.data_dir.resolve()
    if actual_data_dir != expected_data_dir.resolve():
        raise RuntimeError("Configured data directory does not match --expected-data-dir")
    if not (actual_data_dir / "study.sqlite3").is_file():
        raise RuntimeError("Study database is unavailable on this application host")
    db = StudyDB(settings)
    run = db.study_run(run_slug)
    if run is None or run["data_origin"] != "simulation":
        raise RuntimeError("Network-update run is absent or not simulation-only")
    return db, run, settings


def validate_local_fixture_lifecycle(
    run_slug: str,
    expected_data_dir: Path,
    rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    db, run, settings = _local_fixture_db(run_slug, expected_data_dir)
    participants = db.participants_in_run(str(run["id"]))
    if len(participants) != 2:
        raise RuntimeError("Network-update run does not contain exactly two participants")
    if any(int(participant["enabled"]) != 1 for participant in participants):
        raise RuntimeError("Network-update fixture is not fresh and fully enabled")

    by_id = {str(participant["id"]): participant for participant in participants}
    if set(by_id) != {row["participant_id"] for row in rows.values()}:
        raise RuntimeError("Fixture participant UUIDs do not match the local run")
    for purpose, row in rows.items():
        participant = by_id[row["participant_id"]]
        expected_group = EXPECTED_ROWS[purpose][0]
        checks = {
            "group_name": expected_group,
            "assignment_id": row["assignment_id"],
            "run_slug": run_slug,
        }
        for field, expected in checks.items():
            if str(participant.get(field, "")) != expected:
                raise RuntimeError(f"Fixture database {field} mismatch for {purpose}")

        personal_dir = settings.data_dir / "models" / row["participant_id"]
        if (personal_dir / "latest_encoder.pth").exists() or (
            personal_dir / "latest_main_model.pth"
        ).exists():
            raise RuntimeError(f"Fresh fixture already has a personal checkpoint: {purpose}")

    with db.connect() as connection:
        session_row = connection.execute(
            "SELECT COUNT(*) AS session_count FROM sessions WHERE run_id=?",
            (str(run["id"]),),
        ).fetchone()
    if session_row is None or int(session_row["session_count"]) != 0:
        raise RuntimeError("Network-update fixture is not fresh; sessions already exist")

    base_encoder = Path(settings.base_encoder_checkpoint)
    base_main = Path(settings.base_main_checkpoint)
    if not base_encoder.is_file() or not base_main.is_file():
        raise RuntimeError("Locked DTCN base checkpoints are unavailable")
    return {
        "participant_count": 2,
        "groups": ["frozen", "personalized"],
        "purposes": sorted(rows),
        "session_count": 0,
        "personal_checkpoints_before": 0,
        "agent_mode": settings.agent_mode,
    }


def disable_local_fixture(run_slug: str, expected_data_dir: Path) -> dict[str, Any]:
    db, _run, _settings = _local_fixture_db(run_slug, expected_data_dir)
    return disable_runner_smoke_fixture(db, run_slug)


def timing_url(base_url: str, token: str, session_id: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_root = urllib.parse.urlunsplit((ws_scheme, parsed.netloc, parsed.path, "", ""))
    query = urllib.parse.urlencode({"token": token, "session_id": session_id})
    return f"{ws_root}/ws/timing?{query}"


def open_timing(base_url: str, token: str, session_id: str) -> tuple[Any, dict[str, Any]]:
    options: dict[str, Any] = {"timeout": 90, "enable_multithread": True}
    if base_url.startswith("https://"):
        options["sslopt"] = {"cert_reqs": ssl.CERT_REQUIRED}
    socket = websocket.create_connection(
        timing_url(base_url, token, session_id), **options
    )
    ready = receive_json(socket)
    if ready.get("type") != "timing_ready" or ready.get("mode") != "dtcn":
        socket.close()
        raise RuntimeError(f"DTCN timing socket was not ready: {ready}")
    return socket, ready


def commit_phase(socket: Any, phase: str) -> dict[str, Any]:
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
    acknowledgement = receive_json(socket)
    if (
        acknowledgement.get("type") != "phase_ack"
        or acknowledgement.get("transition_id") != transition_id
        or acknowledgement.get("phase") != phase
    ):
        raise RuntimeError(f"Timing phase was not committed: {acknowledgement}")
    return acknowledgement


def send_manual_feedback(
    socket: Any, acknowledgement: dict[str, Any]
) -> dict[str, Any]:
    tap_id = str(uuid.uuid4())
    socket.send(
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
                "expected_activation_id": acknowledgement.get("server_activation_id")
                or "",
                "expected_state_version": int(
                    acknowledgement["server_state_version"]
                ),
            }
        )
    )
    response = receive_json(socket)
    if response.get("type") != "tap_result" or response.get("tap_id") != tap_id:
        raise RuntimeError(f"Unexpected manual-feedback response: {response}")
    return response


def validate_feedback_sequence(
    group: str, results: list[dict[str, Any]]
) -> None:
    if len(results) != 3:
        raise RuntimeError("Network-update smoke must produce exactly three taps")
    mismatches: dict[str, Any] = {}
    for index, result in enumerate(results, start=1):
        for field, expected in {"action": "manual_activate", "label": 1}.items():
            if result.get(field) != expected:
                mismatches[f"tap{index}.{field}"] = {
                    "expected": expected,
                    "actual": result.get(field),
                }

    if group == "personalized":
        expected_scales = [0.0, 0.0, 0.05]
        for index, (result, scale) in enumerate(
            zip(results, expected_scales, strict=True), start=1
        ):
            expected = {
                "training_eligible": True,
                "training_applied": True,
                "personal_calibration_applied": True,
                "network_update_scale": scale,
                "current_network_update_applied": index == 3,
                "requested_training_steps": 1 if index == 3 else 0,
            }
            for field, value in expected.items():
                if result.get(field) != value:
                    mismatches[f"tap{index}.{field}"] = {
                        "expected": value,
                        "actual": result.get(field),
                    }
        third = results[2]
        before = int(third.get("optimizer_step_before", -1))
        after = int(third.get("optimizer_step_after", -1))
        if after - before != 1 or int(third.get("training_steps", -1)) != 1:
            mismatches["tap3.optimizer"] = {
                "expected": "exactly one persisted optimizer step",
                "actual": [before, after, third.get("training_steps")],
            }
        if third.get("parameter_state_interpolated") is not True or not math.isclose(
            float(third.get("parameter_interpolation_scale", 0.0)), 0.05
        ):
            mismatches["tap3.parameter_interpolation"] = {
                "expected": [True, 0.05],
                "actual": [
                    third.get("parameter_state_interpolated"),
                    third.get("parameter_interpolation_scale"),
                ],
            }
        before_lrs = [
            float(value) for value in third.get("optimizer_learning_rates_before", [])
        ]
        during_lrs = [
            float(value) for value in third.get("optimizer_learning_rates_during", [])
        ]
        after_lrs = [
            float(value) for value in third.get("optimizer_learning_rates_after", [])
        ]
        if not before_lrs or not (
            len(before_lrs) == len(during_lrs) == len(after_lrs)
        ):
            mismatches["tap3.optimizer_learning_rates"] = "missing"
        elif any(
            not math.isclose(during, before_lr * 0.05, rel_tol=1e-9, abs_tol=1e-12)
            or not math.isclose(after_lr, before_lr, rel_tol=1e-12, abs_tol=1e-15)
            for before_lr, during, after_lr in zip(
                before_lrs, during_lrs, after_lrs, strict=True
            )
        ):
            mismatches["tap3.optimizer_learning_rates"] = {
                "before": before_lrs,
                "during": during_lrs,
                "after": after_lrs,
            }
        if not math.isfinite(float(third.get("loss", math.nan))):
            mismatches["tap3.loss"] = third.get("loss")
    elif group == "frozen":
        for index, result in enumerate(results, start=1):
            expected = {
                "training_eligible": False,
                "training_applied": False,
                "quarantined": True,
                "quarantine_reason": "activation_not_learning_current",
            }
            for field, value in expected.items():
                if result.get(field) != value:
                    mismatches[f"tap{index}.{field}"] = {
                        "expected": value,
                        "actual": result.get(field),
                    }
            if result.get("current_network_update_applied") not in {None, False}:
                mismatches[f"tap{index}.current_network_update_applied"] = result.get(
                    "current_network_update_applied"
                )
            optimizer_before = result.get("optimizer_step_before")
            optimizer_after = result.get("optimizer_step_after")
            if optimizer_before not in {None, 0} or optimizer_after not in {None, 0}:
                mismatches[f"tap{index}.optimizer"] = [
                    optimizer_before,
                    optimizer_after,
                ]
    else:
        raise ValueError("Network-update smoke group must be personalized or frozen")
    if mismatches:
        raise RuntimeError(f"Invalid {group} three-feedback sequence: {mismatches}")


def _start_day(
    base_url: str,
    study_token: str,
    companion_token: str,
    day: int,
    questionnaire_version: str,
) -> dict[str, Any]:
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
            str(study_session["id"]),
            questionnaire_version,
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
        "study_session": study_session,
        "companion_session": companion_session,
        "study_pair": study_pair,
        "companion_pair": companion_pair,
    }


def _complete_day_one(
    base_url: str,
    group: str,
    assignment_id: str,
    questionnaire_version: str,
    study_token: str,
    companion_token: str,
    study_session_id: str,
    companion_session_id: str,
) -> dict[str, Any]:
    events = [
        event
        for event in simulated_events(assignment_id, "pilot", group, 1, "ABC")
        if event["event_type"] != "topic.selected"
    ]
    request_json(
        base_url,
        "/api/events/batch",
        method="POST",
        token=study_token,
        body={"session_id": study_session_id, "events": events},
    )
    primary = request_json(
        base_url,
        "/api/chat",
        method="POST",
        token=study_token,
        body={
            "session_id": study_session_id,
            "source": "primary",
            "transcript": "这是隔离的网络更新冒烟测试，不进入正式分析。",
        },
    )
    tts = request_bytes(
        base_url,
        "/api/tts",
        token=study_token,
        body={"session_id": study_session_id, "text": str(primary["text"])},
    )
    request_json(
        base_url,
        "/api/chat",
        method="POST",
        token=companion_token,
        body={
            "session_id": companion_session_id,
            "source": "companion",
            "transcript": "这是隔离的模拟伴随端消息。",
        },
    )
    request_json(
        base_url,
        "/api/responses",
        method="POST",
        token=study_token,
        body={
            "session_id": study_session_id,
            "instrument": "daily_day_1",
            "version": questionnaire_version,
            "answers": {
                "timing_today": 4,
                "effort_today": 4,
                "annoyance_today": 4,
                "predictable_today": 4,
                "speed_today": 4,
                "like_today": 4,
                "fatigue_today": 4,
                "control_today": 4,
                "timing_example": "simulation-only network update smoke",
                "technical_issue": "simulation-only network update smoke",
            },
        },
    )
    completion = request_json(
        base_url,
        f"/api/sessions/{study_session_id}/end",
        method="POST",
        token=study_token,
    )
    if completion.get("completed") is not True:
        raise RuntimeError("Compressed Day 1 did not complete before reload validation")
    return {"completed": True, "tts_bytes": len(tts)}


def run_account(
    base_url: str,
    row: dict[str, str],
    pcm: bytes,
    expected_data_dir: Path,
) -> dict[str, Any]:
    group = row["internal_group"]
    study_login = login(base_url, row["participant_code"], "study")
    companion_login = login(base_url, row["participant_code"], "companion")
    study_token = str(study_login["token"])
    companion_token = str(companion_login["token"])
    bootstrap = ensure_consent(base_url, study_token)
    questionnaire_version = str(bootstrap["questionnaire_version"])
    current_session_id: str | None = None
    sockets: dict[str, Any] = {}
    pending_error: BaseException | None = None
    try:
        day_one = _start_day(
            base_url,
            study_token,
            companion_token,
            1,
            questionnaire_version,
        )
        current_session_id = str(day_one["study_session"]["id"])
        sockets.update(
            {
                "study_pair": day_one["study_pair"],
                "companion_pair": day_one["companion_pair"],
            }
        )
        request_json(
            base_url,
            "/api/events",
            method="POST",
            token=study_token,
            body={
                "session_id": current_session_id,
                "event_type": "topic.selected",
                "payload": {
                    "topic_id": "free_topic",
                    "category_id": "free",
                    "reused_topic_id": False,
                    "simulation": True,
                },
            },
        )
        timing_socket, ready_one = open_timing(
            base_url, study_token, current_session_id
        )
        sockets["timing_socket"] = timing_socket
        initial_fingerprint = dict(ready_one.get("model_fingerprint") or {})
        initial_personal = dict(
            ready_one.get("calibration", {}).get("personal", {})
        )
        if (initial_personal.get("enabled") is True) != (group == "personalized"):
            raise RuntimeError("Timing calibration policy does not match fixture group")

        tap_results: list[dict[str, Any]] = []
        for _index in range(3):
            commit_phase(timing_socket, "micro_probe")
            stream_timing_pcm(timing_socket, pcm, receive_json, pace=True)
            learning = commit_phase(timing_socket, "learning")
            tap_results.append(send_manual_feedback(timing_socket, learning))
        validate_feedback_sequence(group, tap_results)
        commit_phase(timing_socket, "complete")

        db, _run, settings = _local_fixture_db(
            row["run_slug"], expected_data_dir
        )
        del db
        personal_dir = settings.data_dir / "models" / row["participant_id"]
        encoder = personal_dir / "latest_encoder.pth"
        main = personal_dir / "latest_main_model.pth"
        from app.integrity import file_sha256

        if group == "personalized":
            if not encoder.is_file() or not main.is_file():
                raise RuntimeError("Third feedback did not persist a personal checkpoint pair")
            persisted_fingerprint = {
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
            }
            if persisted_fingerprint == initial_fingerprint:
                raise RuntimeError("Personal checkpoint fingerprint did not diverge")
        else:
            if encoder.exists() or main.exists():
                raise RuntimeError("Frozen feedback unexpectedly persisted a checkpoint")
            persisted_fingerprint = initial_fingerprint

        day_one_completion = _complete_day_one(
            base_url,
            group,
            row["assignment_id"],
            questionnaire_version,
            study_token,
            companion_token,
            current_session_id,
            str(day_one["companion_session"]["id"]),
        )
        for name in ("timing_socket", "study_pair", "companion_pair"):
            socket = sockets.pop(name, None)
            if socket is not None:
                socket.close()

        day_two = _start_day(
            base_url,
            study_token,
            companion_token,
            2,
            questionnaire_version,
        )
        current_session_id = str(day_two["study_session"]["id"])
        sockets.update(
            {
                "study_pair": day_two["study_pair"],
                "companion_pair": day_two["companion_pair"],
            }
        )
        timing_socket, ready_two = open_timing(
            base_url, study_token, current_session_id
        )
        sockets["timing_socket"] = timing_socket
        reloaded_fingerprint = dict(ready_two.get("model_fingerprint") or {})
        if reloaded_fingerprint != persisted_fingerprint:
            raise RuntimeError(
                "New Day-2 runtime did not reload the expected checkpoint fingerprint"
            )
        reloaded_personal = dict(
            ready_two.get("calibration", {}).get("personal", {})
        )
        if group == "personalized":
            expected_bias = float(tap_results[-1]["personal_logit_bias_after"])
            if not math.isclose(
                float(reloaded_personal.get("logit_bias", math.nan)),
                expected_bias,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ) or (
                int(reloaded_personal.get("update_count", -1)) != 3
                or int(reloaded_personal.get("positive_updates", -1)) != 3
                or int(reloaded_personal.get("negative_updates", -1)) != 0
            ):
                raise RuntimeError("New runtime did not reload personal calibration state")
        elif (
            reloaded_personal.get("enabled") is not False
            or int(reloaded_personal.get("update_count", 0)) != 0
            or int(reloaded_personal.get("positive_updates", 0)) != 0
            or int(reloaded_personal.get("negative_updates", 0)) != 0
        ):
            raise RuntimeError("Frozen reload unexpectedly changed calibration state")

        commit_phase(timing_socket, "micro_probe")
        reload_predictions = stream_timing_pcm(
            timing_socket, pcm, receive_json, pace=True
        )
        if not reload_predictions or not math.isfinite(
            float(reload_predictions[-1].get("raw_logit", math.nan))
        ):
            raise RuntimeError("Reloaded runtime did not produce a finite DTCN prediction")
        return {
            "ok": True,
            "group": group,
            "day_one_session_id": str(day_one["study_session"]["id"]),
            "day_two_session_id": current_session_id,
            "initial_model_fingerprint": initial_fingerprint,
            "persisted_model_fingerprint": persisted_fingerprint,
            "reloaded_model_fingerprint": reloaded_fingerprint,
            "tap_results": tap_results,
            "day_one_completion": day_one_completion,
            "reload_prediction_raw_logit": reload_predictions[-1].get("raw_logit"),
            "reload_calibration_update_count": reloaded_personal.get("update_count"),
        }
    except BaseException as error:
        pending_error = error
        raise
    finally:
        cleanup_errors = cleanup_smoke_run(
            base_url,
            study_token,
            current_session_id,
            tuple((name, socket) for name, socket in sockets.items()),
        )
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            if pending_error is not None:
                pending_error.add_note(f"Account cleanup also failed: {detail}")
            else:
                raise RuntimeError(f"Account cleanup failed: {detail}")


def execute_fixture(
    base_url: str,
    run_slug: str,
    rows: dict[str, dict[str, str]],
    expected_data_dir: Path,
    text: str,
    *,
    synthesizer: Callable[[str], tuple[bytes, dict[str, Any]]] = synthesize_xunfei_pcm,
    account_runner: Callable[..., dict[str, Any]] = run_account,
    lifecycle_preflight: Callable[
        [str, Path, dict[str, dict[str, str]]], dict[str, Any]
    ] = validate_local_fixture_lifecycle,
    disabler: Callable[[str, Path], dict[str, Any]] = disable_local_fixture,
) -> dict[str, Any]:
    pending_error: BaseException | None = None
    lifecycle: dict[str, Any] | None = None
    audio_metadata: dict[str, Any] | None = None
    results: dict[str, dict[str, Any]] = {}
    cleanup: dict[str, Any] | None = None
    try:
        lifecycle = lifecycle_preflight(run_slug, expected_data_dir, rows)
        if lifecycle.get("agent_mode") not in {None, "dtcn"}:
            raise RuntimeError("Network-update public smoke requires AGENT_MODE=dtcn")
        pcm, audio_metadata = synthesizer(text)
        frame_bytes = int(audio_metadata.get("frame_bytes_max", 0))
        if frame_bytes <= 0 or frame_bytes > 6_400:
            raise RuntimeError("Synthesized PCM frame metadata is outside (0, 6400]")
        for purpose in (
            "network_update_personalized",
            "network_update_frozen",
        ):
            results[purpose] = account_runner(
                base_url, rows[purpose], pcm, expected_data_dir
            )
    except BaseException as error:
        pending_error = error
        raise
    finally:
        try:
            cleanup = disabler(run_slug, expected_data_dir)
        except BaseException as cleanup_error:
            if pending_error is not None:
                pending_error.add_note(
                    "Fixture cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise

    assert lifecycle is not None
    assert audio_metadata is not None
    assert cleanup is not None
    if int(cleanup.get("participant_count", -1)) != 2 or int(
        cleanup.get("enabled_after", -1)
    ) != 0:
        raise RuntimeError("Network-update fixture cleanup was incomplete")
    return {
        "ok": True,
        "simulation_only": True,
        "run_slug": run_slug,
        "fixture_design": "network-update",
        "lifecycle_preflight": lifecycle,
        "audio": audio_metadata,
        **results,
        "fixture_cleanup": cleanup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simulation-only three-feedback public DTCN update/reload gate. "
            "iFLYTEK credentials are read only from process environment variables."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_UTTERANCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acknowledge-one-shot-fixture", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_one_shot_fixture:
        parser.error(
            "Pass --acknowledge-one-shot-fixture to confirm both QA accounts "
            "will be disabled in finally"
        )
    run_slug, rows = load_private_fixture(args.code_file)
    result = execute_fixture(
        args.base_url.rstrip("/"),
        run_slug,
        rows,
        args.expected_data_dir,
        args.text,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
