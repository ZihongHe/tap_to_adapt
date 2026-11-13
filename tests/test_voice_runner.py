"""Regression tests for voice runner.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import array
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts import smoke_xunfei_voiced_public as voiced_runner
from scripts.smoke_xunfei_voiced_public import (
    execute_voiced_fixture,
    fixture_cells,
    load_private_fixture,
    validate_local_fixture_lifecycle,
    validate_scenario_result,
)
from scripts.voice_pcm import (
    PCM_SAMPLE_RATE_HZ,
    TIMING_FRAME_BYTES,
    decode_mp3_to_pcm16,
    iter_timing_pcm_frames,
    stream_timing_pcm,
    synthesize_xunfei_pcm,
    validate_voiced_pcm,
)


RUN_SLUG = "simulation-runner-smoke-v11-1784451600123-a1b2c3d4"


def voiced_pcm(seconds: float = 0.6) -> bytes:
    samples = array.array(
        "h",
        (
            4_000 if index % 2 else -4_000
            for index in range(int(PCM_SAMPLE_RATE_HZ * seconds))
        ),
    )
    return samples.tobytes()


def fixture_rows() -> dict[str, dict[str, str]]:
    return {
        "positive": {
            "participant_code": "TTA-POSITIVE-TEST",
            "participant_id": "positive-participant-uuid",
            "internal_group": "personalized",
        },
        "negative": {
            "participant_code": "TTA-NEGATIVE-TEST",
            "participant_id": "negative-participant-uuid",
            "internal_group": "frozen",
        },
    }


def four_way_fixture_rows() -> dict[str, dict[str, str]]:
    return {
        f"{group}_{sign}": {
            "participant_code": f"TTA-{group}-{sign}",
            "participant_id": f"{group}-{sign}-uuid",
            "assignment_id": f"QSMK-{group}-{sign}",
            "internal_group": group,
            "smoke_scenario": (
                "quick_timing" if sign == "positive" else "long_quarantine"
            ),
            "purpose": f"{group}_{sign}",
            "analysis_use": "simulation_runner_smoke_v11",
            "fixture_design": "four-way",
            "run_slug": RUN_SLUG,
        }
        for group in ("personalized", "frozen")
        for sign in ("positive", "negative")
    }


def scenario_result(group: str, sign: str) -> dict[str, Any]:
    personalized = group == "personalized"
    result: dict[str, Any] = {
        "ok": True,
        "training_eligible": personalized,
        "training_applied": personalized,
        "personal_calibration_enabled": personalized,
        "personal_logit_bias_before": 0.0,
        "personal_logit_bias_after": (
            (0.1 if sign == "positive" else -0.1) if personalized else 0.0
        ),
    }
    if personalized:
        result["personal_calibration_applied"] = True
    if sign == "positive":
        result["stale_manual_action"] = "conditional_tap_cancelled"
    else:
        result["stale_interrupt_action"] = "conditional_tap_cancelled"
    return result


def test_mp3_decode_uses_only_pipes_and_forces_timing_format() -> None:
    calls: list[dict[str, Any]] = []

    class Result:
        returncode = 0
        stdout = voiced_pcm()
        stderr = b""

    def fake_run(command: list[str], **kwargs: Any) -> Result:
        calls.append({"command": command, **kwargs})
        return Result()

    result = decode_mp3_to_pcm16(b"synthetic-mp3", "ffmpeg-test", runner=fake_run)

    assert result == Result.stdout
    assert len(calls) == 1
    call = calls[0]
    assert call["input"] == b"synthetic-mp3"
    assert call["capture_output"] is True
    assert ["-ac", "1"] == call["command"][call["command"].index("-ac") :][:2]
    assert ["-ar", "16000"] == call["command"][call["command"].index("-ar") :][:2]
    assert call["command"][-1] == "pipe:1"
    assert all(not item.startswith("file:") for item in call["command"])


def test_voiced_pcm_validation_and_frames_match_public_proxy_limit() -> None:
    pcm = voiced_pcm(0.61)
    metadata = validate_voiced_pcm(pcm)
    frames = list(iter_timing_pcm_frames(pcm))

    assert metadata["sample_rate_hz"] == 16_000
    assert metadata["channels"] == 1
    assert metadata["rms"] > 0.004
    assert b"".join(frames) == pcm
    assert frames
    assert max(map(len, frames)) <= TIMING_FRAME_BYTES
    assert all(len(frame) % 2 == 0 for frame in frames)


def test_stream_pcm_waits_for_prediction_for_every_browser_sized_frame() -> None:
    pcm = voiced_pcm(0.61)

    class Socket:
        def __init__(self) -> None:
            self.frames: list[bytes] = []

        def send_binary(self, frame: bytes) -> None:
            self.frames.append(frame)

    socket = Socket()
    sleeps: list[float] = []
    responses = stream_timing_pcm(
        socket,
        pcm,
        lambda _: {"type": "prediction", "activate": False},
        sleeper=sleeps.append,
    )

    assert b"".join(socket.frames) == pcm
    assert len(responses) == len(socket.frames)
    assert max(map(len, socket.frames)) <= 6_400
    assert len(sleeps) == len(socket.frames) - 1
    assert all(0 < delay <= 0.2 for delay in sleeps)


def test_stream_pcm_can_stop_immediately_when_system_activates() -> None:
    pcm = voiced_pcm(1.0)

    class Socket:
        def __init__(self) -> None:
            self.frames: list[bytes] = []

        def send_binary(self, frame: bytes) -> None:
            self.frames.append(frame)

    socket = Socket()
    calls = 0

    def receive(_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"type": "prediction", "activate": calls == 2}

    responses = stream_timing_pcm(
        socket,
        pcm,
        receive,
        pace=False,
        stop_on_activation=True,
    )

    assert len(responses) == 2
    assert len(socket.frames) == 2
    assert len(b"".join(socket.frames)) < len(pcm)
    assert responses[-1]["activate"] is True


def test_stream_pcm_restores_pair_and_resends_the_unprocessed_frame() -> None:
    pcm = voiced_pcm(0.4)

    class Socket:
        def __init__(self) -> None:
            self.frames: list[bytes] = []

        def send_binary(self, frame: bytes) -> None:
            self.frames.append(frame)

    socket = Socket()
    replies = iter(
        [
            {"type": "pair_required"},
            {"type": "prediction", "activate": False},
            {"type": "prediction", "activate": False},
        ]
    )
    reconnects: list[bool] = []

    responses = stream_timing_pcm(
        socket,
        pcm,
        lambda _: next(replies),
        pace=False,
        on_pair_required=lambda: reconnects.append(True),
    )

    assert reconnects == [True]
    assert socket.frames[0] == socket.frames[1]
    assert b"".join(socket.frames[1:]) == pcm
    assert len(responses) == 2


@dataclass(frozen=True)
class FakeSettings:
    tts_mode: str = "browser"
    xunfei_app_id: str = ""
    xunfei_api_key: str = ""
    xunfei_api_secret: str = ""
    ffmpeg_binary: str = "ffmpeg"


def test_xunfei_credentials_are_env_only_and_provider_errors_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = {
        "XUNFEI_APP_ID": "secret-app-id",
        "XUNFEI_API_KEY": "secret-api-key",
        "XUNFEI_API_SECRET": "secret-api-secret",
    }
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)

    class FailingService:
        def __init__(self, settings: FakeSettings) -> None:
            assert settings.xunfei_app_id == credentials["XUNFEI_APP_ID"]
            assert settings.xunfei_api_key == credentials["XUNFEI_API_KEY"]
            assert settings.xunfei_api_secret == credentials["XUNFEI_API_SECRET"]

        def synthesize(self, text: str) -> bytes:
            raise RuntimeError(
                "signed-url=" + credentials["XUNFEI_API_SECRET"] + " text=" + text
            )

    with pytest.raises(RuntimeError) as captured:
        synthesize_xunfei_pcm(
            "credential redaction test",
            settings_factory=FakeSettings,
            service_factory=FailingService,
        )

    rendered = str(captured.value)
    assert "credential-bearing details were suppressed" in rendered
    assert captured.value.__cause__ is None
    assert all(secret not in rendered for secret in credentials.values())


def test_private_fixture_requires_personalized_positive_and_frozen_negative(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private-fixture.csv"
    fieldnames = [
        "participant_code",
        "participant_id",
        "internal_group",
        "smoke_scenario",
        "purpose",
        "analysis_use",
        "run_slug",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "participant_code": "PRIVATE-POSITIVE",
                    "participant_id": "private-positive-uuid",
                    "internal_group": "personalized",
                    "smoke_scenario": "quick_timing",
                    "purpose": "positive",
                    "analysis_use": "simulation_runner_smoke_v11",
                    "run_slug": RUN_SLUG,
                },
                {
                    "participant_code": "PRIVATE-NEGATIVE",
                    "participant_id": "private-negative-uuid",
                    "internal_group": "frozen",
                    "smoke_scenario": "long_quarantine",
                    "purpose": "negative",
                    "analysis_use": "simulation_runner_smoke_v11",
                    "run_slug": RUN_SLUG,
                },
            ]
        )

    run_slug, rows = load_private_fixture(path)

    assert run_slug == RUN_SLUG
    assert set(rows) == {"positive", "negative"}
    assert rows["negative"]["internal_group"] == "frozen"


def test_voiced_fixture_covers_both_groups_and_disables_in_finally(tmp_path: Path) -> None:
    pcm = voiced_pcm()
    calls: list[tuple[str, Any]] = []

    def synthesize(text: str) -> tuple[bytes, dict[str, Any]]:
        calls.append(("synthesize", text))
        return pcm, {"provider": "iflytek", "frame_bytes_max": 6_400}

    def positive(base_url: str, code: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(("positive", (base_url, code, kwargs)))
        return scenario_result("personalized", "positive")

    def negative(
        base_url: str, code: str, agent_state: str, **kwargs: Any
    ) -> dict[str, Any]:
        calls.append(("negative", (base_url, code, agent_state, kwargs)))
        return scenario_result("frozen", "negative")

    def disable(run_slug: str, data_dir: Path) -> dict[str, Any]:
        calls.append(("disable", (run_slug, data_dir)))
        return {"participant_count": 2, "enabled_after": 0}

    result = execute_voiced_fixture(
        "https://study.example",
        RUN_SLUG,
        fixture_rows(),
        tmp_path,
        "voiced test",
        synthesizer=synthesize,
        positive_runner=positive,
        negative_runner=negative,
        lifecycle_preflight=lambda slug, _, rows: {
            "run_slug": slug,
            "participant_count": len(rows),
        },
        disabler=disable,
    )

    assert result["ok"] is True
    assert result["fixture_cleanup"]["enabled_after"] == 0
    assert [name for name, _ in calls] == [
        "synthesize",
        "positive",
        "negative",
        "disable",
    ]
    positive_kwargs = calls[1][1][2]
    negative_kwargs = calls[2][1][3]
    assert positive_kwargs["voiced_pcm"] == pcm
    assert positive_kwargs["expect_training"] is True
    assert negative_kwargs["voiced_pcm"] == pcm
    assert negative_kwargs["expect_training"] is False
    assert result["personalized_positive"] is result["positive"]
    assert result["frozen_negative"] is result["negative"]


def test_voiced_fixture_still_disables_when_a_scenario_fails(tmp_path: Path) -> None:
    disabled: list[str] = []

    def fail_positive(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic scenario failure")

    with pytest.raises(RuntimeError, match="synthetic scenario failure"):
        execute_voiced_fixture(
            "https://study.example",
            RUN_SLUG,
            fixture_rows(),
            tmp_path,
            "voiced test",
            synthesizer=lambda _: (
                voiced_pcm(),
                {"provider": "iflytek", "frame_bytes_max": 6_400},
            ),
            positive_runner=fail_positive,
            negative_runner=lambda *args, **kwargs: {"ok": True},
            lifecycle_preflight=lambda slug, _, rows: {
                "run_slug": slug,
                "participant_count": len(rows),
            },
            disabler=lambda slug, _: disabled.append(slug)
            or {"participant_count": 2, "enabled_after": 0},
        )

    assert disabled == [RUN_SLUG]


def test_four_way_fixture_crosses_group_and_feedback_without_confounding() -> None:
    design, cells = fixture_cells(four_way_fixture_rows())

    assert design == "four-way"
    assert cells == {
        "personalized_positive": ("personalized", "quick_timing"),
        "personalized_negative": ("personalized", "long_quarantine"),
        "frozen_positive": ("frozen", "quick_timing"),
        "frozen_negative": ("frozen", "long_quarantine"),
    }


def test_four_way_private_csv_requires_exact_purpose_group_mapping(
    tmp_path: Path,
) -> None:
    rows = list(four_way_fixture_rows().values())
    path = tmp_path / "four-way.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run_slug, loaded = load_private_fixture(path)

    assert run_slug == RUN_SLUG
    assert set(loaded) == set(four_way_fixture_rows())

    rows[0]["internal_group"] = "frozen"
    invalid_path = tmp_path / "four-way-swapped.csv"
    with invalid_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="must use the personalized group"):
        load_private_fixture(invalid_path)


def test_lifecycle_preflight_binds_each_uuid_group_assignment_and_zero_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = four_way_fixture_rows()
    participants = [
        {
            "id": row["participant_id"],
            "assignment_id": row["assignment_id"],
            "group_name": row["internal_group"],
            "run_slug": RUN_SLUG,
            "enabled": 1,
        }
        for row in rows.values()
    ]

    class Cursor:
        def __init__(self, count: int) -> None:
            self.count = count

        def fetchone(self) -> dict[str, int]:
            return {"session_count": self.count}

    class Connection:
        def __init__(self, count: int) -> None:
            self.count = count

        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, parameters: tuple[str]) -> Cursor:
            assert "FROM sessions WHERE run_id=?" in sql
            assert parameters == ("run-id",)
            return Cursor(self.count)

    class DB:
        session_count = 0

        def participants_in_run(self, run_id: str) -> list[dict[str, Any]]:
            assert run_id == "run-id"
            return [dict(participant) for participant in participants]

        def connect(self) -> Connection:
            return Connection(self.session_count)

    db = DB()
    monkeypatch.setattr(
        voiced_runner,
        "_local_fixture_db",
        lambda slug, data_dir: (db, {"id": "run-id", "slug": slug}),
    )

    result = validate_local_fixture_lifecycle(RUN_SLUG, tmp_path, rows)
    assert result["participant_count"] == 4
    assert result["session_count"] == 0

    participants[0]["group_name"] = "frozen"
    with pytest.raises(RuntimeError, match="exact four-way group layout"):
        validate_local_fixture_lifecycle(RUN_SLUG, tmp_path, rows)
    participants[0]["group_name"] = "personalized"

    participants[0]["assignment_id"] = "WRONG"
    with pytest.raises(RuntimeError, match="assignment id mismatch"):
        validate_local_fixture_lifecycle(RUN_SLUG, tmp_path, rows)
    participants[0]["assignment_id"] = rows["personalized_positive"]["assignment_id"]

    db.session_count = 1
    with pytest.raises(RuntimeError, match="sessions already exist"):
        validate_local_fixture_lifecycle(RUN_SLUG, tmp_path, rows)


def test_four_way_voiced_fixture_runs_all_cells_and_disables_once(tmp_path: Path) -> None:
    calls: list[tuple[str, str, bool]] = []
    disabled: list[str] = []

    def positive(_base: str, code: str, **kwargs: Any) -> dict[str, Any]:
        group = "personalized" if "personalized" in code else "frozen"
        calls.append((group, "positive", kwargs["expect_training"]))
        assert max(len(frame) for frame in iter_timing_pcm_frames(kwargs["voiced_pcm"])) <= 6_400
        return scenario_result(group, "positive")

    def negative(
        _base: str, code: str, _state: str, **kwargs: Any
    ) -> dict[str, Any]:
        group = "personalized" if "personalized" in code else "frozen"
        calls.append((group, "negative", kwargs["expect_training"]))
        return scenario_result(group, "negative")

    result = execute_voiced_fixture(
        "https://study.example",
        RUN_SLUG,
        four_way_fixture_rows(),
        tmp_path,
        "four-way voice test",
        synthesizer=lambda _: (
            voiced_pcm(),
            {"provider": "iflytek", "frame_bytes_max": 6_400},
        ),
        positive_runner=positive,
        negative_runner=negative,
        lifecycle_preflight=lambda slug, _, rows: {
            "run_slug": slug,
            "participant_count": len(rows),
            "fixture_design": "four-way",
        },
        disabler=lambda slug, _: disabled.append(slug)
        or {"participant_count": 4, "enabled_after": 0},
    )

    assert calls == [
        ("personalized", "positive", True),
        ("personalized", "negative", True),
        ("frozen", "positive", False),
        ("frozen", "negative", False),
    ]
    assert disabled == [RUN_SLUG]
    assert result["fixture_design"] == "four-way"
    assert set(four_way_fixture_rows()).issubset(result)


def test_voiced_fixture_disables_even_when_preflight_fails(tmp_path: Path) -> None:
    disabled: list[str] = []

    with pytest.raises(RuntimeError, match="sessions already exist"):
        execute_voiced_fixture(
            "https://study.example",
            RUN_SLUG,
            four_way_fixture_rows(),
            tmp_path,
            "four-way voice test",
            lifecycle_preflight=lambda *_: (_ for _ in ()).throw(
                RuntimeError("sessions already exist")
            ),
            disabler=lambda slug, _: disabled.append(slug)
            or {"participant_count": 4, "enabled_after": 0},
        )

    assert disabled == [RUN_SLUG]


def test_policy_assertion_rejects_any_frozen_update() -> None:
    invalid = scenario_result("frozen", "positive")
    invalid["training_applied"] = True

    with pytest.raises(RuntimeError, match="frozen_positive policy assertion failed"):
        validate_scenario_result("frozen_positive", "frozen", invalid)


def test_voiced_runner_source_never_loads_dotenv_or_persists_audio() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    voice_source = (scripts / "voice_pcm.py").read_text(encoding="utf-8")
    runner_source = (scripts / "smoke_xunfei_voiced_public.py").read_text(
        encoding="utf-8"
    )

    assert "load_dotenv" not in voice_source + runner_source
    assert "NamedTemporaryFile" not in voice_source + runner_source
    assert "write_bytes" not in voice_source + runner_source
    assert '"pipe:0"' in voice_source
    assert '"pipe:1"' in voice_source
    assert "--acknowledge-one-shot-fixture" in runner_source
