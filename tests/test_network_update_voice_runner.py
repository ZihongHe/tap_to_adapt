"""Regression tests for network update voice runner.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import array
import csv
from pathlib import Path
from typing import Any

import pytest

from scripts.provision_runner_smoke_accounts import build_rows
from scripts.smoke_xunfei_network_update_public import (
    ANALYSIS_USE,
    execute_fixture,
    load_private_fixture,
    validate_feedback_sequence,
)
from scripts.voice_pcm import PCM_SAMPLE_RATE_HZ, iter_timing_pcm_frames


RUN_SLUG = "simulation-runner-smoke-v11-1784451600789-c2d3e4f5"


def pcm(seconds: float = 0.6) -> bytes:
    samples = array.array(
        "h",
        (
            4_000 if index % 2 else -4_000
            for index in range(int(PCM_SAMPLE_RATE_HZ * seconds))
        ),
    )
    return samples.tobytes()


def fixture_rows() -> dict[str, dict[str, str]]:
    rows = build_rows(
        [
            ("TTA-NET-PERS", "personalized-uuid", "QSMKNETP"),
            ("TTA-NET-FROZ", "frozen-uuid", "QSMKNETF"),
        ],
        RUN_SLUG,
        "network-update",
    )
    return {row["purpose"]: row for row in rows}


def personalized_results() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, scale in enumerate((0.0, 0.0, 0.05)):
        results.append(
            {
                "action": "manual_activate",
                "label": 1,
                "training_eligible": True,
                "training_applied": True,
                "personal_calibration_applied": True,
                "network_update_scale": scale,
                "current_network_update_applied": index == 2,
                "requested_training_steps": 1 if index == 2 else 0,
                "training_steps": 1 if index == 2 else 0,
                "optimizer_step_before": 0,
                "optimizer_step_after": 1 if index == 2 else 0,
                "parameter_state_interpolated": index == 2,
                "parameter_interpolation_scale": scale if index == 2 else 1.0,
                "optimizer_learning_rates_before": [0.001] if index == 2 else [],
                "optimizer_learning_rates_during": [0.00005] if index == 2 else [],
                "optimizer_learning_rates_after": [0.001] if index == 2 else [],
                "loss": 0.25 if index == 2 else 0.0,
            }
        )
    return results


def frozen_results() -> list[dict[str, Any]]:
    return [
        {
            "action": "manual_activate",
            "label": 1,
            "training_eligible": False,
            "training_applied": False,
            "quarantined": True,
            "quarantine_reason": "activation_not_learning_current",
        }
        for _ in range(3)
    ]


def test_network_update_provision_rows_are_isolated_from_existing_designs() -> None:
    rows = fixture_rows()

    assert set(rows) == {
        "network_update_personalized",
        "network_update_frozen",
    }
    assert rows["network_update_personalized"]["internal_group"] == "personalized"
    assert rows["network_update_frozen"]["internal_group"] == "frozen"
    assert all(row["fixture_design"] == "network-update" for row in rows.values())
    assert all(row["analysis_use"] == ANALYSIS_USE for row in rows.values())


def test_network_update_private_fixture_requires_exact_metadata(tmp_path: Path) -> None:
    rows = list(fixture_rows().values())
    path = tmp_path / "network-update.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    run_slug, loaded = load_private_fixture(path)

    assert run_slug == RUN_SLUG
    assert set(loaded) == {
        "network_update_personalized",
        "network_update_frozen",
    }

    rows[0]["analysis_use"] = "simulation_runner_smoke_v11"
    invalid = tmp_path / "invalid-network-update.csv"
    with invalid.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Invalid network_update_personalized"):
        load_private_fixture(invalid)


def test_personalized_sequence_requires_third_feedback_network_update() -> None:
    validate_feedback_sequence("personalized", personalized_results())

    invalid = personalized_results()
    invalid[2]["current_network_update_applied"] = False
    with pytest.raises(RuntimeError, match="tap3.current_network_update_applied"):
        validate_feedback_sequence("personalized", invalid)


def test_personalized_sequence_rejects_optimizer_or_schedule_drift() -> None:
    invalid = personalized_results()
    invalid[1]["network_update_scale"] = 0.05
    invalid[2]["optimizer_step_after"] = 2

    with pytest.raises(RuntimeError) as captured:
        validate_feedback_sequence("personalized", invalid)

    assert "tap2.network_update_scale" in str(captured.value)
    assert "tap3.optimizer" in str(captured.value)


def test_frozen_sequence_rejects_any_update() -> None:
    validate_feedback_sequence("frozen", frozen_results())

    invalid = frozen_results()
    invalid[2]["optimizer_step_after"] = 1
    with pytest.raises(RuntimeError, match="tap3.optimizer"):
        validate_feedback_sequence("frozen", invalid)


def test_execute_fixture_uses_one_pcm_for_both_cells_and_always_disables(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Any]] = []
    audio = pcm()

    def account_runner(
        _base_url: str,
        row: dict[str, str],
        voiced_pcm: bytes,
        expected_data_dir: Path,
    ) -> dict[str, Any]:
        calls.append((row["purpose"], expected_data_dir))
        assert voiced_pcm == audio
        assert max(map(len, iter_timing_pcm_frames(voiced_pcm))) <= 6_400
        return {"ok": True, "group": row["internal_group"]}

    result = execute_fixture(
        "https://study.example",
        RUN_SLUG,
        fixture_rows(),
        tmp_path,
        "network update",
        synthesizer=lambda _: (
            audio,
            {"provider": "iflytek", "frame_bytes_max": 6_400},
        ),
        account_runner=account_runner,
        lifecycle_preflight=lambda slug, _, rows: {
            "run_slug": slug,
            "participant_count": len(rows),
            "agent_mode": "dtcn",
        },
        disabler=lambda slug, _: calls.append(("disable", slug))
        or {"participant_count": 2, "enabled_after": 0},
    )

    assert result["ok"] is True
    assert [name for name, _ in calls] == [
        "network_update_personalized",
        "network_update_frozen",
        "disable",
    ]


def test_execute_fixture_disables_when_preflight_or_account_fails(
    tmp_path: Path,
) -> None:
    disabled: list[str] = []

    with pytest.raises(RuntimeError, match="stale fixture"):
        execute_fixture(
            "https://study.example",
            RUN_SLUG,
            fixture_rows(),
            tmp_path,
            "network update",
            lifecycle_preflight=lambda *_: (_ for _ in ()).throw(
                RuntimeError("stale fixture")
            ),
            disabler=lambda slug, _: disabled.append(slug)
            or {"participant_count": 2, "enabled_after": 0},
        )
    assert disabled == [RUN_SLUG]

    disabled.clear()
    with pytest.raises(RuntimeError, match="public scenario failed"):
        execute_fixture(
            "https://study.example",
            RUN_SLUG,
            fixture_rows(),
            tmp_path,
            "network update",
            synthesizer=lambda _: (
                pcm(),
                {"provider": "iflytek", "frame_bytes_max": 6_400},
            ),
            lifecycle_preflight=lambda *_: {"agent_mode": "dtcn"},
            account_runner=lambda *_: (_ for _ in ()).throw(
                RuntimeError("public scenario failed")
            ),
            disabler=lambda slug, _: disabled.append(slug)
            or {"participant_count": 2, "enabled_after": 0},
        )
    assert disabled == [RUN_SLUG]


def test_network_update_runner_never_loads_dotenv_or_persists_audio() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "smoke_xunfei_network_update_public.py"
    ).read_text(encoding="utf-8")

    assert "load_dotenv" not in source
    assert "write_bytes" not in source
    assert "NamedTemporaryFile" not in source
    assert "XUNFEI_API" not in source
    assert "--acknowledge-one-shot-fixture" in source
