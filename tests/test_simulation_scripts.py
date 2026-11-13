"""Regression tests for simulation scripts.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pytest

from scripts.simulate_feasibility import SimulationError, simulated_events, validate_roster_rows
from scripts.provision_runner_smoke_accounts import (
    build_rows as build_runner_smoke_rows,
    create_private_placeholder,
    runner_smoke_specs,
    validate_runner_smoke_run_slug,
    validate_private_output_path,
)
from scripts.disable_simulation_run import (
    disable_runner_smoke_fixture,
    is_canonical_runner_smoke_run_slug,
)
from scripts.smoke_negative_feedback_public import cleanup_smoke_run, pcm_tone, receive_json
from scripts.smoke_timing_public import pcm_tone as timing_pcm_tone
from scripts.validate_calibration_lock import validate


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def roster_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    blocks = [
        ("A", 12, "study_a", "seed_donor"),
        ("A", 16, "study_a", "evaluator_donor"),
        ("A", 8, "study_a", "holdout_evaluator"),
        ("BP", 18, "study_b", "personalized"),
        ("BF", 18, "study_b", "frozen"),
    ]
    counters: dict[str, int] = {}
    for prefix, count, cohort, group in blocks:
        for index in range(1, count + 1):
            counters[prefix] = counters.get(prefix, 0) + 1
            rows.append(
                {
                    "assignment_id": f"{prefix}{counters[prefix]:02d}",
                    "participant_code": f"TTA-{len(rows):04d}-SIMX",
                    "cohort": cohort,
                    "internal_group": group,
                    "data_origin": "simulation",
                    "protocol_mode": "compressed",
                    "run_slug": "simulation-test-v6",
                    "day1_probe_order": "ABC",
                    "day2_probe_order": "BCA",
                    "day3_probe_order": "CAB",
                }
            )
    return rows


def test_public_feedback_smokes_use_atomic_conditional_taps() -> None:
    positive = (SCRIPTS / "smoke_positive_feedback_public.py").read_text(
        encoding="utf-8"
    )
    negative = (SCRIPTS / "smoke_negative_feedback_public.py").read_text(
        encoding="utf-8"
    )
    timing = (SCRIPTS / "smoke_timing_public.py").read_text(encoding="utf-8")

    for script in (positive, negative, timing):
        assert '"intended_action"' in script
        assert '"expected_state"' in script
        assert '"expected_activation_id"' in script
        assert '"expected_state_version"' in script
        assert '"monotonic_ms": time.perf_counter() * 1000' in script

    assert '"intended_action": "manual_activate"' in positive
    assert '"/api/chat"' in positive
    assert '"/api/tts"' in positive
    assert '"tts_bytes": len(tts_audio)' in positive
    assert '"intended_action": "interrupt"' in negative
    assert '"action") != "conditional_tap_cancelled"' in negative
    assert "pcm_tone(0.6)" not in positive
    assert "pcm_tone(0.6)" not in negative
    assert '"monotonic_ms": 1000' not in timing


def test_public_feedback_smoke_pcm_frame_stays_below_production_limit() -> None:
    assert len(pcm_tone(0.2)) == 6_400
    assert len(pcm_tone(0.2)) <= 8_192
    assert len(timing_pcm_tone()) == 6_400
    assert len(timing_pcm_tone()) <= 8_192


def test_public_feedback_smoke_reports_early_websocket_close() -> None:
    class ClosedSocket:
        def recv(self) -> str:
            return ""

    with pytest.raises(RuntimeError, match="maximum frame size"):
        receive_json(ClosedSocket())  # type: ignore[arg-type]


def test_smoke_cleanup_attempts_every_socket_and_session_abort(monkeypatch) -> None:
    class Socket:
        def __init__(self, failure: Exception | None = None):
            self.failure = failure
            self.closed = False

        def close(self) -> None:
            self.closed = True
            if self.failure is not None:
                raise self.failure

    good = Socket()
    bad = Socket(RuntimeError("synthetic close failure"))
    abort_calls: list[tuple[str, str, str]] = []

    def abort(base_url: str, path: str, *, method: str, token: str):
        abort_calls.append((base_url, path, method))
        assert token == "opaque-study-token"
        return {"ended": True}

    monkeypatch.setattr("scripts.smoke_negative_feedback_public.request_json", abort)
    errors = cleanup_smoke_run(
        "https://study.example",
        "opaque-study-token",
        "session-id",
        (("bad", bad), ("good", good), ("missing", None)),
    )

    assert bad.closed is True
    assert good.closed is True
    assert abort_calls == [
        ("https://study.example", "/api/sessions/session-id/abort", "POST")
    ]
    assert len(errors) == 1
    assert errors[0].startswith("bad: RuntimeError")


def test_simulation_roster_rejects_formal_accounts_before_execution() -> None:
    rows = roster_rows()
    rows[0]["data_origin"] = "formal"
    rows[0]["protocol_mode"] = "strict"
    with pytest.raises(SimulationError, match="immutable simulation/compressed"):
        validate_roster_rows(rows)


def test_simulation_roster_accepts_only_complete_isolated_allocation() -> None:
    assert validate_roster_rows(roster_rows()) == "simulation-test-v6"


def test_simulated_events_match_v7_seed_phase_tokens() -> None:
    events = simulated_events("A01", "study_a", "seed_donor", 1, "BCA")
    starts = [
        event["payload"]
        for event in events
        if event["event_type"] == "phase.start"
    ]
    assert [(item["phase"], item["policy_token"]) for item in starts] == [
        ("frozen", "A"),
        ("learning", "A"),
        ("learning", "A"),
        ("learning", "A"),
        ("learning", "A"),
        ("micro_probe", "B"),
        ("micro_probe", "C"),
        ("micro_probe", "A"),
    ]


def test_current_calibration_lock_has_no_label_quota_and_passes_mechanical_checks() -> None:
    protocol_path = Path(__file__).resolve().parents[1] / "content" / "study_protocol.json"
    result = validate(json.loads(protocol_path.read_text(encoding="utf-8")))

    assert result["passed"] is True
    assert result["checks"]["runtime_label_quota_is_disabled"] is True
    assert result["checks"]["personal_intercept_excluded_from_shared_prior"] is True


def test_runner_smoke_fixture_is_fresh_one_shot_and_scenario_driven() -> None:
    rows = build_runner_smoke_rows(
        [
            ("TTA-AAAA-BBBB", "participant-uuid-1", "QSMKTEST0001"),
            ("TTA-CCCC-DDDD", "participant-uuid-2", "QSMKTEST0002"),
        ],
        "simulation-runner-smoke-v11-123",
    )

    assert [row["smoke_scenario"] for row in rows] == [
        "quick_timing",
        "long_quarantine",
    ]
    assert [row["purpose"] for row in rows] == ["positive", "negative"]
    assert all(row["analysis_use"] == "simulation_runner_smoke_v11" for row in rows)
    assert [row["internal_group"] for row in rows] == ["personalized", "frozen"]
    assert all(row["run_slug"] == "simulation-runner-smoke-v11-123" for row in rows)
    assert len({row["assignment_id"] for row in rows}) == 2
    assert [row["participant_id"] for row in rows] == [
        "participant-uuid-1",
        "participant-uuid-2",
    ]
    assert all(row["fixture_design"] == "pair" for row in rows)


def test_runner_smoke_four_way_fixture_crosses_group_and_feedback() -> None:
    accounts = [
        (f"TTA-FOUR-{index}", f"participant-uuid-{index}", f"QSMKFOUR{index}")
        for index in range(1, 5)
    ]

    rows = build_runner_smoke_rows(
        accounts,
        "simulation-runner-smoke-v11-123",
        "four-way",
    )

    assert runner_smoke_specs("four-way") == [
        ("personalized", "personalized_positive", "quick_timing", "impatient"),
        ("personalized", "personalized_negative", "long_quarantine", "ordinary"),
        ("frozen", "frozen_positive", "quick_timing", "impatient"),
        ("frozen", "frozen_negative", "long_quarantine", "ordinary"),
    ]
    assert [row["purpose"] for row in rows] == [
        "personalized_positive",
        "personalized_negative",
        "frozen_positive",
        "frozen_negative",
    ]
    assert all(row["fixture_design"] == "four-way" for row in rows)
    assert len({row["participant_id"] for row in rows}) == 4


def test_runner_smoke_four_way_rejects_partial_account_list() -> None:
    with pytest.raises(ValueError, match="requires exactly 4"):
        build_runner_smoke_rows(
            [
                ("TTA-ONE", "participant-one", "QSMKONE"),
                ("TTA-TWO", "participant-two", "QSMKTWO"),
            ],
            "simulation-runner-smoke-v11-123",
            "four-way",
        )


def test_runner_smoke_credentials_cannot_be_written_under_static(tmp_path: Path) -> None:
    static = tmp_path / "static"
    static.mkdir()
    with pytest.raises(ValueError, match="public directory"):
        validate_private_output_path(static / "qa.csv", [static])
    validate_private_output_path(tmp_path / "private" / "qa.csv", [static])


def test_runner_smoke_private_placeholder_is_exclusive(tmp_path: Path) -> None:
    target = tmp_path / "qa.csv"
    create_private_placeholder(target)
    assert target.exists()
    if os.name != "nt":
        assert target.stat().st_mode & 0o077 == 0
    with pytest.raises(FileExistsError):
        create_private_placeholder(target)


def test_runner_smoke_accepts_strict_caller_known_run_slug() -> None:
    assert (
        validate_runner_smoke_run_slug(
            "simulation-runner-smoke-v11-1784451600123-a1b2c3d4"
        )
        == "A1B2C3D4"
    )


@pytest.mark.parametrize(
    "run_slug",
    [
        "simulation-runner-smoke-v11-178445160012-a1b2c3d4",
        "simulation-runner-smoke-v11-17844516001234-a1b2c3d4",
        "simulation-runner-smoke-v11-1784451600123-A1B2C3D4",
        "simulation-runner-smoke-v11-1784451600123-a1b2c3d",
        "simulation-runner-smoke-v10-1784451600123-a1b2c3d4",
        "simulation-runner-smoke-v11-1784451600123-a1b2c3d4-extra",
    ],
)
def test_runner_smoke_rejects_noncanonical_run_slug(run_slug: str) -> None:
    with pytest.raises(ValueError, match="run slug must match"):
        validate_runner_smoke_run_slug(run_slug)


def test_runner_smoke_disable_accepts_only_canonical_run_slug() -> None:
    assert is_canonical_runner_smoke_run_slug(
        "simulation-runner-smoke-v11-1784451600123-a1b2c3d4"
    )
    assert not is_canonical_runner_smoke_run_slug(
        "simulation-runner-smoke-v11-1784451600123-a1b2c3d4-extra"
    )
    assert not is_canonical_runner_smoke_run_slug(
        "simulation-runner-smoke-v11-1784451600123-A1B2C3D4"
    )


def test_runner_smoke_disable_core_rejects_noncanonical_slug() -> None:
    db = _FakeSmokeDB([], run_exists=False)

    with pytest.raises(ValueError, match="canonical runner-smoke-v11 namespace"):
        disable_runner_smoke_fixture(
            db,
            "simulation-runner-smoke-v11-1784451600123-a1b2c3d4-extra",
        )


def test_runner_smoke_disable_rejects_foreign_data_origin() -> None:
    db = _FakeSmokeDB([], data_origin="formal")

    with pytest.raises(ValueError, match="Only a simulation run"):
        disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert db.actions == []


class _FakeCursor:
    def __init__(self, *, rowcount: int = -1, row: dict[str, int] | None = None) -> None:
        self.rowcount = rowcount
        self._row = row

    def fetchone(self) -> dict[str, int] | None:
        return self._row


class _FakeConnection(AbstractContextManager["_FakeConnection"]):
    def __init__(self, participants: list[dict[str, Any]]) -> None:
        self.participants = participants

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, parameters: tuple[str]) -> _FakeCursor:
        assert parameters == ("run-id",)
        normalized = " ".join(sql.split())
        if normalized.startswith("UPDATE participants SET enabled=0"):
            disabled = sum(
                1 for participant in self.participants if participant["enabled"] == 1
            )
            for participant in self.participants:
                participant["enabled"] = 0
            return _FakeCursor(rowcount=disabled)
        if normalized.startswith("SELECT COUNT(*) AS enabled_count"):
            enabled = sum(
                1 for participant in self.participants if participant["enabled"] == 1
            )
            return _FakeCursor(row={"enabled_count": enabled})
        raise AssertionError(f"Unexpected SQL: {normalized}")


class _FakeSmokeDB:
    def __init__(
        self,
        participants: list[dict[str, Any]],
        *,
        run_exists: bool = True,
        data_origin: str = "simulation",
    ) -> None:
        self.participants = participants
        self.run_exists = run_exists
        self.data_origin = data_origin
        self.actions: list[dict[str, Any]] = []

    def study_run(self, run_slug: str) -> dict[str, str] | None:
        if not self.run_exists:
            return None
        return {"id": "run-id", "slug": run_slug, "data_origin": self.data_origin}

    def participants_in_run(self, run_id: str) -> list[dict[str, Any]]:
        assert run_id == "run-id"
        return [dict(participant) for participant in self.participants]

    def connect(self) -> _FakeConnection:
        return _FakeConnection(self.participants)

    def record_pipeline_action(
        self,
        run_id: str,
        action: str,
        *,
        actor: str,
        payload: dict[str, Any],
    ) -> None:
        self.actions.append(
            {
                "run_id": run_id,
                "action": action,
                "actor": actor,
                "payload": payload,
            }
        )


RUNNER_SMOKE_SLUG = "simulation-runner-smoke-v11-1784451600123-a1b2c3d4"


@pytest.mark.parametrize("run_exists", [False, True])
def test_runner_smoke_disable_allows_zero_participant_pending_cleanup(
    run_exists: bool,
) -> None:
    db = _FakeSmokeDB([], run_exists=run_exists)

    result = disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert result == {
        "run_slug": RUNNER_SMOKE_SLUG,
        "participant_count": 0,
        "enabled_before": 0,
        "disabled_now": 0,
        "enabled_after": 0,
    }
    assert len(db.actions) == int(run_exists)


def test_runner_smoke_disable_is_idempotent_for_valid_two_account_fixture() -> None:
    db = _FakeSmokeDB(
        [
            {"group_name": "qa_runner_smoke", "enabled": 1},
            {"group_name": "qa_runner_smoke", "enabled": 1},
        ]
    )

    first = disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)
    replay = disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert first["participant_count"] == 2
    assert (first["enabled_before"], first["disabled_now"], first["enabled_after"]) == (
        2,
        2,
        0,
    )
    assert (replay["enabled_before"], replay["disabled_now"], replay["enabled_after"]) == (
        0,
        0,
        0,
    )
    assert db.actions[-1]["payload"]["idempotent_replay"] is True


def test_runner_smoke_disable_accepts_personalized_frozen_pair() -> None:
    db = _FakeSmokeDB(
        [
            {"group_name": "personalized", "enabled": 1},
            {"group_name": "frozen", "enabled": 1},
        ]
    )

    result = disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert result["participant_count"] == 2
    assert result["disabled_now"] == 2
    assert result["enabled_after"] == 0


def test_runner_smoke_disable_accepts_only_exact_four_way_group_layout() -> None:
    db = _FakeSmokeDB(
        [
            {"group_name": "personalized", "enabled": 1},
            {"group_name": "personalized", "enabled": 1},
            {"group_name": "frozen", "enabled": 1},
            {"group_name": "frozen", "enabled": 1},
        ]
    )

    result = disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert result["participant_count"] == 4
    assert result["disabled_now"] == 4
    assert result["enabled_after"] == 0


def test_runner_smoke_disable_rejects_wrong_four_account_group_balance() -> None:
    participants = [
        {"group_name": "personalized", "enabled": 1},
        {"group_name": "personalized", "enabled": 1},
        {"group_name": "personalized", "enabled": 1},
        {"group_name": "frozen", "enabled": 1},
    ]
    db = _FakeSmokeDB(participants)

    with pytest.raises(ValueError, match="exact two-personalized/two-frozen"):
        disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert all(participant["enabled"] == 1 for participant in participants)


@pytest.mark.parametrize(
    "participants",
    [
        [{"group_name": "qa_runner_smoke", "enabled": 1}],
        [
            {"group_name": "qa_runner_smoke", "enabled": 1},
            {"group_name": "qa_runner_smoke", "enabled": 1},
            {"group_name": "qa_runner_smoke", "enabled": 1},
        ],
        [
            {"group_name": "qa_runner_smoke", "enabled": 1},
            {"group_name": "other", "enabled": 1},
        ],
    ],
)
def test_runner_smoke_disable_rejects_partial_oversized_or_foreign_fixture(
    participants: list[dict[str, Any]],
) -> None:
    db = _FakeSmokeDB(participants)

    with pytest.raises(ValueError, match="zero pending participants or exactly two"):
        disable_runner_smoke_fixture(db, RUNNER_SMOKE_SLUG)

    assert all(participant["enabled"] == 1 for participant in participants)
    assert db.actions == []
