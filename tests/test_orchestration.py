"""Regression tests for orchestration.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import os
import sys
import tempfile
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

TEST_DATA = Path(tempfile.mkdtemp(prefix="tta-orch-unit-"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STUDY_DATA_DIR", str(TEST_DATA))
os.environ.setdefault("STUDY_TOKEN_SECRET", "test-token-secret")
os.environ.setdefault("STUDY_CODE_PEPPER", "test-code-pepper")
os.environ.setdefault("STUDY_IP_PEPPER", "test-ip-pepper")
os.environ.setdefault("STUDY_ADMIN_TOKEN", "test-admin-token")
os.environ.setdefault(
    "CONSENT_EVIDENCE_SIGNING_KEY",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
)
os.environ.setdefault(
    "COMPENSATION_VAULT_KEY",
    "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
)

import pytest
import numpy as np

from app.config import Settings
from app.db import StudyDB
from app.protocol import Protocol
from app.study_state import StudyState
import app.orchestration as orchestration_module
import app.session_lock as session_lock_module
from app.orchestration import (
    ARM_COLD_LEARN,
    ARM_SHARED_FROZEN,
    ARM_SHARED_LEARN,
    AutoOrchestrator,
    OrchestrationError,
    SHARE_ARRIVAL_ENTRY_MS,
    SHARE_SETUP_TIMEOUT_MS,
    SHARE_WALK_IN_LEASE_MS,
    ShareAppointmentError,
    _behavior_fit_summary,
    _share_fit_remaining_estimate,
    _share_fit_stop_reason,
    _share_fit_series_trend,
    _spearman_trend,
)


def _mark_end_to_end_complete(db: StudyDB, participant_id: str) -> None:
    study_session = db.start_session(participant_id, 1, "study")
    companion_session = db.start_session(participant_id, 1, "companion")
    if (
        study_session.get("activated_ms") is None
        and str(study_session.get("state") or "")
        not in {"completed", "ended", "aborted"}
    ):
        try:
            db.activate_paired_sessions(
                participant_id, study_session["id"], companion_session["id"]
            )
        except ValueError:
            pass
        study_session = db.session(study_session["id"], participant_id) or study_session
    participant = db.participant(participant_id)
    assert participant is not None
    rules = db._timed_interaction_rules(
        participant.get("run_protocol_json"),
        str(participant.get("protocol_mode") or "strict"),
    )
    min_ms = int(float(rules["min_session_seconds"]) * 1000)
    now_ms = int(time.time() * 1000)
    activated_ms = now_ms - max(min_ms, 1) - 60_000
    with db.connect() as connection:
        connection.execute(
            """UPDATE sessions
               SET activated_ms=?, state='active', ended_ms=NULL
               WHERE id IN (?, ?)""",
            (activated_ms, study_session["id"], companion_session["id"]),
        )
        connection.commit()
    schedule = rules["block_schedule"]
    if schedule.get("enabled"):
        blocks = int(schedule.get("active_blocks") or 0)
        include_final_rest = bool(schedule.get("rest_after_final_block"))
        for block in range(1, blocks + 1):
            for event_type, extra in (
                ("phase.start", {"phase": "learning", "analysis_eligible": True}),
                ("phase.end", {"phase": "learning", "analysis_eligible": True}),
                (
                    "companion.window_start",
                    {"window": block, "analysis_eligible": True},
                ),
                (
                    "companion.window_end",
                    {"window": block, "analysis_eligible": True},
                ),
            ):
                db.append_event(
                    participant_id,
                    study_session["id"],
                    event_type,
                    None,
                    None,
                    {"block": block, **extra},
                )
            if block < blocks or include_final_rest:
                rest = {
                    "block": block,
                    "excluded_from_analysis": True,
                    "analysis_eligible": False,
                }
                db.append_event(
                    participant_id,
                    study_session["id"],
                    "rest.start",
                    None,
                    None,
                    rest,
                )
                db.append_event(
                    participant_id,
                    study_session["id"],
                    "rest.end",
                    None,
                    None,
                    rest,
                )
    db.end_session(study_session["id"], participant_id)


def test_contrast_assignment_cycles_three_arms(tmp_path: Path) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data",
        public_enrollment_run_slug="orch-cycle-run",
        public_enrollment_cohort="study_a",
        public_enrollment_group="personalized",
        instance_id="formal",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    state = orch._load()
    state["phase"] = "contrast"
    state["share_fit"]["locked_n"] = 2
    state["share_fit"]["locked_prior_id"] = "prior-lock"
    orch._write(state)

    arms = []
    for index in range(6):
        assignment = orch.assign_enrollment()
        arms.append(assignment["arm"])
        orch.commit_enrollment(
            participant_id=f"p-{index}",
            assignment=assignment,
        )
    assert arms == [
        ARM_COLD_LEARN,
        ARM_SHARED_LEARN,
        ARM_SHARED_FROZEN,
        ARM_COLD_LEARN,
        ARM_SHARED_LEARN,
        ARM_SHARED_FROZEN,
    ]
    status = orch.status()
    assert status["contrast"]["counts"] == {
        ARM_COLD_LEARN: 2,
        ARM_SHARED_LEARN: 2,
        ARM_SHARED_FROZEN: 2,
    }
    assert status["contrast"]["next_arm"] == ARM_COLD_LEARN


def test_contrast_assignment_fills_the_shortest_completed_arm(
    tmp_path: Path,
) -> None:
    from app.orchestration import choose_contrast_arm

    assert choose_contrast_arm(
        {"cold_learn": 1, "shared_learn": 1, "shared_frozen": 0},
        {"cold_learn": 2, "shared_learn": 1, "shared_frozen": 1},
        {"cold_learn": 32, "shared_learn": 32, "shared_frozen": 32},
    ) == ARM_SHARED_FROZEN
    assert choose_contrast_arm(
        {"cold_learn": 1, "shared_learn": 0, "shared_frozen": 0},
        {"cold_learn": 1, "shared_learn": 1, "shared_frozen": 0},
        {"cold_learn": 32, "shared_learn": 32, "shared_frozen": 32},
    ) == ARM_SHARED_FROZEN

    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-shortest-arm",
        public_enrollment_run_slug="orch-shortest-arm-run",
        public_enrollment_cohort="study_a",
        public_enrollment_group="personalized",
        instance_id="formal",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    state = orch._load()
    state["phase"] = "contrast"
    state["share_fit"]["locked_n"] = 2
    state["share_fit"]["locked_prior_id"] = "prior-lock"
    orch._write(state)

    codes = db.create_participants(
        2,
        "pilot",
        "personalized",
        data_origin="pilot",
        protocol_mode="strict",
        run_slug="orch-shortest-arm-run",
    )
    people = [db.participant_by_code(code) for code in codes]
    assert all(person is not None for person in people)
    first = orch.assign_enrollment()
    orch.commit_enrollment(participant_id=people[0]["id"], assignment=first)
    assert first["arm"] == ARM_COLD_LEARN
    second = orch.assign_enrollment()
    orch.commit_enrollment(participant_id=people[1]["id"], assignment=second)
    assert second["arm"] == ARM_SHARED_LEARN
    _mark_end_to_end_complete(db, people[0]["id"])
    _mark_end_to_end_complete(db, people[1]["id"])
    orch.on_session_completed(people[0]["id"])
    orch.on_session_completed(people[1]["id"])
    # ① and ② finished; ③ was never seated, so the next seat is ③.
    third = orch.assign_enrollment()
    assert third["arm"] == ARM_SHARED_FROZEN
    orch.abandon_enrollment(third)
    again = orch.assign_enrollment()
    assert again["arm"] == ARM_SHARED_FROZEN
    assert orch.status()["contrast"]["next_arm"] == ARM_SHARED_FROZEN


def test_default_orchestration_state_enables_auto(tmp_path: Path) -> None:
    from app.orchestration import (
        MIN_DONORS_FOR_AUTO,
        MAX_DONORS_FOR_AUTO,
        BEHAVIOR_WINDOW,
        REL_DELTA_THRESHOLD,
        STABLE_REQUIRED,
        TAP_DELTA_TOLERANCE,
        OPENING_IMPROVE_EPSILON,
        OPENING_EQUIV_EPSILON,
        PLATEAU_PERSIST_AFTER_NONOVERLAP,
        CONFIRM_HORIZON_DONORS,
    )

    base = Settings.from_env()
    settings = replace(base, data_dir=tmp_path / "data-default")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    orch = AutoOrchestrator(settings, StudyDB(settings))
    status = orch.status()
    assert status["auto_enabled"] is True
    assert status["phase"] == "share_fit"
    assert status["enrollment_open"] is True
    assert status["share_fit"]["min_donors_for_auto"] == MIN_DONORS_FOR_AUTO == 8
    assert status["share_fit"]["stable_required"] == STABLE_REQUIRED == 3
    assert status["share_fit"]["rel_delta_threshold"] == REL_DELTA_THRESHOLD == 0.02
    assert status["share_fit"]["max_donors_for_auto"] == MAX_DONORS_FOR_AUTO == 50
    assert status["share_fit"]["behavior_window"] == BEHAVIOR_WINDOW == 3
    assert status["share_fit"]["opening_improve_epsilon"] == OPENING_IMPROVE_EPSILON == 0.002
    assert status["share_fit"]["opening_equiv_epsilon"] == OPENING_EQUIV_EPSILON
    assert (
        status["share_fit"]["plateau_persist_after_nonoverlap"]
        == PLATEAU_PERSIST_AFTER_NONOVERLAP
        == 3
    )
    assert status["share_fit"]["confirm_horizon_donors"] == CONFIRM_HORIZON_DONORS == 22
    assert status["share_fit"]["tap_delta_tolerance"] == TAP_DELTA_TOLERANCE == 0.02
    orch.pause(actor="test")
    assert orch.status()["enrollment_open"] is False
    orch.resume(actor="test")
    assert orch.status()["enrollment_open"] is True


def _create_share_participants(
    settings: Settings, db: StudyDB, orch: AutoOrchestrator, count: int
) -> list[dict[str, object]]:
    codes = db.create_participants(
        count,
        "study_a",
        "personalized",
        data_origin="formal",
        protocol_mode="strict",
        run_slug=settings.public_enrollment_run_slug,
    )
    participants = [db.participant_by_code(code) for code in codes]
    assert all(participant is not None for participant in participants)
    resolved = [participant for participant in participants if participant is not None]
    state = orch._load()
    state["share_fit"]["enrolled_participant_ids"] = [
        str(participant["id"]) for participant in resolved
    ]
    orch._write(state)
    return resolved


def _next_half_hour_slot(now_ms: int) -> int:
    slot_ms = 30 * 60 * 1000
    return ((int(now_ms) // slot_ms) + 1) * slot_ms


def test_share_setup_over_thirty_minutes_releases_and_disables_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-setup-timeout",
        public_enrollment_run_slug="share-setup-timeout-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant = _create_share_participants(settings, db, orch, 1)[0]
    participant_id = str(participant["id"])
    clock = {"seconds": 1_800_000_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])

    claimed = orch.claim_share_session_start(participant_id)
    assert claimed["holder_is_self"] is True
    session = db.start_session(participant_id, 1, "study")
    assert session["state"] == "started"

    live_status = orch.share_appointment_status(participant_id)
    assert live_status["setup_timer_active"] is True
    assert live_status["setup_timeout_ms"] == 30 * 60 * 1000
    assert live_status["setup_started_ms"] == int(clock["seconds"] * 1000)
    assert live_status["setup_deadline_ms"] == (
        live_status["setup_started_ms"] + live_status["setup_timeout_ms"]
    )
    assert live_status["setup_remaining_ms"] == 30 * 60 * 1000

    clock["seconds"] = (live_status["setup_deadline_ms"] / 1000) + 1
    status = orch.share_appointment_status(participant_id)
    assert status["setup_timed_out"] is True
    assert status["setup_timer_active"] is False
    assert status["setup_deadline_ms"] is None
    assert status["serial_required"] is False
    assert status["can_start_setup"] is False
    assert status["holder_present"] is False
    assert db.session(session["id"], participant_id)["state"] == "aborted"
    with pytest.raises(ShareAppointmentError) as blocked:
        orch.claim_share_session_start(participant_id)
    assert blocked.value.code == "share_setup_timed_out"


def test_voluntary_exit_releases_share_slot_immediately_and_retires_identifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-voluntary-exit",
        public_enrollment_run_slug="share-voluntary-exit-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second = _create_share_participants(settings, db, orch, 2)
    first_id = str(first["id"])
    second_id = str(second["id"])
    clock = {"seconds": 1_800_000_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])

    claimed = orch.claim_share_session_start(first_id)
    assert claimed["holder_is_self"] is True
    session = db.start_session(first_id, 1, "study")
    db.abort_session(str(session["id"]), first_id)
    released = orch.on_participant_exit(first_id)

    assert released["released"] is True
    assert released["holder_released"] is True
    assert released["participant_retired"] is True
    status = orch.share_appointment_status(first_id)
    assert status["participant_exited"] is True
    assert status["status"] == "participant_exited"
    assert status["serial_required"] is False
    assert status["appointment_required"] is False
    assert status["can_start_setup"] is False
    assert status["holder_present"] is False
    with pytest.raises(ShareAppointmentError) as blocked:
        orch.claim_share_session_start(first_id)
    assert blocked.value.code == "share_participant_exited"

    # No 30-minute wait applies: the next serial participant can claim now.
    next_status = orch.share_appointment_status(second_id)
    assert next_status["walk_in_available"] is True
    assert orch.claim_share_session_start(second_id)["holder_is_self"] is True


def test_share_serial_appointments_enforce_spacing_and_release_late_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-appointments",
        public_enrollment_run_slug="share-appointment-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 3)
    first, second, third = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_000_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)

    # Enrollment and status reads never allocate the serial position. The
    # position is claimed only when the participant actually starts a session
    # after completing consent and compensation/contact registration.
    assert orch.register_share_enrollment(first)["status"] == "walk_in_available"
    assert orch.register_share_enrollment(second)["walk_in_available"] is True
    assert orch._load()["share_fit"]["current_holder"] is None
    assert orch.claim_share_session_start(first)["holder_is_self"] is True
    assert orch.share_appointment_status(second)["appointment_required"] is True
    assert orch.share_appointment_status(second)["can_book_future"] is True
    second_start = _next_half_hour_slot(now_ms) + (2 * 60 * 60 * 1000)
    with pytest.raises(ShareAppointmentError) as invalid_slot:
        orch.book_share_appointment(second, second_start + 60_000)
    assert invalid_slot.value.code == "share_appointment_invalid_slot"
    booked = orch.book_share_appointment(second, second_start)
    assert booked["status"] == "booked"
    assert booked["scheduled_start_ms"] == second_start
    assert len(booked["voucher"]) == 4
    assert booked["voucher"].isdigit()
    assert orch.share_appointment_status(second)["voucher"] is None
    assert len(booked["time_slots"]) in {24, 25}
    assert all(
        int(row["start_ms"]) % (30 * 60 * 1000) == 0
        for row in booked["time_slots"]
    )
    admin_appointments = orch.status()["share_fit"]["serial_schedule"]["appointments"]
    assert any(
        row["participant_id"] == second and row["voucher"] == booked["voucher"]
        for row in admin_appointments
    )

    with pytest.raises(ShareAppointmentError) as conflict:
        orch.book_share_appointment(third, second_start + (30 * 60 * 1000))
    assert conflict.value.code == "share_appointment_conflict"
    third_start = second_start + (60 * 60 * 1000)
    assert orch.book_share_appointment(third, third_start)["status"] == "booked"

    clock["seconds"] = (second_start + (10 * 60 * 1000) + 1) / 1000
    released = orch.share_appointment_status(second)
    assert released["status"] == "unbooked"
    assert released["appointment_required"] is True


def test_share_walk_in_starts_directly_when_no_current_run_or_due_booking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-free-walk-in",
        public_enrollment_run_slug="share-free-walk-in-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 3)
    first, future_booked, walk_in = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_025_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)

    orch.register_share_enrollment(first)
    future = orch.book_share_appointment(
        future_booked, _next_half_hour_slot(now_ms) + (3 * 60 * 60 * 1000)
    )
    assert future["status"] == "booked"
    orch.on_session_aborted(first)

    available = orch.share_appointment_status(walk_in)
    assert available["holder_present"] is False
    assert available["walk_in_available"] is True
    assert available["appointment_required"] is False
    assert available["can_book_future"] is True
    assert available["can_start_setup"] is True
    claimed = orch.claim_share_session_start(walk_in)
    assert claimed["holder_is_self"] is True


def test_legacy_pre_contact_walk_in_holder_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-legacy-pre-contact-holder",
        public_enrollment_run_slug="share-legacy-pre-contact-holder-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second = [
        str(row["id"])
        for row in _create_share_participants(settings, db, orch, 2)
    ]
    clock = {"seconds": 1_800_040_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    state = orch._load()
    state["share_fit"]["current_holder"] = {
        "participant_id": first,
        "mode": "walk_in",
        "reference_start_ms": now_ms,
        "granted_ms": now_ms,
        "expires_ms": now_ms + SHARE_WALK_IN_LEASE_MS,
    }
    orch._write(state)

    available = orch.share_appointment_status(second)

    assert available["holder_present"] is False
    assert available["walk_in_available"] is True
    assert available["appointment_required"] is False
    assert any(
        row.get("action") == "share_legacy_pre_contact_holder_released"
        and row.get("participant_id") == first
        for row in orch._load()["history"]
    )


def test_share_status_never_regrants_aborted_walk_in_and_next_claim_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-walk-in-once",
        public_enrollment_run_slug="share-walk-in-once-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 2)
    first, second = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_050_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])

    assert orch.register_share_enrollment(first)["status"] == "walk_in_available"
    assert orch.claim_share_session_start(first)["status"] == "walk_in_claimed"
    first_session = db.start_session(first, 1, "study")
    db.abort_session(str(first_session["id"]), first)

    # Reproduce the v138 persisted schema and its aborted ghost holder. The
    # migration must remember the original grant before cleanup releases it.
    state = orch._load()
    state["share_fit"].pop("walk_in_granted_participant_ids", None)
    orch._write(state)

    available = orch.share_appointment_status(second)
    assert available["status"] == "walk_in_available"
    assert available["can_start_setup"] is True
    assert available["holder_present"] is False
    persisted = orch._load()
    assert persisted["share_fit"]["current_holder"] is None
    assert persisted["share_fit"]["walk_in_granted_participant_ids"] == [first]

    # Repeated bootstrap/status reads remain allocation-free, and the old
    # participant cannot receive a second automatic walk-in lease.
    assert orch.share_appointment_status(second)["holder_present"] is False
    first_after_abort = orch.share_appointment_status(first)
    assert first_after_abort["status"] == "unbooked"
    assert first_after_abort["appointment_required"] is True
    assert sum(
        row.get("action") == "share_serial_walk_in"
        and row.get("participant_id") == first
        for row in orch._load()["history"]
    ) == 1

    claimed = orch.claim_share_session_start(second)
    assert claimed["status"] == "walk_in_claimed"
    assert claimed["holder_is_self"] is True
    assert orch._load()["share_fit"]["walk_in_granted_participant_ids"] == [
        first,
        second,
    ]


def test_due_booking_grace_blocks_another_participants_walk_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-due-booking",
        public_enrollment_run_slug="share-due-booking-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 3)
    first, booked_participant, waiting_participant = [
        str(row["id"]) for row in participants
    ]
    clock = {"seconds": 1_800_075_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)

    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(booked_participant, scheduled)
    orch.on_session_aborted(first)
    clock["seconds"] = (scheduled + (5 * 60 * 1000)) / 1000

    waiting = orch.share_appointment_status(waiting_participant)
    assert waiting["holder_present"] is False
    assert waiting["walk_in_available"] is False
    assert waiting["appointment_required"] is True

    with pytest.raises(ShareAppointmentError) as invalid_voucher:
        orch.claim_share_appointment_arrival(booked_participant, "9999")
    assert invalid_voucher.value.code == "share_appointment_voucher_invalid"
    arrived = orch.claim_share_appointment_arrival(
        booked_participant, str(booked["voucher"])
    )
    assert arrived["status"] == "claimed"
    assert arrived["holder_is_self"] is True


def test_historical_abort_does_not_release_new_verified_appointment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-historical-abort-recovery",
        public_enrollment_run_slug="share-historical-abort-recovery-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant = _create_share_participants(settings, db, orch, 1)[0]
    participant_id = str(participant["id"])
    clock = {"seconds": 1_800_080_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])

    old_session = db.start_session(participant_id, 1, "study")
    clock["seconds"] += 5 * 60
    db.abort_session(str(old_session["id"]), participant_id)
    old_abort_ms = int(clock["seconds"] * 1000)

    clock["seconds"] += 5 * 60
    now_ms = int(clock["seconds"] * 1000)
    scheduled = _next_half_hour_slot(now_ms)
    booked = orch.book_share_appointment(participant_id, scheduled)
    assert int(booked["scheduled_start_ms"]) == scheduled
    clock["seconds"] = scheduled / 1000

    arrived = orch.claim_share_appointment_arrival(
        participant_id, str(booked["voucher"])
    )
    assert arrived["status"] == "claimed"
    assert arrived["holder_is_self"] is True
    assert orch.can_restart_aborted_session(participant_id) is True

    # A status reconciliation must not attach the historical abort to the new
    # appointment and must therefore keep the verified serial position.
    reconciled = orch.share_appointment_status(participant_id)
    assert reconciled["status"] == "claimed"
    assert reconciled["holder_is_self"] is True

    study_state = StudyState(db, Protocol(settings))
    allow_restart = orch.can_restart_aborted_session(participant_id)
    assert allow_restart is True
    study_state.assert_can_start(
        participant, 1, "study", allow_aborted_restart=allow_restart
    )
    assert orch.claim_share_session_start(participant_id)["holder_is_self"] is True
    restarted = db.restart_aborted_session_day(participant_id, 1)
    assert restarted is not None
    assert restarted["state"] == "started"
    assert int(restarted["started_ms"]) > old_abort_ms
    assert restarted["ended_ms"] is None
    assert orch.can_restart_aborted_session(participant_id) is False
    after_restart = orch.share_appointment_status(participant_id)
    assert after_restart["status"] == "claimed"
    assert after_restart["holder_is_self"] is True


def test_abort_from_current_appointment_releases_verified_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-current-abort",
        public_enrollment_run_slug="share-current-abort-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant = _create_share_participants(settings, db, orch, 1)[0]
    participant_id = str(participant["id"])
    clock = {"seconds": 1_800_090_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    scheduled = _next_half_hour_slot(now_ms)
    booked = orch.book_share_appointment(participant_id, scheduled)
    clock["seconds"] = scheduled / 1000

    arrived = orch.claim_share_appointment_arrival(
        participant_id, str(booked["voucher"])
    )
    assert arrived["status"] == "claimed"
    current_session = db.start_session(participant_id, 1, "study")
    db.abort_session(str(current_session["id"]), participant_id)

    released = orch.share_appointment_status(participant_id)
    assert released["status"] == "unbooked"
    assert released["holder_is_self"] is False
    appointment = next(
        row
        for row in reversed(orch._load()["share_fit"]["appointments"])
        if str(row.get("participant_id") or "") == participant_id
    )
    assert appointment["status"] == "released"
    assert appointment["release_reason"] == "session_aborted"


def test_on_time_arrival_is_preserved_when_predecessor_runs_past_booking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-arrived-waiting",
        public_enrollment_run_slug="share-arrived-waiting-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 2)
    first, second = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_085_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)

    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(second, scheduled)
    clock["seconds"] = scheduled / 1000

    waiting = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    first_holder = orch._load()["share_fit"]["current_holder"]
    expected_deadline = int(first_holder["expires_ms"])
    assert waiting["status"] == "booked"
    assert waiting["arrived_ms"] == scheduled
    assert waiting["holder_is_self"] is False
    assert waiting["credential_verified"] is True
    assert waiting["predecessor_delay"] is True
    assert waiting["retry_after_ms"] == expected_deadline
    assert waiting["predecessor_deadline_ms"] == expected_deadline

    clock["seconds"] = (scheduled + (11 * 60 * 1000)) / 1000
    preserved = orch.share_appointment_status(second)
    assert preserved["status"] == "booked"
    assert preserved["arrived_ms"] == scheduled

    orch.on_session_aborted(first)
    claimed = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert claimed["status"] == "claimed"
    assert claimed["holder_is_self"] is True


def test_predecessor_wait_uses_setup_deadline_and_requires_voucher_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-wait-deadline",
        public_enrollment_run_slug="share-wait-deadline-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second = [
        str(row["id"]) for row in _create_share_participants(settings, db, orch, 2)
    ]
    clock = {"seconds": 1_800_200_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)

    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(second, scheduled)
    clock["seconds"] = scheduled / 1000
    db.start_session(first, 1, "study")
    first_state = orch._load()
    setup_deadline = orch._share_session_timeout_window_locked(
        first_state, first
    )["deadline_ms"]
    holder_expires = int(first_state["share_fit"]["current_holder"]["expires_ms"])
    expected_deadline = min(setup_deadline, holder_expires)

    waiting = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert waiting["predecessor_delay"] is True
    assert waiting["predecessor_deadline_ms"] == expected_deadline
    assert waiting["arrival_reentry_required"] is False

    orch.on_session_aborted(first)
    opened = orch.share_appointment_status(second)
    assert opened["predecessor_delay"] is False
    assert opened["arrival_reentry_required"] is True
    assert opened["entry_deadline_ms"] == opened["entry_open_ms"] + SHARE_ARRIVAL_ENTRY_MS

    with pytest.raises(ShareAppointmentError) as missing:
        orch.claim_share_appointment_arrival(second, None)
    assert missing.value.code == "share_appointment_voucher_invalid"
    claimed = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert claimed["status"] == "claimed"
    assert claimed["holder_is_self"] is True


def test_arrival_entry_expires_fifteen_minutes_after_predecessor_frees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-entry-expire",
        public_enrollment_run_slug="share-entry-expire-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second = [
        str(row["id"]) for row in _create_share_participants(settings, db, orch, 2)
    ]
    clock = {"seconds": 1_800_260_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(second, scheduled)
    clock["seconds"] = scheduled / 1000
    orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    orch.on_session_aborted(first)
    opened = orch.share_appointment_status(second)
    assert opened["arrival_reentry_required"] is True
    clock["seconds"] = (int(opened["entry_open_ms"]) + SHARE_ARRIVAL_ENTRY_MS + 1000) / 1000
    expired = orch.share_appointment_status(second)
    assert expired["status"] in {"unbooked", "walk_in_available"}
    appointment = next(
        row
        for row in reversed(orch._load()["share_fit"]["appointments"])
        if str(row.get("participant_id") or "") == second
    )
    assert appointment["status"] == "released"
    assert appointment["release_reason"] == "arrival_entry_expired"
    with pytest.raises(ShareAppointmentError) as released:
        orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert released.value.code == "share_appointment_released"


def test_legacy_arrived_waiter_does_not_use_new_entry_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-legacy-entry",
        public_enrollment_run_slug="share-legacy-entry-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second = [
        str(row["id"]) for row in _create_share_participants(settings, db, orch, 2)
    ]
    clock = {"seconds": 1_800_320_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(second, scheduled)
    clock["seconds"] = scheduled / 1000
    orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    state = orch._load()
    own = next(
        row
        for row in reversed(state["share_fit"]["appointments"])
        if str(row.get("participant_id") or "") == second
    )
    own.pop("arrival_entry_policy", None)
    orch._write(state)
    orch.on_session_aborted(first)
    opened = orch.share_appointment_status(second)
    assert opened["arrival_reentry_required"] is False
    clock["seconds"] = (scheduled + (40 * 60 * 1000)) / 1000
    still_booked = orch.share_appointment_status(second)
    assert still_booked["status"] == "booked"
    claimed = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert claimed["status"] == "claimed"


def test_global_reservation_survives_predecessor_wait_past_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    coordination = tmp_path / "shared-scheduler" / "wait-grace.sqlite3"
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-global-wait",
        public_enrollment_run_slug="share-global-wait-run",
        instance_id="formal",
        share_global_coordination_path=str(coordination),
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second = [
        str(row["id"]) for row in _create_share_participants(settings, db, orch, 2)
    ]
    clock = {"seconds": 1_800_380_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(second, scheduled)
    clock["seconds"] = scheduled / 1000
    orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    clock["seconds"] = (scheduled + (11 * 60 * 1000)) / 1000
    appointment_id = next(
        str(item["appointment_id"])
        for item in orch._load()["share_fit"]["appointments"]
        if str(item.get("participant_id") or "") == second
    )
    snapshot = orch.global_coordination.snapshot(
        orch._global_owner_key(second), now_ms=int(clock["seconds"] * 1000)
    )
    matching = [
        row
        for row in snapshot["appointments"]
        if str(row.get("reservation_key") or "") == appointment_id
    ]
    assert matching
    assert int(matching[0]["arrived_ms"] or 0) > 0
    orch.on_session_aborted(first)
    claimed = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert claimed["status"] == "claimed"
    assert claimed["holder_is_self"] is True


def test_share_serial_claim_blocks_parallel_start_and_allows_due_appointment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-claim",
        public_enrollment_run_slug="share-claim-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 3)
    first, second, third = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_100_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)

    orch.register_share_enrollment(first)
    second_start = _next_half_hour_slot(now_ms) + (2 * 60 * 60 * 1000)
    third_start = second_start + (60 * 60 * 1000)
    second_booking = orch.book_share_appointment(second, second_start)
    orch.book_share_appointment(third, third_start)
    assert orch.claim_share_session_start(first)["holder_is_self"] is True
    with pytest.raises(ShareAppointmentError) as occupied:
        orch.claim_share_session_start(second)
    assert occupied.value.code == "share_serial_occupied"

    orch.on_session_aborted(first)
    clock["seconds"] = second_start / 1000
    arrived = orch.claim_share_appointment_arrival(
        second, str(second_booking["voucher"])
    )
    assert arrived["status"] == "claimed"
    claimed = orch.claim_share_session_start(second)
    assert claimed["status"] == "claimed"
    assert claimed["holder_is_self"] is True
    db.start_session(second, 1, "study")
    assert orch.require_share_activation(second)["status"] == "active"
    with pytest.raises(ShareAppointmentError) as parallel:
        orch.require_share_activation(third)
    assert parallel.value.code == "share_serial_not_claimed"
    clock["seconds"] = (second_start + (60 * 60 * 1000)) / 1000
    still_held = orch.share_appointment_status(second)
    assert still_held["status"] == "active"
    assert still_held["holder_is_self"] is True

    clock["seconds"] = (second_start + (91 * 60 * 1000)) / 1000
    after_timeout = orch.status()
    assert after_timeout["share_fit"]["serial_schedule"]["holder_present"] is False
    assert db.sessions(second)[0]["state"] == "aborted"
    assert any(
        row.get("action") == "share_stale_session_released"
        and row.get("participant_id") == second
        for row in after_timeout["history"]
    )


def test_share_setup_slot_is_global_across_formal_and_test_instances(
    tmp_path: Path,
) -> None:
    base = Settings.from_env()
    coordination = tmp_path / "shared-scheduler" / "share.sqlite3"
    formal_settings = replace(
        base,
        data_dir=tmp_path / "formal-data",
        public_enrollment_run_slug="global-formal-run",
        public_enrollment_cohort="study_a",
        instance_id="formal",
        share_global_coordination_path=str(coordination),
    )
    test_settings = replace(
        base,
        data_dir=tmp_path / "test-data",
        public_enrollment_run_slug="global-test-run",
        public_enrollment_cohort="pilot",
        instance_id="test",
        share_global_coordination_path=str(coordination),
    )
    formal_settings.data_dir.mkdir(parents=True, exist_ok=True)
    test_settings.data_dir.mkdir(parents=True, exist_ok=True)
    formal_db = StudyDB(formal_settings)
    test_db = StudyDB(test_settings)
    formal = AutoOrchestrator(formal_settings, formal_db)
    test = AutoOrchestrator(test_settings, test_db)
    formal_participant = _create_share_participants(
        formal_settings, formal_db, formal, 1
    )[0]
    test_participant = _create_share_participants(
        test_settings, test_db, test, 1
    )[0]
    formal_id = str(formal_participant["id"])
    test_id = str(test_participant["id"])

    assert formal.claim_share_session_start(formal_id)["holder_is_self"] is True
    blocked_status = test.share_appointment_status(test_id)
    assert blocked_status["global_external_holder_present"] is True
    assert blocked_status["walk_in_available"] is False
    assert blocked_status["appointment_required"] is True
    with pytest.raises(ShareAppointmentError) as blocked:
        test.claim_share_session_start(test_id)
    assert blocked.value.code == "share_serial_occupied"

    formal.on_session_aborted(formal_id)
    assert test.share_appointment_status(test_id)["walk_in_available"] is True
    assert test.claim_share_session_start(test_id)["holder_is_self"] is True


def test_simultaneous_cross_instance_share_claim_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    base = Settings.from_env()
    coordination = tmp_path / "shared-scheduler" / "race.sqlite3"
    settings_by_instance = {
        instance: replace(
            base,
            data_dir=tmp_path / f"{instance}-race-data",
            public_enrollment_run_slug=f"global-{instance}-race-run",
            public_enrollment_cohort=("study_a" if instance == "formal" else "pilot"),
            instance_id=instance,
            share_global_coordination_path=str(coordination),
        )
        for instance in ("formal", "test")
    }
    orchestrators: dict[str, AutoOrchestrator] = {}
    participant_ids: dict[str, str] = {}
    for instance, instance_settings in settings_by_instance.items():
        instance_settings.data_dir.mkdir(parents=True, exist_ok=True)
        instance_db = StudyDB(instance_settings)
        instance_orchestrator = AutoOrchestrator(instance_settings, instance_db)
        orchestrators[instance] = instance_orchestrator
        participant_ids[instance] = str(
            _create_share_participants(
                instance_settings, instance_db, instance_orchestrator, 1
            )[0]["id"]
        )

    def claim(instance: str) -> tuple[str, str]:
        try:
            result = orchestrators[instance].claim_share_session_start(
                participant_ids[instance]
            )
            return instance, "won" if result["holder_is_self"] else "unexpected"
        except ShareAppointmentError as exc:
            return instance, exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(claim, ("formal", "test")))

    assert list(results.values()).count("won") == 1
    assert list(results.values()).count("share_serial_occupied") == 1
    snapshots = [
        orchestrators[instance].global_coordination.snapshot(
            orchestrators[instance]._global_owner_key(participant_ids[instance])
        )
        for instance in ("formal", "test")
    ]
    assert all(len(snapshot["holders"]) == 1 for snapshot in snapshots)


def test_global_share_appointment_spacing_crosses_instance_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    coordination = tmp_path / "shared-scheduler" / "appointments.sqlite3"
    formal_settings = replace(
        base,
        data_dir=tmp_path / "formal-appointments",
        public_enrollment_run_slug="appointment-formal-run",
        public_enrollment_cohort="study_a",
        instance_id="formal",
        share_global_coordination_path=str(coordination),
    )
    test_settings = replace(
        base,
        data_dir=tmp_path / "test-appointments",
        public_enrollment_run_slug="appointment-test-run",
        instance_id="test",
        share_global_coordination_path=str(coordination),
    )
    formal_settings.data_dir.mkdir(parents=True, exist_ok=True)
    test_settings.data_dir.mkdir(parents=True, exist_ok=True)
    formal_db = StudyDB(formal_settings)
    test_db = StudyDB(test_settings)
    formal = AutoOrchestrator(formal_settings, formal_db)
    test = AutoOrchestrator(test_settings, test_db)
    formal_id = str(
        _create_share_participants(formal_settings, formal_db, formal, 1)[0]["id"]
    )
    test_id = str(
        _create_share_participants(test_settings, test_db, test, 1)[0]["id"]
    )
    clock = {"seconds": 1_800_500_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    start = _next_half_hour_slot(int(clock["seconds"] * 1000)) + 2 * 60 * 60 * 1000

    formal.book_share_appointment(formal_id, start)
    with pytest.raises(ShareAppointmentError) as conflict:
        test.book_share_appointment(test_id, start + 30 * 60 * 1000)
    assert conflict.value.code == "share_appointment_conflict"
    assert test.book_share_appointment(
        test_id, start + 60 * 60 * 1000
    )["status"] == "booked"

def test_share_activation_retires_legacy_stale_active_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-legacy-stale",
        public_enrollment_run_slug="share-legacy-stale-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 2)
    previous, current = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_200_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])

    orch.register_share_enrollment(previous)
    orch.claim_share_session_start(previous)
    study = db.start_session(previous, 1, "study")
    companion = db.start_session(previous, 1, "companion")
    orch.require_share_activation(previous)
    db.activate_paired_sessions(previous, study["id"], companion["id"])

    # Reproduce a pre-scheduler ghost: the new participant owns the persisted
    # holder while a different, old SQLite session still says active.
    clock["seconds"] += 60 * 60
    current_study = db.start_session(current, 1, "study")
    db.start_session(current, 1, "companion")
    state = orch._load()
    now_ms = int(clock["seconds"] * 1000)
    state["share_fit"]["current_holder"] = {
        "participant_id": current,
        "mode": "walk_in",
        "reference_start_ms": now_ms,
        "claimed_ms": now_ms,
        "expires_ms": now_ms + (90 * 60 * 1000),
    }
    orch._write(state)
    with pytest.raises(ShareAppointmentError) as occupied:
        orch.require_share_activation(current)
    assert occupied.value.code == "share_serial_occupied"
    assert occupied.value.status["appointment_required"] is True
    assert occupied.value.status["holder_is_self"] is False
    assert db.sessions(previous)[0]["state"] == "active"

    booked_start = _next_half_hour_slot(int(clock["seconds"] * 1000)) + (
        3 * 60 * 60 * 1000
    )
    booked = orch.book_share_appointment(current, booked_start)
    assert booked["status"] == "booked"

    # Waiting for a future booking must not invalidate the already-created
    # setup session at the ordinary 90-minute session timeout boundary.
    clock["seconds"] += 91 * 60
    orch.status()
    assert db.session(current_study["id"], current)["state"] == "started"
    assert {row["state"] for row in db.sessions(previous)} == {"aborted"}

    clock["seconds"] = booked_start / 1000
    claimed = orch.claim_share_session_start(current)
    assert claimed["status"] == "claimed"
    activated = orch.require_share_activation(current)
    assert activated["status"] == "active"
    assert activated["holder_is_self"] is True


def test_share_serial_claim_assigns_latest_completed_prefix_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-chain-prior",
        public_enrollment_run_slug="share-chain-prior-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 2)
    first, second = [str(row["id"]) for row in participants]
    state = orch._load()
    state["share_fit"]["completed_donor_ids"] = [first]
    state["share_fit"]["delta_history"] = [
        {"n": 1, "prior_id": "prefix-prior-1", "rel_delta": None}
    ]
    orch._write(state)
    monkeypatch.setattr(
        db,
        "end_to_end_completion_records",
        lambda participant_ids: (
            {first: {"participant_id": first, "completed_ms": 1}}
            if first in participant_ids
            else {}
        ),
    )
    assigned: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        db,
        "assign_prior",
        lambda participant_id, prior_id, *, actor: assigned.append(
            (participant_id, prior_id, actor)
        ),
    )
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {
            "id": "prefix-prior-1",
            "run_id": str(participants[0]["run_id"]),
            "donor_ids": list(donor_ids),
        },
    )

    assert orch.register_share_enrollment(second)["status"] == "walk_in_available"
    claimed = orch.claim_share_session_start(second)

    assert claimed["latest_chain_prior_id"] == "prefix-prior-1"
    assert assigned == [(second, "prefix-prior-1", "share-serial-orchestration")]


def test_due_appointment_waits_and_does_not_bind_prior_while_predecessor_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-wait-latest-prior",
        public_enrollment_run_slug="share-wait-latest-prior-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 2)
    first, second = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_300_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    assigned: list[tuple[str, str]] = []
    monkeypatch.setattr(
        db,
        "assign_prior",
        lambda participant_id, prior_id, *, actor: assigned.append(
            (participant_id, prior_id)
        ),
    )
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {"id": "prefix-prior-1", "donor_ids": list(donor_ids)},
    )

    orch.register_share_enrollment(first)
    orch.claim_share_session_start(first)
    scheduled = _next_half_hour_slot(now_ms) + (60 * 60 * 1000)
    booked = orch.book_share_appointment(second, scheduled)
    clock["seconds"] = scheduled / 1000
    waiting = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert waiting["predecessor_delay"] is True
    assert waiting["status"] == "booked"
    assert db.participant(second)["prior_id"] is None
    with pytest.raises(ShareAppointmentError) as occupied:
        orch.claim_share_session_start(second)
    assert occupied.value.code == "share_serial_occupied"
    assert assigned == []
    assert db.participant(second)["prior_id"] is None


def test_stale_start_prior_rebinds_to_latest_prefix_after_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-rebind-latest-prior",
        public_enrollment_run_slug="share-rebind-latest-prior-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 3)
    first, second, third = [str(row["id"]) for row in participants]
    clock = {"seconds": 1_800_400_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    assigned: list[tuple[str, str]] = []
    rebound: list[tuple[str, str]] = []

    def fake_assign(participant_id: str, prior_id: str, *, actor: str) -> None:
        assigned.append((participant_id, prior_id))

    def fake_rebind(
        participant_id: str, prior_id: str, *, actor: str, reason: str
    ) -> dict[str, object]:
        rebound.append((participant_id, prior_id))
        return {
            "participant_id": participant_id,
            "from_prior_id": "prefix-prior-1",
            "to_prior_id": prior_id,
            "unchanged": False,
        }

    monkeypatch.setattr(db, "assign_prior", fake_assign)
    monkeypatch.setattr(db, "rebind_share_start_prior", fake_rebind)
    current_prior = {"id": "prefix-prior-1"}
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {"id": current_prior["id"], "donor_ids": list(donor_ids)},
    )

    def participant_with_prior(participant_id: str) -> dict[str, object] | None:
        row = original_participant(participant_id)
        if row is None:
            return None
        copied = dict(row)
        if participant_id == second and rebound:
            copied["prior_id"] = current_prior["id"]
        elif participant_id == second and assigned:
            copied["prior_id"] = "prefix-prior-1"
        return copied

    original_participant = db.participant
    monkeypatch.setattr(db, "participant", participant_with_prior)
    complete_ids = {first}
    monkeypatch.setattr(
        db,
        "end_to_end_completion_records",
        lambda participant_ids: {
            pid: {"participant_id": pid, "completed_ms": 1}
            for pid in participant_ids
            if pid in complete_ids
        },
    )

    state = orch._load()
    state["share_fit"]["completed_donor_ids"] = [first]
    orch._write(state)
    orch.register_share_enrollment(second)
    first_claim = orch.claim_share_session_start(second)
    assert first_claim["holder_is_self"] is True
    assert assigned == [(second, "prefix-prior-1")]
    orch.on_session_aborted(second)

    current_prior["id"] = "prefix-prior-2"
    complete_ids.add(third)
    later = _next_half_hour_slot(int(clock["seconds"] * 1000)) + (2 * 60 * 60 * 1000)
    booked = orch.book_share_appointment(second, later)
    clock["seconds"] = later / 1000
    arrived = orch.claim_share_appointment_arrival(second, str(booked["voucher"]))
    assert arrived["status"] == "claimed"
    state = orch._load()
    state["share_fit"]["completed_donor_ids"] = [first, third]
    orch._write(state)
    claimed = orch.claim_share_session_start(second)
    assert claimed["holder_is_self"] is True
    assert rebound == [(second, "prefix-prior-2")]
    assert claimed["latest_chain_prior_id"] == "prefix-prior-2"


def test_live_session_keeps_existing_start_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-live-prior-guard",
        public_enrollment_run_slug="share-live-prior-guard-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participants = _create_share_participants(settings, db, orch, 2)
    first, second = [str(row["id"]) for row in participants]
    rebound: list[tuple[str, str]] = []
    monkeypatch.setattr(
        db,
        "rebind_share_start_prior",
        lambda participant_id, prior_id, *, actor, reason: rebound.append(
            (participant_id, prior_id)
        ),
    )
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {"id": "prefix-prior-2", "donor_ids": list(donor_ids)},
    )

    def participant_with_stale_prior(participant_id: str) -> dict[str, object] | None:
        row = original_participant(participant_id)
        if row is None:
            return None
        copied = dict(row)
        if participant_id == second:
            copied["prior_id"] = "prefix-prior-1"
        return copied

    original_participant = db.participant
    monkeypatch.setattr(db, "participant", participant_with_stale_prior)
    monkeypatch.setattr(
        db,
        "end_to_end_completion_records",
        lambda participant_ids: (
            {first: {"participant_id": first, "completed_ms": 1}}
            if first in participant_ids
            else {}
        ),
    )
    state = orch._load()
    state["share_fit"]["completed_donor_ids"] = [first]
    orch._write(state)
    orch.register_share_enrollment(second)
    db.start_session(second, 1, "study")
    now_ms = int(time.time() * 1000)
    state = orch._load()
    state["share_fit"]["current_holder"] = {
        "participant_id": second,
        "mode": "active",
        "reference_start_ms": now_ms,
        "claimed_ms": now_ms,
        "expires_ms": now_ms + (90 * 60 * 1000),
    }
    orch._write(state)
    with pytest.raises(ShareAppointmentError) as mismatch:
        orch.claim_share_session_start(second)
    assert mismatch.value.code == "share_chain_prior_mismatch"
    assert rebound == []


def test_completed_share_participant_cannot_rebook_or_reclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-share-completed-route",
        public_enrollment_run_slug="share-completed-route-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant = _create_share_participants(settings, db, orch, 1)[0]
    participant_id = str(participant["id"])
    monkeypatch.setattr(
        db,
        "participant_end_to_end_complete",
        lambda candidate_id: candidate_id == participant_id,
    )

    status = orch.register_share_enrollment(participant_id)
    assert status["status"] == "completed"
    assert status["end_to_end_complete"] is True
    assert status["serial_required"] is False
    assert status["appointment_required"] is False
    assert status["can_start_setup"] is False
    assert orch._load()["share_fit"]["current_holder"] is None

    with pytest.raises(ShareAppointmentError) as reclaim:
        orch.claim_share_session_start(participant_id)
    assert reclaim.value.code == "participant_already_complete"


def test_share_fit_progress_rejects_incomplete_and_accepts_end_to_end_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-e2e-share",
        public_enrollment_run_slug="orch-e2e-share-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    code = db.create_participants(
        1,
        "pilot",
        "personalized",
        data_origin="pilot",
        protocol_mode="strict",
        run_slug="orch-e2e-share-run",
    )[0]
    participant = db.participant_by_code(code)
    assert participant is not None
    assignment = orch.assign_enrollment()
    orch.commit_enrollment(participant_id=participant["id"], assignment=assignment)

    rejected = orch.on_session_completed(participant["id"])
    assert rejected["share_fit"]["completed"] == 0

    _mark_end_to_end_complete(db, participant["id"])
    pending = orch.on_session_completed(participant["id"])
    assert pending["share_fit"]["completed"] == 0
    assert pending["share_fit"]["protocol_complete_pending_ids"] == [
        participant["id"]
    ]

    receipt = {
        "participant_id": participant["id"],
        "base_prior_id": None,
        "pointer_sha256": "pointer",
        "personal_head_sha256": "head",
        "personal_head_fingerprint": "fingerprint",
        "replay_sha256": "replay",
        "update_count": 0,
        "optimizer_step_count": 0,
        "release_manifest_sha256": "release",
        "shared_head_sha256": "shared",
        "eligible_feedbacks": 0,
        "expected_network_updates": 0,
        "zero_network_update": True,
        "finalized_ms": 123,
    }
    monkeypatch.setattr(
        orch,
        "_personal_checkpoint_receipt",
        lambda _participant: {
            key: value
            for key, value in receipt.items()
            if key not in {
                "participant_id",
                "base_prior_id",
                "eligible_feedbacks",
                "expected_network_updates",
                "zero_network_update",
                "finalized_ms",
            }
        },
    )
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {
            "id": f"prior-{len(donor_ids)}",
            "run_id": participant["run_id"],
        },
    )
    monkeypatch.setattr(
        orch, "_refresh_share_fit_locked", lambda _state, *, reconcile=False: None
    )
    committed = orch.commit_share_completion(participant["id"], receipt)
    assert committed["committed"] is True
    assert orch.status()["share_fit"]["completed"] == 1


def test_share_chain_ignores_questionnaire_without_timed_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-e2e-timed-only",
        public_enrollment_run_slug="orch-e2e-timed-only-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant = _create_share_participants(settings, db, orch, 1)[0]
    participant_id = str(participant["id"])
    study_session = db.start_session(participant_id, 1, "study")
    db.start_session(participant_id, 1, "companion")
    db.save_response(participant_id, study_session["id"], "final", "test", {})
    db.ensure_final_questionnaire_recovery(participant_id, study_session["id"])

    assert db.participant_end_to_end_complete(participant_id) is False
    rejected = orch.on_session_completed(participant_id)
    assert rejected["share_fit"]["completed"] == 0
    assert rejected["share_fit"]["protocol_complete_pending_ids"] == []

    receipt = {
        "participant_id": participant_id,
        "base_prior_id": None,
        "pointer_sha256": "pointer",
        "personal_head_sha256": "head",
        "personal_head_fingerprint": "fingerprint",
        "replay_sha256": "replay",
        "update_count": 0,
        "optimizer_step_count": 0,
        "release_manifest_sha256": "release",
        "shared_head_sha256": "shared",
        "eligible_feedbacks": 0,
        "expected_network_updates": 0,
        "zero_network_update": True,
        "finalized_ms": 123,
    }
    monkeypatch.setattr(
        orch,
        "_personal_checkpoint_receipt",
        lambda _participant: {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "participant_id",
                "base_prior_id",
                "eligible_feedbacks",
                "expected_network_updates",
                "zero_network_update",
                "finalized_ms",
            }
        },
    )
    with pytest.raises(OrchestrationError, match="timed real-time experiment"):
        orch.commit_share_completion(participant_id, receipt)

    _mark_end_to_end_complete(db, participant_id)
    assert db.participant_end_to_end_complete(participant_id) is True
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {
            "id": f"prior-{len(donor_ids)}",
            "run_id": participant["run_id"],
        },
    )
    monkeypatch.setattr(
        orch, "_refresh_share_fit_locked", lambda _state, *, reconcile=False: None
    )
    committed = orch.commit_share_completion(participant_id, receipt)
    assert committed["committed"] is True
    assert orch.status()["share_fit"]["completed"] == 1


def test_contrast_completed_quota_ignores_incomplete_assignment(
    tmp_path: Path,
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-e2e-contrast",
        public_enrollment_run_slug="orch-e2e-contrast-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    state = orch._load()
    state["phase"] = "contrast"
    state["share_fit"]["locked_n"] = 1
    state["share_fit"]["locked_prior_id"] = "prior-lock"
    orch._write(state)
    codes = db.create_participants(
        2,
        "pilot",
        "personalized",
        data_origin="pilot",
        protocol_mode="strict",
        run_slug="orch-e2e-contrast-run",
    )
    participants = [db.participant_by_code(code) for code in codes]
    assert all(participant is not None for participant in participants)
    for participant in participants:
        assert participant is not None
        assignment = orch.assign_enrollment()
        orch.commit_enrollment(
            participant_id=participant["id"], assignment=assignment
        )

    status = orch.status()
    assert sum(status["contrast"]["counts"].values()) == 2
    assert sum(status["contrast"]["completed_counts"].values()) == 0

    first = participants[0]
    assert first is not None
    _mark_end_to_end_complete(db, first["id"])
    status = orch.on_session_completed(first["id"])
    assert sum(status["contrast"]["completed_counts"].values()) == 1


def test_share_chain_prior_failure_keeps_holder_and_retry_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-chain-retry",
        public_enrollment_run_slug="chain-retry-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant = _create_share_participants(settings, db, orch, 1)[0]
    participant_id = str(participant["id"])
    orch.claim_share_session_start(participant_id)
    _mark_end_to_end_complete(db, participant_id)
    receipt = {
        "participant_id": participant_id,
        "base_prior_id": None,
        "pointer_sha256": "pointer",
        "personal_head_sha256": "head",
        "personal_head_fingerprint": "fingerprint",
        "replay_sha256": "replay",
        "update_count": 0,
        "optimizer_step_count": 0,
        "release_manifest_sha256": "release",
        "shared_head_sha256": "shared",
        "eligible_feedbacks": 0,
        "expected_network_updates": 0,
        "zero_network_update": True,
        "finalized_ms": 123,
    }
    actual = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "participant_id",
            "base_prior_id",
            "eligible_feedbacks",
            "expected_network_updates",
            "zero_network_update",
            "finalized_ms",
        }
    }
    monkeypatch.setattr(orch, "_personal_checkpoint_receipt", lambda _p: actual)
    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donor_ids: {
            "id": f"prior-{len(donor_ids)}",
            "run_id": participant["run_id"],
        },
    )
    monkeypatch.setattr(
        orch,
        "_refresh_share_fit_locked",
        lambda _state, *, reconcile=False: (_ for _ in ()).throw(
            OrchestrationError("injected prior write failure")
        ),
    )

    with pytest.raises(OrchestrationError, match="injected prior write failure"):
        orch.commit_share_completion(participant_id, receipt)
    failed = orch._load()["share_fit"]
    assert failed["completed_donor_ids"] == []
    assert failed["current_holder"]["participant_id"] == participant_id
    assert "injected prior write failure" in failed["last_chain_error"]

    monkeypatch.setattr(
        orch, "_refresh_share_fit_locked", lambda _state, *, reconcile=False: None
    )
    committed = orch.commit_share_completion(participant_id, receipt)
    assert committed["committed_n"] == 1
    final = orch._load()["share_fit"]
    assert final["completed_donor_ids"] == [participant_id]
    assert final["current_holder"] is None


def test_existing_zero_budget_prior_is_valid_for_share_chain(
    tmp_path: Path,
) -> None:
    base = Settings.from_env()
    settings = replace(base, data_dir=tmp_path / "data-zero-budget-prior")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    orch = AutoOrchestrator(settings, StudyDB(settings))
    release = SimpleNamespace(
        feature_dim=2,
        revision="release-revision",
        manifest_sha256="release-manifest",
        shared_head_sha256="shared-head",
    )
    head = orchestration_module.LinearHeadState(
        feature_mean=np.zeros(2, dtype=np.float32),
        feature_scale=np.ones(2, dtype=np.float32),
        weight=np.array([0.25, -0.5], dtype=np.float32),
        bias=0.125,
    )
    artifact_dir = tmp_path / "prior-artifacts"
    artifact_dir.mkdir()
    head_path = artifact_dir / "personal_head.npz"
    orchestration_module.write_head_artifact(head_path, head)
    pointer_path = artifact_dir / "distilhubert-prior.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema": "tap-to-adapt-distilhubert-shared-prior/v1",
                "model_family": orchestration_module.MODEL_FAMILY,
                "revision": release.revision,
                "release_manifest_sha256": release.manifest_sha256,
                "shared_head_sha256": release.shared_head_sha256,
                "personal_head_path": head_path.name,
                "personal_head_sha256": orchestration_module.file_sha256(head_path),
                "personal_head_fingerprint": orchestration_module._head_fingerprint(head),
                "donor_count": 1,
                "algorithm": orchestration_module.DISTILHUBERT_PRIOR_ALGORITHM,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prior = {
        "donor_count": 1,
        "opportunity_budget": 0,
        "algorithm": orchestration_module.DISTILHUBERT_PRIOR_ALGORITHM,
        "donor_ids": ["donor-1"],
        "encoder_path": str(head_path),
        "main_path": str(pointer_path),
        "encoder_sha256": orchestration_module.file_sha256(head_path),
        "main_sha256": orchestration_module.file_sha256(pointer_path),
    }

    assert orch._validate_prior_record(prior, ["donor-1"], release) is prior


def test_share_fit_curve_accumulates_tap_rates(tmp_path: Path) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-curve",
        public_enrollment_run_slug="orch-curve-run",
        public_enrollment_cohort="pilot",
        instance_id="test",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    codes = db.create_participants(
        2,
        "pilot",
        "personalized",
        preview_mode=False,
        data_origin="pilot",
        protocol_mode="strict",
        run_slug="orch-curve-run",
        analysis_excluded=True,
    )
    participants = [db.participant_by_code(code) for code in codes]
    assert all(row is not None for row in participants)
    for index, participant in enumerate(participants):
        assert participant is not None
        session = db.start_session(participant["id"], 1, "study")
        for _ in range(10):
            db.append_event(
                participant["id"],
                session["id"],
                "model.prediction",
                None,
                None,
                {"analysis_eligible": True, "decision_eligible": True},
            )
        for _ in range(index + 1):
            db.append_event(
                participant["id"],
                session["id"],
                "tap.manual_activate",
                None,
                None,
                {"cancelled": False, "training_eligible": True},
            )
        db.start_session(participant["id"], 1, "companion")
        db.save_response(
            participant["id"], session["id"], "final", "test", {}
        )
        db.end_session(session["id"], participant["id"])
    state = orch._load()
    state["share_fit"]["completed_donor_ids"] = [row["id"] for row in participants]
    state["share_fit"]["model_finalizations"] = {
        str(row["id"]): {"participant_id": str(row["id"])} for row in participants
    }
    orch._write(state)
    curve = orch.share_fit_curve()
    assert curve["completed"] == 2
    assert curve["points"][0]["tap_count"] == 1
    assert curve["points"][1]["tap_count"] == 2
    assert curve["points"][0]["taps_per_minute"] == pytest.approx(1 / 30, abs=1e-5)
    assert curve["points"][1]["cumulative_mean_taps_per_minute"] == pytest.approx(
        ((1 / 30) + (2 / 30)) / 2, abs=1e-5
    )
    assert curve["points"][0]["correction_burden"] == pytest.approx(0.1)
    assert curve["points"][1]["correction_burden"] == pytest.approx(0.2)


def test_behavior_fit_summary_distinguishes_plateau_improvement_and_rebound() -> None:
    def rows(values: list[float]) -> list[dict[str, float]]:
        return [{"opening_correction_burden": value} for value in values]

    plateau = _behavior_fit_summary(
        rows([0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.12, 0.10, 0.10, 0.10])
    )
    assert plateau["current_state"] == "plateau"
    assert plateau["candidate_n"] == 10
    assert plateau["persist_ready"] is False
    assert plateau["persist_need_n"] == 15

    improving = _behavior_fit_summary(
        rows([0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.12, 0.10, 0.08, 0.00])
    )
    assert improving["current_state"] == "improving"
    assert improving["candidate_n"] == 11

    rebound = _behavior_fit_summary(
        rows([0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.12, 0.10, 0.35, 0.40])
    )
    assert rebound["current_state"] == "rebound"
    assert rebound["candidate_n"] == 9
    assert rebound["current_tap_delta"] == pytest.approx(0.35 - 0.12)
    assert rebound["rebound_confirmed"] is False

    # A single high opener must not flip the three-donor median into rebound.
    spike = _behavior_fit_summary(
        rows([0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.12, 0.10, 0.10, 0.40])
    )
    assert spike["current_state"] == "plateau"
    assert spike["candidate_n"] == 10


def test_behavior_fit_rejects_early_local_min_until_nonoverlap_persist() -> None:
    formal_open = [
        0.02740, 0.01798, 0.03671, 0.01732, 0.00739, 0.00928,
        0.03426, 0.01327, 0.05758, 0.01235, 0.02427, 0.00674,
        0.02404, 0.01057, 0.00258, 0.01487,
    ]
    at16 = _behavior_fit_summary(
        [{"opening_correction_burden": value} for value in formal_open]
    )
    assert at16["candidate_n"] == 14
    assert at16["persist_ready"] is False
    assert at16["persist_need_n"] == 19
    assert at16["current_state"] == "plateau"

    continued = formal_open + [0.01057, 0.01057, 0.01057]
    at19 = _behavior_fit_summary(
        [{"opening_correction_burden": value} for value in continued]
    )
    assert at19["candidate_n"] == 14
    assert at19["persist_ready"] is True
    assert at19["current_state"] == "plateau"

    remaining = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(16)],
            "delta_history": [{"n": 1, "rel_delta": None}]
            + [
                {"n": 8, "rel_delta": 0.0289},
                {"n": 9, "rel_delta": 0.0455},
                {"n": 10, "rel_delta": 0.0292},
                {"n": 11, "rel_delta": 0.0512},
                {"n": 12, "rel_delta": 0.0150},
                {"n": 13, "rel_delta": 0.0223},
                {"n": 14, "rel_delta": 0.0143},
                {"n": 15, "rel_delta": 0.0096},
                {"n": 16, "rel_delta": 0.0170},
            ],
        },
        opening_stats=[
            {"opening_correction_burden": value} for value in formal_open
        ],
    )
    assert remaining["stop_reason_if_ready"] is None
    assert remaining["estimated_additional_donors"] == 3
    assert remaining["behavior_candidate_n"] == 14
    assert remaining["parameter_candidate_n"] == 14
    assert remaining["candidate_n"] == 14
    assert remaining["rebound_confirmed"] is False
    assert remaining["confidence"] < 1


def test_single_rebound_window_does_not_lock_or_zero_remaining() -> None:
    formal_open = [
        0.02740, 0.01798, 0.03671, 0.01732, 0.00739, 0.00928,
        0.03426, 0.01327, 0.05758,
    ]
    summary = _behavior_fit_summary(
        [{"opening_correction_burden": value} for value in formal_open]
    )
    assert summary["current_state"] == "rebound"
    assert summary["rebound_confirmed"] is False
    assert summary["rebound_need_n"] is not None
    assert int(summary["rebound_need_n"]) > 9

    remaining = _share_fit_remaining_estimate(
        {"completed_donor_ids": [f"p{index}" for index in range(9)]},
        opening_stats=[
            {"opening_correction_burden": value} for value in formal_open
        ],
    )
    assert remaining["current_behavior_state"] == "rebound"
    assert remaining["rebound_confirmed"] is False
    assert remaining["stop_reason_if_ready"] is None
    assert remaining["estimated_additional_donors"] >= 1
    assert remaining["confidence"] < 1
    assert remaining["history"][9]["estimated_additional_donors"] >= 1
    assert remaining["history"][9]["confidence"] < 1


def test_consecutive_rebound_after_min_plus_window_confirms_lock() -> None:
    values = [
        0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.10,
        0.18, 0.40, 0.40, 0.40,
    ]
    at11 = _behavior_fit_summary(
        [{"opening_correction_burden": value} for value in values[:11]]
    )
    assert at11["current_state"] == "rebound"
    assert at11["rebound_confirmed"] is False
    assert at11["n_first_min"] == 8

    at12 = _behavior_fit_summary(
        [{"opening_correction_burden": value} for value in values]
    )
    assert at12["current_state"] == "rebound"
    assert at12["rebound_confirmed"] is True
    assert at12["rebound_need_n"] == 12

    remaining = _share_fit_remaining_estimate(
        {"completed_donor_ids": [f"p{index}" for index in range(12)]},
        opening_stats=[{"opening_correction_burden": value} for value in values],
    )
    assert remaining["rebound_confirmed"] is True
    assert remaining["stop_reason_if_ready"] == "behavior_rebound"
    assert remaining["estimated_additional_donors"] == 0
    assert remaining["confidence"] == 1.0

    recovered = _behavior_fit_summary(
        [{"opening_correction_burden": value} for value in values + [0.18, 0.18]]
    )
    assert recovered["current_state"] == "plateau"
    assert recovered["rebound_confirmed"] is False


def test_share_fit_enrollment_hard_caps_at_fifty_model_donors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-curve",
        public_enrollment_run_slug="orch-curve-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    state = orch._load()
    donor_ids = [f"p-{index}" for index in range(50)]
    state["share_fit"]["enrolled_participant_ids"] = donor_ids
    state["share_fit"]["completed_donor_ids"] = donor_ids
    orch._write(state)
    monkeypatch.setattr(
        db,
        "end_to_end_completion_records",
        lambda participant_ids: {
            participant_id: {
                "participant_id": participant_id,
                "completed_ms": index,
            }
            for index, participant_id in enumerate(participant_ids)
        },
    )
    monkeypatch.setattr(
        orch, "_participant_in_current_public_run", lambda _participant_id: True
    )
    with pytest.raises(OrchestrationError, match="50 名模型 donor 上限"):
        orch.assign_enrollment()


def test_incomplete_share_enrollments_do_not_consume_completed_donor_cap(
    tmp_path: Path,
) -> None:
    base = Settings.from_env()
    settings = replace(base, data_dir=tmp_path / "data-incomplete-cap")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    orch = AutoOrchestrator(settings, StudyDB(settings))
    state = orch._load()
    state["share_fit"]["enrolled_participant_ids"] = [
        f"incomplete-{index}" for index in range(50)
    ]
    orch._write(state)

    assignment = orch.assign_enrollment()

    assert assignment["phase"] == "share_fit"


def test_operator_blocked_slots_reject_new_booking_and_keep_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta, timezone

    cst = timezone(timedelta(hours=8))
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-operator-blocked",
        public_enrollment_run_slug="operator-blocked-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    first, second, third = [
        str(row["id"])
        for row in _create_share_participants(settings, db, orch, 3)
    ]
    clock = {"seconds": 1_800_000_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    existing_start = _next_half_hour_slot(now_ms) + (2 * 60 * 60 * 1000)
    blocked_start = existing_start + (2 * 60 * 60 * 1000)
    open_start = blocked_start + (60 * 60 * 1000)
    blocked_date = datetime.fromtimestamp(blocked_start / 1000, cst).strftime(
        "%Y-%m-%d"
    )

    booked = orch.book_share_appointment(first, existing_start)
    assert booked["status"] == "booked"
    orch.replace_operator_blocked_slots(blocked_date, [blocked_start])
    with pytest.raises(ShareAppointmentError) as blocked:
        orch.book_share_appointment(second, blocked_start)
    assert blocked.value.code == "share_appointment_operator_blocked"
    assert orch.share_appointment_status(first)["status"] == "booked"
    assert orch.book_share_appointment(third, open_start)["status"] == "booked"
    slot_row = next(
        row
        for row in orch.share_appointment_status(second)["time_slots"]
        if int(row["start_ms"]) == blocked_start
    )
    assert slot_row["available"] is False
    assert slot_row["reason"] == "operator_blocked"
    admin = orch.status()["share_fit"]
    assert any(
        int(row["scheduled_start_ms"]) == blocked_start
        for row in admin["operator_blocked_slots"]
    )


def test_operator_blocked_slots_replace_one_date_only(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    cst = timezone(timedelta(hours=8))
    first = int(datetime(2026, 8, 27, 10, 0, tzinfo=cst).timestamp() * 1000)
    second = int(datetime(2026, 8, 28, 10, 0, tzinfo=cst).timestamp() * 1000)
    base = Settings.from_env()
    settings = replace(base, data_dir=tmp_path / "data-operator-blocked-date")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    orch = AutoOrchestrator(settings, StudyDB(settings))
    orch.replace_operator_blocked_slots("2026-08-27", [first])
    orch.replace_operator_blocked_slots("2026-08-28", [second])
    orch.replace_operator_blocked_slots("2026-08-27", [])
    starts = {
        int(row["scheduled_start_ms"])
        for row in orch.status()["share_fit"]["operator_blocked_slots"]
    }
    assert starts == {second}


def test_share_fit_remaining_estimate_uses_lock_gates_without_writes() -> None:
    improving = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(10)],
            "parameter_candidate_n": 8,
            "behavior_candidate_n": 8,
            "candidate_n": 8,
            "delta_history": [{"n": 1, "rel_delta": None}]
            + [{"n": index, "rel_delta": 0.01} for index in range(2, 11)],
            "behavior_history": [{"n": 10, "state": "improving"}],
        }
    )
    assert improving["readonly"] is True
    assert improving["estimated_additional_donors"] == 1
    assert improving["parameter_additional_donors"] == 0
    assert improving["behavior_additional_donors"] == 1
    assert improving["stop_reason_if_ready"] is None
    assert 0 <= improving["confidence"] <= 1
    assert improving["confidence"] < 1
    assert [row["n"] for row in improving["history"]] == list(range(0, 11))
    assert improving["history"][0]["estimated_additional_donors"] == 8
    assert improving["history"][-1]["estimated_additional_donors"] == 1

    ready = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(15)],
            "parameter_candidate_n": 8,
            "behavior_history": [
                {
                    "n": index,
                    "state": "plateau" if index >= 10 else "baseline",
                    "opening_correction_burden": 0.10 if index >= 9 else 0.20,
                    "rolling_burden": 0.10 if index >= 10 else 0.18,
                }
                for index in range(1, 16)
            ],
        }
    )
    assert ready["estimated_additional_donors"] == 0
    assert ready["stop_reason_if_ready"] == "parameter_and_behavior_plateau"
    assert ready["fit_point_reached"] is True
    assert ready["persist_ready"] is True

    not_yet = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(10)],
            "parameter_candidate_n": 8,
            "behavior_history": [{"n": 10, "state": "plateau"}],
        }
    )
    assert not_yet["stop_reason_if_ready"] is None
    assert not_yet["estimated_additional_donors"] >= 1
    assert not_yet["fit_point_reached"] is False

    early = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(6)],
            "delta_history": [{"n": 1, "rel_delta": None}]
            + [{"n": index, "rel_delta": 0.05} for index in range(2, 7)],
            "behavior_history": [{"n": 6, "state": "insufficient"}],
        }
    )
    assert early["estimated_additional_donors"] == 3
    assert early["parameter_additional_donors"] == 3
    assert early["behavior_additional_donors"] == 2

    locked = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(12)],
            "locked_n": 9,
            "locked_prior_id": "prior-9",
            "lock_reason": "manual",
            "behavior_history": [{"n": 12, "state": "improving"}],
        }
    )
    assert locked["estimated_additional_donors"] == 0
    assert locked["locked"] is True
    assert locked["confidence"] == 1.0

    stale_plateau = _share_fit_remaining_estimate(
        {
            "completed_donor_ids": [f"p{index}" for index in range(10)],
            "parameter_candidate_n": 8,
            "behavior_candidate_n": 8,
            "behavior_history": [{"n": 10, "state": "plateau"}],
            "delta_history": [{"n": 1, "rel_delta": None}]
            + [{"n": index, "rel_delta": 0.01} for index in range(2, 11)],
        },
        opening_stats=[{"opening_correction_burden": 0.20 - 0.01 * index} for index in range(10)],
    )
    assert stale_plateau["current_behavior_state"] == "improving"
    assert stale_plateau["estimated_additional_donors"] == 1
    assert stale_plateau["stop_reason_if_ready"] is None
    assert stale_plateau["history"][0]["estimated_additional_donors"] == 8
    assert stale_plateau["history"][-1]["confidence"] == stale_plateau["confidence"]


def test_share_fit_series_trend_drops_iqr_outliers_and_reports_p() -> None:
    trend = _share_fit_series_trend(
        list(range(1, 8)),
        [0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 1.80],
        ["a", "b", "c", "d", "e", "f", "out"],
        lower_is_better=True,
    )
    assert "out" in trend["dropped_codes"]
    assert trend["p_value"] is not None
    assert trend["significant_label"] in {"显著", "不显著"}
    assert "向好" in trend["reading"]
    spearman = _spearman_trend([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert spearman["rho"] == -1.0
    assert spearman["p_value"] is not None
    assert spearman["p_value"] < 0.05


def test_peek_share_fit_trend_board_does_not_write(tmp_path: Path) -> None:
    settings = replace(
        Settings.from_env(),
        data_dir=tmp_path / "peek-trend",
        media_dir=tmp_path / "peek-trend" / "media",
        public_enrollment_run_slug="trend-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    empty = orch.peek_share_fit_trend_board(final_items={})
    assert empty["readonly"] is True
    assert empty["writes_orchestration"] is False
    assert not orch.path.is_file()

    codes = db.create_participants(
        3,
        "pilot",
        "personalized",
        preview_mode=False,
        data_origin="pilot",
        protocol_mode="strict",
        run_slug="trend-run",
        analysis_excluded=True,
    )
    participants = [db.participant_by_code(code) for code in codes]
    donor_ids = [str(row["id"]) for row in participants if row is not None]
    for index, participant in enumerate(participants):
        assert participant is not None
        session = db.start_session(participant["id"], 1, "study")
        for block in range(1, 6):
            for _ in range(10):
                db.append_event(
                    participant["id"],
                    session["id"],
                    "model.prediction",
                    None,
                    None,
                    {
                        "analysis_eligible": True,
                        "decision_eligible": True,
                        "active_block_index": block,
                    },
                )
            for _ in range(max(1, 4 - index)):
                db.append_event(
                    participant["id"],
                    session["id"],
                    "tap.manual_activate",
                    None,
                    None,
                    {
                        "analysis_eligible": True,
                        "training_eligible": True,
                        "active_block_index": block,
                    },
                )
        db.save_response(
            participant["id"],
            session["id"],
            "final",
            "test",
            {"response_predictability": 5 + index, "start_stop_control": 4},
        )
    payload = {
        "schema": "tap-to-adapt-auto-orchestration/v1",
        "status": "running",
        "phase": "share_fit",
        "share_fit": {
            "completed_donor_ids": donor_ids,
            "delta_history": [{"n": 1, "rel_delta": None}]
            + [{"n": index, "rel_delta": 0.05} for index in range(2, 4)],
            "behavior_history": [{"n": 3, "state": "insufficient"}],
        },
    }
    orch.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    before = orch.path.read_bytes()
    board = orch.peek_share_fit_trend_board(
        final_items={
            "response_predictability": {
                "id": "response_predictability",
                "dimension": "predictability",
            },
            "start_stop_control": {
                "id": "start_stop_control",
                "dimension": "perceived_control",
            },
        }
    )
    assert orch.path.read_bytes() == before
    assert board["readonly"] is True
    assert board["completed_n"] == 3
    assert len(board["donors"]) == 3
    assert board["donors"][0]["blocks"]["1"]["q"] is not None
    assert board["block_trends"][0]["taps_per_minute"]["reading"]
    assert any(row["dimension"] == "predictability" for row in board["questionnaire_trends"])
    remaining = board["fit_remaining_estimate"]
    assert remaining["estimated_additional_donors"] == 5
    assert remaining["history"][0]["n"] == 0
    assert remaining["history"][-1]["n"] == 3
    assert 0 <= remaining["confidence"] <= 1


def test_share_fit_stop_rule_is_bounded_and_keeps_safety_priority() -> None:
    assert _share_fit_stop_reason(
        completed_n=50,
        max_n=50,
        behavior_state="improving",
        parameter_candidate_n=None,
    ) == "max_donors"
    assert (
        _share_fit_stop_reason(
            completed_n=9,
            max_n=50,
            behavior_state="rebound",
            parameter_candidate_n=None,
        )
        is None
    )
    assert _share_fit_stop_reason(
        completed_n=12,
        max_n=50,
        behavior_state="rebound",
        parameter_candidate_n=None,
        rebound_confirmed=True,
    ) == "behavior_rebound"
    assert (
        _share_fit_stop_reason(
            completed_n=9,
            max_n=50,
            behavior_state="plateau",
            parameter_candidate_n=7,
        )
        is None
    )
    assert _share_fit_stop_reason(
        completed_n=9,
        max_n=50,
        behavior_state="plateau",
        parameter_candidate_n=7,
        persist_ready=True,
    ) == "parameter_and_behavior_plateau"
    assert _share_fit_stop_reason(
        completed_n=9,
        max_n=50,
        behavior_state="improving",
        parameter_candidate_n=7,
        persist_ready=True,
    ) is None
    assert _share_fit_stop_reason(
        completed_n=22,
        max_n=50,
        behavior_state="improving",
        parameter_candidate_n=14,
        persist_ready=False,
    ) == "confirm_horizon"


def test_plateau_locks_terminal_min_after_nonoverlap_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.distilhubert_online import LinearHeadState

    base = Settings.from_env()
    settings = replace(base, data_dir=tmp_path / "data-lock")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    donor_ids = [f"donor-{index}" for index in range(1, 16)]
    state = orch._load()
    state["share_fit"]["completed_donor_ids"] = donor_ids

    monkeypatch.setattr(
        orch,
        "_ensure_prior_for_donors",
        lambda donors: {
            "id": f"prior-{len(donors)}",
            "run_id": "run-1",
            "encoder_path": f"head-{len(donors)}.npz",
        },
    )
    monkeypatch.setattr(
        orchestration_module.DistilHuBERTRelease,
        "load",
        lambda _path: type("Release", (), {"feature_dim": 1})(),
    )
    monkeypatch.setattr(
        orchestration_module,
        "_validated_head",
        lambda _path, _dim: LinearHeadState(
            feature_mean=np.zeros(1, dtype=np.float32),
            feature_scale=np.ones(1, dtype=np.float32),
            weight=np.ones(1, dtype=np.float32),
            bias=0.0,
        ),
    )
    burdens = dict(
        zip(
            donor_ids,
            [0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            strict=True,
        )
    )
    monkeypatch.setattr(
        orch,
        "_participant_tap_stats",
        lambda participant_id: {
            "correction_burden": burdens[participant_id],
            "opening_correction_burden": burdens[participant_id],
        },
    )
    selected: list[tuple[str, str]] = []
    monkeypatch.setattr(
        db,
        "select_shared_prior",
        lambda _run_id, prior_id, *, actor, reason: selected.append((prior_id, reason)),
    )
    monkeypatch.setattr(
        db,
        "end_to_end_completion_records",
        lambda participant_ids: {
            participant_id: {
                "participant_id": participant_id,
                "completed_ms": index,
            }
            for index, participant_id in enumerate(participant_ids)
        },
    )
    monkeypatch.setattr(
        orch, "_participant_in_current_public_run", lambda _participant_id: True
    )

    orch._refresh_share_fit_locked(state)

    assert state["phase"] == "contrast"
    assert state["share_fit"]["locked_n"] == 10
    assert state["share_fit"]["lock_reason"] == "parameter_and_behavior_plateau"
    assert selected == [("prior-10", "auto share-fit parameter_and_behavior_plateau N*=10")]


def test_share_fit_weight_snapshot_is_read_only(tmp_path: Path) -> None:
    from app.integrity import file_sha256
    from app.orchestration import DISTILHUBERT_PRIOR_ALGORITHM

    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-weights",
        public_enrollment_run_slug="orch-weights-run",
        instance_id="formal",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    run_id = "run-weights"
    with db.connect() as conn:
        conn.execute(
            """INSERT INTO study_runs
               (id, slug, data_origin, protocol_version, protocol_sha256,
                protocol_json, protocol_mode, created_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                "orch-weights-run",
                "formal",
                "test-protocol",
                "0" * 64,
                "{}",
                "strict",
                0,
            ),
        )
    prior_id = "prior-readonly-1"
    prior_dir = settings.data_dir / "priors" / run_id / prior_id
    prior_dir.mkdir(parents=True, exist_ok=True)
    head_path = prior_dir / "personal_head.npz"
    pointer_path = prior_dir / "distilhubert-prior.json"
    head_bytes = b"FAKE-HEAD-BYTES"
    pointer_bytes = b'{"schema":"tap-to-adapt-distilhubert-shared-prior/v1"}\n'
    head_path.write_bytes(head_bytes)
    pointer_path.write_bytes(pointer_bytes)
    db.create_shared_prior(
        {
            "id": prior_id,
            "run_id": run_id,
            "donor_count": 3,
            "opportunity_budget": 0,
            "status": "candidate",
            "algorithm": DISTILHUBERT_PRIOR_ALGORITHM,
            "donor_ids": [],
            "encoder_path": str(head_path),
            "main_path": str(pointer_path),
            "encoder_sha256": file_sha256(head_path),
            "main_sha256": file_sha256(pointer_path),
            "manifest": {"donors": []},
        }
    )
    state = {
        "schema": "tap-to-adapt-auto-orchestration/v1",
        "share_fit": {
            "completed_donor_ids": ["donor-a", "donor-b", "donor-c"],
            "locked_prior_id": prior_id,
        },
    }
    orch.path.write_text(json.dumps(state), encoding="utf-8")
    before_head = (head_path.stat().st_mtime_ns, head_path.read_bytes())
    before_pointer = (pointer_path.stat().st_mtime_ns, pointer_path.read_bytes())
    before_state = orch.path.read_text(encoding="utf-8")
    files_before = {path.relative_to(settings.data_dir) for path in settings.data_dir.rglob("*") if path.is_file()}

    snapshot = orch.existing_share_fit_weight_snapshot()

    assert snapshot["prior_id"] == prior_id
    assert snapshot["source"] == "locked"
    assert snapshot["head_bytes"] == head_bytes
    assert snapshot["pointer_bytes"] == pointer_bytes
    assert (head_path.stat().st_mtime_ns, head_path.read_bytes()) == before_head
    assert (pointer_path.stat().st_mtime_ns, pointer_path.read_bytes()) == before_pointer
    assert orch.path.read_text(encoding="utf-8") == before_state
    files_after = {path.relative_to(settings.data_dir) for path in settings.data_dir.rglob("*") if path.is_file()}
    assert files_after == files_before


def test_session_lock_blocks_walk_in_and_slots_without_writing_orch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-session-lock",
        public_enrollment_run_slug="session-lock-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    clock = {"seconds": 1_800_100_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    monkeypatch.setattr(session_lock_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    lock_start = _next_half_hour_slot(now_ms)
    lock_end = lock_start + (2 * 60 * 60 * 1000)
    open_start = lock_end + (60 * 60 * 1000)
    orch._load()
    before_orch = orch.path.read_text(encoding="utf-8")
    before_chain = json.loads(before_orch)["share_fit"]["completed_donor_ids"]

    locked = orch.set_session_lock(lock_start, lock_end, actor="admin")
    assert locked["active"] is False
    assert locked["upcoming"] is True
    assert locked["password"]
    assert orch.path.read_text(encoding="utf-8") == before_orch
    password = locked["password"]
    first, second, third = [
        str(row["id"])
        for row in _create_share_participants(settings, db, orch, 3)
    ]

    slot_row = next(
        row
        for row in orch.share_appointment_status(second)["time_slots"]
        if int(row["start_ms"]) == lock_start
    )
    assert slot_row["available"] is False
    assert slot_row["reason"] == "session_lock"
    with pytest.raises(ShareAppointmentError) as blocked_book:
        orch.book_share_appointment(second, lock_start)
    assert blocked_book.value.code == "share_appointment_session_lock"
    assert orch.book_share_appointment(third, open_start)["status"] == "booked"

    clock["seconds"] = (lock_start + 60_000) / 1000
    assert orch.peek_session_lock()["active"] is True
    available = orch.share_appointment_status(first)
    assert available["walk_in_available"] is False
    with pytest.raises(ShareAppointmentError) as blocked_start:
        orch.claim_share_session_start(first)
    assert blocked_start.value.code == "session_lock_active"
    claimed = orch.claim_share_session_start(first, group_session_password=password)
    assert claimed["holder_is_self"] is True

    after_lock = json.loads(orch.path.read_text(encoding="utf-8"))
    assert after_lock["share_fit"]["completed_donor_ids"] == before_chain
    assert after_lock["share_fit"]["chain_commit_version"] == 3
    cleared = orch.clear_session_lock(actor="admin")
    assert cleared["active"] is False
    assert cleared["configured"] is False


def test_session_lock_supports_two_future_day_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-session-lock-days",
        public_enrollment_run_slug="session-lock-days-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    clock = {"seconds": 1_800_300_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    monkeypatch.setattr(session_lock_module.time, "time", lambda: clock["seconds"])
    now_ms = int(clock["seconds"] * 1000)
    first_start = _next_half_hour_slot(now_ms)
    first_end = first_start + (14 * 60 * 60 * 1000)
    second_start = first_start + (3 * 24 * 60 * 60 * 1000)
    second_end = second_start + (14 * 60 * 60 * 1000)
    orch._load()
    before = orch.path.read_text(encoding="utf-8")
    locked = orch.set_session_lock(
        windows=[(first_start, first_end), (second_start, second_end)],
        actor="admin",
    )
    assert orch.path.read_text(encoding="utf-8") == before
    assert locked["upcoming"] is True
    assert len(locked["windows"]) == 2
    assert locked["windows"][0]["start_ms"] == first_start
    assert locked["windows"][1]["start_ms"] == second_start
    first = str(_create_share_participants(settings, db, orch, 1)[0]["id"])
    slot_a = next(
        row
        for row in orch.share_appointment_status(first)["time_slots"]
        if int(row["start_ms"]) == first_start
    )
    assert slot_a["reason"] == "session_lock"
    assert orch.session_lock.blocks_slot(second_start, now_ms=now_ms) is True
    assert orch.session_lock.blocks_slot(first_end, now_ms=now_ms) is False


def test_session_lock_keeps_current_holder_without_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = Settings.from_env()
    settings = replace(
        base,
        data_dir=tmp_path / "data-session-lock-holder",
        public_enrollment_run_slug="session-lock-holder-run",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = StudyDB(settings)
    orch = AutoOrchestrator(settings, db)
    participant_id = str(_create_share_participants(settings, db, orch, 1)[0]["id"])
    clock = {"seconds": 1_800_200_000.0}
    monkeypatch.setattr(orchestration_module.time, "time", lambda: clock["seconds"])
    monkeypatch.setattr(session_lock_module.time, "time", lambda: clock["seconds"])
    claimed = orch.claim_share_session_start(participant_id)
    assert claimed["holder_is_self"] is True
    now_ms = int(clock["seconds"] * 1000)
    orch.set_session_lock(now_ms - 60_000, now_ms + (2 * 60 * 60 * 1000))
    again = orch.claim_share_session_start(participant_id)
    assert again["holder_is_self"] is True
