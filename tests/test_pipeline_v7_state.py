"""Regression tests for pipeline v7 state.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from app.config import settings as default_settings
from app.db import StudyDB
from app.integrity import file_sha256
from app.pipeline import FormalPipeline, PipelineError


def make_settings(tmp_path: Path):
    base_encoder = tmp_path / "base" / "encoder.pth"
    base_main = tmp_path / "base" / "main.pth"
    base_encoder.parent.mkdir(parents=True)
    base_encoder.write_bytes(b"encoder-m0")
    base_main.write_bytes(b"main-m0")
    return replace(
        default_settings,
        data_dir=tmp_path / "data",
        media_dir=tmp_path / "data" / "media",
        token_secret="pipeline-v7-token",
        code_pepper="pipeline-v7-code",
        ip_pepper="pipeline-v7-ip",
        admin_token="pipeline-v7-admin",
        base_encoder_checkpoint=str(base_encoder),
        base_main_checkpoint=str(base_main),
        matched_model_manifest=str(tmp_path / "data" / "matched-manifest.json"),
        matched_model_ready_sentinel=str(tmp_path / "data" / "matched-self-test.json"),
        model_code_dir=str(PROJECT_ROOT),
    )


def create_block(
    db: StudyDB,
    run_slug: str,
    *,
    count: int,
    cohort: str,
    group: str,
    prefix: str,
    start: int,
    wave: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, code in enumerate(
        db.create_participants(
            count,
            cohort,
            group,
            data_origin="formal",
            protocol_mode="strict",
            run_slug=run_slug,
        )
    ):
        index = start + offset
        participant = db.participant_by_code(code)
        assert participant is not None
        db.configure_participant(
            participant["id"],
            assignment_id=f"{prefix}{index:02d}",
            neutral_id=f"N-{prefix}-{index:02d}",
            enrollment_wave=wave,
            release_state="held",
        )
        refreshed = db.participant(participant["id"])
        assert refreshed is not None
        rows.append(refreshed)
    return rows


def create_roster(db: StudyDB, run_slug: str) -> list[dict[str, Any]]:
    study_a = []
    study_a += create_block(
        db,
        run_slug,
        count=12,
        cohort="study_a",
        group="seed_donor",
        prefix="A",
        start=1,
        wave=1,
    )
    study_a += create_block(
        db,
        run_slug,
        count=8,
        cohort="study_a",
        group="evaluator_donor",
        prefix="A",
        start=13,
        wave=2,
    )
    study_a += create_block(
        db,
        run_slug,
        count=8,
        cohort="study_a",
        group="evaluator_donor",
        prefix="A",
        start=21,
        wave=3,
    )
    study_a += create_block(
        db,
        run_slug,
        count=8,
        cohort="study_a",
        group="holdout_evaluator",
        prefix="A",
        start=29,
        wave=4,
    )
    for prefix, group in (("BP", "personalized"), ("BF", "frozen")):
        create_block(
            db,
            run_slug,
            count=16,
            cohort="study_b",
            group=group,
            prefix=prefix,
            start=1,
            wave=5,
        )
        create_block(
            db,
            run_slug,
            count=2,
            cohort="study_b",
            group=group,
            prefix=prefix,
            start=17,
            wave=6,
        )
    return study_a


def fake_shared_prior(
    db: StudyDB,
    settings: Any,
    run_id: str,
    donor_ids: list[str],
    budget: int,
    budget_kind: str,
    budget_semantics: str,
    algorithm: str,
) -> dict[str, Any]:
    existing = [
        row
        for row in db.shared_priors(run_id)
        if int(row["donor_count"]) == len(donor_ids)
        and int(row["opportunity_budget"]) == budget
    ]
    if existing:
        return existing[0]
    snapshots = {
        row["participant_id"]: row
        for row in db.model_snapshots(
            run_id,
            source="donor_shadow",
            budget=budget,
            budget_kind=budget_kind,
        )
    }
    assert set(donor_ids).issubset(snapshots)
    prior_id = str(uuid.uuid4())
    target = settings.data_dir / "priors" / run_id / prior_id
    target.mkdir(parents=True)
    encoder = target / "encoder.pth"
    main = target / "main_model.pth"
    shutil.copy2(snapshots[donor_ids[0]]["encoder_path"], encoder)
    shutil.copy2(snapshots[donor_ids[0]]["main_path"], main)
    return db.create_shared_prior(
        {
            "id": prior_id,
            "run_id": run_id,
            "donor_count": len(donor_ids),
            "opportunity_budget": budget,
            "status": "candidate",
            "algorithm": algorithm,
            "donor_ids": donor_ids,
            "encoder_path": str(encoder.resolve()),
            "main_path": str(main.resolve()),
            "encoder_sha256": file_sha256(encoder),
            "main_sha256": file_sha256(main),
            "manifest": {
                "budget_kind": budget_kind,
                "budget_semantics": budget_semantics,
                "fixture": True,
            },
        }
    )


def fake_delta(
    _base: Path, _shared: Path, donor: Path, output: Path, _algorithm: str
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(donor, output)


def finish_participant(
    db: StudyDB, settings: Any, participant: dict[str, Any], *, donor: bool
) -> None:
    for day in range(1, 4):
        study = db.start_session(participant["id"], day, "study")
        db.start_session(participant["id"], day, "companion")
        if day == 1:
            for assignment in db.prior_probe_assignments(
                participant_id=participant["id"], day=1
            ):
                payload = {
                    "phase": "candidate_probe",
                    "policy_token": assignment["probe_token"],
                }
                for event_type in (
                    "phase.start",
                    "phase.server_committed",
                    "phase.end",
                ):
                    db.append_event(
                        participant["id"], study["id"], event_type, None, None, payload
                    )
        db.end_session(study["id"], participant["id"])
    if not donor:
        return
    for label in (1, 1, 0, 1, 1, 0, 1, 1):
        db.next_activation_opportunity(participant["id"])
        db.next_eligible_feedback(participant["id"], label)
    for budget in (2, 4, 8):
        target = settings.data_dir / "snapshots" / participant["id"] / f"q-{budget}"
        target.mkdir(parents=True)
        encoder = target / "encoder.pth"
        main = target / "main.pth"
        encoder.write_bytes(f"encoder-{participant['assignment_id']}-{budget}".encode())
        main.write_bytes(f"main-{participant['assignment_id']}-{budget}".encode())
        db.record_model_snapshot(
            {
                "participant_id": participant["id"],
                "source": "donor_shadow",
                "opportunity_budget": budget,
                "budget_kind": "eligible_feedback",
                "eligible_feedbacks": budget,
                "positive_feedbacks": budget - (1 if budget >= 4 else 0),
                "negative_feedbacks": 1 if budget >= 4 else 0,
                "encoder_path": str(encoder.resolve()),
                "main_path": str(main.resolve()),
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
            }
        )


def test_v7_full_state_machine_and_probe_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.pipeline.build_shared_prior", fake_shared_prior)
    monkeypatch.setattr("app.pipeline.delta_transport_checkpoint", fake_delta)
    settings = make_settings(tmp_path)
    db = StudyDB(settings)
    study = json.loads(
        (settings.content_dir / "study_protocol.json").read_text(encoding="utf-8")
    )
    pipeline = FormalPipeline(db, settings, study)
    run_slug = "formal-v7-state-machine"
    study_a = create_roster(db, run_slug)

    assert len(pipeline.release_study_a_wave(run_slug, 1)["released"]) == 12
    with pytest.raises(PipelineError):
        pipeline.release_study_a_wave(run_slug, 2)

    for wave in range(1, 5):
        current = [row for row in study_a if int(row["enrollment_wave"]) == wave]
        for participant in current:
            finish_participant(
                db,
                settings,
                participant,
                donor=participant["group_name"] != "holdout_evaluator",
            )
        if wave < 4:
            released = pipeline.release_study_a_wave(run_slug, wave + 1)
            assert len(released["released"]) == 8

    run = db.study_run(run_slug)
    assert run is not None
    probes = db.prior_probe_assignments(run_id=run["id"], day=1)
    assert len(probes) == 24 * 4
    assert not {
        row["participant_id"] for row in probes
    }.intersection({row["id"] for row in study_a[:12]})
    assert all(
        len(
            db.prior_probe_assignments(participant_id=participant["id"], day=1)
        )
        == 4
        for participant in study_a[12:]
    )
    holdout_ids = {row["id"] for row in study_a[28:]}
    assert not any(
        row["participant_id"] in holdout_ids
        for row in db.model_snapshots(run["id"], source="donor_shadow")
    )

    first_probe = probes[0]
    with pytest.raises(sqlite3.IntegrityError):
        with db.connect() as conn:
            conn.execute(
                "UPDATE prior_probe_assignments SET order_index=99 WHERE id=?",
                (first_probe["id"],),
            )

    prepared = pipeline.prepare_study_b(
        run_slug,
        20,
        4,
        "Blinded holdout decision selected the preregistered saturation cell M(20,4).",
    )
    assert prepared["selected_n"] == 20
    assert prepared["selected_q"] == 4
    assert len(prepared["released"]) == 32
    participants = db.participants_in_run(run["id"])
    study_b = [row for row in participants if row["cohort"] == "study_b"]
    assert {row["prior_id"] for row in study_b} == {prepared["selected_prior_id"]}
    assert sum(row["release_state"] == "released" for row in study_b) == 32
    assert sum(row["release_state"] == "held" for row in study_b) == 4

    status = pipeline.status(run_slug)
    assert status["candidate_probes"]["participants_assigned"] == 24
    assert status["candidate_probes"]["participants_complete"] == 24
    assert status["snapshot_counts"] == {"2": 28, "4": 28, "8": 28}
    analysis = pipeline.analysis(run_slug)
    assert {(row["n"], row["q"]) for row in analysis["candidate_grid"]} == {
        (0, 0),
        (12, 2),
        (12, 4),
        (12, 8),
        (20, 4),
        (20, 8),
        (28, 4),
        (28, 8),
    }
    assert analysis["contribution_decomposition"]["estimable"] is False
