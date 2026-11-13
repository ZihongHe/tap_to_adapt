"""Regression tests for audit model divergence.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from app.config import settings as default_settings
from app.db import StudyDB
from scripts.audit_model_divergence import (
    AuditFailure,
    CheckpointInspector,
    audit_model_divergence,
    file_sha256,
)


def write_checkpoint(
    path: Path, weight: float, *, checkpoint_pair_id: str | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "model_state_dict": {
                "weight": torch.tensor([weight], dtype=torch.float32),
                "running_mean": torch.tensor([123.0], dtype=torch.float32),
                "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
            },
            "trainable_parameter_names": ["weight"],
            "config": {"model_structure": "audit-test"},
        }
    if checkpoint_pair_id is not None:
        payload["checkpoint_pair_id"] = checkpoint_pair_id
    torch.save(payload, path)


def write_pair(directory: Path, weight: float, *, latest: bool = False) -> tuple[Path, Path]:
    encoder_name = "latest_encoder.pth" if latest else "encoder.pth"
    main_name = "latest_main_model.pth" if latest else "main_model.pth"
    encoder = directory / encoder_name
    main = directory / main_name
    write_checkpoint(encoder, weight)
    write_checkpoint(main, weight)
    return encoder.resolve(), main.resolve()


def test_checkpoint_inspector_rejects_crossed_pair_ids(tmp_path: Path) -> None:
    encoder = tmp_path / "encoder.pth"
    main = tmp_path / "main.pth"
    write_checkpoint(encoder, 1.0, checkpoint_pair_id="pair-a")
    write_checkpoint(main, 1.0, checkpoint_pair_id="pair-b")
    inspector = CheckpointInspector()
    pair = inspector.pair("crossed", encoder.resolve(), main.resolve())

    with pytest.raises(AuditFailure, match="pair IDs do not match"):
        inspector.compare_pairs(pair, pair)


@dataclass
class Fixture:
    data_dir: Path
    run_slug: str
    run_id: str
    db: StudyDB
    personalized: dict[str, Any]
    frozen: dict[str, Any]
    selected_prior: dict[str, Any]
    selected_paths: tuple[Path, Path]
    personalized_paths: tuple[Path, Path]
    manifest_path: Path


def make_settings(tmp_path: Path):
    return replace(
        default_settings,
        data_dir=tmp_path / "data",
        media_dir=tmp_path / "data" / "media",
        token_secret="audit-test-token",
        code_pepper="audit-test-code",
        ip_pepper="audit-test-ip",
        admin_token="audit-test-admin",
        model_code_dir=str(PROJECT_ROOT),
    )


def add_participant(
    db: StudyDB,
    run_slug: str,
    cohort: str,
    group: str,
    assignment: str,
) -> dict[str, Any]:
    code = db.create_participants(
        1,
        cohort,
        group,
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug=run_slug,
    )[0]
    participant = db.participant_by_code(code)
    assert participant is not None
    db.configure_participant(
        participant["id"],
        assignment_id=assignment,
        neutral_id=assignment,
        enrollment_wave=1,
        release_state="released",
    )
    refreshed = db.participant(participant["id"])
    assert refreshed is not None
    return refreshed


def create_prior(
    db: StudyDB,
    run_id: str,
    directory: Path,
    *,
    weight: float,
    donor_count: int,
    budget: int,
    status: str,
) -> tuple[dict[str, Any], tuple[Path, Path]]:
    encoder, main = write_pair(directory, weight)
    prior = db.create_shared_prior(
        {
            "run_id": run_id,
            "donor_count": donor_count,
            "opportunity_budget": budget,
            "status": status,
            "algorithm": "audit-fixture",
            "donor_ids": [],
            "encoder_path": str(encoder),
            "main_path": str(main),
            "encoder_sha256": file_sha256(encoder),
            "main_sha256": file_sha256(main),
            "manifest": {"fixture": True},
        }
    )
    return prior, (encoder, main)


def build_fixture(tmp_path: Path, *, shared_weight: float = 1.0) -> Fixture:
    settings = make_settings(tmp_path)
    db = StudyDB(settings)
    run_slug = "model-divergence-audit"
    donor = add_participant(db, run_slug, "study_a", "donor", "A01")
    personalized = add_participant(
        db, run_slug, "study_b", "personalized", "BP01"
    )
    frozen = add_participant(db, run_slug, "study_b", "frozen", "BF01")
    run = db.study_run(run_slug)
    assert run is not None
    run_id = str(run["id"])

    base_prior, _ = create_prior(
        db,
        run_id,
        settings.data_dir / "priors" / run_id / "m0",
        weight=0.0,
        donor_count=0,
        budget=0,
        status="candidate",
    )
    selected_prior, selected_paths = create_prior(
        db,
        run_id,
        settings.data_dir / "priors" / run_id / "selected",
        weight=shared_weight,
        donor_count=1,
        budget=8,
        status="selected",
    )
    assert base_prior["id"] != selected_prior["id"]
    for participant in (personalized, frozen):
        db.assign_prior(participant["id"], selected_prior["id"], actor="audit-test")
    personalized = db.participant(personalized["id"])
    frozen = db.participant(frozen["id"])
    assert personalized is not None and frozen is not None

    personalized_paths = write_pair(
        settings.data_dir / "models" / personalized["id"], 2.0, latest=True
    )
    for budget, weight in ((2, 1.25), (4, 1.5), (8, 2.0)):
        snapshot_paths = write_pair(
            settings.data_dir
            / "snapshots"
            / personalized["id"]
            / "personalized"
            / f"q-{budget}",
            weight,
        )
        db.record_model_snapshot(
            {
                "participant_id": personalized["id"],
                "source": "personalized",
                "opportunity_budget": budget,
                "budget_kind": "eligible_feedback",
                "eligible_feedbacks": budget,
                "positive_feedbacks": budget - (budget // 3),
                "negative_feedbacks": budget // 3,
                "base_prior_id": selected_prior["id"],
                "encoder_path": str(snapshot_paths[0]),
                "main_path": str(snapshot_paths[1]),
                "encoder_sha256": file_sha256(snapshot_paths[0]),
                "main_sha256": file_sha256(snapshot_paths[1]),
                "metadata": {"fixture": True},
            }
        )

    protocol = json.loads(str(run["protocol_json"]))
    calibration_version = protocol["initialization"]["version"]
    protocol_sha256 = str(run["protocol_sha256"])
    db.calibration_state(
        personalized["id"],
        calibration_version=calibration_version,
        protocol_sha256=protocol_sha256,
        initial_logit_bias=0.0,
    )
    for update_index in range(8):
        label = 0 if update_index in {2, 5} else 1
        db.update_calibration_state(
            personalized["id"],
            calibration_version=calibration_version,
            protocol_sha256=protocol_sha256,
            expected_update_count=update_index,
            personal_logit_bias=0.025 * (update_index + 1),
            label=label,
        )
        db.next_activation_opportunity(personalized["id"])
        db.next_eligible_feedback(personalized["id"], label)

    expected_fingerprint = {
        "encoder_sha256": selected_prior["encoder_sha256"],
        "main_sha256": selected_prior["main_sha256"],
    }
    db.append_event(
        frozen["id"],
        None,
        "model.ready",
        None,
        None,
        {"model_fingerprint": expected_fingerprint},
    )

    assignments: dict[str, dict[str, Any]] = {}
    tested_assignments: dict[str, dict[str, Any]] = {}
    budgets = {1: 2, 2: 4, 3: 8}
    for participant in (personalized, frozen):
        participant_id = str(participant["id"])
        assignments[participant_id] = {}
        tested_assignments[participant_id] = {}
        for day, budget in budgets.items():
            matched_paths = write_pair(
                settings.data_dir / "matched" / run_id / donor["id"] / f"day-{day}",
                3.0 + day,
            )
            encoder_hash = file_sha256(matched_paths[0])
            main_hash = file_sha256(matched_paths[1])
            assignments[participant_id][str(day)] = {
                "donor_participant_id": donor["id"],
                "matching_strata": {
                    "shared_prior_id": selected_prior["id"],
                    "opportunity_budget": budget,
                    "budget_kind": "eligible_feedback",
                    "day": day,
                },
                "encoder_checkpoint": str(matched_paths[0]),
                "encoder_sha256": encoder_hash,
                "main_checkpoint": str(matched_paths[1]),
                "main_sha256": main_hash,
            }
            tested_assignments[participant_id][str(day)] = {
                "donor_participant_id": donor["id"],
                "encoder_sha256": encoder_hash,
                "main_sha256": main_hash,
            }

    matched_root = settings.data_dir / "matched" / run_id
    manifest_path = matched_root / "manifest.json"
    sentinel_path = matched_root / "self-test.json"
    manifest = {
        "version": "audit-fixture-v1",
        "approved": True,
        "run_id": run_id,
        "run_slug": run_slug,
        "assignment_rule": "fixed audit fixture",
        "frozen_at_ms": 1,
        "assignments": assignments,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sentinel = {
        "passed": True,
        "manifest_sha256": file_sha256(manifest_path),
        "assignments": tested_assignments,
    }
    sentinel_path.write_text(
        json.dumps(sentinel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return Fixture(
        data_dir=settings.data_dir,
        run_slug=run_slug,
        run_id=run_id,
        db=db,
        personalized=personalized,
        frozen=frozen,
        selected_prior=selected_prior,
        selected_paths=selected_paths,
        personalized_paths=personalized_paths,
        manifest_path=manifest_path,
    )


def test_audit_passes_and_does_not_modify_database(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    database = fixture.data_dir / "study.sqlite3"
    database_hash_before = file_sha256(database)

    report = audit_model_divergence(fixture.data_dir, fixture.run_slug)

    assert file_sha256(database) == database_hash_before
    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["assertions"] == {
        "all_recorded_hashes_match": True,
        "shared_differs_from_m0": True,
        "all_personalized_checkpoints_differ_from_prior": True,
        "all_frozen_have_no_personal_updates": True,
        "all_frozen_effective_checkpoints_equal_prior": True,
        "all_matched_assignments_present_and_hash_bound": True,
    }
    assert report["shared"]["selected_shared_vs_m0"]["combined_l2_distance"] > 0
    personalized = report["participants"]["personalized"][0]
    assert personalized["latest_vs_prior"]["combined_l2_distance"] > 0
    assert personalized["latest_vs_latest_snapshot"]["exact_tensor_equal"] is True
    frozen = report["participants"]["frozen"][0]
    assert frozen["model_ready_fingerprints_match_prior"] is True
    assert frozen["personalized_snapshot_count"] == 0
    assert report["matched"]["assignment_count"] == 6
    assert all(
        row["matched_vs_selected_shared"]["combined_l2_distance"] > 0
        for row in report["matched"]["assignments"]
    )


def test_audit_fails_when_shared_prior_equals_m0(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path, shared_weight=0.0)

    with pytest.raises(
        AuditFailure,
        match="selected shared prior must differ from M0",
    ):
        audit_model_divergence(fixture.data_dir, fixture.run_slug)


def test_audit_fails_when_personalized_checkpoint_equals_prior(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    for path in fixture.personalized_paths:
        write_checkpoint(path, 1.0)

    with pytest.raises(
        AuditFailure,
        match="latest checkpoint still equals its prior",
    ):
        audit_model_divergence(fixture.data_dir, fixture.run_slug)


def test_audit_fails_when_frozen_has_personal_update(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    run = fixture.db.study_run(fixture.run_slug)
    assert run is not None
    protocol = json.loads(str(run["protocol_json"]))
    calibration_version = protocol["initialization"]["version"]
    fixture.db.calibration_state(
        fixture.frozen["id"],
        calibration_version=calibration_version,
        protocol_sha256=run["protocol_sha256"],
        initial_logit_bias=0.0,
    )
    fixture.db.update_calibration_state(
        fixture.frozen["id"],
        calibration_version=calibration_version,
        protocol_sha256=run["protocol_sha256"],
        expected_update_count=0,
        personal_logit_bias=-0.1,
        label=0,
    )

    with pytest.raises(
        AuditFailure,
        match="forbidden personal calibration updates",
    ):
        audit_model_divergence(fixture.data_dir, fixture.run_slug)


def test_audit_fails_closed_when_matched_manifest_is_missing(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    fixture.manifest_path.unlink()

    with pytest.raises(AuditFailure, match="matched manifest is missing"):
        audit_model_divergence(fixture.data_dir, fixture.run_slug)


def test_audit_fails_when_recorded_checkpoint_hash_is_stale(tmp_path: Path) -> None:
    fixture = build_fixture(tmp_path)
    manifest = json.loads(fixture.manifest_path.read_text(encoding="utf-8"))
    participant_assignments = manifest["assignments"][fixture.personalized["id"]]
    matched_encoder = Path(participant_assignments["1"]["encoder_checkpoint"])
    write_checkpoint(matched_encoder, 99.0)

    with pytest.raises(AuditFailure, match="SHA-256 does not match its immutable record"):
        audit_model_divergence(fixture.data_dir, fixture.run_slug)
