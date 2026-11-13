"""Regression tests for pipeline.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import json
import math
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from app.config import settings as default_settings
from app.db import StudyDB
from app.integrity import file_sha256
from app.matched import matched_assignment, matched_paths
from app.pipeline import FormalPipeline, PipelineError
from app.priors import (
    average_checkpoint_files,
    create_base_prior,
    delta_transport_checkpoint,
    prior_readiness,
)
from app.timing import TimingService
from scripts.audit_model_divergence import audit_model_divergence


def write_checkpoint(path: Path, weight: float, counter: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                "weight": torch.tensor([weight], dtype=torch.float32),
                "num_batches_tracked": torch.tensor(counter, dtype=torch.int64),
            },
            "config": {"model_structure": "test"},
        },
        path,
    )


def checkpoint_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]


def test_checkpoint_aggregation_and_delta_buffer_rules(tmp_path: Path) -> None:
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    averaged = tmp_path / "averaged.pth"
    transported = tmp_path / "transported.pth"
    write_checkpoint(first, 1.0, 2)
    write_checkpoint(second, 3.0, 7)

    average_checkpoint_files([first, second], averaged)
    state = checkpoint_state(averaged)
    assert state["weight"].item() == pytest.approx(2.0)
    assert state["num_batches_tracked"].item() == 7

    delta_transport_checkpoint(first, averaged, second, transported)
    state = checkpoint_state(transported)
    assert state["weight"].item() == pytest.approx(4.0)
    assert state["num_batches_tracked"].item() == 7


def test_zero_coordinate_base_prior_is_ready_after_assignment(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db = StudyDB(settings)
    run_slug = "base-prior-readiness"
    participant = add_block(db, run_slug, 1, "study_a", "donor", "A")[0]
    base = create_base_prior(db, settings, participant["run_id"])
    db.assign_prior(participant["id"], base["id"], actor="test")
    db.set_release_state(participant["id"], "released", actor="test", reason="fixture")
    refreshed = db.participant(participant["id"])
    assert refreshed is not None
    checks = prior_readiness(db, refreshed)
    assert all(checks.values())


def make_settings(tmp_path: Path):
    base_encoder = tmp_path / "base" / "encoder.pth"
    base_main = tmp_path / "base" / "main.pth"
    write_checkpoint(base_encoder, 0.0, 0)
    write_checkpoint(base_main, 0.0, 0)
    return replace(
        default_settings,
        data_dir=tmp_path / "data",
        media_dir=tmp_path / "data" / "media",
        token_secret="pipeline-test-token",
        code_pepper="pipeline-test-code",
        ip_pepper="pipeline-test-ip",
        admin_token="pipeline-test-admin",
        base_encoder_checkpoint=str(base_encoder),
        base_main_checkpoint=str(base_main),
        matched_model_manifest=str(tmp_path / "data" / "matched-manifest.json"),
        matched_model_ready_sentinel=str(tmp_path / "data" / "matched-self-test.json"),
        model_code_dir=str(PROJECT_ROOT),
    )


def make_dtcn_settings(tmp_path: Path):
    base_encoder = tmp_path / "base-dtcn" / "encoder.pth"
    base_main = tmp_path / "base-dtcn" / "main_model.pth"
    base_encoder.parent.mkdir(parents=True)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model import initialize_models

    config = {
        "model_structure": "dilated_tcn",
        "input_type": "waveform",
        "original_sr": 16_000,
        "target_sr": 16_000,
        "max_audio_seconds": 2.5,
        "z_dim": 2,
        "tcn_channels": [1],
        "tcn_kernel_size": 3,
        "tcn_dilation_base": 2,
        "tcn_dropout": 0.0,
        "time_weighting": False,
        "rear_bias_attention": False,
        "main_hidden": 4,
        "if_main_transformer": False,
        "LEARNING_RATE": 1e-2,
        "num_of_step": 1,
        "if_historical_sample_rehearsal": False,
        "if_step_balance": False,
        "if_only_current": False,
        "ENABLE_SLIDING_WINDOW_SAMPLING": False,
        "use_instance_normalization": False,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260720)
        encoder, main = initialize_models(config)
    pair_id = uuid.uuid4().hex
    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "trainable_parameter_names": sorted(dict(encoder.named_parameters())),
            "config": config,
            "checkpoint_pair_id": pair_id,
        },
        base_encoder,
    )
    torch.save(
        {
            "model_state_dict": main.state_dict(),
            "trainable_parameter_names": sorted(dict(main.named_parameters())),
            "config": config,
            "checkpoint_pair_id": pair_id,
        },
        base_main,
    )
    return replace(
        default_settings,
        data_dir=tmp_path / "data",
        media_dir=tmp_path / "data" / "media",
        token_secret="pipeline-dtcn-test-token",
        code_pepper="pipeline-dtcn-test-code",
        ip_pepper="pipeline-dtcn-test-ip",
        admin_token="pipeline-dtcn-test-admin",
        agent_mode="dtcn",
        base_encoder_checkpoint=str(base_encoder),
        base_main_checkpoint=str(base_main),
        matched_model_manifest=str(tmp_path / "data" / "matched-manifest.json"),
        matched_model_ready_sentinel=str(tmp_path / "data" / "matched-self-test.json"),
        model_code_dir=str(PROJECT_ROOT),
    )


def write_perturbed_dtcn_pair(
    settings,
    target: Path,
    *,
    donor_ordinal: int,
    budget: int,
) -> tuple[Path, Path]:
    pair_id = uuid.uuid4().hex
    delta = donor_ordinal * 1e-3 + budget * 1e-4
    outputs = []
    for source, name in (
        (Path(settings.base_encoder_checkpoint), "encoder.pth"),
        (Path(settings.base_main_checkpoint), "main_model.pth"),
    ):
        payload = torch.load(source, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"]
        parameter_names = payload["trainable_parameter_names"]
        parameter_name = next(
            value
            for value in parameter_names
            if state[value].is_floating_point() and state[value].numel()
        )
        updated = state[parameter_name].detach().clone()
        updated.view(-1)[0] += delta
        state[parameter_name] = updated
        payload["checkpoint_pair_id"] = pair_id
        payload["fixture_update"] = {
            "analysis_eligible": False,
            "donor_ordinal": donor_ordinal,
            "opportunity_budget": budget,
        }
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)
        outputs.append(output.resolve())
    return outputs[0], outputs[1]


def test_pipeline_from_run_survives_a_newer_disk_protocol(tmp_path: Path) -> None:
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    source_protocol = json.loads(
        (default_settings.content_dir / "study_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    protocol_path = content_dir / "study_protocol.json"
    protocol_path.write_text(
        json.dumps(source_protocol, ensure_ascii=False), encoding="utf-8"
    )
    settings = replace(make_settings(tmp_path), content_dir=content_dir)
    db = StudyDB(settings)
    run_slug = "immutable-pipeline-protocol"
    db.create_participants(
        1,
        "study_a",
        "seed_donor",
        data_origin="formal",
        protocol_mode="strict",
        run_slug=run_slug,
    )

    newer = dict(source_protocol)
    newer["version"] = f"{source_protocol['version']}-newer"
    protocol_path.write_text(json.dumps(newer), encoding="utf-8")

    pipeline = FormalPipeline.from_run(db, settings, run_slug)

    assert pipeline.study["version"] == source_protocol["version"]
    assert pipeline.bound_run_id == db.study_run(run_slug)["id"]


def add_block(
    db: StudyDB,
    run_slug: str,
    count: int,
    cohort: str,
    group: str,
    prefix: str,
) -> list[dict]:
    rows = []
    if prefix == "A" and count == 36:
        blocks = [
            (12, "seed_donor", 1, 1),
            (8, "evaluator_donor", 2, 13),
            (8, "evaluator_donor", 3, 21),
            (8, "holdout_evaluator", 4, 29),
        ]
    elif prefix in {"BP", "BF"} and count == 18:
        blocks = [(16, group, 5, 1), (2, group, 6, 17)]
    else:
        blocks = [(count, group, 1, 1)]
    for block_count, block_group, wave, start in blocks:
        codes = db.create_participants(
            block_count,
            cohort,
            block_group,
            data_origin="formal",
            protocol_mode="strict",
            run_slug=run_slug,
        )
        for offset, code in enumerate(codes):
            index = start + offset
            participant = db.participant_by_code(code)
            assert participant is not None
            assignment = f"{prefix}{index:02d}"
            neutral = f"N-{prefix}-{index:02d}"
            db.configure_participant(
                participant["id"],
                assignment_id=assignment,
                neutral_id=neutral,
                enrollment_wave=wave,
                release_state="held",
            )
            refreshed = db.participant(participant["id"])
            assert refreshed is not None
            rows.append(refreshed)
    return rows


def complete_donor(db: StudyDB, settings, donor: dict, ordinal: int) -> None:
    for day in range(1, 4):
        study = db.start_session(donor["id"], day, "study")
        db.start_session(donor["id"], day, "companion")
        if day == 1:
            for assignment in db.prior_probe_assignments(
                participant_id=donor["id"], day=1
            ):
                payload = {
                    "phase": "candidate_probe",
                    "policy_token": assignment["probe_token"],
                }
                db.append_event(donor["id"], study["id"], "phase.start", None, None, payload)
                db.append_event(
                    donor["id"], study["id"], "phase.server_committed", None, None, payload
                )
                db.append_event(donor["id"], study["id"], "phase.end", None, None, payload)
        db.end_session(study["id"], donor["id"])
    for index in range(8):
        db.next_activation_opportunity(donor["id"])
        db.next_eligible_feedback(donor["id"], 0 if index in {3, 7} else 1)
    for budget in (2, 4, 8):
        negative_at_budget = sum(index in {3, 7} for index in range(budget))
        target = settings.data_dir / "fixture-snapshots" / donor["id"] / f"q-{budget}"
        encoder = target / "encoder.pth"
        main = target / "main.pth"
        write_checkpoint(encoder, ordinal + budget / 100.0, budget + ordinal)
        write_checkpoint(main, ordinal * 2 + budget / 100.0, budget + ordinal)
        db.record_model_snapshot(
            {
                "participant_id": donor["id"],
                "source": "donor_shadow",
                "opportunity_budget": budget,
                "budget_kind": "eligible_feedback",
                "eligible_feedbacks": budget,
                "positive_feedbacks": budget - negative_at_budget,
                "negative_feedbacks": negative_at_budget,
                "base_prior_id": donor.get("prior_id"),
                "encoder_path": str(encoder.resolve()),
                "main_path": str(main.resolve()),
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
            }
        )


def complete_holdout(db: StudyDB, holdout: dict) -> None:
    for day in range(1, 4):
        study = db.start_session(holdout["id"], day, "study")
        db.start_session(holdout["id"], day, "companion")
        if day == 1:
            for assignment in db.prior_probe_assignments(
                participant_id=holdout["id"], day=1
            ):
                payload = {
                    "phase": "candidate_probe",
                    "policy_token": assignment["probe_token"],
                }
                db.append_event(holdout["id"], study["id"], "phase.start", None, None, payload)
                db.append_event(
                    holdout["id"], study["id"], "phase.server_committed", None, None, payload
                )
                db.append_event(holdout["id"], study["id"], "phase.end", None, None, payload)
        db.end_session(study["id"], holdout["id"])


def complete_dtcn_donor(
    db: StudyDB,
    settings,
    donor: dict,
    ordinal: int,
) -> None:
    for day in range(1, 4):
        study = db.start_session(donor["id"], day, "study")
        db.start_session(donor["id"], day, "companion")
        if day == 1:
            for assignment in db.prior_probe_assignments(
                participant_id=donor["id"], day=1
            ):
                payload = {
                    "phase": "candidate_probe",
                    "policy_token": assignment["probe_token"],
                }
                db.append_event(
                    donor["id"], study["id"], "phase.start", None, None, payload
                )
                db.append_event(
                    donor["id"],
                    study["id"],
                    "phase.server_committed",
                    None,
                    None,
                    payload,
                )
                db.append_event(
                    donor["id"], study["id"], "phase.end", None, None, payload
                )
        db.end_session(study["id"], donor["id"])
    for index in range(8):
        db.next_activation_opportunity(donor["id"])
        db.next_eligible_feedback(donor["id"], 0 if index in {3, 7} else 1)
    for budget in (2, 4, 8):
        target = settings.data_dir / "dtcn-donor-snapshots" / donor["id"] / f"q-{budget}"
        encoder, main = write_perturbed_dtcn_pair(
            settings,
            target,
            donor_ordinal=ordinal,
            budget=budget,
        )
        negative_feedbacks = sum(index in {3, 7} for index in range(budget))
        db.record_model_snapshot(
            {
                "participant_id": donor["id"],
                "source": "donor_shadow",
                "opportunity_budget": budget,
                "budget_kind": "eligible_feedback",
                "eligible_feedbacks": budget,
                "positive_feedbacks": budget - negative_feedbacks,
                "negative_feedbacks": negative_feedbacks,
                "base_prior_id": donor.get("prior_id"),
                "encoder_path": str(encoder),
                "main_path": str(main),
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
                "metadata": {
                    "analysis_eligible": False,
                    "fixture_kind": "real-dtcn-pipeline-integration",
                },
            }
        )


def test_per_donor_cap_builds_all_q_priors_without_forcing_max_q(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    db = StudyDB(settings)
    study = json.loads(
        (settings.content_dir / "study_protocol.json").read_text(encoding="utf-8")
    )
    pipeline = FormalPipeline(db, settings, study)
    run_slug = "capped-feedback-prior"
    donor = add_block(db, run_slug, 1, "study_a", "donor", "A")[0]
    for day in range(1, 4):
        session = db.start_session(donor["id"], day, "study")
        db.start_session(donor["id"], day, "companion")
        db.end_session(session["id"], donor["id"])
    for label in (1, 1, 0):
        db.next_activation_opportunity(donor["id"])
        db.next_eligible_feedback(donor["id"], label)

    for cap, realized in ((2, 2), (4, 3), (8, 3)):
        target = settings.data_dir / "capped-fixtures" / f"q-{cap}"
        encoder = target / "encoder.pth"
        main = target / "main.pth"
        write_checkpoint(encoder, float(realized), realized)
        write_checkpoint(main, float(realized * 2), realized)
        db.record_model_snapshot(
            {
                "participant_id": donor["id"],
                "source": "donor_shadow",
                "opportunity_budget": cap,
                "budget_kind": "eligible_feedback",
                "eligible_feedbacks": realized,
                "positive_feedbacks": min(realized, 2),
                "negative_feedbacks": int(realized == 3),
                "metadata": {
                    "budget_semantics": "per_donor_cap",
                    "cap_budget": cap,
                    "realized_feedbacks": realized,
                    "cap_attained": realized >= cap,
                },
                "encoder_path": str(encoder.resolve()),
                "main_path": str(main.resolve()),
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
            }
        )

    priors = pipeline.build_nested_priors(run_slug, 1)
    assert [int(row["opportunity_budget"]) for row in priors] == [2, 4, 8]
    manifests = {
        int(row["opportunity_budget"]): json.loads(row["manifest"])
        for row in priors
    }
    assert manifests[2]["cap_attainment_rate"] == 1.0
    assert manifests[4]["cap_attainment_rate"] == 0.0
    assert manifests[8]["realized_feedback_mean"] == 3.0
    assert all(
        manifest["snapshots"][0]["participant_id"] == donor["id"]
        for manifest in manifests.values()
    )


def test_full_72_account_formal_pipeline_is_ordered_and_idempotent(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db = StudyDB(settings)
    study = json.loads((settings.content_dir / "study_protocol.json").read_text(encoding="utf-8"))
    pipeline = FormalPipeline(db, settings, study)
    run_slug = "formal-pipeline-e2e"
    donors = add_block(db, run_slug, 36, "study_a", "donor", "A")
    personalized = add_block(db, run_slug, 18, "study_b", "personalized", "BP")
    frozen = add_block(db, run_slug, 18, "study_b", "frozen", "BF")
    assert len(donors + personalized + frozen) == 72

    wave_one = pipeline.release_study_a_wave(run_slug, 1)
    assert len(wave_one["released"]) == 12
    with pytest.raises(PipelineError):
        pipeline.release_study_a_wave(run_slug, 2)

    for wave in range(1, 5):
        current = [
            row
            for row in db.participants_in_run(donors[0]["run_id"])
            if row["cohort"] == "study_a" and int(row["enrollment_wave"]) == wave
        ]
        for participant in current:
            if participant["group_name"] == "holdout_evaluator":
                complete_holdout(db, participant)
            else:
                complete_donor(
                    db,
                    settings,
                    participant,
                    int(participant["assignment_id"][1:]),
                )
        if wave < 4:
            released = pipeline.release_study_a_wave(run_slug, wave + 1)
            assert len(released["released"]) == 8

    donor_status = pipeline.status(run_slug)
    assert donor_status["waves"]["1"]["eligible_for_prior"] == 12
    assert donor_status["waves"]["1"]["prior_blockers"] == []

    prepared = pipeline.prepare_study_b(
        run_slug,
        28,
        8,
        "Blind locked decision: maximum checkpoint selected for fixture validation.",
    )
    assert prepared["selected_n"] == 28
    assert prepared["selected_q"] == 8
    assert prepared["matched_assignments"] == 36
    assert len(prepared["released"]) == 32
    assert prepared["idempotent"] is False
    manifest_path, _ = matched_paths(settings, donors[0]["run_id"])
    manifest_hash = file_sha256(manifest_path)

    repeated = pipeline.prepare_study_b(
        run_slug,
        28,
        8,
        "Blind locked decision: maximum checkpoint selected for fixture validation.",
    )
    assert repeated["idempotent"] is True
    assert file_sha256(manifest_path) == manifest_hash

    rows = db.participants_in_run(donors[0]["run_id"])
    study_b = [row for row in rows if row["cohort"] == "study_b"]
    assert sum(row["release_state"] == "released" for row in study_b) == 32
    assert sum(row["release_state"] == "held" for row in study_b) == 4
    selected = {row["prior_id"] for row in study_b}
    assert selected == {prepared["selected_prior_id"]}
    for participant in study_b:
        for day in pipeline.session_days:
            assignment, checks = matched_assignment(
                settings, participant["id"], day, participant["run_id"]
            )
            assert assignment is not None
            assert all(checks.values())
            assert assignment["donor_participant_id"] != participant["id"]

    reserves = pipeline.release_study_b_reserve(run_slug, "replace preregistered attrition")
    assert len(reserves["released"]) == 4
    status = pipeline.status(run_slug)
    assert status["waves"]["6"]["released"] == 4
    assert status["snapshot_counts"] == {"2": 28, "4": 28, "8": 28}


def test_study_b_is_blocked_until_holdout_finishes_but_can_select_smaller_n(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    db = StudyDB(settings)
    study = json.loads((settings.content_dir / "study_protocol.json").read_text(encoding="utf-8"))
    pipeline = FormalPipeline(db, settings, study)
    run_slug = "formal-pipeline-early-stop"
    donors = add_block(db, run_slug, 36, "study_a", "donor", "A")
    add_block(db, run_slug, 18, "study_b", "personalized", "BP")
    add_block(db, run_slug, 18, "study_b", "frozen", "BF")

    pipeline.release_study_a_wave(run_slug, 1)
    for wave in range(1, 4):
        current = [row for row in donors if int(row["enrollment_wave"]) == wave]
        for donor in current:
            complete_donor(db, settings, donor, int(donor["assignment_id"][1:]))
        pipeline.release_study_a_wave(run_slug, wave + 1)
    with pytest.raises(PipelineError, match="holdout evaluation"):
        pipeline.prepare_study_b(
            run_slug,
            12,
            4,
            "Blind saturation record selects the smallest supported donor checkpoint.",
        )
    for holdout in donors[28:]:
        complete_holdout(db, holdout)

    prepared = pipeline.prepare_study_b(
        run_slug,
        12,
        4,
        "Blind saturation record selects the smallest supported donor checkpoint.",
    )
    assert prepared["selected_n"] == 12
    assert prepared["selected_q"] == 4
    assert len(prepared["released"]) == 32
    assert all(db.sessions(participant["id"]) for participant in donors)


def test_real_dtcn_pipeline_personalized_frozen_divergence_audit(
    tmp_path: Path,
) -> None:
    settings = make_dtcn_settings(tmp_path)
    db = StudyDB(settings)
    study = json.loads(
        (settings.content_dir / "study_protocol.json").read_text(encoding="utf-8")
    )
    pipeline = FormalPipeline(db, settings, study)
    run_slug = "real-dtcn-divergence-e2e"
    donors = add_block(db, run_slug, 36, "study_a", "donor", "A")
    add_block(db, run_slug, 18, "study_b", "personalized", "BP")
    add_block(db, run_slug, 18, "study_b", "frozen", "BF")

    pipeline.release_study_a_wave(run_slug, 1)
    for wave in range(1, 5):
        current = [
            participant
            for participant in donors
            if int(participant["enrollment_wave"]) == wave
        ]
        for participant in current:
            if participant["group_name"] == "holdout_evaluator":
                complete_holdout(db, participant)
            else:
                complete_dtcn_donor(
                    db,
                    settings,
                    participant,
                    int(participant["assignment_id"][1:]),
                )
        if wave < 4:
            pipeline.release_study_a_wave(run_slug, wave + 1)

    prepared = pipeline.prepare_study_b(
        run_slug,
        12,
        8,
        "Blind integration fixture selects the locked n=12, q=8 candidate.",
    )
    assert prepared["matched_assignments"] == 36
    assert prepared["selected_n"] == 12
    assert prepared["selected_q"] == 8

    run = db.study_run(run_slug)
    assert run is not None
    rows = db.participants_in_run(run["id"])
    personalized = next(row for row in rows if row["assignment_id"] == "BP01")
    frozen = next(row for row in rows if row["assignment_id"] == "BF01")
    target_ids = {personalized["id"], frozen["id"]}
    for participant in rows:
        if (
            participant["cohort"] == "study_b"
            and participant["release_state"] == "released"
            and participant["id"] not in target_ids
        ):
            db.set_release_state(
                participant["id"],
                "held",
                actor="integration-test",
                reason="Scope the expensive real-DTCN audit to one participant per arm.",
            )
    personalized = db.participant(personalized["id"])
    frozen = db.participant(frozen["id"])
    assert personalized is not None and frozen is not None
    assert personalized["release_state"] == "released"
    assert frozen["release_state"] == "released"

    selected_prior = db.shared_prior(prepared["selected_prior_id"])
    assert selected_prior is not None
    selected_hashes = (
        file_sha256(Path(selected_prior["encoder_path"])),
        file_sha256(Path(selected_prior["main_path"])),
    )

    service = TimingService(settings, db)
    personalized_session = db.start_session(personalized["id"], 1, "study")
    db.start_session(personalized["id"], 1, "companion")
    personalized_runtime = service.runtime(
        personalized,
        day=1,
        session_id=personalized_session["id"],
    )
    assert personalized_runtime.set_phase("learning", "current")["committed"] is True
    audio = np.sin(
        2 * np.pi * 220 * np.arange(8_000, dtype=np.float32) / 16_000
    ).astype(np.float32)
    personalized_runtime.ring.append(audio)
    personalized_results = []
    for index in range(8):
        client_ms = float(1_000 + index * 1_000)
        result = personalized_runtime.tap(
            client_ms,
            pointer_agent_state="waiting",
            page_visible=True,
            intended_action="manual_activate",
            expected_state="waiting",
            expected_activation_id="",
            expected_state_version=personalized_runtime.state_version,
        )
        personalized_results.append(result)
        db.append_event(
            personalized["id"],
            personalized_session["id"],
            "tap.manual_activate",
            client_ms,
            client_ms,
            result,
        )
        personalized_runtime.set_state("waiting")

    assert all(result["training_eligible"] for result in personalized_results)
    assert all(result["training_applied"] for result in personalized_results)
    assert all(
        result["personal_calibration_applied"] for result in personalized_results
    )
    assert [
        result["current_network_update_applied"]
        for result in personalized_results[:3]
    ] == [False, False, True]
    assert personalized_results[2]["optimizer_step_after"] > personalized_results[2][
        "optimizer_step_before"
    ]

    latest_encoder = settings.data_dir / "models" / personalized["id"] / "latest_encoder.pth"
    latest_main = settings.data_dir / "models" / personalized["id"] / "latest_main_model.pth"
    assert latest_encoder.is_file() and latest_main.is_file()
    assert (
        file_sha256(latest_encoder),
        file_sha256(latest_main),
    ) != selected_hashes
    snapshots = [
        row
        for row in db.model_snapshots(run["id"], source="personalized")
        if row["participant_id"] == personalized["id"]
    ]
    assert {int(row["opportunity_budget"]) for row in snapshots} >= {2, 4, 8}
    assert db.feedback_counts(personalized["id"])["eligible_feedbacks"] == 8

    reloaded_runtime = TimingService(settings, db).runtime(personalized, day=1)
    assert reloaded_runtime.model is not None
    probability, logit = reloaded_runtime.model.predict(audio)
    assert math.isfinite(probability) and math.isfinite(logit)

    frozen_session = db.start_session(frozen["id"], 1, "study")
    db.start_session(frozen["id"], 1, "companion")
    frozen_runtime = service.runtime(frozen, day=1, session_id=frozen_session["id"])
    assert frozen_runtime.model_identity == {
        "encoder_sha256": selected_prior["encoder_sha256"],
        "main_sha256": selected_prior["main_sha256"],
    }
    assert frozen_runtime.set_phase("learning", "current")["committed"] is True
    db.append_event(
        frozen["id"],
        frozen_session["id"],
        "model.ready",
        None,
        None,
        frozen_runtime.ready_payload(),
    )
    frozen_runtime.ring.append(audio)
    frozen_result = frozen_runtime.tap(
        1_000.0,
        pointer_agent_state="waiting",
        page_visible=True,
        intended_action="manual_activate",
        expected_state="waiting",
        expected_activation_id="",
        expected_state_version=frozen_runtime.state_version,
    )
    db.append_event(
        frozen["id"],
        frozen_session["id"],
        "tap.manual_activate",
        1_000.0,
        1_000.0,
        frozen_result,
    )
    frozen_runtime.set_state("waiting")
    assert frozen_result["training_eligible"] is False
    assert frozen_result["training_applied"] is False
    assert frozen_result["quarantine_reason"] == "activation_not_learning_current"
    assert db.feedback_counts(frozen["id"])["eligible_feedbacks"] == 0
    assert not (settings.data_dir / "models" / frozen["id"]).exists()
    assert not [
        row
        for row in db.model_snapshots(run["id"], source="personalized")
        if row["participant_id"] == frozen["id"]
    ]
    assert (
        file_sha256(Path(selected_prior["encoder_path"])),
        file_sha256(Path(selected_prior["main_path"])),
    ) == selected_hashes

    db.end_session(personalized_session["id"], personalized["id"])
    db.end_session(frozen_session["id"], frozen["id"])
    database = settings.data_dir / "study.sqlite3"
    database_hash_before = file_sha256(database)
    report = audit_model_divergence(settings.data_dir, run_slug)

    assert file_sha256(database) == database_hash_before
    assert report["passed"] is True
    assert report["assertions"] == {
        "all_recorded_hashes_match": True,
        "shared_differs_from_m0": True,
        "all_personalized_checkpoints_differ_from_prior": True,
        "all_frozen_have_no_personal_updates": True,
        "all_frozen_effective_checkpoints_equal_prior": True,
        "all_matched_assignments_present_and_hash_bound": True,
    }
    assert len(report["participants"]["personalized"]) == 1
    assert len(report["participants"]["frozen"]) == 1
