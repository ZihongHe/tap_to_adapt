"""Materialize an isolated simulation-only Study B E2E fixture.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _trainable_float_tensor_name(payload: dict[str, Any], torch: Any) -> str:
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Synthetic fixture source has no model_state_dict")
    declared = payload.get("trainable_parameter_names")
    if isinstance(declared, list):
        candidates = [str(name) for name in declared if str(name) in state]
    else:
        candidates = []
    candidates.extend(
        name
        for name in sorted(state)
        if name not in candidates
        and torch.is_tensor(state[name])
        and state[name].is_floating_point()
        and state[name].numel() > 0
        and not name.endswith(("running_mean", "running_var"))
    )
    for name in candidates:
        value = state.get(name)
        if (
            torch.is_tensor(value)
            and value.is_floating_point()
            and value.numel() > 0
        ):
            return name
    raise RuntimeError("Synthetic fixture source has no trainable floating tensor")


def _write_synthetic_checkpoint(
    source: Path,
    target: Path,
    *,
    pair_id: str,
    delta: float,
    donor_rank: int,
    budget: int,
) -> dict[str, Any]:
    """Write a tiny deterministic simulation-only update without touching M0.

    This is an engineering fixture, not learned evidence.  A real copied tensor
    is changed so the A-to-B aggregation and matched-delta paths cannot pass by
    merely routing byte-identical M0 files.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build the simulation E2E fixture") from exc
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite synthetic checkpoint: {target}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid synthetic fixture source checkpoint: {source}")
    payload = copy.deepcopy(payload)
    tensor_name = _trainable_float_tensor_name(payload, torch)
    tensor = payload["model_state_dict"][tensor_name].detach().cpu().clone()
    flattened = tensor.reshape(-1)
    before = float(flattened[0].item())
    flattened[0] = flattened[0] + float(delta)
    after = float(flattened[0].item())
    if after == before:
        raise RuntimeError(
            f"Synthetic perturbation was below checkpoint precision for {tensor_name}"
        )
    payload["model_state_dict"][tensor_name] = tensor
    payload["checkpoint_pair_id"] = pair_id
    payload["synthetic_fixture"] = {
        "schema": "tap-to-adapt-simulation-checkpoint-perturbation/v1",
        "analysis_eligible": False,
        "donor_rank": donor_rank,
        "opportunity_budget": budget,
        "tensor_name": tensor_name,
        "requested_delta": float(delta),
        "realized_delta": after - before,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{pair_id}.tmp")
    try:
        torch.save(payload, temporary)
        verified = torch.load(temporary, map_location="cpu", weights_only=False)
        if (
            verified.get("checkpoint_pair_id") != pair_id
            or not isinstance(verified.get("model_state_dict"), dict)
        ):
            raise RuntimeError("Synthetic checkpoint verification failed")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return payload["synthetic_fixture"]


def write_synthetic_checkpoint_pair(
    encoder_source: Path,
    main_source: Path,
    encoder_target: Path,
    main_target: Path,
    *,
    donor_rank: int,
    budget: int,
) -> dict[str, Any]:
    if encoder_target.exists() or main_target.exists():
        raise RuntimeError("Refusing a partial or pre-existing synthetic checkpoint pair")
    pair_id = uuid.uuid4().hex
    # Use a clearly representable but numerically tiny offset.  Donor rank and
    # budget keep every synthetic snapshot distinct while preserving validity.
    delta = ((donor_rank + 1) * 100 + int(budget)) * 1e-5
    encoder_metadata = _write_synthetic_checkpoint(
        encoder_source,
        encoder_target,
        pair_id=pair_id,
        delta=delta,
        donor_rank=donor_rank,
        budget=budget,
    )
    try:
        main_metadata = _write_synthetic_checkpoint(
            main_source,
            main_target,
            pair_id=pair_id,
            delta=delta,
            donor_rank=donor_rank,
            budget=budget,
        )
    except BaseException:
        encoder_target.unlink(missing_ok=True)
        raise
    return {
        "checkpoint_pair_id": pair_id,
        "encoder": encoder_metadata,
        "main": main_metadata,
    }


def complete_fixture_donor(
    db: Any,
    settings: Any,
    donor: dict[str, Any],
    q_levels: list[int],
    budget_kind: str,
    donor_rank: int,
) -> None:
    day_three_session_id: str | None = None
    for day in range(1, 4):
        study = db.start_session(donor["id"], day, "study")
        db.start_session(donor["id"], day, "companion")
        if study["state"] not in {"completed", "ended"}:
            db.end_session(study["id"], donor["id"])
        if day == 3:
            day_three_session_id = str(study["id"])

    if budget_kind == "eligible_feedback":
        while db.feedback_counts(donor["id"])["eligible_feedbacks"] < max(q_levels):
            counts = db.feedback_counts(donor["id"])
            db.next_activation_opportunity(donor["id"])
            db.next_eligible_feedback(
                donor["id"], 0 if counts["eligible_feedbacks"] % 4 == 3 else 1
            )
    else:
        while db.learning_opportunity_count(donor["id"]) < max(q_levels):
            db.next_learning_opportunity(donor["id"])

    existing = {
        int(row["opportunity_budget"])
        for row in db.model_snapshots(
            donor["run_id"], source="donor_shadow", budget_kind=budget_kind
        )
        if row["participant_id"] == donor["id"]
    }
    from app.integrity import file_sha256

    for budget in q_levels:
        if budget in existing:
            continue
        target = settings.data_dir / "simulation-fixtures" / donor["run_id"] / donor["id"] / f"q-{budget}"
        encoder = target / "encoder.pth"
        main = target / "main_model.pth"
        perturbation = write_synthetic_checkpoint_pair(
            Path(settings.base_encoder_checkpoint),
            Path(settings.base_main_checkpoint),
            encoder,
            main,
            donor_rank=donor_rank,
            budget=budget,
        )
        db.record_model_snapshot(
            {
                "participant_id": donor["id"],
                "session_id": day_three_session_id,
                "source": "donor_shadow",
                "opportunity_budget": budget,
                "budget_kind": budget_kind,
                "eligible_feedbacks": budget if budget_kind == "eligible_feedback" else 0,
                "positive_feedbacks": budget - (budget // 4),
                "negative_feedbacks": budget // 4,
                "metadata": {
                    "analysis_eligible": False,
                    "fixture": True,
                    "budget_kind": budget_kind,
                    "checkpoint_source": "deterministic_nonanalytic_m0_perturbation",
                    "synthetic_perturbation": perturbation,
                },
                "base_prior_id": donor.get("prior_id"),
                "encoder_path": str(encoder.resolve()),
                "main_path": str(main.resolve()),
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize an isolated simulation-only Study B E2E fixture."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-n", type=int, default=20)
    parser.add_argument("--selected-q", type=int, default=8)
    parser.add_argument("--acknowledge-synthetic-donors", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_synthetic_donors:
        raise SystemExit(
            "Pass --acknowledge-synthetic-donors; this fixture is never analysis eligible"
        )

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.matched import matched_assignment, matched_paths
    from app.pipeline import FormalPipeline
    from app.protocol import Protocol
    from app.priors import prior_readiness

    db = StudyDB(settings)
    protocol = Protocol(settings)
    run = db.study_run(args.run_slug)
    if run is None:
        raise SystemExit(f"Unknown run slug: {args.run_slug}")
    if run["data_origin"] != "simulation" or run["protocol_mode"] != "compressed":
        raise SystemExit("Refusing to materialize synthetic donors outside a simulation run")

    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = list(csv.DictReader(handle))
    by_assignment = {row["assignment_id"]: row for row in roster}
    participants = db.participants_in_run(run["id"])
    donors = sorted(
        [row for row in participants if row["cohort"] == "study_a"],
        key=lambda row: str(row["assignment_id"]),
    )[: args.selected_n]
    if len(donors) != args.selected_n:
        raise SystemExit("The simulation roster does not contain enough Study A donors")

    pipeline_rules = protocol.study["pipeline"]
    budget_kind = str(pipeline_rules.get("budget_kind") or "activation_opportunity")
    q_key = "q_feedback_levels" if budget_kind == "eligible_feedback" else "q_opportunity_levels"
    q_levels = [int(value) for value in pipeline_rules[q_key]]
    for donor_rank, donor in enumerate(donors):
        complete_fixture_donor(
            db, settings, donor, q_levels, budget_kind, donor_rank
        )
    db.record_pipeline_action(
        run["id"],
        "simulation.fixture_donors",
        actor="e2e-fixture",
        payload={
            "analysis_eligible": False,
            "synthetic_donor_count": len(donors),
            "opportunity_levels": q_levels,
            "budget_kind": budget_kind,
            "checkpoint_source": "deterministic_nonanalytic_m0_perturbation",
        },
    )

    pipeline = FormalPipeline(db, settings, protocol.study, allow_simulation=True)
    prepared = pipeline.prepare_study_b(
        args.run_slug,
        args.selected_n,
        args.selected_q,
        "Simulation-only E2E lock used to verify shared-prior and matched routing.",
    )
    refreshed = db.participants_in_run(run["id"])
    test_accounts: list[dict[str, Any]] = []
    for assignment_id in ("BP01", "BF01"):
        participant = next(row for row in refreshed if row["assignment_id"] == assignment_id)
        roster_row = by_assignment[assignment_id]
        matched_days = {}
        for day in range(1, 4):
            assignment, checks = matched_assignment(
                settings, participant["id"], day, run["id"]
            )
            if assignment is None or not all(checks.values()):
                raise SystemExit(f"Matched assignment is not ready for {assignment_id} day {day}")
            matched_days[str(day)] = assignment["donor_participant_id"]
        prior_checks = prior_readiness(db, participant)
        if not all(prior_checks.values()):
            raise SystemExit(f"Prior is not ready for {assignment_id}: {prior_checks}")
        test_accounts.append(
            {
                "assignment_id": assignment_id,
                "participant_code": roster_row["participant_code"],
                "neutral_id": participant["neutral_id"],
                "group": participant["group_name"],
                "prior_id": participant["prior_id"],
                "prior_n": participant["prior_n"],
                "prior_q": participant["prior_q"],
                "matched_donor_by_day": matched_days,
            }
        )

    manifest_path, sentinel_path = matched_paths(settings, run["id"])
    result = {
        "simulation_only": True,
        "analysis_eligible": False,
        "run_id": run["id"],
        "run_slug": run["slug"],
        "prepared": prepared,
        "matched_manifest": str(manifest_path),
        "matched_self_test": str(sentinel_path),
        "accounts": test_accounts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
