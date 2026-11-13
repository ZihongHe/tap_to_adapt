"""Compatible checkpoint aggregation and shared-initialization artifacts.

Shared priors are built from validated checkpoints with matching model shapes.
The helpers preserve the distinction between an immutable initialization and
a participant's subsequent online updates; file paths are supplied explicitly.
"""

from __future__ import annotations

import copy
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

from .integrity import file_sha256

if TYPE_CHECKING:
    from .config import Settings
    from .db import StudyDB


class PriorError(RuntimeError):
    pass


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise PriorError("PyTorch is required for shared-prior operations") from exc
    return torch


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PriorError(f"Checkpoint is missing: {path}")
    checkpoint = _torch().load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise PriorError(f"Invalid checkpoint: {path}")
    return checkpoint


def _assert_compatible(checkpoints: list[dict[str, Any]]) -> list[str]:
    if not checkpoints:
        raise PriorError("At least one checkpoint is required")
    keys = list(checkpoints[0]["model_state_dict"])
    for checkpoint in checkpoints[1:]:
        state = checkpoint["model_state_dict"]
        if list(state) != keys:
            raise PriorError("Checkpoint state dictionaries do not have identical keys")
        for key in keys:
            if state[key].shape != checkpoints[0]["model_state_dict"][key].shape:
                raise PriorError(f"Checkpoint tensor shape mismatch: {key}")
    return keys


def average_checkpoint_files(
    inputs: Iterable[Path],
    output: Path,
    algorithm: str = "class-balanced-equal-donor-shadow-fedavg-from-locked-m0-v3",
) -> None:
    paths = list(inputs)
    checkpoints = [_load_checkpoint(path) for path in paths]
    keys = _assert_compatible(checkpoints)
    torch = _torch()
    averaged: dict[str, Any] = {}
    for key in keys:
        values = [checkpoint["model_state_dict"][key] for checkpoint in checkpoints]
        if torch.is_floating_point(values[0]):
            value = torch.stack([item.detach().cpu().to(torch.float64) for item in values]).mean(0)
            averaged[key] = value.to(values[0].dtype)
        else:
            # Integer state entries are running counters (for example BatchNorm's
            # num_batches_tracked), not learnable weights. Preserve a deterministic,
            # valid counter without treating it as a parameter estimate.
            stacked = torch.stack([item.detach().cpu() for item in values])
            averaged[key] = torch.amax(stacked, dim=0).to(values[0].dtype)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(checkpoints[0])
    payload["model_state_dict"] = averaged
    payload["aggregation"] = {
        "algorithm": algorithm,
        "non_floating_buffer_rule": "elementwise_max",
        "input_sha256": [file_sha256(path) for path in paths],
        "created_ms": int(time.time() * 1000),
    }
    torch.save(payload, output)


def delta_transport_checkpoint(
    base: Path,
    shared: Path,
    donor: Path,
    output: Path,
    algorithm: str = "shared-prior-plus-donor-shadow-delta-v2",
) -> None:
    checkpoints = [_load_checkpoint(path) for path in (base, shared, donor)]
    keys = _assert_compatible(checkpoints)
    torch = _torch()
    base_state, shared_state, donor_state = [item["model_state_dict"] for item in checkpoints]
    transported: dict[str, Any] = {}
    for key in keys:
        base_value = base_state[key]
        shared_value = shared_state[key]
        donor_value = donor_state[key]
        if torch.is_floating_point(base_value):
            value = shared_value.detach().cpu().to(torch.float64)
            value += donor_value.detach().cpu().to(torch.float64)
            value -= base_value.detach().cpu().to(torch.float64)
            transported[key] = value.to(shared_value.dtype)
        else:
            # Non-floating state is not a learnable delta. Keep the shared-prior
            # buffer so matched models differ only by transported floating updates.
            transported[key] = shared_value.detach().cpu().clone()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(checkpoints[1])
    payload["model_state_dict"] = transported
    payload["aggregation"] = {
        "algorithm": algorithm,
        "non_floating_buffer_rule": "inherit_shared_prior",
        "base_sha256": file_sha256(base),
        "shared_sha256": file_sha256(shared),
        "donor_sha256": file_sha256(donor),
        "created_ms": int(time.time() * 1000),
    }
    torch.save(payload, output)


def create_base_prior(db: "StudyDB", settings: "Settings", run_id: str) -> dict[str, Any]:
    existing = [
        row
        for row in db.shared_priors(run_id)
        if int(row["donor_count"]) == 0 and int(row["opportunity_budget"]) == 0
    ]
    if existing:
        return existing[0]
    prior_id = str(uuid.uuid4())
    target = settings.data_dir / "priors" / run_id / prior_id
    encoder = target / "encoder.pth"
    main = target / "main_model.pth"
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(settings.base_encoder_checkpoint, encoder)
    shutil.copy2(settings.base_main_checkpoint, main)
    return db.create_shared_prior(
        {
            "id": prior_id,
            "run_id": run_id,
            "donor_count": 0,
            "opportunity_budget": 0,
            "status": "candidate",
            "algorithm": "locked-m0-v1",
            "donor_ids": [],
            "encoder_path": str(encoder.resolve()),
            "main_path": str(main.resolve()),
            "encoder_sha256": file_sha256(encoder),
            "main_sha256": file_sha256(main),
            "manifest": {
                "source_encoder_sha256": file_sha256(Path(settings.base_encoder_checkpoint)),
                "source_main_sha256": file_sha256(Path(settings.base_main_checkpoint)),
            },
        }
    )


def build_shared_prior(
    db: "StudyDB",
    settings: "Settings",
    run_id: str,
    donor_ids: list[str],
    opportunity_budget: int,
    budget_kind: str = "activation_opportunity",
    budget_semantics: str = "exact_count",
    algorithm: str = "class-balanced-equal-donor-shadow-fedavg-from-locked-m0-v3",
) -> dict[str, Any]:
    if not donor_ids:
        raise PriorError("A shared prior requires at least one donor")
    snapshots = db.model_snapshots(
        run_id,
        source="donor_shadow",
        budget=opportunity_budget,
        budget_kind=budget_kind,
    )
    by_participant = {row["participant_id"]: row for row in snapshots}
    missing = [participant_id for participant_id in donor_ids if participant_id not in by_participant]
    if missing:
        raise PriorError(
            f"Missing q={opportunity_budget} donor-shadow snapshots for {len(missing)} donors"
        )
    existing = [
        row
        for row in db.shared_priors(run_id)
        if int(row["donor_count"]) == len(donor_ids)
        and int(row["opportunity_budget"]) == opportunity_budget
    ]
    if existing:
        return existing[0]
    prior_id = str(uuid.uuid4())
    target = settings.data_dir / "priors" / run_id / prior_id
    encoder = target / "encoder.pth"
    main = target / "main_model.pth"
    encoder_inputs = [Path(by_participant[value]["encoder_path"]) for value in donor_ids]
    main_inputs = [Path(by_participant[value]["main_path"]) for value in donor_ids]
    average_checkpoint_files(encoder_inputs, encoder, algorithm)
    average_checkpoint_files(main_inputs, main, algorithm)
    snapshot_manifest = [
        {
            "participant_id": participant_id,
            "snapshot_id": by_participant[participant_id]["id"],
            "cap_budget": opportunity_budget,
            "realized_feedbacks": int(
                by_participant[participant_id]["eligible_feedbacks"]
            ),
            "cap_attained": int(
                by_participant[participant_id]["eligible_feedbacks"]
            )
            >= opportunity_budget,
            "encoder_sha256": by_participant[participant_id]["encoder_sha256"],
            "main_sha256": by_participant[participant_id]["main_sha256"],
        }
        for participant_id in donor_ids
    ]
    return db.create_shared_prior(
        {
            "id": prior_id,
            "run_id": run_id,
            "donor_count": len(donor_ids),
            "opportunity_budget": opportunity_budget,
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
                "feedback_budget": opportunity_budget,
                "same_donor_set_across_q": True,
                "cap_attainment_rate": sum(
                    int(row["cap_attained"]) for row in snapshot_manifest
                )
                / len(snapshot_manifest),
                "realized_feedback_mean": sum(
                    int(row["realized_feedbacks"]) for row in snapshot_manifest
                )
                / len(snapshot_manifest),
                "snapshots": snapshot_manifest,
            },
        }
    )


def prior_readiness(db: "StudyDB", participant: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "participant_released": participant.get("release_state") in {"released", "completed"},
        "prior_assigned": bool(participant.get("prior_id")),
        "prior_record_present": False,
        "prior_same_run": False,
        "prior_coordinates_match": False,
        "prior_files_present": False,
        "prior_hashes_current": False,
        "study_b_prior_selected": participant.get("cohort") != "study_b",
    }
    if not participant.get("prior_id"):
        return checks
    prior = db.shared_prior(str(participant["prior_id"]))
    if prior is None:
        return checks
    checks["prior_record_present"] = True
    checks["prior_same_run"] = prior["run_id"] == participant["run_id"]
    participant_n = participant.get("prior_n")
    participant_q = participant.get("prior_q")
    checks["prior_coordinates_match"] = (
        participant_n is not None
        and participant_q is not None
        and int(prior["donor_count"]) == int(participant_n)
        and int(prior["opportunity_budget"]) == int(participant_q)
    )
    encoder = Path(prior["encoder_path"])
    main = Path(prior["main_path"])
    checks["prior_files_present"] = encoder.is_file() and main.is_file()
    if checks["prior_files_present"]:
        checks["prior_hashes_current"] = (
            file_sha256(encoder) == prior["encoder_sha256"]
            and file_sha256(main) == prior["main_sha256"]
        )
    if participant.get("cohort") == "study_b":
        checks["study_b_prior_selected"] = prior["status"] == "selected"
    return checks


def prior_paths(db: "StudyDB", participant: dict[str, Any]) -> tuple[Path, Path]:
    checks = prior_readiness(db, participant)
    required = {key: value for key, value in checks.items() if key != "participant_released"}
    if not all(required.values()):
        missing = [name for name, passed in required.items() if not passed]
        raise PriorError(f"Participant prior is not ready: {', '.join(missing)}")
    prior = db.shared_prior(str(participant["prior_id"]))
    assert prior is not None
    return Path(prior["encoder_path"]), Path(prior["main_path"])
