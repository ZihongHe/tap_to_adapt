"""Command-line utility: runtime self test.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")

    from app.config import settings
    from app.db import StudyDB
    from app.timing import DTCNModel, TimingService

    if settings.agent_mode != "dtcn":
        raise RuntimeError("AGENT_MODE must be dtcn")

    db = StudyDB(settings)
    code = db.create_participants(1, "pilot", "qa_runtime")[0]
    participant = db.participant_by_code(code)
    if participant is None:
        raise RuntimeError("QA participant could not be loaded")

    participant_id = participant["id"]
    service = TimingService(settings, db)
    try:
        runtime = service.runtime(participant, day=1)
        if runtime.model is None:
            raise RuntimeError("DTCN model was not loaded by TimingService")

        runtime.set_phase("learning", "current")
        rng = np.random.default_rng(20260713)
        sample = rng.normal(0.0, 0.015, 16_000 * 30).astype(np.float32)
        runtime.ring.append(sample)
        probability_before, logit_before = runtime.model.predict(sample)
        results = []
        for index, timestamp in enumerate((1000.0, 2000.0, 3000.0)):
            result = runtime.tap(timestamp)
            results.append(result)
            if result.get("action") != "manual_activate" or not result.get(
                "training_eligible"
            ):
                raise RuntimeError(
                    f"Runtime tap {index + 1} was not eligible for learning: {result}"
                )
            if result.get("personal_calibration_applied") is not True:
                raise RuntimeError(
                    f"Runtime tap {index + 1} did not apply personal calibration"
                )
            if index < 2:
                if float(result.get("network_update_scale", -1.0)) != 0.0:
                    raise RuntimeError("The first two feedbacks must be intercept-only")
                if result.get("current_network_update_applied") is not False:
                    raise RuntimeError("Intercept-only feedback unexpectedly updated DTCN")
                if int(result.get("requested_training_steps", -1)) != 0:
                    raise RuntimeError("Intercept-only feedback requested optimizer steps")
            if index < 2:
                runtime.set_state("waiting")

        first, second, result = results

        snapshot_dir = settings.data_dir / "models" / participant_id
        snapshot_encoder = snapshot_dir / "latest_encoder.pth"
        snapshot_main = snapshot_dir / "latest_main_model.pth"
        if not (
            float(first["personal_logit_bias_after"])
            < float(second["personal_logit_bias_after"])
            < float(result["personal_logit_bias_after"])
        ):
            raise RuntimeError("Positive feedback did not monotonically raise the intercept")
        if float(result.get("network_update_scale", 0.0)) != 0.05:
            raise RuntimeError("The third-feedback network update schedule is not locked")
        if int(result.get("requested_training_steps", 0)) != 1:
            raise RuntimeError("The third feedback did not use one DTCN optimizer step")
        lr_before = [float(value) for value in result["optimizer_learning_rates_before"]]
        lr_during = [float(value) for value in result["optimizer_learning_rates_during"]]
        lr_after = [float(value) for value in result["optimizer_learning_rates_after"]]
        if not lr_before or len(lr_before) != len(lr_during):
            raise RuntimeError("The DTCN learning-rate trust region was not recorded")
        if any(
            not math.isclose(during, before * 0.05, rel_tol=1e-9, abs_tol=1e-12)
            for before, during in zip(lr_before, lr_during, strict=True)
        ):
            raise RuntimeError("The third feedback did not scale the optimizer learning rate")
        if any(
            not math.isclose(after, before, rel_tol=1e-12, abs_tol=1e-15)
            for before, after in zip(lr_before, lr_after, strict=True)
        ):
            raise RuntimeError("The optimizer learning rate was not restored")
        if result.get("parameter_state_interpolated") is not True or not math.isclose(
            float(result.get("parameter_interpolation_scale", 0.0)),
            0.05,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError("The DTCN parameter-state trust region was not applied")
        if not math.isfinite(float(result.get("loss", math.nan))):
            raise RuntimeError("Runtime update returned a non-finite loss")
        if not snapshot_encoder.is_file() or not snapshot_main.is_file():
            raise RuntimeError("Participant snapshot was not persisted")
        snapshot_dir.chmod(0o700)
        snapshot_encoder.chmod(0o600)
        snapshot_main.chmod(0o600)

        base_encoder = Path(settings.base_encoder_checkpoint)
        base_main = Path(settings.base_main_checkpoint)
        changed = (
            file_sha256(snapshot_encoder) != file_sha256(base_encoder)
            or file_sha256(snapshot_main) != file_sha256(base_main)
        )
        if not changed:
            raise RuntimeError("Persisted participant snapshot did not change")

        reloaded = DTCNModel(settings, str(snapshot_encoder), str(snapshot_main))
        probability_after, logit_after = reloaded.predict(sample)
        if not all(
            map(
                math.isfinite,
                (probability_before, logit_before, probability_after, logit_after),
            )
        ):
            raise RuntimeError("Runtime prediction or reloaded prediction was non-finite")

        persisted_bias = float(result["personal_logit_bias_after"])
        calibrated_before = float(result["feedback_probability_before"])
        calibrated_after = 1.0 / (
            1.0
            + math.exp(
                -(
                    logit_after / runtime.temperature
                    + runtime.logit_bias
                    + persisted_bias
                )
            )
        )
        calibrated_delta = calibrated_after - calibrated_before
        if not 0 < calibrated_delta <= 0.1:
            raise RuntimeError(
                "The first network update exceeded the locked calibrated trust region: "
                f"delta={calibrated_delta:.6f}"
            )
        service.release(participant_id)
        restarted_service = TimingService(settings, db)
        restarted_runtime = restarted_service.runtime(participant, day=2)
        if restarted_runtime.personal_logit_bias != persisted_bias:
            raise RuntimeError("Personal calibration did not survive runtime recreation")
        restarted_service.release(participant_id)

        sentinel = {
            "passed": True,
            "created_ms": int(time.time() * 1000),
            "agent_mode": settings.agent_mode,
            "participant_id": participant_id,
            "participant_cohort": participant["cohort"],
            "participant_group": participant["group_name"],
            "phase": runtime.phase,
            "policy": runtime.policy,
            "action": result["action"],
            "training_eligible": result["training_eligible"],
            "loss": result["loss"],
            "first_two_intercept_only": True,
            "first_two_network_scales": [
                first["network_update_scale"],
                second["network_update_scale"],
            ],
            "network_update_scale": result["network_update_scale"],
            "requested_training_steps": result["requested_training_steps"],
            "training_steps": result["training_steps"],
            "optimizer_learning_rates_before": lr_before,
            "optimizer_learning_rates_during": lr_during,
            "optimizer_learning_rates_after": lr_after,
            "parameter_state_interpolated": result["parameter_state_interpolated"],
            "parameter_interpolation_scale": result["parameter_interpolation_scale"],
            "personal_logit_bias_before": first["personal_logit_bias_before"],
            "personal_logit_bias_after": persisted_bias,
            "personal_calibration_persisted": True,
            "probability_before": probability_before,
            "logit_before": logit_before,
            "calibrated_probability_before_network_update": calibrated_before,
            "calibrated_probability_after_network_update": calibrated_after,
            "calibrated_probability_delta": calibrated_delta,
            "snapshot_encoder_sha256": file_sha256(snapshot_encoder),
            "snapshot_main_sha256": file_sha256(snapshot_main),
            "snapshot_differs_from_base": changed,
            "reloaded_probability": probability_after,
            "reloaded_logit": logit_after,
        }
        target = settings.data_dir / "runtime-self-test.json"
        target.write_text(
            json.dumps(sentinel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        target.chmod(0o600)
        print(json.dumps(sentinel, ensure_ascii=False, indent=2))
    finally:
        service.release(participant_id)
        with db.connect() as conn:
            conn.execute("UPDATE participants SET enabled=0 WHERE id=?", (participant_id,))


if __name__ == "__main__":
    main()
