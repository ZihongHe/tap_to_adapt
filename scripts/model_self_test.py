"""Command-line utility: model self test.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
    from app.timing import DTCNModel

    encoder = Path(settings.base_encoder_checkpoint)
    main_model = Path(settings.base_main_checkpoint)
    model = DTCNModel(settings, str(encoder), str(main_model))
    rng = np.random.default_rng(20260713)
    sample = rng.normal(0.0, 0.015, 16_000 * 30).astype(np.float32)
    probability, logit = model.predict(sample)
    if not math.isfinite(probability) or not math.isfinite(logit):
        raise RuntimeError("Model returned a non-finite prediction")
    if probability < 0 or probability > 1:
        raise RuntimeError("Model probability is outside [0, 1]")

    model.torch.manual_seed(20260713)
    parameters_before = {
        f"encoder.{name}": parameter.detach().cpu().clone()
        for name, parameter in model.encoder.named_parameters()
    }
    parameters_before.update(
        {
            f"main.{name}": parameter.detach().cpu().clone()
            for name, parameter in model.main_model.named_parameters()
        }
    )
    update_loss = model.train(sample, 1)
    changed_parameters = [
        f"encoder.{name}"
        for name, parameter in model.encoder.named_parameters()
        if not model.torch.equal(parameters_before[f"encoder.{name}"], parameter.detach().cpu())
    ]
    changed_parameters.extend(
        f"main.{name}"
        for name, parameter in model.main_model.named_parameters()
        if not model.torch.equal(parameters_before[f"main.{name}"], parameter.detach().cpu())
    )
    weights_changed = bool(changed_parameters)
    if not math.isfinite(update_loss) or not weights_changed:
        raise RuntimeError("Online update did not produce a finite loss and changed weights")

    snapshot_dir = settings.data_dir / "model-self-test-snapshot"
    model.save(snapshot_dir)
    snapshot_encoder = snapshot_dir / "latest_encoder.pth"
    snapshot_main = snapshot_dir / "latest_main_model.pth"
    snapshot_dir.chmod(0o700)
    snapshot_encoder.chmod(0o600)
    snapshot_main.chmod(0o600)
    reloaded = DTCNModel(settings, str(snapshot_encoder), str(snapshot_main))
    probability_after, logit_after = reloaded.predict(sample)
    if not all(map(math.isfinite, (probability_after, logit_after))):
        raise RuntimeError("Reloaded online snapshot returned a non-finite prediction")

    sentinel = {
        "passed": True,
        "created_ms": int(time.time() * 1000),
        "encoder_path": str(encoder.resolve()),
        "encoder_sha256": file_sha256(encoder),
        "main_path": str(main_model.resolve()),
        "main_sha256": file_sha256(main_model),
        "probability": probability,
        "logit": logit,
        "online_update_loss": update_loss,
        "online_weights_changed": weights_changed,
        "online_changed_parameter_count": len(changed_parameters),
        "online_changed_parameter_examples": changed_parameters[:12],
        "snapshot_encoder_sha256": file_sha256(snapshot_encoder),
        "snapshot_main_sha256": file_sha256(snapshot_main),
        "snapshot_probability": probability_after,
        "snapshot_logit": logit_after,
        "config_model_structure": model.config.get("model_structure"),
        "config_input_type": model.config.get("input_type"),
        "device": model.config.get("device"),
    }
    target = Path(settings.model_ready_sentinel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sentinel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target.chmod(0o600)
    print(json.dumps(sentinel, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
