#!/usr/bin/env python3
"""Run the optional DistilHuBERT release through its real online contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distilhubert_online import DistilHuBERTOnlineModel  # noqa: E402
from app.integrity import file_sha256  # noqa: E402


def read_wav(path: Path) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required by this engineering self-test") from exc
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16_000 or np.asarray(audio).ndim != 1:
        raise RuntimeError("Self-test audio must be mono 16 kHz")
    return np.asarray(audio, dtype=np.float32)


def run(args: argparse.Namespace) -> dict[str, object]:
    def progress(message: str) -> None:
        print(f"[distilhubert-self-test] {message}", file=sys.stderr, flush=True)

    audio = read_wav(args.audio.resolve())
    common = {
        "cache_dir": args.cache_dir,
        "allow_download": args.allow_download,
        "device": None if args.device == "auto" else args.device,
    }
    progress("loading learning runtime")
    learning = DistilHuBERTOnlineModel(
        args.manifest.resolve(), update_enabled=True, **common
    )
    progress("predicting shared/personal baseline")
    before = learning.predict_both(audio)
    progress("applying personal-head update")
    loss = learning.train(audio, args.label)
    progress("predicting after update")
    after = learning.predict_both(audio)
    if not learning.last_update_applied:
        raise RuntimeError(f"Personal-head update failed: {learning.last_update_reason}")
    if before.shared_head_fingerprint != after.shared_head_fingerprint:
        raise RuntimeError("Immutable shared head changed during online update")
    if before.personal_head_fingerprint == after.personal_head_fingerprint:
        raise RuntimeError("Personal head did not diverge after online update")
    progress("saving personal checkpoint")
    pointer, head = learning.save(args.checkpoint_dir.resolve())
    progress("reloading personal checkpoint")
    restored = DistilHuBERTOnlineModel(
        args.manifest.resolve(),
        update_enabled=True,
        personal_checkpoint_dir=pointer.parent,
        **common,
    )
    progress("predicting reloaded checkpoint")
    restored_prediction = restored.predict_both(audio)
    if restored_prediction.personal_probability != after.personal_probability:
        raise RuntimeError("Saved personal head does not reproduce its prediction")

    progress("loading frozen comparator")
    frozen = DistilHuBERTOnlineModel(
        args.manifest.resolve(), update_enabled=False, **common
    )
    frozen_before = frozen.identity()
    frozen_loss = frozen.train(audio, args.label)
    frozen_after = frozen.identity()
    if frozen_loss != 0.0 or frozen.last_update_reason != "frozen_condition":
        raise RuntimeError("Frozen condition did not reject its update")
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen condition changed model state")

    progress("verifying frozen backbone")
    feature_encoder = getattr(learning, "feature_encoder", None)
    backbone = getattr(feature_encoder, "model", None)
    backbone_all_frozen = bool(
        backbone is not None
        and not any(parameter.requires_grad for parameter in backbone.parameters())
    )
    if not backbone_all_frozen:
        raise RuntimeError("Backbone is not observably frozen")
    progress("all checks passed")
    return {
        "passed": True,
        "model_identity": learning.identity(),
        "audio": str(args.audio.resolve()),
        "label": args.label,
        "probability_before": before.personal_probability,
        "probability_after": after.personal_probability,
        "shared_probability_before": before.shared_probability,
        "shared_probability_after": after.shared_probability,
        "loss": loss,
        "backbone_all_frozen": backbone_all_frozen,
        "checkpoint_manifest": str(pointer),
        "checkpoint_manifest_sha256": file_sha256(pointer),
        "personal_head": str(head),
        "personal_head_sha256": file_sha256(head),
        "reload_probability_equal": True,
        "frozen_update_rejected": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--audio", type=Path, required=True)
    value.add_argument("--checkpoint-dir", type=Path, required=True)
    value.add_argument("--cache-dir", type=Path)
    value.add_argument("--allow-download", action="store_true")
    value.add_argument("--device", default="auto")
    value.add_argument("--label", type=int, choices=(0, 1), default=1)
    return value


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
