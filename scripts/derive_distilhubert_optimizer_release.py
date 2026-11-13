"""Derive an immutable DistilHuBERT release with a locked online optimizer.

The shared head and pinned backbone stay unchanged.  Only the runtime replay
contract is replaced.  A separate audit file binds the new release to the
model-selection evidence without adding unsupported fields to ``release.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.distilhubert_online import DistilHuBERTRelease  # noqa: E402
from app.integrity import file_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--proximal-to-shared", type=float, required=True)
    parser.add_argument("--replay-per-class", type=int, required=True)
    parser.add_argument("--replay-batch-size", type=int, required=True)
    parser.add_argument("--single-class-update-scale", type=float, required=True)
    args = parser.parse_args()

    base_manifest = args.base_manifest.resolve()
    evidence = args.selection_evidence.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Refusing to overwrite immutable release: {output_dir}")
    if not evidence.is_file():
        raise SystemExit(f"Selection evidence is missing: {evidence}")

    # Validate the base before copying any artifact.
    base_release = DistilHuBERTRelease.load(base_manifest)
    raw = json.loads(base_manifest.read_text(encoding="utf-8"))
    if raw["schema"] != "tap-to-adapt-distilhubert-release/v2":
        raise SystemExit("Base release is not the replay-capable v2 schema")
    if not raw["training_provenance"]["development_only"]:
        raise SystemExit("This derivation tool is restricted to development releases")

    optimizer = dict(raw["optimizer"])
    optimizer.update(
        {
            "learning_rate": args.learning_rate,
            "proximal_to_shared": args.proximal_to_shared,
            "replay_per_class": args.replay_per_class,
            "replay_batch_size": args.replay_batch_size,
            "require_both_replay_classes": False,
            "single_class_update_scale": args.single_class_update_scale,
        }
    )
    raw["optimizer"] = optimizer

    temporary = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.mkdir(parents=True)
        shared_name = str(raw["shared_head"]["path"])
        shutil.copy2(base_release.shared_head_path, temporary / shared_name)
        if base_release.replay_seed_path is not None:
            replay_name = str(raw["replay_seed"]["path"])
            shutil.copy2(base_release.replay_seed_path, temporary / replay_name)
        manifest = temporary / "release.json"
        manifest.write_text(
            json.dumps(raw, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        derived = DistilHuBERTRelease.load(manifest)
        audit = {
            "schema": "tap-to-adapt-derived-optimizer-release-audit/v1",
            "base_manifest": str(base_manifest),
            "base_manifest_sha256": file_sha256(base_manifest),
            "selection_evidence": str(evidence),
            "selection_evidence_sha256": file_sha256(evidence),
            "derived_manifest_sha256": derived.manifest_sha256,
            "shared_head_unchanged": (
                derived.shared_head_sha256 == base_release.shared_head_sha256
            ),
            "optimizer": optimizer,
            "simulation_only": True,
            "human_effect_evidence": False,
        }
        (temporary / "derivation-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(
        json.dumps(
            {
                "release_manifest": str(output_dir / "release.json"),
                "release_manifest_sha256": file_sha256(
                    output_dir / "release.json"
                ),
                "audit": str(output_dir / "derivation-audit.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
