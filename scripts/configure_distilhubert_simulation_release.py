"""Atomically select a validated DistilHuBERT release for simulation-only use."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def replace_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{prefix}{value}"
            break
    else:
        lines.append(f"{prefix}{value}")
    temporary = path.with_name(f".{path.name}.new")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_simulation_only:
        raise SystemExit("Pass --acknowledge-simulation-only")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.distilhubert_online import DistilHuBERTRelease
    from app.integrity import file_sha256

    if settings.formal_study_enabled:
        raise SystemExit("Refusing a simulation release while FORMAL_STUDY_ENABLED=1")
    manifest = args.manifest.resolve()
    evidence = args.selection_evidence.resolve()
    release = DistilHuBERTRelease.load(manifest)
    if not release.training_provenance["development_only"]:
        raise SystemExit("Expected a development-only simulation release")
    if not evidence.is_file():
        raise SystemExit("Selection evidence is missing")

    audit = {
        "schema": "tap-to-adapt-simulation-release-selection/v1",
        "configured_ms": int(time.time() * 1000),
        "simulation_only": True,
        "human_effect_evidence": False,
        "release_manifest": str(manifest),
        "release_manifest_sha256": release.manifest_sha256,
        "selection_evidence": str(evidence),
        "selection_evidence_sha256": file_sha256(evidence),
        "optimizer": {
            "learning_rate": release.optimizer.learning_rate,
            "proximal_to_shared": release.optimizer.proximal_to_shared,
            "replay_per_class": release.optimizer.replay_per_class,
            "replay_batch_size": release.optimizer.replay_batch_size,
            "single_class_update_scale": (
                release.optimizer.single_class_update_scale
            ),
        },
    }
    audit_path = settings.data_dir / "simulation-release-selection.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    audit_path.chmod(0o600)
    replace_env_value(root / ".env", "AGENT_MODE", "distilhubert")
    replace_env_value(
        root / ".env",
        "DISTILHUBERT_RELEASE_MANIFEST",
        str(manifest),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
