"""Score every frozen PCM utterance with the immutable shared head.

The full waveform is evaluated once, without streaming truncation, personal
updates, or activation-state effects.  The resulting map is a stimulus
difficulty audit artifact, not duration or human-effect evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.distilhubert_online import DistilHuBERTOnlineModel  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm_waveform(path: Path, entry: dict[str, Any]) -> np.ndarray:
    pcm = path.read_bytes()
    if (
        len(pcm) != int(entry["pcm_bytes"])
        or hashlib.sha256(pcm).hexdigest() != str(entry["pcm_sha256"])
        or len(pcm) % 2
    ):
        raise RuntimeError(f"Frozen PCM checksum failed: {path}")
    return (
        np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    if not args.bundle_root.is_dir():
        raise SystemExit("Frozen PCM bundle is absent")
    bundle_manifest_path = args.bundle_root / "manifest.json"
    if not bundle_manifest_path.is_file():
        raise SystemExit("Frozen PCM bundle root manifest is absent")
    bundle_manifest = json.loads(
        bundle_manifest_path.read_text(encoding="utf-8")
    )
    if (
        bundle_manifest.get("schema")
        != "tap-to-adapt-frozen-pcm-bundle/v1"
        or bundle_manifest.get("cohort") not in {"selection8", "duration32"}
    ):
        raise SystemExit("Frozen PCM bundle root manifest is invalid")
    expected_participants = int(bundle_manifest.get("participants", 0))
    cohort_expected = 8 if bundle_manifest["cohort"] == "selection8" else 32
    if expected_participants != cohort_expected:
        raise SystemExit("Frozen PCM bundle participant count is invalid")

    load_dotenv(ROOT / ".env")
    settings = Settings.from_env()
    release_manifest = (
        args.release_manifest
        if args.release_manifest is not None
        else Path(settings.distilhubert_release_manifest)
    ).resolve()
    model = DistilHuBERTOnlineModel(
        release_manifest,
        update_enabled=False,
        cache_dir=settings.distilhubert_cache_dir or None,
        allow_download=False,
        device=(
            None
            if settings.distilhubert_device == "auto"
            else settings.distilhubert_device
        ),
    )

    manifests = []
    rows = []
    for participant_index in range(1, expected_participants + 1):
        participant_dir = (
            args.bundle_root / f"participant-{participant_index:02d}"
        )
        manifest_path = participant_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema")
            != "tap-to-adapt-frozen-pcm-participant/v1"
            or int(manifest["participant_index"]) != participant_index
        ):
            raise SystemExit(f"Invalid PCM manifest for participant {participant_index}")
        manifest_sha256 = file_sha256(manifest_path)
        manifests.append(
            {
                "participant_index": participant_index,
                "participant_manifest_sha256": manifest_sha256,
                "entries": len(manifest["entries"]),
            }
        )
        for entry in manifest["entries"]:
            waveform = pcm_waveform(
                participant_dir / str(entry["filename"]), entry
            )
            prediction = model.predict_both(waveform)
            rows.append(
                {
                    "participant_index": participant_index,
                    "cache_key": str(entry["cache_key"]),
                    "speaker_role": str(entry["speaker_role"]),
                    "text_sha256": str(entry["text_sha256"]),
                    "pcm_sha256": str(entry["pcm_sha256"]),
                    "pcm_samples": int(waveform.size),
                    "shared_probability": prediction.shared_probability,
                    "shared_logit": prediction.shared_logit,
                }
            )
        print(
            f"participant {participant_index}: {len(manifest['entries'])} utterances",
            file=sys.stderr,
            flush=True,
        )

    bundle_sha256 = hashlib.sha256(
        json.dumps(
            manifests,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "tap-to-adapt-full-utterance-shared-score-map/v1",
        "simulation_only": True,
        "duration_evidence": False,
        "scoring_mode": "full_utterance_shared_only_no_truncation",
        "release_manifest_sha256": model.release.manifest_sha256,
        "shared_head_sha256": model.release.shared_head_sha256,
        "shared_head_fingerprint": model.identity()["shared_head_fingerprint"],
        "bundle_aggregate_manifest_sha256": bundle_sha256,
        "participant_manifests": manifests,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"participant_manifests", "rows"}
            }
            | {"participants": len(manifests), "utterances": len(rows)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
