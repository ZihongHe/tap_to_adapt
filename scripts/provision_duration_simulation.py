"""Provision a participant-isolated M0+Personalised duration simulation.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


RUN_SLUG_PATTERN = re.compile(
    r"^simulation-duration-v1-(?P<created_at_ms>[0-9]{13})-(?P<fixture_id>[a-f0-9]{8})$"
)
FIELDNAMES = [
    "participant_code",
    "participant_id",
    "participant_index",
    "voice_id",
    "voice_name",
    "background_voice_id",
    "background_voice_name",
    "speed",
    "run_slug",
    "data_origin",
    "protocol_version",
    "analysis_use",
]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_private_output(path: Path, public_root: Path) -> None:
    resolved = path.resolve()
    public = public_root.resolve()
    if resolved == public or public in resolved.parents:
        raise ValueError("Credential CSV must be outside the public static directory")


def reserve_private_file(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    path.chmod(0o600)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision a participant-isolated M0+Personalised duration simulation."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-overlay",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "content"
        / "duration_sim_protocol.overlay.json",
    )
    parser.add_argument(
        "--voice-ids",
        type=parse_ints,
        default=parse_ints("3,21,42,64,95,117,143,171"),
    )
    parser.add_argument(
        "--speeds",
        type=parse_floats,
        default=parse_floats("0.90,0.95,1.00,1.05,1.10,0.92,1.08,0.98"),
    )
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_simulation_only:
        raise SystemExit("Pass --acknowledge-simulation-only")
    if RUN_SLUG_PATTERN.fullmatch(args.run_slug) is None:
        raise SystemExit(
            "run slug must match simulation-duration-v1-<13digitms>-<8hex>"
        )
    if len(args.voice_ids) < 4 or len(args.voice_ids) != len(set(args.voice_ids)):
        raise SystemExit("Provide at least four unique AISHELL3 voice ids")
    if len(args.speeds) != len(args.voice_ids):
        raise SystemExit("Provide exactly one speed for every voice id")
    if any(not 0.75 <= value <= 1.25 for value in args.speeds):
        raise SystemExit("Every TTS speed must be in [0.75, 1.25]")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    actual_data_dir = settings.data_dir.resolve()
    if actual_data_dir != args.expected_data_dir.resolve():
        raise SystemExit(
            f"Configured data directory mismatch: {actual_data_dir}"
        )
    output = args.output.resolve()
    validate_private_output(output, settings.static_dir)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite credential CSV: {output}")

    base = json.loads(
        (settings.content_dir / "study_protocol.json").read_text(encoding="utf-8")
    )
    overlay = json.loads(args.protocol_overlay.read_text(encoding="utf-8"))
    merged = deep_merge(base, overlay)
    protocol_text = json.dumps(
        merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    protocol_version = str(merged["version"])
    protocol_sha256 = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()

    db = StudyDB(settings)
    if db.study_run(args.run_slug) is not None:
        raise SystemExit(f"Refusing to reuse duration run: {args.run_slug}")
    db.protocol_text = protocol_text
    db.protocol_version = protocol_version
    db.protocol_sha256 = protocol_sha256

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    reserve_private_file(temporary)
    run_id: str | None = None
    try:
        codes = db.create_participants(
            len(args.voice_ids),
            "pilot",
            "personalized",
            preview_mode=True,
            data_origin="simulation",
            protocol_mode="compressed",
            run_slug=args.run_slug,
        )
        run = db.study_run(args.run_slug)
        if run is None:
            raise RuntimeError("Duration simulation run was not created")
        run_id = str(run["id"])
        speakers_path = Path(settings.sherpa_tts_model_dir) / "speakers.txt"
        speakers = [
            line.strip()
            for line in speakers_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip()
        ]
        rows: list[dict[str, Any]] = []
        for index, (code, voice_id, speed) in enumerate(
            zip(codes, args.voice_ids, args.speeds, strict=True), start=1
        ):
            participant = db.participant_by_code(code)
            if participant is None:
                raise RuntimeError("Created duration participant cannot be resolved")
            background_voice_id = args.voice_ids[index % len(args.voice_ids)]
            if not 0 <= voice_id < len(speakers) or not 0 <= background_voice_id < len(
                speakers
            ):
                raise RuntimeError("Requested voice id is outside speakers.txt")
            rows.append(
                {
                    "participant_code": code,
                    "participant_id": participant["id"],
                    "participant_index": index,
                    "voice_id": voice_id,
                    "voice_name": speakers[voice_id],
                    "background_voice_id": background_voice_id,
                    "background_voice_name": speakers[background_voice_id],
                    "speed": speed,
                    "run_slug": args.run_slug,
                    "data_origin": "simulation",
                    "protocol_version": protocol_version,
                    "analysis_use": "duration_simulation_engineering_only",
                }
            )
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output)
        output.chmod(0o600)
        db.record_pipeline_action(
            run_id,
            "simulation.duration_fixture_provisioned",
            actor="duration-simulation-provisioner",
            payload={
                "participant_count": len(rows),
                "voice_ids": args.voice_ids,
                "protocol_version": protocol_version,
                "protocol_sha256": protocol_sha256,
                "analysis_use": "duration_simulation_engineering_only",
                "credential_path": str(output),
            },
        )
    except BaseException:
        if run_id is not None:
            with db.connect() as connection:
                connection.execute(
                    "UPDATE participants SET enabled=0 WHERE run_id=?", (run_id,)
                )
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "simulation_only": True,
                "run_slug": args.run_slug,
                "participant_count": len(args.voice_ids),
                "protocol_version": protocol_version,
                "protocol_sha256": protocol_sha256,
                "voice_ids": args.voice_ids,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
