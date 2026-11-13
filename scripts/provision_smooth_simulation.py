"""Provision isolated accounts for the smooth 4+2 simulation.

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
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .smooth_simulation_protocol import RUN_SLUG_PATTERN
except ImportError:
    from smooth_simulation_protocol import RUN_SLUG_PATTERN


FIELDNAMES = [
    "participant_code",
    "participant_id",
    "participant_index",
    "voice_id",
    "voice_name",
    "experimental_agent_voice_id",
    "experimental_agent_voice_name",
    "companion_agent_voice_id",
    "companion_agent_voice_name",
    "speed",
    "run_slug",
    "cohort",
    "data_origin",
    "protocol_version",
    "analysis_use",
]
DEFAULT_DURATION_VOICES = (
    "3,8,13,18,23,28,33,38,43,48,53,58,63,68,73,78,"
    "83,88,93,98,103,108,113,118,123,128,133,138,143,148,153,158"
)
DEFAULT_DURATION_SPEEDS = (
    "1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,"
    "1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,"
    "1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,"
    "1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00"
)
DEFAULT_SELECTION_VOICES = "1,21,41,61,81,101,121,141"
DEFAULT_SELECTION_SPEEDS = "1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00"


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def reserve(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision isolated accounts for the smooth 4+2 simulation."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--cohort", choices=("selection8", "duration32"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-overlay",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "content"
        / "smooth_duration_sim_protocol.overlay.json",
    )
    parser.add_argument("--voice-ids", type=parse_ints)
    parser.add_argument("--speeds", type=parse_floats)
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_simulation_only:
        raise SystemExit("Pass --acknowledge-simulation-only")
    if RUN_SLUG_PATTERN.fullmatch(args.run_slug) is None:
        raise SystemExit("run slug is outside simulation-intent-v3")
    if args.cohort == "duration32":
        voice_ids = args.voice_ids or parse_ints(DEFAULT_DURATION_VOICES)
        speeds = args.speeds or parse_floats(DEFAULT_DURATION_SPEEDS)
        expected = 32
    else:
        voice_ids = args.voice_ids or parse_ints(DEFAULT_SELECTION_VOICES)
        speeds = args.speeds or parse_floats(DEFAULT_SELECTION_SPEEDS)
        expected = 8
    if not (len(voice_ids) == len(speeds) == expected):
        raise SystemExit(f"{args.cohort} requires exactly {expected} voices and speeds")
    if len(set(voice_ids)) != expected:
        raise SystemExit("Participant speaker IDs must all be distinct")
    if any(not 0.75 <= value <= 1.25 for value in speeds):
        raise SystemExit("Every TTS speed must be in [0.75, 1.25]")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    if settings.data_dir.resolve() != args.expected_data_dir.resolve():
        raise SystemExit("Configured data directory does not match --expected-data-dir")
    agent_voice_ids = {
        int(settings.sherpa_tts_primary_speaker_id),
        int(settings.sherpa_tts_companion_speaker_id),
    }
    if len(agent_voice_ids) != 2 or agent_voice_ids.intersection(voice_ids):
        raise SystemExit(
            "Agent voices must be distinct and excluded from participant voices"
        )
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    public_root = settings.static_dir.resolve()
    if output == public_root or public_root in output.parents:
        raise SystemExit("Credential CSV must remain outside the public static directory")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    base = json.loads(
        (settings.content_dir / "study_protocol.json").read_text(encoding="utf-8")
    )
    overlay = json.loads(args.protocol_overlay.read_text(encoding="utf-8"))
    merged = deep_merge(base, overlay)
    merged["duration_simulation"]["cohort"] = args.cohort
    protocol_text = json.dumps(
        merged, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    protocol_version = str(merged["version"])
    protocol_sha256 = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()

    db = StudyDB(settings)
    if db.study_run(args.run_slug) is not None:
        raise SystemExit(f"Refusing to reuse {args.run_slug}")
    db.protocol_text = protocol_text
    db.protocol_version = protocol_version
    db.protocol_sha256 = protocol_sha256
    speakers_path = Path(settings.sherpa_tts_model_dir) / "speakers.txt"
    speakers = [
        row.strip()
        for row in speakers_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if row.strip()
    ]
    if any(not 0 <= value < len(speakers) for value in voice_ids):
        raise SystemExit("A requested speaker ID is outside speakers.txt")

    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    reserve(temporary)
    run_id: str | None = None
    try:
        codes = db.create_participants(
            expected,
            "pilot",
            "personalized",
            preview_mode=True,
            data_origin="simulation",
            protocol_mode="compressed",
            run_slug=args.run_slug,
        )
        run = db.study_run(args.run_slug)
        if run is None:
            raise RuntimeError("Smooth simulation run was not created")
        run_id = str(run["id"])
        rows = []
        for index, (code, voice_id, speed) in enumerate(
            zip(codes, voice_ids, speeds, strict=True), start=1
        ):
            participant = db.participant_by_code(code)
            if participant is None:
                raise RuntimeError("Created participant cannot be resolved")
            rows.append(
                {
                    "participant_code": code,
                    "participant_id": participant["id"],
                    "participant_index": index,
                    "voice_id": voice_id,
                    "voice_name": speakers[voice_id],
                    "experimental_agent_voice_id": settings.sherpa_tts_primary_speaker_id,
                    "experimental_agent_voice_name": (
                        f"aishell3-sid{settings.sherpa_tts_primary_speaker_id}"
                    ),
                    "companion_agent_voice_id": settings.sherpa_tts_companion_speaker_id,
                    "companion_agent_voice_name": (
                        f"aishell3-sid{settings.sherpa_tts_companion_speaker_id}"
                    ),
                    "speed": speed,
                    "run_slug": args.run_slug,
                    "cohort": args.cohort,
                    "data_origin": "simulation",
                    "protocol_version": protocol_version,
                    "analysis_use": "smooth_4plus2_engineering_only",
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
            "simulation.smooth_fixture_provisioned",
            actor="smooth-simulation-provisioner",
            payload={
                "cohort": args.cohort,
                "participant_count": expected,
                "target_voice_ids": voice_ids,
                "experimental_agent_voice": (
                    f"aishell3-sid{settings.sherpa_tts_primary_speaker_id}"
                ),
                "companion_agent_voice": (
                    f"aishell3-sid{settings.sherpa_tts_companion_speaker_id}"
                ),
                "protocol_sha256": protocol_sha256,
                "analysis_use": "smooth_4plus2_engineering_only",
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
                "cohort": args.cohort,
                "participant_count": expected,
                "target_voice_ids": voice_ids,
                "experimental_agent_voice": (
                    f"aishell3-sid{settings.sherpa_tts_primary_speaker_id}"
                ),
                "companion_agent_voice": (
                    f"aishell3-sid{settings.sherpa_tts_companion_speaker_id}"
                ),
                "protocol_version": protocol_version,
                "protocol_sha256": protocol_sha256,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
