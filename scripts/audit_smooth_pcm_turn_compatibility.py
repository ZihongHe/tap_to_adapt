"""Audit whether frozen smooth-turn text is present in every PCM manifest.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_entries(bundle_root: Path, participant_index: int) -> dict[str, Any]:
    manifest_path = (
        bundle_root / f"participant-{participant_index:02d}" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "tap-to-adapt-frozen-pcm-participant/v1"
        or int(manifest["participant_index"]) != participant_index
    ):
        raise ValueError(f"Invalid PCM manifest for participant {participant_index}")
    return {
        str(row["cache_key"]): row
        for row in manifest["entries"]
    }


def participant_missing(
    *,
    entries: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        role = str(event["speaker_role"])
        text = str(event["text"])
        cache_key = hashlib.sha256(f"{role}\0{text}".encode("utf-8")).hexdigest()
        if cache_key in seen:
            continue
        seen.add(cache_key)
        entry = entries.get(cache_key)
        if (
            entry is None
            or entry.get("speaker_role") != role
            or entry.get("text_sha256")
            != hashlib.sha256(text.encode("utf-8")).hexdigest()
        ):
            missing.append(
                {
                    "opportunity_index": int(event["opportunity_index"]),
                    "speaker_role": role,
                    "text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether frozen smooth-turn text is present in every PCM manifest."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--turns", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for turns_path in args.turns:
        payload = json.loads(turns_path.read_text(encoding="utf-8"))
        participants = payload.get("participants")
        if not isinstance(participants, list) or len(participants) != 32:
            raise SystemExit(f"{turns_path} must contain 32 participants")
        participant_reports = []
        for participant in participants:
            participant_index = int(participant["participant_index"])
            missing = participant_missing(
                entries=load_entries(args.bundle_root, participant_index),
                events=participant["events"],
            )
            participant_reports.append(
                {
                    "participant_index": participant_index,
                    "missing_unique_entries": len(missing),
                    "missing": missing,
                }
            )
        reports.append(
            {
                "turns": str(turns_path),
                "turns_sha256": hashlib.sha256(turns_path.read_bytes()).hexdigest(),
                "compatible_participants": sum(
                    row["missing_unique_entries"] == 0
                    for row in participant_reports
                ),
                "missing_unique_entries": sum(
                    row["missing_unique_entries"] for row in participant_reports
                ),
                "participants": participant_reports,
            }
        )
    print(json.dumps({"reports": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
