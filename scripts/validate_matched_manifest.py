"""Validate a frozen matched-other manifest.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frozen matched-other manifest.")
    parser.add_argument("--run-slug")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")

    from app.config import settings
    from app.db import StudyDB
    from app.matched import file_sha256, load_json, matched_paths

    db = StudyDB(settings)
    run = db.study_run(args.run_slug) if args.run_slug else None
    if args.run_slug and run is None:
        raise SystemExit(f"Unknown run slug: {args.run_slug}")
    run_id = str(run["id"]) if run else None
    expected_days = {"1", "2", "3"}
    if run:
        try:
            protocol = json.loads(str(run["protocol_json"]))
            session_design = protocol.get("session_design")
            total_sessions = (
                int(session_design.get("total_sessions") or 3)
                if isinstance(session_design, dict)
                else 3
            )
            expected_days = {str(day) for day in range(1, total_sessions + 1)}
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit("Run protocol has invalid session_design") from exc
    manifest_path, target = matched_paths(settings, run_id)
    manifest = load_json(manifest_path)
    if manifest is None:
        raise SystemExit(f"Invalid or missing manifest: {manifest_path}")
    if manifest.get("approved") is not True:
        raise SystemExit("Manifest must set approved=true after the assignment is frozen")
    if not manifest.get("assignment_rule") or int(manifest.get("frozen_at_ms") or 0) <= 0:
        raise SystemExit("Manifest requires assignment_rule and frozen_at_ms")
    if not isinstance(manifest.get("matching_variables"), list) or not manifest["matching_variables"]:
        raise SystemExit("Manifest requires a non-empty matching_variables list")
    if manifest.get("random_seed") is None:
        raise SystemExit("Manifest requires the frozen random_seed")

    assignments = manifest.get("assignments")
    if not isinstance(assignments, dict) or not assignments:
        raise SystemExit("Manifest has no participant assignments")

    roster = {row["id"]: row for row in db.admin_roster()}
    tested: dict[str, dict[str, dict[str, str]]] = {}
    for participant_id, day_assignments in assignments.items():
        if participant_id not in roster:
            raise SystemExit(f"Unknown target participant: {participant_id}")
        if not isinstance(day_assignments, dict) or set(day_assignments) != expected_days:
            raise SystemExit(
                f"Assignments for {participant_id} must cover protocol days "
                f"{', '.join(sorted(expected_days, key=int))}"
            )

        sessions = db.sessions(participant_id)
        if sessions and int(manifest["frozen_at_ms"]) >= min(row["started_ms"] for row in sessions):
            raise SystemExit(f"Assignment was not frozen before the first session: {participant_id}")

        tested[participant_id] = {}
        for day, assignment in day_assignments.items():
            if not isinstance(assignment, dict):
                raise SystemExit(f"Invalid day {day} assignment for {participant_id}")
            donor_id = str(assignment.get("donor_participant_id") or "")
            if donor_id == participant_id or donor_id not in roster:
                raise SystemExit(
                    f"Donor must be another registered participant: {participant_id}, day {day}"
                )
            if not isinstance(assignment.get("matching_strata"), dict):
                raise SystemExit(f"matching_strata is required for {participant_id}, day {day}")

            encoder = Path(str(assignment.get("encoder_checkpoint") or ""))
            main = Path(str(assignment.get("main_checkpoint") or ""))
            if (
                not encoder.is_absolute()
                or not main.is_absolute()
                or not encoder.is_file()
                or not main.is_file()
            ):
                raise SystemExit(
                    f"Checkpoint is missing or not absolute for {participant_id}, day {day}"
                )
            encoder_hash = file_sha256(encoder)
            main_hash = file_sha256(main)
            if encoder_hash != str(assignment.get("encoder_sha256") or "").lower():
                raise SystemExit(f"Encoder hash mismatch for {participant_id}, day {day}")
            if main_hash != str(assignment.get("main_sha256") or "").lower():
                raise SystemExit(f"Main-model hash mismatch for {participant_id}, day {day}")
            tested[participant_id][day] = {
                "donor_participant_id": donor_id,
                "encoder_sha256": encoder_hash,
                "main_sha256": main_hash,
            }

    sentinel = {
        "passed": True,
        "created_ms": int(time.time() * 1000),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_version": manifest.get("version"),
        "assignments": tested,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sentinel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target.chmod(0o600)
    print(json.dumps(sentinel, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
