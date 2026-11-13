"""Disable every account in one isolated smooth simulation run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from .smooth_simulation_protocol import RUN_SLUG_PATTERN
except ImportError:
    from smooth_simulation_protocol import RUN_SLUG_PATTERN


def disable_smooth_fixture(db: Any, run_slug: str) -> dict[str, Any]:
    if RUN_SLUG_PATTERN.fullmatch(run_slug) is None:
        raise ValueError(
            "Refusing to disable a run outside the simulation-intent-v3 namespace"
        )
    run = db.study_run(run_slug)
    if run is None:
        raise ValueError("Smooth simulation run does not exist")
    if str(run["data_origin"]) != "simulation":
        raise ValueError("Only a simulation-origin run can be disabled")

    run_id = str(run["id"])
    participants = db.participants_in_run(run_id)
    participant_count = len(participants)
    if participant_count not in (8, 32):
        raise ValueError(
            "Smooth simulation cleanup requires exactly 8 or 32 participants"
        )
    if any(str(row["data_origin"]) != "simulation" for row in participants):
        raise ValueError("Every participant must have simulation data origin")
    if any(int(row["preview_mode"]) != 1 for row in participants):
        raise ValueError("Every participant must be preview-only")

    enabled_before = sum(
        1 for participant in participants if int(participant["enabled"]) == 1
    )
    with db.connect() as connection:
        cursor = connection.execute(
            "UPDATE participants SET enabled=0 WHERE run_id=? AND enabled=1",
            (run_id,),
        )
        disabled_now = int(cursor.rowcount)
        enabled_row = connection.execute(
            "SELECT COUNT(*) AS enabled_count FROM participants "
            "WHERE run_id=? AND enabled=1",
            (run_id,),
        ).fetchone()
    if enabled_row is None:
        raise RuntimeError("Could not verify smooth simulation cleanup")
    enabled_after = int(enabled_row["enabled_count"])
    if disabled_now != enabled_before or enabled_after != 0:
        raise RuntimeError(
            "Smooth simulation cleanup did not disable every enabled participant"
        )

    result = {
        "run_slug": run_slug,
        "participant_count": participant_count,
        "enabled_before": enabled_before,
        "disabled_now": disabled_now,
        "enabled_after": enabled_after,
    }
    db.record_pipeline_action(
        run_id,
        "simulation.smooth_run_disabled",
        actor="smooth-simulation-cleanup",
        payload={
            **result,
            "idempotent_replay": enabled_before == 0,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disable every account in one isolated smooth simulation run."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-disable", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_disable:
        raise SystemExit("Pass --acknowledge-disable")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    actual_data_dir = settings.data_dir.resolve()
    expected_data_dir = args.expected_data_dir.resolve()
    if actual_data_dir != expected_data_dir:
        raise SystemExit(
            f"Configured data directory mismatch: actual={actual_data_dir} "
            f"expected={expected_data_dir}"
        )
    try:
        result = disable_smooth_fixture(StudyDB(settings), args.run_slug)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    result["data_dir"] = str(actual_data_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
