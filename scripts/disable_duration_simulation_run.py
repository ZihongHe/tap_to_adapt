"""Disable, without deleting, one exact duration-simulation run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


RUN_SLUG_PATTERN = re.compile(
    r"^simulation-duration-v1-[0-9]{13}-[a-f0-9]{8}$"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Disable, without deleting, one exact duration-simulation run."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-disable", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_disable:
        raise SystemExit("Pass --acknowledge-disable")
    if RUN_SLUG_PATTERN.fullmatch(args.run_slug) is None:
        raise SystemExit("Refusing a run outside the duration-simulation namespace")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    if settings.data_dir.resolve() != args.expected_data_dir.resolve():
        raise SystemExit("Configured data directory does not match --expected-data-dir")
    db = StudyDB(settings)
    run = db.study_run(args.run_slug)
    if run is None or run["data_origin"] != "simulation":
        raise SystemExit("Duration simulation run is absent or not simulation-only")
    participants = db.participants_in_run(str(run["id"]))
    if not participants:
        raise SystemExit("Duration simulation run has no participants")
    enabled_before = sum(int(row["enabled"]) for row in participants)
    with db.connect() as connection:
        connection.execute(
            "UPDATE participants SET enabled=0 WHERE run_id=?", (str(run["id"]),)
        )
    db.record_pipeline_action(
        str(run["id"]),
        "simulation.duration_fixture_disabled",
        actor="duration-simulation-disabler",
        payload={
            "participant_count": len(participants),
            "enabled_before": enabled_before,
            "enabled_after": 0,
        },
    )
    print(
        json.dumps(
            {
                "simulation_only": True,
                "run_slug": args.run_slug,
                "participant_count": len(participants),
                "enabled_before": enabled_before,
                "enabled_after": 0,
                "recoverable": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
