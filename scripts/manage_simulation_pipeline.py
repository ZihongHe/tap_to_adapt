"""Manage an isolated simulation-only A-to-B pipeline.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage an isolated simulation-only A-to-B pipeline."
    )
    parser.add_argument(
        "action",
        choices=["status", "release-study-a", "prepare-study-b", "release-reserve"],
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--wave", type=int)
    parser.add_argument("--selected-n", type=int)
    parser.add_argument("--selected-q", type=int)
    parser.add_argument("--decision-record")
    parser.add_argument("--reason")
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.pipeline import FormalPipeline

    db = StudyDB(settings)
    run = db.study_run(args.run_slug)
    if run is None:
        raise SystemExit(f"Unknown run slug: {args.run_slug}")
    if run["data_origin"] != "simulation" or run["protocol_mode"] != "compressed":
        raise SystemExit("This command accepts only simulation/compressed runs")
    if args.action != "status" and not args.acknowledge_simulation_only:
        raise SystemExit(
            "Write actions require --acknowledge-simulation-only; outputs are never analysis eligible"
        )

    pipeline = FormalPipeline.from_run(
        db, settings, args.run_slug, allow_simulation=True
    )
    if args.action == "status":
        result = pipeline.status(args.run_slug)
    elif args.action == "release-study-a":
        if args.wave is None:
            raise SystemExit("--wave is required")
        result = pipeline.release_study_a_wave(args.run_slug, args.wave)
    elif args.action == "prepare-study-b":
        if args.selected_n is None or args.selected_q is None or not args.decision_record:
            raise SystemExit(
                "--selected-n, --selected-q and --decision-record are required"
            )
        result = pipeline.prepare_study_b(
            args.run_slug,
            args.selected_n,
            args.selected_q,
            args.decision_record,
        )
    else:
        if not args.reason or len(args.reason.strip()) < 10:
            raise SystemExit("--reason must document why simulation reserves are released")
        result = pipeline.release_study_b_reserve(args.run_slug, args.reason.strip())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
