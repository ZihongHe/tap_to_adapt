"""Manage the fail-closed formal A-to-B pipeline.

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
    parser = argparse.ArgumentParser(description="Manage the fail-closed formal A-to-B pipeline.")
    parser.add_argument("action", choices=["status", "release-study-a", "prepare-study-b", "release-reserve"])
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--wave", type=int)
    parser.add_argument("--selected-n", type=int)
    parser.add_argument("--selected-q", type=int)
    parser.add_argument("--decision-record")
    parser.add_argument("--reason")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.pipeline import FormalPipeline

    db = StudyDB(settings)
    pipeline = FormalPipeline.from_run(db, settings, args.run_slug)
    if args.action == "status":
        result = pipeline.status(args.run_slug)
    elif args.action == "release-study-a":
        if args.wave is None:
            raise SystemExit("--wave is required")
        result = pipeline.release_study_a_wave(args.run_slug, args.wave)
    elif args.action == "prepare-study-b":
        if args.selected_n is None or args.selected_q is None or not args.decision_record:
            raise SystemExit(
                "--selected-n, --selected-q and --decision-record are required for prepare-study-b"
            )
        result = pipeline.prepare_study_b(
            args.run_slug, args.selected_n, args.selected_q, args.decision_record
        )
    else:
        if not args.reason or len(args.reason.strip()) < 10:
            raise SystemExit("--reason must document the blinded reserve-release rule")
        result = pipeline.release_study_b_reserve(args.run_slug, args.reason.strip())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
