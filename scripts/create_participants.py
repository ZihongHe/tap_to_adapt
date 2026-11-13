"""Create non-sequential participant codes.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    parser = argparse.ArgumentParser(description="Create non-sequential participant codes.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--cohort", choices=["pilot", "study_a", "study_b"], required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Create an analysis-ineligible formal-flow preview account.",
    )
    parser.add_argument("--origin", choices=["formal", "pilot", "simulation"])
    parser.add_argument("--protocol-mode", choices=["strict", "compressed"])
    parser.add_argument("--run-slug")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    codes = StudyDB(settings).create_participants(
        args.count,
        args.cohort,
        args.group,
        preview_mode=args.preview,
        data_origin=args.origin,
        protocol_mode=args.protocol_mode,
        run_slug=args.run_slug,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "participant_code",
                "cohort",
                "group",
                "preview_mode",
                "data_origin",
                "protocol_mode",
                "run_slug",
            ]
        )
        writer.writerows(
            (
                code,
                args.cohort,
                args.group,
                int(args.preview),
                args.origin or ("simulation" if args.preview else "pilot" if args.cohort == "pilot" else "formal"),
                args.protocol_mode or ("compressed" if args.preview or args.cohort == "pilot" else "strict"),
                args.run_slug or "default",
            )
            for code in codes
        )
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass
    print(f"Created {len(codes)} codes in {args.output}. This is the only full-code export.")


if __name__ == "__main__":
    main()
