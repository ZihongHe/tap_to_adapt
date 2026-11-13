"""Build a locked, cycle-balanced 4+2 minute LLM-text schedule.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from .smooth_simulation_protocol import build_turn_payload
except ImportError:
    from smooth_simulation_protocol import build_turn_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a locked, cycle-balanced 4+2 minute LLM-text schedule."
    )
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-pools", type=Path, required=True)
    parser.add_argument("--difficulty-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", choices=("selection8", "duration32"), required=True)
    parser.add_argument("--max-minutes", type=int, default=72)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        payload = build_turn_payload(
            roster_rows=rows,
            source_pools=args.source_pools,
            difficulty_map=args.difficulty_map,
            cohort=args.cohort,
            max_minutes=args.max_minutes,
        )
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from None
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "simulation_only": True,
                "run_slug": payload["run_slug"],
                "cohort": payload["cohort"],
                "participants": len(payload["participants"]),
                "events_per_participant": len(payload["participants"][0]["events"]),
                "max_minutes": payload["max_minutes"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
