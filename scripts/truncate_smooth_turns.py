"""Create a shorter, immutable 4+2 schedule from a frozen template.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .smooth_simulation_protocol import validate_schedule
except ImportError:
    from smooth_simulation_protocol import validate_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a shorter, immutable 4+2 schedule from a frozen template."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-minutes", type=int, required=True)
    args = parser.parse_args()
    if args.maximum_minutes < 18 or args.maximum_minutes % 6:
        raise SystemExit("--maximum-minutes must be a multiple of 6 and at least 18")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    document = json.loads(args.input.read_text(encoding="utf-8"))
    maximum_cycle = args.maximum_minutes // 6
    participants = []
    for participant in document["participants"]:
        events = [
            event
            for event in participant["events"]
            if int(event["cycle_index"]) <= maximum_cycle
        ]
        validate_schedule(events, max_minutes=args.maximum_minutes)
        participants.append({**participant, "events": events})
    output = {
        **document,
        "max_minutes": args.maximum_minutes,
        "participants": participants,
        "truncation": {
            "source": str(args.input.resolve()),
            "maximum_minutes": args.maximum_minutes,
            "preserved_fields": "text, intent, profile, timing, difficulty, voice calibration",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "participants": len(participants),
                "maximum_minutes": args.maximum_minutes,
                "events_per_participant": len(participants[0]["events"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
