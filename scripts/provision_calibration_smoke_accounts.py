"""Create isolated simulation-only accounts for public calibration smoke tests.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create isolated simulation-only accounts for public calibration smoke tests."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_simulation_only:
        raise SystemExit("Pass --acknowledge-simulation-only")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    db = StudyDB(settings)
    run_slug = f"simulation-calibration-smoke-v10-3-{int(time.time())}"
    codes = db.create_participants(
        2,
        "pilot",
        "qa_calibration_public",
        preview_mode=True,
        data_origin="simulation",
        protocol_mode="compressed",
        run_slug=run_slug,
    )
    rows = [
        {"purpose": "positive", "participant_code": codes[0]},
        {"purpose": "negative", "participant_code": codes[1]},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["purpose", "participant_code"])
        writer.writeheader()
        writer.writerows(rows)
    try:
        args.output.parent.chmod(0o700)
        args.output.chmod(0o600)
    except OSError:
        pass
    print(
        json.dumps(
            {
                "simulation_only": True,
                "run_slug": run_slug,
                "protocol_version": db.protocol_version,
                "output": str(args.output),
                "accounts": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
