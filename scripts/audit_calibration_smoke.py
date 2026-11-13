"""Audit persisted calibration state for an isolated QA run.

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
        description="Audit persisted calibration state for an isolated QA run."
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    db = StudyDB(settings)
    run = db.study_run(args.run_slug)
    if run is None:
        raise SystemExit(f"Unknown run: {args.run_slug}")
    participants = db.participants_in_run(run["id"])
    rows = []
    for participant in participants:
        state = db.calibration_state(
            participant["id"],
            calibration_version="engineering-locked-temperature-personal-intercept-v4",
            protocol_sha256=run["protocol_sha256"],
            initial_logit_bias=0.0,
        )
        rows.append(
            {
                "participant_id": participant["id"],
                "feedback_counts": db.feedback_counts(participant["id"]),
                "personal_logit_bias": state["personal_logit_bias"],
                "update_count": state["update_count"],
                "positive_updates": state["positive_updates"],
                "negative_updates": state["negative_updates"],
                "calibration_version": state["calibration_version"],
                "protocol_sha256": state["protocol_sha256"],
            }
        )
    checks = {
        "simulation_origin": run["data_origin"] == "simulation",
        "protocol_is_v10_3": run["protocol_version"].endswith("-v10.3"),
        "two_accounts": len(rows) == 2,
        "one_positive_and_one_negative": sorted(
            (row["positive_updates"], row["negative_updates"]) for row in rows
        )
        == [(0, 1), (1, 0)],
        "one_persisted_update_each": all(row["update_count"] == 1 for row in rows),
        "counter_matches_calibration": all(
            row["feedback_counts"]["eligible_feedbacks"] == row["update_count"]
            for row in rows
        ),
        "protocol_hash_matches": all(
            row["protocol_sha256"] == run["protocol_sha256"] for row in rows
        ),
        "bias_directions_are_correct": any(
            row["positive_updates"] == 1 and row["personal_logit_bias"] > 0
            for row in rows
        )
        and any(
            row["negative_updates"] == 1 and row["personal_logit_bias"] < 0
            for row in rows
        ),
    }
    result = {
        "passed": all(checks.values()),
        "run_slug": args.run_slug,
        "protocol_version": run["protocol_version"],
        "checks": checks,
        "participants": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
