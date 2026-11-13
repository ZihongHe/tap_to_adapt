"""Verify that live runtimes use each run's immutable protocol snapshot.

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


def expected_configuration(study: dict) -> dict:
    initialization = study.get("initialization") or {}
    schedule = [
        (int(item["eligible_feedbacks_lt"]), float(item["scale"]))
        for item in initialization.get("network_update_schedule") or []
    ]
    return {
        "temperature": float(initialization.get("temperature", 1.0)),
        "logit_bias": float(initialization.get("logit_bias", 0.0)),
        "activation_threshold": float(initialization.get("activation_threshold", 0.5)),
        "network_update_schedule": schedule,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that live runtimes use each run's immutable protocol snapshot."
    )
    parser.add_argument("--run-slug", action="append", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.timing import TimingService

    db = StudyDB(settings)
    service = TimingService(settings, db)
    rows = []
    for slug in args.run_slug:
        run = db.study_run(slug)
        if run is None:
            raise SystemExit(f"Unknown run: {slug}")
        participants = db.participants_in_run(run["id"])
        if not participants:
            raise SystemExit(f"Run has no participants: {slug}")
        participant = participants[0]
        actual = service._runtime_configuration(participant)
        expected = expected_configuration(json.loads(run["protocol_json"]))
        checks = {
            "protocol_hash_matches": actual["protocol_sha256"]
            == run["protocol_sha256"],
            "temperature_matches": actual["temperature"] == expected["temperature"],
            "logit_bias_matches": actual["logit_bias"] == expected["logit_bias"],
            "threshold_matches": actual["activation_threshold"]
            == expected["activation_threshold"],
            "network_schedule_matches": list(actual["network_update_schedule"])
            == expected["network_update_schedule"],
        }
        rows.append(
            {
                "run_slug": slug,
                "protocol_version": run["protocol_version"],
                "passed": all(checks.values()),
                "checks": checks,
                "configuration": {
                    "temperature": actual["temperature"],
                    "logit_bias": actual["logit_bias"],
                    "activation_threshold": actual["activation_threshold"],
                    "network_update_schedule": actual["network_update_schedule"],
                },
            }
        )
    result = {"passed": all(row["passed"] for row in rows), "runs": rows}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
