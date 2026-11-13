"""Disable every account in a QA simulation run.

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
from typing import Any

from dotenv import load_dotenv


RUNNER_SMOKE_RUN_SLUG_PATTERN = re.compile(
    r"^simulation-runner-smoke-v11-[0-9]{13}-[a-f0-9]{8}$"
)


def is_canonical_runner_smoke_run_slug(run_slug: str) -> bool:
    return RUNNER_SMOKE_RUN_SLUG_PATTERN.fullmatch(run_slug) is not None


def disable_runner_smoke_fixture(db: Any, run_slug: str) -> dict[str, Any]:
    """Disable a canonical two- or four-account fixture with safe replays.

    A caller can know the canonical slug before provisioning begins.  Therefore a
    missing run (or a simulation run with no participants yet) is a valid no-op
    during pending-ledger cleanup.  Any partially provisioned or foreign fixture
    remains a hard failure.
    """
    if not is_canonical_runner_smoke_run_slug(run_slug):
        raise ValueError(
            "Refusing to disable a run outside the canonical runner-smoke-v11 namespace"
        )

    run = db.study_run(run_slug)
    if run is None:
        return {
            "run_slug": run_slug,
            "participant_count": 0,
            "enabled_before": 0,
            "disabled_now": 0,
            "enabled_after": 0,
        }
    if run["data_origin"] != "simulation":
        raise ValueError("Only a simulation run can be disabled")

    run_id = str(run["id"])
    participants = db.participants_in_run(run_id)
    participant_count = len(participants)
    if participant_count == 0:
        result = {
            "run_slug": run_slug,
            "participant_count": 0,
            "enabled_before": 0,
            "disabled_now": 0,
            "enabled_after": 0,
        }
    else:
        groups = [str(participant["group_name"]) for participant in participants]
        valid_legacy_pair = (
            participant_count == 2 and groups.count("qa_runner_smoke") == 2
        )
        valid_group_pair = (
            participant_count == 2
            and sorted(groups) == ["frozen", "personalized"]
        )
        valid_four_way = (
            participant_count == 4
            and groups.count("personalized") == 2
            and groups.count("frozen") == 2
        )
        if not (valid_legacy_pair or valid_group_pair or valid_four_way):
            raise ValueError(
                "Runner smoke cleanup requires zero pending participants or exactly "
                "two legacy qa_runner_smoke participants, one personalized/frozen "
                "pair, or the exact two-personalized/two-frozen four-way layout"
            )
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
            raise RuntimeError("Could not verify runner smoke cleanup")
        enabled_after = int(enabled_row["enabled_count"])
        if disabled_now != enabled_before or enabled_after != 0:
            raise RuntimeError(
                "Runner smoke cleanup did not disable every enabled participant"
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
        "simulation.qa_run_disabled",
        actor="runner-smoke-gate",
        payload={
            "participant_count": result["participant_count"],
            "enabled_before": result["enabled_before"],
            "disabled_now": result["disabled_now"],
            "enabled_after": result["enabled_after"],
            "idempotent_replay": result["enabled_before"] == 0,
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Disable every account in a QA simulation run.")
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument("--acknowledge-disable", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_disable:
        raise SystemExit("Pass --acknowledge-disable")
    if not is_canonical_runner_smoke_run_slug(args.run_slug):
        raise SystemExit(
            "Refusing to disable a run outside the canonical runner-smoke-v11 namespace"
        )

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB

    actual_data_dir = settings.data_dir.resolve()
    expected_data_dir = args.expected_data_dir.resolve()
    if actual_data_dir != expected_data_dir:
        raise SystemExit(
            f"Configured data directory mismatch: actual={actual_data_dir} expected={expected_data_dir}"
        )
    db = StudyDB(settings)
    try:
        result = disable_runner_smoke_fixture(db, args.run_slug)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    result["data_dir"] = str(actual_data_dir)
    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
