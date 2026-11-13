"""Audit a synthetic feasibility run.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a synthetic feasibility run.")
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")

    from app.config import settings
    from app.db import StudyDB
    from app.matched import matched_assignment

    with args.roster.open("r", encoding="utf-8-sig", newline="") as handle:
        roster = [
            row
            for row in csv.DictReader(handle)
            if row["cohort"] in {"study_a", "study_b"}
            and row["analysis_use"] == "formal_when_unlocked"
        ]
    if len(roster) != 72:
        raise SystemExit(f"Expected 72 formal rows, found {len(roster)}")

    db = StudyDB(settings)
    failures: list[dict[str, Any]] = []
    totals = {
        "participants": 0,
        "sessions": 0,
        "pair_intervals": 0,
        "quality_checks": 0,
        "responses": 0,
        "events": 0,
        "simulation_events": 0,
        "chat_turns": 0,
        "matched_assignments": 0,
    }
    for roster_row in roster:
        label = roster_row["assignment_id"]
        participant = db.participant_by_code(roster_row["participant_code"])
        if participant is None:
            failures.append({"label": label, "reason": "participant_missing"})
            continue
        totals["participants"] += 1
        reasons: list[str] = []
        if participant.get("preview_mode") != 1:
            reasons.append("preview_mode_not_set")
        if db.latest_consent(participant["id"]) is None:
            reasons.append("consent_missing")

        sessions = db.sessions(participant["id"])
        totals["sessions"] += len(sessions)
        session_keys = {(int(row["day"]), str(row["client_role"])) for row in sessions}
        expected_session_keys = {
            (day, role) for day in range(1, 4) for role in ("study", "companion")
        }
        if session_keys != expected_session_keys:
            reasons.append("session_coverage_incomplete")
        if any(
            row["ended_ms"] is None or row["state"] not in {"completed", "ended"}
            for row in sessions
        ):
            reasons.append("session_not_ended")

        intervals = db.pair_intervals(participant["id"])
        totals["pair_intervals"] += len(intervals)
        if {int(row["day"]) for row in intervals} != {1, 2, 3}:
            reasons.append("pair_day_coverage_incomplete")
        if any(row["ended_ms"] is None for row in intervals):
            reasons.append("pair_interval_still_open")

        with db.connect() as connection:
            quality = connection.execute(
                "SELECT kind FROM quality_checks WHERE participant_id=?",
                (participant["id"],),
            ).fetchall()
            responses = connection.execute(
                "SELECT instrument FROM responses WHERE participant_id=?",
                (participant["id"],),
            ).fetchall()
            events = connection.execute(
                "SELECT event_type, payload FROM events WHERE participant_id=? ORDER BY seq",
                (participant["id"],),
            ).fetchall()
        totals["quality_checks"] += len(quality)
        totals["responses"] += len(responses)
        totals["events"] += len(events)
        totals["chat_turns"] += sum(row["event_type"] == "chat.turn" for row in events)
        totals["simulation_events"] += sum(
            bool(json.loads(row["payload"]).get("simulation")) for row in events
        )
        if len(quality) < 12:
            reasons.append("quality_checks_incomplete")
        instruments = {str(row["instrument"]) for row in responses}
        if not {"daily_day_1", "daily_day_2", "daily_day_3", "final"}.issubset(instruments):
            reasons.append("questionnaires_incomplete")
        if sum(row["event_type"] == "chat.turn" for row in events) < 6:
            reasons.append("chat_turns_incomplete")
        if not any(json.loads(row["payload"]).get("simulation") for row in events):
            reasons.append("simulation_marker_missing")

        if roster_row["cohort"] == "study_b":
            for day in range(1, 4):
                assignment, checks = matched_assignment(
                    settings, participant["id"], day, participant.get("run_id")
                )
                if assignment is None or not all(checks.values()):
                    reasons.append(f"matched_assignment_day_{day}_invalid")
                else:
                    totals["matched_assignments"] += 1

        if reasons:
            failures.append({"label": label, "reasons": reasons})

    report = {
        "simulation_only": True,
        "participants_expected": 72,
        "participants_complete": 72 - len(failures),
        "participant_days_expected": 216,
        "expected_minimums": {
            "sessions": 432,
            "pair_intervals": 216,
            "quality_checks": 864,
            "responses": 288,
            "chat_turns": 432,
            "matched_assignments": 108,
        },
        "totals": totals,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
