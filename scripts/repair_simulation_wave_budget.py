"""Audit or materialize missing q snapshots for a completed simulation-only Study A wave. This command refuses formal and pilot data.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def sessions_complete(db: Any, participant_id: str) -> bool:
    expected = {(day, role) for day in range(1, 4) for role in ("study", "companion")}
    observed = {
        (int(row["day"]), str(row["client_role"]))
        for row in db.sessions(participant_id)
        if row["state"] in {"completed", "ended"}
    }
    return observed == expected


def source_checkpoints(settings: Any, db: Any, donor: dict[str, Any]) -> tuple[Path, Path, str]:
    shadow = settings.data_dir / "donor-shadow" / donor["id"] / "latest"
    shadow_encoder = shadow / "latest_encoder.pth"
    shadow_main = shadow / "latest_main_model.pth"
    if shadow_encoder.is_file() and shadow_main.is_file():
        return shadow_encoder, shadow_main, "latest_donor_shadow"

    snapshots = sorted(
        (
            row
            for row in db.model_snapshots(donor["run_id"], source="donor_shadow")
            if row["participant_id"] == donor["id"]
        ),
        key=lambda row: int(row["opportunity_budget"]),
        reverse=True,
    )
    for snapshot in snapshots:
        encoder = Path(snapshot["encoder_path"])
        main = Path(snapshot["main_path"])
        if encoder.is_file() and main.is_file():
            return encoder, main, f"existing_q{snapshot['opportunity_budget']}_snapshot"

    return (
        Path(settings.base_encoder_checkpoint),
        Path(settings.base_main_checkpoint),
        "locked_m0",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or materialize missing q snapshots for a completed simulation-only "
            "Study A wave. This command refuses formal and pilot data."
        )
    )
    parser.add_argument("--run-slug", required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialize-synthetic-budget", action="store_true")
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()

    if args.materialize_synthetic_budget and not args.acknowledge_simulation_only:
        raise SystemExit(
            "Writing synthetic q snapshots requires --acknowledge-simulation-only"
        )

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.db import StudyDB
    from app.integrity import file_sha256
    from app.protocol import Protocol

    db = StudyDB(settings)
    protocol = Protocol(settings)
    run = db.study_run(args.run_slug)
    if run is None:
        raise SystemExit(f"Unknown run slug: {args.run_slug}")
    if run["data_origin"] != "simulation" or run["protocol_mode"] != "compressed":
        raise SystemExit("Refusing to repair anything outside simulation/compressed")
    if protocol.study["pipeline"].get("budget_kind") == "eligible_feedback":
        raise SystemExit(
            "v6 eligible-feedback budgets cannot be repaired or imputed. "
            "Collect additional natural tap interactions or report the checkpoint as censored."
        )

    q_levels = sorted(
        int(value) for value in protocol.study["pipeline"]["q_opportunity_levels"]
    )
    donors = sorted(
        (
            row
            for row in db.participants_in_run(run["id"])
            if row["cohort"] == "study_a"
            and row["group_name"] == "donor"
            and int(row.get("enrollment_wave") or 0) == args.wave
        ),
        key=lambda row: str(row.get("assignment_id") or ""),
    )
    if not donors:
        raise SystemExit(f"No Study A donors found in wave {args.wave}")
    incomplete = [row["assignment_id"] for row in donors if not sessions_complete(db, row["id"])]
    if incomplete:
        raise SystemExit(f"Refusing to repair incomplete donors: {incomplete}")

    donor_reports: list[dict[str, Any]] = []
    changed = False
    for donor in donors:
        before_count = db.learning_opportunity_count(donor["id"])
        existing_rows = [
            row
            for row in db.model_snapshots(run["id"], source="donor_shadow")
            if row["participant_id"] == donor["id"]
        ]
        before_snapshots = sorted(int(row["opportunity_budget"]) for row in existing_rows)
        missing = sorted(set(q_levels) - set(before_snapshots))
        source_encoder, source_main, source_kind = source_checkpoints(settings, db, donor)
        created: list[dict[str, Any]] = []

        if args.materialize_synthetic_budget:
            while db.learning_opportunity_count(donor["id"]) < max(q_levels):
                db.next_learning_opportunity(donor["id"])

            day_three = next(
                (
                    row
                    for row in db.sessions(donor["id"])
                    if int(row["day"]) == 3 and row["client_role"] == "study"
                ),
                None,
            )
            if day_three is None:
                raise SystemExit(f"Missing Day 3 study session for {donor['assignment_id']}")
            for budget in missing:
                target = (
                    settings.data_dir
                    / "simulation-budget-repairs"
                    / run["id"]
                    / donor["id"]
                    / f"q-{budget}"
                )
                encoder = target / "encoder.pth"
                main = target / "main_model.pth"
                link_or_copy(source_encoder, encoder)
                link_or_copy(source_main, main)
                snapshot = db.record_model_snapshot(
                    {
                        "participant_id": donor["id"],
                        "session_id": day_three["id"],
                        "source": "donor_shadow",
                        "opportunity_budget": budget,
                        "base_prior_id": donor.get("prior_id"),
                        "encoder_path": str(encoder.resolve()),
                        "main_path": str(main.resolve()),
                        "encoder_sha256": file_sha256(encoder),
                        "main_sha256": file_sha256(main),
                    }
                )
                created.append(
                    {
                        "snapshot_id": snapshot["id"],
                        "budget": budget,
                        "source_kind": source_kind,
                        "encoder_sha256": snapshot["encoder_sha256"],
                        "main_sha256": snapshot["main_sha256"],
                    }
                )
            changed = changed or before_count < max(q_levels) or bool(created)

        after_rows = [
            row
            for row in db.model_snapshots(run["id"], source="donor_shadow")
            if row["participant_id"] == donor["id"]
        ]
        after_snapshots = sorted(int(row["opportunity_budget"]) for row in after_rows)
        after_count = db.learning_opportunity_count(donor["id"])
        donor_reports.append(
            {
                "assignment_id": donor["assignment_id"],
                "participant_id": donor["id"],
                "before": {
                    "learning_opportunities": before_count,
                    "snapshots": before_snapshots,
                },
                "created": created,
                "after": {
                    "learning_opportunities": after_count,
                    "snapshots": after_snapshots,
                    "eligible_for_prior": after_count >= max(q_levels)
                    and set(q_levels).issubset(after_snapshots),
                },
            }
        )

    report = {
        "simulation_only": True,
        "analysis_eligible": False,
        "run_slug": run["slug"],
        "run_id": run["id"],
        "wave": args.wave,
        "mode": "materialize" if args.materialize_synthetic_budget else "audit",
        "scientific_warning": (
            "Synthetic budget repair validates pipeline orchestration only. Repaired counters "
            "and checkpoints must never be used to estimate J(n,q), shared-prior benefit, or "
            "personalization effects."
        ),
        "q_levels": q_levels,
        "donors": donor_reports,
        "ready": all(row["after"]["eligible_for_prior"] for row in donor_reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        os.chmod(args.output, 0o600)
    except OSError:
        pass

    if args.materialize_synthetic_budget and changed:
        db.record_pipeline_action(
            run["id"],
            "simulation.synthetic_budget_repair",
            actor="simulation-repair-script",
            payload={
                "analysis_eligible": False,
                "wave": args.wave,
                "q_levels": q_levels,
                "report_path": str(args.output.resolve()),
                "donor_count": len(donors),
                "repaired_donors": sum(bool(row["created"]) for row in donor_reports),
            },
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
