"""Create fresh one-shot simulation accounts for the browser runner hard gate.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv


FIELDNAMES = [
    "assignment_id",
    "participant_code",
    "participant_id",
    "cohort",
    "phase",
    "internal_group",
    "analysis_use",
    "behavior_profile",
    "smoke_scenario",
    "purpose",
    "fixture_design",
    "run_slug",
]

RUN_SLUG_PATTERN = re.compile(
    r"^simulation-runner-smoke-v11-(?P<created_at_ms>[0-9]{13})-(?P<fixture_id>[a-f0-9]{8})$"
)


def validate_runner_smoke_run_slug(run_slug: str) -> str:
    """Validate a caller-known, one-shot namespace and return its fixture id."""
    match = RUN_SLUG_PATTERN.fullmatch(run_slug)
    if match is None:
        raise ValueError(
            "run slug must match simulation-runner-smoke-v11-<13digitms>-<8hex>"
        )
    return match.group("fixture_id").upper()


def validate_private_output_path(output: Path, public_roots: list[Path]) -> None:
    """Reject credential files that could be served by the application."""
    resolved_output = output.resolve()
    for public_root in public_roots:
        resolved_root = public_root.resolve()
        if resolved_output == resolved_root or resolved_root in resolved_output.parents:
            raise ValueError(
                f"Credential output must not be inside a public directory: {resolved_root}"
            )


def create_private_placeholder(path: Path) -> None:
    """Atomically reserve a credential path with owner-only POSIX permissions."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def runner_smoke_specs(design: str) -> list[tuple[str, str, str, str]]:
    """Return the exact group/feedback cells for a one-shot fixture.

    ``pair`` is the historical two-account hard gate.  ``four-way`` removes
    the historical confound between participant group and feedback sign by
    crossing both dimensions.  Keeping the legacy design explicit avoids
    silently changing existing provisioning callers.
    """
    if design == "pair":
        return [
            ("personalized", "positive", "quick_timing", "impatient"),
            ("frozen", "negative", "long_quarantine", "ordinary"),
        ]
    if design == "four-way":
        return [
            (
                "personalized",
                "personalized_positive",
                "quick_timing",
                "impatient",
            ),
            (
                "personalized",
                "personalized_negative",
                "long_quarantine",
                "ordinary",
            ),
            ("frozen", "frozen_positive", "quick_timing", "impatient"),
            ("frozen", "frozen_negative", "long_quarantine", "ordinary"),
        ]
    if design == "network-update":
        return [
            (
                "personalized",
                "network_update_personalized",
                "three_feedback_network_update",
                "controlled",
            ),
            (
                "frozen",
                "network_update_frozen",
                "three_feedback_network_control",
                "controlled",
            ),
        ]
    raise ValueError(
        "runner smoke design must be 'pair', 'four-way', or 'network-update'"
    )


def runner_smoke_analysis_use(design: str) -> str:
    if design == "network-update":
        return "simulation_network_update_smoke_v11"
    if design in {"pair", "four-way"}:
        return "simulation_runner_smoke_v11"
    raise ValueError(
        "runner smoke design must be 'pair', 'four-way', or 'network-update'"
    )


def build_rows(
    accounts: list[tuple[str, str, str]],
    run_slug: str,
    design: str = "pair",
) -> list[dict[str, str]]:
    specs = runner_smoke_specs(design)
    analysis_use = runner_smoke_analysis_use(design)
    if len(accounts) != len(specs):
        raise ValueError(
            f"runner smoke {design} design requires exactly {len(specs)} fresh accounts"
        )
    rows: list[dict[str, str]] = []
    for (code, participant_id, assignment_id), spec in zip(
        accounts, specs, strict=True
    ):
        group_name, purpose, smoke_scenario, behavior_profile = spec
        rows.append(
            {
                "assignment_id": assignment_id,
                "participant_code": code,
                "participant_id": participant_id,
                "cohort": "SMOKE",
                "phase": run_slug,
                "internal_group": group_name,
                "analysis_use": analysis_use,
                "behavior_profile": behavior_profile,
                "smoke_scenario": smoke_scenario,
                "purpose": purpose,
                "fixture_design": design,
                "run_slug": run_slug,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fresh one-shot simulation accounts for the browser runner hard gate."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-slug",
        required=True,
        help="Caller-known unique simulation-runner-smoke-v11 namespace.",
    )
    parser.add_argument(
        "--expected-data-dir",
        type=Path,
        required=True,
        help="Fail closed unless this resolves to the configured study data directory.",
    )
    parser.add_argument(
        "--design",
        choices=("pair", "four-way", "network-update"),
        default="pair",
        help=(
            "pair preserves the historical two-account gate; four-way crosses "
            "personalized/frozen with positive/negative feedback; network-update "
            "creates a separate three-feedback personalized/control pair"
        ),
    )
    parser.add_argument("--acknowledge-simulation-only", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_simulation_only:
        raise SystemExit("Pass --acknowledge-simulation-only")
    try:
        fixture_id = validate_runner_smoke_run_slug(args.run_slug)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    run_slug = args.run_slug

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

    output = args.output.resolve()
    try:
        validate_private_output_path(output, [settings.static_dir])
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing credential file: {output}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fixture_uuid = uuid.uuid4().hex
    temporary_output = output.with_name(f".{output.name}.{fixture_uuid}.tmp")

    db = StudyDB(settings)
    if db.study_run(run_slug) is not None:
        raise SystemExit(f"Refusing to reuse existing runner smoke run: {run_slug}")
    create_private_placeholder(temporary_output)
    run_id: str | None = None
    try:
        # The browser hard gate must exercise both longitudinal branches.  The
        # former fixture put both accounts in qa_runner_smoke, which made both
        # update-enabled and could never prove that Frozen remains unchanged.
        specs = runner_smoke_specs(args.design)
        codes = []
        for group_name, _purpose, _scenario, _profile in specs:
            codes.extend(
                db.create_participants(
                    1,
                    "pilot",
                    group_name,
                    preview_mode=True,
                    data_origin="simulation",
                    protocol_mode="compressed",
                    run_slug=run_slug,
                )
            )
            # Capture the run id after the first insert so a later partial
            # provisioning failure still disables every already-created code.
            if run_id is None:
                partial_run = db.study_run(run_slug)
                if partial_run is None:
                    raise RuntimeError("smoke run was not created")
                run_id = str(partial_run["id"])
        run = db.study_run(run_slug)
        if run is None:
            raise RuntimeError("smoke run was not created")
        if str(run["id"]) != run_id:
            raise RuntimeError("smoke run identity changed during provisioning")
        participant_rows = [db.participant_by_code(code) for code in codes]
        if any(participant is None for participant in participant_rows):
            raise RuntimeError("created smoke participant could not be resolved")

        # Keep credentials unusable until the private CSV has been atomically published.
        with db.connect() as connection:
            connection.execute("UPDATE participants SET enabled=0 WHERE run_id=?", (run_id,))

        accounts: list[tuple[str, str, str]] = []
        for index, (code, participant) in enumerate(
            zip(codes, participant_rows, strict=True), start=1
        ):
            assert participant is not None
            assignment_id = f"QSMK{fixture_id}{index}"
            db.configure_participant(
                str(participant["id"]),
                assignment_id=assignment_id,
                neutral_id=assignment_id,
                enrollment_wave=1,
                release_state="released",
            )
            accounts.append((code, str(participant["id"]), assignment_id))
        rows = build_rows(accounts, run_slug, args.design)

        with temporary_output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            temporary_output.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary_output, output)
        try:
            output.chmod(0o600)
        except OSError:
            pass

        with db.connect() as connection:
            cursor = connection.execute(
                "UPDATE participants SET enabled=1 WHERE run_id=? AND enabled=0",
                (run_id,),
            )
            if int(cursor.rowcount) != len(specs):
                raise RuntimeError(
                    f"failed to enable exactly {len(specs)} smoke accounts"
                )
        db.record_pipeline_action(
            run_id,
            "simulation.runner_smoke_provisioned",
            actor="runner-smoke-gate",
            payload={
                "fixture_id": fixture_id,
                "account_count": len(specs),
                "credential_path": str(output),
                "cleanup_required": True,
                "fixture_design": args.design,
                "groups": [spec[0] for spec in specs],
                "purposes": [spec[1] for spec in specs],
            },
        )
    except BaseException:
        cleanup_run_id = run_id
        if cleanup_run_id is None:
            partial_run = db.study_run(run_slug)
            if partial_run is not None and partial_run["data_origin"] == "simulation":
                cleanup_run_id = str(partial_run["id"])
        if cleanup_run_id is not None:
            with db.connect() as connection:
                connection.execute(
                    "UPDATE participants SET enabled=0 WHERE run_id=?",
                    (cleanup_run_id,),
                )
        temporary_output.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "simulation_only": True,
                "fixture_policy": "single-gate-use; disable run in finally",
                "cleanup_required": True,
                "run_slug": run_slug,
                "protocol_version": db.protocol_version,
                "data_dir": str(actual_data_dir),
                "output": str(output),
                "account_count": len(rows),
                "assignments": [row["assignment_id"] for row in rows],
                "fixture_design": args.design,
                "groups": [row["internal_group"] for row in rows],
                "purposes": [row["purpose"] for row in rows],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
