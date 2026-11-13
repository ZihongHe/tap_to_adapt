"""Run iFLYTEK-voiced personalized/frozen public QA and always disable the one-shot fixture. Credentials are read only from process env.

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
from typing import Any, Callable

try:
    from .disable_simulation_run import (
        disable_runner_smoke_fixture,
        is_canonical_runner_smoke_run_slug,
    )
    from .smoke_negative_feedback_public import run as run_negative
    from .smoke_positive_feedback_public import run as run_positive
    from .voice_pcm import synthesize_xunfei_pcm
except ImportError:  # Direct execution from the scripts directory.
    from disable_simulation_run import (
        disable_runner_smoke_fixture,
        is_canonical_runner_smoke_run_slug,
    )
    from smoke_negative_feedback_public import run as run_negative
    from smoke_positive_feedback_public import run as run_positive
    from voice_pcm import synthesize_xunfei_pcm


DEFAULT_UTTERANCE = (
    "我刚才想到一个相关的例子，想补充说明一下自己的看法。"
)

PAIR_CELLS = {
    "positive": ("personalized", "quick_timing"),
    "negative": ("frozen", "long_quarantine"),
}
FOUR_WAY_CELLS = {
    "personalized_positive": ("personalized", "quick_timing"),
    "personalized_negative": ("personalized", "long_quarantine"),
    "frozen_positive": ("frozen", "quick_timing"),
    "frozen_negative": ("frozen", "long_quarantine"),
}


def fixture_cells(rows: dict[str, dict[str, str]]) -> tuple[str, dict[str, tuple[str, str]]]:
    purposes = set(rows)
    if purposes == set(PAIR_CELLS):
        return "pair", PAIR_CELLS
    if purposes == set(FOUR_WAY_CELLS):
        return "four-way", FOUR_WAY_CELLS
    raise ValueError(
        "Runner fixture must be either the legacy positive/negative pair or "
        "the exact personalized/frozen x positive/negative four-way design"
    )


def load_private_fixture(path: Path) -> tuple[str, dict[str, dict[str, str]]]:
    if not path.is_file():
        raise ValueError("Runner fixture CSV does not exist")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) not in {2, 4}:
        raise ValueError("Voiced runner requires exactly two or four one-shot accounts")

    run_slugs = {str(row.get("run_slug", "")) for row in rows}
    if len(run_slugs) != 1:
        raise ValueError("Runner fixture rows must share exactly one run slug")
    run_slug = next(iter(run_slugs))
    if not is_canonical_runner_smoke_run_slug(run_slug):
        raise ValueError("Runner fixture is outside the canonical smoke-v11 namespace")

    by_purpose: dict[str, dict[str, str]] = {}
    for row in rows:
        purpose = str(row.get("purpose", ""))
        if purpose in by_purpose:
            raise ValueError("Runner fixture purposes must be unique")
        by_purpose[purpose] = {str(key): str(value) for key, value in row.items()}
    design, expected = fixture_cells(by_purpose)
    for purpose, (group, scenario) in expected.items():
        row = by_purpose[purpose]
        if row.get("internal_group") != group:
            raise ValueError(f"{purpose} fixture must use the {group} group")
        if row.get("smoke_scenario") != scenario:
            raise ValueError(f"{purpose} fixture has the wrong smoke scenario")
        if row.get("analysis_use") != "simulation_runner_smoke_v11":
            raise ValueError("Runner fixture is not marked simulation-only")
        declared_design = row.get("fixture_design", "")
        if declared_design not in {"", design}:
            raise ValueError(f"{purpose} fixture declares the wrong design")
        if design == "four-way" and declared_design != "four-way":
            raise ValueError("Four-way fixtures must explicitly declare fixture_design")
        if not row.get("participant_code"):
            raise ValueError(f"{purpose} fixture has no participant code")
        if not row.get("participant_id"):
            raise ValueError(f"{purpose} fixture has no participant UUID")
        if design == "four-way" and not row.get("assignment_id"):
            raise ValueError(f"{purpose} four-way fixture has no assignment id")
    if len({row["participant_code"] for row in by_purpose.values()}) != len(rows):
        raise ValueError("Runner fixture participant codes must be unique")
    if len({row["participant_id"] for row in by_purpose.values()}) != len(rows):
        raise ValueError("Runner fixture participant UUIDs must be unique")
    return run_slug, by_purpose


def _local_fixture_db(run_slug: str, expected_data_dir: Path) -> tuple[Any, dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from app.config import Settings
    from app.db import StudyDB

    settings = Settings.from_env()
    actual_data_dir = settings.data_dir.resolve()
    required_data_dir = expected_data_dir.resolve()
    if actual_data_dir != required_data_dir:
        raise RuntimeError(
            "Configured data directory does not match --expected-data-dir"
        )
    if not (actual_data_dir / "study.sqlite3").is_file():
        raise RuntimeError(
            "Study database is unavailable; run this lifecycle gate on the application host"
        )
    db = StudyDB(settings)
    run = db.study_run(run_slug)
    if run is None:
        raise RuntimeError("Runner fixture run is absent from the local study database")
    if run["data_origin"] != "simulation":
        raise RuntimeError("Runner fixture is not simulation-only")
    return db, run


def validate_local_fixture_lifecycle(
    run_slug: str,
    expected_data_dir: Path,
    rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Fail before login unless this process can later disable the exact fixture."""
    db, run = _local_fixture_db(run_slug, expected_data_dir)
    participants = db.participants_in_run(str(run["id"]))
    design, expected_cells = fixture_cells(rows)
    groups = sorted(str(participant["group_name"]) for participant in participants)
    expected_groups = sorted(group for group, _scenario in expected_cells.values())
    if len(participants) != len(rows) or groups != expected_groups:
        raise RuntimeError(f"Local runner fixture is not the exact {design} group layout")
    if any(int(participant["enabled"]) != 1 for participant in participants):
        raise RuntimeError("Local runner fixture is not fresh and fully enabled")
    database_ids = {str(participant["id"]) for participant in participants}
    fixture_ids = {row["participant_id"] for row in rows.values()}
    if database_ids != fixture_ids:
        raise RuntimeError("Fixture CSV participant UUIDs do not match the local run")
    participants_by_id = {
        str(participant["id"]): participant for participant in participants
    }
    for purpose, row in rows.items():
        participant = participants_by_id[row["participant_id"]]
        expected_group = expected_cells[purpose][0]
        if row["internal_group"] != expected_group:
            raise RuntimeError(f"Fixture purpose/group mapping is invalid for {purpose}")
        if str(participant["group_name"]) != expected_group:
            raise RuntimeError(
                f"Fixture participant UUID maps to the wrong database group for {purpose}"
            )
        if str(participant.get("run_slug", run_slug)) != run_slug:
            raise RuntimeError(f"Fixture participant belongs to the wrong run for {purpose}")
        assignment_id = row.get("assignment_id", "")
        if assignment_id and str(participant.get("assignment_id", "")) != assignment_id:
            raise RuntimeError(f"Fixture assignment id mismatch for {purpose}")
    with db.connect() as connection:
        session_row = connection.execute(
            "SELECT COUNT(*) AS session_count FROM sessions WHERE run_id=?",
            (str(run["id"]),),
        ).fetchone()
    if session_row is None or int(session_row["session_count"]) != 0:
        raise RuntimeError("Local runner fixture is not fresh; sessions already exist")
    return {
        "participant_count": len(participants),
        "fixture_design": design,
        "groups": groups,
        "purposes": sorted(rows),
        "session_count": 0,
    }


def disable_local_fixture(run_slug: str, expected_data_dir: Path) -> dict[str, Any]:
    """Disable the fixture through the application host's own database."""
    db, _ = _local_fixture_db(run_slug, expected_data_dir)
    return disable_runner_smoke_fixture(db, run_slug)


def validate_scenario_result(
    purpose: str,
    group: str,
    result: dict[str, Any],
) -> None:
    """Assert the group policy independently of each scenario implementation."""
    mismatches: dict[str, dict[str, Any]] = {}

    def expect(field: str, expected: Any) -> None:
        actual = result.get(field)
        if actual != expected:
            mismatches[field] = {"expected": expected, "actual": actual}

    expect("ok", True)
    personalized = group == "personalized"
    expect("training_eligible", personalized)
    expect("training_applied", personalized)
    expect("personal_calibration_enabled", personalized)
    if personalized:
        expect("personal_calibration_applied", True)
    else:
        before = float(result.get("personal_logit_bias_before", 0.0))
        after = float(result.get("personal_logit_bias_after", before))
        if after != before:
            mismatches["personal_logit_bias"] = {
                "expected": "unchanged",
                "actual": [before, after],
            }
        if result.get("current_network_update_applied") not in {None, False}:
            mismatches["current_network_update_applied"] = {
                "expected": "absent or false",
                "actual": result.get("current_network_update_applied"),
            }

    if purpose.endswith("positive") or purpose == "positive":
        expect("stale_manual_action", "conditional_tap_cancelled")
    else:
        expect("stale_interrupt_action", "conditional_tap_cancelled")
    if mismatches:
        raise RuntimeError(f"Voiced {purpose} policy assertion failed: {mismatches}")


def execute_voiced_fixture(
    base_url: str,
    run_slug: str,
    rows: dict[str, dict[str, str]],
    expected_data_dir: Path,
    text: str,
    *,
    synthesizer: Callable[[str], tuple[bytes, dict[str, Any]]] = synthesize_xunfei_pcm,
    positive_runner: Callable[..., dict[str, Any]] = run_positive,
    negative_runner: Callable[..., dict[str, Any]] = run_negative,
    lifecycle_preflight: Callable[
        [str, Path, dict[str, dict[str, str]]], dict[str, Any]
    ] = (
        validate_local_fixture_lifecycle
    ),
    disabler: Callable[[str, Path], dict[str, Any]] = disable_local_fixture,
) -> dict[str, Any]:
    """Run every declared cell and disable all accounts on every exit path."""
    design, expected_cells = fixture_cells(rows)
    pending_error: BaseException | None = None
    lifecycle: dict[str, Any] | None = None
    audio_metadata: dict[str, Any] | None = None
    scenario_results: dict[str, dict[str, Any]] = {}
    cleanup_result: dict[str, Any] | None = None
    try:
        # This fail-closed check occurs before any login or provider call, but
        # still lives inside the cleanup envelope: a stale/consumed otherwise
        # valid one-shot fixture must be disabled, not left reusable.
        lifecycle = lifecycle_preflight(run_slug, expected_data_dir, rows)
        pcm, audio_metadata = synthesizer(text)
        frame_bytes_max = int(audio_metadata.get("frame_bytes_max", 0))
        if frame_bytes_max <= 0 or frame_bytes_max > 6_400:
            raise RuntimeError("Synthesized PCM exceeds the public timing frame limit")
        for purpose, row in rows.items():
            group = expected_cells[purpose][0]
            expect_training = group == "personalized"
            if purpose.endswith("positive") or purpose == "positive":
                result = positive_runner(
                    base_url,
                    row["participant_code"],
                    expect_training=expect_training,
                    voiced_pcm=pcm,
                    pace_voiced_pcm=True,
                )
            else:
                result = negative_runner(
                    base_url,
                    row["participant_code"],
                    "thinking",
                    expect_training=expect_training,
                    voiced_pcm=pcm,
                    pace_voiced_pcm=True,
                )
            validate_scenario_result(purpose, group, result)
            scenario_results[purpose] = result
    except BaseException as error:
        pending_error = error
        raise
    finally:
        try:
            cleanup_result = disabler(run_slug, expected_data_dir)
        except BaseException as cleanup_error:
            if pending_error is not None:
                pending_error.add_note(
                    "Runner account cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise

    assert lifecycle is not None
    assert audio_metadata is not None
    assert len(scenario_results) == len(rows)
    assert cleanup_result is not None
    if int(cleanup_result.get("participant_count", -1)) != len(rows):
        raise RuntimeError("Fixture cleanup covered the wrong participant count")
    if int(cleanup_result.get("enabled_after", -1)) != 0:
        raise RuntimeError("Fixture cleanup left one or more accounts enabled")
    compatibility_results: dict[str, dict[str, Any]] = {}
    if design == "pair":
        compatibility_results = {
            "personalized_positive": scenario_results["positive"],
            "frozen_negative": scenario_results["negative"],
        }
    return {
        "ok": True,
        "simulation_only": True,
        "run_slug": run_slug,
        "fixture_design": design,
        "lifecycle_preflight": lifecycle,
        "audio": audio_metadata,
        **scenario_results,
        **compatibility_results,
        "fixture_cleanup": cleanup_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run iFLYTEK-voiced personalized/frozen public QA and always disable "
            "the one-shot fixture. Credentials are read only from process env."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--expected-data-dir", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_UTTERANCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acknowledge-one-shot-fixture", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_one_shot_fixture:
        parser.error(
            "Pass --acknowledge-one-shot-fixture to confirm the simulation accounts "
            "will be disabled in finally"
        )

    run_slug, rows = load_private_fixture(args.code_file)
    result = execute_voiced_fixture(
        args.base_url.rstrip("/"),
        run_slug,
        rows,
        args.expected_data_dir,
        args.text,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
