"""Paths and assignments for version-locked matched model conditions.

Manifest fields connect experimental assignments to their encoder and decision
head artifacts. These helpers read configuration and do not train models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings
from .integrity import file_sha256


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def matched_paths(settings: Settings, run_id: str | None = None) -> tuple[Path, Path]:
    if run_id:
        root = settings.data_dir / "matched" / run_id
        return root / "manifest.json", root / "self-test.json"
    return Path(settings.matched_model_manifest), Path(settings.matched_model_ready_sentinel)


def matched_assignment(
    settings: Settings,
    participant_id: str,
    day: int,
    run_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, bool]]:
    manifest_path, sentinel_path = matched_paths(settings, run_id)
    manifest = load_json(manifest_path)
    sentinel = load_json(sentinel_path)
    checks = {
        "manifest_present": manifest is not None,
        "manifest_same_run": run_id is None,
        "manifest_approved": False,
        "assignment_rule_frozen": False,
        "participant_day_assignment_present": False,
        "donor_is_other_participant": False,
        "checkpoint_files_present": False,
        "checkpoint_hashes_declared": False,
        "checkpoint_hashes_current": False,
        "manifest_self_test_passed": False,
        "assignment_self_test_passed": False,
    }
    if manifest is None:
        return None, checks

    checks["manifest_approved"] = manifest.get("approved") is True
    checks["manifest_same_run"] = run_id is None or manifest.get("run_id") == run_id
    if not checks["manifest_same_run"]:
        return None, checks
    checks["assignment_rule_frozen"] = bool(manifest.get("assignment_rule")) and int(
        manifest.get("frozen_at_ms") or 0
    ) > 0
    assignments = manifest.get("assignments")
    participant_assignments = (
        assignments.get(participant_id) if isinstance(assignments, dict) else None
    )
    assignment = (
        participant_assignments.get(str(day))
        if isinstance(participant_assignments, dict)
        else None
    )
    checks["participant_day_assignment_present"] = isinstance(assignment, dict)
    if not isinstance(assignment, dict):
        return None, checks

    donor_id = str(assignment.get("donor_participant_id") or "")
    checks["donor_is_other_participant"] = bool(donor_id) and donor_id != participant_id
    encoder = Path(str(assignment.get("encoder_checkpoint") or ""))
    main = Path(str(assignment.get("main_checkpoint") or ""))
    checks["checkpoint_files_present"] = (
        encoder.is_absolute() and main.is_absolute() and encoder.is_file() and main.is_file()
    )
    encoder_hash = str(assignment.get("encoder_sha256") or "").lower()
    main_hash = str(assignment.get("main_sha256") or "").lower()
    checks["checkpoint_hashes_declared"] = len(encoder_hash) == 64 and len(main_hash) == 64
    if checks["checkpoint_files_present"] and checks["checkpoint_hashes_declared"]:
        checks["checkpoint_hashes_current"] = (
            file_sha256(encoder) == encoder_hash and file_sha256(main) == main_hash
        )

    if sentinel is not None and manifest_path.is_file():
        checks["manifest_self_test_passed"] = (
            sentinel.get("passed") is True
            and sentinel.get("manifest_sha256") == file_sha256(manifest_path)
        )
        tested = sentinel.get("assignments", {}).get(participant_id, {}).get(str(day), {})
        checks["assignment_self_test_passed"] = (
            isinstance(tested, dict)
            and tested.get("encoder_sha256") == encoder_hash
            and tested.get("main_sha256") == main_hash
            and tested.get("donor_participant_id") == donor_id
        )

    if not all(checks.values()):
        return None, checks
    return {
        **assignment,
        "encoder_checkpoint": str(encoder),
        "main_checkpoint": str(main),
    }, checks
