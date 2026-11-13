"""Readiness guards for pipeline-managed participant records.

The predicates distinguish ordinary pilot records from locked study-pipeline
assignments and expose the checks required before a managed session starts.
"""

from __future__ import annotations

from typing import Any


def is_pipeline_managed(participant: dict[str, Any]) -> bool:
    return (
        participant.get("cohort") in {"study_a", "study_b"}
        and bool(participant.get("assignment_id"))
        and participant.get("enrollment_wave") is not None
    )


def requires_locked_pipeline(participant: dict[str, Any]) -> bool:
    # Public Study A donors start from the locked cold model and therefore do
    # not need a pre-assigned shared prior. Pipeline-managed Study A accounts
    # and every Study B account still require the full prior/release guard.
    return is_pipeline_managed(participant) or (
        participant.get("data_origin") == "formal"
        and participant.get("cohort") == "study_b"
    )


PIPELINE_READINESS_KEYS = {
    "protocol_snapshot_match",
    "participant_released",
    "prior_assigned",
    "prior_record_present",
    "prior_same_run",
    "prior_coordinates_match",
    "prior_files_present",
    "prior_hashes_current",
    "candidate_probe_assignments_ready",
    "study_b_prior_selected",
}


def pipeline_readiness(checks: dict[str, bool]) -> dict[str, bool]:
    return {
        name: passed
        for name, passed in checks.items()
        if name in PIPELINE_READINESS_KEYS or name.startswith("matched_day")
    }
