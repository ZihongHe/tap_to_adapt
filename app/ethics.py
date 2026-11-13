"""Consent-status interpretation and recruitment-readiness predicates.

These helpers implement the study application's configured status policy.
Template status fields describe software configuration, not an authorization
to recruit participants; an independent deployment needs its own approvals.
"""

from __future__ import annotations

from typing import Any

ETHICS_STATUSES = frozenset({"pending", "approved", "rejected"})
FORMAL_READINESS_ADVISORY_KEYS = frozenset(
    {
        "consent_ethics_finalized",
        "compensation_ethics_finalized",
    }
)


def normalize_ethics_status(document: dict[str, Any]) -> str:
    raw = str(document.get("ethics_status") or "").strip().lower()
    if raw in ETHICS_STATUSES:
        return raw
    return "approved" if document.get("approved") else "pending"


def ethics_blocks_recruitment(document: dict[str, Any]) -> bool:
    return normalize_ethics_status(document) == "rejected"


def consent_recruitment_ready(
    document: dict[str, Any],
    *,
    consent_version: str,
    approved_consent_version: str,
) -> bool:
    status = normalize_ethics_status(document)
    if status == "rejected":
        return False
    if status == "pending":
        return True
    return (
        bool(document.get("approved"))
        and bool(approved_consent_version)
        and approved_consent_version == consent_version
    )


def consent_ethics_finalized(
    document: dict[str, Any],
    *,
    consent_version: str,
    approved_consent_version: str,
) -> bool:
    status = normalize_ethics_status(document)
    if status != "approved":
        return False
    return (
        bool(document.get("approved"))
        and bool(approved_consent_version)
        and approved_consent_version == consent_version
    )


def compensation_recruitment_ready(document: dict[str, Any]) -> bool:
    status = normalize_ethics_status(document)
    if status == "rejected":
        return False
    if status == "pending":
        return True
    return bool(document.get("approved"))


def compensation_ethics_finalized(document: dict[str, Any]) -> bool:
    return normalize_ethics_status(document) == "approved" and bool(document.get("approved"))


def should_show_document_status_notice(document: dict[str, Any]) -> bool:
    status = normalize_ethics_status(document)
    if status in {"pending", "approved"}:
        return False
    if status == "rejected":
        return True
    return not bool(document.get("approved"))


def formal_readiness_is_ready(checks: dict[str, bool]) -> bool:
    return all(
        passed
        for key, passed in checks.items()
        if key not in FORMAL_READINESS_ADVISORY_KEYS
    )
