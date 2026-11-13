"""Consent-aware export of research records into controlled release tiers.

The exporter checks run eligibility, optional consent, privacy-review decisions,
and requested data tier before writing pseudonymous records. It separates
controlled artifacts from public derived data and excludes compensation data.
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SAFE_EVENT_FIELDS = (
    "action",
    "accepted",
    "label",
    "training_eligible",
    "training_applied",
    "quarantined",
    "quarantine_reason",
    "feedback_attribution",
    "phase",
    "policy",
    "policy_effective",
    "probability",
    "raw_probability",
    "activation_threshold",
    "silence_ms",
    "decision_eligible",
    "eligible",
    "latency_ms",
    "completed_seconds",
    "planned_seconds",
    "elapsed_seconds",
    "active_block_index",
    "analysis_eligible",
    "scheduled_rest",
    "window",
    "block",
)
TEXT_FIELDS = ("text", "transcript", "reply", "message", "answer")
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
    "url": re.compile(r"\b(?:https?://|www\.)\S+", re.I),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "long_number": re.compile(r"(?<!\d)\d{8,}(?!\d)"),
}


PUBLIC_EVENT_COLUMNS = (
    "participant_id",
    "session_id",
    "split",
    "cohort",
    "condition",
    "day",
    "seq",
    "event_type",
    "relative_ms",
    *SAFE_EVENT_FIELDS,
)
REST_COLUMNS = (
    "participant_id",
    "session_id",
    "day",
    "event_type",
    "relative_ms",
    "block",
    "completed_seconds",
    "planned_seconds",
    "analysis_eligible",
)
AUDIO_COLUMNS = (
    "audio_id",
    "participant_id",
    "session_id",
    "day",
    "kind",
    "content_type",
    "bytes",
    "sha256",
    "review_status",
    "exported_relative_path",
    "ai_training_allowed",
    "commercial_research_allowed",
)
REVIEW_COLUMNS = (
    "review_id",
    "participant_id",
    "session_id",
    "source_type",
    "source_id",
    "field",
    "detected_risks",
    "redacted_text",
    "decision",
    "reviewer",
    "reviewed_at",
)


@dataclass(frozen=True)
class ReleaseResult:
    output_dir: Path
    manifest_path: Path
    public_participants: int
    controlled_participants: int
    controlled_audio_files: int


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }


def _assert_run_release_eligible(
    connection: sqlite3.Connection, run_id: str
) -> None:
    participant_columns = _columns(connection, "participants")
    if "analysis_excluded" in participant_columns:
        excluded = connection.execute(
            """SELECT COUNT(*) AS n FROM participants
               WHERE run_id=? AND analysis_excluded=1""",
            (run_id,),
        ).fetchone()
        if excluded is not None and int(excluded["n"]) > 0:
            raise ValueError(
                "personal-debug runs are permanently excluded from dataset release"
            )
    batch_columns = _columns(connection, "account_batches")
    if batch_columns:
        clauses = []
        if "stage" in batch_columns:
            clauses.append("stage='personal_debug'")
        if "analysis_excluded" in batch_columns:
            clauses.append("analysis_excluded=1")
        if clauses:
            excluded_batch = connection.execute(
                f"""SELECT 1 FROM account_batches
                    WHERE run_id=? AND ({' OR '.join(clauses)}) LIMIT 1""",
                (run_id,),
            ).fetchone()
            if excluded_batch is not None:
                raise ValueError(
                    "personal-debug runs are permanently excluded from dataset release"
                )


def _row_bool(row: sqlite3.Row | dict[str, Any], key: str) -> bool:
    keys = row.keys()
    return bool(row[key]) if key in keys else False


def _pseudonym(pepper: str, namespace: str, value: str, length: int = 16) -> str:
    digest = hmac.new(
        pepper.encode("utf-8"),
        f"{namespace}:{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{namespace[:1].upper()}-{digest[:length]}"


def _split(pepper: str, participant_id: str) -> str:
    digest = hmac.new(
        pepper.encode("utf-8"),
        f"split:{participant_id}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "train" if bucket < 70 else ("validation" if bucket < 85 else "test")


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _redact(text: str) -> tuple[str, list[str]]:
    redacted = text
    risks: list[str] = []
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(redacted):
            risks.append(name)
            redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    # Names, uncommon places, voices, and third-party speech cannot be reliably
    # de-identified by regex. Every non-empty text/audio item remains in review.
    if text.strip():
        risks.append("human_semantic_and_identity_review_required")
    return redacted, risks


def _latest_consents(connection: sqlite3.Connection, run_id: str) -> dict[str, sqlite3.Row]:
    rows = connection.execute(
        """
        SELECT c.*
        FROM consents c
        JOIN (
          SELECT participant_id, MAX(signed_ms) AS signed_ms
          FROM consents
          WHERE run_id=?
          GROUP BY participant_id
        ) latest
        ON latest.participant_id=c.participant_id AND latest.signed_ms=c.signed_ms
        WHERE c.run_id=?
        """,
        (run_id, run_id),
    ).fetchall()
    return {str(row["participant_id"]): row for row in rows}


def _review_decisions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    decisions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audio_id = str(row.get("audio_id") or "").strip()
            status = str(row.get("status") or "").strip().lower()
            if audio_id and status in {"approved", "excluded"}:
                decisions[audio_id] = status
    return decisions


def export_research_dataset(
    *,
    study_db: Path,
    consent_db: Path,
    media_dir: Path,
    output_dir: Path,
    run_slug: str,
    export_pepper: str,
    include_controlled_audio: bool = False,
    audio_review_csv: Path | None = None,
) -> ReleaseResult:
    """Build a fail-closed, consent-aware research release.

    Public files never include text, absolute timestamps, raw audio, recruitment
    identifiers, IP/user-agent data, or participant-specific model weights.
    Controlled audio is copied only after both consent and per-file human review.
    """

    if len(export_pepper) < 16:
        raise ValueError("export_pepper must contain at least 16 characters")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output_dir must be absent or empty")

    study = _connect(study_db)
    try:
        run = study.execute(
            "SELECT * FROM study_runs WHERE slug=?", (run_slug,)
        ).fetchone()
        if run is None:
            raise ValueError(f"unknown run_slug: {run_slug}")
        run_id = str(run["id"])
        _assert_run_release_eligible(study, run_id)
    except Exception:
        study.close()
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = output_dir / "public"
    controlled_dir = output_dir / "controlled"
    restricted_dir = output_dir / "_restricted_review"
    public_dir.mkdir()
    controlled_dir.mkdir()
    restricted_dir.mkdir()

    consent = _connect(consent_db)
    try:
        participants = {
            str(row["id"]): row
            for row in study.execute(
                "SELECT * FROM participants WHERE run_id=? ORDER BY created_ms", (run_id,)
            ).fetchall()
        }
        consents = _latest_consents(consent, run_id)
        public_ids = {
            participant_id
            for participant_id, row in consents.items()
            if _row_bool(row, "participation") and _row_bool(row, "public_derived")
        }
        controlled_ids = {
            participant_id
            for participant_id, row in consents.items()
            if _row_bool(row, "participation") and _row_bool(row, "controlled_sharing")
        }
        sessions = {
            str(row["id"]): row
            for row in study.execute(
                "SELECT * FROM sessions WHERE run_id=?", (run_id,)
            ).fetchall()
        }

        public_rows: list[dict[str, Any]] = []
        rest_rows: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []
        events = study.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY participant_id, seq", (run_id,)
        ).fetchall()
        for event in events:
            participant_id = str(event["participant_id"])
            session_id = str(event["session_id"] or "")
            session = sessions.get(session_id)
            started_ms = int(session["started_ms"]) if session else int(event["server_ms"])
            payload = json.loads(str(event["payload"]) or "{}")
            participant_code = _pseudonym(export_pepper, "participant", participant_id)
            session_code = _pseudonym(export_pepper, "session", session_id or participant_id)
            event_type = str(event["event_type"])
            analysis_eligible = bool(
                payload.get("analysis_eligible", event_type not in {"rest.start", "rest.end"})
            )
            common = {
                "participant_id": participant_code,
                "session_id": session_code,
                "day": int(session["day"]) if session else "",
                "event_type": event_type,
                "relative_ms": max(0, int(event["server_ms"]) - started_ms),
            }
            if event_type in {"rest.start", "rest.end"} and participant_id in public_ids:
                rest_rows.append(
                    {
                        **common,
                        "block": payload.get("block", payload.get("active_block_index", "")),
                        "completed_seconds": payload.get("completed_seconds", ""),
                        "planned_seconds": payload.get("planned_seconds", ""),
                        "analysis_eligible": False,
                    }
                )
            elif participant_id in public_ids and analysis_eligible:
                participant = participants[participant_id]
                row = {
                    **common,
                    "split": _split(export_pepper, participant_id),
                    "cohort": participant["cohort"],
                    "condition": participant["group_name"],
                    "seq": int(event["seq"]),
                }
                for field in SAFE_EVENT_FIELDS:
                    row[field] = payload.get(field, "")
                row["analysis_eligible"] = True
                public_rows.append(row)

            if participant_id in controlled_ids:
                for field in TEXT_FIELDS:
                    value = payload.get(field)
                    if isinstance(value, str) and value.strip():
                        redacted, risks = _redact(value)
                        review_rows.append(
                            {
                                "review_id": _pseudonym(
                                    export_pepper,
                                    "review",
                                    f"{event['id']}:{field}",
                                ),
                                "participant_id": participant_code,
                                "session_id": session_code,
                                "source_type": "event_text",
                                "source_id": _pseudonym(
                                    export_pepper, "event", str(event["id"])
                                ),
                                "field": field,
                                "detected_risks": "|".join(risks),
                                "redacted_text": redacted,
                                "decision": "",
                                "reviewer": "",
                                "reviewed_at": "",
                            }
                        )

        _write_csv(public_dir / "interaction_events.csv", PUBLIC_EVENT_COLUMNS, public_rows)
        _write_csv(public_dir / "rest_intervals.csv", REST_COLUMNS, rest_rows)
        _write_csv(restricted_dir / "privacy_review_queue.csv", REVIEW_COLUMNS, review_rows)

        decisions = _review_decisions(audio_review_csv)
        if include_controlled_audio and audio_review_csv is None:
            raise ValueError(
                "audio_review_csv is required when include_controlled_audio is enabled"
            )
        audio_rows: list[dict[str, Any]] = []
        copied_audio = 0
        audio_dir = controlled_dir / "audio"
        if include_controlled_audio:
            audio_dir.mkdir()
        for audio in study.execute(
            "SELECT * FROM audio_files WHERE run_id=? ORDER BY created_ms", (run_id,)
        ).fetchall():
            participant_id = str(audio["participant_id"])
            consent_row = consents.get(participant_id)
            if (
                participant_id not in controlled_ids
                or consent_row is None
                or not _row_bool(consent_row, "raw_audio")
            ):
                continue
            audio_id = str(audio["id"])
            review_status = decisions.get(audio_id, "pending")
            exported_relative_path = ""
            if include_controlled_audio and review_status == "approved":
                source = (media_dir / str(audio["relative_path"])).resolve()
                if media_dir.resolve() not in source.parents or not source.is_file():
                    raise ValueError(f"invalid or missing audio source for {audio_id}")
                suffix = source.suffix.lower()[:10]
                target = audio_dir / f"{_pseudonym(export_pepper, 'audio', audio_id)}{suffix}"
                shutil.copy2(source, target)
                if hashlib.sha256(target.read_bytes()).hexdigest() != str(audio["sha256"]):
                    target.unlink(missing_ok=True)
                    raise ValueError(f"sha256 mismatch for {audio_id}")
                exported_relative_path = str(target.relative_to(output_dir)).replace("\\", "/")
                copied_audio += 1
            session = sessions.get(str(audio["session_id"] or ""))
            audio_rows.append(
                {
                    "audio_id": _pseudonym(export_pepper, "audio", audio_id),
                    "participant_id": _pseudonym(
                        export_pepper, "participant", participant_id
                    ),
                    "session_id": _pseudonym(
                        export_pepper, "session", str(audio["session_id"] or participant_id)
                    ),
                    "day": int(session["day"]) if session else "",
                    "kind": audio["kind"],
                    "content_type": audio["content_type"],
                    "bytes": int(audio["bytes"]),
                    "sha256": audio["sha256"],
                    "review_status": review_status,
                    "exported_relative_path": exported_relative_path,
                    "ai_training_allowed": _row_bool(consent_row, "ai_training"),
                    "commercial_research_allowed": _row_bool(
                        consent_row, "commercial_research"
                    ),
                }
            )
        _write_csv(controlled_dir / "audio_manifest.csv", AUDIO_COLUMNS, audio_rows)

        dataset_card = {
            "title": "Tap-to-Adapt Interaction Timing Dataset",
            "release_status": "researcher-generated candidate; requires ethics and PI approval",
            "run_slug": run_slug,
            "protocol_version": run["protocol_version"],
            "public_tier": (
                "Pseudonymized participant-disjoint event labels and model outputs; "
                "no text, raw audio, absolute timestamps, IP/user-agent, recruitment "
                "mapping, or participant-specific weights."
            ),
            "controlled_tier": (
                "Consent-filtered audio manifest. Audio bytes appear only after an "
                "approved per-file human privacy review and must remain access-controlled."
            ),
            "rest_policy": (
                "rest.start/rest.end are retained only in rest_intervals.csv for audit; "
                "rest data are excluded from interaction_events.csv and online learning."
            ),
            "known_reidentification_risks": [
                "Voice biometrics, accent, health cues, background speech, names and places",
                "Rare interaction sequences and small demographic cells",
                "Automated redaction cannot establish anonymity",
            ],
            "prohibited_uses": [
                "speaker identification or re-identification",
                "surveillance, employment, insurance, credit, or law-enforcement decisions",
                "inferring sensitive traits without separate ethics approval",
                "contacting or attempting to locate participants",
            ],
            "counts": {
                "enrolled_in_run": len(participants),
                "public_derived_consented": len(public_ids),
                "controlled_sharing_consented": len(controlled_ids),
                "public_event_rows": len(public_rows),
                "rest_audit_rows": len(rest_rows),
                "controlled_audio_manifest_rows": len(audio_rows),
                "controlled_audio_files_copied": copied_audio,
                "privacy_review_items": len(review_rows),
            },
            "split_policy": "HMAC participant-level 70/15/15; no participant crosses splits",
            "license": "Not assigned. Release requires institution-approved terms.",
        }
        (public_dir / "DATASET_CARD.json").write_text(
            json.dumps(dataset_card, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        files = []
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                files.append(
                    {
                        "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "access": (
                            "restricted"
                            if "_restricted_review" in path.parts
                            else ("controlled" if "controlled" in path.parts else "public")
                        ),
                    }
                )
        manifest = {
            "schema_version": "tap-to-adapt-release-v1",
            "run_slug": run_slug,
            "files": files,
            "public_audio_exported": False,
            "participant_specific_weights_exported": False,
        }
        manifest_path = output_dir / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ReleaseResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            public_participants=len(public_ids),
            controlled_participants=len(controlled_ids),
            controlled_audio_files=copied_audio,
        )
    finally:
        study.close()
        consent.close()
