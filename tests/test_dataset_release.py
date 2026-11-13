"""Regression tests for dataset release.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset_release import export_research_dataset


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    study_path = tmp_path / "study.sqlite3"
    consent_path = tmp_path / "consent.sqlite3"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    audio = media_dir / "sample.webm"
    audio.write_bytes(b"voice-bytes")
    audio_sha = hashlib.sha256(audio.read_bytes()).hexdigest()

    study = sqlite3.connect(study_path)
    study.executescript(
        """
        CREATE TABLE study_runs (
          id TEXT, slug TEXT, protocol_version TEXT
        );
        CREATE TABLE participants (
          id TEXT, run_id TEXT, cohort TEXT, group_name TEXT, created_ms INTEGER
        );
        CREATE TABLE sessions (
          id TEXT, run_id TEXT, participant_id TEXT, day INTEGER, started_ms INTEGER
        );
        CREATE TABLE events (
          id INTEGER, run_id TEXT, participant_id TEXT, session_id TEXT, seq INTEGER,
          event_type TEXT, server_ms INTEGER, payload TEXT
        );
        CREATE TABLE audio_files (
          id TEXT, run_id TEXT, participant_id TEXT, session_id TEXT, kind TEXT,
          relative_path TEXT, content_type TEXT, bytes INTEGER, sha256 TEXT,
          created_ms INTEGER
        );
        """
    )
    study.execute("INSERT INTO study_runs VALUES ('run-1','formal-v1','v1')")
    study.execute(
        "INSERT INTO participants VALUES ('person-1','run-1','study_b','P',1)"
    )
    study.execute(
        "INSERT INTO sessions VALUES ('session-1','run-1','person-1',1,1000)"
    )
    events = [
        (
            1,
            "run-1",
            "person-1",
            "session-1",
            1,
            "asr.final",
            1100,
            json.dumps(
                {
                    "text": "联系我 test@example.com",
                    "active_block_index": 1,
                    "analysis_eligible": True,
                }
            ),
        ),
        (
            2,
            "run-1",
            "person-1",
            "session-1",
            2,
            "rest.start",
            2000,
            json.dumps(
                {
                    "block": 1,
                    "planned_seconds": 60,
                    "analysis_eligible": False,
                }
            ),
        ),
        (
            3,
            "run-1",
            "person-1",
            "session-1",
            3,
            "tap.result",
            2100,
            json.dumps(
                {
                    "action": "manual_activate",
                    "label": 1,
                    "analysis_eligible": False,
                    "scheduled_rest": True,
                }
            ),
        ),
        (
            4,
            "run-1",
            "person-1",
            "session-1",
            4,
            "rest.end",
            62000,
            json.dumps(
                {
                    "block": 1,
                    "completed_seconds": 60,
                    "analysis_eligible": False,
                }
            ),
        ),
    ]
    study.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)", events)
    study.execute(
        "INSERT INTO audio_files VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            "audio-1",
            "run-1",
            "person-1",
            "session-1",
            "study_stream",
            "sample.webm",
            "audio/webm",
            audio.stat().st_size,
            audio_sha,
            1200,
        ),
    )
    study.commit()
    study.close()

    consent = sqlite3.connect(consent_path)
    consent.executescript(
        """
        CREATE TABLE consents (
          id TEXT, run_id TEXT, participant_id TEXT, participation INTEGER,
          raw_audio INTEGER, public_derived INTEGER, controlled_sharing INTEGER,
          ai_training INTEGER, commercial_research INTEGER, signed_ms INTEGER
        );
        """
    )
    consent.execute(
        "INSERT INTO consents VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("c1", "run-1", "person-1", 1, 1, 1, 1, 0, 0, 100),
    )
    consent.commit()
    consent.close()
    return study_path, consent_path, media_dir


def test_public_release_removes_text_absolute_time_and_rest_rows(tmp_path: Path) -> None:
    study, consent, media = _fixture(tmp_path)
    output = tmp_path / "release"
    result = export_research_dataset(
        study_db=study,
        consent_db=consent,
        media_dir=media,
        output_dir=output,
        run_slug="formal-v1",
        export_pepper="this-is-a-test-export-pepper",
    )

    rows = list(
        csv.DictReader(
            (output / "public" / "interaction_events.csv").open(
                encoding="utf-8-sig"
            )
        )
    )
    assert len(rows) == 1
    assert rows[0]["event_type"] == "asr.final"
    assert "text" not in rows[0]
    assert "server_ms" not in rows[0]
    assert rows[0]["participant_id"] != "person-1"
    assert rows[0]["relative_ms"] == "100"
    assert "test@example.com" not in (
        output / "public" / "interaction_events.csv"
    ).read_text(encoding="utf-8-sig")
    assert result.controlled_audio_files == 0
    assert not (output / "controlled" / "audio").exists()

    rest = list(
        csv.DictReader(
            (output / "public" / "rest_intervals.csv").open(encoding="utf-8-sig")
        )
    )
    assert [row["event_type"] for row in rest] == ["rest.start", "rest.end"]
    assert all(row["analysis_eligible"] == "False" for row in rest)


def test_personal_debug_run_is_rejected_before_output_creation(
    tmp_path: Path,
) -> None:
    study, consent, media = _fixture(tmp_path)
    connection = sqlite3.connect(study)
    connection.execute(
        "ALTER TABLE participants ADD COLUMN analysis_excluded INTEGER NOT NULL DEFAULT 0"
    )
    connection.execute(
        "UPDATE participants SET analysis_excluded=1 WHERE id='person-1'"
    )
    connection.commit()
    connection.close()
    output = tmp_path / "debug-release"

    with pytest.raises(ValueError, match="permanently excluded"):
        export_research_dataset(
            study_db=study,
            consent_db=consent,
            media_dir=media,
            output_dir=output,
            run_slug="formal-v1",
            export_pepper="this-is-a-test-export-pepper",
        )
    assert not output.exists()


def test_controlled_audio_fails_closed_then_copies_only_after_review(
    tmp_path: Path,
) -> None:
    study, consent, media = _fixture(tmp_path)
    with pytest.raises(ValueError, match="audio_review_csv"):
        export_research_dataset(
            study_db=study,
            consent_db=consent,
            media_dir=media,
            output_dir=tmp_path / "missing-review",
            run_slug="formal-v1",
            export_pepper="this-is-a-test-export-pepper",
            include_controlled_audio=True,
        )

    review = tmp_path / "review.csv"
    review.write_text("audio_id,status\naudio-1,approved\n", encoding="utf-8")
    result = export_research_dataset(
        study_db=study,
        consent_db=consent,
        media_dir=media,
        output_dir=tmp_path / "approved",
        run_slug="formal-v1",
        export_pepper="this-is-a-test-export-pepper",
        include_controlled_audio=True,
        audio_review_csv=review,
    )
    assert result.controlled_audio_files == 1
    exported = list((tmp_path / "approved" / "controlled" / "audio").iterdir())
    assert len(exported) == 1
    assert exported[0].read_bytes() == b"voice-bytes"
