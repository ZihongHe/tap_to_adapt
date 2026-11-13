"""Regression tests for consent evidence.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from app.config import settings
from app.db import StudyDB


def test_existing_consent_database_migrates_before_ledger_index(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    consent_path = data_dir / "consent.sqlite3"
    with sqlite3.connect(consent_path) as conn:
        conn.execute(
            """CREATE TABLE consents (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, participant_id TEXT NOT NULL,
                version TEXT NOT NULL, document_hash TEXT NOT NULL,
                document_json TEXT NOT NULL DEFAULT '', participation INTEGER NOT NULL,
                raw_audio INTEGER NOT NULL, future_sharing INTEGER NOT NULL,
                internal_reuse INTEGER NOT NULL DEFAULT 0,
                controlled_sharing INTEGER NOT NULL DEFAULT 0,
                public_derived INTEGER NOT NULL DEFAULT 0,
                public_audio INTEGER NOT NULL DEFAULT 0,
                ai_training INTEGER NOT NULL DEFAULT 0,
                commercial_research INTEGER NOT NULL DEFAULT 0,
                affirmation TEXT NOT NULL, signed_ms INTEGER NOT NULL,
                ip_hash TEXT NOT NULL, user_agent TEXT NOT NULL
            )"""
        )

    database = StudyDB(replace(settings, data_dir=data_dir))
    with database.connect_consent() as conn:
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(consents)")
        }
        indexes = {
            str(row["name"]) for row in conn.execute("PRAGMA index_list(consents)")
        }
    assert {"evidence_json", "evidence_key_id", "ledger_seq", "record_sha256"} <= columns
    assert "idx_consents_ledger_seq" in indexes
