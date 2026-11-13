"""SQLite persistence for participant state and auditable study events.

StudyDB owns schema initialization, transactional participant/session updates,
quality and questionnaire records, and event-chain bookkeeping. Participant
identity and reimbursement plaintext are handled by the separate vault.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import string
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .auth import code_hash, is_full_participant_code
from .config import Settings


class StudyCapacityExceeded(RuntimeError):
    """Raised when an atomic study-pair activation would exceed its cap."""


TIMED_ACTIVE_BLOCK_MS = 6 * 60 * 1000
TIMED_REST_MS = 60 * 1000
TIMED_ACTIVE_BLOCKS = 5


def estimated_timed_progress(
    activated_ms: int | None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Estimate the 5×6-minute block from official start time only.

    Does not read events, models, or orchestration state.
    """
    clock_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    if not activated_ms:
        return {
            "phase": "not_started",
            "block": None,
            "elapsed_ms": None,
            "remaining_ms": None,
            "label": "尚未正式计时",
        }
    elapsed_ms = max(0, clock_ms - int(activated_ms))
    cycle_ms = TIMED_ACTIVE_BLOCK_MS + TIMED_REST_MS
    total_ms = (TIMED_ACTIVE_BLOCKS - 1) * cycle_ms + TIMED_ACTIVE_BLOCK_MS
    remaining_ms = max(0, total_ms - elapsed_ms)
    for index in range(1, TIMED_ACTIVE_BLOCKS + 1):
        start_ms = (index - 1) * cycle_ms
        if elapsed_ms < start_ms + TIMED_ACTIVE_BLOCK_MS:
            return {
                "phase": "interaction",
                "block": index,
                "elapsed_ms": elapsed_ms,
                "remaining_ms": remaining_ms,
                "label": f"第{index}窗口互动中",
            }
        if index < TIMED_ACTIVE_BLOCKS and elapsed_ms < start_ms + cycle_ms:
            return {
                "phase": "rest",
                "block": index,
                "elapsed_ms": elapsed_ms,
                "remaining_ms": remaining_ms,
                "label": f"第{index}窗口后休息",
            }
    return {
        "phase": "closing",
        "block": TIMED_ACTIVE_BLOCKS,
        "elapsed_ms": elapsed_ms,
        "remaining_ms": remaining_ms,
        "label": "计时将结束/收尾",
    }


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS study_runs (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    data_origin TEXT NOT NULL CHECK (data_origin IN ('formal','pilot','simulation')),
    protocol_version TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    protocol_json TEXT NOT NULL,
    protocol_mode TEXT NOT NULL CHECK (protocol_mode IN ('strict','compressed')),
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS participants (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    code_hash TEXT NOT NULL UNIQUE,
    code_hint TEXT NOT NULL,
    cohort TEXT NOT NULL CHECK (cohort IN ('pilot','study_a','study_b')),
    group_name TEXT NOT NULL,
    probe_schedule TEXT NOT NULL,
    prior_id TEXT,
    prior_n INTEGER,
    prior_q INTEGER,
    preview_mode INTEGER NOT NULL DEFAULT 0,
    analysis_excluded INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    day INTEGER NOT NULL CHECK (day BETWEEN 1 AND 3),
    client_role TEXT NOT NULL CHECK (client_role IN ('study','companion')),
    state TEXT NOT NULL DEFAULT 'started',
    started_ms INTEGER NOT NULL,
    activated_ms INTEGER,
    ended_ms INTEGER,
    UNIQUE(participant_id, day, client_role)
);

CREATE TABLE IF NOT EXISTS pair_intervals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    day INTEGER NOT NULL CHECK (day BETWEEN 1 AND 3),
    study_session_id TEXT NOT NULL REFERENCES sessions(id),
    companion_session_id TEXT NOT NULL REFERENCES sessions(id),
    started_ms INTEGER NOT NULL,
    ended_ms INTEGER,
    end_reason TEXT,
    qualified INTEGER NOT NULL DEFAULT 0,
    qualified_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pair_intervals_participant_day
    ON pair_intervals(participant_id, day, started_ms);

CREATE TABLE IF NOT EXISTS companion_presence (
    participant_id TEXT PRIMARY KEY,
    companion_session_id TEXT NOT NULL,
    day INTEGER NOT NULL,
    last_seen_ms INTEGER NOT NULL,
    ready INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    session_id TEXT REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    server_ms INTEGER NOT NULL,
    client_ms REAL,
    monotonic_ms REAL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    UNIQUE(participant_id, seq)
);

CREATE TABLE IF NOT EXISTS quality_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    session_id TEXT REFERENCES sessions(id),
    kind TEXT NOT NULL,
    passed INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    session_id TEXT REFERENCES sessions(id),
    instrument TEXT NOT NULL,
    version TEXT NOT NULL,
    answers TEXT NOT NULL,
    answer_labels TEXT NOT NULL DEFAULT '[]',
    question_order TEXT NOT NULL DEFAULT '[]',
    locale TEXT NOT NULL DEFAULT 'zh-CN',
    created_ms INTEGER NOT NULL,
    UNIQUE(participant_id, session_id, instrument)
);

CREATE TABLE IF NOT EXISTS final_questionnaire_recoveries (
    participant_id TEXT PRIMARY KEY REFERENCES participants(id),
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id),
    credential TEXT NOT NULL UNIQUE,
    entered_ms INTEGER NOT NULL,
    model_finalized_ms INTEGER,
    chain_committed_ms INTEGER,
    final_submitted_ms INTEGER,
    last_error TEXT,
    updated_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_final_questionnaire_recoveries_run
    ON final_questionnaire_recoveries(run_id, entered_ms);

CREATE TABLE IF NOT EXISTS audio_files (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    session_id TEXT REFERENCES sessions(id),
    kind TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    content_type TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS shared_priors (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    donor_count INTEGER NOT NULL,
    opportunity_budget INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate','selected','retired')),
    algorithm TEXT NOT NULL,
    donor_ids TEXT NOT NULL,
    encoder_path TEXT NOT NULL,
    main_path TEXT NOT NULL,
    encoder_sha256 TEXT NOT NULL,
    main_sha256 TEXT NOT NULL,
    manifest TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    selected_ms INTEGER,
    UNIQUE(run_id, donor_count, opportunity_budget)
);

CREATE TABLE IF NOT EXISTS model_snapshots (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    session_id TEXT REFERENCES sessions(id),
    source TEXT NOT NULL CHECK (source IN ('donor_shadow','personalized')),
    opportunity_budget INTEGER NOT NULL,
    budget_kind TEXT NOT NULL DEFAULT 'activation_opportunity',
    eligible_feedbacks INTEGER NOT NULL DEFAULT 0,
    positive_feedbacks INTEGER NOT NULL DEFAULT 0,
    negative_feedbacks INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}',
    base_prior_id TEXT,
    encoder_path TEXT NOT NULL,
    main_path TEXT NOT NULL,
    encoder_sha256 TEXT NOT NULL,
    main_sha256 TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE(participant_id, source, opportunity_budget)
);

CREATE TABLE IF NOT EXISTS pipeline_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prior_probe_assignments (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    participant_id TEXT NOT NULL REFERENCES participants(id),
    day INTEGER NOT NULL CHECK (day BETWEEN 1 AND 3),
    probe_token TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    prior_id TEXT NOT NULL REFERENCES shared_priors(id),
    assignment_sha256 TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    UNIQUE(participant_id, day, probe_token),
    UNIQUE(participant_id, day, order_index)
);
CREATE TABLE IF NOT EXISTS model_counters (
    participant_id TEXT PRIMARY KEY REFERENCES participants(id),
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    learning_opportunities INTEGER NOT NULL DEFAULT 0,
    activation_opportunities INTEGER NOT NULL DEFAULT 0,
    eligible_feedbacks INTEGER NOT NULL DEFAULT 0,
    positive_feedbacks INTEGER NOT NULL DEFAULT 0,
    negative_feedbacks INTEGER NOT NULL DEFAULT 0,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS participant_calibrations (
    participant_id TEXT PRIMARY KEY REFERENCES participants(id),
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    calibration_version TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    personal_logit_bias REAL NOT NULL DEFAULT 0.0,
    update_count INTEGER NOT NULL DEFAULT 0,
    positive_updates INTEGER NOT NULL DEFAULT 0,
    negative_updates INTEGER NOT NULL DEFAULT 0,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_memory (
    participant_id TEXT NOT NULL REFERENCES participants(id),
    source TEXT NOT NULL CHECK (source IN ('primary','companion')),
    paragraph TEXT NOT NULL DEFAULT '',
    turn_count INTEGER NOT NULL DEFAULT 0,
    updated_ms INTEGER NOT NULL,
    PRIMARY KEY (participant_id, source)
);
CREATE INDEX IF NOT EXISTS idx_conversation_memory_participant
    ON conversation_memory(participant_id, source);
CREATE TABLE IF NOT EXISTS account_batches (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES study_runs(id),
    stage TEXT NOT NULL,
    label TEXT NOT NULL,
    cohort TEXT NOT NULL,
    group_name TEXT NOT NULL,
    data_origin TEXT NOT NULL,
    protocol_mode TEXT NOT NULL,
    requested_count INTEGER NOT NULL,
    participant_ids TEXT NOT NULL,
    is_pre_experiment INTEGER NOT NULL DEFAULT 0,
    analysis_excluded INTEGER NOT NULL DEFAULT 0,
    storage_namespace TEXT NOT NULL DEFAULT 'pre_experiments',
    archive_relative_path TEXT,
    created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_account_batches_run
    ON account_batches(run_id, created_ms);
CREATE INDEX IF NOT EXISTS idx_shared_priors_run ON shared_priors(run_id, donor_count, opportunity_budget);
CREATE INDEX IF NOT EXISTS idx_model_snapshots_run ON model_snapshots(run_id, participant_id, source);
CREATE INDEX IF NOT EXISTS idx_pipeline_actions_run ON pipeline_actions(run_id, created_ms);
CREATE INDEX IF NOT EXISTS idx_prior_probe_assignments_run
    ON prior_probe_assignments(run_id, participant_id, day, order_index);
CREATE INDEX IF NOT EXISTS idx_participant_calibrations_run
    ON participant_calibrations(run_id, participant_id);
"""

CONSENT_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS consents (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    participant_id TEXT NOT NULL,
    version TEXT NOT NULL,
    document_hash TEXT NOT NULL,
    document_json TEXT NOT NULL,
    participation INTEGER NOT NULL,
    raw_audio INTEGER NOT NULL,
    future_sharing INTEGER NOT NULL,
    internal_reuse INTEGER NOT NULL DEFAULT 0,
    controlled_sharing INTEGER NOT NULL DEFAULT 0,
    public_derived INTEGER NOT NULL DEFAULT 0,
    public_audio INTEGER NOT NULL DEFAULT 0,
    ai_training INTEGER NOT NULL DEFAULT 0,
    commercial_research INTEGER NOT NULL DEFAULT 0,
    typed_name TEXT NOT NULL DEFAULT '',
    affirmation TEXT NOT NULL,
    signed_ms INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    client_nonce TEXT NOT NULL DEFAULT '',
    auth_context_sha256 TEXT NOT NULL DEFAULT '',
    evidence_json TEXT NOT NULL DEFAULT '',
    evidence_sha256 TEXT NOT NULL DEFAULT '',
    evidence_key_id TEXT NOT NULL DEFAULT '',
    ledger_seq INTEGER,
    previous_record_sha256 TEXT NOT NULL DEFAULT '',
    record_sha256 TEXT NOT NULL DEFAULT '',
    server_instance TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_consents_participant ON consents(participant_id, signed_ms);
"""


class StudyDB:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.study_path = settings.data_dir / "study.sqlite3"
        self.consent_path = settings.data_dir / "consent.sqlite3"
        self.login_codes_path = settings.data_dir / "participant_login_codes.sqlite3"
        protocol_path = settings.content_dir / "study_protocol.json"
        self.protocol_text = protocol_path.read_text(encoding="utf-8")
        self.protocol_version = str(json.loads(self.protocol_text)["version"])
        self.protocol_sha256 = hashlib.sha256(
            self.protocol_text.encode("utf-8")
        ).hexdigest()
        settings.ensure_directories()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            participant_columns = self._columns(conn, "participants")
            if "preview_mode" not in participant_columns:
                conn.execute(
                    "ALTER TABLE participants ADD COLUMN preview_mode INTEGER NOT NULL DEFAULT 0"
                )
            participant_migrations = {
                "assignment_id": "TEXT",
                "neutral_id": "TEXT",
                "enrollment_wave": "INTEGER",
                "release_state": "TEXT NOT NULL DEFAULT 'released'",
                "analysis_excluded": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in participant_migrations.items():
                if column not in participant_columns:
                    conn.execute(f"ALTER TABLE participants ADD COLUMN {column} {declaration}")
            run_columns = self._columns(conn, "study_runs")
            for column in ("protocol_sha256", "protocol_json"):
                if column not in run_columns:
                    conn.execute(f"ALTER TABLE study_runs ADD COLUMN {column} TEXT")
            counter_columns = self._columns(conn, "model_counters")
            activation_added = "activation_opportunities" not in counter_columns
            counter_migrations = {
                "activation_opportunities": "INTEGER NOT NULL DEFAULT 0",
                "eligible_feedbacks": "INTEGER NOT NULL DEFAULT 0",
                "positive_feedbacks": "INTEGER NOT NULL DEFAULT 0",
                "negative_feedbacks": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in counter_migrations.items():
                if column not in counter_columns:
                    conn.execute(f"ALTER TABLE model_counters ADD COLUMN {column} {declaration}")
            if activation_added:
                conn.execute(
                    "UPDATE model_counters SET activation_opportunities=learning_opportunities"
                )
            snapshot_columns = self._columns(conn, "model_snapshots")
            snapshot_migrations = {
                "budget_kind": "TEXT NOT NULL DEFAULT 'activation_opportunity'",
                "eligible_feedbacks": "INTEGER NOT NULL DEFAULT 0",
                "positive_feedbacks": "INTEGER NOT NULL DEFAULT 0",
                "negative_feedbacks": "INTEGER NOT NULL DEFAULT 0",
                "metadata": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column, declaration in snapshot_migrations.items():
                if column not in snapshot_columns:
                    conn.execute(f"ALTER TABLE model_snapshots ADD COLUMN {column} {declaration}")
            pair_columns = self._columns(conn, "pair_intervals")
            pair_migrations = {
                "qualified": "INTEGER NOT NULL DEFAULT 0",
                "qualified_ms": "INTEGER",
                "study_last_seen_ms": "INTEGER",
                "companion_last_seen_ms": "INTEGER",
                "companion_ready": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, declaration in pair_migrations.items():
                if column not in pair_columns:
                    conn.execute(
                        f"ALTER TABLE pair_intervals ADD COLUMN {column} {declaration}"
                    )
            batch_columns = self._columns(conn, "account_batches")
            batch_migrations = {
                "is_pre_experiment": "INTEGER NOT NULL DEFAULT 0",
                "analysis_excluded": "INTEGER NOT NULL DEFAULT 0",
                "storage_namespace": "TEXT NOT NULL DEFAULT 'pre_experiments'",
                "archive_relative_path": "TEXT",
            }
            for column, declaration in batch_migrations.items():
                if column not in batch_columns:
                    conn.execute(
                        f"ALTER TABLE account_batches ADD COLUMN {column} {declaration}"
                    )
            response_columns = self._columns(conn, "responses")
            response_migrations = {
                "answer_labels": "TEXT NOT NULL DEFAULT '[]'",
                "question_order": "TEXT NOT NULL DEFAULT '[]'",
                "locale": "TEXT NOT NULL DEFAULT 'zh-CN'",
            }
            for column, declaration in response_migrations.items():
                if column not in response_columns:
                    conn.execute(
                        f"ALTER TABLE responses ADD COLUMN {column} {declaration}"
                    )
            self._migrate_study_runs(conn)
        with self.connect_consent() as conn:
            conn.executescript(CONSENT_SCHEMA)
            conn.execute("DROP TRIGGER IF EXISTS consents_run_immutable")
            conn.execute("DROP TRIGGER IF EXISTS consents_immutable")
            conn.execute("DROP TRIGGER IF EXISTS consents_delete_forbidden")
            if "run_id" not in self._columns(conn, "consents"):
                conn.execute("ALTER TABLE consents ADD COLUMN run_id TEXT")
            consent_columns = self._columns(conn, "consents")
            if "document_json" not in consent_columns:
                conn.execute(
                    "ALTER TABLE consents ADD COLUMN document_json TEXT NOT NULL DEFAULT ''"
                )
                consent_columns.add("document_json")
            for column in (
                "internal_reuse",
                "controlled_sharing",
                "public_derived",
                "public_audio",
                "ai_training",
                "commercial_research",
            ):
                if column not in consent_columns:
                    conn.execute(
                        f"ALTER TABLE consents ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                    )
            consent_columns = self._columns(conn, "consents")
            if "typed_name" not in consent_columns:
                conn.execute(
                    "ALTER TABLE consents ADD COLUMN typed_name TEXT NOT NULL DEFAULT ''"
                )
                consent_columns.add("typed_name")
            evidence_migrations = {
                "client_nonce": "TEXT NOT NULL DEFAULT ''",
                "auth_context_sha256": "TEXT NOT NULL DEFAULT ''",
                "evidence_json": "TEXT NOT NULL DEFAULT ''",
                "evidence_sha256": "TEXT NOT NULL DEFAULT ''",
                "evidence_key_id": "TEXT NOT NULL DEFAULT ''",
                "ledger_seq": "INTEGER",
                "previous_record_sha256": "TEXT NOT NULL DEFAULT ''",
                "record_sha256": "TEXT NOT NULL DEFAULT ''",
                "server_instance": "TEXT NOT NULL DEFAULT ''",
            }
            for column, declaration in evidence_migrations.items():
                if column not in consent_columns:
                    conn.execute(
                        f"ALTER TABLE consents ADD COLUMN {column} {declaration}"
                    )
            conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_consents_ledger_seq
                   ON consents(ledger_seq) WHERE ledger_seq IS NOT NULL"""
            )
            with self.connect() as study_conn:
                mappings = study_conn.execute("SELECT id, run_id FROM participants").fetchall()
            for row in mappings:
                conn.execute(
                    "UPDATE consents SET run_id=? WHERE participant_id=? AND run_id IS NULL",
                    (row["run_id"], row["id"]),
                )
            conn.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS consents_run_immutable
                BEFORE UPDATE OF run_id ON consents
                BEGIN SELECT RAISE(ABORT, 'consent run_id is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS consents_run_required
                BEFORE INSERT ON consents WHEN NEW.run_id IS NULL
                BEGIN SELECT RAISE(ABORT, 'consent run_id is required'); END;
                CREATE TRIGGER IF NOT EXISTS consents_immutable
                BEFORE UPDATE ON consents
                BEGIN SELECT RAISE(ABORT, 'consent records are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS consents_delete_forbidden
                BEFORE DELETE ON consents
                BEGIN SELECT RAISE(ABORT, 'consent records cannot be deleted in place'); END;
                """
            )

    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}

    def _run_row_compatible(
        self,
        row: sqlite3.Row,
        data_origin: str,
        protocol_mode: str,
    ) -> bool:
        if (
            row["data_origin"] != data_origin
            or row["protocol_mode"] != protocol_mode
            or row["protocol_version"] != self.protocol_version
        ):
            return False
        if row["protocol_sha256"] is not None and row["protocol_sha256"] != self.protocol_sha256:
            return False
        if row["protocol_json"] is not None:
            actual = hashlib.sha256(row["protocol_json"].encode("utf-8")).hexdigest()
            if actual != self.protocol_sha256:
                return False
        return True

    def protocol_scoped_run_slug(self, preferred: str) -> str:
        """Rewrite a stale enrollment slug onto the current protocol short version."""
        short = str(self.protocol_version).rsplit("-", 1)[-1]
        if not re.fullmatch(r"v\d+", short):
            short = "v1"
        base = re.sub(r"-v\d+-v\d+$", "", preferred.strip())
        base = re.sub(r"-v\d+$", "", base)
        base = base.rstrip("-") or preferred.strip() or "public-run"
        return f"{base}-{short}-v1"

    def resolve_enrollment_run_slug(
        self,
        preferred: str,
        data_origin: str,
        protocol_mode: str,
    ) -> str:
        """Return a run slug compatible with the current protocol snapshot.

        Public enrollment env slugs are often left pointing at the previous
        protocol after a content bump. Prefer the configured slug when it is
        still compatible; otherwise move to a protocol-scoped sibling slug.
        """
        preferred = (preferred or "").strip() or f"public-{data_origin}-v1"
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM study_runs WHERE slug=?", (preferred,)
            ).fetchone()
            if row is None or self._run_row_compatible(row, data_origin, protocol_mode):
                if row is not None and row["protocol_sha256"] is None and not preferred.startswith("legacy-"):
                    raise ValueError(
                        f"Study run {preferred} predates protocol snapshots and cannot accept new accounts"
                    )
                return preferred
            candidate = self.protocol_scoped_run_slug(preferred)
            if candidate == preferred:
                candidate = f"{preferred}-{str(self.protocol_version).rsplit('-', 1)[-1]}-v1"
            existing = conn.execute(
                "SELECT * FROM study_runs WHERE slug=?", (candidate,)
            ).fetchone()
            if existing is None or self._run_row_compatible(
                existing, data_origin, protocol_mode
            ):
                return candidate
            raise ValueError(
                f"Study run {preferred} exists with incompatible immutable metadata"
            )

    def _ensure_run_in_conn(
        self,
        conn: sqlite3.Connection,
        slug: str,
        data_origin: str,
        protocol_mode: str,
    ) -> str:
        row = conn.execute("SELECT * FROM study_runs WHERE slug=?", (slug,)).fetchone()
        if row is not None:
            if not self._run_row_compatible(row, data_origin, protocol_mode):
                raise ValueError(f"Study run {slug} exists with incompatible immutable metadata")
            if row["protocol_sha256"] is None and not slug.startswith("legacy-"):
                raise ValueError(
                    f"Study run {slug} predates protocol snapshots and cannot accept new accounts"
                )
            return str(row["id"])
        run_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO study_runs
               (id, slug, data_origin, protocol_version, protocol_sha256,
                protocol_json, protocol_mode, created_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                slug,
                data_origin,
                self.protocol_version,
                self.protocol_sha256,
                self.protocol_text,
                protocol_mode,
                int(time.time() * 1000),
            ),
        )
        return run_id

    def _migrate_study_runs(self, conn: sqlite3.Connection) -> None:
        if "run_id" not in self._columns(conn, "participants"):
            conn.execute("ALTER TABLE participants ADD COLUMN run_id TEXT")
        child_tables = (
            "sessions",
            "pair_intervals",
            "events",
            "quality_checks",
            "responses",
            "final_questionnaire_recoveries",
            "audio_files",
        )
        for table in child_tables:
            if "run_id" not in self._columns(conn, table):
                conn.execute(f"ALTER TABLE {table} ADD COLUMN run_id TEXT")
        if "activated_ms" not in self._columns(conn, "sessions"):
            conn.execute("ALTER TABLE sessions ADD COLUMN activated_ms INTEGER")

        legacy_runs = {
            "formal": self._ensure_run_in_conn(
                conn, f"legacy-formal-{self.protocol_version}", "formal", "strict"
            ),
            "pilot": self._ensure_run_in_conn(
                conn, f"legacy-pilot-{self.protocol_version}", "pilot", "compressed"
            ),
            "simulation": self._ensure_run_in_conn(
                conn, f"legacy-simulation-{self.protocol_version}", "simulation", "compressed"
            ),
        }
        rows = conn.execute(
            "SELECT id, cohort, preview_mode FROM participants WHERE run_id IS NULL"
        ).fetchall()
        for row in rows:
            if bool(row["preview_mode"]):
                origin = "simulation"
            elif row["cohort"] == "pilot":
                origin = "pilot"
            else:
                origin = "formal"
            conn.execute(
                "UPDATE participants SET run_id=? WHERE id=?",
                (legacy_runs[origin], row["id"]),
            )
        for table in child_tables:
            conn.execute(
                f"""UPDATE {table} SET run_id=(
                        SELECT p.run_id FROM participants p WHERE p.id={table}.participant_id
                    ) WHERE run_id IS NULL"""
            )

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_participants_run ON participants(run_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_run ON sessions(run_id);
            CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
            CREATE INDEX IF NOT EXISTS idx_responses_run ON responses(run_id);

            DROP TRIGGER IF EXISTS study_run_identity_immutable;
            CREATE TRIGGER study_run_identity_immutable
            BEFORE UPDATE OF id, slug, data_origin, protocol_version, protocol_sha256,
                             protocol_json, protocol_mode ON study_runs
            BEGIN SELECT RAISE(ABORT, 'study run identity is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS participant_run_immutable
            BEFORE UPDATE OF run_id ON participants
            BEGIN SELECT RAISE(ABORT, 'participant run_id is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS participant_run_required
            BEFORE INSERT ON participants WHEN NEW.run_id IS NULL
            BEGIN SELECT RAISE(ABORT, 'participant run_id is required'); END;
            CREATE TRIGGER IF NOT EXISTS participant_analysis_exclusion_immutable
            BEFORE UPDATE OF analysis_excluded ON participants
            BEGIN SELECT RAISE(ABORT, 'participant analysis exclusion is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS participant_assignment_immutable
            BEFORE UPDATE OF cohort, group_name, probe_schedule, preview_mode,
                             assignment_id, neutral_id, enrollment_wave ON participants
            WHEN (OLD.assignment_id IS NOT NULL OR OLD.neutral_id IS NOT NULL OR OLD.enrollment_wave IS NOT NULL)
                 AND (NEW.cohort IS NOT OLD.cohort
                      OR NEW.group_name IS NOT OLD.group_name
                      OR NEW.probe_schedule IS NOT OLD.probe_schedule
                      OR NEW.preview_mode IS NOT OLD.preview_mode
                      OR NEW.assignment_id IS NOT OLD.assignment_id
                      OR NEW.neutral_id IS NOT OLD.neutral_id
                      OR NEW.enrollment_wave IS NOT OLD.enrollment_wave)
            BEGIN SELECT RAISE(ABORT, 'participant assignment is immutable'); END;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_run_assignment
            ON participants(run_id, assignment_id) WHERE assignment_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_run_neutral
            ON participants(run_id, neutral_id) WHERE neutral_id IS NOT NULL;
            CREATE TRIGGER IF NOT EXISTS participant_prior_immutable
            BEFORE UPDATE OF prior_id, prior_n, prior_q ON participants
            WHEN OLD.prior_id IS NOT NULL AND (
                NEW.prior_id IS NOT OLD.prior_id
                OR NEW.prior_n IS NOT OLD.prior_n
                OR NEW.prior_q IS NOT OLD.prior_q
            )
            BEGIN SELECT RAISE(ABORT, 'participant prior assignment is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS shared_prior_identity_immutable
            BEFORE UPDATE OF id, run_id, donor_count, opportunity_budget, algorithm, donor_ids,
                             encoder_path, main_path, encoder_sha256, main_sha256, manifest, created_ms
            ON shared_priors
            BEGIN SELECT RAISE(ABORT, 'shared prior identity is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS model_snapshot_immutable
            BEFORE UPDATE ON model_snapshots
            BEGIN SELECT RAISE(ABORT, 'model snapshots are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS model_snapshot_delete_forbidden
            BEFORE DELETE ON model_snapshots
            BEGIN SELECT RAISE(ABORT, 'model snapshots cannot be deleted in place'); END;
            CREATE TRIGGER IF NOT EXISTS pipeline_action_immutable
            BEFORE UPDATE ON pipeline_actions
            BEGIN SELECT RAISE(ABORT, 'pipeline actions are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS pipeline_action_delete_forbidden
            BEFORE DELETE ON pipeline_actions
            BEGIN SELECT RAISE(ABORT, 'pipeline actions cannot be deleted in place'); END;
            CREATE TRIGGER IF NOT EXISTS prior_probe_assignment_immutable
            BEFORE UPDATE ON prior_probe_assignments
            BEGIN SELECT RAISE(ABORT, 'prior probe assignments are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS prior_probe_assignment_delete_forbidden
            BEFORE DELETE ON prior_probe_assignments
            BEGIN SELECT RAISE(ABORT, 'prior probe assignments cannot be deleted in place'); END;
            CREATE TRIGGER IF NOT EXISTS prior_probe_assignment_run_consistent
            BEFORE INSERT ON prior_probe_assignments
            WHEN NEW.run_id IS NULL
                 OR NEW.run_id != (SELECT run_id FROM participants WHERE id=NEW.participant_id)
                 OR NEW.run_id != (SELECT run_id FROM shared_priors WHERE id=NEW.prior_id)
            BEGIN SELECT RAISE(ABORT, 'prior probe assignment run_id mismatch'); END;
            CREATE TRIGGER IF NOT EXISTS events_immutable
            BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT, 'event records are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS events_delete_forbidden
            BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT, 'event records cannot be deleted in place'); END;
            CREATE TRIGGER IF NOT EXISTS model_counter_run_consistent
            BEFORE INSERT ON model_counters
            WHEN NEW.run_id != (SELECT run_id FROM participants WHERE id=NEW.participant_id)
            BEGIN SELECT RAISE(ABORT, 'model counter run_id mismatch'); END;
            CREATE TRIGGER IF NOT EXISTS participant_calibration_run_immutable
            BEFORE UPDATE OF run_id, calibration_version, protocol_sha256
            ON participant_calibrations
            BEGIN SELECT RAISE(ABORT, 'participant calibration identity is immutable'); END;
            CREATE TRIGGER IF NOT EXISTS participant_calibration_run_consistent
            BEFORE INSERT ON participant_calibrations
            WHEN NEW.run_id != (SELECT run_id FROM participants WHERE id=NEW.participant_id)
                 OR NEW.protocol_sha256 != (
                     SELECT protocol_sha256 FROM study_runs WHERE id=NEW.run_id
                 )
            BEGIN SELECT RAISE(ABORT, 'participant calibration run mismatch'); END;
            CREATE TRIGGER IF NOT EXISTS account_batches_immutable
            BEFORE UPDATE ON account_batches
            BEGIN SELECT RAISE(ABORT, 'account batches are immutable'); END;
            CREATE TRIGGER IF NOT EXISTS account_batches_delete_forbidden
            BEFORE DELETE ON account_batches
            BEGIN SELECT RAISE(ABORT, 'account batches cannot be deleted in place'); END;
            """
        )
        for table in child_tables:
            conn.executescript(
                f"""
                CREATE TRIGGER IF NOT EXISTS {table}_run_immutable
                BEFORE UPDATE OF run_id ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} run_id is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS {table}_run_consistent
                BEFORE INSERT ON {table}
                WHEN NEW.run_id IS NULL OR NEW.run_id != (
                    SELECT run_id FROM participants WHERE id=NEW.participant_id
                )
                BEGIN SELECT RAISE(ABORT, '{table} run_id mismatch'); END;
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.study_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @contextmanager
    def connect_consent(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.consent_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _run_id(conn: sqlite3.Connection, participant_id: str) -> str:
        row = conn.execute(
            "SELECT run_id FROM participants WHERE id=?", (participant_id,)
        ).fetchone()
        if row is None or not row["run_id"]:
            raise ValueError("Participant is not assigned to a study run")
        return str(row["run_id"])

    def create_participants(
        self,
        count: int,
        cohort: str,
        group_name: str,
        preview_mode: bool = False,
        data_origin: str | None = None,
        protocol_mode: str | None = None,
        run_slug: str | None = None,
        analysis_excluded: bool = False,
    ) -> list[str]:
        if count < 1 or count > 1000:
            raise ValueError("count must be between 1 and 1000")
        if cohort not in {"pilot", "study_a", "study_b"}:
            raise ValueError("invalid cohort")
        data_origin = data_origin or (
            "simulation" if preview_mode else "pilot" if cohort == "pilot" else "formal"
        )
        protocol_mode = protocol_mode or ("strict" if data_origin == "formal" else "compressed")
        if data_origin not in {"formal", "pilot", "simulation"}:
            raise ValueError("invalid data origin")
        if protocol_mode not in {"strict", "compressed"}:
            raise ValueError("invalid protocol mode")
        if data_origin == "formal" and (cohort == "pilot" or preview_mode or protocol_mode != "strict"):
            raise ValueError("formal runs require Study A/B, preview disabled, and strict protocol")
        if data_origin == "pilot" and cohort != "pilot":
            raise ValueError("pilot origin requires the pilot cohort")
        if data_origin == "simulation" and (not preview_mode or protocol_mode != "compressed"):
            raise ValueError("simulation runs require preview mode and compressed protocol")
        run_slug = run_slug or f"default-{data_origin}-{self.protocol_version}"
        alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        schedules = [
            ["ABC", "BCA", "CAB"],
            ["BCA", "CAB", "ABC"],
            ["CAB", "ABC", "BCA"],
            ["ACB", "CBA", "BAC"],
            ["CBA", "BAC", "ACB"],
            ["BAC", "ACB", "CBA"],
        ]
        created: list[str] = []
        created_pairs: list[tuple[str, str]] = []
        with self.connect() as conn:
            run_id = self._ensure_run_in_conn(conn, run_slug, data_origin, protocol_mode)
            for index in range(count):
                while True:
                    raw = "".join(secrets.choice(alphabet) for _ in range(8))
                    code = f"TTA-{raw[:4]}-{raw[4:]}"
                    digest = code_hash(code, self.settings)
                    exists = conn.execute(
                        "SELECT 1 FROM participants WHERE code_hash=?", (digest,)
                    ).fetchone()
                    if not exists:
                        break
                participant_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO participants
                       (id, run_id, code_hash, code_hint, cohort, group_name, probe_schedule,
                        preview_mode, analysis_excluded, release_state, created_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        participant_id,
                        run_id,
                        digest,
                        code[-4:],
                        cohort,
                        group_name,
                        json.dumps(schedules[index % len(schedules)]),
                        int(preview_mode),
                        int(analysis_excluded),
                        "held" if data_origin == "formal" else "released",
                        int(time.time() * 1000),
                    ),
                )
                created.append(code)
                created_pairs.append((participant_id, code))
        for participant_id, code in created_pairs:
            self.remember_participant_login_code(participant_id, code)
        return created

    def remember_participant_login_code(self, participant_id: str, code: str) -> None:
        if not participant_id or not is_full_participant_code(code):
            return
        path = self.login_codes_path
        created = not path.exists()
        conn = sqlite3.connect(path, timeout=30)
        try:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS participant_login_codes (
                    participant_id TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    created_ms INTEGER NOT NULL
                )"""
            )
            conn.execute(
                """INSERT OR IGNORE INTO participant_login_codes
                   (participant_id, code, created_ms) VALUES (?, ?, ?)""",
                (participant_id, code, int(time.time() * 1000)),
            )
            conn.commit()
        finally:
            conn.close()
        if created:
            try:
                path.chmod(0o600)
            except OSError:
                pass

    def participant_login_code(self, participant_id: str) -> str | None:
        path = self.login_codes_path
        if not participant_id or not path.exists():
            return None
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT code FROM participant_login_codes WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        code = str(row["code"] or "")
        if not is_full_participant_code(code):
            return None
        participant = self.participant(participant_id)
        if participant is None:
            return None
        if code_hash(code, self.settings) != participant["code_hash"]:
            return None
        if not code.endswith(str(participant.get("code_hint") or "")):
            return None
        return code

    def record_account_batch(
        self,
        *,
        run_id: str,
        stage: str,
        label: str,
        cohort: str,
        group_name: str,
        data_origin: str,
        protocol_mode: str,
        participant_ids: list[str],
        is_pre_experiment: bool,
        analysis_excluded: bool = False,
        storage_namespace: str = "pre_experiments",
    ) -> str:
        if analysis_excluded and not is_pre_experiment:
            raise ValueError("analysis-excluded batches must use an isolated archive")
        if storage_namespace not in {"pre_experiments", "personal_debug"}:
            raise ValueError("invalid account batch storage namespace")
        if analysis_excluded and storage_namespace != "personal_debug":
            raise ValueError("analysis-excluded batches require personal_debug storage")
        if storage_namespace == "personal_debug" and not analysis_excluded:
            raise ValueError("personal_debug storage requires analysis exclusion")
        batch_id = str(uuid.uuid4())
        with self.connect() as conn:
            run = conn.execute(
                "SELECT slug FROM study_runs WHERE id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError("study run not found")
            if not participant_ids:
                raise ValueError("account batch cannot be empty")
            placeholders = ",".join("?" for _ in participant_ids)
            rows = conn.execute(
                f"""SELECT id, analysis_excluded FROM participants
                    WHERE run_id=? AND id IN ({placeholders})""",
                (run_id, *participant_ids),
            ).fetchall()
            if {str(row["id"]) for row in rows} != set(participant_ids):
                raise ValueError("account batch participant/run mismatch")
            if any(
                bool(row["analysis_excluded"]) != bool(analysis_excluded)
                for row in rows
            ):
                raise ValueError("account batch analysis exclusion mismatch")
            archive_relative_path = (
                str(Path(storage_namespace) / batch_id)
                if is_pre_experiment
                else None
            )
            conn.execute(
                """INSERT INTO account_batches
                   (id, run_id, stage, label, cohort, group_name, data_origin,
                    protocol_mode, requested_count, participant_ids,
                    is_pre_experiment, analysis_excluded, storage_namespace,
                    archive_relative_path, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    batch_id,
                    run_id,
                    stage,
                    label,
                    cohort,
                    group_name,
                    data_origin,
                    protocol_mode,
                    len(participant_ids),
                    json.dumps(participant_ids),
                    int(is_pre_experiment),
                    int(analysis_excluded),
                    storage_namespace,
                    archive_relative_path,
                    int(time.time() * 1000),
                ),
            )
        return batch_id

    def account_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT ab.*, r.slug AS run_slug
                   FROM account_batches ab
                   JOIN study_runs r ON r.id=ab.run_id
                   WHERE ab.id=?""",
                (batch_id,),
            ).fetchone()
        return dict(row) if row else None

    def account_batch_for_participant(
        self, participant_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT ab.*, r.slug AS run_slug
                   FROM account_batches ab
                   JOIN study_runs r ON r.id=ab.run_id
                   WHERE ab.is_pre_experiment=1
                   ORDER BY ab.created_ms DESC"""
            ).fetchall()
        for row in rows:
            try:
                participant_ids = json.loads(str(row["participant_ids"]))
            except (TypeError, json.JSONDecodeError):
                continue
            if participant_id in participant_ids:
                return dict(row)
        return None

    def participants_by_code_hint(self, hint: str) -> list[dict[str, Any]]:
        needle = str(hint or "").strip().upper()
        if len(needle) != 4:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.*, r.slug AS run_slug, r.data_origin, r.protocol_version,
                          r.protocol_sha256 AS run_protocol_sha256,
                          r.protocol_json AS run_protocol_json, r.protocol_mode
                   FROM participants p JOIN study_runs r ON r.id=p.run_id
                   WHERE p.enabled=1 AND UPPER(p.code_hint)=?""",
                (needle,),
            ).fetchall()
        return [dict(row) for row in rows]

    def typed_events_for_participant(
        self, participant_id: str, event_types: list[str]
    ) -> list[dict[str, Any]]:
        if not event_types:
            return []
        placeholders = ",".join("?" for _ in event_types)
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT id, session_id, event_type, server_ms, payload
                    FROM events
                    WHERE participant_id=? AND event_type IN ({placeholders})
                    ORDER BY server_ms, id""",
                (participant_id, *event_types),
            ).fetchall()
        return [dict(row) for row in rows]

    def interrupt_continue_summaries(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.code_hint AS code_hint,
                          e.participant_id AS participant_id,
                          e.server_ms AS server_ms,
                          e.event_type AS event_type
                   FROM events e
                   JOIN participants p ON p.id=e.participant_id
                   WHERE e.event_type IN (
                     'session.interrupt_continue_chosen',
                     'session.interrupt_restart_chosen',
                     'session.interrupted'
                   )
                   ORDER BY e.server_ms""",
            ).fetchall()
        by_pid: dict[str, dict[str, Any]] = {}
        for row in rows:
            pid = str(row["participant_id"])
            bucket = by_pid.setdefault(
                pid,
                {
                    "participant_id": pid,
                    "code_hint": str(row["code_hint"] or ""),
                    "interrupt_count": 0,
                    "continue_count": 0,
                    "restart_count": 0,
                    "continue_timestamps_ms": [],
                    "last_ms": None,
                },
            )
            kind = str(row["event_type"] or "")
            if kind == "session.interrupted":
                bucket["interrupt_count"] += 1
            elif kind == "session.interrupt_continue_chosen":
                bucket["continue_count"] += 1
                bucket["continue_timestamps_ms"].append(int(row["server_ms"] or 0))
            elif kind == "session.interrupt_restart_chosen":
                bucket["restart_count"] += 1
            bucket["last_ms"] = int(row["server_ms"] or 0)
        return sorted(
            by_pid.values(),
            key=lambda item: int(item.get("last_ms") or 0),
            reverse=True,
        )

    def participant_by_code(self, code: str) -> dict[str, Any] | None:
        digest = code_hash(code, self.settings)
        with self.connect() as conn:
            row = conn.execute(
                """SELECT p.*, r.slug AS run_slug, r.data_origin, r.protocol_version,
                          r.protocol_sha256 AS run_protocol_sha256,
                          r.protocol_json AS run_protocol_json, r.protocol_mode
                   FROM participants p JOIN study_runs r ON r.id=p.run_id
                   WHERE p.code_hash=? AND p.enabled=1""",
                (digest,),
            ).fetchone()
        return dict(row) if row else None

    def participant(self, participant_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT p.*, r.slug AS run_slug, r.data_origin, r.protocol_version,
                          r.protocol_sha256 AS run_protocol_sha256,
                          r.protocol_json AS run_protocol_json, r.protocol_mode
                   FROM participants p JOIN study_runs r ON r.id=p.run_id
                   WHERE p.id=? AND p.enabled=1""",
                (participant_id,),
            ).fetchone()
        return dict(row) if row else None

    def study_run(self, slug: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM study_runs WHERE slug=?", (slug,)).fetchone()
        return dict(row) if row else None

    def participants_in_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT p.*, r.slug AS run_slug, r.data_origin, r.protocol_version,
                          r.protocol_sha256 AS run_protocol_sha256,
                          r.protocol_json AS run_protocol_json, r.protocol_mode
                   FROM participants p JOIN study_runs r ON r.id=p.run_id
                   WHERE p.run_id=? ORDER BY p.enrollment_wave, p.assignment_id, p.created_ms""",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def configure_participant(
        self,
        participant_id: str,
        *,
        assignment_id: str,
        neutral_id: str,
        enrollment_wave: int,
        release_state: str = "held",
    ) -> None:
        if release_state not in {"held", "released", "completed", "withdrawn"}:
            raise ValueError("invalid participant release state")
        with self.connect() as conn:
            conn.execute(
                """UPDATE participants
                   SET assignment_id=?, neutral_id=?, enrollment_wave=?, release_state=?
                   WHERE id=?""",
                (assignment_id, neutral_id, enrollment_wave, release_state, participant_id),
            )

    def set_release_state(
        self, participant_id: str, release_state: str, *, actor: str, reason: str
    ) -> None:
        if release_state not in {"held", "released", "completed", "withdrawn"}:
            raise ValueError("invalid participant release state")
        with self.connect() as conn:
            participant = conn.execute(
                "SELECT run_id, release_state FROM participants WHERE id=?", (participant_id,)
            ).fetchone()
            if participant is None:
                raise ValueError("participant not found")
            conn.execute(
                "UPDATE participants SET release_state=? WHERE id=?",
                (release_state, participant_id),
            )
            conn.execute(
                """INSERT INTO pipeline_actions (id, run_id, action, actor, payload, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    participant["run_id"],
                    "participant.release_state",
                    actor,
                    json.dumps(
                        {
                            "participant_id": participant_id,
                            "from": participant["release_state"],
                            "to": release_state,
                            "reason": reason,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    int(time.time() * 1000),
                ),
            )

    def assign_prior(self, participant_id: str, prior_id: str, *, actor: str) -> None:
        with self.connect() as conn:
            participant = conn.execute(
                "SELECT run_id, prior_id FROM participants WHERE id=?", (participant_id,)
            ).fetchone()
            prior = conn.execute("SELECT * FROM shared_priors WHERE id=?", (prior_id,)).fetchone()
            if participant is None or prior is None:
                raise ValueError("participant or prior not found")
            if participant["run_id"] != prior["run_id"]:
                raise ValueError("participant and prior belong to different runs")
            if participant["prior_id"] is not None:
                raise ValueError("participant already has an immutable prior assignment")
            conn.execute(
                "UPDATE participants SET prior_id=?, prior_n=?, prior_q=? WHERE id=?",
                (prior_id, prior["donor_count"], prior["opportunity_budget"], participant_id),
            )
            conn.execute(
                """INSERT INTO pipeline_actions (id, run_id, action, actor, payload, created_ms)
                   VALUES (?, ?, 'participant.assign_prior', ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    participant["run_id"],
                    actor,
                    json.dumps(
                        {"participant_id": participant_id, "prior_id": prior_id},
                        sort_keys=True,
                    ),
                    int(time.time() * 1000),
                ),
            )

    def rebind_share_start_prior(
        self,
        participant_id: str,
        prior_id: str,
        *,
        actor: str,
        reason: str,
    ) -> dict[str, Any]:
        """Replace a stale unused start prior with the current Share chain tip.

        The first assignment remains the normal path.  This exists only for
        aborted / unstarted Share participants whose booked slot outlived the
        prefix they first saw.  A live or finished study session cannot move.
        """
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            participant = conn.execute(
                "SELECT run_id, prior_id, prior_n, prior_q FROM participants WHERE id=?",
                (participant_id,),
            ).fetchone()
            prior = conn.execute(
                "SELECT * FROM shared_priors WHERE id=?", (prior_id,)
            ).fetchone()
            if participant is None or prior is None:
                raise ValueError("participant or prior not found")
            if participant["run_id"] != prior["run_id"]:
                raise ValueError("participant and prior belong to different runs")
            previous_id = participant["prior_id"]
            if previous_id is None:
                raise ValueError("participant has no start prior to rebind")
            if str(previous_id) == str(prior_id):
                return {
                    "participant_id": participant_id,
                    "from_prior_id": str(previous_id),
                    "to_prior_id": str(prior_id),
                    "unchanged": True,
                }
            live_session = conn.execute(
                """SELECT id, state FROM sessions
                   WHERE participant_id=? AND client_role='study'
                     AND state IN ('started', 'active', 'ended', 'completed',
                                   'questionnaire')
                   LIMIT 1""",
                (participant_id,),
            ).fetchone()
            if live_session is not None:
                raise ValueError(
                    "cannot rebind start prior while a live or finished session exists"
                )
            trigger = conn.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='trigger' AND name='participant_prior_immutable'"""
            ).fetchone()
            if trigger is None or not trigger["sql"]:
                raise ValueError("prior immutability trigger missing")
            conn.execute("DROP TRIGGER participant_prior_immutable")
            updated = conn.execute(
                """UPDATE participants
                   SET prior_id=?, prior_n=?, prior_q=?
                   WHERE id=? AND prior_id=?""",
                (
                    prior_id,
                    prior["donor_count"],
                    prior["opportunity_budget"],
                    participant_id,
                    previous_id,
                ),
            )
            if int(updated.rowcount or 0) != 1:
                conn.execute(trigger["sql"])
                raise ValueError("expected exactly one start-prior rebind")
            conn.execute(trigger["sql"])
            conn.execute(
                """UPDATE model_counters
                   SET learning_opportunities=0,
                       activation_opportunities=0,
                       eligible_feedbacks=0,
                       positive_feedbacks=0,
                       negative_feedbacks=0,
                       updated_ms=?
                   WHERE participant_id=?""",
                (now_ms, participant_id),
            )
            conn.execute(
                """INSERT INTO pipeline_actions (id, run_id, action, actor, payload, created_ms)
                   VALUES (?, ?, 'participant.rebind_share_start_prior', ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    participant["run_id"],
                    actor,
                    json.dumps(
                        {
                            "participant_id": participant_id,
                            "from_prior_id": previous_id,
                            "to_prior_id": prior_id,
                            "reason": reason,
                        },
                        sort_keys=True,
                    ),
                    now_ms,
                ),
            )
        return {
            "participant_id": participant_id,
            "from_prior_id": str(previous_id),
            "to_prior_id": str(prior_id),
            "unchanged": False,
        }

    def create_shared_prior(
        self,
        record: dict[str, Any],
        *,
        allow_analysis_excluded: bool = False,
    ) -> dict[str, Any]:
        prior_id = str(record.get("id") or uuid.uuid4())
        donor_ids = [str(value) for value in record.get("donor_ids", [])]
        with self.connect() as conn:
            if donor_ids:
                placeholders = ",".join("?" for _ in donor_ids)
                donors = conn.execute(
                    f"""SELECT id, run_id, analysis_excluded FROM participants
                        WHERE id IN ({placeholders})""",
                    donor_ids,
                ).fetchall()
                if len(donors) != len(set(donor_ids)):
                    raise ValueError("shared-prior donor list contains unknown participants")
                if any(str(row["run_id"]) != str(record["run_id"]) for row in donors):
                    raise ValueError("shared-prior donors must belong to the target run")
                if not allow_analysis_excluded and any(
                    bool(row["analysis_excluded"]) for row in donors
                ):
                    raise ValueError(
                        "analysis-excluded participants cannot contribute to a shared prior"
                    )
            conn.execute(
                """INSERT INTO shared_priors
                   (id, run_id, donor_count, opportunity_budget, status, algorithm, donor_ids,
                    encoder_path, main_path, encoder_sha256, main_sha256, manifest, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prior_id,
                    record["run_id"],
                    int(record["donor_count"]),
                    int(record["opportunity_budget"]),
                    record.get("status", "candidate"),
                    record["algorithm"],
                    json.dumps(donor_ids, sort_keys=True),
                    record["encoder_path"],
                    record["main_path"],
                    record["encoder_sha256"],
                    record["main_sha256"],
                    json.dumps(record.get("manifest", {}), ensure_ascii=False, sort_keys=True),
                    int(record.get("created_ms") or time.time() * 1000),
                ),
            )
            row = conn.execute("SELECT * FROM shared_priors WHERE id=?", (prior_id,)).fetchone()
        return dict(row)

    def shared_prior(self, prior_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM shared_priors WHERE id=?", (prior_id,)).fetchone()
        return dict(row) if row else None

    def shared_priors(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT * FROM shared_priors WHERE run_id=?
                   ORDER BY donor_count, opportunity_budget""",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def select_shared_prior(self, run_id: str, prior_id: str, *, actor: str, reason: str) -> None:
        with self.connect() as conn:
            prior = conn.execute(
                "SELECT * FROM shared_priors WHERE id=? AND run_id=?", (prior_id, run_id)
            ).fetchone()
            if prior is None:
                raise ValueError("shared prior not found in run")
            now = int(time.time() * 1000)
            conn.execute(
                "UPDATE shared_priors SET status='retired' WHERE run_id=? AND status='selected'",
                (run_id,),
            )
            conn.execute(
                "UPDATE shared_priors SET status='selected', selected_ms=? WHERE id=?",
                (now, prior_id),
            )
            conn.execute(
                """INSERT INTO pipeline_actions (id, run_id, action, actor, payload, created_ms)
                   VALUES (?, ?, 'shared_prior.select', ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    run_id,
                    actor,
                    json.dumps({"prior_id": prior_id, "reason": reason}, ensure_ascii=False),
                    now,
                ),
            )

    def record_model_snapshot(self, record: dict[str, Any]) -> dict[str, Any]:
        snapshot_id = str(record.get("id") or uuid.uuid4())
        with self.connect() as conn:
            run_id = self._run_id(conn, record["participant_id"])
            conn.execute(
                """INSERT INTO model_snapshots
                   (id, run_id, participant_id, session_id, source, opportunity_budget,
                    budget_kind, eligible_feedbacks, positive_feedbacks, negative_feedbacks,
                    metadata, base_prior_id, encoder_path, main_path, encoder_sha256,
                    main_sha256, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    run_id,
                    record["participant_id"],
                    record.get("session_id"),
                    record["source"],
                    int(record["opportunity_budget"]),
                    str(record.get("budget_kind") or "activation_opportunity"),
                    int(record.get("eligible_feedbacks") or 0),
                    int(record.get("positive_feedbacks") or 0),
                    int(record.get("negative_feedbacks") or 0),
                    json.dumps(record.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
                    record.get("base_prior_id"),
                    record["encoder_path"],
                    record["main_path"],
                    record["encoder_sha256"],
                    record["main_sha256"],
                    int(record.get("created_ms") or time.time() * 1000),
                ),
            )
            row = conn.execute("SELECT * FROM model_snapshots WHERE id=?", (snapshot_id,)).fetchone()
        return dict(row)

    def learning_opportunity_count(self, participant_id: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT learning_opportunities FROM model_counters WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        return int(row["learning_opportunities"]) if row else 0

    def next_learning_opportunity(self, participant_id: str) -> int:
        with self.connect() as conn:
            run_id = self._run_id(conn, participant_id)
            now = int(time.time() * 1000)
            conn.execute(
                """INSERT INTO model_counters
                   (participant_id, run_id, learning_opportunities,
                    activation_opportunities, updated_ms)
                   VALUES (?, ?, 1, 1, ?)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     learning_opportunities=model_counters.learning_opportunities + 1,
                     activation_opportunities=model_counters.activation_opportunities + 1,
                     updated_ms=excluded.updated_ms""",
                (participant_id, run_id, now),
            )
            row = conn.execute(
                "SELECT learning_opportunities FROM model_counters WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        return int(row["learning_opportunities"])

    def feedback_counts(self, participant_id: str) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT activation_opportunities, eligible_feedbacks,
                          positive_feedbacks, negative_feedbacks
                   FROM model_counters WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
        if row is None:
            return {
                "activation_opportunities": 0,
                "eligible_feedbacks": 0,
                "positive_feedbacks": 0,
                "negative_feedbacks": 0,
            }
        return {name: int(row[name]) for name in row.keys()}

    def calibration_state(
        self,
        participant_id: str,
        *,
        calibration_version: str,
        protocol_sha256: str,
        initial_logit_bias: float,
    ) -> dict[str, Any]:
        now = int(time.time() * 1000)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run_id = self._run_id(conn, participant_id)
            conn.execute(
                """INSERT INTO participant_calibrations
                   (participant_id, run_id, calibration_version, protocol_sha256,
                    personal_logit_bias, updated_ms)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(participant_id) DO NOTHING""",
                (
                    participant_id,
                    run_id,
                    calibration_version,
                    protocol_sha256,
                    float(initial_logit_bias),
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM participant_calibrations WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Participant calibration state was not created")
            if (
                row["run_id"] != run_id
                or row["calibration_version"] != calibration_version
                or row["protocol_sha256"] != protocol_sha256
            ):
                raise ValueError(
                    "Participant calibration state does not match the immutable run protocol"
                )
        return dict(row)

    def update_calibration_state(
        self,
        participant_id: str,
        *,
        calibration_version: str,
        protocol_sha256: str,
        expected_update_count: int,
        personal_logit_bias: float,
        label: int,
    ) -> dict[str, Any]:
        if label not in {0, 1}:
            raise ValueError("calibration label must be 0 or 1")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = int(time.time() * 1000)
            cursor = conn.execute(
                """UPDATE participant_calibrations
                   SET personal_logit_bias=?,
                       update_count=update_count + 1,
                       positive_updates=positive_updates + ?,
                       negative_updates=negative_updates + ?,
                       updated_ms=?
                   WHERE participant_id=?
                     AND calibration_version=?
                     AND protocol_sha256=?
                     AND update_count=?""",
                (
                    float(personal_logit_bias),
                    int(label == 1),
                    int(label == 0),
                    now,
                    participant_id,
                    calibration_version,
                    protocol_sha256,
                    int(expected_update_count),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    "Concurrent or protocol-mismatched participant calibration update"
                )
            row = conn.execute(
                "SELECT * FROM participant_calibrations WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("Participant calibration state disappeared after update")
        return dict(row)

    def next_activation_opportunity(self, participant_id: str) -> int:
        with self.connect() as conn:
            run_id = self._run_id(conn, participant_id)
            now = int(time.time() * 1000)
            conn.execute(
                """INSERT INTO model_counters
                   (participant_id, run_id, activation_opportunities, updated_ms)
                   VALUES (?, ?, 1, ?)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     activation_opportunities=model_counters.activation_opportunities + 1,
                     updated_ms=excluded.updated_ms""",
                (participant_id, run_id, now),
            )
            row = conn.execute(
                "SELECT activation_opportunities FROM model_counters WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        return int(row["activation_opportunities"])

    def next_eligible_feedback(self, participant_id: str, label: int) -> dict[str, int]:
        if label not in {0, 1}:
            raise ValueError("feedback label must be 0 or 1")
        positive = 1 if label == 1 else 0
        negative = 1 if label == 0 else 0
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run_id = self._run_id(conn, participant_id)
            now = int(time.time() * 1000)
            conn.execute(
                """INSERT INTO model_counters
                   (participant_id, run_id, eligible_feedbacks, positive_feedbacks,
                    negative_feedbacks, updated_ms)
                   VALUES (?, ?, 1, ?, ?, ?)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     eligible_feedbacks=model_counters.eligible_feedbacks + 1,
                     positive_feedbacks=model_counters.positive_feedbacks + excluded.positive_feedbacks,
                     negative_feedbacks=model_counters.negative_feedbacks + excluded.negative_feedbacks,
                     updated_ms=excluded.updated_ms""",
                (participant_id, run_id, positive, negative, now),
            )
            row = conn.execute(
                """SELECT activation_opportunities, eligible_feedbacks,
                          positive_feedbacks, negative_feedbacks
                   FROM model_counters WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
        return {name: int(row[name]) for name in row.keys()}

    def model_snapshots(
        self,
        run_id: str,
        *,
        source: str | None = None,
        budget: int | None = None,
        budget_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["run_id=?"]
        values: list[Any] = [run_id]
        if source is not None:
            clauses.append("source=?")
            values.append(source)
        if budget is not None:
            clauses.append("opportunity_budget=?")
            values.append(int(budget))
        if budget_kind is not None:
            clauses.append("budget_kind=?")
            values.append(budget_kind)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM model_snapshots WHERE {' AND '.join(clauses)} ORDER BY created_ms",
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def pipeline_actions(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pipeline_actions WHERE run_id=? ORDER BY created_ms", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def record_pipeline_action(
        self, run_id: str, action: str, *, actor: str, payload: dict[str, Any]
    ) -> str:
        action_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO pipeline_actions (id, run_id, action, actor, payload, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    action_id,
                    run_id,
                    action,
                    actor,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    int(time.time() * 1000),
                ),
            )
        return action_id

    def create_prior_probe_assignments(
        self,
        participant_id: str,
        day: int,
        prior_ids: list[str],
    ) -> list[dict[str, Any]]:
        if day not in {1, 2, 3}:
            raise ValueError("probe day must be between 1 and 3")
        if not prior_ids or len(prior_ids) != len(set(prior_ids)):
            raise ValueError("probe prior ids must be a non-empty unique list")
        now = int(time.time() * 1000)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run_id = self._run_id(conn, participant_id)
            existing = conn.execute(
                """SELECT ppa.*, sp.donor_count, sp.opportunity_budget
                   FROM prior_probe_assignments ppa
                   JOIN shared_priors sp ON sp.id=ppa.prior_id
                   WHERE ppa.participant_id=? AND ppa.day=?
                   ORDER BY ppa.order_index""",
                (participant_id, day),
            ).fetchall()
            if existing:
                if [str(row["prior_id"]) for row in existing] != prior_ids:
                    raise ValueError("participant already has a different immutable probe plan")
                return [dict(row) for row in existing]
            rows: list[dict[str, Any]] = []
            for order_index, prior_id in enumerate(prior_ids, start=1):
                prior = conn.execute(
                    "SELECT * FROM shared_priors WHERE id=? AND run_id=?",
                    (prior_id, run_id),
                ).fetchone()
                if prior is None:
                    raise ValueError("probe prior is missing or belongs to another run")
                probe_id = str(uuid.uuid4())
                probe_token = secrets.token_urlsafe(18)
                material = "|".join(
                    [run_id, participant_id, str(day), str(order_index), prior_id, probe_token]
                )
                assignment_sha256 = hashlib.sha256(material.encode("utf-8")).hexdigest()
                conn.execute(
                    """INSERT INTO prior_probe_assignments
                       (id, run_id, participant_id, day, probe_token, order_index, prior_id,
                        assignment_sha256, created_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        probe_id,
                        run_id,
                        participant_id,
                        day,
                        probe_token,
                        order_index,
                        prior_id,
                        assignment_sha256,
                        now,
                    ),
                )
                rows.append(
                    {
                        "id": probe_id,
                        "run_id": run_id,
                        "participant_id": participant_id,
                        "day": day,
                        "probe_token": probe_token,
                        "order_index": order_index,
                        "prior_id": prior_id,
                        "assignment_sha256": assignment_sha256,
                        "created_ms": now,
                        "donor_count": int(prior["donor_count"]),
                        "opportunity_budget": int(prior["opportunity_budget"]),
                    }
                )
        return rows

    def prior_probe_assignments(
        self,
        *,
        participant_id: str | None = None,
        run_id: str | None = None,
        day: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if participant_id is not None:
            clauses.append("ppa.participant_id=?")
            values.append(participant_id)
        if run_id is not None:
            clauses.append("ppa.run_id=?")
            values.append(run_id)
        if day is not None:
            clauses.append("ppa.day=?")
            values.append(day)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT ppa.*, sp.donor_count, sp.opportunity_budget,
                           sp.encoder_path, sp.main_path, sp.encoder_sha256, sp.main_sha256
                    FROM prior_probe_assignments ppa
                    JOIN shared_priors sp ON sp.id=ppa.prior_id
                    {where}
                    ORDER BY ppa.participant_id, ppa.day, ppa.order_index""",
                values,
            ).fetchall()
        return [dict(row) for row in rows]

    def prior_probe_assignment(
        self, participant_id: str, day: int, probe_token: str
    ) -> dict[str, Any] | None:
        rows = self.prior_probe_assignments(participant_id=participant_id, day=day)
        return next((row for row in rows if row["probe_token"] == probe_token), None)

    def latest_consent(self, participant_id: str) -> dict[str, Any] | None:
        with self.connect_consent() as conn:
            row = conn.execute(
                "SELECT * FROM consents WHERE participant_id=? ORDER BY signed_ms DESC LIMIT 1",
                (participant_id,),
            ).fetchone()
        return dict(row) if row else None

    def admin_consent_names(
        self, participant_ids: list[str] | tuple[str, ...] | set[str]
    ) -> dict[str, str]:
        """Latest typed consent name for the authenticated admin dashboard."""
        ids = list(dict.fromkeys(str(value) for value in participant_ids if value))
        if not ids:
            return {}
        names: dict[str, str] = {}
        with self.connect_consent() as conn:
            for offset in range(0, len(ids), 800):
                chunk = ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""SELECT participant_id, typed_name, signed_ms
                        FROM consents
                        WHERE participant_id IN ({placeholders})
                          AND typed_name != ''
                        ORDER BY signed_ms DESC""",
                    chunk,
                ).fetchall()
                for row in rows:
                    participant_id = str(row["participant_id"] or "")
                    if not participant_id or participant_id in names:
                        continue
                    name = re.sub(
                        r"\s+", " ", str(row["typed_name"] or "")
                    ).strip()
                    if name:
                        names[participant_id] = name
        return names

    def admin_window_tap_counts(
        self,
        *,
        run_ids: set[str] | list[str] | tuple[str, ...] | None = None,
        participant_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Per-subject tap/interrupt counts across the five timed windows."""
        run_list = list(dict.fromkeys(str(value) for value in (run_ids or []) if value))
        participant_list = list(
            dict.fromkeys(str(value) for value in (participant_ids or []) if value)
        )
        if not run_list and not participant_list:
            return {}

        def empty_record() -> dict[str, Any]:
            return {
                "windows": [
                    {"window": index, "taps": 0, "interrupts": 0}
                    for index in range(1, 6)
                ],
                "tap_total": 0,
                "interrupt_total": 0,
            }

        records: dict[str, dict[str, Any]] = {
            participant_id: empty_record() for participant_id in participant_list
        }
        clauses: list[str] = [
            "event_type IN ('tap.manual_activate','tap.interrupt')"
        ]
        params: list[Any] = []
        if run_list:
            clauses.append(
                f"run_id IN ({','.join('?' for _ in run_list)})"
            )
            params.extend(run_list)
        if participant_list:
            clauses.append(
                f"participant_id IN ({','.join('?' for _ in participant_list)})"
            )
            params.extend(participant_list)
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT participant_id, event_type, payload
                    FROM events
                    WHERE {' AND '.join(clauses)}
                    ORDER BY id""",
                params,
            ).fetchall()
        for row in rows:
            participant_id = str(row["participant_id"] or "")
            if not participant_id:
                continue
            try:
                payload = json.loads(str(row["payload"] or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            if payload.get("cancelled") is True:
                continue
            raw_window = payload.get(
                "active_block_index",
                payload.get("block", payload.get("window")),
            )
            try:
                window = int(raw_window)
            except (TypeError, ValueError):
                continue
            if window not in {1, 2, 3, 4, 5}:
                continue
            record = records.setdefault(participant_id, empty_record())
            slot = record["windows"][window - 1]
            if str(row["event_type"]) == "tap.manual_activate":
                slot["taps"] += 1
                record["tap_total"] += 1
            else:
                slot["interrupts"] += 1
                record["interrupt_total"] += 1
        return records

    def find_participant_ids_by_consent_name(self, name: str) -> list[str]:
        needle = re.sub(r"\s+", " ", (name or "").strip()).casefold()
        if len(needle) < 2:
            return []
        with self.connect_consent() as conn:
            rows = conn.execute(
                """SELECT participant_id, typed_name, signed_ms
                   FROM consents
                   WHERE typed_name != ''
                   ORDER BY signed_ms DESC"""
            ).fetchall()
        seen: set[str] = set()
        matches: list[str] = []
        for row in rows:
            participant_id = str(row["participant_id"] or "")
            if not participant_id or participant_id in seen:
                continue
            seen.add(participant_id)
            stored = re.sub(r"\s+", " ", str(row["typed_name"] or "").strip()).casefold()
            if stored == needle:
                matches.append(participant_id)
        return matches

    def consent(self, consent_id: str) -> dict[str, Any] | None:
        with self.connect_consent() as conn:
            row = conn.execute(
                "SELECT * FROM consents WHERE id=?",
                (consent_id,),
            ).fetchone()
        return dict(row) if row else None

    def consent_chain_tip(self) -> dict[str, Any]:
        with self.connect_consent() as conn:
            row = conn.execute(
                """SELECT ledger_seq, record_sha256 FROM consents
                   WHERE ledger_seq IS NOT NULL AND record_sha256 != ''
                   ORDER BY ledger_seq DESC LIMIT 1"""
            ).fetchone()
        if row is None:
            return {"ledger_seq": 0, "record_sha256": ""}
        return dict(row)

    def save_consent(self, record: dict[str, Any]) -> str:
        consent_id = str(record.get("id") or uuid.uuid4())
        with self.connect_consent() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if record.get("ledger_seq") is not None:
                tip = conn.execute(
                    """SELECT ledger_seq, record_sha256 FROM consents
                       WHERE ledger_seq IS NOT NULL AND record_sha256 != ''
                       ORDER BY ledger_seq DESC LIMIT 1"""
                ).fetchone()
                actual_seq = int(tip["ledger_seq"]) if tip else 0
                actual_hash = str(tip["record_sha256"]) if tip else ""
                if int(record["ledger_seq"]) != actual_seq + 1:
                    raise ValueError("Consent evidence ledger sequence changed")
                if str(record.get("previous_record_sha256") or "") != actual_hash:
                    raise ValueError("Consent evidence chain tip changed")
            conn.execute(
                """INSERT INTO consents
                   (id, run_id, participant_id, version, document_hash, document_json,
                    participation, raw_audio,
                    future_sharing, internal_reuse, controlled_sharing, public_derived,
                    public_audio, ai_training, commercial_research, typed_name, affirmation, signed_ms,
                    ip_hash, user_agent, client_nonce, auth_context_sha256,
                    evidence_json, evidence_sha256, evidence_key_id, ledger_seq,
                    previous_record_sha256, record_sha256, server_instance)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    consent_id,
                    record["run_id"],
                    record["participant_id"],
                    record["version"],
                    record["document_hash"],
                    record["document_json"],
                    int(record["participation"]),
                    int(record["raw_audio"]),
                    int(record["future_sharing"]),
                    int(record.get("internal_reuse", False)),
                    int(record.get("controlled_sharing", False)),
                    int(record.get("public_derived", False)),
                    int(record.get("public_audio", False)),
                    int(record.get("ai_training", False)),
                    int(record.get("commercial_research", False)),
                    str(record.get("typed_name") or ""),
                    record["affirmation"],
                    record["signed_ms"],
                    record["ip_hash"],
                    record["user_agent"],
                    record.get("client_nonce", ""),
                    record.get("auth_context_sha256", ""),
                    record.get("evidence_json", ""),
                    record.get("evidence_sha256", ""),
                    record.get("evidence_key_id", ""),
                    record.get("ledger_seq"),
                    record.get("previous_record_sha256", ""),
                    record.get("record_sha256", ""),
                    record.get("server_instance", ""),
                ),
            )
        return consent_id

    def start_session(self, participant_id: str, day: int, client_role: str) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE participant_id=? AND day=? AND client_role=?",
                (participant_id, day, client_role),
            ).fetchone()
            if row:
                return dict(row)
            session_id = str(uuid.uuid4())
            now = int(time.time() * 1000)
            run_id = self._run_id(conn, participant_id)
            conn.execute(
                """INSERT INTO sessions
                   (id, run_id, participant_id, day, client_role, started_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, run_id, participant_id, day, client_role, now),
            )
            row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row)

    def reopen_interrupted_aborted_day(
        self, participant_id: str, day: int
    ) -> dict[str, Any] | None:
        """Reopen an aborted day so an interrupt checkpoint can resume.

        Keeps the original started/activated timestamps and session IDs so
        continue-from-interrupt uses the same cursor. Does not rewrite
        orchestration, models, or participant rows.
        """
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE participant_id=? AND day=?""",
                (participant_id, int(day)),
            ).fetchall()
            if not rows:
                return None
            sessions = [dict(row) for row in rows]
            study = next(
                (row for row in sessions if row["client_role"] == "study"),
                None,
            )
            if study is None or str(study.get("state") or "") != "aborted":
                return None
            if any(
                str(row.get("state") or "") in {"completed", "ended", "questionnaire"}
                for row in sessions
            ):
                return None
            conn.execute(
                """UPDATE sessions
                   SET state=CASE
                         WHEN activated_ms IS NULL THEN 'started'
                         ELSE 'active'
                       END,
                       ended_ms=NULL
                   WHERE participant_id=? AND day=? AND state='aborted'""",
                (participant_id, int(day)),
            )
            reopened = conn.execute(
                """SELECT * FROM sessions
                   WHERE participant_id=? AND day=? AND client_role='study'""",
                (participant_id, int(day)),
            ).fetchone()
        return dict(reopened) if reopened else None

    def restart_aborted_session_day(
        self, participant_id: str, day: int
    ) -> dict[str, Any] | None:
        """Restart an aborted single-session attempt without deleting its audit trail.

        The orchestration layer must first prove that a newly verified
        appointment supersedes the old abort.  Existing row identifiers are
        retained so historical events remain attributable, while the session
        timing boundary is reset for the new attempt.
        """
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE participant_id=? AND day=?""",
                (participant_id, int(day)),
            ).fetchall()
            if not rows:
                return None
            sessions = [dict(row) for row in rows]
            study = next(
                (row for row in sessions if row["client_role"] == "study"),
                None,
            )
            if study is None or str(study.get("state") or "") != "aborted":
                return None
            if any(
                str(row.get("state") or "") in {"completed", "ended", "questionnaire"}
                for row in sessions
            ):
                return None
            recovery = conn.execute(
                """SELECT 1 FROM final_questionnaire_recoveries
                   WHERE participant_id=? LIMIT 1""",
                (participant_id,),
            ).fetchone()
            if recovery is not None:
                return None
            conn.execute(
                """UPDATE sessions
                   SET state='started', started_ms=?, activated_ms=NULL, ended_ms=NULL
                   WHERE participant_id=? AND day=? AND state='aborted'""",
                (now_ms, participant_id, int(day)),
            )
            restarted = conn.execute(
                """SELECT * FROM sessions
                   WHERE participant_id=? AND day=? AND client_role='study'""",
                (participant_id, int(day)),
            ).fetchone()
        return dict(restarted) if restarted else None

    def sessions(self, participant_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE participant_id=? ORDER BY day, client_role",
                (participant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def activate_paired_sessions(
        self,
        participant_id: str,
        study_session_id: str,
        companion_session_id: str,
        *,
        max_concurrent: int | None = None,
        active_window_ms: int = 50 * 60 * 1000,
    ) -> dict[str, Any]:
        with self.connect() as conn:
            # Serialize the count-and-activate boundary across requests and
            # processes. An advisory capacity GET alone cannot prevent two
            # participants from both observing the sixth slot as available.
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE participant_id=? AND id IN (?, ?)""",
                (participant_id, study_session_id, companion_session_id),
            ).fetchall()
            sessions = {str(row["client_role"]): dict(row) for row in rows}
            study = sessions.get("study")
            companion = sessions.get("companion")
            if (
                study is None
                or companion is None
                or study["id"] != study_session_id
                or companion["id"] != companion_session_id
                or study["day"] != companion["day"]
                or study["state"] in {"ended", "completed", "aborted"}
                or companion["state"] in {"ended", "completed", "aborted"}
            ):
                raise ValueError("Study and companion sessions are not an active same-day pair")
            if (
                study["state"] == "active"
                or companion["state"] == "active"
                or study["activated_ms"] is not None
                or companion["activated_ms"] is not None
            ):
                raise ValueError("This same-day pair has already been activated")
            activated_ms = int(time.time() * 1000)
            if max_concurrent is not None:
                cutoff_ms = activated_ms - max(60_000, int(active_window_ms))
                active_count = int(
                    conn.execute(
                        """SELECT COUNT(*) AS count
                           FROM sessions
                           WHERE client_role='study'
                             AND state='active'
                             AND activated_ms IS NOT NULL
                             AND activated_ms >= ?
                             AND ended_ms IS NULL
                             AND id != ?""",
                        (cutoff_ms, study_session_id),
                    ).fetchone()["count"]
                )
                if active_count >= max(1, int(max_concurrent)):
                    raise StudyCapacityExceeded(
                        f"active study sessions {active_count} >= {max_concurrent}"
                    )
            conn.execute(
                """UPDATE sessions SET state='active', activated_ms=COALESCE(activated_ms, ?)
                   WHERE participant_id=? AND id IN (?, ?)""",
                (activated_ms, participant_id, study_session_id, companion_session_id),
            )
        return {
            "day": int(study["day"]),
            "study_session_id": study_session_id,
            "companion_session_id": companion_session_id,
            "activated_ms": activated_ms,
            "state": "active",
        }

    def upsert_companion_presence(
        self,
        participant_id: str,
        companion_session_id: str,
        day: int,
        *,
        ready: bool = True,
        last_seen_ms: int | None = None,
    ) -> dict[str, Any]:
        seen_ms = int(last_seen_ms if last_seen_ms is not None else time.time() * 1000)
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO companion_presence
                   (participant_id, companion_session_id, day, last_seen_ms, ready)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(participant_id) DO UPDATE SET
                     companion_session_id=excluded.companion_session_id,
                     day=excluded.day,
                     last_seen_ms=excluded.last_seen_ms,
                     ready=excluded.ready""",
                (
                    participant_id,
                    companion_session_id,
                    int(day),
                    seen_ms,
                    int(bool(ready)),
                ),
            )
            row = conn.execute(
                "SELECT * FROM companion_presence WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        return dict(row)

    def live_companion_presence(
        self, participant_id: str, *, timeout_ms: int
    ) -> dict[str, Any] | None:
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            row = conn.execute(
                """SELECT * FROM companion_presence WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
        if row is None:
            return None
        if not int(row["ready"] or 0):
            return None
        if now_ms - int(row["last_seen_ms"]) > timeout_ms:
            return None
        return dict(row)

    def bind_live_companion_for_activation(
        self,
        participant_id: str,
        study_session_id: str,
        companion_session_id: str,
        day: int,
    ) -> str:
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            existing = conn.execute(
                """SELECT * FROM pair_intervals
                   WHERE participant_id=? AND study_session_id=?
                     AND companion_session_id=? AND ended_ms IS NULL
                   ORDER BY started_ms DESC LIMIT 1""",
                (participant_id, study_session_id, companion_session_id),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    """UPDATE pair_intervals
                       SET study_last_seen_ms=?, companion_last_seen_ms=?,
                           companion_ready=1
                       WHERE id=?""",
                    (now_ms, now_ms, existing["id"]),
                )
                return str(existing["id"])
        interval_id = self.open_pair_interval(
            participant_id, day, study_session_id, companion_session_id
        )
        self.update_pair_role_presence(interval_id, "study", now_ms)
        self.update_pair_role_presence(
            interval_id, "companion", now_ms, companion_ready=True
        )
        return interval_id

    def open_pair_interval(
        self,
        participant_id: str,
        day: int,
        study_session_id: str,
        companion_session_id: str,
    ) -> str:
        interval_id = str(uuid.uuid4())
        with self.connect() as conn:
            run_id = self._run_id(conn, participant_id)
            started_ms = int(time.time() * 1000)
            conn.execute(
                """UPDATE pair_intervals
                   SET ended_ms=?, end_reason='server_pair_state_recovered'
                   WHERE participant_id=? AND day=? AND ended_ms IS NULL""",
                (started_ms, participant_id, day),
            )
            conn.execute(
                """INSERT INTO pair_intervals
                   (id, run_id, participant_id, day, study_session_id, companion_session_id, started_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    interval_id,
                    run_id,
                    participant_id,
                    day,
                    study_session_id,
                    companion_session_id,
                    started_ms,
                ),
            )
        return interval_id

    def pair_interval_companion_ready(
        self,
        participant_id: str,
        companion_session_id: str,
        *,
        study_session_id: str | None = None,
    ) -> bool:
        with self.connect() as conn:
            if study_session_id:
                row = conn.execute(
                    """SELECT companion_ready FROM pair_intervals
                       WHERE participant_id=? AND study_session_id=?
                         AND companion_session_id=? AND ended_ms IS NULL
                       ORDER BY started_ms DESC LIMIT 1""",
                    (participant_id, study_session_id, companion_session_id),
                ).fetchone()
            else:
                row = conn.execute(
                    """SELECT companion_ready FROM pair_intervals
                       WHERE participant_id=? AND companion_session_id=?
                         AND ended_ms IS NULL
                       ORDER BY started_ms DESC LIMIT 1""",
                    (participant_id, companion_session_id),
                ).fetchone()
            return bool(row and int(row["companion_ready"] or 0))

    def update_pair_role_presence(
        self,
        interval_id: str,
        role: str,
        last_seen_ms: int,
        *,
        companion_ready: bool | None = None,
    ) -> None:
        if role == "study":
            column = "study_last_seen_ms"
        elif role == "companion":
            column = "companion_last_seen_ms"
        else:
            raise ValueError(f"Unknown pair role: {role}")
        with self.connect() as conn:
            if companion_ready is None:
                conn.execute(
                    f"UPDATE pair_intervals SET {column}=? WHERE id=? AND ended_ms IS NULL",
                    (int(last_seen_ms), interval_id),
                )
                return
            conn.execute(
                f"""UPDATE pair_intervals
                    SET {column}=?, companion_ready=?
                    WHERE id=? AND ended_ms IS NULL""",
                (int(last_seen_ms), int(companion_ready), interval_id),
            )

    def activation_ready_interval(
        self,
        participant_id: str,
        study_session_id: str,
        *,
        timeout_ms: int,
        require_qualified: bool = False,
    ) -> dict[str, Any] | None:
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            row = conn.execute(
                """SELECT * FROM pair_intervals
                   WHERE participant_id=? AND study_session_id=? AND ended_ms IS NULL""",
                (participant_id, study_session_id),
            ).fetchone()
            if row is None:
                return None
            if require_qualified and not int(row["qualified"]):
                return None
            study_seen = row["study_last_seen_ms"]
            companion_seen = row["companion_last_seen_ms"]
            if study_seen is None or companion_seen is None:
                return None
            if now_ms - int(study_seen) > timeout_ms:
                return None
            if now_ms - int(companion_seen) > timeout_ms:
                return None
            if not int(row["companion_ready"]):
                return None
            sessions = conn.execute(
                """SELECT id, day, client_role FROM sessions
                   WHERE participant_id=? AND id IN (?, ?)""",
                (participant_id, study_session_id, row["companion_session_id"]),
            ).fetchall()
            by_role = {str(item["client_role"]): item for item in sessions}
            study = by_role.get("study")
            companion = by_role.get("companion")
            if (
                study is None
                or companion is None
                or int(study["day"]) != int(companion["day"])
                or int(study["day"]) != int(row["day"])
            ):
                return None
            return dict(row)

    def close_pair_interval(self, interval_id: str, reason: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE pair_intervals SET ended_ms=?, end_reason=?
                   WHERE id=? AND ended_ms IS NULL""",
                (int(time.time() * 1000), reason[:80], interval_id),
            )

    def qualify_pair_interval(
        self,
        interval_id: str,
        participant_id: str,
        study_session_id: str,
        companion_session_id: str,
        qualified_ms: int,
    ) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM pair_intervals
                   WHERE id=? AND participant_id=? AND study_session_id=?
                     AND companion_session_id=?""",
                (
                    interval_id,
                    participant_id,
                    study_session_id,
                    companion_session_id,
                ),
            ).fetchone()
            if row is None or row["ended_ms"] is not None:
                raise ValueError("The activation pair interval is no longer active")
            existing_pairs = conn.execute(
                """SELECT DISTINCT study_session_id, companion_session_id
                   FROM pair_intervals
                   WHERE participant_id=? AND day=? AND qualified=1""",
                (participant_id, int(row["day"])),
            ).fetchall()
            if any(
                str(existing["study_session_id"]) != study_session_id
                or str(existing["companion_session_id"]) != companion_session_id
                for existing in existing_pairs
            ):
                raise ValueError(
                    "This day already has a different qualified device pair"
                )
            conn.execute(
                """UPDATE pair_intervals
                   SET qualified=1, qualified_ms=?, companion_ready=1
                   WHERE id=?""",
                (int(qualified_ms), interval_id),
            )
            updated = conn.execute(
                "SELECT * FROM pair_intervals WHERE id=?", (interval_id,)
            ).fetchone()
        return dict(updated)

    def qualify_reconnected_pair_interval(
        self,
        interval_id: str,
        participant_id: str,
        study_session_id: str,
        companion_session_id: str,
    ) -> dict[str, Any] | None:
        """Qualify a reconnect only for the exact pair activated earlier that day."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            interval = conn.execute(
                """SELECT * FROM pair_intervals
                   WHERE id=? AND participant_id=? AND study_session_id=?
                     AND companion_session_id=?""",
                (
                    interval_id,
                    participant_id,
                    study_session_id,
                    companion_session_id,
                ),
            ).fetchone()
            if interval is None or interval["ended_ms"] is not None or interval["qualified"]:
                return None

            rows = conn.execute(
                """SELECT * FROM sessions
                   WHERE participant_id=? AND id IN (?, ?)""",
                (participant_id, study_session_id, companion_session_id),
            ).fetchall()
            sessions = {str(row["client_role"]): row for row in rows}
            study = sessions.get("study")
            companion = sessions.get("companion")
            if (
                study is None
                or companion is None
                or str(study["id"]) != study_session_id
                or str(companion["id"]) != companion_session_id
                or int(study["day"]) != int(companion["day"])
                or study["state"] != "active"
                or companion["state"] != "active"
                or study["activated_ms"] is None
                or companion["activated_ms"] is None
            ):
                return None

            prior = conn.execute(
                """SELECT study_session_id, companion_session_id
                   FROM pair_intervals
                   WHERE participant_id=? AND day=? AND qualified=1
                   ORDER BY qualified_ms LIMIT 1""",
                (participant_id, int(interval["day"])),
            ).fetchone()
            if prior is None:
                return None
            if (
                str(prior["study_session_id"]) != study_session_id
                or str(prior["companion_session_id"]) != companion_session_id
            ):
                raise ValueError("Reconnect does not match the originally activated pair")

            qualified_ms = int(time.time() * 1000)
            conn.execute(
                """UPDATE pair_intervals
                   SET qualified=1, qualified_ms=?, companion_ready=1
                   WHERE id=?""",
                (qualified_ms, interval_id),
            )
            updated = conn.execute(
                "SELECT * FROM pair_intervals WHERE id=?", (interval_id,)
            ).fetchone()
        return dict(updated)

    def pair_interval(self, interval_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM pair_intervals WHERE id=?", (interval_id,)
            ).fetchone()
        return dict(row) if row else None

    def pair_intervals(self, participant_id: str, day: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM pair_intervals WHERE participant_id=?"
        params: tuple[Any, ...] = (participant_id,)
        if day is not None:
            query += " AND day=?"
            params += (day,)
        query += " ORDER BY started_ms"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def quality_history(self, participant_id: str, kind: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT q.kind, q.passed, q.metrics, q.created_ms, s.day, q.session_id
                   FROM quality_checks q
                   LEFT JOIN sessions s ON s.id=q.session_id
                   WHERE q.participant_id=? AND q.kind=?
                   ORDER BY q.created_ms DESC, q.id DESC""",
                (participant_id, kind),
            ).fetchall()
        return [dict(row) for row in rows]

    def session(self, session_id: str, participant_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id=? AND participant_id=?",
                (session_id, participant_id),
            ).fetchone()
        return dict(row) if row else None

    def end_session(self, session_id: str, participant_id: str) -> None:
        with self.connect() as conn:
            session = conn.execute(
                "SELECT day FROM sessions WHERE id=? AND participant_id=?",
                (session_id, participant_id),
            ).fetchone()
            if session is None:
                return
            conn.execute(
                """UPDATE sessions SET state='completed', ended_ms=?
                   WHERE participant_id=? AND day=?""",
                (int(time.time() * 1000), participant_id, int(session["day"])),
            )

    def abort_session(self, session_id: str, participant_id: str) -> None:
        now = int(time.time() * 1000)
        with self.connect() as conn:
            session = conn.execute(
                "SELECT day FROM sessions WHERE id=? AND participant_id=?",
                (session_id, participant_id),
            ).fetchone()
            if session is None:
                return
            day = int(session["day"])
            conn.execute(
                """UPDATE sessions SET state='aborted', ended_ms=?
                   WHERE participant_id=? AND day=? AND state NOT IN ('completed','ended')""",
                (now, participant_id, day),
            )
            conn.execute(
                """UPDATE pair_intervals SET ended_ms=?, end_reason='participant_ended_early'
                   WHERE participant_id=? AND day=? AND ended_ms IS NULL""",
                (now, participant_id, day),
            )

    def append_event(
        self,
        participant_id: str,
        session_id: str | None,
        event_type: str,
        client_ms: float | None,
        monotonic_ms: float | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        server_ms = int(time.time() * 1000)
        canonical_payload = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run_id = self._run_id(conn, participant_id)
            last = conn.execute(
                "SELECT seq, event_hash FROM events WHERE participant_id=? ORDER BY seq DESC LIMIT 1",
                (participant_id,),
            ).fetchone()
            seq = int(last["seq"]) + 1 if last else 1
            prev_hash = str(last["event_hash"]) if last else "0" * 64
            material = "|".join(
                [prev_hash, participant_id, str(session_id or ""), str(seq), event_type, str(server_ms), canonical_payload]
            )
            event_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()
            conn.execute(
                """INSERT INTO events
                   (run_id, participant_id, session_id, seq, event_type, server_ms, client_ms,
                    monotonic_ms, payload, prev_hash, event_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    participant_id,
                    session_id,
                    seq,
                    event_type,
                    server_ms,
                    client_ms,
                    monotonic_ms,
                    canonical_payload,
                    prev_hash,
                    event_hash,
                ),
            )
        return {"seq": seq, "server_ms": server_ms, "event_hash": event_hash}

    def append_events(
        self, participant_id: str, session_id: str | None, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not events:
            return []
        saved: list[dict[str, Any]] = []
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run_id = self._run_id(conn, participant_id)
            last = conn.execute(
                "SELECT seq, event_hash FROM events WHERE participant_id=? ORDER BY seq DESC LIMIT 1",
                (participant_id,),
            ).fetchone()
            seq = int(last["seq"]) if last else 0
            prev_hash = str(last["event_hash"]) if last else "0" * 64
            for event in events:
                seq += 1
                server_ms = int(time.time() * 1000)
                event_type = str(event["event_type"])
                payload = event.get("payload", {})
                canonical_payload = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                material = "|".join(
                    [
                        prev_hash,
                        participant_id,
                        str(session_id or ""),
                        str(seq),
                        event_type,
                        str(server_ms),
                        canonical_payload,
                    ]
                )
                event_hash = hashlib.sha256(material.encode("utf-8")).hexdigest()
                conn.execute(
                    """INSERT INTO events
                       (run_id, participant_id, session_id, seq, event_type, server_ms, client_ms,
                        monotonic_ms, payload, prev_hash, event_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        participant_id,
                        session_id,
                        seq,
                        event_type,
                        server_ms,
                        event.get("client_ms"),
                        event.get("monotonic_ms"),
                        canonical_payload,
                        prev_hash,
                        event_hash,
                    ),
                )
                saved.append({"seq": seq, "server_ms": server_ms, "event_hash": event_hash})
                prev_hash = event_hash
        return saved

    def save_quality(
        self,
        participant_id: str,
        session_id: str | None,
        kind: str,
        passed: bool,
        metrics: dict[str, Any],
    ) -> None:
        with self.connect() as conn:
            run_id = self._run_id(conn, participant_id)
            conn.execute(
                """INSERT INTO quality_checks
                   (run_id, participant_id, session_id, kind, passed, metrics, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    participant_id,
                    session_id,
                    kind,
                    int(passed),
                    json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                    int(time.time() * 1000),
                ),
            )

    def save_response(
        self,
        participant_id: str,
        session_id: str | None,
        instrument: str,
        version: str,
        answers: dict[str, Any],
        answer_labels: list[dict[str, Any]] | None = None,
        question_order: list[str] | None = None,
        locale: str = "zh-CN",
    ) -> None:
        with self.connect() as conn:
            run_id = self._run_id(conn, participant_id)
            conn.execute(
                """INSERT INTO responses
                   (run_id, participant_id, session_id, instrument, version, answers,
                    answer_labels, question_order, locale, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(participant_id, session_id, instrument)
                   DO UPDATE SET answers=excluded.answers, version=excluded.version,
                                 answer_labels=excluded.answer_labels,
                                 question_order=excluded.question_order,
                                 locale=excluded.locale,
                                 created_ms=excluded.created_ms""",
                (
                    run_id,
                    participant_id,
                    session_id,
                    instrument,
                    version,
                    json.dumps(answers, ensure_ascii=False, sort_keys=True),
                    json.dumps(answer_labels or [], ensure_ascii=False),
                    json.dumps(question_order or [], ensure_ascii=False),
                    locale,
                    int(time.time() * 1000),
                ),
            )

    def ensure_final_questionnaire_recovery(
        self, participant_id: str, session_id: str
    ) -> dict[str, Any]:
        """Create or return the participant's high-entropy final-form credential."""
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            existing = conn.execute(
                """SELECT * FROM final_questionnaire_recoveries
                   WHERE participant_id=? OR session_id=?""",
                (participant_id, session_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["participant_id"]) != participant_id
                    or str(existing["session_id"]) != session_id
                ):
                    raise ValueError("Final-questionnaire recovery binding conflict")
                return dict(existing)
            run_id = self._run_id(conn, participant_id)
            for _ in range(100):
                credential = "".join(secrets.choice(alphabet) for _ in range(8))
                try:
                    conn.execute(
                        """INSERT INTO final_questionnaire_recoveries
                           (participant_id, run_id, session_id, credential, entered_ms,
                            updated_ms)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            participant_id,
                            run_id,
                            session_id,
                            credential,
                            now_ms,
                            now_ms,
                        ),
                    )
                except sqlite3.IntegrityError:
                    continue
                row = conn.execute(
                    "SELECT * FROM final_questionnaire_recoveries WHERE participant_id=?",
                    (participant_id,),
                ).fetchone()
                assert row is not None
                return dict(row)
        raise RuntimeError("Unable to allocate a unique final-questionnaire credential")

    def final_questionnaire_recovery(
        self, participant_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM final_questionnaire_recoveries WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        return dict(row) if row else None

    def final_questionnaire_recovery_by_credential(
        self, credential: str
    ) -> dict[str, Any] | None:
        normalized = re.sub(r"[^A-Z0-9]", "", str(credential).upper())
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM final_questionnaire_recoveries WHERE credential=?",
                (normalized,),
            ).fetchone()
        return dict(row) if row else None

    def update_final_questionnaire_recovery(
        self,
        participant_id: str,
        *,
        model_finalized_ms: int | None = None,
        chain_committed_ms: int | None = None,
        final_submitted_ms: int | None = None,
        last_error: str | None = None,
        clear_error: bool = False,
    ) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        assignments = ["updated_ms=?"]
        values: list[Any] = [now_ms]
        for column, value in (
            ("model_finalized_ms", model_finalized_ms),
            ("chain_committed_ms", chain_committed_ms),
            ("final_submitted_ms", final_submitted_ms),
        ):
            if value is not None:
                assignments.append(f"{column}=COALESCE({column}, ?)")
                values.append(int(value))
        if clear_error:
            assignments.append("last_error=NULL")
        elif last_error is not None:
            assignments.append("last_error=?")
            values.append(str(last_error)[:2000])
        values.append(participant_id)
        with self.connect() as conn:
            conn.execute(
                f"""UPDATE final_questionnaire_recoveries
                    SET {', '.join(assignments)} WHERE participant_id=?""",
                values,
            )
            row = conn.execute(
                "SELECT * FROM final_questionnaire_recoveries WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Final-questionnaire recovery record is missing")
        return dict(row)

    def final_questionnaire_stage_records(
        self, participant_ids: list[str] | tuple[str, ...] | set[str]
    ) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(str(value) for value in participant_ids if value))
        if not ids:
            return {}
        result: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for offset in range(0, len(ids), 800):
                chunk = ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""SELECT * FROM final_questionnaire_recoveries
                        WHERE participant_id IN ({placeholders})""",
                    chunk,
                ).fetchall()
                result.update(
                    (str(row["participant_id"]), dict(row)) for row in rows
                )
        return result

    def admin_final_questionnaire_recoveries(
        self, run_ids: set[str] | None = None
    ) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = ()
        where = ""
        if run_ids:
            ordered = sorted(run_ids)
            where = f"WHERE f.run_id IN ({','.join('?' for _ in ordered)})"
            params = tuple(ordered)
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT f.*, p.code_hint
                    FROM final_questionnaire_recoveries f
                    JOIN participants p ON p.id=f.participant_id
                    {where}
                    ORDER BY f.entered_ms DESC""",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def enter_final_questionnaire(
        self, participant_id: str, session_id: str
    ) -> int:
        """Close timed interaction while leaving the study form recoverable."""
        now_ms = int(time.time() * 1000)
        with self.connect() as conn:
            session = conn.execute(
                """SELECT day FROM sessions
                   WHERE id=? AND participant_id=? AND client_role='study'""",
                (session_id, participant_id),
            ).fetchone()
            if session is None:
                raise ValueError("Study session is missing")
            day = int(session["day"])
            conn.execute(
                """UPDATE sessions SET state='questionnaire'
                   WHERE id=? AND participant_id=?
                     AND state IN ('started','active','questionnaire')""",
                (session_id, participant_id),
            )
            conn.execute(
                """UPDATE sessions SET state='completed', ended_ms=COALESCE(ended_ms, ?)
                   WHERE participant_id=? AND day=? AND client_role='companion'
                     AND state NOT IN ('completed','ended','aborted')""",
                (now_ms, participant_id, day),
            )
            conn.execute(
                """UPDATE pair_intervals SET ended_ms=COALESCE(ended_ms, ?),
                                             end_reason=COALESCE(end_reason, 'final_questionnaire')
                   WHERE participant_id=? AND day=? AND ended_ms IS NULL""",
                (now_ms, participant_id, day),
            )
        return now_ms

    def response_exists(
        self, participant_id: str, session_id: str | None, instrument: str
    ) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT 1 FROM responses
                   WHERE participant_id=? AND session_id IS ? AND instrument=?""",
                (participant_id, session_id, instrument),
            ).fetchone()
        return row is not None

    @staticmethod
    def _timed_interaction_rules(protocol_json: str | None, protocol_mode: str) -> dict[str, Any]:
        try:
            protocol = json.loads(str(protocol_json or "{}"))
        except (TypeError, json.JSONDecodeError):
            protocol = {}
        design = protocol.get("session_design") or {}
        rules = ((protocol.get("modes") or {}).get(protocol_mode) or {})
        schedule = rules.get("block_schedule") or {}
        try:
            final_day = int(design.get("total_sessions", 1))
        except (TypeError, ValueError):
            final_day = 1
        try:
            min_session_seconds = float(rules.get("min_session_seconds") or 0)
        except (TypeError, ValueError):
            min_session_seconds = 0.0
        return {
            "final_day": max(1, min(3, final_day)),
            "min_session_seconds": max(0.0, min_session_seconds),
            "block_schedule": schedule if isinstance(schedule, dict) else {},
        }

    @staticmethod
    def _timed_block_milestones_present(
        events: list[sqlite3.Row],
        schedule: dict[str, Any],
    ) -> bool:
        if not schedule.get("enabled"):
            return True
        try:
            blocks = int(schedule.get("active_blocks") or 0)
        except (TypeError, ValueError):
            return False
        if blocks < 1:
            return True
        include_final_rest = bool(schedule.get("rest_after_final_block"))
        needed: set[tuple[str, int]] = set()
        for block in range(1, blocks + 1):
            needed.update(
                (
                    ("phase.start", block),
                    ("phase.end", block),
                    ("companion.window_start", block),
                    ("companion.window_end", block),
                )
            )
            if block < blocks or include_final_rest:
                needed.update((("rest.start", block), ("rest.end", block)))
        seen: set[tuple[str, int]] = set()
        for row in events:
            try:
                payload = json.loads(str(row["payload"] or "{}"))
            except (TypeError, json.JSONDecodeError):
                payload = {}
            raw_block = payload.get(
                "block",
                payload.get("window", payload.get("active_block_index")),
            )
            try:
                block = int(raw_block)
            except (TypeError, ValueError):
                continue
            seen.add((str(row["event_type"]), block))
        return needed.issubset(seen)

    def end_to_end_completion_records(
        self, participant_ids: list[str] | tuple[str, ...] | set[str]
    ) -> dict[str, dict[str, Any]]:
        """Return server-authoritative timed-interaction completions.

        End-to-end completion and Share-chain eligibility require the
        protocol's real-time experiment (strict mode: five 6-minute blocks
        plus rests, 34 minutes / 2040 seconds).  The final questionnaire is
        recorded when present but does not grant or withhold this receipt.
        Abandoned or unactivated sessions cannot qualify.
        """
        ids = list(dict.fromkeys(str(value) for value in participant_ids if value))
        if not ids:
            return {}

        participant_rows: list[sqlite3.Row] = []
        session_rows: list[sqlite3.Row] = []
        response_rows: list[sqlite3.Row] = []
        event_rows: list[sqlite3.Row] = []
        now_ms = int(time.time() * 1000)
        # Stay below SQLite's common 999-variable limit for large exports/runs.
        with self.connect() as conn:
            for offset in range(0, len(ids), 800):
                chunk = ids[offset : offset + 800]
                placeholders = ",".join("?" for _ in chunk)
                participant_rows.extend(
                    conn.execute(
                        f"""SELECT p.id, r.protocol_json, r.protocol_mode
                            FROM participants p
                            JOIN study_runs r ON r.id=p.run_id
                            WHERE p.id IN ({placeholders}) AND p.enabled=1""",
                        chunk,
                    ).fetchall()
                )
                session_rows.extend(
                    conn.execute(
                        f"""SELECT id, participant_id, day, client_role, state,
                                   started_ms, activated_ms, ended_ms
                            FROM sessions
                            WHERE participant_id IN ({placeholders})""",
                        chunk,
                    ).fetchall()
                )
                response_rows.extend(
                    conn.execute(
                        f"""SELECT participant_id, session_id, created_ms
                            FROM responses
                            WHERE participant_id IN ({placeholders})
                              AND instrument='final'""",
                        chunk,
                    ).fetchall()
                )
                event_rows.extend(
                    conn.execute(
                        f"""SELECT participant_id, session_id, event_type, payload, server_ms
                            FROM events
                            WHERE participant_id IN ({placeholders})
                              AND event_type IN (
                                  'phase.start','phase.end',
                                  'companion.window_start','companion.window_end',
                                  'rest.start','rest.end'
                              )""",
                        chunk,
                    ).fetchall()
                )

        rules_by_participant: dict[str, dict[str, Any]] = {}
        for row in participant_rows:
            rules_by_participant[str(row["id"])] = self._timed_interaction_rules(
                row["protocol_json"],
                str(row["protocol_mode"] or "strict"),
            )

        study_by_participant: dict[str, list[sqlite3.Row]] = {}
        companion_days: set[tuple[str, int]] = set()
        for row in session_rows:
            participant_id = str(row["participant_id"])
            rules = rules_by_participant.get(participant_id)
            if rules is None or int(row["day"]) != rules["final_day"]:
                continue
            state = str(row["state"] or "")
            if state == "aborted":
                continue
            if str(row["client_role"]) == "study":
                study_by_participant.setdefault(participant_id, []).append(row)
            elif str(row["client_role"]) == "companion":
                companion_days.add((participant_id, int(row["day"])))

        events_by_session: dict[str, list[sqlite3.Row]] = {}
        last_block_end_by_session: dict[str, int] = {}
        for row in event_rows:
            session_id = str(row["session_id"] or "")
            if not session_id:
                continue
            events_by_session.setdefault(session_id, []).append(row)
            if str(row["event_type"]) in {
                "companion.window_end",
                "phase.end",
                "rest.end",
            }:
                last_block_end_by_session[session_id] = max(
                    int(last_block_end_by_session.get(session_id) or 0),
                    int(row["server_ms"] or 0),
                )

        final_response_by_session = {
            (str(row["participant_id"]), str(row["session_id"])): row
            for row in response_rows
            if row["session_id"] is not None
        }
        records: dict[str, dict[str, Any]] = {}
        for participant_id, study_rows in study_by_participant.items():
            rules = rules_by_participant[participant_id]
            final_day = int(rules["final_day"])
            if (participant_id, final_day) not in companion_days:
                continue
            eligible: list[sqlite3.Row] = []
            for row in study_rows:
                activated_ms = row["activated_ms"]
                if activated_ms is None:
                    continue
                ended_ms = row["ended_ms"]
                reference_ms = int(ended_ms) if ended_ms is not None else now_ms
                elapsed_ms = max(0, reference_ms - int(activated_ms))
                if elapsed_ms < int(float(rules["min_session_seconds"]) * 1000):
                    continue
                if not self._timed_block_milestones_present(
                    events_by_session.get(str(row["id"]), []),
                    rules["block_schedule"],
                ):
                    continue
                eligible.append(row)
            if not eligible:
                continue
            study_row = max(
                eligible,
                key=lambda row: (
                    int(row["ended_ms"] or last_block_end_by_session.get(str(row["id"])) or 0),
                    str(row["id"]),
                ),
            )
            response = final_response_by_session.get(
                (participant_id, str(study_row["id"]))
            )
            completed_ms = int(
                study_row["ended_ms"]
                or last_block_end_by_session.get(str(study_row["id"]))
                or study_row["activated_ms"]
                or 0
            )
            records[participant_id] = {
                "participant_id": participant_id,
                "study_session_id": str(study_row["id"]),
                "final_day": final_day,
                "entered_ms": completed_ms,
                "completed_ms": completed_ms,
                "final_response_ms": (
                    int(response["created_ms"]) if response is not None else None
                ),
            }
        return records

    def participant_end_to_end_complete(self, participant_id: str) -> bool:
        return participant_id in self.end_to_end_completion_records([participant_id])

    def save_audio_record(
        self,
        participant_id: str,
        session_id: str | None,
        kind: str,
        relative_path: str,
        content_type: str,
        size: int,
        digest: str,
    ) -> str:
        audio_id = str(uuid.uuid4())
        with self.connect() as conn:
            run_id = self._run_id(conn, participant_id)
            conn.execute(
                """INSERT INTO audio_files
                   (id, run_id, participant_id, session_id, kind, relative_path, content_type,
                    bytes, sha256, created_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    audio_id,
                    run_id,
                    participant_id,
                    session_id,
                    kind,
                    relative_path,
                    content_type,
                    size,
                    digest,
                    int(time.time() * 1000),
                ),
            )
        return audio_id

    def concurrent_study_load(
        self,
        *,
        active_window_ms: int = 50 * 60 * 1000,
    ) -> dict[str, Any]:
        """Count recently activated, unfinished study sessions.

        Uses only the sessions table (indexed-friendly, no events scan) so the
        public capacity/health endpoints stay fast under large event logs.
        Slots free automatically about ``active_window_ms`` after activation if
        the session was abandoned without an orderly end/abort.
        """
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - max(60_000, int(active_window_ms))
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT id AS session_id,
                          participant_id AS participant_id,
                          activated_ms AS activated_ms
                   FROM sessions
                   WHERE client_role = 'study'
                     AND state = 'active'
                     AND activated_ms IS NOT NULL
                     AND activated_ms >= ?
                     AND ended_ms IS NULL
                   ORDER BY activated_ms DESC""",
                (cutoff_ms,),
            ).fetchall()
        active = [
            {
                "session_id": str(row["session_id"]),
                "participant_id": str(row["participant_id"]),
                "activated_ms": int(row["activated_ms"] or 0),
            }
            for row in rows
        ]
        return {
            "active_count": len(active),
            "sessions": active,
            "server_ms": now_ms,
            "active_window_ms": max(60_000, int(active_window_ms)),
        }

    def live_experiment_sessions(
        self,
        *,
        active_window_ms: int = 90 * 60 * 1000,
        setup_window_ms: int = 12 * 60 * 60 * 1000,
    ) -> dict[str, Any]:
        """Read unfinished study/companion sessions without scanning events.

        Officially timed sessions stay visible for ``active_window_ms`` after
        ``activated_ms`` so abandoned historical rows do not flood the board.
        Unactivated setup sessions older than ``setup_window_ms`` are omitted.
        """
        now_ms = int(time.time() * 1000)
        active_cutoff_ms = now_ms - max(60_000, int(active_window_ms))
        setup_cutoff_ms = now_ms - max(60_000, int(setup_window_ms))
        with self.connect() as conn:
            session_rows = conn.execute(
                """SELECT s.id AS session_id,
                          s.participant_id AS participant_id,
                          s.client_role AS client_role,
                          s.state AS state,
                          s.day AS day,
                          s.started_ms AS started_ms,
                          s.activated_ms AS activated_ms,
                          s.ended_ms AS ended_ms,
                          p.code_hint AS code_hint,
                          p.group_name AS group_name,
                          r.slug AS run_slug
                   FROM sessions s
                   JOIN participants p ON p.id = s.participant_id
                   JOIN study_runs r ON r.id = p.run_id
                   WHERE s.ended_ms IS NULL
                     AND s.state IN ('started', 'active')
                     AND (
                       (s.activated_ms IS NOT NULL AND s.activated_ms >= ?)
                       OR (s.activated_ms IS NULL AND COALESCE(s.started_ms, 0) >= ?)
                     )
                   ORDER BY COALESCE(s.activated_ms, s.started_ms) DESC""",
                (active_cutoff_ms, setup_cutoff_ms),
            ).fetchall()
            pair_rows = conn.execute(
                """SELECT participant_id
                   FROM pair_intervals
                   WHERE ended_ms IS NULL"""
            ).fetchall()
        pair_live = {
            str(row["participant_id"]) for row in pair_rows if row["participant_id"]
        }
        by_pid: dict[str, dict[str, Any]] = {}
        sessions: list[dict[str, Any]] = []
        for row in session_rows:
            payload = {
                "session_id": str(row["session_id"]),
                "participant_id": str(row["participant_id"]),
                "client_role": str(row["client_role"]),
                "state": str(row["state"]),
                "day": int(row["day"] or 1),
                "started_ms": row["started_ms"],
                "activated_ms": row["activated_ms"],
                "ended_ms": row["ended_ms"],
                "code_hint": str(row["code_hint"] or ""),
                "group_name": str(row["group_name"] or ""),
                "run_slug": str(row["run_slug"] or ""),
            }
            sessions.append(payload)
            bucket = by_pid.setdefault(
                payload["participant_id"],
                {
                    "participant_id": payload["participant_id"],
                    "code_hint": payload["code_hint"],
                    "group_name": payload["group_name"],
                    "run_slug": payload["run_slug"],
                    "study": None,
                    "companion": None,
                    "pair_live": payload["participant_id"] in pair_live,
                },
            )
            role_payload = {
                "session_id": payload["session_id"],
                "state": payload["state"],
                "day": payload["day"],
                "started_ms": payload["started_ms"],
                "activated_ms": payload["activated_ms"],
            }
            if payload["client_role"] == "study":
                bucket["study"] = role_payload
            elif payload["client_role"] == "companion":
                bucket["companion"] = role_payload
        return {
            "server_ms": now_ms,
            "participants": list(by_pid.values()),
            "sessions": sessions,
        }

    def participants_brief(
        self, participant_ids: list[str] | tuple[str, ...] | set[str]
    ) -> dict[str, dict[str, Any]]:
        ids = [str(value) for value in dict.fromkeys(participant_ids) if value]
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT p.id AS participant_id,
                           p.code_hint AS code_hint,
                           p.group_name AS group_name,
                           r.slug AS run_slug
                    FROM participants p
                    JOIN study_runs r ON r.id = p.run_id
                    WHERE p.id IN ({placeholders})""",
                ids,
            ).fetchall()
        return {str(row["participant_id"]): dict(row) for row in rows}

    def participant_progress(self, participant_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            sessions = conn.execute(
                """SELECT id, day, client_role, state, started_ms, activated_ms, ended_ms
                   FROM sessions WHERE participant_id=? ORDER BY day, client_role""",
                (participant_id,),
            ).fetchall()
            quality = conn.execute(
                """SELECT q.session_id, s.day, q.kind, q.passed, q.metrics, q.created_ms
                   FROM quality_checks q
                   LEFT JOIN sessions s ON s.id=q.session_id
                   WHERE q.participant_id=?
                   ORDER BY q.created_ms DESC, q.id DESC LIMIT 40""",
                (participant_id,),
            ).fetchall()
            responses = conn.execute(
                """SELECT instrument, version, created_ms FROM responses
                   WHERE participant_id=? ORDER BY created_ms""",
                (participant_id,),
            ).fetchall()
            pair_intervals = conn.execute(
                """SELECT day, study_session_id, companion_session_id, started_ms,
                          ended_ms, end_reason, qualified, qualified_ms
                   FROM pair_intervals WHERE participant_id=? ORDER BY started_ms""",
                (participant_id,),
            ).fetchall()
        return {
            "sessions": [dict(row) for row in sessions],
            "quality": [dict(row) for row in quality],
            "responses": [dict(row) for row in responses],
            "pair_intervals": [dict(row) for row in pair_intervals],
        }

    def selected_topics(self, participant_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT s.day, e.server_ms, e.event_type, e.payload
                   FROM events e
                   JOIN sessions s ON s.id=e.session_id
                   WHERE e.participant_id=?
                     AND e.event_type IN ('topic.selected', 'topic.changed')
                   ORDER BY e.seq""",
                (participant_id,),
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload"])
            result.append(
                {
                    "day": int(row["day"]),
                    "server_ms": int(row["server_ms"]),
                    "event_type": str(row["event_type"]),
                    "topic_id": payload.get("topic_id"),
                }
            )
        return result

    def protocol_snapshot(self, participant_id: str, session_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE participant_id=? AND id=?",
                (participant_id, session_id),
            ).fetchone()
            if session is None:
                return {
                    "session": None,
                    "events": [],
                    "quality": [],
                    "responses": [],
                    "pair_intervals": [],
                }
            day = int(session["day"])
            events = conn.execute(
                """SELECT event_type, server_ms, client_ms, monotonic_ms, payload
                   FROM events WHERE participant_id=? AND session_id=? ORDER BY seq""",
                (participant_id, session_id),
            ).fetchall()
            quality = conn.execute(
                """SELECT kind, passed, metrics, created_ms FROM quality_checks
                   WHERE participant_id=? AND session_id=? ORDER BY created_ms DESC, id DESC""",
                (participant_id, session_id),
            ).fetchall()
            responses = conn.execute(
                """SELECT instrument, version, answers, created_ms FROM responses
                   WHERE participant_id=? AND session_id=? ORDER BY created_ms""",
                (participant_id, session_id),
            ).fetchall()
            intervals = conn.execute(
                """SELECT * FROM pair_intervals
                   WHERE participant_id=? AND day=? ORDER BY started_ms""",
                (participant_id, day),
            ).fetchall()
        return {
            "session": dict(session),
            "events": [dict(row) for row in events],
            "quality": [dict(row) for row in quality],
            "responses": [dict(row) for row in responses],
            "pair_intervals": [dict(row) for row in intervals],
        }

    def admin_roster(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """WITH session_summary AS (
                       SELECT participant_id,
                              COUNT(DISTINCT day) AS days_started,
                              MAX(started_ms) AS last_session_ms
                       FROM sessions GROUP BY participant_id
                   ),
                   event_summary AS (
                       SELECT participant_id, COUNT(*) AS event_count
                       FROM events GROUP BY participant_id
                   ),
                   pair_summary AS (
                       SELECT participant_id,
                              COUNT(*) AS pair_interval_count,
                              SUM(CASE WHEN ended_ms IS NULL THEN 1 ELSE 0 END) AS pair_live,
                              COALESCE(SUM(
                                  MAX(0, COALESCE(
                                      ended_ms,
                                      CAST(strftime('%s','now') AS INTEGER) * 1000
                                  ) - started_ms)
                              ), 0) AS pair_overlap_ms
                       FROM pair_intervals GROUP BY participant_id
                   )
                   SELECT p.id, p.run_id, p.code_hint, p.cohort, p.group_name, p.probe_schedule,
                          p.assignment_id, p.neutral_id, p.enrollment_wave, p.release_state,
                          p.prior_id, p.prior_n, p.prior_q, p.preview_mode,
                          p.analysis_excluded, p.enabled, p.created_ms,
                          r.slug AS run_slug, r.data_origin,
                          r.protocol_version, r.protocol_sha256 AS run_protocol_sha256,
                          r.protocol_mode,
                          COALESCE(ss.days_started, 0) AS days_started,
                          ss.last_session_ms,
                          COALESCE(es.event_count, 0) AS event_count,
                          COALESCE(ps.pair_interval_count, 0) AS pair_interval_count,
                          COALESCE(ps.pair_live, 0) AS pair_live,
                          COALESCE(ps.pair_overlap_ms, 0) AS pair_overlap_ms
                   FROM participants p
                   JOIN study_runs r ON r.id=p.run_id
                   LEFT JOIN session_summary ss ON ss.participant_id=p.id
                   LEFT JOIN event_summary es ON es.participant_id=p.id
                   LEFT JOIN pair_summary ps ON ps.participant_id=p.id
                   ORDER BY p.created_ms"""
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_events_and_run_counts(
        self, run_ids: set[str], *, limit: int = 200
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        if not run_ids:
            return [], {}
        ordered_ids = sorted(run_ids)
        placeholders = ",".join("?" for _ in ordered_ids)
        with self.connect() as conn:
            counts = conn.execute(
                f"""SELECT run_id, COUNT(*) AS event_count
                    FROM events
                    WHERE run_id IN ({placeholders})
                    GROUP BY run_id""",
                ordered_ids,
            ).fetchall()
            recent = conn.execute(
                f"""SELECT * FROM events
                    WHERE run_id IN ({placeholders})
                    ORDER BY id DESC LIMIT ?""",
                (*ordered_ids, max(1, min(int(limit), 1000))),
            ).fetchall()
        return (
            [dict(row) for row in recent],
            {str(row["run_id"]): int(row["event_count"]) for row in counts},
        )

    def participant_table_rows(
        self, table: str, participant_id: str
    ) -> list[dict[str, Any]]:
        allowed = {
            "sessions",
            "events",
            "quality_checks",
            "responses",
            "audio_files",
            "final_questionnaire_recoveries",
        }
        if table not in allowed:
            raise ValueError("invalid participant table")
        order = "id"
        if table == "sessions":
            order = "started_ms"
        elif table in {"quality_checks", "responses", "audio_files"}:
            order = "created_ms"
        elif table == "final_questionnaire_recoveries":
            order = "entered_ms"
        with self.connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM {table}
                    WHERE participant_id=? ORDER BY {order}""",
                (participant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def table_rows(self, table: str) -> list[dict[str, Any]]:
        allowed = {
            "study_runs",
            "participants",
            "sessions",
            "pair_intervals",
            "events",
            "quality_checks",
            "responses",
            "final_questionnaire_recoveries",
            "audio_files",
            "shared_priors",
            "model_snapshots",
            "pipeline_actions",
            "prior_probe_assignments",
            "model_counters",
            "participant_calibrations",
            "account_batches",
        }
        if table not in allowed:
            raise ValueError("invalid table")
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        return [dict(row) for row in rows]

    def events_in_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT e.*, s.day, s.client_role
                   FROM events e
                   LEFT JOIN sessions s ON s.id=e.session_id
                   WHERE e.run_id=? ORDER BY e.server_ms, e.id""",
                (run_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            try:
                value["payload"] = json.loads(value["payload"])
            except (TypeError, json.JSONDecodeError):
                value["payload"] = {}
            result.append(value)
        return result

    def participant_event_rows(self, participant_id: str) -> list[dict[str, Any]]:
        """Return one participant's event rows without crossing run boundaries."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT e.id, e.seq, e.session_id, e.event_type, e.server_ms,
                          e.client_ms, e.monotonic_ms, e.payload,
                          s.day, s.client_role
                   FROM events e
                   LEFT JOIN sessions s ON s.id=e.session_id
                   WHERE e.participant_id=?
                   ORDER BY e.server_ms, e.id""",
                (participant_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation_memory(self, participant_id: str, source: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT paragraph FROM conversation_memory
                   WHERE participant_id=? AND source=?""",
                (participant_id, source),
            ).fetchone()
        if row is None:
            return ""
        return str(row["paragraph"] or "").strip()

    def upsert_conversation_memory(
        self,
        participant_id: str,
        source: str,
        paragraph: str,
        *,
        turn_delta: int = 1,
    ) -> None:
        cleaned = " ".join(paragraph.strip().split())
        if len(cleaned) > 240:
            cleaned = cleaned[:240]
        now = int(time.time() * 1000)
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO conversation_memory
                   (participant_id, source, paragraph, turn_count, updated_ms)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(participant_id, source) DO UPDATE SET
                     paragraph=excluded.paragraph,
                     turn_count=conversation_memory.turn_count + excluded.turn_count,
                     updated_ms=excluded.updated_ms""",
                (participant_id, source, cleaned, max(turn_delta, 0), now),
            )

    def clear_conversation_memory(self, participant_id: str) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_memory WHERE participant_id=?",
                (participant_id,),
            )
            return int(cursor.rowcount or 0)
