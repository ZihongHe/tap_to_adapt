"""Minimal cross-instance mutex for the Share-fit scheduling boundary.

The formal and test studies keep separate databases, orchestration state,
models and participant records.  This optional SQLite registry stores only
opaque owner hashes, instance labels and scheduling timestamps so both HTTP
processes can nevertheless enforce one global Share-fit setup/experiment slot.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA = "tap-to-adapt-share-global-coordination/v1"


class ShareGlobalCoordinator:
    def __init__(self, path: str | Path | None, instance_id: str) -> None:
        raw = str(path or "").strip()
        self.path = Path(raw).resolve() if raw else None
        self.instance_id = str(instance_id or "").strip().lower()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def owner_key(self, participant_id: str) -> str:
        material = f"{SCHEMA}|{self.instance_id}|{participant_id}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        if self.path is None:
            raise RuntimeError("global Share coordination is disabled")
        conn = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS holders (
                    owner_key TEXT PRIMARY KEY,
                    instance_id TEXT NOT NULL,
                    reference_start_ms INTEGER NOT NULL,
                    claimed_ms INTEGER NOT NULL,
                    expires_ms INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS appointments (
                    reservation_key TEXT PRIMARY KEY,
                    owner_key TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    scheduled_start_ms INTEGER NOT NULL,
                    grace_ms INTEGER NOT NULL,
                    arrived_ms INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_global_share_appointments_start
                    ON appointments(status, scheduled_start_ms);
                """
            )
            row = conn.execute(
                "SELECT value FROM metadata WHERE key='schema'"
            ).fetchone()
            if row is not None and str(row["value"]) != SCHEMA:
                raise RuntimeError("unsupported global Share coordination schema")
            conn.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('schema',?)",
                (SCHEMA,),
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _cleanup(conn: sqlite3.Connection, now_ms: int) -> None:
        conn.execute("DELETE FROM holders WHERE expires_ms<=?", (now_ms,))
        conn.execute(
            """DELETE FROM appointments
               WHERE status='booked' AND arrived_ms=0
                 AND scheduled_start_ms+grace_ms<?""",
            (now_ms,),
        )
        conn.execute(
            "DELETE FROM appointments WHERE status IN ('released','completed')"
        )

    def snapshot(self, owner_key: str, *, now_ms: int | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "holders": [],
                "self_holder": None,
                "external_holders": [],
                "appointments": [],
            }
        now_ms = int(now_ms or time.time() * 1000)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            holders = [dict(row) for row in conn.execute(
                "SELECT * FROM holders ORDER BY claimed_ms, owner_key"
            ).fetchall()]
            appointments = [dict(row) for row in conn.execute(
                """SELECT * FROM appointments
                   WHERE status IN ('booked','claimed','active')
                   ORDER BY scheduled_start_ms, reservation_key"""
            ).fetchall()]
            conn.commit()
        return {
            "enabled": True,
            "holders": holders,
            "self_holder": next(
                (row for row in holders if str(row["owner_key"]) == owner_key),
                None,
            ),
            "external_holders": [
                row for row in holders if str(row["owner_key"]) != owner_key
            ],
            "appointments": appointments,
        }

    def claim_holder(
        self,
        owner_key: str,
        *,
        reference_start_ms: int,
        expires_ms: int,
        now_ms: int,
    ) -> bool:
        if not self.enabled:
            return True
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            other = conn.execute(
                "SELECT 1 FROM holders WHERE owner_key<>? LIMIT 1", (owner_key,)
            ).fetchone()
            if other is not None:
                conn.rollback()
                return False
            conn.execute(
                """INSERT INTO holders
                   (owner_key,instance_id,reference_start_ms,claimed_ms,
                    expires_ms,stage,updated_ms)
                   VALUES(?,?,?,?,?,'setup',?)
                   ON CONFLICT(owner_key) DO UPDATE SET
                     reference_start_ms=excluded.reference_start_ms,
                     expires_ms=MAX(holders.expires_ms,excluded.expires_ms),
                     updated_ms=excluded.updated_ms""",
                (
                    owner_key,
                    self.instance_id,
                    int(reference_start_ms),
                    int(now_ms),
                    int(expires_ms),
                    int(now_ms),
                ),
            )
            conn.commit()
        return True

    def register_existing_holder(
        self,
        owner_key: str,
        *,
        reference_start_ms: int,
        claimed_ms: int,
        expires_ms: int,
        stage: str,
    ) -> None:
        """Deployment-only grandfathering for holders created pre-registry.

        Multiple rows may exist only through this migration. Runtime claims
        remain fail-closed while any different live row exists.
        """
        if not self.enabled:
            return
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            conn.execute(
                """INSERT INTO holders
                   (owner_key,instance_id,reference_start_ms,claimed_ms,
                    expires_ms,stage,updated_ms)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(owner_key) DO UPDATE SET
                     reference_start_ms=excluded.reference_start_ms,
                     claimed_ms=excluded.claimed_ms,
                     expires_ms=excluded.expires_ms,
                     stage=excluded.stage,
                     updated_ms=excluded.updated_ms""",
                (
                    owner_key,
                    self.instance_id,
                    int(reference_start_ms),
                    int(claimed_ms),
                    int(expires_ms),
                    "active" if stage == "active" else "setup",
                    now_ms,
                ),
            )
            conn.commit()

    def activate_holder(self, owner_key: str, *, expires_ms: int, now_ms: int) -> bool:
        if not self.enabled:
            return True
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            own = conn.execute(
                "SELECT 1 FROM holders WHERE owner_key=?", (owner_key,)
            ).fetchone()
            if own is None:
                conn.rollback()
                return False
            conn.execute(
                """UPDATE holders SET stage='active', expires_ms=?, updated_ms=?
                   WHERE owner_key=?""",
                (int(expires_ms), int(now_ms), owner_key),
            )
            conn.execute(
                "UPDATE appointments SET status='active', updated_ms=? WHERE owner_key=?",
                (int(now_ms), owner_key),
            )
            conn.commit()
        return True

    def release_owner(self, owner_key: str) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM holders WHERE owner_key=?", (owner_key,))
            conn.execute("DELETE FROM appointments WHERE owner_key=?", (owner_key,))
            conn.commit()

    def book_appointment(
        self,
        owner_key: str,
        reservation_key: str,
        *,
        scheduled_start_ms: int,
        grace_ms: int,
        spacing_ms: int,
        now_ms: int,
    ) -> bool:
        if not self.enabled:
            return True
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            references = [
                int(row["reference_start_ms"])
                for row in conn.execute(
                    "SELECT reference_start_ms FROM holders WHERE owner_key<>?",
                    (owner_key,),
                ).fetchall()
            ]
            references.extend(
                int(row["scheduled_start_ms"])
                for row in conn.execute(
                    """SELECT scheduled_start_ms FROM appointments
                       WHERE owner_key<>? AND status IN ('booked','claimed','active')""",
                    (owner_key,),
                ).fetchall()
            )
            if any(
                abs(int(scheduled_start_ms) - reference) < int(spacing_ms)
                for reference in references
            ):
                conn.rollback()
                return False
            conn.execute("DELETE FROM appointments WHERE owner_key=?", (owner_key,))
            conn.execute(
                """INSERT INTO appointments
                   (reservation_key,owner_key,instance_id,scheduled_start_ms,
                    grace_ms,arrived_ms,status,updated_ms)
                   VALUES(?,?,?,?,?,0,'booked',?)""",
                (
                    reservation_key,
                    owner_key,
                    self.instance_id,
                    int(scheduled_start_ms),
                    int(grace_ms),
                    int(now_ms),
                ),
            )
            conn.commit()
        return True

    def _upsert_booked_reservation(
        self,
        conn: sqlite3.Connection,
        owner_key: str,
        reservation_key: str,
        *,
        scheduled_start_ms: int,
        grace_ms: int,
        arrived_ms: int,
        now_ms: int,
    ) -> None:
        existing = conn.execute(
            "SELECT * FROM appointments WHERE reservation_key=?",
            (reservation_key,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO appointments
                   (reservation_key,owner_key,instance_id,scheduled_start_ms,
                    grace_ms,arrived_ms,status,updated_ms)
                   VALUES(?,?,?,?,?,?,'booked',?)""",
                (
                    reservation_key,
                    owner_key,
                    self.instance_id,
                    int(scheduled_start_ms),
                    int(grace_ms),
                    int(arrived_ms or 0),
                    int(now_ms),
                ),
            )
            return
        if str(existing["owner_key"]) != owner_key:
            return
        conn.execute(
            """UPDATE appointments
               SET arrived_ms=CASE
                     WHEN arrived_ms=0 AND ? > 0 THEN ?
                     ELSE arrived_ms
                   END,
                   updated_ms=?
               WHERE reservation_key=?""",
            (int(arrived_ms or 0), int(arrived_ms or 0), int(now_ms), reservation_key),
        )

    def mark_appointment_arrived(
        self,
        owner_key: str,
        reservation_key: str,
        *,
        scheduled_start_ms: int,
        grace_ms: int,
        now_ms: int,
    ) -> None:
        """Stamp a local on-time arrival onto the global row so grace cleanup
        cannot treat a waiting predecessor-blocked booking as a no-show."""
        if not self.enabled or not reservation_key:
            return
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            self._upsert_booked_reservation(
                conn,
                owner_key,
                reservation_key,
                scheduled_start_ms=scheduled_start_ms,
                grace_ms=grace_ms,
                arrived_ms=now_ms,
                now_ms=now_ms,
            )
            conn.commit()

    def arrive_and_claim(
        self,
        owner_key: str,
        reservation_key: str,
        *,
        reference_start_ms: int,
        expires_ms: int,
        now_ms: int,
        grace_ms: int | None = None,
    ) -> bool:
        if not self.enabled:
            return True
        if not reservation_key:
            return False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._cleanup(conn, now_ms)
            self._upsert_booked_reservation(
                conn,
                owner_key,
                reservation_key,
                scheduled_start_ms=int(reference_start_ms),
                grace_ms=int(grace_ms or 10 * 60 * 1000),
                arrived_ms=now_ms,
                now_ms=now_ms,
            )
            appointment = conn.execute(
                """SELECT * FROM appointments
                   WHERE reservation_key=? AND owner_key=?""",
                (reservation_key, owner_key),
            ).fetchone()
            if appointment is None:
                conn.rollback()
                return False
            conn.execute(
                "UPDATE appointments SET arrived_ms=?,updated_ms=? WHERE reservation_key=?",
                (int(now_ms), int(now_ms), reservation_key),
            )
            other = conn.execute(
                "SELECT 1 FROM holders WHERE owner_key<>? LIMIT 1", (owner_key,)
            ).fetchone()
            if other is not None:
                conn.commit()
                return False
            conn.execute(
                """INSERT INTO holders
                   (owner_key,instance_id,reference_start_ms,claimed_ms,
                    expires_ms,stage,updated_ms)
                   VALUES(?,?,?,?,?,'setup',?)
                   ON CONFLICT(owner_key) DO UPDATE SET
                     expires_ms=excluded.expires_ms,updated_ms=excluded.updated_ms""",
                (
                    owner_key,
                    self.instance_id,
                    int(reference_start_ms),
                    int(now_ms),
                    int(expires_ms),
                    int(now_ms),
                ),
            )
            conn.execute(
                "UPDATE appointments SET status='claimed',updated_ms=? WHERE reservation_key=?",
                (int(now_ms), reservation_key),
            )
            conn.commit()
        return True

    def release_reservation(self, reservation_key: str) -> None:
        if not self.enabled or not reservation_key:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM appointments WHERE reservation_key=?", (reservation_key,))

