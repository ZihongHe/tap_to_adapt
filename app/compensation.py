"""Encrypted storage for compensation identity and contact records.

The vault is separate from the research event database. Decryption is reserved
for explicit authorized processing; routine research exports contain neither
identity-document numbers nor signatures nor contact plaintext.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from cryptography.fernet import Fernet, InvalidToken


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS compensation_identities (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    participant_id TEXT NOT NULL UNIQUE,
    document_type TEXT NOT NULL,
    full_name_ciphertext BLOB NOT NULL,
    document_number_ciphertext BLOB NOT NULL,
    document_number_digest TEXT NOT NULL,
    signature_png_ciphertext BLOB NOT NULL,
    notice_version TEXT NOT NULL,
    notice_hash TEXT NOT NULL,
    notice_json TEXT NOT NULL,
    affirmation TEXT NOT NULL,
    separate_consent INTEGER NOT NULL,
    submitted_ms INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compensation_identity_run
    ON compensation_identities(run_id, submitted_ms);
CREATE TRIGGER IF NOT EXISTS compensation_identities_immutable
BEFORE UPDATE ON compensation_identities
BEGIN SELECT RAISE(ABORT, 'compensation identity records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS compensation_identities_delete_forbidden
BEFORE DELETE ON compensation_identities
BEGIN SELECT RAISE(ABORT, 'compensation identity records cannot be deleted in place'); END;
CREATE TABLE IF NOT EXISTS compensation_registrations (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    participant_id TEXT NOT NULL UNIQUE,
    decision TEXT NOT NULL CHECK(decision IN ('provide', 'decline')),
    identity_record_id TEXT,
    notice_version TEXT NOT NULL,
    notice_hash TEXT NOT NULL,
    notice_json TEXT NOT NULL,
    affirmation TEXT NOT NULL,
    submitted_ms INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compensation_registration_run
    ON compensation_registrations(run_id, submitted_ms);
CREATE TRIGGER IF NOT EXISTS compensation_registrations_immutable
BEFORE UPDATE ON compensation_registrations
BEGIN SELECT RAISE(ABORT, 'compensation registration records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS compensation_registrations_delete_forbidden
BEFORE DELETE ON compensation_registrations
BEGIN SELECT RAISE(ABORT, 'compensation registration records cannot be deleted in place'); END;
CREATE TABLE IF NOT EXISTS compensation_contacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    participant_id TEXT NOT NULL UNIQUE,
    email_ciphertext BLOB,
    phone_ciphertext BLOB,
    wechat_ciphertext BLOB,
    contact_channels TEXT NOT NULL,
    notice_version TEXT NOT NULL,
    notice_hash TEXT NOT NULL,
    submitted_ms INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,
    user_agent TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_compensation_contact_run
    ON compensation_contacts(run_id, submitted_ms);
CREATE TRIGGER IF NOT EXISTS compensation_contacts_immutable
BEFORE UPDATE ON compensation_contacts
BEGIN SELECT RAISE(ABORT, 'compensation contact records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS compensation_contacts_delete_forbidden
BEFORE DELETE ON compensation_contacts
BEGIN SELECT RAISE(ABORT, 'compensation contact records cannot be deleted in place'); END;
"""


class CompensationVaultUnavailable(RuntimeError):
    pass


class CompensationVault:
    """Encrypted identity store kept outside study and consent databases.

    The database only exposes participant/run pseudonyms and ciphertext.
    Identity documents stay offline.  The authenticated admin dashboard may
    decrypt names and contact channels to operate the live study; generic
    dataset export must not call those helpers.
    """

    def __init__(self, data_dir: Path, key: str):
        self.directory = data_dir / "compensation"
        self.path = self.directory / "compensation.sqlite3"
        self._fernet: Fernet | None = None
        self._digest_key: bytes | None = None
        normalized = key.strip().encode("ascii", errors="ignore")
        if normalized:
            try:
                decoded = base64.urlsafe_b64decode(normalized)
                if len(decoded) != 32:
                    raise ValueError("vault key must decode to 32 bytes")
                self._fernet = Fernet(normalized)
                self._digest_key = hashlib.sha256(
                    decoded + b"tap-to-adapt-compensation-digest-v1"
                ).digest()
            except (TypeError, ValueError) as exc:
                raise CompensationVaultUnavailable(
                    "COMPENSATION_VAULT_KEY is invalid"
                ) from exc
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(compensation_identities)"
                ).fetchall()
            }
            if "notice_json" not in columns:
                connection.execute(
                    "ALTER TABLE compensation_identities "
                    "ADD COLUMN notice_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "affirmation" not in columns:
                connection.execute(
                    "ALTER TABLE compensation_identities "
                    "ADD COLUMN affirmation TEXT NOT NULL DEFAULT ''"
                )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @property
    def ready(self) -> bool:
        return self._fernet is not None and self._digest_key is not None

    def require_ready(self) -> None:
        if not self.ready:
            raise CompensationVaultUnavailable(
                "Encrypted compensation vault is not configured"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def status(self, participant_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            contact = connection.execute(
                """SELECT id, contact_channels FROM compensation_contacts
                   WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
            registration = connection.execute(
                """SELECT id, participant_id, run_id, decision, identity_record_id,
                          notice_version, notice_hash, submitted_ms
                   FROM compensation_registrations WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
            if registration is not None:
                return {
                    **dict(registration),
                    "contact_complete": contact is not None,
                    "contact_record_id": str(contact["id"]) if contact else None,
                }
            row = connection.execute(
                """SELECT id, participant_id, run_id, document_type,
                          notice_version, notice_hash, separate_consent, submitted_ms
                   FROM compensation_identities WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
        if row is None:
            if contact is None:
                return None
            return {
                "participant_id": participant_id,
                "decision": None,
                "identity_record_id": None,
                "contact_complete": True,
                "contact_record_id": str(contact["id"]),
                "contact_channels": str(contact["contact_channels"]),
            }
        return {
            **dict(row),
            "decision": "provide",
            "identity_record_id": str(row["id"]),
            "contact_complete": contact is not None,
            "contact_record_id": str(contact["id"]) if contact else None,
        }

    def save_contact(
        self,
        *,
        run_id: str,
        participant_id: str,
        email: str,
        phone: str,
        wechat: str,
        notice_version: str,
        notice_hash: str,
        ip_hash: str,
        user_agent: str,
    ) -> tuple[str, bool]:
        """Store at least one contact channel outside the research database."""
        self.require_ready()
        assert self._fernet is not None
        normalized = {
            "email": email.strip(),
            "phone": phone.strip(),
            "wechat": wechat.strip(),
        }
        channels = sorted(key for key, value in normalized.items() if value)
        if not channels:
            raise ValueError("at least one contact channel is required")
        payload_sha256 = hashlib.sha256(
            b"\x00".join(
                (
                    normalized["email"].encode("utf-8"),
                    normalized["phone"].encode("utf-8"),
                    normalized["wechat"].encode("utf-8"),
                    notice_hash.encode("ascii"),
                )
            )
        ).hexdigest()
        record_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id FROM compensation_contacts WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
            if existing is not None:
                return str(existing["id"]), False
            connection.execute(
                """INSERT INTO compensation_contacts
                   (id, run_id, participant_id, email_ciphertext,
                    phone_ciphertext, wechat_ciphertext, contact_channels,
                    notice_version, notice_hash, submitted_ms, ip_hash,
                    user_agent, payload_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    run_id,
                    participant_id,
                    self._fernet.encrypt(normalized["email"].encode("utf-8"))
                    if normalized["email"]
                    else None,
                    self._fernet.encrypt(normalized["phone"].encode("utf-8"))
                    if normalized["phone"]
                    else None,
                    self._fernet.encrypt(normalized["wechat"].encode("utf-8"))
                    if normalized["wechat"]
                    else None,
                    ",".join(channels),
                    notice_version,
                    notice_hash,
                    int(time.time() * 1000),
                    ip_hash,
                    user_agent[:500],
                    payload_sha256,
                ),
            )
        return record_id, True

    def count(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM compensation_identities"
            ).fetchone()
        return int(row["count"] if row else 0)

    def registration_count(self, decision: str | None = None) -> int:
        with self.connect() as connection:
            if decision is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM compensation_registrations"
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT COUNT(*) AS count FROM compensation_registrations
                       WHERE decision=?""",
                    (decision,),
                ).fetchone()
        return int(row["count"] if row else 0)

    def contact_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM compensation_contacts"
            ).fetchone()
        return int(row["count"] if row else 0)

    @staticmethod
    def normalize_person_name(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip()).casefold()

    @staticmethod
    def normalize_contact_email(value: str) -> str:
        return (value or "").strip().casefold()

    @staticmethod
    def normalize_contact_phone(value: str) -> str:
        return re.sub(r"\D", "", value or "")

    @staticmethod
    def normalize_contact_wechat(value: str) -> str:
        return (value or "").strip().casefold()

    def match_participant_ids_by_contact(
        self, *, email: str = "", phone: str = "", wechat: str = ""
    ) -> list[str]:
        """Return participant IDs whose encrypted contact matches all given fields."""
        wanted_email = self.normalize_contact_email(email)
        wanted_phone = self.normalize_contact_phone(phone)
        wanted_wechat = self.normalize_contact_wechat(wechat)
        if not (wanted_email or wanted_phone or wanted_wechat):
            return []
        self.require_ready()
        assert self._fernet is not None
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT participant_id, email_ciphertext, phone_ciphertext,
                          wechat_ciphertext
                   FROM compensation_contacts"""
            ).fetchall()
        matches: list[str] = []
        for row in rows:
            try:
                stored_email = (
                    self._fernet.decrypt(bytes(row["email_ciphertext"])).decode("utf-8")
                    if row["email_ciphertext"] is not None
                    else ""
                )
                stored_phone = (
                    self._fernet.decrypt(bytes(row["phone_ciphertext"])).decode("utf-8")
                    if row["phone_ciphertext"] is not None
                    else ""
                )
                stored_wechat = (
                    self._fernet.decrypt(bytes(row["wechat_ciphertext"])).decode("utf-8")
                    if row["wechat_ciphertext"] is not None
                    else ""
                )
            except InvalidToken:
                continue
            if wanted_email and self.normalize_contact_email(stored_email) != wanted_email:
                continue
            if wanted_phone and self.normalize_contact_phone(stored_phone) != wanted_phone:
                continue
            if wanted_wechat and self.normalize_contact_wechat(stored_wechat) != wanted_wechat:
                continue
            matches.append(str(row["participant_id"]))
        return list(dict.fromkeys(matches))

    def match_participant_ids_by_full_name(self, name: str) -> list[str]:
        needle = self.normalize_person_name(name)
        if len(needle) < 2:
            return []
        self.require_ready()
        assert self._fernet is not None
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT participant_id, full_name_ciphertext FROM compensation_identities"
            ).fetchall()
        matches: list[str] = []
        for row in rows:
            try:
                stored = self._fernet.decrypt(
                    bytes(row["full_name_ciphertext"])
                ).decode("utf-8")
            except InvalidToken:
                continue
            if self.normalize_person_name(stored) == needle:
                matches.append(str(row["participant_id"]))
        return list(dict.fromkeys(matches))

    def contact_metadata(
        self, participant_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        """Return non-sensitive contact completion metadata for Admin views."""
        ids = sorted({str(value) for value in participant_ids if value})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT participant_id, contact_channels
                    FROM compensation_contacts
                    WHERE participant_id IN ({placeholders})""",
                ids,
            ).fetchall()
        return {
            str(row["participant_id"]): {
                "complete": True,
                "channels": [
                    value
                    for value in str(row["contact_channels"] or "").split(",")
                    if value
                ],
            }
            for row in rows
        }

    def admin_contact_directory(
        self, participant_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        """Decrypt live-study contact fields for the authenticated admin UI."""
        ids = list(dict.fromkeys(str(value) for value in participant_ids if value))
        if not ids or not self.ready:
            return {}
        assert self._fernet is not None
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT participant_id, email_ciphertext, phone_ciphertext,
                           wechat_ciphertext, contact_channels
                    FROM compensation_contacts
                    WHERE participant_id IN ({placeholders})""",
                ids,
            ).fetchall()

        def decrypt_optional(value: Any) -> str:
            if value is None:
                return ""
            return self._fernet.decrypt(bytes(value)).decode("utf-8")

        directory: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                email = decrypt_optional(row["email_ciphertext"])
                phone = decrypt_optional(row["phone_ciphertext"])
                wechat = decrypt_optional(row["wechat_ciphertext"])
            except InvalidToken:
                continue
            channels = [
                value
                for value in str(row["contact_channels"] or "").split(",")
                if value
            ]
            directory[str(row["participant_id"])] = {
                "complete": True,
                "channels": channels,
                "email": email,
                "phone": phone,
                "wechat": wechat,
            }
        return directory

    def admin_identity_names(
        self, participant_ids: Iterable[str]
    ) -> dict[str, str]:
        """Decrypt reimbursement names only; never identity-document numbers."""
        ids = list(dict.fromkeys(str(value) for value in participant_ids if value))
        if not ids or not self.ready:
            return {}
        assert self._fernet is not None
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""SELECT participant_id, full_name_ciphertext
                    FROM compensation_identities
                    WHERE participant_id IN ({placeholders})""",
                ids,
            ).fetchall()
        names: dict[str, str] = {}
        for row in rows:
            try:
                name = re.sub(
                    r"\s+",
                    " ",
                    self._fernet.decrypt(
                        bytes(row["full_name_ciphertext"])
                    ).decode("utf-8"),
                ).strip()
            except InvalidToken:
                continue
            if name:
                names[str(row["participant_id"])] = name
        return names

    def save(
        self,
        *,
        run_id: str,
        participant_id: str,
        document_type: str,
        full_name: str,
        document_number: str,
        signature_png: bytes,
        notice_version: str,
        notice_hash: str,
        notice_json: str,
        affirmation: str,
        separate_consent: bool,
        ip_hash: str,
        user_agent: str,
    ) -> tuple[str, bool]:
        self.require_ready()
        if not separate_consent:
            raise ValueError("separate compensation identity consent is required")
        assert self._fernet is not None
        assert self._digest_key is not None
        normalized_document = "".join(document_number.upper().split())
        document_digest = hmac.new(
            self._digest_key,
            normalized_document.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        payload_sha256 = hashlib.sha256(
            b"\x00".join(
                (
                    full_name.encode("utf-8"),
                    normalized_document.encode("utf-8"),
                    signature_png,
                    notice_hash.encode("ascii"),
                    notice_json.encode("utf-8"),
                    affirmation.encode("utf-8"),
                )
            )
        ).hexdigest()
        record_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_registration = connection.execute(
                """SELECT id, identity_record_id FROM compensation_registrations
                   WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
            if existing_registration is not None:
                existing_id = (
                    existing_registration["identity_record_id"]
                    or existing_registration["id"]
                )
                return str(existing_id), False
            existing_identity = connection.execute(
                """SELECT id FROM compensation_identities
                   WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
            if existing_identity is not None:
                return str(existing_identity["id"]), False
            connection.execute(
                """INSERT INTO compensation_identities
                   (id, run_id, participant_id, document_type,
                    full_name_ciphertext, document_number_ciphertext,
                    document_number_digest, signature_png_ciphertext,
                    notice_version, notice_hash, notice_json, affirmation,
                    separate_consent,
                    submitted_ms, ip_hash, user_agent, payload_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    run_id,
                    participant_id,
                    document_type,
                    self._fernet.encrypt(full_name.encode("utf-8")),
                    self._fernet.encrypt(normalized_document.encode("utf-8")),
                    document_digest,
                    self._fernet.encrypt(signature_png),
                    notice_version,
                    notice_hash,
                    notice_json,
                    affirmation,
                    1,
                    int(time.time() * 1000),
                    ip_hash,
                    user_agent[:500],
                    payload_sha256,
                ),
            )
            connection.execute(
                """INSERT INTO compensation_registrations
                   (id, run_id, participant_id, decision, identity_record_id,
                    notice_version, notice_hash, notice_json, affirmation,
                    submitted_ms, ip_hash, user_agent, payload_sha256)
                   VALUES (?, ?, ?, 'provide', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    run_id,
                    participant_id,
                    record_id,
                    notice_version,
                    notice_hash,
                    notice_json,
                    affirmation,
                    int(time.time() * 1000),
                    ip_hash,
                    user_agent[:500],
                    payload_sha256,
                ),
            )
        return record_id, True

    def save_decline(
        self,
        *,
        run_id: str,
        participant_id: str,
        notice_version: str,
        notice_hash: str,
        notice_json: str,
        affirmation: str,
        ip_hash: str,
        user_agent: str,
    ) -> tuple[str, bool]:
        self.require_ready()
        payload_sha256 = hashlib.sha256(
            b"\x00".join(
                (
                    b"decline",
                    notice_hash.encode("ascii"),
                    notice_json.encode("utf-8"),
                    affirmation.encode("utf-8"),
                )
            )
        ).hexdigest()
        record_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_registration = connection.execute(
                """SELECT id FROM compensation_registrations
                   WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
            if existing_registration is not None:
                return str(existing_registration["id"]), False
            existing_identity = connection.execute(
                """SELECT id FROM compensation_identities
                   WHERE participant_id=?""",
                (participant_id,),
            ).fetchone()
            if existing_identity is not None:
                return str(existing_identity["id"]), False
            connection.execute(
                """INSERT INTO compensation_registrations
                   (id, run_id, participant_id, decision, identity_record_id,
                    notice_version, notice_hash, notice_json, affirmation,
                    submitted_ms, ip_hash, user_agent, payload_sha256)
                   VALUES (?, ?, ?, 'decline', NULL, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record_id,
                    run_id,
                    participant_id,
                    notice_version,
                    notice_hash,
                    notice_json,
                    affirmation,
                    int(time.time() * 1000),
                    ip_hash,
                    user_agent[:500],
                    payload_sha256,
                ),
            )
        return record_id, True

    def decrypt_for_offline_authorised_processing(
        self, record_id: str
    ) -> dict[str, Any]:
        """Explicit offline-only primitive; never expose this through FastAPI."""
        self.require_ready()
        assert self._fernet is not None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM compensation_identities WHERE id=?", (record_id,)
            ).fetchone()
        if row is None:
            raise KeyError(record_id)
        try:
            decrypted = {
                **dict(row),
                "full_name": self._fernet.decrypt(
                    bytes(row["full_name_ciphertext"])
                ).decode("utf-8"),
                "document_number": self._fernet.decrypt(
                    bytes(row["document_number_ciphertext"])
                ).decode("utf-8"),
                "signature_png": self._fernet.decrypt(
                    bytes(row["signature_png_ciphertext"])
                ),
            }
            contact = self.decrypt_contact_for_offline_authorised_processing(
                str(row["participant_id"])
            )
            return {**decrypted, **contact}
        except InvalidToken as exc:
            raise CompensationVaultUnavailable(
                "Compensation record could not be decrypted with this key"
            ) from exc

    def decrypt_contact_for_offline_authorised_processing(
        self, participant_id: str
    ) -> dict[str, Any]:
        """Offline-only contact retrieval; never expose through FastAPI."""
        self.require_ready()
        assert self._fernet is not None
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM compensation_contacts WHERE participant_id=?",
                (participant_id,),
            ).fetchone()
        if row is None:
            raise KeyError(participant_id)
        try:
            def decrypt_optional(column: str) -> str:
                value = row[column]
                if value is None:
                    return ""
                return self._fernet.decrypt(bytes(value)).decode("utf-8")

            return {
                "contact_record_id": str(row["id"]),
                "contact_email": decrypt_optional("email_ciphertext"),
                "contact_phone": decrypt_optional("phone_ciphertext"),
                "contact_wechat": decrypt_optional("wechat_ciphertext"),
                "contact_channels": str(row["contact_channels"]),
            }
        except InvalidToken as exc:
            raise CompensationVaultUnavailable(
                "Compensation contact could not be decrypted with this key"
            ) from exc
