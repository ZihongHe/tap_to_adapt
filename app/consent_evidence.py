"""Canonical consent records, hashes, and digital-signature verification.

The signer creates verifiable evidence for a versioned consent decision.
Verification uses public-key material; the private signing seed remains an
instance secret and is never distributed with a research export.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


EVIDENCE_SCHEMA_VERSION = "tap-to-adapt-consent-evidence-v2"
SIGNATURE_ALGORITHM = "Ed25519"
ASSURANCE_SCOPE = "pseudonymous-research-consent"


class ConsentEvidenceUnavailable(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decode_urlsafe(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _encode_urlsafe(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def public_key_id(public_key_raw: bytes) -> str:
    return f"ed25519-sha256:{hashlib.sha256(public_key_raw).hexdigest()}"


@dataclass(frozen=True)
class ConsentEvidenceSigner:
    """Instance-scoped signer for third-party-verifiable consent receipts.

    The signature authenticates the study server's receipt, document snapshot,
    permission choices and append-only chain position.  It deliberately does
    not claim to certify a participant's civil identity.
    """

    instance_id: str
    private_key: Ed25519PrivateKey | None

    @classmethod
    def from_secret(cls, instance_id: str, secret: str) -> "ConsentEvidenceSigner":
        normalized = secret.strip()
        if not normalized:
            return cls(instance_id=instance_id, private_key=None)
        try:
            raw = _decode_urlsafe(normalized)
            if len(raw) != 32:
                raise ValueError("Ed25519 seed must contain 32 bytes")
            private_key = Ed25519PrivateKey.from_private_bytes(raw)
        except (TypeError, ValueError) as exc:
            raise ConsentEvidenceUnavailable(
                "CONSENT_EVIDENCE_SIGNING_KEY is invalid"
            ) from exc
        return cls(instance_id=instance_id, private_key=private_key)

    @property
    def ready(self) -> bool:
        return self.private_key is not None

    def require_ready(self) -> None:
        if not self.ready:
            raise ConsentEvidenceUnavailable(
                "Consent evidence signing is not configured"
            )

    @property
    def public_key_raw(self) -> bytes:
        self.require_ready()
        assert self.private_key is not None
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @property
    def public_key_b64(self) -> str:
        return _encode_urlsafe(self.public_key_raw)

    @property
    def key_id(self) -> str:
        return public_key_id(self.public_key_raw)

    def descriptor(self) -> dict[str, Any]:
        self.require_ready()
        return {
            "schema_version": "tap-to-adapt-consent-evidence-key-v1",
            "instance_id": self.instance_id,
            "algorithm": SIGNATURE_ALGORITHM,
            "public_key_base64url": self.public_key_b64,
            "key_id": self.key_id,
            "assurance_scope": ASSURANCE_SCOPE,
            "identity_certification": False,
        }

    def sign(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.require_ready()
        if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("Consent evidence payload schema is invalid")
        if payload.get("instance_id") != self.instance_id:
            raise ValueError("Consent evidence payload belongs to another instance")
        assert self.private_key is not None
        canonical = canonical_json(payload).encode("utf-8")
        signature = self.private_key.sign(canonical)
        return {
            "schema_version": "tap-to-adapt-consent-evidence-bundle-v1",
            "payload": payload,
            "payload_sha256": hashlib.sha256(canonical).hexdigest(),
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "public_key_base64url": self.public_key_b64,
                "key_id": self.key_id,
                "value_base64url": _encode_urlsafe(signature),
            },
        }


def verify_evidence_bundle(
    bundle: dict[str, Any],
    *,
    expected_key_id: str | None = None,
    expected_instance_id: str | None = None,
) -> dict[str, Any]:
    if bundle.get("schema_version") != "tap-to-adapt-consent-evidence-bundle-v1":
        raise ValueError("Unsupported consent evidence bundle schema")
    payload = bundle.get("payload")
    signature = bundle.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature, dict):
        raise ValueError("Consent evidence bundle is incomplete")
    if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported consent evidence payload schema")
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        raise ValueError("Unsupported consent evidence signature algorithm")
    try:
        public_raw = _decode_urlsafe(str(signature["public_key_base64url"]))
        signature_raw = _decode_urlsafe(str(signature["value_base64url"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Consent evidence signature encoding is invalid") from exc
    if len(public_raw) != 32 or len(signature_raw) != 64:
        raise ValueError("Consent evidence signature length is invalid")
    actual_key_id = public_key_id(public_raw)
    if signature.get("key_id") != actual_key_id:
        raise ValueError("Consent evidence key fingerprint does not match the public key")
    if expected_key_id is not None and actual_key_id != expected_key_id:
        raise ValueError("Consent evidence was signed by an unexpected key")
    if (
        expected_instance_id is not None
        and payload.get("instance_id") != expected_instance_id
    ):
        raise ValueError("Consent evidence belongs to an unexpected study instance")
    canonical = canonical_json(payload).encode("utf-8")
    actual_sha256 = hashlib.sha256(canonical).hexdigest()
    if bundle.get("payload_sha256") != actual_sha256:
        raise ValueError("Consent evidence payload hash does not match")
    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(
            signature_raw,
            canonical,
        )
    except InvalidSignature as exc:
        raise ValueError("Consent evidence signature is invalid") from exc
    return {
        "valid": True,
        "instance_id": payload.get("instance_id"),
        "consent_id": payload.get("consent_id"),
        "ledger_sequence": payload.get("ledger_sequence"),
        "payload_sha256": actual_sha256,
        "key_id": actual_key_id,
        "assurance_scope": payload.get("assurance_scope"),
    }
