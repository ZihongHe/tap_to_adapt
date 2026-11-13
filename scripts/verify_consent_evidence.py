#!/usr/bin/env python3
"""Offline verification for a Tap-to-Adapt consent evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from app.consent_evidence import canonical_json, verify_evidence_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--expected-key-id")
    parser.add_argument("--expected-instance-id")
    parser.add_argument(
        "--signed-document",
        type=Path,
        help="Optional signed-document.json whose canonical SHA-256 must match the evidence",
    )
    args = parser.parse_args()
    try:
        bundle = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = verify_evidence_bundle(
            bundle,
            expected_key_id=args.expected_key_id,
            expected_instance_id=args.expected_instance_id,
        )
        if args.signed_document:
            document = json.loads(args.signed_document.read_text(encoding="utf-8"))
            document_sha256 = hashlib.sha256(
                json.dumps(document, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            expected_document_sha256 = str(
                bundle["payload"].get("document", {}).get("sha256") or ""
            )
            if document_sha256 != expected_document_sha256:
                raise ValueError("signed document hash does not match the evidence payload")
            result["signed_document_valid"] = True
        result["evidence_file_sha256"] = hashlib.sha256(
            canonical_json(bundle).encode("utf-8")
        ).hexdigest()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
