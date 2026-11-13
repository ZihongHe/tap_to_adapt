"""Create a private local mock configuration with independent random secrets.

Existing configuration is never overwritten. Generated values are never printed.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import secrets


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / ".env"
    if destination.exists():
        raise SystemExit("Existing .env retained.")
    settings = {
        key: secrets.token_urlsafe(32)
        for key in (
            "STUDY_TOKEN_SECRET", "STUDY_CODE_PEPPER",
            "STUDY_IP_PEPPER", "STUDY_ADMIN_TOKEN",
        )
    }
    settings.update({
        "COMPENSATION_VAULT_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        "CONSENT_EVIDENCE_SIGNING_KEY": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        "AGENT_MODE": "mock", "ASR_MODE": "browser",
        "SERVER_ASR_ENABLED": "0", "TTS_MODE": "browser",
        "FORMAL_STUDY_ENABLED": "0", "STUDY_DATA_DIR": "./runtime-data",
        "STUDY_MEDIA_DIR": "./runtime-data/media",
    })
    lines = (root / ".env.example").read_text(encoding="utf-8").splitlines()
    output = []
    for line in lines:
        key = line.partition("=")[0]
        output.append(f"{key}={settings.pop(key)}" if key in settings else line)
    output.extend(f"{key}={value}" for key, value in settings.items())
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(output) + "\n")
    print("Created local .env. Keep it private.")


if __name__ == "__main__":
    main()
