"""Command-line utility: configure compensation vault.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet


def configure(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    key_name = "COMPENSATION_VAULT_KEY"
    existing = ""
    target_index: int | None = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key_name:
            target_index = index
            existing = value.strip()
            break
    if existing:
        os.chmod(path, 0o600)
        return "already_configured"

    generated = Fernet.generate_key().decode("ascii")
    replacement = f"{key_name}={generated}"
    if target_index is None:
        lines.append(replacement)
    else:
        lines[target_index] = replacement
    temporary = path.with_name(f".{path.name}.compensation-vault-new")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    return "configured"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: configure_compensation_vault.py /path/to/.env")
    result = configure(Path(sys.argv[1]).resolve())
    print(f"compensation_vault={result}")
