"""Command-line utility: setup server.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from __future__ import annotations

import secrets
import shutil
from pathlib import Path


SECRET_KEYS = {
    "STUDY_TOKEN_SECRET": 48,
    "STUDY_CODE_PEPPER": 48,
    "STUDY_IP_PEPPER": 48,
    "STUDY_ADMIN_TOKEN": 36,
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        values = {key: secrets.token_urlsafe(size) for key, size in SECRET_KEYS.items()}
        lines: list[str] = []
        for line in (root / ".env.example").read_text(encoding="utf-8").splitlines():
            key = line.split("=", 1)[0] if "=" in line else ""
            lines.append(f"{key}={values[key]}" if key in values else line)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        current = env_path.read_text(encoding="utf-8").splitlines()
        existing = {
            line.split("=", 1)[0]
            for line in current
            if "=" in line and not line.lstrip().startswith("#")
        }
        additions = [
            line
            for line in (root / ".env.example").read_text(encoding="utf-8").splitlines()
            if "=" in line and line.split("=", 1)[0] not in existing
        ]
        if additions:
            env_path.write_text(
                "\n".join(current + ["", "# Added by setup_server.py"] + additions) + "\n",
                encoding="utf-8",
            )

    env_path.chmod(0o600)
    data_dir = root / "data"
    media_dir = data_dir / "media"
    data_dir.mkdir(mode=0o700, exist_ok=True)
    media_dir.mkdir(mode=0o700, exist_ok=True)
    data_dir.chmod(0o700)
    media_dir.chmod(0o700)

    service_dir = Path.home() / ".config" / "systemd" / "user"
    service_dir.mkdir(parents=True, exist_ok=True)
    for service_source in (root / "deploy").glob("*.service"):
        shutil.copy2(service_source, service_dir / service_source.name)

    print(f"Prepared {root}")
    print("Formal study remains disabled until the readiness gate passes.")


if __name__ == "__main__":
    main()
