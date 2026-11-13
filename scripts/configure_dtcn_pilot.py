"""Configure a DTCN engineering-pilot prior.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def replace_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    prefix = f"{key}="
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = f"{prefix}{value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{prefix}{value}")
    temporary = path.with_suffix(".new")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    temporary = target.with_suffix(target.suffix + ".new")
    shutil.copy2(source, temporary)
    temporary.chmod(0o600)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure a DTCN engineering-pilot prior.")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--main-model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--acknowledge-engineering-prior", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_engineering_prior:
        raise SystemExit("Pass --acknowledge-engineering-prior after confirming this is pilot-only")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.integrity import file_sha256

    if settings.formal_study_enabled:
        raise SystemExit("Refusing to replace the prior while FORMAL_STUDY_ENABLED=1")
    if not args.encoder.is_file() or not args.main_model.is_file():
        raise SystemExit("Source checkpoint is missing")

    target_encoder = Path(settings.base_encoder_checkpoint)
    target_main = Path(settings.base_main_checkpoint)
    atomic_copy(args.encoder, target_encoder)
    atomic_copy(args.main_model, target_main)
    provenance = {
        "configured_ms": int(time.time() * 1000),
        "purpose": "engineering_pilot_only",
        "label": args.label,
        "source_encoder": str(args.encoder.resolve()),
        "source_main_model": str(args.main_model.resolve()),
        "encoder_sha256": file_sha256(target_encoder),
        "main_sha256": file_sha256(target_main),
    }
    provenance_path = settings.data_dir / "pilot-model-provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    provenance_path.chmod(0o600)

    subprocess.run([sys.executable, root / "scripts" / "model_self_test.py"], check=True)
    replace_env_value(root / ".env", "AGENT_MODE", "dtcn")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print("AGENT_MODE=dtcn written after prediction, online-update, save, and reload checks passed.")


if __name__ == "__main__":
    main()
