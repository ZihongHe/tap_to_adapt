"""Atomically lock the two locally selected agent voices in the server .env."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    if (
        audit.get("schema") != "tap-to-adapt-local-agent-voice-selection/v1"
        or audit.get("all_voices_mutually_exclusive") is not True
        or audit.get("silent_synthesis") is not True
    ):
        raise SystemExit("Voice-selection audit is absent or invalid")
    primary = int(audit["experimental_agent"]["speaker_id"])
    companion = int(audit["companion_agent"]["speaker_id"])
    if primary == companion or primary in audit["excluded_participant_ids"] or companion in audit["excluded_participant_ids"]:
        raise SystemExit("Voice-selection audit violates role exclusivity")

    replacements = {
        "TTS_MODE": "sherpa_onnx",
        "SHERPA_TTS_SPEAKER_ID": str(primary),
        "SHERPA_TTS_PRIMARY_SPEAKER_ID": str(primary),
        "SHERPA_TTS_COMPANION_SPEAKER_ID": str(companion),
    }
    original = args.env_file.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = []
    for line in original:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in replacements:
            output.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in replacements.items():
        if key not in seen:
            output.append(f"{key}={value}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.env_file.name}.", dir=args.env_file.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(output) + "\n")
        os.chmod(temporary_name, args.env_file.stat().st_mode)
        os.replace(temporary_name, args.env_file)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "tts_mode": "sherpa_onnx",
                "experimental_agent_voice": f"aishell3-sid{primary}",
                "companion_agent_voice": f"aishell3-sid{companion}",
                "credentials_read_or_printed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
