"""Silent self-test for the two locked local agent voices."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.speech import SpeechService

    audit = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    service = SpeechService(settings)
    voices = {
        "study": service.voice_for("study"),
        "companion": service.voice_for("companion"),
    }
    records = {}
    for role, text in {
        "study": "我听到了，你愿意继续说说具体发生了什么吗？",
        "companion": "我们换个角度聊聊，你最近有没有新的计划？",
    }.items():
        pcm = service.synthesize_agent_pcm(text, client_role=role)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        records[role] = {
            "voice": voices[role],
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "pcm_bytes": len(pcm),
            "duration_seconds": round(samples.size / 16000.0, 3),
            "rms": round(float(np.sqrt(np.mean(samples**2))), 6),
        }
    expected = {
        audit["experimental_agent"]["voice"],
        audit["companion_agent"]["voice"],
    }
    passed = (
        service.provider == "sherpa_onnx_vits_aishell3"
        and set(voices.values()) == expected
        and len(set(voices.values())) == 2
        and records["study"]["pcm_sha256"] != records["companion"]["pcm_sha256"]
        and all(record["pcm_bytes"] > 3200 for record in records.values())
        and all(record["rms"] > 0.001 for record in records.values())
    )
    result = {
        "schema": "tap-to-adapt-local-agent-voice-smoke/v1",
        "passed": passed,
        "silent_playback": True,
        "provider": service.provider,
        "roles": records,
        "selection_audit_sha256": hashlib.sha256(
            args.selection_audit.read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
