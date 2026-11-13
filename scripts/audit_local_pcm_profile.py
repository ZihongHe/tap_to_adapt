"""Silently audit every unique utterance for one local-TTS simulation profile."""

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
    parser.add_argument("--turns", type=Path, required=True)
    parser.add_argument("--participant-index", type=int, required=True)
    parser.add_argument("--speaker-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.speech import SpeechService

    turns = json.loads(args.turns.read_text(encoding="utf-8"))
    participant = next(
        row
        for row in turns["participants"]
        if int(row["participant_index"]) == args.participant_index
    )
    service = SpeechService(settings)
    records = []
    seen = set()
    for event in participant["events"]:
        role = str(event["speaker_role"])
        text = str(event["text"])
        key = (role, text)
        if key in seen:
            continue
        seen.add(key)
        if role == "participant":
            pcm = service.synthesize_pcm(text, speaker_id=args.speaker_id, speed=1.0)
        else:
            pcm = service.synthesize_agent_pcm(text, client_role="companion")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
        records.append(
            {
                "speaker_role": role,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "samples": int(samples.size),
                "duration_ms": round(samples.size * 1000 / 16000.0, 1),
                "rms": round(rms, 8),
                "passes_streaming_qa": samples.size >= 6400 and rms >= 0.004,
            }
        )
    failures = [row for row in records if not row["passes_streaming_qa"]]
    payload = {
        "schema": "tap-to-adapt-local-pcm-profile-audit/v1",
        "simulation_only": True,
        "silent_playback": True,
        "participant_index": args.participant_index,
        "speaker_id": args.speaker_id,
        "utterances": len(records),
        "failures": failures,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "participant_index": args.participant_index,
                "utterances": len(records),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
