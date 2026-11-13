"""One-shot TTS provider smoke test; credentials are read only from the environment.

Run with --help for required input paths, selection options, and output files.
Supply review-local artifacts and endpoints when exercising this utility.
Runtime files and credentials are not part of the source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def validate_mp3(audio: bytes, ffmpeg_binary: str) -> None:
    result = subprocess.run(
        [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "mp3",
            "-i",
            "pipe:0",
            "-f",
            "null",
            "-",
        ],
        input=audio,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Generated audio is not a valid MP3: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot TTS provider smoke test; credentials are read only from the environment."
    )
    parser.add_argument(
        "--text",
        default="这是语音交互预实验的连通性测试，请在状态切换后停止操作。",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.speech import SpeechService

    if settings.tts_mode not in {"xunfei", "sherpa_onnx"}:
        raise RuntimeError("TTS_MODE must be xunfei or sherpa_onnx")
    service = SpeechService(settings)
    if not service.available:
        raise RuntimeError(f"{settings.tts_mode} is not fully configured")

    started = time.perf_counter()
    audio = service.synthesize(args.text)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    validate_mp3(audio, settings.ffmpeg_binary)
    print(
        json.dumps(
            {
                "passed": True,
                "provider": service.provider,
                "voice": service.voice,
                "format": service.output_format,
                "sample_rate_hz": service.sample_rate_hz,
                "audio_bytes": len(audio),
                "audio_sha256": hashlib.sha256(audio).hexdigest(),
                "latency_ms": latency_ms,
                "decode_validation": "passed",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
