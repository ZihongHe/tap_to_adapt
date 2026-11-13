"""Command-line utility: tts self test.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mp3(audio: bytes, ffmpeg_binary: str) -> None:
    decoded = subprocess.run(
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
    if decoded.returncode != 0:
        detail = decoded.stderr.decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Generated MP3 failed decode validation: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(Path(args.env_file) if args.env_file else root / ".env", override=True)

    from app.config import settings
    from app.speech import SpeechService

    if settings.tts_mode != "sherpa_onnx":
        raise RuntimeError("TTS_MODE must be sherpa_onnx for the v6 self-test")
    service = SpeechService(settings)
    if not service.available:
        raise RuntimeError("The locked local TTS model is incomplete")

    started = time.perf_counter()
    audio = service.synthesize("这是语音系统的固定连通性测试，今天很高兴和你聊天。")
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    validate_mp3(audio, settings.ffmpeg_binary)

    def concurrent_request(index: int) -> dict[str, float | int]:
        request_started = time.perf_counter()
        value = service.synthesize(f"这是第{index + 1}路并发语音测试。")
        validate_mp3(value, settings.ffmpeg_binary)
        return {
            "bytes": len(value),
            "latency_ms": round((time.perf_counter() - request_started) * 1000, 1),
        }

    concurrency_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as executor:
        concurrent_results = list(executor.map(concurrent_request, range(6)))
    concurrency_elapsed_ms = round(
        (time.perf_counter() - concurrency_started) * 1000, 1
    )

    model_path = Path(settings.sherpa_tts_model_dir) / "model.onnx"
    sample_path = settings.data_dir / "tts-self-test.mp3"
    sample_path.write_bytes(audio)
    sample_path.chmod(0o600)
    sentinel = {
        "passed": True,
        "created_ms": int(time.time() * 1000),
        "provider": service.provider,
        "voice": service.voice,
        "speed": service.speed,
        "sample_rate_hz": service.sample_rate_hz,
        "format": service.output_format,
        "model_path": str(model_path.resolve()),
        "model_sha256": file_sha256(model_path),
        "audio_bytes": len(audio),
        "audio_sha256": hashlib.sha256(audio).hexdigest(),
        "latency_ms": latency_ms,
        "decode_validation": "passed",
        "concurrency": {
            "workers": 6,
            "passed": len(concurrent_results) == 6,
            "elapsed_ms": concurrency_elapsed_ms,
            "results": concurrent_results,
        },
    }
    target = Path(settings.tts_ready_sentinel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(sentinel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    target.chmod(0o600)
    print(json.dumps(sentinel, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
