"""Command-line utility: voice pcm.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable


PCM_SAMPLE_RATE_HZ = 16_000
PCM_SAMPLE_WIDTH_BYTES = 2
TIMING_FRAME_BYTES = 6_400
MIN_VOICED_SECONDS = 0.4
MIN_PCM_RMS = 0.004
XUNFEI_ENV_NAMES = (
    "XUNFEI_APP_ID",
    "XUNFEI_API_KEY",
    "XUNFEI_API_SECRET",
)


def decode_mp3_to_pcm16(
    audio: bytes,
    ffmpeg_binary: str,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> bytes:
    """Decode an in-memory MP3 to 16 kHz mono signed little-endian PCM.

    The audio is deliberately passed through stdin/stdout.  The QA runner must
    never leave either provider credentials or synthesized speech on disk.
    """
    if not audio:
        raise RuntimeError("iFLYTEK returned empty audio")
    try:
        result = runner(
            [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "mp3",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(PCM_SAMPLE_RATE_HZ),
                "-f",
                "s16le",
                "pipe:1",
            ],
            input=audio,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("ffmpeg could not decode the synthesized speech") from error
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg could not decode the synthesized speech: {detail}")
    pcm = bytes(result.stdout)
    if len(pcm) % PCM_SAMPLE_WIDTH_BYTES:
        raise RuntimeError("Decoded PCM has a partial int16 sample")
    return pcm


def _pcm_rms(pcm: bytes) -> float:
    # Keep this runner import-light: the study server has numpy, but decoding
    # validation should also work in a minimal provisioning environment.
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    mean_square = sum((value / 32768.0) ** 2 for value in samples) / len(samples)
    return mean_square**0.5


def validate_voiced_pcm(pcm: bytes) -> dict[str, float | int]:
    if len(pcm) % PCM_SAMPLE_WIDTH_BYTES:
        raise ValueError("PCM must contain complete int16 samples")
    samples = len(pcm) // PCM_SAMPLE_WIDTH_BYTES
    minimum_samples = int(PCM_SAMPLE_RATE_HZ * MIN_VOICED_SECONDS)
    if samples < minimum_samples:
        raise ValueError(
            f"Decoded speech is too short: need at least {minimum_samples} samples"
        )
    rms = _pcm_rms(pcm)
    if rms < MIN_PCM_RMS:
        raise ValueError("Decoded speech is silent or below the voiced QA threshold")
    return {
        "sample_rate_hz": PCM_SAMPLE_RATE_HZ,
        "channels": 1,
        "sample_width_bytes": PCM_SAMPLE_WIDTH_BYTES,
        "samples": samples,
        "duration_ms": round(samples * 1000 / PCM_SAMPLE_RATE_HZ, 1),
        "rms": round(rms, 6),
    }


def iter_timing_pcm_frames(
    pcm: bytes, *, frame_bytes: int = TIMING_FRAME_BYTES
) -> Iterable[bytes]:
    """Yield browser-sized PCM frames accepted by the public WebSocket proxy."""
    if frame_bytes <= 0 or frame_bytes > TIMING_FRAME_BYTES:
        raise ValueError(f"frame_bytes must be in [1, {TIMING_FRAME_BYTES}]")
    if frame_bytes % PCM_SAMPLE_WIDTH_BYTES:
        raise ValueError("frame_bytes must preserve int16 sample boundaries")
    if len(pcm) % PCM_SAMPLE_WIDTH_BYTES:
        raise ValueError("PCM must contain complete int16 samples")
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if frame:
            yield frame


def stream_timing_pcm(
    socket: Any,
    pcm: bytes,
    receive_json: Callable[[Any], dict[str, Any]],
    *,
    pace: bool = True,
    stop_on_activation: bool = False,
    on_pair_required: Callable[[], None] | None = None,
    pair_recovery_attempts: int = 3,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Send voiced PCM with the same framing/cadence as a browser recorder."""
    if pair_recovery_attempts < 0:
        raise ValueError("pair_recovery_attempts cannot be negative")
    responses: list[dict[str, Any]] = []
    frames = list(iter_timing_pcm_frames(pcm))
    for index, frame in enumerate(frames):
        recoveries = 0
        while True:
            socket.send_binary(frame)
            response = receive_json(socket)
            if response.get("type") != "pair_required":
                break
            if (
                on_pair_required is None
                or recoveries >= pair_recovery_attempts
            ):
                raise RuntimeError(
                    "Timing PCM was paused because pair presence could not be restored"
                )
            on_pair_required()
            recoveries += 1
        if response.get("type") != "prediction":
            raise RuntimeError(f"Unexpected timing PCM response type: {response.get('type')}")
        responses.append(response)
        if stop_on_activation and response.get("activate") is True:
            break
        if pace and index + 1 < len(frames):
            sleeper(len(frame) / (PCM_SAMPLE_RATE_HZ * PCM_SAMPLE_WIDTH_BYTES))
    return responses


def synthesize_xunfei_pcm(
    text: str,
    *,
    settings_factory: Callable[[], Any] | None = None,
    service_factory: Callable[[Any], Any] | None = None,
) -> tuple[bytes, dict[str, float | int | str]]:
    """Synthesize speech with credentials read strictly from process env.

    Provider exceptions are intentionally replaced with a credential-free
    message because some transport libraries may include a signed URL in their
    exception text.  No dotenv file is loaded by this function.
    """
    missing = [name for name in XUNFEI_ENV_NAMES if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required iFLYTEK environment variables: " + ", ".join(missing)
        )

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if settings_factory is None or service_factory is None:
        from app.config import Settings
        from app.speech import SpeechService

        settings_factory = settings_factory or Settings.from_env
        service_factory = service_factory or SpeechService

    base_settings = settings_factory()
    settings = replace(
        base_settings,
        tts_mode="xunfei",
        xunfei_app_id=os.environ["XUNFEI_APP_ID"],
        xunfei_api_key=os.environ["XUNFEI_API_KEY"],
        xunfei_api_secret=os.environ["XUNFEI_API_SECRET"],
    )
    service = service_factory(settings)
    try:
        mp3 = service.synthesize(text)
    except BaseException:
        # Never chain the provider exception: it can contain a signed URL.
        raise RuntimeError(
            "iFLYTEK synthesis failed; credential-bearing details were suppressed"
        ) from None
    pcm = decode_mp3_to_pcm16(mp3, settings.ffmpeg_binary)
    metadata = validate_voiced_pcm(pcm)
    return pcm, {
        **metadata,
        "provider": "iflytek",
        "frame_bytes_max": TIMING_FRAME_BYTES,
        "frame_count": sum(1 for _ in iter_timing_pcm_frames(pcm)),
    }
