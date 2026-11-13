"""Cancellable server-side speech recognition for short PCM utterances.

Whisper replicas are loaded lazily and protected against concurrent use.
Recognition results supply conversational content after activation; they are
separate from the timing model's decision and corrective-tap labels.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class ServerASRUnavailable(RuntimeError):
    """Raised when the server-side recognizer cannot be used."""


class ServerASRCancelled(RuntimeError):
    """Raised when a transcription is abandoned before it occupies a replica."""


@dataclass(frozen=True)
class ServerASRResult:
    text: str
    model: str
    language: str
    duration_seconds: float
    latency_ms: float
    signal_rms: float


class WhisperASRService:
    """In-memory Whisper transcription for short 16 kHz PCM utterances.

    The service never writes participant audio to disk.  Independent model
    replicas decode in parallel so concurrent subjects do not share one lock.
    A cancelled or superseded turn leaves the replica queue without starting
    another decode.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        model_name: str,
        device: str,
        download_root: str,
        max_seconds: float,
        min_rms: float,
        replica_count: int = 1,
        devices: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_name = str(model_name).strip() or "large-v3"
        self.device = str(device).strip() or "cuda"
        self.devices = tuple(
            str(item).strip()
            for item in (devices or (self.device,))
            if str(item).strip()
        ) or (self.device,)
        self.download_root = str(download_root).strip()
        self.max_seconds = max(1.0, float(max_seconds))
        self.min_rms = max(0.0, float(min_rms))
        self.replica_count = max(1, int(replica_count or 1))
        self._model: Any | None = None
        self._replicas: list[Any] = []
        self._replica_devices: list[str] = []
        self._pool: queue.Queue[Any] = queue.Queue()
        self._lock = threading.RLock()

    def _create_model(self, device: str) -> Any:
        if not self.enabled:
            raise ServerASRUnavailable("Server-side ASR is disabled")
        try:
            import whisper

            kwargs: dict[str, Any] = {"device": device}
            if self.download_root:
                kwargs["download_root"] = str(Path(self.download_root).expanduser())
            return whisper.load_model(self.model_name, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on deployment GPU
            logger.exception("Server-side Whisper model could not be loaded")
            raise ServerASRUnavailable("Server-side ASR is unavailable") from exc

    def _ensure_pool(self) -> None:
        if self._replicas:
            return
        with self._lock:
            if self._replicas:
                return
            for index in range(self.replica_count):
                device = self.devices[index % len(self.devices)]
                try:
                    replica = self._create_model(device)
                except Exception:
                    logger.exception(
                        "Whisper replica %s/%s on %s could not be loaded; keeping %s",
                        index + 1,
                        self.replica_count,
                        device,
                        len(self._replicas),
                    )
                    break
                self._replicas.append(replica)
                self._replica_devices.append(device)
                self._pool.put(replica)
            if self._replicas:
                self._model = self._replicas[0]

    def _uses_injected_model(self) -> bool:
        return self._model is not None and not self._replicas

    def _acquire_model(
        self,
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[Any, bool]:
        if cancel_check and cancel_check():
            raise ServerASRCancelled("ASR cancelled before a replica was free")
        if self._uses_injected_model():
            return self._model, False
        self._ensure_pool()
        while True:
            if cancel_check and cancel_check():
                raise ServerASRCancelled("ASR cancelled before a replica was free")
            try:
                model = self._pool.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        if cancel_check and cancel_check():
            self._pool.put(model)
            raise ServerASRCancelled("ASR cancelled before decode started")
        return model, True

    def warmup(self) -> None:
        """Load replicas without making service startup depend on them."""

        self._ensure_pool()

    def transcribe_pcm16(
        self,
        pcm16: bytes,
        *,
        sample_rate_hz: int,
        language: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ServerASRResult:
        if sample_rate_hz != 16_000:
            raise ValueError("Server ASR accepts only 16 kHz PCM")
        if language not in {"zh", "en"}:
            raise ValueError("ASR language must be zh or en")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("PCM payload must contain complete Int16 samples")

        sample_count = len(pcm16) // 2
        duration_seconds = sample_count / float(sample_rate_hz)
        if duration_seconds > self.max_seconds + 0.05:
            raise ValueError("PCM payload exceeds the configured ASR duration")
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        signal_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
        # Whisper can hallucinate fluent text on digital silence. This guard
        # only suppresses an unusable transcript; it never participates in the
        # timing model's response decision. The caller still responds with an
        # explicit request to repeat when both ASR paths are empty.
        if signal_rms < self.min_rms:
            return ServerASRResult(
                text="",
                model=f"whisper-{self.model_name}",
                language=language,
                duration_seconds=round(duration_seconds, 3),
                latency_ms=0.0,
                signal_rms=round(signal_rms, 6),
            )

        started = time.perf_counter()
        model, pooled = self._acquire_model(cancel_check)
        try:
            result = model.transcribe(
                audio,
                language=language,
                task="transcribe",
                fp16=self.device.startswith("cuda"),
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=False,
            )
        except ServerASRCancelled:
            raise
        except Exception as exc:  # pragma: no cover - provider/runtime failure
            logger.exception("Server-side Whisper transcription failed")
            raise ServerASRUnavailable("Server-side ASR transcription failed") from exc
        finally:
            if pooled:
                self._pool.put(model)
        text = str((result or {}).get("text") or "").strip()
        return ServerASRResult(
            text=text,
            model=f"whisper-{self.model_name}",
            language=language,
            duration_seconds=round(duration_seconds, 3),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            signal_rms=round(signal_rms, 6),
        )
