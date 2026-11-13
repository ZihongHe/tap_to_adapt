"""Regression tests for runtime isolation.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from app.config import Settings
from app.distilhubert_online import SharedEncoderBusy
from app.timing import ParticipantRuntime


class SlowPredictModel:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._calls = 0
        self._lock = threading.Lock()

    def predict(self, audio: np.ndarray, encoder_timeout: float | None = None):
        del audio, encoder_timeout
        with self._lock:
            self._calls += 1
            first = self._calls == 1
        if first:
            self.entered.set()
            assert self.release.wait(2.0)
        return 0.1, -2.0

    def can_train(self, audio: np.ndarray) -> bool:
        del audio
        return True


class BusyEncoderModel:
    def predict(self, audio: np.ndarray, encoder_timeout: float | None = None):
        del audio, encoder_timeout
        raise SharedEncoderBusy("shared DistilHuBERT encoder is busy")

    def can_train(self, audio: np.ndarray) -> bool:
        del audio
        return True


def _runtime(model: object) -> ParticipantRuntime:
    instance = ParticipantRuntime(
        participant_id="isolation-test",
        group_name="frozen",
        cohort="pilot",
        settings=Settings.from_env(),
        phase="learning",
        activation_threshold=0.99,
        voice_confirm_frames=1,
        voice_rms_ema_alpha=1.0,
    )
    instance.model = model  # type: ignore[assignment]
    return instance


def test_busy_shared_encoder_skips_prediction_without_activating() -> None:
    runtime = _runtime(BusyEncoderModel())
    result = runtime.process_pcm(np.full(3_200, 2_000, dtype="<i2").tobytes(), 2_000.0)
    assert result["skipped"] is True
    assert result["skip_reason"] == "shared_encoder_busy"
    assert result["activate"] is False


def test_tap_is_not_blocked_by_in_flight_shared_predict() -> None:
    model = SlowPredictModel()
    runtime = _runtime(model)
    worker = threading.Thread(
        target=runtime.process_pcm,
        args=(np.full(3_200, 2_000, dtype="<i2").tobytes(), 2_000.0),
    )
    worker.start()
    assert model.entered.wait(1.0)
    started = time.monotonic()
    tap = runtime.tap(2_100.0)
    elapsed = time.monotonic() - started
    model.release.set()
    worker.join(2.0)
    assert worker.is_alive() is False
    assert elapsed < 0.3
    assert tap["action"] in {
        "manual_activate",
        "conditional_tap_cancelled",
        "tap_burst_ignored",
    }


def test_gpu_encoder_waits_longer_than_cpu() -> None:
    from app.timing import (
        SHARED_ENCODER_GPU_WAIT_S,
        SHARED_ENCODER_WAIT_S,
        _shared_encoder_wait_s,
    )

    assert _shared_encoder_wait_s("cpu") == SHARED_ENCODER_WAIT_S
    assert _shared_encoder_wait_s("cuda:2") == SHARED_ENCODER_GPU_WAIT_S
    assert _shared_encoder_wait_s("cuda:3") == SHARED_ENCODER_GPU_WAIT_S
