"""Causal ≤1 min tap-timing history fusion for cold-start personalisation.

Production lock: A10_lr_high_0.05 from the cold-start feature/backbone sweep
(experiments/sweep_tap_history_coldstart_features.py, outputs/tap_history_feature_sweep_v1).

Fusion: threshold_and_early — asymmetric threshold state machine + bounded
early-fusion residual on the acoustic logit. Co-adapted with replay-v1 on the
same eligible acoustic taps.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

CorrectionKind = Literal["manual_activate", "interrupt"]
FeatureMode = Literal["full", "timing_only", "minimal", "rich_decay"]
FusionMode = Literal["threshold_and_early", "threshold_only", "early_only", "logit_clip"]

STATE_FILENAME = "tap-history-state.json"
SCHEMA = "tap-to-adapt-tap-history-runtime/v1"
LOCKED_VARIANT_ID = "A10_lr_high_0.05"

FEATURE_NAMES = (
    "log_time_since_activate",
    "missing_activate",
    "log_time_since_interrupt",
    "missing_interrupt",
    "decay_activate_horizon",
    "decay_interrupt_horizon",
    "decay_activate_half",
    "decay_interrupt_half",
    "decay_activate_quarter",
    "decay_interrupt_quarter",
    "last_correction_direction",
    "log_last_interrupt_delay",
    "missing_interrupt_delay",
    "has_recent_tap",
    "interrupt_pressure",
    "activate_pressure",
    "interrupt_acoustic_similarity",
    "missing_interrupt_acoustic",
    "interrupt_abruptness",
    "missing_interrupt_abruptness",
)


@dataclass(frozen=True)
class TapHistoryConfig:
    variant_id: str = LOCKED_VARIANT_ID
    enabled: bool = True
    horizon_s: float = 60.0
    feature_mode: FeatureMode = "full"
    fusion_mode: FusionMode = "threshold_and_early"
    state_lr: float = 0.05
    k_interrupt0: float = 0.08
    k_activate0: float = 0.04
    early_fusion_delta: float = 0.50
    logit_clip_delta: float = 0.75

    @classmethod
    def locked_a10(cls) -> "TapHistoryConfig":
        return cls()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "TapHistoryConfig":
        if not raw:
            return cls.locked_a10()
        base = cls.locked_a10()
        return cls(
            variant_id=str(raw.get("variant_id", base.variant_id)),
            enabled=bool(raw.get("enabled", base.enabled)),
            horizon_s=float(raw.get("horizon_s", base.horizon_s)),
            feature_mode=str(raw.get("feature_mode", base.feature_mode)),  # type: ignore[arg-type]
            fusion_mode=str(raw.get("fusion_mode", base.fusion_mode)),  # type: ignore[arg-type]
            state_lr=float(raw.get("state_lr", base.state_lr)),
            k_interrupt0=float(raw.get("k_interrupt0", base.k_interrupt0)),
            k_activate0=float(raw.get("k_activate0", base.k_activate0)),
            early_fusion_delta=float(
                raw.get("early_fusion_delta", base.early_fusion_delta)
            ),
            logit_clip_delta=float(raw.get("logit_clip_delta", base.logit_clip_delta)),
        )


@dataclass
class TapEvent:
    time_seconds: float
    kind: CorrectionKind
    interrupt_delay_ms: float | None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TapTimingMemory:
    def __init__(self, config: TapHistoryConfig) -> None:
        self.config = config
        self.use_acoustic = config.feature_mode not in {"timing_only", "minimal"}
        self._events: list[TapEvent] = []
        self._last_interrupt_feature: np.ndarray | None = None
        self._last_interrupt_abruptness: float | None = None
        self._last_interrupt_time: float | None = None

    def _prune(self, now_seconds: float) -> None:
        cutoff = now_seconds - self.config.horizon_s
        self._events = [e for e in self._events if e.time_seconds >= cutoff]
        if (
            self._last_interrupt_time is not None
            and self._last_interrupt_time < cutoff
        ):
            self._last_interrupt_feature = None
            self._last_interrupt_abruptness = None
            self._last_interrupt_time = None

    def _decay(self, kind: CorrectionKind, now: float, tau: float) -> float:
        total = 0.0
        for event in self._events:
            if event.kind != kind:
                continue
            age = max(0.0, now - event.time_seconds)
            if age > self.config.horizon_s:
                continue
            total += math.exp(-age / max(tau, 1e-6))
        return float(total)

    def features(
        self, *, now_seconds: float, current_feature: np.ndarray | None
    ) -> np.ndarray:
        self._prune(now_seconds)
        horizon = self.config.horizon_s
        activates = [e for e in self._events if e.kind == "manual_activate"]
        interrupts = [e for e in self._events if e.kind == "interrupt"]
        last_activate = activates[-1].time_seconds if activates else None
        last_interrupt = interrupts[-1].time_seconds if interrupts else None
        last_event = self._events[-1] if self._events else None
        last_delay = None
        for event in reversed(self._events):
            if event.kind == "interrupt" and event.interrupt_delay_ms is not None:
                last_delay = float(event.interrupt_delay_ms)
                break

        def log_time(ts: float | None) -> tuple[float, float, float]:
            if ts is None:
                return 0.0, 1.0, float("inf")
            age = max(0.0, now_seconds - ts)
            return float(math.log1p(age) / math.log1p(horizon)), 0.0, age

        act_log, act_miss, act_age = log_time(last_activate)
        int_log, int_miss, int_age = log_time(last_interrupt)
        if last_event is None:
            direction = 0.0
        elif last_event.kind == "manual_activate":
            direction = 1.0
        else:
            direction = -1.0
        if last_delay is None:
            delay_log, delay_miss = 0.0, 1.0
        else:
            delay_log = float(math.log1p(max(0.0, last_delay)) / math.log1p(3000.0))
            delay_miss = 0.0

        interrupt_pressure = (
            float(math.exp(-int_age / max(horizon / 2.4, 1.0)))
            if int_age < horizon
            else 0.0
        )
        activate_pressure = (
            float(math.exp(-act_age / max(horizon / 1.5, 1.0)))
            if act_age < horizon
            else 0.0
        )

        if (
            self.use_acoustic
            and current_feature is not None
            and self._last_interrupt_feature is not None
        ):
            sim = 0.5 * (
                _cosine(current_feature, self._last_interrupt_feature) + 1.0
            )
            miss_sim = 0.0
        else:
            sim, miss_sim = 0.0, 1.0
        if self.use_acoustic and self._last_interrupt_abruptness is not None:
            abrupt = float(np.clip(self._last_interrupt_abruptness, 0.0, 1.0))
            miss_abrupt = 0.0
        else:
            abrupt, miss_abrupt = 0.0, 1.0

        full = np.asarray(
            [
                act_log,
                act_miss,
                int_log,
                int_miss,
                self._decay("manual_activate", now_seconds, horizon),
                self._decay("interrupt", now_seconds, horizon),
                self._decay("manual_activate", now_seconds, horizon / 2),
                self._decay("interrupt", now_seconds, horizon / 2),
                self._decay("manual_activate", now_seconds, horizon / 4),
                self._decay("interrupt", now_seconds, horizon / 4),
                direction,
                delay_log,
                delay_miss,
                1.0 if self._events else 0.0,
                interrupt_pressure,
                activate_pressure,
                sim,
                miss_sim,
                abrupt,
                miss_abrupt,
            ],
            dtype=np.float64,
        )
        mode = self.config.feature_mode
        if mode == "full":
            values = full.copy()
            values[FEATURE_NAMES.index("decay_activate_quarter")] = 0.0
            values[FEATURE_NAMES.index("decay_interrupt_quarter")] = 0.0
            return values
        if mode == "rich_decay":
            return full
        if mode == "timing_only":
            values = full.copy()
            for name in (
                "interrupt_acoustic_similarity",
                "missing_interrupt_acoustic",
                "interrupt_abruptness",
                "missing_interrupt_abruptness",
                "decay_activate_quarter",
                "decay_interrupt_quarter",
            ):
                values[FEATURE_NAMES.index(name)] = (
                    1.0 if name.startswith("missing_") else 0.0
                )
            return values
        values = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        values[FEATURE_NAMES.index("interrupt_pressure")] = interrupt_pressure
        values[FEATURE_NAMES.index("activate_pressure")] = activate_pressure
        values[FEATURE_NAMES.index("last_correction_direction")] = direction
        values[FEATURE_NAMES.index("has_recent_tap")] = 1.0 if self._events else 0.0
        values[FEATURE_NAMES.index("missing_activate")] = act_miss
        values[FEATURE_NAMES.index("missing_interrupt")] = int_miss
        return values

    def explicit_threshold_shift(
        self,
        features: np.ndarray,
        *,
        k_interrupt: float,
        k_activate: float,
    ) -> float:
        idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        interrupt_pressure = float(features[idx["interrupt_pressure"]])
        activate_pressure = float(features[idx["activate_pressure"]])
        sim = float(features[idx["interrupt_acoustic_similarity"]])
        abrupt = float(features[idx["interrupt_abruptness"]])
        miss_sim = float(features[idx["missing_interrupt_acoustic"]])
        acoustic_scale = 0.55 + 0.45 * sim if miss_sim < 0.5 else 0.70
        abrupt_scale = 0.60 + 0.40 * abrupt if miss_sim < 0.5 else 0.75
        raw = (
            k_interrupt * interrupt_pressure * acoustic_scale * abrupt_scale
            - k_activate * activate_pressure
        )
        return float(np.clip(raw, -0.05, 0.10))

    def observe(
        self,
        *,
        time_seconds: float,
        kind: CorrectionKind,
        interrupt_delay_ms: float | None,
        interrupt_feature: np.ndarray | None = None,
        prior_feature: np.ndarray | None = None,
    ) -> None:
        self._events.append(TapEvent(time_seconds, kind, interrupt_delay_ms))
        if kind == "interrupt" and interrupt_feature is not None and self.use_acoustic:
            self._last_interrupt_feature = np.asarray(
                interrupt_feature, dtype=np.float64
            )
            self._last_interrupt_time = float(time_seconds)
            if prior_feature is not None:
                delta = np.asarray(interrupt_feature, dtype=np.float64) - np.asarray(
                    prior_feature, dtype=np.float64
                )
                self._last_interrupt_abruptness = float(
                    np.clip(np.linalg.norm(delta) / 8.0, 0.0, 1.0)
                )
            else:
                self._last_interrupt_abruptness = 0.35
        self._prune(time_seconds + 1e-6)


@dataclass
class TapTimingController:
    config: TapHistoryConfig
    k_interrupt: float = 0.0
    k_activate: float = 0.0
    threshold_bias: float = 0.0
    early_weight: np.ndarray = field(default_factory=lambda: np.zeros(0))
    update_count: int = 0

    def __post_init__(self) -> None:
        self.k_interrupt = self.config.k_interrupt0
        self.k_activate = self.config.k_activate0
        if self.early_weight.size != len(FEATURE_NAMES):
            self.early_weight = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    def _active_mask(self, features: np.ndarray) -> np.ndarray:
        mask = np.ones(len(FEATURE_NAMES), dtype=bool)
        if self.config.feature_mode == "minimal":
            keep = {
                "interrupt_pressure",
                "activate_pressure",
                "last_correction_direction",
                "has_recent_tap",
            }
            mask = np.asarray([name in keep for name in FEATURE_NAMES], dtype=bool)
        elif self.config.feature_mode == "timing_only":
            for name in (
                "interrupt_acoustic_similarity",
                "missing_interrupt_acoustic",
                "interrupt_abruptness",
                "missing_interrupt_abruptness",
            ):
                mask[FEATURE_NAMES.index(name)] = False
        return mask

    def apply(
        self, features: np.ndarray, memory: TapTimingMemory
    ) -> tuple[float, float]:
        """Return (early_or_logit_fusion, threshold_shift)."""

        explicit = memory.explicit_threshold_shift(
            features,
            k_interrupt=self.k_interrupt,
            k_activate=self.k_activate,
        )
        fused = float(features @ self.early_weight)
        mode = self.config.fusion_mode
        if mode == "threshold_and_early":
            shift = float(np.clip(explicit + self.threshold_bias, -0.06, 0.12))
            early = float(
                np.clip(
                    fused,
                    -self.config.early_fusion_delta,
                    self.config.early_fusion_delta,
                )
            )
            return early, shift
        if mode == "threshold_only":
            shift = float(np.clip(explicit + self.threshold_bias, -0.06, 0.12))
            return 0.0, shift
        if mode == "early_only":
            early = float(
                np.clip(
                    fused,
                    -self.config.early_fusion_delta,
                    self.config.early_fusion_delta,
                )
            )
            return early, 0.0
        logit = float(
            np.clip(
                -explicit + fused + self.threshold_bias,
                -self.config.logit_clip_delta,
                self.config.logit_clip_delta,
            )
        )
        return logit, 0.0

    def update(self, kind: CorrectionKind, features: np.ndarray) -> None:
        lr = self.config.state_lr
        mask = self._active_mask(features)
        selected = features * mask
        if kind == "interrupt":
            self.k_interrupt = float(np.clip(self.k_interrupt + lr, 0.02, 0.18))
            self.threshold_bias = float(
                np.clip(self.threshold_bias + 0.5 * lr, -0.04, 0.10)
            )
            self.early_weight = np.clip(
                self.early_weight - lr * selected, -0.20, 0.20
            )
        else:
            self.k_activate = float(np.clip(self.k_activate + lr, 0.01, 0.12))
            self.threshold_bias = float(
                np.clip(self.threshold_bias - 0.5 * lr, -0.04, 0.10)
            )
            self.early_weight = np.clip(
                self.early_weight + lr * selected, -0.20, 0.20
            )
        self.update_count += 1


class TapHistoryRuntime:
    """Per-participant tap-timing memory + online controller."""

    def __init__(self, config: TapHistoryConfig) -> None:
        self.config = config
        self.memory = TapTimingMemory(config)
        self.controller = TapTimingController(config)
        self.last_early_fusion = 0.0
        self.last_threshold_shift = 0.0
        self.last_features: np.ndarray | None = None

    def decide(
        self,
        *,
        now_seconds: float,
        current_feature: np.ndarray | None,
        base_threshold: float,
    ) -> dict[str, float]:
        features = self.memory.features(
            now_seconds=now_seconds, current_feature=current_feature
        )
        early, shift = self.controller.apply(features, self.memory)
        self.last_early_fusion = early
        self.last_threshold_shift = shift
        self.last_features = features
        effective = float(np.clip(base_threshold + shift, 0.05, 0.99))
        return {
            "early_fusion_logit": early,
            "threshold_shift": shift,
            "effective_activation_threshold": effective,
        }

    def observe_and_update(
        self,
        *,
        now_seconds: float,
        kind: CorrectionKind,
        interrupt_delay_ms: float | None,
        interrupt_feature: np.ndarray | None = None,
        prior_feature: np.ndarray | None = None,
        current_feature: np.ndarray | None = None,
    ) -> None:
        features = self.memory.features(
            now_seconds=now_seconds,
            current_feature=current_feature
            if current_feature is not None
            else interrupt_feature,
        )
        self.controller.update(kind, features)
        self.memory.observe(
            time_seconds=now_seconds,
            kind=kind,
            interrupt_delay_ms=interrupt_delay_ms,
            interrupt_feature=interrupt_feature,
            prior_feature=prior_feature,
        )

    def audit_state(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "variant_id": self.config.variant_id,
            "enabled": self.config.enabled,
            "horizon_s": self.config.horizon_s,
            "feature_mode": self.config.feature_mode,
            "fusion_mode": self.config.fusion_mode,
            "state_lr": self.config.state_lr,
            "k_interrupt": self.controller.k_interrupt,
            "k_activate": self.controller.k_activate,
            "threshold_bias": self.controller.threshold_bias,
            "update_count": self.controller.update_count,
            "event_count": len(self.memory._events),
            "last_early_fusion_logit": self.last_early_fusion,
            "last_threshold_shift": self.last_threshold_shift,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "controller": {
                "k_interrupt": self.controller.k_interrupt,
                "k_activate": self.controller.k_activate,
                "threshold_bias": self.controller.threshold_bias,
                "early_weight": self.controller.early_weight.tolist(),
                "update_count": self.controller.update_count,
            },
            "memory": {
                "events": [
                    {
                        "time_seconds": e.time_seconds,
                        "kind": e.kind,
                        "interrupt_delay_ms": e.interrupt_delay_ms,
                    }
                    for e in self.memory._events
                ],
                "last_interrupt_feature": (
                    None
                    if self.memory._last_interrupt_feature is None
                    else self.memory._last_interrupt_feature.tolist()
                ),
                "last_interrupt_abruptness": self.memory._last_interrupt_abruptness,
                "last_interrupt_time": self.memory._last_interrupt_time,
            },
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / STATE_FILENAME
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
        return path

    @classmethod
    def load_or_create(
        cls, directory: Path | None, config: TapHistoryConfig
    ) -> "TapHistoryRuntime":
        runtime = cls(config)
        if directory is None:
            return runtime
        path = directory / STATE_FILENAME
        if not path.is_file():
            return runtime
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return runtime
        if payload.get("schema") != SCHEMA:
            return runtime
        controller = payload.get("controller") or {}
        runtime.controller.k_interrupt = float(
            controller.get("k_interrupt", config.k_interrupt0)
        )
        runtime.controller.k_activate = float(
            controller.get("k_activate", config.k_activate0)
        )
        runtime.controller.threshold_bias = float(
            controller.get("threshold_bias", 0.0)
        )
        weight = np.asarray(
            controller.get("early_weight") or [], dtype=np.float64
        )
        if weight.shape == (len(FEATURE_NAMES),):
            runtime.controller.early_weight = weight
        runtime.controller.update_count = int(controller.get("update_count", 0))
        memory = payload.get("memory") or {}
        for raw in memory.get("events") or []:
            kind = str(raw.get("kind") or "")
            if kind not in {"manual_activate", "interrupt"}:
                continue
            runtime.memory._events.append(
                TapEvent(
                    time_seconds=float(raw["time_seconds"]),
                    kind=kind,  # type: ignore[arg-type]
                    interrupt_delay_ms=(
                        None
                        if raw.get("interrupt_delay_ms") is None
                        else float(raw["interrupt_delay_ms"])
                    ),
                )
            )
        feat = memory.get("last_interrupt_feature")
        if feat is not None:
            runtime.memory._last_interrupt_feature = np.asarray(feat, dtype=np.float64)
        abrupt = memory.get("last_interrupt_abruptness")
        runtime.memory._last_interrupt_abruptness = (
            None if abrupt is None else float(abrupt)
        )
        t = memory.get("last_interrupt_time")
        runtime.memory._last_interrupt_time = None if t is None else float(t)
        return runtime
