"""Streaming response-initiation inference and state-bound correction learning.

Each ParticipantRuntime manages the audio buffer, immutable shared prediction,
personal prediction, causal tap history, and correction eligibility.
TimingService shares frozen feature extraction while isolating participant
heads and checkpoints. Prediction precedes feedback; the server state and
activation identifier bind each eligible tap to the decision it corrects.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np

from .config import Settings
from .integrity import file_sha256
from .matched import matched_assignment
from .pipeline_guard import requires_locked_pipeline
from .priors import PriorError, prior_paths
from .tap_history import TapHistoryConfig, TapHistoryRuntime

if TYPE_CHECKING:
    from .db import StudyDB


SAMPLE_RATE = 16_000
HISTORY_SECONDS = 40


class ModelUnavailable(RuntimeError):
    pass


SHARED_ENCODER_WAIT_S = 0.08
SHARED_ENCODER_GPU_WAIT_S = 0.25


def _shared_encoder_wait_s(device: str | None) -> float:
    if str(device or "").startswith("cuda"):
        return SHARED_ENCODER_GPU_WAIT_S
    return SHARED_ENCODER_WAIT_S


def _is_shared_encoder_busy(exc: BaseException) -> bool:
    return type(exc).__name__ == "SharedEncoderBusy"


@dataclass(frozen=True)
class CalibratedPrediction:
    probability: float
    logit: float
    raw_probability: float
    raw_logit: float
    policy_effective: str
    personal_logit_bias: float


class AudioRing:
    def __init__(self, max_samples: int = SAMPLE_RATE * HISTORY_SECONDS):
        self.max_samples = max_samples
        self.parts: deque[np.ndarray] = deque()
        self.samples = 0

    def append(self, values: np.ndarray) -> None:
        self.parts.append(values.copy())
        self.samples += len(values)
        while self.parts and self.samples - len(self.parts[0]) >= self.max_samples:
            self.samples -= len(self.parts.popleft())

    def recent(self, seconds: float) -> np.ndarray:
        needed = min(int(seconds * SAMPLE_RATE), self.samples)
        if needed <= 0:
            return np.zeros(0, dtype=np.float32)
        result: list[np.ndarray] = []
        remaining = needed
        for part in reversed(self.parts):
            if remaining <= 0:
                break
            result.append(part[-remaining:])
            remaining -= len(result[-1])
        return np.concatenate(list(reversed(result))) if result else np.zeros(0, dtype=np.float32)


class DTCNModel:
    def __init__(self, settings: Settings, encoder_path: str, main_path: str):
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable("PyTorch is not installed") from exc

        code_dir = Path(settings.model_code_dir)
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        try:
            from learner import OnlineTrainer
            from model import initialize_models
            from utils import AdaptiveAudioProcessor
        except ImportError as exc:
            raise ModelUnavailable(f"Cannot import model code from {code_dir}") from exc

        self.torch = torch
        self.encoder_path = Path(encoder_path)
        self.main_path = Path(main_path)
        if not self.encoder_path.is_file() or not self.main_path.is_file():
            raise ModelUnavailable("Configured DTCN checkpoint is missing")

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        encoder_checkpoint = torch.load(self.encoder_path, map_location=device_name, weights_only=False)
        main_checkpoint = torch.load(self.main_path, map_location=device_name, weights_only=False)
        encoder_pair_id = encoder_checkpoint.get("checkpoint_pair_id")
        main_pair_id = main_checkpoint.get("checkpoint_pair_id")
        if (encoder_pair_id is None) != (main_pair_id is None) or (
            encoder_pair_id is not None and encoder_pair_id != main_pair_id
        ):
            raise ModelUnavailable("Encoder/main checkpoint pair is incomplete or mismatched")
        self.config = copy.deepcopy(encoder_checkpoint["config"])
        self.config.update(
            {
                "device": device_name,
                "sim_mode": False,
                "data_mode": False,
                "audio_saving": False,
                "save_model": False,
                "LEARNING_RATE": self.config.get(
                    "LEARNING_RATE", self.config.get("RLHF_LEARNING_RATE", 5e-5)
                ),
            }
        )
        self.encoder, self.main_model = initialize_models(self.config)
        state = main_checkpoint["model_state_dict"]
        if "input_proj.weight" in state and not hasattr(self.main_model, "input_proj"):
            self.main_model._init_proj(int(state["input_proj.weight"].shape[1]))
        self.encoder.load_state_dict(encoder_checkpoint["model_state_dict"], strict=True)
        self.main_model.load_state_dict(state, strict=True)
        self.encoder.to(device_name).eval()
        self.main_model.to(device_name).eval()
        self.processor = AdaptiveAudioProcessor(self.config, self.encoder)
        self.trainer = OnlineTrainer(self.encoder, self.main_model, self.config)
        self.last_update_applied = False
        self.last_update_reason: str | None = None
        self.last_optimizer_step_before = 0
        self.last_optimizer_step_after = 0
        self.last_requested_training_steps = 0
        self.last_training_steps = 0
        self.last_optimizer_learning_rates_before: list[float] = []
        self.last_optimizer_learning_rates_during: list[float] = []
        self.last_optimizer_learning_rates_after: list[float] = []
        self.last_parameter_state_interpolated = False
        self.last_parameter_interpolation_scale = 1.0
        self.lock = threading.Lock()

    def predict(self, audio: np.ndarray) -> tuple[float, float]:
        with self.lock, self.torch.no_grad():
            tensor = self.processor.preprocess_audio(audio)
            z, _, frame_feats = self.encoder(tensor)
            probability, logit = self.main_model(z, frame_feats)
            return (
                float(np.clip(probability.squeeze().detach().cpu().item(), 0.0, 1.0)),
                float(logit.squeeze().detach().cpu().item()),
            )

    def can_train(self, audio: np.ndarray) -> bool:
        return bool(self.processor.is_audio_compatible(audio))

    def _optimizer_step_count(self) -> int:
        optimizer = self.trainer.optimizer
        if optimizer is None:
            return 0
        maximum = 0
        for state in optimizer.state.values():
            raw_step = state.get("step", 0)
            try:
                step = int(raw_step.item())
            except AttributeError:
                step = int(raw_step)
            maximum = max(maximum, step)
        return maximum

    def _snapshot_model_state(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                name: value.detach().clone()
                for name, value in self.encoder.state_dict().items()
            },
            {
                name: value.detach().clone()
                for name, value in self.main_model.state_dict().items()
            },
        )

    def _interpolate_model_state(
        self,
        snapshot: tuple[dict[str, Any], dict[str, Any]],
        scale: float,
    ) -> None:
        with self.torch.no_grad():
            for module, before_state in zip(
                (self.encoder, self.main_model), snapshot, strict=True
            ):
                after_state = module.state_dict()
                blended = {}
                for name, before in before_state.items():
                    after = after_state[name]
                    if after.is_floating_point() or after.is_complex():
                        blended[name] = before + (after - before) * float(scale)
                    else:
                        blended[name] = before
                module.load_state_dict(blended, strict=True)

    def train(
        self,
        audio: np.ndarray,
        label: int,
        weight: float = 1.0,
        update_scale: float = 1.0,
    ) -> float:
        with self.lock:
            self.last_update_applied = False
            if not 0 < float(update_scale) <= 1:
                self.last_update_reason = "invalid_network_update_scale"
                return 0.0
            if not self.can_train(audio):
                self.last_update_reason = "training_audio_too_short"
                return 0.0
            self.last_optimizer_step_before = self._optimizer_step_count()
            self.trainer.add_training_sample(audio, label, weight=float(weight), step=0)
            state_before = (
                self._snapshot_model_state() if float(update_scale) < 1.0 else None
            )
            self.last_parameter_state_interpolated = state_before is not None
            self.last_parameter_interpolation_scale = float(update_scale)
            configured_steps = int(self.config.get("num_of_step", 1))
            self.last_requested_training_steps = max(
                1, int(round(configured_steps * float(update_scale)))
            )
            optimizer = self.trainer.optimizer
            if optimizer is None:
                self.last_update_reason = "optimizer_unavailable"
                return 0.0
            self.last_optimizer_learning_rates_before = [
                float(group["lr"]) for group in optimizer.param_groups
            ]
            self.last_optimizer_learning_rates_during = [
                value * float(update_scale)
                for value in self.last_optimizer_learning_rates_before
            ]
            for group, scaled_lr in zip(
                optimizer.param_groups,
                self.last_optimizer_learning_rates_during,
                strict=True,
            ):
                group["lr"] = scaled_lr
            try:
                loss = float(
                    self.trainer.train_step(
                        num_steps=self.last_requested_training_steps,
                        current_positive_audio=audio if label == 1 else None,
                    )
                )
            finally:
                for group, original_lr in zip(
                    optimizer.param_groups,
                    self.last_optimizer_learning_rates_before,
                    strict=True,
                ):
                    group["lr"] = original_lr
                self.last_optimizer_learning_rates_after = [
                    float(group["lr"]) for group in optimizer.param_groups
                ]
            if state_before is not None:
                self._interpolate_model_state(state_before, float(update_scale))
            self.last_optimizer_step_after = self._optimizer_step_count()
            self.last_training_steps = max(
                0,
                self.last_optimizer_step_after - self.last_optimizer_step_before,
            )
            self.last_update_applied = (
                self.last_optimizer_step_after > self.last_optimizer_step_before
            )
            self.last_update_reason = (
                None if self.last_update_applied else "optimizer_update_not_applied"
            )
            return loss

    def save(self, target_dir: Path) -> tuple[Path, Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = target_dir / "latest_encoder.pth"
        main_path = target_dir / "latest_main_model.pth"
        pair_id = uuid.uuid4().hex
        encoder_temporary = target_dir / f".latest_encoder.{pair_id}.tmp"
        main_temporary = target_dir / f".latest_main_model.{pair_id}.tmp"
        with self.lock:
            try:
                self.torch.save(
                    {
                        "model_state_dict": self.encoder.state_dict(),
                        "config": self.config,
                        "checkpoint_pair_id": pair_id,
                    },
                    encoder_temporary,
                )
                self.torch.save(
                    {
                        "model_state_dict": self.main_model.state_dict(),
                        "config": self.config,
                        "checkpoint_pair_id": pair_id,
                    },
                    main_temporary,
                )
                encoder_check = self.torch.load(
                    encoder_temporary, map_location="cpu", weights_only=False
                )
                main_check = self.torch.load(
                    main_temporary, map_location="cpu", weights_only=False
                )
                if (
                    encoder_check.get("checkpoint_pair_id") != pair_id
                    or main_check.get("checkpoint_pair_id") != pair_id
                    or not isinstance(encoder_check.get("model_state_dict"), dict)
                    or not isinstance(main_check.get("model_state_dict"), dict)
                ):
                    raise ModelUnavailable("Checkpoint verification failed before publication")
                os.replace(encoder_temporary, encoder_path)
                os.replace(main_temporary, main_path)
            finally:
                encoder_temporary.unlink(missing_ok=True)
                main_temporary.unlink(missing_ok=True)
        return encoder_path, main_path


@dataclass
class ParticipantRuntime:
    participant_id: str
    group_name: str
    cohort: str
    settings: Settings
    model_storage_dir: Path | None = None
    runtime_instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    base_prior_id: str | None = None
    session_id: str | None = None
    day: int = 1
    phase: str = "frozen"
    policy: str = "current"
    active_block_index: int | None = None
    analysis_eligible: bool = True
    agent_state: str = "waiting"
    state_version: int = 0
    last_state_change_ms: float = 0.0
    state_transition_tap_guard_ms: float = 500.0
    ring: AudioRing = field(default_factory=AudioRing)
    model: DTCNModel | None = None
    donor_shadow_model: DTCNModel | None = None
    matched_model: DTCNModel | None = None
    candidate_model: DTCNModel | None = None
    candidate_probe_token: str | None = None
    model_identity: dict[str, Any] = field(default_factory=dict)
    budget_kind: str = "activation_opportunity"
    budget_semantics: str = "exact_count"
    q_levels: tuple[int, ...] = ()
    snapshot_levels: tuple[int, ...] = ()
    calibration_version: str = "uncalibrated"
    protocol_sha256: str = ""
    temperature: float = 1.0
    logit_bias: float = 0.0
    activation_threshold: float = 0.5
    minimum_end_of_turn_silence_ms: float = 0.0
    explicit_turn_end_required: bool = False
    adaptive_voice_threshold_enabled: bool = False
    voice_rms_floor: float = 0.020
    voice_noise_multiplier: float = 1.5
    voice_noise_percentile: float = 20.0
    voice_noise_window_frames: int = 50
    voice_calibration_frames: int = 5
    voice_rms_ema_alpha: float = 0.35
    voice_confirm_frames: int = 2
    decision_context_required: bool = False
    reply_needed_threshold: float = 0.5
    decision_context_ready: bool = True
    conversation_target: str = "experimental_agent"
    reply_needed_probability: float = 1.0
    decision_context_source: str = "legacy_unfactorized"
    decision_context_text_sha256: str | None = None
    decision_context_version: int = 0
    warmup_feedbacks: int = 0
    warmup_update_scale: float = 1.0
    network_update_schedule: tuple[tuple[int, float], ...] = ()
    personal_calibration_enabled: bool = False
    personal_logit_bias: float = 0.0
    personal_calibration_learning_rate: float = 0.0
    personal_calibration_l2: float = 0.0
    personal_logit_bias_min: float = -1.0
    personal_logit_bias_max: float = 1.0
    personal_calibration_update_count: int = 0
    personal_positive_updates: int = 0
    personal_negative_updates: int = 0
    tap_history: TapHistoryRuntime | None = None
    tap_history_enabled: bool = False
    last_backbone_feature: np.ndarray | None = None
    activation_backbone_feature: np.ndarray | None = None
    opportunity_count: int = 0
    activation_count: int = 0
    feedback_count: int = 0
    positive_feedbacks: int = 0
    negative_feedbacks: int = 0
    opportunity_counter: Callable[[], int] | None = None
    activation_counter: Callable[[], int] | None = None
    feedback_counter: Callable[[int], dict[str, int]] | None = None
    calibration_recorder: Callable[[int, int, float], dict[str, Any]] | None = None
    snapshot_recorder: Callable[[str, int, DTCNModel, dict[str, Any]], None] | None = None
    pending_snapshot_budgets: set[int] = field(default_factory=set)
    voice_rms_history: deque[float] = field(default_factory=deque)
    voice_rms_ema: float | None = None
    voice_confirm_count: int = 0
    last_voice_ms: float | None = None
    last_asr_final_ms: float | None = None
    last_prediction_ms: float = 0.0
    last_activation_ms: float = 0.0
    last_tap_ms: float = 0.0
    activation_audio: np.ndarray | None = None
    activation_id: str | None = None
    activation_source: str | None = None
    activation_phase: str | None = None
    activation_policy: str | None = None
    activation_policy_effective: str | None = None
    activation_probability: float | None = None
    activation_logit: float | None = None
    activation_raw_probability: float | None = None
    activation_raw_logit: float | None = None
    has_voice_since_activation: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def update_enabled(self) -> bool:
        return (
            self.group_name.lower()
            not in {"frozen", "f", "holdout", "holdout_evaluator"}
            and self.phase == "learning"
        )

    @property
    def prediction_interval_ms(self) -> float:
        # The production CPU path is intentionally slower than the GPU path.
        # Keeping the 180 ms cadence with a ~235 ms backbone would accumulate
        # queued PCM indefinitely. 400 ms remains below the end-of-turn decision
        # window while providing measured headroom on the shared server.
        if (
            self.settings.agent_mode == "distilhubert"
            and self.settings.distilhubert_device == "cpu"
        ):
            return 400.0
        return 180.0

    @property
    def activation_update_enabled(self) -> bool:
        return (
            self.group_name.lower()
            not in {"frozen", "f", "holdout", "holdout_evaluator"}
            and self.activation_phase == "learning"
            and self.activation_policy == "current"
        )

    def set_phase(
        self,
        phase: str,
        policy: str,
        *,
        drain_active: bool = False,
        active_block_index: int | None = None,
        analysis_eligible: bool = True,
    ) -> dict[str, Any]:
        allowed_phases = {
            "frozen",
            "learning",
            "micro_probe",
            "candidate_probe",
            "rest",
            "complete",
        }
        allowed_policies = {
            "current",
            "base",
            "matched_other",
            "vad",
            "candidate_prior",
        }
        if phase not in allowed_phases or policy not in allowed_policies:
            raise ValueError("invalid phase or policy")
        with self.lock:
            previous_state = self.agent_state
            if self.agent_state != "waiting":
                if not drain_active:
                    return {"committed": False, "drained_state": None}
                self._set_agent_state_locked("waiting")
                self._finalize_learning_opportunity()
                self._clear_activation()
                self._invalidate_decision_context()
            self.phase = phase
            self.policy = policy
            self.active_block_index = active_block_index
            self.analysis_eligible = bool(analysis_eligible and phase != "rest")
            if phase == "rest":
                self.ring = AudioRing()
                self.last_voice_ms = None
                self.last_prediction_ms = 0.0
                self.has_voice_since_activation = False
                if self.decision_context_required:
                    self.decision_context_ready = False
                    self.decision_context_text_sha256 = None
        return {
            "committed": True,
            "drained_state": previous_state if previous_state != "waiting" else None,
        }

    def set_state(self, state: str) -> None:
        if state not in {"waiting", "thinking", "speaking"}:
            raise ValueError("invalid agent state")
        with self.lock:
            previous = self.agent_state
            self._set_agent_state_locked(state)
            if state == "waiting" and previous != "waiting":
                self._finalize_learning_opportunity()
                self._clear_activation()
                self._invalidate_decision_context()

    def set_decision_context(
        self,
        *,
        conversation_target: str,
        reply_needed_probability: float,
        source: str,
        text_sha256: str | None,
    ) -> dict[str, Any]:
        if conversation_target not in {"experimental_agent", "companion_agent"}:
            raise ValueError("invalid conversation target")
        if not 0.0 <= reply_needed_probability <= 1.0:
            raise ValueError("reply-needed probability must be in [0, 1]")
        if not source or len(source) > 100:
            raise ValueError("invalid decision context source")
        if text_sha256 is not None and (
            len(text_sha256) != 64
            or any(character not in "0123456789abcdef" for character in text_sha256)
        ):
            raise ValueError("invalid decision context text digest")
        with self.lock:
            if self.agent_state != "waiting":
                raise ValueError("decision context requires the waiting state")
            self.conversation_target = conversation_target
            self.reply_needed_probability = float(reply_needed_probability)
            self.decision_context_source = source
            self.decision_context_text_sha256 = text_sha256
            self.decision_context_ready = True
            self.decision_context_version += 1
            return self._decision_context_snapshot()

    def _decision_context_snapshot(self) -> dict[str, Any]:
        # v27: addressee + reply-needed no longer gate activation. The streaming
        # acoustic timing model holds activation authority; ASR text is used
        # after activation for chat content only.
        target_allowed = True
        reply_allowed = True
        return {
            "decision_context_required": self.decision_context_required,
            "decision_context_ready": self.decision_context_ready,
            "decision_context_version": self.decision_context_version,
            "conversation_target": self.conversation_target,
            "reply_needed_probability": self.reply_needed_probability,
            "reply_needed_threshold": self.reply_needed_threshold,
            "decision_context_source": self.decision_context_source,
            "decision_context_text_sha256": self.decision_context_text_sha256,
            "context_target_allowed": target_allowed,
            "context_reply_allowed": reply_allowed,
            "context_allows_activation": True,
        }

    def reset_audio_after_simulated_gap(self) -> dict[str, Any]:
        """Fast-forward a simulation-only silent gap without model inference.

        A browser continuously sends silence between turns.  Replaying thirty
        seconds of silent 200 ms frames would waste accelerator time while
        producing the same DistilHuBERT input as an empty ring followed by
        left-zero-padding.  The WebSocket layer exposes this only to isolated
        preview/simulation accounts.
        """

        with self.lock:
            if self.agent_state != "waiting":
                raise ValueError("simulated gap requires the waiting state")
            discarded_samples = int(self.ring.samples)
            self.ring = AudioRing()
            self.last_voice_ms = None
            self.last_prediction_ms = 0.0
            self.has_voice_since_activation = False
            if self.decision_context_required:
                self._invalidate_decision_context()
            return {
                "discarded_samples": discarded_samples,
                **self._decision_context_snapshot(),
                "server_agent_state": self.agent_state,
                "server_state_version": self.state_version,
                "server_activation_id": self.activation_id,
            }

    def _set_agent_state_locked(
        self, state: str, server_monotonic_ms: float | None = None
    ) -> None:
        if state != self.agent_state:
            self.agent_state = state
            self.state_version += 1
            self.last_state_change_ms = (
                float(server_monotonic_ms)
                if server_monotonic_ms is not None
                else time.monotonic() * 1000
            )

    def _server_state_snapshot(self) -> dict[str, Any]:
        return {
            "server_agent_state": self.agent_state,
            "server_state_version": self.state_version,
            "server_activation_id": self.activation_id,
            "active_block_index": self.active_block_index,
            "analysis_eligible": self.analysis_eligible,
        }

    def _activation_provenance(self) -> dict[str, Any]:
        return {
            "activation_id": self.activation_id,
            "activation_source": self.activation_source,
            "activation_phase": self.activation_phase,
            "activation_policy": self.activation_policy,
            "activation_policy_effective": self.activation_policy_effective,
            "activation_probability": self.activation_probability,
            "activation_logit": self.activation_logit,
            "activation_raw_probability": self.activation_raw_probability,
            "activation_raw_logit": self.activation_raw_logit,
            "activation_state_version": self.state_version,
            "candidate_probe_token": self.candidate_probe_token
            if self.activation_policy == "candidate_prior"
            else None,
        }

    def ready_payload(self) -> dict[str, Any]:
        audit_reader = getattr(self.model, "audit_state", None)
        model_head_state = audit_reader() if callable(audit_reader) else None
        return {
            "latency_clock": "server_monotonic_activation_commit_v2",
            "client_timestamps_for_audit_only": True,
            "runtime_instance_id": self.runtime_instance_id,
            "prediction_interval_ms": self.prediction_interval_ms,
            "model_head_state": model_head_state,
            "negative_label_semantics": "observable_activation_interrupt_v2",
            "negative_label_latency_cutoff_ms": None,
            "negative_label_requires_pre_tap_speech": False,
            "state_transition_tap_guard_ms": self.state_transition_tap_guard_ms,
            "model_fingerprint": self.model_identity,
            "decision_factorization": {
                "minimum_end_of_turn_silence_ms": self.minimum_end_of_turn_silence_ms,
                "explicit_turn_end_required": self.explicit_turn_end_required,
                "acoustic_vad": {
                    "adaptive_threshold_enabled": self.adaptive_voice_threshold_enabled,
                    "minimum_rms_threshold": self.voice_rms_floor,
                    "noise_multiplier": self.voice_noise_multiplier,
                    "noise_percentile": self.voice_noise_percentile,
                    "window_frames": self.voice_noise_window_frames,
                    "calibration_frames": self.voice_calibration_frames,
                    "rms_ema_alpha": self.voice_rms_ema_alpha,
                    "voice_confirm_frames": self.voice_confirm_frames,
                },
                **self._decision_context_snapshot(),
            },
            **self._server_state_snapshot(),
            "calibration": {
                "version": self.calibration_version,
                "protocol_sha256": self.protocol_sha256,
                "temperature": self.temperature,
                "logit_bias": self.logit_bias,
                "activation_threshold": self.activation_threshold,
                "warmup_feedbacks": self.warmup_feedbacks,
                "warmup_update_scale": self.warmup_update_scale,
                "network_update_schedule": [
                    {"eligible_feedbacks_lt": limit, "scale": scale}
                    for limit, scale in self.network_update_schedule
                ],
                "personal": {
                    "enabled": self.personal_calibration_enabled,
                    "logit_bias": self.personal_logit_bias,
                    "update_count": self.personal_calibration_update_count,
                    "positive_updates": self.personal_positive_updates,
                    "negative_updates": self.personal_negative_updates,
                    "minimum": self.personal_logit_bias_min,
                    "maximum": self.personal_logit_bias_max,
                },
                "tap_timing_history": (
                    self.tap_history.audit_state()
                    if self.tap_history is not None
                    else {"enabled": False}
                ),
            },
        }

    def _extract_backbone_feature(self, audio: np.ndarray | None) -> np.ndarray | None:
        if audio is None or self.model is None or not len(audio):
            return None
        extractor = getattr(self.model, "extract_features", None)
        if not callable(extractor):
            return None
        try:
            return np.asarray(extractor(audio), dtype=np.float64)
        except Exception:
            return None

    def _apply_tap_history_fusion(
        self,
        prediction: CalibratedPrediction,
        *,
        now_seconds: float,
        audio: np.ndarray,
    ) -> tuple[CalibratedPrediction, dict[str, float]]:
        if self.tap_history is None or not self.tap_history_enabled:
            return prediction, {
                "early_fusion_logit": 0.0,
                "threshold_shift": 0.0,
                "effective_activation_threshold": self.activation_threshold,
            }
        feature = self._extract_backbone_feature(audio)
        if feature is not None:
            self.last_backbone_feature = feature
        decision = self.tap_history.decide(
            now_seconds=now_seconds,
            current_feature=feature,
            base_threshold=self.activation_threshold,
        )
        fused_logit = float(prediction.logit) + float(decision["early_fusion_logit"])
        fused = CalibratedPrediction(
            probability=self._sigmoid(fused_logit),
            logit=fused_logit,
            raw_probability=prediction.raw_probability,
            raw_logit=prediction.raw_logit,
            policy_effective=prediction.policy_effective,
            personal_logit_bias=prediction.personal_logit_bias,
        )
        return fused, decision

    def _coadapt_tap_history(
        self,
        *,
        client_ms: float,
        kind: str,
        interrupt_delay_ms: float | None,
        result: dict[str, Any],
    ) -> None:
        if (
            self.tap_history is None
            or not self.tap_history_enabled
            or not result.get("training_applied")
        ):
            return
        if kind not in {"manual_activate", "interrupt"}:
            return
        audio = self.activation_audio
        interrupt_feature = (
            self.activation_backbone_feature
            if self.activation_backbone_feature is not None
            else self._extract_backbone_feature(audio)
        )
        prior_feature = self.last_backbone_feature
        self.tap_history.observe_and_update(
            now_seconds=float(client_ms) / 1000.0,
            kind=kind,  # type: ignore[arg-type]
            interrupt_delay_ms=interrupt_delay_ms,
            interrupt_feature=interrupt_feature if kind == "interrupt" else None,
            prior_feature=prior_feature if kind == "interrupt" else None,
            current_feature=interrupt_feature,
        )
        storage = self.model_storage_dir or (
            self.settings.data_dir / "models" / self.participant_id
        )
        self.tap_history.save(storage)
        result["tap_history_update_applied"] = True
        result["tap_history"] = self.tap_history.audit_state()

    def _begin_activation(
        self,
        client_ms: float,
        source: str,
        policy_effective: str,
        prediction: CalibratedPrediction | None = None,
    ) -> None:
        self._set_agent_state_locked("thinking", client_ms)
        self.last_activation_ms = client_ms
        self.activation_audio = self.ring.recent(30.0)
        self.activation_backbone_feature = self._extract_backbone_feature(
            self.activation_audio
        )
        self.activation_id = str(uuid.uuid4())
        self.activation_source = source
        self.activation_phase = self.phase
        self.activation_policy = self.policy
        self.activation_policy_effective = policy_effective
        self.activation_probability = (
            prediction.probability if prediction is not None else None
        )
        self.activation_logit = prediction.logit if prediction is not None else None
        self.activation_raw_probability = (
            prediction.raw_probability if prediction is not None else None
        )
        self.activation_raw_logit = (
            prediction.raw_logit if prediction is not None else None
        )
        self.has_voice_since_activation = False
        self._start_learning_opportunity()

    def mark_activation_observable(
        self, activation_id: str, server_monotonic_ms: float
    ) -> bool:
        """Move the latency origin to the instant an activation is committed to the client."""
        with self.lock:
            if (
                not activation_id
                or self.activation_id != activation_id
                or self.agent_state == "waiting"
            ):
                return False
            self.last_activation_ms = server_monotonic_ms
            return True

    def _clear_activation(self) -> None:
        self.activation_audio = None
        self.activation_backbone_feature = None
        self.activation_id = None
        self.activation_source = None
        self.activation_phase = None
        self.activation_policy = None
        self.activation_policy_effective = None
        self.activation_probability = None
        self.activation_logit = None
        self.activation_raw_probability = None
        self.activation_raw_logit = None
        self.has_voice_since_activation = False

    def _invalidate_decision_context(self) -> None:
        """Require a fresh completed user turn before another automatic reply.

        A reply-needed decision belongs to exactly one user turn. Reusing it
        after the agent finishes or is interrupted can otherwise create a loop
        of automatic activations from the same stale transcript.
        """
        if not self.decision_context_required:
            return
        self.decision_context_ready = False
        self.decision_context_text_sha256 = None
        self.reply_needed_probability = 0.0
        self.decision_context_source = "awaiting_fresh_user_turn"

    def _start_learning_opportunity(self) -> None:
        if self.phase != "learning":
            return
        if self.budget_kind == "eligible_feedback":
            self.activation_count = (
                self.activation_counter()
                if self.activation_counter
                else self.activation_count + 1
            )
            return
        self.opportunity_count = (
            self.opportunity_counter() if self.opportunity_counter else self.opportunity_count + 1
        )
        if self.opportunity_count in self.q_levels:
            self.pending_snapshot_budgets.add(self.opportunity_count)

    def _finalize_learning_opportunity(self) -> None:
        if not self.pending_snapshot_budgets or self.snapshot_recorder is None:
            return
        for budget in sorted(self.pending_snapshot_budgets):
            if self.donor_shadow_model is not None:
                self.snapshot_recorder("donor_shadow", budget, self.donor_shadow_model, {})
            if (
                self.cohort == "study_b"
                and self.activation_update_enabled
                and self.model is not None
            ):
                self.snapshot_recorder("personalized", budget, self.model, {})
        self.pending_snapshot_budgets.clear()

    def _balanced_feedback_weight(self, label: int) -> float:
        same = self.positive_feedbacks if label == 1 else self.negative_feedbacks
        opposite = self.negative_feedbacks if label == 1 else self.positive_feedbacks
        return float(np.clip(math.sqrt((opposite + 1.0) / (same + 1.0)), 0.35, 3.0))

    def _record_eligible_feedback(self, label: int) -> dict[str, int]:
        if self.feedback_counter is None:
            counts = {
                "activation_opportunities": self.activation_count,
                "eligible_feedbacks": self.feedback_count + 1,
                "positive_feedbacks": self.positive_feedbacks + int(label == 1),
                "negative_feedbacks": self.negative_feedbacks + int(label == 0),
            }
        else:
            counts = self.feedback_counter(label)
        self.activation_count = int(counts["activation_opportunities"])
        self.feedback_count = int(counts["eligible_feedbacks"])
        self.positive_feedbacks = int(counts["positive_feedbacks"])
        self.negative_feedbacks = int(counts["negative_feedbacks"])
        if self.feedback_count in self.snapshot_levels and self.snapshot_recorder is not None:
            metadata = {
                "budget_kind": "eligible_feedback",
                "budget_semantics": self.budget_semantics,
                "cap_budget": self.feedback_count,
                "realized_feedbacks": self.feedback_count,
                "observed_feedbacks": self.feedback_count,
                "cap_attained": True,
                "capped_final": False,
                "personal_calibration": self._personal_calibration_metadata(),
                **counts,
            }
            if self.donor_shadow_model is not None:
                self.snapshot_recorder(
                    "donor_shadow", self.feedback_count, self.donor_shadow_model, metadata
                )
            if (
                self.cohort == "study_b"
                and self.activation_update_enabled
                and self.model is not None
            ):
                self.snapshot_recorder(
                    "personalized", self.feedback_count, self.model, metadata
                )
        return counts

    def _personal_calibration_metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.personal_calibration_enabled,
            "version": self.calibration_version,
            "protocol_sha256": self.protocol_sha256,
            "logit_bias": self.personal_logit_bias,
            "update_count": self.personal_calibration_update_count,
            "positive_updates": self.personal_positive_updates,
            "negative_updates": self.personal_negative_updates,
            "shared_prior_inclusion": False,
        }

    def _network_scale(self) -> tuple[float, str]:
        for limit, scale in self.network_update_schedule:
            if self.feedback_count < limit:
                return float(scale), f"feedback_lt_{limit}"
        if self.network_update_schedule:
            return 1.0, "fully_unfrozen"
        if self.feedback_count < self.warmup_feedbacks:
            return float(self.warmup_update_scale), "legacy_warmup"
        return 1.0, "full_update"

    def _feedback_prediction(self, audio: np.ndarray) -> CalibratedPrediction:
        assert self.model is not None
        raw_probability, raw_logit = self.model.predict(audio)
        return self._calibrate(
            raw_probability,
            raw_logit,
            policy_effective="current",
            include_personal=True,
        )

    def _update_personal_calibration(
        self,
        *,
        label: int,
        probability_before: float,
        class_weight: float,
        result: dict[str, Any],
    ) -> None:
        result["personal_calibration_applied"] = False
        result["personal_logit_bias_before"] = self.personal_logit_bias
        result["feedback_probability_before"] = probability_before
        if not self.personal_calibration_enabled:
            result["personal_logit_bias_after"] = self.personal_logit_bias
            return

        gradient = (
            class_weight * (float(label) - float(probability_before))
            - self.personal_calibration_l2 * self.personal_logit_bias
        )
        step = self.personal_calibration_learning_rate * gradient
        proposed = float(
            np.clip(
                self.personal_logit_bias + step,
                self.personal_logit_bias_min,
                self.personal_logit_bias_max,
            )
        )
        if self.calibration_recorder is not None:
            state = self.calibration_recorder(
                self.personal_calibration_update_count,
                label,
                proposed,
            )
            proposed = float(state["personal_logit_bias"])
            self.personal_calibration_update_count = int(state["update_count"])
            self.personal_positive_updates = int(state["positive_updates"])
            self.personal_negative_updates = int(state["negative_updates"])
        else:
            self.personal_calibration_update_count += 1
            self.personal_positive_updates += int(label == 1)
            self.personal_negative_updates += int(label == 0)
        self.personal_logit_bias = proposed
        result.update(
            {
                "personal_calibration_applied": True,
                "personal_calibration_gradient": gradient,
                "personal_calibration_step": step,
                "personal_logit_bias_after": proposed,
                "personal_calibration_update_count": self.personal_calibration_update_count,
                "personal_positive_updates": self.personal_positive_updates,
                "personal_negative_updates": self.personal_negative_updates,
            }
        )

    def _apply_training_update(
        self, audio: np.ndarray, label: int, result: dict[str, Any]
    ) -> None:
        base_weight = (
            self._balanced_feedback_weight(label)
            if self.budget_kind == "eligible_feedback"
            else 1.0
        )
        assert self.model is not None
        feedback_prediction = self._feedback_prediction(audio)
        audit_reader = getattr(self.model, "audit_state", None)
        if callable(audit_reader):
            result["model_head_comparison_before_update"] = audit_reader()
        network_scale, network_stage = self._network_scale()
        model_weight = base_weight * network_scale
        result.update(
            {
                "base_class_weight": base_weight,
                "class_weight": model_weight,
                "network_update_scale": network_scale,
                "network_update_stage": network_stage,
                "warmup_update_applied": network_scale < 1.0,
                "warmup_update_scale": network_scale,
                "feedback_raw_probability_before": feedback_prediction.raw_probability,
                "feedback_raw_logit_before": feedback_prediction.raw_logit,
                "feedback_logit_before": feedback_prediction.logit,
            }
        )
        current_network_applied = False
        if network_scale > 0:
            if isinstance(self.model, DTCNModel):
                result["loss"] = self.model.train(
                    audio,
                    label,
                    weight=model_weight,
                    update_scale=network_scale,
                )
            elif bool(getattr(self.model, "supports_update_scale", False)):
                result["loss"] = self.model.train(
                    audio,
                    label,
                    weight=model_weight,
                    update_scale=network_scale,
                )
            else:
                result["loss"] = self.model.train(audio, label, weight=model_weight)
            current_network_applied = bool(
                getattr(self.model, "last_update_applied", True)
            )
            if callable(audit_reader):
                result["model_head_comparison_after_update"] = audit_reader()
            for field in (
                "last_optimizer_step_before",
                "last_optimizer_step_after",
                "last_requested_training_steps",
                "last_training_steps",
            ):
                if hasattr(self.model, field):
                    result[field.removeprefix("last_")] = int(
                        getattr(self.model, field)
                    )
            for field in (
                "last_optimizer_learning_rates_before",
                "last_optimizer_learning_rates_during",
                "last_optimizer_learning_rates_after",
            ):
                if hasattr(self.model, field):
                    result[field.removeprefix("last_")] = list(
                        getattr(self.model, field)
                    )
            result["parameter_state_interpolated"] = bool(
                getattr(self.model, "last_parameter_state_interpolated", False)
            )
            result["parameter_interpolation_scale"] = float(
                getattr(self.model, "last_parameter_interpolation_scale", 1.0)
            )
        else:
            result.update(
                {
                    "loss": 0.0,
                    "optimizer_step_before": 0,
                    "optimizer_step_after": 0,
                    "requested_training_steps": 0,
                    "training_steps": 0,
                }
            )
        result["current_network_update_applied"] = current_network_applied
        if network_scale > 0 and not current_network_applied:
            result["training_applied"] = False
            result["training_eligible"] = False
            result["quarantined"] = True
            result["quarantine_reason"] = str(
                getattr(self.model, "last_update_reason", None)
                or "optimizer_update_not_applied"
            )
            return
        shadow_network_applied = False
        if self.donor_shadow_model is not None:
            result["shadow_loss"] = self.donor_shadow_model.train(
                audio, label, weight=base_weight
            )
            shadow_network_applied = bool(
                getattr(self.donor_shadow_model, "last_update_applied", True)
            )
            if not shadow_network_applied:
                result["training_applied"] = False
                result["training_eligible"] = False
                result["quarantined"] = True
                result["quarantine_reason"] = "donor_shadow_optimizer_update_not_applied"
                return
            self._save_shadow_model()
        result["shadow_network_update_applied"] = shadow_network_applied
        self._update_personal_calibration(
            label=label,
            probability_before=feedback_prediction.probability,
            class_weight=base_weight,
            result=result,
        )
        result["training_applied"] = bool(
            current_network_applied or result["personal_calibration_applied"]
        )
        if not result["training_applied"]:
            result["training_eligible"] = False
            result["quarantined"] = True
            result["quarantine_reason"] = "no_locked_adaptation_step_applied"
            return
        if current_network_applied:
            self._save_model()
        if self.budget_kind == "eligible_feedback":
            result.update(self._record_eligible_feedback(label))

    def flush_pending_snapshots(self) -> None:
        with self.lock:
            self._finalize_learning_opportunity()

    def finalize_capped_snapshots(self) -> None:
        if self.budget_semantics != "per_donor_cap" or self.snapshot_recorder is None:
            return
        with self.lock:
            self._finalize_learning_opportunity()
            counts = {
                "activation_opportunities": self.activation_count,
                "eligible_feedbacks": self.feedback_count,
                "positive_feedbacks": self.positive_feedbacks,
                "negative_feedbacks": self.negative_feedbacks,
            }
            for cap in self.q_levels:
                if self.feedback_count >= cap:
                    continue
                metadata = {
                    "budget_kind": self.budget_kind,
                    "budget_semantics": self.budget_semantics,
                    "cap_budget": cap,
                    "realized_feedbacks": self.feedback_count,
                    "observed_feedbacks": self.feedback_count,
                    "cap_attained": False,
                    "capped_final": True,
                    "finalized_day": self.day,
                    "personal_calibration": self._personal_calibration_metadata(),
                    **counts,
                }
                if self.donor_shadow_model is not None:
                    self.snapshot_recorder(
                        "donor_shadow", cap, self.donor_shadow_model, metadata
                    )
                if (
                    self.cohort == "study_b"
                    and self.group_name.lower()
                    not in {"frozen", "f", "holdout", "holdout_evaluator"}
                    and self.model is not None
                ):
                    self.snapshot_recorder("personalized", cap, self.model, metadata)

    def _voice_activity(self, rms: float) -> tuple[bool, float, float]:
        alpha = float(np.clip(self.voice_rms_ema_alpha, 0.05, 1.0))
        if self.voice_rms_ema is None:
            self.voice_rms_ema = float(rms)
        else:
            self.voice_rms_ema = float(alpha * rms + (1.0 - alpha) * self.voice_rms_ema)
        level = float(self.voice_rms_ema)
        noise_floor = 0.0

        if not self.adaptive_voice_threshold_enabled:
            threshold = float(self.voice_rms_floor)
            above = level >= threshold
        else:
            self.voice_rms_history.append(level)
            while len(self.voice_rms_history) > self.voice_noise_window_frames:
                self.voice_rms_history.popleft()
            noise_floor = float(
                np.percentile(
                    np.asarray(self.voice_rms_history, dtype=np.float32),
                    self.voice_noise_percentile,
                )
            )
            threshold = max(
                float(self.voice_rms_floor),
                noise_floor * float(self.voice_noise_multiplier),
            )
            calibrated = len(self.voice_rms_history) >= self.voice_calibration_frames
            if not calibrated:
                self.voice_confirm_count = 0
                return False, threshold, noise_floor
            above = level >= threshold

        if above:
            self.voice_confirm_count += 1
        else:
            self.voice_confirm_count = 0
        confirm_needed = max(1, int(self.voice_confirm_frames))
        return self.voice_confirm_count >= confirm_needed, threshold, noise_floor

    def note_asr_final(self, client_ms: float) -> dict[str, Any]:
        with self.lock:
            self.last_asr_final_ms = float(client_ms)
            return {
                "last_asr_final_ms": self.last_asr_final_ms,
                "last_activation_ms": self.last_activation_ms,
                "explicit_turn_end_required": self.explicit_turn_end_required,
            }

    def _asr_turn_end_ready(self, client_ms: float) -> bool:
        del client_ms
        if not self.explicit_turn_end_required:
            return True
        if self.last_asr_final_ms is None:
            return False
        return float(self.last_asr_final_ms) > float(self.last_activation_ms)

    def process_pcm(self, pcm: bytes, client_ms: float) -> dict[str, Any]:
        values = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if not len(values):
            return {"probability": 0.0, "activate": False}
        with self.lock:
            if self.phase == "rest" or not self.analysis_eligible:
                return {
                    "probability": 0.0,
                    "activate": False,
                    "skipped": True,
                    "skip_reason": "scheduled_rest_excluded",
                    "phase": self.phase,
                    **self._server_state_snapshot(),
                }
            self.ring.append(values)
            rms = float(np.sqrt(np.mean(values * values) + 1e-12))
            voice_detected, voice_rms_threshold, voice_noise_floor = self._voice_activity(rms)
            if voice_detected:
                self.last_voice_ms = client_ms
                if self.agent_state != "waiting":
                    self.has_voice_since_activation = True

            elapsed = client_ms - self.last_prediction_ms
            if elapsed < self.prediction_interval_ms:
                return {
                    "rms": rms,
                    "voice_detected": voice_detected,
                    "voice_rms_threshold": voice_rms_threshold,
                    "voice_noise_floor": voice_noise_floor,
                    "activate": False,
                    "skipped": True,
                }
            self.last_prediction_ms = client_ms

            silence_ms = (
                max(0.0, client_ms - self.last_voice_ms) if self.last_voice_ms is not None else 0.0
            )
            # Telemetry only: silence / ASR-final no longer gate activation timing.
            asr_turn_end_ready = self._asr_turn_end_ready(client_ms)
            recent_audio = self.ring.recent(30.0)

        # Shared DistilHuBERT inference must not hold the per-subject lock.
        # Another participant's GPU turn used to block this subject's tap()
        # acknowledgements and close their timing socket.
        try:
            prediction = self._predict_audio(
                recent_audio,
                silence_ms,
                encoder_timeout=_shared_encoder_wait_s(
                    getattr(self.settings, "distilhubert_device", None)
                ),
            )
        except Exception as exc:
            if not _is_shared_encoder_busy(exc):
                raise
            return {
                "rms": rms,
                "voice_detected": voice_detected,
                "voice_rms_threshold": voice_rms_threshold,
                "voice_noise_floor": voice_noise_floor,
                "activate": False,
                "skipped": True,
                "skip_reason": "shared_encoder_busy",
            }

        with self.lock:
            if self.policy == "current":
                prediction, tap_decision = self._apply_tap_history_fusion(
                    prediction,
                    now_seconds=float(client_ms) / 1000.0,
                    audio=recent_audio,
                )
            else:
                tap_decision = {
                    "early_fusion_logit": 0.0,
                    "threshold_shift": 0.0,
                    "effective_activation_threshold": self.activation_threshold,
                }
            effective_threshold = float(
                tap_decision["effective_activation_threshold"]
            )
            context = self._decision_context_snapshot()
            activation_audio_trainable = bool(
                self.policy != "current"
                or self.model is None
                or not self.update_enabled
                or self._training_audio_compatible(recent_audio)
            )
            # Continuous realtime path: the timing model holds activation
            # authority.  VAD remains diagnostic telemetry only; microphone
            # gain must not decide whether a model prediction is eligible.
            # EOU silence/ASR-final, addressee, and reply-needed gates are off.
            can_activate = (
                self.agent_state == "waiting"
                and self.phase not in {"complete", "rest"}
                and self.analysis_eligible
                and client_ms - self.last_activation_ms >= 1500
                and activation_audio_trainable
            )
            gate_reasons: list[str] = []
            if self.agent_state != "waiting":
                gate_reasons.append("agent_active")
            if self.phase == "complete":
                gate_reasons.append("phase_complete")
            if client_ms - self.last_activation_ms < 1500:
                gate_reasons.append("refractory_period")
            if not activation_audio_trainable:
                gate_reasons.append("training_audio_not_ready")
            activate = can_activate and prediction.probability > effective_threshold
            if activate:
                source = "model" if self.policy != "vad" else "vad"
                self._begin_activation(
                    client_ms,
                    source,
                    prediction.policy_effective,
                    prediction,
                )
            threshold_logit = math.log(
                max(effective_threshold, 1e-6)
                / max(1.0 - effective_threshold, 1e-6)
            )
            raw_logit_threshold = self.temperature * (
                threshold_logit
                - self.logit_bias
                - prediction.personal_logit_bias
                - float(tap_decision["early_fusion_logit"])
            )
            result = {
                "rms": rms,
                "voice_detected": voice_detected,
                "voice_rms_threshold": voice_rms_threshold,
                "voice_noise_floor": voice_noise_floor,
                "voice_rms_ema": self.voice_rms_ema,
                "adaptive_voice_threshold_enabled": self.adaptive_voice_threshold_enabled,
                "silence_ms": silence_ms,
                "asr_turn_end_ready": asr_turn_end_ready,
                "last_asr_final_ms": self.last_asr_final_ms,
                "probability": prediction.probability,
                "logit": prediction.logit,
                "raw_probability": prediction.raw_probability,
                "raw_logit": prediction.raw_logit,
                "temperature": self.temperature,
                "global_logit_bias": self.logit_bias,
                "personal_logit_bias": prediction.personal_logit_bias,
                "activation_threshold": self.activation_threshold,
                "effective_activation_threshold": effective_threshold,
                "tap_history_early_fusion_logit": float(
                    tap_decision["early_fusion_logit"]
                ),
                "tap_history_threshold_shift": float(tap_decision["threshold_shift"]),
                "minimum_end_of_turn_silence_ms": self.minimum_end_of_turn_silence_ms,
                "explicit_turn_end_required": self.explicit_turn_end_required,
                "raw_logit_threshold": raw_logit_threshold,
                "decision_eligible": can_activate,
                "decision_gate_reasons": gate_reasons,
                "policy": self.policy,
                "policy_effective": prediction.policy_effective,
                "phase": self.phase,
                "candidate_probe_token": (
                    self.candidate_probe_token
                    if self.policy == "candidate_prior"
                    else None
                ),
                "activate": activate,
                **context,
            }
            if self.policy == "current" and self.model is not None:
                audit_reader = getattr(self.model, "audit_state", None)
                if callable(audit_reader):
                    # This records the immutable shared-head and current
                    # personal-head outputs at the same decision opportunity.
                    result["model_head_comparison"] = audit_reader()
            if activate:
                result.update(self._activation_provenance())
            return result

    @staticmethod
    def _sigmoid(logit: float) -> float:
        if logit >= 0:
            inverse = math.exp(-logit)
            return 1.0 / (1.0 + inverse)
        value = math.exp(logit)
        return value / (1.0 + value)

    def _calibrate(
        self,
        probability: float,
        logit: float,
        *,
        policy_effective: str,
        include_personal: bool,
    ) -> CalibratedPrediction:
        personal_bias = self.personal_logit_bias if include_personal else 0.0
        calibrated_logit = (
            float(logit) / max(self.temperature, 1e-6)
            + self.logit_bias
            + personal_bias
        )
        return CalibratedPrediction(
            probability=self._sigmoid(calibrated_logit),
            logit=calibrated_logit,
            raw_probability=float(probability),
            raw_logit=float(logit),
            policy_effective=policy_effective,
            personal_logit_bias=personal_bias,
        )

    def _model_predict(
        self,
        model: Any,
        audio: np.ndarray,
        *,
        encoder_timeout: float | None,
    ) -> tuple[float, float]:
        if encoder_timeout is None:
            return model.predict(audio)
        try:
            return model.predict(audio, encoder_timeout=encoder_timeout)
        except TypeError:
            return model.predict(audio)

    def _predict(self, silence_ms: float) -> CalibratedPrediction:
        return self._predict_audio(self.ring.recent(30.0), silence_ms)

    def _predict_audio(
        self,
        audio: np.ndarray,
        silence_ms: float,
        *,
        encoder_timeout: float | None = None,
    ) -> CalibratedPrediction:
        if self.policy == "vad" or self.model is None:
            probability = 1.0 / (1.0 + math.exp(-(silence_ms - 1000.0) / 180.0))
            effective = "vad" if self.policy == "vad" else "pilot_vad_fallback"
            logit = math.log(
                max(probability, 1e-6) / max(1.0 - probability, 1e-6)
            )
            return CalibratedPrediction(
                probability=probability,
                logit=logit,
                raw_probability=probability,
                raw_logit=logit,
                policy_effective=effective,
                personal_logit_bias=0.0,
            )
        if self.policy == "matched_other":
            if self.model_identity.get("matched_other_supported") is False:
                raise ModelUnavailable(
                    "This model-family prototype has no matched-other policy"
                )
            if self.matched_model is None:
                probability, logit = self._model_predict(
                    self.model, audio, encoder_timeout=encoder_timeout
                )
                return self._calibrate(
                    probability,
                    logit,
                    policy_effective="current_missing_matched_fallback",
                    include_personal=True,
                )
            probability, logit = self._model_predict(
                self.matched_model, audio, encoder_timeout=encoder_timeout
            )
            return self._calibrate(
                probability,
                logit,
                policy_effective="matched_other",
                include_personal=False,
            )
        if self.policy == "base":
            if self.matched_model is None:
                raise ModelUnavailable("Locked M0 comparison model is missing")
            probability, logit = self._model_predict(
                self.matched_model, audio, encoder_timeout=encoder_timeout
            )
            return self._calibrate(
                probability,
                logit,
                policy_effective="base",
                include_personal=False,
            )
        if self.policy == "candidate_prior":
            if self.candidate_model is None or not self.candidate_probe_token:
                raise ModelUnavailable("Locked candidate-prior model is missing")
            probability, logit = self._model_predict(
                self.candidate_model, audio, encoder_timeout=encoder_timeout
            )
            return self._calibrate(
                probability,
                logit,
                policy_effective="candidate_prior",
                include_personal=False,
            )
        probability, logit = self._model_predict(
            self.model, audio, encoder_timeout=encoder_timeout
        )
        return self._calibrate(
            probability,
            logit,
            policy_effective="current",
            include_personal=True,
        )

    def _training_audio_compatible(self, audio: np.ndarray | None) -> bool:
        if self.model is None or audio is None:
            return False
        checker = getattr(self.model, "can_train", None)
        return bool(checker(audio)) if callable(checker) else True

    def tap(
        self,
        client_ms: float,
        *,
        pointer_agent_state: str | None = None,
        page_visible: bool | None = None,
        intended_action: str | None = None,
        expected_state: str | None = None,
        expected_activation_id: str | None = None,
        expected_state_version: int | None = None,
        feedback_attribution: str = "acoustic_personalizer",
    ) -> dict[str, Any]:
        with self.lock:
            cancel_reason: str | None = None
            conditional_values = (
                intended_action,
                expected_state,
                expected_activation_id,
                expected_state_version,
            )
            conditional_requested = any(value is not None for value in conditional_values)
            conditional_complete = all(value is not None for value in conditional_values)
            if conditional_requested and not conditional_complete:
                cancel_reason = "conditional_fields_required"
            elif intended_action not in {None, "manual_activate", "interrupt"}:
                cancel_reason = "invalid_intended_action"
            elif expected_state is not None and expected_state != self.agent_state:
                cancel_reason = "expected_state_mismatch"
            elif (
                expected_activation_id is not None
                and expected_activation_id != (self.activation_id or "")
            ):
                cancel_reason = "expected_activation_mismatch"
            elif (
                expected_state_version is not None
                and expected_state_version != self.state_version
            ):
                cancel_reason = "expected_state_version_mismatch"
            elif intended_action == "manual_activate" and self.agent_state != "waiting":
                cancel_reason = "manual_requires_waiting"
            elif intended_action == "interrupt" and self.agent_state not in {"thinking", "speaking"}:
                cancel_reason = "interrupt_requires_active_agent"
            elif intended_action == "interrupt" and not self.activation_id:
                cancel_reason = "interrupt_requires_activation"
            elif feedback_attribution not in {
                "acoustic_personalizer",
                "reply_needed_context",
            }:
                cancel_reason = "invalid_feedback_attribution"
            if cancel_reason is not None:
                return {
                    "action": "conditional_tap_cancelled",
                    "accepted": False,
                    "cancel_reason": cancel_reason,
                    "intended_action": intended_action,
                    "expected_state": expected_state,
                    "expected_activation_id": expected_activation_id,
                    "expected_state_version": expected_state_version,
                    "feedback_attribution": feedback_attribution,
                    "training_eligible": False,
                    "training_applied": False,
                    "label": None,
                    "quarantined": True,
                    "quarantine_reason": "conditional_tap_cancelled",
                    **self._server_state_snapshot(),
                }
            if self.phase in {"complete", "rest"}:
                return {
                    "action": (
                        "scheduled_rest_ignored"
                        if self.phase == "rest"
                        else "after_complete_ignored"
                    ),
                    "training_eligible": False,
                    "training_applied": False,
                    "label": None,
                    "quarantined": True,
                    "quarantine_reason": (
                        "scheduled_rest_excluded"
                        if self.phase == "rest"
                        else "interaction_complete"
                    ),
                    "phase": self.phase,
                    "policy": self.policy,
                    "tap_phase": self.phase,
                    "tap_policy": self.policy,
                    **self._server_state_snapshot(),
                }
            state_age_ms = client_ms - self.last_state_change_ms
            if (
                conditional_complete
                and self.last_state_change_ms > 0
                and 0 <= state_age_ms < self.state_transition_tap_guard_ms
            ):
                return {
                    "action": "state_transition_mistouch",
                    "accepted": False,
                    "cancel_reason": "state_transition_guard",
                    "state_age_ms": state_age_ms,
                    "state_transition_tap_guard_ms": self.state_transition_tap_guard_ms,
                    "intended_action": intended_action,
                    "expected_state": expected_state,
                    "expected_activation_id": expected_activation_id,
                    "expected_state_version": expected_state_version,
                    "feedback_attribution": feedback_attribution,
                    "training_eligible": False,
                    "training_applied": False,
                    "label": None,
                    "quarantined": True,
                    "quarantine_reason": "state_transition_mistouch",
                    **self._server_state_snapshot(),
                }
            # A complete conditional tap is already atomically bound to the exact
            # server state/version/activation. The legacy 700 ms debounce is only
            # needed for unbound UI bursts; applying it here can discard a valid
            # correction after a distinct state transition.
            if not conditional_complete and client_ms - self.last_tap_ms < 700:
                return {
                    "action": "tap_burst_ignored",
                    "training_eligible": False,
                    **self._server_state_snapshot(),
                }
            self.last_tap_ms = client_ms

            if self.agent_state == "waiting":
                self._begin_activation(client_ms, "manual", "manual")
                activation_audio = self.activation_audio
                audio_compatible = self._training_audio_compatible(activation_audio)
                eligible = bool(
                    feedback_attribution == "acoustic_personalizer"
                    and
                    self.activation_update_enabled
                    and self.model is not None
                    and activation_audio is not None
                    and audio_compatible
                )
                if feedback_attribution != "acoustic_personalizer":
                    quarantine_reason = "feedback_attributed_to_context_gate"
                elif not self.activation_update_enabled:
                    quarantine_reason = "activation_not_learning_current"
                elif self.model is None or activation_audio is None:
                    quarantine_reason = "training_audio_missing"
                elif not audio_compatible:
                    quarantine_reason = "training_audio_too_short"
                else:
                    quarantine_reason = None
                result = {
                    "action": "manual_activate",
                    "training_eligible": eligible,
                    "training_applied": False,
                    "training_audio_samples": (
                        int(len(activation_audio)) if activation_audio is not None else 0
                    ),
                    "training_audio_compatible": audio_compatible,
                    "label": 1,
                    "quarantined": not eligible,
                    "quarantine_reason": quarantine_reason,
                    "phase": self.activation_phase,
                    "policy": self.activation_policy,
                    "tap_phase": self.phase,
                    "tap_policy": self.policy,
                    "intended_action": intended_action,
                    "feedback_attribution": feedback_attribution,
                    **self._activation_provenance(),
                    **self._server_state_snapshot(),
                }
                if result["training_eligible"]:
                    assert activation_audio is not None
                    self._apply_training_update(activation_audio, 1, result)
                    self._coadapt_tap_history(
                        client_ms=client_ms,
                        kind="manual_activate",
                        interrupt_delay_ms=None,
                        result=result,
                    )
                return result

            latency_ms = client_ms - self.last_activation_ms
            manual_conflict = self.activation_source == "manual"
            provenance = self._activation_provenance()
            activation_audio = self.activation_audio
            audio_compatible = self._training_audio_compatible(activation_audio)
            interrupt_agent_state = self.agent_state
            pointer_state_active = pointer_agent_state in {"thinking", "speaking"}
            frontend_observable = bool(page_visible is True and pointer_state_active)
            if frontend_observable:
                label_confidence = "high"
            elif pointer_state_active or page_visible is True:
                label_confidence = "standard"
            else:
                label_confidence = "telemetry_unverified"
            timing_attribution_confidence = (
                "speech_overlap_observed"
                if self.has_voice_since_activation
                else "no_pre_tap_speech_observed"
            )
            self._set_agent_state_locked("waiting", client_ms)
            eligible = bool(
                feedback_attribution == "acoustic_personalizer"
                and
                self.activation_source == "model"
                and self.activation_update_enabled
                and self.model is not None
                and activation_audio is not None
                and audio_compatible
            )
            if feedback_attribution != "acoustic_personalizer":
                quarantine_reason = "feedback_attributed_to_context_gate"
            elif manual_conflict:
                quarantine_reason = "manual_activate_interrupt_conflict"
            elif self.activation_source != "model":
                quarantine_reason = "non_model_activation"
            elif not self.activation_update_enabled:
                quarantine_reason = "activation_not_learning_current"
            elif activation_audio is None:
                quarantine_reason = "training_audio_missing"
            elif not audio_compatible:
                quarantine_reason = "training_audio_too_short"
            else:
                quarantine_reason = None
            result = {
                "action": "interrupt",
                "activation_to_tap_ms": latency_ms,
                "user_speech_resumed": self.has_voice_since_activation,
                "interrupt_agent_state": interrupt_agent_state,
                "pointer_agent_state": pointer_agent_state,
                "page_visible": page_visible,
                "frontend_observable": frontend_observable,
                "label_semantics": "observable_activation_interrupt_v2",
                "label_confidence": label_confidence,
                "label_confidence_basis": {
                    "activation_id_bound": bool(self.activation_id),
                    "pointer_state_active": pointer_state_active,
                    "page_visible": page_visible,
                },
                "timing_attribution_confidence": timing_attribution_confidence,
                "training_eligible": eligible,
                "training_applied": False,
                "training_audio_samples": (
                    int(len(activation_audio)) if activation_audio is not None else 0
                ),
                "training_audio_compatible": audio_compatible,
                "label": 0,
                "quarantined": not eligible,
                "quarantine_reason": quarantine_reason,
                "phase": self.activation_phase,
                "policy": self.activation_policy,
                "tap_phase": self.phase,
                "tap_policy": self.policy,
                "intended_action": intended_action,
                "feedback_attribution": feedback_attribution,
                **provenance,
            }
            if result["training_eligible"]:
                assert activation_audio is not None
                self._apply_training_update(activation_audio, 0, result)
                self._coadapt_tap_history(
                    client_ms=client_ms,
                    kind="interrupt",
                    interrupt_delay_ms=float(latency_ms),
                    result=result,
                )
            self._finalize_learning_opportunity()
            self._clear_activation()
            result.update(self._server_state_snapshot())
            return result

    def _save_model(self) -> None:
        if self.model is not None:
            self.model.save(
                self.model_storage_dir
                or self.settings.data_dir / "models" / self.participant_id
            )

    def _save_shadow_model(self) -> None:
        if self.donor_shadow_model is not None:
            self.donor_shadow_model.save(
                self.settings.data_dir / "donor-shadow" / self.participant_id / "latest"
            )


class TimingService:
    def __init__(self, settings: Settings, db: "StudyDB"):
        self.settings = settings
        self.db = db
        self.runtimes: dict[tuple[str, int], ParticipantRuntime] = {}
        self.protocol_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.distilhubert_feature_encoder: Any | None = None
        self._encoder_lock = threading.Lock()
        self._runtime_build_locks: dict[tuple[str, int], threading.Lock] = {}
        # Fail before accepting any WebSocket/session when the pinned release
        # or frozen backbone cannot be loaded.  Runtime creation then reuses
        # this process-wide ready encoder and only clones small linear heads.
        self._ensure_distilhubert_encoder()

    def _ensure_distilhubert_encoder(self) -> None:
        if self.settings.agent_mode != "distilhubert":
            return
        if self.distilhubert_feature_encoder is not None:
            return
        with self._encoder_lock:
            if self.distilhubert_feature_encoder is not None:
                return
            manifest_path = Path(self.settings.distilhubert_release_manifest)
            if (
                not self.settings.distilhubert_release_manifest
                or not manifest_path.is_file()
            ):
                raise ModelUnavailable("DISTILHUBERT_RELEASE_MANIFEST is missing")
            try:
                from .distilhubert_online import (
                    DistilHuBERTOnlineModel,
                    DistilHuBERTUnavailable,
                )

                replica_count = 1
                if self.settings.listener_role != "http":
                    replica_count = max(
                        1,
                        int(getattr(self.settings, "distilhubert_encoder_replicas", 1) or 1),
                    )
                encoder_devices = getattr(
                    self.settings, "distilhubert_encoder_devices", None
                )
                if not encoder_devices:
                    encoder_devices = (
                        None
                        if self.settings.distilhubert_device == "auto"
                        else (self.settings.distilhubert_device,)
                    )
                self.distilhubert_feature_encoder = (
                    DistilHuBERTOnlineModel.preload_feature_encoder(
                        manifest_path,
                        cache_dir=(self.settings.distilhubert_cache_dir or None),
                        allow_download=self.settings.distilhubert_allow_download,
                        device=(
                            None
                            if self.settings.distilhubert_device == "auto"
                            else self.settings.distilhubert_device
                        ),
                        replica_count=replica_count,
                        devices=encoder_devices,
                    )
                )
                DistilHuBERTOnlineModel(
                    manifest_path,
                    update_enabled=False,
                    feature_encoder=self.distilhubert_feature_encoder,
                )
            except DistilHuBERTUnavailable as exc:
                raise ModelUnavailable(str(exc)) from exc

    def _personal_model_dir(self, participant: dict[str, Any]) -> Path:
        participant_id = str(participant["id"])
        if not bool(participant.get("analysis_excluded")):
            return self.settings.data_dir / "models" / participant_id
        # The public /tap-test instance already has its own physical data root.
        # Its self-enrolled participants are intentionally excluded from formal
        # analysis, but (unlike the retired admin-created debug batches) do not
        # have an account_batch row.  Keep their personal heads in a dedicated
        # per-participant namespace inside data-test instead of rejecting the
        # timing WebSocket before it can open.
        if self.settings.instance_id == "test":
            return (
                self.settings.data_dir
                / "personal_debug"
                / "public_enrollment"
                / participant_id
                / "models"
            )
        batch = self.db.account_batch_for_participant(participant_id)
        if (
            batch is None
            or not bool(batch.get("analysis_excluded"))
            or str(batch.get("storage_namespace")) != "personal_debug"
            or not batch.get("archive_relative_path")
        ):
            raise ModelUnavailable(
                "Analysis-excluded participant is missing its personal-debug isolation record"
            )
        data_root = self.settings.data_dir.resolve()
        archive_root = (
            data_root / str(batch["archive_relative_path"])
        ).resolve()
        if not archive_root.is_relative_to(data_root):
            raise ModelUnavailable("Personal-debug archive escapes the study data directory")
        return archive_root / "models" / participant_id

    def _runtime_configuration(self, participant: dict[str, Any]) -> dict[str, Any]:
        protocol_text = str(participant.get("run_protocol_json") or "")
        protocol_sha256 = str(participant.get("run_protocol_sha256") or "")
        if not protocol_text or not protocol_sha256:
            raise ModelUnavailable("Study run is missing its immutable protocol snapshot")
        actual_sha256 = hashlib.sha256(protocol_text.encode("utf-8")).hexdigest()
        if actual_sha256 != protocol_sha256:
            raise ModelUnavailable("Study run protocol snapshot integrity check failed")
        cache_key = (str(participant["run_id"]), protocol_sha256)
        cached = self.protocol_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            study = json.loads(protocol_text)
        except json.JSONDecodeError as exc:
            raise ModelUnavailable("Study run protocol snapshot is invalid JSON") from exc
        if str(study.get("version") or "") != str(participant.get("protocol_version") or ""):
            raise ModelUnavailable("Study run protocol version does not match its snapshot")

        pipeline = study.get("pipeline", {})
        budget_kind = str(pipeline.get("budget_kind") or "activation_opportunity")
        budget_semantics = str(pipeline.get("budget_semantics") or "exact_count")
        levels = (
            pipeline.get("q_feedback_levels")
            if budget_kind == "eligible_feedback"
            else pipeline.get("q_opportunity_levels")
        )
        q_levels = tuple(int(value) for value in (levels or []))
        snapshot_max = int(pipeline.get("snapshot_every_feedback_up_to") or 0)
        snapshot_levels = tuple(range(1, snapshot_max + 1)) or q_levels
        initialization = study.get("initialization", {})
        temperature = float(initialization.get("temperature", 1.0))
        activation_threshold = float(initialization.get("activation_threshold", 0.5))
        if temperature <= 0 or not 0 < activation_threshold < 1:
            raise ModelUnavailable("Study run contains an invalid locked operating point")
        factorization = initialization.get("decision_factorization") or {}
        acoustic_vad = factorization.get("acoustic_vad") or {}
        reply_needed_threshold = float(
            factorization.get("reply_needed_threshold", 0.5)
        )
        # Hard-disable legacy EOU / reply-needed activation gates for all
        # enrolled protocol snapshots. Timing is continuous model-driven only.
        minimum_end_of_turn_silence_ms = 0.0
        explicit_turn_end_required = False
        decision_context_required = False
        adaptive_voice_threshold_enabled = bool(
            acoustic_vad.get("adaptive_threshold_enabled", False)
        )
        voice_rms_floor = float(acoustic_vad.get("minimum_rms_threshold", 0.020))
        # Harden legacy enrolled protocols that still pin the old 0.012 floor.
        voice_rms_floor = max(voice_rms_floor, 0.020)
        voice_noise_multiplier = float(acoustic_vad.get("noise_multiplier", 1.5))
        voice_noise_percentile = float(acoustic_vad.get("noise_percentile", 20.0))
        voice_noise_window_frames = int(acoustic_vad.get("window_frames", 50))
        voice_calibration_frames = int(acoustic_vad.get("calibration_frames", 5))
        # Older snapshots may omit smoothing knobs; default to the hardened EOU path.
        voice_rms_ema_alpha = float(acoustic_vad.get("rms_ema_alpha", 0.35))
        voice_confirm_frames = int(acoustic_vad.get("voice_confirm_frames", 2))
        feedback = study.get("feedback") or {}
        state_transition_tap_guard_ms = float(
            feedback.get("state_transition_tap_guard_ms", 500.0)
        )
        if (
            minimum_end_of_turn_silence_ms < 0
            or minimum_end_of_turn_silence_ms > 5_000
            or not 0.0 <= reply_needed_threshold <= 1.0
            or not 0.0 < voice_rms_floor <= 0.2
            or not 1.0 <= voice_noise_multiplier <= 8.0
            or not 1.0 <= voice_noise_percentile <= 50.0
            or not 5 <= voice_noise_window_frames <= 200
            or not 1 <= voice_calibration_frames <= voice_noise_window_frames
            or not 0.05 <= voice_rms_ema_alpha <= 1.0
            or not 1 <= voice_confirm_frames <= 8
            or not 0.0 <= state_transition_tap_guard_ms <= 5_000
        ):
            raise ModelUnavailable(
                "Study run contains invalid decision-factorization settings"
            )

        schedule: list[tuple[int, float]] = []
        previous_limit = 0
        for item in initialization.get("network_update_schedule") or []:
            limit = int(item["eligible_feedbacks_lt"])
            scale = float(item["scale"])
            if limit <= previous_limit or not 0 <= scale <= 1:
                raise ModelUnavailable("Study run contains an invalid network update schedule")
            schedule.append((limit, scale))
            previous_limit = limit

        personal = initialization.get("personal_calibration") or {}
        bias_min = float(personal.get("minimum_logit_bias", -1.0))
        bias_max = float(personal.get("maximum_logit_bias", 1.0))
        learning_rate = float(personal.get("learning_rate", 0.0))
        l2 = float(personal.get("l2", 0.0))
        initial_personal_bias = float(personal.get("initial_logit_bias", 0.0))
        if (
            bias_min >= bias_max
            or not bias_min <= initial_personal_bias <= bias_max
            or learning_rate < 0
            or l2 < 0
        ):
            raise ModelUnavailable("Study run contains invalid personal calibration limits")

        tap_raw = initialization.get("tap_timing_history")
        if isinstance(tap_raw, dict):
            tap_config = TapHistoryConfig.from_mapping(tap_raw)
        else:
            # Existing enrolled protocol snapshots may omit the block; fall back
            # to the locked A10 defaults gated by process env / agent mode.
            tap_config = TapHistoryConfig.locked_a10()
            tap_config = TapHistoryConfig(
                variant_id=self.settings.tap_history_variant or tap_config.variant_id,
                enabled=bool(self.settings.tap_history_enabled),
                horizon_s=tap_config.horizon_s,
                feature_mode=tap_config.feature_mode,
                fusion_mode=tap_config.fusion_mode,
                state_lr=tap_config.state_lr,
                k_interrupt0=tap_config.k_interrupt0,
                k_activate0=tap_config.k_activate0,
                early_fusion_delta=tap_config.early_fusion_delta,
                logit_clip_delta=tap_config.logit_clip_delta,
            )
        if self.settings.agent_mode != "distilhubert":
            tap_config = TapHistoryConfig.from_mapping(
                {**asdict(tap_config), "enabled": False}
            )

        configuration = {
            "budget_kind": budget_kind,
            "budget_semantics": budget_semantics,
            "q_levels": q_levels,
            "snapshot_levels": snapshot_levels,
            "calibration_version": str(
                initialization.get("version") or "uncalibrated"
            ),
            "protocol_sha256": protocol_sha256,
            "temperature": temperature,
            "logit_bias": float(initialization.get("logit_bias", 0.0)),
            "activation_threshold": activation_threshold,
            "minimum_end_of_turn_silence_ms": minimum_end_of_turn_silence_ms,
            "explicit_turn_end_required": explicit_turn_end_required,
            "adaptive_voice_threshold_enabled": adaptive_voice_threshold_enabled,
            "voice_rms_floor": voice_rms_floor,
            "voice_noise_multiplier": voice_noise_multiplier,
            "voice_noise_percentile": voice_noise_percentile,
            "voice_noise_window_frames": voice_noise_window_frames,
            "voice_calibration_frames": voice_calibration_frames,
            "voice_rms_ema_alpha": voice_rms_ema_alpha,
            "voice_confirm_frames": voice_confirm_frames,
            "state_transition_tap_guard_ms": state_transition_tap_guard_ms,
            "decision_context_required": decision_context_required,
            "reply_needed_threshold": reply_needed_threshold,
            "warmup_feedbacks": int(
                initialization.get("warmup_eligible_feedbacks", 0)
            ),
            "warmup_update_scale": float(
                initialization.get("warmup_update_scale", 1.0)
            ),
            "network_update_schedule": tuple(schedule),
            "personal_enabled": bool(personal.get("enabled", False)),
            "personal_initial_logit_bias": initial_personal_bias,
            "personal_learning_rate": learning_rate,
            "personal_l2": l2,
            "personal_bias_min": bias_min,
            "personal_bias_max": bias_max,
            "tap_history_config": tap_config,
        }
        self.protocol_cache[cache_key] = configuration
        return configuration

    def _record_snapshot(
        self,
        participant: dict[str, Any],
        session_id: str | None,
        source: str,
        budget: int,
        model: DTCNModel,
        metadata: dict[str, Any],
        budget_kind: str,
    ) -> None:
        existing = [
            row
            for row in self.db.model_snapshots(
                participant["run_id"],
                source=source,
                budget=budget,
                budget_kind=budget_kind,
            )
            if row["participant_id"] == participant["id"]
        ]
        if existing:
            return
        if bool(participant.get("analysis_excluded")):
            target = (
                self._personal_model_dir(participant)
                / "snapshots"
                / source
                / f"q-{budget}"
            )
        else:
            target = (
                self.settings.data_dir
                / "snapshots"
                / participant["id"]
                / source
                / f"q-{budget}"
            )
        encoder, main = model.save(target)
        self.db.record_model_snapshot(
            {
                "participant_id": participant["id"],
                "session_id": session_id,
                "source": source,
                "opportunity_budget": budget,
                "budget_kind": budget_kind,
                "eligible_feedbacks": metadata.get("eligible_feedbacks", 0),
                "positive_feedbacks": metadata.get("positive_feedbacks", 0),
                "negative_feedbacks": metadata.get("negative_feedbacks", 0),
                "metadata": metadata,
                "base_prior_id": participant.get("prior_id"),
                "encoder_path": str(encoder.resolve()),
                "main_path": str(main.resolve()),
                "encoder_sha256": file_sha256(encoder),
                "main_sha256": file_sha256(main),
            }
        )

    def runtime(
        self, participant: dict[str, Any], day: int = 1, session_id: str | None = None
    ) -> ParticipantRuntime:
        if self.settings.agent_mode == "distilhubert":
            self._ensure_distilhubert_encoder()
        participant_id = participant["id"]
        runtime_key = (participant_id, day)
        configuration = self._runtime_configuration(participant)
        personal_model_dir = self._personal_model_dir(participant)
        with self.lock:
            runtime = self.runtimes.get(runtime_key)
            if runtime is not None:
                return runtime
            key_lock = self._runtime_build_locks.setdefault(runtime_key, threading.Lock())
        with key_lock:
            with self.lock:
                runtime = self.runtimes.get(runtime_key)
                if runtime is not None:
                    return runtime
            if runtime is None:
                feedback_counts = self.db.feedback_counts(participant_id)
                personal_enabled = bool(configuration["personal_enabled"]) and (
                    str(participant["group_name"]).lower()
                    not in {"frozen", "f", "holdout", "holdout_evaluator"}
                )
                calibration_state: dict[str, Any] | None = None
                if personal_enabled:
                    calibration_state = self.db.calibration_state(
                        participant_id,
                        calibration_version=configuration["calibration_version"],
                        protocol_sha256=configuration["protocol_sha256"],
                        initial_logit_bias=configuration["personal_initial_logit_bias"],
                    )
                runtime = ParticipantRuntime(
                    participant_id=participant_id,
                    group_name=participant["group_name"],
                    cohort=participant["cohort"],
                    settings=self.settings,
                    model_storage_dir=personal_model_dir,
                    base_prior_id=participant.get("prior_id"),
                    session_id=session_id,
                    day=day,
                    budget_kind=configuration["budget_kind"],
                    budget_semantics=configuration["budget_semantics"],
                    q_levels=configuration["q_levels"],
                    snapshot_levels=configuration["snapshot_levels"],
                    calibration_version=configuration["calibration_version"],
                    protocol_sha256=configuration["protocol_sha256"],
                    temperature=configuration["temperature"],
                    logit_bias=configuration["logit_bias"],
                    activation_threshold=configuration["activation_threshold"],
                    minimum_end_of_turn_silence_ms=configuration[
                        "minimum_end_of_turn_silence_ms"
                    ],
                    explicit_turn_end_required=configuration[
                        "explicit_turn_end_required"
                    ],
                    adaptive_voice_threshold_enabled=configuration[
                        "adaptive_voice_threshold_enabled"
                    ],
                    voice_rms_floor=configuration["voice_rms_floor"],
                    voice_noise_multiplier=configuration["voice_noise_multiplier"],
                    voice_noise_percentile=configuration["voice_noise_percentile"],
                    voice_noise_window_frames=configuration[
                        "voice_noise_window_frames"
                    ],
                    voice_calibration_frames=configuration[
                        "voice_calibration_frames"
                    ],
                    voice_rms_ema_alpha=configuration["voice_rms_ema_alpha"],
                    voice_confirm_frames=configuration["voice_confirm_frames"],
                    state_transition_tap_guard_ms=configuration[
                        "state_transition_tap_guard_ms"
                    ],
                    decision_context_required=configuration[
                        "decision_context_required"
                    ],
                    reply_needed_threshold=configuration[
                        "reply_needed_threshold"
                    ],
                    decision_context_ready=not configuration[
                        "decision_context_required"
                    ],
                    warmup_feedbacks=configuration["warmup_feedbacks"],
                    warmup_update_scale=configuration["warmup_update_scale"],
                    network_update_schedule=configuration["network_update_schedule"],
                    personal_calibration_enabled=personal_enabled,
                    personal_logit_bias=(
                        float(calibration_state["personal_logit_bias"])
                        if calibration_state is not None
                        else 0.0
                    ),
                    personal_calibration_learning_rate=configuration[
                        "personal_learning_rate"
                    ],
                    personal_calibration_l2=configuration["personal_l2"],
                    personal_logit_bias_min=configuration["personal_bias_min"],
                    personal_logit_bias_max=configuration["personal_bias_max"],
                    personal_calibration_update_count=(
                        int(calibration_state["update_count"])
                        if calibration_state is not None
                        else 0
                    ),
                    personal_positive_updates=(
                        int(calibration_state["positive_updates"])
                        if calibration_state is not None
                        else 0
                    ),
                    personal_negative_updates=(
                        int(calibration_state["negative_updates"])
                        if calibration_state is not None
                        else 0
                    ),
                    tap_history=(
                        TapHistoryRuntime.load_or_create(
                            personal_model_dir,
                            configuration["tap_history_config"],
                        )
                        if bool(configuration["tap_history_config"].enabled)
                        else None
                    ),
                    tap_history_enabled=bool(
                        configuration["tap_history_config"].enabled
                    ),
                    opportunity_count=self.db.learning_opportunity_count(participant_id),
                    activation_count=feedback_counts["activation_opportunities"],
                    feedback_count=feedback_counts["eligible_feedbacks"],
                    positive_feedbacks=feedback_counts["positive_feedbacks"],
                    negative_feedbacks=feedback_counts["negative_feedbacks"],
                    opportunity_counter=lambda: self.db.next_learning_opportunity(participant_id),
                    activation_counter=lambda: self.db.next_activation_opportunity(participant_id),
                    feedback_counter=lambda label: self.db.next_eligible_feedback(
                        participant_id, label
                    ),
                    calibration_recorder=(
                        lambda expected, label, bias: self.db.update_calibration_state(
                            participant_id,
                            calibration_version=configuration["calibration_version"],
                            protocol_sha256=configuration["protocol_sha256"],
                            expected_update_count=expected,
                            personal_logit_bias=bias,
                            label=label,
                        )
                        if personal_enabled
                        else None
                    ),
                )
                if self.settings.agent_mode == "dtcn":
                    personal_dir = personal_model_dir
                    encoder_path = personal_dir / "latest_encoder.pth"
                    main_path = personal_dir / "latest_main_model.pth"
                    if not encoder_path.is_file() or participant["group_name"].lower() in {
                        "frozen",
                        "f",
                        "holdout",
                        "holdout_evaluator",
                    }:
                        if participant.get("prior_id"):
                            try:
                                encoder_path, main_path = prior_paths(self.db, participant)
                            except PriorError as exc:
                                raise ModelUnavailable(str(exc)) from exc
                        elif requires_locked_pipeline(participant):
                            raise ModelUnavailable("Pipeline-managed participant has no locked prior")
                        else:
                            encoder_path = Path(self.settings.base_encoder_checkpoint)
                            main_path = Path(self.settings.base_main_checkpoint)
                    runtime.model = DTCNModel(
                        self.settings, str(encoder_path), str(main_path)
                    )
                    runtime.model_identity = {
                        "encoder_sha256": (
                            file_sha256(Path(encoder_path))
                            if Path(encoder_path).is_file()
                            else None
                        ),
                        "main_sha256": (
                            file_sha256(Path(main_path))
                            if Path(main_path).is_file()
                            else None
                        ),
                    }
                    if (
                        participant.get("cohort") == "study_a"
                        and participant["group_name"].lower()
                        in {"donor", "seed_donor", "evaluator_donor"}
                        and participant.get("prior_id")
                    ):
                        shadow_dir = (
                            self.settings.data_dir / "donor-shadow" / participant_id / "latest"
                        )
                        shadow_encoder = shadow_dir / "latest_encoder.pth"
                        shadow_main = shadow_dir / "latest_main_model.pth"
                        if not shadow_encoder.is_file() or not shadow_main.is_file():
                            shadow_encoder = Path(self.settings.base_encoder_checkpoint)
                            shadow_main = Path(self.settings.base_main_checkpoint)
                        runtime.donor_shadow_model = DTCNModel(
                            self.settings, str(shadow_encoder), str(shadow_main)
                        )
                    runtime.snapshot_recorder = (
                        lambda source, budget, model, metadata: self._record_snapshot(
                            participant,
                            session_id,
                            source,
                            budget,
                            model,
                            metadata,
                            configuration["budget_kind"],
                        )
                    )
                    assignment, matched_checks = matched_assignment(
                        self.settings, participant_id, day, participant.get("run_id")
                    )
                    if assignment is not None:
                        runtime.matched_model = DTCNModel(
                            self.settings,
                            assignment["encoder_checkpoint"],
                            assignment["main_checkpoint"],
                        )
                    elif requires_locked_pipeline(participant) and participant.get("cohort") == "study_b":
                        missing = [name for name, passed in matched_checks.items() if not passed]
                        raise ModelUnavailable(
                            f"Locked matched-other assignment is not ready: {', '.join(missing)}"
                        )
                    elif participant.get("cohort") == "study_a":
                        runtime.matched_model = DTCNModel(
                            self.settings,
                            self.settings.base_encoder_checkpoint,
                            self.settings.base_main_checkpoint,
                        )
                    elif participant["cohort"] == "pilot":
                        matched_encoder = Path(self.settings.matched_encoder_checkpoint)
                        matched_main = Path(self.settings.matched_main_checkpoint)
                        if matched_encoder.is_file() and matched_main.is_file():
                            runtime.matched_model = DTCNModel(
                                self.settings, str(matched_encoder), str(matched_main)
                            )
                elif self.settings.agent_mode == "distilhubert":
                    # Public auto-orchestration supports cold-start and shared-start
                    # DistilHuBERT personal heads. Legacy locked DTCN Study B
                    # pipelines remain unavailable for this model family.
                    if requires_locked_pipeline(participant):
                        raise ModelUnavailable(
                            "DistilHuBERT locked study pipelines remain unavailable"
                        )
                    if participant.get("cohort") not in {"pilot", "study_a", "study_b"}:
                        raise ModelUnavailable(
                            "DistilHuBERT supports pilot/study_a/study_b public arms only"
                        )
                    manifest_path = Path(self.settings.distilhubert_release_manifest)
                    if self.distilhubert_feature_encoder is None:
                        self._ensure_distilhubert_encoder()
                    try:
                        from .distilhubert_online import (
                            DistilHuBERTOnlineModel,
                            DistilHuBERTUnavailable,
                        )
                        from .orchestration import seed_initial_head_from_prior

                        frozen = participant["group_name"].lower() in {
                            "frozen",
                            "f",
                            "holdout",
                            "holdout_evaluator",
                        }
                        personal_dir = personal_model_dir
                        pointer = personal_dir / "distilhubert-personal.json"
                        runtime.model = DistilHuBERTOnlineModel(
                            manifest_path,
                            update_enabled=not frozen,
                            personal_checkpoint_dir=(
                                personal_dir if pointer.is_file() else None
                            ),
                            feature_encoder=self.distilhubert_feature_encoder,
                        )
                        if (
                            not pointer.is_file()
                            and participant.get("prior_id")
                        ):
                            prior = self.db.shared_prior(str(participant["prior_id"]))
                            if prior is None:
                                raise ModelUnavailable("Assigned shared prior is missing")
                            seed_initial_head_from_prior(
                                runtime.model, Path(prior["main_path"])
                            )
                        # The comparison policy uses a separate frozen clone of
                        # the exact same shared head and cached frozen backbone.
                        runtime.matched_model = DistilHuBERTOnlineModel(
                            manifest_path,
                            update_enabled=False,
                            feature_encoder=self.distilhubert_feature_encoder,
                        )
                        runtime.model_identity = {
                            **runtime.model.identity(),
                            "comparison_policy": "shared_base_clone",
                            "matched_other_supported": False,
                            "shared_prior_id": participant.get("prior_id"),
                        }
                    except DistilHuBERTUnavailable as exc:
                        raise ModelUnavailable(str(exc)) from exc
                with self.lock:
                    self.runtimes[runtime_key] = runtime
            return runtime

    def select_candidate_prior(
        self,
        runtime: ParticipantRuntime,
        participant: dict[str, Any],
        day: int,
        probe_token: str,
    ) -> None:
        if self.settings.agent_mode != "dtcn":
            raise ModelUnavailable(
                "Candidate-prior probes are not implemented for this model family"
            )
        assignment = self.db.prior_probe_assignment(
            participant["id"], day, probe_token
        )
        if assignment is None:
            raise ModelUnavailable("Unknown or out-of-order candidate-prior probe token")
        if runtime.candidate_probe_token == probe_token and runtime.candidate_model is not None:
            return
        encoder_path = Path(assignment["encoder_path"])
        main_path = Path(assignment["main_path"])
        if (
            not encoder_path.is_file()
            or not main_path.is_file()
            or file_sha256(encoder_path) != assignment["encoder_sha256"]
            or file_sha256(main_path) != assignment["main_sha256"]
        ):
            raise ModelUnavailable("Candidate-prior checkpoint integrity check failed")
        model = DTCNModel(self.settings, str(encoder_path), str(main_path))
        with runtime.lock:
            runtime.candidate_model = model
            runtime.candidate_probe_token = probe_token

    def flush_pending(self, participant_id: str) -> None:
        with self.lock:
            runtimes = [
                runtime
                for key, runtime in self.runtimes.items()
                if key[0] == participant_id
            ]
        for runtime in runtimes:
            runtime.flush_pending_snapshots()

    def finalize_capped_snapshots(self, participant_id: str, day: int) -> None:
        with self.lock:
            runtime = self.runtimes.get((participant_id, day))
        if runtime is None:
            raise ModelUnavailable("Participant timing runtime is unavailable for cap finalization")
        runtime.finalize_capped_snapshots()

    @staticmethod
    def _expected_distilhubert_network_updates(
        configuration: dict[str, Any], eligible_feedbacks: int
    ) -> int:
        schedule = tuple(configuration.get("network_update_schedule") or ())
        warmup_feedbacks = int(configuration.get("warmup_feedbacks") or 0)
        warmup_scale = float(configuration.get("warmup_update_scale") or 0.0)
        expected = 0
        for completed_feedbacks in range(max(0, int(eligible_feedbacks))):
            scale: float | None = None
            for limit, candidate in schedule:
                if completed_feedbacks < int(limit):
                    scale = float(candidate)
                    break
            if scale is None:
                scale = (
                    1.0
                    if schedule or completed_feedbacks >= warmup_feedbacks
                    else warmup_scale
                )
            if scale > 0:
                expected += 1
        return expected

    def finalize_personal_checkpoint(
        self,
        participant: dict[str, Any],
        day: int,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Publish and independently validate the session-end personal head.

        A zero-network-update participant still receives a real checkpoint:
        its final head is the exact cold/shared starting head.  Reconstruction
        after a process restart is allowed only when the immutable protocol and
        server counters prove that every eligible feedback had network scale
        zero.  Otherwise a missing pointer fails closed instead of fabricating
        learned weights.
        """

        if self.settings.agent_mode != "distilhubert":
            raise ModelUnavailable(
                "Share-chain personal finalization requires DistilHuBERT"
            )
        participant_id = str(participant["id"])
        personal_dir = self._personal_model_dir(participant)
        pointer = personal_dir / "distilhubert-personal.json"
        configuration = self._runtime_configuration(participant)
        feedback_counts = self.db.feedback_counts(participant_id)
        eligible_feedbacks = int(feedback_counts.get("eligible_feedbacks") or 0)
        expected_updates = self._expected_distilhubert_network_updates(
            configuration, eligible_feedbacks
        )

        with self.lock:
            runtime = self.runtimes.get((participant_id, int(day)))
        if runtime is None and not pointer.is_file():
            if expected_updates > 0:
                raise ModelUnavailable(
                    "Personal checkpoint is missing after one or more network "
                    "updates; automatic reconstruction is unsafe"
                )
            # Safe recovery for the all-zero-scale prefix: runtime() recreates
            # the immutable cold/shared start and carries no learned network
            # state that could have been lost.
            runtime = self.runtime(
                participant, day=int(day), session_id=session_id
            )

        if runtime is not None:
            runtime.flush_pending_snapshots()
            from .distilhubert_online import DistilHuBERTOnlineModel

            if not isinstance(runtime.model, DistilHuBERTOnlineModel):
                raise ModelUnavailable(
                    "Share-chain runtime is not a DistilHuBERT online model"
                )
            runtime.model.save(personal_dir)

        try:
            from .distilhubert_online import validated_personal_checkpoint

            receipt = validated_personal_checkpoint(
                self.settings.distilhubert_release_manifest, personal_dir
            )
        except Exception as exc:
            raise ModelUnavailable(
                f"Personal checkpoint finalization failed: {exc}"
            ) from exc
        actual_updates = int(receipt.get("update_count") or 0)
        if actual_updates != expected_updates:
            raise ModelUnavailable(
                "Personal checkpoint update count does not match the "
                f"server-authoritative feedback schedule ({actual_updates} != "
                f"{expected_updates})"
            )
        return {
            **receipt,
            "participant_id": participant_id,
            "base_prior_id": participant.get("prior_id"),
            "eligible_feedbacks": eligible_feedbacks,
            "expected_network_updates": expected_updates,
            "zero_network_update": expected_updates == 0,
            "finalized_ms": int(time.time() * 1000),
        }

    def release(self, participant_id: str) -> None:
        self.flush_pending(participant_id)
        with self.lock:
            for key in [key for key in self.runtimes if key[0] == participant_id]:
                self.runtimes.pop(key, None)
