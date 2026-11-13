"""Fail-closed DistilHuBERT backend with an immutable shared linear head.

This module is intentionally optional.  Importing it does not import
``transformers`` or download model weights.  A real backbone is loaded only
after a versioned release manifest, the shared-head artifact, and every pinned
backbone file hash have been validated.

The online contract is:

* the DistilHuBERT backbone is always in ``eval`` mode and never trainable;
* ``shared_head`` is immutable for the lifetime of a release;
* ``personal_head`` starts as a byte-for-byte numerical clone of the shared
  head and is the only state changed by feedback in the learning condition;
* frozen-condition ``train`` calls fail closed without changing either head;
* personal checkpoints are versioned, hash-bound to the release manifest and
  shared head, and published atomically through a small JSON pointer.

The representation matches ``experiments/fewshot_ssl_benchmark.py``:
last-hidden-state mean and standard deviation over the whole 15 s clip and the
last 8/4/2/1/0.5/0.25 seconds (14 * 768 = 10,752 features).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .integrity import file_sha256


MODEL_FAMILY = "distilhubert-frozen-standardized-linear-head"
LEGACY_RELEASE_SCHEMA = "tap-to-adapt-distilhubert-release/v1"
RELEASE_SCHEMA = "tap-to-adapt-distilhubert-release/v2"
LEGACY_CHECKPOINT_SCHEMA = "tap-to-adapt-distilhubert-personal-head/v1"
CHECKPOINT_SCHEMA = "tap-to-adapt-distilhubert-personal-head/v2"
HEAD_FORMAT_VERSION = 1
MODEL_ID = "ntu-spml/distilhubert"
REPRESENTATION = "last-hidden.temporal-mean-std-recent-v1"
REQUIRED_BACKBONE_FILES = frozenset(
    {"config.json", "model.safetensors", "preprocessor_config.json"}
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class DistilHuBERTUnavailable(RuntimeError):
    """Raised when an optional SSL release cannot be loaded safely."""


class SharedEncoderBusy(RuntimeError):
    """Raised when another participant is using the process-wide backbone."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DistilHuBERTUnavailable(
            f"{context} fields do not match schema (missing={missing}, extra={extra})"
        )


def _sha256(value: Any, context: str) -> str:
    digest = str(value or "").lower()
    if SHA256_RE.fullmatch(digest) is None:
        raise DistilHuBERTUnavailable(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _safe_relative_path(root: Path, value: Any, context: str) -> Path:
    relative = Path(str(value or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise DistilHuBERTUnavailable(f"{context} must be a contained relative path")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DistilHuBERTUnavailable(f"{context} escapes the release directory") from exc
    return resolved


@dataclass(frozen=True)
class HeadOptimizer:
    learning_rate: float
    l2: float
    proximal_to_shared: float
    max_gradient_norm: float
    steps_per_feedback: int
    replay_per_class: int
    replay_batch_size: int
    replay_seed: int
    require_both_replay_classes: bool
    single_class_update_scale: float


@dataclass(frozen=True)
class DistilHuBERTRelease:
    path: Path
    manifest_sha256: str
    model_id: str
    revision: str
    sample_rate: int
    clip_seconds: float
    hidden_size: int
    feature_dim: int
    shared_head_path: Path
    shared_head_sha256: str
    replay_seed_path: Path | None
    replay_seed_sha256: str | None
    backbone_files: dict[str, str]
    optimizer: HeadOptimizer
    training_provenance: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "DistilHuBERTRelease":
        manifest_path = Path(path).resolve()
        if not manifest_path.is_file():
            raise DistilHuBERTUnavailable(
                f"DistilHuBERT release manifest is missing: {manifest_path}"
            )
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DistilHuBERTUnavailable("DistilHuBERT release manifest is invalid JSON") from exc
        if not isinstance(raw, dict):
            raise DistilHuBERTUnavailable("DistilHuBERT release manifest must be an object")
        optional_release_keys = {"replay_seed"} & set(raw)
        _exact_keys(
            raw,
            {
                "schema",
                "model_family",
                "model_id",
                "revision",
                "sample_rate",
                "clip_seconds",
                "representation",
                "hidden_size",
                "feature_dim",
                "backbone_files",
                "shared_head",
                "optimizer",
                "training_provenance",
            }
            | optional_release_keys,
            "release manifest",
        )
        schema = str(raw["schema"])
        if schema not in {LEGACY_RELEASE_SCHEMA, RELEASE_SCHEMA} or raw["model_family"] != MODEL_FAMILY:
            raise DistilHuBERTUnavailable("Unsupported DistilHuBERT release schema/family")
        if raw["model_id"] != MODEL_ID:
            raise DistilHuBERTUnavailable("Release does not pin ntu-spml/distilhubert")
        revision = str(raw["revision"]).lower()
        if REVISION_RE.fullmatch(revision) is None:
            raise DistilHuBERTUnavailable("Backbone revision must be a full 40-character commit")
        if raw["representation"] != REPRESENTATION:
            raise DistilHuBERTUnavailable("Unsupported DistilHuBERT representation")
        sample_rate = int(raw["sample_rate"])
        clip_seconds = float(raw["clip_seconds"])
        hidden_size = int(raw["hidden_size"])
        feature_dim = int(raw["feature_dim"])
        if (
            sample_rate != 16_000
            or clip_seconds != 15.0
            or hidden_size != 768
            or feature_dim != 14 * hidden_size
        ):
            raise DistilHuBERTUnavailable("Release dimensions do not match the screened representation")

        backbone = raw["backbone_files"]
        if not isinstance(backbone, dict) or set(backbone) != REQUIRED_BACKBONE_FILES:
            raise DistilHuBERTUnavailable(
                "backbone_files must hash config.json, model.safetensors, and preprocessor_config.json"
            )
        backbone_files = {
            str(name): _sha256(value, f"backbone_files.{name}")
            for name, value in backbone.items()
        }

        shared = raw["shared_head"]
        if not isinstance(shared, dict):
            raise DistilHuBERTUnavailable("shared_head must be an object")
        _exact_keys(shared, {"path", "sha256", "format"}, "shared_head")
        if shared["format"] != "numpy-npz-no-pickle/v1":
            raise DistilHuBERTUnavailable("Unsupported shared-head format")
        shared_path = _safe_relative_path(manifest_path.parent, shared["path"], "shared_head.path")
        shared_hash = _sha256(shared["sha256"], "shared_head.sha256")
        if not shared_path.is_file() or file_sha256(shared_path) != shared_hash:
            raise DistilHuBERTUnavailable("Shared-head artifact is missing or has the wrong hash")
        replay_seed_path = None
        replay_seed_hash = None
        if "replay_seed" in raw:
            replay_seed = raw["replay_seed"]
            if not isinstance(replay_seed, dict):
                raise DistilHuBERTUnavailable("replay_seed must be an object")
            _exact_keys(replay_seed, {"path", "sha256", "format"}, "replay_seed")
            if replay_seed["format"] != "standardized-replay-seed-npz/v1":
                raise DistilHuBERTUnavailable("Unsupported replay-seed format")
            replay_seed_path = _safe_relative_path(
                manifest_path.parent, replay_seed["path"], "replay_seed.path"
            )
            replay_seed_hash = _sha256(
                replay_seed["sha256"], "replay_seed.sha256"
            )
            if (
                not replay_seed_path.is_file()
                or file_sha256(replay_seed_path) != replay_seed_hash
            ):
                raise DistilHuBERTUnavailable(
                    "Replay-seed artifact is missing or has the wrong hash"
                )

        optimizer = raw["optimizer"]
        if not isinstance(optimizer, dict):
            raise DistilHuBERTUnavailable("optimizer must be an object")
        legacy_optimizer_keys = {
            "learning_rate",
            "l2",
            "proximal_to_shared",
            "max_gradient_norm",
            "steps_per_feedback",
        }
        replay_optimizer_keys = legacy_optimizer_keys | {
            "replay_per_class",
            "replay_batch_size",
            "replay_seed",
        }
        replay_optional_policy_keys = {
            "require_both_replay_classes",
            "single_class_update_scale",
        }
        _exact_keys(
            optimizer,
            legacy_optimizer_keys
            if schema == LEGACY_RELEASE_SCHEMA
            else (
                replay_optimizer_keys
                | (replay_optional_policy_keys & set(optimizer))
            ),
            "optimizer",
        )
        parsed_optimizer = HeadOptimizer(
            learning_rate=float(optimizer["learning_rate"]),
            l2=float(optimizer["l2"]),
            proximal_to_shared=float(optimizer["proximal_to_shared"]),
            max_gradient_norm=float(optimizer["max_gradient_norm"]),
            steps_per_feedback=int(optimizer["steps_per_feedback"]),
            replay_per_class=int(optimizer.get("replay_per_class", 0)),
            replay_batch_size=int(optimizer.get("replay_batch_size", 0)),
            replay_seed=int(optimizer.get("replay_seed", 0)),
            require_both_replay_classes=bool(
                optimizer.get("require_both_replay_classes", False)
            ),
            single_class_update_scale=float(
                optimizer.get("single_class_update_scale", 1.0)
            ),
        )
        if (
            not 0 < parsed_optimizer.learning_rate <= 1
            or not 0 <= parsed_optimizer.l2 <= 10
            or not 0 <= parsed_optimizer.proximal_to_shared <= 10
            or not 0 < parsed_optimizer.max_gradient_norm <= 1_000
            or not 1 <= parsed_optimizer.steps_per_feedback <= 32
            or not 0 <= parsed_optimizer.replay_per_class <= 256
            or not 0 <= parsed_optimizer.replay_batch_size <= 512
            or parsed_optimizer.replay_batch_size
            > (
                4 * parsed_optimizer.replay_per_class
                if replay_seed_path is not None
                else 2 * parsed_optimizer.replay_per_class
            )
            or not 0 <= parsed_optimizer.replay_seed <= 2**31 - 1
            or not 0 < parsed_optimizer.single_class_update_scale <= 1
        ):
            raise DistilHuBERTUnavailable("Head optimizer settings are outside guarded limits")
        if schema == RELEASE_SCHEMA and (
            parsed_optimizer.replay_per_class < 1
            or parsed_optimizer.replay_batch_size < 2
        ):
            raise DistilHuBERTUnavailable("Replay releases require a non-empty class-balanced buffer")

        provenance = raw["training_provenance"]
        if not isinstance(provenance, dict):
            raise DistilHuBERTUnavailable("training_provenance must be an object")
        _exact_keys(
            provenance,
            {
                "source",
                "source_manifest_sha256",
                "feature_cache_sha256",
                "head_fitter",
                "regularization_c",
                "fit_examples",
                "positive_examples",
                "negative_examples",
                "participant_disjoint",
                "development_only",
            },
            "training_provenance",
        )
        _sha256(provenance["source_manifest_sha256"], "source_manifest_sha256")
        _sha256(provenance["feature_cache_sha256"], "feature_cache_sha256")
        fit_examples = int(provenance["fit_examples"])
        positive_examples = int(provenance["positive_examples"])
        negative_examples = int(provenance["negative_examples"])
        if (
            not str(provenance["source"]).strip()
            or provenance["head_fitter"] != "standard-scaler-liblinear-logistic-v1"
            or float(provenance["regularization_c"]) <= 0
            or fit_examples <= 0
            or positive_examples <= 0
            or negative_examples <= 0
            or positive_examples + negative_examples != fit_examples
            or not isinstance(provenance["participant_disjoint"], bool)
            or not isinstance(provenance["development_only"], bool)
        ):
            raise DistilHuBERTUnavailable("training_provenance is internally inconsistent")
        if not provenance["development_only"] and not provenance["participant_disjoint"]:
            raise DistilHuBERTUnavailable(
                "Non-development releases must be trained on participant-disjoint data"
            )

        return cls(
            path=manifest_path,
            manifest_sha256=file_sha256(manifest_path),
            model_id=MODEL_ID,
            revision=revision,
            sample_rate=sample_rate,
            clip_seconds=clip_seconds,
            hidden_size=hidden_size,
            feature_dim=feature_dim,
            shared_head_path=shared_path,
            shared_head_sha256=shared_hash,
            replay_seed_path=replay_seed_path,
            replay_seed_sha256=replay_seed_hash,
            backbone_files=backbone_files,
            optimizer=parsed_optimizer,
            training_provenance=dict(provenance),
        )


@dataclass
class LinearHeadState:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    bias: float

    def clone(self) -> "LinearHeadState":
        return LinearHeadState(
            feature_mean=self.feature_mean.copy(),
            feature_scale=self.feature_scale.copy(),
            weight=self.weight.copy(),
            bias=float(self.bias),
        )


def _validated_head(path: Path, feature_dim: int) -> LinearHeadState:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {
                "format_version",
                "feature_mean",
                "feature_scale",
                "weight",
                "bias",
            }:
                raise DistilHuBERTUnavailable("Head artifact fields do not match schema")
            version = int(np.asarray(payload["format_version"]).reshape(()))
            state = LinearHeadState(
                feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32).copy(),
                feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32).copy(),
                weight=np.asarray(payload["weight"], dtype=np.float32).copy(),
                bias=float(np.asarray(payload["bias"], dtype=np.float32).reshape(())),
            )
    except DistilHuBERTUnavailable:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise DistilHuBERTUnavailable(f"Cannot read head artifact: {path}") from exc
    if version != HEAD_FORMAT_VERSION:
        raise DistilHuBERTUnavailable("Unsupported head artifact version")
    expected = (feature_dim,)
    if any(array.shape != expected for array in (state.feature_mean, state.feature_scale, state.weight)):
        raise DistilHuBERTUnavailable("Head artifact feature dimensions do not match release")
    if (
        not np.isfinite(state.feature_mean).all()
        or not np.isfinite(state.feature_scale).all()
        or not np.isfinite(state.weight).all()
        or not math.isfinite(state.bias)
        or np.any(state.feature_scale <= 0)
    ):
        raise DistilHuBERTUnavailable("Head artifact contains invalid numeric values")
    return state


def write_head_artifact(path: str | Path, state: LinearHeadState) -> Path:
    """Write a safe, non-pickle head artifact atomically.

    Release tooling can use this after fitting a StandardScaler + logistic
    regression probe.  The release manifest must still bind the resulting file
    hash before runtime loading.
    """

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                format_version=np.asarray(HEAD_FORMAT_VERSION, dtype=np.int64),
                feature_mean=np.asarray(state.feature_mean, dtype=np.float32),
                feature_scale=np.asarray(state.feature_scale, dtype=np.float32),
                weight=np.asarray(state.weight, dtype=np.float32),
                bias=np.asarray(state.bias, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
        # Validate before publishing; infer dimension from the supplied state.
        _validated_head(temporary, int(np.asarray(state.weight).size))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _head_fingerprint(state: LinearHeadState) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("feature_mean", state.feature_mean),
        ("feature_scale", state.feature_scale),
        ("weight", state.weight),
        ("bias", np.asarray([state.bias], dtype=np.float32)),
    ):
        array = np.ascontiguousarray(value, dtype=np.float32)
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _window_slices(length: int, frames_per_second: float) -> list[slice]:
    windows = [slice(0, length)]
    for seconds in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25):
        frames = max(1, min(length, int(round(seconds * frames_per_second))))
        windows.append(slice(length - frames, length))
    return windows


def _temporal_pool(hidden: Any, clip_seconds: float, torch: Any) -> np.ndarray:
    hidden = hidden.float()
    if hidden.ndim != 2 or hidden.shape[0] <= 0:
        raise DistilHuBERTUnavailable("Backbone returned an invalid hidden-state tensor")
    length = int(hidden.shape[0])
    chunks = []
    for frame_slice in _window_slices(length, length / clip_seconds):
        part = hidden[frame_slice]
        chunks.extend((part.mean(dim=0), part.std(dim=0, unbiased=False)))
    return torch.cat(chunks).detach().cpu().numpy().astype(np.float32, copy=False)


class DistilHuBERTFeatureEncoder:
    """Pinned local-cache-only Hugging Face feature encoder by default."""

    def __init__(
        self,
        release: DistilHuBERTRelease,
        *,
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
        device: str | None = None,
        replica_count: int = 1,
        devices: tuple[str, ...] | list[str] | None = None,
    ):
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise DistilHuBERTUnavailable(
                "DistilHuBERT mode requires torch, transformers, safetensors, and huggingface_hub"
            ) from exc

        self.torch = torch
        self.release = release
        device_list = tuple(
            str(item).strip()
            for item in (devices or ((device,) if device else ()))
            if str(item).strip()
        )
        self.device = device_list[0] if device_list else (
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not device_list:
            device_list = (self.device,)
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=release.model_id,
                    revision=release.revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=not allow_download,
                    allow_patterns=sorted(REQUIRED_BACKBONE_FILES),
                )
            ).resolve()
        except Exception as exc:
            mode = "download-enabled" if allow_download else "local-cache-only"
            raise DistilHuBERTUnavailable(
                f"Pinned DistilHuBERT snapshot is unavailable in {mode} mode"
            ) from exc
        for name, expected_hash in release.backbone_files.items():
            artifact = snapshot / name
            if not artifact.is_file() or file_sha256(artifact) != expected_hash:
                raise DistilHuBERTUnavailable(
                    f"Pinned backbone artifact is missing or changed: {name}"
                )
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(
                snapshot,
                trust_remote_code=False,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                snapshot,
                trust_remote_code=False,
                use_safetensors=True,
                local_files_only=True,
            ).to(self.device)
        except Exception as exc:
            raise DistilHuBERTUnavailable("Pinned DistilHuBERT snapshot cannot be loaded") from exc
        if (
            int(getattr(self.extractor, "sampling_rate", 0)) != release.sample_rate
            or int(getattr(self.model.config, "hidden_size", 0)) != release.hidden_size
        ):
            raise DistilHuBERTUnavailable("Loaded backbone dimensions differ from the release manifest")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise DistilHuBERTUnavailable("Backbone freezing failed")
        self._replicas = [self.model]
        self._replica_devices = [self.device]
        self._available: queue.Queue[int] = queue.Queue()
        self._available.put(0)
        for index in range(1, max(1, int(replica_count or 1))):
            replica_device = device_list[index % len(device_list)]
            extra = AutoModel.from_pretrained(
                snapshot,
                trust_remote_code=False,
                use_safetensors=True,
                local_files_only=True,
            ).to(replica_device)
            extra.eval()
            for parameter in extra.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in extra.parameters()):
                raise DistilHuBERTUnavailable("Backbone freezing failed")
            self._available.put(len(self._replicas))
            self._replicas.append(extra)
            self._replica_devices.append(replica_device)
        self.lock = threading.Lock()

    def __call__(self, audio: np.ndarray, *, timeout: float | None = None) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1 or not np.isfinite(waveform).all():
            raise DistilHuBERTUnavailable("Audio must be a finite mono float32 waveform")
        target_samples = int(self.release.sample_rate * self.release.clip_seconds)
        if waveform.size > target_samples:
            waveform = waveform[-target_samples:]
        elif waveform.size < target_samples:
            waveform = np.pad(waveform, (target_samples - waveform.size, 0))
        try:
            if timeout is None:
                replica_index = self._available.get()
            else:
                replica_index = self._available.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            raise SharedEncoderBusy("shared DistilHuBERT encoder is busy") from None
        model = self._replicas[replica_index]
        replica_device = self._replica_devices[replica_index]
        try:
            inputs = self.extractor(
                waveform,
                sampling_rate=self.release.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(replica_device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(replica_device)
            model.eval()
            amp_enabled = str(replica_device).startswith("cuda")
            with self.torch.inference_mode(), self.torch.autocast(
                device_type="cuda", dtype=self.torch.float16, enabled=amp_enabled
            ):
                outputs = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                )
            features = _temporal_pool(
                outputs.last_hidden_state[0], self.release.clip_seconds, self.torch
            )
        finally:
            self._available.put(replica_index)
        if features.shape != (self.release.feature_dim,) or not np.isfinite(features).all():
            raise DistilHuBERTUnavailable("Backbone produced incompatible pooled features")
        return features


@dataclass(frozen=True)
class HeadPrediction:
    shared_probability: float
    shared_logit: float
    personal_probability: float
    personal_logit: float
    shared_head_fingerprint: str
    personal_head_fingerprint: str
    update_count: int


class DistilHuBERTOnlineModel:
    """Timing-model adapter for frozen DistilHuBERT + online personal head."""

    supports_update_scale = True
    _encoder_cache: dict[tuple, DistilHuBERTFeatureEncoder] = {}
    _encoder_cache_lock = threading.Lock()

    @classmethod
    def preload_feature_encoder(
        cls,
        manifest_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
        device: str | None = None,
        replica_count: int = 1,
        devices: tuple[str, ...] | list[str] | None = None,
    ) -> DistilHuBERTFeatureEncoder:
        """Load and validate the heavy backbone before serving WebSockets."""

        release = DistilHuBERTRelease.load(manifest_path)
        replicas = max(1, int(replica_count or 1))
        device_key = tuple(
            str(item).strip()
            for item in (devices or ((device,) if device else ()))
            if str(item).strip()
        )
        cache_key = (
            release.manifest_sha256,
            str(Path(cache_dir).resolve()) if cache_dir else "",
            str(device or "auto"),
            release.revision,
            bool(allow_download),
            replicas,
            device_key,
        )
        with cls._encoder_cache_lock:
            encoder = cls._encoder_cache.get(cache_key)
            if encoder is None:
                encoder = DistilHuBERTFeatureEncoder(
                    release,
                    cache_dir=cache_dir,
                    allow_download=allow_download,
                    device=device,
                    replica_count=replicas,
                    devices=device_key or None,
                )
                cls._encoder_cache[cache_key] = encoder
        return encoder

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        update_enabled: bool,
        personal_checkpoint_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        allow_download: bool = False,
        device: str | None = None,
        feature_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.release = DistilHuBERTRelease.load(manifest_path)
        shared = _validated_head(self.release.shared_head_path, self.release.feature_dim)
        # Shared arrays are immutable even if a future code path receives a reference.
        for array in (shared.feature_mean, shared.feature_scale, shared.weight):
            array.setflags(write=False)
        self.shared_head = shared
        self.personal_head = shared.clone()
        for array in (self.personal_head.feature_mean, self.personal_head.feature_scale):
            array.setflags(write=False)
        self.update_enabled = bool(update_enabled)
        self.update_count = 0
        self.optimizer_step_count = 0
        self.lock = threading.Lock()
        self._last_audio_digest: str | None = None
        self._last_features: np.ndarray | None = None
        self._replay_features: dict[int, list[np.ndarray]] = {0: [], 1: []}
        self._shared_replay_features: dict[int, list[np.ndarray]] = {0: [], 1: []}
        self._replay_seen: dict[int, int] = {0: 0, 1: 0}
        if self.release.replay_seed_path is not None:
            try:
                with np.load(
                    self.release.replay_seed_path, allow_pickle=False
                ) as replay_seed:
                    if set(replay_seed.files) != {
                        "format_version",
                        "standardized_features",
                        "labels",
                    }:
                        raise DistilHuBERTUnavailable(
                            "Replay-seed fields do not match schema"
                        )
                    if int(np.asarray(replay_seed["format_version"]).reshape(())) != 1:
                        raise DistilHuBERTUnavailable(
                            "Unsupported replay-seed artifact version"
                        )
                    seed_features = np.asarray(
                        replay_seed["standardized_features"], dtype=np.float32
                    )
                    seed_labels = np.asarray(replay_seed["labels"], dtype=np.int64)
            except DistilHuBERTUnavailable:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise DistilHuBERTUnavailable(
                    "Cannot read replay-seed artifact"
                ) from exc
            if (
                seed_features.ndim != 2
                or seed_features.shape[1] != self.release.feature_dim
                or seed_labels.shape != (seed_features.shape[0],)
                or not np.isfinite(seed_features).all()
                or not set(seed_labels.tolist()) <= {0, 1}
            ):
                raise DistilHuBERTUnavailable(
                    "Replay-seed artifact is internally inconsistent"
                )
            for replay_label in (0, 1):
                selected = seed_features[seed_labels == replay_label]
                if not 1 <= len(selected) <= self.release.optimizer.replay_per_class:
                    raise DistilHuBERTUnavailable(
                        "Replay seed must provide each class within buffer capacity"
                    )
                self._shared_replay_features[replay_label] = [
                    row.copy() for row in selected
                ]
        self.last_prediction: HeadPrediction | None = None
        self.last_update_applied = False
        self.last_update_reason: str | None = None
        self.last_optimizer_step_before = 0
        self.last_optimizer_step_after = 0
        self.last_requested_training_steps = 0
        self.last_training_steps = 0
        self.last_optimizer_learning_rates_before = [
            self.release.optimizer.learning_rate
        ]
        self.last_optimizer_learning_rates_during = [
            self.release.optimizer.learning_rate
        ]
        self.last_optimizer_learning_rates_after = [
            self.release.optimizer.learning_rate
        ]
        self.last_parameter_state_interpolated = False
        self.last_parameter_interpolation_scale = 1.0

        if personal_checkpoint_dir is not None:
            # Frozen arms may still restore a shared-start or resumed head; train()
            # remains fail-closed when update_enabled is false.
            self._load_personal_checkpoint(Path(personal_checkpoint_dir))

        self._locked_shared_fingerprint = _head_fingerprint(self.shared_head)
        self._assert_head_invariants()

        if feature_encoder is not None:
            if (
                isinstance(feature_encoder, DistilHuBERTFeatureEncoder)
                and feature_encoder.release.manifest_sha256
                != self.release.manifest_sha256
            ):
                raise DistilHuBERTUnavailable(
                    "Preloaded backbone belongs to a different release manifest"
                )
            self.feature_encoder = feature_encoder
        else:
            self.feature_encoder = self.preload_feature_encoder(
                self.release.path,
                cache_dir=cache_dir,
                allow_download=allow_download,
                device=device,
            )

    def can_train(self, audio: np.ndarray) -> bool:
        waveform = np.asarray(audio)
        return bool(
            waveform.ndim == 1
            and waveform.size >= int(0.25 * self.release.sample_rate)
            and np.isfinite(waveform).all()
        )

    def _assert_head_invariants(self) -> None:
        locked = getattr(self, "_locked_shared_fingerprint", None)
        if locked is not None and _head_fingerprint(self.shared_head) != locked:
            raise DistilHuBERTUnavailable("Immutable shared head changed in memory")
        if (
            not np.array_equal(
                self.personal_head.feature_mean, self.shared_head.feature_mean
            )
            or not np.array_equal(
                self.personal_head.feature_scale, self.shared_head.feature_scale
            )
        ):
            raise DistilHuBERTUnavailable("Personal head changed the locked feature scaler")
        # Frozen arms may start from a shared-prior personal head (S_sat) that
        # already diverges from the release shared head; updates remain disabled.

    def _features(self, audio: np.ndarray, *, timeout: float | None = None) -> np.ndarray:
        waveform = np.ascontiguousarray(audio, dtype=np.float32)
        digest = hashlib.sha256(waveform.tobytes()).hexdigest()
        if digest == self._last_audio_digest and self._last_features is not None:
            return self._last_features.copy()
        if isinstance(self.feature_encoder, DistilHuBERTFeatureEncoder):
            raw_features = self.feature_encoder(waveform, timeout=timeout)
        else:
            raw_features = self.feature_encoder(waveform)
        features = np.asarray(raw_features, dtype=np.float32)
        if features.shape != (self.release.feature_dim,) or not np.isfinite(features).all():
            raise DistilHuBERTUnavailable("Feature encoder returned an invalid vector")
        self._last_audio_digest = digest
        self._last_features = features.copy()
        return features

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Public backbone pooled features for tap-timing acoustic memory."""

        with self.lock:
            return self._features(audio)

    def _standardized(self, features: np.ndarray) -> np.ndarray:
        value = (features - self.shared_head.feature_mean) / self.shared_head.feature_scale
        if not np.isfinite(value).all():
            raise DistilHuBERTUnavailable("Standardized SSL features are not finite")
        return value.astype(np.float32, copy=False)

    def _replay_add(self, standardized: np.ndarray, label: int) -> None:
        capacity = self.release.optimizer.replay_per_class
        if capacity <= 0:
            return
        self._replay_seen[label] += 1
        seen = self._replay_seen[label]
        bucket = self._replay_features[label]
        value = np.asarray(standardized, dtype=np.float32).copy()
        if len(bucket) < capacity:
            bucket.append(value)
            return
        digest = hashlib.sha256(
            value.tobytes()
            + int(label).to_bytes(1, "little")
            + int(seen).to_bytes(8, "little")
            + int(self.release.optimizer.replay_seed).to_bytes(8, "little")
        ).digest()
        candidate = int.from_bytes(digest[:8], "little") % seen
        if candidate < capacity:
            bucket[candidate] = value

    def _replay_batch(self) -> tuple[np.ndarray, np.ndarray]:
        optimizer = self.release.optimizer
        if optimizer.replay_per_class <= 0:
            raise DistilHuBERTUnavailable("Replay batch requested by a legacy non-replay release")
        per_class_limit = max(1, optimizer.replay_batch_size // 2)
        features: list[np.ndarray] = []
        labels: list[int] = []
        for label in (0, 1):
            bucket = self._replay_features[label]
            anchors = self._shared_replay_features[label]
            remaining = max(0, per_class_limit - len(anchors))
            selected = anchors + (bucket[-remaining:] if remaining else [])
            features.extend(selected)
            labels.extend([label] * len(selected))
        if not features:
            raise DistilHuBERTUnavailable("Replay buffer is empty after an eligible feedback")
        return (
            np.asarray(features, dtype=np.float64),
            np.asarray(labels, dtype=np.int64),
        )

    @staticmethod
    def _head_logit(head: LinearHeadState, standardized: np.ndarray) -> float:
        return float(np.dot(head.weight.astype(np.float64), standardized.astype(np.float64)) + head.bias)

    def predict_both(
        self, audio: np.ndarray, *, encoder_timeout: float | None = None
    ) -> HeadPrediction:
        with self.lock:
            self._assert_head_invariants()
            standardized = self._standardized(
                self._features(audio, timeout=encoder_timeout)
            )
            shared_logit = self._head_logit(self.shared_head, standardized)
            personal_logit = self._head_logit(self.personal_head, standardized)
            prediction = HeadPrediction(
                shared_probability=_sigmoid(shared_logit),
                shared_logit=shared_logit,
                personal_probability=_sigmoid(personal_logit),
                personal_logit=personal_logit,
                shared_head_fingerprint=_head_fingerprint(self.shared_head),
                personal_head_fingerprint=_head_fingerprint(self.personal_head),
                update_count=self.update_count,
            )
            self.last_prediction = prediction
            return prediction

    def predict(
        self, audio: np.ndarray, *, encoder_timeout: float | None = None
    ) -> tuple[float, float]:
        prediction = self.predict_both(audio, encoder_timeout=encoder_timeout)
        return prediction.personal_probability, prediction.personal_logit

    def train(
        self,
        audio: np.ndarray,
        label: int,
        weight: float = 1.0,
        update_scale: float = 1.0,
    ) -> float:
        with self.lock:
            self._assert_head_invariants()
            self.last_update_applied = False
            self.last_update_reason = None
            self.last_optimizer_step_before = self.optimizer_step_count
            self.last_optimizer_step_after = self.optimizer_step_count
            self.last_training_steps = 0
            self.last_requested_training_steps = 0
            self.last_parameter_state_interpolated = False
            self.last_parameter_interpolation_scale = float(update_scale)
            if not self.update_enabled:
                self.last_update_reason = "frozen_condition"
                return 0.0
            if label not in (0, 1):
                self.last_update_reason = "invalid_label"
                return 0.0
            if not 0 < float(weight) <= 10 or not 0 < float(update_scale) <= 1:
                self.last_update_reason = "invalid_update_weight_or_scale"
                return 0.0
            if not self.can_train(audio):
                self.last_update_reason = "training_audio_too_short_or_invalid"
                return 0.0

            standardized = self._standardized(self._features(audio))
            self._replay_add(standardized, label)
            effective_update_scale = float(update_scale)
            if (
                not self._replay_features[0]
                or not self._replay_features[1]
            ):
                if self.release.optimizer.require_both_replay_classes:
                    self.last_update_reason = "awaiting_both_replay_classes"
                    return 0.0
                effective_update_scale = min(
                    effective_update_scale,
                    self.release.optimizer.single_class_update_scale,
                )
            self.last_parameter_interpolation_scale = effective_update_scale
            if self.release.optimizer.replay_per_class > 0:
                replay_features, replay_labels = self._replay_batch()
            else:
                replay_features = standardized.astype(np.float64, copy=False)[None, :]
                replay_labels = np.asarray([label], dtype=np.int64)
            personal_weight = self.personal_head.weight.astype(np.float64).copy()
            personal_bias = float(self.personal_head.bias)
            shared_weight = self.shared_head.weight.astype(np.float64)
            shared_bias = float(self.shared_head.bias)
            optimizer = self.release.optimizer
            learning_rate = optimizer.learning_rate * effective_update_scale
            self.last_optimizer_learning_rates_before = [optimizer.learning_rate]
            self.last_optimizer_learning_rates_during = [learning_rate]
            self.last_optimizer_learning_rates_after = [optimizer.learning_rate]
            requested_steps = max(
                1,
                int(
                    round(
                        optimizer.steps_per_feedback * effective_update_scale
                    )
                ),
            )
            self.last_requested_training_steps = requested_steps
            state_before = (personal_weight.copy(), personal_bias)
            last_loss = 0.0
            for _ in range(requested_steps):
                logits = replay_features @ personal_weight + personal_bias
                probabilities = np.asarray([_sigmoid(float(value)) for value in logits])
                losses = -(
                    replay_labels * np.log(np.maximum(probabilities, 1e-8))
                    + (1 - replay_labels) * np.log(np.maximum(1.0 - probabilities, 1e-8))
                )
                class_gradients: list[np.ndarray] = []
                class_bias_gradients: list[float] = []
                class_losses: list[float] = []
                for replay_label in (0, 1):
                    mask = replay_labels == replay_label
                    if not mask.any():
                        continue
                    class_weight = float(weight) if replay_label == label else 1.0
                    errors = class_weight * (
                        probabilities[mask] - replay_labels[mask].astype(np.float64)
                    )
                    class_gradients.append(
                        np.mean(errors[:, None] * replay_features[mask], axis=0)
                    )
                    class_bias_gradients.append(float(np.mean(errors)))
                    class_losses.append(float(np.mean(losses[mask])))
                error_gradient = np.mean(np.asarray(class_gradients), axis=0)
                error_bias_gradient = float(np.mean(class_bias_gradients))
                last_loss = float(np.mean(class_losses))
                weight_gradient = (
                    error_gradient
                    + optimizer.l2 * personal_weight
                    + optimizer.proximal_to_shared * (personal_weight - shared_weight)
                )
                bias_gradient = (
                    error_bias_gradient
                    + optimizer.l2 * personal_bias
                    + optimizer.proximal_to_shared * (personal_bias - shared_bias)
                )
                gradient_norm = math.sqrt(
                    float(np.dot(weight_gradient, weight_gradient)) + bias_gradient**2
                )
                if not math.isfinite(gradient_norm):
                    self.last_update_reason = "non_finite_head_gradient"
                    return 0.0
                gradient_scale = min(
                    1.0, optimizer.max_gradient_norm / max(gradient_norm, 1e-12)
                )
                personal_weight -= learning_rate * gradient_scale * weight_gradient
                personal_bias -= learning_rate * gradient_scale * bias_gradient
                if not np.isfinite(personal_weight).all() or not math.isfinite(personal_bias):
                    self.last_update_reason = "non_finite_personal_head"
                    return 0.0

            if effective_update_scale < 1.0:
                # Match the locked DTCN schedule contract: scaled optimizer
                # steps/LR are additionally interpolated back toward the
                # pre-update state before publication.
                before_weight, before_bias = state_before
                personal_weight = before_weight + (
                    personal_weight - before_weight
                ) * effective_update_scale
                personal_bias = before_bias + (
                    personal_bias - before_bias
                ) * effective_update_scale
                self.last_parameter_state_interpolated = True

            self.personal_head.weight = personal_weight.astype(np.float32)
            self.personal_head.bias = float(np.float32(personal_bias))
            self.update_count += 1
            self.optimizer_step_count += requested_steps
            self.last_optimizer_step_after = self.optimizer_step_count
            self.last_training_steps = requested_steps
            self.last_update_applied = True
            self.last_prediction = None
            return float(last_loss)

    def audit_state(self) -> dict[str, Any]:
        self._assert_head_invariants()
        prediction = self.last_prediction
        return {
            "model_family": MODEL_FAMILY,
            "revision": self.release.revision,
            "release_manifest_sha256": self.release.manifest_sha256,
            "shared_head_artifact_sha256": self.release.shared_head_sha256,
            "shared_head_fingerprint": _head_fingerprint(self.shared_head),
            "personal_head_fingerprint": _head_fingerprint(self.personal_head),
            "heads_diverged": _head_fingerprint(self.shared_head)
            != _head_fingerprint(self.personal_head),
            "update_enabled": self.update_enabled,
            "update_count": self.update_count,
            "development_only": bool(
                self.release.training_provenance["development_only"]
            ),
            "participant_disjoint_training": bool(
                self.release.training_provenance["participant_disjoint"]
            ),
            "replay_per_class": self.release.optimizer.replay_per_class,
            "replay_counts": {
                "negative": len(self._replay_features[0]),
                "positive": len(self._replay_features[1]),
            },
            "replay_seen": {
                "negative": self._replay_seen[0],
                "positive": self._replay_seen[1],
            },
            "shared_probability": (
                prediction.shared_probability if prediction is not None else None
            ),
            "shared_logit": prediction.shared_logit if prediction is not None else None,
            "personal_probability": (
                prediction.personal_probability if prediction is not None else None
            ),
            "personal_logit": prediction.personal_logit if prediction is not None else None,
        }

    def identity(self) -> dict[str, Any]:
        state = self.audit_state()
        return {
            key: state[key]
            for key in (
                "model_family",
                "revision",
                "release_manifest_sha256",
                "shared_head_artifact_sha256",
                "shared_head_fingerprint",
                "personal_head_fingerprint",
                "heads_diverged",
                "update_enabled",
                "update_count",
                "development_only",
                "participant_disjoint_training",
                "replay_per_class",
                "replay_counts",
                "replay_seen",
            )
        }

    def _load_personal_checkpoint(self, directory: Path) -> None:
        pointer = directory.resolve() / "distilhubert-personal.json"
        if not pointer.is_file():
            raise DistilHuBERTUnavailable("Personal-head checkpoint manifest is missing")
        try:
            raw = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DistilHuBERTUnavailable("Personal-head checkpoint manifest is invalid") from exc
        if not isinstance(raw, dict):
            raise DistilHuBERTUnavailable("Personal-head checkpoint manifest must be an object")
        schema = str(raw.get("schema") or "")
        legacy_keys = {
            "schema",
            "model_family",
            "revision",
            "release_manifest_sha256",
            "shared_head_sha256",
            "personal_head_path",
            "personal_head_sha256",
            "personal_head_fingerprint",
            "update_count",
            "optimizer_step_count",
        }
        replay_keys = legacy_keys | {
            "replay_path",
            "replay_sha256",
            "replay_seen_negative",
            "replay_seen_positive",
        }
        _exact_keys(
            raw,
            legacy_keys if schema == LEGACY_CHECKPOINT_SCHEMA else replay_keys,
            "personal checkpoint",
        )
        if (
            schema not in {LEGACY_CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA}
            or raw["model_family"] != MODEL_FAMILY
            or raw["revision"] != self.release.revision
            or raw["release_manifest_sha256"] != self.release.manifest_sha256
            or raw["shared_head_sha256"] != self.release.shared_head_sha256
        ):
            raise DistilHuBERTUnavailable("Personal checkpoint belongs to a different release")
        artifact = _safe_relative_path(pointer.parent, raw["personal_head_path"], "personal_head_path")
        artifact_hash = _sha256(raw["personal_head_sha256"], "personal_head_sha256")
        if not artifact.is_file() or file_sha256(artifact) != artifact_hash:
            raise DistilHuBERTUnavailable("Personal-head artifact is missing or changed")
        personal = _validated_head(artifact, self.release.feature_dim)
        if (
            not np.array_equal(personal.feature_mean, self.shared_head.feature_mean)
            or not np.array_equal(personal.feature_scale, self.shared_head.feature_scale)
        ):
            raise DistilHuBERTUnavailable(
                "Personal checkpoint changed the release feature scaler"
            )
        for array in (personal.feature_mean, personal.feature_scale):
            array.setflags(write=False)
        if _head_fingerprint(personal) != _sha256(
            raw["personal_head_fingerprint"], "personal_head_fingerprint"
        ):
            raise DistilHuBERTUnavailable("Personal-head numerical fingerprint changed")
        update_count = int(raw["update_count"])
        optimizer_step_count = int(raw["optimizer_step_count"])
        if update_count < 0 or optimizer_step_count < update_count:
            raise DistilHuBERTUnavailable("Personal checkpoint counters are invalid")
        if optimizer_step_count > update_count * self.release.optimizer.steps_per_feedback:
            raise DistilHuBERTUnavailable(
                "Personal checkpoint optimizer step count exceeds the locked schedule"
            )
        self.personal_head = personal
        self.update_count = update_count
        self.optimizer_step_count = optimizer_step_count
        if schema == CHECKPOINT_SCHEMA:
            replay_artifact = _safe_relative_path(
                pointer.parent, raw["replay_path"], "replay_path"
            )
            replay_hash = _sha256(raw["replay_sha256"], "replay_sha256")
            if (
                not replay_artifact.is_file()
                or file_sha256(replay_artifact) != replay_hash
            ):
                raise DistilHuBERTUnavailable("Personal replay artifact is missing or changed")
            try:
                with np.load(replay_artifact, allow_pickle=False) as payload:
                    if set(payload.files) != {
                        "format_version",
                        "negative_features",
                        "positive_features",
                    }:
                        raise DistilHuBERTUnavailable(
                            "Personal replay artifact fields do not match schema"
                        )
                    if int(np.asarray(payload["format_version"]).reshape(())) != 1:
                        raise DistilHuBERTUnavailable(
                            "Unsupported personal replay artifact version"
                        )
                    negative = np.asarray(
                        payload["negative_features"], dtype=np.float32
                    ).copy()
                    positive = np.asarray(
                        payload["positive_features"], dtype=np.float32
                    ).copy()
            except DistilHuBERTUnavailable:
                raise
            except (OSError, ValueError, TypeError) as exc:
                raise DistilHuBERTUnavailable(
                    "Cannot read personal replay artifact"
                ) from exc
            capacity = self.release.optimizer.replay_per_class
            for label, values in ((0, negative), (1, positive)):
                if (
                    values.ndim != 2
                    or values.shape[1] != self.release.feature_dim
                    or values.shape[0] > capacity
                    or not np.isfinite(values).all()
                ):
                    raise DistilHuBERTUnavailable(
                        "Personal replay artifact dimensions are invalid"
                    )
                self._replay_features[label] = [row.copy() for row in values]
            self._replay_seen = {
                0: int(raw["replay_seen_negative"]),
                1: int(raw["replay_seen_positive"]),
            }
            if any(
                self._replay_seen[label] < len(self._replay_features[label])
                for label in (0, 1)
            ):
                raise DistilHuBERTUnavailable(
                    "Personal replay counters are smaller than stored buffers"
                )
        elif self.release.optimizer.replay_per_class > 0:
            raise DistilHuBERTUnavailable(
                "Replay-enabled release cannot restore a legacy checkpoint"
            )

    def save(self, target_dir: str | Path) -> tuple[Path, Path]:
        target = Path(target_dir).resolve()
        target.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self._assert_head_invariants()
            identifier = uuid.uuid4().hex
            head_path = target / f"personal_head-{identifier}.npz"
            write_head_artifact(head_path, self.personal_head)
            head_hash = file_sha256(head_path)
            replay_path = target / f"personal_replay-{identifier}.npz"
            negative = np.asarray(
                self._replay_features[0], dtype=np.float32
            ).reshape((-1, self.release.feature_dim))
            positive = np.asarray(
                self._replay_features[1], dtype=np.float32
            ).reshape((-1, self.release.feature_dim))
            np.savez_compressed(
                replay_path,
                format_version=np.asarray(1, dtype=np.int64),
                negative_features=negative,
                positive_features=positive,
            )
            replay_hash = file_sha256(replay_path)
            pointer = target / "distilhubert-personal.json"
            temporary = target / f".{pointer.name}.{identifier}.tmp"
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "model_family": MODEL_FAMILY,
                "revision": self.release.revision,
                "release_manifest_sha256": self.release.manifest_sha256,
                "shared_head_sha256": self.release.shared_head_sha256,
                "personal_head_path": head_path.name,
                "personal_head_sha256": head_hash,
                "personal_head_fingerprint": _head_fingerprint(self.personal_head),
                "update_count": self.update_count,
                "optimizer_step_count": self.optimizer_step_count,
                "replay_path": replay_path.name,
                "replay_sha256": replay_hash,
                "replay_seen_negative": self._replay_seen[0],
                "replay_seen_positive": self._replay_seen[1],
            }
            try:
                temporary.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                # Re-read the complete pointer before atomically publishing it.
                if json.loads(temporary.read_text(encoding="utf-8")) != payload:
                    raise DistilHuBERTUnavailable("Personal checkpoint verification failed")
                os.replace(temporary, pointer)
            finally:
                temporary.unlink(missing_ok=True)
            return pointer, head_path


def validated_personal_checkpoint(
    manifest_path: str | Path,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Fully validate a published personal checkpoint without loading the backbone.

    ``DistilHuBERTOnlineModel`` already owns the fail-closed schema, release,
    artifact, replay and numerical-fingerprint checks.  Supplying a dummy
    feature encoder lets closeout and orchestration reuse those exact checks
    without downloading or instantiating the heavy acoustic backbone.
    """

    directory = Path(checkpoint_dir).resolve()
    pointer = directory / "distilhubert-personal.json"
    model = DistilHuBERTOnlineModel(
        manifest_path,
        update_enabled=True,
        personal_checkpoint_dir=directory,
        feature_encoder=lambda _audio: np.zeros(1, dtype=np.float32),
    )
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # defensive; constructor checked it
        raise DistilHuBERTUnavailable(
            "Personal-head checkpoint manifest is invalid"
        ) from exc
    head_path = _safe_relative_path(
        directory, payload["personal_head_path"], "personal_head_path"
    )
    replay_path = _safe_relative_path(
        directory, payload["replay_path"], "replay_path"
    )
    identity = model.identity()
    return {
        "schema": str(payload["schema"]),
        "pointer_path": str(pointer),
        "pointer_sha256": file_sha256(pointer),
        "personal_head_path": str(head_path),
        "personal_head_sha256": file_sha256(head_path),
        "personal_head_fingerprint": str(
            identity["personal_head_fingerprint"]
        ),
        "replay_path": str(replay_path),
        "replay_sha256": file_sha256(replay_path),
        "update_count": int(identity["update_count"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "replay_counts": dict(identity["replay_counts"]),
        "replay_seen": dict(identity["replay_seen"]),
        "revision": str(identity["revision"]),
        "release_manifest_sha256": str(
            identity["release_manifest_sha256"]
        ),
        "shared_head_sha256": str(
            identity["shared_head_artifact_sha256"]
        ),
    }
