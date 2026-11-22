"""Numerical tests of frozen encoders, balanced replay, and head persistence."""
from __future__ import annotations
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from tap_to_adapt.distilhubert_online import (
    MODEL_FAMILY, MODEL_ID, RELEASE_SCHEMA, REPRESENTATION,
    DistilHuBERTOnlineModel, DistilHuBERTRelease, DistilHuBERTUnavailable,
    LinearHeadState, write_head_artifact,
)
from tap_to_adapt.integrity import file_sha256
from tap_to_adapt.tap_history import TapHistoryConfig, TapHistoryRuntime

FEATURE_DIM = 10752
REVISION = "fa87d96265d6b7af66e112faff6ff44df419cec9"
def make_release(
    tmp_path: Path,
    *,
    learning_rate: float = 0.1,
    steps_per_feedback: int = 2,
) -> Path:
    head_path = tmp_path / "shared_head.npz"
    write_head_artifact(
        head_path,
        LinearHeadState(
            feature_mean=np.zeros(FEATURE_DIM, dtype=np.float32),
            feature_scale=np.ones(FEATURE_DIM, dtype=np.float32),
            weight=np.zeros(FEATURE_DIM, dtype=np.float32),
            bias=0.0,
        ),
    )
    backbone_hashes: dict[str, str] = {}
    for name, content in (
        ("config.json", b"{}"),
        ("model.safetensors", b"not-a-real-weight-file"),
        ("preprocessor_config.json", b"{}"),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        backbone_hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema": RELEASE_SCHEMA,
        "model_family": MODEL_FAMILY,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "sample_rate": 16_000,
        "clip_seconds": 15.0,
        "representation": REPRESENTATION,
        "hidden_size": 768,
        "feature_dim": FEATURE_DIM,
        "backbone_files": backbone_hashes,
        "shared_head": {
            "path": head_path.name,
            "sha256": file_sha256(head_path),
            "format": "numpy-npz-no-pickle/v1",
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "l2": 0.0001,
            "proximal_to_shared": 0.01,
            "max_gradient_norm": 2.0,
            "steps_per_feedback": steps_per_feedback,
            "replay_per_class": 12,
            "replay_batch_size": 24,
            "replay_seed": 17,
        },
        "training_provenance": {
            "source": "unit-test-fixture",
            "source_manifest_sha256": "1" * 64,
            "feature_cache_sha256": "2" * 64,
            "head_fitter": "standard-scaler-liblinear-logistic-v1",
            "regularization_c": 0.01,
            "fit_examples": 4,
            "positive_examples": 2,
            "negative_examples": 2,
            "participant_disjoint": False,
            "development_only": True,
        },
    }
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path

def fixed_encoder(_: np.ndarray) -> np.ndarray:
    return np.full(FEATURE_DIM, 1.0 / np.sqrt(FEATURE_DIM), dtype=np.float32)

def test_shared_head_is_cloned_then_only_personal_head_updates(tmp_path: Path) -> None:
    manifest = make_release(tmp_path)
    model = DistilHuBERTOnlineModel(
        manifest, update_enabled=True, feature_encoder=fixed_encoder
    )
    audio = np.zeros(4_000, dtype=np.float32)

    initial = model.predict_both(audio)
    shared_before = initial.shared_head_fingerprint
    assert initial.shared_probability == pytest.approx(0.5)
    assert initial.personal_probability == pytest.approx(initial.shared_probability)
    assert initial.personal_head_fingerprint == shared_before

    loss = model.train(audio, 1, weight=1.0)
    assert loss > 0
    assert model.last_update_applied is True
    assert model.last_training_steps == 2
    after = model.predict_both(audio)
    assert after.shared_probability == pytest.approx(0.5)
    assert after.shared_head_fingerprint == shared_before
    assert after.personal_probability > after.shared_probability
    assert after.personal_head_fingerprint != shared_before
    assert model.audit_state()["heads_diverged"] is True

def test_fixed_head_refuses_update_without_mutation(tmp_path: Path) -> None:
    model = DistilHuBERTOnlineModel(
        make_release(tmp_path), update_enabled=False, feature_encoder=fixed_encoder
    )
    audio = np.zeros(4_000, dtype=np.float32)
    before = model.audit_state()
    assert model.train(audio, 1) == 0.0
    after = model.audit_state()

    assert model.last_update_applied is False
    assert model.last_update_reason == "frozen_condition"
    assert after["update_count"] == 0
    assert after["shared_head_fingerprint"] == before["shared_head_fingerprint"]
    assert after["personal_head_fingerprint"] == before["personal_head_fingerprint"]
    assert after["heads_diverged"] is False

def test_personal_checkpoint_round_trip_is_hash_bound(tmp_path: Path) -> None:
    manifest = make_release(tmp_path / "release")
    model = DistilHuBERTOnlineModel(
        manifest, update_enabled=True, feature_encoder=fixed_encoder
    )
    audio = np.zeros(4_000, dtype=np.float32)
    model.train(audio, 1)
    expected = model.predict(audio)
    pointer, artifact = model.save(tmp_path / "participant")

    assert pointer.is_file() and artifact.is_file()
    restored = DistilHuBERTOnlineModel(
        manifest,
        update_enabled=True,
        personal_checkpoint_dir=pointer.parent,
        feature_encoder=fixed_encoder,
    )
    assert restored.predict(audio) == pytest.approx(expected)
    assert restored.update_count == 1
    assert restored.identity()["heads_diverged"] is True
    assert restored.identity()["replay_counts"] == {
        "negative": 0,
        "positive": 1,
    }
    # A resumed checkpoint may be loaded for frozen inference. Updates stay disabled.
    frozen = DistilHuBERTOnlineModel(
        manifest,
        update_enabled=False,
        personal_checkpoint_dir=pointer.parent,
        feature_encoder=fixed_encoder,
    )
    before = frozen.audit_state()
    assert frozen.predict(audio) == pytest.approx(expected)
    assert frozen.train(audio, 0) == 0.0
    after = frozen.audit_state()
    assert after["update_count"] == before["update_count"]
    assert after["personal_head_fingerprint"] == before["personal_head_fingerprint"]
    assert frozen.last_update_reason == "frozen_condition"

    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    pointer_payload["optimizer_step_count"] = 3
    pointer.write_text(json.dumps(pointer_payload), encoding="utf-8")
    with pytest.raises(DistilHuBERTUnavailable, match="exceeds the locked schedule"):
        DistilHuBERTOnlineModel(
            manifest,
            update_enabled=True,
            personal_checkpoint_dir=pointer.parent,
            feature_encoder=fixed_encoder,
        )
    # Restore the valid pointer before separately tampering with the artifact.
    pointer_payload["optimizer_step_count"] = 2
    pointer.write_text(json.dumps(pointer_payload), encoding="utf-8")

    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(DistilHuBERTUnavailable, match="missing or changed"):
        DistilHuBERTOnlineModel(
            manifest,
            update_enabled=True,
            personal_checkpoint_dir=pointer.parent,
            feature_encoder=fixed_encoder,
        )

def test_replay_is_class_balanced_and_persists_across_restart(tmp_path: Path) -> None:
    manifest = make_release(tmp_path / "release")

    def positive_encoder(_: np.ndarray) -> np.ndarray:
        value = np.zeros(FEATURE_DIM, dtype=np.float32)
        value[0] = 1.0
        return value

    def negative_encoder(_: np.ndarray) -> np.ndarray:
        value = np.zeros(FEATURE_DIM, dtype=np.float32)
        value[0] = -1.0
        return value

    audio = np.zeros(4_000, dtype=np.float32)
    model = DistilHuBERTOnlineModel(
        manifest, update_enabled=True, feature_encoder=positive_encoder
    )
    model.train(audio, 1)
    model.feature_encoder = negative_encoder
    model._last_audio_digest = None
    model._last_features = None
    model.train(audio, 0)
    pointer, _ = model.save(tmp_path / "participant")

    restored = DistilHuBERTOnlineModel(
        manifest,
        update_enabled=True,
        personal_checkpoint_dir=pointer.parent,
        feature_encoder=positive_encoder,
    )
    assert restored.identity()["replay_counts"] == {
        "negative": 1,
        "positive": 1,
    }
    assert restored.identity()["replay_seen"] == {
        "negative": 1,
        "positive": 1,
    }

def test_release_manifest_rejects_unpinned_or_tampered_artifacts(tmp_path: Path) -> None:
    manifest = make_release(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["revision"] = "main"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DistilHuBERTUnavailable, match="full 40-character"):
        DistilHuBERTRelease.load(manifest)

    manifest = make_release(tmp_path / "second")
    release = DistilHuBERTRelease.load(manifest)
    release.shared_head_path.write_bytes(release.shared_head_path.read_bytes() + b"tamper")
    with pytest.raises(DistilHuBERTUnavailable, match="wrong hash"):
        DistilHuBERTRelease.load(manifest)


def test_interrupt_raises_threshold_activate_lowers_it() -> None:
    runtime = TapHistoryRuntime(TapHistoryConfig.default())
    feature = np.zeros(16, dtype=np.float64)
    before = runtime.decide(
        now_seconds=10.0, current_feature=feature, base_threshold=0.70
    )
    assert before["threshold_shift"] == 0.0

    runtime.observe_and_update(
        now_seconds=10.0,
        kind="interrupt",
        interrupt_delay_ms=250.0,
        interrupt_feature=feature,
        prior_feature=feature,
        current_feature=feature,
    )
    after_int = runtime.decide(
        now_seconds=12.0, current_feature=feature, base_threshold=0.70
    )
    assert after_int["threshold_shift"] > 0.0
    assert after_int["effective_activation_threshold"] > 0.70

    runtime.observe_and_update(
        now_seconds=20.0,
        kind="manual_activate",
        interrupt_delay_ms=None,
        current_feature=feature,
    )
    # Fresh activate pressure should pull the threshold back toward sensitive.
    after_act = runtime.decide(
        now_seconds=21.0, current_feature=feature, base_threshold=0.70
    )
    assert after_act["threshold_shift"] < after_int["threshold_shift"]

def test_persistence_roundtrip(tmp_path) -> None:
    runtime = TapHistoryRuntime(TapHistoryConfig.default())
    feature = np.ones(8, dtype=np.float64)
    runtime.observe_and_update(
        now_seconds=5.0,
        kind="interrupt",
        interrupt_delay_ms=100.0,
        interrupt_feature=feature,
        prior_feature=np.zeros(8),
        current_feature=feature,
    )
    runtime.save(tmp_path)
    restored = TapHistoryRuntime.load_or_create(tmp_path, TapHistoryConfig.default())
    assert restored.controller.update_count == 1
    assert restored.controller.k_interrupt > TapHistoryConfig.default().k_interrupt0
    assert len(restored.memory._events) == 1
