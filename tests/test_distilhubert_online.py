"""Regression tests for distilhubert online.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.distilhubert_online import (
    MODEL_FAMILY,
    MODEL_ID,
    RELEASE_SCHEMA,
    REPRESENTATION,
    DistilHuBERTOnlineModel,
    DistilHuBERTRelease,
    DistilHuBERTUnavailable,
    LinearHeadState,
    write_head_artifact,
)
from app.integrity import file_sha256


FEATURE_DIM = 10_752
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
            "replay_seed": 20260723,
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


def test_frozen_condition_refuses_update_without_mutation(tmp_path: Path) -> None:
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


class FakeDB:
    def feedback_counts(self, participant_id: str) -> dict[str, int]:
        return {
            "activation_opportunities": 0,
            "eligible_feedbacks": 0,
            "positive_feedbacks": 0,
            "negative_feedbacks": 0,
        }

    def learning_opportunity_count(self, participant_id: str) -> int:
        return 0

    def next_learning_opportunity(self, participant_id: str) -> int:
        return 1

    def next_activation_opportunity(self, participant_id: str) -> int:
        return 1

    def next_eligible_feedback(self, participant_id: str, label: int) -> dict[str, int]:
        raise AssertionError("not used while constructing a runtime")


def participant(group: str) -> dict[str, Any]:
    protocol = {
        "version": "distil-pilot-v1",
        "pipeline": {},
        "initialization": {"activation_threshold": 0.5},
    }
    protocol_text = json.dumps(protocol, sort_keys=True)
    return {
        "id": f"pilot-{group}",
        "run_id": "pilot-run",
        "group_name": group,
        "cohort": "pilot",
        "data_origin": "simulation",
        "prior_id": None,
        "run_protocol_json": protocol_text,
        "run_protocol_sha256": hashlib.sha256(protocol_text.encode()).hexdigest(),
        "protocol_version": protocol["version"],
    }


def test_timing_service_opt_in_maps_personalized_and_frozen_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Import lazily: test_api establishes its isolated application environment
    # during collection, and this model-only test must not pre-initialize the
    # process-global FastAPI settings ahead of that fixture.
    from app.config import settings
    from app.timing import ModelUnavailable, TimingService

    manifest = make_release(tmp_path / "release")
    calls: list[bool] = []
    preload_calls: list[dict[str, Any]] = []
    preloaded_encoder = object()

    class FakeOnlineModel:
        @classmethod
        def preload_feature_encoder(cls, *args: Any, **kwargs: Any) -> object:
            preload_calls.append(kwargs)
            return preloaded_encoder

        def __init__(self, *args: Any, update_enabled: bool, **kwargs: Any):
            assert kwargs.get("feature_encoder") is preloaded_encoder
            assert "cache_dir" not in kwargs
            assert "allow_download" not in kwargs
            assert "device" not in kwargs
            calls.append(update_enabled)
            self.update_enabled = update_enabled

        def identity(self) -> dict[str, Any]:
            return {"model_family": MODEL_FAMILY, "update_enabled": self.update_enabled}

        def save(self, target: Path) -> tuple[Path, Path]:
            return target / "manifest", target / "head"

    monkeypatch.setattr(
        "app.distilhubert_online.DistilHuBERTOnlineModel", FakeOnlineModel
    )
    configured = replace(
        settings,
        agent_mode="distilhubert",
        data_dir=tmp_path / "data",
        distilhubert_release_manifest=str(manifest),
        distilhubert_cache_dir="",
        distilhubert_allow_download=False,
        distilhubert_device="auto",
    )
    service = TimingService(configured, FakeDB())
    learning = service.runtime(participant("personalized"))
    frozen = service.runtime(participant("frozen"))

    # Startup preloads once and validates one frozen head. Participant runtime
    # creation only clones small heads from that already-ready encoder.
    assert len(preload_calls) == 1
    assert calls == [False, True, False, False, False]
    assert learning.model_identity["update_enabled"] is True
    assert frozen.model_identity["update_enabled"] is False
    assert learning.model_identity["comparison_policy"] == "shared_base_clone"
    assert learning.model_identity["matched_other_supported"] is False

    learning.set_phase("micro_probe", "matched_other")
    with pytest.raises(ModelUnavailable, match="no matched-other"):
        learning._predict(0.0)

    formal = participant("personalized")
    formal.update({"id": "formal-study-a", "cohort": "study_a", "data_origin": "formal"})
    study_a = service.runtime(formal)
    assert study_a.model_identity["update_enabled"] is True

    formal = participant("personalized")
    formal.update({"id": "formal-study-b", "cohort": "study_b", "data_origin": "formal"})
    with pytest.raises(ModelUnavailable, match="locked study pipelines remain unavailable"):
        service.runtime(formal)


def test_timing_service_fails_before_websocket_when_preload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings
    from app.timing import ModelUnavailable, TimingService

    manifest = make_release(tmp_path / "release")

    class FailingOnlineModel:
        @classmethod
        def preload_feature_encoder(cls, *args: Any, **kwargs: Any) -> object:
            raise DistilHuBERTUnavailable("preload failed")

    monkeypatch.setattr(
        "app.distilhubert_online.DistilHuBERTOnlineModel", FailingOnlineModel
    )
    configured = replace(
        settings,
        agent_mode="distilhubert",
        distilhubert_release_manifest=str(manifest),
    )
    with pytest.raises(ModelUnavailable, match="preload failed"):
        TimingService(configured, FakeDB())


@pytest.mark.parametrize(
    ("schedule", "warmup_feedbacks", "warmup_scale", "expected_stage"),
    [
        ((), 5, 0.25, "legacy_warmup"),
        (((5, 0.25),), 0, 1.0, "feedback_lt_5"),
    ],
)
def test_locked_network_scale_changes_steps_delta_and_logging(
    tmp_path: Path,
    schedule: tuple[tuple[int, float], ...],
    warmup_feedbacks: int,
    warmup_scale: float,
    expected_stage: str,
) -> None:
    from app.config import settings
    from app.timing import ParticipantRuntime

    manifest = make_release(
        tmp_path / "release", learning_rate=0.1, steps_per_feedback=8
    )
    scaled_model = DistilHuBERTOnlineModel(
        manifest, update_enabled=True, feature_encoder=fixed_encoder
    )
    full_model = DistilHuBERTOnlineModel(
        manifest, update_enabled=True, feature_encoder=fixed_encoder
    )
    common = {
        "participant_id": "scaled",
        "group_name": "personalized",
        "cohort": "pilot",
        "settings": replace(settings, data_dir=tmp_path / "data"),
        "phase": "learning",
        "network_update_schedule": schedule,
        "warmup_feedbacks": warmup_feedbacks,
        "warmup_update_scale": warmup_scale,
    }
    scaled_runtime = ParticipantRuntime(model=scaled_model, **common)
    full_runtime = ParticipantRuntime(
        model=full_model,
        **{
            **common,
            "participant_id": "full",
            "network_update_schedule": (),
            "warmup_feedbacks": 0,
            "warmup_update_scale": 1.0,
        },
    )
    audio = np.zeros(4_000, dtype=np.float32)
    scaled_before = scaled_model.personal_head.weight.copy()
    full_before = full_model.personal_head.weight.copy()
    scaled_result: dict[str, Any] = {}
    full_result: dict[str, Any] = {}
    scaled_runtime._apply_training_update(audio, 1, scaled_result)
    full_runtime._apply_training_update(audio, 1, full_result)

    scaled_delta = float(
        np.linalg.norm(scaled_model.personal_head.weight - scaled_before)
    )
    full_delta = float(np.linalg.norm(full_model.personal_head.weight - full_before))
    assert 0 < scaled_delta < full_delta
    assert scaled_result["network_update_scale"] == pytest.approx(0.25)
    assert scaled_result["network_update_stage"] == expected_stage
    assert scaled_result["class_weight"] == pytest.approx(0.25)
    assert scaled_result["requested_training_steps"] == 2
    assert scaled_result["training_steps"] == 2
    assert scaled_result["optimizer_step_after"] - scaled_result["optimizer_step_before"] == 2
    assert scaled_result["optimizer_learning_rates_before"] == pytest.approx([0.1])
    assert scaled_result["optimizer_learning_rates_during"] == pytest.approx([0.025])
    assert scaled_result["optimizer_learning_rates_after"] == pytest.approx([0.1])
    assert scaled_result["parameter_state_interpolated"] is True
    assert scaled_result["parameter_interpolation_scale"] == pytest.approx(0.25)
    assert full_result["requested_training_steps"] == 8
    assert full_result["training_steps"] == 8


def test_distil_micro_probe_b_is_shared_base_not_matched_other() -> None:
    from app.main import _micro_probe_policy

    assert _micro_probe_policy("B", cohort="pilot", agent_mode="distilhubert") == "base"
    assert _micro_probe_policy("B", cohort="study_b", agent_mode="dtcn") == "matched_other"
    assert _micro_probe_policy("B", cohort="study_a", agent_mode="dtcn") == "base"


def test_timing_payload_exposes_same_opportunity_shared_personal_comparison(
    tmp_path: Path,
) -> None:
    from app.config import settings
    from app.timing import ParticipantRuntime

    model = DistilHuBERTOnlineModel(
        make_release(tmp_path / "release"),
        update_enabled=True,
        feature_encoder=fixed_encoder,
    )
    runtime = ParticipantRuntime(
        participant_id="distil-pilot",
        group_name="personalized",
        cohort="pilot",
        settings=replace(settings, data_dir=tmp_path / "data"),
        phase="learning",
        model=model,
        activation_threshold=0.99,
    )
    pcm = np.full(3_200, 2_000, dtype="<i2").tobytes()
    prediction = runtime.process_pcm(pcm, client_ms=2_000.0)
    comparison = prediction["model_head_comparison"]
    assert comparison["shared_probability"] == pytest.approx(
        comparison["personal_probability"]
    )
    assert comparison["heads_diverged"] is False

    result: dict[str, Any] = {}
    runtime._apply_training_update(
        np.zeros(4_000, dtype=np.float32), label=1, result=result
    )
    assert result["model_head_comparison_before_update"]["heads_diverged"] is False
    assert result["model_head_comparison_after_update"]["heads_diverged"] is True
    assert result["current_network_update_applied"] is True


def test_distil_activation_waits_until_negative_feedback_audio_is_trainable(
    tmp_path: Path,
) -> None:
    from app.config import settings
    from app.timing import ParticipantRuntime

    model = DistilHuBERTOnlineModel(
        make_release(tmp_path / "release"),
        update_enabled=True,
        feature_encoder=fixed_encoder,
    )
    runtime = ParticipantRuntime(
        participant_id="distil-trainable-gate",
        group_name="personalized",
        cohort="pilot",
        settings=replace(settings, data_dir=tmp_path / "data"),
        phase="learning",
        model=model,
        activation_threshold=0.49,
    )
    first = runtime.process_pcm(
        np.full(3_200, 2_000, dtype="<i2").tobytes(), client_ms=2_000.0
    )
    assert first["activate"] is False
    assert "training_audio_not_ready" in first["decision_gate_reasons"]

    second = runtime.process_pcm(
        np.full(3_200, 2_000, dtype="<i2").tobytes(), client_ms=2_200.0
    )
    assert second["activate"] is True
    assert "training_audio_not_ready" not in second["decision_gate_reasons"]
