"""Select two role-locked local agent voices by acoustic maximin distance.

The script synthesizes one neutral audit sentence for every AISHELL-3 speaker,
extracts pitch, spectral and temporal descriptors in memory, excludes all
participant speakers, and selects the pair that maximizes the smallest
standardized distance to each other and to any participant voice. It never
opens an audio output device and writes only JSON audit data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

DEFAULT_PARTICIPANT_IDS = (
    "1,3,8,13,18,21,23,28,33,38,41,43,48,53,58,61,63,68,73,78,"
    "81,83,88,93,98,101,103,108,113,118,121,123,128,133,138,141,"
    "143,148,153,158"
)
AUDIT_TEXT = "今天我们来聊聊你的计划，也可以说说最近让你印象深刻的事情。"


def parse_ids(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def _frames(samples: np.ndarray, frame: int = 400, hop: int = 160) -> np.ndarray:
    if samples.size < frame:
        samples = np.pad(samples, (0, frame - samples.size))
    count = 1 + (samples.size - frame) // hop
    indices = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return samples[indices] * np.hanning(frame)[None, :]


def acoustic_features(samples: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
    frames = _frames(samples.astype(np.float64, copy=False))
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    voiced = frames[rms >= max(np.quantile(rms, 0.35), 1e-4)]
    if not voiced.size:
        voiced = frames

    spectrum = np.abs(np.fft.rfft(voiced, axis=1)) + 1e-10
    power = spectrum**2
    frequencies = np.fft.rfftfreq(frames.shape[1], 1.0 / sample_rate)
    norm = power.sum(axis=1, keepdims=True)
    centroid = (power * frequencies[None, :]).sum(axis=1) / norm[:, 0]
    cumulative = np.cumsum(power, axis=1)
    rolloff_indices = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff = frequencies[rolloff_indices]
    zcr = np.mean(np.diff(np.signbit(voiced), axis=1), axis=1)

    f0_values: list[float] = []
    min_lag = max(1, sample_rate // 400)
    max_lag = min(frames.shape[1] - 2, sample_rate // 70)
    for frame_values in voiced:
        centered = frame_values - frame_values.mean()
        corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        if corr[0] <= 1e-9:
            continue
        region = corr[min_lag : max_lag + 1]
        lag_offset = int(np.argmax(region))
        lag = min_lag + lag_offset
        periodicity = float(region[lag_offset] / corr[0])
        if periodicity >= 0.2:
            f0_values.append(sample_rate / lag)
    f0 = np.asarray(f0_values or [0.0], dtype=np.float64)

    edges = np.geomspace(80.0, sample_rate / 2.0, 9)
    band_values = []
    mean_power = power.mean(axis=0)
    total_power = float(mean_power.sum() + 1e-12)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (frequencies >= low) & (frequencies < high)
        band_values.append(np.log10(float(mean_power[mask].sum()) / total_power + 1e-8))

    vector = np.asarray(
        [
            np.log(max(float(np.median(f0)), 1.0)),
            np.log1p(float(np.subtract(*np.percentile(f0, [75, 25])))),
            np.log(max(float(np.median(centroid)), 1.0)),
            np.log1p(float(np.std(centroid))),
            np.log(max(float(np.median(rolloff)), 1.0)),
            float(np.median(zcr)),
            np.log(max(float(np.median(rms)), 1e-8)),
            *band_values,
        ],
        dtype=np.float64,
    )
    summary = {
        "median_f0_hz": round(float(np.median(f0)), 3),
        "f0_iqr_hz": round(float(np.subtract(*np.percentile(f0, [75, 25]))), 3),
        "median_spectral_centroid_hz": round(float(np.median(centroid)), 3),
        "median_rolloff85_hz": round(float(np.median(rolloff)), 3),
        "median_zcr": round(float(np.median(zcr)), 6),
        "duration_seconds": round(float(samples.size / sample_rate), 3),
    }
    return vector, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--participant-ids", type=parse_ids, default=parse_ids(DEFAULT_PARTICIPANT_IDS))
    parser.add_argument("--text", default=AUDIT_TEXT)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    load_dotenv(root / ".env")
    from app.config import settings
    from app.integrity import file_sha256
    from app.speech import SpeechService

    local_settings = replace(settings, tts_mode="sherpa_onnx")
    service = SpeechService(local_settings)
    model_dir = Path(local_settings.sherpa_tts_model_dir)
    speaker_lines = [
        row.strip()
        for row in (model_dir / "speakers.txt").read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    speaker_count = len(speaker_lines)
    participants = sorted(set(args.participant_ids))
    if any(not 0 <= value < speaker_count for value in participants):
        raise SystemExit("Participant ID outside speakers.txt")

    vectors: list[np.ndarray] = []
    summaries: dict[int, dict] = {}
    for speaker_id in range(speaker_count):
        samples, sample_rate = service._sherpa_samples(
            args.text, speaker_id=speaker_id, speed=1.0
        )
        vector, summary = acoustic_features(samples, sample_rate)
        vectors.append(vector)
        summaries[speaker_id] = summary
        print(json.dumps({"speaker_id": speaker_id, **summary}), flush=True)

    matrix = np.vstack(vectors)
    center = np.median(matrix, axis=0)
    scale = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
    scale[scale < 1e-6] = np.std(matrix, axis=0)[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    standardized = (matrix - center) / scale

    participant_matrix = standardized[participants]
    candidates = [speaker_id for speaker_id in range(speaker_count) if speaker_id not in participants]
    distance_to_participants = {
        speaker_id: float(
            np.min(np.linalg.norm(standardized[speaker_id] - participant_matrix, axis=1))
        )
        for speaker_id in candidates
    }
    pair_rows = []
    for offset, first in enumerate(candidates):
        for second in candidates[offset + 1 :]:
            between = float(np.linalg.norm(standardized[first] - standardized[second]))
            maximin = min(
                between,
                distance_to_participants[first],
                distance_to_participants[second],
            )
            pair_rows.append((maximin, between, first, second))
    maximin, between, first, second = max(pair_rows)
    if summaries[first]["median_f0_hz"] <= summaries[second]["median_f0_hz"]:
        primary, companion = first, second
    else:
        primary, companion = second, first

    audit = {
        "schema": "tap-to-adapt-local-agent-voice-selection/v1",
        "simulation_only": True,
        "silent_synthesis": True,
        "selection_rule": (
            "maximize the minimum robust-standardized Euclidean acoustic distance "
            "between agents and from either agent to every excluded participant voice"
        ),
        "feature_family": [
            "median_and_iqr_f0",
            "spectral_centroid",
            "spectral_rolloff85",
            "zero_crossing_rate",
            "rms",
            "eight_log_spectral_bands",
        ],
        "audit_text_sha256": hashlib.sha256(args.text.encode("utf-8")).hexdigest(),
        "tts_model": "vits-icefall-zh-aishell3",
        "tts_model_sha256": file_sha256(model_dir / "model.onnx"),
        "speaker_count_screened": speaker_count,
        "excluded_participant_ids": participants,
        "candidate_count": len(candidates),
        "experimental_agent": {
            "speaker_id": primary,
            "voice": f"aishell3-sid{primary}",
            "features": summaries[primary],
            "nearest_participant_distance": round(distance_to_participants[primary], 6),
        },
        "companion_agent": {
            "speaker_id": companion,
            "voice": f"aishell3-sid{companion}",
            "features": summaries[companion],
            "nearest_participant_distance": round(distance_to_participants[companion], 6),
        },
        "agent_pair_distance": round(between, 6),
        "maximin_distance": round(maximin, 6),
        "all_voices_mutually_exclusive": len(set(participants + [primary, companion]))
        == len(participants) + 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
