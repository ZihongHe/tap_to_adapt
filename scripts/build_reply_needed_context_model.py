#!/usr/bin/env python3
"""Build a participant-disjoint reply-needed text gate for the 4+2 simulator.

The eight-profile fixture is the development set.  Candidate text models are
scored by leave-one-profile-out predictions, and the selected model is then
refit on all eight profiles before it is applied to the separate 32-profile
duration fixture.  The exported JSON contains enough information for a small
dependency-free runtime implementation of the character TF-IDF logistic head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
)


@dataclass(frozen=True)
class Candidate:
    ngram_min: int
    ngram_max: int
    max_features: int
    c_value: float

    @property
    def candidate_id(self) -> str:
        return (
            f"char{self.ngram_min}-{self.ngram_max}"
            f"-f{self.max_features}-c{self.c_value:g}"
        )


CANDIDATES = tuple(
    Candidate(ngram_min, ngram_max, max_features, c_value)
    for ngram_min, ngram_max in ((1, 3), (1, 4), (2, 5))
    for max_features in (4_000, 8_000)
    for c_value in (0.25, 0.5, 1.0, 2.0, 5.0)
)


def load_primary_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for participant in payload["participants"]:
        participant_index = int(participant["participant_index"])
        profile_id = str(participant["profile"]["profile_id"])
        for event in participant["events"]:
            if str(event["phase"]) != "primary":
                continue
            text = " ".join(str(event["text"]).strip().split())
            rows.append(
                {
                    "participant_index": participant_index,
                    "profile_id": profile_id,
                    "opportunity_index": int(event["opportunity_index"]),
                    "cycle_index": int(event["cycle_index"]),
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "label": int(bool(event["response_expected"])),
                }
            )
    if not rows or len({row["label"] for row in rows}) != 2:
        raise ValueError(f"Primary reply-needed fixture is invalid: {path}")
    return rows


def vectorizer(candidate: Candidate) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(candidate.ngram_min, candidate.ngram_max),
        min_df=2,
        max_features=candidate.max_features,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )


def fit_model(
    candidate: Candidate, rows: list[dict[str, Any]]
) -> tuple[TfidfVectorizer, LogisticRegression]:
    encoder = vectorizer(candidate)
    features = encoder.fit_transform([row["text"] for row in rows])
    model = LogisticRegression(
        C=candidate.c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=3_000,
        random_state=20260728,
    ).fit(features, [row["label"] for row in rows])
    return encoder, model


def crossfit(
    candidate: Candidate, rows: list[dict[str, Any]]
) -> np.ndarray:
    probabilities = np.zeros(len(rows), dtype=np.float64)
    participant_ids = sorted({int(row["participant_index"]) for row in rows})
    for participant_index in participant_ids:
        train_indices = [
            index
            for index, row in enumerate(rows)
            if int(row["participant_index"]) != participant_index
        ]
        test_indices = [
            index
            for index, row in enumerate(rows)
            if int(row["participant_index"]) == participant_index
        ]
        train = [rows[index] for index in train_indices]
        encoder, model = fit_model(candidate, train)
        probabilities[test_indices] = model.predict_proba(
            encoder.transform([rows[index]["text"] for index in test_indices])
        )[:, 1]
    return probabilities


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(
        np.r_[np.arange(0.05, 0.951, 0.005), probabilities]
    )
    scored: list[tuple[tuple[float, ...], float]] = []
    for threshold in candidates:
        prediction = probabilities >= threshold
        positive = labels == 1
        negative = ~positive
        false_negative_rate = float(np.mean(~prediction[positive]))
        false_positive_rate = float(np.mean(prediction[negative]))
        correction_rate = float(np.mean(prediction != labels))
        score = (
            -max(false_negative_rate, false_positive_rate),
            -correction_rate,
            float(balanced_accuracy_score(labels, prediction)),
            -abs(float(threshold) - 0.5),
        )
        scored.append((score, float(threshold)))
    return max(scored, key=lambda item: item[0])[1]


def metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    prediction = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "n": int(len(labels)),
        "positive_n": int(np.sum(positive)),
        "negative_n": int(np.sum(negative)),
        "threshold": float(threshold),
        "correction_rate": float(np.mean(prediction != labels)),
        "false_negative_rate": float(np.mean(~prediction[positive])),
        "false_positive_rate": float(np.mean(prediction[negative])),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, prediction)
        ),
        "f1": float(f1_score(labels, prediction)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s\s+", " ", text.strip().lower())


def char_ngrams(text: str, lower: int, upper: int) -> list[str]:
    normalized = normalize_text(text)
    return [
        normalized[start : start + size]
        for size in range(lower, upper + 1)
        for start in range(0, max(0, len(normalized) - size + 1))
    ]


def dependency_free_probability(artifact: dict[str, Any], text: str) -> float:
    vocabulary = artifact["vectorizer"]["vocabulary"]
    idf = artifact["vectorizer"]["idf"]
    weight = artifact["head"]["weight"]
    counts: dict[int, int] = {}
    for token in char_ngrams(
        text,
        int(artifact["vectorizer"]["ngram_min"]),
        int(artifact["vectorizer"]["ngram_max"]),
    ):
        index = vocabulary.get(token)
        if index is not None:
            counts[int(index)] = counts.get(int(index), 0) + 1
    values: dict[int, float] = {
        index: (1.0 + math.log(count)) * float(idf[index])
        for index, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in values.values()))
    logit = float(artifact["head"]["bias"])
    if norm > 0:
        logit += sum(
            float(weight[index]) * value / norm
            for index, value in values.items()
        )
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))


def prediction_manifest(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    threshold: float,
    *,
    source: str,
    model_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema": "tap-to-adapt-reply-needed-predictions/v1",
        "source": source,
        "model_sha256": model_sha256,
        "threshold": float(threshold),
        "rows": [
            {
                **row,
                "probability": float(probability),
                "prediction": bool(probability >= threshold),
            }
            for row, probability in zip(rows, probabilities, strict=True)
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection8-turns", type=Path, required=True)
    parser.add_argument("--duration32-turns", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development = load_primary_rows(args.selection8_turns)
    held_out = load_primary_rows(args.duration32_turns)
    development_labels = np.asarray(
        [row["label"] for row in development], dtype=np.int64
    )
    held_out_labels = np.asarray(
        [row["label"] for row in held_out], dtype=np.int64
    )

    candidate_rows: list[dict[str, Any]] = []
    crossfit_cache: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        probabilities = crossfit(candidate, development)
        threshold = choose_threshold(development_labels, probabilities)
        row = {
            "candidate_id": candidate.candidate_id,
            "ngram_min": candidate.ngram_min,
            "ngram_max": candidate.ngram_max,
            "max_features": candidate.max_features,
            "c_value": candidate.c_value,
            **metrics(development_labels, probabilities, threshold),
        }
        candidate_rows.append(row)
        crossfit_cache[candidate.candidate_id] = probabilities
    candidate_rows.sort(
        key=lambda row: (
            float(row["correction_rate"]),
            max(
                float(row["false_negative_rate"]),
                float(row["false_positive_rate"]),
            ),
            float(row["brier"]),
        )
    )
    selected_row = candidate_rows[0]
    selected = next(
        candidate
        for candidate in CANDIDATES
        if candidate.candidate_id == selected_row["candidate_id"]
    )
    threshold = float(selected_row["threshold"])
    encoder, model = fit_model(selected, development)
    held_out_probabilities = model.predict_proba(
        encoder.transform([row["text"] for row in held_out])
    )[:, 1]

    vocabulary = {
        token: int(index) for token, index in encoder.vocabulary_.items()
    }
    artifact = {
        "schema": "tap-to-adapt-reply-needed-char-tfidf/v1",
        "version": "selection8-lopo-20260728-v1",
        "simulation_only_training": True,
        "human_effect_evidence": False,
        "training_profiles": 8,
        "training_rows": len(development),
        "selection_rule": (
            "leave-one-profile-out; minimize the larger of false-negative and "
            "false-positive rates, then total correction rate and Brier score"
        ),
        "selected_candidate": selected.candidate_id,
        "threshold": threshold,
        "vectorizer": {
            "analyzer": "char",
            "lowercase": True,
            "ngram_min": selected.ngram_min,
            "ngram_max": selected.ngram_max,
            "min_df": 2,
            "max_features": selected.max_features,
            "sublinear_tf": True,
            "norm": "l2",
            "vocabulary": vocabulary,
            "idf": [float(value) for value in encoder.idf_],
        },
        "head": {
            "type": "logistic_regression",
            "c_value": selected.c_value,
            "class_weight": "balanced",
            "weight": [float(value) for value in model.coef_[0]],
            "bias": float(model.intercept_[0]),
        },
        "development_lopo_metrics": {
            key: value
            for key, value in selected_row.items()
            if key
            not in {
                "candidate_id",
                "ngram_min",
                "ngram_max",
                "max_features",
                "c_value",
            }
        },
        "duration32_heldout_metrics": metrics(
            held_out_labels, held_out_probabilities, threshold
        ),
    }
    artifact_path = args.output_dir / "reply_needed_model.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    crossfit_probabilities = crossfit_cache[selected.candidate_id]
    runtime_probabilities = np.asarray(
        [
            dependency_free_probability(artifact, row["text"])
            for row in held_out
        ]
    )
    max_runtime_difference = float(
        np.max(np.abs(runtime_probabilities - held_out_probabilities))
    )
    if max_runtime_difference > 1e-10:
        raise RuntimeError(
            "Dependency-free inference does not match sklearn: "
            f"{max_runtime_difference}"
        )

    selection_manifest = prediction_manifest(
        development,
        crossfit_probabilities,
        threshold,
        source="selection8_leave_one_profile_out",
        model_sha256=None,
    )
    duration_manifest = prediction_manifest(
        held_out,
        held_out_probabilities,
        threshold,
        source="selection8_refit_applied_to_heldout_duration32",
        model_sha256=artifact_sha256,
    )
    (args.output_dir / "selection8_context_predictions.json").write_text(
        json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "duration32_context_predictions.json").write_text(
        json.dumps(duration_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "candidate_metrics.csv", candidate_rows)
    summary = {
        "schema": "tap-to-adapt-reply-needed-selection/v1",
        "artifact_sha256": artifact_sha256,
        "selected_candidate": selected.candidate_id,
        "threshold": threshold,
        "selection8_lopo": artifact["development_lopo_metrics"],
        "duration32_heldout": artifact["duration32_heldout_metrics"],
        "dependency_free_max_probability_difference": max_runtime_difference,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
