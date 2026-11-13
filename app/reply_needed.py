"""Optional text-context reply-needed model loader and scorer.

This component provides a separate diagnostic or legacy context score.
The deployed continuous timing path assigns activation authority to acoustic
timing inference and causal tap history rather than this text-context score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


class ReplyNeededUnavailable(RuntimeError):
    pass


class ReplyNeededModel:
    """Dependency-free character TF-IDF logistic context gate."""

    def __init__(self, artifact_path: str | Path):
        path = Path(artifact_path)
        if not path.is_file():
            raise ReplyNeededUnavailable("Reply-needed model artifact is missing")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "tap-to-adapt-reply-needed-char-tfidf/v1":
                raise ValueError("unexpected schema")
            vectorizer = payload["vectorizer"]
            head = payload["head"]
            self.vocabulary = {
                str(token): int(index)
                for token, index in vectorizer["vocabulary"].items()
            }
            self.idf = tuple(float(value) for value in vectorizer["idf"])
            self.weight = tuple(float(value) for value in head["weight"])
            self.bias = float(head["bias"])
            self.ngram_min = int(vectorizer["ngram_min"])
            self.ngram_max = int(vectorizer["ngram_max"])
            self.threshold = float(payload["threshold"])
            self.version = str(payload["version"])
            if (
                len(self.idf) != len(self.weight)
                or len(self.idf) != len(self.vocabulary)
                or not 0.0 <= self.threshold <= 1.0
                or self.ngram_min < 1
                or self.ngram_max < self.ngram_min
            ):
                raise ValueError("invalid dimensions or operating point")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplyNeededUnavailable(
                "Reply-needed model artifact is invalid"
            ) from exc
        self.path = path.resolve()
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"\s\s+", " ", text.strip().lower())

    def _features(self, text: str) -> dict[int, float]:
        normalized = self.normalize(text)
        counts: dict[int, int] = {}
        for size in range(self.ngram_min, self.ngram_max + 1):
            for start in range(0, max(0, len(normalized) - size + 1)):
                index = self.vocabulary.get(normalized[start : start + size])
                if index is not None:
                    counts[index] = counts.get(index, 0) + 1
        values = {
            index: (1.0 + math.log(count)) * self.idf[index]
            for index, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in values.values()))
        if norm <= 0:
            return {}
        return {index: value / norm for index, value in values.items()}

    def predict(self, text: str) -> dict[str, Any]:
        cleaned = self.normalize(text)
        if not cleaned:
            raise ValueError("empty transcript")
        features = self._features(cleaned)
        logit = self.bias + sum(
            self.weight[index] * value for index, value in features.items()
        )
        probability = 1.0 / (
            1.0 + math.exp(-max(-30.0, min(30.0, logit)))
        )
        return {
            "probability": probability,
            "prediction": probability >= self.threshold,
            "threshold": self.threshold,
            "version": self.version,
            "artifact_sha256": self.sha256,
            "text_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
            "recognized_features": len(features),
        }
