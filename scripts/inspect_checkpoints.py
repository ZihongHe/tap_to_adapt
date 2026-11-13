"""Command-line utility: inspect checkpoints.

This module supports the study's local tools and test fixtures. Inspect the
entry point and explicit path arguments before running data-writing operations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


CONFIG_KEYS = (
    "model_structure",
    "input_type",
    "encoder_type",
    "feature_normalization",
    "use_transformer",
    "dilated",
    "z_dim",
    "main_hidden",
    "num_of_step",
    "LEARNING_RATE",
    "RLHF_LEARNING_RATE",
)


def main() -> None:
    for raw_path in sys.argv[1:]:
        path = Path(raw_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") or {}
        state = checkpoint.get("model_state_dict") or {}
        print(
            json.dumps(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "config": {key: config.get(key) for key in CONFIG_KEYS},
                    "state_keys": len(state),
                    "sample_state_keys": list(state)[:8],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
