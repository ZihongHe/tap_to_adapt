"""Regression tests for prepare simulation e2e fixture.

The cases exercise the corresponding application or utility contracts using
explicit test inputs and fixtures. Assertions document expected state, output,
and failure behavior; this module contains no collected participant records.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_simulation_e2e_fixture import write_synthetic_checkpoint_pair


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_source(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {
                "layer.weight": torch.tensor([value, value + 1], dtype=torch.float32),
                "running_mean": torch.tensor([123.0], dtype=torch.float32),
            },
            "trainable_parameter_names": ["layer.weight"],
        },
        path,
    )


def test_synthetic_pair_diverges_without_mutating_m0(tmp_path: Path) -> None:
    encoder_source = tmp_path / "m0" / "encoder.pth"
    main_source = tmp_path / "m0" / "main.pth"
    write_source(encoder_source, 1.0)
    write_source(main_source, 2.0)
    source_hashes = (sha256(encoder_source), sha256(main_source))
    encoder_target = tmp_path / "q-8" / "encoder.pth"
    main_target = tmp_path / "q-8" / "main.pth"

    metadata = write_synthetic_checkpoint_pair(
        encoder_source,
        main_source,
        encoder_target,
        main_target,
        donor_rank=3,
        budget=8,
    )

    assert (sha256(encoder_source), sha256(main_source)) == source_hashes
    encoder = torch.load(encoder_target, map_location="cpu", weights_only=False)
    main = torch.load(main_target, map_location="cpu", weights_only=False)
    assert encoder["checkpoint_pair_id"] == main["checkpoint_pair_id"]
    assert encoder["model_state_dict"]["layer.weight"][0].item() != 1.0
    assert main["model_state_dict"]["layer.weight"][0].item() != 2.0
    assert metadata["checkpoint_pair_id"] == encoder["checkpoint_pair_id"]
    assert metadata["encoder"]["analysis_eligible"] is False
    assert metadata["main"]["analysis_eligible"] is False
    assert metadata["encoder"]["opportunity_budget"] == 8

    with pytest.raises(RuntimeError, match="pre-existing"):
        write_synthetic_checkpoint_pair(
            encoder_source,
            main_source,
            encoder_target,
            main_target,
            donor_rank=3,
            budget=8,
        )
