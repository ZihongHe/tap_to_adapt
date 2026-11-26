# Tap-to-Adapt

Response initiation timing models for speech agents and a corpus of 1,000 human-recorded speech clips.
BPRHT (Balanced Proximal Replay for History-conditioned Timing) combines frozen
DistilHuBERT representations, individual logistic decision heads, balanced
proximal replay, and recent correction history.

## Contents

| Path | Contents |
| --- | --- |
| `tap_to_adapt/distilhubert_online.py` | Frozen speech features, linear heads, replay, and online updates |
| `tap_to_adapt/tap_history.py` | Temporal and acoustic feedback history, threshold and logit adjustments |
| `data/pilot_1000/` | 1,000 audio clips, binary scenario labels, and cached acoustic features |
| `scripts/extract_public_pilot_features.py` | Feature extraction with a pinned DistilHuBERT revision |
| `tests/` | Numerical tests with generated fixtures |

## Installation

Use Python 3.11 or later in a virtual environment:

```sh
python -m pip install -e ".[audio,dev]"
python -m pytest tests -q
```

The model module accepts local backbone files and a fitted linear-head artifact
through `DistilHuBERTRelease`. The backbone identifier is `ntu-spml/distilhubert`,
pinned to revision `fa87d96265d6b7af66e112faff6ff44df419cec9`.
`write_head_artifact` writes the feature standardization parameters, weights,
and bias of a head fitted to the selected training partition. Backbone model
files are obtained from their original provider under the provider's terms.

## Corpus

Each recording is 15 seconds of mono PCM16 audio at 16 kHz. The labels contain
253 examples of beginning a response and 747 examples of continuing to wait.
The dataset card describes collection, ordering, representation, and evaluation.

```python
from pathlib import Path
import numpy as np

parts = {key: [] for key in ("features", "labels", "event_indices")}
for path in sorted(Path("data/pilot_1000/features").glob("*.npz")):
    with np.load(path, allow_pickle=False) as corpus:
        for key in parts:
            parts[key].append(corpus[key])
features, labels, event_indices = (np.concatenate(parts[key]) for key in parts)
```

To extract a new feature matrix:

```sh
python scripts/extract_public_pilot_features.py --device cpu --output pilot_features.npz
```

## License

The code and corpus use CC BY-NC 4.0. Third-party models retain their own terms.
