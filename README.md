# Tap-to-Adapt

Tap-to-Adapt learns when to initiate a speech response from corrective taps during interaction. Manual activation supplies a positive timing label; manual interruption supplies a negative label for the response's initiation. A frozen DistilHuBERT encoder, an individual decision head, class-balanced proximal experience replay (CBPER), and causal tap-history adaptation (CTHA) implement the timing path. ASR, conversational generation, and speech synthesis provide response content.

This repository contains the study implementation: a desktop client, a mobile push-to-talk companion, and a FastAPI server. English and Simplified Chinese interfaces are included. The [preprint](https://arxiv.org/abs/2603.14449) describes the research; consult its version history for the corresponding manuscript.

## Contents

| Path | Contents |
| --- | --- |
| `app/distilhubert_online.py` | Frozen encoder, individual heads, replay, proximal regularization, and checkpoint validation |
| `app/tap_history.py` | Causal feedback features and timing modulation |
| `app/main.py` | HTTP/WebSocket endpoints and interaction orchestration |
| `app/` | Enrollment, consent, storage, analysis, speech, and scheduling services |
| `static/` | Desktop, companion, and operator interfaces |
| `content/` | Protocol, questionnaires, topics, and deployment templates |
| `scripts/` | Simulation, model preparation, validation, and export utilities |
| `tests/` | Unit and integration tests with synthetic fixtures |
| `docs/IMPLEMENTATION.md` | Runtime architecture, model dependencies, and validation notes |
| `data/pilot_1000/` | Authorized 1,000-record pilot corpus, labels, and acoustic features |
| `data/public_acoustic_reference/` | Acoustic prototypes fitted only to the public pilot |
| `data/user_corrections_dp_v1/` | Participant-level differentially private joint statistics |

## Research data

The [pilot corpus](data/pilot_1000/DATASET_CARD.md) provides 1,000 human-recorded,
15-second audio clips (253 begin-responding and 747 continue-waiting labels),
their collection order, hashes, and 10,752-dimensional DistilHuBERT features.
The data owner authorized public distribution. Audio retains spoken content
and voice characteristics; this corpus is not a differentially private release.

The [user-correction summary](data/user_corrections_dp_v1/DATASET_CARD.md) contains
16 noisy joint weights over eight public acoustic prototypes and two correction
labels. Its mechanism uses participant-level differential privacy with epsilon
1 and delta 0 under add/remove-participant adjacency. It contains neither
individual study feature vectors nor generated observation rows. It supports
inspection of a coarse acoustic/feedback distribution, with substantial
resolution and noise limitations; it does not establish a useful replacement
for the original feature-label training data. Individual user-study recordings,
transcripts, features, and trajectories remain private. See [data documentation](data/README.md).

## Run locally

Use Python 3.11 in an isolated environment. Run the following from this directory:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python scripts/init_local_env.py
python -m uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8765
```

The initializer creates independent local secrets and selects mock timing, browser speech, and disabled recruitment. It leaves an existing `.env` untouched. This mode exercises the interfaces and protocol; it does not reproduce the paper's acoustic predictions.

- Desktop: `http://127.0.0.1:8765/?lang=en`
- Mobile companion: `http://127.0.0.1:8765/companion?lang=en`
- Operator dashboard: `http://127.0.0.1:8765/admin?lang=en` (requires the locally generated admin token)

For acoustic inference, install a suitable PyTorch build, obtain the DistilHuBERT artifacts, and configure `AGENT_MODE=distilhubert` with a validated release manifest. ASR/TTS models and conversational API credentials are configured separately. Participant checkpoints, deployed credentials, recordings, and conversation text are not distributed. See [implementation details](docs/IMPLEMENTATION.md).

## Validation

```sh
python -m pytest tests/test_consent_evidence.py tests/test_agent.py tests/test_dataset_release.py tests/test_distilhubert_online.py tests/test_study_analysis.py tests/test_block_schedule.py tests/test_runtime_isolation.py -q
node --check static/app.js
node --check static/companion.js
node --check static/i18n.js
```

The selected tests cover consent evidence, export fields, head updates and freezing, checkpoint integrity, active blocks, and runtime isolation. Fixtures use temporary storage and test-only credentials. Additional integration tests and simulation utilities are included; optional model and speech providers require separate setup.

## License

The repository retains its [CC BY-NC 4.0 license](LICENSE). Third-party models, libraries, and media retain their respective licenses and must be obtained separately.
