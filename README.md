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
