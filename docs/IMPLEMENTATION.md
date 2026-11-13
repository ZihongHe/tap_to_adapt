# Implementation guide

## Interaction and learning

The desktop and mobile clients share a scheduled study session. Desktop taps supervise response initiation: activation labels the current waiting context positively; interruption labels the initiating context negatively. Eligibility checks associate each label with a valid decision. The frozen speech representation feeds a per-user head; CBPER balances replay classes and constrains updates toward the fixed base parameters. Recent Tap-History Adaptation (RTHA) uses corrective events available before the current decision and bounded timing adjustments. Content generation and conversational memory run separately from the timing path.

Core entry points are `app/distilhubert_online.py`, `app/tap_history.py`, `app/timing.py`, and `app/main.py`. `app/study_analysis.py` defines active-block analyses. `content/study_protocol.json` specifies the study schedule. `app/orchestration.py` implements assignment and session management.

## Configuration and external artifacts

The public configuration is a local template. Researcher contacts and institution fields must be configured for a new deployment. Consent documents are templates; they do not grant approval to recruit participants.

The local mock mode requires no participant files or remote service credentials. Real acoustic inference requires the frozen DistilHuBERT files and a hash-bound release manifest containing a compatible shared head and normalization parameters. The model loader verifies dimensions, hashes, and update settings. Shared and participant-specific checkpoints are not included in this distribution. Optional Whisper recognition, AISHELL-3 VITS speech synthesis, and conversational API access need separately obtained dependencies and configuration. The default background-audio manifest references media that must also be obtained separately.

`static/i18n.js` and the English JSON documents provide localization. Use `?lang=en` for English and `?lang=zh-CN` for Simplified Chinese.

## Validation scope

The README lists the focused checks run for this release. Additional historical flow tests cover earlier questionnaire and interface snapshots and may require adaptation to the current content. Optional-provider integration tests require their corresponding local models or services. Passing local mock tests does not establish deployment readiness or reproduce a user study.

## Data handling

The runtime stores research events separately from encrypted identity/compensation records. Public configuration contains no deployed keys, study database, raw speech, transcripts, or participant model states. Generic export utilities require explicit local input and output paths or configuration; their outputs require consent and disclosure review before redistribution. Coarse acoustic features and pseudonyms alone do not establish anonymity.
