# Authorized pilot speech corpus

This corpus contains 1,000 recordings from scripted prototype interactions.
The first author led collection and primarily assigned the scenario-intention
labels, with an external collaborator assisting collection. Labels received
no independent review. The data owner authorized public distribution of this
corpus. These are human-recorded interactions, not a differentially private
synthetic dataset. Voices and spoken content remain audible.

## Files

- `audio/*.wav`: 1,000 mono PCM16 recordings at 16 kHz, 15 seconds each;
  480,044,000 bytes in total. Only standard WAV format and audio-data chunks
  are included. Audio sample values are unchanged.
- `labels.csv`: event index (1–1,000), binary label, label name, relative audio
  path, duration in seconds, and SHA256 of each WAV file.
- `acoustic_features.npz`: `features` (1,000 × 10,752), `labels` (1,000),
  and `event_indices` (1,000). Load using `allow_pickle=False`.
- `MANIFEST.json`: counts, encoder identity/revision, and archive/label hashes.

Label 1 means the scripted context called for beginning a response (253 clips).
Label 0 means continuing to wait (747 clips). These scenario labels differ
from actual user-study corrective taps and should not be conflated with them.
There are no speaker identifiers or defensible participant-disjoint splits in
this corpus. Do not interpret event-disjoint evaluation as speaker generalization.

The paper's sequential protocol predicts before updating in collection order.
Its final metrics cover examples 801–1,000 during continuing learning. This is
not a conventional frozen-model holdout. Preserve event order when reproducing
that protocol. Labels are visible in filenames, so models must consume audio
or acoustic arrays, not filenames as predictive features.

## Acoustic representation

The encoder is `ntu-spml/distilhubert`, revision
`fa87d96265d6b7af66e112faff6ff44df419cec9`. Its last hidden layer has 768 units.
For the full clip and trailing 8, 4, 2, 1, 0.5, and 0.25 seconds, concatenate
the elementwise temporal mean and population standard deviation, yielding
7 × 2 × 768 = 10,752 values. These cached features precede per-model
normalization. The pinned encoder is obtained separately under its own terms.

`scripts/extract_public_pilot_features.py` regenerates this representation.
Small numerical differences can occur across hardware/library versions. The
provided matrix has a file hash in the manifest and corresponds to the supplied
ordered recordings. No user-study replay features occur in this archive.

The repository's CC BY-NC 4.0 license applies to the distributed corpus.
