# Controlled-access user correction pairs

This encrypted archive contains 926 measured acoustic-feature and correction-label
pairs from 43 participants in the analyzed user study. It preserves the original
feature values and labels for joint acoustic/feedback modeling. Downloading the
archive does not provide access to its plaintext. An authorized researcher needs
a separately supplied decryption key.

## Contents and provenance

The records come from the final native online-training replay caches, verified
against their saved SHA256 pointers. The caches retain at most 12 examples per
class per participant. The archive contains 470 interruption labels and 456
activation labels, with 10,752 standardized pooled DistilHuBERT features per row.
They are partial training reservoirs rather than complete conversations or all
decision opportunities. The frozen condition has no online replay cache.

Only analyzed participants with affirmative formal derived-data permission and
an affirmative latest interface academic-release option contribute. This corpus
is separate from the public 1,000-record pilot. Original source mappings remain
with the data custodian.

| Array | Meaning |
| --- | --- |
| `features` | `float32`, shape `(926, 10752)`, native standardized acoustic inputs |
| `labels` | `uint8`, shape `(926,)`, activation `1`, interruption `0` |
| `group` | `uint16`, shape `(926,)`, randomly assigned release-local participant group |

Manual activation pairs label 1 with the current acoustic input. Manual
interruption pairs label 0 with the stored input at the interrupted response's
initiation. Feedback can reflect timing, recognition, or response-content issues.
These are observed corrective actions, not independently annotated ideal timing.

Row order is independently shuffled. Enrollment identifiers, session identifiers,
timestamps, condition and demographic fields, waveforms, transcripts, model
weights, and private provenance files are absent. Random group identifiers allow
participant-disjoint evaluation; they do not establish anonymity. Native duplicate
observations and labels are preserved. Feature vectors are pooled over seven
trailing windows and retain the runtime's fixed standardization. Independent
pilot features use a different preprocessing stage; do not combine the arrays
without accounting for normalization.

## Research use and limits

Use participant-disjoint training and test partitions, fitting further transforms
on the training data only. The selected correction reservoirs support studies of
acoustic representations and corrective labels. Missing uncorrected opportunities
and event timestamps prevent prospective timing evaluation, temporal train/test
splits, and estimates of intervention rates from this archive alone. Reservoir
membership and class caps also alter the observed class distribution.

## Privacy and access

The file uses AES-256-GCM with a separately generated key and nonce. The key is
held outside GitHub. Decryption authenticates the complete archive. Removing
direct identifiers and encrypting transport/storage does not make the decrypted
acoustic features anonymous. They may retain speaker or linguistic information.
This archive has no differential-privacy guarantee. The separately published DP
summary has its own documented scope; it does not protect decrypted records or
the exact aggregate metadata in this card.

Request access through the corresponding-author contact in the
[preprint](https://arxiv.org/abs/2603.14449). State the research purpose, requested
use, data-security arrangements, and responsible institution. Access requires
consent-compatible review and a data-use arrangement covering secure storage,
no re-identification or conversation reconstruction, no redistribution, and
retention. Do not post participant data, keys, or credentials to GitHub issues.
The archive is a controlled-access resource; it is not covered by the
repository's public-data redistribution permission.

## Reading authorized data

Install `numpy` and `cryptography`. Keep the separately supplied binary key
outside this checkout. The following command validates the container and reports
array shapes without saving plaintext.

```sh
python scripts/read_controlled_features.py \
  data/user_corrections_controlled_v1/feature_label_pairs.aes256gcm \
  --key-file /secure/location/authorized-key.bin
```

Import `load_pairs` from that script for in-memory modeling. Researchers remain
responsible for protecting process memory, outputs, backups, and any derived
models. The public `MANIFEST.json` gives the ciphertext hash and array schema.
