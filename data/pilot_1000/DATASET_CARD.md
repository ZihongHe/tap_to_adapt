# Pilot speech corpus

The corpus contains 1,000 recordings from scripted speech interactions.
The recordings and labels were collected and annotated by the data owner,
with collection assistance from a collaborator. Public distribution is authorized
by the data owner. The audio preserves spoken content and vocal characteristics.

## Files

- `audio/*.wav`: 1,000 mono PCM16 recordings, each 15 seconds at 16 kHz.
- `labels.csv`: event index, binary label, label name, relative audio path,
  duration in seconds, and SHA256 of each recording.
- `features/*.npz`: ten ordered parts, each with 100 recordings. Concatenating
  their arrays yields `features` of shape `(1000, 10752)`, `labels` of shape
  `(1000,)`, and `event_indices` of shape `(1000,)`.
- `MANIFEST.json`: corpus dimensions, model revision, and file hashes.

Label 1 indicates that the scripted scenario calls for beginning a response
(253 clips); label 0 indicates continuing to wait (747 clips). Labels encode
scenario intent. The event index preserves collection order.

## Evaluation

Sequential evaluation predicts before updating, following collection order.
The final evaluation interval comprises examples 801 through 1,000.
Use audio or feature values as predictive inputs; filenames contain label prefixes.
Fit additional preprocessing using the chosen training partition.
Independence between speakers is not established by splitting event indices.

## Acoustic representation

The encoder is `ntu-spml/distilhubert`, revision
`fa87d96265d6b7af66e112faff6ff44df419cec9`. The final hidden layer has 768 units.
Temporal means and population standard deviations over the whole clip and the
trailing 8, 4, 2, 1, 0.5, and 0.25 seconds yield 10,752 pooled values.
The supplied cache precedes model-specific standardization.

The cached features use CUDA float16 inference and float32 pooling. CPU
extraction uses float32; hardware and numerical precision can affect results.
Load the NumPy archive with `allow_pickle=False`.

The code and corpus use CC BY-NC 4.0.
