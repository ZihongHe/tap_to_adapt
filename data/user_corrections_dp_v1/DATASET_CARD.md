# Participant-private acoustic/correction summary

This file provides differentially private joint statistics derived from real
user-study replay caches. It is not a table of collected individual records
and contains no newly generated observation rows. Acoustic prototypes come
only from the separately authorized public pilot corpus.

## Contents and use

`weights.csv` has 16 rows, with `prototype_id` (0–7), `correction_label` (0 or 1),
and `noisy_weight`. Join prototype IDs to `prototypes` in
`../public_acoustic_reference/reference.npz`. The weights describe a coarse,
noisy joint distribution over acoustic cells and corrective labels. Use them
as nonnegative weights; do not interpret them as exact event or participant
counts. Zero weights can result from clipping a negative noise realization.

Label 1 denotes manual activation. Label 0 denotes manual interruption linked
by the runtime to the interrupted response's initiation audio. The source
caches retain at most 12 examples per class per participant, using bounded
replay reservoirs. This selection is not a census of corrections or decision
opportunities. It supplies no untapped negative examples, no per-user learning
curves, and no recording/event timestamp reconstruction. Frozen-condition
participants do not contribute an online replay cache.

The projection and privacy noise impose substantial information loss. This
summary is useful for inspecting the release mechanism and coarse joint
weights; it has not established useful predictive performance for acoustic
timing models and cannot replace the private feature-label records for
personalization experiments. The repository provides the original pilot for
audio/representation research and the DP summary as a separate resource.

## Consent and preprocessing

Only records within the paper's analysis cohort with affirmative formal
public-derived-data consent and an affirmative session-interface academic-release
preference enter this mechanism. Raw audio, transcripts, user identifiers,
individual embeddings, model weights, timestamps, group membership, and exact
contributor/sample counts are not released by this mechanism.

Native replay features are first returned to raw pooled-feature coordinates
using the fixed synthetic-development normalizer. The public-pilot reference
then maps each feature to one of eight cells. The reference and its boundaries
are fixed before accessing private feature-label pairs. No private normalization,
PCA component, classifier parameter, or acoustic prototype is published.

## Privacy mechanism

Neighboring input datasets differ by adding or removing all contributed records
of one participant. Each participant supplies at most 24 records. The complete
8 × 2 count vector therefore has L1 sensitivity at most 24. Independent Laplace
noise with scale 24 is added to every cell, giving epsilon=1, delta=0 for this
vector under the Laplace mechanism. Negative weights are clipped to zero as
postprocessing. Replacing one participant's entire contribution has the
conservative epsilon bound 2. Numerical sampling uses diffprivlib 0.6.6's
Laplace mechanism with system randomness and its finite-precision mitigation.

The bound concerns the entire released vector, not a separate epsilon budget
per cell: for neighboring count vectors h and h', the joint density ratio is
bounded by exp(||h-h'||_1 / 24), hence by exp(1). The fixed public reference
adds no privacy expenditure for user-study participants. Arbitrary processing
of this same noisy vector adds no further expenditure; recomputing noise for
another release does. The private ledger records this one release; noise is
not resampled to obtain more favorable values.

This mechanism does not make the original recordings anonymous, guarantee
that identification is impossible, or remove information disclosed elsewhere.
Its guarantee is local to the specified released statistic, not a claim that
the entire paper or all earlier repository versions are differentially private.
Original features remain private; encrypted storage is a separate safeguard.

`MANIFEST.json` records the parameters, file hash, and reference hash.
`scripts/dp_acoustic_release.py` implements the mechanism. Tests exercise
contribution bounds, input validation, and repeat-release safeguards. Tests do
not substitute for the mathematical sensitivity argument or independent audit.

References: [NIST SP 800-226](https://csrc.nist.gov/pubs/sp/800/226/final) and
[diffprivlib mechanisms](https://diffprivlib.readthedocs.io/en/latest/modules/mechanisms.html).
