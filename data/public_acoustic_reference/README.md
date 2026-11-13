# Public acoustic reference

`reference.npz` contains a transformation and eight acoustic prototypes fitted
exclusively to the authorized 1,000-record pilot. No user-study record affects
this file. Load it with `allow_pickle=False`.

Processing: L2-normalize each raw 10,752-dimensional vector; apply a standard
scaler fitted on the public pilot; calculate a logistic score (C=0.1) and the
first principal-component score. Public-pilot score quartiles define four
categories; the public median principal-component score defines two categories.
The cell index is `2 * score_category + acoustic_category`. Each prototype is
the mean raw acoustic vector of the public pilot samples in that cell.

Arrays: `mean`, `scale`, `weight`, `bias`, `score_edges`,
`acoustic_direction`, `acoustic_edge`, `prototypes`, and `public_cell_counts`.
Prototype identifiers 0–7 index rows of `prototypes`. They are not participant
identifiers. Reference fitting uses all public pilot labels and therefore must
not be included in a claim of held-out evaluation on that same pilot.

Rebuild into a new file:

```sh
python scripts/dp_acoustic_release.py --public-features data/pilot_1000/acoustic_features.npz --output-reference new-reference.npz
```

The released reference's SHA256 is bound in the DP summary manifest. Rebuilding
can vary numerically across library versions. The supplied reference is the
authoritative mapping for the supplied weights.
