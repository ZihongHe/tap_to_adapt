# Research data

| Directory | Source and contents | Interpretation |
| --- | --- | --- |
| `pilot_1000` | 1,000 authorized pilot recordings, binary scenario labels, ordered manifest, and acoustic features | Original pilot observations, publicly released with authorization |
| `public_acoustic_reference` | Eight prototypes and a feature-to-cell transformation fitted only on `pilot_1000` | Public reference representation, not user-study samples |
| `user_corrections_dp_v1` | Noisy joint weights derived from consented user-study replay caches | Participant-level DP statistics, not individual records or synthetic observations |
| `user_corrections_controlled_v1` | Encrypted 926 native acoustic-feature/correction-label pairs from 43 eligible participants | Original observations; authorized key access required |

The pilot's filenames retain their original event order and label prefix for
compatibility with the sequential evaluation. The `sim_data` name in earlier
code denotes offline replay; it does not establish synthetic audio provenance.

The public pilot includes real voices and spoken content. Its authorization is
separate from user-study consent. Public user-study derivatives require both
the formal derived-data permission and the session-interface academic-release
option. Study recordings, text, model states, timestamps, demographics, and
source identifiers are not shipped. High-dimensional feature/label pairs and
release-local random grouping are distributed only as authenticated ciphertext.

Real user-study feature access requires a purpose-specific request to the
corresponding author, the applicable participant permission, privacy review,
and a data-use arrangement. Access is not automatic and is not promised for
records without the necessary authorization. Do not attach participant data to
GitHub issues. Encryption alone does not anonymize decrypted features.

Data use is subject to the repository's CC BY-NC 4.0 license and applicable
third-party rights. The original study results use original records; noisy
public statistics must not be substituted into those reported results.
