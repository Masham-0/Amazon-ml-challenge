# Public research — 26 September 2026

We searched public challenge discussions and repositories for methodology,
dataset provenance, and leakage reports. We found no verified release of private
test labels in the searched sources. No external business records, competitor
predictions, competitor model weights, or leaked labels were incorporated.

[A public solution's experimental notes](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/FINAL_RECOMMENDATION.md)
report a local macro F0.5 of 0.98703 using candidate competition and collective
evidence. This is that author's reported validation result, not our result or a
verified leaderboard score. It motivated independent experiments with related
ideas; our code was written separately and assessed on the supplied labels.

[Its data analysis](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/DATA_ANALYSIS.md)
reports negative ID and row-order leakage checks. Our own checks on the local
files found zero train/test ID overlap in all three sources. Numeric ID rank
correlations in our training sample were 0.00112 and 0.00240 for Source 2 and
Source 3, respectively; see `leakage_checks.json`.

[Public submission discussion](https://www.reddit.com/r/hackathon/comments/1wpekwd/submissions_limit_for_amazon_ml_challenge_2026/)
primarily concerns upload delays, runtime, and submission limits. It does not
establish a reproducible high-scoring method or a label leak.

All reported Shawarma metrics are computed from the supplied training labels.
Public and private leaderboard scores require organizer evaluation.

Additional search, 26 September 2026:
- https://github.com/Sid-techweb/AmazonML-New reports a local 0.9783 resolver and improved Indic-script retrieval from a fitting-only transliteration dictionary. This is a self-reported development result, not a verified leaderboard score. It supports testing conservative learned text aliases; no code or predictions were imported.
- https://huggingface.co/datasets/logicalguy/amazon-ml-challenge-2026 lists the supplied challenge data. The search returned no verified private test labels.
- Additional public discussions concern upload access and submission limits. No verified shortcut to a private-test score above 0.99 was found.

## Deadline research — 27 September 2026

No searched source establishes the leading teams' methods or provides a verified
reproducible 0.99 public leaderboard solution. A high leaderboard score alone is
not evidence of cheating.

- [gojosatorou999/amazonml handoff](https://github.com/gojosatorou999/amazonml/blob/main/HANDOFF.md)
  and [summary](https://github.com/gojosatorou999/amazonml/blob/main/SUMMARY.md)
  explicitly describe the 99% claims as measurements on a small synthetic
  harness, and document that the implementation is not ready for the supplied
  full data on this memory budget. Useful ideas include multi-view retrieval,
  related-record evidence, and expected-F decisions; the claimed score is not
  a verified result on this competition's hidden labels.
- [Ayan's validation strategy](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/main/VALIDATION_STRATEGY.md)
  states that normalization dictionaries included validation labels. Its local
  0.98703 therefore cannot be treated as an independent estimate of hidden-test
  performance or as proof that the methodology reaches that leaderboard score.
- [Abimanyu-A/amazon_ml_challenge](https://github.com/Abimanyu-A/amazon_ml_challenge)
  and [Sid-techweb/AmazonML-New](https://github.com/Sid-techweb/AmazonML-New)
  report local experiments with reverse retrieval and transfer checks. These
  self-reports do not establish the leading teams' public scores.

Our independent tuning experiments rejected genetic decision optimization
because it reduced separate development macro F0.5 from 0.984504 to 0.984311.
Expected-F decisions also lost on the tuning set, despite a tiny development
gain, and were not selected. Broader name-only retrieval recovered 122 additional
development true pairs; this is a recall measurement, not a final model score.

The selected trial ideas are broader fitting-only text aliases, an increased
candidate budget for records with missing addresses, matching reference density
to the supplied test population, freshly fitted first-stage models, and explicit
exclusion of reserved audit targets. All code and measurements are local and use
the supplied records. No competitor outputs, models, weights, or private labels
were imported.

### Private compute trial, 27 September

The user authorized using their Kaggle account for a private CPU training run. The planned upload contains only derived numeric fitting features, binary supplied-training labels, fitting/tuning flags and numeric match counts; no raw business text or entity identifiers. Dataset and notebook privacy are set explicitly and the remote dataset privacy is verified before notebook dispatch. No Kaggle or external business dataset is an input. Credentials remain outside the submission. Remote binary logloss is not reported as a challenge score; any downloaded model requires local complete-context calibration before selection. Official client documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md and https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md .

Private compute setup: the live Kaggle server rejected the documented copyright-authors value. A tiny private bootstrap validated Other, with the original challenge restrictions preserved in its description, and metadata confirmed isPrivate=true. Official reference: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md . Numeric feature versions and both CPU notebooks remain private. The additional direct classifier uses 134 active features and full supplied owner labels on the frozen retrieved comparisons; it reserves groups100–199 and their targets. This is an internal method trial, not a known winner method or leaderboard result.
