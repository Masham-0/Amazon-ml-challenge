# Team Shawarma submission history

| Version | Upload time (IST) | Public macro F0.5 | Evidence |
|---|---|---:|---|
| v3, 160-feature collective model | 26 September 2026, 18:29 | 0.975816 | User confirmed the upload time; supplied leaderboard screenshot shows rank 501. |

The uploaded files and their consistent model package are preserved in
`work/submission_v3/`. Local development macro F0.5 for that model is 0.982246895.
The public result demonstrates that local development performance is optimistic
for the leaderboard population. Country weighting, unseen France, population
density, and feature/model transfer remain possible contributors; none has yet
been isolated using test labels, which are unavailable.

The v4 model, inference, bounded official validation, and archive inference checks
are complete. Its immutable submission is preserved in `work/submission_v4/`.

## v4 — 26 September, 23:01 IST

User screenshot shows **0.977**, rounded to three decimals, evaluated. Previous screenshot showed v3 0.976 rounded, with exact v3 score 0.975816 known. Exact v4 score is not available yet. Upload: work/submission_v4/matching_results.tsv; SHA256 0f5f85c0ae4db507014f15a72266996686145a5ad8c36bdf00c038df9d338431. Local development 0.9845043889. The public improvement is modest; local gain did not transfer fully.

## Pending v9 / private CPU comparison — 27 September

v9 broadens fitting-only dictionaries, expands only the missing-address candidate budget, uses the test-like reference density, and refits all three first-stage teachers with strict reserved-target exclusion. A private Kaggle CPU trial is queued on the same frozen numeric feature matrix with entity weights and 255 bins. One fixed 50/50 raw-log-odds ensemble is also prepared; its numerical identity passed a supplied-record fixture, which is not a challenge-score measurement.

Final-model decisions are calibrated on complete scoring context: CRC100–102 for tuning and CRC103–104 for development. CRC105–106 are reserved for a fresh audit and will not select the model. The new submission will be published only after native/context checks, full prediction generation, organizer ID validation and actual extracted-archive inference. There is no new public score yet.

## Verified v9 — 27 September 22:45 IST

Immutable upload: `/home/mash/Projects/ml-challenge/work/submission_v9/matching_results.tsv`. Local development macro F0.5 **0.980665**; public leaderboard score unknown. Organizer ID validation, native/context equivalence and actual extracted-archive inference passed. Match SHA256 `028008079f7e3b81fa448293ef1440b557958d7ccad55c30bef0b65544805a70`; archive SHA256 `66348b640f6203085828341ab4ca2c58184502d5e28d82a150a2dcf87ada8275`.

## Public feedback — 27 September 23:16 IST

User screenshot: v9 submitted at 23:15 IST scored **0.976 rounded**, below the previous **0.977 rounded** v4. Local model selection did not predict this public regression. A separate deadline candidate using the completed Kaggle weighted model is being generated; public performance is unknown. v4 and v9 immutable artifacts remain preserved.
