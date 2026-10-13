# Business entity resolution — Amazon ML Challenge 2026

This pipeline uses only the supplied business records and training labels. It
retrieves a small set of plausible pairs, scores them with an MIT-licensed
LightGBM resolver, and writes the two required TSV files. It handles country labels
as strings and discovers every country in the input; France is included.

## Environment

Linux, Python 3.12, GCC/G++ with C++17 support (tested with GCC 13.3), about 16 GB
RAM, and at least 20 GB free disk space for inference or 70 GB for full training
and validation. No GPU or external business lookup is
needed. The packaged model can be used for inference without retraining.

If `training_device: gpu` is present in the packaged configuration, append
`--device gpu` when reproducing expanded training. It requires an OpenCL-capable
GPU; the current GPU fitting settings use double precision and 63 bins. Inference
still runs on CPU. A CPU retrain is supported but can follow a different fitting
trajectory.

If `first_stage_training_device` differs from `training_device`, also pass
`--teacher-device` with that value. A private remote CPU trial can reuse the
frozen GPU teachers and fit the second stage on CPU with 255 bins; its own
complete-context measurements are recorded before selection. Kaggle access is
not required to reproduce inference or CPU second-stage training.

If `blend_component_models` is present, the final model combines the unweighted
GPU resolver and the entity-weighted CPU resolver by averaging their raw
log-odds. Reproduce the combination from the included fitted components with:

```sh
.venv/bin/python code/business_entity_resolution/src/blend_model.py \
  --first code/business_entity_resolution/models/unweighted_component_model.txt \
  --second code/business_entity_resolution/models/weighted_component_model.txt \
  --second-weight 0.5 --output work/reproduced_blend_model.txt
```

To retrain the components, run the expanded training sequence separately with
`--device gpu` for the unweighted model, and `--device cpu --teacher-device gpu
--entity-weighted` for the weighted model. Both use the same frozen GPU teachers,
aliases, candidate population and feature definitions. Calibrate the combined
model's decisions with complete context before prediction. CPU inference
automatically consumes the shipped combined model without a special option.

If `name_only_candidate_limit` is present, also pass its value as
`--name-only-candidate-limit` and the configured `name_only_gap` as
`--name-only-gap` to expanded training. This expands retrieval only for target
records whose address is missing; inference reads both values from the model
configuration automatically.

From the unzipped submission directory (containing `code/` and `output/`):

```sh
python3 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce from the supplied data

Pass the directory containing the supplied `train/` and `test/` folders:

```sh
.venv/bin/python code/business_entity_resolution/src/prepare.py \
  --data /absolute/path/to/student_resource/dataset --workers 3
.venv/bin/python code/business_entity_resolution/src/run.py --stage train
.venv/bin/python code/business_entity_resolution/src/run.py --stage predict
```

If the packaged configuration also has `copies: true`, add `--copies` to the
expanded training command. This adds four features describing other candidates
with the same normalized name and address, including their source and average
first-stage probability. The current query is excluded from these counts.

If the packaged configuration has `entity_weighted: true`, also add
`--entity-weighted` to expanded training. It weights fitting comparisons by the
inverse of the reference's true match count, with a minimum count of one, so
references with many matches do not dominate fitting. This option changes
training only; the final decision still uses the calibrated model probability.

The basic training command fits the first-stage models and a collective resolver.
If the packaged configuration has `tfidf: true`, reproduce the expanded resolver
with the following sequence instead. The first stage fits about 9% of references;
the expanded second stage fits about 29%. By default, normalization uses the
original fitting group. If the shipped configuration has `expanded_aliases:
true`, add `--expanded-aliases --alias-fitting-stop 1000` to the expanded training
command. That variant fits its dictionaries on groups 0–19, 30–99 and 200–999;
groups 20–29 and 100–199 are still excluded. Groups 105–109 remain reserved for a
fresh audit.

If `density_matched: true` is present, also add `--density-matched` to the expanded
training command. It samples the training reference population to match the
supplied test target/reference ratios, keeping all target records and every
development/audit reference. Retrieval and features are recomputed for that
population. Test inference always includes the full supplied test population.

If `strict_audit_targets: true` is present, add `--strict-audit-targets`. The
expanded training command refits three first-stage teachers for 1,600 iterations
on the baseline's original feature matrix, excluding all targets belonging to
reserved reference groups 100–199, even as negative examples. The second stage
also excludes those target rows from supervised fitting. Their labels are used
only to identify records to withhold, never as fitting examples or to choose
features, thresholds or model parameters.

If `teacher_candidate_population: current_expanded` is present, also add
`--teacher-current-candidates`. The strict first-stage teachers then use the
current expanded candidate matrix and its sampled reference population. The
baseline reports still provide the separately measured country-transfer result.

If `context_decision_calibration` is present, reproduce the final calibration
after expanded training and before prediction:

```sh
.venv/bin/python code/business_entity_resolution/src/subgraph_audit.py \
  --model-work work --audit-work work/calibration_context \
  --report reports/context_calibration_initial_audit.json \
  --audit-start 100 --audit-buckets 5
.venv/bin/python code/business_entity_resolution/src/calibrate_context.py \
  --model-work work --audit-work work/calibration_context --reports reports
```

This uses frozen final models with complete related-record and competing-reference
context. Groups 100–102 choose cutoffs; groups 103–104 are development measurements.
Their labels remain excluded from both model fitting stages and alias learning.

```sh
.venv/bin/python code/business_entity_resolution/src/prepare.py \
  --data /absolute/path/to/student_resource/dataset --work work/baseline
.venv/bin/python code/business_entity_resolution/src/run.py \
  --stage train --work work/baseline --reports reports/baseline
.venv/bin/python code/business_entity_resolution/src/prepare.py \
  --data /absolute/path/to/student_resource/dataset --expanded-sample
.venv/bin/python code/business_entity_resolution/src/run.py \
  --stage train --expanded --baseline-work work/baseline \
  --baseline-reports reports/baseline
.venv/bin/python code/business_entity_resolution/src/run.py --stage predict
```

Intermediate normalized files, candidate features, and fitted models are saved
in `work/` alongside `code/`. Validation results are saved in `reports/`.
Predictions are written to `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. Processing countries sequentially keeps the memory
requirement manageable; all countries share the fitted matching models.
Full training and prediction take several hours on a CPU of this size.

For inference with the shipped fitted model, run preparation with `--split test`
and then `run.py --stage predict`. The inference command imports the packaged
models, normalization aliases, and configuration into `work/` automatically.
Compilation flags and the feature schema are selected from that configuration;
the basic and expanded model formats are both supported.

Validate both outputs with the organizer's supplied validator:

```sh
python3 /absolute/path/to/student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir /absolute/path/to/student_resource/dataset/test --check-ids
```

## Pipeline and validation

- `prepare.py`: ASCII transliteration, punctuation cleanup, common legal-form
  and street/state abbreviation normalization. Record IDs are never features.
- `learn_aliases.py`: learns recurring token spelling and transliteration
  variants from fitting entities only, using conservative support and agreement
  thresholds. It does not use held-out entity labels for alias fitting.
- `learn_whole_aliases.py`: whole-name variants require at least three distinct
  fitting references and 95% agreement. Whole names are not memorized from test
  labels or external business records.
- `engine.cpp`: country-aware inverted indexes for names, addresses, rare words,
  phonetic tokens, name trigrams, combined name words, building numbers with
  address words, and address word pairs. A deterministic similarity pass reduces
  candidates before the final model. C++ exports exactly the pairs the model
  scores, including rejected pairs.
- Matching features preserve mixed building numbers, floor/unit distinctions,
  branch directions, and the original name alongside broad normalization.
  Weighted and fuzzy token similarities help distinguish business names that
  share generic words.
- `train.py`: entity-disjoint fitting, threshold tuning, local macro F0.5
  evaluation including singletons, and a US-to-India country transfer check.
  The sampled entity groups are fixed by CRC32 of the identifier; identifiers
  are used only for reproducible sampling and join operations.
- `crossfit.py`: three entity-disjoint first-stage models provide out-of-fold
  fitting predictions. The second stage adds related-record evidence and
  competing-reference probabilities. Query records are excluded from their own
  related-record prototypes.
- `expanded_train.py`: reuses those frozen first-stage models and fits more
  second-stage references. It adds six word/character TF-IDF similarities;
  frequencies are calculated only from the supplied reference records and are
  scoped by country. Training feature construction processes countries
  separately while preserving every candidate and competitor. Prediction runs
  in chunks to bound memory.
- `collective_batch.py`: bounds the number of active reference records in
  memory, retains their complete scoring context, and saves each finished
  group. Byte equality with unbatched feature construction and inference is
  checked on a three-country supplied-record fixture.
- `training_runtime.py`: constructs fitting matrices on disk and saves a
  checkpoint every 100 boosting rounds. Restarting the same experiment resumes
  a verified checkpoint; changed input signatures are rejected. A resumed
  stochastic fit is evaluated again before model selection.
- `decision.py`: validates separate cutoffs for a reference's first match and
  its additional matches. This rule is adopted only when it improves tuning
  macro F0.5 by more than 0.0003.
- `verify.py`: native/Python probability agreement, pair and feature alignment,
  independent related-record aggregates, learned-name features, and TF-IDF
  similarity checks on supplied-record fixtures.
- `export.py`: reports every scored candidate and selects the highest-scoring
  reference for each target above the chosen threshold. Every Source 1 entity
  receives a row, including entities with no final matches.
- `run.py`: compilation, country discovery, sequential retrieval, model fitting,
  inference through LightGBM's native API, and export.

The macro F0.5 calculation is `1.25*TP/(predicted_count + 0.25*true_count)` per
reference. A true singleton with no predicted matches scores 1; a false merge
on a singleton scores 0. Thresholds and blocking cutoffs are tuned on one group
of entities, then evaluated on a separate group. The final model is refitted
after recording the local evaluation.

Local development validation is not a private leaderboard score. There are no
supplied labels for France, so its accuracy cannot be established locally.
Development examples were inspected while designing features, so those results
are labeled as development evaluation. See the submission methodology document
and included reports for measured results and the frozen-model fresh audit.

## Model license

The matching model is a small gradient-boosted tree model trained from scratch
with [LightGBM](https://github.com/microsoft/LightGBM), which is MIT licensed.
No pretrained language model, external business database, geocoding API, or
external entity-resolution service is used. Python package licenses are
separate from the model license; see their published metadata.

### Deadline private CPU component

The optional private CPU trial is reproduced by `src/export_weighted_training.py` followed by `src/train_weighted_private.py` in a private Kaggle script notebook attached to the numeric export. The private CPU component uses the verified bundled Kaggle LightGBM4.6.0 with internet disabled; local inference uses LightGBM4.7.0 and must pass native/context checks. The final stage uses inverse true-count weights, 127 bins, learning rate0.05, a maximum of800 rounds and60-round stopping patience; its own logloss selects the final refit iteration. Kaggle may unpack gzip uploads, so the training script accepts compressed or unpacked files and verifies their dimensions and metadata hash. No credentials, IDs or business text are included in this compute input. Its challenge score is measured locally with the frozen complete-context protocol; remote logloss is not a leaderboard score.
