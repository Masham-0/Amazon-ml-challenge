"""Write measured methodology and package code, model, and checked outputs."""
import argparse,json,shutil,zipfile,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def main(work,reports):
    validation=json.loads((reports/'validation_report.json').read_text());submission=json.loads((ROOT/'reports/submission_report.json').read_text());config=json.loads((work/'config.json').read_text())
    for name,digest in submission.get('model_hashes',{}).items():
        if hashlib.sha256((work/name).read_bytes()).hexdigest()!=digest:raise ValueError('Output/model package mismatch: '+name)
    code=ROOT/'code/business_entity_resolution';models=code/'models';models.mkdir(exist_ok=True)
    for path in models.iterdir():
        if path.is_file():path.unlink()
    names=['config.json','aliases.tsv']+config.get('ensemble',['model.txt'])
    if config.get('collective'):names+=['collective_model.txt']
    for name in config.get('blend_component_models',[]):
        assert hashlib.sha256((work/name).read_bytes()).hexdigest()==config['blend_component_sha256'][name]
        names.append(name)
    for name in names:shutil.copyfile(work/name,models/name)
    license_text='''MIT License

Copyright (c) 2026 Team Shawarma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    (models/'LICENSE').write_text(license_text)
    (code/'LICENSE').write_text(license_text)
    copied=code/'reports';copied.mkdir(exist_ok=True)
    for path in copied.iterdir():
        if path.is_file():path.unlink()
    for name in ['validation_report.json','remote_training_report.json','blend_report.json','selected_private_resolver.json','deadline_model_selection.json','collective_validation_report.json','decision_validation_report.json','context_decision_validation_report.json','context_calibration_initial_audit.json','fresh_audit.json','fresh_audit.entities.tsv','frozen_model_comparison.json','baseline_comparison_audit.json','trial_comparison_audit.json','baseline_comparison_audit.entities.tsv','trial_comparison_audit.entities.tsv']:
        if (reports/name).exists():shutil.copyfile(reports/name,copied/name)
    for name in ['dataset_summary.json','submission_report.json','runtime_verification.json','context_verification.json','country_partition_verification.json','reference_batch_verification.json','training_checkpoint_verification.json','bounded_submission_validation.json','copy_append_verification.json','inference_feature_cache_verification.json','context_calibration_verification.json','query_context_equivalence.json','private_selection_verification.json','kaggle_export_verification.json','full_reference_export_verification.json','blend_model_verification.json','alias_learner_verification.json','package_verification.json','submission_validation.log','public_research.md','submission_history.md','public_submission_diagnosis.json','leakage_checks.json']:
        if (ROOT/'reports'/name).exists():shutil.copyfile(ROOT/'reports'/name,copied/name)
    h=validation['heldout'];b=validation['blocking'];c=validation['countries'];transfer=validation['country_transfer_US_to_India']
    total_targets=4887273+5082316;cartesian=submission['test_reference_entities']*total_targets;reduction=1-submission['scored_candidate_pairs']/cartesian
    matching_rule=f"The matching threshold **{validation['threshold']:.6f}** is chosen by directly optimizing macro F0.5, including singleton entities. Each target is assigned to its highest-scoring reference only when its score exceeds the threshold."
    if config.get('adaptive_thresholds'):
        cut=config['adaptive_thresholds'];matching_rule=f"Target ownership is resolved by the highest model score first. The highest eligible candidate of a reference uses cutoff **{cut['first_match']:.2f}**, and its additional candidates use cutoff **{cut['additional_matches']:.2f}**. These cutoffs are chosen by directly optimizing macro F0.5, including singleton entities."
    evaluation_roles='Groups 20–24 tune decisions; groups 25–29 measure development performance. Development errors were inspected to improve features, so this is not presented as an untouched final test. The final model is refitted on the sampled labeled data after these development measurements.'
    if config.get('context_decision_calibration'):
        evaluation_roles='Groups 20–24 select the initial boosting iteration and sampled decision cutoffs; groups 25–29 support inspected development. The final matching model is refitted including groups20–29. Final decision calibration then uses frozen models with complete competitor and related-record context: groups100–102 choose cutoffs, and groups103–104 produce the headline development results below. Both fitting stages and alias learning exclude groups100–199. This calibration followed public-score feedback and is not described as an untouched final test.'
    doc=f'''# ML Challenge 2026: Business Entity Resolution — Shawarma

**Team:** Shawarma  
**Leader:** Mohammad Masham  
**Members:** Ahaan Avi Arya, Krishna Agarwal, Yash Kumar  
**Submission date:** {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%d %B %Y')}

## 1. Executive Summary

We combine country-aware inverted retrieval, conservative spelling normalization,
and a supervised LightGBM pair classifier. Local entity-disjoint development
validation achieved **macro F0.5 = {h['macro_f05']:.6f}**; this is a validation
measurement, not a public or private leaderboard result. The final candidate set
contains **{submission['mean_candidates_per_reference']:.2f} candidates per
reference on average**, and includes exactly the pairs scored by the model.

## 2. Methodology

The supplied training set contains 2,206,821 reference businesses, 10,320,219
Source 2/3 records, and 123,247 true singleton references. The test set contains
1,732,544 references and 9,969,589 targets. France contributes 259,452 test
references and has no supplied labels. Names contain legal-form changes, spelling
errors, transliterations, domain names, unrelated trade names, and missing fields;
addresses contain abbreviations, reordered components, additions, and omissions.

We transliterate and normalize strings, standardize common legal and address
forms, and learn recurring token spelling variants from labeled fitting entities
only. No business identity database, external entity lookup, geocoding API, or
external business data is used. Country strings are discovered from the input,
used to scope retrieval, and never restricted to the training countries.

## 3. Candidate Generation

We index normalized names and addresses, name/address words, name phonetic
tokens and trigrams, combinations of name words, number/address-word
combinations, and adjacent address words. Querying rare keys recovers matches
even when a single common token provides little information. Posting limits and
query budgets prevent extremely common keys from producing enormous blocks.

A cheap similarity pass examines at most {200 if config.get('wide_blocking') else 100} retrieved references per target,
then retains at most {config.get('candidate_limit',6)} candidates per addressed target, and at most
{config.get('name_only_candidate_limit',config.get('candidate_limit',6))} candidates per target without an address, above an absolute cutoff and within a
margin of the best candidate. The final absolute cutoff is
**{validation['gate']:.2f}**, and the margin is **{validation['gap']:.2f}**. These
cutoffs are chosen during development to reduce candidate count while
preserving macro F0.5. This deterministic pass occurs **before** final model
inference; the reported candidates are never trimmed using model predictions.

On held-out development entities, candidate pair recall is
**{b['pair_recall']:.6f}**, and the macro F0.5 ceiling imposed by blocking is
**{b['macro_f05_ceiling']:.6f}**. Test inference scores
**{submission['scored_candidate_pairs']:,} pairs**, giving an all-pairs reduction
ratio of **{reduction:.9f}**. Median and 95th-percentile candidate counts per
reference are {submission['median_candidates_per_reference']:.0f} and
{submission['p95_candidates_per_reference']:.0f}, respectively.

## 4. Matching Model

The final model is an MIT-licensed LightGBM gradient-boosted tree classifier
trained from scratch, with {len(validation['feature_importance'])} numeric features and
{validation['best_iteration']} boosting iterations. It is well below the
8-billion-parameter limit. Features include name and address token overlap,
character trigrams, original and sorted-string similarities, phonetic name
similarity, number agreement, missing-field indicators, retrieval evidence,
candidate rank, and the gap to competing candidates. IDs are not model inputs.

{matching_rule} A
reference may receive zero, one, or many targets from either source.

## 5. Results and Error Analysis

References are sampled using CRC32(identifier) modulo 1,000. Alias fitting uses
groups {validation.get('alias_fitting_reference_buckets','0–19, 30–99')}. First-stage classifiers use groups 0–19 and
30–99. {evaluation_roles}
Aliases are learned only from fitting entities. Evaluation compares full match sets against all supplied
ground truth, including matches not recovered by blocking.

| Development evaluation | Macro F0.5 |
|---|---:|
| Overall ({h['entities']:,} entities) | {h['macro_f05']:.6f} |
| US | {c['US']['macro_f05']:.6f} |
| India | {c['India']['macro_f05']:.6f} |
| US-only classifier evaluated on India, with shared fitting aliases | {transfer['macro_f05']:.6f} |

Overall pair precision is **{h['pair_precision']:.6f}**, pair recall is
**{h['pair_recall']:.6f}**, and singleton correctness is
**{h['singleton_correct_fraction']:.6f}**. The standard error of the local macro
score is {h['standard_error']:.6f}; the approximate sampling interval does not
account for distribution shift to France.

Observed difficult cases include generic names with little address information,
shared addresses, severe spelling/transliteration errors, and trade names that
share little text with the reference. Some missing-field cases are intrinsically
ambiguous. Local development metrics cannot establish the unknown France score
or guarantee a private leaderboard result. We did not access test labels.

## 6. Reproduction and Submission

The package contains both output TSVs, complete source under
`code/business_entity_resolution/src/`, pinned dependencies, a reproduction
README, the fitted model and its configuration, and measured evaluation reports.
The pipeline processes countries sequentially and queries in parallel to keep
memory manageable. The native inference path and parallel feature extraction
are checked against independent Python/scalar execution on actual supplied
records. The organizer's validator is run with ID-existence checks on both final
files. Every test reference, including France and predicted singletons, has
exactly one output row. Final matches are a subset of the scored candidates.

During the challenge upload only `output/matching_results.tsv`; submit this
complete archive as the final reproducible solution package.
'''
    extra='\n## Additional model and validation details\n\n'
    if (ROOT/'reports/bounded_submission_validation.json').exists():
        extra+='To bound validation memory, the organizer\'s unchanged per-file validation functions run on batches of 75,000 output rows. Required-reference coverage and duplicate rows are checked globally before partitioning; every candidate and matched ID is checked against the complete supplied target ID set, and final matches are checked to be a subset of candidates. Output hashes and peak memory are recorded in the validation report.\n\n'
    extra+='Related-record feature construction processes bounded groups of references while retaining every candidate of each active reference and the first-stage scores of all competing references. Features and scores match the unbatched implementation byte for byte on a supplied-record fixture. Each completed group is checkpointed. Fitting uses a disk-backed matrix and saves boosting checkpoints every 100 rounds; interrupted fits are re-evaluated after resumption.\n\n'
    if config.get('collective') and not config.get('direct_full_reference_model'):
        extra+='The first stage has 134 numeric features. Three entity-disjoint folds supply out-of-fold probabilities for fitting references and averaged predictions for development evaluation. All rows for evaluation targets are excluded from first-stage fitting. The second LightGBM classifier adds 26 inputs, giving 160 features: probabilities of other records for the same reference, leave-one-record-out name/address/number similarities to up to five confident records, source counts, competing-reference probabilities, and a conservatively learned whole-name dictionary. No labels enter these inference features. Final first-stage inference averages the three packaged fold models.\n\n'
    if config.get('tfidf') and not config.get('direct_full_reference_model'):
        extra+='The expanded second stage fits references in groups 0–19, 30–99 and 200–399, and adds six similarities for a total of 166 inputs. These include name/address word and character TF-IDF cosine similarities, and name character inverse-frequency coverage. Frequency tables use only supplied reference records, are scoped by country, and are computed separately for training and inference. First-stage classifiers retain their original fitting groups. The final second stage is refitted including groups 20–29 after development results and iteration counts are recorded.\n\n'
    if config.get('expanded_aliases'):
        extra+=f"The expanded spelling and whole-name dictionaries use fitting groups **{config['alias_fitting_reference_buckets']}**. Groups 20–29 and 100–199 are excluded from dictionary learning. Both training retrieval and test inference use these same frozen dictionaries; all features are regenerated for their normalization.\n\n"
    if config.get('density_matched'):
        extra+='Training retrieval samples the supplied reference population to match the supplied test target/reference ratios by country, while retaining every target record. An independent identifier hash selects the population without labels; all development and reserved audit references are preserved. Candidate retrieval, reference frequencies, competitor features and related-record evidence are regenerated for the sampled population. The test reference population is always used in full. Development scores for this variant use this density protocol and should not be compared directly with scores from the original population.\n\n'
    if config.get('copies') and not config.get('direct_full_reference_model'):
        extra+='Four further inputs describe other candidate records sharing the same normalized whole name and address: their count, their Source 2 and Source 3 counts, and their mean first-stage probability. The current query is excluded. These inputs bring the second stage to 170 features and use no labels during inference.\n\n'
    if config.get('entity_weighted'):
        extra+='Second-stage fitting weights each comparison by 1/max(1, true match count of its fitting reference), rescaled to mean weight one. This gives more weight to references with few matches and aligns fitting more closely with the macro entity metric. The decision thresholds are calibrated separately; no true match counts are required during inference.\n\n'
    if config.get('wide_blocking'):
        extra+='Broader retrieval also indexes address character trigrams and conservatively learned whole-name variants. Its posting budget is 3,000, and the rough shortlist contains at most 200 references per target. These additional views precede model inference. The candidate output includes the complete scored set.\n\n'
    if 'name_only_candidate_limit' in config:
        extra+=f"Targets without addresses have a separate candidate budget of **{config['name_only_candidate_limit']}** and similarity margin **{config['name_only_gap']:.2f}**. Addressed targets retain the smaller budget of **{config.get('candidate_limit',6)}**. The expansion is applied before ML scoring and is present in both fitting and inference; every scored pair appears in the candidate TSV.\n\n"
    if config.get('training_device')=='gpu':
        extra+='Training uses the OpenCL GPU learner with double-precision histogram sums and max_bin=63. Both validation and final models use these settings. Test inference uses the native CPU predictor and does not require a GPU. GPU and CPU training trajectories are not claimed to be identical.\n\n'
    if config.get('first_stage_training_device')=='gpu' and config.get('training_device')=='cpu' and not config.get('direct_full_reference_model'):
        bins=validation.get('training_parameters',{}).get('max_bin',255)
        extra+=f'The frozen first-stage teachers use the OpenCL GPU learner with double-precision histogram sums and 63 bins. The selected weighted second-stage model was fitted in a private Kaggle CPU notebook with {bins} bins, using only derived numeric features and supplied training labels. No source record IDs or business text were uploaded for that trial. Local frozen-model calibration and audit results, rather than remote binary logloss, measure its challenge metric. The model and inference require no Kaggle credentials.\n\n'
    if config.get('blend_component_models'):
        weight=config['blend_second_weight']
        extra+=f"The final model combines component raw log-odds with fixed weights **{1-weight:.2f}** and **{weight:.2f}**, then applies the sigmoid. Leaf values are scaled and trees combined through LightGBM's native BoosterMerge API, yielding one standard model for native inference. The reported tree count is the sum of component tree counts, not a new fit of that many rounds. Original components and their hashes are included, and `blend_model.py` reproduces the combination. Tuning groups100–102 select among the measured models; fresh audit labels are not used for selection.\n\n"
    if config.get('direct_full_reference_model') or config.get('full_reference_auxiliary'):
        extra+='The full-reference CPU component uses all supplied labels for references in the frozen retrieval population, with fitting groups 0–19,30–99,200–999 and final refit groups 0–99,200–999. Owners of tuning groups20–29 and audit groups100–199 are excluded even as negative fitting candidates; audit targets also remain excluded from final refit. Its active inputs are the 134 direct comparison features. The remaining 36 schema columns are constant zero during fitting and never appear in its trees; their inference values cannot affect that component. It uses entity weights and 127 bins, with the iteration count selected on groups20–24. The private export contains numeric features and flags only. `export_full_reference_training.py` and `train_full_reference_private.py` reproduce this component. It uses no related-record probabilities as inputs; those contextual features still apply to a separately fitted collective blend component when included.\n\n'
    if config.get('teacher_candidate_population')=='current_expanded':
        extra+='The first-stage teachers were freshly fitted on this resolver\'s expanded candidate matrix and sampled reference population. Reserved audit targets are excluded even from negative fitting rows. Original baseline teachers are not reused by this variant.\n\n'
    if config.get('adaptive_thresholds'):
        cut=config['adaptive_thresholds'];extra+=f"Reference-level decision calibration uses cutoff **{cut['first_match']:.2f}** for the highest-scoring eligible candidate of a reference, and **{cut['additional_matches']:.2f}** for its remaining candidates. Target ownership is resolved first. This optional rule is selected on tuning references, not on the development evaluation or test labels.\n\n"
    if (reports/'fresh_audit.json').exists():
        audit=json.loads((reports/'fresh_audit.json').read_text());ah=audit['heldout']
        extra+=f"The frozen resolver is also measured on previously unused entity groups: macro F0.5 **{ah['macro_f05']:.6f}** over **{ah['entities']:,}** references. These labels were excluded from aliases, classifier training, calibration and threshold selection. Full-population candidate generation is used; a second pass recovers complete scoring context for the audit references and every competitor on an audit query. This reduction was verified to reproduce full inference exactly on supplied-record fixtures, without consulting their labels. Model hashes and detailed results are included in `reports/fresh_audit.json`.\n"
        if not audit.get('strict_audit_targets'):
            extra+='Audit target records may have appeared as negative fitting examples. This result is not described as a strictly record-disjoint audit.\n'
    if (reports/'frozen_model_comparison.json').exists():
        comparison=json.loads((reports/'frozen_model_comparison.json').read_text())
        extra+=f"A separate frozen-model comparison used the same sampled reference population and all supplied targets for both resolvers. Over **{comparison['entities']:,}** reserved references, the original resolver measured **{comparison['baseline_macro_f05']:.6f}** and the trial measured **{comparison['trial_macro_f05']:.6f}**. Complete competitor scoring context was recovered for both. This comparison informed model selection and is not presented as an unbiased evaluation after selection; France remains unlabeled. Detailed paired differences and exclusion protocols are included in the comparison reports.\n\n"
    (ROOT/'Documentation_template.md').write_text(doc+extra)
    package=ROOT/'Shawarma_submission.zip'
    paths=[ROOT/'Documentation_template.md',ROOT/'output/matching_results.tsv',ROOT/'output/candidate_pairs.tsv']
    paths+=sorted(p for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:archive.write(path,path.relative_to(ROOT))
    with zipfile.ZipFile(package) as archive:
        bad=archive.testzip()
        if bad:raise RuntimeError('corrupt archive member '+bad)
    print(package,package.stat().st_size,'bytes',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--reports',type=Path,default=ROOT/'reports');a=ap.parse_args();main(a.work,a.reports)
