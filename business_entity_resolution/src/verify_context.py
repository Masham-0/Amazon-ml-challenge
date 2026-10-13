"""Compare the audit's complete-context subgraph against full inference."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import DTYPE
from export import DTYPE as SCORE_DTYPE
from expanded_train import predict_chunks
from decision import selected

def main(work,fixture,report):
    cfg=json.loads((work/'config.json').read_text());lib=Path(lgb.__file__).parent/'lib/lib_lightgbm.so';models=cfg.get('ensemble',['model.txt'])
    def retrieve(prefix,cohort=None):
        truth='-'
        if cohort is not None:
            truth=str(prefix)+'.membership.tsv';Path(truth).write_text(''.join(sid+'\t\n' for sid in sorted(cohort)))
        args=[str(work/'engine'),str(fixture/'source1.tsv'),str(fixture/'source2.tsv'),str(fixture/'source3.tsv'),str(prefix),truth,','.join(str(work/name) for name in models) if cohort is None else '-',str(lib) if cohort is None else '-',str(cfg['gate']),str(cfg['gap']),str(work/'aliases.tsv'),'India']
        if 'name_only_candidate_limit' in cfg:
            args+=['dump' if cohort is None else '-',str(cfg.get('candidate_limit',6)),str(cfg['name_only_candidate_limit']),str(cfg.get('name_only_gap',cfg['gap']))]
        elif cohort is None:args+=['dump']
        subprocess.run(args,check=True)
        return np.fromfile(str(prefix)+'.features.bin',dtype=DTYPE)
    def resolve(prefix,features,p):
        p.astype(np.float32).tofile(str(prefix)+'.prob.bin')
        subprocess.run([str(work/'collective'),str(prefix)+'.features.bin',str(prefix)+'.prob.bin',str(fixture/'source2.tsv'),str(fixture/'source3.tsv'),str(prefix)+'.refs.tsv',str(prefix)+'.resolved.bin',str(work/'aliases.tsv'),'score',str(work/'collective_model.txt'),str(lib),str(fixture/'source1.tsv')],check=True)
        return np.fromfile(str(prefix)+'.resolved.bin',dtype=SCORE_DTYPE)
    full_prefix=fixture/'context_full';full_features=retrieve(full_prefix);first_prob=np.fromfile(str(full_prefix)+'.scores.bin',dtype=SCORE_DTYPE)['p'];full=resolve(full_prefix,full_features,first_prob)
    ids=[line.split('\t',1)[0] for line in Path(str(full_prefix)+'.refs.tsv').open()];primary=set(ids[:20]);first_prefix=fixture/'context_first';first=retrieve(first_prefix,primary);cohort=primary|{ids[int(r)] for r in np.unique(first['ref'])};context_prefix=fixture/'context_complete';context_features=retrieve(context_prefix,cohort)
    probability=np.zeros(len(context_features),dtype=np.float64)
    for name in models:probability+=predict_chunks(lgb.Booster(model_file=str(work/name)),context_features['x'])/len(models)
    context=resolve(context_prefix,context_features,probability)
    primary_index=np.array([sid in primary for sid in ids]);fm=primary_index[full['ref']];cm=primary_index[context['ref']]
    def keys(rows):return (rows['ref'].astype(np.uint64)<<32)|rows['target']
    fk=keys(full[fm]);ck=keys(context[cm]);fi=np.argsort(fk);ci=np.argsort(ck);assert len(fk)>0 and np.array_equal(fk[fi],ck[ci]);assert np.array_equal(full_features['x'][fm][fi],context_features['x'][cm][ci])
    delta=float(np.max(abs(full['p'][fm][fi]-context['p'][cm][ci])));assert delta<1e-6
    fs=selected(full,full['p'],cfg)[fm][fi];cs=selected(context,context['p'],cfg)[cm][ci];assert np.array_equal(fs,cs)
    result={'primary_references':len(primary),'context_references':len(cohort),'full_pairs':len(full),'context_pairs':len(context),'primary_pairs':len(fk),'primary_candidates_identical':True,'primary_base_features_identical':True,'maximum_primary_probability_error':delta,'primary_final_decisions_identical':True,'labels_used':False}
    one_probability=np.zeros(len(first),dtype=np.float64)
    for name in models:one_probability+=predict_chunks(lgb.Booster(model_file=str(work/name)),first['x'])/len(models)
    one=resolve(first_prefix,first,one_probability);om=primary_index[one['ref']];oi=np.argsort(keys(one[om]))
    assert np.array_equal(fk[fi],keys(one[om])[oi])
    one_selected=selected(one,one['p'],cfg)[om][oi]
    one_lookup={int(k):float(p) for k,p in zip(keys(one),one['p'])}
    common=np.array([abs(float(p)-one_lookup[int(k)]) for k,p in zip(keys(full),full['p']) if int(k) in one_lookup])
    result['single_pass_comparison']={'pairs':len(one),'maximum_probability_difference_in_shared_pairs':float(common.max()),'shared_pairs_with_probability_difference_above_1e_6':int((common>1e-6).sum()),'primary_decisions_differing_from_full_context':int((fs!=one_selected).sum()),'note':'No truth labels used; this measures context differences, not their direction of score bias.'}
    report.parent.mkdir(exist_ok=True,parents=True);report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--fixture',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);a=ap.parse_args();main(a.work,a.fixture,a.report)
