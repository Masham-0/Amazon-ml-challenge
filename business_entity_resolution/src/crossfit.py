"""Train the collective resolver using entity-disjoint first-stage predictions."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import FEATURES,DTYPE,load_metadata,truth_data,metric,exclusive
ROOT=Path(__file__).resolve().parents[3]
SUPPORT='stage1_probability other_max_probability other_second_probability other_mean_probability other_count50 other_count80 other_count95 other_count99 other_source2_count80 other_source3_count80 sibling_name_lcs sibling_name_compact_lcs sibling_address_trigram sibling_address_lcs sibling_hybrid sibling_numbers_jaccard sibling_numbers_containment sibling_numbers_exact sibling_exact_name_count sibling_exact_address_count sibling_no_number_count sibling_best_name_weighted_probability sibling_best_address_weighted_probability'.split()
SUPPORT+='stage1_competitor_probability stage1_competitor_margin'.split()
SUPPORT+=['dictionary_name_lcs']
SECOND_FEATURES=FEATURES+SUPPORT
SECOND_DTYPE=np.dtype([('ref','u4'),('target','u4'),('x','f4',(len(SECOND_FEATURES),))])

def train(work,reports):
    ids,countries,groups=load_metadata(work);truth,pairs,sizes=truth_data(work,ids)
    rows=np.memmap(work/'train.features.bin',dtype=DTYPE,mode='r');rg=groups[rows['ref']];y=np.isin((rows['ref'].astype(np.uint64)<<32)|rows['target'],pairs)
    fitting=(rg<20)|((rg>=30)&(rg<100));tuneref=(groups>=20)&(groups<25);hold=(groups>=25)&(groups<30)
    owner=(pairs&np.uint64(0xffffffff)).astype(np.uint32);og=groups[(pairs>>np.uint64(32)).astype(np.int64)];order=np.argsort(owner);owner=owner[order];og=og[order]
    loc=np.searchsorted(owner,rows['target']);safe=np.minimum(loc,len(owner)-1);valid=(loc<len(owner))&(owner[safe]==rows['target']);tg=np.full(len(rows),999);tg[valid]=og[safe[valid]]
    base=lgb.Booster(model_file=str(work/'validation_model.txt'))
    params=dict(objective='binary',metric='binary_logloss',learning_rate=.05,num_leaves=127,min_data_in_leaf=100,feature_fraction=.90,lambda_l2=5,num_threads=6,verbosity=-1,seed=2026,deterministic=True,force_col_wise=True)
    rounds=base.current_iteration();oof=np.zeros(len(rows),np.float32);fold=rg%3;eval_or_unknown=~fitting
    for k in range(3):
        path=work/f'fold{k}.model.txt'
        if path.exists():model=lgb.Booster(model_file=str(path))
        else:
            fitmask=fitting&(fold!=k)&((tg==999)|(((tg<20)|((tg>=30)&(tg<100)))&(tg%3!=k)))
            print('crossfit fold',k,'pairs',int(fitmask.sum()),'rounds',rounds,flush=True)
            model=lgb.train(params,lgb.Dataset(rows['x'][fitmask],label=y[fitmask],feature_name=FEATURES),num_boost_round=rounds,callbacks=[lgb.log_evaluation(200)])
            model.save_model(str(path))
        p=model.predict(rows['x'],num_threads=6).astype(np.float32);oof[fitting&(fold==k)]=p[fitting&(fold==k)];oof[eval_or_unknown]+=p[eval_or_unknown]/3
        print('crossfit fold completed',k,flush=True)
    oof.tofile(work/'crossfit.raw.bin')
    lib=Path(lgb.__file__).parent/'lib/lib_lightgbm.so'
    subprocess.run([str(work/'collective'),str(work/'train.features.bin'),str(work/'crossfit.raw.bin'),str(work/'train_source2.norm.tsv'),str(work/'train_source3.norm.tsv'),str(work/'train.refs.tsv'),str(work/'collective.features.bin'),str(work/'aliases.tsv'),'train','-','-',str(work/'train_source1.norm.tsv')],check=True)
    second=np.memmap(work/'collective.features.bin',dtype=SECOND_DTYPE,mode='r');assert np.array_equal(second['ref'],rows['ref']) and np.array_equal(second['target'],rows['target'])
    fitmask=fitting&((tg==999)|(tg<20)|((tg>=30)&(tg<100)))
    par=dict(objective='binary',metric='binary_logloss',learning_rate=.04,num_leaves=63,min_data_in_leaf=100,feature_fraction=.9,lambda_l2=5,num_threads=6,verbosity=-1,seed=2026,deterministic=True,force_col_wise=True)
    ds=lgb.Dataset(second['x'][fitmask],label=y[fitmask],feature_name=SECOND_FEATURES);val=lgb.Dataset(second['x'][tuneref[rows['ref']]],label=y[tuneref[rows['ref']]],reference=ds)
    model=lgb.train(par,ds,num_boost_round=2200,valid_sets=[val],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(100)])
    raw=model.predict(second['x'],num_threads=6).astype(np.float32);p=exclusive(rows,raw)
    sweep=[{'threshold':float(t),**metric(rows,p,y,sizes,tuneref,t)[0]} for t in np.linspace(.3,.95,66)];best=max(sweep,key=lambda r:r['macro_f05'])
    report={'tuning':best,'heldout':metric(rows,p,y,sizes,hold,best['threshold'])[0],'countries':{},'first_stage_iterations':rounds,'best_iteration':model.best_iteration,'threshold':best['threshold'],'gate':.35,'gap':.1,'threshold_sweep':sweep,'feature_importance':dict(zip(SECOND_FEATURES,map(float,model.feature_importance(importance_type='gain'))))}
    for country in sorted(set(countries)):report['countries'][country]=metric(rows,p,y,sizes,hold&(countries==country),best['threshold'])[0]
    (reports/'collective_validation_report.json').write_text(json.dumps(report,indent=2));model.save_model(str(work/'collective_validation_model.txt'));np.save(work/'collective_validation_raw.npy',raw)
    print(json.dumps({k:v for k,v in report.items() if k not in ['threshold_sweep','feature_importance']},indent=2),flush=True)
    final=lgb.train(par,lgb.Dataset(second['x'][rg<100],label=y[rg<100],feature_name=SECOND_FEATURES),num_boost_round=model.best_iteration);final.save_model(str(work/'collective_model.txt'))
    baseline=json.loads((reports/'validation_report.json').read_text())
    if best['macro_f05']>baseline['tuning']['macro_f05']+.0001:
        config=json.loads((work/'config.json').read_text());config.update(collective=True,ensemble=[f'fold{k}.model.txt' for k in range(3)],threshold=best['threshold'],second_stage_features=SECOND_FEATURES)
        (work/'config.json').write_text(json.dumps(config,indent=2))
        baseline['first_stage_validation']={k:baseline[k] for k in ['heldout','tuning','countries','best_iteration']}
        baseline.update({k:report[k] for k in ['heldout','tuning','countries','best_iteration','threshold','feature_importance']});baseline['collective']=True;baseline['first_stage_iterations']=rounds
        (reports/'validation_report.json').write_text(json.dumps(baseline,indent=2));print('Collective model selected by tuning score.',flush=True)
    else:print('Base model retained by tuning score.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--reports',type=Path,default=ROOT/'reports');a=ap.parse_args();a.reports.mkdir(parents=True,exist_ok=True);train(a.work,a.reports)
