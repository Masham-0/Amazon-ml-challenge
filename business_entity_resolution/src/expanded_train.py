"""Fit more labeled references with fixed, cross-fitted first-stage models."""
import argparse,json,shutil,subprocess,time,mmap,gc,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import DTYPE,FEATURES,load_metadata,truth_data,metric,exclusive
from crossfit import SECOND_FEATURES
from collective_batch import run_batched
from training_runtime import selected_dataset, train_checkpointed, signature
ROOT=Path(__file__).resolve().parents[3]
EXTRA_FEATURES='name_character_tfidf_cosine address_character_tfidf_cosine name_token_tfidf_cosine address_token_tfidf_cosine name_character_idf_coverage_ref name_character_idf_coverage_target'.split()
COPY_FEATURES='exact_attribute_peer_count exact_attribute_source2_peer_count exact_attribute_source3_peer_count exact_attribute_peer_mean_probability'.split()
FEATURE_NAMES=SECOND_FEATURES+EXTRA_FEATURES
SECOND_DTYPE=np.dtype([('ref','u4'),('target','u4'),('x','f4',(len(FEATURE_NAMES),))])

def predict_chunks(model,x,label=None):
    result=np.empty(len(x),dtype=np.float64)
    for start in range(0,len(x),100000):
        result[start:start+100000]=model.predict(np.ascontiguousarray(x[start:start+100000]),num_threads=6)
        if label and (start+100000)%1000000==0:print(label,'predicted',min(start+100000,len(x)),'/',len(x),flush=True)
    return result

def predict_evaluation(model,rows,x,reference_mask):
    # Every competitor for every evaluation query must be predicted. Queries
    # unrelated to evaluated references cannot affect their final decisions.
    targets=np.unique(rows['target'][reference_mask[rows['ref']]])
    selected=np.flatnonzero(np.isin(rows['target'],targets))
    result=np.full(len(rows),-1.,dtype=np.float64)
    print('Evaluation context pairs',len(selected),'of',len(rows),flush=True)
    for start in range(0,len(selected),100000):
        indices=selected[start:start+100000]
        result[indices]=model.predict(np.ascontiguousarray(x[indices]),num_threads=6)
    assert (result[reference_mask[rows['ref']]]>=0).all()
    return result

def build_features(work):
    """Process independent countries separately, preserving full context."""
    parts=[p for p in sorted(work.glob('train_part_*.features.bin')) if Path(str(p).removesuffix('.features.bin')+'.refs.tsv').exists()]
    candidate=json.loads((work/'candidate_config.json').read_text())
    population=work/candidate.get('reference_population_file','train_source1.norm.tsv')
    def native(features,probabilities,refs,output):
        run_batched(work/'collective',features,probabilities,work/'train_source2.norm.tsv',work/'train_source3.norm.tsv',refs,output,work/'aliases.tsv',population,feature_count=len(FEATURE_NAMES))
    if not parts:
        native(work/'train.features.bin',work/'crossfit.raw.bin',work/'train.refs.tsv',work/'collective.features.bin')
        return
    probabilities=np.memmap(work/'crossfit.raw.bin',dtype=np.float32,mode='r')
    alias_hash=hashlib.sha256((work/'aliases.tsv').read_bytes()).hexdigest()
    probability_hash=hashlib.sha256(probabilities).hexdigest()
    normalized=[[(work/f'train_source{s}.norm.tsv').stat().st_size,(work/f'train_source{s}.norm.tsv').stat().st_mtime_ns] for s in [1,2,3]]
    row_offset=0;ref_offset=0;outputs=[]
    combined=np.memmap(work/'train.features.bin',dtype=DTYPE,mode='r')
    for part in parts:
        prefix=str(part).removesuffix('.features.bin');refs=Path(prefix+'.refs.tsv');n=part.stat().st_size//DTYPE.itemsize
        original=np.memmap(part,dtype=DTYPE,mode='r')
        for start in range(0,n,100000):
            end=min(n,start+100000);assert np.array_equal(original['ref'][start:end]+ref_offset,combined['ref'][row_offset+start:row_offset+end]);assert np.array_equal(original['target'][start:end],combined['target'][row_offset+start:row_offset+end])
        output=Path(prefix+'.collective.features.bin');marker=Path(prefix+'.collective.features.done')
        protocol={'rows':n,'width':len(FEATURE_NAMES),'aliases':alias_hash,'probabilities':probability_hash,'candidate_size':part.stat().st_size,'candidate_mtime':part.stat().st_mtime_ns,'normalized':normalized,'reference_population':signature(population)}
        if not(marker.exists() and json.loads(marker.read_text())==protocol and output.exists() and output.stat().st_size==n*SECOND_DTYPE.itemsize):
            probfile=Path(prefix+'.probabilities.bin');probabilities[row_offset:row_offset+n].tofile(probfile)
            print('Building training features:',part.name,'pairs',n,flush=True);native(part,probfile,refs,output)
            assert output.stat().st_size==n*SECOND_DTYPE.itemsize
            marker.write_text(json.dumps(protocol,indent=2))
        outputs.append((output,ref_offset));row_offset+=n
        ref_offset+=sum(1 for _ in refs.open())
        del original
    assert row_offset==len(combined)==len(probabilities)
    with (work/'collective.features.partial').open('wb') as out:
        for output,offset in outputs:
            rows=np.memmap(output,dtype=SECOND_DTYPE,mode='r')
            for start in range(0,len(rows),100000):
                block=np.array(rows[start:start+100000]);block['ref']+=offset;block.tofile(out)
            del rows
    (work/'collective.features.partial').replace(work/'collective.features.bin')

def train(work,reports,baseline_work,baseline_reports,copies=False,entity_weighted=False,device='cpu'):
    global FEATURE_NAMES,SECOND_DTYPE
    if copies:
        FEATURE_NAMES=SECOND_FEATURES+EXTRA_FEATURES+COPY_FEATURES;SECOND_DTYPE=np.dtype([('ref','u4'),('target','u4'),('x','f4',(len(FEATURE_NAMES),))])
    reports.mkdir(exist_ok=True,parents=True);ids,countries,groups=load_metadata(work);truth,pairs,sizes=truth_data(work,ids)
    rows=np.memmap(work/'train.features.bin',dtype=DTYPE,mode='r');rg=groups[rows['ref']];y=np.isin((rows['ref'].astype(np.uint64)<<32)|rows['target'],pairs)
    oldfit=(rg<20)|((rg>=30)&(rg<100));fitting=oldfit|((rg>=200)&(rg<400));tuning=(groups>=20)&(groups<25);heldout=(groups>=25)&(groups<30)
    if not (work/'crossfit.raw.bin').exists():
        raw=np.zeros(len(rows),dtype=np.float64)
        for k in range(3):
            name=f'fold{k}.model.txt'
            if not (work/name).exists():shutil.copyfile(baseline_work/name,work/name)
            prediction=work/f'fold{k}.training_predictions.bin';marker=work/f'fold{k}.training_predictions.json'
            prediction_protocol={'features':signature(work/'train.features.bin'),'model_sha256':hashlib.sha256((work/name).read_bytes()).hexdigest(),'rows':len(rows)}
            if not (prediction.exists() and marker.exists() and json.loads(marker.read_text())==prediction_protocol and prediction.stat().st_size==len(rows)*8):
                model=lgb.Booster(model_file=str(work/name));p=predict_chunks(model,rows['x'],f'fold{k}')
                temporary=prediction.with_suffix('.partial');p.tofile(temporary);temporary.replace(prediction);marker.write_text(json.dumps(prediction_protocol,indent=2));del model,p
            p=np.memmap(prediction,dtype=np.float64,mode='r');selected=oldfit&(rg%3==k);raw[selected]=p[selected];raw[~oldfit]+=p[~oldfit]/3
            del p
            print('Expanded first-stage predictions',k,'pairs',len(rows),flush=True)
        raw.astype(np.float32).tofile(work/'crossfit.raw.bin')
    if not (work/'collective.features.bin').exists():
        build_features(work)
    second=np.memmap(work/'collective.features.bin',dtype=SECOND_DTYPE,mode='r');assert len(second)==len(rows) and np.array_equal(second['ref'],rows['ref']) and np.array_equal(second['target'],rows['target'])
    owner=(pairs&np.uint64(0xffffffff)).astype(np.uint32);og=groups[(pairs>>np.uint64(32)).astype(np.int64)];order=np.argsort(owner);owner=owner[order];og=og[order]
    pos=np.searchsorted(owner,rows['target']);safe=np.minimum(pos,len(owner)-1);valid=(pos<len(owner))&(owner[safe]==rows['target']);tg=np.full(len(rows),999,dtype=np.int32);tg[valid]=og[safe[valid]]
    fit_target=(tg<20)|((tg>=30)&(tg<100))|((tg>=200)&(tg<400))|(tg==999);fitmask=fitting&fit_target
    reserved_targets=None
    if (work/'reserved_audit_targets.bin').exists():
        reserved_targets=np.fromfile(work/'reserved_audit_targets.bin',dtype=np.uint32);fitmask&=~np.isin(rows['target'],reserved_targets)
    params=dict(objective='binary',metric='binary_logloss',learning_rate=.04,num_leaves=95,min_data_in_leaf=200,feature_fraction=.9,lambda_l2=8,num_threads=6,verbosity=-1,seed=2026,deterministic=True,force_col_wise=True)
    if device=='gpu':params.update(device_type='gpu',gpu_use_dp=True,max_bin=63)
    weights=None
    if entity_weighted:
        weights=1./np.maximum(1,sizes[rows['ref']]);weights/=weights[fitmask].mean()
    print('Expanded fitting pairs',int(fitmask.sum()),'total',len(rows),'features',len(FEATURE_NAMES),flush=True)
    rows._mmap.madvise(mmap.MADV_DONTNEED)
    ds=selected_dataset(second,fitmask,y,FEATURE_NAMES,params,work/'fit_matrix.scratch.bin',weight=weights)
    tunemask=tuning[rows['ref']]
    val=selected_dataset(second,tunemask,y,FEATURE_NAMES,params,work/'tuning_matrix.scratch.bin',weight=weights,reference=ds)
    protocol={'features':signature(work/'collective.features.bin'),'feature_names':FEATURE_NAMES,'fitting_mask_sha256':hashlib.sha256(fitmask).hexdigest(),'tuning_mask_sha256':hashlib.sha256(tunemask).hexdigest(),'labels_sha256':hashlib.sha256(y).hexdigest(),'entity_weighted':entity_weighted}
    model=train_checkpointed(params,ds,1600,work,'validation',protocol,valid_sets=[val],patience=80)
    raw=predict_evaluation(model,rows,second['x'],tuning|heldout);p=exclusive(rows,raw);thresholds=np.linspace(.3,.95,66)
    sweep=[{'threshold':float(t),**metric(rows,p,y,sizes,tuning,t)[0]} for t in thresholds];best=max(sweep,key=lambda r:r['macro_f05'])
    candidate=json.loads((work/'candidate_config.json').read_text())
    report={'tuning':best,'heldout':metric(rows,p,y,sizes,heldout,best['threshold'])[0],'countries':{},'best_iteration':model.best_iteration,'threshold':best['threshold'],**candidate,'threshold_sweep':sweep,'feature_importance':dict(zip(FEATURE_NAMES,map(float,model.feature_importance(importance_type='gain')))),'collective':True,'tfidf':True,'fitting_reference_buckets':'0–19, 30–99, 200–399','alias_fitting_reference_buckets':'0–19, 30–99','first_stage_model_reference_buckets':'0–19, 30–99; three entity-disjoint folds','first_stage_iterations':lgb.Booster(model_file=str(work/'fold0.model.txt')).current_iteration()}
    alias_config=json.loads((work/'alias_config.json').read_text()) if (work/'alias_config.json').exists() else {}
    report.update(alias_config)
    report['training_device']=device
    report['training_parameters']=params
    report['validation_training_resumed']=json.loads((work/'validation.training.json').read_text())['resumed']
    if reserved_targets is not None:report['strict_audit_targets']=True;report['reserved_audit_reference_buckets']='100–199'
    for country in sorted(set(countries)):report['countries'][country]=metric(rows,p,y,sizes,heldout&(countries==country),best['threshold'])[0]
    ceiling=np.bincount(rows['ref'][y],minlength=len(ids));upper=np.divide(1.25*ceiling,ceiling+.25*sizes,out=np.ones(len(sizes)),where=sizes>0)
    report['blocking']={'pair_recall':float(ceiling[heldout].sum()/sizes[heldout].sum()),'macro_f05_ceiling':float(upper[heldout].mean()),'candidate_pairs_per_reference':float(np.bincount(rows['ref'],minlength=len(ids))[heldout].mean())}
    baseline=json.loads((baseline_reports/'validation_report.json').read_text());report['country_transfer_US_to_India']=baseline['country_transfer_US_to_India'];report['country_transfer_note']='First-stage-only US classifier from baseline; shared fitting aliases. Not a test of the expanded resolver.'
    if copies:report['copies']=True
    if entity_weighted:report['entity_weighted']=True
    report['selected_by_tuning']=best['macro_f05']>baseline['tuning']['macro_f05']+.0001
    model.save_model(str(work/'collective_validation_model.txt'));np.save(work/'collective_validation_raw.npy',raw);(reports/'validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['threshold_sweep','feature_importance']},indent=2),flush=True)
    del model,ds,val;gc.collect()
    labeled=fitting|tuning[rows['ref']]|heldout[rows['ref']]
    if reserved_targets is not None:labeled&=~np.isin(rows['target'],reserved_targets)
    rows._mmap.madvise(mmap.MADV_DONTNEED)
    final_weight=None
    if entity_weighted:
        final_weight=1./np.maximum(1,sizes[rows['ref']]);final_weight/=final_weight[labeled].mean()
    ds=selected_dataset(second,labeled,y,FEATURE_NAMES,params,work/'final_matrix.scratch.bin',weight=final_weight)
    final_protocol={**protocol,'fitting_mask_sha256':hashlib.sha256(labeled).hexdigest()}
    final=train_checkpointed(params,ds,report['best_iteration'],work,'final',final_protocol)
    final.save_model(str(work/'collective_model.partial'));(work/'collective_model.partial').replace(work/'collective_model.txt')
    cfg={'threshold':best['threshold'],'features':FEATURES,'second_stage_features':FEATURE_NAMES,'ensemble':[f'fold{k}.model.txt' for k in range(3)],'collective':True,'tfidf':True,**candidate,**alias_config}
    cfg['training_device']=device
    if copies:cfg['copies']=True
    if entity_weighted:cfg['entity_weighted']=True
    if reserved_targets is not None:cfg['strict_audit_targets']=True
    (work/'config.json').write_text(json.dumps(cfg,indent=2));print('Expanded final model saved.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--reports',type=Path,required=True);ap.add_argument('--baseline-work',type=Path,default=ROOT/'work/v3');ap.add_argument('--baseline-reports',type=Path,default=ROOT/'reports/v3');ap.add_argument('--copies',action='store_true');ap.add_argument('--entity-weighted',action='store_true');ap.add_argument('--device',choices=['cpu','gpu'],default='cpu');a=ap.parse_args();train(a.work,a.reports,a.baseline_work,a.baseline_reports,a.copies,a.entity_weighted,a.device)
