"""Entity-disjoint fitting, threshold tuning, and development evaluation."""
import argparse,csv,json,zlib,time
from pathlib import Path
import numpy as np
import lightgbm as lgb

ROOT=Path(__file__).resolve().parents[3]
FEATURES='name_exact name_lcs name_sorted_lcs name_jaccard name_dice name_containment name_overlap_max name_trigram name_phonetic name_length_ratio name_overlap_count name_ref_tokens name_target_tokens address_exact address_jaccard address_dice address_containment address_overlap_max address_trigram address_sorted_lcs address_overlap_count address_ref_tokens address_target_tokens numbers_jaccard numbers_containment numbers_exact numbers_ref_count numbers_target_count numbers_missing name_missing address_missing retrieval_weight retrieval_rank retrieval_gap source2 name_total_length'.split()
FEATURES+='raw_name_exact raw_name_lcs raw_name_sorted_lcs raw_name_jaccard raw_name_dice raw_name_containment raw_name_overlap_max raw_name_length_ratio raw_name_extra_target raw_name_extra_ref name_fuzzy_mean_min name_fuzzy_mean_max name_fuzzy_min_ref name_fuzzy_min_target name_first_token_lcs name_first_token_phonetic name_weighted_overlap name_weighted_missing_ref name_weighted_missing_target alnum_number_jaccard alnum_number_dice alnum_number_containment alnum_number_overlap_max alnum_number_exact alnum_number_ref_count alnum_number_target_count alnum_number_missing_ref alnum_number_missing_target first_number_exact first_number_lcs digit_multiset_jaccard digit_multiset_dice digit_multiset_exact digit_multiset_missing_ref digit_multiset_missing_target floor_match floor_conflict floor_missing unit_match unit_conflict unit_missing name_direction_conflict name_direction_extra_target number_leading_zero_conflict'.split()
FEATURES+='name_compact_lcs name_compact_containment name_initials_match name_core_initials_match name_ref_frequency name_target_frequency name_core_jaccard name_core_exact name_core_ref_count name_core_target_count name_core_extra_target name_core_extra_ref name_raw_first_lcs name_raw_first_exact address_weighted_jaccard address_weighted_dice address_weighted_coverage_ref address_weighted_coverage_target address_max_shared_idf address_core_jaccard address_core_exact address_core_ref_count address_core_target_count numbers_fuzzy_mean numbers_fuzzy_min numbers_same_digits number_deletion_count number_substitution_count number_transposition_count'.split()
COMPONENTS='name address name_containment address_containment raw_name name_idf numbers hybrid'.split()
FEATURES+=['competitor_count']+[f'other_best_{n}' for n in COMPONENTS]+[f'other_margin_{n}' for n in COMPONENTS]+[f'other_close_{n}' for n in COMPONENTS]
DTYPE=np.dtype([('ref','<u4'),('target','<u4'),('x','<f4',(len(FEATURES),))])

def load_metadata(work,prefix='train'):
    ids=[];countries=[];groups=[]
    with (work/f'{prefix}.refs.tsv').open() as inp:
        for line in inp:
            sid,country,sample=line.rstrip('\n').split('\t');ids.append(sid);countries.append(country)
            groups.append(zlib.crc32(sid.encode())%1000 if sample=='1' else 999)
    return np.array(ids),np.array(countries),np.array(groups)

def truth_data(work,ids):
    refs={x:i for i,x in enumerate(ids)};truth={};pairs=[];sizes=np.zeros(len(ids),dtype=np.int32)
    for line in (work/'sample_truth.tsv').read_text().splitlines():
        sid,targets=line.split('\t');i=refs[sid];ts=set()
        for t in targets.split(',') if targets else []:
            num=int(t[3:])|((1<<31) if t.startswith('S3-') else 0);ts.add(num);pairs.append((int(i)<<32)|num)
        truth[i]=ts;sizes[i]=len(ts)
    return truth,np.array(pairs,dtype=np.uint64),sizes

def metric(rows,prob,y,true_sizes,mask,threshold):
    keep=prob>=threshold
    pred=np.bincount(rows['ref'][keep],minlength=len(true_sizes))
    tp=np.bincount(rows['ref'][keep&y],minlength=len(true_sizes))
    den=pred+.25*true_sizes
    scores=np.divide(1.25*tp,den,out=np.zeros(len(den)),where=den>0)
    scores[(pred==0)&(true_sizes==0)]=1
    selected=scores[mask]
    return {'macro_f05':float(selected.mean()),'standard_error':float(selected.std()/np.sqrt(len(selected))),'entities':int(mask.sum()),'pair_precision':float(tp[mask].sum()/max(1,pred[mask].sum())),'pair_recall':float(tp[mask].sum()/max(1,true_sizes[mask].sum())),'false_positive_pairs':int((pred[mask]-tp[mask]).sum()),'false_negative_pairs':int((true_sizes[mask]-tp[mask]).sum()),'singleton_correct_fraction':float((pred[mask&(true_sizes==0)]==0).mean())},scores,pred,tp

def exclusive(rows,prob):
    starts=np.r_[0,np.flatnonzero(rows['target'][1:]!=rows['target'][:-1])+1]
    maximum=np.repeat(np.maximum.reduceat(prob,starts),np.diff(np.r_[starts,len(rows)]))
    eligible=np.flatnonzero(prob==maximum)
    eligible=eligible[np.r_[True,rows['target'][eligible[1:]]!=rows['target'][eligible[:-1]]]]
    result=np.full(len(rows),-1,dtype=np.float32);result[eligible]=prob[eligible]
    return result

def fit(x,y,fitmask,tunemask,rounds=2400):
    ds=lgb.Dataset(x[fitmask],label=y[fitmask],feature_name=FEATURES)
    val=lgb.Dataset(x[tunemask],label=y[tunemask],reference=ds)
    params=dict(objective='binary',metric='binary_logloss',learning_rate=.05,num_leaves=127,min_data_in_leaf=100,feature_fraction=.90,lambda_l2=5,num_threads=6,verbosity=-1,seed=2026,deterministic=True,force_col_wise=True)
    return lgb.train(params,ds,num_boost_round=rounds,valid_sets=[val],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(100)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--reports',type=Path,default=ROOT/'reports');a=ap.parse_args();a.reports.mkdir(parents=True,exist_ok=True)
    ids,countries,groups=load_metadata(a.work);truth,true_pairs,sizes=truth_data(a.work,ids)
    rows=np.memmap(a.work/'train.features.bin',dtype=DTYPE,mode='r');packed=(rows['ref'].astype(np.uint64)<<32)|rows['target'];y=np.isin(packed,true_pairs);rg=groups[rows['ref']]
    # A target belonging to a tuning/evaluation entity cannot enter fitting,
    # even as a negative candidate for a different reference.
    owner_targets=(true_pairs&np.uint64(0xffffffff)).astype(np.uint32)
    owner_groups=groups[(true_pairs>>np.uint64(32)).astype(np.int64)]
    sort_owner=np.argsort(owner_targets);owner_targets=owner_targets[sort_owner];owner_groups=owner_groups[sort_owner]
    location=np.searchsorted(owner_targets,rows['target']);valid=location<len(owner_targets);safe=np.minimum(location,len(owner_targets)-1)
    valid&=owner_targets[safe]==rows['target'];target_group=np.full(len(rows),999,dtype=np.int32);target_group[valid]=owner_groups[safe[valid]]
    fitgroup=(groups<20)|((groups>=30)&(groups<100))
    target_fit=(target_group<20)|((target_group>=30)&(target_group<100))
    fitting=fitgroup[rows['ref']]&(target_fit|(target_group==999))
    print('pairs',len(rows),'positive',y.sum(),'references',len(truth),flush=True)
    model=fit(rows['x'],y,fitting,(rg>=20)&(rg<25));raw_p=model.predict(rows['x'],num_threads=6).astype(np.float32);p=exclusive(rows,raw_p)
    thresholds=np.unique(np.r_[np.linspace(.1,.95,35),.97,.98,.99,.995,.999])
    sweep=[]
    for t in thresholds:
        result,*_=metric(rows,p,y,sizes,(groups>=20)&(groups<25),float(t));sweep.append({'threshold':float(t),**result})
    best=max(sweep,key=lambda r:r['macro_f05']);threshold=best['threshold'];hold=(groups>=25)&(groups<30)
    # Tune the last deterministic blocking pass before any final ML scoring.
    x=rows['x'];ad=np.maximum(x[:,18],.88*x[:,16]);ns=np.maximum.reduce([x[:,2],.90*x[:,8],.90*x[:,5]])
    cheap=np.maximum(ns*np.where(x[:,30]>0,.72,.42+.58*ad),.84*ad)
    pruning=[];base=best['macro_f05'];chosen=None
    for gate in [.35]:
        for gap in [.10]:
            allowed=(cheap>=gate)&(x[:,33]>=-gap)
            pm=exclusive(rows,np.where(allowed,raw_p,-1));r=metric(rows,pm,y,sizes,(groups>=20)&(groups<25),threshold)[0]
            mean=float(np.bincount(rows['ref'][allowed],minlength=len(ids))[(groups>=20)&(groups<25)].mean())
            trial={'gate':gate,'gap':gap,'mean_candidates':mean,**r};pruning.append(trial)
            if r['macro_f05']>=base-.00003 and (chosen is None or mean<chosen['mean_candidates']):chosen=trial
    if chosen is None:chosen=pruning[0]
    gate,gap=chosen['gate'],chosen['gap'];allowed=(cheap>=gate)&(x[:,33]>=-gap);p=exclusive(rows,np.where(allowed,raw_p,-1))
    result,scores,pred,tp=metric(rows,p,y,sizes,hold,threshold)
    report={'threshold':threshold,'gate':gate,'gap':gap,'best_iteration':model.best_iteration,'tuning':chosen,'heldout':result,'countries':{},'threshold_sweep':sweep,'blocking_sweep':pruning,'feature_importance':dict(zip(FEATURES,map(float,model.feature_importance(importance_type='gain'))))}
    for country in sorted(set(countries)):
        report['countries'][country]=metric(rows,p,y,sizes,hold&(countries==country),threshold)[0]
    ceiling=np.bincount(rows['ref'][y&allowed],minlength=len(ids));upper=np.divide(1.25*ceiling,ceiling+.25*sizes,out=np.ones(len(sizes)),where=sizes>0)
    report['blocking']={'pair_recall':float(ceiling[hold].sum()/sizes[hold].sum()),'macro_f05_ceiling':float(upper[hold].mean()),'candidate_pairs_per_reference':float(np.bincount(rows['ref'][allowed],minlength=len(ids))[hold].mean())}
    model.save_model(str(a.work/'validation_model.txt'))
    np.save(a.work/'validation_raw.npy',raw_p)
    # An explicit country-held-out test probes transfer to unseen country patterns.
    us=countries[rows['ref']]=='US';transfer=fit(rows['x'],y,fitting&us,(rg>=20)&(rg<25)&us,rounds=model.best_iteration)
    pt=exclusive(rows,np.where(allowed,transfer.predict(rows['x'],num_threads=6).astype(np.float32),-1))
    transfer_t=max(thresholds,key=lambda t:metric(rows,pt,y,sizes,(groups>=20)&(groups<25)&(countries=='US'),t)[0]['macro_f05'])
    report['country_transfer_US_to_India']=metric(rows,pt,y,sizes,hold&(countries=='India'),transfer_t)[0]
    report['country_transfer_threshold']=float(transfer_t)
    with (a.reports/'validation_errors.tsv').open('w') as out:
        out.write('source1_entity_id\tcountry\tf05\ttrue_count\tpredicted_count\ttrue_positive_count\n')
        wrong=np.flatnonzero(hold&(scores<1));wrong=wrong[np.argsort(scores[wrong])]
        for i in wrong[:1000]:out.write(f'{ids[i]}\t{countries[i]}\t{scores[i]:.6f}\t{sizes[i]}\t{pred[i]}\t{tp[i]}\n')
    with (a.reports/'validation_pair_errors.tsv').open('w') as out:
        out.write('source1_entity_id\ttarget_id\tcountry\ttrue_match\tprobability\tname_similarity\taddress_similarity\tnumber_overlap\n')
        mistakes=hold[rows['ref']]&(((p>=threshold)&~y)|((p<threshold)&y))
        ix=np.flatnonzero(mistakes);ix=ix[np.argsort(-p[ix])]
        for j in ix[:1000]:
            i=int(rows['ref'][j]);t=int(rows['target'][j]);tid=('S3-' if t>>31 else 'S2-')+str(t&0x7fffffff)
            out.write(f'{ids[i]}\t{tid}\t{countries[i]}\t{int(y[j])}\t{p[j]:.6f}\t{x[j,2]:.6f}\t{x[j,18]:.6f}\t{x[j,23]:.6f}\n')
    np.savez(a.work/'validation_predictions.npz',p=p,y=y,groups=groups)
    (a.reports/'validation_report.json').write_text(json.dumps(report,indent=2))
    model.save_model(str(a.work/'validation_model.txt'))
    print(json.dumps({k:v for k,v in report.items() if k not in ['threshold_sweep','feature_importance']},indent=2),flush=True)
    # Refit only after local evaluation has been recorded; iteration count is fixed.
    final_params=model.params.copy();final_params.pop('num_iterations',None)
    labeled=rg<100
    final=lgb.train(final_params,lgb.Dataset(rows['x'][labeled],label=y[labeled],feature_name=FEATURES),num_boost_round=model.best_iteration)
    final.save_model(str(a.work/'model.txt'));(a.work/'config.json').write_text(json.dumps({'threshold':threshold,'features':FEATURES,'gate':gate,'gap':gap},indent=2))

if __name__=='__main__':main()
