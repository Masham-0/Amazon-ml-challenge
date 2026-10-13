"""Calibrate frozen decisions on references with complete competitor context."""
import argparse,json
from pathlib import Path
import numpy as np
from train import load_metadata,truth_data,metric,exclusive
from export import DTYPE
from decision import selected

def calibrate(model_work,audit_work,reports):
    cfg=json.loads((model_work/'config.json').read_text())
    ids,countries,groups=load_metadata(audit_work,'test')
    # Audit labels were written only after frozen inference completed.
    _,pairs,sizes=truth_data(audit_work,ids)
    rows=np.memmap(audit_work/'test.scores.bin',dtype=DTYPE,mode='r')
    packed=(rows['ref'].astype(np.uint64)<<32)|rows['target'];y=np.isin(packed,pairs)
    probability=exclusive(rows,rows['p'])
    # Resolve ownership with every competitor before reducing to calibration
    # references. The compact arrays preserve all candidates of these refs.
    reference_indices=np.flatnonzero((groups>=100)&(groups<105))
    row_indices=np.flatnonzero((groups[rows['ref']]>=100)&(groups[rows['ref']]<105))
    compact=np.empty(len(row_indices),dtype=DTYPE)
    compact['ref']=np.searchsorted(reference_indices,rows['ref'][row_indices])
    compact['target']=rows['target'][row_indices];compact['p']=probability[row_indices]
    probability=compact['p'];y=y[row_indices];rows=compact
    sizes=sizes[reference_indices];groups=groups[reference_indices];countries=countries[reference_indices];ids=ids[reference_indices]
    tuning=(groups>=100)&(groups<103);development=(groups>=103)&(groups<105)
    assert tuning.any() and development.any()
    baseline=selected(rows,probability,cfg,exclusive_already=True)
    top=np.full(len(ids),-1,dtype=np.float32);np.maximum.at(top,rows['ref'],probability)
    head=probability==top[rows['ref']]
    before=metric(rows,baseline.astype(np.float32),y,sizes,tuning,.5)[0]
    trials=[];best=None;seen=set()
    def evaluate(first,additional):
        nonlocal best
        first,additional=round(float(first),2),round(float(additional),2)
        if (first,additional) in seen:return
        seen.add((first,additional))
        keep=probability>=np.where(head,first,additional)
        measurement=metric(rows,keep.astype(np.float32),y,sizes,tuning,.5)[0]
        trial={'first_match':first,'additional_matches':additional,**measurement};trials.append(trial)
        if best is None or trial['macro_f05']>best['macro_f05']:best=trial
    for first in np.arange(.4,.951,.05):
        for additional in np.arange(.4,.951,.05):evaluate(first,additional)
    # One local refinement of the two chosen cutoffs, using tuning only.
    coarse=best.copy()
    for first in np.arange(max(.30,coarse['first_match']-.04),min(.99,coarse['first_match']+.04)+.001,.01):
        for additional in np.arange(max(.30,coarse['additional_matches']-.04),min(.99,coarse['additional_matches']+.04)+.001,.01):evaluate(first,additional)
    applied=best['macro_f05']>before['macro_f05']+.0003
    if applied:
        cfg['adaptive_thresholds']={k:best[k] for k in ['first_match','additional_matches']}
    cfg['context_decision_calibration']={'tuning_buckets':'100–102','development_buckets':'103–104','complete_competitor_context':True}
    active=selected(rows,probability,cfg,exclusive_already=True)
    active_tuning=metric(rows,active.astype(np.float32),y,sizes,tuning,.5)[0]
    active_development=metric(rows,active.astype(np.float32),y,sizes,development,.5)[0]
    country_metrics={c:metric(rows,active.astype(np.float32),y,sizes,development&(countries==c),.5)[0] for c in sorted(set(countries))}
    report={'protocol':'Frozen final models; complete related-record and competing-reference context. Groups100–102 select cutoffs by a coarse grid and one local 0.01 refinement; groups103–104 assess development. These groups are excluded from both fitting stages and alias learning. Previously designated diagnostic buckets are used for calibration, not an untouched final test.','selected_by_tuning':bool(applied),'thresholds_before':{'threshold':cfg['threshold'],'adaptive_thresholds':json.loads((model_work/'config.json').read_text()).get('adaptive_thresholds')},'tuning_before':before,'development_before':metric(rows,baseline.astype(np.float32),y,sizes,development,.5)[0],'best_coarse_tuning_trial':coarse,'best_tuning_trial':best,'active_tuning':active_tuning,'active_development':active_development,'active_countries':country_metrics,'trials':trials,'models_refitted':False,'test_labels_used':False}
    reports.mkdir(exist_ok=True,parents=True)
    (reports/'context_decision_validation_report.json').write_text(json.dumps(report,indent=2))
    # The packaged headline must describe the decisions that are actually used.
    validation=json.loads((reports/'validation_report.json').read_text())
    validation.setdefault('before_complete_context_calibration',{k:validation[k] for k in ['tuning','heldout','countries','blocking']})
    validation.update(tuning=active_tuning,heldout=active_development,countries=country_metrics,decision_evaluation_protocol='Complete scoring context; CRC100–102 tuning, CRC103–104 development')
    ceiling=np.bincount(rows['ref'][y],minlength=len(ids));upper=np.divide(1.25*ceiling,ceiling+.25*sizes,out=np.ones(len(sizes)),where=sizes>0)
    validation['blocking']={'pair_recall':float(ceiling[development].sum()/sizes[development].sum()),'macro_f05_ceiling':float(upper[development].mean()),'candidate_pairs_per_reference':float(np.bincount(rows['ref'],minlength=len(ids))[development].mean())}
    if cfg.get('adaptive_thresholds'):validation['adaptive_thresholds']=cfg['adaptive_thresholds']
    (reports/'validation_report.json').write_text(json.dumps(validation,indent=2))
    (model_work/'config.json').write_text(json.dumps(cfg,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-work',type=Path,required=True);parser.add_argument('--audit-work',type=Path,required=True);parser.add_argument('--reports',type=Path,required=True)
    args=parser.parse_args();calibrate(args.model_work,args.audit_work,args.reports)
