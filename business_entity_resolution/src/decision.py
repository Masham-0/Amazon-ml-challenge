"""Validate an optional threshold for a reference's first match separately."""
import argparse,json
from pathlib import Path
import numpy as np
from train import DTYPE,load_metadata,truth_data,exclusive,metric
ROOT=Path(__file__).resolve().parents[3]

def selected(rows,prob,config,exclusive_already=False):
    p=prob if exclusive_already else exclusive(rows,prob)
    if not config.get('adaptive_thresholds'):return p>=config['threshold']
    highest=np.full(int(rows['ref'].max())+1,-1,dtype=np.float32);np.maximum.at(highest,rows['ref'],p)
    head=p==highest[rows['ref']];cut=config['adaptive_thresholds'];return p>=np.where(head,cut['first_match'],cut['additional_matches'])

def tune(work,reports):
    ids,countries,groups=load_metadata(work);_,pairs,sizes=truth_data(work,ids);rows=np.memmap(work/'train.features.bin',dtype=DTYPE,mode='r');cfg=json.loads((work/'config.json').read_text())
    raw=np.load(work/('collective_validation_raw.npy' if cfg.get('collective') else 'validation_raw.npy'));p=exclusive(rows,raw)
    refix=np.flatnonzero((groups>=20)&(groups<30));rowix=np.flatnonzero((groups[rows['ref']]>=20)&(groups[rows['ref']]<30));small=np.zeros(len(rowix),dtype=[('ref','u4'),('target','u4')]);small['ref']=np.searchsorted(refix,rows['ref'][rowix]);small['target']=rows['target'][rowix];y=np.isin((rows['ref'][rowix].astype(np.uint64)<<32)|rows['target'][rowix],pairs);p=p[rowix];size=sizes[refix];g=groups[refix];country=countries[refix];tune=g<25;held=g>=25
    top=np.full(len(refix),-1,dtype=np.float32);np.maximum.at(top,small['ref'],p);head=p==top[small['ref']]
    baseline=metric(small,p,y,size,tune,cfg['threshold'])[0];trials=[]
    for first in [.4,.5,.6,.65,.7,.75,.8,.85,.9]:
        for other in [.6,.65,.7,.75,.8,.85,.9,.95]:
            keep=p>=np.where(head,first,other);r=metric(small,keep.astype(np.float32),y,size,tune,.5)[0];trials.append({'first_match':first,'additional_matches':other,**r})
    best=max(trials,key=lambda r:r['macro_f05']);keep=p>=np.where(head,best['first_match'],best['additional_matches']);report={'tuning':best,'baseline_tuning':baseline,'heldout':metric(small,keep.astype(np.float32),y,size,held,.5)[0],'baseline_heldout':metric(small,p,y,size,held,cfg['threshold'])[0],'selected_by_tuning':best['macro_f05']>baseline['macro_f05']+.0003,'trials':trials,'countries':{}}
    for c in sorted(set(country)):report['countries'][c]=metric(small,keep.astype(np.float32),y,size,held&(country==c),.5)[0]
    reports.mkdir(exist_ok=True,parents=True);(reports/'decision_validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)
    if report['selected_by_tuning']:
        cfg['adaptive_thresholds']={k:best[k] for k in ['first_match','additional_matches']};(work/'config.json').write_text(json.dumps(cfg,indent=2))
        base=json.loads((reports/'validation_report.json').read_text());base['before_adaptive_decisions']={k:base[k] for k in ['tuning','heldout','countries']};base.update(heldout=report['heldout'],countries=report['countries'],tuning={**base['tuning'],**best},adaptive_thresholds=cfg['adaptive_thresholds']);(reports/'validation_report.json').write_text(json.dumps(base,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--reports',type=Path,required=True);a=ap.parse_args();tune(a.work,a.reports)
