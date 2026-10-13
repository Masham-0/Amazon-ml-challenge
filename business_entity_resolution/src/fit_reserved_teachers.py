"""Fit frozen teachers excluding reserved audit targets even as negatives."""
import argparse,hashlib,json,mmap,gc,zlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import DTYPE,FEATURES,load_metadata,truth_data
from training_runtime import selected_dataset, train_checkpointed

def train(baseline,output,device='cpu'):
    output.mkdir(exist_ok=True,parents=True)
    ids,_,groups=load_metadata(baseline);_,pairs,_=truth_data(baseline,ids)
    rows=np.memmap(baseline/'train.features.bin',dtype=DTYPE,mode='r');rg=groups[rows['ref']]
    y=np.isin((rows['ref'].astype(np.uint64)<<32)|rows['target'],pairs)
    data=Path((baseline/'data_path.txt').read_text().strip());reserved=[]
    with (data/'train/train_ground_truth.tsv').open() as f:
        for line in f:
            sid,targets=line.rstrip('\n').split('\t')
            if sid=='source1_entity_id':continue
            if 100<=zlib.crc32(sid.encode())%1000<200:
                reserved.extend(int(t[3:])|((1<<31) if t.startswith('S3-') else 0) for t in targets.split(',') if t)
    reserved=np.unique(np.array(reserved,dtype=np.uint32));reserved.tofile(output/'reserved_audit_targets.bin')
    blocked=np.isin(rows['target'],reserved)
    assert not (y&blocked).any()
    fitting=(rg<20)|((rg>=30)&(rg<100));fold=rg%3
    owner=(pairs&np.uint64(0xffffffff)).astype(np.uint32);og=groups[(pairs>>np.uint64(32)).astype(np.int64)];order=np.argsort(owner);owner=owner[order];og=og[order]
    loc=np.searchsorted(owner,rows['target']);safe=np.minimum(loc,len(owner)-1);valid=(loc<len(owner))&(owner[safe]==rows['target']);tg=np.full(len(rows),999);tg[valid]=og[safe[valid]]
    params=dict(objective='binary',metric='binary_logloss',learning_rate=.05,num_leaves=127,min_data_in_leaf=100,feature_fraction=.9,lambda_l2=5,num_threads=6,verbosity=-1,seed=2026,deterministic=True,force_col_wise=True)
    if device=='gpu':params.update(device_type='gpu',gpu_use_dp=True,max_bin=63)
    protocol={'reserved_groups':[100,200],'reserved_targets':int(len(reserved)),'rounds':1600,'parameters':params,'candidate_size':(baseline/'train.features.bin').stat().st_size,'candidate_mtime':(baseline/'train.features.bin').stat().st_mtime_ns,'aliases_sha256':hashlib.sha256((baseline/'aliases.tsv').read_bytes()).hexdigest(),'removed_fitting_negative_pairs':int((blocked&fitting).sum())}
    print('Strict teacher protocol:',json.dumps(protocol),flush=True)
    for k in range(3):
        path=output/f'fold{k}.model.txt';marker=output/f'fold{k}.strict.done'
        if path.exists() and marker.exists() and json.loads(marker.read_text())==protocol:continue
        mask=fitting&(fold!=k)&~blocked&((tg==999)|(((tg<20)|((tg>=30)&(tg<100)))&(tg%3!=k)))
        ds=selected_dataset(rows,mask,y,FEATURES,params,output/f'fold{k}.scratch.bin')
        model=train_checkpointed(params,ds,1600,output,f'fold{k}',{**protocol,'mask_sha256':hashlib.sha256(mask).hexdigest()});temporary=output/f'fold{k}.model.partial';model.save_model(str(temporary));temporary.replace(path)
        marker.write_text(json.dumps(protocol,indent=2));del model,ds;gc.collect();print('Strict teacher fold finished',k,flush=True)
    (output/'strict_teacher_protocol.json').write_text(json.dumps(protocol,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-work',type=Path,required=True);ap.add_argument('--output-work',type=Path,required=True);ap.add_argument('--device',choices=['cpu','gpu'],default='cpu');a=ap.parse_args();train(a.baseline_work,a.output_work,a.device)
