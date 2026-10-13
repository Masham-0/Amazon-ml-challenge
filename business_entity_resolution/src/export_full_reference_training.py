"""Export supplied-data base features with full reference labels for private CPU fitting.

The existing retrieval population is frozen. All owners in the supplied truth
are used to exclude tuning/audit targets even when they are negative candidates.
Only numeric features, labels, and split/count flags leave this exporter.
"""
import argparse,array,csv,gzip,hashlib,json,time,zlib
from pathlib import Path
import numpy as np
from train import DTYPE,FEATURES
from crossfit import SECOND_FEATURES
from expanded_train import EXTRA_FEATURES,COPY_FEATURES

ROOT=Path(__file__).resolve().parents[3]

def fitting_groups(groups):
    return (groups<20)|((groups>=30)&(groups<100))|(groups>=200)

def selected_flags(reference_groups,owner_groups):
    final=(reference_groups<100)|(reference_groups>=200)
    # Owners are always known in the full supplied ground truth. Targets absent
    # from truth are unmatched records, denoted by 1000.
    final &= (owner_groups<100)|(owner_groups>=200)
    fit=fitting_groups(reference_groups)&(fitting_groups(owner_groups)|(owner_groups==1000))
    tune=(reference_groups>=20)&(reference_groups<25)&final
    return final,fit,tune

def export(work,output):
    assert not (output/'export_complete.json').exists(),'Existing numeric exports are preserved'
    started=time.time();refs={};groups=[]
    with (work/'train.refs.tsv').open() as inp:
        for index,line in enumerate(inp):
            sid=line.split('\t',1)[0];refs[sid]=index;groups.append(zlib.crc32(sid.encode())%1000)
    groups=np.array(groups,dtype=np.uint16);sizes=np.zeros(len(groups),dtype=np.uint8)
    packed=array.array('Q');owners=array.array('I');owner_groups=array.array('H')
    assert packed.itemsize==8 and owners.itemsize==4 and owner_groups.itemsize==2
    data=Path((work/'data_path.txt').read_text().strip())
    truth_rows=0;mapped_references=0
    with (data/'train/train_ground_truth.tsv').open(newline='') as inp:
        for row in csv.DictReader(inp,delimiter='\t'):
            sid=row['source1_entity_id'];group=zlib.crc32(sid.encode())%1000;ref=refs.get(sid)
            targets=row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
            assert len(targets)<=255
            if ref is not None:sizes[ref]=len(targets);mapped_references+=1
            for tid in targets:
                source,number=tid.split('-',1);assert source in {'S2','S3'}
                target=int(number)|((1<<31) if source=='S3' else 0)
                owners.append(target);owner_groups.append(group)
                if ref is not None:packed.append((ref<<32)|target)
            truth_rows+=1
    assert mapped_references==len(refs),'Every retrieved reference must have supplied labels'
    pairs=np.frombuffer(packed,dtype=np.uint64).copy();pairs.sort();del packed,refs
    targets=np.frombuffer(owners,dtype=np.uint32);tg=np.frombuffer(owner_groups,dtype=np.uint16)
    order=np.argsort(targets);targets=targets[order];tg=tg[order];del owners,owner_groups,order
    assert (targets[1:]!=targets[:-1]).all(),'Target ownership must be unique'
    rows=np.memmap(work/'train.features.bin',dtype=DTYPE,mode='r')
    output.mkdir(parents=True,exist_ok=True)
    metadata_dtype=np.dtype([('y','u1'),('fit','u1'),('tuning','u1'),('size','u1')])
    counts={'rows':0,'fitting':0,'tuning':0,'positive':0,'reserved_target_pairs_excluded':0,'all_ground_truth_references':truth_rows}
    digest=hashlib.sha256()
    with gzip.open(output/'features.f32.gz','wb',compresslevel=1) as xf,gzip.open(output/'labels_flags.u8.gz','wb',compresslevel=1) as mf:
        for start in range(0,len(rows),50000):
            block=rows[start:start+50000];rg=groups[block['ref']]
            loc=np.searchsorted(targets,block['target']);safe=np.minimum(loc,len(targets)-1)
            valid=(loc<len(targets))&(targets[safe]==block['target'])
            owner=np.full(len(block),1000,dtype=np.uint16);owner[valid]=tg[safe[valid]]
            final,fit,tune=selected_flags(rg,owner)
            counts['reserved_target_pairs_excluded']+=int(((owner>=100)&(owner<200)&((rg<100)|(rg>=200))).sum())
            keys=(block['ref'].astype(np.uint64)<<np.uint64(32))|block['target']
            loc=np.searchsorted(pairs,keys);safe=np.minimum(loc,len(pairs)-1)
            positive=(loc<len(pairs))&(pairs[safe]==keys)
            ix=np.flatnonzero(final);x=np.ascontiguousarray(block['x'][ix],dtype='<f4')
            assert np.isfinite(x).all()
            metadata=np.empty(len(ix),dtype=metadata_dtype)
            metadata['y']=positive[ix];metadata['fit']=fit[ix];metadata['tuning']=tune[ix];metadata['size']=sizes[block['ref'][ix]]
            assert not (((owner[ix]>=100)&(owner[ix]<200))|((rg[ix]>=100)&(rg[ix]<200))).any()
            assert not (metadata['fit']&metadata['tuning']).any()
            xf.write(x.tobytes());mf.write(metadata.tobytes());digest.update(metadata.tobytes())
            for key,value in [('rows',len(ix)),('fitting',metadata['fit'].sum()),('tuning',metadata['tuning'].sum()),('positive',metadata['y'].sum())]:counts[key]+=int(value)
            if (start+50000)%2000000==0:print('Exported supplied input pairs',start+50000,'/',len(rows),flush=True)
    names=SECOND_FEATURES+EXTRA_FEATURES+COPY_FEATURES
    assert names[:len(FEATURES)]==FEATURES
    manifest={**counts,'active_features':len(FEATURES),'features_per_pair':len(FEATURES),'model_features':len(names),'feature_names':names,'inactive_features':'Remaining columns are constant zero in fitting; cannot enter trees.','metadata_fields':['y','fit','tuning','size'],'metadata_sha256':digest.hexdigest(),'reference_ids_exported':False,'source_text_exported':False,'fitting_reference_buckets':'0–19,30–99,200–999','tuning_reference_buckets':'20–24','final_reference_buckets':'0–99,200–999','audit_reference_and_target_buckets_excluded':'100–199','fitting_target_owner_buckets_excluded':'20–29,100–199','lightgbm_version':'4.7.0','source_rows':len(rows),'source_feature_bytes':(work/'train.features.bin').stat().st_size,'seconds':time.time()-started}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'export_complete.json').write_text(json.dumps({'complete':True,**counts},indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='feature_names'},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work/v9');ap.add_argument('--output',type=Path,default=ROOT/'work/kaggle_full_reference_data');a=ap.parse_args();export(a.work.resolve(),a.output.resolve())
