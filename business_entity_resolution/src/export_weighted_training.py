"""Export numeric final-stage fitting data without source IDs or source text."""
import argparse, gzip, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'code/business_entity_resolution/src'))
from train import load_metadata, truth_data
from expanded_train import SECOND_FEATURES, EXTRA_FEATURES, COPY_FEATURES

def export(work, output):
    if (output/'export_complete.json').exists():
        raise ValueError('Use an empty export directory; existing export is preserved')
    names = SECOND_FEATURES + EXTRA_FEATURES + COPY_FEATURES
    dtype = np.dtype([('ref','<u4'),('target','<u4'),('x','<f4',(len(names),))])
    features = work/'collective.features.bin'
    assert features.exists() and features.stat().st_size % dtype.itemsize == 0
    # This exporter must only consume the fully assembled fitting matrix.
    assert (work/'validation.training.json').exists()
    ids, countries, groups = load_metadata(work)
    truth, pairs, sizes = truth_data(work, ids)
    del truth, countries, ids
    packed_truth = np.sort(pairs)
    target = (pairs & np.uint64(0xffffffff)).astype(np.uint32)
    owner_group = groups[(pairs >> np.uint64(32)).astype(np.int64)]
    order = np.argsort(target); target = target[order]; owner_group = owner_group[order]
    reserved = np.sort(np.fromfile(work/'reserved_audit_targets.bin',dtype=np.uint32))
    del pairs, order
    rows = np.memmap(features,dtype=dtype,mode='r')
    output.mkdir(parents=True,exist_ok=True)
    counts = dict(rows=0, fitting=0, tuning=0, positive=0)
    meta_dtype = np.dtype([('y','u1'),('fit','u1'),('tuning','u1'),('size','u1')])
    digest = hashlib.sha256(); started = time.time()
    with gzip.open(output/'features.f32.gz','wb',compresslevel=1) as xf, gzip.open(output/'labels_flags.u8.gz','wb',compresslevel=1) as mf:
        for start in range(0,len(rows),50000):
            block=rows[start:start+50000]; rg=groups[block['ref']]
            fitting=(rg<20)|((rg>=30)&(rg<100))|((rg>=200)&(rg<400))
            tuning=(rg>=20)&(rg<25)
            labeled=fitting|((rg>=20)&(rg<30))
            target_key=block['target']
            loc=np.searchsorted(reserved,target_key); safe=np.minimum(loc,len(reserved)-1)
            excluded=(loc<len(reserved))&(reserved[safe]==target_key)
            labeled &= ~excluded
            loc=np.searchsorted(target,target_key); safe=np.minimum(loc,len(target)-1)
            tg=np.full(len(block),999,dtype=np.int32); valid=(loc<len(target))&(target[safe]==target_key)
            tg[valid]=owner_group[safe[valid]]
            fit_target=(tg<20)|((tg>=30)&(tg<100))|((tg>=200)&(tg<400))|(tg==999)
            fit=fitting&fit_target&~excluded
            keys=(block['ref'].astype(np.uint64)<<np.uint64(32))|target_key
            loc=np.searchsorted(packed_truth,keys); safe=np.minimum(loc,len(packed_truth)-1)
            y=(loc<len(packed_truth))&(packed_truth[safe]==keys)
            selected=np.flatnonzero(labeled)
            x=np.ascontiguousarray(block['x'][selected],dtype='<f4')
            assert np.isfinite(x).all()
            entity_size=sizes[block['ref'][selected]]
            assert (entity_size<=255).all()
            meta=np.empty(len(selected),dtype=meta_dtype)
            meta['y']=y[selected];meta['fit']=fit[selected];meta['tuning']=tuning[selected];meta['size']=entity_size
            xf.write(x.tobytes());mf.write(meta.tobytes());digest.update(meta.tobytes())
            counts['rows']+=len(selected);counts['fitting']+=int(meta['fit'].sum());counts['tuning']+=int(meta['tuning'].sum());counts['positive']+=int(meta['y'].sum())
            if (start+50000)%1000000==0:print('Exported input pairs',start+50000,'/',len(rows),flush=True)
    manifest={**counts,'feature_names':names,'features_per_pair':len(names),'metadata_fields':['y','fit','tuning','size'],'metadata_sha256':digest.hexdigest(),'source_feature_bytes':features.stat().st_size,'source_feature_mtime_ns':features.stat().st_mtime_ns,'reference_ids_exported':False,'source_text_exported':False,'audit_buckets_excluded':'100–199; reserved targets excluded even as negatives','seconds':time.time()-started,'lightgbm_version':'4.7.0'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'export_complete.json').write_text(json.dumps({'complete':True,**counts},indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='feature_names'},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work/v9');ap.add_argument('--output',type=Path,default=ROOT/'work/kaggle_training_data');a=ap.parse_args();export(a.work.resolve(),a.output.resolve())
