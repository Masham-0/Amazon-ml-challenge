"""Score complete frozen features only for evaluation queries and all competitors.

Query keys come from the first, complete candidate pass for primary references.
Their features still come from the second pass, which contains every related
record needed by those competing references. No labels are read here.
"""
import hashlib,json,mmap
from pathlib import Path
import lightgbm as lgb
import numpy as np
from train import DTYPE as BASE_DTYPE
from export import DTYPE as SCORE_DTYPE

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def query_indices(features,queries,width):
    features,queries=Path(features),Path(queries)
    dtype=np.dtype([('ref','<u4'),('target','<u4'),('x','<f4',(width,))])
    assert features.stat().st_size%dtype.itemsize==0 and queries.stat().st_size%BASE_DTYPE.itemsize==0
    signature={'features':[features.stat().st_size,features.stat().st_mtime_ns],'queries_sha256':digest(queries),'width':width}
    indices=Path(str(features)+'.query_indices.bin');marker=Path(str(indices)+'.json')
    if marker.exists() and indices.exists() and json.loads(marker.read_text())==signature:
        return np.fromfile(indices,dtype='<u8')
    primary=np.memmap(queries,mode='r',dtype=BASE_DTYPE)
    keys=(primary['ref'].astype(np.uint64)<<np.uint64(32))|primary['target'];keys.sort()
    assert len(keys)>0 and (keys[1:]!=keys[:-1]).all();del primary
    rows=np.memmap(features,mode='r',dtype=dtype);parts=[]
    for start in range(0,len(rows),100000):
        block=rows[start:start+100000]
        packed=(block['ref'].astype(np.uint64)<<np.uint64(32))|block['target']
        location=np.searchsorted(keys,packed);safe=np.minimum(location,len(keys)-1)
        parts.append((np.flatnonzero((location<len(keys))&(keys[safe]==packed))+start).astype('<u8'))
        if hasattr(rows._mmap,'madvise'):rows._mmap.madvise(mmap.MADV_DONTNEED)
    selected=np.concatenate(parts);assert len(selected)==len(keys),'Every original candidate and competitor must be recovered exactly once'
    tmp=Path(str(indices)+'.partial');selected.tofile(tmp);tmp.replace(indices);marker.write_text(json.dumps(signature,indent=2))
    print('Exact evaluation-query context pairs',len(selected),'of',len(rows),flush=True)
    return selected

def score_queries(features,queries,output,model,width,num_threads=1):
    features,queries,output,model=map(Path,[features,queries,output,model]);selected=query_indices(features,queries,width)
    dtype=np.dtype([('ref','<u4'),('target','<u4'),('x','<f4',(width,))]);rows=np.memmap(features,mode='r',dtype=dtype)
    booster=lgb.Booster(model_file=str(model));assert booster.num_feature()==width
    temporary=Path(str(output)+'.partial');count=0
    with temporary.open('wb') as out:
        for start in range(0,len(selected),50000):
            block=rows[selected[start:start+50000]]
            if hasattr(rows._mmap,'madvise'):rows._mmap.madvise(mmap.MADV_DONTNEED)
            score=np.empty(len(block),dtype=SCORE_DTYPE);score['ref']=block['ref'];score['target']=block['target']
            score['p']=booster.predict(np.ascontiguousarray(block['x']),num_threads=num_threads).astype(np.float32)
            assert np.isfinite(score['p']).all();score.tofile(out);count+=len(score)
    assert count==len(selected);temporary.replace(output)
    result={'complete_context_feature_pairs':len(rows),'scored_query_pairs':count,'all_primary_query_competitors_retained':True,'all_related_record_features_reused':True,'labels_used':False,'model_sha256':digest(model),'query_sha256':digest(queries),'prediction_threads':num_threads}
    Path(str(output)+'.query_context.json').write_text(json.dumps(result,indent=2));return result
