"""Bound support-record memory while preserving complete per-reference context."""
import hashlib,json,mmap,subprocess
from pathlib import Path
import numpy as np
from train import DTYPE

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def run_batched(helper,features,probabilities,source2,source3,refs,output,aliases,source1,feature_count=166,model=None,library=None,max_references=75000,resume=True):
    paths=list(map(Path,[helper,features,probabilities,source2,source3,refs,output,aliases,source1]))
    helper,features,probabilities,source2,source3,refs,output,aliases,source1=paths
    nrefs=sum(1 for _ in refs.open());n=features.stat().st_size//DTYPE.itemsize
    dtype=np.dtype([('ref','u4'),('target','u4'),('p','f4')]) if model else np.dtype([('ref','u4'),('target','u4'),('x','f4',(feature_count,))])
    protocol={'rows':n,'references':nrefs,'itemsize':dtype.itemsize,'batch_references':max_references,'batching':'active_reference_ranges_v2','features':[features.stat().st_size,features.stat().st_mtime_ns],'probabilities':digest(probabilities),'aliases':digest(aliases),'refs':digest(refs),'helper':digest(helper),'normalized':[[p.stat().st_size,p.stat().st_mtime_ns] for p in [source1,source2,source3]],'model':digest(model) if model else None}
    statefile=Path(str(output)+'.batch_state.json');partial=Path(str(output)+'.partial')
    state=json.loads(statefile.read_text()) if resume and statefile.exists() else {}
    valid=state.get('protocol')==protocol
    if valid and state.get('complete') and output.exists() and output.stat().st_size==n*dtype.itemsize:return
    valid=valid and partial.exists() and partial.stat().st_size==n*dtype.itemsize
    if not valid:state={'protocol':protocol,'completed':[],'complete':False}
    dst=np.memmap(partial,dtype=dtype,mode='r+' if valid else 'w+',shape=(n,))
    base=np.memmap(features,dtype=DTYPE,mode='r')
    active_refs=np.unique(base['ref']);base._mmap.madvise(mmap.MADV_DONTNEED)
    for offset in range(0,len(active_refs),max_references):
        start=int(active_refs[offset]);stop=int(active_refs[min(len(active_refs),offset+max_references)-1])+1
        if start in state['completed']:continue
        indices=np.flatnonzero((base['ref']>=start)&(base['ref']<stop))
        base._mmap.madvise(mmap.MADV_DONTNEED)
        batch=Path(str(output)+'.batch.bin')
        if len(indices):
            mode='score_range' if model else 'train_range'
            args=[helper,features,probabilities,source2,source3,refs,batch,aliases,mode,model or '-',library or '-',source1,str(start),str(stop)]
            print('Related-record batch',start,stop,'pairs',len(indices),flush=True)
            subprocess.run(list(map(str,args)),check=True)
            part=np.memmap(batch,dtype=dtype,mode='r');assert len(part)==len(indices)
            for pos in range(0,len(indices),100000):
                ix=indices[pos:pos+100000];block=part[pos:pos+100000]
                assert np.array_equal(block['ref'],base['ref'][ix]) and np.array_equal(block['target'],base['target'][ix])
                dst[ix]=block
            del block,part;batch.unlink()
        dst.flush();dst._mmap.madvise(mmap.MADV_DONTNEED);base._mmap.madvise(mmap.MADV_DONTNEED)
        state['completed'].append(start)
        temporary=statefile.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(statefile)
    del dst,base
    partial.replace(output);state['complete']=True
    temporary=statefile.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(statefile)
