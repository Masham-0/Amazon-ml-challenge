"""Apply leave-one-record-out matching support separately in each country."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from export import DTYPE
from run import countries
from collective_batch import run_batched
ROOT=Path(__file__).resolve().parents[3]
def score_cached_features(features,output,model_path,feature_count,resume=False):
    """Reuse label-free features when comparing only second-stage models."""
    features,output,model_path=map(Path,[features,output,model_path])
    dtype=np.dtype([('ref','u4'),('target','u4'),('x','f4',(feature_count,))])
    assert features.stat().st_size%dtype.itemsize==0
    count=features.stat().st_size//dtype.itemsize
    protocol={'features':[str(features.resolve()),features.stat().st_size,features.stat().st_mtime_ns],'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'features_per_pair':feature_count,'pairs':count}
    marker=Path(str(output)+'.cached_model.json')
    if resume and marker.exists() and json.loads(marker.read_text())==protocol and output.exists() and output.stat().st_size==count*DTYPE.itemsize:return
    model=lgb.Booster(model_file=str(model_path));assert model.num_feature()==feature_count
    rows=np.memmap(features,dtype=dtype,mode='r');temporary=Path(str(output)+'.partial')
    with temporary.open('wb') as out:
        for start in range(0,count,100000):
            end=min(count,start+100000);block=np.empty(end-start,dtype=DTYPE)
            block['ref']=rows['ref'][start:end];block['target']=rows['target'][start:end]
            block['p']=model.predict(np.ascontiguousarray(rows['x'][start:end]),num_threads=6)
            assert np.isfinite(block['p']).all();block.tofile(out)
            if end%1000000==0:print('Cached second-stage predictions',end,'/',count,flush=True)
    temporary.replace(output);marker.write_text(json.dumps(protocol,indent=2))

def apply(work,resume=False,retain_features=False):
    cfg=json.loads((work/'config.json').read_text());parts=[]
    final_classifier_frozen=False
    for country in countries(work/'test_source1.norm.tsv'):
        import re
        prefix=work/('test_part_'+re.sub(r'[^a-zA-Z0-9]','_',country));parts.append(prefix)
        rows=np.memmap(str(prefix)+'.scores.bin',dtype=DTYPE,mode='r');np.array(rows['p']).tofile(str(prefix)+'.probabilities.bin')
        library=Path(lgb.__file__).parent/'lib/lib_lightgbm.so'
        done=Path(str(prefix)+'.collective.done')
        if retain_features:
            features=Path(str(prefix)+'.collective.features.bin')
            run_batched(work/'collective',str(prefix)+'.features.bin',str(prefix)+'.probabilities.bin',work/'test_source2.norm.tsv',work/'test_source3.norm.tsv',str(prefix)+'.refs.tsv',features,work/'aliases.tsv',work/'test_source1.norm.tsv',feature_count=len(cfg['second_stage_features']),resume=resume)
            if not final_classifier_frozen:
                from select_private_resolver import choose_before_scoring
                choose_before_scoring(work)
                cfg=json.loads((work/'config.json').read_text())
                final_classifier_frozen=True
            score_cached_features(features,Path(str(prefix)+'.collective_scores.bin'),work/'collective_model.txt',len(cfg['second_stage_features']),resume=resume)
            done.write_text('Completed successfully with reusable inference features.\n')
        elif not (resume and done.exists()):
            run_batched(work/'collective',str(prefix)+'.features.bin',str(prefix)+'.probabilities.bin',work/'test_source2.norm.tsv',work/'test_source3.norm.tsv',str(prefix)+'.refs.tsv',str(prefix)+'.collective_scores.bin',work/'aliases.tsv',work/'test_source1.norm.tsv',feature_count=len(cfg['second_stage_features']),model=work/'collective_model.txt',library=library,resume=resume)
            done.write_text('Completed successfully.\n')
    offset=0
    with (work/'test.scores.bin').open('wb') as out:
        for prefix in parts:
            rows=np.memmap(str(prefix)+'.collective_scores.bin',dtype=DTYPE,mode='r')
            for start in range(0,len(rows),500000):
                block=np.array(rows[start:start+500000]);block['ref']+=offset;block.tofile(out)
            offset+=sum(1 for _ in Path(str(prefix)+'.refs.tsv').open())
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,default=ROOT/'work');ap.add_argument('--resume',action='store_true');ap.add_argument('--retain-features',action='store_true');a=ap.parse_args();apply(a.work,a.resume,a.retain_features)
