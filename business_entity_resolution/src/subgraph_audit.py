"""Audit frozen models with exact context for a sampled reference cohort.

The first candidate pass identifies every competing reference for an audit
query. The second pass recovers all candidate records of those references.
Related-record features for every competitor on an audit query then equal
full-population inference. No truth labels are read until scoring is finished.
"""
import argparse,hashlib,json,shutil,subprocess,sys,zlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import DTYPE
from export import DTYPE as SCORE_DTYPE
from run import retrieve
from expanded_train import predict_chunks
from collective_batch import run_batched
from collective import score_cached_features
SRC=Path(__file__).resolve().parent

def link(source,destination):
    if destination.is_symlink():destination.unlink()
    elif destination.exists():raise ValueError('existing non-link '+str(destination))
    destination.symlink_to(source.resolve())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()

def main(model_work,audit_work,report,start,buckets,resume,retain_features=True):
    cfg=json.loads((model_work/'config.json').read_text());stop=start+buckets
    audit_groups=np.arange(start,stop)
    fitted=(audit_groups<100)
    if cfg.get('tfidf'):fitted|=(audit_groups>=200)&(audit_groups<400)
    if cfg.get('expanded_aliases'):fitted|=(audit_groups>=200)&(audit_groups<cfg.get('alias_fitting_stop',1000))
    if start<0 or stop>1000 or fitted.any():raise ValueError('audit groups overlap fitting groups or fall outside CRC32 sampling buckets')
    audit_work.mkdir(exist_ok=True,parents=True)
    names=cfg.get('ensemble',['model.txt'])+(['collective_model.txt'] if cfg.get('collective') else [])+['aliases.tsv','config.json'];hashes={name:sha(model_work/name) for name in names};protocol={'model_hashes':hashes,'audit_start':start,'audit_buckets':buckets}
    marker=audit_work/'protocol.json'
    if resume and marker.exists() and json.loads(marker.read_text())!=protocol:raise ValueError('audit inputs changed; use a new audit directory')
    marker.write_text(json.dumps(protocol,indent=2));(audit_work/'config.json').write_text(json.dumps(cfg,indent=2))
    data=(model_work/'data_path.txt').read_text().strip() if (model_work/'data_path.txt').exists() else str(Path(__file__).resolve().parents[3]/'dataset/student_resource/dataset')
    (audit_work/'data_path.txt').write_text(data)
    population=model_work/cfg.get('reference_population_file','train_source1.norm.tsv')
    primary=set()
    with population.open() as f:
        for line in f:
            sid=line.split('\t',1)[0]
            if start<=zlib.crc32(sid.encode())%1000<stop:primary.add(sid)
    def setup(directory,cohort):
        directory.mkdir(exist_ok=True)
        for source in [1,2,3]:link(population if source==1 else model_work/f'train_source{source}.norm.tsv',directory/f'train_source{source}.norm.tsv')
        link(model_work/'engine',directory/'engine');link(model_work/'aliases.tsv',directory/'aliases.tsv')
        candidate_keys=['gate','gap','candidate_limit','name_only_candidate_limit','name_only_gap']
        (directory/'candidate_config.json').write_text(json.dumps({k:cfg[k] for k in candidate_keys if k in cfg}))
        # Empty lists mark reference membership only; they are not labels.
        (directory/'sample_truth.tsv').write_text(''.join(sid+'\t\n' for sid in sorted(cohort)))
    first=audit_work/'query_candidates';setup(first,primary);retrieve(first,'train',resume)
    ids=[line.split('\t',1)[0] for line in (first/'train.refs.tsv').open()];rows=np.memmap(first/'train.features.bin',dtype=DTYPE,mode='r');cohort=primary|{ids[int(r)] for r in np.unique(rows['ref'])}
    print('Audit primary references',len(primary),'complete-context cohort',len(cohort),flush=True)
    second=audit_work/'complete_context';setup(second,cohort);retrieve(second,'train',resume)
    context=np.memmap(second/'train.features.bin',dtype=DTYPE,mode='r');assert np.array_equal(np.array(ids),np.array([line.split('\t',1)[0] for line in (second/'train.refs.tsv').open()]))
    first_keys=(rows['ref'].astype(np.uint64)<<32)|rows['target'];second_keys=(context['ref'].astype(np.uint64)<<32)|context['target'];assert np.isin(first_keys,second_keys).all()
    del first_keys,second_keys,rows
    probability=np.zeros(len(context),dtype=np.float64);models=cfg.get('ensemble',['model.txt'])
    for name in models:
        model=lgb.Booster(model_file=str(model_work/name));probability+=predict_chunks(model,context['x'],'audit '+name)/len(models)
    probability.astype(np.float32).tofile(second/'probabilities.bin')
    if cfg.get('collective'):
        lib=Path(lgb.__file__).parent/'lib/lib_lightgbm.so'
        if retain_features:
            features=second/'collective.features.bin'
            run_batched(model_work/'collective',second/'train.features.bin',second/'probabilities.bin',model_work/'train_source2.norm.tsv',model_work/'train_source3.norm.tsv',second/'train.refs.tsv',features,model_work/'aliases.tsv',population,feature_count=len(cfg['second_stage_features']),resume=resume)
            score_cached_features(features,second/'resolved.scores.bin',model_work/'collective_model.txt',len(cfg['second_stage_features']),resume=resume)
        else:
            run_batched(model_work/'collective',second/'train.features.bin',second/'probabilities.bin',model_work/'train_source2.norm.tsv',model_work/'train_source3.norm.tsv',second/'train.refs.tsv',second/'resolved.scores.bin',model_work/'aliases.tsv',population,feature_count=len(cfg['second_stage_features']),model=model_work/'collective_model.txt',library=lib,resume=resume)
    else:
        out=np.lib.format.open_memmap(second/'temporary.npy',dtype=SCORE_DTYPE,mode='w+',shape=(len(context),));out['ref']=context['ref'];out['target']=context['target'];out['p']=probability;out.tofile(second/'resolved.scores.bin');del out;(second/'temporary.npy').unlink()
    link(second/'resolved.scores.bin',audit_work/'test.scores.bin');link(second/'train.refs.tsv',audit_work/'test.refs.tsv')
    subprocess.run([sys.executable,str(SRC/'audit.py'),'--work',str(audit_work),'--report',str(report),'--audit-start',str(start),'--audit-buckets',str(buckets)],check=True)
    result=json.loads(report.read_text());result.update(model_hashes=hashes,context={'audit_references':len(primary),'context_references':len(cohort),'scored_context_pairs':len(context),'primary_pairs_retained':True,'truth_used_only_after_frozen_inference':True,'collective_features_retained':bool(retain_features and cfg.get('collective'))});report.write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-work',type=Path,required=True);ap.add_argument('--audit-work',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--audit-start',type=int,default=100);ap.add_argument('--audit-buckets',type=int,default=5);ap.add_argument('--resume',action='store_true');ap.add_argument('--no-feature-cache',action='store_true',help='Score directly without retaining reusable second-stage features.');a=ap.parse_args();main(a.model_work.resolve(),a.audit_work.resolve(),a.report.resolve(),a.audit_start,a.audit_buckets,a.resume,not a.no_feature_cache)
