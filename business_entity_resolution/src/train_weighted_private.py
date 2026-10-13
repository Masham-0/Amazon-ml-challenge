"""Private CPU trial: macro-entity weighting on frozen provided-data features."""
import gc,gzip,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

import lightgbm as lgb
assert lgb.__version__=='4.6.0','Use the verified bundled Kaggle LightGBM image; no network install is required'

source=next(Path('/kaggle/input').rglob('manifest.json')).parent
manifest=json.loads((source/'manifest.json').read_text())
out=Path('/kaggle/working');temp=Path('/tmp/shawarma_weighted');temp.mkdir(exist_ok=True)
for name in ['features.f32','labels_flags.u8']:
    # Kaggle may automatically unpack gzip uploads during dataset processing.
    if (source/(name+'.gz')).exists():
        with gzip.open(source/(name+'.gz'),'rb') as inp,(temp/name).open('wb') as dest:shutil.copyfileobj(inp,dest,8*1024*1024)
    else:shutil.copyfile(source/name,temp/name)
width=manifest['features_per_pair'];count=manifest['rows']
assert (temp/'features.f32').stat().st_size==count*width*4
assert (temp/'labels_flags.u8').stat().st_size==count*4
meta=np.memmap(temp/'labels_flags.u8',mode='r',dtype=[('y','u1'),('fit','u1'),('tuning','u1'),('size','u1')])
assert hashlib.sha256(meta).hexdigest()==manifest['metadata_sha256']
x=np.memmap(temp/'features.f32',mode='r',dtype='<f4',shape=(count,width))
params=dict(objective='binary',metric='binary_logloss',learning_rate=.05,num_leaves=95,min_data_in_leaf=200,feature_fraction=.9,lambda_l2=8,num_threads=min(6,os.cpu_count() or 4),verbosity=-1,seed=2026,deterministic=True,force_col_wise=True,max_bin=127)
weights=1./np.maximum(1,meta['size'].astype(np.float32))

def dataset(mask,reference=None):
    ix=np.flatnonzero(mask);matrix=np.empty((len(ix),width),dtype=np.float32)
    for start in range(0,len(ix),50000):matrix[start:start+50000]=x[ix[start:start+50000]]
    ds=lgb.Dataset(matrix,label=meta['y'][ix],weight=weights[ix]/weights[ix].mean(),feature_name=manifest['feature_names'],params=params,reference=reference,free_raw_data=True)
    ds.construct();del matrix,ix;gc.collect();return ds

def checkpoint(label):
    def save(env):
        if (env.iteration+1)%100==0:
            env.model.save_model(str(out/(label+'_checkpoint.txt')),num_iteration=-1)
            (out/'progress.json').write_text(json.dumps({'stage':label,'iteration':env.iteration+1,'time':time.time()},indent=2))
    save.order=20;return save

print('PRIVATE CPU TRAINING',count,'rows',width,'features',flush=True)
started=time.time();fit=dataset(meta['fit']>0);val=dataset(meta['tuning']>0,reference=fit)
model=lgb.train(params,fit,num_boost_round=800,valid_sets=[val],callbacks=[lgb.early_stopping(60),lgb.log_evaluation(100),checkpoint('validation')])
best=model.best_iteration;model.save_model(str(out/'collective_validation_model.txt'),num_iteration=best)
report={'best_iteration':best,'validation_binary_logloss':float(model.best_score['valid_0']['binary_logloss']),'training_parameters':params,'entity_weighted':True,'training_device':'cpu','first_stage_training_device':'gpu','feature_names':manifest['feature_names'],'export_metadata_sha256':manifest['metadata_sha256'],'local_complete_context_evaluation_required':True,'reported_score_is_not_challenge_f05':True,'lightgbm_version':lgb.__version__}
(out/'remote_training_report.json').write_text(json.dumps(report,indent=2))
del model,fit,val;gc.collect()
final=dataset(np.ones(count,dtype=bool))
model=lgb.train(params,final,num_boost_round=best,callbacks=[lgb.log_evaluation(100),checkpoint('final')])
model.save_model(str(out/'collective_model.txt'));report['seconds']=time.time()-started
report['model_sha256']=hashlib.sha256((out/'collective_model.txt').read_bytes()).hexdigest()
(out/'remote_training_report.json').write_text(json.dumps(report,indent=2))
(out/'complete.json').write_text(json.dumps({'complete':True,'model_sha256':report['model_sha256']},indent=2))
print('COMPLETE',json.dumps(report),flush=True)
