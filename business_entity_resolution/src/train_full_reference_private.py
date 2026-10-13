"""Private Kaggle CPU trial from frozen supplied-data direct pair features."""
import gc,gzip,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

import lightgbm as lgb
assert lgb.__version__=='4.6.0','Use the verified bundled Kaggle LightGBM image; no network install is required'

source=next(Path('/kaggle/input').rglob('manifest.json')).parent
manifest=json.loads((source/'manifest.json').read_text())
out=Path('/kaggle/working');temp=Path('/tmp/shawarma_full_reference');temp.mkdir(exist_ok=True)
for name in ['features.f32','labels_flags.u8']:
    if (source/(name+'.gz')).exists():
        with gzip.open(source/(name+'.gz'),'rb') as inp,(temp/name).open('wb') as dest:shutil.copyfileobj(inp,dest,8*1024*1024)
    else:shutil.copyfile(source/name,temp/name)
count=manifest['rows'];active=manifest['active_features'];width=manifest['model_features']
assert (temp/'features.f32').stat().st_size==count*active*4
metadata=np.memmap(temp/'labels_flags.u8',mode='r',dtype=[('y','u1'),('fit','u1'),('tuning','u1'),('size','u1')])
assert len(metadata)==count and hashlib.sha256(metadata).hexdigest()==manifest['metadata_sha256']
x=np.memmap(temp/'features.f32',mode='r',dtype='<f4',shape=(count,active))
params=dict(objective='binary',metric='binary_logloss',learning_rate=.05,num_leaves=95,min_data_in_leaf=200,feature_fraction=.9,lambda_l2=8,num_threads=min(6,os.cpu_count() or 4),verbosity=-1,seed=2026,deterministic=True,force_col_wise=True,max_bin=127)
weights=1./np.maximum(1,metadata['size'].astype(np.float32))

def dataset(mask,reference=None):
    ix=np.flatnonzero(mask);matrix=np.zeros((len(ix),width),dtype=np.float32)
    for start in range(0,len(ix),50000):matrix[start:start+50000,:active]=x[ix[start:start+50000]]
    ds=lgb.Dataset(matrix,label=metadata['y'][ix],weight=weights[ix]/weights[ix].mean(),feature_name=manifest['feature_names'],params=params,reference=reference,free_raw_data=True)
    ds.construct();del matrix,ix;gc.collect();return ds

def checkpoint(label):
    def save(env):
        if (env.iteration+1)%100==0:
            env.model.save_model(str(out/(label+'_checkpoint.txt')),num_iteration=-1)
            (out/'progress.json').write_text(json.dumps({'stage':label,'iteration':env.iteration+1,'time':time.time()},indent=2))
    save.order=20;return save

def validate_active_features(model):
    assert model.num_feature()==width and model.feature_name()==manifest['feature_names']
    def walk(node):
        if 'split_feature' in node:
            assert node['split_feature']<active
            walk(node['left_child']);walk(node['right_child'])
    for tree in model.dump_model()['tree_info']:walk(tree['tree_structure'])

started=time.time();print('PRIVATE FULL-REFERENCE CPU TRAINING',count,'rows',active,'active features',flush=True)
fit=dataset(metadata['fit']>0);validation=dataset(metadata['tuning']>0,reference=fit)
model=lgb.train(params,fit,num_boost_round=1000,valid_sets=[validation],callbacks=[lgb.early_stopping(60),lgb.log_evaluation(100),checkpoint('validation')])
best=model.best_iteration;validate_active_features(model);model.save_model(str(out/'collective_validation_model.txt'),num_iteration=best)
report={'best_iteration':best,'validation_binary_logloss':float(model.best_score['valid_0']['binary_logloss']),'training_parameters':params,'entity_weighted':True,'training_device':'cpu','feature_names':manifest['feature_names'],'active_features':active,'inactive_features_constant_zero':True,'export_metadata_sha256':manifest['metadata_sha256'],'full_reference_labels':True,'local_complete_context_evaluation_required':True,'reported_score_is_not_challenge_f05':True,'lightgbm_version':lgb.__version__}
(out/'remote_training_report.json').write_text(json.dumps(report,indent=2))
del model,fit,validation;gc.collect()
fit=dataset(np.ones(count,dtype=bool));model=lgb.train(params,fit,num_boost_round=best,callbacks=[lgb.log_evaluation(100),checkpoint('final')])
validate_active_features(model);model.save_model(str(out/'collective_model.txt'))
report['seconds']=time.time()-started;report['model_sha256']=hashlib.sha256((out/'collective_model.txt').read_bytes()).hexdigest()
(out/'remote_training_report.json').write_text(json.dumps(report,indent=2));(out/'complete.json').write_text(json.dumps({'complete':True,'model_sha256':report['model_sha256']},indent=2))
print('COMPLETE',json.dumps(report),flush=True)
