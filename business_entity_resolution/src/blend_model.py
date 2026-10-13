"""Combine two binary resolvers by a weighted mean of their raw log-odds.

Uses LightGBM's documented leaf-output and BoosterMerge C APIs. The result is
a normal single model consumed by the existing native inference path.
"""
import argparse,hashlib,json
from pathlib import Path
import lightgbm as lgb
from lightgbm.basic import _LIB,_safe_call

def scale(model,weight):
    description=model.model_to_string(num_iteration=-1)
    assert 'objective=binary sigmoid:1\n' in description
    assert model.num_model_per_iteration()==1
    tree=None
    for line in description.splitlines():
        if line.startswith('Tree='):tree=int(line.split('=',1)[1])
        elif line.startswith('leaf_value='):
            assert tree is not None
            for leaf,value in enumerate(line.split('=',1)[1].split()):
                model.set_leaf_output(tree,leaf,float(value)*weight)

def blend(first,second,output,second_weight=.5):
    if not 0<second_weight<1:raise ValueError('Both models must have a nonzero weight')
    if Path(output).exists():raise ValueError('Existing blended models are preserved')
    a=lgb.Booster(model_file=str(first));b=lgb.Booster(model_file=str(second))
    assert a.feature_name()==b.feature_name()
    trees=a.num_trees()+b.num_trees()
    scale(a,1-second_weight);scale(b,second_weight)
    _safe_call(_LIB.LGBM_BoosterMerge(a._handle,b._handle))
    assert a.num_trees()==trees
    a.save_model(str(output),num_iteration=-1)
    return {'first_model_sha256':hashlib.sha256(Path(first).read_bytes()).hexdigest(),'second_model_sha256':hashlib.sha256(Path(second).read_bytes()).hexdigest(),'second_weight':second_weight,'combination':'sigmoid((1-w)*raw_first + w*raw_second)','trees':trees,'features':a.num_feature(),'blended_model_sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest()}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--first',type=Path,required=True);ap.add_argument('--second',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--second-weight',type=float,default=.5);a=ap.parse_args()
    print(json.dumps(blend(a.first,a.second,a.output,a.second_weight),indent=2))
