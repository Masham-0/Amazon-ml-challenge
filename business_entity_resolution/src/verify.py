"""Check native inference and feature alignment on supplied-record fixtures."""
import argparse,json,subprocess
import collections,math
from pathlib import Path
import numpy as np
import lightgbm as lgb
from rapidfuzz.fuzz import ratio
from train import DTYPE,FEATURES
from crossfit import SECOND_DTYPE,SECOND_FEATURES
from export import DTYPE as SCORE_DTYPE

def main(work,fixture,report,features_only=False,copies=False):
    cfg=(json.loads((work/'config.json').read_text()) if not features_only else {'gate':.30,'gap':.15,'ensemble':[f'fold{k}.model.txt' for k in range(3)],'collective':True,'tfidf':True,'copies':copies});lib=Path(lgb.__file__).parent/'lib/lib_lightgbm.so'
    second_dtype=SECOND_DTYPE;second_features=SECOND_FEATURES
    if cfg.get('tfidf'):
        from expanded_train import SECOND_DTYPE as second_dtype,FEATURE_NAMES as second_features
    if cfg.get('copies'):
        from expanded_train import COPY_FEATURES
        second_features=second_features+COPY_FEATURES
        second_dtype=np.dtype([('ref','u4'),('target','u4'),('x','f4',(len(second_features),))])
    prefix=fixture/'verified';models=cfg.get('ensemble',['model.txt'])
    engine_args=[str(work/'engine'),str(fixture/'source1.tsv'),str(fixture/'source2.tsv'),str(fixture/'source3.tsv'),str(prefix),'-',','.join(str(work/p) for p in models),str(lib),str(cfg['gate']),str(cfg['gap']),str(work/'aliases.tsv'),'India','dump']
    if 'name_only_candidate_limit' in cfg:engine_args += [str(cfg.get('candidate_limit',6)),str(cfg['name_only_candidate_limit']),str(cfg.get('name_only_gap',cfg['gap']))]
    subprocess.run(engine_args,check=True)
    features=np.fromfile(str(prefix)+'.features.bin',dtype=DTYPE);scores=np.fromfile(str(prefix)+'.scores.bin',dtype=SCORE_DTYPE)
    assert len(features)>0 and np.array_equal(features['ref'],scores['ref']) and np.array_equal(features['target'],scores['target'])
    assert np.isfinite(features['x']).all()
    expected=np.mean([lgb.Booster(model_file=str(work/p)).predict(features['x'],num_threads=4) for p in models],axis=0)
    delta=float(np.max(np.abs(expected-scores['p'])));assert delta<1e-6
    result={'fixture_pairs':len(features),'base_features':len(FEATURES),'ensemble_models':len(models),'maximum_native_probability_error':delta,'finite_features':True,'pair_alignment':True}
    if cfg.get('collective'):
        scores['p'].tofile(fixture/'verified.prob.bin')
        args=[str(work/'collective'),str(prefix)+'.features.bin',str(fixture/'verified.prob.bin'),str(fixture/'source2.tsv'),str(fixture/'source3.tsv'),str(prefix)+'.refs.tsv',str(fixture/'verified.collective.features.bin'),str(work/'aliases.tsv'),'train','-','-',str(fixture/'source1.tsv')]
        subprocess.run(args,check=True);second=np.fromfile(fixture/'verified.collective.features.bin',dtype=second_dtype)
        assert np.array_equal(second['ref'],features['ref']) and np.array_equal(second['target'],features['target']) and np.array_equal(second['x'][:,:len(FEATURES)],features['x'])
        assert np.isfinite(second['x']).all() and np.array_equal(second['x'][:,134],scores['p'])
        # These aggregate inputs are computed independently from the native
        # related-record feature builder, including removal of the query row.
        for r in np.unique(features['ref']):
            ix=np.flatnonzero(features['ref']==r)
            for j in ix:
                others=ix[features['target'][ix]!=features['target'][j]];p=scores['p'][others]
                assert abs(second['x'][j,137]-(p.mean() if len(p) else 0))<1e-6
                for k,t in enumerate([.5,.8,.95,.99]):assert second['x'][j,138+k]==(p>=t).sum()
        aliases=dict(line.rstrip('\n').split('\t') for line in (work/'aliases.tsv').open())
        def canonical(name):
            words=[]
            for token in name.split():
                value=aliases.get(token,token)
                if value!='-':words.extend(value.split())
            value=' '.join(sorted(set(words)));return aliases.get('%'+value,value)
        refs={};targets={}
        for line in (fixture/'source1.tsv').open():
            x=line.rstrip('\n').split('\t');refs[x[0]]=canonical(x[2])
        for s in [2,3]:
            for line in (fixture/f'source{s}.tsv').open():
                x=line.rstrip('\n').split('\t');targets[int(x[0][3:])|((1<<31) if s==3 else 0)]=canonical(x[2])
        ids=[line.split('\t',1)[0] for line in Path(str(prefix)+'.refs.tsv').open()]
        dictionary=np.array([ratio(refs[ids[r]],targets[t])/100 if refs[ids[r]] and targets[t] else 0 for r,t in zip(features['ref'],features['target'])])
        assert np.max(abs(dictionary-second['x'][:,159]))<1e-6
        if cfg.get('tfidf'):
            def text(line):
                fields=line.rstrip('\n').split('\t');sets=[]
                for address,n in [(False,fields[2]),(True,fields[3])]:
                    words=[]
                    for token in n.split():
                        value=aliases.get(('@' if address else '')+token,token)
                        if value!='-':words.extend(value.split())
                    words=set(words);value=' '.join(sorted(words));value=value if address else value.replace(' ','');sets.extend([set(value[i:i+3] for i in range(len(value)-2)),words])
                return fields[1],(sets[0],sets[2],sets[1],sets[3])
            reftext={line.split('\t',1)[0]:text(line) for line in (fixture/'source1.tsv').open()};targettext={}
            for source in [2,3]:
                for line in (fixture/f'source{source}.tsv').open():targettext[int(line.split('\t',1)[0][3:])|((1<<31) if source==3 else 0)]=text(line)
            df=collections.defaultdict(collections.Counter);count=collections.Counter()
            for sid in ids:
                country,sets=reftext[sid];count[country]+=1
                for kind,values in enumerate(sets):df[country,kind].update(values)
            expected=np.empty((len(features),6));weights={}
            def summary(values,country,kind):
                key=(frozenset(values),country,kind)
                if key not in weights:
                    w={token:math.log1p(count[country]/max(1,df[country,kind][token])) for token in values};weights[key]=(w,sum(v*v for v in w.values()),sum(w.values()))
                return weights[key]
            for j,(r,t) in enumerate(zip(features['ref'],features['target'])):
                country,a=reftext[ids[r]];_,b=targettext[t]
                for kind in range(4):
                    wa,na,sa=summary(a[kind],country,kind);wb,nb,sb=summary(b[kind],country,kind);shared=a[kind]&b[kind];dot=sum(wa[token]**2 for token in shared);expected[j,kind]=dot/math.sqrt(na*nb) if na and nb else 0
                    if kind==0:
                        overlap=sum(wa[token] for token in shared);expected[j,4]=overlap/sa if sa else 0;expected[j,5]=overlap/sb if sb else 0
            error=float(np.max(abs(expected-second['x'][:,160:166])));assert error<2e-6;result['maximum_tfidf_feature_error']=error
            if cfg.get('copies'):
                signatures={}
                for source in [2,3]:
                    for line in (fixture/f'source{source}.tsv').open():
                        fields=line.rstrip('\n').split('\t');country,sets=text(line)
                        name=canonical(fields[2]);address=' '.join(sorted(sets[3]))
                        target=int(fields[0][3:])|((1<<31) if source==3 else 0)
                        signatures[target]=(name,address) if name and address else None
                expected=np.zeros((len(features),4))
                for r in np.unique(features['ref']):
                    ix=np.flatnonzero(features['ref']==r)
                    for j in ix:
                        signature=signatures[features['target'][j]]
                        peers=[k for k in ix if k!=j and signature is not None and signatures[features['target'][k]]==signature]
                        if peers:
                            expected[j]=[len(peers),sum(features['target'][k]>>31==0 for k in peers),sum(features['target'][k]>>31==1 for k in peers),scores['p'][peers].mean()]
                error=float(np.max(abs(expected-second['x'][:,166:170])));assert error<2e-6;result['maximum_copy_feature_error']=error
        if not features_only:
            args[6]=str(fixture/'verified.collective.scores.bin');args[8]='score';args[9]=str(work/'collective_model.txt');args[10]=str(lib);subprocess.run(args,check=True)
            native=np.fromfile(fixture/'verified.collective.scores.bin',dtype=SCORE_DTYPE);expected=lgb.Booster(model_file=str(work/'collective_model.txt')).predict(second['x'],num_threads=4)
            delta=float(np.max(abs(expected-native['p'])));assert delta<1e-6;result['maximum_collective_probability_error']=delta
        result.update(collective_features=len(second_features),leave_query_out_aggregates=True,dictionary_feature_verified=True)
    report.parent.mkdir(exist_ok=True,parents=True);report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--fixture',type=Path,required=True);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--tfidf-features-only',action='store_true');ap.add_argument('--copies-features-only',action='store_true');a=ap.parse_args();main(a.work,a.fixture,a.report,a.tfidf_features_only or a.copies_features_only,a.copies_features_only)
