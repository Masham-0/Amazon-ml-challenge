"""Reproduce training, test inference, and submission artifacts."""
import argparse,subprocess,sys,re,json,shutil,os
from pathlib import Path
import lightgbm
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
SRC=Path(__file__).resolve().parent
def run(args):print('Running:',*map(str,args),flush=True);subprocess.run(list(map(str,args)),check=True,env={**os.environ,'PYTHONUNBUFFERED':'1'})
def countries(path):
    found=set()
    with path.open() as f:
        for line in f:found.add(line.split('\t',2)[1])
    return sorted(found)
def retrieve(work,split,resume=False):
    from train import DTYPE as FEATURE_DTYPE
    from export import DTYPE as SCORE_DTYPE
    training=split=='train';dtype=FEATURE_DTYPE if training else SCORE_DTYPE
    suffix='.features.bin' if training else '.scores.bin'
    library=Path(lightgbm.__file__).parent/'lib/lib_lightgbm.so';parts=[]
    config=(json.loads((work/'candidate_config.json').read_text()) if (work/'candidate_config.json').exists() else {'gate':.35,'gap':.10}) if training else json.loads((work/'config.json').read_text())
    population=work/config.get('reference_population_file',f'{split}_source1.norm.tsv') if training else work/f'{split}_source1.norm.tsv'
    for country in countries(population):
        prefix=work/(split+'_part_'+re.sub(r'[^a-zA-Z0-9]','_',country));parts.append(prefix)
        done=Path(str(prefix)+'.done')
        if resume and done.exists():
            print('Reusing completed country:',country,flush=True);continue
        model='-' if training else ','.join(str(work/m) for m in config.get('ensemble',['model.txt']))
        args=[work/'engine',population,work/f'{split}_source2.norm.tsv',work/f'{split}_source3.norm.tsv',prefix,work/'sample_truth.tsv' if training else '-',model,'-' if training else library,str(config['gate']),str(config['gap']),work/'aliases.tsv',country]
        if 'name_only_candidate_limit' in config:
            args+=['dump' if not training and config.get('collective') else '-',str(config.get('candidate_limit',6)),str(config['name_only_candidate_limit']),str(config.get('name_only_gap',config['gap']))]
        elif not training and config.get('collective'):args+=['dump']
        run(args)
        done.write_text('Completed successfully.\n')
    offset=0
    with (work/f'{split}.refs.tsv').open('w') as rf,(work/f'{split}{suffix}').open('wb') as bf:
        for prefix in parts:
            lines=Path(str(prefix)+'.refs.tsv').read_text();rf.write(lines)
            rows=np.memmap(str(prefix)+suffix,dtype=dtype,mode='r')
            for start in range(0,len(rows),500000):
                block=np.array(rows[start:start+500000]);block['ref']+=offset;block.tofile(bf)
            offset+=len(lines.splitlines())
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['prepare','train','predict','all'],default='all')
    ap.add_argument('--work',type=Path,default=ROOT/'work')
    ap.add_argument('--reports',type=Path,default=ROOT/'reports')
    ap.add_argument('--expanded',action='store_true')
    ap.add_argument('--copies',action='store_true')
    ap.add_argument('--entity-weighted',action='store_true')
    ap.add_argument('--expanded-aliases',action='store_true',help='Fit dictionaries on additional groups, excluding development and audit groups.')
    ap.add_argument('--alias-fitting-stop',type=int,default=1000)
    ap.add_argument('--density-matched',action='store_true',help='Sample the training reference population to the supplied test target/reference ratio.')
    ap.add_argument('--strict-audit-targets',action='store_true',help='Exclude reserved audit records from supervised fitting even as negative examples.')
    ap.add_argument('--teacher-current-candidates',action='store_true',help='Refit strict first-stage teachers on the current expanded candidate matrix.')
    ap.add_argument('--device',choices=['cpu','gpu'],default='cpu',help='Training device; test inference uses the native CPU predictor.')
    ap.add_argument('--teacher-device',choices=['cpu','gpu'],help='Optional separate device for freshly fitted first-stage teachers.')
    ap.add_argument('--name-only-candidate-limit',type=int,help='Optional expanded candidate budget for targets lacking an address.')
    ap.add_argument('--name-only-gap',type=float,default=.25,help='Similarity margin for the optional name-only candidate budget.')
    ap.add_argument('--retain-inference-features',action='store_true',help='Keep label-free second-stage features for efficient frozen-model comparisons; requires extra disk.')
    ap.add_argument('--baseline-work',type=Path,default=ROOT/'work/baseline')
    ap.add_argument('--baseline-reports',type=Path,default=ROOT/'reports/baseline')
    ap.add_argument('--resume',action='store_true',help='Reuse completed country artifacts only when inputs and settings have not changed.')
    a=ap.parse_args();work=a.work.resolve();work.mkdir(exist_ok=True)
    if a.teacher_current_candidates and not (a.expanded and a.strict_audit_targets):ap.error('--teacher-current-candidates requires --expanded --strict-audit-targets')
    if (a.copies or a.entity_weighted or a.expanded_aliases or a.density_matched or a.strict_audit_targets) and not a.expanded:ap.error('Additional training variants require --expanded')
    if not 200 <= a.alias_fitting_stop <= 1000:ap.error('--alias-fitting-stop must be between 200 and 1000')
    if a.stage in ['prepare','all']:run([sys.executable,SRC/'prepare.py','--work',work]+(['--expanded-sample'] if a.expanded else []))
    if a.stage in ['train','all']:
        if a.expanded:
            data=Path((work/'data_path.txt').read_text().strip()) if (work/'data_path.txt').exists() else ROOT/'dataset/student_resource/dataset'
            run([sys.executable,SRC/'prepare.py','--work',work,'--data',data,'--split','train','--expanded-sample'])
            candidate={'gate':.30,'gap':.15,'wide_blocking':True}
            if a.name_only_candidate_limit is not None:
                candidate.update(candidate_limit=6,name_only_candidate_limit=a.name_only_candidate_limit,name_only_gap=a.name_only_gap)
            (work/'candidate_config.json').write_text(json.dumps(candidate))
        alias_args=['--expanded-fitting','--fitting-stop',str(a.alias_fitting_stop)] if a.expanded_aliases else []
        run([sys.executable,SRC/'learn_aliases.py','--work',work]+alias_args)
        run([sys.executable,SRC/'learn_whole_aliases.py','--work',work]+alias_args)
        (work/'alias_config.json').write_text(json.dumps({'alias_fitting_reference_buckets':f'0–19, 30–99, 200–{a.alias_fitting_stop-1}' if a.expanded_aliases else '0–19, 30–99','expanded_aliases':a.expanded_aliases,'alias_fitting_stop':a.alias_fitting_stop if a.expanded_aliases else 100},indent=2))
        if a.teacher_current_candidates:
            metadata=json.loads((work/'alias_config.json').read_text());metadata['teacher_candidate_population']='current_expanded';(work/'alias_config.json').write_text(json.dumps(metadata,indent=2))
        if a.teacher_device:
            metadata=json.loads((work/'alias_config.json').read_text());metadata['first_stage_training_device']=a.teacher_device;(work/'alias_config.json').write_text(json.dumps(metadata,indent=2))
        if a.density_matched:run([sys.executable,SRC/'prepare_density.py','--work',work])
        run(['g++','-O3','-std=c++17','-fopenmp']+(['-DWIDE_BLOCKING'] if a.expanded else [])+[SRC/'engine.cpp','-ldl','-o',work/'engine'])
        retrieve(work,'train',a.resume)
        if a.expanded:
            if a.strict_audit_targets:run([sys.executable,SRC/'fit_reserved_teachers.py','--baseline-work',work if a.teacher_current_candidates else a.baseline_work,'--output-work',work,'--device',a.teacher_device or a.device])
            run(['g++','-O3','-std=c++17','-fopenmp','-DEXTRA_TFIDF']+(['-DCOPY_SUPPORT'] if a.copies else [])+[SRC/'collective.cpp','-ldl','-o',work/'collective'])
            run([sys.executable,SRC/'expanded_train.py','--work',work,'--reports',a.reports,'--baseline-work',a.baseline_work,'--baseline-reports',a.baseline_reports,'--device',a.device]+(['--copies'] if a.copies else [])+(['--entity-weighted'] if a.entity_weighted else []))
        else:
            run([sys.executable,SRC/'train.py','--work',work,'--reports',a.reports])
            run(['g++','-O3','-std=c++17','-fopenmp',SRC/'collective.cpp','-ldl','-o',work/'collective'])
            run([sys.executable,SRC/'crossfit.py','--work',work,'--reports',a.reports])
        run([sys.executable,SRC/'decision.py','--work',work,'--reports',a.reports])
    if a.stage in ['predict','all']:
        for artifact in ['aliases.tsv','config.json']:
            destination=work/artifact
            if not destination.exists():shutil.copyfile(SRC.parent/'models'/artifact,destination)
        library=Path(lightgbm.__file__).parent/'lib/lib_lightgbm.so'
        config=json.loads((work/'config.json').read_text())
        for artifact in config.get('ensemble',['model.txt'])+(['collective_model.txt'] if config.get('collective') else []):
            if not (work/artifact).exists():shutil.copyfile(SRC.parent/'models'/artifact,work/artifact)
        run(['g++','-O3','-std=c++17','-fopenmp']+(['-DWIDE_BLOCKING'] if config.get('wide_blocking') else [])+[SRC/'engine.cpp','-ldl','-o',work/'engine'])
        retrieve(work,'test',a.resume)
        if config.get('collective'):
            run(['g++','-O3','-std=c++17','-fopenmp']+(['-DEXTRA_TFIDF'] if config.get('tfidf') else [])+(['-DCOPY_SUPPORT'] if config.get('copies') else [])+[SRC/'collective.cpp','-ldl','-o',work/'collective'])
            run([sys.executable,SRC/'collective.py','--work',work]+(['--resume'] if a.resume else [])+(['--retain-features'] if a.retain_inference_features else []))
        run([sys.executable,SRC/'export.py','--work',work])
if __name__=='__main__':main()
